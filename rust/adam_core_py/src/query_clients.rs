use crate::http::{get_text, get_text_with_status, percent_encode};
use adam_core_rs_coords::{
    cometary_to_cartesian6, optical_obs80_record_batch, parse_optical_obs80_file,
    transform_with_covariance_flat6, CoordinateBatch, CoordinateRepresentation, CovarianceBatch,
    CovarianceUnits, DataFrame, IntoNestedRecordBatch, ObjectId, OrbitBatch, OrbitId,
    OrbitVariantBatch, OriginArray, OriginId, PhysicalParametersBatch, Representation, TimeArray,
    TimeScale, VariantId,
};
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use arrow_array::{
    Array, ArrayRef, Int64Array, Int8Array, LargeStringArray, RecordBatch, StructArray,
};
use arrow_schema::{DataType, Field, Schema};
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

const MU_SUN: f64 = 0.000_295_912_208_284_119_56;
const JD_MINUS_MJD: f64 = 2_400_000.5;
const NEOCC_URL: &str = "https://neo.ssa.esa.int/PSDB-portlet/download";
const SCOUT_URL: &str = "https://ssd-api.jpl.nasa.gov/scout.api";
const SBDB_URL: &str = "https://ssd-api.jpl.nasa.gov/sbdb.api";

fn mjd_values(times: &TimeArray) -> Vec<f64> {
    times
        .epochs
        .iter()
        .map(|epoch| epoch.days as f64 + epoch.nanos as f64 / 86_400_000_000_000.0)
        .collect()
}

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn parse_json(text: &str, context: &str) -> PyResult<Value> {
    serde_json::from_str(text).map_err(|err| value_error(format!("invalid {context} JSON: {err}")))
}

fn field<'a>(value: &'a Value, name: &str) -> PyResult<&'a Value> {
    value
        .get(name)
        .ok_or_else(|| value_error(format!("missing field {name:?}")))
}

fn string_field(value: &Value, name: &str) -> PyResult<String> {
    field(value, name)?
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| value_error(format!("field {name:?} must be a string")))
}

fn number_field(value: &Value, name: &str) -> PyResult<f64> {
    field(value, name)?
        .as_f64()
        .ok_or_else(|| value_error(format!("field {name:?} must be numeric")))
}

fn number_array(value: &Value, name: &str) -> PyResult<Vec<f64>> {
    field(value, name)?
        .as_array()
        .ok_or_else(|| value_error(format!("field {name:?} must be an array")))?
        .iter()
        .map(|entry| {
            entry
                .as_f64()
                .ok_or_else(|| value_error(format!("{name} entries must be numeric")))
        })
        .collect()
}

fn matrix_rows(value: &Value, name: &str) -> PyResult<Vec<[f64; 6]>> {
    field(value, name)?
        .as_array()
        .ok_or_else(|| value_error(format!("field {name:?} must be an array")))?
        .iter()
        .map(|row| {
            let row = row
                .as_array()
                .ok_or_else(|| value_error(format!("{name} rows must be arrays")))?;
            if row.len() != 6 {
                return Err(value_error(format!("{name} rows must have length 6")));
            }
            let mut output = [0.0; 6];
            for (slot, item) in output.iter_mut().zip(row) {
                *slot = item
                    .as_f64()
                    .ok_or_else(|| value_error(format!("{name} entries must be numeric")))?;
            }
            Ok(output)
        })
        .collect()
}

fn string_array(value: &Value, name: &str) -> PyResult<Vec<String>> {
    field(value, name)?
        .as_array()
        .ok_or_else(|| value_error(format!("field {name:?} must be an array")))?
        .iter()
        .map(|entry| {
            entry
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| value_error(format!("{name} entries must be strings")))
        })
        .collect()
}

fn next_recorded(
    recorded: Option<&[String]>,
    cursor: &mut usize,
    url: &str,
    service: &str,
) -> PyResult<String> {
    if let Some(recorded) = recorded {
        let payload = recorded
            .get(*cursor)
            .ok_or_else(|| value_error(format!("not enough recorded {service} responses")))?
            .clone();
        *cursor += 1;
        Ok(payload)
    } else {
        get_text(url, Duration::from_secs(60), service)
    }
}

fn scout_error(
    kind: &str,
    object_id: &str,
    upstream_status: Option<u16>,
    message: impl Into<String>,
) -> PyErr {
    value_error(format!(
        "__SCOUT_ERROR__:{}",
        serde_json::json!({
            "kind": kind,
            "object_id": object_id,
            "upstream_status": upstream_status,
            "message": message.into(),
        })
    ))
}

fn validate_scout_payload(object_id: &str, payload: &str) -> PyResult<Value> {
    let value: Value = serde_json::from_str(payload).map_err(|error| {
        scout_error(
            "response",
            object_id,
            None,
            format!("Scout returned invalid JSON for {object_id:?}: {error}"),
        )
    })?;
    if !value.is_object() {
        return Err(scout_error(
            "response",
            object_id,
            None,
            format!("Scout returned a non-object response for {object_id:?}"),
        ));
    }
    if let Some(detail) = value.get("error").filter(|value| !value.is_null()) {
        let detail = detail
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| detail.to_string());
        let normalized = detail.to_lowercase();
        let kind = if [
            "not found",
            "no object",
            "does not exist",
            "unknown object",
            "invalid designation",
        ]
        .iter()
        .any(|token| normalized.contains(token))
        {
            "not_found"
        } else if [
            "temporar",
            "service unavailable",
            "timeout",
            "try again",
            "rate limit",
            "internal server",
        ]
        .iter()
        .any(|token| normalized.contains(token))
        {
            "service_unavailable"
        } else {
            "response"
        };
        let message = match kind {
            "not_found" => format!("Scout object {object_id:?} is unavailable: {detail}"),
            "service_unavailable" => {
                format!("Scout is temporarily unavailable for {object_id:?}: {detail}")
            }
            _ => format!("Scout returned an error for {object_id:?}: {detail}"),
        };
        return Err(scout_error(kind, object_id, None, message));
    }
    Ok(value)
}

fn require_scout_signature<'a>(payload: &'a Value, object_id: &str) -> PyResult<&'a Value> {
    let signature = payload.get("signature");
    if signature
        .and_then(|value| value.as_object())
        .and_then(|value| value.get("version"))
        .and_then(Value::as_str)
        != Some("1.3")
    {
        return Err(scout_error(
            "response",
            object_id,
            None,
            format!(
                "Unsupported or missing Scout API signature for {object_id}: {}",
                signature.map_or_else(|| "null".to_string(), Value::to_string)
            ),
        ));
    }
    Ok(signature.expect("validated Scout signature exists"))
}

#[allow(clippy::too_many_arguments)]
fn scout_object_payload(
    object_id: &str,
    query: &str,
    timeout_s: f64,
    max_attempts: usize,
    retry_delay_s: f64,
    recorded: Option<&[String]>,
    cursor: &mut usize,
) -> PyResult<Value> {
    if let Some(recorded) = recorded {
        let payload = recorded
            .get(*cursor)
            .ok_or_else(|| value_error("not enough recorded Scout responses"))?;
        *cursor += 1;
        return validate_scout_payload(object_id, payload);
    }
    let url = format!("{SCOUT_URL}?tdes={}&{query}", percent_encode(object_id));
    let attempts = max_attempts.max(1);
    let timeout = Duration::try_from_secs_f64(timeout_s)
        .map_err(|error| value_error(format!("invalid Scout timeout: {error}")))?;
    for attempt in 0..attempts {
        match get_text_with_status(&url, timeout, "Scout") {
            Ok(payload) => return validate_scout_payload(object_id, &payload),
            Err(error) => {
                let status = error.status;
                if status == Some(404) {
                    return Err(scout_error(
                        "not_found",
                        object_id,
                        status,
                        format!("Scout object {object_id:?} was not found"),
                    ));
                }
                let retryable = status.is_none_or(|value| value == 429 || value >= 500);
                if !retryable {
                    return Err(scout_error(
                        "query",
                        object_id,
                        status,
                        format!("Scout request failed for {object_id:?} with HTTP {status:?}"),
                    ));
                }
                if attempt + 1 == attempts {
                    return Err(scout_error(
                        "service_unavailable",
                        object_id,
                        status,
                        format!("Scout is temporarily unavailable for {object_id:?}; retry later"),
                    ));
                }
                std::thread::sleep(
                    Duration::try_from_secs_f64(retry_delay_s * 2.0_f64.powi(attempt as i32))
                        .map_err(|error| {
                            value_error(format!("invalid Scout retry delay: {error}"))
                        })?,
                );
            }
        }
    }
    unreachable!("Scout attempts are always at least one")
}

fn neocc_orbits(
    object_ids: &[String],
    orbit_type: &str,
    orbit_epoch: &str,
    recorded: Option<&[String]>,
) -> PyResult<OrbitBatch> {
    if orbit_type == "eq" {
        return Err(PyNotImplementedError::new_err(
            "Equinoctial elements are not supported yet.",
        ));
    }
    if orbit_type != "ke" {
        return Err(value_error(format!("Invalid orbit type: {orbit_type}")));
    }
    let epoch_suffix = match orbit_epoch {
        "middle" => 0,
        "present-day" => 1,
        other => return Err(value_error(format!("Invalid orbit epoch: {other}"))),
    };
    let mut cursor = 0usize;
    let mut orbit_ids = Vec::new();
    let mut coords = Vec::new();
    let mut covariance = Vec::new();
    let mut epochs = Vec::new();
    let mut h_values = Vec::new();
    let mut g_values = Vec::new();
    for object_id in object_ids {
        let clean = object_id.replace(' ', "");
        let filename = format!("{clean}.{orbit_type}{epoch_suffix}");
        let url = format!("{NEOCC_URL}?file={}", percent_encode(&filename));
        let response = next_recorded(recorded, &mut cursor, &url, "NEOCC")?;
        if response.is_empty() {
            continue;
        }
        let normalized = adam_core_rs_coords::neocc_parse_oef_json(&response)
            .map_err(|err| value_error(err.to_string()))?;
        let parsed = parse_json(&normalized, "normalized NEOCC")?;
        let header = field(&parsed, "header")?;
        let reference = string_field(header, "refsys")?;
        if reference != "ECLM J2000" {
            return Err(value_error(format!(
                "Unsupported reference system: {reference}"
            )));
        }
        let system = string_field(&parsed, "time_system")?;
        if system != "TDT" {
            return Err(value_error(format!("Unsupported time scale: {system}")));
        }
        let elements = field(&parsed, "elements")?;
        coords.push([
            number_field(elements, "a")?,
            number_field(elements, "e")?,
            number_field(elements, "i")?,
            number_field(elements, "node")?,
            number_field(elements, "peri")?,
            number_field(elements, "M")?,
        ]);
        let matrix = number_array(&parsed, "covariance")?;
        if matrix.len() != 36 {
            return Err(value_error("NEOCC covariance must contain 36 values"));
        }
        covariance.extend(matrix);
        epochs.push(number_field(&parsed, "epoch")?);
        let id = string_field(&parsed, "object_id")?;
        orbit_ids.push(id);
        let magnitude = parsed.get("magnitude");
        h_values.push(
            magnitude
                .and_then(|value| value.get("H"))
                .and_then(Value::as_f64),
        );
        g_values.push(
            magnitude
                .and_then(|value| value.get("G"))
                .and_then(Value::as_f64),
        );
    }
    let rows = coords.len();
    let time_array =
        TimeArray::from_mjd(TimeScale::Tt, &epochs).map_err(|err| value_error(err.to_string()))?;
    let rounded_epochs = mjd_values(&time_array);
    let flat: Vec<f64> = coords.iter().flatten().copied().collect();
    let mu = vec![MU_SUN; rows];
    let (cartesian, covariance) = transform_with_covariance_flat6(
        &flat,
        &covariance,
        Representation::Keplerian,
        Representation::Cartesian,
        adam_core_rs_coords::Frame::Ecliptic,
        adam_core_rs_coords::Frame::Ecliptic,
        &rounded_epochs,
        &mu,
        &mu,
        0.0,
        0.0,
        100,
        1e-15,
        None,
    );
    let cartesian = cartesian
        .chunks_exact(6)
        .map(|row| [row[0], row[1], row[2], row[3], row[4], row[5]])
        .collect();
    let coordinates = CoordinateBatch::cartesian(
        cartesian,
        DataFrame::Ecliptic,
        OriginArray::repeat(OriginId::from_code("SUN"), rows),
        Some(time_array),
        Some(
            CovarianceBatch::new(
                rows,
                6,
                covariance,
                CovarianceUnits::Coordinate(CoordinateRepresentation::Cartesian),
            )
            .map_err(|err| value_error(err.to_string()))?,
        ),
    )
    .map_err(|err| value_error(err.to_string()))?;
    let physical = PhysicalParametersBatch {
        h_v: h_values,
        h_v_sigma: vec![Some(f64::NAN); rows],
        g: g_values,
        g_sigma: vec![Some(f64::NAN); rows],
        sigma_eff: vec![None; rows],
        chi2_red: vec![None; rows],
    };
    OrbitBatch::new(
        orbit_ids.iter().cloned().map(OrbitId).collect(),
        orbit_ids.into_iter().map(|id| Some(ObjectId(id))).collect(),
        coordinates,
    )
    .and_then(|orbits| orbits.with_physical_parameters(physical))
    .map_err(|err| value_error(err.to_string()))
}

fn scout_summary_batch(payload: &str) -> PyResult<RecordBatch> {
    let root = parse_json(payload, "Scout summary")?;
    let rows = field(&root, "data")?
        .as_array()
        .ok_or_else(|| value_error("Scout data must be an array"))?;
    let string = |name: &str, nullable: bool| -> PyResult<ArrayRef> {
        let values: PyResult<Vec<Option<String>>> = rows
            .iter()
            .map(|row| match row.get(name) {
                Some(Value::String(value)) => Ok(Some(value.clone())),
                Some(Value::Null) if nullable => Ok(None),
                _ => Err(value_error(format!("Scout field {name} must be a string"))),
            })
            .collect();
        Ok(Arc::new(LargeStringArray::from(values?)))
    };
    let int64 = |name: &str| -> PyResult<ArrayRef> {
        let values: PyResult<Vec<i64>> = rows
            .iter()
            .map(|row| {
                row.get(name)
                    .and_then(Value::as_i64)
                    .ok_or_else(|| value_error(format!("Scout field {name} must be int64")))
            })
            .collect();
        Ok(Arc::new(Int64Array::from(values?)))
    };
    let fields = [
        ("unc", DataType::LargeUtf8, false),
        ("lastRun", DataType::LargeUtf8, false),
        ("dec", DataType::LargeUtf8, false),
        ("H", DataType::LargeUtf8, false),
        ("moid", DataType::LargeUtf8, false),
        ("geocentricScore", DataType::Int64, false),
        ("ra", DataType::LargeUtf8, false),
        ("rating", DataType::Int8, true),
        ("tisserandScore", DataType::Int64, false),
        ("uncP1", DataType::LargeUtf8, false),
        ("ieoScore", DataType::Int64, false),
        ("rate", DataType::LargeUtf8, false),
        ("rmsN", DataType::LargeUtf8, false),
        ("Vmag", DataType::LargeUtf8, false),
        ("neoScore", DataType::Int64, false),
        ("nObs", DataType::Int64, false),
        ("objectName", DataType::LargeUtf8, false),
        ("phaScore", DataType::Int64, false),
        ("tEphem", DataType::LargeUtf8, false),
        ("arc", DataType::LargeUtf8, false),
        ("caDist", DataType::LargeUtf8, true),
        ("elong", DataType::LargeUtf8, false),
        ("vInf", DataType::LargeUtf8, true),
        ("neo1kmScore", DataType::Int64, false),
    ];
    let mut arrays = Vec::with_capacity(fields.len());
    for (name, data_type, nullable) in &fields {
        let array = match data_type {
            DataType::LargeUtf8 => string(name, *nullable)?,
            DataType::Int64 => int64(name)?,
            DataType::Int8 => {
                let values: PyResult<Vec<Option<i8>>> = rows
                    .iter()
                    .map(|row| match row.get(*name) {
                        Some(Value::Null) => Ok(None),
                        Some(value) => value
                            .as_i64()
                            .map(|value| Some(value as i8))
                            .ok_or_else(|| value_error(format!("Scout field {name} must be int8"))),
                        None => Ok(None),
                    })
                    .collect();
                Arc::new(Int8Array::from(values?))
            }
            _ => unreachable!(),
        };
        arrays.push(array);
    }
    RecordBatch::try_new(
        Arc::new(Schema::new(
            fields
                .into_iter()
                .map(|(name, data_type, nullable)| Field::new(name, data_type, nullable))
                .collect::<Vec<_>>(),
        )),
        arrays,
    )
    .map_err(|err| value_error(err.to_string()))
}

fn scout_rows(payload: &str) -> PyResult<Vec<Map<String, Value>>> {
    let root = parse_json(payload, "Scout orbits")?;
    let orbits = field(&root, "orbits")?;
    let fields = field(orbits, "fields")?
        .as_array()
        .ok_or_else(|| value_error("Scout orbit fields must be an array"))?;
    let names: PyResult<Vec<&str>> = fields
        .iter()
        .map(|name| {
            name.as_str()
                .ok_or_else(|| value_error("Scout orbit field names must be strings"))
        })
        .collect();
    let names = names?;
    field(orbits, "data")?
        .as_array()
        .ok_or_else(|| value_error("Scout orbit data must be an array"))?
        .iter()
        .map(|row| {
            let row = row
                .as_array()
                .ok_or_else(|| value_error("Scout orbit rows must be arrays"))?;
            if row.len() != names.len() {
                return Err(value_error("Scout orbit row length does not match fields"));
            }
            Ok(names
                .iter()
                .zip(row)
                .map(|(name, value)| ((*name).to_string(), value.clone()))
                .collect())
        })
        .collect()
}

fn scout_variants(
    object_ids: &[String],
    timeout_s: f64,
    max_attempts: usize,
    retry_delay_s: f64,
    recorded: Option<&[String]>,
) -> PyResult<OrbitVariantBatch> {
    let mut cursor = 0usize;
    let mut orbit_ids = Vec::new();
    let mut variant_ids = Vec::new();
    let mut output_object_ids = Vec::new();
    let mut states = Vec::new();
    let mut epochs_jd = Vec::new();
    for object_id in object_ids {
        let payload = scout_object_payload(
            object_id,
            "orbits=1",
            timeout_s,
            max_attempts,
            retry_delay_s,
            recorded,
            &mut cursor,
        )?;
        require_scout_signature(&payload, object_id)?;
        let rows_value = payload.get("orbits").and_then(|value| value.get("data"));
        if !rows_value.is_some_and(Value::is_array) {
            return Err(scout_error(
                "response",
                object_id,
                None,
                format!("Scout response for {object_id:?} is missing orbits.data"),
            ));
        }
        if rows_value
            .and_then(Value::as_array)
            .is_some_and(Vec::is_empty)
        {
            return Err(scout_error(
                "not_found",
                object_id,
                None,
                format!("Scout returned no orbit samples for {object_id:?}"),
            ));
        }
        let payload_text = payload.to_string();
        let rows = scout_rows(&payload_text).map_err(|error| {
            scout_error(
                "response",
                object_id,
                None,
                format!("Scout returned invalid orbit samples for {object_id:?}: {error}"),
            )
        })?;
        let normalized = adam_core_rs_coords::scout_normalize_orbits_json(
            object_id,
            &serde_json::to_string(&rows).map_err(|err| value_error(err.to_string()))?,
        )
        .map_err(|error| {
            scout_error(
                "response",
                object_id,
                None,
                format!("Scout returned invalid orbit samples for {object_id:?}: {error}"),
            )
        })?;
        let normalized = parse_json(&normalized, "normalized Scout")?;
        let coords = matrix_rows(&normalized, "coords_cometary")?;
        let times = number_array(&normalized, "times_jd")?;
        let ids = string_array(&normalized, "orbit_id")?;
        let variants = string_array(&normalized, "variant_id")?;
        let objects = string_array(&normalized, "object_id")?;
        let time_array = TimeArray::from_mjd(
            TimeScale::Tai,
            &times
                .iter()
                .map(|value| value - JD_MINUS_MJD)
                .collect::<Vec<_>>(),
        )
        .map_err(|err| value_error(err.to_string()))?;
        let rounded = mjd_values(
            &time_array
                .rescale(TimeScale::Tdb)
                .map_err(|err| value_error(err.to_string()))?,
        );
        for (row, cometary) in coords.iter().enumerate() {
            states.push(cometary_to_cartesian6::<f64>(
                cometary,
                rounded[row],
                MU_SUN,
                100,
                1e-15,
            ));
        }
        epochs_jd.extend(times);
        orbit_ids.extend(ids);
        variant_ids.extend(variants);
        output_object_ids.extend(objects);
    }
    let rows = states.len();
    let coordinates = CoordinateBatch::cartesian(
        states,
        DataFrame::Ecliptic,
        OriginArray::repeat(OriginId::from_code("SUN"), rows),
        Some(
            TimeArray::from_mjd(
                TimeScale::Tai,
                &epochs_jd
                    .iter()
                    .map(|value| value - JD_MINUS_MJD)
                    .collect::<Vec<_>>(),
            )
            .map_err(|err| value_error(err.to_string()))?,
        ),
        None,
    )
    .map_err(|err| value_error(err.to_string()))?;
    OrbitVariantBatch::new(
        orbit_ids.into_iter().map(OrbitId).collect(),
        output_object_ids
            .into_iter()
            .map(|id| Some(ObjectId(id)))
            .collect(),
        variant_ids
            .into_iter()
            .map(|id| Some(VariantId(id)))
            .collect(),
        vec![None; rows],
        vec![None; rows],
        coordinates,
    )
    .map_err(|err| value_error(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (object_ids, orbit_type="ke", orbit_epoch="present-day", recorded_responses=None))]
fn query_neocc_arrow<'py>(
    py: Python<'py>,
    object_ids: Vec<String>,
    orbit_type: &str,
    orbit_epoch: &str,
    recorded_responses: Option<Vec<String>>,
) -> PyResult<PyObject> {
    let output = py.allow_threads(|| {
        neocc_orbits(
            &object_ids,
            orbit_type,
            orbit_epoch,
            recorded_responses.as_deref(),
        )
    })?;
    output
        .into_nested_record_batch()
        .map_err(|err| value_error(err.to_string()))?
        .to_pyarrow(py)
        .map_err(|err| value_error(err.to_string()))
}

#[derive(Debug)]
struct ScoutObservationRow {
    object_id: String,
    solution_date_utc: Option<String>,
    declared_n_obs: Option<i64>,
    snapshot_sha256: String,
    snapshot_observation_count: i64,
    signature_version: Option<String>,
    signature_source: Option<String>,
    observation_index: i64,
    observation: adam_core_rs_coords::OpticalObs80Record,
}

#[allow(clippy::too_many_arguments)]
fn scout_observations_batch(
    object_ids: &[String],
    timeout_s: f64,
    max_attempts: usize,
    retry_delay_s: f64,
    recorded: Option<&[String]>,
) -> PyResult<RecordBatch> {
    let mut cursor = 0usize;
    let mut rows = Vec::new();
    for object_id in object_ids {
        let payload = scout_object_payload(
            object_id,
            "file=mpc",
            timeout_s,
            max_attempts,
            retry_delay_s,
            recorded,
            &mut cursor,
        )?;
        let signature = require_scout_signature(&payload, object_id)?;
        let raw_file = payload
            .get("fileMPC")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| {
                scout_error(
                    "response",
                    object_id,
                    None,
                    format!("Scout returned no file=mpc snapshot for {object_id}"),
                )
            })?;
        let observations = parse_optical_obs80_file(raw_file, true).map_err(|error| {
            scout_error(
                "response",
                object_id,
                None,
                format!("Scout returned an invalid file=mpc snapshot for {object_id}: {error}"),
            )
        })?;
        if observations.is_empty() {
            return Err(scout_error(
                "response",
                object_id,
                None,
                format!("Scout returned an empty file=mpc snapshot for {object_id}"),
            ));
        }
        if observations
            .iter()
            .any(|observation| observation.designation != *object_id)
        {
            return Err(scout_error(
                "response",
                object_id,
                None,
                format!("Scout file=mpc designation mismatch for requested object {object_id}"),
            ));
        }
        let declared_n_obs = payload.get("nObs").and_then(|value| {
            value
                .as_i64()
                .or_else(|| value.as_str().and_then(|value| value.parse::<i64>().ok()))
        });
        let hash = format!("{:x}", Sha256::digest(raw_file.as_bytes()));
        let count = observations.len() as i64;
        let solution_date_utc = payload
            .get("lastRun")
            .and_then(Value::as_str)
            .map(str::to_string);
        let signature_version = signature
            .get("version")
            .and_then(Value::as_str)
            .map(str::to_string);
        let signature_source = signature
            .get("source")
            .and_then(Value::as_str)
            .map(str::to_string);
        for (index, observation) in observations.into_iter().enumerate() {
            rows.push(ScoutObservationRow {
                object_id: object_id.clone(),
                solution_date_utc: solution_date_utc.clone(),
                declared_n_obs,
                snapshot_sha256: hash.clone(),
                snapshot_observation_count: count,
                signature_version: signature_version.clone(),
                signature_source: signature_source.clone(),
                observation_index: index as i64,
                observation,
            });
        }
    }

    let observation_batch = optical_obs80_record_batch(
        &rows
            .iter()
            .map(|row| row.observation.clone())
            .collect::<Vec<_>>(),
    )
    .map_err(|error| value_error(error.to_string()))?;
    let observation = StructArray::try_new(
        observation_batch.schema().fields().clone(),
        observation_batch.columns().to_vec(),
        None,
    )
    .map_err(|error| value_error(error.to_string()))?;
    let fields = vec![
        Field::new("object_id", DataType::LargeUtf8, false),
        Field::new("solution_date_utc", DataType::LargeUtf8, true),
        Field::new("declared_n_obs", DataType::Int64, true),
        Field::new("snapshot_sha256", DataType::LargeUtf8, false),
        Field::new("snapshot_observation_count", DataType::Int64, false),
        Field::new("signature_version", DataType::LargeUtf8, true),
        Field::new("signature_source", DataType::LargeUtf8, true),
        Field::new("observation_index", DataType::Int64, false),
        Field::new("observation", observation.data_type().clone(), true),
    ];
    let strings =
        |values: Vec<&str>| -> ArrayRef { Arc::new(LargeStringArray::from_iter_values(values)) };
    let optional_strings =
        |values: Vec<Option<&str>>| -> ArrayRef { Arc::new(LargeStringArray::from_iter(values)) };
    let arrays: Vec<ArrayRef> = vec![
        strings(rows.iter().map(|row| row.object_id.as_str()).collect()),
        optional_strings(
            rows.iter()
                .map(|row| row.solution_date_utc.as_deref())
                .collect(),
        ),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.declared_n_obs)
                .collect::<Vec<_>>(),
        )),
        strings(
            rows.iter()
                .map(|row| row.snapshot_sha256.as_str())
                .collect(),
        ),
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|row| row.snapshot_observation_count),
        )),
        optional_strings(
            rows.iter()
                .map(|row| row.signature_version.as_deref())
                .collect(),
        ),
        optional_strings(
            rows.iter()
                .map(|row| row.signature_source.as_deref())
                .collect(),
        ),
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|row| row.observation_index),
        )),
        Arc::new(observation),
    ];
    let mut metadata = HashMap::new();
    metadata.insert(
        "adam_core_schema".to_string(),
        "ScoutObservations.nested.quivr.v1".to_string(),
    );
    metadata.insert("adam_core_schema_version".to_string(), "1".to_string());
    metadata.insert("observation.time.scale".to_string(), "utc".to_string());
    RecordBatch::try_new(
        Arc::new(Schema::new_with_metadata(fields, metadata)),
        arrays,
    )
    .map_err(|error| value_error(error.to_string()))
}

#[pyfunction]
#[pyo3(signature = (object_ids, recorded_responses=None, timeout_s=30.0, max_attempts=3, retry_delay_s=0.5))]
fn query_scout_observations_arrow<'py>(
    py: Python<'py>,
    object_ids: Vec<String>,
    recorded_responses: Option<Vec<String>>,
    timeout_s: f64,
    max_attempts: usize,
    retry_delay_s: f64,
) -> PyResult<PyObject> {
    let batch = py.allow_threads(|| {
        scout_observations_batch(
            &object_ids,
            timeout_s,
            max_attempts,
            retry_delay_s,
            recorded_responses.as_deref(),
        )
    })?;
    batch
        .to_pyarrow(py)
        .map_err(|error| value_error(error.to_string()))
}

#[pyfunction]
#[pyo3(signature = (recorded_response=None, timeout_s=30.0, max_attempts=3, retry_delay_s=0.5))]
fn get_scout_objects_arrow<'py>(
    py: Python<'py>,
    recorded_response: Option<String>,
    timeout_s: f64,
    max_attempts: usize,
    retry_delay_s: f64,
) -> PyResult<PyObject> {
    let payload = match recorded_response {
        Some(payload) => payload,
        None => {
            let timeout = Duration::try_from_secs_f64(timeout_s)
                .map_err(|error| value_error(format!("invalid Scout timeout: {error}")))?;
            let attempts = max_attempts.max(1);
            let mut last_error = None;
            let mut result = None;
            for attempt in 0..attempts {
                match get_text_with_status(SCOUT_URL, timeout, "Scout") {
                    Ok(payload) => {
                        result = Some(payload);
                        break;
                    }
                    Err(error) => {
                        let retryable = error
                            .status
                            .is_none_or(|status| status == 429 || status >= 500);
                        last_error = Some(error.to_string());
                        if !retryable || attempt + 1 == attempts {
                            break;
                        }
                        std::thread::sleep(
                            Duration::try_from_secs_f64(
                                retry_delay_s * 2.0_f64.powi(attempt as i32),
                            )
                            .map_err(|error| {
                                value_error(format!("invalid Scout retry delay: {error}"))
                            })?,
                        );
                    }
                }
            }
            result.ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "Scout query failed after {attempts} attempts: {}",
                    last_error.unwrap_or_else(|| "unknown error".to_string())
                ))
            })?
        }
    };
    scout_summary_batch(&payload)?
        .to_pyarrow(py)
        .map_err(|err| value_error(err.to_string()))
}

#[pyfunction]
fn scout_orbits_to_variants_arrow<'py>(
    py: Python<'py>,
    object_id: &str,
    scout_batch: &Bound<'py, PyAny>,
) -> PyResult<PyObject> {
    let batch = RecordBatch::from_pyarrow_bound(scout_batch)
        .map_err(|err| value_error(format!("invalid ScoutOrbit batch: {err}")))?;
    let names: Vec<String> = batch
        .schema()
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
    let mut rows = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let mut values = Vec::with_capacity(batch.num_columns());
        for column in batch.columns() {
            if column.is_null(row) {
                values.push(Value::Null);
            } else if let Some(strings) = column.as_any().downcast_ref::<LargeStringArray>() {
                values.push(Value::String(strings.value(row).to_string()));
            } else if let Some(integers) = column.as_any().downcast_ref::<Int64Array>() {
                values.push(Value::from(integers.value(row)));
            } else {
                return Err(value_error("unsupported ScoutOrbit Arrow column type"));
            }
        }
        rows.push(Value::Array(values));
    }
    let payload = serde_json::json!({
        "signature": {"version": "1.3", "source": "recorded"},
        "orbits": {"fields": names, "data": rows}
    })
    .to_string();
    let output = scout_variants(&[object_id.to_string()], 30.0, 1, 0.0, Some(&[payload]))?;
    output
        .into_nested_record_batch()
        .map_err(|err| value_error(err.to_string()))?
        .to_pyarrow(py)
        .map_err(|err| value_error(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (object_ids, recorded_responses=None, timeout_s=30.0, max_attempts=3, retry_delay_s=0.5))]
fn query_scout_arrow<'py>(
    py: Python<'py>,
    object_ids: Vec<String>,
    recorded_responses: Option<Vec<String>>,
    timeout_s: f64,
    max_attempts: usize,
    retry_delay_s: f64,
) -> PyResult<PyObject> {
    let output = py.allow_threads(|| {
        scout_variants(
            &object_ids,
            timeout_s,
            max_attempts,
            retry_delay_s,
            recorded_responses.as_deref(),
        )
    })?;
    output
        .into_nested_record_batch()
        .map_err(|err| value_error(err.to_string()))?
        .to_pyarrow(py)
        .map_err(|err| value_error(err.to_string()))
}

fn optional_number_array(value: &Value, name: &str) -> PyResult<Vec<Option<f64>>> {
    field(value, name)?
        .as_array()
        .ok_or_else(|| value_error(format!("field {name:?} must be an array")))?
        .iter()
        .map(|entry| match entry {
            Value::Null => Ok(None),
            _ => entry
                .as_f64()
                .map(Some)
                .ok_or_else(|| value_error(format!("{name} entries must be numeric or null"))),
        })
        .collect()
}

fn sbdb_payload(
    object_id: &str,
    physical_parameters: bool,
    timeout_s: f64,
    max_attempts: usize,
) -> PyResult<String> {
    if object_id.trim().is_empty() {
        return Err(value_error("object_id must be non-empty"));
    }
    if timeout_s <= 0.0 {
        return Err(value_error("timeout_s must be > 0"));
    }
    if max_attempts == 0 {
        return Err(value_error("max_attempts must be > 0"));
    }
    let url = format!(
        "{SBDB_URL}?sstr={}&cov=mat&full-prec=true&phys-par={}",
        percent_encode(object_id.trim()),
        if physical_parameters { "true" } else { "false" }
    );
    let timeout = Duration::from_secs_f64(timeout_s);
    let mut last_error = None;
    for attempt in 0..max_attempts {
        match get_text(&url, timeout, "SBDB") {
            Ok(payload) => return Ok(payload),
            Err(error) => last_error = Some(error),
        }
        if attempt + 1 < max_attempts {
            std::thread::sleep(Duration::from_secs_f64(
                8.0_f64.min(0.5 * 2.0_f64.powi(attempt as i32)),
            ));
        }
    }
    Err(PyRuntimeError::new_err(format!(
        "SBDB query failed after {max_attempts} attempts: {}",
        last_error
            .map(|error| error.to_string())
            .unwrap_or_else(|| "unknown error".to_string())
    )))
}

fn sbdb_orbits(
    ids: &[String],
    physical_parameters: bool,
    timeout_s: f64,
    max_attempts: usize,
    allow_missing: bool,
    orbit_id_from_input: bool,
    recorded: Option<&[String]>,
) -> PyResult<OrbitBatch> {
    let mut cursor = 0usize;
    let mut kept_ids = Vec::new();
    let mut payloads = Vec::new();
    for id in ids {
        let text = if let Some(recorded) = recorded {
            let payload = recorded
                .get(cursor)
                .ok_or_else(|| value_error("not enough recorded SBDB responses"))?
                .clone();
            cursor += 1;
            payload
        } else {
            sbdb_payload(id, physical_parameters, timeout_s, max_attempts)?
        };
        let payload = parse_json(&text, "SBDB response")?;
        if payload.get("object").is_none() {
            if allow_missing {
                continue;
            }
            return Err(value_error(format!("__NOT_FOUND__:{id}")));
        }
        kept_ids.push(id.clone());
        payloads.push(text);
    }
    let ids_json = serde_json::to_string(&kept_ids).map_err(|err| value_error(err.to_string()))?;
    let payloads_json = format!("[{}]", payloads.join(","));
    let normalized = adam_core_rs_coords::sbdb_normalize_payloads_json(&ids_json, &payloads_json)
        .map_err(|err| value_error(err.to_string()))?;
    let normalized = parse_json(&normalized, "normalized SBDB")?;
    let mut orbit_ids = string_array(&normalized, "orbit_id")?;
    if orbit_id_from_input {
        orbit_ids.clone_from(&kept_ids);
    }
    let object_ids = string_array(&normalized, "object_id")?;
    let cometary = matrix_rows(&normalized, "coords_cometary")?;
    let times_jd = number_array(&normalized, "times_jd")?;
    let covariance_rows = field(&normalized, "covariances_cometary")?
        .as_array()
        .ok_or_else(|| value_error("covariances_cometary must be an array"))?;
    let mut covariance = Vec::with_capacity(covariance_rows.len() * 36);
    for row in covariance_rows {
        let values = row
            .as_array()
            .ok_or_else(|| value_error("SBDB covariance rows must be arrays"))?;
        if values.len() != 36 {
            return Err(value_error("SBDB covariance rows must contain 36 values"));
        }
        for value in values {
            covariance.push(value.as_f64().unwrap_or(f64::NAN));
        }
    }
    let rows = cometary.len();
    let flat: Vec<f64> = cometary.iter().flatten().copied().collect();
    let epochs_mjd: Vec<f64> = times_jd.iter().map(|value| value - JD_MINUS_MJD).collect();
    let time_array = TimeArray::from_mjd(TimeScale::Tdb, &epochs_mjd)
        .map_err(|err| value_error(err.to_string()))?;
    let mu = vec![MU_SUN; rows];
    let (cartesian, covariance) = transform_with_covariance_flat6(
        &flat,
        &covariance,
        Representation::Cometary,
        Representation::Cartesian,
        adam_core_rs_coords::Frame::Ecliptic,
        adam_core_rs_coords::Frame::Ecliptic,
        &epochs_mjd,
        &mu,
        &mu,
        0.0,
        0.0,
        100,
        1e-15,
        None,
    );
    let states = cartesian
        .chunks_exact(6)
        .map(|row| [row[0], row[1], row[2], row[3], row[4], row[5]])
        .collect();
    let coordinates = CoordinateBatch::cartesian(
        states,
        DataFrame::Ecliptic,
        OriginArray::repeat(OriginId::from_code("SUN"), rows),
        Some(time_array),
        Some(
            CovarianceBatch::new(
                rows,
                6,
                covariance,
                CovarianceUnits::Coordinate(CoordinateRepresentation::Cartesian),
            )
            .map_err(|err| value_error(err.to_string()))?,
        ),
    )
    .map_err(|err| value_error(err.to_string()))?;
    let physical = field(&normalized, "physical_parameters")?;
    let nan_values = |name: &str| -> PyResult<Vec<Option<f64>>> {
        Ok(optional_number_array(physical, name)?
            .into_iter()
            .map(|value| Some(value.unwrap_or(f64::NAN)))
            .collect())
    };
    let physical = PhysicalParametersBatch {
        h_v: nan_values("H_v")?,
        h_v_sigma: nan_values("H_v_sigma")?,
        g: nan_values("G")?,
        g_sigma: nan_values("G_sigma")?,
        sigma_eff: vec![None; rows],
        chi2_red: vec![None; rows],
    };
    OrbitBatch::new(
        orbit_ids.into_iter().map(OrbitId).collect(),
        object_ids
            .into_iter()
            .map(|id| Some(ObjectId(id)))
            .collect(),
        coordinates,
    )
    .and_then(|orbits| orbits.with_physical_parameters(physical))
    .map_err(|err| value_error(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (ids, physical_parameters, timeout_s, max_attempts, allow_missing, orbit_id_from_input, recorded_responses=None))]
#[allow(clippy::too_many_arguments)]
fn query_sbdb_arrow<'py>(
    py: Python<'py>,
    ids: Vec<String>,
    physical_parameters: bool,
    timeout_s: f64,
    max_attempts: usize,
    allow_missing: bool,
    orbit_id_from_input: bool,
    recorded_responses: Option<Vec<String>>,
) -> PyResult<PyObject> {
    let output = py.allow_threads(|| {
        sbdb_orbits(
            &ids,
            physical_parameters,
            timeout_s,
            max_attempts,
            allow_missing,
            orbit_id_from_input,
            recorded_responses.as_deref(),
        )
    })?;
    output
        .into_nested_record_batch()
        .map_err(|err| value_error(err.to_string()))?
        .to_pyarrow(py)
        .map_err(|err| value_error(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (kind, payloads, reps, trials, warmup_reps=1))]
fn benchmark_query_client_processing(
    kind: &str,
    payloads: Vec<String>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    if reps == 0 || trials == 0 {
        return Err(value_error("reps and trials must be >= 1"));
    }
    let run = || -> PyResult<()> {
        match kind {
            "neocc" => {
                black_box(neocc_orbits(
                    &vec!["recorded".to_string(); payloads.len()],
                    "ke",
                    "present-day",
                    Some(&payloads),
                )?);
            }
            "scout" => {
                black_box(scout_variants(
                    &vec!["recorded".to_string(); payloads.len()],
                    30.0,
                    1,
                    0.0,
                    Some(&payloads),
                )?);
            }
            "scout-summary" => {
                for payload in &payloads {
                    black_box(scout_summary_batch(payload)?);
                }
            }
            "scout-observations" => {
                let object_ids = payloads
                    .iter()
                    .map(|payload| -> PyResult<String> {
                        let value = parse_json(payload, "recorded Scout observations")?;
                        let raw_file = value
                            .get("fileMPC")
                            .and_then(Value::as_str)
                            .ok_or_else(|| value_error("recorded Scout payload missing fileMPC"))?;
                        parse_optical_obs80_file(raw_file, true)
                            .map_err(|error| value_error(error.to_string()))?
                            .first()
                            .map(|record| record.designation.clone())
                            .ok_or_else(|| value_error("recorded Scout fileMPC is empty"))
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                black_box(scout_observations_batch(
                    &object_ids,
                    30.0,
                    1,
                    0.0,
                    Some(&payloads),
                )?);
            }
            "sbdb" => {
                black_box(sbdb_orbits(
                    &vec!["recorded".to_string(); payloads.len()],
                    true,
                    60.0,
                    1,
                    false,
                    false,
                    Some(&payloads),
                )?);
            }
            other => return Err(value_error(format!("unknown query kind: {other}"))),
        }
        Ok(())
    };
    let mut output = Vec::with_capacity(trials);
    for _ in 0..trials {
        for _ in 0..warmup_reps {
            run()?;
        }
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let started = Instant::now();
            run()?;
            samples.push(started.elapsed().as_secs_f64());
        }
        output.push(samples);
    }
    Ok(output)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(query_neocc_arrow, module)?)?;
    module.add_function(wrap_pyfunction!(get_scout_objects_arrow, module)?)?;
    module.add_function(wrap_pyfunction!(scout_orbits_to_variants_arrow, module)?)?;
    module.add_function(wrap_pyfunction!(query_scout_arrow, module)?)?;
    module.add_function(wrap_pyfunction!(query_scout_observations_arrow, module)?)?;
    module.add_function(wrap_pyfunction!(query_sbdb_arrow, module)?)?;
    module.add_function(wrap_pyfunction!(benchmark_query_client_processing, module)?)?;
    Ok(())
}
