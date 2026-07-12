use crate::http::{get_text, percent_encode};
use adam_core_rs_coords::{
    cometary_to_cartesian6, transform_with_covariance_flat6, CoordinateBatch,
    CoordinateRepresentation, CovarianceBatch, CovarianceUnits, DataFrame, IntoNestedRecordBatch,
    ObjectId, OrbitBatch, OrbitId, OrbitVariantBatch, OriginArray, OriginId,
    PhysicalParametersBatch, Representation, TimeArray, TimeScale, VariantId,
};
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use arrow_array::{Array, ArrayRef, Int64Array, Int8Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use serde_json::{Map, Value};
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
    recorded: Option<&[String]>,
) -> PyResult<OrbitVariantBatch> {
    let mut cursor = 0usize;
    let mut orbit_ids = Vec::new();
    let mut variant_ids = Vec::new();
    let mut output_object_ids = Vec::new();
    let mut states = Vec::new();
    let mut epochs_jd = Vec::new();
    for object_id in object_ids {
        let url = format!("{SCOUT_URL}?tdes={}&orbits=1", percent_encode(object_id));
        let payload = next_recorded(recorded, &mut cursor, &url, "Scout")?;
        let rows = scout_rows(&payload)?;
        let normalized = adam_core_rs_coords::scout_normalize_orbits_json(
            object_id,
            &serde_json::to_string(&rows).map_err(|err| value_error(err.to_string()))?,
        )
        .map_err(|err| value_error(err.to_string()))?;
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
        let rounded = mjd_values(&time_array);
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

#[pyfunction]
#[pyo3(signature = (recorded_response=None))]
fn get_scout_objects_arrow<'py>(
    py: Python<'py>,
    recorded_response: Option<String>,
) -> PyResult<PyObject> {
    let payload = match recorded_response {
        Some(payload) => payload,
        None => get_text(SCOUT_URL, Duration::from_secs(60), "Scout")?,
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
        "orbits": {"fields": names, "data": rows}
    })
    .to_string();
    let output = scout_variants(&[object_id.to_string()], Some(&[payload]))?;
    output
        .into_nested_record_batch()
        .map_err(|err| value_error(err.to_string()))?
        .to_pyarrow(py)
        .map_err(|err| value_error(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (object_ids, recorded_responses=None))]
fn query_scout_arrow<'py>(
    py: Python<'py>,
    object_ids: Vec<String>,
    recorded_responses: Option<Vec<String>>,
) -> PyResult<PyObject> {
    let output = py.allow_threads(|| scout_variants(&object_ids, recorded_responses.as_deref()))?;
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
                    Some(&payloads),
                )?);
            }
            "scout-summary" => {
                for payload in &payloads {
                    black_box(scout_summary_batch(payload)?);
                }
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
    module.add_function(wrap_pyfunction!(query_sbdb_arrow, module)?)?;
    module.add_function(wrap_pyfunction!(benchmark_query_client_processing, module)?)?;
    Ok(())
}
