//! Typed Rust access to the JPL Small-Body Database (SBDB) orbit product.
//!
//! The live and recorded-response entrypoints share the same deterministic
//! response-to-[`OrbitBatch`] assembly. Recorded responses provide a hermetic
//! seam for semantic tests and downstream consumers that archive provider
//! responses.

use crate::coords::{
    transform_with_covariance_flat, CoordinateBatch, CoordinateRepresentation, CovarianceBatch,
    CovarianceUnits, DataFrame, Frame, NonGravitationalParametersBatch, NongravCovarianceAxes,
    ObjectId, OrbitBatch, OrbitId, OriginArray, OriginId, PhysicalParametersBatch, Representation,
    TimeArray, TimeScale, Validity,
};
use serde_json::Value;
use std::fmt;
use std::time::Duration;

const MU_SUN: f64 = 0.000_295_912_208_284_119_56;
const JD_MINUS_MJD: f64 = 2_400_000.5;
const SBDB_URL: &str = "https://ssd-api.jpl.nasa.gov/sbdb.api";

/// Explicit behavior controls for one ordered SBDB query.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SbdbQueryOptions {
    /// Ask SBDB to include physical parameters in each response.
    pub physical_parameters: bool,
    /// Per-request HTTP timeout, in seconds.
    pub timeout_s: f64,
    /// Maximum HTTP attempts for each requested object.
    pub max_attempts: usize,
    /// Omit provider responses that do not contain an `object` record.
    pub allow_missing: bool,
    /// Use retained input identifiers as orbit IDs instead of generated row IDs.
    pub orbit_id_from_input: bool,
}

/// Complete canonical product returned by an SBDB query.
#[derive(Debug, Clone, PartialEq)]
pub struct SbdbQueryResult {
    pub orbits: OrbitBatch,
    pub warnings: Vec<String>,
}

/// Typed failure from validation, HTTP transport, response decoding, or a
/// provider-level missing-object response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SbdbError {
    InvalidInput(String),
    Request(String),
    InvalidResponse(String),
    NotFound { object_id: String },
}

impl SbdbError {
    /// Whether this failure belongs to the live HTTP request/retry boundary.
    pub fn is_request(&self) -> bool {
        matches!(self, Self::Request(_))
    }
}

impl fmt::Display for SbdbError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message)
            | Self::Request(message)
            | Self::InvalidResponse(message) => formatter.write_str(message),
            Self::NotFound { object_id } => write!(formatter, "__NOT_FOUND__:{object_id}"),
        }
    }
}

impl std::error::Error for SbdbError {}

/// Query SBDB over HTTP, preserving input order in the returned product.
pub fn query_sbdb(
    object_ids: &[String],
    options: SbdbQueryOptions,
) -> Result<SbdbQueryResult, SbdbError> {
    query_sbdb_with_source(object_ids, options, ResponseSource::Live)
}

/// Build the complete SBDB product from ordered recorded JSON responses.
///
/// Exactly one response is consumed for each input identifier. Extra responses
/// are ignored, matching the live/Python binding contract; too few responses
/// fail explicitly.
pub fn query_sbdb_recorded(
    object_ids: &[String],
    options: SbdbQueryOptions,
    recorded_responses: &[String],
) -> Result<SbdbQueryResult, SbdbError> {
    query_sbdb_with_source(
        object_ids,
        options,
        ResponseSource::Recorded(recorded_responses),
    )
}

enum ResponseSource<'a> {
    Live,
    Recorded(&'a [String]),
}

fn query_sbdb_with_source(
    object_ids: &[String],
    options: SbdbQueryOptions,
    source: ResponseSource<'_>,
) -> Result<SbdbQueryResult, SbdbError> {
    let mut cursor = 0usize;
    let mut kept_ids = Vec::new();
    let mut payloads = Vec::new();
    for id in object_ids {
        let text = match source {
            ResponseSource::Live => sbdb_payload(id, options)?,
            ResponseSource::Recorded(recorded) => {
                let payload = recorded.get(cursor).ok_or_else(|| {
                    SbdbError::InvalidInput("not enough recorded SBDB responses".to_string())
                })?;
                cursor += 1;
                payload.clone()
            }
        };
        let payload = parse_json(&text, "SBDB response")?;
        if payload.get("object").is_none() {
            if options.allow_missing {
                continue;
            }
            return Err(SbdbError::NotFound {
                object_id: id.clone(),
            });
        }
        kept_ids.push(id.clone());
        payloads.push(text);
    }
    assemble_product(&kept_ids, &payloads, options.orbit_id_from_input)
}

fn sbdb_payload(object_id: &str, options: SbdbQueryOptions) -> Result<String, SbdbError> {
    if object_id.trim().is_empty() {
        return Err(SbdbError::InvalidInput(
            "object_id must be non-empty".to_string(),
        ));
    }
    if options.timeout_s <= 0.0 {
        return Err(SbdbError::InvalidInput("timeout_s must be > 0".to_string()));
    }
    if options.max_attempts == 0 {
        return Err(SbdbError::InvalidInput(
            "max_attempts must be > 0".to_string(),
        ));
    }
    let url = sbdb_url(object_id, options.physical_parameters);
    let timeout = Duration::from_secs_f64(options.timeout_s);
    let mut last_error = None;
    for attempt in 0..options.max_attempts {
        match get_text(&url, timeout) {
            Ok(payload) => return Ok(payload),
            Err(error) => last_error = Some(error),
        }
        if attempt + 1 < options.max_attempts {
            std::thread::sleep(Duration::from_secs_f64(
                8.0_f64.min(0.5 * 2.0_f64.powi(attempt as i32)),
            ));
        }
    }
    Err(SbdbError::Request(format!(
        "SBDB query failed after {} attempts: {}",
        options.max_attempts,
        last_error.unwrap_or_else(|| "unknown error".to_string())
    )))
}

fn sbdb_url(object_id: &str, physical_parameters: bool) -> String {
    format!(
        "{SBDB_URL}?sstr={}&cov=mat&full-prec=true&phys-par={}",
        percent_encode(object_id.trim()),
        if physical_parameters { "true" } else { "false" }
    )
}

fn percent_encode(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            output.push(byte as char);
        } else {
            output.push_str(&format!("%{byte:02X}"));
        }
    }
    output
}

fn get_text(url: &str, timeout: Duration) -> Result<String, String> {
    let agent = ureq::AgentBuilder::new().timeout(timeout).build();
    let response = agent
        .get(url)
        .call()
        .map_err(|error| format!("SBDB request failed: {error}"))?;
    response
        .into_string()
        .map_err(|error| format!("SBDB response read failed: {error}"))
}

fn assemble_product(
    kept_ids: &[String],
    payloads: &[String],
    orbit_id_from_input: bool,
) -> Result<SbdbQueryResult, SbdbError> {
    let ids_json = serde_json::to_string(kept_ids)
        .map_err(|error| SbdbError::InvalidResponse(error.to_string()))?;
    let payloads_json = format!("[{}]", payloads.join(","));
    let normalized = crate::coords::sbdb_normalize_payloads_json(&ids_json, &payloads_json)
        .map_err(|error| SbdbError::InvalidResponse(error.to_string()))?;
    let normalized = parse_json(&normalized, "normalized SBDB")?;

    let mut orbit_ids = string_array(&normalized, "orbit_id")?;
    if orbit_id_from_input {
        orbit_ids = kept_ids.to_vec();
    }
    let object_ids = string_array(&normalized, "object_id")?;
    let cometary = matrix_rows(&normalized, "coords_cometary")?;
    let times_jd = number_array(&normalized, "times_jd")?;
    let covariance_rows = field(&normalized, "covariances_cometary")?
        .as_array()
        .ok_or_else(|| invalid_response("covariances_cometary must be an array"))?;
    let covariance_validity = bool_array(&normalized, "covariance_validity")?;
    let nongrav_covariance_axis_mask =
        nongrav_covariance_axis_masks(&normalized, "nongrav_covariance_axis_mask")?;
    let covariance_dimension = if covariance_rows
        .iter()
        .any(|row| row.as_array().is_some_and(|values| values.len() == 81))
    {
        9
    } else {
        6
    };
    let mut covariance =
        vec![f64::NAN; covariance_rows.len() * covariance_dimension * covariance_dimension];
    for (row_index, row) in covariance_rows.iter().enumerate() {
        let values = row
            .as_array()
            .ok_or_else(|| invalid_response("SBDB covariance rows must be arrays"))?;
        if !matches!(values.len(), 36 | 81) {
            return Err(invalid_response(
                "SBDB covariance rows must contain 36 or 81 values",
            ));
        }
        let parsed = values
            .iter()
            .map(|value| value.as_f64().unwrap_or(f64::NAN))
            .collect::<Vec<_>>();
        let row_start = row_index * covariance_dimension * covariance_dimension;
        if values.len() == covariance_dimension * covariance_dimension {
            covariance[row_start..row_start + parsed.len()].copy_from_slice(&parsed);
        } else {
            for coordinate_row in 0..6 {
                let source = &parsed[coordinate_row * 6..coordinate_row * 6 + 6];
                let start = row_start + coordinate_row * covariance_dimension;
                covariance[start..start + 6].copy_from_slice(source);
            }
        }
    }

    let rows = cometary.len();
    let flat = cometary.iter().flatten().copied().collect::<Vec<_>>();
    let epochs_mjd = times_jd
        .iter()
        .map(|value| value - JD_MINUS_MJD)
        .collect::<Vec<_>>();
    let time_array = TimeArray::from_mjd(TimeScale::Tdb, &epochs_mjd)
        .map_err(|error| invalid_response(error.to_string()))?;
    let mu = vec![MU_SUN; rows];
    let (cartesian, covariance) = transform_with_covariance_flat(
        &flat,
        &covariance,
        covariance_dimension,
        Representation::Cometary,
        Representation::Cartesian,
        Frame::Ecliptic,
        Frame::Ecliptic,
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
    let covariance = if covariance_validity.iter().any(|valid| *valid) {
        Some(
            CovarianceBatch::new(
                rows,
                covariance_dimension,
                covariance,
                CovarianceUnits::Coordinate(CoordinateRepresentation::Cartesian),
            )
            .and_then(|covariance| {
                covariance.with_row_validity(Validity::from_bools(&covariance_validity))
            })
            .and_then(|covariance| covariance.with_nongrav_axis_mask(nongrav_covariance_axis_mask))
            .map_err(|error| invalid_response(error.to_string()))?,
        )
    } else {
        None
    };
    let coordinates = CoordinateBatch::cartesian(
        states,
        DataFrame::Ecliptic,
        OriginArray::repeat(OriginId::from_code("SUN"), rows),
        Some(time_array),
        covariance,
    )
    .map_err(|error| invalid_response(error.to_string()))?;

    let physical = field(&normalized, "physical_parameters")?;
    let nan_values = |name: &str| -> Result<Vec<Option<f64>>, SbdbError> {
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
    let warnings = field(&normalized, "warnings")?
        .as_array()
        .ok_or_else(|| invalid_response("warnings must be an array"))?
        .iter()
        .map(|warning| {
            warning
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| invalid_response("warning entries must be strings"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let nongrav_rows = field(&normalized, "non_gravitational_parameters")?
        .as_array()
        .ok_or_else(|| invalid_response("non_gravitational_parameters must be an array"))?;
    let nongrav_float = |name: &str| {
        nongrav_rows
            .iter()
            .map(|row| row.get(name).and_then(Value::as_f64))
            .collect::<Vec<_>>()
    };
    let nongrav = NonGravitationalParametersBatch {
        source: nongrav_rows
            .iter()
            .map(|row| {
                row.get("source")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .collect(),
        a1: nongrav_float("A1"),
        a2: nongrav_float("A2"),
        a3: nongrav_float("A3"),
        aln: nongrav_float("ALN"),
        nk: nongrav_float("NK"),
        nm: nongrav_float("NM"),
        nn: nongrav_float("NN"),
        r0: nongrav_float("R0"),
    };
    let orbits = OrbitBatch::new(
        orbit_ids.into_iter().map(OrbitId).collect(),
        object_ids
            .into_iter()
            .map(|id| Some(ObjectId(id)))
            .collect(),
        coordinates,
    )
    .and_then(|orbits| orbits.with_physical_parameters(physical))
    .and_then(|orbits| orbits.with_non_gravitational_parameters(nongrav))
    .map_err(|error| invalid_response(error.to_string()))?;
    Ok(SbdbQueryResult { orbits, warnings })
}

fn invalid_response(message: impl Into<String>) -> SbdbError {
    SbdbError::InvalidResponse(message.into())
}

fn parse_json(text: &str, context: &str) -> Result<Value, SbdbError> {
    serde_json::from_str(text)
        .map_err(|error| invalid_response(format!("invalid {context} JSON: {error}")))
}

fn field<'a>(value: &'a Value, name: &str) -> Result<&'a Value, SbdbError> {
    value
        .get(name)
        .ok_or_else(|| invalid_response(format!("missing field {name:?}")))
}

fn string_array(value: &Value, name: &str) -> Result<Vec<String>, SbdbError> {
    field(value, name)?
        .as_array()
        .ok_or_else(|| invalid_response(format!("field {name:?} must be an array")))?
        .iter()
        .map(|entry| {
            entry
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| invalid_response(format!("{name} entries must be strings")))
        })
        .collect()
}

fn number_array(value: &Value, name: &str) -> Result<Vec<f64>, SbdbError> {
    field(value, name)?
        .as_array()
        .ok_or_else(|| invalid_response(format!("field {name:?} must be an array")))?
        .iter()
        .map(|entry| {
            entry
                .as_f64()
                .ok_or_else(|| invalid_response(format!("{name} entries must be numeric")))
        })
        .collect()
}

fn optional_number_array(value: &Value, name: &str) -> Result<Vec<Option<f64>>, SbdbError> {
    field(value, name)?
        .as_array()
        .ok_or_else(|| invalid_response(format!("field {name:?} must be an array")))?
        .iter()
        .map(|entry| match entry {
            Value::Null => Ok(None),
            _ => entry
                .as_f64()
                .map(Some)
                .ok_or_else(|| invalid_response(format!("{name} entries must be numeric or null"))),
        })
        .collect()
}

fn bool_array(value: &Value, name: &str) -> Result<Vec<bool>, SbdbError> {
    field(value, name)?
        .as_array()
        .ok_or_else(|| invalid_response(format!("field {name:?} must be an array")))?
        .iter()
        .map(|entry| {
            entry
                .as_bool()
                .ok_or_else(|| invalid_response(format!("{name} entries must be booleans")))
        })
        .collect()
}

fn matrix_rows(value: &Value, name: &str) -> Result<Vec<[f64; 6]>, SbdbError> {
    field(value, name)?
        .as_array()
        .ok_or_else(|| invalid_response(format!("field {name:?} must be an array")))?
        .iter()
        .map(|row| {
            let row = row
                .as_array()
                .ok_or_else(|| invalid_response(format!("{name} rows must be arrays")))?;
            if row.len() != 6 {
                return Err(invalid_response(format!("{name} rows must have length 6")));
            }
            let mut output = [0.0; 6];
            for (slot, item) in output.iter_mut().zip(row) {
                *slot = item
                    .as_f64()
                    .ok_or_else(|| invalid_response(format!("{name} entries must be numeric")))?;
            }
            Ok(output)
        })
        .collect()
}

fn nongrav_covariance_axis_masks(
    value: &Value,
    name: &str,
) -> Result<Vec<NongravCovarianceAxes>, SbdbError> {
    field(value, name)?
        .as_array()
        .ok_or_else(|| invalid_response(format!("field {name:?} must be an array")))?
        .iter()
        .map(|entry| {
            let bits = entry
                .as_u64()
                .and_then(|bits| u8::try_from(bits).ok())
                .ok_or_else(|| invalid_response(format!("{name} entries must be uint8 values")))?;
            NongravCovarianceAxes::from_bits(bits).ok_or_else(|| {
                invalid_response(format!("{name} entry has unsupported bits {bits:#010b}"))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::{IntoNestedRecordBatch, TryFromNestedRecordBatch};
    use serde_json::json;

    const CERES: &str =
        include_str!("../../../../src/adam_core/orbits/query/tests/testdata/sbdb/Ceres.json");
    const VB_2001: &str =
        include_str!("../../../../src/adam_core/orbits/query/tests/testdata/sbdb/2001VB.json");
    const YORP: &str =
        include_str!("../../../../src/adam_core/orbits/query/tests/testdata/sbdb/54509.json");
    const CHURYUMOV_GERASIMENKO: &str =
        include_str!("../../../../src/adam_core/orbits/query/tests/testdata/sbdb/67P_phys.json");
    const APOPHIS: &str =
        include_str!("../../../../src/adam_core/orbits/query/tests/testdata/sbdb/99942_phys.json");
    const MISSING: &str =
        include_str!("../../../../src/adam_core/orbits/query/tests/testdata/sbdb/missing.json");

    fn options() -> SbdbQueryOptions {
        SbdbQueryOptions {
            physical_parameters: true,
            timeout_s: 60.0,
            max_attempts: 1,
            allow_missing: false,
            orbit_id_from_input: true,
        }
    }

    #[test]
    fn url_preserves_existing_request_shape_and_encoding() {
        assert_eq!(
            sbdb_url(" 67P/Churyumov Gerasimenko ", true),
            "https://ssd-api.jpl.nasa.gov/sbdb.api?sstr=67P%2FChuryumov%20Gerasimenko&cov=mat&full-prec=true&phys-par=true"
        );
        assert!(sbdb_url("Ceres", false).ends_with("phys-par=false"));
    }

    #[test]
    fn recorded_product_preserves_order_identities_and_complete_science_fields() {
        let ids = [
            "Ceres".to_string(),
            "2001VB".to_string(),
            "54509".to_string(),
            "67P".to_string(),
            "99942".to_string(),
        ];
        let responses = [CERES, VB_2001, YORP, CHURYUMOV_GERASIMENKO, APOPHIS].map(str::to_string);
        let result = query_sbdb_recorded(&ids, options(), &responses).unwrap();

        assert_eq!(
            result
                .orbits
                .orbit_id
                .iter()
                .map(|id| id.0.as_str())
                .collect::<Vec<_>>(),
            ids.iter().map(String::as_str).collect::<Vec<_>>()
        );
        assert_eq!(
            result.orbits.object_id[0].as_ref().unwrap().0,
            "1 Ceres (A801 AA)"
        );
        assert_eq!(
            result.orbits.object_id[4].as_ref().unwrap().0,
            "99942 Apophis (2004 MN4)"
        );
        assert_eq!(result.orbits.coordinates.values.len(), ids.len());

        let physical = result.orbits.physical_parameters.as_ref().unwrap();
        assert_eq!(physical.h_v[4], Some(19.09));
        assert_eq!(physical.g[4], Some(0.24));

        let nongrav = result.orbits.non_gravitational_parameters.as_ref().unwrap();
        assert_eq!(nongrav.source[3].as_deref(), Some("SBDB"));
        assert_eq!(nongrav.a1[3], Some(1.042451026100725e-9));
        assert_eq!(nongrav.aln[3], Some(0.1112620426));
        assert_eq!(nongrav.nk[3], Some(4.6142));
        assert_eq!(nongrav.r0[3], Some(2.808));

        let covariance = result.orbits.coordinates.covariance.as_ref().unwrap();
        assert_eq!(covariance.dimension, 9);
        assert!(covariance
            .row_values(0)
            .iter()
            .all(|value| { value.is_finite() || value.is_nan() }));
        assert_eq!(
            covariance.nongrav_axis_mask,
            Some(vec![
                NongravCovarianceAxes::NONE,
                NongravCovarianceAxes::NONE,
                NongravCovarianceAxes::A2,
                NongravCovarianceAxes::A1 | NongravCovarianceAxes::A2 | NongravCovarianceAxes::A3,
                NongravCovarianceAxes::A1 | NongravCovarianceAxes::A2,
            ])
        );
        assert!(covariance.is_row_valid(0));
        assert!(covariance.is_row_valid(4));
        assert!(covariance.row_values(4)[8 * 9..9 * 9]
            .iter()
            .all(|value| *value == 0.0));
        assert!(result.warnings.iter().any(|warning| warning.contains("DT")));

        let batch = result.orbits.clone().into_nested_record_batch().unwrap();
        let round_trip = OrbitBatch::try_from_nested_record_batch(&batch).unwrap();
        assert_eq!(round_trip.orbit_id, result.orbits.orbit_id);
        assert_eq!(round_trip.object_id, result.orbits.object_id);
        assert_eq!(
            round_trip.coordinates.values,
            result.orbits.coordinates.values
        );
        let round_trip_covariance = round_trip.coordinates.covariance.unwrap();
        assert_eq!(round_trip_covariance.rows, covariance.rows);
        assert_eq!(round_trip_covariance.dimension, covariance.dimension);
        assert_eq!(
            round_trip_covariance.nongrav_axis_mask,
            covariance.nongrav_axis_mask
        );
        assert_eq!(
            round_trip_covariance
                .values_row_major
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            covariance
                .values_row_major
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert!((0..covariance.rows).all(|row| {
            round_trip_covariance.is_row_valid(row) == covariance.is_row_valid(row)
        }));
    }

    #[test]
    fn recorded_missing_behavior_is_ordered_and_explicit() {
        let ids = ["missing".to_string(), "Ceres".to_string()];
        let responses = [MISSING.to_string(), CERES.to_string()];
        let error = query_sbdb_recorded(&ids, options(), &responses).unwrap_err();
        assert_eq!(
            error,
            SbdbError::NotFound {
                object_id: "missing".to_string()
            }
        );
        assert_eq!(error.to_string(), "__NOT_FOUND__:missing");

        let mut allow = options();
        allow.allow_missing = true;
        let result = query_sbdb_recorded(&ids, allow, &responses).unwrap();
        assert_eq!(result.orbits.orbit_id, vec![OrbitId("Ceres".to_string())]);

        allow.orbit_id_from_input = false;
        let generated = query_sbdb_recorded(&ids, allow, &responses).unwrap();
        assert_eq!(
            generated.orbits.orbit_id,
            vec![OrbitId("00000".to_string())]
        );

        let all_missing =
            query_sbdb_recorded(&["missing".to_string()], allow, &[MISSING.to_string()]).unwrap();
        assert!(all_missing.orbits.orbit_id.is_empty());
        assert!(all_missing.orbits.coordinates.covariance.is_none());
    }

    #[test]
    fn recorded_mixed_validity_and_masks_remain_canonical() {
        let state_labels = ["e", "q", "tp", "node", "peri", "i"];
        let a2_labels = ["e", "q", "tp", "node", "peri", "i", "A2"];
        let payload = |labels: Option<&[&str]>, missing_sigma: bool| {
            let names = ["e", "q", "tp", "om", "w", "i"];
            let values = [0.1, 1.0, 2_460_000.5, 10.0, 20.0, 5.0];
            let elements = names
                .iter()
                .zip(values)
                .enumerate()
                .map(|(index, (name, value))| {
                    json!({
                        "name": name,
                        "value": value,
                        "sigma": if missing_sigma && index == 5 { Value::Null } else { json!(0.01) },
                    })
                })
                .collect::<Vec<_>>();
            let mut orbit = json!({
                "epoch": 2_460_000.5,
                "elements": elements,
                "model_pars": [],
            });
            if let Some(labels) = labels {
                let dimension = labels.len();
                orbit["covariance"] = json!({
                    "labels": labels,
                    "data": (0..dimension).map(|row| {
                        (0..dimension).map(|column| if row == column { 1.0 } else { 0.0 }).collect::<Vec<_>>()
                    }).collect::<Vec<_>>(),
                    "epoch": 2_460_000.5,
                });
            }
            json!({"object": {"fullname": "test object"}, "orbit": orbit}).to_string()
        };
        let ids = ["state".to_string(), "a2".to_string(), "missing".to_string()];
        let responses = [
            payload(Some(&state_labels), false),
            payload(Some(&a2_labels), false),
            payload(None, true),
        ];
        let result = query_sbdb_recorded(&ids, options(), &responses).unwrap();
        let covariance = result.orbits.coordinates.covariance.unwrap();
        assert_eq!(covariance.dimension, 9);
        assert_eq!(covariance.row_dimension(0), 6);
        assert_eq!(covariance.row_dimension(1), 9);
        assert!(covariance.is_row_valid(0));
        assert!(covariance.is_row_valid(1));
        assert!(!covariance.is_row_valid(2));
        assert_eq!(
            covariance.nongrav_axis_mask,
            Some(vec![
                NongravCovarianceAxes::NONE,
                NongravCovarianceAxes::A2,
                NongravCovarianceAxes::NONE,
            ])
        );
        assert!(covariance.row_values(2).iter().all(|value| value.is_nan()));
    }

    #[test]
    fn live_input_validation_precedes_http() {
        let error = query_sbdb(&["  ".to_string()], options()).unwrap_err();
        assert_eq!(
            error,
            SbdbError::InvalidInput("object_id must be non-empty".to_string())
        );

        let mut invalid_timeout = options();
        invalid_timeout.timeout_s = 0.0;
        let error = query_sbdb(&["Ceres".to_string()], invalid_timeout).unwrap_err();
        assert_eq!(
            error,
            SbdbError::InvalidInput("timeout_s must be > 0".to_string())
        );

        let mut no_attempts = options();
        no_attempts.max_attempts = 0;
        let error = query_sbdb(&["Ceres".to_string()], no_attempts).unwrap_err();
        assert_eq!(
            error,
            SbdbError::InvalidInput("max_attempts must be > 0".to_string())
        );
    }

    #[test]
    fn recorded_input_errors_fail_loudly_without_network() {
        let error = query_sbdb_recorded(&["Ceres".to_string()], options(), &[]).unwrap_err();
        assert_eq!(
            error,
            SbdbError::InvalidInput("not enough recorded SBDB responses".to_string())
        );
        let error =
            query_sbdb_recorded(&["Ceres".to_string()], options(), &["{".to_string()]).unwrap_err();
        assert!(matches!(error, SbdbError::InvalidResponse(_)));
        assert!(error.to_string().starts_with("invalid SBDB response JSON:"));
    }
}
