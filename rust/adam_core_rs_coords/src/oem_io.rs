//! CCSDS OEM (Orbit Ephemeris Message) KVN writer/parser (bead personal-cmy.28).
//!
//! Replaces the third-party Python `oem` package for the surfaces adam-core
//! uses: writing single-segment KVN files (header, metadata, state lines,
//! optional lower-triangle covariance blocks) and parsing KVN files back into
//! per-segment arrays. The layout follows the `oem` package:
//!
//! * floats via Python `f"{v:.14e}"`, or `f"{v:.15e}"` in OEM 3.0 (the 16
//!   significant digits of CCSDS 7.5.7), with a two-digit signed exponent;
//! * state/covariance epochs via astropy `Time.strftime("%Y-%m-%dT%H:%M:%S.%f")`
//!   (legacy astropy default millisecond precision through ERFA `d2dtf`, with
//!   astropy's `day_frac` two-sum jd splitting replicated exactly);
//! * header/metadata `KEY = value` lines in order, `META_START`/`META_STOP`,
//!   blank-line separators, and `COV_REF_FRAME` emitted only when it differs
//!   from `REF_FRAME`.
//!
//! [`oem_render_kvn`] writes OEM 2.0, the legacy product, byte for byte as the
//! `oem` package did, or OEM 3.0, which rounds epochs to the millisecond, ties
//! to even, and refuses duplicate epochs, mixed centers, non-finite values and
//! KVN values that are not printable ASCII.
//!
//! The parser mirrors the package's semantics for the files adam-core reads:
//! epoch strings -> ERFA `dtf2d` -> the exact `Timestamp.from_astropy`
//! integer split; lower-triangle covariance reconstruction to a symmetric
//! 6x6; `COMMENT` lines skipped; multiple segments supported.

use crate::local_frames::{local_frame_covariances, LocalFrame};
use crate::types::{
    origin_mu_au3_day2, CoordinateBatch, CoordinateRepresentation, CovarianceBatch,
    CovarianceUnits, Epoch, Frame, ObjectId, OrbitBatch, OrbitId, OriginArray, OriginId,
    SchemaError, SchemaResult, TimeArray, TimeScale, Validity, KM_PER_AU, SECONDS_PER_DAY,
};
use std::fmt::Write as _;
use std::path::Path;

const NANOS_PER_DAY_F64: f64 = 86_400e9;
const MJD_JD_OFFSET: f64 = 2_400_000.5;

fn invalid(message: String) -> SchemaError {
    SchemaError::InvalidRecordBatch(message)
}

// --- float / epoch formatting ---------------------------------------------------

/// Python `f"{value:.{decimals}e}"`: a mantissa of `decimals + 1` significant
/// digits and a two digit signed exponent.
fn py_sci(value: f64, decimals: usize) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    let raw = format!("{value:.prec$e}", prec = decimals);
    let (mantissa, exponent) = raw.split_once('e').expect("exponent");
    let exponent: i32 = exponent.parse().expect("exponent digits");
    format!(
        "{mantissa}e{}{:02}",
        if exponent < 0 { "-" } else { "+" },
        exponent.abs()
    )
}

/// astropy `two_sum` (utils in astropy.time): exact float addition.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let b_virtual = s - a;
    let a_virtual = s - b_virtual;
    let b_error = b - b_virtual;
    let a_error = a - a_virtual;
    (s, a_error + b_error)
}

/// astropy `day_frac(val1, val2)`.
fn day_frac(val1: f64, val2: f64) -> (f64, f64) {
    let (sum12, err12) = two_sum(val1, val2);
    let mut day = sum12.round_ties_even();
    let (extra, frac0) = two_sum(sum12, -day);
    let mut frac = frac0 + (extra + err12);
    let excess = frac.round_ties_even();
    day += excess;
    let (extra, frac0) = two_sum(sum12, -day);
    frac = frac0 + (extra + err12);
    (day, frac)
}

/// astropy `Time(days, fractional_days, format="mjd", scale=...)` jd1/jd2 split.
fn mjd_to_jd_pair(days: i64, nanos: i64) -> (f64, f64) {
    let (day, frac) = day_frac(days as f64, nanos as f64 / NANOS_PER_DAY_F64);
    (day + MJD_JD_OFFSET, frac)
}

/// astropy `Time.strftime("%Y-%m-%dT%H:%M:%S.%f")` on a `to_astropy()` Time:
/// astropy's strftime renders `%f` with `Time.precision` digits, and
/// `Timestamp.to_astropy` leaves the default precision of 3, so legacy OEM
/// epochs carry milliseconds. Civil time goes through ERFA `d2dtf`
/// (leap-second aware for UTC). This is also byte-identical to the legacy
/// `Timestamp.to_iso8601` (astropy `isot`, default precision 3), which the
/// fused reader uses for the legacy per-state `orbit_id` strings.
pub fn format_epoch(days: i64, nanos: i64, scale: TimeScale) -> SchemaResult<String> {
    let (jd1, jd2) = mjd_to_jd_pair(days, nanos);
    let ((iy, im, id, ih, imin, isec, ifrac), _warning) =
        erfars::timescales::D2dtf(scale == TimeScale::Utc, 3, jd1, jd2)
            .map_err(|code| invalid(format!("eraD2dtf failed with ERFA status {code}")))?;
    Ok(format!(
        "{iy:04}-{im:02}-{id:02}T{ih:02}:{imin:02}:{isec:02}.{ifrac:03}"
    ))
}

/// astropy `Time(days, nanos/86400e9, format="mjd", scale=...).mjd` float:
/// `TimeMJD.set_jds` runs `day_frac` and adds the MJD/JD offset to `jd1`;
/// the `.mjd` float output subtracts the offset from `jd1` and sums with
/// `jd2` in that order. Used by the OpenSpace SBDB epoch strings, whose
/// fractional part is the fractional part of Python `repr(mjd)`.
pub fn astropy_mjd_float(days: i64, nanos: i64) -> f64 {
    let (day, frac) = day_frac(days as f64, nanos as f64 / NANOS_PER_DAY_F64);
    ((day + MJD_JD_OFFSET) - MJD_JD_OFFSET) + frac
}

/// ISOT epoch string -> legacy `Timestamp.from_astropy` (days, nanos).
///
/// Public so the Python timestamp veneer and fused OEM parser share exactly
/// one ERFA-backed civil-time implementation.
pub fn parse_epoch(isot: &str, scale: TimeScale) -> SchemaResult<(i64, i64)> {
    let bad = |what: &str| invalid(format!("invalid OEM epoch {isot:?}: {what}"));
    let (date, time) = isot.split_once('T').ok_or_else(|| bad("missing 'T'"))?;
    let mut date_parts = date.split('-');
    let year: i32 = date_parts
        .next()
        .ok_or_else(|| bad("missing year"))?
        .parse()
        .map_err(|_| bad("bad year"))?;
    let month: i32 = date_parts
        .next()
        .ok_or_else(|| bad("missing month"))?
        .parse()
        .map_err(|_| bad("bad month"))?;
    let day: i32 = date_parts
        .next()
        .ok_or_else(|| bad("missing day"))?
        .parse()
        .map_err(|_| bad("bad day"))?;
    let mut time_parts = time.split(':');
    let hour: i32 = time_parts
        .next()
        .ok_or_else(|| bad("missing hour"))?
        .parse()
        .map_err(|_| bad("bad hour"))?;
    let minute: i32 = time_parts
        .next()
        .ok_or_else(|| bad("missing minute"))?
        .parse()
        .map_err(|_| bad("bad minute"))?;
    let seconds: f64 = time_parts
        .next()
        .unwrap_or("0")
        .parse()
        .map_err(|_| bad("bad seconds"))?;
    let ((jd1, jd2), _warning) = erfars::timescales::Dtf2d(
        scale == TimeScale::Utc,
        year,
        month,
        day,
        hour,
        minute,
        seconds,
    )
    .map_err(|code| invalid(format!("eraDtf2d failed with ERFA status {code}")))?;

    // Timestamp.from_astropy integer split.
    let value = jd1 - MJD_JD_OFFSET;
    let mut days = value.floor();
    let mut remainder = value - days;
    remainder += jd2;
    if remainder < 0.0 {
        remainder += 1.0;
        days -= 1.0;
    }
    if remainder >= 1.0 {
        remainder -= 1.0;
        days += 1.0;
    }
    let mut nanos = (remainder * NANOS_PER_DAY_F64).round_ties_even();
    if nanos == NANOS_PER_DAY_F64 {
        days += 1.0;
        nanos = 0.0;
    }
    Ok((days as i64, nanos as i64))
}

// --- parser -----------------------------------------------------------------------

/// One parsed OEM covariance block: symmetric 6x6 matrix (row-major, 36
/// values, km/km-s units) plus its epoch and reference frame (defaulted to
/// the segment `REF_FRAME` when `COV_REF_FRAME` is absent).
#[derive(Debug, Clone, PartialEq)]
pub struct OemParsedCovariance {
    pub days: i64,
    pub nanos: i64,
    pub frame: Option<String>,
    pub matrix: Vec<f64>,
}

/// One parsed OEM segment: metadata `KEY -> value` strings in file order
/// plus per-state epoch integers and km/km-s state vectors.
#[derive(Debug, Clone, Default)]
pub struct OemSegment {
    pub metadata: serde_json::Map<String, serde_json::Value>,
    pub days: Vec<i64>,
    pub nanos: Vec<i64>,
    pub values_km: Vec<Vec<f64>>,
    pub covariances: Vec<OemParsedCovariance>,
}

/// A parsed OEM document.
#[derive(Debug, Clone, Default)]
pub struct OemDocument {
    pub header: serde_json::Map<String, serde_json::Value>,
    pub segments: Vec<OemSegment>,
}

/// Parse a KVN OEM file into a JSON payload:
/// `{header: {...}, segments: [{metadata: {...}, states: {days, nanos,
/// values_km}, covariances: [{days, nanos, frame, matrix (36, symmetric,
/// km)}]}]}`. Epoch integers use the exact legacy `Timestamp.from_astropy`
/// split in the segment's TIME_SYSTEM scale.
pub fn oem_parse_kvn(path: &Path) -> SchemaResult<String> {
    use serde_json::json;

    let document = oem_parse_kvn_structured(path)?;
    let payload = json!({
        "header": document.header,
        "segments": document.segments
            .into_iter()
            .map(|segment| {
                json!({
                    "metadata": segment.metadata,
                    "states": {
                        "days": segment.days,
                        "nanos": segment.nanos,
                        "values_km": segment.values_km,
                    },
                    "covariances": segment.covariances
                        .into_iter()
                        .map(|covariance| {
                            json!({
                                "days": covariance.days,
                                "nanos": covariance.nanos,
                                "frame": covariance.frame,
                                "matrix": covariance.matrix,
                            })
                        })
                        .collect::<Vec<_>>(),
                })
            })
            .collect::<Vec<_>>(),
    });
    serde_json::to_string(&payload).map_err(|err| invalid(format!("encode failed: {err}")))
}

/// Structured form of [`oem_parse_kvn`] for Rust consumers (the fused
/// orbit-product reader).
pub fn oem_parse_kvn_structured(path: &Path) -> SchemaResult<OemDocument> {
    use serde_json::{Map, Value};

    let text = std::fs::read_to_string(path)
        .map_err(|err| invalid(format!("failed to read {}: {err}", path.display())))?;

    type Segment = OemSegment;

    let mut header: Map<String, Value> = Map::new();
    let mut segments: Vec<Segment> = Vec::new();
    let mut current: Option<Segment> = None;
    let mut in_meta = false;
    let mut in_cov = false;
    let mut cov_epoch: Option<(i64, i64)> = None;
    let mut cov_frame: Option<String> = None;
    let mut cov_values: Vec<f64> = Vec::new();

    let scale_of = |segment: &Segment| -> SchemaResult<TimeScale> {
        let system = segment
            .metadata
            .get("TIME_SYSTEM")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("OEM segment missing TIME_SYSTEM".to_string()))?;
        TimeScale::parse(&system.to_lowercase())
    };

    let flush_cov = |segment: &mut Segment,
                     cov_epoch: &mut Option<(i64, i64)>,
                     cov_frame: &mut Option<String>,
                     cov_values: &mut Vec<f64>|
     -> SchemaResult<()> {
        if let Some((days, nanos)) = cov_epoch.take() {
            if cov_values.len() != 21 {
                return Err(invalid(format!(
                    "OEM covariance block must have 21 lower-triangle values, got {}",
                    cov_values.len()
                )));
            }
            // Symmetric 6x6 reconstruction (row-major, matching the oem
            // package's Covariance.matrix).
            let mut matrix = vec![0.0f64; 36];
            for (index, &(row, col)) in LOWER_TRIANGLE_INDICES.iter().enumerate() {
                let value = cov_values[index];
                matrix[row * 6 + col] = value;
                matrix[col * 6 + row] = value;
            }
            let frame = cov_frame.take().or_else(|| {
                segment
                    .metadata
                    .get("REF_FRAME")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            });
            segment.covariances.push(OemParsedCovariance {
                days,
                nanos,
                frame,
                matrix,
            });
            cov_values.clear();
        }
        Ok(())
    };

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with("COMMENT") {
            continue;
        }
        if line == "META_START" {
            if let Some(mut segment) = current.take() {
                flush_cov(
                    &mut segment,
                    &mut cov_epoch,
                    &mut cov_frame,
                    &mut cov_values,
                )?;
                segments.push(segment);
            }
            current = Some(Segment::default());
            in_meta = true;
            in_cov = false;
            continue;
        }
        if line == "META_STOP" {
            in_meta = false;
            continue;
        }
        if line == "COVARIANCE_START" {
            in_cov = true;
            continue;
        }
        if line == "COVARIANCE_STOP" {
            if let Some(segment) = current.as_mut() {
                flush_cov(segment, &mut cov_epoch, &mut cov_frame, &mut cov_values)?;
            }
            in_cov = false;
            continue;
        }

        if let Some((key, value)) = line.split_once('=') {
            let key = key.trim();
            let value = value.trim();
            if current.is_none() {
                header.insert(key.to_string(), Value::String(value.to_string()));
                continue;
            }
            if in_meta {
                if let Some(segment) = current.as_mut() {
                    segment
                        .metadata
                        .insert(key.to_string(), Value::String(value.to_string()));
                }
                continue;
            }
            if in_cov {
                let segment = current.as_mut().expect("segment");
                match key {
                    "EPOCH" => {
                        flush_cov(segment, &mut cov_epoch, &mut cov_frame, &mut cov_values)?;
                        let scale = scale_of(segment)?;
                        cov_epoch = Some(parse_epoch(value, scale)?);
                    }
                    "COV_REF_FRAME" => {
                        cov_frame = Some(value.to_string());
                    }
                    other => {
                        return Err(invalid(format!(
                            "unexpected key in OEM covariance block: {other}"
                        )));
                    }
                }
                continue;
            }
            continue;
        }

        // Data lines: either state rows or covariance triangle rows.
        let segment = current.as_mut().ok_or_else(|| {
            invalid(format!(
                "unexpected OEM data line before META_START: {line}"
            ))
        })?;
        if in_cov {
            for token in line.split_whitespace() {
                cov_values.push(
                    token
                        .parse::<f64>()
                        .map_err(|_| invalid(format!("invalid OEM covariance value: {token}")))?,
                );
            }
            continue;
        }
        let mut tokens = line.split_whitespace();
        let epoch = tokens
            .next()
            .ok_or_else(|| invalid("empty OEM state line".to_string()))?;
        let scale = scale_of(segment)?;
        let (days, nanos) = parse_epoch(epoch, scale)?;
        let values: Vec<f64> = tokens
            .map(|token| {
                token
                    .parse::<f64>()
                    .map_err(|_| invalid(format!("invalid OEM state value: {token}")))
            })
            .collect::<SchemaResult<_>>()?;
        if values.len() < 6 {
            return Err(invalid(format!(
                "OEM state line must have at least 6 values, got {}",
                values.len()
            )));
        }
        segment.days.push(days);
        segment.nanos.push(nanos);
        segment.values_km.push(values[..6].to_vec());
    }
    if let Some(mut segment) = current.take() {
        flush_cov(
            &mut segment,
            &mut cov_epoch,
            &mut cov_frame,
            &mut cov_values,
        )?;
        segments.push(segment);
    }

    Ok(OemDocument { header, segments })
}

// --- fused orbit products (bead personal-cmy.37.4.4) -------------------------------

/// `np.tril_indices(6)` order (row sweep of the lower triangle), shared by
/// the parser's symmetric reconstruction and the writer's extraction.
const LOWER_TRIANGLE_INDICES: [(usize, usize); 21] = [
    (0, 0),
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 1),
    (2, 2),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
    (4, 0),
    (4, 1),
    (4, 2),
    (4, 3),
    (4, 4),
    (5, 0),
    (5, 1),
    (5, 2),
    (5, 3),
    (5, 4),
    (5, 5),
];

/// Legacy `_adam_to_oem_center` (exact error message).
fn adam_to_oem_center(code: &str) -> SchemaResult<&'static str> {
    match code {
        "SOLAR_SYSTEM_BARYCENTER" => Ok("SOLAR SYSTEM BARYCENTER"),
        "MERCURY_BARYCENTER" => Ok("MERCURY BARYCENTER"),
        "VENUS_BARYCENTER" => Ok("VENUS BARYCENTER"),
        "EARTH_MOON_BARYCENTER" => Ok("EARTH BARYCENTER"),
        "MARS_BARYCENTER" => Ok("MARS BARYCENTER"),
        "JUPITER_BARYCENTER" => Ok("JUPITER BARYCENTER"),
        "SATURN_BARYCENTER" => Ok("SATURN BARYCENTER"),
        "URANUS_BARYCENTER" => Ok("URANUS BARYCENTER"),
        "NEPTUNE_BARYCENTER" => Ok("NEPTUNE BARYCENTER"),
        "SUN" => Ok("SUN"),
        "MERCURY" => Ok("MERCURY"),
        "VENUS" => Ok("VENUS"),
        "EARTH" => Ok("EARTH"),
        "MOON" => Ok("MOON"),
        "MARS" => Ok("MARS"),
        "JUPITER" => Ok("JUPITER"),
        "SATURN" => Ok("SATURN"),
        "URANUS" => Ok("URANUS"),
        "NEPTUNE" => Ok("NEPTUNE"),
        other => Err(invalid(format!(
            "Unsupported origin code for OEM conversion: {other}"
        ))),
    }
}

/// Legacy `_oem_to_adam_center` map in dict insertion order (the error
/// message renders the key list exactly like Python).
const OEM_TO_ADAM_CENTERS: [(&str, &str); 19] = [
    ("SOLAR SYSTEM BARYCENTER", "SOLAR_SYSTEM_BARYCENTER"),
    ("MERCURY BARYCENTER", "MERCURY_BARYCENTER"),
    ("VENUS BARYCENTER", "VENUS_BARYCENTER"),
    ("EARTH BARYCENTER", "EARTH_MOON_BARYCENTER"),
    ("MARS BARYCENTER", "MARS_BARYCENTER"),
    ("JUPITER BARYCENTER", "JUPITER_BARYCENTER"),
    ("SATURN BARYCENTER", "SATURN_BARYCENTER"),
    ("URANUS BARYCENTER", "URANUS_BARYCENTER"),
    ("NEPTUNE BARYCENTER", "NEPTUNE_BARYCENTER"),
    ("SUN", "SUN"),
    ("MERCURY", "MERCURY"),
    ("VENUS", "VENUS"),
    ("EARTH", "EARTH"),
    ("MOON", "MOON"),
    ("MARS", "MARS"),
    ("JUPITER", "JUPITER"),
    ("SATURN", "SATURN"),
    ("URANUS", "URANUS"),
    ("NEPTUNE", "NEPTUNE"),
];

fn oem_to_adam_center(center: &str) -> SchemaResult<&'static str> {
    let upper = center.to_uppercase();
    for (oem, adam) in OEM_TO_ADAM_CENTERS {
        if oem == upper {
            return Ok(adam);
        }
    }
    let keys = OEM_TO_ADAM_CENTERS
        .iter()
        .map(|(key, _)| format!("'{key}'"))
        .collect::<Vec<_>>()
        .join(", ");
    Err(invalid(format!(
        "Unsupported OEM center name: {center}. Supported centers are [{keys}]."
    )))
}

/// `_oem_to_adam_frame`: ICRF, J2000 and GCRF are read as the J2000 axes
/// adam_core calls equatorial.
fn oem_to_adam_frame(frame: &str) -> SchemaResult<Frame> {
    match frame {
        "EME2000" | "ICRF" | "J2000" | "GCRF" => Ok(Frame::Equatorial),
        "ITRF-93" => Ok(Frame::Itrf93),
        other => Err(invalid(format!(
            "Unsupported OEM frame: {other}. Supported frames are \
             ['EME2000', 'ICRF', 'J2000', 'GCRF', 'ITRF-93']."
        ))),
    }
}

/// Legacy `convert_cartesian_values_au_to_km` row operation order.
fn values_au_to_km(row: &[f64; 6]) -> [f64; 6] {
    [
        row[0] * KM_PER_AU,
        row[1] * KM_PER_AU,
        row[2] * KM_PER_AU,
        row[3] * KM_PER_AU / SECONDS_PER_DAY,
        row[4] * KM_PER_AU / SECONDS_PER_DAY,
        row[5] * KM_PER_AU / SECONDS_PER_DAY,
    ]
}

/// Legacy covariance unit conversion (outer-product factors, exact order).
fn convert_covariance_matrix(matrix: &[f64], au_to_km: bool) -> Vec<f64> {
    let unit = if au_to_km {
        [
            KM_PER_AU,
            KM_PER_AU,
            KM_PER_AU,
            KM_PER_AU / SECONDS_PER_DAY,
            KM_PER_AU / SECONDS_PER_DAY,
            KM_PER_AU / SECONDS_PER_DAY,
        ]
    } else {
        [
            1.0 / KM_PER_AU,
            1.0 / KM_PER_AU,
            1.0 / KM_PER_AU,
            SECONDS_PER_DAY / KM_PER_AU,
            SECONDS_PER_DAY / KM_PER_AU,
            SECONDS_PER_DAY / KM_PER_AU,
        ]
    };
    let mut out = vec![0.0_f64; 36];
    for j in 0..6 {
        for k in 0..6 {
            out[j * 6 + k] = matrix[j * 6 + k] * (unit[j] * unit[k]);
        }
    }
    out
}

fn time_system_upper(scale: TimeScale) -> String {
    scale.as_str().to_uppercase()
}

// --- writer -----------------------------------------------------------------------

/// Covariance frame labels of CCSDS 502.0-B-3 table 5-4.
const TABLE_COVARIANCE_FRAMES: [&str; 3] = ["RSW", "RTN", "TNW"];
/// CCSDS 502.0-B-3 7.3.2: a COMMENT line, keyword included, is at most 254 characters.
const MAX_COMMENT_LINE: usize = 254;
const NANOS_PER_MILLISECOND: i64 = 1_000_000;

/// Options of [`oem_render_kvn`].
#[derive(Debug, Clone, PartialEq)]
pub struct OemWriteOptions {
    /// `CCSDS_OEM_VERS`, "2.0" or "3.0".
    pub version: String,
    pub originator: String,
    pub creation_date: String,
    /// `OBJECT_NAME` and `OBJECT_ID`; None or empty means the orbits' object_id.
    pub object_name: Option<String>,
    pub object_id: Option<String>,
    /// `COMMENT` lines written right after `META_START`.
    pub comments: Vec<String>,
    pub include_covariance: bool,
    /// None or the REF_FRAME label writes the state covariance; any other
    /// label names a local orbital frame.
    pub covariance_frame: Option<String>,
    /// Refuse covariance frames outside table 5-4 (always on for 2.0).
    pub table_frames_only: bool,
}

impl OemWriteOptions {
    /// The legacy product: OEM 2.0 with state covariance blocks.
    pub fn legacy(originator: &str, creation_date: &str) -> Self {
        Self {
            version: "2.0".to_string(),
            originator: originator.to_string(),
            creation_date: creation_date.to_string(),
            object_name: None,
            object_id: None,
            comments: Vec::new(),
            include_covariance: true,
            covariance_frame: None,
            table_frames_only: false,
        }
    }
}

/// Nearest millisecond, ties to even.
fn round_to_millisecond(epoch: &Epoch) -> Epoch {
    let half = NANOS_PER_MILLISECOND / 2;
    let quotient = epoch.nanos.div_euclid(NANOS_PER_MILLISECOND);
    let remainder = epoch.nanos.rem_euclid(NANOS_PER_MILLISECOND);
    let up = remainder > half || (remainder == half && quotient % 2 == 1);
    Epoch::new(
        epoch.days,
        (quotient + i64::from(up)) * NANOS_PER_MILLISECOND,
    )
}

/// Distinct values in first-seen order.
fn distinct<T: Eq + std::hash::Hash + Clone>(items: impl IntoIterator<Item = T>) -> Vec<T> {
    let mut seen = std::collections::HashSet::new();
    items
        .into_iter()
        .filter(|item| seen.insert(item.clone()))
        .collect()
}

/// Top left 6x6 block of the covariance of `row` (9x9 rows carry the
/// non-gravitational parameters after the state) when the row is valid and
/// the block is not all NaN (legacy rule).
fn state_covariance(covariance: Option<&CovarianceBatch>, row: usize) -> Option<[f64; 36]> {
    let covariance = covariance.filter(|covariance| covariance.dimension >= 6)?;
    let (values, dimension) = (covariance.row_values(row), covariance.dimension);
    let matrix: [f64; 36] = std::array::from_fn(|index| values[index / 6 * dimension + index % 6]);
    (covariance.is_row_valid(row) && !matrix.iter().all(|value| value.is_nan())).then_some(matrix)
}

/// Render one object's states as a single-segment KVN OEM and return the text
/// with the number of epochs moved onto the millisecond grid (3.0 only).
/// Ecliptic input is rotated to equatorial and states are stably sorted by
/// time. The covariance is written in REF_FRAME or rotated into a local
/// orbital frame, `_ROTATING` frames with the center's gravitational
/// parameter. Errors name states by their input row.
pub fn oem_render_kvn(
    orbits: &OrbitBatch,
    options: &OemWriteOptions,
) -> SchemaResult<(String, usize)> {
    let version_3 = match options.version.as_str() {
        "2.0" => false,
        "3.0" => true,
        other => {
            return Err(invalid(format!(
                "OEM version must be \"2.0\" or \"3.0\", got \"{other}\""
            )))
        }
    };
    let rotated;
    let orbits = match orbits.coordinates.frame {
        Frame::Equatorial => orbits,
        Frame::Ecliptic => {
            rotated = orbits.rotate_frame(Frame::Equatorial)?;
            &rotated
        }
        other => {
            return Err(invalid(format!(
                "OEM writer requires equatorial or ecliptic coordinates, got {other:?}; \
                 transform first"
            )))
        }
    };
    let times = orbits
        .coordinates
        .times
        .as_ref()
        .ok_or_else(|| invalid("OEM writer requires coordinate times".to_string()))?;
    let values = orbits
        .coordinates
        .values
        .cartesian()
        .ok_or_else(|| invalid("OEM writer requires Cartesian coordinates".to_string()))?;
    let n = values.len();
    if n == 0 {
        return Err(invalid(
            "OEM writer requires at least one state".to_string(),
        ));
    }
    let scale = times.scale;

    // One object about one center (CCSDS 502.0-B-3 5.1.3); 2.0 without a
    // covariance frame writes the first center, as the legacy writer did.
    let object_ids = distinct(
        orbits
            .object_id
            .iter()
            .map(|id| id.as_ref().map(|id| id.0.as_str())),
    );
    let origin = &orbits.coordinates.origins.origins[0];
    let origins = distinct(
        orbits
            .coordinates
            .origins
            .origins
            .iter()
            .map(OriginId::code),
    );
    let one_center = origins.len() == 1 || !(version_3 || options.covariance_frame.is_some());
    let object_id = match object_ids.as_slice() {
        [Some(object_id)] if one_center => object_id.to_string(),
        _ => {
            return Err(invalid(format!(
                "An OEM needs a non-null object_id and carries one object about one center \
                 per file, got {} object_ids and {} origins.",
                object_ids.len(),
                origins.len()
            )))
        }
    };

    // 3.0 rounds so state and covariance EPOCH strings agree and read back
    // exactly; 2.0 leaves the epochs to ERFA's formatting, as the legacy writer did.
    let (epochs, off_grid_epochs): (Vec<Epoch>, usize) = if version_3 {
        let off_grid = times
            .epochs
            .iter()
            .filter(|epoch| epoch.nanos % NANOS_PER_MILLISECOND != 0)
            .count();
        (
            times.epochs.iter().map(round_to_millisecond).collect(),
            off_grid,
        )
    } else {
        (times.epochs.clone(), 0)
    };
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| (epochs[i].days, epochs[i].nanos));
    if version_3 {
        if let Some(pair) = order
            .windows(2)
            .find(|pair| epochs[pair[0]] == epochs[pair[1]])
        {
            let epoch = epochs[pair[0]];
            return Err(invalid(format!(
                "Epochs must be unique within an OEM after rounding to the millisecond, \
                 {} appears twice.",
                format_epoch(epoch.days, epoch.nanos, scale)?
            )));
        }
        if let Some(i) = (0..n).find(|&i| !values[i].iter().all(|value| value.is_finite())) {
            return Err(invalid(format!("State {i} has a non-finite value.")));
        }
    }

    let ref_frame = if version_3 { "ICRF" } else { "EME2000" };
    // Parsed whenever given, so a bad name fails without a covariance block too.
    let local_frame = match options
        .covariance_frame
        .as_deref()
        .map(|label| label.trim().to_uppercase())
    {
        Some(label) if label != ref_frame => Some((LocalFrame::parse(&label)?, label)),
        _ => None,
    };
    let center = adam_to_oem_center(&origin.code())?;
    let names = [&options.object_name, &options.object_id]
        .map(|name| name.as_deref().filter(|text| !text.is_empty()));
    let (first, last) = (epochs[order[0]], epochs[order[n - 1]]);
    let header = [
        ("CCSDS_OEM_VERS", options.version.clone()),
        ("CREATION_DATE", options.creation_date.clone()),
        ("ORIGINATOR", options.originator.clone()),
    ];
    let metadata = [
        (
            "OBJECT_NAME",
            names[0].unwrap_or(object_id.as_str()).to_string(),
        ),
        (
            "OBJECT_ID",
            names[1].unwrap_or(object_id.as_str()).to_string(),
        ),
        ("CENTER_NAME", center.to_string()),
        ("REF_FRAME", ref_frame.to_string()),
        ("TIME_SYSTEM", time_system_upper(scale)),
        ("START_TIME", format_epoch(first.days, first.nanos, scale)?),
        ("STOP_TIME", format_epoch(last.days, last.nanos, scale)?),
    ];

    let mut comments = options.comments.clone();
    let mut label = ref_frame.to_string();
    let mut records: Vec<(Epoch, [f64; 21])> = Vec::new();
    if options.include_covariance {
        let covariance = orbits.coordinates.covariance.as_ref();
        // Input row order, so kernel errors name the caller's rows.
        let mut rows: Vec<Option<[f64; 36]>> =
            (0..n).map(|i| state_covariance(covariance, i)).collect();
        if let Some((frame, given)) = local_frame {
            if rows.iter().all(Option::is_none) {
                return Err(invalid("The states carry no covariance.".to_string()));
            }
            // Table 5-4 labels are written as given, any other as its registry name.
            label = if TABLE_COVARIANCE_FRAMES.contains(&given.as_str()) {
                given
            } else {
                let name = frame.canonical_name();
                let note = format!(
                    "COV_REF_FRAME {name} follows the SANA orbit-relative reference frames \
                     registry (CCSDS 502.0-B-3 annex B5)."
                );
                if !version_3 || options.table_frames_only {
                    let remedy = if version_3 {
                        "Pass table_frames_only=False to write it anyway."
                    } else {
                        "Annex B5 is an OEM 3.0 provision, pass version=\"3.0\" to write it."
                    };
                    return Err(invalid(format!(
                        "{note} It is outside the {} set of table 5-4. {remedy}",
                        TABLE_COVARIANCE_FRAMES.join(", ")
                    )));
                }
                comments.push(note);
                name.to_string()
            };
            let mu = if frame.rotating {
                let mu = origin_mu_au3_day2(origin).map_err(|_| {
                    invalid(format!(
                        "{} needs the gravitational parameter of the center, which adam_core \
                         does not have for {}.",
                        frame.canonical_name(),
                        origin.code()
                    ))
                })?;
                vec![mu; n]
            } else {
                Vec::new()
            };
            let flat_values: Vec<f64> = values.iter().flatten().copied().collect();
            let flat_covariances: Vec<f64> = rows
                .iter()
                .flat_map(|row| row.unwrap_or([f64::NAN; 36]))
                .collect();
            let local = local_frame_covariances(&flat_values, &flat_covariances, &mu, frame)?;
            for (row, rotated) in rows.iter_mut().zip(local.chunks_exact(36)) {
                if let Some(matrix) = row {
                    matrix.copy_from_slice(rotated);
                }
            }
        }
        records = order
            .iter()
            .filter_map(|&i| {
                let matrix_km = convert_covariance_matrix(&rows[i]?, true);
                Some((
                    epochs[i],
                    LOWER_TRIANGLE_INDICES.map(|(row, col)| matrix_km[row * 6 + col]),
                ))
            })
            .collect();
    }
    if version_3 {
        if let Some((epoch, _)) = records
            .iter()
            .find(|(_, lower)| !lower.iter().all(|value| value.is_finite()))
        {
            return Err(invalid(format!(
                "Covariance at epoch {} has a non-finite entry.",
                format_epoch(epoch.days, epoch.nanos, scale)?
            )));
        }
    }

    // CCSDS 502.0-B-3 7.3.4 for comments and the caller's names, and for every
    // header and metadata value in 3.0; 2.0 writes the rest verbatim, as the
    // legacy writer did.
    let kvn_values = header
        .iter()
        .chain(&metadata)
        .filter(|_| version_3)
        .map(|(key, value)| (*key, value.as_str()))
        .chain(
            ["OBJECT_NAME", "OBJECT_ID"]
                .into_iter()
                .zip(names)
                .filter_map(|(key, name)| Some((key, name?))),
        )
        .chain(comments.iter().map(|comment| ("COMMENT", comment.as_str())));
    for (key, value) in kvn_values {
        if value.trim().is_empty() || !value.bytes().all(|byte| (b' '..=b'~').contains(&byte)) {
            return Err(invalid(format!(
                "OEM value for {key} must be non-empty printable ASCII on one line."
            )));
        }
    }
    for comment in &comments {
        let length = "COMMENT ".len() + comment.chars().count();
        if length > MAX_COMMENT_LINE {
            return Err(invalid(format!(
                "A COMMENT line must be at most {MAX_COMMENT_LINE} characters including the \
                 keyword (CCSDS 502.0-B-3 7.3.2), got {length}."
            )));
        }
    }

    // The `oem` package's KVN layout, comments right after META_START.
    let decimals = if version_3 { 15 } else { 14 };
    let mut out = String::new();
    for (key, value) in &header {
        let _ = writeln!(out, "{key} = {value}");
    }
    out.push_str("\nMETA_START\n");
    for comment in &comments {
        let _ = writeln!(out, "COMMENT {comment}");
    }
    for (key, value) in &metadata {
        let _ = writeln!(out, "{key} = {value}");
    }
    out.push_str("META_STOP\n\n");
    for &i in &order {
        let state = values_au_to_km(&values[i]).map(|value| py_sci(value, decimals));
        let epoch = format_epoch(epochs[i].days, epochs[i].nanos, scale)?;
        let _ = writeln!(out, "{epoch} {}", state.join(" "));
    }
    out.push('\n');
    if !records.is_empty() {
        out.push_str("COVARIANCE_START\n");
        for (epoch, lower) in &records {
            let _ = writeln!(
                out,
                "EPOCH = {}",
                format_epoch(epoch.days, epoch.nanos, scale)?
            );
            if label != ref_frame {
                let _ = writeln!(out, "COV_REF_FRAME = {label}");
            }
            for row in 0..6 {
                let start = row * (row + 1) / 2;
                let rendered: Vec<String> = lower[start..=start + row]
                    .iter()
                    .map(|&value| py_sci(value, decimals))
                    .collect();
                let _ = writeln!(out, "{}", rendered.join(" "));
            }
        }
        out.push_str("COVARIANCE_STOP\n\n");
    }
    Ok((out, off_grid_epochs))
}

/// Render (see [`oem_render_kvn`]) and write the file; returns the number of
/// epochs moved onto the millisecond grid.
pub fn oem_write_kvn_file(
    path: &Path,
    orbits: &OrbitBatch,
    options: &OemWriteOptions,
) -> SchemaResult<usize> {
    let (text, off_grid_epochs) = oem_render_kvn(orbits, options)?;
    std::fs::write(path, text)
        .map_err(|err| invalid(format!("failed to write {}: {err}", path.display())))?;
    Ok(off_grid_epochs)
}

/// Fused legacy `orbit_from_oem`: parse the KVN file and assemble the
/// complete `OrbitBatch` (frame/center mapping with exact legacy errors,
/// km->AU conversion, epoch-and-frame covariance matching with last-match
/// precedence, legacy per-state orbit ids). Returns `None` for files with
/// no segments (the caller returns `Orbits.empty()`), and a dedicated
/// "mixed" error when segments disagree on frame or time system (the
/// caller falls back to the legacy per-state composition).
pub fn oem_read_orbits(path: &Path) -> SchemaResult<Option<OrbitBatch>> {
    use serde_json::Value;

    let document = oem_parse_kvn_structured(path)?;
    if document.segments.is_empty() {
        return Ok(None);
    }

    let mut orbit_ids: Vec<OrbitId> = Vec::new();
    let mut object_ids: Vec<Option<ObjectId>> = Vec::new();
    let mut rows: Vec<[f64; 6]> = Vec::new();
    let mut days: Vec<i64> = Vec::new();
    let mut nanos: Vec<i64> = Vec::new();
    let mut origins: Vec<OriginId> = Vec::new();
    let mut covariance_values: Vec<f64> = Vec::new();
    let mut covariance_validity: Vec<bool> = Vec::new();
    let mut frame_out: Option<Frame> = None;
    let mut scale_out: Option<TimeScale> = None;

    for (segment_index, segment) in document.segments.iter().enumerate() {
        let meta = |key: &str| -> SchemaResult<&str> {
            segment
                .metadata
                .get(key)
                .and_then(Value::as_str)
                .ok_or_else(|| invalid(format!("OEM segment missing {key}")))
        };
        let object_id = meta("OBJECT_ID")?;
        let ref_frame = meta("REF_FRAME")?;
        let frame = oem_to_adam_frame(ref_frame)?;
        let origin_code = oem_to_adam_center(meta("CENTER_NAME")?)?;
        let scale = TimeScale::parse(&meta("TIME_SYSTEM")?.to_lowercase())?;
        if *frame_out.get_or_insert(frame) != frame || *scale_out.get_or_insert(scale) != scale {
            return Err(invalid(
                "OEM segments have mixed reference frames or time systems".to_string(),
            ));
        }

        for j in 0..segment.days.len() {
            let state_days = segment.days[j];
            let state_nanos = segment.nanos[j];
            let value = &segment.values_km[j];
            rows.push([
                value[0] / KM_PER_AU,
                value[1] / KM_PER_AU,
                value[2] / KM_PER_AU,
                value[3] / KM_PER_AU * SECONDS_PER_DAY,
                value[4] / KM_PER_AU * SECONDS_PER_DAY,
                value[5] / KM_PER_AU * SECONDS_PER_DAY,
            ]);
            days.push(state_days);
            nanos.push(state_nanos);
            origins.push(OriginId::from_code(origin_code));

            // Legacy last-match-wins epoch+frame covariance join.
            let mut matched: Option<Vec<f64>> = None;
            for covariance in &segment.covariances {
                if covariance.days == state_days
                    && covariance.nanos == state_nanos
                    && covariance.frame.as_deref() == Some(ref_frame)
                {
                    matched = Some(convert_covariance_matrix(&covariance.matrix, false));
                }
            }
            match matched {
                Some(matrix) => {
                    covariance_values.extend(matrix);
                    covariance_validity.push(true);
                }
                None => {
                    covariance_values.extend([f64::NAN; 36]);
                    covariance_validity.push(false);
                }
            }

            orbit_ids.push(OrbitId(format!(
                "{object_id}_seg_{segment_index}_{}",
                format_epoch(state_days, state_nanos, scale)?
            )));
            object_ids.push(Some(ObjectId(object_id.to_string())));
        }
    }

    let n = rows.len();
    let covariance = CovarianceBatch::new(
        n,
        6,
        covariance_values,
        CovarianceUnits::Coordinate(CoordinateRepresentation::Cartesian),
    )?
    .with_row_validity(Validity::from_bools(&covariance_validity))?;
    let coordinates = CoordinateBatch::cartesian(
        rows,
        frame_out.expect("at least one segment"),
        OriginArray::new(origins),
        Some(TimeArray::from_parts(
            scale_out.expect("at least one segment"),
            days,
            nanos,
        )?),
        Some(covariance),
    )?;
    Ok(Some(OrbitBatch::new(orbit_ids, object_ids, coordinates)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three unsorted states of one object about the Sun. Row 1 has an
    /// all-NaN covariance and row 2 an invalid one, so only row 0 has a block.
    fn fixture(frame: Frame, scale: TimeScale, with_covariance: bool) -> OrbitBatch {
        let values = vec![
            [1.1, -0.3, 0.05, 0.001, 0.017, -0.0003],
            [0.9, 0.4, -0.02, -0.009, 0.015, 0.0002],
            [1.3, 0.1, 0.07, 0.004, -0.012, 0.0001],
        ];
        let covariance = with_covariance.then(|| {
            let mut matrices = Vec::with_capacity(3 * 36);
            for row in 0..3 {
                for j in 0..6 {
                    for k in 0..6 {
                        let scale_j = if j < 3 { 1e-6 } else { 1e-8 };
                        let scale_k = if k < 3 { 1e-6 } else { 1e-8 };
                        let shape = if j == k {
                            2.0 + row as f64
                        } else {
                            0.1 * ((j + k + row) as f64).sin()
                        };
                        matrices.push(if row == 1 {
                            f64::NAN
                        } else {
                            shape * scale_j * scale_k
                        });
                    }
                }
            }
            CovarianceBatch::new(
                3,
                6,
                matrices,
                CovarianceUnits::Coordinate(CoordinateRepresentation::Cartesian),
            )
            .unwrap()
            .with_row_validity(Validity::from_bools(&[true, true, false]))
            .unwrap()
        });
        let coordinates = CoordinateBatch::cartesian(
            values,
            frame,
            OriginArray::repeat(OriginId::from_code("SUN"), 3),
            Some(
                TimeArray::from_parts(
                    scale,
                    vec![60002, 60000, 60001],
                    vec![0, 43_200_000_000_000, 123_000_000],
                )
                .unwrap(),
            ),
            covariance,
        )
        .unwrap();
        OrbitBatch::new(
            (0..3).map(|i| OrbitId(format!("o{i}"))).collect(),
            vec![Some(ObjectId("TEST OBJECT".to_string())); 3],
            coordinates,
        )
        .unwrap()
    }

    /// Output of the legacy `oem_render_orbits_kvn` for the ecliptic fixture,
    /// captured before the options driven renderer replaced it.
    const LEGACY_WITH_COVARIANCE: &str = "CCSDS_OEM_VERS = 2.0\nCREATION_DATE = 2026-10-08T00:00:00\nORIGINATOR = TEST ORIGINATOR\n\nMETA_START\nOBJECT_NAME = TEST OBJECT\nOBJECT_ID = TEST OBJECT\nCENTER_NAME = SUN\nREF_FRAME = EME2000\nTIME_SYSTEM = TDB\nSTART_TIME = 2023-02-25T12:00:00.000\nSTOP_TIME = 2023-02-27T00:00:00.000\nMETA_STOP\n\n2023-02-25T12:00:00.000 1.34638083630000e+08 5.60914774672083e+07 2.10575789583866e+07 -1.55831115312500e+01 2.36909620400095e+01 1.06487257602539e+01\n2023-02-26T00:00:00.123 1.94477231910000e+08 9.55987320126335e+06 1.55583969564213e+07 6.92582734722222e+00 -1.91318404658101e+01 -8.10594965505896e+00\n2023-02-27T00:00:00.000 1.64557657770000e+08 -4.41513396443243e+07 -1.09893165176051e+07 1.73145683680556e+00 2.72124902061235e+01 1.12319034180726e+01\n\nCOVARIANCE_START\nEPOCH = 2023-02-27T00:00:00.000\n4.47590458359478e+04\n9.18314159531114e+02 4.45285267798468e+04\n2.61612597581151e+03 2.15877603187907e+02 4.49895648920489e+04\n3.65532228537091e-05 -8.10518309248941e-05 -3.05862433001877e-04 5.99588555544140e-10\n-8.10518309248941e-05 -1.29328986101921e-04 -2.02225334394984e-04 6.27257498216484e-12 5.90570501161385e-10\n-3.05862433001877e-04 -2.02225334394984e-04 5.11204081089073e-05 3.50475311361241e-11 8.44527128687302e-12 6.08606609926895e-10\nCOVARIANCE_STOP\n\n";
    const CREATION_DATE: &str = "2026-10-08T00:00:00";

    fn legacy_options() -> OemWriteOptions {
        OemWriteOptions::legacy("TEST ORIGINATOR", CREATION_DATE)
    }

    fn options_3() -> OemWriteOptions {
        OemWriteOptions {
            version: "3.0".to_string(),
            ..legacy_options()
        }
    }

    fn render(orbits: &OrbitBatch, options: &OemWriteOptions) -> String {
        oem_render_kvn(orbits, options).unwrap().0
    }

    fn render_error(orbits: &OrbitBatch, options: &OemWriteOptions) -> String {
        match oem_render_kvn(orbits, options).unwrap_err() {
            SchemaError::InvalidRecordBatch(message) => message,
            other => other.to_string(),
        }
    }

    fn set_state(orbits: &mut OrbitBatch, row: usize, state: [f64; 6]) {
        let crate::types::CoordinateValues::Cartesian(values) = &mut orbits.coordinates.values
        else {
            unreachable!("the fixture is Cartesian")
        };
        values[row] = state;
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("adam_core_oem_{}_{name}.oem", std::process::id()))
    }

    fn assert_close(actual: &[f64], expected: &[f64], rtol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() <= rtol * e.abs(), "{a} != {e}");
        }
    }

    #[test]
    fn legacy_options_reproduce_the_legacy_writer() {
        let orbits = fixture(Frame::Ecliptic, TimeScale::Tdb, true);
        assert_eq!(
            oem_render_kvn(&orbits, &legacy_options()).unwrap(),
            (LEGACY_WITH_COVARIANCE.to_string(), 0)
        );
    }

    #[test]
    fn version_3_layout() {
        let orbits = fixture(Frame::Equatorial, TimeScale::Tdb, true);
        let mut options = options_3();
        options.comments = vec!["first comment".to_string(), "second comment".to_string()];
        options.object_name = Some("NAME".to_string());
        options.covariance_frame = Some("icrf".to_string());
        let text = render(&orbits, &options);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(
            lines[..15],
            [
                "CCSDS_OEM_VERS = 3.0",
                "CREATION_DATE = 2026-10-08T00:00:00",
                "ORIGINATOR = TEST ORIGINATOR",
                "",
                "META_START",
                "COMMENT first comment",
                "COMMENT second comment",
                "OBJECT_NAME = NAME",
                "OBJECT_ID = TEST OBJECT",
                "CENTER_NAME = SUN",
                "REF_FRAME = ICRF",
                "TIME_SYSTEM = TDB",
                "START_TIME = 2023-02-25T12:00:00.000",
                "STOP_TIME = 2023-02-27T00:00:00.000",
                "META_STOP",
            ]
        );
        // A covariance_frame equal to REF_FRAME writes state frame blocks
        // without a COV_REF_FRAME line.
        assert!(text.contains("COVARIANCE_START\nEPOCH = 2023-02-27T00:00:00.000\n4."));
        assert_eq!(text.matches("EPOCH = ").count(), 1);
        assert!(!text.contains("COV_REF_FRAME"));
        // 16 significant digits for states and covariances.
        let data = text.split("META_STOP\n").nth(1).unwrap();
        let numbers: Vec<&str> = data
            .lines()
            .filter(|line| !line.is_empty() && !line.contains('=') && !line.contains("COVARIANCE"))
            .flat_map(str::split_whitespace)
            .filter(|token| !token.contains('T'))
            .collect();
        assert_eq!(numbers.len(), 3 * 6 + 21);
        for token in numbers {
            let mantissa = token.trim_start_matches('-').split('e').next().unwrap();
            assert_eq!(mantissa.len(), 17, "{token}");
        }

        // include_covariance false writes no block, but the frame name is still checked.
        let mut options = options_3();
        options.include_covariance = false;
        assert!(!render(&orbits, &options).contains("COVARIANCE_START"));
        options.covariance_frame = Some("lvlh".to_string());
        assert!(render_error(&orbits, &options).starts_with("Unknown local orbital frame 'LVLH'"));
    }

    #[test]
    fn validation_errors() {
        let orbits = fixture(Frame::Equatorial, TimeScale::Tdb, true);
        let legacy = legacy_options();

        let mut two_ids = orbits.clone();
        two_ids.object_id[2] = Some(ObjectId("OTHER".to_string()));
        assert_eq!(
            render_error(&two_ids, &legacy),
            "An OEM needs a non-null object_id and carries one object about one center per \
             file, got 2 object_ids and 1 origins."
        );
        // 2.0 writes the first center, as the legacy writer did, unless a
        // covariance frame is given.
        let mut two_origins = orbits.clone();
        two_origins.coordinates.origins.origins[1] = OriginId::SolarSystemBarycenter;
        assert!(render(&two_origins, &legacy).contains("\nCENTER_NAME = SUN\n"));
        let mut options = legacy_options();
        options.covariance_frame = Some("TNW".to_string());
        for options in [options, options_3()] {
            assert!(
                render_error(&two_origins, &options).ends_with("got 1 object_ids and 2 origins.")
            );
        }

        // 400 microseconds apart round to the same millisecond; 2.0 writes both.
        let mut duplicates = orbits.clone();
        duplicates.coordinates.times = Some(
            TimeArray::from_parts(
                TimeScale::Tdb,
                vec![60000, 60001, 60000],
                vec![0, 0, 400_000],
            )
            .unwrap(),
        );
        assert_eq!(
            render_error(&duplicates, &options_3()),
            "Epochs must be unique within an OEM after rounding to the millisecond, \
             2023-02-25T00:00:00.000 appears twice."
        );
        let text = render(&duplicates, &legacy);
        assert_eq!(text.matches("\n2023-02-25T00:00:00.000 ").count(), 2);

        let mut unspecified = orbits.clone();
        unspecified.coordinates.frame = Frame::Unspecified;
        assert!(render_error(&unspecified, &legacy).ends_with("transform first"));
        let mut options = options_3();
        options.version = "1.0".to_string();
        assert_eq!(
            render_error(&orbits, &options),
            "OEM version must be \"2.0\" or \"3.0\", got \"1.0\""
        );

        // CCSDS 7.3.2: 254 characters including "COMMENT ".
        let mut options = options_3();
        options.comments = vec!["x".repeat(246)];
        assert!(oem_render_kvn(&orbits, &options).is_ok());
        options.comments = vec!["x".repeat(247)];
        assert_eq!(
            render_error(&orbits, &options),
            "A COMMENT line must be at most 254 characters including the keyword \
             (CCSDS 502.0-B-3 7.3.2), got 255."
        );

        // CCSDS 7.3.4: every value in 3.0, comments and the caller's names in 2.0.
        let error = |key: &str| {
            format!("OEM value for {key} must be non-empty printable ASCII on one line.")
        };
        options.comments = vec!["two\nlines".to_string()];
        assert_eq!(render_error(&orbits, &options), error("COMMENT"));
        let mut options = options_3();
        options.originator = "\u{00c9}QUIPE".to_string();
        assert_eq!(render_error(&orbits, &options), error("ORIGINATOR"));
        let mut legacy = OemWriteOptions::legacy("TEST ORIGINATOR", "");
        assert!(render(&orbits, &legacy).contains("\nCREATION_DATE = \n"));
        legacy.object_name = Some("A\nCENTER_NAME = MOON".to_string());
        assert_eq!(render_error(&orbits, &legacy), error("OBJECT_NAME"));
    }

    #[test]
    fn off_grid_epochs_are_rounded_to_the_millisecond() {
        for (nanos, expected) in [
            (499_999, (0, 0)),
            (500_000, (0, 0)),
            (500_001, (0, 1_000_000)),
            (1_500_000, (0, 2_000_000)),
            (2_500_000, (0, 2_000_000)),
            (86_399_999_600_000, (1, 0)),
        ] {
            let rounded = round_to_millisecond(&Epoch::new(0, nanos));
            assert_eq!((rounded.days, rounded.nanos), expected, "{nanos}");
        }

        let on_grid = fixture(Frame::Equatorial, TimeScale::Tdb, true);
        let mut off_grid = on_grid.clone();
        off_grid.coordinates.times = Some(
            TimeArray::from_parts(
                TimeScale::Tdb,
                vec![60002, 60000, 60001],
                vec![400_000, 43_200_000_000_000, 122_600_000],
            )
            .unwrap(),
        );
        let (text, rounded) = oem_render_kvn(&off_grid, &options_3()).unwrap();
        assert_eq!(rounded, 2);
        assert_eq!(text, render(&on_grid, &options_3()));
        // Half a millisecond rounds to even in 3.0 and up in ERFA's
        // formatting, which 2.0 keeps, as the legacy writer did.
        let mut ties = on_grid.clone();
        ties.coordinates.times = Some(
            TimeArray::from_parts(
                TimeScale::Tdb,
                vec![60000, 60001, 60002],
                vec![500_000, 1_500_000, 0],
            )
            .unwrap(),
        );
        let (text, rounded) = oem_render_kvn(&ties, &legacy_options()).unwrap();
        assert_eq!(rounded, 0);
        assert!(text.contains("START_TIME = 2023-02-25T00:00:00.001\n"));
        assert!(render(&ties, &options_3()).contains("START_TIME = 2023-02-25T00:00:00.000\n"));
    }

    #[test]
    fn reader_accepts_version_3_files() {
        for label in ["ICRF", "J2000", "GCRF", "EME2000"] {
            assert_eq!(oem_to_adam_frame(label).unwrap(), Frame::Equatorial);
        }

        // A 3.0 file with comments and a state frame block reads back exactly.
        let orbits = fixture(Frame::Equatorial, TimeScale::Tdb, true);
        let mut options = options_3();
        options.comments = vec!["a comment".to_string()];
        let path = temp_path("version_3");
        assert_eq!(oem_write_kvn_file(&path, &orbits, &options).unwrap(), 0);
        let read = oem_read_orbits(&path).unwrap().unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(read.coordinates.frame, Frame::Equatorial);
        let times = read.coordinates.times.as_ref().unwrap();
        assert_eq!(times.scale, TimeScale::Tdb);
        let source = orbits.coordinates.times.as_ref().unwrap();
        for (row, i) in [1, 2, 0].into_iter().enumerate() {
            assert_eq!(times.epochs[row], source.epochs[i]);
            assert_close(
                &read.coordinates.values.cartesian().unwrap()[row],
                &orbits.coordinates.values.cartesian().unwrap()[i],
                4e-15,
            );
        }
        let covariance = read.coordinates.covariance.as_ref().unwrap();
        assert_eq!(
            (0..3)
                .map(|row| covariance.is_row_valid(row))
                .collect::<Vec<_>>(),
            [false, false, true]
        );
        let source = orbits.coordinates.covariance.as_ref().unwrap();
        assert_close(covariance.row_values(2), source.row_values(0), 4e-15);
    }

    #[test]
    fn local_frame_covariance_blocks() {
        let orbits = fixture(Frame::Equatorial, TimeScale::Tdb, true);
        let mut options = options_3();
        options.covariance_frame = Some("vnc_rotating".to_string());
        let text = render(&orbits, &options);
        assert!(text.contains(
            "META_START\nCOMMENT COV_REF_FRAME VNC_ROTATING follows the SANA orbit-relative \
             reference frames registry (CCSDS 502.0-B-3 annex B5).\nOBJECT_NAME"
        ));
        assert!(text.contains("EPOCH = 2023-02-27T00:00:00.000\nCOV_REF_FRAME = VNC_ROTATING\n"));
        assert_eq!(text.matches("COV_REF_FRAME = ").count(), 1);

        // The block is the kernel's rotation in km with the Sun's mu, and the
        // reader leaves it unjoined.
        let path = temp_path("vnc_rotating");
        std::fs::write(&path, &text).unwrap();
        let document = oem_parse_kvn_structured(&path).unwrap();
        let read = oem_read_orbits(&path).unwrap().unwrap();
        std::fs::remove_file(&path).unwrap();
        let expected = local_frame_covariances(
            &orbits.coordinates.values.cartesian().unwrap()[0],
            orbits
                .coordinates
                .covariance
                .as_ref()
                .unwrap()
                .row_values(0),
            &[origin_mu_au3_day2(&OriginId::from_code("SUN")).unwrap()],
            LocalFrame::parse("VNC_ROTATING").unwrap(),
        )
        .unwrap();
        let block = &document.segments[0].covariances[0];
        assert_eq!(block.frame.as_deref(), Some("VNC_ROTATING"));
        assert_close(
            &block.matrix,
            &convert_covariance_matrix(&expected, true),
            4e-15,
        );
        let covariance = read.coordinates.covariance.as_ref().unwrap();
        assert!((0..3).all(|row| !covariance.is_row_valid(row)));

        // Labels outside table 5-4 are written as their registry names; TNW
        // is in it, so it stays TNW with no note, also on 2.0.
        options.covariance_frame = Some("VNC".to_string());
        assert!(render(&orbits, &options).contains("\nCOV_REF_FRAME = VNC_INERTIAL\n"));
        let mut legacy = legacy_options();
        legacy.covariance_frame = Some(" tnw ".to_string());
        let text = render(&orbits, &legacy);
        assert_eq!(text.matches("COV_REF_FRAME = TNW\n").count(), 1);
        assert!(!text.contains("SANA"));
    }

    #[test]
    fn local_frame_errors() {
        let orbits = fixture(Frame::Equatorial, TimeScale::Tdb, true);
        let mut options = options_3();
        // An unknown label is reported before the table 5-4 rule.
        options.covariance_frame = Some("lvlh".to_string());
        options.table_frames_only = true;
        assert!(render_error(&orbits, &options).starts_with("Unknown local orbital frame 'LVLH'"));

        // table_frames_only, always on for 2.0, refuses names outside table 5-4.
        let note = "COV_REF_FRAME VNC_ROTATING follows the SANA orbit-relative reference frames \
                    registry (CCSDS 502.0-B-3 annex B5). It is outside the RSW, RTN, TNW set \
                    of table 5-4.";
        options.covariance_frame = Some("vnc_rotating".to_string());
        assert_eq!(
            render_error(&orbits, &options),
            format!("{note} Pass table_frames_only=False to write it anyway.")
        );
        let mut legacy = legacy_options();
        legacy.covariance_frame = Some("VNC_ROTATING".to_string());
        assert_eq!(
            render_error(&orbits, &legacy),
            format!("{note} Annex B5 is an OEM 3.0 provision, pass version=\"3.0\" to write it.")
        );

        options.table_frames_only = false;
        assert_eq!(
            render_error(&fixture(Frame::Equatorial, TimeScale::Tdb, false), &options),
            "The states carry no covariance."
        );
        let mut barycenter = orbits.clone();
        barycenter.coordinates.origins =
            OriginArray::repeat(OriginId::from_code("EARTH_MOON_BARYCENTER"), 3);
        assert_eq!(
            render_error(&barycenter, &options),
            "VNC_ROTATING needs the gravitational parameter of the center, which adam_core \
             does not have for EARTH_MOON_BARYCENTER."
        );

        // Kernel errors name the input row; row 0 is written last.
        let mut radial = orbits.clone();
        let state = orbits.coordinates.values.cartesian().unwrap()[0];
        set_state(
            &mut radial,
            0,
            [state[0], state[1], state[2], state[0], state[1], state[2]],
        );
        assert_eq!(
            render_error(&radial, &options),
            "State 0 is not finite or has no orbit plane."
        );
    }

    #[test]
    fn non_finite_values_are_refused_in_version_3() {
        let orbits = fixture(Frame::Equatorial, TimeScale::Tdb, true);
        let mut nan_state = orbits.clone();
        let mut state = orbits.coordinates.values.cartesian().unwrap()[2];
        state[4] = f64::NAN;
        set_state(&mut nan_state, 2, state);
        assert_eq!(
            render_error(&nan_state, &options_3()),
            "State 2 has a non-finite value."
        );
        // 2.0 writes it, as the legacy writer did.
        assert!(render(&nan_state, &legacy_options()).contains(" nan "));

        // One NaN in the lower triangle of the only block, in REF_FRAME or rotated.
        let mut nan_entry = orbits.clone();
        nan_entry
            .coordinates
            .covariance
            .as_mut()
            .unwrap()
            .values_row_major[6] = f64::NAN;
        let expected = "Covariance at epoch 2023-02-27T00:00:00.000 has a non-finite entry.";
        let mut options = options_3();
        assert_eq!(render_error(&nan_entry, &options), expected);
        options.covariance_frame = Some("TNW".to_string());
        assert_eq!(render_error(&nan_entry, &options), expected);
        assert!(render(&nan_entry, &legacy_options()).contains("\nnan "));
        // The all-NaN row 1 is skipped silently.
        assert_eq!(render(&orbits, &options).matches("EPOCH = ").count(), 1);
    }

    #[test]
    fn py_sci_matches_python_format() {
        assert_eq!(py_sci(0.0, 14), "0.00000000000000e+00");
        assert_eq!(py_sci(123456.789, 14), "1.23456789000000e+05");
        assert_eq!(py_sci(-1.5e-7, 14), "-1.50000000000000e-07");
        assert_eq!(py_sci(std::f64::consts::TAU, 14), "6.28318530717959e+00");
        assert_eq!(py_sci(std::f64::consts::TAU, 15), "6.283185307179586e+00");
    }

    #[test]
    fn epoch_format_parse_round_trip() {
        // The writer is millisecond precision (legacy Time.precision = 3),
        // so round-trips are exact for ms-representable epochs.
        for &(days, nanos) in &[
            (60000_i64, 0_i64),
            (60000, 43_200_000_000_000),
            (60000, 123_000_000_000),
            (53734, 86_399_000_000_000),
        ] {
            for &scale in &[TimeScale::Tdb, TimeScale::Utc, TimeScale::Tt] {
                let isot = format_epoch(days, nanos, scale).unwrap();
                let (rt_days, rt_nanos) = parse_epoch(&isot, scale).unwrap();
                assert_eq!((rt_days, rt_nanos), (days, nanos), "{isot} {scale:?}");
            }
        }
    }
}
