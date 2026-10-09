//! CCSDS OEM (Orbit Ephemeris Message) KVN writer/parser (bead personal-cmy.28).
//!
//! Replaces the third-party Python `oem` package for the surfaces adam-core
//! uses: writing single-segment KVN files (header, metadata, state lines,
//! optional lower-triangle covariance blocks) and parsing KVN files back into
//! per-segment arrays. Formatting replicates the `oem` package byte-for-byte:
//!
//! * floats via Python `f"{v:.14e}"` (two-digit signed exponent);
//! * state/covariance epochs via astropy `Time.strftime("%Y-%m-%dT%H:%M:%S.%f")`
//!   (legacy astropy default millisecond precision through ERFA `d2dtf`, with
//!   astropy's `day_frac` two-sum jd splitting replicated exactly);
//! * header/metadata `KEY = value` lines in insertion order (serde_json
//!   preserve_order), `META_START`/`META_STOP`, blank-line separators, and
//!   `COV_REF_FRAME` emitted only when it differs from `REF_FRAME`.
//!
//! The parser mirrors the package's semantics for the files adam-core reads:
//! epoch strings -> ERFA `dtf2d` -> the exact `Timestamp.from_astropy`
//! integer split; lower-triangle covariance reconstruction to a symmetric
//! 6x6; `COMMENT` lines skipped; multiple segments supported.
//!
//! OEM 3.0 output (CCSDS 502.0-B-3) differs in REF_FRAME ICRF, 16 significant
//! digits, epochs rounded to the millisecond, COMMENT lines, stricter
//! validation and covariance blocks in a local orbital frame.

use crate::local_frames::{local_frame_covariances, LocalFrame};
use crate::types::{
    origin_mu_au3_day2, CoordinateBatch, CoordinateRepresentation, CovarianceBatch,
    CovarianceUnits, Epoch, Frame, ObjectId, OrbitBatch, OrbitId, OriginArray, OriginId,
    SchemaError, SchemaResult, TimeArray, TimeScale, Validity, KM_PER_AU, SECONDS_PER_DAY,
};
use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::Path;

const NANOS_PER_DAY_F64: f64 = 86_400e9;
const MJD_JD_OFFSET: f64 = 2_400_000.5;

fn invalid(message: String) -> SchemaError {
    SchemaError::InvalidRecordBatch(message)
}

// --- float / epoch formatting ---------------------------------------------------

/// Python `f"{value:.{decimals}e}"`.
fn py_sci(value: f64, decimals: usize) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    let raw = format!("{value:.decimals$e}");
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

// --- writer -----------------------------------------------------------------------

/// One segment's covariance record: epoch + frame + 21 lower-triangle values (km).
pub struct OemCovarianceRecord {
    pub days: i64,
    pub nanos: i64,
    pub frame: String,
    pub lower_triangle: [f64; 21],
}

/// Render a single-segment KVN OEM byte-identically to the Python `oem`
/// package's `save_as(..., file_format="kvn")` for the structures adam-core
/// writes. `header` and `metadata` are insertion-ordered `KEY -> value`
/// JSON objects (values rendered verbatim). `comments` follow META_START and
/// numbers carry `decimals + 1` significant digits.
pub fn oem_to_kvn(
    header_json: &str,
    metadata_json: &str,
    time_scale: TimeScale,
    days: &[i64],
    nanos: &[i64],
    states_km: &[f64],
    covariances: &[OemCovarianceRecord],
    comments: &[String],
    decimals: usize,
) -> SchemaResult<String> {
    let header: serde_json::Map<String, serde_json::Value> = serde_json::from_str(header_json)
        .map_err(|err| invalid(format!("invalid OEM header payload: {err}")))?;
    let metadata: serde_json::Map<String, serde_json::Value> = serde_json::from_str(metadata_json)
        .map_err(|err| invalid(format!("invalid OEM metadata payload: {err}")))?;
    if states_km.len() != days.len() * 6 || nanos.len() != days.len() {
        return Err(invalid("states/days/nanos length mismatch".to_string()));
    }

    let scalar = |value: &serde_json::Value| -> String {
        match value {
            serde_json::Value::String(text) => text.clone(),
            other => other.to_string(),
        }
    };

    let mut out = String::new();
    // HeaderSection._to_string: CCSDS_OEM_VERS first, remaining fields in
    // order, then a trailing newline; _to_kvn_oem adds one blank line.
    if let Some(version) = header.get("CCSDS_OEM_VERS") {
        let _ = writeln!(out, "CCSDS_OEM_VERS = {}", scalar(version));
    }
    let mut remaining = Vec::new();
    for (key, value) in &header {
        if key != "CCSDS_OEM_VERS" {
            remaining.push(format!("{key} = {}", scalar(value)));
        }
    }
    out.push_str(&remaining.join("\n"));
    out.push('\n');
    out.push('\n');

    // MetaDataSection._to_string + segment separator newline.
    out.push_str("META_START\n");
    for comment in comments {
        let _ = writeln!(out, "COMMENT {comment}");
    }
    let meta_lines: Vec<String> = metadata
        .iter()
        .map(|(key, value)| format!("{key} = {}", scalar(value)))
        .collect();
    out.push_str(&meta_lines.join("\n"));
    out.push('\n');
    out.push_str("META_STOP\n");
    out.push('\n');

    for row in 0..days.len() {
        let epoch = format_epoch(days[row], nanos[row], time_scale)?;
        let _ = write!(out, "{epoch} ");
        let state = &states_km[row * 6..row * 6 + 6];
        let rendered: Vec<String> = state.iter().map(|&value| py_sci(value, decimals)).collect();
        out.push_str(&rendered.join(" "));
        out.push('\n');
    }
    out.push('\n');

    if !covariances.is_empty() {
        let ref_frame = metadata.get("REF_FRAME").map(scalar).unwrap_or_default();
        out.push_str("COVARIANCE_START\n");
        for record in covariances {
            let epoch = format_epoch(record.days, record.nanos, time_scale)?;
            let _ = writeln!(out, "EPOCH = {epoch}");
            if record.frame != ref_frame {
                let _ = writeln!(out, "COV_REF_FRAME = {}", record.frame);
            }
            let cov = &record.lower_triangle;
            let rows: [&[f64]; 6] = [
                &cov[0..1],
                &cov[1..3],
                &cov[3..6],
                &cov[6..10],
                &cov[10..15],
                &cov[15..21],
            ];
            for row in rows {
                let rendered: Vec<String> =
                    row.iter().map(|&value| py_sci(value, decimals)).collect();
                out.push_str(&rendered.join(" "));
                out.push('\n');
            }
        }
        out.push_str("COVARIANCE_STOP\n");
        out.push('\n');
    }

    Ok(out)
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

/// CCSDS OEM version written by the fused product writer; mirrors the
/// public `adam_core.orbits.oem_io.OEM_VERSION` compatibility constant.
const OEM_VERSION: &str = "2.0";

/// Covariance frame labels of table 5-4, written as given.
const TABLE_COVARIANCE_FRAMES: [&str; 3] = ["RSW", "RTN", "TNW"];

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

/// `_oem_to_adam_frame` (exact error message); ICRF, J2000 and GCRF read as equatorial.
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

/// Options of [`oem_render_orbits_kvn`]; the default is the legacy OEM 2.0 product.
#[derive(Debug, Clone, PartialEq)]
pub struct OemWriteOptions {
    /// "2.0" or "3.0".
    pub version: String,
    /// OBJECT_NAME and OBJECT_ID; None or empty means the orbits' object_id.
    pub object_name: Option<String>,
    pub object_id: Option<String>,
    /// COMMENT lines written right after META_START.
    pub comments: Vec<String>,
    pub include_covariance: bool,
    /// None or the REF_FRAME label writes the state covariance, any other
    /// label names a local orbital frame.
    pub covariance_frame: Option<String>,
    /// Refuse covariance frames outside table 5-4 (always on for 2.0).
    pub table_frames_only: bool,
}

impl Default for OemWriteOptions {
    fn default() -> Self {
        Self {
            version: OEM_VERSION.to_string(),
            object_name: None,
            object_id: None,
            comments: Vec::new(),
            include_covariance: true,
            covariance_frame: None,
            table_frames_only: false,
        }
    }
}

/// Fused legacy `orbit_to_oem` tail: equatorial rotation (ecliptic input),
/// stable time sort, metadata assembly, AU->km conversion, covariance
/// extraction, and KVN rendering, all in Rust. The caller performs the
/// legacy Python-side assertions and the SPICE-dependent ITRF93 transform.
/// Returns the text and the number of epochs OEM 3.0 rounded to the millisecond.
pub fn oem_render_orbits_kvn(
    orbits: &OrbitBatch,
    originator: &str,
    creation_date: &str,
    options: &OemWriteOptions,
) -> SchemaResult<(String, usize)> {
    use serde_json::{Map, Value};

    let version_3 = match options.version.as_str() {
        "2.0" => false,
        "3.0" => true,
        other => {
            return Err(invalid(format!(
                "OEM version must be \"2.0\" or \"3.0\", got \"{other}\""
            )))
        }
    };
    let orbits = match orbits.coordinates.frame {
        Frame::Equatorial => orbits.clone(),
        Frame::Ecliptic => orbits.rotate_frame(Frame::Equatorial)?,
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

    // 3.0 rounds to the millisecond, ties to even, so EPOCH strings read back exactly.
    let step = if version_3 { 1_000_000 } else { 1 };
    let round = |nanos: i64| (nanos as f64 / step as f64).round_ties_even() as i64 * step;
    let epochs: Vec<Epoch> = (times.epochs.iter())
        .map(|epoch| Epoch::new(epoch.days, round(epoch.nanos)))
        .collect();
    let rounded = (0..n).filter(|&i| epochs[i] != times.epochs[i]).count();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| (epochs[i].days, epochs[i].nanos));
    let scale = times.scale;

    // One object about one center per file; 2.0 without a covariance frame
    // takes the first center.
    let object_ids: HashSet<Option<&str>> = (orbits.object_id.iter())
        .map(|id| id.as_ref().map(|id| id.0.as_str()))
        .collect();
    let origin = &orbits.coordinates.origins.origins[0];
    let centers: HashSet<String> = (orbits.coordinates.origins.origins.iter())
        .map(OriginId::code)
        .collect();
    let one_center = centers.len() == 1 || !(version_3 || options.covariance_frame.is_some());
    let object_id = match object_ids.iter().next() {
        Some(Some(object_id)) if object_ids.len() == 1 && one_center => object_id.to_string(),
        _ => {
            return Err(invalid(format!(
                "An OEM needs a non-null object_id and carries one object about one center \
                 per file, got {} object_ids and {} origins.",
                object_ids.len(),
                centers.len()
            )))
        }
    };
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
    let oem_frame = if version_3 { "ICRF" } else { "EME2000" };
    // Parsed whenever given, so a bad name fails without a covariance block too.
    let local_frame =
        match (options.covariance_frame.as_deref()).map(|label| label.trim().to_uppercase()) {
            Some(label) if label != oem_frame => Some((LocalFrame::parse(&label)?, label)),
            _ => None,
        };
    let center = adam_to_oem_center(&origin.code())?;
    let first = *order.first().expect("non-empty");
    let last = *order.last().expect("non-empty");
    let start = format_epoch(epochs[first].days, epochs[first].nanos, scale)?;
    let stop = format_epoch(epochs[last].days, epochs[last].nanos, scale)?;

    let mut header = Map::new();
    header.insert(
        "CCSDS_OEM_VERS".to_string(),
        Value::String(options.version.clone()),
    );
    header.insert(
        "CREATION_DATE".to_string(),
        Value::String(creation_date.to_string()),
    );
    header.insert(
        "ORIGINATOR".to_string(),
        Value::String(originator.to_string()),
    );

    let names = [&options.object_name, &options.object_id]
        .map(|name| name.as_deref().filter(|text| !text.is_empty()));
    let mut metadata = Map::new();
    for (key, name) in ["OBJECT_NAME", "OBJECT_ID"].into_iter().zip(names) {
        let value = name.unwrap_or(&object_id).to_string();
        metadata.insert(key.to_string(), Value::String(value));
    }
    metadata.insert("CENTER_NAME".to_string(), Value::String(center.to_string()));
    metadata.insert(
        "REF_FRAME".to_string(),
        Value::String(oem_frame.to_string()),
    );
    metadata.insert(
        "TIME_SYSTEM".to_string(),
        Value::String(time_system_upper(scale)),
    );
    metadata.insert("START_TIME".to_string(), Value::String(start));
    metadata.insert("STOP_TIME".to_string(), Value::String(stop));

    let header_json =
        serde_json::to_string(&header).map_err(|err| invalid(format!("encode failed: {err}")))?;
    let metadata_json =
        serde_json::to_string(&metadata).map_err(|err| invalid(format!("encode failed: {err}")))?;

    // The top left 6x6 block of each valid covariance (9x9 rows carry the
    // non-gravitational parameters after the state) unless it is all NaN.
    let covariance = orbits.coordinates.covariance.as_ref();
    let mut rows: Vec<Option<[f64; 36]>> = (0..n)
        .map(|i| {
            let covariance = covariance.filter(|covariance| {
                options.include_covariance
                    && covariance.dimension >= 6
                    && covariance.is_row_valid(i)
            })?;
            let (values, dimension) = (covariance.row_values(i), covariance.dimension);
            let matrix: [f64; 36] = std::array::from_fn(|k| values[k / 6 * dimension + k % 6]);
            (!matrix.iter().all(|value| value.is_nan())).then_some(matrix)
        })
        .collect();
    let mut comments = options.comments.clone();
    let mut label = oem_frame.to_string();
    if let Some((frame, given)) = local_frame.filter(|_| options.include_covariance) {
        if rows.iter().all(Option::is_none) {
            return Err(invalid("The states carry no covariance.".to_string()));
        }
        label = if TABLE_COVARIANCE_FRAMES.contains(&given.as_str()) {
            given
        } else {
            let name = frame.canonical_name();
            let note = format!(
                "COV_REF_FRAME {name} follows the SANA orbit-relative reference frames \
                 registry (CCSDS 502.0-B-3 annex B5)."
            );
            if !version_3 || options.table_frames_only {
                let remedy = match version_3 {
                    true => "Pass table_frames_only=False to write it anyway.",
                    false => "Annex B5 is an OEM 3.0 provision, pass version=\"3.0\" to write it.",
                };
                return Err(invalid(format!(
                    "{note} It is outside the RSW, RTN, TNW set of table 5-4. {remedy}"
                )));
            }
            comments.push(note);
            name.to_string()
        };
        let mu = match frame.rotating {
            true => origin_mu_au3_day2(origin).map_err(|_| {
                invalid(format!(
                    "{} needs the gravitational parameter of the center, which adam_core \
                     does not have for {}.",
                    frame.canonical_name(),
                    origin.code()
                ))
            })?,
            false => 0.0,
        };
        let flat_values: Vec<f64> = values.iter().flatten().copied().collect();
        let flat: Vec<f64> = rows
            .iter()
            .flat_map(|row| row.unwrap_or([f64::NAN; 36]))
            .collect();
        let local = local_frame_covariances(&flat_values, &flat, &vec![mu; n], frame)?;
        for (row, rotated) in rows.iter_mut().zip(local.chunks_exact(36)) {
            if let Some(matrix) = row {
                matrix.copy_from_slice(rotated);
            }
        }
    }

    let mut days = Vec::with_capacity(n);
    let mut nanos = Vec::with_capacity(n);
    let mut states_km = Vec::with_capacity(n * 6);
    let mut covariances = Vec::new();
    for &i in &order {
        days.push(epochs[i].days);
        nanos.push(epochs[i].nanos);
        states_km.extend_from_slice(&values_au_to_km(&values[i]));
        if let Some(matrix) = rows[i] {
            let matrix_km = convert_covariance_matrix(&matrix, true);
            let mut lower_triangle = [0.0_f64; 21];
            for (index, &(row, col)) in LOWER_TRIANGLE_INDICES.iter().enumerate() {
                lower_triangle[index] = matrix_km[row * 6 + col];
            }
            if version_3 && !lower_triangle.iter().all(|value| value.is_finite()) {
                return Err(invalid(format!(
                    "Covariance at epoch {} has a non-finite entry.",
                    format_epoch(epochs[i].days, epochs[i].nanos, scale)?
                )));
            }
            covariances.push(OemCovarianceRecord {
                days: epochs[i].days,
                nanos: epochs[i].nanos,
                frame: label.clone(),
                lower_triangle,
            });
        }
    }

    // KVN values are printable ASCII on one line (7.3.4): every header and
    // metadata value in 3.0, the caller's names and the comments always.
    let given = ["OBJECT_NAME", "OBJECT_ID"].into_iter().zip(names);
    let kvn_values = (header.iter().chain(&metadata))
        .filter(|_| version_3)
        .map(|(key, value)| (key.as_str(), value.as_str().unwrap_or_default()))
        .chain(given.filter_map(|(key, name)| Some((key, name?))))
        .chain(comments.iter().map(|comment| ("COMMENT", comment.as_str())));
    for (key, value) in kvn_values {
        if value.trim().is_empty() || !value.bytes().all(|byte| (b' '..=b'~').contains(&byte)) {
            return Err(invalid(format!(
                "OEM value for {key} must be non-empty printable ASCII on one line."
            )));
        }
    }
    for comment in &comments {
        let length = "COMMENT ".len() + comment.len();
        if length > 254 {
            return Err(invalid(format!(
                "A COMMENT line must be at most 254 characters including the keyword \
                 (CCSDS 502.0-B-3 7.3.2), got {length}."
            )));
        }
    }

    let decimals = if version_3 { 15 } else { 14 };
    let text = oem_to_kvn(
        &header_json,
        &metadata_json,
        scale,
        &days,
        &nanos,
        &states_km,
        &covariances,
        &comments,
        decimals,
    )?;
    Ok((text, rounded))
}

/// Render and write the fused OEM product (see [`oem_render_orbits_kvn`]);
/// returns the number of epochs rounded to the millisecond.
pub fn oem_write_orbits_kvn(
    path: &Path,
    orbits: &OrbitBatch,
    originator: &str,
    creation_date: &str,
    options: &OemWriteOptions,
) -> SchemaResult<usize> {
    let (text, rounded) = oem_render_orbits_kvn(orbits, originator, creation_date, options)?;
    std::fs::write(path, text)
        .map_err(|err| invalid(format!("failed to write {}: {err}", path.display())))?;
    Ok(rounded)
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

    #[test]
    fn writer_layout_matches_oem_package_shape() {
        let text = oem_to_kvn(
            "{\"CCSDS_OEM_VERS\": \"3.0\", \"CREATION_DATE\": \"2026-01-01T00:00:00\", \"ORIGINATOR\": \"TEST\"}",
            "{\"OBJECT_NAME\": \"X\", \"OBJECT_ID\": \"X\", \"CENTER_NAME\": \"SUN\", \"REF_FRAME\": \"EME2000\", \"TIME_SYSTEM\": \"TDB\", \"START_TIME\": \"t0\", \"STOP_TIME\": \"t1\"}",
            TimeScale::Tdb,
            &[60000],
            &[0],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[],
            &[],
            14,
        )
        .unwrap();
        let expected = "CCSDS_OEM_VERS = 3.0\nCREATION_DATE = 2026-01-01T00:00:00\nORIGINATOR = TEST\n\nMETA_START\nOBJECT_NAME = X\nOBJECT_ID = X\nCENTER_NAME = SUN\nREF_FRAME = EME2000\nTIME_SYSTEM = TDB\nSTART_TIME = t0\nSTOP_TIME = t1\nMETA_STOP\n\n2023-02-25T00:00:00.000 1.00000000000000e+00 2.00000000000000e+00 3.00000000000000e+00 4.00000000000000e+00 5.00000000000000e+00 6.00000000000000e+00\n\n";
        assert_eq!(text, expected);
    }

    /// Three unsorted states of one object about the Sun. Row 1 has an
    /// all-NaN covariance and row 2 an invalid one, so only row 0 has a block.
    fn fixture(frame: Frame) -> OrbitBatch {
        let values = vec![
            [1.1, -0.3, 0.05, 0.001, 0.017, -0.0003],
            [0.9, 0.4, -0.02, -0.009, 0.015, 0.0002],
            [1.3, 0.1, 0.07, 0.004, -0.012, 0.0001],
        ];
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
        let covariance = CovarianceBatch::new(
            3,
            6,
            matrices,
            CovarianceUnits::Coordinate(CoordinateRepresentation::Cartesian),
        )
        .unwrap()
        .with_row_validity(Validity::from_bools(&[true, true, false]))
        .unwrap();
        let times = TimeArray::from_parts(
            TimeScale::Tdb,
            vec![60002, 60000, 60001],
            vec![0, 43_200_000_000_000, 123_000_000],
        );
        let coordinates = CoordinateBatch::cartesian(
            values,
            frame,
            OriginArray::repeat(OriginId::from_code("SUN"), 3),
            Some(times.unwrap()),
            Some(covariance),
        )
        .unwrap();
        let object_ids = vec![Some(ObjectId("TEST OBJECT".to_string())); 3];
        let orbit_ids = (0..3).map(|i| OrbitId(format!("o{i}"))).collect();
        OrbitBatch::new(orbit_ids, object_ids, coordinates).unwrap()
    }

    /// Output of the legacy writer for the ecliptic fixture.
    const LEGACY_WITH_COVARIANCE: &str = "CCSDS_OEM_VERS = 2.0\nCREATION_DATE = 2026-10-08T00:00:00\nORIGINATOR = TEST ORIGINATOR\n\nMETA_START\nOBJECT_NAME = TEST OBJECT\nOBJECT_ID = TEST OBJECT\nCENTER_NAME = SUN\nREF_FRAME = EME2000\nTIME_SYSTEM = TDB\nSTART_TIME = 2023-02-25T12:00:00.000\nSTOP_TIME = 2023-02-27T00:00:00.000\nMETA_STOP\n\n2023-02-25T12:00:00.000 1.34638083630000e+08 5.60914774672083e+07 2.10575789583866e+07 -1.55831115312500e+01 2.36909620400095e+01 1.06487257602539e+01\n2023-02-26T00:00:00.123 1.94477231910000e+08 9.55987320126335e+06 1.55583969564213e+07 6.92582734722222e+00 -1.91318404658101e+01 -8.10594965505896e+00\n2023-02-27T00:00:00.000 1.64557657770000e+08 -4.41513396443243e+07 -1.09893165176051e+07 1.73145683680556e+00 2.72124902061235e+01 1.12319034180726e+01\n\nCOVARIANCE_START\nEPOCH = 2023-02-27T00:00:00.000\n4.47590458359478e+04\n9.18314159531114e+02 4.45285267798468e+04\n2.61612597581151e+03 2.15877603187907e+02 4.49895648920489e+04\n3.65532228537091e-05 -8.10518309248941e-05 -3.05862433001877e-04 5.99588555544140e-10\n-8.10518309248941e-05 -1.29328986101921e-04 -2.02225334394984e-04 6.27257498216484e-12 5.90570501161385e-10\n-3.05862433001877e-04 -2.02225334394984e-04 5.11204081089073e-05 3.50475311361241e-11 8.44527128687302e-12 6.08606609926895e-10\nCOVARIANCE_STOP\n\n";

    fn render(orbits: &OrbitBatch, options: &OemWriteOptions) -> SchemaResult<(String, usize)> {
        oem_render_orbits_kvn(orbits, "TEST ORIGINATOR", "2026-10-08T00:00:00", options)
    }

    fn options_3() -> OemWriteOptions {
        let version = "3.0".to_string();
        OemWriteOptions {
            version,
            ..Default::default()
        }
    }

    fn states(orbits: &mut OrbitBatch) -> &mut Vec<[f64; 6]> {
        match &mut orbits.coordinates.values {
            crate::types::CoordinateValues::Cartesian(values) => values,
            _ => unreachable!("the fixture is Cartesian"),
        }
    }

    #[test]
    fn default_options_reproduce_the_legacy_writer() {
        let text = render(&fixture(Frame::Ecliptic), &OemWriteOptions::default()).unwrap();
        assert_eq!(text, (LEGACY_WITH_COVARIANCE.to_string(), 0));
    }

    #[test]
    fn version_3_layout() {
        let options = OemWriteOptions {
            object_name: Some("NAME".to_string()),
            comments: vec!["a comment".to_string()],
            covariance_frame: Some("vnc_rotating".to_string()),
            ..options_3()
        };
        let (text, _) = render(&fixture(Frame::Equatorial), &options).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(
            lines[..16],
            [
                "CCSDS_OEM_VERS = 3.0",
                "CREATION_DATE = 2026-10-08T00:00:00",
                "ORIGINATOR = TEST ORIGINATOR",
                "",
                "META_START",
                "COMMENT a comment",
                "COMMENT COV_REF_FRAME VNC_ROTATING follows the SANA orbit-relative reference \
                 frames registry (CCSDS 502.0-B-3 annex B5).",
                "OBJECT_NAME = NAME",
                "OBJECT_ID = TEST OBJECT",
                "CENTER_NAME = SUN",
                "REF_FRAME = ICRF",
                "TIME_SYSTEM = TDB",
                "START_TIME = 2023-02-25T12:00:00.000",
                "STOP_TIME = 2023-02-27T00:00:00.000",
                "META_STOP",
                "",
            ]
        );
        assert!(text.contains("\nEPOCH = 2023-02-27T00:00:00.000\nCOV_REF_FRAME = VNC_ROTATING\n"));
        assert_eq!(text.matches("COV_REF_FRAME = ").count(), 1);
        // 16 significant digits for the 3 states and the 21 covariance entries.
        let numbers: Vec<&str> = (text.split_whitespace())
            .filter(|token| token.contains('e') && token.parse::<f64>().is_ok())
            .collect();
        assert_eq!(numbers.len(), 3 * 6 + 21);
        for token in numbers {
            assert_eq!(token.trim_start_matches('-').find('e'), Some(17), "{token}");
        }
    }

    #[test]
    fn version_3_rounds_epochs_ties_to_even_and_refuses_duplicates() {
        let mut orbits = fixture(Frame::Equatorial);
        let nanos = vec![500_000, 1_500_000, 86_399_999_600_000];
        let times = TimeArray::from_parts(TimeScale::Tdb, vec![60000, 60001, 60002], nanos);
        orbits.coordinates.times = Some(times.unwrap());
        let (text, rounded) = render(&orbits, &options_3()).unwrap();
        assert_eq!(rounded, 3);
        for epoch in [
            "2023-02-25T00:00:00.000",
            "2023-02-26T00:00:00.002",
            "2023-02-28T00:00:00.000",
        ] {
            assert!(text.contains(&format!("\n{epoch} ")), "{epoch}");
        }
        // 2.0 keeps ERFA's formatting, which rounds half a millisecond up.
        let (text, rounded) = render(&orbits, &OemWriteOptions::default()).unwrap();
        assert_eq!(rounded, 0);
        assert!(text.contains("\n2023-02-25T00:00:00.001 "));

        // 400 microseconds apart round to the same millisecond; 2.0 writes both.
        let times = TimeArray::from_parts(
            TimeScale::Tdb,
            vec![60000, 60001, 60000],
            vec![0, 0, 400_000],
        );
        orbits.coordinates.times = Some(times.unwrap());
        let error = "Epochs must be unique within an OEM after rounding to the millisecond, \
                     2023-02-25T00:00:00.000 appears twice.";
        assert_eq!(
            render(&orbits, &options_3()),
            Err(invalid(error.to_string()))
        );
        let (text, _) = render(&orbits, &OemWriteOptions::default()).unwrap();
        assert_eq!(text.matches("\n2023-02-25T00:00:00.000 ").count(), 2);
    }

    #[test]
    fn reader_accepts_icrf_and_ignores_local_frame_blocks() {
        let orbits = fixture(Frame::Equatorial);
        let path = std::env::temp_dir().join(format!("adam_core_oem_{}.oem", std::process::id()));
        for (frame, valid) in [(None, [false, false, true]), (Some("RSW"), [false; 3])] {
            let covariance_frame = frame.map(str::to_string);
            let options = OemWriteOptions {
                covariance_frame,
                ..options_3()
            };
            oem_write_orbits_kvn(&path, &orbits, "O", "2026-10-08T00:00:00", &options).unwrap();
            let read = oem_read_orbits(&path).unwrap().unwrap();
            assert_eq!(read.coordinates.frame, Frame::Equatorial);
            let source = &orbits.coordinates.times.as_ref().unwrap().epochs;
            assert_eq!(
                read.coordinates.times.unwrap().epochs,
                [1, 2, 0].map(|i| source[i])
            );
            let covariance = read.coordinates.covariance.unwrap();
            let rows: Vec<bool> = (0..3).map(|row| covariance.is_row_valid(row)).collect();
            assert_eq!(rows, valid);
        }
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn validation_errors() {
        type Case = fn(&mut OrbitBatch, &mut OemWriteOptions);
        let note = "COV_REF_FRAME VNC_ROTATING follows the SANA orbit-relative reference frames \
                    registry (CCSDS 502.0-B-3 annex B5). It is outside the RSW, RTN, TNW set of \
                    table 5-4.";
        let one_center = "An OEM needs a non-null object_id and carries one object about one \
                          center per file, got";
        let ascii = "must be non-empty printable ASCII on one line.";
        let cases: [(Case, String); 15] = [
            (
                |_, o| o.version = "1.0".into(),
                r#"OEM version must be "2.0" or "3.0", got "1.0""#.into(),
            ),
            (
                |b, _| b.coordinates.frame = Frame::Unspecified,
                "OEM writer requires equatorial or ecliptic coordinates, got Unspecified; \
                 transform first"
                    .into(),
            ),
            (
                |b, _| b.object_id[2] = Some(ObjectId("OTHER".into())),
                format!("{one_center} 2 object_ids and 1 origins."),
            ),
            (
                |b, o| {
                    b.coordinates.origins.origins[1] = OriginId::SolarSystemBarycenter;
                    *o = OemWriteOptions {
                        covariance_frame: Some("TNW".into()),
                        ..Default::default()
                    };
                },
                format!("{one_center} 1 object_ids and 2 origins."),
            ),
            (
                |b, _| states(b)[2][4] = f64::NAN,
                "State 2 has a non-finite value.".into(),
            ),
            (
                |_, o| o.covariance_frame = Some(" lvlh ".into()),
                "Unknown local orbital frame 'LVLH', expected one of [\"RSW_ROTATING\", \
                 \"RSW_INERTIAL\", \"TNW_ROTATING\", \"TNW_INERTIAL\", \"VNC_ROTATING\", \
                 \"VNC_INERTIAL\"]"
                    .into(),
            ),
            (
                |b, o| (b.coordinates.covariance, o.covariance_frame) = (None, Some("TNW".into())),
                "The states carry no covariance.".into(),
            ),
            (
                |_, o| {
                    (o.covariance_frame, o.table_frames_only) = (Some("vnc_rotating".into()), true)
                },
                format!("{note} Pass table_frames_only=False to write it anyway."),
            ),
            (
                |_, o| {
                    (o.covariance_frame, o.version) = (Some("VNC_ROTATING".into()), "2.0".into())
                },
                format!(
                    "{note} Annex B5 is an OEM 3.0 provision, pass version=\"3.0\" to write it."
                ),
            ),
            (
                |b, o| {
                    let origin = OriginId::from_code("EARTH_MOON_BARYCENTER");
                    b.coordinates.origins = OriginArray::repeat(origin, 3);
                    o.covariance_frame = Some("VNC_ROTATING".into());
                },
                "VNC_ROTATING needs the gravitational parameter of the center, which adam_core \
                 does not have for EARTH_MOON_BARYCENTER."
                    .into(),
            ),
            (
                |b, o| {
                    let state = &mut states(b)[0];
                    state.copy_within(0..3, 3);
                    o.covariance_frame = Some("VNC".into());
                },
                "State 0 is not finite or has no orbit plane.".into(),
            ),
            (
                |b, _| b.coordinates.covariance.as_mut().unwrap().values_row_major[6] = f64::NAN,
                "Covariance at epoch 2023-02-27T00:00:00.000 has a non-finite entry.".into(),
            ),
            (
                |b, _| b.object_id = vec![Some(ObjectId("\u{c9}".into())); 3],
                format!("OEM value for OBJECT_NAME {ascii}"),
            ),
            (
                |_, o| (o.version, o.comments) = ("2.0".into(), vec!["two\nlines".into()]),
                format!("OEM value for COMMENT {ascii}"),
            ),
            (
                |_, o| o.comments = vec!["x".repeat(247)],
                "A COMMENT line must be at most 254 characters including the keyword \
                 (CCSDS 502.0-B-3 7.3.2), got 255."
                    .into(),
            ),
        ];
        for (case, expected) in cases {
            let (mut orbits, mut options) = (fixture(Frame::Equatorial), options_3());
            case(&mut orbits, &mut options);
            assert_eq!(render(&orbits, &options).unwrap_err(), invalid(expected));
        }
        // 2.0 writes what 3.0 refuses.
        let mut orbits = fixture(Frame::Equatorial);
        states(&mut orbits)[2][4] = f64::NAN;
        orbits.coordinates.origins.origins[1] = OriginId::SolarSystemBarycenter;
        orbits.object_id = vec![Some(ObjectId("\u{c9}".into())); 3];
        assert!(render(&orbits, &OemWriteOptions::default()).is_ok());
        let options = OemWriteOptions {
            comments: vec!["x".repeat(246)],
            ..options_3()
        };
        assert!(render(&fixture(Frame::Equatorial), &options).is_ok());
    }
}
