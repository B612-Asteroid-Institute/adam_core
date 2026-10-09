//! CCSDS OEM (Orbit Ephemeris Message) KVN writer/parser (bead personal-cmy.28).
//!
//! Replaces the third-party Python `oem` package for the surfaces adam-core
//! uses: writing single-segment KVN files (header, metadata, state lines,
//! optional lower-triangle covariance blocks) and parsing KVN files back into
//! per-segment arrays. The layout follows the `oem` package:
//!
//! * floats via Python `f"{v:.{d}e}"` with 15 significant digits for the legacy
//!   product and up to 16 (CCSDS 7.5.7) for the message writer (two-digit signed exponent);
//! * state/covariance epochs via astropy `Time.strftime("%Y-%m-%dT%H:%M:%S.%f")`
//!   (legacy astropy default millisecond precision through ERFA `d2dtf`, with
//!   astropy's `day_frac` two-sum jd splitting replicated exactly);
//! * header/metadata `KEY = value` lines in order, `META_START`/`META_STOP`,
//!   blank-line separators, and `COV_REF_FRAME` emitted only when it differs
//!   from `REF_FRAME`.
//!
//! [`oem_render_kvn`] writes an `OrbitBatch` as OEM 2.0 or 3.0 from one
//! options struct. Version 2.0 with [`OemWriteOptions::legacy`] is the legacy
//! product and matches the `oem` package byte for byte: epochs are formatted
//! as given and values are written verbatim. Version 3.0 rounds epochs to the
//! millisecond first, ties to even, so state and covariance `EPOCH` strings
//! agree, and refuses duplicate epochs, mixed centers, non-finite values and
//! KVN values that are not printable ASCII.
//!
//! The parser mirrors the package's semantics for the files adam-core reads:
//! epoch strings -> ERFA `dtf2d` -> the exact `Timestamp.from_astropy`
//! integer split; lower-triangle covariance reconstruction to a symmetric
//! 6x6; `COMMENT` lines skipped; multiple segments supported.

use crate::local_frames::{local_frame_covariances, LocalFrame};
use crate::types::{
    CoordinateBatch, CoordinateRepresentation, CovarianceBatch, CovarianceUnits, Epoch, Frame,
    ObjectId, OrbitBatch, OrbitId, OriginArray, OriginId, SchemaError, SchemaResult, TimeArray,
    TimeScale, Validity, KM_PER_AU, SECONDS_PER_DAY,
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

// --- writer -----------------------------------------------------------------------

/// One segment's covariance record: epoch + frame + 21 lower-triangle values (km).
struct OemCovarianceRecord {
    days: i64,
    nanos: i64,
    frame: String,
    lower_triangle: [f64; 21],
}

/// The KVN layout of the `oem` package's `save_as(..., file_format="kvn")`;
/// values are written verbatim and `comments` become `COMMENT` lines right
/// after `META_START`.
fn render_kvn(
    header: &[(String, String)],
    comments: &[String],
    metadata: &[(String, String)],
    time_scale: TimeScale,
    days: &[i64],
    nanos: &[i64],
    states_km: &[f64],
    covariances: &[OemCovarianceRecord],
    significant_digits: usize,
) -> SchemaResult<String> {
    // CCSDS 502.0-B-3 7.5.7 allows a mantissa of at most 16 digits.
    if !(1..=16).contains(&significant_digits) {
        return Err(invalid(format!(
            "significant_digits must be between 1 and 16, got {significant_digits}"
        )));
    }
    let decimals = significant_digits - 1;
    if states_km.len() != days.len() * 6 || nanos.len() != days.len() {
        return Err(invalid("states/days/nanos length mismatch".to_string()));
    }

    let mut out = String::new();
    // HeaderSection._to_string: CCSDS_OEM_VERS first, remaining fields in
    // order, then a trailing newline; _to_kvn_oem adds one blank line.
    if let Some((_, version)) = header.iter().find(|(key, _)| key == "CCSDS_OEM_VERS") {
        let _ = writeln!(out, "CCSDS_OEM_VERS = {version}");
    }
    let remaining: Vec<String> = header
        .iter()
        .filter(|(key, _)| key != "CCSDS_OEM_VERS")
        .map(|(key, value)| format!("{key} = {value}"))
        .collect();
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
        .map(|(key, value)| format!("{key} = {value}"))
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
        let ref_frame = metadata
            .iter()
            .find(|(key, _)| key == "REF_FRAME")
            .map(|(_, value)| value.as_str())
            .unwrap_or_default();
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

fn write_text(path: &Path, text: &str) -> SchemaResult<()> {
    std::fs::write(path, text)
        .map_err(|err| invalid(format!("failed to write {}: {err}", path.display())))
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

/// Covariance frame labels of CCSDS 502.0-B-3 table 5-4.
const TABLE_COVARIANCE_FRAMES: [&str; 3] = ["RSW", "RTN", "TNW"];
/// CCSDS 502.0-B-3 7.3.2: a COMMENT line, keyword included, is at most 254 characters.
const MAX_COMMENT_LINE: usize = 254;
const NANOS_PER_MILLISECOND: i64 = 1_000_000;
const OPTION_KEYS: [&str; 11] = [
    "version",
    "originator",
    "creation_date",
    "object_name",
    "object_id",
    "comments",
    "include_covariance",
    "covariance_frame",
    "table_frames_only",
    "significant_digits",
    "ref_frame_label",
];

/// Options of [`oem_render_kvn`].
#[derive(Debug, Clone, PartialEq)]
pub struct OemWriteOptions {
    /// `CCSDS_OEM_VERS`, "2.0" or "3.0".
    pub version: String,
    pub originator: String,
    /// `CREATION_DATE`, written verbatim.
    pub creation_date: String,
    /// `OBJECT_NAME`; None or empty means the orbits' object_id.
    pub object_name: Option<String>,
    /// `OBJECT_ID`; None or empty means the orbits' object_id.
    pub object_id: Option<String>,
    /// `COMMENT` lines written right after `META_START`.
    pub comments: Vec<String>,
    /// False writes no covariance block.
    pub include_covariance: bool,
    /// Covariance label, trimmed and uppercased; None or REF_FRAME writes the
    /// state covariance, other labels are local orbital frames.
    pub covariance_frame: Option<String>,
    /// Refuse covariance labels outside the RSW, RTN, TNW set of table 5-4.
    pub table_frames_only: bool,
    /// Mantissa digits of states and covariances, 1 to 16.
    pub significant_digits: usize,
    /// `REF_FRAME`, trimmed and uppercased: EME2000, ICRF, J2000 or GCRF.
    /// None or empty means EME2000 (2.0) or ICRF (3.0).
    pub ref_frame_label: Option<String>,
}

impl OemWriteOptions {
    /// The legacy product: OEM 2.0, EME2000, 15 digits, state covariance blocks.
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
            significant_digits: 15,
            ref_frame_label: None,
        }
    }

    /// Options from a JSON object. Missing or null keys take the defaults:
    /// version "2.0", originator "ADAM CORE USER", creation_date the current
    /// UTC time, include_covariance true, table_frames_only false,
    /// significant_digits 15 for 2.0 and 16 for 3.0, the rest None or empty.
    pub fn from_json(json: &str) -> SchemaResult<Self> {
        use serde_json::Value;

        let payload: Value = serde_json::from_str(json)
            .map_err(|err| invalid(format!("invalid OEM options payload: {err}")))?;
        let map = payload
            .as_object()
            .ok_or_else(|| invalid("OEM options must be a JSON object".to_string()))?;
        if let Some(key) = map.keys().find(|key| !OPTION_KEYS.contains(&key.as_str())) {
            return Err(invalid(format!(
                "Unknown OEM option '{key}', expected one of {}.",
                OPTION_KEYS.join(", ")
            )));
        }
        let get = |key: &str| map.get(key).filter(|value| !value.is_null());
        let wrong = |key: &str, kind: &str, value: &Value| {
            invalid(format!("OEM option '{key}' must be {kind}, got {value}"))
        };
        let text = |key: &str| -> SchemaResult<Option<String>> {
            match get(key) {
                None => Ok(None),
                Some(Value::String(text)) => Ok(Some(text.clone())),
                Some(other) => Err(wrong(key, "a string", other)),
            }
        };
        let flag = |key: &str, default: bool| -> SchemaResult<bool> {
            match get(key) {
                None => Ok(default),
                Some(Value::Bool(flag)) => Ok(*flag),
                Some(other) => Err(wrong(key, "a boolean", other)),
            }
        };

        let version = text("version")?.unwrap_or_else(|| "2.0".to_string());
        let significant_digits = match get("significant_digits") {
            None if version == "3.0" => 16,
            None => 15,
            Some(value) => value
                .as_u64()
                .and_then(|digits| usize::try_from(digits).ok())
                .ok_or_else(|| wrong("significant_digits", "a non-negative integer", value))?,
        };
        let comments = match get("comments") {
            None => Vec::new(),
            Some(value) => value
                .as_array()
                .and_then(|items| {
                    items
                        .iter()
                        .map(|item| item.as_str().map(str::to_string))
                        .collect::<Option<Vec<_>>>()
                })
                .ok_or_else(|| wrong("comments", "a list of strings", value))?,
        };
        let creation_date = match text("creation_date")? {
            Some(date) => date,
            None => utc_now()?,
        };
        Ok(Self {
            originator: text("originator")?.unwrap_or_else(|| "ADAM CORE USER".to_string()),
            creation_date,
            object_name: text("object_name")?,
            object_id: text("object_id")?,
            comments,
            include_covariance: flag("include_covariance", true)?,
            covariance_frame: text("covariance_frame")?,
            table_frames_only: flag("table_frames_only", false)?,
            significant_digits,
            ref_frame_label: text("ref_frame_label")?,
            version,
        })
    }
}

/// The current UTC time as `YYYY-MM-DDTHH:MM:SS`.
fn utc_now() -> SchemaResult<String> {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs() as i64);
    // Unix day 0 is MJD 40587.
    let text = format_epoch(
        40_587 + seconds / 86_400,
        seconds % 86_400 * 1_000_000_000,
        TimeScale::Utc,
    )?;
    Ok(text[..19].to_string())
}

/// [`oem_render_kvn`] output: the KVN text and the number of epochs that
/// were moved onto the millisecond grid.
#[derive(Debug, Clone, PartialEq)]
pub struct OemRendered {
    pub text: String,
    pub off_grid_epochs: usize,
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

/// Python `repr` of a list of optional strings.
fn py_list<'a>(items: impl IntoIterator<Item = Option<&'a str>>) -> String {
    let items: Vec<String> = items
        .into_iter()
        .map(|item| item.map_or("None".to_string(), |text| format!("'{text}'")))
        .collect();
    format!("[{}]", items.join(", "))
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

/// A km covariance record from a 6x6 covariance in AU units.
fn covariance_record(epoch: Epoch, frame: &str, matrix_au: &[f64]) -> OemCovarianceRecord {
    let matrix_km = convert_covariance_matrix(matrix_au, true);
    let mut lower_triangle = [0.0_f64; 21];
    for (index, &(row, col)) in LOWER_TRIANGLE_INDICES.iter().enumerate() {
        lower_triangle[index] = matrix_km[row * 6 + col];
    }
    OemCovarianceRecord {
        days: epoch.days,
        nanos: epoch.nanos,
        frame: frame.to_string(),
        lower_triangle,
    }
}

/// Render one object's states as a single-segment KVN OEM. Ecliptic input
/// is rotated to equatorial, states are stably sorted by time (3.0 rounds
/// the epochs to the millisecond first), and the covariance is written in
/// REF_FRAME or rotated into a local orbital frame. `mu` (AU^3/day^2, one
/// entry per input row) is required for `_ROTATING` covariance frames.
/// Errors name states by their input row.
pub fn oem_render_kvn(
    orbits: &OrbitBatch,
    options: &OemWriteOptions,
    mu: Option<&[f64]>,
) -> SchemaResult<OemRendered> {
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
                "An OEM needs an object_id and carries one object about one center per file, \
                 got object_ids {} and origins {}.",
                py_list(object_ids.iter().copied()),
                py_list(origins.iter().map(|code| Some(code.as_str()))),
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

    let non_empty = |value: &Option<String>| {
        value
            .as_deref()
            .filter(|text| !text.is_empty())
            .map(str::to_string)
    };
    let ref_frame = match non_empty(&options.ref_frame_label) {
        None if version_3 => "ICRF".to_string(),
        None => "EME2000".to_string(),
        Some(label) => {
            let allowed = ["EME2000", "ICRF", "J2000", "GCRF"];
            let upper = label.trim().to_uppercase();
            if !allowed.contains(&upper.as_str()) {
                return Err(invalid(format!(
                    "ref_frame_label {label} is not valid for equatorial states, expected one of {}",
                    py_list(allowed.iter().map(|name| Some(*name)))
                )));
            }
            upper
        }
    };
    // Parsed whenever given, so a bad name fails without a covariance block too.
    let local_frame = match options
        .covariance_frame
        .as_deref()
        .map(|label| label.trim().to_uppercase())
    {
        Some(label) if label != ref_frame => Some((LocalFrame::parse(&label)?, label)),
        _ => None,
    };
    let center = adam_to_oem_center(&orbits.coordinates.origins.origins[0].code())?;
    let first = epochs[order[0]];
    let last = epochs[order[n - 1]];
    let header = [
        ("CCSDS_OEM_VERS", options.version.clone()),
        ("CREATION_DATE", options.creation_date.clone()),
        ("ORIGINATOR", options.originator.clone()),
    ]
    .map(|(key, value)| (key.to_string(), value));
    let metadata = [
        (
            "OBJECT_NAME",
            non_empty(&options.object_name).unwrap_or_else(|| object_id.clone()),
        ),
        (
            "OBJECT_ID",
            non_empty(&options.object_id).unwrap_or_else(|| object_id.clone()),
        ),
        ("CENTER_NAME", center.to_string()),
        ("REF_FRAME", ref_frame.clone()),
        ("TIME_SYSTEM", time_system_upper(scale)),
        ("START_TIME", format_epoch(first.days, first.nanos, scale)?),
        ("STOP_TIME", format_epoch(last.days, last.nanos, scale)?),
    ]
    .map(|(key, value)| (key.to_string(), value));

    let mut comments = options.comments.clone();
    let mut records = Vec::new();
    if options.include_covariance {
        let covariance = orbits.coordinates.covariance.as_ref();
        // Input row order, so kernel errors name the caller's rows.
        let rows: Vec<Option<[f64; 36]>> =
            (0..n).map(|i| state_covariance(covariance, i)).collect();
        let (label, local) = match local_frame {
            None => (ref_frame.clone(), None),
            Some((local_frame, label)) => {
                if rows.iter().all(Option::is_none) {
                    return Err(invalid("The states carry no covariance.".to_string()));
                }
                // Table 5-4 labels are written as given, any other as its registry name.
                let label = if TABLE_COVARIANCE_FRAMES.contains(&label.as_str()) {
                    label
                } else {
                    let name = local_frame.canonical_name();
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
                let mu = match mu {
                    Some(mu) if mu.len() != n => {
                        return Err(invalid(format!(
                            "mu must have one entry per state, got {} for {n} states",
                            mu.len()
                        )))
                    }
                    Some(mu) => mu,
                    None if local_frame.rotating => {
                        return Err(invalid(
                            "mu is required for a _ROTATING covariance frame".to_string(),
                        ))
                    }
                    None => &[],
                };
                let flat_values: Vec<f64> = values.iter().flatten().copied().collect();
                let flat_covariances: Vec<f64> = rows
                    .iter()
                    .flat_map(|row| row.unwrap_or([f64::NAN; 36]))
                    .collect();
                let local =
                    local_frame_covariances(&flat_values, &flat_covariances, mu, local_frame)?;
                (label, Some(local))
            }
        };
        for &i in &order {
            if let Some(matrix) = &rows[i] {
                let matrix = local
                    .as_ref()
                    .map_or(&matrix[..], |local| &local[i * 36..(i + 1) * 36]);
                records.push(covariance_record(epochs[i], &label, matrix));
            }
        }
    }
    if version_3 {
        if let Some(record) = records
            .iter()
            .find(|record| !record.lower_triangle.iter().all(|value| value.is_finite()))
        {
            return Err(invalid(format!(
                "Covariance at epoch {} has a non-finite entry.",
                format_epoch(record.days, record.nanos, scale)?
            )));
        }
    }

    // CCSDS 502.0-B-3 7.3.4 for comments and the caller's names, and for every
    // header and metadata value in 3.0; 2.0 writes the rest verbatim, as the
    // legacy writer did.
    let names = [
        ("OBJECT_NAME", non_empty(&options.object_name)),
        ("OBJECT_ID", non_empty(&options.object_id)),
    ];
    let kvn_values = header
        .iter()
        .chain(&metadata)
        .filter(|_| version_3)
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .chain(
            names
                .iter()
                .filter_map(|(key, value)| Some((*key, value.as_deref()?))),
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

    let mut days = Vec::with_capacity(n);
    let mut nanos = Vec::with_capacity(n);
    let mut states_km = Vec::with_capacity(n * 6);
    for &i in &order {
        days.push(epochs[i].days);
        nanos.push(epochs[i].nanos);
        states_km.extend_from_slice(&values_au_to_km(&values[i]));
    }
    let text = render_kvn(
        &header,
        &comments,
        &metadata,
        scale,
        &days,
        &nanos,
        &states_km,
        &records,
        options.significant_digits,
    )?;
    Ok(OemRendered {
        text,
        off_grid_epochs,
    })
}

/// Render (see [`oem_render_kvn`]) and write the file; returns the number of
/// epochs moved onto the millisecond grid.
pub fn oem_write_kvn_file(
    path: &Path,
    orbits: &OrbitBatch,
    options: &OemWriteOptions,
    mu: Option<&[f64]>,
) -> SchemaResult<usize> {
    let rendered = oem_render_kvn(orbits, options, mu)?;
    write_text(path, &rendered.text)?;
    Ok(rendered.off_grid_epochs)
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

    /// Output of the legacy `oem_render_orbits_kvn` for the fixtures, captured
    /// before the options driven renderer replaced it.
    const LEGACY_WITH_COVARIANCE: &str = "CCSDS_OEM_VERS = 2.0\nCREATION_DATE = 2026-10-08T00:00:00\nORIGINATOR = TEST ORIGINATOR\n\nMETA_START\nOBJECT_NAME = TEST OBJECT\nOBJECT_ID = TEST OBJECT\nCENTER_NAME = SUN\nREF_FRAME = EME2000\nTIME_SYSTEM = TDB\nSTART_TIME = 2023-02-25T12:00:00.000\nSTOP_TIME = 2023-02-27T00:00:00.000\nMETA_STOP\n\n2023-02-25T12:00:00.000 1.34638083630000e+08 5.60914774672083e+07 2.10575789583866e+07 -1.55831115312500e+01 2.36909620400095e+01 1.06487257602539e+01\n2023-02-26T00:00:00.123 1.94477231910000e+08 9.55987320126335e+06 1.55583969564213e+07 6.92582734722222e+00 -1.91318404658101e+01 -8.10594965505896e+00\n2023-02-27T00:00:00.000 1.64557657770000e+08 -4.41513396443243e+07 -1.09893165176051e+07 1.73145683680556e+00 2.72124902061235e+01 1.12319034180726e+01\n\nCOVARIANCE_START\nEPOCH = 2023-02-27T00:00:00.000\n4.47590458359478e+04\n9.18314159531114e+02 4.45285267798468e+04\n2.61612597581151e+03 2.15877603187907e+02 4.49895648920489e+04\n3.65532228537091e-05 -8.10518309248941e-05 -3.05862433001877e-04 5.99588555544140e-10\n-8.10518309248941e-05 -1.29328986101921e-04 -2.02225334394984e-04 6.27257498216484e-12 5.90570501161385e-10\n-3.05862433001877e-04 -2.02225334394984e-04 5.11204081089073e-05 3.50475311361241e-11 8.44527128687302e-12 6.08606609926895e-10\nCOVARIANCE_STOP\n\n";
    const LEGACY_WITHOUT_COVARIANCE: &str = "CCSDS_OEM_VERS = 2.0\nCREATION_DATE = 2026-10-08T00:00:00\nORIGINATOR = TEST ORIGINATOR\n\nMETA_START\nOBJECT_NAME = TEST OBJECT\nOBJECT_ID = TEST OBJECT\nCENTER_NAME = SUN\nREF_FRAME = EME2000\nTIME_SYSTEM = UTC\nSTART_TIME = 2023-02-25T12:00:00.000\nSTOP_TIME = 2023-02-27T00:00:00.000\nMETA_STOP\n\n2023-02-25T12:00:00.000 1.34638083630000e+08 5.98391482800000e+07 -2.99195741400000e+06 -1.55831115312500e+01 2.59718525520833e+01 3.46291367361111e-01\n2023-02-26T00:00:00.123 1.94477231910000e+08 1.49597870700000e+07 1.04718509490000e+07 6.92582734722222e+00 -2.07774820416667e+01 1.73145683680556e-01\n2023-02-27T00:00:00.000 1.64557657770000e+08 -4.48793612100000e+07 7.47989353500000e+06 1.73145683680556e+00 2.94347662256944e+01 -5.19437051041667e-01\n\n";
    const CREATION_DATE: &str = "2026-10-08T00:00:00";
    const MU_SUN: f64 = 2.959_122_082_841_2e-4;

    fn legacy_options() -> OemWriteOptions {
        OemWriteOptions::legacy("TEST ORIGINATOR", CREATION_DATE)
    }

    fn options_3() -> OemWriteOptions {
        OemWriteOptions {
            version: "3.0".to_string(),
            significant_digits: 16,
            ..legacy_options()
        }
    }

    fn message(err: SchemaError) -> String {
        match err {
            SchemaError::InvalidRecordBatch(message) => message,
            other => other.to_string(),
        }
    }

    fn render_error(orbits: &OrbitBatch, options: &OemWriteOptions) -> String {
        message(oem_render_kvn(orbits, options, None).unwrap_err())
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
        for (orbits, expected) in [
            (
                fixture(Frame::Ecliptic, TimeScale::Tdb, true),
                LEGACY_WITH_COVARIANCE,
            ),
            (
                fixture(Frame::Equatorial, TimeScale::Utc, false),
                LEGACY_WITHOUT_COVARIANCE,
            ),
        ] {
            let rendered = oem_render_kvn(&orbits, &legacy_options(), None).unwrap();
            assert_eq!(rendered.text, expected);
            assert_eq!(rendered.off_grid_epochs, 0);
        }
        // The JSON defaults are the legacy options.
        let from_json = OemWriteOptions::from_json(
            r#"{"originator": "TEST ORIGINATOR", "creation_date": "2026-10-08T00:00:00"}"#,
        )
        .unwrap();
        assert_eq!(from_json, legacy_options());
    }

    #[test]
    fn version_3_layout() {
        let orbits = fixture(Frame::Equatorial, TimeScale::Tdb, true);
        let mut options = options_3();
        options.comments = vec!["first comment".to_string(), "second comment".to_string()];
        options.object_name = Some("NAME".to_string());
        options.covariance_frame = Some("icrf".to_string());
        let rendered = oem_render_kvn(&orbits, &options, None).unwrap();
        let text = rendered.text;
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

        // With 15 digits only the version and frame labels differ from legacy.
        let mut options = options_3();
        options.significant_digits = 15;
        options.covariance_frame = Some("ICRF".to_string());
        let text = oem_render_kvn(
            &fixture(Frame::Ecliptic, TimeScale::Tdb, true),
            &options,
            None,
        )
        .unwrap()
        .text;
        assert_eq!(
            text,
            LEGACY_WITH_COVARIANCE
                .replace("CCSDS_OEM_VERS = 2.0", "CCSDS_OEM_VERS = 3.0")
                .replace("REF_FRAME = EME2000", "REF_FRAME = ICRF")
        );

        // An explicit REF_FRAME label, matched case insensitively.
        let mut options = options_3();
        options.ref_frame_label = Some("EME2000".to_string());
        options.covariance_frame = Some("eme2000".to_string());
        let text = oem_render_kvn(&orbits, &options, None).unwrap().text;
        assert!(text.contains("REF_FRAME = EME2000\n"));
        assert_eq!(text.matches("EPOCH = ").count(), 1);
        assert!(!text.contains("COV_REF_FRAME"));

        // include_covariance false writes no block, but the frame name is still checked.
        let mut options = options_3();
        options.include_covariance = false;
        let text = oem_render_kvn(&orbits, &options, None).unwrap().text;
        assert!(!text.contains("COVARIANCE_START"));
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
            "An OEM needs an object_id and carries one object about one center per file, \
             got object_ids ['TEST OBJECT', 'OTHER'] and origins ['SUN']."
        );
        let mut no_ids = orbits.clone();
        no_ids.object_id = vec![None; 3];
        assert_eq!(
            render_error(&no_ids, &options_3()),
            "An OEM needs an object_id and carries one object about one center per file, \
             got object_ids [None] and origins ['SUN']."
        );
        let mut two_origins = orbits.clone();
        two_origins.coordinates.origins.origins[1] = OriginId::SolarSystemBarycenter;
        assert_eq!(
            render_error(&two_origins, &options_3()),
            "An OEM needs an object_id and carries one object about one center per file, \
             got object_ids ['TEST OBJECT'] and origins ['SUN', 'SOLAR_SYSTEM_BARYCENTER']."
        );
        // 2.0 writes the first center, as the legacy writer did, unless a
        // covariance frame is given.
        let text = oem_render_kvn(&two_origins, &legacy, None).unwrap().text;
        assert!(text.contains("\nCENTER_NAME = SUN\n"));
        let mut options = legacy_options();
        options.covariance_frame = Some("TNW".to_string());
        assert!(render_error(&two_origins, &options).contains("one object about one center"));

        // 400 microseconds apart round to the same millisecond.
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
        // 2.0 writes both, as the legacy writer did.
        let text = oem_render_kvn(&duplicates, &legacy, None).unwrap().text;
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
        let mut options = options_3();
        options.significant_digits = 17;
        assert_eq!(
            render_error(&orbits, &options),
            "significant_digits must be between 1 and 16, got 17"
        );

        // CCSDS 7.3.2: 254 characters including "COMMENT ".
        let mut options = options_3();
        options.comments = vec!["x".repeat(246)];
        assert!(oem_render_kvn(&orbits, &options, None).is_ok());
        options.comments = vec!["x".repeat(247)];
        assert_eq!(
            render_error(&orbits, &options),
            "A COMMENT line must be at most 254 characters including the keyword \
             (CCSDS 502.0-B-3 7.3.2), got 255."
        );
        options.comments = vec!["two\nlines".to_string()];
        assert_eq!(
            render_error(&orbits, &options),
            "OEM value for COMMENT must be non-empty printable ASCII on one line."
        );

        let mut options = options_3();
        options.covariance_frame = Some("vnc_rotating".to_string());
        options.table_frames_only = true;
        assert_eq!(
            render_error(&orbits, &options),
            "COV_REF_FRAME VNC_ROTATING follows the SANA orbit-relative reference frames \
             registry (CCSDS 502.0-B-3 annex B5). It is outside the RSW, RTN, TNW set of \
             table 5-4. Pass table_frames_only=False to write it anyway."
        );
        options.table_frames_only = false;
        assert_eq!(
            render_error(&fixture(Frame::Equatorial, TimeScale::Tdb, false), &options),
            "The states carry no covariance."
        );
        // 2.0 takes the table 5-4 frames only.
        let mut options = legacy_options();
        options.covariance_frame = Some("VNC".to_string());
        assert_eq!(
            render_error(&orbits, &options),
            "COV_REF_FRAME VNC_INERTIAL follows the SANA orbit-relative reference frames \
             registry (CCSDS 502.0-B-3 annex B5). It is outside the RSW, RTN, TNW set of \
             table 5-4. Annex B5 is an OEM 3.0 provision, pass version=\"3.0\" to write it."
        );
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
        let options = options_3();
        let rendered = oem_render_kvn(&off_grid, &options, None).unwrap();
        assert_eq!(rendered.off_grid_epochs, 2);
        assert_eq!(
            rendered.text,
            oem_render_kvn(&on_grid, &options, None).unwrap().text
        );
    }

    #[test]
    fn options_from_json() {
        let options = OemWriteOptions::from_json(
            r#"{"version": "3.0", "originator": "O", "creation_date": "D",
                "object_name": null, "object_id": "ID", "comments": ["a", "b"],
                "include_covariance": true, "covariance_frame": "VNC_ROTATING",
                "table_frames_only": true, "ref_frame_label": "ICRF"}"#,
        )
        .unwrap();
        assert_eq!(
            options,
            OemWriteOptions {
                version: "3.0".to_string(),
                originator: "O".to_string(),
                creation_date: "D".to_string(),
                object_name: None,
                object_id: Some("ID".to_string()),
                comments: vec!["a".to_string(), "b".to_string()],
                include_covariance: true,
                covariance_frame: Some("VNC_ROTATING".to_string()),
                table_frames_only: true,
                significant_digits: 16,
                ref_frame_label: Some("ICRF".to_string()),
            }
        );

        let defaults = OemWriteOptions::from_json("{}").unwrap();
        let date = defaults.creation_date.clone();
        assert_eq!(defaults, OemWriteOptions::legacy("ADAM CORE USER", &date));
        assert_eq!(date.len(), 19);
        assert_eq!((&date[4..5], &date[10..11], &date[16..17]), ("-", "T", ":"));
        assert!(date.as_str() > "2026-01-01");
        let digits = OemWriteOptions::from_json(r#"{"significant_digits": 12}"#).unwrap();
        assert_eq!(digits.significant_digits, 12);

        let error = |json: &str| message(OemWriteOptions::from_json(json).unwrap_err());
        assert_eq!(
            error(r#"{"frame": "ICRF"}"#),
            "Unknown OEM option 'frame', expected one of version, originator, creation_date, \
             object_name, object_id, comments, include_covariance, covariance_frame, \
             table_frames_only, significant_digits, ref_frame_label."
        );
        assert_eq!(
            error(r#"{"version": 3}"#),
            "OEM option 'version' must be a string, got 3"
        );
        assert_eq!(
            error(r#"{"include_covariance": "yes"}"#),
            "OEM option 'include_covariance' must be a boolean, got \"yes\""
        );
        assert_eq!(
            error(r#"{"comments": "a"}"#),
            "OEM option 'comments' must be a list of strings, got \"a\""
        );
        assert_eq!(
            error(r#"{"comments": ["a", 1]}"#),
            "OEM option 'comments' must be a list of strings, got [\"a\",1]"
        );
        assert_eq!(
            error(r#"{"significant_digits": 15.5}"#),
            "OEM option 'significant_digits' must be a non-negative integer, got 15.5"
        );
        assert_eq!(error("[]"), "OEM options must be a JSON object");
    }

    #[test]
    fn reader_accepts_version_3_files() {
        for label in ["ICRF", "J2000", "GCRF", "EME2000"] {
            assert_eq!(oem_to_adam_frame(label).unwrap(), Frame::Equatorial);
        }
        assert_eq!(
            message(oem_to_adam_frame("TOD").unwrap_err()),
            "Unsupported OEM frame: TOD. Supported frames are \
             ['EME2000', 'ICRF', 'J2000', 'GCRF', 'ITRF-93']."
        );

        // A 3.0 file with comments and a state frame block reads back exactly.
        let orbits = fixture(Frame::Equatorial, TimeScale::Tdb, true);
        let mut options = options_3();
        options.comments = vec!["a comment".to_string()];
        let path = temp_path("version_3");
        assert_eq!(
            oem_write_kvn_file(&path, &orbits, &options, None).unwrap(),
            0
        );
        let read = oem_read_orbits(&path).unwrap().unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(read.coordinates.frame, Frame::Equatorial);
        let times = read.coordinates.times.as_ref().unwrap();
        assert_eq!(times.scale, TimeScale::Tdb);
        let order = [1, 2, 0];
        let source = orbits.coordinates.times.as_ref().unwrap();
        for (row, &i) in order.iter().enumerate() {
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
        assert_close(
            covariance.row_values(2),
            orbits
                .coordinates
                .covariance
                .as_ref()
                .unwrap()
                .row_values(0),
            4e-15,
        );

        // A block in another frame is kept by the parser but not joined.
        let header = [("CCSDS_OEM_VERS", "3.0"), ("ORIGINATOR", "T")]
            .map(|(key, value)| (key.to_string(), value.to_string()));
        let metadata = [
            ("OBJECT_NAME", "X"),
            ("OBJECT_ID", "X"),
            ("CENTER_NAME", "SUN"),
            ("REF_FRAME", "ICRF"),
            ("TIME_SYSTEM", "TDB"),
            ("START_TIME", "2023-02-25T00:00:00.000"),
            ("STOP_TIME", "2023-02-26T00:00:00.000"),
        ]
        .map(|(key, value)| (key.to_string(), value.to_string()));
        let identity: Vec<f64> = (0..36)
            .map(|index| if index % 7 == 0 { 1.0 } else { 0.0 })
            .collect();
        let records = [(60000, "ICRF"), (60001, "TNW")]
            .map(|(days, frame)| covariance_record(Epoch::new(days, 0), frame, &identity));
        let text = render_kvn(
            &header,
            &["a comment".to_string()],
            &metadata,
            TimeScale::Tdb,
            &[60000, 60001],
            &[0, 0],
            &[
                1.0e8, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 1.0e8, 0.0, -30.0, 0.0, 0.0,
            ],
            &records,
            16,
        )
        .unwrap();
        assert!(text.contains("META_START\nCOMMENT a comment\nOBJECT_NAME = X\n"));
        assert!(text.contains("EPOCH = 2023-02-26T00:00:00.000\nCOV_REF_FRAME = TNW\n"));
        assert_eq!(text.matches("COV_REF_FRAME").count(), 1);
        let path = temp_path("foreign_block");
        std::fs::write(&path, text).unwrap();
        let document = oem_parse_kvn_structured(&path).unwrap();
        let read = oem_read_orbits(&path).unwrap().unwrap();
        std::fs::remove_file(&path).unwrap();
        let frames: Vec<Option<&str>> = document.segments[0]
            .covariances
            .iter()
            .map(|covariance| covariance.frame.as_deref())
            .collect();
        assert_eq!(frames, [Some("ICRF"), Some("TNW")]);
        let covariance = read.coordinates.covariance.as_ref().unwrap();
        assert!(covariance.is_row_valid(0) && !covariance.is_row_valid(1));
    }

    #[test]
    fn local_frame_covariance_blocks() {
        let orbits = fixture(Frame::Equatorial, TimeScale::Tdb, true);
        let mu = [MU_SUN; 3];
        let mut options = options_3();
        options.covariance_frame = Some("vnc_rotating".to_string());
        assert_eq!(
            render_error(&orbits, &options),
            "mu is required for a _ROTATING covariance frame"
        );
        assert_eq!(
            message(oem_render_kvn(&orbits, &options, Some(&mu[..2])).unwrap_err()),
            "mu must have one entry per state, got 2 for 3 states"
        );
        let text = oem_render_kvn(&orbits, &options, Some(&mu)).unwrap().text;
        assert!(text.contains(
            "META_START\nCOMMENT COV_REF_FRAME VNC_ROTATING follows the SANA orbit-relative \
             reference frames registry (CCSDS 502.0-B-3 annex B5).\nOBJECT_NAME"
        ));
        assert!(text.contains("EPOCH = 2023-02-27T00:00:00.000\nCOV_REF_FRAME = VNC_ROTATING\n"));
        assert_eq!(text.matches("COV_REF_FRAME = ").count(), 1);

        // The block is the kernel's rotation in km, and the reader leaves it unjoined.
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
            &mu[..1],
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

        // TNW is in table 5-4: no note, and the label stays TNW.
        options.covariance_frame = Some("tnw".to_string());
        options.table_frames_only = true;
        let text = oem_render_kvn(&orbits, &options, None).unwrap().text;
        assert_eq!(text.matches("COV_REF_FRAME = TNW\n").count(), 1);
        assert!(!text.contains("SANA"));
    }

    #[test]
    fn version_3_rounds_ties_to_even_and_version_2_formats_epochs_as_given() {
        let mut orbits = fixture(Frame::Equatorial, TimeScale::Tdb, false);
        orbits.coordinates.times = Some(
            TimeArray::from_parts(
                TimeScale::Tdb,
                vec![60000, 60001, 60002],
                vec![500_000, 1_500_000, 0],
            )
            .unwrap(),
        );
        let rendered = oem_render_kvn(&orbits, &options_3(), None).unwrap();
        assert_eq!(rendered.off_grid_epochs, 2);
        assert!(rendered.text.contains("\n2023-02-25T00:00:00.000 "));
        assert!(rendered.text.contains("\n2023-02-26T00:00:00.002 "));

        let rendered = oem_render_kvn(&orbits, &legacy_options(), None).unwrap();
        assert_eq!(rendered.off_grid_epochs, 0);
        for (days, nanos) in [(60000, 500_000), (60001, 1_500_000)] {
            let epoch = format_epoch(days, nanos, TimeScale::Tdb).unwrap();
            assert!(rendered.text.contains(&format!("\n{epoch} ")), "{epoch}");
        }
        assert!(rendered
            .text
            .contains("START_TIME = 2023-02-25T00:00:00.001\n"));
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
        let text = oem_render_kvn(&nan_state, &legacy_options(), None)
            .unwrap()
            .text;
        assert!(text.contains(" nan "));

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
        let text = oem_render_kvn(&nan_entry, &legacy_options(), None)
            .unwrap()
            .text;
        assert!(text.contains("\nnan "));
        // The all-NaN row 1 is skipped silently.
        let text = oem_render_kvn(&orbits, &options, None).unwrap().text;
        assert_eq!(text.matches("EPOCH = ").count(), 1);
    }

    #[test]
    fn local_frame_errors() {
        let orbits = fixture(Frame::Equatorial, TimeScale::Tdb, true);
        let mut options = options_3();
        // An unknown label is reported before the table 5-4 rule.
        options.covariance_frame = Some("lvlh".to_string());
        options.table_frames_only = true;
        assert!(render_error(&orbits, &options).starts_with("Unknown local orbital frame 'LVLH'"));

        // Labels are trimmed.
        options.covariance_frame = Some(" tnw ".to_string());
        let trimmed = oem_render_kvn(&orbits, &options, None).unwrap().text;
        options.covariance_frame = Some("TNW".to_string());
        assert_eq!(
            trimmed,
            oem_render_kvn(&orbits, &options, None).unwrap().text
        );

        // Labels outside table 5-4 are written as their registry names.
        let mut options = options_3();
        options.covariance_frame = Some("VNC".to_string());
        let text = oem_render_kvn(&orbits, &options, None).unwrap().text;
        assert!(text.contains("\nCOV_REF_FRAME = VNC_INERTIAL\n"));
        options.covariance_frame = Some("TNW".to_string());
        options.table_frames_only = true;

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
        options.covariance_frame = Some("VNC_ROTATING".to_string());
        options.table_frames_only = false;
        assert_eq!(
            message(oem_render_kvn(&orbits, &options, Some(&[0.0, MU_SUN, MU_SUN])).unwrap_err()),
            "mu must be finite and positive for a _ROTATING frame, got 0 at state 0"
        );
        // Rows without a covariance are not rotated and need no mu.
        assert!(oem_render_kvn(&orbits, &options, Some(&[MU_SUN, f64::NAN, f64::NAN])).is_ok());
    }

    #[test]
    fn ref_frame_label_is_validated() {
        let orbits = fixture(Frame::Ecliptic, TimeScale::Tdb, true);
        let mut options = options_3();
        options.ref_frame_label = Some(" gcrf ".to_string());
        let text = oem_render_kvn(&orbits, &options, None).unwrap().text;
        assert!(text.contains("\nREF_FRAME = GCRF\n"));
        options.ref_frame_label = Some("TOD".to_string());
        assert_eq!(
            render_error(&orbits, &options),
            "ref_frame_label TOD is not valid for equatorial states, expected one of \
             ['EME2000', 'ICRF', 'J2000', 'GCRF']"
        );
    }

    #[test]
    fn kvn_values_are_printable_ascii_on_one_line_in_version_3() {
        let orbits = fixture(Frame::Equatorial, TimeScale::Tdb, true);
        let error = |key: &str| {
            format!("OEM value for {key} must be non-empty printable ASCII on one line.")
        };
        let mut options = options_3();
        options.creation_date = String::new();
        assert_eq!(render_error(&orbits, &options), error("CREATION_DATE"));
        let mut options = options_3();
        options.originator = "\u{00c9}QUIPE".to_string();
        assert_eq!(render_error(&orbits, &options), error("ORIGINATOR"));
        let mut options = options_3();
        options.object_name = Some("A\tB".to_string());
        assert_eq!(render_error(&orbits, &options), error("OBJECT_NAME"));
        let mut options = options_3();
        options.comments = vec![" ".to_string()];
        assert_eq!(render_error(&orbits, &options), error("COMMENT"));
        // 2.0 writes the other header and metadata values verbatim, as the
        // legacy writer did, but checks the caller's names.
        let mut legacy = OemWriteOptions::legacy("TEST ORIGINATOR", "");
        let text = oem_render_kvn(&orbits, &legacy, None).unwrap().text;
        assert!(text.contains("\nCREATION_DATE = \n"));
        legacy.object_name = Some("A\nCENTER_NAME = MOON".to_string());
        assert_eq!(render_error(&orbits, &legacy), error("OBJECT_NAME"));
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

    #[test]
    fn writer_layout_matches_oem_package_shape() {
        let pairs = |items: &[(&str, &str)]| -> Vec<(String, String)> {
            items
                .iter()
                .map(|(key, value)| (key.to_string(), value.to_string()))
                .collect()
        };
        let text = render_kvn(
            &pairs(&[
                ("CCSDS_OEM_VERS", "3.0"),
                ("CREATION_DATE", "2026-01-01T00:00:00"),
                ("ORIGINATOR", "TEST"),
            ]),
            &[],
            &pairs(&[
                ("OBJECT_NAME", "X"),
                ("OBJECT_ID", "X"),
                ("CENTER_NAME", "SUN"),
                ("REF_FRAME", "EME2000"),
                ("TIME_SYSTEM", "TDB"),
                ("START_TIME", "t0"),
                ("STOP_TIME", "t1"),
            ]),
            TimeScale::Tdb,
            &[60000],
            &[0],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[],
            15,
        )
        .unwrap();
        let expected = "CCSDS_OEM_VERS = 3.0\nCREATION_DATE = 2026-01-01T00:00:00\nORIGINATOR = TEST\n\nMETA_START\nOBJECT_NAME = X\nOBJECT_ID = X\nCENTER_NAME = SUN\nREF_FRAME = EME2000\nTIME_SYSTEM = TDB\nSTART_TIME = t0\nSTOP_TIME = t1\nMETA_STOP\n\n2023-02-25T00:00:00.000 1.00000000000000e+00 2.00000000000000e+00 3.00000000000000e+00 4.00000000000000e+00 5.00000000000000e+00 6.00000000000000e+00\n\n";
        assert_eq!(text, expected);
    }
}
