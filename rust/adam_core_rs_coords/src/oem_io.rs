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

use crate::types::{SchemaError, SchemaResult, TimeScale};
use std::fmt::Write as _;
use std::path::Path;

const NANOS_PER_DAY_F64: f64 = 86_400e9;
const MJD_JD_OFFSET: f64 = 2_400_000.5;

fn invalid(message: String) -> SchemaError {
    SchemaError::InvalidRecordBatch(message)
}

// --- float / epoch formatting ---------------------------------------------------

/// Python `f"{value:.14e}"`.
fn py_sci14(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    let raw = format!("{value:.14e}");
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
/// (leap-second aware for UTC).
fn format_epoch(days: i64, nanos: i64, scale: TimeScale) -> SchemaResult<String> {
    let (jd1, jd2) = mjd_to_jd_pair(days, nanos);
    let ((iy, im, id, ih, imin, isec, ifrac), _warning) =
        erfars::timescales::D2dtf(scale == TimeScale::Utc, 3, jd1, jd2)
            .map_err(|code| invalid(format!("eraD2dtf failed with ERFA status {code}")))?;
    Ok(format!(
        "{iy:04}-{im:02}-{id:02}T{ih:02}:{imin:02}:{isec:02}.{ifrac:03}"
    ))
}

/// ISOT epoch string -> legacy `Timestamp.from_astropy` (days, nanos).
fn parse_epoch(isot: &str, scale: TimeScale) -> SchemaResult<(i64, i64)> {
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
/// JSON objects (values rendered verbatim).
pub fn oem_to_kvn(
    header_json: &str,
    metadata_json: &str,
    time_scale: TimeScale,
    days: &[i64],
    nanos: &[i64],
    states_km: &[f64],
    covariances: &[OemCovarianceRecord],
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
        let rendered: Vec<String> = state.iter().map(|&value| py_sci14(value)).collect();
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
                let rendered: Vec<String> = row.iter().map(|&value| py_sci14(value)).collect();
                out.push_str(&rendered.join(" "));
                out.push('\n');
            }
        }
        out.push_str("COVARIANCE_STOP\n");
        out.push('\n');
    }

    Ok(out)
}

/// Write a KVN OEM file (see [`oem_to_kvn`]).
pub fn oem_write_kvn(
    path: &Path,
    header_json: &str,
    metadata_json: &str,
    time_scale: TimeScale,
    days: &[i64],
    nanos: &[i64],
    states_km: &[f64],
    covariances: &[OemCovarianceRecord],
) -> SchemaResult<()> {
    let text = oem_to_kvn(
        header_json,
        metadata_json,
        time_scale,
        days,
        nanos,
        states_km,
        covariances,
    )?;
    std::fs::write(path, text)
        .map_err(|err| invalid(format!("failed to write {}: {err}", path.display())))
}

// --- parser -----------------------------------------------------------------------

/// Parse a KVN OEM file into a JSON payload:
/// `{header: {...}, segments: [{metadata: {...}, states: {days, nanos,
/// values_km}, covariances: [{days, nanos, frame, matrix (36, symmetric,
/// km)}]}]}`. Epoch integers use the exact legacy `Timestamp.from_astropy`
/// split in the segment's TIME_SYSTEM scale.
pub fn oem_parse_kvn(path: &Path) -> SchemaResult<String> {
    use serde_json::{json, Map, Value};

    let text = std::fs::read_to_string(path)
        .map_err(|err| invalid(format!("failed to read {}: {err}", path.display())))?;

    #[derive(Default)]
    struct Segment {
        metadata: Map<String, Value>,
        days: Vec<i64>,
        nanos: Vec<i64>,
        values_km: Vec<Vec<f64>>,
        covariances: Vec<Value>,
    }

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
            segment.covariances.push(json!({
                "days": days,
                "nanos": nanos,
                "frame": frame,
                "matrix": matrix,
            }));
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

    let payload = json!({
        "header": header,
        "segments": segments
            .into_iter()
            .map(|segment| {
                json!({
                    "metadata": segment.metadata,
                    "states": {
                        "days": segment.days,
                        "nanos": segment.nanos,
                        "values_km": segment.values_km,
                    },
                    "covariances": segment.covariances,
                })
            })
            .collect::<Vec<_>>(),
    });
    serde_json::to_string(&payload).map_err(|err| invalid(format!("encode failed: {err}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn py_sci14_matches_python_format() {
        assert_eq!(py_sci14(0.0), "0.00000000000000e+00");
        assert_eq!(py_sci14(123456.789), "1.23456789000000e+05");
        assert_eq!(py_sci14(-1.5e-7), "-1.50000000000000e-07");
        assert_eq!(py_sci14(std::f64::consts::TAU), "6.28318530717959e+00");
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
        )
        .unwrap();
        let expected = "CCSDS_OEM_VERS = 3.0\nCREATION_DATE = 2026-01-01T00:00:00\nORIGINATOR = TEST\n\nMETA_START\nOBJECT_NAME = X\nOBJECT_ID = X\nCENTER_NAME = SUN\nREF_FRAME = EME2000\nTIME_SYSTEM = TDB\nSTART_TIME = t0\nSTOP_TIME = t1\nMETA_STOP\n\n2023-02-25T00:00:00.000 1.00000000000000e+00 2.00000000000000e+00 3.00000000000000e+00 4.00000000000000e+00 5.00000000000000e+00 6.00000000000000e+00\n\n";
        assert_eq!(text, expected);
    }
}
