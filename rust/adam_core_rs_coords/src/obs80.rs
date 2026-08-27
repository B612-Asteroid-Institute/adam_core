//! Rust-owned MPC 80-column optical-observation parsing.

use crate::types::{SchemaError, SchemaResult, NANOS_PER_DAY};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, LargeStringArray, RecordBatch,
    StructArray,
};
use arrow_schema::{DataType, Field, Fields, Schema};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

const UNSUPPORTED_TWO_LINE_TYPES: &[char] = &['R', 'S', 'V', 'r', 's', 'v'];

#[derive(Debug, Clone, PartialEq)]
pub struct OpticalObs80Record {
    pub raw_line: String,
    pub designation: String,
    pub discovery: bool,
    pub note1: Option<String>,
    pub note2: Option<String>,
    pub observatory_code: String,
    pub days: i64,
    pub nanos: i64,
    pub ra_deg: f64,
    pub dec_deg: f64,
    pub mag: Option<f64>,
    pub band: Option<String>,
    pub astrometric_catalog: Option<String>,
    pub reference: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Obs80Error(pub String);

impl fmt::Display for Obs80Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for Obs80Error {}

fn field(chars: &[char], start: usize, end: usize) -> String {
    chars[start..end].iter().collect()
}

fn optional_field(chars: &[char], start: usize, end: usize) -> Option<String> {
    let value = field(chars, start, end).trim().to_string();
    (!value.is_empty()).then_some(value)
}

fn leap_year(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn days_in_month(year: i64, month: i64) -> Option<i64> {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => Some(31),
        4 | 6 | 9 | 11 => Some(30),
        2 => Some(if leap_year(year) { 29 } else { 28 }),
        _ => None,
    }
}

/// Days since 1970-01-01 for a validated proleptic-Gregorian date.
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let adjusted_year = year - i64::from(month <= 2);
    let era = if adjusted_year >= 0 {
        adjusted_year
    } else {
        adjusted_year - 399
    } / 400;
    let year_of_era = adjusted_year - era * 400;
    let shifted_month = month + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * shifted_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

fn rounded_fractional_day_nanos(fraction: &str) -> Result<i64, Obs80Error> {
    if fraction.is_empty() {
        return Ok(0);
    }
    if !fraction.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(Obs80Error(format!(
            "invalid fractional MPC day: {fraction:?}"
        )));
    }
    let numerator = fraction
        .parse::<u128>()
        .map_err(|_| Obs80Error(format!("invalid fractional MPC day: {fraction:?}")))?;
    let denominator = 10_u128
        .checked_pow(fraction.len() as u32)
        .ok_or_else(|| Obs80Error("MPC day has too many decimal places".to_string()))?;
    let scaled = numerator
        .checked_mul(NANOS_PER_DAY as u128)
        .ok_or_else(|| Obs80Error("MPC day is out of range".to_string()))?;
    let quotient = scaled / denominator;
    let remainder = scaled % denominator;
    let doubled = remainder * 2;
    let rounded = quotient
        + u128::from(doubled > denominator || (doubled == denominator && quotient % 2 == 1));
    i64::try_from(rounded).map_err(|_| Obs80Error("MPC day is out of range".to_string()))
}

fn parse_date(value: &str) -> Result<(i64, i64), Obs80Error> {
    let parts: Vec<&str> = value.split_whitespace().collect();
    if parts.len() != 3 {
        return Err(Obs80Error("invalid observation date".to_string()));
    }
    let year = parts[0]
        .parse::<i64>()
        .map_err(|_| Obs80Error("invalid observation date".to_string()))?;
    let month = parts[1]
        .parse::<i64>()
        .map_err(|_| Obs80Error("invalid observation date".to_string()))?;
    let day_text = parts[2];
    let (whole, fraction) = day_text.split_once('.').unwrap_or((day_text, ""));
    let day = whole
        .parse::<i64>()
        .map_err(|_| Obs80Error("invalid observation day".to_string()))?;
    let max_day = days_in_month(year, month)
        .ok_or_else(|| Obs80Error("invalid observation date".to_string()))?;
    if !(1..=max_day).contains(&day) {
        return Err(Obs80Error("invalid observation date".to_string()));
    }
    let mut nanos = rounded_fractional_day_nanos(fraction)
        .map_err(|_| Obs80Error("invalid observation day".to_string()))?;
    let mut mjd = days_from_civil(year, month, day) + 40_587;
    if nanos == NANOS_PER_DAY {
        mjd += 1;
        nanos = 0;
    }
    Ok((mjd, nanos))
}

fn parse_right_ascension(value: &str) -> Result<f64, Obs80Error> {
    let parts: Vec<&str> = value.split_whitespace().collect();
    if parts.len() != 3 {
        return Err(Obs80Error("invalid right ascension".to_string()));
    }
    let parse = |part: &str| {
        part.parse::<f64>()
            .map_err(|_| Obs80Error("invalid right ascension".to_string()))
    };
    let hours = parse(parts[0])?;
    let minutes = parse(parts[1])?;
    let seconds = parse(parts[2])?;
    if !(0.0..24.0).contains(&hours)
        || !(0.0..60.0).contains(&minutes)
        || !(0.0..60.0).contains(&seconds)
    {
        return Err(Obs80Error(
            "right ascension outside valid range".to_string(),
        ));
    }
    Ok(15.0 * (hours + minutes / 60.0 + seconds / 3600.0))
}

fn parse_declination(value: &str) -> Result<f64, Obs80Error> {
    let parts: Vec<&str> = value.split_whitespace().collect();
    let first = parts.first().copied().unwrap_or_default();
    if parts.len() != 3 || !(first.starts_with('+') || first.starts_with('-')) {
        return Err(Obs80Error("invalid declination".to_string()));
    }
    let sign = if first.starts_with('-') { -1.0 } else { 1.0 };
    let parse = |part: &str| {
        part.parse::<f64>()
            .map_err(|_| Obs80Error("invalid declination".to_string()))
    };
    let degrees = parse(&first[1..])?;
    let minutes = parse(parts[1])?;
    let seconds = parse(parts[2])?;
    if !(0.0..=90.0).contains(&degrees)
        || !(0.0..60.0).contains(&minutes)
        || !(0.0..60.0).contains(&seconds)
    {
        return Err(Obs80Error("declination outside valid range".to_string()));
    }
    let output = sign * (degrees + minutes / 60.0 + seconds / 3600.0);
    if !(-90.0..=90.0).contains(&output) {
        return Err(Obs80Error("declination outside valid range".to_string()));
    }
    Ok(output)
}

fn parse_line(raw_line: &str) -> Result<OpticalObs80Record, Obs80Error> {
    let length = raw_line.chars().count();
    if length < 80 {
        return Err(Obs80Error("record is shorter than 80 columns".to_string()));
    }
    let normalized: String = raw_line.chars().take(80).collect();
    let chars: Vec<char> = normalized.chars().collect();

    let designation = field(&chars, 0, 12).trim().to_string();
    let note2 = optional_field(&chars, 14, 15);
    if note2
        .as_deref()
        .and_then(|value| value.chars().next())
        .is_some_and(|value| UNSUPPORTED_TWO_LINE_TYPES.contains(&value))
    {
        return Err(Obs80Error(format!(
            "unsupported two-line record type {}",
            note2.as_deref().unwrap_or_default()
        )));
    }
    let observatory_code = field(&chars, 77, 80).trim().to_string();
    if designation.is_empty() {
        return Err(Obs80Error("missing designation".to_string()));
    }
    if observatory_code.chars().count() != 3 {
        return Err(Obs80Error("invalid observatory code".to_string()));
    }
    let (days, nanos) = parse_date(field(&chars, 15, 32).trim())?;
    let ra_deg = parse_right_ascension(&field(&chars, 32, 44))?;
    let dec_deg = parse_declination(&field(&chars, 44, 56))?;

    let mag = optional_field(&chars, 65, 70)
        .map(|value| {
            value
                .parse::<f64>()
                .map_err(|_| Obs80Error("invalid magnitude".to_string()))
        })
        .transpose()?;

    Ok(OpticalObs80Record {
        raw_line: normalized,
        designation,
        discovery: field(&chars, 12, 13) == "*",
        note1: optional_field(&chars, 13, 14),
        note2,
        observatory_code,
        days,
        nanos,
        ra_deg,
        dec_deg,
        mag,
        band: optional_field(&chars, 70, 71),
        astrometric_catalog: optional_field(&chars, 71, 72),
        reference: optional_field(&chars, 72, 77),
    })
}

pub fn parse_optical_obs80_line(raw: &str) -> Result<OpticalObs80Record, Obs80Error> {
    parse_line(raw)
}

pub fn parse_optical_obs80_file(
    raw: &str,
    strict: bool,
) -> Result<Vec<OpticalObs80Record>, Obs80Error> {
    let mut output = Vec::new();
    for (index, line) in raw.lines().enumerate() {
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() {
            continue;
        }
        match parse_line(line) {
            Ok(record) => output.push(record),
            Err(error) if strict => return Err(Obs80Error(format!("line {}: {error}", index + 1))),
            Err(_) => {}
        }
    }
    Ok(output)
}

fn required_strings<'a>(values: impl Iterator<Item = &'a str>) -> ArrayRef {
    Arc::new(LargeStringArray::from_iter_values(values))
}

fn optional_strings<'a>(values: impl Iterator<Item = Option<&'a str>>) -> ArrayRef {
    Arc::new(LargeStringArray::from_iter(values))
}

pub fn optical_obs80_record_batch(records: &[OpticalObs80Record]) -> SchemaResult<RecordBatch> {
    let time = StructArray::try_new(
        Fields::from(vec![
            Field::new("days", DataType::Int64, true),
            Field::new("nanos", DataType::Int64, true),
        ]),
        vec![
            Arc::new(Int64Array::from_iter_values(
                records.iter().map(|row| row.days),
            )) as ArrayRef,
            Arc::new(Int64Array::from_iter_values(
                records.iter().map(|row| row.nanos),
            )) as ArrayRef,
        ],
        None,
    )
    .map_err(|error| SchemaError::Arrow(error.to_string()))?;
    let fields = vec![
        Field::new("raw_line", DataType::LargeUtf8, false),
        Field::new("designation", DataType::LargeUtf8, false),
        Field::new("discovery", DataType::Boolean, false),
        Field::new("note1", DataType::LargeUtf8, true),
        Field::new("note2", DataType::LargeUtf8, true),
        Field::new("observatory_code", DataType::LargeUtf8, false),
        Field::new("time", time.data_type().clone(), true),
        Field::new("ra_deg", DataType::Float64, false),
        Field::new("dec_deg", DataType::Float64, false),
        Field::new("mag", DataType::Float64, true),
        Field::new("band", DataType::LargeUtf8, true),
        Field::new("astrometric_catalog", DataType::LargeUtf8, true),
        Field::new("reference", DataType::LargeUtf8, true),
    ];
    let arrays: Vec<ArrayRef> = vec![
        required_strings(records.iter().map(|row| row.raw_line.as_str())),
        required_strings(records.iter().map(|row| row.designation.as_str())),
        Arc::new(BooleanArray::from(
            records.iter().map(|row| row.discovery).collect::<Vec<_>>(),
        )),
        optional_strings(records.iter().map(|row| row.note1.as_deref())),
        optional_strings(records.iter().map(|row| row.note2.as_deref())),
        required_strings(records.iter().map(|row| row.observatory_code.as_str())),
        Arc::new(time),
        Arc::new(Float64Array::from_iter_values(
            records.iter().map(|row| row.ra_deg),
        )),
        Arc::new(Float64Array::from_iter_values(
            records.iter().map(|row| row.dec_deg),
        )),
        Arc::new(Float64Array::from_iter(records.iter().map(|row| row.mag))),
        optional_strings(records.iter().map(|row| row.band.as_deref())),
        optional_strings(records.iter().map(|row| row.astrometric_catalog.as_deref())),
        optional_strings(records.iter().map(|row| row.reference.as_deref())),
    ];
    let mut metadata = HashMap::new();
    metadata.insert(
        "adam_core_schema".to_string(),
        "OpticalObs80.nested.quivr.v1".to_string(),
    );
    metadata.insert("adam_core_schema_version".to_string(), "1".to_string());
    metadata.insert("adam_core_time_scale".to_string(), "utc".to_string());
    metadata.insert("time.scale".to_string(), "utc".to_string());
    RecordBatch::try_new(
        Arc::new(Schema::new_with_metadata(fields, metadata)),
        arrays,
    )
    .map_err(|error| SchemaError::Arrow(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_mpc_line_and_rounds_fractional_day() {
        let line =
            "     A11EpSe*0C2026 07 08.17725719 41 24.185-30 19 19.42         19.35oVNEOCPW68";
        assert_eq!(line.len(), 80);
        let row = parse_optical_obs80_line(line).unwrap();
        assert_eq!(row.designation, "A11EpSe");
        assert_eq!(row.days, 61_229);
        assert_eq!(row.nanos, 15_315_004_800_000);
        assert!((row.ra_deg - 295.350_770_833_333_3).abs() < 1e-12);
        assert!((row.dec_deg - -30.322_061_111_111_11).abs() < 1e-12);
        assert_eq!(row.observatory_code, "W68");
    }

    #[test]
    fn rejects_non_eighty_column_strict_line() {
        let error = parse_optical_obs80_line("short").unwrap_err();
        assert_eq!(error.to_string(), "record is shorter than 80 columns");
    }

    #[test]
    fn randomized_fixed_column_values_round_trip() {
        fn put(line: &mut [u8], start: usize, value: &str) {
            line[start..start + value.len()].copy_from_slice(value.as_bytes());
        }

        let mut state = 0x5eed_u64;
        for _ in 0..128 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let fraction = state % 1_000_000;
            let hours = (state >> 8) % 24;
            let minutes = (state >> 16) % 60;
            let seconds_milli = (state >> 24) % 60_000;
            let degrees = (state >> 32) % 90;
            let dec_minutes = (state >> 40) % 60;
            let dec_seconds_hundredth = (state >> 48) % 6_000;
            let sign = if state & 1 == 0 { '+' } else { '-' };

            let mut line = vec![b' '; 80];
            put(&mut line, 0, "     TEST001");
            put(&mut line, 14, "C");
            put(&mut line, 15, &format!("2026 07 08.{fraction:06}"));
            put(
                &mut line,
                32,
                &format!(
                    "{hours:02} {minutes:02} {:06.3}",
                    seconds_milli as f64 / 1_000.0
                ),
            );
            put(
                &mut line,
                44,
                &format!(
                    "{sign}{degrees:02} {dec_minutes:02} {:05.2}",
                    dec_seconds_hundredth as f64 / 100.0
                ),
            );
            put(&mut line, 65, "19.50");
            put(&mut line, 70, "V");
            put(&mut line, 71, "U");
            put(&mut line, 72, "RANDM");
            put(&mut line, 77, "500");
            let line = String::from_utf8(line).unwrap();
            let record = parse_optical_obs80_line(&line).unwrap();

            let expected_nanos =
                ((u128::from(fraction) * NANOS_PER_DAY as u128) / 1_000_000) as i64;
            let expected_ra = 15.0
                * (hours as f64 + minutes as f64 / 60.0 + seconds_milli as f64 / 1_000.0 / 3_600.0);
            let expected_dec = if sign == '-' { -1.0 } else { 1.0 }
                * (degrees as f64
                    + dec_minutes as f64 / 60.0
                    + dec_seconds_hundredth as f64 / 100.0 / 3_600.0);
            assert_eq!(record.nanos, expected_nanos);
            assert!((record.ra_deg - expected_ra).abs() < 1e-12);
            assert!((record.dec_deg - expected_dec).abs() < 1e-12);
            assert_eq!(record.astrometric_catalog.as_deref(), Some("U"));
        }
    }
}
