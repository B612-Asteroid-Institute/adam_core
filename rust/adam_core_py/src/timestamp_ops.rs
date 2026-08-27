//! Rust-owned kernels for the remaining `Timestamp` computational surface:
//! unit reductions, rounding, JD/ET conversions, grouping keys/signatures,
//! precision-tolerant equality, extrema, and uniquing. Scale/precision
//! validation errors stay at the Python boundary with their legacy messages.

use adam_core_rs_coords::oem_io::{format_epoch, parse_epoch};
use adam_core_rs_coords::{Epoch, TimeArray, TimeScale};
use chrono::{Offset, Utc};
use chrono_tz::Tz;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::hint::black_box;
use std::time::Instant;

const NANOS_IN_DAY: i64 = 86_400_000_000_000;
const NANOS_IN_SECOND: i64 = 1_000_000_000;
const NANOS_IN_MICROSECOND: i64 = 1_000;
const HALF_DAY_NANOS: i64 = NANOS_IN_DAY / 2;
const UNIX_EPOCH_MJD: i64 = 40_587;
const J2000_TDB_MJD: f64 = 51_544.5;
const SECONDS_IN_DAY: f64 = 86_400.0;

fn int_column(values: &PyReadonlyArray1<'_, i64>, label: &str) -> PyResult<Vec<i64>> {
    values
        .as_array()
        .as_slice()
        .map(<[i64]>::to_vec)
        .ok_or_else(|| PyValueError::new_err(format!("{label} must be contiguous")))
}

pub(crate) fn time_array(days: &[i64], nanos: &[i64], scale: &str) -> PyResult<TimeArray> {
    let scale = TimeScale::parse(scale).map_err(|err| PyValueError::new_err(err.to_string()))?;
    TimeArray::new(
        scale,
        days.iter()
            .zip(nanos.iter())
            .map(|(&days, &nanos)| Epoch::new(days, nanos))
            .collect(),
    )
    .map_err(|err| PyValueError::new_err(err.to_string()))
}

fn rescaled(days: &[i64], nanos: &[i64], scale: &str, target: Option<&str>) -> PyResult<TimeArray> {
    let times = time_array(days, nanos, scale)?;
    match target {
        None => Ok(times),
        Some(target) => {
            let target =
                TimeScale::parse(target).map_err(|err| PyValueError::new_err(err.to_string()))?;
            times
                .rescale(target)
                .map_err(|err| PyValueError::new_err(err.to_string()))
        }
    }
}

fn keys(times: &TimeArray) -> Vec<i64> {
    times
        .epochs
        .iter()
        .map(|epoch| epoch.days * NANOS_IN_DAY + epoch.nanos)
        .collect()
}

fn bench<F, T>(
    reps: usize,
    trials: usize,
    warmup_reps: usize,
    mut run: F,
) -> PyResult<Vec<Vec<f64>>>
where
    F: FnMut() -> T,
{
    if reps == 0 || trials == 0 {
        return Err(PyValueError::new_err("reps and trials must be >= 1"));
    }
    let mut trial_samples = Vec::with_capacity(trials);
    for _ in 0..trials {
        for _ in 0..warmup_reps {
            black_box(run());
        }
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let started = Instant::now();
            black_box(run());
            samples.push(started.elapsed().as_secs_f64());
        }
        trial_samples.push(samples);
    }
    Ok(trial_samples)
}

fn parse_scale(scale: &str) -> PyResult<TimeScale> {
    TimeScale::parse(scale).map_err(|err| PyValueError::new_err(err.to_string()))
}

fn format_iso_values(days: &[i64], nanos: &[i64], scale: TimeScale) -> PyResult<Vec<String>> {
    days.iter()
        .zip(nanos.iter())
        .map(|(&day, &nano)| {
            format_epoch(day, nano, scale).map_err(|err| PyValueError::new_err(err.to_string()))
        })
        .collect()
}

fn parse_iso_values(values: &[String], scale: TimeScale) -> PyResult<(Vec<i64>, Vec<i64>)> {
    values
        .iter()
        .map(|value| {
            let value = value.strip_suffix('Z').unwrap_or(value);
            let value = if value.contains('T') {
                Cow::Borrowed(value)
            } else {
                Cow::Owned(format!("{value}T00:00:00"))
            };
            parse_epoch(&value, scale).map_err(|err| PyValueError::new_err(err.to_string()))
        })
        .collect::<PyResult<Vec<_>>>()
        .map(|parts| parts.into_iter().unzip())
}

pub(crate) fn parse_timezones(names: &[String]) -> PyResult<Vec<Tz>> {
    let mut parsed = HashMap::<&str, Tz>::new();
    names
        .iter()
        .map(|name| {
            if let Some(timezone) = parsed.get(name.as_str()) {
                return Ok(*timezone);
            }
            let timezone = name
                .parse::<Tz>()
                .map_err(|_| PyValueError::new_err(format!("unknown IANA timezone: {name}")))?;
            parsed.insert(name.as_str(), timezone);
            Ok(timezone)
        })
        .collect()
}

pub(crate) fn observing_nights(
    days: &[i64],
    nanos: &[i64],
    timezones: &[Tz],
) -> PyResult<Vec<i64>> {
    if days.len() != nanos.len() || days.len() != timezones.len() {
        return Err(PyValueError::new_err(
            "days, nanos, and timezones must have equal length",
        ));
    }
    days.iter()
        .zip(nanos.iter())
        .zip(timezones.iter())
        .map(|((&day, &nano), timezone)| {
            // Legacy crossed through Astropy's Python `datetime`, whose
            // nearest-microsecond conversion can carry into the next day.
            let mut rounded_day = day;
            let mut micros = nano.div_euclid(NANOS_IN_MICROSECOND);
            let remainder = nano.rem_euclid(NANOS_IN_MICROSECOND);
            if remainder > NANOS_IN_MICROSECOND / 2
                || (remainder == NANOS_IN_MICROSECOND / 2 && micros % 2 != 0)
            {
                micros += 1;
            }
            let mut rounded_nano = micros * NANOS_IN_MICROSECOND;
            if rounded_nano == NANOS_IN_DAY {
                rounded_day += 1;
                rounded_nano = 0;
            }

            let unix_seconds = rounded_day
                .checked_sub(UNIX_EPOCH_MJD)
                .and_then(|value| value.checked_mul(86_400))
                .and_then(|value| value.checked_add(rounded_nano.div_euclid(NANOS_IN_SECOND)))
                .ok_or_else(|| PyValueError::new_err("timestamp is outside datetime range"))?;
            let subsecond_nanos = rounded_nano.rem_euclid(NANOS_IN_SECOND) as u32;
            let utc = chrono::DateTime::<Utc>::from_timestamp(unix_seconds, subsecond_nanos)
                .ok_or_else(|| PyValueError::new_err("timestamp is outside datetime range"))?;
            let offset_seconds =
                utc.with_timezone(timezone).offset().fix().local_minus_utc() as i64;
            let local_noon_shift = rounded_nano
                .checked_add(offset_seconds * NANOS_IN_SECOND)
                .and_then(|value| value.checked_sub(HALF_DAY_NANOS))
                .ok_or_else(|| PyValueError::new_err("timestamp arithmetic overflow"))?;
            Ok(rounded_day + local_noon_shift.div_euclid(NANOS_IN_DAY))
        })
        .collect()
}

type TimeParts<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>);

/// Rust/ERFA-owned ISO formatting at Astropy's legacy default millisecond
/// precision.
#[pyfunction]
fn timestamp_to_iso8601_numpy(
    days: PyReadonlyArray1<'_, i64>,
    nanos: PyReadonlyArray1<'_, i64>,
    scale: &str,
) -> PyResult<Vec<String>> {
    let days = int_column(&days, "days")?;
    let nanos = int_column(&nanos, "nanos")?;
    format_iso_values(&days, &nanos, parse_scale(scale)?)
}

/// Rust/ERFA-owned ISOT parsing with legacy half-even nanosecond rounding.
#[pyfunction]
fn timestamp_from_iso8601_numpy<'py>(
    py: Python<'py>,
    values: Vec<String>,
    scale: &str,
) -> PyResult<TimeParts<'py>> {
    let (days, nanos) = parse_iso_values(&values, parse_scale(scale)?)?;
    Ok((days.into_pyarray(py), nanos.into_pyarray(py)))
}

/// Observing-night integer MJD using IANA timezone offsets (including DST).
#[pyfunction]
fn calculate_observing_nights_numpy<'py>(
    py: Python<'py>,
    days: PyReadonlyArray1<'py, i64>,
    nanos: PyReadonlyArray1<'py, i64>,
    timezone_names: Vec<String>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let days = int_column(&days, "days")?;
    let nanos = int_column(&nanos, "nanos")?;
    let timezones = parse_timezones(&timezone_names)?;
    Ok(observing_nights(&days, &nanos, &timezones)?.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (days, nanos, scale, values, reps, trials, warmup_reps=1))]
fn benchmark_timestamp_iso_numpy(
    days: PyReadonlyArray1<'_, i64>,
    nanos: PyReadonlyArray1<'_, i64>,
    scale: &str,
    values: Vec<String>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let days = int_column(&days, "days")?;
    let nanos = int_column(&nanos, "nanos")?;
    let scale = parse_scale(scale)?;
    parse_iso_values(&values, scale)?;
    format_iso_values(&days, &nanos, scale)?;
    bench(reps, trials, warmup_reps, || {
        let formatted = format_iso_values(&days, &nanos, scale).expect("validated ISO format");
        let parsed = parse_iso_values(&values, scale).expect("validated ISO input");
        (formatted, parsed)
    })
}

#[pyfunction]
#[pyo3(signature = (days, nanos, timezone_names, reps, trials, warmup_reps=1))]
fn benchmark_observing_nights_numpy(
    days: PyReadonlyArray1<'_, i64>,
    nanos: PyReadonlyArray1<'_, i64>,
    timezone_names: Vec<String>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let days = int_column(&days, "days")?;
    let nanos = int_column(&nanos, "nanos")?;
    let timezones = parse_timezones(&timezone_names)?;
    observing_nights(&days, &nanos, &timezones)?;
    bench(reps, trials, warmup_reps, || {
        observing_nights(&days, &nanos, &timezones).expect("validated observing-night input")
    })
}

#[pyfunction]
fn timestamp_unit_floor_numpy<'py>(
    py: Python<'py>,
    nanos: PyReadonlyArray1<'py, i64>,
    divisor: i64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    if divisor <= 0 {
        return Err(PyValueError::new_err("divisor must be positive"));
    }
    let nanos = int_column(&nanos, "nanos")?;
    Ok(nanos
        .iter()
        .map(|value| value / divisor)
        .collect::<Vec<_>>()
        .into_pyarray(py))
}

#[pyfunction]
fn timestamp_rounded_nanos_numpy<'py>(
    py: Python<'py>,
    nanos: PyReadonlyArray1<'py, i64>,
    divisor: i64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    if divisor <= 0 {
        return Err(PyValueError::new_err("divisor must be positive"));
    }
    let nanos = int_column(&nanos, "nanos")?;
    Ok(nanos
        .iter()
        .map(|value| value / divisor * divisor)
        .collect::<Vec<_>>()
        .into_pyarray(py))
}

#[pyfunction]
fn timestamp_fractional_days_numpy<'py>(
    py: Python<'py>,
    nanos: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let nanos = int_column(&nanos, "nanos")?;
    Ok(nanos
        .iter()
        .map(|&value| value as f64 / NANOS_IN_DAY as f64)
        .collect::<Vec<_>>()
        .into_pyarray(py))
}

#[pyfunction]
fn timestamp_jd_numpy<'py>(
    py: Python<'py>,
    days: PyReadonlyArray1<'py, i64>,
    nanos: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let days = int_column(&days, "days")?;
    let nanos = int_column(&nanos, "nanos")?;
    let times = time_array(&days, &nanos, "tdb")?;
    Ok(times
        .mjd_values()
        .into_iter()
        .map(|mjd| mjd + 2_400_000.5)
        .collect::<Vec<_>>()
        .into_pyarray(py))
}

/// Fused elementwise MJD difference between two timestamp columns
/// (`minuend.mjd() - subtrahend.mjd()`) in one crossing.
#[pyfunction]
fn timestamp_mjd_difference_numpy<'py>(
    py: Python<'py>,
    minuend_days: PyReadonlyArray1<'py, i64>,
    minuend_nanos: PyReadonlyArray1<'py, i64>,
    minuend_scale: &str,
    subtrahend_days: PyReadonlyArray1<'py, i64>,
    subtrahend_nanos: PyReadonlyArray1<'py, i64>,
    subtrahend_scale: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let minuend_days = int_column(&minuend_days, "minuend_days")?;
    let minuend_nanos = int_column(&minuend_nanos, "minuend_nanos")?;
    let subtrahend_days = int_column(&subtrahend_days, "subtrahend_days")?;
    let subtrahend_nanos = int_column(&subtrahend_nanos, "subtrahend_nanos")?;
    if minuend_days.len() != subtrahend_days.len() {
        return Err(PyValueError::new_err(
            "minuend and subtrahend must have equal length",
        ));
    }
    let minuend = time_array(&minuend_days, &minuend_nanos, minuend_scale)?;
    let subtrahend = time_array(&subtrahend_days, &subtrahend_nanos, subtrahend_scale)?;
    Ok(minuend
        .mjd_values()
        .into_iter()
        .zip(subtrahend.mjd_values())
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>()
        .into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (minuend_days, minuend_nanos, minuend_scale, subtrahend_days, subtrahend_nanos, subtrahend_scale, reps, trials, warmup_reps=1))]
#[allow(clippy::too_many_arguments)]
fn benchmark_timestamp_mjd_difference_numpy(
    minuend_days: PyReadonlyArray1<'_, i64>,
    minuend_nanos: PyReadonlyArray1<'_, i64>,
    minuend_scale: &str,
    subtrahend_days: PyReadonlyArray1<'_, i64>,
    subtrahend_nanos: PyReadonlyArray1<'_, i64>,
    subtrahend_scale: &str,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let minuend_days = int_column(&minuend_days, "minuend_days")?;
    let minuend_nanos = int_column(&minuend_nanos, "minuend_nanos")?;
    let subtrahend_days = int_column(&subtrahend_days, "subtrahend_days")?;
    let subtrahend_nanos = int_column(&subtrahend_nanos, "subtrahend_nanos")?;
    let minuend = time_array(&minuend_days, &minuend_nanos, minuend_scale)?;
    let subtrahend = time_array(&subtrahend_days, &subtrahend_nanos, subtrahend_scale)?;
    if minuend.len() != subtrahend.len() {
        return Err(PyValueError::new_err(
            "minuend and subtrahend must have equal length",
        ));
    }
    bench(reps, trials, warmup_reps, || {
        minuend
            .mjd_values()
            .into_iter()
            .zip(subtrahend.mjd_values())
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>()
    })
}

/// Fused `Timestamp.et`: rescale to TDB and convert MJD to ET seconds in one
/// crossing.
#[pyfunction]
fn timestamp_et_numpy<'py>(
    py: Python<'py>,
    days: PyReadonlyArray1<'py, i64>,
    nanos: PyReadonlyArray1<'py, i64>,
    scale: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let days = int_column(&days, "days")?;
    let nanos = int_column(&nanos, "nanos")?;
    let times = rescaled(&days, &nanos, scale, Some("tdb"))?;
    Ok(times
        .mjd_values()
        .into_iter()
        .map(|mjd| (mjd - J2000_TDB_MJD) * SECONDS_IN_DAY)
        .collect::<Vec<_>>()
        .into_pyarray(py))
}

/// Fused `Timestamp.key`: optional rescale plus int64 key assembly.
#[pyfunction]
#[pyo3(signature = (days, nanos, scale, target_scale=None))]
fn timestamp_key_numpy<'py>(
    py: Python<'py>,
    days: PyReadonlyArray1<'py, i64>,
    nanos: PyReadonlyArray1<'py, i64>,
    scale: &str,
    target_scale: Option<&str>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let days = int_column(&days, "days")?;
    let nanos = int_column(&nanos, "nanos")?;
    let times = rescaled(&days, &nanos, scale, target_scale)?;
    Ok(keys(&times).into_pyarray(py))
}

/// Fused `Timestamp.signature`: `(n, first_key, last_key, sum_mod)`.
#[pyfunction]
#[pyo3(signature = (days, nanos, scale, target_scale=None))]
fn timestamp_signature_numpy(
    days: PyReadonlyArray1<'_, i64>,
    nanos: PyReadonlyArray1<'_, i64>,
    scale: &str,
    target_scale: Option<&str>,
) -> PyResult<(i64, i64, i64, i64)> {
    let days = int_column(&days, "days")?;
    let nanos = int_column(&nanos, "nanos")?;
    if days.is_empty() {
        return Ok((0, 0, 0, 0));
    }
    let times = rescaled(&days, &nanos, scale, target_scale)?;
    let keys = keys(&times);
    let sum_mod = keys
        .iter()
        .fold(0_i64, |accumulator, &key| accumulator.wrapping_add(key))
        & 0x7FFF_FFFF_FFFF_FFFF;
    Ok((keys.len() as i64, keys[0], keys[keys.len() - 1], sum_mod))
}

/// Precision-tolerant equality over Rust timestamp differences, matching the
/// legacy `(delta_days, delta_nanos)` tolerance composition.
#[pyfunction]
fn timestamp_equals_numpy<'py>(
    py: Python<'py>,
    days: PyReadonlyArray1<'py, i64>,
    nanos: PyReadonlyArray1<'py, i64>,
    other_days: PyReadonlyArray1<'py, i64>,
    other_nanos: PyReadonlyArray1<'py, i64>,
    max_nanos_deviation: i64,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let days = int_column(&days, "days")?;
    let nanos = int_column(&nanos, "nanos")?;
    let other_days = int_column(&other_days, "other_days")?;
    let other_nanos = int_column(&other_nanos, "other_nanos")?;
    let broadcast = other_days.len() == 1;
    if !broadcast && (other_days.len() != days.len() || other_nanos.len() != nanos.len()) {
        return Err(PyValueError::new_err(
            "Timestamps must have the same length",
        ));
    }
    let out: Vec<bool> = days
        .iter()
        .zip(nanos.iter())
        .enumerate()
        .map(|(row, (&day, &nano))| {
            let index = if broadcast { 0 } else { row };
            // Difference with the legacy normalization: nanos in [0, day).
            let mut delta_days = day - other_days[index];
            let mut delta_nanos = nano - other_nanos[index];
            if delta_nanos < 0 {
                delta_days -= 1;
                delta_nanos += NANOS_IN_DAY;
            }
            if max_nanos_deviation == 0 {
                delta_days == 0 && delta_nanos == 0
            } else {
                (delta_days == 0 && delta_nanos.abs() < max_nanos_deviation)
                    || (delta_days == -1
                        && delta_nanos.abs() as f64
                            >= NANOS_IN_DAY as f64 - max_nanos_deviation as f64)
            }
        })
        .collect();
    Ok(out.into_pyarray(py))
}

/// First-occurrence extremum row index (max when `is_max`, else min),
/// ordering by (days, nanos).
#[pyfunction]
fn timestamp_extremum_index_numpy(
    days: PyReadonlyArray1<'_, i64>,
    nanos: PyReadonlyArray1<'_, i64>,
    is_max: bool,
) -> PyResult<usize> {
    let days = int_column(&days, "days")?;
    let nanos = int_column(&nanos, "nanos")?;
    if days.is_empty() {
        return Err(PyValueError::new_err("cannot reduce an empty Timestamp"));
    }
    let mut best = 0usize;
    for row in 1..days.len() {
        let candidate = (days[row], nanos[row]);
        let current = (days[best], nanos[best]);
        let better = if is_max {
            candidate > current
        } else {
            candidate < current
        };
        if better {
            best = row;
        }
    }
    Ok(best)
}

type UniquePairs<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>);

/// Unique (days, nanos) pairs in first-appearance order.
#[pyfunction]
fn timestamp_unique_numpy<'py>(
    py: Python<'py>,
    days: PyReadonlyArray1<'py, i64>,
    nanos: PyReadonlyArray1<'py, i64>,
) -> PyResult<UniquePairs<'py>> {
    let days = int_column(&days, "days")?;
    let nanos = int_column(&nanos, "nanos")?;
    let mut seen = HashSet::with_capacity(days.len());
    let mut days_out = Vec::new();
    let mut nanos_out = Vec::new();
    for (&day, &nano) in days.iter().zip(nanos.iter()) {
        if seen.insert((day, nano)) {
            days_out.push(day);
            nanos_out.push(nano);
        }
    }
    Ok((days_out.into_pyarray(py), nanos_out.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (days, nanos, scale, reps, trials, warmup_reps=1))]
fn benchmark_timestamp_ops_numpy(
    days: PyReadonlyArray1<'_, i64>,
    nanos: PyReadonlyArray1<'_, i64>,
    scale: &str,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let days = int_column(&days, "days")?;
    let nanos = int_column(&nanos, "nanos")?;
    let scale = TimeScale::parse(scale).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let times = TimeArray::new(
        scale,
        days.iter()
            .zip(nanos.iter())
            .map(|(&days, &nanos)| Epoch::new(days, nanos))
            .collect(),
    )
    .map_err(|err| PyValueError::new_err(err.to_string()))?;
    bench(reps, trials, warmup_reps, || {
        let rescaled = times.rescale(TimeScale::Tdb).expect("rescale");
        let keys = keys(&rescaled);
        let floors: Vec<i64> = nanos.iter().map(|value| value / 1_000_000).collect();
        let et: Vec<f64> = rescaled
            .mjd_values()
            .into_iter()
            .map(|mjd| (mjd - J2000_TDB_MJD) * SECONDS_IN_DAY)
            .collect();
        (keys, floors, et)
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(timestamp_to_iso8601_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(timestamp_from_iso8601_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_observing_nights_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_timestamp_iso_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_observing_nights_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(timestamp_unit_floor_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(timestamp_rounded_nanos_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(timestamp_fractional_days_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(timestamp_jd_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(timestamp_mjd_difference_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(
        benchmark_timestamp_mjd_difference_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(timestamp_et_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(timestamp_key_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(timestamp_signature_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(timestamp_equals_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(timestamp_extremum_index_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(timestamp_unique_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_timestamp_ops_numpy, m)?)?;
    Ok(())
}
