//! Rust-owned computation cores for the public `VariantEphemeris` collapse
//! surfaces. Python veneers extract columns once, make one crossing, and wrap
//! the returned arrays; grouping, circular longitude statistics, weighted
//! means/covariances, and magnitude reductions all execute here.

use adam_core_rs_coords::weighted_covariance_flat;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

fn rows6(values: &PyReadonlyArray2<'_, f64>, label: &str) -> PyResult<Vec<[f64; 6]>> {
    let view = values.as_array();
    if view.ncols() != 6 {
        return Err(PyValueError::new_err(format!(
            "{label} must have shape (N, 6)"
        )));
    }
    Ok(view
        .rows()
        .into_iter()
        .map(|row| [row[0], row[1], row[2], row[3], row[4], row[5]])
        .collect())
}

fn scalars(values: &PyReadonlyArray1<'_, f64>, label: &str) -> PyResult<Vec<f64>> {
    values
        .as_array()
        .as_slice()
        .map(<[f64]>::to_vec)
        .ok_or_else(|| PyValueError::new_err(format!("{label} must be contiguous")))
}

fn int_column(values: &PyReadonlyArray1<'_, i64>, label: &str) -> PyResult<Vec<i64>> {
    values
        .as_array()
        .as_slice()
        .map(<[i64]>::to_vec)
        .ok_or_else(|| PyValueError::new_err(format!("{label} must be contiguous")))
}

struct CollapseCovarianceInputs {
    orbit_keys: Vec<(String, i64, i64)>,
    orbit_means: Vec<[f64; 6]>,
    orbit_aberrated_means: Option<Vec<[f64; 6]>>,
    variant_keys: Vec<(String, i64, i64)>,
    variant_values: Vec<[f64; 6]>,
    variant_weights_cov: Vec<f64>,
    variant_aberrated: Option<Vec<[f64; 6]>>,
}

fn collapse_covariances(inputs: &CollapseCovarianceInputs) -> (Vec<f64>, Option<Vec<f64>>) {
    let mut groups: HashMap<(&str, i64, i64), Vec<usize>> = HashMap::new();
    for (row, key) in inputs.variant_keys.iter().enumerate() {
        groups
            .entry((key.0.as_str(), key.1, key.2))
            .or_default()
            .push(row);
    }
    let rows = inputs.orbit_keys.len();
    let mut covariances = vec![0.0_f64; rows * 36];
    let mut aberrated_covariances = inputs
        .orbit_aberrated_means
        .as_ref()
        .map(|_| vec![0.0_f64; rows * 36]);
    for row in 0..rows {
        let key = &inputs.orbit_keys[row];
        let indices: &[usize] = groups
            .get(&(key.0.as_str(), key.1, key.2))
            .map_or(&[], Vec::as_slice);
        let weights: Vec<f64> = indices
            .iter()
            .map(|&index| inputs.variant_weights_cov[index])
            .collect();
        let samples: Vec<f64> = indices
            .iter()
            .flat_map(|&index| inputs.variant_values[index].iter().copied())
            .collect();
        let covariance = weighted_covariance_flat(
            &inputs.orbit_means[row],
            &samples,
            &weights,
            indices.len(),
            6,
        );
        covariances[row * 36..(row + 1) * 36].copy_from_slice(&covariance);
        if let (Some(aberrated_means), Some(aberrated_out), Some(aberrated_values)) = (
            inputs.orbit_aberrated_means.as_ref(),
            aberrated_covariances.as_mut(),
            inputs.variant_aberrated.as_ref(),
        ) {
            let samples: Vec<f64> = indices
                .iter()
                .flat_map(|&index| aberrated_values[index].iter().copied())
                .collect();
            let covariance = weighted_covariance_flat(
                &aberrated_means[row],
                &samples,
                &weights,
                indices.len(),
                6,
            );
            aberrated_out[row * 36..(row + 1) * 36].copy_from_slice(&covariance);
        }
    }
    (covariances, aberrated_covariances)
}

#[allow(clippy::too_many_arguments)]
fn collapse_covariance_inputs(
    orbit_ids: Vec<String>,
    orbit_days: &PyReadonlyArray1<'_, i64>,
    orbit_millis: &PyReadonlyArray1<'_, i64>,
    orbit_means: &PyReadonlyArray2<'_, f64>,
    orbit_aberrated_means: Option<&PyReadonlyArray2<'_, f64>>,
    variant_ids: Vec<String>,
    variant_days: &PyReadonlyArray1<'_, i64>,
    variant_millis: &PyReadonlyArray1<'_, i64>,
    variant_values: &PyReadonlyArray2<'_, f64>,
    variant_weights_cov: &PyReadonlyArray1<'_, f64>,
    variant_aberrated: Option<&PyReadonlyArray2<'_, f64>>,
) -> PyResult<CollapseCovarianceInputs> {
    let orbit_days = int_column(orbit_days, "orbit_days")?;
    let orbit_millis = int_column(orbit_millis, "orbit_millis")?;
    let variant_days = int_column(variant_days, "variant_days")?;
    let variant_millis = int_column(variant_millis, "variant_millis")?;
    if orbit_ids.len() != orbit_days.len() || orbit_ids.len() != orbit_millis.len() {
        return Err(PyValueError::new_err("orbit key columns must align"));
    }
    if variant_ids.len() != variant_days.len() || variant_ids.len() != variant_millis.len() {
        return Err(PyValueError::new_err("variant key columns must align"));
    }
    Ok(CollapseCovarianceInputs {
        orbit_keys: orbit_ids
            .into_iter()
            .zip(orbit_days)
            .zip(orbit_millis)
            .map(|((orbit_id, days), millis)| (orbit_id, days, millis))
            .collect(),
        orbit_means: rows6(orbit_means, "orbit_means")?,
        orbit_aberrated_means: orbit_aberrated_means
            .map(|values| rows6(values, "orbit_aberrated_means"))
            .transpose()?,
        variant_keys: variant_ids
            .into_iter()
            .zip(variant_days)
            .zip(variant_millis)
            .map(|((orbit_id, days), millis)| (orbit_id, days, millis))
            .collect(),
        variant_values: rows6(variant_values, "variant_values")?,
        variant_weights_cov: scalars(variant_weights_cov, "variant_weights_cov")?,
        variant_aberrated: variant_aberrated
            .map(|values| rows6(values, "variant_aberrated"))
            .transpose()?,
    })
}

/// Rust-owned `VariantEphemeris.collapse` core: linkage grouping and weighted
/// covariance for the spherical and optional aberrated coordinates.
#[pyfunction]
#[pyo3(signature = (
    orbit_ids, orbit_days, orbit_millis, orbit_means,
    variant_ids, variant_days, variant_millis, variant_values,
    variant_weights_cov, orbit_aberrated_means=None, variant_aberrated=None
))]
#[allow(clippy::too_many_arguments)]
pub fn collapse_variant_ephemeris_covariances_numpy<'py>(
    py: Python<'py>,
    orbit_ids: Vec<String>,
    orbit_days: PyReadonlyArray1<'py, i64>,
    orbit_millis: PyReadonlyArray1<'py, i64>,
    orbit_means: PyReadonlyArray2<'py, f64>,
    variant_ids: Vec<String>,
    variant_days: PyReadonlyArray1<'py, i64>,
    variant_millis: PyReadonlyArray1<'py, i64>,
    variant_values: PyReadonlyArray2<'py, f64>,
    variant_weights_cov: PyReadonlyArray1<'py, f64>,
    orbit_aberrated_means: Option<PyReadonlyArray2<'py, f64>>,
    variant_aberrated: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Option<Bound<'py, PyArray2<f64>>>)> {
    let inputs = collapse_covariance_inputs(
        orbit_ids,
        &orbit_days,
        &orbit_millis,
        &orbit_means,
        orbit_aberrated_means.as_ref(),
        variant_ids,
        &variant_days,
        &variant_millis,
        &variant_values,
        &variant_weights_cov,
        variant_aberrated.as_ref(),
    )?;
    let rows = inputs.orbit_keys.len();
    let (covariances, aberrated) = py.allow_threads(|| collapse_covariances(&inputs));
    let to_array = |values: Vec<f64>| {
        ndarray::Array2::from_shape_vec((rows, 36), values)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    };
    Ok((
        to_array(covariances)?.into_pyarray(py),
        aberrated
            .map(|values| to_array(values).map(|array| array.into_pyarray(py)))
            .transpose()?,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    orbit_ids, orbit_days, orbit_millis, orbit_means,
    variant_ids, variant_days, variant_millis, variant_values,
    variant_weights_cov, reps, trials, warmup_reps=1,
    orbit_aberrated_means=None, variant_aberrated=None
))]
#[allow(clippy::too_many_arguments)]
pub fn benchmark_collapse_variant_ephemeris_covariances_numpy<'py>(
    orbit_ids: Vec<String>,
    orbit_days: PyReadonlyArray1<'py, i64>,
    orbit_millis: PyReadonlyArray1<'py, i64>,
    orbit_means: PyReadonlyArray2<'py, f64>,
    variant_ids: Vec<String>,
    variant_days: PyReadonlyArray1<'py, i64>,
    variant_millis: PyReadonlyArray1<'py, i64>,
    variant_values: PyReadonlyArray2<'py, f64>,
    variant_weights_cov: PyReadonlyArray1<'py, f64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
    orbit_aberrated_means: Option<PyReadonlyArray2<'py, f64>>,
    variant_aberrated: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Vec<Vec<f64>>> {
    if reps == 0 || trials == 0 {
        return Err(PyValueError::new_err("reps and trials must be >= 1"));
    }
    let inputs = collapse_covariance_inputs(
        orbit_ids,
        &orbit_days,
        &orbit_millis,
        &orbit_means,
        orbit_aberrated_means.as_ref(),
        variant_ids,
        &variant_days,
        &variant_millis,
        &variant_values,
        &variant_weights_cov,
        variant_aberrated.as_ref(),
    )?;
    let mut trial_samples = Vec::with_capacity(trials);
    for _ in 0..trials {
        for _ in 0..warmup_reps {
            black_box(collapse_covariances(&inputs));
        }
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let started = Instant::now();
            black_box(collapse_covariances(&inputs));
            samples.push(started.elapsed().as_secs_f64());
        }
        trial_samples.push(samples);
    }
    Ok(trial_samples)
}

struct GroupedCollapseInputs {
    object_ids: Vec<Option<String>>,
    days: Vec<i64>,
    nanos: Vec<i64>,
    origin_codes: Vec<String>,
    values: Vec<[f64; 6]>,
    weights: Vec<f64>,
    weights_cov: Vec<f64>,
    magnitudes: Vec<f64>,
    aberrated: Option<(Vec<[f64; 6]>, Vec<f64>)>,
}

struct GroupedCollapseOutput {
    sorted_indices: Vec<i64>,
    group_starts: Vec<i64>,
    means: Vec<f64>,
    covariances: Vec<f64>,
    magnitudes: Vec<f64>,
    aberrated: Option<(Vec<f64>, Vec<f64>)>,
}

fn normalize_weights(weights: &[f64], fallback: Option<&[f64]>) -> Vec<f64> {
    let n = weights.len();
    let all_finite = weights.iter().all(|value| value.is_finite());
    if !all_finite {
        return match fallback {
            Some(fallback) => fallback.to_vec(),
            None => vec![1.0 / n as f64; n],
        };
    }
    let sum: f64 = weights.iter().sum();
    if sum.is_finite() && sum != 0.0 {
        weights.iter().map(|value| value / sum).collect()
    } else {
        match fallback {
            Some(fallback) => fallback.to_vec(),
            None => vec![1.0 / n as f64; n],
        }
    }
}

fn collapse_grouped(inputs: &GroupedCollapseInputs) -> GroupedCollapseOutput {
    let rows = inputs.values.len();
    let mut order: Vec<usize> = (0..rows).collect();
    order.sort_by(|&a, &b| {
        let key = |row: usize| {
            (
                inputs.object_ids[row].is_none(),
                inputs.object_ids[row].as_deref().unwrap_or(""),
                inputs.days[row],
                inputs.nanos[row],
                inputs.origin_codes[row].as_str(),
            )
        };
        key(a).cmp(&key(b))
    });

    let mut group_starts: Vec<i64> = vec![0];
    for position in 1..rows {
        let previous = order[position - 1];
        let current = order[position];
        let changed = inputs.object_ids[previous] != inputs.object_ids[current]
            || inputs.days[previous] != inputs.days[current]
            || inputs.nanos[previous] != inputs.nanos[current]
            || inputs.origin_codes[previous] != inputs.origin_codes[current];
        if changed {
            group_starts.push(position as i64);
        }
    }
    let n_groups = group_starts.len();
    let mut bounds = group_starts.clone();
    bounds.push(rows as i64);

    let mut means = vec![0.0_f64; n_groups * 6];
    let mut covariances = vec![0.0_f64; n_groups * 36];
    let mut magnitudes = vec![f64::NAN; n_groups];
    let mut aberrated_out = inputs
        .aberrated
        .as_ref()
        .map(|_| (vec![0.0_f64; n_groups * 6], vec![f64::NAN; n_groups]));

    for group in 0..n_groups {
        let start = bounds[group] as usize;
        let end = bounds[group + 1] as usize;
        let group_rows = &order[start..end];
        let n = group_rows.len();

        let raw_mean: Vec<f64> = group_rows.iter().map(|&row| inputs.weights[row]).collect();
        let w_mean = normalize_weights(&raw_mean, None);
        let raw_cov: Vec<f64> = group_rows
            .iter()
            .map(|&row| inputs.weights_cov[row])
            .collect();
        let w_cov = normalize_weights(&raw_cov, Some(&w_mean));

        // Circular weighted mean for longitude (degrees).
        let mut s_sin = 0.0_f64;
        let mut s_cos = 0.0_f64;
        for (position, &row) in group_rows.iter().enumerate() {
            let lon = inputs.values[row][1].to_radians();
            s_sin += w_mean[position] * lon.sin();
            s_cos += w_mean[position] * lon.cos();
        }
        let lon_mean = (s_sin.atan2(s_cos).to_degrees() + 360.0) % 360.0;

        // Wrap longitude samples around the circular mean.
        let samples: Vec<f64> = group_rows
            .iter()
            .flat_map(|&row| {
                let mut values = inputs.values[row];
                let lon = values[1];
                values[1] = lon_mean + (((lon - lon_mean + 180.0) % 360.0 + 360.0) % 360.0 - 180.0);
                values
            })
            .collect();

        let mut mean = [0.0_f64; 6];
        for (position, chunk) in samples.chunks_exact(6).enumerate() {
            for (k, value) in chunk.iter().enumerate() {
                mean[k] += w_mean[position] * value;
            }
        }
        mean[1] = lon_mean;
        means[group * 6..(group + 1) * 6].copy_from_slice(&mean);

        let covariance = weighted_covariance_flat(&mean, &samples, &w_cov, n, 6);
        covariances[group * 36..(group + 1) * 36].copy_from_slice(&covariance);

        // Weighted magnitude mean ignoring non-finite entries.
        let mut denominator = 0.0_f64;
        let mut numerator = 0.0_f64;
        for (position, &row) in group_rows.iter().enumerate() {
            let magnitude = inputs.magnitudes[row];
            if magnitude.is_finite() {
                denominator += w_mean[position];
                numerator += w_mean[position] * magnitude;
            }
        }
        if denominator > 0.0 {
            magnitudes[group] = numerator / denominator;
        }

        if let (Some((aberrated_values, light_times)), Some((out_ab, out_lt))) =
            (inputs.aberrated.as_ref(), aberrated_out.as_mut())
        {
            let mut mean_ab = [0.0_f64; 6];
            for (position, &row) in group_rows.iter().enumerate() {
                for (k, value) in aberrated_values[row].iter().enumerate() {
                    mean_ab[k] += w_mean[position] * value;
                }
            }
            out_ab[group * 6..(group + 1) * 6].copy_from_slice(&mean_ab);
            let mut denominator = 0.0_f64;
            let mut numerator = 0.0_f64;
            for (position, &row) in group_rows.iter().enumerate() {
                let light_time = light_times[row];
                if light_time.is_finite() {
                    denominator += w_mean[position];
                    numerator += w_mean[position] * light_time;
                }
            }
            if denominator > 0.0 {
                out_lt[group] = numerator / denominator;
            }
        }
    }

    GroupedCollapseOutput {
        sorted_indices: order.into_iter().map(|row| row as i64).collect(),
        group_starts,
        means,
        covariances,
        magnitudes,
        aberrated: aberrated_out,
    }
}

#[allow(clippy::too_many_arguments)]
fn grouped_collapse_inputs(
    object_ids: Vec<Option<String>>,
    days: &PyReadonlyArray1<'_, i64>,
    nanos: &PyReadonlyArray1<'_, i64>,
    origin_codes: Vec<String>,
    values: &PyReadonlyArray2<'_, f64>,
    weights: &PyReadonlyArray1<'_, f64>,
    weights_cov: &PyReadonlyArray1<'_, f64>,
    magnitudes: &PyReadonlyArray1<'_, f64>,
    aberrated_values: Option<&PyReadonlyArray2<'_, f64>>,
    light_times: Option<&PyReadonlyArray1<'_, f64>>,
) -> PyResult<GroupedCollapseInputs> {
    let values = rows6(values, "values")?;
    let rows = values.len();
    let inputs = GroupedCollapseInputs {
        object_ids,
        days: int_column(days, "days")?,
        nanos: int_column(nanos, "nanos")?,
        origin_codes,
        values,
        weights: scalars(weights, "weights")?,
        weights_cov: scalars(weights_cov, "weights_cov")?,
        magnitudes: scalars(magnitudes, "magnitudes")?,
        aberrated: match (aberrated_values, light_times) {
            (Some(values), Some(light_times)) => Some((
                rows6(values, "aberrated_values")?,
                scalars(light_times, "light_times")?,
            )),
            (None, None) => None,
            _ => {
                return Err(PyValueError::new_err(
                    "aberrated_values and light_times must be provided together",
                ))
            }
        },
    };
    for (label, len) in [
        ("object_ids", inputs.object_ids.len()),
        ("days", inputs.days.len()),
        ("nanos", inputs.nanos.len()),
        ("origin_codes", inputs.origin_codes.len()),
        ("weights", inputs.weights.len()),
        ("weights_cov", inputs.weights_cov.len()),
        ("magnitudes", inputs.magnitudes.len()),
    ] {
        if len != rows {
            return Err(PyValueError::new_err(format!(
                "{label} must have length {rows}"
            )));
        }
    }
    Ok(inputs)
}

type GroupedCollapseResult<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Option<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)>,
);

/// Rust-owned `VariantEphemeris.collapse_by_object_id` core: stable grouping
/// by (object_id, time, origin), circular-longitude weighted statistics, and
/// optional aberrated/light-time weighted means.
#[pyfunction]
#[pyo3(signature = (
    object_ids, days, nanos, origin_codes, values, weights, weights_cov,
    magnitudes, aberrated_values=None, light_times=None
))]
#[allow(clippy::too_many_arguments)]
pub fn collapse_variant_ephemeris_by_object_numpy<'py>(
    py: Python<'py>,
    object_ids: Vec<Option<String>>,
    days: PyReadonlyArray1<'py, i64>,
    nanos: PyReadonlyArray1<'py, i64>,
    origin_codes: Vec<String>,
    values: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    weights_cov: PyReadonlyArray1<'py, f64>,
    magnitudes: PyReadonlyArray1<'py, f64>,
    aberrated_values: Option<PyReadonlyArray2<'py, f64>>,
    light_times: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<GroupedCollapseResult<'py>> {
    let inputs = grouped_collapse_inputs(
        object_ids,
        &days,
        &nanos,
        origin_codes,
        &values,
        &weights,
        &weights_cov,
        &magnitudes,
        aberrated_values.as_ref(),
        light_times.as_ref(),
    )?;
    let output = py.allow_threads(|| collapse_grouped(&inputs));
    let n_groups = output.group_starts.len();
    let means = ndarray::Array2::from_shape_vec((n_groups, 6), output.means)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let covariances = ndarray::Array2::from_shape_vec((n_groups, 36), output.covariances)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let aberrated = output
        .aberrated
        .map(|(values, light_times)| {
            ndarray::Array2::from_shape_vec((n_groups, 6), values)
                .map_err(|err| PyValueError::new_err(err.to_string()))
                .map(|array| (array.into_pyarray(py), light_times.into_pyarray(py)))
        })
        .transpose()?;
    Ok((
        output.sorted_indices.into_pyarray(py),
        output.group_starts.into_pyarray(py),
        means.into_pyarray(py),
        covariances.into_pyarray(py),
        output.magnitudes.into_pyarray(py),
        aberrated,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    object_ids, days, nanos, origin_codes, values, weights, weights_cov,
    magnitudes, reps, trials, warmup_reps=1, aberrated_values=None, light_times=None
))]
#[allow(clippy::too_many_arguments)]
pub fn benchmark_collapse_variant_ephemeris_by_object_numpy<'py>(
    object_ids: Vec<Option<String>>,
    days: PyReadonlyArray1<'py, i64>,
    nanos: PyReadonlyArray1<'py, i64>,
    origin_codes: Vec<String>,
    values: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    weights_cov: PyReadonlyArray1<'py, f64>,
    magnitudes: PyReadonlyArray1<'py, f64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
    aberrated_values: Option<PyReadonlyArray2<'py, f64>>,
    light_times: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Vec<Vec<f64>>> {
    if reps == 0 || trials == 0 {
        return Err(PyValueError::new_err("reps and trials must be >= 1"));
    }
    let inputs = grouped_collapse_inputs(
        object_ids,
        &days,
        &nanos,
        origin_codes,
        &values,
        &weights,
        &weights_cov,
        &magnitudes,
        aberrated_values.as_ref(),
        light_times.as_ref(),
    )?;
    let mut trial_samples = Vec::with_capacity(trials);
    for _ in 0..trials {
        for _ in 0..warmup_reps {
            black_box(collapse_grouped(&inputs));
        }
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let started = Instant::now();
            black_box(collapse_grouped(&inputs));
            samples.push(started.elapsed().as_secs_f64());
        }
        trial_samples.push(samples);
    }
    Ok(trial_samples)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        collapse_variant_ephemeris_covariances_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        benchmark_collapse_variant_ephemeris_covariances_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        collapse_variant_ephemeris_by_object_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        benchmark_collapse_variant_ephemeris_by_object_numpy,
        m
    )?)?;
    Ok(())
}
