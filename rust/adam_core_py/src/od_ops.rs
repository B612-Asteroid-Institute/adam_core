//! Rust-owned kernels for orbit-determination utility surfaces: C3,
//! outlier policy, worst-observation selection, IOD observation selection,
//! and linkage/member sorting.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

fn rows3(values: &PyReadonlyArray2<'_, f64>, label: &str) -> PyResult<Vec<f64>> {
    let view = values.as_array();
    if view.ncols() != 3 {
        return Err(PyValueError::new_err(format!(
            "{label} must have shape (N, 3)"
        )));
    }
    view.as_slice()
        .map(<[f64]>::to_vec)
        .ok_or_else(|| PyValueError::new_err(format!("{label} must be contiguous")))
}

fn scalars(values: &PyReadonlyArray1<'_, f64>, label: &str) -> PyResult<Vec<f64>> {
    values
        .as_array()
        .as_slice()
        .map(<[f64]>::to_vec)
        .ok_or_else(|| PyValueError::new_err(format!("{label} must be contiguous")))
}

fn vinf_values(v1: &[f64], body_v: &[f64]) -> Vec<f64> {
    v1.chunks_exact(3)
        .zip(body_v.chunks_exact(3))
        .map(|(v1, body_v)| {
            let dx = v1[0] - body_v[0];
            let dy = v1[1] - body_v[1];
            let dz = v1[2] - body_v[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .collect()
}

fn c3_values(v1: &[f64], body_v: &[f64]) -> Vec<f64> {
    // Matches the legacy norm-then-square expression exactly.
    vinf_values(v1, body_v)
        .into_iter()
        .map(|norm| norm * norm)
        .collect()
}

/// Relative-velocity magnitude (v-infinity) per row, one Rust crossing.
#[pyfunction]
fn vinf_numpy<'py>(
    py: Python<'py>,
    v1: PyReadonlyArray2<'py, f64>,
    body_v: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let v1 = rows3(&v1, "v1")?;
    let body_v = rows3(&body_v, "body_v")?;
    if v1.len() != body_v.len() {
        return Err(PyValueError::new_err("v1 and body_v must have equal shape"));
    }
    Ok(vinf_values(&v1, &body_v).into_pyarray(py))
}

#[pyfunction]
fn calculate_c3_numpy<'py>(
    py: Python<'py>,
    v1: PyReadonlyArray2<'py, f64>,
    body_v: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let v1 = rows3(&v1, "v1")?;
    let body_v = rows3(&body_v, "body_v")?;
    if v1.len() != body_v.len() {
        return Err(PyValueError::new_err("v1 and body_v must have equal shape"));
    }
    Ok(c3_values(&v1, &body_v).into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (v1, body_v, reps, trials, warmup_reps=1))]
fn benchmark_calculate_c3_numpy(
    v1: PyReadonlyArray2<'_, f64>,
    body_v: PyReadonlyArray2<'_, f64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let v1 = rows3(&v1, "v1")?;
    let body_v = rows3(&body_v, "body_v")?;
    bench(reps, trials, warmup_reps, || c3_values(&v1, &body_v))
}

#[pyfunction]
fn calculate_max_outliers_numpy(
    num_obs: i64,
    min_obs: i64,
    contamination_percentage: f64,
) -> PyResult<i64> {
    let max_outliers = num_obs as f64 * (contamination_percentage / 100.0);
    Ok(max_outliers.min((num_obs - min_obs) as f64) as i64)
}

/// Worst-observation policy: index of the lowest-probability row; ties broken
/// by the largest NaN-ignoring squared-residual norm. NaN probabilities make
/// the minimum undefined, matching the legacy empty-candidate error.
#[pyfunction]
fn lowest_probability_observation_index_numpy(
    probabilities: PyReadonlyArray1<'_, f64>,
    residual_values: PyReadonlyArray2<'_, f64>,
) -> PyResult<usize> {
    let probabilities = scalars(&probabilities, "probabilities")?;
    let residuals = residual_values.as_array();
    if residuals.nrows() != probabilities.len() {
        return Err(PyValueError::new_err(
            "residual_values must have one row per probability",
        ));
    }
    if probabilities.is_empty() || probabilities.iter().any(|value| value.is_nan()) {
        return Err(PyValueError::new_err(
            "Could not identify a lowest-probability observation.",
        ));
    }
    let min_probability = probabilities.iter().cloned().fold(f64::INFINITY, f64::min);
    let candidates: Vec<usize> = probabilities
        .iter()
        .enumerate()
        .filter(|(_, &value)| value == min_probability)
        .map(|(row, _)| row)
        .collect();
    if candidates.len() == 1 {
        return Ok(candidates[0]);
    }
    let mut best = candidates[0];
    let mut best_norm = f64::NEG_INFINITY;
    for &row in &candidates {
        let mut norm = 0.0_f64;
        for value in residuals.row(row) {
            if !value.is_nan() {
                norm += value * value;
            }
        }
        if norm > best_norm {
            best_norm = norm;
            best = row;
        }
    }
    Ok(best)
}

fn nearest_percentile_indices(times: &[f64], quantiles: &[f64]) -> Vec<usize> {
    // np.percentile(..., interpolation='nearest') on the sorted copy, then
    // np.intersect1d first-occurrence indices in the original array.
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let mut selected_values: Vec<f64> = quantiles
        .iter()
        .map(|q| {
            let rank = q / 100.0 * (n as f64 - 1.0);
            // numpy 'nearest' uses round-half-to-even on the fractional rank.
            let index = round_half_even(rank).clamp(0.0, n as f64 - 1.0) as usize;
            sorted[index]
        })
        .collect();
    selected_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    selected_values.dedup();
    // First occurrence of each selected value in the original order,
    // returned sorted by value (intersect1d semantics).
    selected_values
        .iter()
        .filter_map(|value| times.iter().position(|t| t == value))
        .collect()
}

fn round_half_even(value: f64) -> f64 {
    let rounded = value.round();
    if (value - value.trunc()).abs() == 0.5 && rounded % 2.0 != 0.0 {
        rounded - value.signum()
    } else {
        rounded
    }
}

type SelectedTriplets<'py> = Bound<'py, PyArray2<i64>>;

/// IOD observation-triplet selection (indices into the input order).
#[pyfunction]
fn select_observation_triplets_numpy<'py>(
    py: Python<'py>,
    times: PyReadonlyArray1<'py, f64>,
    method: &str,
) -> PyResult<SelectedTriplets<'py>> {
    let times = scalars(&times, "times")?;
    let n = times.len();
    let mut triplets: Vec<[usize; 3]> = Vec::new();
    match method {
        "first+middle+last" => {
            let indices = nearest_percentile_indices(&times, &[0.0, 50.0, 100.0]);
            if indices.len() == 3 {
                triplets.push([indices[0], indices[1], indices[2]]);
            }
        }
        "thirds" => {
            let indices = nearest_percentile_indices(&times, &[100.0 / 6.0, 50.0, 500.0 / 6.0]);
            if indices.len() == 3 {
                triplets.push([indices[0], indices[1], indices[2]]);
            }
        }
        "combinations" => {
            for first in 0..n {
                for second in first + 1..n {
                    for third in second + 1..n {
                        triplets.push([first, second, third]);
                    }
                }
            }
            // Sort by descending arc length, then ascending mid-point offset
            // (np.lexsort with keys (time_from_mid, -arc_length)).
            let mut keyed: Vec<(f64, f64, [usize; 3])> = triplets
                .into_iter()
                .map(|triplet| {
                    let arc = times[triplet[2]] - times[triplet[0]];
                    let mid =
                        ((times[triplet[2]] + times[triplet[0]]) / 2.0 - times[triplet[1]]).abs();
                    (arc, mid, triplet)
                })
                .collect();
            keyed.sort_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap()
                    .then(a.1.partial_cmp(&b.1).unwrap())
            });
            triplets = keyed.into_iter().map(|(_, _, triplet)| triplet).collect();
        }
        _ => {
            return Err(PyValueError::new_err(
                "method should be one of {'first+middle+last', 'thirds'}",
            ))
        }
    }
    // Keep only triplets with three unique observation times.
    let kept: Vec<i64> = triplets
        .into_iter()
        .filter(|triplet| {
            let a = times[triplet[0]];
            let b = times[triplet[1]];
            let c = times[triplet[2]];
            a != b && a != c && b != c
        })
        .flat_map(|triplet| [triplet[0] as i64, triplet[1] as i64, triplet[2] as i64])
        .collect();
    let rows = kept.len() / 3;
    ndarray::Array2::from_shape_vec((rows, 3), kept)
        .map(|array| array.into_pyarray(py))
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

type DuplicateAssignment<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<bool>>);

/// `assign_duplicate_observations` core: stable priority ordering of orbits
/// by (num_obs desc, arc_length desc, reduced_chi2 asc); each observation is
/// kept only on its highest-priority orbit; orbits with no surviving members
/// are dropped. Returns (sorted+filtered orbit take indices, member keep mask).
#[pyfunction]
fn assign_duplicate_observations_numpy<'py>(
    py: Python<'py>,
    orbit_ids: Vec<String>,
    num_obs: PyReadonlyArray1<'py, i64>,
    arc_length: PyReadonlyArray1<'py, f64>,
    reduced_chi2: PyReadonlyArray1<'py, f64>,
    member_orbit_ids: Vec<String>,
    member_obs_ids: Vec<String>,
) -> PyResult<DuplicateAssignment<'py>> {
    let num_obs = num_obs
        .as_array()
        .as_slice()
        .map(<[i64]>::to_vec)
        .ok_or_else(|| PyValueError::new_err("num_obs must be contiguous"))?;
    let arc_length = scalars(&arc_length, "arc_length")?;
    let reduced_chi2 = scalars(&reduced_chi2, "reduced_chi2")?;
    if orbit_ids.len() != num_obs.len()
        || orbit_ids.len() != arc_length.len()
        || orbit_ids.len() != reduced_chi2.len()
    {
        return Err(PyValueError::new_err("orbit columns must align"));
    }
    if member_orbit_ids.len() != member_obs_ids.len() {
        return Err(PyValueError::new_err("member columns must align"));
    }

    let mut order: Vec<usize> = (0..orbit_ids.len()).collect();
    order.sort_by(|&a, &b| {
        num_obs[b]
            .cmp(&num_obs[a])
            .then(arc_length[b].partial_cmp(&arc_length[a]).unwrap())
            .then(reduced_chi2[a].partial_cmp(&reduced_chi2[b]).unwrap())
    });
    let mut rank_by_orbit: HashMap<&str, usize> = HashMap::with_capacity(order.len());
    for (rank, &row) in order.iter().enumerate() {
        rank_by_orbit.insert(orbit_ids[row].as_str(), rank);
    }

    // Highest-priority (lowest-rank) orbit containing each observation.
    let mut best_rank_by_obs: HashMap<&str, usize> = HashMap::new();
    for (row, obs_id) in member_obs_ids.iter().enumerate() {
        if let Some(&rank) = rank_by_orbit.get(member_orbit_ids[row].as_str()) {
            best_rank_by_obs
                .entry(obs_id.as_str())
                .and_modify(|best| {
                    if rank < *best {
                        *best = rank;
                    }
                })
                .or_insert(rank);
        }
    }

    let member_keep: Vec<bool> = member_obs_ids
        .iter()
        .enumerate()
        .map(|(row, obs_id)| {
            match (
                rank_by_orbit.get(member_orbit_ids[row].as_str()),
                best_rank_by_obs.get(obs_id.as_str()),
            ) {
                (Some(rank), Some(best)) => rank == best,
                // Members whose orbit is not present are never removed.
                _ => true,
            }
        })
        .collect();

    let surviving: std::collections::HashSet<&str> = member_orbit_ids
        .iter()
        .zip(member_keep.iter())
        .filter(|(_, &keep)| keep)
        .map(|(orbit_id, _)| orbit_id.as_str())
        .collect();
    let orbit_take: Vec<i64> = order
        .into_iter()
        .filter(|&row| surviving.contains(orbit_ids[row].as_str()))
        .map(|row| row as i64)
        .collect();

    Ok((orbit_take.into_pyarray(py), member_keep.into_pyarray(py)))
}

type SortOrders<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>);

/// `sort_by_id_and_time` ordering: linkages sorted by id; members stably
/// sorted by (linkage id, observation days, observation nanos).
#[pyfunction]
fn sort_linkages_by_id_and_time_numpy<'py>(
    py: Python<'py>,
    linkage_ids: Vec<String>,
    member_linkage_ids: Vec<String>,
    member_obs_ids: Vec<String>,
    observation_ids: Vec<String>,
    observation_days: PyReadonlyArray1<'py, i64>,
    observation_nanos: PyReadonlyArray1<'py, i64>,
) -> PyResult<SortOrders<'py>> {
    let days = observation_days
        .as_array()
        .as_slice()
        .map(<[i64]>::to_vec)
        .ok_or_else(|| PyValueError::new_err("observation_days must be contiguous"))?;
    let nanos = observation_nanos
        .as_array()
        .as_slice()
        .map(<[i64]>::to_vec)
        .ok_or_else(|| PyValueError::new_err("observation_nanos must be contiguous"))?;
    if observation_ids.len() != days.len() || observation_ids.len() != nanos.len() {
        return Err(PyValueError::new_err("observation columns must align"));
    }
    if member_linkage_ids.len() != member_obs_ids.len() {
        return Err(PyValueError::new_err("member columns must align"));
    }
    let mut observation_times = HashMap::with_capacity(observation_ids.len());
    for (row, id) in observation_ids.iter().enumerate() {
        observation_times.insert(id.as_str(), (days[row], nanos[row]));
    }

    let mut linkage_order: Vec<i64> = (0..linkage_ids.len() as i64).collect();
    linkage_order.sort_by(|&a, &b| linkage_ids[a as usize].cmp(&linkage_ids[b as usize]));

    let mut member_order: Vec<i64> = (0..member_linkage_ids.len() as i64).collect();
    let member_key = |row: i64| -> PyResult<(&str, i64, i64)> {
        let row = row as usize;
        let times = observation_times
            .get(member_obs_ids[row].as_str())
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "member observation id {} not present in observations",
                    member_obs_ids[row]
                ))
            })?;
        Ok((member_linkage_ids[row].as_str(), times.0, times.1))
    };
    // Validate all keys first so sorting is infallible.
    for row in 0..member_order.len() as i64 {
        member_key(row)?;
    }
    member_order.sort_by(|&a, &b| {
        let a = member_key(a).expect("validated");
        let b = member_key(b).expect("validated");
        a.cmp(&b)
    });
    Ok((
        linkage_order.into_pyarray(py),
        member_order.into_pyarray(py),
    ))
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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_c3_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(vinf_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_calculate_c3_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_max_outliers_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(
        lowest_probability_observation_index_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(select_observation_triplets_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(sort_linkages_by_id_and_time_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(assign_duplicate_observations_numpy, m)?)?;
    Ok(())
}
