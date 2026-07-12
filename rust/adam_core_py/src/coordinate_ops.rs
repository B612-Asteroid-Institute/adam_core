//! Rust-owned kernels for adam-core coordinate table properties: derived
//! vectors, RIC rotations, unit conversions, covariance access, derived
//! orbital elements, unit-sphere projection, and origin gravitational
//! parameters. Python veneers extract Arrow columns once and make one
//! crossing per public property/function.

use adam_core_rs_coords::{
    bound_longitude_residuals_flat, calc_apoapsis_distance, calc_mean_motion, calc_period,
    calc_semi_latus_rectum, calc_semi_major_axis, chi2_survival, compute_residuals_chi2_flat,
    origin_mu_au3_day2, sample_coordinate_covariances_flat, sample_covariance_random_flat,
    sample_covariance_sigma_points_flat, OrbitVariantSamplingMethod, OriginId,
};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::hint::black_box;
use std::time::Instant;

const KM_P_AU: f64 = 149_597_870.7;
const S_P_DAY: f64 = 86_400.0;
const SPECIFIC_ANGULAR_MOMENTUM_TOLERANCE: f64 = 1e-20;

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

fn rows6(values: &PyReadonlyArray2<'_, f64>, label: &str) -> PyResult<Vec<f64>> {
    let view = values.as_array();
    if view.ncols() != 6 {
        return Err(PyValueError::new_err(format!(
            "{label} must have shape (N, 6)"
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

fn covariance_rows(
    values: &PyReadonlyArray3<'_, f64>,
    label: &str,
) -> PyResult<(Vec<f64>, usize, usize)> {
    let view = values.as_array();
    let shape = view.shape();
    if shape[1] != shape[2] {
        return Err(PyValueError::new_err(format!(
            "{label} must have shape (N, D, D)"
        )));
    }
    let flat = view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err(format!("{label} must be contiguous")))?
        .to_vec();
    Ok((flat, shape[0], shape[1]))
}

fn array1<'py>(py: Python<'py>, values: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
    values.into_pyarray(py)
}

fn array2<'py>(
    py: Python<'py>,
    values: Vec<f64>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    ndarray::Array2::from_shape_vec((rows, cols), values)
        .map(|array| array.into_pyarray(py))
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

fn array3<'py>(
    py: Python<'py>,
    values: Vec<f64>,
    rows: usize,
    dim: usize,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    ndarray::Array3::from_shape_vec((rows, dim, dim), values)
        .map(|array| array.into_pyarray(py))
        .map_err(|err| PyValueError::new_err(err.to_string()))
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

// ---------------------------------------------------------------------------
// Vector kernels
// ---------------------------------------------------------------------------

fn norm3(flat: &[f64]) -> Vec<f64> {
    flat.chunks_exact(3)
        .map(|row| (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt())
        .collect()
}

fn unit3(flat: &[f64]) -> Vec<f64> {
    flat.chunks_exact(3)
        .flat_map(|row| {
            let norm = (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt();
            [row[0] / norm, row[1] / norm, row[2] / norm]
        })
        .collect()
}

fn cross3(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.chunks_exact(3)
        .zip(b.chunks_exact(3))
        .flat_map(|(a, b)| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        })
        .collect()
}

fn h_mag_from_values(values: &[f64]) -> Vec<f64> {
    values
        .chunks_exact(6)
        .map(|row| {
            let h = [
                row[1] * row[5] - row[2] * row[4],
                row[2] * row[3] - row[0] * row[5],
                row[0] * row[4] - row[1] * row[3],
            ];
            (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]).sqrt()
        })
        .collect()
}

#[pyfunction]
fn row_norm3_numpy<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let flat = rows3(&values, "values")?;
    Ok(array1(py, norm3(&flat)))
}

#[pyfunction]
fn row_unit3_numpy<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let flat = rows3(&values, "values")?;
    let rows = flat.len() / 3;
    array2(py, unit3(&flat), rows, 3)
}

#[pyfunction]
fn row_cross3_numpy<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a = rows3(&a, "a")?;
    let b = rows3(&b, "b")?;
    if a.len() != b.len() {
        return Err(PyValueError::new_err("a and b must have the same shape"));
    }
    let rows = a.len() / 3;
    array2(py, cross3(&a, &b), rows, 3)
}

#[pyfunction]
fn cartesian_h_mag_numpy<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let flat = rows6(&values, "values")?;
    Ok(array1(py, h_mag_from_values(&flat)))
}

// ---------------------------------------------------------------------------
// RIC rotation matrices
// ---------------------------------------------------------------------------

fn ric3_flat(values: &[f64]) -> Vec<f64> {
    let rows = values.len() / 6;
    let mut out = vec![0.0_f64; rows * 9];
    for row in 0..rows {
        let state = &values[row * 6..row * 6 + 6];
        let mut radial = [state[0], state[1], state[2]];
        let velocity = [state[3], state[4], state[5]];
        let mut cross_track = [
            radial[1] * velocity[2] - radial[2] * velocity[1],
            radial[2] * velocity[0] - radial[0] * velocity[2],
            radial[0] * velocity[1] - radial[1] * velocity[0],
        ];
        let mut r_mag =
            (radial[0] * radial[0] + radial[1] * radial[1] + radial[2] * radial[2]).sqrt();
        let mut h_mag = (cross_track[0] * cross_track[0]
            + cross_track[1] * cross_track[1]
            + cross_track[2] * cross_track[2])
            .sqrt();
        // Degenerate orbital plane: identity policy matching the legacy
        // implementation (radial -> X, cross-track -> Z).
        if h_mag < SPECIFIC_ANGULAR_MOMENTUM_TOLERANCE {
            radial = [1.0, 0.0, 0.0];
            r_mag = 1.0;
            cross_track = [0.0, 0.0, 1.0];
            h_mag = 1.0;
        }
        let radial = [radial[0] / r_mag, radial[1] / r_mag, radial[2] / r_mag];
        let cross_track = [
            cross_track[0] / h_mag,
            cross_track[1] / h_mag,
            cross_track[2] / h_mag,
        ];
        let in_track = [
            cross_track[1] * radial[2] - cross_track[2] * radial[1],
            cross_track[2] * radial[0] - cross_track[0] * radial[2],
            cross_track[0] * radial[1] - cross_track[1] * radial[0],
        ];
        let base = row * 9;
        out[base..base + 3].copy_from_slice(&radial);
        out[base + 3..base + 6].copy_from_slice(&in_track);
        out[base + 6..base + 9].copy_from_slice(&cross_track);
    }
    out
}

fn ric6_flat(values: &[f64]) -> Vec<f64> {
    let rows = values.len() / 6;
    let ric3 = ric3_flat(values);
    let mut out = vec![0.0_f64; rows * 36];
    for row in 0..rows {
        for j in 0..3 {
            for k in 0..3 {
                let value = ric3[row * 9 + j * 3 + k];
                out[row * 36 + j * 6 + k] = value;
                out[row * 36 + (j + 3) * 6 + (k + 3)] = value;
            }
        }
    }
    out
}

#[pyfunction]
fn ric3_matrix_numpy<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let flat = rows6(&values, "values")?;
    let rows = flat.len() / 6;
    array3(py, ric3_flat(&flat), rows, 3)
}

#[pyfunction]
fn ric6_matrix_numpy<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let flat = rows6(&values, "values")?;
    let rows = flat.len() / 6;
    array3(py, ric6_flat(&flat), rows, 6)
}

#[pyfunction]
#[pyo3(signature = (values, reps, trials, warmup_reps=1))]
fn benchmark_ric_matrices_numpy(
    values: PyReadonlyArray2<'_, f64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let flat = rows6(&values, "values")?;
    bench(reps, trials, warmup_reps, || ric6_flat(&flat))
}

// ---------------------------------------------------------------------------
// Translation and unit conversions
// ---------------------------------------------------------------------------

fn translate6(values: &[f64], vector: &[f64]) -> Vec<f64> {
    let broadcast = vector.len() == 6;
    values
        .chunks_exact(6)
        .enumerate()
        .flat_map(|(row, state)| {
            let offset = if broadcast { 0 } else { row * 6 };
            [
                state[0] + vector[offset],
                state[1] + vector[offset + 1],
                state[2] + vector[offset + 2],
                state[3] + vector[offset + 3],
                state[4] + vector[offset + 4],
                state[5] + vector[offset + 5],
            ]
        })
        .collect()
}

#[pyfunction]
fn translate_cartesian_numpy<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    vector: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let values = rows6(&values, "values")?;
    let vector = rows6(&vector, "vector")?;
    let rows = values.len() / 6;
    if vector.len() != 6 && vector.len() != values.len() {
        return Err(PyValueError::new_err(
            "vector must have shape (1, 6) or (N, 6)",
        ));
    }
    array2(py, translate6(&values, &vector), rows, 6)
}

#[pyfunction]
fn au_to_km_numpy<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = scalars(&values, "values")?;
    Ok(array1(
        py,
        values.iter().map(|value| value * KM_P_AU).collect(),
    ))
}

#[pyfunction]
fn km_to_au_numpy<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = scalars(&values, "values")?;
    Ok(array1(
        py,
        values.iter().map(|value| value / KM_P_AU).collect(),
    ))
}

#[pyfunction]
fn au_per_day_to_km_per_s_numpy<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = scalars(&values, "values")?;
    Ok(array1(
        py,
        values
            .iter()
            .map(|value| value * KM_P_AU / S_P_DAY)
            .collect(),
    ))
}

#[pyfunction]
fn km_per_s_to_au_per_day_numpy<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = scalars(&values, "values")?;
    Ok(array1(
        py,
        values
            .iter()
            .map(|value| value / KM_P_AU * S_P_DAY)
            .collect(),
    ))
}

fn convert_values(flat: &[f64], au_to_km: bool) -> Vec<f64> {
    flat.chunks_exact(6)
        .flat_map(|row| {
            if au_to_km {
                [
                    row[0] * KM_P_AU,
                    row[1] * KM_P_AU,
                    row[2] * KM_P_AU,
                    row[3] * KM_P_AU / S_P_DAY,
                    row[4] * KM_P_AU / S_P_DAY,
                    row[5] * KM_P_AU / S_P_DAY,
                ]
            } else {
                [
                    row[0] / KM_P_AU,
                    row[1] / KM_P_AU,
                    row[2] / KM_P_AU,
                    row[3] / KM_P_AU * S_P_DAY,
                    row[4] / KM_P_AU * S_P_DAY,
                    row[5] / KM_P_AU * S_P_DAY,
                ]
            }
        })
        .collect()
}

fn convert_covariance(flat: &[f64], au_to_km: bool) -> Vec<f64> {
    // Matches the legacy outer-product conversion factor exactly.
    let unit = if au_to_km {
        [
            KM_P_AU,
            KM_P_AU,
            KM_P_AU,
            KM_P_AU / S_P_DAY,
            KM_P_AU / S_P_DAY,
            KM_P_AU / S_P_DAY,
        ]
    } else {
        [
            1.0 / KM_P_AU,
            1.0 / KM_P_AU,
            1.0 / KM_P_AU,
            S_P_DAY / KM_P_AU,
            S_P_DAY / KM_P_AU,
            S_P_DAY / KM_P_AU,
        ]
    };
    flat.chunks_exact(36)
        .flat_map(|matrix| {
            let mut out = [0.0_f64; 36];
            for j in 0..6 {
                for k in 0..6 {
                    out[j * 6 + k] = matrix[j * 6 + k] * (unit[j] * unit[k]);
                }
            }
            out
        })
        .collect()
}

#[pyfunction]
fn convert_cartesian_values_au_to_km_numpy<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let flat = rows6(&values, "values")?;
    let rows = flat.len() / 6;
    array2(py, convert_values(&flat, true), rows, 6)
}

#[pyfunction]
fn convert_cartesian_values_km_to_au_numpy<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let flat = rows6(&values, "values")?;
    let rows = flat.len() / 6;
    array2(py, convert_values(&flat, false), rows, 6)
}

#[pyfunction]
fn convert_cartesian_covariance_au_to_km_numpy<'py>(
    py: Python<'py>,
    covariances: PyReadonlyArray3<'py, f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let (flat, rows, dim) = covariance_rows(&covariances, "covariances")?;
    if dim != 6 {
        return Err(PyValueError::new_err(
            "covariances must have shape (N, 6, 6)",
        ));
    }
    array3(py, convert_covariance(&flat, true), rows, 6)
}

#[pyfunction]
fn convert_cartesian_covariance_km_to_au_numpy<'py>(
    py: Python<'py>,
    covariances: PyReadonlyArray3<'py, f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let (flat, rows, dim) = covariance_rows(&covariances, "covariances")?;
    if dim != 6 {
        return Err(PyValueError::new_err(
            "covariances must have shape (N, 6, 6)",
        ));
    }
    array3(py, convert_covariance(&flat, false), rows, 6)
}

#[pyfunction]
#[pyo3(signature = (values, covariances, reps, trials, warmup_reps=1))]
fn benchmark_cartesian_unit_conversions_numpy(
    values: PyReadonlyArray2<'_, f64>,
    covariances: PyReadonlyArray3<'_, f64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let values = rows6(&values, "values")?;
    let (covariances, _, _) = covariance_rows(&covariances, "covariances")?;
    bench(reps, trials, warmup_reps, || {
        (
            convert_values(&values, true),
            convert_values(&values, false),
            convert_covariance(&covariances, true),
            convert_covariance(&covariances, false),
        )
    })
}

// ---------------------------------------------------------------------------
// Covariance construction and access
// ---------------------------------------------------------------------------

fn covariance_sigmas(flat: &[f64], rows: usize, dim: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; rows * dim];
    for row in 0..rows {
        for j in 0..dim {
            out[row * dim + j] = flat[row * dim * dim + j * dim + j].sqrt();
        }
    }
    out
}

fn sigma_block_norm(flat: &[f64], rows: usize, dim: usize, start: usize, len: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; rows];
    for row in 0..rows {
        let mut total = 0.0_f64;
        for j in start..start + len {
            let sigma = flat[row * dim * dim + j * dim + j].sqrt();
            total += sigma * sigma;
        }
        out[row] = total.sqrt();
    }
    out
}

fn sigmas_to_covariances_flat(sigmas: &[f64], rows: usize, dim: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; rows * dim * dim];
    for row in 0..rows {
        for j in 0..dim {
            let sigma = sigmas[row * dim + j];
            out[row * dim * dim + j * dim + j] = sigma * sigma;
        }
    }
    out
}

#[pyfunction]
fn covariance_sigmas_numpy<'py>(
    py: Python<'py>,
    covariances: PyReadonlyArray3<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (flat, rows, dim) = covariance_rows(&covariances, "covariances")?;
    array2(py, covariance_sigmas(&flat, rows, dim), rows, dim)
}

#[pyfunction]
fn covariance_sigma_block_norm_numpy<'py>(
    py: Python<'py>,
    covariances: PyReadonlyArray3<'py, f64>,
    start: usize,
    len: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (flat, rows, dim) = covariance_rows(&covariances, "covariances")?;
    if start + len > dim {
        return Err(PyValueError::new_err("sigma block exceeds dimension"));
    }
    Ok(array1(py, sigma_block_norm(&flat, rows, dim, start, len)))
}

#[pyfunction]
fn sigmas_to_covariances_numpy<'py>(
    py: Python<'py>,
    sigmas: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let view = sigmas.as_array();
    let rows = view.nrows();
    let dim = view.ncols();
    let flat = view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("sigmas must be contiguous"))?;
    array3(py, sigmas_to_covariances_flat(flat, rows, dim), rows, dim)
}

#[pyfunction]
fn covariance_is_all_nan_numpy(covariances: PyReadonlyArray3<'_, f64>) -> PyResult<bool> {
    let (flat, _, _) = covariance_rows(&covariances, "covariances")?;
    Ok(flat.iter().all(|value| value.is_nan()))
}

#[pyfunction]
#[pyo3(signature = (covariances, reps, trials, warmup_reps=1))]
fn benchmark_covariance_sigmas_numpy(
    covariances: PyReadonlyArray3<'_, f64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let (flat, rows, dim) = covariance_rows(&covariances, "covariances")?;
    bench(reps, trials, warmup_reps, || {
        covariance_sigmas(&flat, rows, dim)
    })
}

#[pyfunction]
#[pyo3(signature = (sigmas, reps, trials, warmup_reps=1))]
fn benchmark_sigmas_to_covariances_numpy(
    sigmas: PyReadonlyArray2<'_, f64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let view = sigmas.as_array();
    let rows = view.nrows();
    let dim = view.ncols();
    let flat = view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("sigmas must be contiguous"))?
        .to_vec();
    bench(reps, trials, warmup_reps, || {
        sigmas_to_covariances_flat(&flat, rows, dim)
    })
}

// ---------------------------------------------------------------------------
// Derived orbital elements
// ---------------------------------------------------------------------------

fn map2(a: &[f64], b: &[f64], op: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&a, &b)| op(a, b)).collect()
}

#[pyfunction]
fn cometary_apoapsis_distance_numpy<'py>(
    py: Python<'py>,
    q: PyReadonlyArray1<'py, f64>,
    e: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let q = scalars(&q, "q")?;
    let e = scalars(&e, "e")?;
    Ok(array1(
        py,
        map2(&q, &e, |q, e| {
            calc_apoapsis_distance(calc_semi_major_axis(q, e), e)
        }),
    ))
}

#[pyfunction]
fn cometary_semi_latus_rectum_numpy<'py>(
    py: Python<'py>,
    q: PyReadonlyArray1<'py, f64>,
    e: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let q = scalars(&q, "q")?;
    let e = scalars(&e, "e")?;
    Ok(array1(
        py,
        map2(&q, &e, |q, e| {
            calc_semi_latus_rectum(calc_semi_major_axis(q, e), e)
        }),
    ))
}

#[pyfunction]
fn cometary_period_numpy<'py>(
    py: Python<'py>,
    q: PyReadonlyArray1<'py, f64>,
    e: PyReadonlyArray1<'py, f64>,
    mu: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let q = scalars(&q, "q")?;
    let e = scalars(&e, "e")?;
    let mu = scalars(&mu, "mu")?;
    Ok(array1(
        py,
        q.iter()
            .zip(e.iter())
            .zip(mu.iter())
            .map(|((&q, &e), &mu)| calc_period(calc_semi_major_axis(q, e), mu))
            .collect(),
    ))
}

#[pyfunction]
fn mean_motion_degrees_numpy<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    mu: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = scalars(&a, "a")?;
    let mu = scalars(&mu, "mu")?;
    Ok(array1(
        py,
        map2(&a, &mu, |a, mu| calc_mean_motion(a, mu).to_degrees()),
    ))
}

#[pyfunction]
fn cometary_mean_motion_degrees_numpy<'py>(
    py: Python<'py>,
    q: PyReadonlyArray1<'py, f64>,
    e: PyReadonlyArray1<'py, f64>,
    mu: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let q = scalars(&q, "q")?;
    let e = scalars(&e, "e")?;
    let mu = scalars(&mu, "mu")?;
    Ok(array1(
        py,
        q.iter()
            .zip(e.iter())
            .zip(mu.iter())
            .map(|((&q, &e), &mu)| calc_mean_motion(calc_semi_major_axis(q, e), mu).to_degrees())
            .collect(),
    ))
}

#[pyfunction]
fn period_from_origin_numpy<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    codes: Vec<String>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = scalars(&a, "a")?;
    let mu = origin_mu_values(&codes).map_err(PyValueError::new_err)?;
    if a.len() != mu.len() {
        return Err(PyValueError::new_err("a and codes must have equal length"));
    }
    Ok(array1(py, map2(&a, &mu, calc_period)))
}

#[pyfunction]
fn mean_motion_degrees_from_origin_numpy<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    codes: Vec<String>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = scalars(&a, "a")?;
    let mu = origin_mu_values(&codes).map_err(PyValueError::new_err)?;
    if a.len() != mu.len() {
        return Err(PyValueError::new_err("a and codes must have equal length"));
    }
    Ok(array1(
        py,
        map2(&a, &mu, |a, mu| calc_mean_motion(a, mu).to_degrees()),
    ))
}

#[pyfunction]
fn cometary_period_from_origin_numpy<'py>(
    py: Python<'py>,
    q: PyReadonlyArray1<'py, f64>,
    e: PyReadonlyArray1<'py, f64>,
    codes: Vec<String>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let q = scalars(&q, "q")?;
    let e = scalars(&e, "e")?;
    let mu = origin_mu_values(&codes).map_err(PyValueError::new_err)?;
    if q.len() != mu.len() || e.len() != mu.len() {
        return Err(PyValueError::new_err("q, e, codes must have equal length"));
    }
    Ok(array1(
        py,
        q.iter()
            .zip(e.iter())
            .zip(mu.iter())
            .map(|((&q, &e), &mu)| calc_period(calc_semi_major_axis(q, e), mu))
            .collect(),
    ))
}

#[pyfunction]
fn cometary_mean_motion_degrees_from_origin_numpy<'py>(
    py: Python<'py>,
    q: PyReadonlyArray1<'py, f64>,
    e: PyReadonlyArray1<'py, f64>,
    codes: Vec<String>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let q = scalars(&q, "q")?;
    let e = scalars(&e, "e")?;
    let mu = origin_mu_values(&codes).map_err(PyValueError::new_err)?;
    if q.len() != mu.len() || e.len() != mu.len() {
        return Err(PyValueError::new_err("q, e, codes must have equal length"));
    }
    Ok(array1(
        py,
        q.iter()
            .zip(e.iter())
            .zip(mu.iter())
            .map(|((&q, &e), &mu)| calc_mean_motion(calc_semi_major_axis(q, e), mu).to_degrees())
            .collect(),
    ))
}

#[pyfunction]
#[pyo3(signature = (q, e, mu, reps, trials, warmup_reps=1))]
fn benchmark_derived_elements_numpy(
    q: PyReadonlyArray1<'_, f64>,
    e: PyReadonlyArray1<'_, f64>,
    mu: PyReadonlyArray1<'_, f64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let q = scalars(&q, "q")?;
    let e = scalars(&e, "e")?;
    let mu = scalars(&mu, "mu")?;
    bench(reps, trials, warmup_reps, || {
        q.iter()
            .zip(e.iter())
            .zip(mu.iter())
            .map(|((&q, &e), &mu)| {
                let a = calc_semi_major_axis(q, e);
                (
                    calc_apoapsis_distance(a, e),
                    calc_semi_latus_rectum(a, e),
                    calc_period(a, mu),
                    calc_mean_motion(a, mu).to_degrees(),
                )
            })
            .collect::<Vec<_>>()
    })
}

// ---------------------------------------------------------------------------
// Unit sphere and origin mu
// ---------------------------------------------------------------------------

fn to_unit_sphere(flat: &[f64], only_missing: bool) -> Vec<f64> {
    flat.chunks_exact(6)
        .flat_map(|row| {
            let rho = if !only_missing || row[0].is_nan() {
                1.0
            } else {
                row[0]
            };
            let vrho = if !only_missing || row[3].is_nan() {
                0.0
            } else {
                row[3]
            };
            [rho, row[1], row[2], vrho, row[4], row[5]]
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (values, only_missing=false))]
fn spherical_to_unit_sphere_numpy<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    only_missing: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let flat = rows6(&values, "values")?;
    let rows = flat.len() / 6;
    array2(py, to_unit_sphere(&flat, only_missing), rows, 6)
}

fn origin_mu_values(codes: &[String]) -> Result<Vec<f64>, String> {
    codes
        .iter()
        .map(|code| {
            origin_mu_au3_day2(&OriginId::from_code(code.clone()))
                .map_err(|_| format!("Unknown origin code: {code}"))
        })
        .collect()
}

#[pyfunction]
fn origin_mu_numpy<'py>(
    py: Python<'py>,
    codes: Vec<String>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    origin_mu_values(&codes)
        .map(|values| array1(py, values))
        .map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (codes, reps, trials, warmup_reps=1))]
fn benchmark_origin_mu_numpy(
    codes: Vec<String>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    bench(reps, trials, warmup_reps, || origin_mu_values(&codes))
}

// ---------------------------------------------------------------------------
// Residual convenience surface
// ---------------------------------------------------------------------------

fn broadcast_predicted(predicted: &[f64], rows: usize) -> Result<Vec<f64>, String> {
    let predicted_rows = predicted.len() / 6;
    if predicted_rows == rows {
        Ok(predicted.to_vec())
    } else if predicted_rows == 1 {
        Ok(predicted.repeat(rows))
    } else {
        Err(format!(
            "Predicted coordinates must have length 1 or match observed length ({rows}), got {predicted_rows}."
        ))
    }
}

fn residual_values(
    observed: &[f64],
    predicted: &[f64],
    is_spherical: bool,
) -> Result<Vec<f64>, String> {
    let rows = observed.len() / 6;
    let predicted = broadcast_predicted(predicted, rows)?;
    let mut residuals: Vec<f64> = observed
        .iter()
        .zip(predicted.iter())
        .map(|(observed, predicted)| observed - predicted)
        .collect();
    if is_spherical {
        bound_longitude_residuals_flat(observed, &mut residuals, rows, 6);
        for row in 0..rows {
            let cos_lat = observed[row * 6 + 2].to_radians().cos();
            residuals[row * 6 + 1] *= cos_lat;
            residuals[row * 6 + 4] *= cos_lat;
        }
    }
    Ok(residuals)
}

#[pyfunction]
#[pyo3(signature = (observed, predicted, is_spherical=false))]
fn compute_residual_values_numpy<'py>(
    py: Python<'py>,
    observed: PyReadonlyArray2<'py, f64>,
    predicted: PyReadonlyArray2<'py, f64>,
    is_spherical: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let observed = rows6(&observed, "observed")?;
    let predicted = rows6(&predicted, "predicted")?;
    let rows = observed.len() / 6;
    let residuals =
        residual_values(&observed, &predicted, is_spherical).map_err(PyValueError::new_err)?;
    array2(py, residuals, rows, 6)
}

#[pyfunction]
#[pyo3(signature = (observed, predicted, is_spherical, reps, trials, warmup_reps=1))]
fn benchmark_compute_residual_values_numpy(
    observed: PyReadonlyArray2<'_, f64>,
    predicted: PyReadonlyArray2<'_, f64>,
    is_spherical: bool,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let observed = rows6(&observed, "observed")?;
    let predicted = rows6(&predicted, "predicted")?;
    bench(reps, trials, warmup_reps, || {
        residual_values(&observed, &predicted, is_spherical)
    })
}

#[pyfunction]
fn reduced_chi2_numpy(
    chi2: PyReadonlyArray1<'_, f64>,
    dof: PyReadonlyArray1<'_, i64>,
    parameters: i64,
) -> PyResult<f64> {
    let chi2 = scalars(&chi2, "chi2")?;
    let dof = dof
        .as_array()
        .as_slice()
        .map(<[i64]>::to_vec)
        .ok_or_else(|| PyValueError::new_err("dof must be contiguous"))?;
    let chi2_total: f64 = chi2.iter().sum();
    let dof_total: i64 = dof.iter().sum::<i64>() - parameters;
    Ok(chi2_total / dof_total as f64)
}

#[pyfunction]
fn chi2_survival_numpy<'py>(
    py: Python<'py>,
    chi2: PyReadonlyArray1<'py, f64>,
    dof: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let chi2 = scalars(&chi2, "chi2")?;
    let dof = scalars(&dof, "dof")?;
    if chi2.len() != dof.len() {
        return Err(PyValueError::new_err("chi2 and dof must have equal length"));
    }
    Ok(array1(
        py,
        chi2.iter()
            .zip(dof.iter())
            .map(|(&chi2, &dof)| {
                if chi2.is_nan() {
                    f64::NAN
                } else {
                    chi2_survival(chi2, dof)
                }
            })
            .collect(),
    ))
}

type ResidualsProbabilityResult<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    bool,
);

/// One-crossing custom-coordinates fallback for `Residuals.calculate`:
/// broadcast, NaN-covariance policy, fused residual/chi2 kernel, and
/// chi-squared survival probability all execute here.
#[pyfunction]
#[pyo3(signature = (observed, predicted, observed_covariances, predicted_covariances=None, is_spherical=false))]
fn compute_residuals_chi2_probability_numpy<'py>(
    py: Python<'py>,
    observed: PyReadonlyArray2<'py, f64>,
    predicted: PyReadonlyArray2<'py, f64>,
    observed_covariances: PyReadonlyArray3<'py, f64>,
    predicted_covariances: Option<PyReadonlyArray3<'py, f64>>,
    is_spherical: bool,
) -> PyResult<ResidualsProbabilityResult<'py>> {
    let observed = rows6(&observed, "observed")?;
    let rows = observed.len() / 6;
    let predicted = broadcast_predicted(&rows6(&predicted, "predicted")?, rows)
        .map_err(PyValueError::new_err)?;
    let (observed_cov, cov_rows, dim) =
        covariance_rows(&observed_covariances, "observed_covariances")?;
    if cov_rows != rows || dim != 6 {
        return Err(PyValueError::new_err(
            "observed_covariances must have shape (N, 6, 6)",
        ));
    }
    let predicted_cov = match predicted_covariances {
        Some(values) => {
            let (flat, pred_rows, dim) = covariance_rows(&values, "predicted_covariances")?;
            if dim != 6 {
                return Err(PyValueError::new_err(
                    "predicted_covariances must have shape (N, 6, 6)",
                ));
            }
            let mut broadcast = if pred_rows == rows {
                flat
            } else if pred_rows == 1 {
                flat.repeat(rows)
            } else {
                return Err(PyValueError::new_err(format!(
                    "Predicted covariance length must be 1 or {rows}, got {pred_rows}"
                )));
            };
            for value in broadcast.iter_mut() {
                if value.is_nan() {
                    *value = 0.0;
                }
            }
            broadcast
        }
        None => vec![0.0_f64; rows * 36],
    };
    let output = compute_residuals_chi2_flat(
        &observed,
        &predicted,
        &observed_cov,
        &predicted_cov,
        rows,
        6,
        is_spherical,
    )
    .map_err(|err| PyValueError::new_err(format!("residual computation failed: {err:?}")))?;
    let probability = output
        .chi2
        .iter()
        .zip(output.dof.iter())
        .map(|(&chi2, &dof)| {
            if chi2.is_nan() {
                f64::NAN
            } else {
                chi2_survival(chi2, dof as f64)
            }
        })
        .collect::<Vec<_>>();
    Ok((
        array2(py, output.residuals, rows, 6)?,
        array1(py, output.chi2),
        output.dof.into_pyarray(py),
        array1(py, probability),
        output.had_off_diagonal_nan,
    ))
}

// ---------------------------------------------------------------------------
// Orbit evaluation orchestration
// ---------------------------------------------------------------------------

type EvaluateOrbitsResult<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<bool>>,
    bool,
);

struct OrbitEvaluation {
    residuals: Vec<f64>,
    residual_chi2: Vec<f64>,
    residual_dof: Vec<i64>,
    residual_probability: Vec<f64>,
    orbit_chi2: Vec<f64>,
    reduced_chi2: Vec<f64>,
    arc_length: Vec<f64>,
    num_obs: Vec<i64>,
    observation_indices: Vec<i64>,
    member_outliers: Vec<bool>,
    had_off_diagonal_nan: bool,
}

#[allow(clippy::too_many_arguments)]
fn evaluate_orbits_flat(
    orbit_ids: &[String],
    ephemeris_orbit_ids: &[String],
    observation_ids: &[String],
    observed_origin_ids: &[String],
    predicted_origin_ids: &[String],
    observed_frame: &str,
    predicted_frame: &str,
    observed: &[f64],
    predicted: &[f64],
    observed_covariances: &[f64],
    predicted_covariances: &[f64],
    observation_days: &[i64],
    observation_nanos: &[i64],
    ignore: &[String],
    parameters: i64,
) -> Result<OrbitEvaluation, String> {
    let num_orbits = orbit_ids.len();
    let num_observations = observation_ids.len();
    let rows = num_orbits
        .checked_mul(num_observations)
        .ok_or_else(|| "orbit/observation product is too large".to_string())?;
    if observed.len() != num_observations * 6
        || observed_covariances.len() != num_observations * 36
        || observation_days.len() != num_observations
        || observation_nanos.len() != num_observations
        || observed_origin_ids.len() != num_observations
    {
        return Err("observation arrays must have equal length".to_string());
    }
    if predicted.len() != rows * 6
        || predicted_covariances.len() != rows * 36
        || ephemeris_orbit_ids.len() != rows
        || predicted_origin_ids.len() != rows
    {
        return Err(
            "Ephemeris rows must be grouped by sorted orbit_id with one block per observation; this is the documented Propagator.generate_ephemeris order."
                .to_string(),
        );
    }

    let expected_ids = orbit_ids
        .iter()
        .flat_map(|orbit_id| std::iter::repeat_n(orbit_id, num_observations));
    if !ephemeris_orbit_ids
        .iter()
        .zip(expected_ids)
        .all(|(actual, expected)| actual == expected)
    {
        return Err(
            "Ephemeris rows must be grouped by sorted orbit_id with one block per observation; this is the documented Propagator.generate_ephemeris order."
                .to_string(),
        );
    }

    if observed_frame != predicted_frame {
        return Err(format!(
            "Observed ({observed_frame}) and predicted ({predicted_frame}) coordinates must have the same frame."
        ));
    }
    let origins_all_differ = (0..rows)
        .all(|row| observed_origin_ids[row % num_observations] != predicted_origin_ids[row]);
    if origins_all_differ {
        return Err("Observed and predicted coordinates must have the same origin.".to_string());
    }

    let mut repeated_observed = Vec::with_capacity(rows * 6);
    let mut repeated_covariances = Vec::with_capacity(rows * 36);
    for _ in 0..num_orbits {
        repeated_observed.extend_from_slice(observed);
        repeated_covariances.extend_from_slice(observed_covariances);
    }
    let mut predicted_covariances = predicted_covariances.to_vec();
    for value in &mut predicted_covariances {
        if value.is_nan() {
            *value = 0.0;
        }
    }
    let output = compute_residuals_chi2_flat(
        &repeated_observed,
        predicted,
        &repeated_covariances,
        &predicted_covariances,
        rows,
        6,
        true,
    )
    .map_err(|err| format!("residual computation failed: {err:?}"))?;
    let residual_probability = output
        .chi2
        .iter()
        .zip(output.dof.iter())
        .map(|(&chi2, &dof)| {
            if chi2.is_nan() {
                f64::NAN
            } else {
                chi2_survival(chi2, dof as f64)
            }
        })
        .collect();

    let included = observation_ids
        .iter()
        .map(|id| !ignore.iter().any(|ignored| ignored == id))
        .collect::<Vec<_>>();
    let num_included = included.iter().filter(|&&value| value).count();
    if num_included == 0 {
        return Err(
            "zero-size array to reduction operation maximum which has no identity".to_string(),
        );
    }
    let included_times = observation_days
        .iter()
        .zip(observation_nanos)
        .zip(&included)
        .filter_map(|((&days, &nanos), &include)| {
            include.then_some(days as f64 + nanos as f64 / (S_P_DAY * 1e9))
        })
        .collect::<Vec<_>>();
    let min_time = included_times.iter().copied().fold(f64::INFINITY, f64::min);
    let max_time = included_times
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let arc = max_time - min_time;

    let mut orbit_chi2 = Vec::with_capacity(num_orbits);
    let mut reduced_chi2 = Vec::with_capacity(num_orbits);
    for orbit in 0..num_orbits {
        let offset = orbit * num_observations;
        let mut chi2 = 0.0;
        let mut dof = 0_i64;
        for (observation, &include) in included.iter().enumerate() {
            if include {
                chi2 += output.chi2[offset + observation];
                dof += output.dof[offset + observation];
            }
        }
        orbit_chi2.push(chi2);
        reduced_chi2.push(chi2 / (dof - parameters) as f64);
    }

    Ok(OrbitEvaluation {
        residuals: output.residuals,
        residual_chi2: output.chi2,
        residual_dof: output.dof,
        residual_probability,
        orbit_chi2,
        reduced_chi2,
        arc_length: vec![arc; num_orbits],
        num_obs: vec![num_included as i64; num_orbits],
        observation_indices: (0..num_orbits)
            .flat_map(|_| 0..num_observations as i64)
            .collect(),
        member_outliers: (0..num_orbits)
            .flat_map(|_| included.iter().map(|&value| !value))
            .collect(),
        had_off_diagonal_nan: output.had_off_diagonal_nan,
    })
}

#[pyfunction]
#[pyo3(signature = (orbit_ids, ephemeris_orbit_ids, observation_ids, observed_origin_ids, predicted_origin_ids, observed_frame, predicted_frame, observed, predicted, observed_covariances, predicted_covariances, observation_days, observation_nanos, ignore, parameters))]
#[allow(clippy::too_many_arguments)]
fn evaluate_orbits_numpy<'py>(
    py: Python<'py>,
    orbit_ids: Vec<String>,
    ephemeris_orbit_ids: Vec<String>,
    observation_ids: Vec<String>,
    observed_origin_ids: Vec<String>,
    predicted_origin_ids: Vec<String>,
    observed_frame: &str,
    predicted_frame: &str,
    observed: PyReadonlyArray2<'py, f64>,
    predicted: PyReadonlyArray2<'py, f64>,
    observed_covariances: PyReadonlyArray3<'py, f64>,
    predicted_covariances: PyReadonlyArray3<'py, f64>,
    observation_days: PyReadonlyArray1<'py, i64>,
    observation_nanos: PyReadonlyArray1<'py, i64>,
    ignore: Vec<String>,
    parameters: i64,
) -> PyResult<EvaluateOrbitsResult<'py>> {
    let observed = rows6(&observed, "observed")?;
    let predicted = rows6(&predicted, "predicted")?;
    let (observed_covariances, _, observed_dim) =
        covariance_rows(&observed_covariances, "observed_covariances")?;
    let (predicted_covariances, _, predicted_dim) =
        covariance_rows(&predicted_covariances, "predicted_covariances")?;
    if observed_dim != 6 || predicted_dim != 6 {
        return Err(PyValueError::new_err(
            "covariance arrays must have shape (N, 6, 6)",
        ));
    }
    let observation_days = observation_days
        .as_array()
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("observation_days must be contiguous"))?
        .to_vec();
    let observation_nanos = observation_nanos
        .as_array()
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("observation_nanos must be contiguous"))?
        .to_vec();
    let output = evaluate_orbits_flat(
        &orbit_ids,
        &ephemeris_orbit_ids,
        &observation_ids,
        &observed_origin_ids,
        &predicted_origin_ids,
        observed_frame,
        predicted_frame,
        &observed,
        &predicted,
        &observed_covariances,
        &predicted_covariances,
        &observation_days,
        &observation_nanos,
        &ignore,
        parameters,
    )
    .map_err(PyValueError::new_err)?;
    let rows = output.residual_chi2.len();
    Ok((
        array2(py, output.residuals, rows, 6)?,
        array1(py, output.residual_chi2),
        output.residual_dof.into_pyarray(py),
        array1(py, output.residual_probability),
        array1(py, output.orbit_chi2),
        array1(py, output.reduced_chi2),
        array1(py, output.arc_length),
        output.num_obs.into_pyarray(py),
        output.observation_indices.into_pyarray(py),
        output.member_outliers.into_pyarray(py),
        output.had_off_diagonal_nan,
    ))
}

#[pyfunction]
#[pyo3(signature = (orbit_ids, ephemeris_orbit_ids, observation_ids, observed_origin_ids, predicted_origin_ids, observed_frame, predicted_frame, observed, predicted, observed_covariances, predicted_covariances, observation_days, observation_nanos, parameters, ignore, reps, trials, warmup_reps=1))]
#[allow(clippy::too_many_arguments)]
fn benchmark_evaluate_orbits_numpy(
    orbit_ids: Vec<String>,
    ephemeris_orbit_ids: Vec<String>,
    observation_ids: Vec<String>,
    observed_origin_ids: Vec<String>,
    predicted_origin_ids: Vec<String>,
    observed_frame: &str,
    predicted_frame: &str,
    observed: PyReadonlyArray2<'_, f64>,
    predicted: PyReadonlyArray2<'_, f64>,
    observed_covariances: PyReadonlyArray3<'_, f64>,
    predicted_covariances: PyReadonlyArray3<'_, f64>,
    observation_days: PyReadonlyArray1<'_, i64>,
    observation_nanos: PyReadonlyArray1<'_, i64>,
    parameters: i64,
    ignore: Vec<String>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let observed = rows6(&observed, "observed")?;
    let predicted = rows6(&predicted, "predicted")?;
    let (observed_covariances, _, observed_dim) =
        covariance_rows(&observed_covariances, "observed_covariances")?;
    let (predicted_covariances, _, predicted_dim) =
        covariance_rows(&predicted_covariances, "predicted_covariances")?;
    if observed_dim != 6 || predicted_dim != 6 {
        return Err(PyValueError::new_err(
            "covariance arrays must have shape (N, 6, 6)",
        ));
    }
    let observation_days = observation_days
        .as_array()
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("observation_days must be contiguous"))?
        .to_vec();
    let observation_nanos = observation_nanos
        .as_array()
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("observation_nanos must be contiguous"))?
        .to_vec();
    let run = || {
        evaluate_orbits_flat(
            &orbit_ids,
            &ephemeris_orbit_ids,
            &observation_ids,
            &observed_origin_ids,
            &predicted_origin_ids,
            observed_frame,
            predicted_frame,
            &observed,
            &predicted,
            &observed_covariances,
            &predicted_covariances,
            &observation_days,
            &observation_nanos,
            &ignore,
            parameters,
        )
    };
    run().map_err(PyValueError::new_err)?;
    bench(reps, trials, warmup_reps, run)
}

// ---------------------------------------------------------------------------
// Covariance sampling
// ---------------------------------------------------------------------------

fn parse_sampling_method(method: &str) -> PyResult<OrbitVariantSamplingMethod> {
    match method {
        "auto" => Ok(OrbitVariantSamplingMethod::Auto),
        "sigma-point" => Ok(OrbitVariantSamplingMethod::SigmaPoint),
        "monte-carlo" => Ok(OrbitVariantSamplingMethod::MonteCarlo),
        other => Err(PyValueError::new_err(format!(
            "Unknown coordinate covariance sampling method: {other}"
        ))),
    }
}

type SamplingResult<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
);

#[pyfunction]
#[pyo3(signature = (means, covariances, method, num_samples=10000, seed=None, alpha=1.0, beta=0.0, kappa=0.0))]
#[allow(clippy::too_many_arguments)]
fn sample_coordinate_variants_numpy<'py>(
    py: Python<'py>,
    means: PyReadonlyArray2<'py, f64>,
    covariances: PyReadonlyArray3<'py, f64>,
    method: &str,
    num_samples: usize,
    seed: Option<u64>,
    alpha: f64,
    beta: f64,
    kappa: f64,
) -> PyResult<SamplingResult<'py>> {
    let means = rows6(&means, "means")?;
    let (covariances, _, dim) = covariance_rows(&covariances, "covariances")?;
    if dim != 6 {
        return Err(PyValueError::new_err(
            "covariances must have shape (N, 6, 6)",
        ));
    }
    let method = parse_sampling_method(method)?;
    let (samples, weights, weights_cov, source_rows) = py
        .allow_threads(|| {
            sample_coordinate_covariances_flat(
                &means,
                &covariances,
                method,
                num_samples,
                seed,
                alpha,
                beta,
                kappa,
            )
        })
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let rows = weights.len();
    Ok((
        array2(py, samples, rows, 6)?,
        array1(py, weights),
        array1(py, weights_cov),
        source_rows.into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (means, covariances, method, reps, trials, warmup_reps=1, num_samples=10000, seed=None, alpha=1.0, beta=0.0, kappa=0.0))]
#[allow(clippy::too_many_arguments)]
fn benchmark_sample_coordinate_variants_numpy(
    means: PyReadonlyArray2<'_, f64>,
    covariances: PyReadonlyArray3<'_, f64>,
    method: &str,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
    num_samples: usize,
    seed: Option<u64>,
    alpha: f64,
    beta: f64,
    kappa: f64,
) -> PyResult<Vec<Vec<f64>>> {
    let means = rows6(&means, "means")?;
    let (covariances, _, _) = covariance_rows(&covariances, "covariances")?;
    let method = parse_sampling_method(method)?;
    bench(reps, trials, warmup_reps, || {
        sample_coordinate_covariances_flat(
            &means,
            &covariances,
            method,
            num_samples,
            seed,
            alpha,
            beta,
            kappa,
        )
    })
}

type SingleSampleResult<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

#[pyfunction]
#[pyo3(signature = (mean, covariance, alpha=1.0, beta=0.0, kappa=0.0))]
fn sample_covariance_sigma_points_numpy<'py>(
    py: Python<'py>,
    mean: PyReadonlyArray1<'py, f64>,
    covariance: PyReadonlyArray2<'py, f64>,
    alpha: f64,
    beta: f64,
    kappa: f64,
) -> PyResult<SingleSampleResult<'py>> {
    let mean = scalars(&mean, "mean")?;
    let view = covariance.as_array();
    let flat = view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("covariance must be contiguous"))?;
    if mean.len() != 6 || view.shape() != [6, 6] {
        return Err(PyValueError::new_err(
            "mean must have shape (6,) and covariance (6, 6)",
        ));
    }
    let (samples, weights, weights_cov) =
        sample_covariance_sigma_points_flat(&mean, flat, alpha, beta, kappa)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let rows = weights.len();
    Ok((
        array2(py, samples, rows, 6)?,
        array1(py, weights),
        array1(py, weights_cov),
    ))
}

#[pyfunction]
#[pyo3(signature = (mean, covariance, num_samples=10000, seed=None))]
fn sample_covariance_random_numpy<'py>(
    py: Python<'py>,
    mean: PyReadonlyArray1<'py, f64>,
    covariance: PyReadonlyArray2<'py, f64>,
    num_samples: usize,
    seed: Option<u64>,
) -> PyResult<SingleSampleResult<'py>> {
    let mean = scalars(&mean, "mean")?;
    let view = covariance.as_array();
    let flat = view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("covariance must be contiguous"))?;
    if mean.len() != 6 || view.shape() != [6, 6] {
        return Err(PyValueError::new_err(
            "mean must have shape (6,) and covariance (6, 6)",
        ));
    }
    let (samples, weights, weights_cov) =
        sample_covariance_random_flat(&mean, flat, num_samples, seed)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let rows = weights.len();
    Ok((
        array2(py, samples, rows, 6)?,
        array1(py, weights),
        array1(py, weights_cov),
    ))
}

// ---------------------------------------------------------------------------
// Observatory geodesy
// ---------------------------------------------------------------------------

type LonLatResult<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

/// Observatory longitude/geodetic-latitude from MPC parallax coefficients,
/// matching the legacy NumPy composition (NaN rows are space-based sites).
#[pyfunction]
fn observatory_lon_lat_numpy<'py>(
    py: Python<'py>,
    longitude: PyReadonlyArray1<'py, f64>,
    cos_phi: PyReadonlyArray1<'py, f64>,
    sin_phi: PyReadonlyArray1<'py, f64>,
    e2: f64,
) -> PyResult<LonLatResult<'py>> {
    let longitude = scalars(&longitude, "longitude")?;
    let cos_phi = scalars(&cos_phi, "cos_phi")?;
    let sin_phi = scalars(&sin_phi, "sin_phi")?;
    if cos_phi.len() != longitude.len() || sin_phi.len() != longitude.len() {
        return Err(PyValueError::new_err(
            "longitude, cos_phi, sin_phi must have equal length",
        ));
    }
    let mut lon_out = Vec::with_capacity(longitude.len());
    let mut lat_out = Vec::with_capacity(longitude.len());
    for row in 0..longitude.len() {
        if longitude[row].is_nan() {
            lon_out.push(f64::NAN);
            lat_out.push(f64::NAN);
            continue;
        }
        let mut lon = longitude[row];
        if lon > 180.0 {
            lon -= 360.0;
        }
        let tan_phi_geo = sin_phi[row] / cos_phi[row];
        let latitude_geodetic = (tan_phi_geo / (1.0 - e2)).atan();
        lon_out.push(lon);
        lat_out.push(latitude_geodetic.to_degrees());
    }
    Ok((array1(py, lon_out), array1(py, lat_out)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(row_norm3_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(row_unit3_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(row_cross3_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(cartesian_h_mag_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(ric3_matrix_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(ric6_matrix_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_ric_matrices_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(translate_cartesian_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(au_to_km_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(km_to_au_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(au_per_day_to_km_per_s_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(km_per_s_to_au_per_day_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(
        convert_cartesian_values_au_to_km_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        convert_cartesian_values_km_to_au_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        convert_cartesian_covariance_au_to_km_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        convert_cartesian_covariance_km_to_au_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        benchmark_cartesian_unit_conversions_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(covariance_sigmas_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(covariance_sigma_block_norm_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(sigmas_to_covariances_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(covariance_is_all_nan_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_covariance_sigmas_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_sigmas_to_covariances_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(cometary_apoapsis_distance_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(cometary_semi_latus_rectum_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(cometary_period_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(mean_motion_degrees_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(cometary_mean_motion_degrees_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(period_from_origin_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(mean_motion_degrees_from_origin_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(cometary_period_from_origin_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(
        cometary_mean_motion_degrees_from_origin_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(benchmark_derived_elements_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(spherical_to_unit_sphere_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(origin_mu_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_origin_mu_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(compute_residual_values_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(
        benchmark_compute_residual_values_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(reduced_chi2_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(chi2_survival_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(
        compute_residuals_chi2_probability_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(evaluate_orbits_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_evaluate_orbits_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(sample_coordinate_variants_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(
        benchmark_sample_coordinate_variants_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(sample_covariance_sigma_points_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(sample_covariance_random_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(observatory_lon_lat_numpy, m)?)?;
    Ok(())
}
