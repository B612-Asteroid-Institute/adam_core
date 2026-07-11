//! Rust-owned `Instant` benchmark entrypoints for flat coordinate kernels.
//!
//! NumPy/PyO3 input extraction happens once at entry. Every recorded interval
//! contains only a direct call to an `adam_core_rs_coords` kernel.

use std::hint::black_box;
use std::time::Instant;

use adam_core_rs_coords::{
    apply_cosine_latitude_correction_flat, bound_longitude_residuals_flat,
    calculate_chi2_flat, cartesian_to_cometary_flat6, cartesian_to_geodetic_flat6,
    cartesian_to_keplerian_flat6, cartesian_to_spherical_flat6, classify_orbits_flat,
    cometary_to_cartesian_flat6, keplerian_to_cartesian_flat6,
    rotate_cartesian_time_varying_flat6, spherical_to_cartesian_flat6,
    weighted_covariance_flat, weighted_mean_flat,
};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn samples<F>(reps: usize, trials: usize, warmup: usize, mut run: F) -> PyResult<Vec<Vec<f64>>>
where
    F: FnMut() -> PyResult<()>,
{
    if reps == 0 || trials == 0 {
        return Err(PyValueError::new_err("reps and trials must be >= 1"));
    }
    let mut trial_samples = Vec::with_capacity(trials);
    for _ in 0..trials {
        for _ in 0..warmup {
            run()?;
        }
        let mut trial = Vec::with_capacity(reps);
        for _ in 0..reps {
            let start = Instant::now();
            run()?;
            trial.push(start.elapsed().as_secs_f64());
        }
        trial_samples.push(trial);
    }
    Ok(trial_samples)
}

fn flat6<'a>(array: &'a PyReadonlyArray2<'_, f64>) -> PyResult<(&'a [f64], usize)> {
    let view = array.as_array();
    if view.ncols() != 6 {
        return Err(PyValueError::new_err("coords must have shape (N, 6)"));
    }
    let rows = view.nrows();
    let flat = view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("array must be contiguous"))?;
    Ok((flat, rows))
}

#[pyfunction]
#[pyo3(signature = (coords, reps, trials, warmup=1))]
fn benchmark_cartesian_to_spherical_native(
    coords: PyReadonlyArray2<'_, f64>, reps: usize, trials: usize, warmup: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let (coords, _) = flat6(&coords)?;
    samples(reps, trials, warmup, || {
        black_box(cartesian_to_spherical_flat6(black_box(coords)));
        Ok(())
    })
}

#[pyfunction]
#[pyo3(signature = (coords, reps, trials, warmup=1))]
fn benchmark_spherical_to_cartesian_native(
    coords: PyReadonlyArray2<'_, f64>, reps: usize, trials: usize, warmup: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let (coords, _) = flat6(&coords)?;
    samples(reps, trials, warmup, || {
        black_box(spherical_to_cartesian_flat6(black_box(coords)));
        Ok(())
    })
}

#[pyfunction]
#[pyo3(signature = (coords, a, f, max_iter, tol, reps, trials, warmup=1))]
fn benchmark_cartesian_to_geodetic_native(
    coords: PyReadonlyArray2<'_, f64>, a: f64, f: f64, max_iter: usize, tol: f64,
    reps: usize, trials: usize, warmup: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let (coords, _) = flat6(&coords)?;
    samples(reps, trials, warmup, || {
        black_box(cartesian_to_geodetic_flat6(coords, a, f, max_iter, tol));
        Ok(())
    })
}

#[pyfunction]
#[pyo3(signature = (coords, t0, mu, reps, trials, warmup=1))]
fn benchmark_cartesian_to_keplerian_native(
    coords: PyReadonlyArray2<'_, f64>, t0: PyReadonlyArray1<'_, f64>,
    mu: PyReadonlyArray1<'_, f64>, reps: usize, trials: usize, warmup: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let (coords, rows) = flat6(&coords)?;
    let t0 = t0.as_slice()?;
    let mu = mu.as_slice()?;
    if t0.len() != rows || mu.len() != rows {
        return Err(PyValueError::new_err("t0 and mu must have length N"));
    }
    samples(reps, trials, warmup, || {
        black_box(cartesian_to_keplerian_flat6(coords, t0, mu));
        Ok(())
    })
}

#[pyfunction]
#[pyo3(signature = (coords, mu, max_iter, tol, reps, trials, warmup=1))]
fn benchmark_keplerian_to_cartesian_native(
    coords: PyReadonlyArray2<'_, f64>, mu: PyReadonlyArray1<'_, f64>,
    max_iter: usize, tol: f64, reps: usize, trials: usize, warmup: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let (coords, rows) = flat6(&coords)?;
    let mu = mu.as_slice()?;
    if mu.len() != rows { return Err(PyValueError::new_err("mu must have length N")); }
    samples(reps, trials, warmup, || {
        black_box(keplerian_to_cartesian_flat6(coords, mu, max_iter, tol));
        Ok(())
    })
}

#[pyfunction]
#[pyo3(signature = (coords, t0, mu, reps, trials, warmup=1))]
fn benchmark_cartesian_to_cometary_native(
    coords: PyReadonlyArray2<'_, f64>, t0: PyReadonlyArray1<'_, f64>,
    mu: PyReadonlyArray1<'_, f64>, reps: usize, trials: usize, warmup: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let (coords, rows) = flat6(&coords)?;
    let t0 = t0.as_slice()?;
    let mu = mu.as_slice()?;
    if t0.len() != rows || mu.len() != rows {
        return Err(PyValueError::new_err("t0 and mu must have length N"));
    }
    samples(reps, trials, warmup, || {
        black_box(cartesian_to_cometary_flat6(coords, t0, mu));
        Ok(())
    })
}

#[pyfunction]
#[pyo3(signature = (coords, t0, mu, max_iter, tol, reps, trials, warmup=1))]
fn benchmark_cometary_to_cartesian_native(
    coords: PyReadonlyArray2<'_, f64>, t0: PyReadonlyArray1<'_, f64>,
    mu: PyReadonlyArray1<'_, f64>, max_iter: usize, tol: f64,
    reps: usize, trials: usize, warmup: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let (coords, rows) = flat6(&coords)?;
    let t0 = t0.as_slice()?;
    let mu = mu.as_slice()?;
    if t0.len() != rows || mu.len() != rows {
        return Err(PyValueError::new_err("t0 and mu must have length N"));
    }
    samples(reps, trials, warmup, || {
        black_box(cometary_to_cartesian_flat6(coords, t0, mu, max_iter, tol));
        Ok(())
    })
}

#[pyfunction]
#[pyo3(signature = (coords, time_index, matrices, covariances, reps, trials, warmup=1))]
fn benchmark_rotate_cartesian_time_varying_native(
    coords: PyReadonlyArray2<'_, f64>, time_index: PyReadonlyArray1<'_, i64>,
    matrices: PyReadonlyArray3<'_, f64>, covariances: PyReadonlyArray2<'_, f64>,
    reps: usize, trials: usize, warmup: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let (coords, rows) = flat6(&coords)?;
    let covariance_view = covariances.as_array();
    if covariance_view.shape() != [rows, 36] {
        return Err(PyValueError::new_err("covariances must have shape (N, 36)"));
    }
    let covariances = covariance_view.as_slice().ok_or_else(|| PyValueError::new_err("covariances must be contiguous"))?;
    let indices: Vec<usize> = time_index.as_slice()?.iter().map(|&value| usize::try_from(value)
        .map_err(|_| PyValueError::new_err("time_index must be non-negative"))).collect::<PyResult<_>>()?;
    if indices.len() != rows { return Err(PyValueError::new_err("time_index must have length N")); }
    let matrix_view = matrices.as_array();
    if matrix_view.shape().get(1..) != Some(&[6, 6][..]) {
        return Err(PyValueError::new_err("matrices must have shape (U, 6, 6)"));
    }
    let matrices = matrix_view.as_slice().ok_or_else(|| PyValueError::new_err("matrices must be contiguous"))?;
    samples(reps, trials, warmup, || {
        black_box(rotate_cartesian_time_varying_flat6(coords, covariances, &indices, matrices)
            .map_err(PyValueError::new_err)?);
        Ok(())
    })
}

#[pyfunction]
#[pyo3(signature = (residuals, covariances, reps, trials, warmup=1))]
fn benchmark_calculate_chi2_native(
    residuals: PyReadonlyArray2<'_, f64>, covariances: PyReadonlyArray3<'_, f64>,
    reps: usize, trials: usize, warmup: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let residual_view = residuals.as_array();
    let n = residual_view.nrows();
    let d = residual_view.ncols();
    let residuals = residual_view.as_slice().ok_or_else(|| PyValueError::new_err("residuals must be contiguous"))?;
    let covariance_view = covariances.as_array();
    if covariance_view.shape() != [n, d, d] { return Err(PyValueError::new_err("covariances must have shape (N, D, D)")); }
    let covariances = covariance_view.as_slice().ok_or_else(|| PyValueError::new_err("covariances must be contiguous"))?;
    samples(reps, trials, warmup, || {
        black_box(calculate_chi2_flat(residuals, covariances, n, d)
            .map_err(|err| PyValueError::new_err(format!("chi2 failed: {err:?}")))?);
        Ok(())
    })
}

#[pyfunction]
#[pyo3(signature = (observed, residuals, reps, trials, warmup=1))]
fn benchmark_bound_longitude_residuals_native(
    observed: PyReadonlyArray2<'_, f64>, residuals: PyReadonlyArray2<'_, f64>,
    reps: usize, trials: usize, warmup: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let observed_view = observed.as_array();
    let residual_view = residuals.as_array();
    if observed_view.shape() != residual_view.shape() { return Err(PyValueError::new_err("shapes must match")); }
    let n = observed_view.nrows();
    let d = observed_view.ncols();
    let observed = observed_view.as_slice().ok_or_else(|| PyValueError::new_err("observed must be contiguous"))?;
    let residuals = residual_view.as_slice().ok_or_else(|| PyValueError::new_err("residuals must be contiguous"))?;
    samples(reps, trials, warmup, || {
        let mut output = residuals.to_vec();
        bound_longitude_residuals_flat(observed, &mut output, n, d);
        black_box(output);
        Ok(())
    })
}

#[pyfunction]
#[pyo3(signature = (lat, residuals, covariances, reps, trials, warmup=1))]
fn benchmark_apply_cosine_latitude_correction_native(
    lat: PyReadonlyArray1<'_, f64>, residuals: PyReadonlyArray2<'_, f64>,
    covariances: PyReadonlyArray3<'_, f64>, reps: usize, trials: usize, warmup: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let lat = lat.as_slice()?;
    let residual_view = residuals.as_array();
    let n = residual_view.nrows();
    let d = residual_view.ncols();
    if lat.len() != n { return Err(PyValueError::new_err("lat must have length N")); }
    let residuals = residual_view.as_slice().ok_or_else(|| PyValueError::new_err("residuals must be contiguous"))?;
    let covariance_view = covariances.as_array();
    if covariance_view.shape() != [n, d, d] { return Err(PyValueError::new_err("covariances must have shape (N, D, D)")); }
    let covariances = covariance_view.as_slice().ok_or_else(|| PyValueError::new_err("covariances must be contiguous"))?;
    samples(reps, trials, warmup, || {
        let mut residual_output = residuals.to_vec();
        let mut covariance_output = covariances.to_vec();
        apply_cosine_latitude_correction_flat(lat, &mut residual_output, &mut covariance_output, n, d);
        black_box((residual_output, covariance_output));
        Ok(())
    })
}

#[pyfunction]
#[pyo3(signature = (values, weights, reps, trials, warmup=1))]
fn benchmark_weighted_mean_native(
    values: PyReadonlyArray2<'_, f64>, weights: PyReadonlyArray1<'_, f64>,
    reps: usize, trials: usize, warmup: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let view = values.as_array();
    let n = view.nrows();
    let d = view.ncols();
    let values = view.as_slice().ok_or_else(|| PyValueError::new_err("samples must be contiguous"))?;
    let weights = weights.as_slice()?;
    if weights.len() != n { return Err(PyValueError::new_err("weights must have length N")); }
    samples(reps, trials, warmup, || {
        black_box(weighted_mean_flat(values, weights, n, d));
        Ok(())
    })
}

#[pyfunction]
#[pyo3(signature = (mean, values, weights, reps, trials, warmup=1))]
fn benchmark_weighted_covariance_native(
    mean: PyReadonlyArray1<'_, f64>, values: PyReadonlyArray2<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>, reps: usize, trials: usize, warmup: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let mean = mean.as_slice()?;
    let view = values.as_array();
    let n = view.nrows();
    let d = view.ncols();
    if mean.len() != d { return Err(PyValueError::new_err("mean must have length D")); }
    let values = view.as_slice().ok_or_else(|| PyValueError::new_err("samples must be contiguous"))?;
    let weights = weights.as_slice()?;
    if weights.len() != n { return Err(PyValueError::new_err("weights must have length N")); }
    samples(reps, trials, warmup, || {
        black_box(weighted_covariance_flat(mean, values, weights, n, d));
        Ok(())
    })
}

#[pyfunction]
#[pyo3(signature = (a, e, q, q_apo, reps, trials, warmup=1))]
fn benchmark_classify_orbits_native(
    a: PyReadonlyArray1<'_, f64>, e: PyReadonlyArray1<'_, f64>,
    q: PyReadonlyArray1<'_, f64>, q_apo: PyReadonlyArray1<'_, f64>,
    reps: usize, trials: usize, warmup: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let a = a.as_slice()?;
    let e = e.as_slice()?;
    let q = q.as_slice()?;
    let q_apo = q_apo.as_slice()?;
    if e.len() != a.len() || q.len() != a.len() || q_apo.len() != a.len() {
        return Err(PyValueError::new_err("a, e, q, Q must have equal length"));
    }
    samples(reps, trials, warmup, || {
        black_box(classify_orbits_flat(a, e, q, q_apo));
        Ok(())
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(benchmark_cartesian_to_spherical_native, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_spherical_to_cartesian_native, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_cartesian_to_geodetic_native, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_cartesian_to_keplerian_native, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_keplerian_to_cartesian_native, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_cartesian_to_cometary_native, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_cometary_to_cartesian_native, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_rotate_cartesian_time_varying_native, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_calculate_chi2_native, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_bound_longitude_residuals_native, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_apply_cosine_latitude_correction_native, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_weighted_mean_native, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_weighted_covariance_native, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_classify_orbits_native, m)?)?;
    Ok(())
}
