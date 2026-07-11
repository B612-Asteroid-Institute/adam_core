use adam_core_rs_coords::{fit_absolute_magnitude_grouped, fit_absolute_magnitude_rows};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::hint::black_box;
use std::time::Instant;

#[pyfunction]
fn fit_absolute_magnitude_rows_numpy<'py>(
    _py: Python<'py>,
    h_rows: PyReadonlyArray1<'py, f64>,
    sigma_rows: PyReadonlyArray1<'py, f64>,
) -> PyResult<(f64, f64, f64, f64, i64)> {
    let h = h_rows.as_array();
    let s = sigma_rows.as_array();
    if h.len() != s.len() {
        return Err(PyValueError::new_err(
            "h_rows and sigma_rows must be equal length",
        ));
    }
    let h_slice = h
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("h_rows must be contiguous"))?;
    let s_slice = s
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("sigma_rows must be contiguous"))?;
    let f = fit_absolute_magnitude_rows(h_slice, s_slice);
    Ok((f.h_hat, f.h_sigma, f.sigma_eff, f.chi2_red, f.n_used))
}

#[pyfunction]
fn fit_absolute_magnitude_grouped_numpy<'py>(
    py: Python<'py>,
    h_rows: PyReadonlyArray1<'py, f64>,
    sigma_rows: PyReadonlyArray1<'py, f64>,
    group_offsets: PyReadonlyArray1<'py, i64>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
)> {
    let h = h_rows.as_array();
    let s = sigma_rows.as_array();
    let off = group_offsets.as_array();
    if h.len() != s.len() {
        return Err(PyValueError::new_err(
            "h_rows and sigma_rows must be equal length",
        ));
    }
    let off_usize: Vec<usize> = off.iter().map(|&v| v as usize).collect();
    let h_slice = h
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("h_rows must be contiguous"))?;
    let s_slice = s
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("sigma_rows must be contiguous"))?;
    let fits = fit_absolute_magnitude_grouped(h_slice, s_slice, &off_usize);
    let h_hat: Vec<f64> = fits.iter().map(|f| f.h_hat).collect();
    let h_sigma: Vec<f64> = fits.iter().map(|f| f.h_sigma).collect();
    let sigma_eff: Vec<f64> = fits.iter().map(|f| f.sigma_eff).collect();
    let chi2_red: Vec<f64> = fits.iter().map(|f| f.chi2_red).collect();
    let n_used: Vec<i64> = fits.iter().map(|f| f.n_used).collect();
    Ok((
        ndarray::Array1::from_vec(h_hat).into_pyarray(py),
        ndarray::Array1::from_vec(h_sigma).into_pyarray(py),
        ndarray::Array1::from_vec(sigma_eff).into_pyarray(py),
        ndarray::Array1::from_vec(chi2_red).into_pyarray(py),
        ndarray::Array1::from_vec(n_used).into_pyarray(py),
    ))
}

#[pyfunction]
fn calculate_phase_angle_numpy<'py>(
    py: Python<'py>,
    object_pos: PyReadonlyArray2<'py, f64>,
    observer_pos: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let obj_arr = object_pos.as_array();
    let obs_arr = observer_pos.as_array();
    if obj_arr.ncols() != 3 {
        return Err(PyValueError::new_err("object_pos must have shape (N, 3)"));
    }
    let n = obj_arr.nrows();
    if obs_arr.nrows() != n || obs_arr.ncols() != 3 {
        return Err(PyValueError::new_err(
            "observer_pos must have shape (N, 3) matching object_pos rows",
        ));
    }
    let obj_slice = obj_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("object_pos must be contiguous"))?;
    let obs_slice = obs_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("observer_pos must be contiguous"))?;
    let out = PyArray1::<f64>::zeros(py, n, false);
    {
        let mut out_rw = numpy::PyArrayMethods::readwrite(&out);
        let out_slice = out_rw
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("allocated phase output must be contiguous"))?;
        adam_core_rs_coords::calculate_phase_angle_into(obj_slice, obs_slice, out_slice);
    }
    Ok(out)
}

#[pyfunction]
fn calculate_apparent_magnitude_v_numpy<'py>(
    py: Python<'py>,
    h_v: PyReadonlyArray1<'py, f64>,
    object_pos: PyReadonlyArray2<'py, f64>,
    observer_pos: PyReadonlyArray2<'py, f64>,
    g: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let obj_arr = object_pos.as_array();
    let obs_arr = observer_pos.as_array();
    let h_arr = h_v.as_array();
    let g_arr = g.as_array();
    if obj_arr.ncols() != 3 {
        return Err(PyValueError::new_err("object_pos must have shape (N, 3)"));
    }
    let n = obj_arr.nrows();
    if obs_arr.nrows() != n || obs_arr.ncols() != 3 {
        return Err(PyValueError::new_err(
            "observer_pos must have shape (N, 3) matching object_pos rows",
        ));
    }
    if h_arr.len() != n || g_arr.len() != n {
        return Err(PyValueError::new_err(
            "h_v and g must each have length N for positions shape (N, 3)",
        ));
    }
    let obj_slice = obj_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("object_pos must be contiguous"))?;
    let obs_slice = obs_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("observer_pos must be contiguous"))?;
    let h_slice = h_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("h_v must be contiguous"))?;
    let g_slice = g_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("g must be contiguous"))?;
    let out = PyArray1::<f64>::zeros(py, n, false);
    {
        let mut out_rw = numpy::PyArrayMethods::readwrite(&out);
        let out_slice = out_rw
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("allocated magnitude output must be contiguous"))?;
        adam_core_rs_coords::calculate_apparent_magnitude_v_into(
            h_slice, obj_slice, obs_slice, g_slice, out_slice,
        );
    }
    Ok(out)
}

#[pyfunction]
fn calculate_apparent_magnitude_v_and_phase_angle_numpy<'py>(
    py: Python<'py>,
    h_v: PyReadonlyArray1<'py, f64>,
    object_pos: PyReadonlyArray2<'py, f64>,
    observer_pos: PyReadonlyArray2<'py, f64>,
    g: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let obj_arr = object_pos.as_array();
    let obs_arr = observer_pos.as_array();
    let h_arr = h_v.as_array();
    let g_arr = g.as_array();
    if obj_arr.ncols() != 3 {
        return Err(PyValueError::new_err("object_pos must have shape (N, 3)"));
    }
    let n = obj_arr.nrows();
    if obs_arr.nrows() != n || obs_arr.ncols() != 3 {
        return Err(PyValueError::new_err(
            "observer_pos must have shape (N, 3) matching object_pos rows",
        ));
    }
    if h_arr.len() != n || g_arr.len() != n {
        return Err(PyValueError::new_err(
            "h_v and g must each have length N for positions shape (N, 3)",
        ));
    }
    let obj_slice = obj_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("object_pos must be contiguous"))?;
    let obs_slice = obs_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("observer_pos must be contiguous"))?;
    let h_slice = h_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("h_v must be contiguous"))?;
    let g_slice = g_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("g must be contiguous"))?;
    let mag_out = PyArray1::<f64>::zeros(py, n, false);
    let alpha_out = PyArray1::<f64>::zeros(py, n, false);
    {
        let mut mag_rw = numpy::PyArrayMethods::readwrite(&mag_out);
        let mut alpha_rw = numpy::PyArrayMethods::readwrite(&alpha_out);
        let mag_slice = mag_rw
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("allocated magnitude output must be contiguous"))?;
        let alpha_slice = alpha_rw
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("allocated phase output must be contiguous"))?;
        adam_core_rs_coords::calculate_apparent_magnitude_v_and_phase_angle_into(
            h_slice,
            obj_slice,
            obs_slice,
            g_slice,
            mag_slice,
            alpha_slice,
        );
    }
    Ok((mag_out, alpha_out))
}

#[pyfunction]
fn predict_magnitudes_bandpass_numpy<'py>(
    py: Python<'py>,
    h_v: PyReadonlyArray1<'py, f64>,
    object_pos: PyReadonlyArray2<'py, f64>,
    observer_pos: PyReadonlyArray2<'py, f64>,
    g: PyReadonlyArray1<'py, f64>,
    target_ids: PyReadonlyArray1<'py, i32>,
    delta_table: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let obj_arr = object_pos.as_array();
    let obs_arr = observer_pos.as_array();
    let h_arr = h_v.as_array();
    let g_arr = g.as_array();
    let tid_arr = target_ids.as_array();
    let dt_arr = delta_table.as_array();
    if obj_arr.ncols() != 3 {
        return Err(PyValueError::new_err("object_pos must have shape (N, 3)"));
    }
    let n = obj_arr.nrows();
    if obs_arr.nrows() != n || obs_arr.ncols() != 3 {
        return Err(PyValueError::new_err(
            "observer_pos must have shape (N, 3) matching object_pos rows",
        ));
    }
    if h_arr.len() != n || g_arr.len() != n || tid_arr.len() != n {
        return Err(PyValueError::new_err(
            "h_v, g, target_ids must each have length N for positions shape (N, 3)",
        ));
    }
    let obj_slice = obj_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("object_pos must be contiguous"))?;
    let obs_slice = obs_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("observer_pos must be contiguous"))?;
    let h_slice = h_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("h_v must be contiguous"))?;
    let g_slice = g_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("g must be contiguous"))?;
    let tid_slice = tid_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("target_ids must be contiguous"))?;
    let dt_slice = dt_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("delta_table must be contiguous"))?;
    let out = PyArray1::<f64>::zeros(py, n, false);
    {
        let mut out_rw = numpy::PyArrayMethods::readwrite(&out);
        let out_slice = out_rw
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("allocated predict output must be contiguous"))?;
        adam_core_rs_coords::predict_magnitudes_bandpass_into(
            h_slice, obj_slice, obs_slice, g_slice, tid_slice, dt_slice, out_slice,
        );
    }
    Ok(out)
}

fn benchmark_samples<T, F>(
    reps: usize,
    trials: usize,
    warmup_reps: usize,
    mut run_once: F,
) -> PyResult<Vec<Vec<f64>>>
where
    F: FnMut() -> T,
{
    if reps == 0 || trials == 0 {
        return Err(PyValueError::new_err("reps and trials must be >= 1"));
    }
    let mut sample_trials = Vec::with_capacity(trials);
    for _ in 0..trials {
        for _ in 0..warmup_reps {
            black_box(run_once());
        }
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let started = Instant::now();
            let output = run_once();
            black_box(&output);
            samples.push(started.elapsed().as_secs_f64());
        }
        sample_trials.push(samples);
    }
    Ok(sample_trials)
}

fn owned_positions(
    object_pos: &PyReadonlyArray2<'_, f64>,
    observer_pos: &PyReadonlyArray2<'_, f64>,
) -> PyResult<(Vec<f64>, Vec<f64>, usize)> {
    let object = object_pos.as_array();
    let observer = observer_pos.as_array();
    if object.ncols() != 3 {
        return Err(PyValueError::new_err("object_pos must have shape (N, 3)"));
    }
    let n = object.nrows();
    if observer.nrows() != n || observer.ncols() != 3 {
        return Err(PyValueError::new_err(
            "observer_pos must have shape (N, 3) matching object_pos rows",
        ));
    }
    let object = object
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("object_pos must be contiguous"))?
        .to_vec();
    let observer = observer
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("observer_pos must be contiguous"))?
        .to_vec();
    Ok((object, observer, n))
}

/// Rust-owned Instant timer for the phase-angle parity surface. NumPy/PyO3
/// extraction happens once above the timing loop; every sample allocates its
/// semantic output and calls the Rust kernel directly.
#[pyfunction]
#[pyo3(signature = (object_pos, observer_pos, reps, trials, warmup_reps=1))]
fn benchmark_calculate_phase_angle_numpy(
    object_pos: PyReadonlyArray2<'_, f64>,
    observer_pos: PyReadonlyArray2<'_, f64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let (object, observer, n) = owned_positions(&object_pos, &observer_pos)?;
    benchmark_samples(reps, trials, warmup_reps, || {
        let mut output = vec![0.0; n];
        adam_core_rs_coords::calculate_phase_angle_into(&object, &observer, &mut output);
        output
    })
}

#[pyfunction]
#[pyo3(signature = (h_v, object_pos, observer_pos, g, reps, trials, warmup_reps=1))]
fn benchmark_calculate_apparent_magnitude_v_numpy(
    h_v: PyReadonlyArray1<'_, f64>,
    object_pos: PyReadonlyArray2<'_, f64>,
    observer_pos: PyReadonlyArray2<'_, f64>,
    g: PyReadonlyArray1<'_, f64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let (object, observer, n) = owned_positions(&object_pos, &observer_pos)?;
    let h = h_v.as_array();
    let g = g.as_array();
    if h.len() != n || g.len() != n {
        return Err(PyValueError::new_err("h_v and g must each have length N"));
    }
    let h = h
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("h_v must be contiguous"))?;
    let g = g
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("g must be contiguous"))?;
    benchmark_samples(reps, trials, warmup_reps, || {
        let mut output = vec![0.0; n];
        adam_core_rs_coords::calculate_apparent_magnitude_v_into(
            h,
            &object,
            &observer,
            g,
            &mut output,
        );
        output
    })
}

#[pyfunction]
#[pyo3(signature = (h_v, object_pos, observer_pos, g, reps, trials, warmup_reps=1))]
fn benchmark_calculate_apparent_magnitude_v_and_phase_angle_numpy(
    h_v: PyReadonlyArray1<'_, f64>,
    object_pos: PyReadonlyArray2<'_, f64>,
    observer_pos: PyReadonlyArray2<'_, f64>,
    g: PyReadonlyArray1<'_, f64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let (object, observer, n) = owned_positions(&object_pos, &observer_pos)?;
    let h = h_v.as_array();
    let g = g.as_array();
    if h.len() != n || g.len() != n {
        return Err(PyValueError::new_err("h_v and g must each have length N"));
    }
    let h = h
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("h_v must be contiguous"))?;
    let g = g
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("g must be contiguous"))?;
    benchmark_samples(reps, trials, warmup_reps, || {
        let mut magnitude = vec![0.0; n];
        let mut phase_angle = vec![0.0; n];
        adam_core_rs_coords::calculate_apparent_magnitude_v_and_phase_angle_into(
            h,
            &object,
            &observer,
            g,
            &mut magnitude,
            &mut phase_angle,
        );
        (magnitude, phase_angle)
    })
}

#[pyfunction]
#[pyo3(signature = (h_v, object_pos, observer_pos, g, target_ids, delta_table, reps, trials, warmup_reps=1))]
fn benchmark_predict_magnitudes_bandpass_numpy(
    h_v: PyReadonlyArray1<'_, f64>,
    object_pos: PyReadonlyArray2<'_, f64>,
    observer_pos: PyReadonlyArray2<'_, f64>,
    g: PyReadonlyArray1<'_, f64>,
    target_ids: PyReadonlyArray1<'_, i32>,
    delta_table: PyReadonlyArray1<'_, f64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let (object, observer, n) = owned_positions(&object_pos, &observer_pos)?;
    let h = h_v.as_array();
    let g = g.as_array();
    let targets = target_ids.as_array();
    let deltas = delta_table.as_array();
    if h.len() != n || g.len() != n || targets.len() != n {
        return Err(PyValueError::new_err(
            "h_v, g, and target_ids must each have length N",
        ));
    }
    let h = h
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("h_v must be contiguous"))?;
    let g = g
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("g must be contiguous"))?;
    let targets = targets
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("target_ids must be contiguous"))?;
    let deltas = deltas
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("delta_table must be contiguous"))?;
    benchmark_samples(reps, trials, warmup_reps, || {
        let mut output = vec![0.0; n];
        adam_core_rs_coords::predict_magnitudes_bandpass_into(
            h,
            &object,
            &observer,
            g,
            targets,
            deltas,
            &mut output,
        );
        output
    })
}

/// Rust-owned timer for one-group fitting. The complete fit, including its
/// per-fit residual/MAD setup and allocations, is inside each semantic sample.
#[pyfunction]
#[pyo3(signature = (h_rows, sigma_rows, reps, trials, warmup_reps=1))]
fn benchmark_fit_absolute_magnitude_rows_numpy(
    h_rows: PyReadonlyArray1<'_, f64>,
    sigma_rows: PyReadonlyArray1<'_, f64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let h = h_rows.as_array();
    let sigma = sigma_rows.as_array();
    if h.len() != sigma.len() {
        return Err(PyValueError::new_err(
            "h_rows and sigma_rows must be equal length",
        ));
    }
    let h = h
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("h_rows must be contiguous"))?;
    let sigma = sigma
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("sigma_rows must be contiguous"))?;
    benchmark_samples(reps, trials, warmup_reps, || {
        fit_absolute_magnitude_rows(h, sigma)
    })
}

/// Rust-owned timer for grouped fitting. Offset conversion/validation is input
/// preparation outside samples; each sample includes grouped orchestration,
/// per-group row-fit setup, Rayon collection, and result allocation.
#[pyfunction]
#[pyo3(signature = (h_rows, sigma_rows, group_offsets, reps, trials, warmup_reps=1))]
fn benchmark_fit_absolute_magnitude_grouped_numpy(
    h_rows: PyReadonlyArray1<'_, f64>,
    sigma_rows: PyReadonlyArray1<'_, f64>,
    group_offsets: PyReadonlyArray1<'_, i64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let h = h_rows.as_array();
    let sigma = sigma_rows.as_array();
    let offsets = group_offsets.as_array();
    if h.len() != sigma.len() {
        return Err(PyValueError::new_err(
            "h_rows and sigma_rows must be equal length",
        ));
    }
    let h = h
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("h_rows must be contiguous"))?;
    let sigma = sigma
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("sigma_rows must be contiguous"))?;
    let offsets = offsets
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("group_offsets must be contiguous"))?;
    if offsets.len() < 2
        || offsets[0] != 0
        || offsets.last().copied() != Some(h.len() as i64)
        || offsets.windows(2).any(|pair| pair[0] > pair[1])
        || offsets.iter().any(|&offset| offset < 0)
    {
        return Err(PyValueError::new_err(
            "group_offsets must be monotonic from 0 through the row count",
        ));
    }
    let offsets: Vec<usize> = offsets.iter().map(|&offset| offset as usize).collect();
    benchmark_samples(reps, trials, warmup_reps, || {
        fit_absolute_magnitude_grouped(h, sigma, &offsets)
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_absolute_magnitude_rows_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(fit_absolute_magnitude_grouped_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_phase_angle_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_apparent_magnitude_v_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(
        calculate_apparent_magnitude_v_and_phase_angle_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(predict_magnitudes_bandpass_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_calculate_phase_angle_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(
        benchmark_calculate_apparent_magnitude_v_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        benchmark_calculate_apparent_magnitude_v_and_phase_angle_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        benchmark_predict_magnitudes_bandpass_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        benchmark_fit_absolute_magnitude_rows_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        benchmark_fit_absolute_magnitude_grouped_numpy,
        m
    )?)?;
    Ok(())
}
