use adam_core_rs_coords::{fit_absolute_magnitude_grouped, fit_absolute_magnitude_rows};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
    Ok(())
}
