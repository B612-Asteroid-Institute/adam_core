use adam_core_rs_orbit_determination::{
    calc_gauss_row, calc_gibbs_row, calc_herrick_gibbs_row, gauss_iod_fused,
    gauss_iod_orbits_from_roots,
};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
fn calc_gibbs_numpy<'py>(
    py: Python<'py>,
    r1: PyReadonlyArray1<'py, f64>,
    r2: PyReadonlyArray1<'py, f64>,
    r3: PyReadonlyArray1<'py, f64>,
    mu: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let r1_arr = r1.as_array();
    let r2_arr = r2.as_array();
    let r3_arr = r3.as_array();
    if r1_arr.len() != 3 || r2_arr.len() != 3 || r3_arr.len() != 3 {
        return Err(PyValueError::new_err(
            "r1, r2, and r3 must each have shape (3,)",
        ));
    }

    let out = calc_gibbs_row(
        [r1_arr[0], r1_arr[1], r1_arr[2]],
        [r2_arr[0], r2_arr[1], r2_arr[2]],
        [r3_arr[0], r3_arr[1], r3_arr[2]],
        mu,
    );

    Ok(ndarray::Array1::from_vec(out.to_vec()).into_pyarray_bound(py))
}

#[pyfunction]
fn calc_herrick_gibbs_numpy<'py>(
    py: Python<'py>,
    r1: PyReadonlyArray1<'py, f64>,
    r2: PyReadonlyArray1<'py, f64>,
    r3: PyReadonlyArray1<'py, f64>,
    t1: f64,
    t2: f64,
    t3: f64,
    mu: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let r1_arr = r1.as_array();
    let r2_arr = r2.as_array();
    let r3_arr = r3.as_array();
    if r1_arr.len() != 3 || r2_arr.len() != 3 || r3_arr.len() != 3 {
        return Err(PyValueError::new_err(
            "r1, r2, and r3 must each have shape (3,)",
        ));
    }

    let out = calc_herrick_gibbs_row(
        [r1_arr[0], r1_arr[1], r1_arr[2]],
        [r2_arr[0], r2_arr[1], r2_arr[2]],
        [r3_arr[0], r3_arr[1], r3_arr[2]],
        t1,
        t2,
        t3,
        mu,
    );

    Ok(ndarray::Array1::from_vec(out.to_vec()).into_pyarray_bound(py))
}

#[pyfunction]
fn calc_gauss_numpy<'py>(
    py: Python<'py>,
    r1: PyReadonlyArray1<'py, f64>,
    r2: PyReadonlyArray1<'py, f64>,
    r3: PyReadonlyArray1<'py, f64>,
    t1: f64,
    t2: f64,
    t3: f64,
    mu: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let r1_arr = r1.as_array();
    let r2_arr = r2.as_array();
    let r3_arr = r3.as_array();
    if r1_arr.len() != 3 || r2_arr.len() != 3 || r3_arr.len() != 3 {
        return Err(PyValueError::new_err(
            "r1, r2, and r3 must each have shape (3,)",
        ));
    }

    let out = calc_gauss_row(
        [r1_arr[0], r1_arr[1], r1_arr[2]],
        [r2_arr[0], r2_arr[1], r2_arr[2]],
        [r3_arr[0], r3_arr[1], r3_arr[2]],
        t1,
        t2,
        t3,
        mu,
    );

    Ok(ndarray::Array1::from_vec(out.to_vec()).into_pyarray_bound(py))
}

#[pyfunction]
fn gauss_iod_orbits_numpy<'py>(
    py: Python<'py>,
    r2_mags: PyReadonlyArray1<'py, f64>,
    q1: PyReadonlyArray1<'py, f64>,
    q2: PyReadonlyArray1<'py, f64>,
    q3: PyReadonlyArray1<'py, f64>,
    rho1_hat: PyReadonlyArray1<'py, f64>,
    rho2_hat: PyReadonlyArray1<'py, f64>,
    rho3_hat: PyReadonlyArray1<'py, f64>,
    t1: f64,
    t2: f64,
    t3: f64,
    v: f64,
    velocity_method: &str,
    light_time: bool,
    mu: f64,
    c: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let r2_mags_arr = r2_mags.as_array();
    let q1_arr = q1.as_array();
    let q2_arr = q2.as_array();
    let q3_arr = q3.as_array();
    let rho1_hat_arr = rho1_hat.as_array();
    let rho2_hat_arr = rho2_hat.as_array();
    let rho3_hat_arr = rho3_hat.as_array();
    if q1_arr.len() != 3
        || q2_arr.len() != 3
        || q3_arr.len() != 3
        || rho1_hat_arr.len() != 3
        || rho2_hat_arr.len() != 3
        || rho3_hat_arr.len() != 3
    {
        return Err(PyValueError::new_err(
            "q1, q2, q3, rho1_hat, rho2_hat, and rho3_hat must each have shape (3,)",
        ));
    }

    let velocity_method_id = if velocity_method == "gauss" {
        0
    } else if velocity_method == "gibbs" {
        1
    } else if velocity_method == "herrick+gibbs" {
        2
    } else {
        return Err(PyValueError::new_err(
            "velocity_method should be one of {'gauss', 'gibbs', 'herrick+gibbs'}",
        ));
    };

    let (epochs, orbits_flat) = gauss_iod_orbits_from_roots(
        r2_mags_arr
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("r2_mags must be contiguous"))?,
        [q1_arr[0], q1_arr[1], q1_arr[2]],
        [q2_arr[0], q2_arr[1], q2_arr[2]],
        [q3_arr[0], q3_arr[1], q3_arr[2]],
        [rho1_hat_arr[0], rho1_hat_arr[1], rho1_hat_arr[2]],
        [rho2_hat_arr[0], rho2_hat_arr[1], rho2_hat_arr[2]],
        [rho3_hat_arr[0], rho3_hat_arr[1], rho3_hat_arr[2]],
        t1,
        t2,
        t3,
        v,
        velocity_method_id,
        light_time,
        mu,
        c,
    );
    let n = epochs.len();
    let epochs_arr = ndarray::Array1::from_vec(epochs).into_pyarray_bound(py);
    let orbits_arr = ndarray::Array2::from_shape_vec((n, 6), orbits_flat)
        .map_err(|e| PyValueError::new_err(format!("failed to shape output: {e}")))?
        .into_pyarray_bound(py);

    Ok((epochs_arr, orbits_arr))
}

#[pyfunction]
#[pyo3(signature = (
    ra_deg, dec_deg, obs_times_mjd, coords_obs,
    velocity_method, light_time, mu, c,
))]
fn gauss_iod_fused_numpy<'py>(
    py: Python<'py>,
    ra_deg: PyReadonlyArray1<'py, f64>,
    dec_deg: PyReadonlyArray1<'py, f64>,
    obs_times_mjd: PyReadonlyArray1<'py, f64>,
    coords_obs: PyReadonlyArray2<'py, f64>,
    velocity_method: &str,
    light_time: bool,
    mu: f64,
    c: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let ra_arr = ra_deg.as_array();
    let dec_arr = dec_deg.as_array();
    let t_arr = obs_times_mjd.as_array();
    let obs_arr = coords_obs.as_array();
    if ra_arr.len() != 3 || dec_arr.len() != 3 || t_arr.len() != 3 {
        return Err(PyValueError::new_err(
            "ra_deg, dec_deg, and obs_times_mjd must each have length 3",
        ));
    }
    if obs_arr.nrows() != 3 || obs_arr.ncols() != 3 {
        return Err(PyValueError::new_err(
            "coords_obs must have shape (3, 3) — three heliocentric observer positions",
        ));
    }

    let velocity_method_id = if velocity_method == "gauss" {
        0
    } else if velocity_method == "gibbs" {
        1
    } else if velocity_method == "herrick+gibbs" {
        2
    } else {
        return Err(PyValueError::new_err(
            "velocity_method should be one of {'gauss', 'gibbs', 'herrick+gibbs'}",
        ));
    };

    let ra_in = [ra_arr[0], ra_arr[1], ra_arr[2]];
    let dec_in = [dec_arr[0], dec_arr[1], dec_arr[2]];
    let t_in = [t_arr[0], t_arr[1], t_arr[2]];
    let obs_in = [
        [obs_arr[[0, 0]], obs_arr[[0, 1]], obs_arr[[0, 2]]],
        [obs_arr[[1, 0]], obs_arr[[1, 1]], obs_arr[[1, 2]]],
        [obs_arr[[2, 0]], obs_arr[[2, 1]], obs_arr[[2, 2]]],
    ];

    let (epochs, orbits_flat) = gauss_iod_fused(
        ra_in,
        dec_in,
        t_in,
        obs_in,
        velocity_method_id,
        light_time,
        mu,
        c,
    );
    let n = epochs.len();
    let epochs_arr = ndarray::Array1::from_vec(epochs).into_pyarray_bound(py);
    let orbits_arr = ndarray::Array2::from_shape_vec((n, 6), orbits_flat)
        .map_err(|e| PyValueError::new_err(format!("failed to shape output: {e}")))?
        .into_pyarray_bound(py);

    Ok((epochs_arr, orbits_arr))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calc_gibbs_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calc_herrick_gibbs_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calc_gauss_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(gauss_iod_orbits_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(gauss_iod_fused_numpy, m)?)?;
    Ok(())
}
