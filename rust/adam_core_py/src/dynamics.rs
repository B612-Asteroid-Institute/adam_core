use adam_core_rs_coords::{
    add_light_time_batch_flat, apply_lagrange_coefficients, apply_stellar_aberration_row,
    calc_apoapsis_distance, calc_chi, calc_lagrange_coefficients, calc_mean_anomaly,
    calc_mean_motion_batch, calc_periapsis_distance, calc_period, calc_semi_latus_rectum,
    calc_semi_major_axis, calc_stumpff, calculate_moid, calculate_moid_batch,
    generate_ephemeris_2body_flat6, generate_ephemeris_2body_with_covariance_flat6,
    izzo_lambert_batch_flat, porkchop_grid_flat, propagate_2body_along_arc,
    propagate_2body_arc_batch_flat6, propagate_2body_flat6, propagate_2body_with_covariance_flat6,
    solve_barker, solve_kepler_true_anomaly, tisserand_parameter_flat,
};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::hint::black_box;
use std::time::Instant;

fn benchmark_samples<T, F>(
    reps: usize,
    trials: usize,
    warmup_reps: usize,
    mut run_once: F,
) -> Vec<Vec<f64>>
where
    F: FnMut() -> T,
{
    let mut trial_samples = Vec::with_capacity(trials);
    for _ in 0..trials {
        for _ in 0..warmup_reps {
            black_box(run_once());
        }
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let started = Instant::now();
            black_box(run_once());
            samples.push(started.elapsed().as_secs_f64());
        }
        trial_samples.push(samples);
    }
    trial_samples
}

#[pyfunction]
fn calc_mean_motion_numpy<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    mu: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a_arr = a.as_array();
    let mu_arr = mu.as_array();
    if a_arr.len() != mu_arr.len() {
        return Err(PyValueError::new_err("a and mu must have the same length"));
    }

    let out = calc_mean_motion_batch(
        a_arr
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("a must be contiguous"))?,
        mu_arr
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("mu must be contiguous"))?,
    );

    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn calc_period_numpy<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    mu: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a_arr = a.as_array();
    let mu_arr = mu.as_array();
    if a_arr.len() != mu_arr.len() {
        return Err(PyValueError::new_err("a and mu must have the same length"));
    }
    let a_slice = a_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("a must be contiguous"))?;
    let mu_slice = mu_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("mu must be contiguous"))?;
    let out: Vec<f64> = a_slice
        .iter()
        .zip(mu_slice)
        .map(|(&ai, &mi)| calc_period(ai, mi))
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn calc_periapsis_distance_numpy<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    e: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a_arr = a.as_array();
    let e_arr = e.as_array();
    if a_arr.len() != e_arr.len() {
        return Err(PyValueError::new_err("a and e must have the same length"));
    }
    let a_slice = a_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("a must be contiguous"))?;
    let e_slice = e_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("e must be contiguous"))?;
    let out: Vec<f64> = a_slice
        .iter()
        .zip(e_slice)
        .map(|(&ai, &ei)| calc_periapsis_distance(ai, ei))
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn calc_apoapsis_distance_numpy<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    e: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a_arr = a.as_array();
    let e_arr = e.as_array();
    if a_arr.len() != e_arr.len() {
        return Err(PyValueError::new_err("a and e must have the same length"));
    }
    let a_slice = a_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("a must be contiguous"))?;
    let e_slice = e_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("e must be contiguous"))?;
    let out: Vec<f64> = a_slice
        .iter()
        .zip(e_slice)
        .map(|(&ai, &ei)| calc_apoapsis_distance(ai, ei))
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn calc_semi_major_axis_numpy<'py>(
    py: Python<'py>,
    q: PyReadonlyArray1<'py, f64>,
    e: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let q_arr = q.as_array();
    let e_arr = e.as_array();
    if q_arr.len() != e_arr.len() {
        return Err(PyValueError::new_err("q and e must have the same length"));
    }
    let q_slice = q_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("q must be contiguous"))?;
    let e_slice = e_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("e must be contiguous"))?;
    let out: Vec<f64> = q_slice
        .iter()
        .zip(e_slice)
        .map(|(&qi, &ei)| calc_semi_major_axis(qi, ei))
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn calc_semi_latus_rectum_numpy<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    e: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a_arr = a.as_array();
    let e_arr = e.as_array();
    if a_arr.len() != e_arr.len() {
        return Err(PyValueError::new_err("a and e must have the same length"));
    }
    let a_slice = a_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("a must be contiguous"))?;
    let e_slice = e_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("e must be contiguous"))?;
    let out: Vec<f64> = a_slice
        .iter()
        .zip(e_slice)
        .map(|(&ai, &ei)| calc_semi_latus_rectum(ai, ei))
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn calc_mean_anomaly_numpy<'py>(
    py: Python<'py>,
    nu: PyReadonlyArray1<'py, f64>,
    e: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let nu_arr = nu.as_array();
    let e_arr = e.as_array();
    if nu_arr.len() != e_arr.len() {
        return Err(PyValueError::new_err("nu and e must have the same length"));
    }
    let nu_slice = nu_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("nu must be contiguous"))?;
    let e_slice = e_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("e must be contiguous"))?;
    let out: Vec<f64> = nu_slice
        .iter()
        .zip(e_slice)
        .map(|(&nui, &ei)| calc_mean_anomaly(nui, ei))
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn solve_barker_numpy<'py>(
    py: Python<'py>,
    m: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let m_arr = m.as_array();
    let m_slice = m_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("m must be contiguous"))?;
    let out: Vec<f64> = m_slice.iter().map(|&mi| solve_barker(mi)).collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (e, m, max_iter=100, tol=1e-15))]
fn solve_kepler_numpy<'py>(
    py: Python<'py>,
    e: PyReadonlyArray1<'py, f64>,
    m: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let e_arr = e.as_array();
    let m_arr = m.as_array();
    if e_arr.len() != m_arr.len() {
        return Err(PyValueError::new_err("e and m must have the same length"));
    }
    let e_slice = e_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("e must be contiguous"))?;
    let m_slice = m_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("m must be contiguous"))?;
    let out: Vec<f64> = e_slice
        .iter()
        .zip(m_slice)
        .map(|(&ei, &mi)| solve_kepler_true_anomaly(ei, mi, max_iter, tol))
        .collect();
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn calc_stumpff_numpy<'py>(
    py: Python<'py>,
    psi: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let psi_arr = psi.as_array();
    let psi_slice = psi_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("psi must be contiguous"))?;
    let mut out = Vec::with_capacity(psi_slice.len() * 6);
    for &psi_i in psi_slice {
        out.extend_from_slice(&calc_stumpff::<f64>(psi_i));
    }
    let shaped = ndarray::Array2::from_shape_vec((psi_slice.len(), 6), out)
        .map_err(|e| PyValueError::new_err(format!("failed to shape output: {e}")))?;
    Ok(shaped.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (r, v, dts, mus, max_iter=100, tol=1e-15))]
fn calc_chi_numpy<'py>(
    py: Python<'py>,
    r: PyReadonlyArray2<'py, f64>,
    v: PyReadonlyArray2<'py, f64>,
    dts: PyReadonlyArray1<'py, f64>,
    mus: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let r_arr = r.as_array();
    let v_arr = v.as_array();
    let dts_arr = dts.as_array();
    let mus_arr = mus.as_array();
    if r_arr.ncols() != 3 || v_arr.ncols() != 3 {
        return Err(PyValueError::new_err("r and v must each have shape (N, 3)"));
    }
    let n = r_arr.nrows();
    if v_arr.nrows() != n || dts_arr.len() != n || mus_arr.len() != n {
        return Err(PyValueError::new_err(
            "v, dts, and mus must have length/rows N matching r",
        ));
    }
    let r_slice = r_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("r must be contiguous"))?;
    let v_slice = v_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("v must be contiguous"))?;
    let dts_slice = dts_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("dts must be contiguous"))?;
    let mus_slice = mus_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("mus must be contiguous"))?;

    let mut out = Vec::with_capacity(n * 7);
    for i in 0..n {
        let base = i * 3;
        let (chi, stumpff) = calc_chi::<f64>(
            [r_slice[base], r_slice[base + 1], r_slice[base + 2]],
            [v_slice[base], v_slice[base + 1], v_slice[base + 2]],
            dts_slice[i],
            mus_slice[i],
            max_iter,
            tol,
        );
        out.push(chi);
        out.extend_from_slice(&stumpff);
    }
    let shaped = ndarray::Array2::from_shape_vec((n, 7), out)
        .map_err(|e| PyValueError::new_err(format!("failed to shape output: {e}")))?;
    Ok(shaped.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (r, v, dts, mus, max_iter=100, tol=1e-15))]
fn calc_lagrange_coefficients_numpy<'py>(
    py: Python<'py>,
    r: PyReadonlyArray2<'py, f64>,
    v: PyReadonlyArray2<'py, f64>,
    dts: PyReadonlyArray1<'py, f64>,
    mus: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let r_arr = r.as_array();
    let v_arr = v.as_array();
    let dts_arr = dts.as_array();
    let mus_arr = mus.as_array();
    if r_arr.ncols() != 3 || v_arr.ncols() != 3 {
        return Err(PyValueError::new_err("r and v must each have shape (N, 3)"));
    }
    let n = r_arr.nrows();
    if v_arr.nrows() != n || dts_arr.len() != n || mus_arr.len() != n {
        return Err(PyValueError::new_err(
            "v, dts, and mus must have length/rows N matching r",
        ));
    }
    let r_slice = r_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("r must be contiguous"))?;
    let v_slice = v_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("v must be contiguous"))?;
    let dts_slice = dts_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("dts must be contiguous"))?;
    let mus_slice = mus_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("mus must be contiguous"))?;

    let mut coeffs_out = Vec::with_capacity(n * 4);
    let mut stumpff_out = Vec::with_capacity(n * 6);
    let mut chi_out = Vec::with_capacity(n);
    for i in 0..n {
        let base = i * 3;
        let (coeffs, stumpff, chi) = calc_lagrange_coefficients::<f64>(
            [r_slice[base], r_slice[base + 1], r_slice[base + 2]],
            [v_slice[base], v_slice[base + 1], v_slice[base + 2]],
            dts_slice[i],
            mus_slice[i],
            max_iter,
            tol,
        );
        coeffs_out.extend_from_slice(&coeffs);
        stumpff_out.extend_from_slice(&stumpff);
        chi_out.push(chi);
    }

    let coeffs = ndarray::Array2::from_shape_vec((n, 4), coeffs_out)
        .map_err(|e| PyValueError::new_err(format!("failed to shape coeffs output: {e}")))?;
    let stumpff = ndarray::Array2::from_shape_vec((n, 6), stumpff_out)
        .map_err(|e| PyValueError::new_err(format!("failed to shape stumpff output: {e}")))?;
    Ok((
        coeffs.into_pyarray(py),
        stumpff.into_pyarray(py),
        ndarray::Array1::from_vec(chi_out).into_pyarray(py),
    ))
}

#[pyfunction]
fn apply_lagrange_coefficients_numpy<'py>(
    py: Python<'py>,
    r: PyReadonlyArray2<'py, f64>,
    v: PyReadonlyArray2<'py, f64>,
    coeffs: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let r_arr = r.as_array();
    let v_arr = v.as_array();
    let coeffs_arr = coeffs.as_array();
    if r_arr.ncols() != 3 || v_arr.ncols() != 3 {
        return Err(PyValueError::new_err("r and v must each have shape (N, 3)"));
    }
    let n = r_arr.nrows();
    if v_arr.nrows() != n || coeffs_arr.nrows() != n || coeffs_arr.ncols() != 4 {
        return Err(PyValueError::new_err(
            "v must have rows N and coeffs must have shape (N, 4)",
        ));
    }
    let r_slice = r_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("r must be contiguous"))?;
    let v_slice = v_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("v must be contiguous"))?;
    let coeffs_slice = coeffs_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("coeffs must be contiguous"))?;

    let mut r_out = Vec::with_capacity(n * 3);
    let mut v_out = Vec::with_capacity(n * 3);
    for i in 0..n {
        let state_base = i * 3;
        let coeff_base = i * 4;
        let (r_new, v_new) = apply_lagrange_coefficients::<f64>(
            [
                r_slice[state_base],
                r_slice[state_base + 1],
                r_slice[state_base + 2],
            ],
            [
                v_slice[state_base],
                v_slice[state_base + 1],
                v_slice[state_base + 2],
            ],
            [
                coeffs_slice[coeff_base],
                coeffs_slice[coeff_base + 1],
                coeffs_slice[coeff_base + 2],
                coeffs_slice[coeff_base + 3],
            ],
        );
        r_out.extend_from_slice(&r_new);
        v_out.extend_from_slice(&v_new);
    }

    let r_shaped = ndarray::Array2::from_shape_vec((n, 3), r_out)
        .map_err(|e| PyValueError::new_err(format!("failed to shape r output: {e}")))?;
    let v_shaped = ndarray::Array2::from_shape_vec((n, 3), v_out)
        .map_err(|e| PyValueError::new_err(format!("failed to shape v output: {e}")))?;
    Ok((r_shaped.into_pyarray(py), v_shaped.into_pyarray(py)))
}

#[pyfunction]
fn add_stellar_aberration_numpy<'py>(
    py: Python<'py>,
    orbits: PyReadonlyArray2<'py, f64>,
    observer_states: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let orbits_arr = orbits.as_array();
    let observers_arr = observer_states.as_array();
    if orbits_arr.ncols() != 6 || observers_arr.ncols() != 6 {
        return Err(PyValueError::new_err(
            "orbits and observer_states must each have shape (N, 6)",
        ));
    }
    let n = orbits_arr.nrows();
    if observers_arr.nrows() != n {
        return Err(PyValueError::new_err(
            "observer_states must have the same row count as orbits",
        ));
    }
    let orbits_slice = orbits_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("orbits must be contiguous"))?;
    let observers_slice = observers_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("observer_states must be contiguous"))?;

    let mut out = Vec::with_capacity(n * 3);
    for i in 0..n {
        let base = i * 6;
        let orbit = [
            orbits_slice[base],
            orbits_slice[base + 1],
            orbits_slice[base + 2],
            orbits_slice[base + 3],
            orbits_slice[base + 4],
            orbits_slice[base + 5],
        ];
        let observer = [
            observers_slice[base],
            observers_slice[base + 1],
            observers_slice[base + 2],
            observers_slice[base + 3],
            observers_slice[base + 4],
            observers_slice[base + 5],
        ];
        let topo = [
            orbit[0] - observer[0],
            orbit[1] - observer[1],
            orbit[2] - observer[2],
            orbit[3] - observer[3],
            orbit[4] - observer[4],
            orbit[5] - observer[5],
        ];
        let aberrated = apply_stellar_aberration_row(topo, observer);
        out.extend_from_slice(&aberrated[..3]);
    }

    let shaped = ndarray::Array2::from_shape_vec((n, 3), out)
        .map_err(|e| PyValueError::new_err(format!("failed to shape output: {e}")))?;
    Ok(shaped.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (orbits, dts, mus, max_iter=100, tol=1e-15))]
fn propagate_2body_numpy<'py>(
    py: Python<'py>,
    orbits: PyReadonlyArray2<'py, f64>,
    dts: PyReadonlyArray1<'py, f64>,
    mus: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let orbits_arr = orbits.as_array();
    let dts_arr = dts.as_array();
    let mus_arr = mus.as_array();
    if orbits_arr.ncols() != 6 {
        return Err(PyValueError::new_err("orbits must have shape (N, 6)"));
    }
    let n = orbits_arr.nrows();
    if dts_arr.len() != n || mus_arr.len() != n {
        return Err(PyValueError::new_err(
            "dts and mus must each have length N for orbits shape (N, 6)",
        ));
    }
    let orbits_flat = orbits_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("orbits must be contiguous"))?;
    let dts_slice = dts_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("dts must be contiguous"))?;
    let mus_slice = mus_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("mus must be contiguous"))?;

    let out_flat = propagate_2body_flat6(orbits_flat, dts_slice, mus_slice, max_iter, tol);
    let shaped = ndarray::Array2::from_shape_vec((n, 6), out_flat)
        .map_err(|e| PyValueError::new_err(format!("failed to shape output: {e}")))?;
    Ok(shaped.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (orbit, dts, mu, max_iter=100, tol=1e-15))]
fn propagate_2body_along_arc_numpy<'py>(
    py: Python<'py>,
    orbit: PyReadonlyArray1<'py, f64>,
    dts: PyReadonlyArray1<'py, f64>,
    mu: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let orbit_arr = orbit.as_array();
    let dts_arr = dts.as_array();
    if orbit_arr.len() != 6 {
        return Err(PyValueError::new_err("orbit must have shape (6,)"));
    }
    let dts_slice = dts_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("dts must be contiguous"))?;
    let orbit_arr_slice = orbit_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("orbit must be contiguous"))?;
    let orbit_state: [f64; 6] = [
        orbit_arr_slice[0],
        orbit_arr_slice[1],
        orbit_arr_slice[2],
        orbit_arr_slice[3],
        orbit_arr_slice[4],
        orbit_arr_slice[5],
    ];
    let rows = propagate_2body_along_arc(orbit_state, dts_slice, mu, max_iter, tol);
    let n = rows.len();
    let mut flat = Vec::with_capacity(n * 6);
    for row in &rows {
        flat.extend_from_slice(row);
    }
    let shaped = ndarray::Array2::from_shape_vec((n, 6), flat)
        .map_err(|e| PyValueError::new_err(format!("failed to shape output: {e}")))?;
    Ok(shaped.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (orbits, dts, mus, max_iter=100, tol=1e-15))]
fn propagate_2body_arc_batch_numpy<'py>(
    py: Python<'py>,
    orbits: PyReadonlyArray2<'py, f64>,
    dts: PyReadonlyArray2<'py, f64>,
    mus: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let orbits_arr = orbits.as_array();
    let dts_arr = dts.as_array();
    let mus_arr = mus.as_array();
    if orbits_arr.ncols() != 6 {
        return Err(PyValueError::new_err("orbits must have shape (N, 6)"));
    }
    let n = orbits_arr.nrows();
    if dts_arr.nrows() != n {
        return Err(PyValueError::new_err(
            "dts must have shape (N, K) matching orbits rows",
        ));
    }
    if mus_arr.len() != n {
        return Err(PyValueError::new_err(
            "mus must have length N matching orbits rows",
        ));
    }
    let k = dts_arr.ncols();
    let orbits_flat = orbits_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("orbits must be contiguous"))?;
    let dts_flat = dts_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("dts must be contiguous"))?;
    let mus_slice = mus_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("mus must be contiguous"))?;

    let out_flat =
        propagate_2body_arc_batch_flat6(orbits_flat, dts_flat, mus_slice, k, max_iter, tol);
    let shaped = ndarray::Array2::from_shape_vec((n * k, 6), out_flat)
        .map_err(|e| PyValueError::new_err(format!("failed to shape output: {e}")))?;
    Ok(shaped.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (orbits, covariances, dts, mus, max_iter=100, tol=1e-15))]
fn propagate_2body_with_covariance_numpy<'py>(
    py: Python<'py>,
    orbits: PyReadonlyArray2<'py, f64>,
    covariances: PyReadonlyArray2<'py, f64>,
    dts: PyReadonlyArray1<'py, f64>,
    mus: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let orbits_arr = orbits.as_array();
    let cov_arr = covariances.as_array();
    let dts_arr = dts.as_array();
    let mus_arr = mus.as_array();
    if orbits_arr.ncols() != 6 {
        return Err(PyValueError::new_err("orbits must have shape (N, 6)"));
    }
    let n = orbits_arr.nrows();
    if cov_arr.nrows() != n || cov_arr.ncols() != 36 {
        return Err(PyValueError::new_err(
            "covariances must have shape (N, 36) with row-major 6x6 per row",
        ));
    }
    if dts_arr.len() != n || mus_arr.len() != n {
        return Err(PyValueError::new_err(
            "dts and mus must each have length N for orbits shape (N, 6)",
        ));
    }
    let orbits_flat = orbits_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("orbits must be contiguous"))?;
    let cov_flat = cov_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("covariances must be contiguous"))?;
    let dts_slice = dts_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("dts must be contiguous"))?;
    let mus_slice = mus_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("mus must be contiguous"))?;

    let (states_out, cov_out) = propagate_2body_with_covariance_flat6(
        orbits_flat,
        cov_flat,
        dts_slice,
        mus_slice,
        max_iter,
        tol,
    );
    let states_shaped = ndarray::Array2::from_shape_vec((n, 6), states_out)
        .map_err(|e| PyValueError::new_err(format!("failed to shape states output: {e}")))?;
    let cov_shaped = ndarray::Array2::from_shape_vec((n, 36), cov_out)
        .map_err(|e| PyValueError::new_err(format!("failed to shape cov output: {e}")))?;
    Ok((states_shaped.into_pyarray(py), cov_shaped.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (
    orbits, observer_states, mus,
    lt_tol=1e-10, max_iter=1000, tol=1e-15,
    stellar_aberration=false, max_lt_iter=10,
))]
fn generate_ephemeris_2body_numpy<'py>(
    py: Python<'py>,
    orbits: PyReadonlyArray2<'py, f64>,
    observer_states: PyReadonlyArray2<'py, f64>,
    mus: PyReadonlyArray1<'py, f64>,
    lt_tol: f64,
    max_iter: usize,
    tol: f64,
    stellar_aberration: bool,
    max_lt_iter: usize,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let orbits_arr = orbits.as_array();
    let observer_arr = observer_states.as_array();
    let mus_arr = mus.as_array();
    if orbits_arr.ncols() != 6 {
        return Err(PyValueError::new_err("orbits must have shape (N, 6)"));
    }
    let n = orbits_arr.nrows();
    if observer_arr.nrows() != n || observer_arr.ncols() != 6 {
        return Err(PyValueError::new_err(
            "observer_states must have shape (N, 6) matching orbits rows",
        ));
    }
    if mus_arr.len() != n {
        return Err(PyValueError::new_err(
            "mus must have length N for orbits shape (N, 6)",
        ));
    }
    let orbits_flat = orbits_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("orbits must be contiguous"))?;
    let observer_flat = observer_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("observer_states must be contiguous"))?;
    let mus_slice = mus_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("mus must be contiguous"))?;

    let (sph_out, lt_out, aberrated_out) = generate_ephemeris_2body_flat6(
        orbits_flat,
        observer_flat,
        mus_slice,
        lt_tol,
        max_iter,
        tol,
        stellar_aberration,
        max_lt_iter,
    );
    let sph_shaped = ndarray::Array2::from_shape_vec((n, 6), sph_out)
        .map_err(|e| PyValueError::new_err(format!("failed to shape spherical output: {e}")))?;
    let lt_shaped = ndarray::Array1::from_vec(lt_out);
    let aberrated_shaped = ndarray::Array2::from_shape_vec((n, 6), aberrated_out)
        .map_err(|e| PyValueError::new_err(format!("failed to shape aberrated output: {e}")))?;
    Ok((
        sph_shaped.into_pyarray(py),
        lt_shaped.into_pyarray(py),
        aberrated_shaped.into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (
    orbits, covariances, observer_states, mus,
    lt_tol=1e-10, max_iter=1000, tol=1e-15,
    stellar_aberration=false, max_lt_iter=10,
))]
fn generate_ephemeris_2body_with_covariance_numpy<'py>(
    py: Python<'py>,
    orbits: PyReadonlyArray2<'py, f64>,
    covariances: PyReadonlyArray2<'py, f64>,
    observer_states: PyReadonlyArray2<'py, f64>,
    mus: PyReadonlyArray1<'py, f64>,
    lt_tol: f64,
    max_iter: usize,
    tol: f64,
    stellar_aberration: bool,
    max_lt_iter: usize,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let orbits_arr = orbits.as_array();
    let cov_arr = covariances.as_array();
    let observer_arr = observer_states.as_array();
    let mus_arr = mus.as_array();
    if orbits_arr.ncols() != 6 {
        return Err(PyValueError::new_err("orbits must have shape (N, 6)"));
    }
    let n = orbits_arr.nrows();
    if cov_arr.nrows() != n || cov_arr.ncols() != 36 {
        return Err(PyValueError::new_err(
            "covariances must have shape (N, 36) with row-major 6x6 per row",
        ));
    }
    if observer_arr.nrows() != n || observer_arr.ncols() != 6 {
        return Err(PyValueError::new_err(
            "observer_states must have shape (N, 6) matching orbits rows",
        ));
    }
    if mus_arr.len() != n {
        return Err(PyValueError::new_err(
            "mus must have length N for orbits shape (N, 6)",
        ));
    }
    let orbits_flat = orbits_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("orbits must be contiguous"))?;
    let cov_flat = cov_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("covariances must be contiguous"))?;
    let observer_flat = observer_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("observer_states must be contiguous"))?;
    let mus_slice = mus_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("mus must be contiguous"))?;

    let (sph_out, lt_out, aberrated_out, cov_out) = generate_ephemeris_2body_with_covariance_flat6(
        orbits_flat,
        cov_flat,
        observer_flat,
        mus_slice,
        lt_tol,
        max_iter,
        tol,
        stellar_aberration,
        max_lt_iter,
    );
    let sph_shaped = ndarray::Array2::from_shape_vec((n, 6), sph_out)
        .map_err(|e| PyValueError::new_err(format!("failed to shape spherical output: {e}")))?;
    let lt_shaped = ndarray::Array1::from_vec(lt_out);
    let aberrated_shaped = ndarray::Array2::from_shape_vec((n, 6), aberrated_out)
        .map_err(|e| PyValueError::new_err(format!("failed to shape aberrated output: {e}")))?;
    let cov_shaped = ndarray::Array2::from_shape_vec((n, 36), cov_out)
        .map_err(|e| PyValueError::new_err(format!("failed to shape cov output: {e}")))?;
    Ok((
        sph_shaped.into_pyarray(py),
        lt_shaped.into_pyarray(py),
        aberrated_shaped.into_pyarray(py),
        cov_shaped.into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (
    orbits, observer_positions, mus,
    lt_tol=1e-10, max_iter=1000, tol=1e-15, max_lt_iter=10,
))]
fn add_light_time_numpy<'py>(
    py: Python<'py>,
    orbits: PyReadonlyArray2<'py, f64>,
    observer_positions: PyReadonlyArray2<'py, f64>,
    mus: PyReadonlyArray1<'py, f64>,
    lt_tol: f64,
    max_iter: usize,
    tol: f64,
    max_lt_iter: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    let orbits_arr = orbits.as_array();
    let obs_arr = observer_positions.as_array();
    let mus_arr = mus.as_array();
    if orbits_arr.ncols() != 6 {
        return Err(PyValueError::new_err("orbits must have shape (N, 6)"));
    }
    let n = orbits_arr.nrows();
    if obs_arr.nrows() != n || obs_arr.ncols() != 3 {
        return Err(PyValueError::new_err(
            "observer_positions must have shape (N, 3)",
        ));
    }
    if mus_arr.len() != n {
        return Err(PyValueError::new_err(
            "mus must have length N for orbits shape (N, 6)",
        ));
    }
    let orbits_slice = orbits_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("orbits must be contiguous"))?;
    let obs_slice = obs_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("observer_positions must be contiguous"))?;
    let mus_slice = mus_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("mus must be contiguous"))?;
    let (ab, lt) = add_light_time_batch_flat(
        orbits_slice,
        obs_slice,
        mus_slice,
        lt_tol,
        max_iter,
        tol,
        max_lt_iter,
    );
    let ab_shaped = ndarray::Array2::from_shape_vec((n, 6), ab)
        .map_err(|e| PyValueError::new_err(format!("failed to shape aberrated output: {e}")))?;
    Ok((
        ab_shaped.into_pyarray(py),
        ndarray::Array1::from_vec(lt).into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (primary_orbit, secondary_orbit, mu, max_iter=100, xtol=1e-10))]
fn calculate_moid_numpy(
    primary_orbit: PyReadonlyArray1<'_, f64>,
    secondary_orbit: PyReadonlyArray1<'_, f64>,
    mu: f64,
    max_iter: usize,
    xtol: f64,
) -> PyResult<(f64, f64)> {
    let p_arr = primary_orbit.as_array();
    let s_arr = secondary_orbit.as_array();
    if p_arr.len() != 6 || s_arr.len() != 6 {
        return Err(PyValueError::new_err(
            "primary_orbit and secondary_orbit must each have length 6",
        ));
    }
    let p = [p_arr[0], p_arr[1], p_arr[2], p_arr[3], p_arr[4], p_arr[5]];
    let s = [s_arr[0], s_arr[1], s_arr[2], s_arr[3], s_arr[4], s_arr[5]];
    Ok(calculate_moid(p, s, mu, max_iter, xtol))
}

#[pyfunction]
#[pyo3(signature = (primary_orbits, secondary_orbits, mus, max_iter=100, xtol=1e-10))]
fn calculate_moid_batch_numpy<'py>(
    py: Python<'py>,
    primary_orbits: PyReadonlyArray2<'py, f64>,
    secondary_orbits: PyReadonlyArray2<'py, f64>,
    mus: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
    xtol: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let p_arr = primary_orbits.as_array();
    let s_arr = secondary_orbits.as_array();
    let mu_arr = mus.as_array();
    if p_arr.ncols() != 6 || s_arr.ncols() != 6 {
        return Err(PyValueError::new_err(
            "primary_orbits and secondary_orbits must each have shape (N, 6)",
        ));
    }
    let n = p_arr.nrows();
    if s_arr.nrows() != n || mu_arr.len() != n {
        return Err(PyValueError::new_err(
            "primary_orbits, secondary_orbits, mus must all have N rows",
        ));
    }
    let p_slice = p_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("primary_orbits must be contiguous"))?;
    let s_slice = s_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("secondary_orbits must be contiguous"))?;
    let mu_slice = mu_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("mus must be contiguous"))?;
    let (moids, dts) = calculate_moid_batch(p_slice, s_slice, mu_slice, max_iter, xtol);
    Ok((
        ndarray::Array1::from_vec(moids).into_pyarray(py),
        ndarray::Array1::from_vec(dts).into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (
    r1, r2, tof, mu,
    m=0, prograde=true, low_path=true,
    maxiter=35, atol=1e-10, rtol=1e-10,
))]
fn izzo_lambert_numpy<'py>(
    py: Python<'py>,
    r1: PyReadonlyArray2<'py, f64>,
    r2: PyReadonlyArray2<'py, f64>,
    tof: PyReadonlyArray1<'py, f64>,
    mu: f64,
    m: u32,
    prograde: bool,
    low_path: bool,
    maxiter: u32,
    atol: f64,
    rtol: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let r1_arr = r1.as_array();
    let r2_arr = r2.as_array();
    let tof_arr = tof.as_array();
    if r1_arr.ncols() != 3 {
        return Err(PyValueError::new_err("r1 must have shape (N, 3)"));
    }
    let n = r1_arr.nrows();
    if r2_arr.nrows() != n || r2_arr.ncols() != 3 {
        return Err(PyValueError::new_err(
            "r2 must have shape (N, 3) matching r1",
        ));
    }
    if tof_arr.len() != n {
        return Err(PyValueError::new_err(
            "tof must have length N for r1 shape (N, 3)",
        ));
    }
    let r1_slice = r1_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("r1 must be contiguous"))?;
    let r2_slice = r2_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("r2 must be contiguous"))?;
    let tof_slice = tof_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("tof must be contiguous"))?;
    let (v1, v2) = izzo_lambert_batch_flat(
        r1_slice, r2_slice, tof_slice, mu, m, prograde, low_path, maxiter, atol, rtol,
    );
    let v1_shaped = ndarray::Array2::from_shape_vec((n, 3), v1)
        .map_err(|e| PyValueError::new_err(format!("failed to shape v1 output: {e}")))?;
    let v2_shaped = ndarray::Array2::from_shape_vec((n, 3), v2)
        .map_err(|e| PyValueError::new_err(format!("failed to shape v2 output: {e}")))?;
    Ok((v1_shaped.into_pyarray(py), v2_shaped.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (
    dep_states, dep_mjds, arr_states, arr_mjds, mu,
    prograde=true, maxiter=35, atol=1e-10, rtol=1e-10,
))]
fn porkchop_grid_numpy<'py>(
    py: Python<'py>,
    dep_states: PyReadonlyArray2<'py, f64>,
    dep_mjds: PyReadonlyArray1<'py, f64>,
    arr_states: PyReadonlyArray2<'py, f64>,
    arr_mjds: PyReadonlyArray1<'py, f64>,
    mu: f64,
    prograde: bool,
    maxiter: u32,
    atol: f64,
    rtol: f64,
) -> PyResult<(
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let dep_arr = dep_states.as_array();
    let arr_arr = arr_states.as_array();
    let dep_mjd_arr = dep_mjds.as_array();
    let arr_mjd_arr = arr_mjds.as_array();
    if dep_arr.ncols() != 6 || arr_arr.ncols() != 6 {
        return Err(PyValueError::new_err(
            "dep_states and arr_states must each have shape (N, 6)",
        ));
    }
    if dep_mjd_arr.len() != dep_arr.nrows() {
        return Err(PyValueError::new_err(
            "dep_mjds must have length N for dep_states shape (N, 6)",
        ));
    }
    if arr_mjd_arr.len() != arr_arr.nrows() {
        return Err(PyValueError::new_err(
            "arr_mjds must have length M for arr_states shape (M, 6)",
        ));
    }
    let dep_slice = dep_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("dep_states must be contiguous"))?;
    let arr_slice = arr_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("arr_states must be contiguous"))?;
    let dep_mjd_slice = dep_mjd_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("dep_mjds must be contiguous"))?;
    let arr_mjd_slice = arr_mjd_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("arr_mjds must be contiguous"))?;
    let (dep_idx, arr_idx, v1, v2) = porkchop_grid_flat(
        dep_slice,
        dep_mjd_slice,
        arr_slice,
        arr_mjd_slice,
        mu,
        prograde,
        maxiter,
        atol,
        rtol,
    );
    let v = dep_idx.len();
    let v1_shaped = ndarray::Array2::from_shape_vec((v, 3), v1)
        .map_err(|e| PyValueError::new_err(format!("failed to shape v1 output: {e}")))?;
    let v2_shaped = ndarray::Array2::from_shape_vec((v, 3), v2)
        .map_err(|e| PyValueError::new_err(format!("failed to shape v2 output: {e}")))?;
    Ok((
        ndarray::Array1::from_vec(dep_idx).into_pyarray(py),
        ndarray::Array1::from_vec(arr_idx).into_pyarray(py),
        v1_shaped.into_pyarray(py),
        v2_shaped.into_pyarray(py),
    ))
}

/// Rust-owned `Instant` timing for the public scalar dynamics helpers.
/// Inputs are packed column-wise into one (N, K) array per kernel so a single
/// dispatcher covers every scalar surface; extraction happens before warmup.
#[pyfunction]
#[pyo3(signature = (kernel, packed, reps, trials, warmup_reps=1))]
fn benchmark_scalar_dynamics_kernel_numpy(
    kernel: &str,
    packed: PyReadonlyArray2<'_, f64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let packed_arr = packed.as_array();
    let n = packed_arr.nrows();
    let cols = packed_arr.ncols();
    let data = packed_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("packed must be contiguous"))?
        .to_vec();
    let column =
        |index: usize| -> Vec<f64> { (0..n).map(|row| data[row * cols + index]).collect() };
    let require_cols = |expected: usize| -> PyResult<()> {
        if cols == expected {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "kernel {kernel} expects {expected} packed columns, got {cols}"
            )))
        }
    };
    let row3 = |base: usize, row: usize| -> [f64; 3] {
        [
            data[row * cols + base],
            data[row * cols + base + 1],
            data[row * cols + base + 2],
        ]
    };

    let samples = match kernel {
        "calc_period"
        | "calc_mean_motion"
        | "calc_periapsis_distance"
        | "calc_apoapsis_distance"
        | "calc_semi_latus_rectum"
        | "calc_semi_major_axis"
        | "calc_mean_anomaly"
        | "solve_kepler" => {
            require_cols(2)?;
            let first = column(0);
            let second = column(1);
            let op: fn(f64, f64) -> f64 = match kernel {
                "calc_period" => calc_period,
                "calc_mean_motion" => |a, mu| calc_mean_motion_batch(&[a], &[mu])[0],
                "calc_periapsis_distance" => calc_periapsis_distance,
                "calc_apoapsis_distance" => calc_apoapsis_distance,
                "calc_semi_latus_rectum" => calc_semi_latus_rectum,
                "calc_semi_major_axis" => calc_semi_major_axis,
                "calc_mean_anomaly" => calc_mean_anomaly,
                _ => |e, m| solve_kepler_true_anomaly(e, m, 100, 1e-15),
            };
            benchmark_samples(reps, trials, warmup_reps, || -> Vec<f64> {
                first
                    .iter()
                    .zip(second.iter())
                    .map(|(&a, &b)| op(a, b))
                    .collect()
            })
        }
        "solve_barker" => {
            require_cols(1)?;
            let m = column(0);
            benchmark_samples(reps, trials, warmup_reps, || -> Vec<f64> {
                m.iter().map(|&mi| solve_barker(mi)).collect()
            })
        }
        "calc_stumpff" => {
            require_cols(1)?;
            let psi = column(0);
            benchmark_samples(reps, trials, warmup_reps, || -> Vec<f64> {
                let mut out = Vec::with_capacity(psi.len() * 6);
                for &psi_i in &psi {
                    out.extend_from_slice(&calc_stumpff::<f64>(psi_i));
                }
                out
            })
        }
        "calc_chi" => {
            require_cols(8)?;
            benchmark_samples(reps, trials, warmup_reps, || -> Vec<f64> {
                let mut out = Vec::with_capacity(n * 7);
                for row in 0..n {
                    let (chi, stumpff) = calc_chi::<f64>(
                        row3(0, row),
                        row3(3, row),
                        data[row * cols + 6],
                        data[row * cols + 7],
                        100,
                        1e-15,
                    );
                    out.push(chi);
                    out.extend_from_slice(&stumpff);
                }
                out
            })
        }
        "calc_lagrange_coefficients" => {
            require_cols(8)?;
            benchmark_samples(reps, trials, warmup_reps, || -> Vec<f64> {
                let mut out = Vec::with_capacity(n * 11);
                for row in 0..n {
                    let (coeffs, stumpff, chi) = calc_lagrange_coefficients::<f64>(
                        row3(0, row),
                        row3(3, row),
                        data[row * cols + 6],
                        data[row * cols + 7],
                        100,
                        1e-15,
                    );
                    out.extend_from_slice(&coeffs);
                    out.extend_from_slice(&stumpff);
                    out.push(chi);
                }
                out
            })
        }
        "apply_lagrange_coefficients" => {
            require_cols(10)?;
            benchmark_samples(reps, trials, warmup_reps, || -> Vec<f64> {
                let mut out = Vec::with_capacity(n * 6);
                for row in 0..n {
                    let base = row * cols;
                    let (r_new, v_new) = apply_lagrange_coefficients::<f64>(
                        row3(0, row),
                        row3(3, row),
                        [
                            data[base + 6],
                            data[base + 7],
                            data[base + 8],
                            data[base + 9],
                        ],
                    );
                    out.extend_from_slice(&r_new);
                    out.extend_from_slice(&v_new);
                }
                out
            })
        }
        "add_stellar_aberration" => {
            require_cols(12)?;
            benchmark_samples(reps, trials, warmup_reps, || -> Vec<f64> {
                let mut out = Vec::with_capacity(n * 3);
                for row in 0..n {
                    let base = row * cols;
                    let orbit: [f64; 6] = data[base..base + 6].try_into().expect("six columns");
                    let observer: [f64; 6] =
                        data[base + 6..base + 12].try_into().expect("six columns");
                    let topo = [
                        orbit[0] - observer[0],
                        orbit[1] - observer[1],
                        orbit[2] - observer[2],
                        orbit[3] - observer[3],
                        orbit[4] - observer[4],
                        orbit[5] - observer[5],
                    ];
                    let aberrated = apply_stellar_aberration_row(topo, observer);
                    out.extend_from_slice(&aberrated[..3]);
                }
                out
            })
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown scalar dynamics kernel: {kernel}"
            )))
        }
    };
    Ok(samples)
}

/// Rust-owned timers for canonical parity dynamics/mission kernels. Every
/// NumPy/PyO3 extraction and owned-input preparation happens before the first
/// warmup; recorded samples contain only direct Rust calls and `Instant`.
#[pyfunction]
#[pyo3(signature = (a, mu, reps, trials, warmup_reps=1))]
fn benchmark_calc_mean_motion_numpy(
    a: PyReadonlyArray1<'_, f64>,
    mu: PyReadonlyArray1<'_, f64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let a = a.as_slice()?.to_vec();
    let mu = mu.as_slice()?.to_vec();
    if a.len() != mu.len() {
        return Err(PyValueError::new_err("a and mu must have the same length"));
    }
    Ok(benchmark_samples(reps, trials, warmup_reps, || {
        calc_mean_motion_batch(&a, &mu)
    }))
}

#[pyfunction]
#[pyo3(signature = (a, e, i_deg, ap, reps, trials, warmup_reps=1))]
fn benchmark_tisserand_parameter_numpy(
    a: PyReadonlyArray1<'_, f64>,
    e: PyReadonlyArray1<'_, f64>,
    i_deg: PyReadonlyArray1<'_, f64>,
    ap: f64,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let a = a.as_slice()?.to_vec();
    let e = e.as_slice()?.to_vec();
    let i_deg = i_deg.as_slice()?.to_vec();
    if a.len() != e.len() || a.len() != i_deg.len() {
        return Err(PyValueError::new_err("a, e, i_deg must have equal length"));
    }
    Ok(benchmark_samples(reps, trials, warmup_reps, || {
        tisserand_parameter_flat(&a, &e, &i_deg, ap)
    }))
}

#[pyfunction]
#[pyo3(signature = (primary_orbits, secondary_orbits, mus, reps, trials, warmup_reps=1, max_iter=100, xtol=1e-10))]
#[allow(clippy::too_many_arguments)]
fn benchmark_calculate_moid_numpy(
    primary_orbits: PyReadonlyArray2<'_, f64>,
    secondary_orbits: PyReadonlyArray2<'_, f64>,
    mus: PyReadonlyArray1<'_, f64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
    max_iter: usize,
    xtol: f64,
) -> PyResult<Vec<Vec<f64>>> {
    let p_shape = primary_orbits.shape();
    let s_shape = secondary_orbits.shape();
    if p_shape.len() != 2 || s_shape.len() != 2 || p_shape[1] != 6 || s_shape != p_shape {
        return Err(PyValueError::new_err(
            "orbit inputs must have matching shape (N, 6)",
        ));
    }
    let p = primary_orbits.as_slice()?.to_vec();
    let s = secondary_orbits.as_slice()?.to_vec();
    let mus = mus.as_slice()?.to_vec();
    if mus.len() != p_shape[0] {
        return Err(PyValueError::new_err("mus must have length N"));
    }
    Ok(benchmark_samples(reps, trials, warmup_reps, || {
        (0..mus.len())
            .map(|row| {
                let base = row * 6;
                let primary: [f64; 6] = p[base..base + 6].try_into().unwrap();
                let secondary: [f64; 6] = s[base..base + 6].try_into().unwrap();
                calculate_moid(primary, secondary, mus[row], max_iter, xtol)
            })
            .collect::<Vec<_>>()
    }))
}

#[pyfunction]
#[pyo3(signature = (primary_orbits, secondary_orbits, mus, reps, trials, warmup_reps=1, max_iter=100, xtol=1e-10))]
#[allow(clippy::too_many_arguments)]
fn benchmark_calculate_moid_batch_numpy(
    primary_orbits: PyReadonlyArray2<'_, f64>,
    secondary_orbits: PyReadonlyArray2<'_, f64>,
    mus: PyReadonlyArray1<'_, f64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
    max_iter: usize,
    xtol: f64,
) -> PyResult<Vec<Vec<f64>>> {
    let p_shape = primary_orbits.shape();
    let s_shape = secondary_orbits.shape();
    if p_shape.len() != 2 || s_shape.len() != 2 || p_shape[1] != 6 || s_shape != p_shape {
        return Err(PyValueError::new_err(
            "orbit inputs must have matching shape (N, 6)",
        ));
    }
    let p = primary_orbits.as_slice()?.to_vec();
    let s = secondary_orbits.as_slice()?.to_vec();
    let mus = mus.as_slice()?.to_vec();
    if mus.len() != p_shape[0] {
        return Err(PyValueError::new_err("mus must have length N"));
    }
    Ok(benchmark_samples(reps, trials, warmup_reps, || {
        calculate_moid_batch(&p, &s, &mus, max_iter, xtol)
    }))
}

#[pyfunction]
#[pyo3(signature = (orbit, dts, mu, reps, trials, warmup_reps=1, max_iter=100, tol=1e-15))]
#[allow(clippy::too_many_arguments)]
fn benchmark_propagate_2body_along_arc_numpy(
    orbit: PyReadonlyArray1<'_, f64>,
    dts: PyReadonlyArray1<'_, f64>,
    mu: f64,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<Vec<Vec<f64>>> {
    let orbit: [f64; 6] = orbit
        .as_slice()?
        .try_into()
        .map_err(|_| PyValueError::new_err("orbit must have shape (6,)"))?;
    let dts = dts.as_slice()?.to_vec();
    Ok(benchmark_samples(reps, trials, warmup_reps, || {
        propagate_2body_along_arc(orbit, &dts, mu, max_iter, tol)
    }))
}

#[pyfunction]
#[pyo3(signature = (orbits, dts, mus, reps, trials, warmup_reps=1, max_iter=100, tol=1e-15))]
#[allow(clippy::too_many_arguments)]
fn benchmark_propagate_2body_arc_batch_numpy(
    orbits: PyReadonlyArray2<'_, f64>,
    dts: PyReadonlyArray2<'_, f64>,
    mus: PyReadonlyArray1<'_, f64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<Vec<Vec<f64>>> {
    let orbit_shape = orbits.shape();
    let dt_shape = dts.shape();
    if orbit_shape[1] != 6 || dt_shape[0] != orbit_shape[0] {
        return Err(PyValueError::new_err("orbits/dts row counts must match"));
    }
    let orbits = orbits.as_slice()?.to_vec();
    let dts = dts.as_slice()?.to_vec();
    let mus = mus.as_slice()?.to_vec();
    if mus.len() != orbit_shape[0] {
        return Err(PyValueError::new_err("mus must have length N"));
    }
    let epochs = dt_shape[1];
    Ok(benchmark_samples(reps, trials, warmup_reps, || {
        propagate_2body_arc_batch_flat6(&orbits, &dts, &mus, epochs, max_iter, tol)
    }))
}

#[pyfunction]
#[pyo3(signature = (orbits, covariances, dts, mus, reps, trials, warmup_reps=1, max_iter=100, tol=1e-15))]
#[allow(clippy::too_many_arguments)]
fn benchmark_propagate_2body_with_covariance_numpy(
    orbits: PyReadonlyArray2<'_, f64>,
    covariances: PyReadonlyArray2<'_, f64>,
    dts: PyReadonlyArray1<'_, f64>,
    mus: PyReadonlyArray1<'_, f64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<Vec<Vec<f64>>> {
    let shape = orbits.shape();
    if shape[1] != 6 || covariances.shape() != [shape[0], 36] {
        return Err(PyValueError::new_err(
            "orbits/covariances must have shapes (N,6)/(N,36)",
        ));
    }
    let orbits = orbits.as_slice()?.to_vec();
    let covariances = covariances.as_slice()?.to_vec();
    let dts = dts.as_slice()?.to_vec();
    let mus = mus.as_slice()?.to_vec();
    if dts.len() != shape[0] || mus.len() != shape[0] {
        return Err(PyValueError::new_err("dts and mus must have length N"));
    }
    Ok(benchmark_samples(reps, trials, warmup_reps, || {
        propagate_2body_with_covariance_flat6(&orbits, &covariances, &dts, &mus, max_iter, tol)
    }))
}

#[pyfunction]
#[pyo3(signature = (r1, r2, tof, mu, m, prograde, low_path, reps, trials, warmup_reps=1, maxiter=35, atol=1e-10, rtol=1e-10))]
#[allow(clippy::too_many_arguments)]
fn benchmark_izzo_lambert_numpy(
    r1: PyReadonlyArray2<'_, f64>,
    r2: PyReadonlyArray2<'_, f64>,
    tof: PyReadonlyArray1<'_, f64>,
    mu: f64,
    m: u32,
    prograde: bool,
    low_path: bool,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
    maxiter: u32,
    atol: f64,
    rtol: f64,
) -> PyResult<Vec<Vec<f64>>> {
    let shape = r1.shape();
    if shape[1] != 3 || r2.shape() != shape || tof.len() != shape[0] {
        return Err(PyValueError::new_err(
            "r1/r2/tof shapes must be (N,3)/(N,3)/(N,)",
        ));
    }
    let r1 = r1.as_slice()?.to_vec();
    let r2 = r2.as_slice()?.to_vec();
    let tof = tof.as_slice()?.to_vec();
    Ok(benchmark_samples(reps, trials, warmup_reps, || {
        izzo_lambert_batch_flat(
            &r1, &r2, &tof, mu, m, prograde, low_path, maxiter, atol, rtol,
        )
    }))
}

#[pyfunction]
#[pyo3(signature = (orbits, observer_positions, mus, reps, trials, warmup_reps=1, lt_tol=1e-10, max_iter=1000, tol=1e-15, max_lt_iter=10))]
#[allow(clippy::too_many_arguments)]
fn benchmark_add_light_time_numpy(
    orbits: PyReadonlyArray2<'_, f64>,
    observer_positions: PyReadonlyArray2<'_, f64>,
    mus: PyReadonlyArray1<'_, f64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
    lt_tol: f64,
    max_iter: usize,
    tol: f64,
    max_lt_iter: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let shape = orbits.shape();
    if shape[1] != 6 || observer_positions.shape() != [shape[0], 3] || mus.len() != shape[0] {
        return Err(PyValueError::new_err(
            "orbits/observers/mus shapes must be (N,6)/(N,3)/(N,)",
        ));
    }
    let orbits = orbits.as_slice()?.to_vec();
    let observers = observer_positions.as_slice()?.to_vec();
    let mus = mus.as_slice()?.to_vec();
    Ok(benchmark_samples(reps, trials, warmup_reps, || {
        add_light_time_batch_flat(
            &orbits,
            &observers,
            &mus,
            lt_tol,
            max_iter,
            tol,
            max_lt_iter,
        )
    }))
}

#[pyfunction]
#[pyo3(signature = (dep_states, dep_mjds, arr_states, arr_mjds, mu, prograde, reps, trials, warmup_reps=1, maxiter=35, atol=1e-10, rtol=1e-10))]
#[allow(clippy::too_many_arguments)]
fn benchmark_porkchop_grid_numpy(
    dep_states: PyReadonlyArray2<'_, f64>,
    dep_mjds: PyReadonlyArray1<'_, f64>,
    arr_states: PyReadonlyArray2<'_, f64>,
    arr_mjds: PyReadonlyArray1<'_, f64>,
    mu: f64,
    prograde: bool,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
    maxiter: u32,
    atol: f64,
    rtol: f64,
) -> PyResult<Vec<Vec<f64>>> {
    if dep_states.shape()[1] != 6
        || arr_states.shape()[1] != 6
        || dep_mjds.len() != dep_states.shape()[0]
        || arr_mjds.len() != arr_states.shape()[0]
    {
        return Err(PyValueError::new_err(
            "porkchop state/time shapes do not match",
        ));
    }
    let dep_states = dep_states.as_slice()?.to_vec();
    let dep_mjds = dep_mjds.as_slice()?.to_vec();
    let arr_states = arr_states.as_slice()?.to_vec();
    let arr_mjds = arr_mjds.as_slice()?.to_vec();
    Ok(benchmark_samples(reps, trials, warmup_reps, || {
        porkchop_grid_flat(
            &dep_states,
            &dep_mjds,
            &arr_states,
            &arr_mjds,
            mu,
            prograde,
            maxiter,
            atol,
            rtol,
        )
    }))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calc_mean_motion_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calc_period_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calc_periapsis_distance_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calc_apoapsis_distance_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calc_semi_major_axis_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calc_semi_latus_rectum_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calc_mean_anomaly_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(solve_barker_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(solve_kepler_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calc_stumpff_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calc_chi_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calc_lagrange_coefficients_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(apply_lagrange_coefficients_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(add_stellar_aberration_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_2body_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_2body_along_arc_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_2body_arc_batch_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_2body_with_covariance_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(generate_ephemeris_2body_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(
        generate_ephemeris_2body_with_covariance_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(add_light_time_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_moid_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_moid_batch_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(izzo_lambert_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(porkchop_grid_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_scalar_dynamics_kernel_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_calc_mean_motion_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_tisserand_parameter_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_calculate_moid_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_calculate_moid_batch_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(
        benchmark_propagate_2body_along_arc_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        benchmark_propagate_2body_arc_batch_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        benchmark_propagate_2body_with_covariance_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(benchmark_izzo_lambert_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_add_light_time_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_porkchop_grid_numpy, m)?)?;
    Ok(())
}
