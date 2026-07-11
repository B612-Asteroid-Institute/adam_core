use adam_core_rs_coords::{
    CoordinateBatch, DataFrame, Epoch, IntoNestedRecordBatch, OrbitBatch, OrbitId, OriginArray,
    OriginId, TimeArray, TimeScale,
};
use adam_core_rs_orbit_determination::{
    calc_gauss_row, calc_gibbs_row, calc_herrick_gibbs_row, gauss_iod_fused,
    gauss_iod_orbits_from_roots,
};
use arrow::pyarrow::ToPyArrow;
use arrow_array::RecordBatch;
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

    Ok(ndarray::Array1::from_vec(out.to_vec()).into_pyarray(py))
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

    Ok(ndarray::Array1::from_vec(out.to_vec()).into_pyarray(py))
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

    Ok(ndarray::Array1::from_vec(out.to_vec()).into_pyarray(py))
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
    let epochs_arr = ndarray::Array1::from_vec(epochs).into_pyarray(py);
    let orbits_arr = ndarray::Array2::from_shape_vec((n, 6), orbits_flat)
        .map_err(|e| PyValueError::new_err(format!("failed to shape output: {e}")))?
        .into_pyarray(py);

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
    let epochs_arr = ndarray::Array1::from_vec(epochs).into_pyarray(py);
    let orbits_arr = ndarray::Array2::from_shape_vec((n, 6), orbits_flat)
        .map_err(|e| PyValueError::new_err(format!("failed to shape output: {e}")))?
        .into_pyarray(py);

    Ok((epochs_arr, orbits_arr))
}

/// Random 32-hex orbit ids matching quivr's uuid4().hex default semantics
/// (unique random hex strings; bit-identity with Python uuid4 is not a
/// contract, uniqueness is).
fn random_hex_orbit_ids(count: usize) -> Vec<String> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let mut state = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos() as u64)
        .unwrap_or(0x9e37_79b9_7f4a_7c15)
        ^ (count as u64).wrapping_mul(0xa076_1d64_78bd_642f);
    let mut next = move || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    (0..count)
        .map(|_| format!("{:016x}{:016x}", next(), next()))
        .collect()
}

fn mjd_to_epoch(mjd: f64) -> Epoch {
    let days = mjd.floor();
    let nanos = ((mjd - days) * 86_400_000_000_000.0_f64).round() as i64;
    Epoch::new(days as i64, nanos)
}

/// Fused Gauss IOD returning the finished Orbits nested RecordBatch:
/// unit-vector rotation, polynomial roots, per-root orbit construction,
/// non-finite filtering, orbit-id generation, and table assembly in Rust.
#[allow(clippy::too_many_arguments)]
fn gauss_iod_orbits_record_batch(
    ra: [f64; 3],
    dec: [f64; 3],
    times: [f64; 3],
    obs: [[f64; 3]; 3],
    velocity_method_id: i32,
    light_time: bool,
    mu: f64,
    c: f64,
) -> Result<RecordBatch, String> {
    let (epochs, orbits_flat) =
        gauss_iod_fused(ra, dec, times, obs, velocity_method_id, light_time, mu, c);
    let mut states: Vec<[f64; 6]> = Vec::with_capacity(epochs.len());
    let mut kept_epochs: Vec<Epoch> = Vec::with_capacity(epochs.len());
    for (row, epoch) in epochs.iter().enumerate() {
        let state: [f64; 6] = std::array::from_fn(|axis| orbits_flat[row * 6 + axis]);
        if state.iter().all(|value| value.is_finite()) {
            states.push(state);
            kept_epochs.push(mjd_to_epoch(*epoch));
        }
    }
    let rows = states.len();
    let coordinates = CoordinateBatch::cartesian(
        states,
        DataFrame::Ecliptic,
        OriginArray::repeat(OriginId::Named("SUN".to_string()), rows),
        Some(
            TimeArray::new(TimeScale::Utc, kept_epochs)
                .map_err(|err| format!("invalid IOD epochs: {err}"))?,
        ),
        None,
    )
    .map_err(|err| format!("failed to build IOD coordinates: {err}"))?;
    let orbits = OrbitBatch::new(
        random_hex_orbit_ids(rows)
            .into_iter()
            .map(OrbitId)
            .collect(),
        vec![None; rows],
        coordinates,
    )
    .map_err(|err| format!("failed to build IOD OrbitBatch: {err}"))?;
    orbits
        .into_nested_record_batch()
        .map_err(|err| format!("failed to encode IOD Orbits: {err}"))
}

fn parse_velocity_method(velocity_method: &str) -> PyResult<i32> {
    match velocity_method {
        "gauss" => Ok(0),
        "gibbs" => Ok(1),
        "herrick+gibbs" => Ok(2),
        _ => Err(PyValueError::new_err(
            "velocity_method should be one of {'gauss', 'gibbs', 'herrick+gibbs'}",
        )),
    }
}

fn triplet_inputs(
    ra_deg: &PyReadonlyArray1<'_, f64>,
    dec_deg: &PyReadonlyArray1<'_, f64>,
    obs_times_mjd: &PyReadonlyArray1<'_, f64>,
    coords_obs: &PyReadonlyArray2<'_, f64>,
) -> PyResult<([f64; 3], [f64; 3], [f64; 3], [[f64; 3]; 3])> {
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
            "coords_obs must have shape (3, 3) - three heliocentric observer positions",
        ));
    }
    Ok((
        [ra_arr[0], ra_arr[1], ra_arr[2]],
        [dec_arr[0], dec_arr[1], dec_arr[2]],
        [t_arr[0], t_arr[1], t_arr[2]],
        [
            [obs_arr[[0, 0]], obs_arr[[0, 1]], obs_arr[[0, 2]]],
            [obs_arr[[1, 0]], obs_arr[[1, 1]], obs_arr[[1, 2]]],
            [obs_arr[[2, 0]], obs_arr[[2, 1]], obs_arr[[2, 2]]],
        ],
    ))
}

/// Arrow-native public Gauss-IOD surface: numpy triplet inputs in, finished
/// Orbits RecordBatch out.
#[pyfunction]
#[pyo3(signature = (
    ra_deg, dec_deg, obs_times_mjd, coords_obs,
    velocity_method, light_time, mu, c,
))]
#[allow(clippy::too_many_arguments)]
fn gauss_iod_orbits_arrow<'py>(
    py: Python<'py>,
    ra_deg: PyReadonlyArray1<'py, f64>,
    dec_deg: PyReadonlyArray1<'py, f64>,
    obs_times_mjd: PyReadonlyArray1<'py, f64>,
    coords_obs: PyReadonlyArray2<'py, f64>,
    velocity_method: &str,
    light_time: bool,
    mu: f64,
    c: f64,
) -> PyResult<PyObject> {
    let velocity_method_id = parse_velocity_method(velocity_method)?;
    let (ra, dec, times, obs) = triplet_inputs(&ra_deg, &dec_deg, &obs_times_mjd, &coords_obs)?;
    let output = py
        .allow_threads(|| {
            gauss_iod_orbits_record_batch(
                ra,
                dec,
                times,
                obs,
                velocity_method_id,
                light_time,
                mu,
                c,
            )
        })
        .map_err(PyValueError::new_err)?;
    output
        .to_pyarrow(py)
        .map_err(|err| PyValueError::new_err(format!("failed to export RecordBatch: {err}")))
}

/// Rust-owned Instant timer for the Arrow-native Gauss-IOD surface. Each
/// timed sample runs every supplied triplet through the full record-batch
/// core (matching the public per-triplet loop shape).
#[pyfunction]
#[pyo3(signature = (
    ra_deg_per_triplet, dec_deg_per_triplet, times_per_triplet, obs_pos_flat,
    reps, trials, warmup_reps=1, velocity_method="gibbs", light_time=true, mu=0.00029591220828559115, c=173.14463267424034,
))]
#[allow(clippy::too_many_arguments)]
fn benchmark_gauss_iod_orbits_arrow<'py>(
    ra_deg_per_triplet: PyReadonlyArray2<'py, f64>,
    dec_deg_per_triplet: PyReadonlyArray2<'py, f64>,
    times_per_triplet: PyReadonlyArray2<'py, f64>,
    obs_pos_flat: PyReadonlyArray2<'py, f64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
    velocity_method: &str,
    light_time: bool,
    mu: f64,
    c: f64,
) -> PyResult<Vec<Vec<f64>>> {
    if reps == 0 || trials == 0 {
        return Err(PyValueError::new_err("reps and trials must be >= 1"));
    }
    let velocity_method_id = parse_velocity_method(velocity_method)?;
    let ra = ra_deg_per_triplet.as_array();
    let dec = dec_deg_per_triplet.as_array();
    let times = times_per_triplet.as_array();
    let obs = obs_pos_flat.as_array();
    let triplets = ra.nrows();
    if dec.nrows() != triplets || times.nrows() != triplets || obs.nrows() != triplets * 3 {
        return Err(PyValueError::new_err(
            "per-triplet arrays must share the same triplet count (obs rows = 3 * triplets)",
        ));
    }
    if ra.ncols() != 3 || dec.ncols() != 3 || times.ncols() != 3 || obs.ncols() != 3 {
        return Err(PyValueError::new_err("triplet arrays must be 3-wide"));
    }
    let run_once = || -> PyResult<()> {
        for triplet in 0..triplets {
            let output = gauss_iod_orbits_record_batch(
                [ra[[triplet, 0]], ra[[triplet, 1]], ra[[triplet, 2]]],
                [dec[[triplet, 0]], dec[[triplet, 1]], dec[[triplet, 2]]],
                [
                    times[[triplet, 0]],
                    times[[triplet, 1]],
                    times[[triplet, 2]],
                ],
                [
                    [
                        obs[[triplet * 3, 0]],
                        obs[[triplet * 3, 1]],
                        obs[[triplet * 3, 2]],
                    ],
                    [
                        obs[[triplet * 3 + 1, 0]],
                        obs[[triplet * 3 + 1, 1]],
                        obs[[triplet * 3 + 1, 2]],
                    ],
                    [
                        obs[[triplet * 3 + 2, 0]],
                        obs[[triplet * 3 + 2, 1]],
                        obs[[triplet * 3 + 2, 2]],
                    ],
                ],
                velocity_method_id,
                light_time,
                mu,
                c,
            )
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
            std::hint::black_box(output);
        }
        Ok(())
    };
    let mut sample_trials = Vec::with_capacity(trials);
    for _ in 0..trials {
        for _ in 0..warmup_reps {
            run_once()?;
        }
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let started = std::time::Instant::now();
            run_once()?;
            samples.push(started.elapsed().as_secs_f64());
        }
        sample_trials.push(samples);
    }
    Ok(sample_trials)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calc_gibbs_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calc_herrick_gibbs_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calc_gauss_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(gauss_iod_orbits_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(gauss_iod_fused_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(gauss_iod_orbits_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_gauss_iod_orbits_arrow, m)?)?;
    Ok(())
}
