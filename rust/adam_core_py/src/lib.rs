#![allow(
    clippy::useless_conversion,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

use adam_core_rs_coords::{
    add_light_time_batch_flat, apply_cosine_latitude_correction_flat, apply_lagrange_coefficients,
    apply_stellar_aberration_row, bound_longitude_residuals_flat, calc_apoapsis_distance, calc_chi,
    calc_lagrange_coefficients, calc_mean_anomaly, calc_mean_motion_batch, calc_periapsis_distance,
    calc_period, calc_semi_latus_rectum, calc_semi_major_axis, calc_stumpff,
    calculate_apparent_magnitude_v_and_phase_angle_flat, calculate_apparent_magnitude_v_flat,
    calculate_chi2_flat, calculate_moid, calculate_moid_batch, calculate_phase_angle_flat,
    cartesian_to_cometary_flat6, cartesian_to_geodetic_flat6, cartesian_to_keplerian_flat6,
    cartesian_to_spherical_flat6, cartesian_to_spherical_row, classify_orbits_flat,
    cometary_to_cartesian_flat6, fit_absolute_magnitude_grouped, fit_absolute_magnitude_rows,
    generate_ephemeris_2body_flat6, generate_ephemeris_2body_with_covariance_flat6,
    izzo_lambert_batch_flat, keplerian_to_cartesian_flat6, porkchop_grid_flat,
    predict_magnitudes_bandpass_flat, propagate_2body_along_arc, propagate_2body_arc_batch_flat6,
    propagate_2body_flat6, propagate_2body_with_covariance_flat6, rotate_cartesian_frame_flat6,
    rotate_cartesian_time_varying_flat6, solve_barker, solve_kepler_true_anomaly,
    spherical_to_cartesian_flat6, spherical_to_cartesian_row, tisserand_parameter_flat,
    transform_with_covariance_flat6, weighted_covariance_flat, weighted_mean_flat, Frame,
    Representation as CoordsRepresentation,
};
use adam_core_rs_orbit_determination::{
    calc_gauss_row, calc_gibbs_row, calc_herrick_gibbs_row, gauss_iod_fused,
    gauss_iod_orbits_from_roots,
};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod spice;

#[derive(Clone, Copy)]
enum Representation {
    Cartesian,
    Spherical,
    Geodetic,
    Keplerian,
    Cometary,
}

fn parse_representation(value: &str) -> PyResult<Representation> {
    match value {
        "cartesian" => Ok(Representation::Cartesian),
        "spherical" => Ok(Representation::Spherical),
        "geodetic" => Ok(Representation::Geodetic),
        "keplerian" => Ok(Representation::Keplerian),
        "cometary" => Ok(Representation::Cometary),
        _ => Err(PyValueError::new_err(format!(
            "unsupported representation: {value}"
        ))),
    }
}

fn to_coords_rep(rep: Representation) -> CoordsRepresentation {
    match rep {
        Representation::Cartesian => CoordsRepresentation::Cartesian,
        Representation::Spherical => CoordsRepresentation::Spherical,
        Representation::Geodetic => CoordsRepresentation::Geodetic,
        Representation::Keplerian => CoordsRepresentation::Keplerian,
        Representation::Cometary => CoordsRepresentation::Cometary,
    }
}

fn parse_frame(value: &str) -> PyResult<Frame> {
    match value {
        "ecliptic" => Ok(Frame::Ecliptic),
        "equatorial" => Ok(Frame::Equatorial),
        "itrf93" => Ok(Frame::Itrf93),
        _ => Err(PyValueError::new_err(format!("unsupported frame: {value}"))),
    }
}

#[pyfunction]
fn cartesian_to_spherical_numpy<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr = coords.as_array();
    if arr.ncols() != 6 {
        return Err(PyValueError::new_err("coords must have shape (N, 6)"));
    }

    if let Some(flat) = arr.as_slice() {
        let out_flat = cartesian_to_spherical_flat6(flat);
        let shaped = ndarray::Array2::from_shape_vec((arr.nrows(), 6), out_flat)
            .map_err(|e| PyValueError::new_err(format!("failed to shape output: {e}")))?;
        return Ok(shaped.into_pyarray_bound(py));
    }

    let mut shaped = ndarray::Array2::<f64>::zeros((arr.nrows(), 6));
    for (i, row) in arr.rows().into_iter().enumerate() {
        let out = cartesian_to_spherical_row(&[row[0], row[1], row[2], row[3], row[4], row[5]]);
        shaped[[i, 0]] = out[0];
        shaped[[i, 1]] = out[1];
        shaped[[i, 2]] = out[2];
        shaped[[i, 3]] = out[3];
        shaped[[i, 4]] = out[4];
        shaped[[i, 5]] = out[5];
    }

    Ok(shaped.into_pyarray_bound(py))
}

#[pyfunction]
fn spherical_to_cartesian_numpy<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr = coords.as_array();
    if arr.ncols() != 6 {
        return Err(PyValueError::new_err("coords must have shape (N, 6)"));
    }

    if let Some(flat) = arr.as_slice() {
        let out_flat = spherical_to_cartesian_flat6(flat);
        let shaped = ndarray::Array2::from_shape_vec((arr.nrows(), 6), out_flat)
            .map_err(|e| PyValueError::new_err(format!("failed to shape output: {e}")))?;
        return Ok(shaped.into_pyarray_bound(py));
    }

    let mut shaped = ndarray::Array2::<f64>::zeros((arr.nrows(), 6));
    for (i, row) in arr.rows().into_iter().enumerate() {
        let out = spherical_to_cartesian_row(&[row[0], row[1], row[2], row[3], row[4], row[5]]);
        shaped[[i, 0]] = out[0];
        shaped[[i, 1]] = out[1];
        shaped[[i, 2]] = out[2];
        shaped[[i, 3]] = out[3];
        shaped[[i, 4]] = out[4];
        shaped[[i, 5]] = out[5];
    }

    Ok(shaped.into_pyarray_bound(py))
}

#[pyfunction]
fn cartesian_to_geodetic_numpy<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    a: f64,
    f: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr = coords.as_array();
    if arr.ncols() != 6 {
        return Err(PyValueError::new_err("coords must have shape (N, 6)"));
    }

    if let Some(flat) = arr.as_slice() {
        let out_flat = cartesian_to_geodetic_flat6(flat, a, f, max_iter, tol);
        let shaped = ndarray::Array2::from_shape_vec((arr.nrows(), 6), out_flat)
            .map_err(|e| PyValueError::new_err(format!("failed to shape output: {e}")))?;
        return Ok(shaped.into_pyarray_bound(py));
    }

    Err(PyValueError::new_err("coords must be contiguous"))
}

#[pyfunction]
fn cartesian_to_keplerian_numpy<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    t0: PyReadonlyArray1<'py, f64>,
    mu: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr = coords.as_array();
    let t0_arr = t0.as_array();
    let mu_arr = mu.as_array();

    if arr.ncols() != 6 {
        return Err(PyValueError::new_err("coords must have shape (N, 6)"));
    }
    if t0_arr.len() != arr.nrows() || mu_arr.len() != arr.nrows() {
        return Err(PyValueError::new_err(
            "t0 and mu must each have length N for coords shape (N, 6)",
        ));
    }

    let flat = arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("coords must be contiguous"))?;
    let t0_slice = t0_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("t0 must be contiguous"))?;
    let mu_slice = mu_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("mu must be contiguous"))?;

    let out_flat = cartesian_to_keplerian_flat6(flat, t0_slice, mu_slice);
    let shaped = ndarray::Array2::from_shape_vec((arr.nrows(), 13), out_flat)
        .map_err(|e| PyValueError::new_err(format!("failed to shape output: {e}")))?;
    Ok(shaped.into_pyarray_bound(py))
}

#[pyfunction]
fn keplerian_to_cartesian_numpy<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    mu: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr = coords.as_array();
    let mu_arr = mu.as_array();

    if arr.ncols() != 6 {
        return Err(PyValueError::new_err("coords must have shape (N, 6)"));
    }
    if mu_arr.len() != arr.nrows() {
        return Err(PyValueError::new_err(
            "mu must have length N for coords shape (N, 6)",
        ));
    }

    let flat = arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("coords must be contiguous"))?;
    let mu_slice = mu_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("mu must be contiguous"))?;

    let out_flat = keplerian_to_cartesian_flat6(flat, mu_slice, max_iter, tol);
    let shaped = ndarray::Array2::from_shape_vec((arr.nrows(), 6), out_flat)
        .map_err(|e| PyValueError::new_err(format!("failed to shape output: {e}")))?;
    Ok(shaped.into_pyarray_bound(py))
}

#[pyfunction]
fn cartesian_to_cometary_numpy<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    t0: PyReadonlyArray1<'py, f64>,
    mu: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr = coords.as_array();
    let t0_arr = t0.as_array();
    let mu_arr = mu.as_array();

    if arr.ncols() != 6 {
        return Err(PyValueError::new_err("coords must have shape (N, 6)"));
    }
    if t0_arr.len() != arr.nrows() || mu_arr.len() != arr.nrows() {
        return Err(PyValueError::new_err(
            "t0 and mu must each have length N for coords shape (N, 6)",
        ));
    }

    let flat = arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("coords must be contiguous"))?;
    let t0_slice = t0_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("t0 must be contiguous"))?;
    let mu_slice = mu_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("mu must be contiguous"))?;

    let out_flat = cartesian_to_cometary_flat6(flat, t0_slice, mu_slice);
    let shaped = ndarray::Array2::from_shape_vec((arr.nrows(), 6), out_flat)
        .map_err(|e| PyValueError::new_err(format!("failed to shape output: {e}")))?;
    Ok(shaped.into_pyarray_bound(py))
}

#[pyfunction]
fn cometary_to_cartesian_numpy<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    t0: PyReadonlyArray1<'py, f64>,
    mu: PyReadonlyArray1<'py, f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr = coords.as_array();
    let t0_arr = t0.as_array();
    let mu_arr = mu.as_array();

    if arr.ncols() != 6 {
        return Err(PyValueError::new_err("coords must have shape (N, 6)"));
    }
    if t0_arr.len() != arr.nrows() || mu_arr.len() != arr.nrows() {
        return Err(PyValueError::new_err(
            "t0 and mu must each have length N for coords shape (N, 6)",
        ));
    }

    let flat = arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("coords must be contiguous"))?;
    let t0_slice = t0_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("t0 must be contiguous"))?;
    let mu_slice = mu_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("mu must be contiguous"))?;

    let out_flat = cometary_to_cartesian_flat6(flat, t0_slice, mu_slice, max_iter, tol);
    let shaped = ndarray::Array2::from_shape_vec((arr.nrows(), 6), out_flat)
        .map_err(|e| PyValueError::new_err(format!("failed to shape output: {e}")))?;
    Ok(shaped.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (coords, representation_in, representation_out, t0=None, mu=None, a=None, f=None, max_iter=100, tol=1e-15, frame_in=None, frame_out=None, translation_vectors=None))]
#[allow(clippy::too_many_arguments)]
fn transform_coordinates_numpy<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    representation_in: &str,
    representation_out: &str,
    t0: Option<PyReadonlyArray1<'py, f64>>,
    mu: Option<PyReadonlyArray1<'py, f64>>,
    a: Option<f64>,
    f: Option<f64>,
    max_iter: usize,
    tol: f64,
    frame_in: Option<&str>,
    frame_out: Option<&str>,
    translation_vectors: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr = coords.as_array();
    if arr.ncols() != 6 {
        return Err(PyValueError::new_err("coords must have shape (N, 6)"));
    }

    let flat = arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("coords must be contiguous"))?;
    let n = arr.nrows();

    let rep_in = parse_representation(representation_in)?;
    let rep_out = parse_representation(representation_out)?;
    let frame_in_name = frame_in.unwrap_or("ecliptic");
    let frame_out_name = frame_out.unwrap_or(frame_in_name);
    let frame_in_value = parse_frame(frame_in_name)?;
    let frame_out_value = parse_frame(frame_out_name)?;

    let cartesian_flat = match rep_in {
        Representation::Cartesian => flat.to_vec(),
        Representation::Spherical => spherical_to_cartesian_flat6(flat),
        Representation::Keplerian => {
            let mu_arr = mu
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("mu is required"))?;
            let mu_view = mu_arr.as_array();
            if mu_view.len() != n {
                return Err(PyValueError::new_err(
                    "mu must have length N for coords shape (N, 6)",
                ));
            }
            let mu_slice = mu_view
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("mu must be contiguous"))?;
            keplerian_to_cartesian_flat6(flat, mu_slice, max_iter, tol)
        }
        Representation::Cometary => {
            let t0_arr = t0
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("t0 is required"))?;
            let mu_arr = mu
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("mu is required"))?;
            let t0_view = t0_arr.as_array();
            let mu_view = mu_arr.as_array();
            if t0_view.len() != n || mu_view.len() != n {
                return Err(PyValueError::new_err(
                    "t0 and mu must each have length N for coords shape (N, 6)",
                ));
            }
            let t0_slice = t0_view
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("t0 must be contiguous"))?;
            let mu_slice = mu_view
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("mu must be contiguous"))?;
            cometary_to_cartesian_flat6(flat, t0_slice, mu_slice, max_iter, tol)
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "unsupported transform path: {representation_in} -> {representation_out}"
            )));
        }
    };
    // Origin translation is a covariance-invariant constant offset applied
    // in the input frame, before any frame rotation. SPICE-resolved
    // translation vectors are passed in as an (N, 6) array by the caller.
    let mut cartesian_translated = cartesian_flat;
    if let Some(tv) = translation_vectors.as_ref() {
        let tv_view = tv.as_array();
        if tv_view.nrows() != n || tv_view.ncols() != 6 {
            return Err(PyValueError::new_err(
                "translation_vectors must have shape (N, 6)",
            ));
        }
        let tv_slice = tv_view
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("translation_vectors must be contiguous"))?;
        for i in 0..(n * 6) {
            cartesian_translated[i] += tv_slice[i];
        }
    }
    let cartesian_in_out_frame = if frame_in_value == frame_out_value {
        cartesian_translated
    } else {
        rotate_cartesian_frame_flat6(&cartesian_translated, frame_in_value, frame_out_value)
            .map_err(PyValueError::new_err)?
    };

    let (out_flat, ncols): (Vec<f64>, usize) = match rep_out {
        Representation::Cartesian => (cartesian_in_out_frame, 6),
        Representation::Spherical => (cartesian_to_spherical_flat6(&cartesian_in_out_frame), 6),
        Representation::Geodetic => {
            let a_value = a.ok_or_else(|| PyValueError::new_err("a is required"))?;
            let f_value = f.ok_or_else(|| PyValueError::new_err("f is required"))?;
            (
                cartesian_to_geodetic_flat6(
                    &cartesian_in_out_frame,
                    a_value,
                    f_value,
                    max_iter,
                    tol,
                ),
                6,
            )
        }
        Representation::Keplerian => {
            let t0_arr = t0.ok_or_else(|| PyValueError::new_err("t0 is required"))?;
            let mu_arr = mu
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("mu is required"))?;
            let t0_view = t0_arr.as_array();
            let mu_view = mu_arr.as_array();
            if t0_view.len() != n || mu_view.len() != n {
                return Err(PyValueError::new_err(
                    "t0 and mu must each have length N for coords shape (N, 6)",
                ));
            }
            let t0_slice = t0_view
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("t0 must be contiguous"))?;
            let mu_slice = mu_view
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("mu must be contiguous"))?;
            (
                cartesian_to_keplerian_flat6(&cartesian_in_out_frame, t0_slice, mu_slice),
                13,
            )
        }
        Representation::Cometary => {
            let t0_arr = t0.ok_or_else(|| PyValueError::new_err("t0 is required"))?;
            let mu_arr = mu
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("mu is required"))?;
            let t0_view = t0_arr.as_array();
            let mu_view = mu_arr.as_array();
            if t0_view.len() != n || mu_view.len() != n {
                return Err(PyValueError::new_err(
                    "t0 and mu must each have length N for coords shape (N, 6)",
                ));
            }
            let t0_slice = t0_view
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("t0 must be contiguous"))?;
            let mu_slice = mu_view
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("mu must be contiguous"))?;
            (
                cartesian_to_cometary_flat6(&cartesian_in_out_frame, t0_slice, mu_slice),
                6,
            )
        }
    };

    let shaped = ndarray::Array2::from_shape_vec((n, ncols), out_flat)
        .map_err(|e| PyValueError::new_err(format!("failed to shape output: {e}")))?;
    Ok(shaped.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (coords, covariances, representation_in, representation_out, t0=None, mu=None, a=None, f=None, max_iter=100, tol=1e-15, frame_in=None, frame_out=None, translation_vectors=None))]
#[allow(clippy::too_many_arguments)]
fn transform_coordinates_with_covariance_numpy<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    covariances: PyReadonlyArray2<'py, f64>,
    representation_in: &str,
    representation_out: &str,
    t0: Option<PyReadonlyArray1<'py, f64>>,
    mu: Option<PyReadonlyArray1<'py, f64>>,
    a: Option<f64>,
    f: Option<f64>,
    max_iter: usize,
    tol: f64,
    frame_in: Option<&str>,
    frame_out: Option<&str>,
    translation_vectors: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let coords_arr = coords.as_array();
    let cov_arr = covariances.as_array();
    if coords_arr.ncols() != 6 {
        return Err(PyValueError::new_err("coords must have shape (N, 6)"));
    }
    let n = coords_arr.nrows();
    if cov_arr.nrows() != n || cov_arr.ncols() != 36 {
        return Err(PyValueError::new_err(
            "covariances must have shape (N, 36) with row-major 6x6 per row",
        ));
    }

    let rep_in = to_coords_rep(parse_representation(representation_in)?);
    let rep_out = to_coords_rep(parse_representation(representation_out)?);
    let frame_in_name = frame_in.unwrap_or("ecliptic");
    let frame_out_name = frame_out.unwrap_or(frame_in_name);
    let frame_in_value = parse_frame(frame_in_name)?;
    let frame_out_value = parse_frame(frame_out_name)?;

    let coords_flat = coords_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("coords must be contiguous"))?;
    let cov_flat = cov_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("covariances must be contiguous"))?;

    // Default t0, mu to zeros when not required by the rep chain but caller supplied None.
    let t0_view = t0.as_ref().map(|arr| arr.as_array());
    let t0_fallback: Vec<f64>;
    let t0_slice: &[f64] = if let Some(view) = t0_view.as_ref() {
        if view.len() != n {
            return Err(PyValueError::new_err(
                "t0 must have length N for coords shape (N, 6)",
            ));
        }
        view.as_slice()
            .ok_or_else(|| PyValueError::new_err("t0 must be contiguous"))?
    } else {
        t0_fallback = vec![0.0_f64; n];
        &t0_fallback
    };
    let mu_view = mu.as_ref().map(|arr| arr.as_array());
    let mu_fallback: Vec<f64>;
    let mu_slice: &[f64] = if let Some(view) = mu_view.as_ref() {
        if view.len() != n {
            return Err(PyValueError::new_err(
                "mu must have length N for coords shape (N, 6)",
            ));
        }
        view.as_slice()
            .ok_or_else(|| PyValueError::new_err("mu must be contiguous"))?
    } else {
        mu_fallback = vec![0.0_f64; n];
        &mu_fallback
    };

    let a_value = a.unwrap_or(0.0);
    let f_value = f.unwrap_or(0.0);

    let tv_view = translation_vectors.as_ref().map(|arr| arr.as_array());
    let translation_slice: Option<&[f64]> = if let Some(view) = tv_view.as_ref() {
        if view.nrows() != n || view.ncols() != 6 {
            return Err(PyValueError::new_err(
                "translation_vectors must have shape (N, 6)",
            ));
        }
        Some(
            view.as_slice()
                .ok_or_else(|| PyValueError::new_err("translation_vectors must be contiguous"))?,
        )
    } else {
        None
    };

    let (coords_out_flat, cov_out_flat) = transform_with_covariance_flat6(
        coords_flat,
        cov_flat,
        rep_in,
        rep_out,
        frame_in_value,
        frame_out_value,
        t0_slice,
        mu_slice,
        a_value,
        f_value,
        max_iter,
        tol,
        translation_slice,
    );

    let coords_shaped = ndarray::Array2::from_shape_vec((n, 6), coords_out_flat)
        .map_err(|e| PyValueError::new_err(format!("failed to shape coords output: {e}")))?;
    let cov_shaped = ndarray::Array2::from_shape_vec((n, 36), cov_out_flat)
        .map_err(|e| PyValueError::new_err(format!("failed to shape covariance output: {e}")))?;
    Ok((
        coords_shaped.into_pyarray_bound(py),
        cov_shaped.into_pyarray_bound(py),
    ))
}

/// Apply per-row 6x6 state-transform matrices to a batch of Cartesian
/// states (and optionally their 6x6 covariances). `time_index[i]`
/// selects which matrix from `matrices` to apply to row `i` — callers
/// with many rows sharing few unique epochs pass a deduplicated
/// `matrices` table and an index map instead of one matrix per row.
///
/// Matrices are applied in the units the caller supplies; no unit
/// conversion is performed here. Covariance NaN rows pass through.
#[pyfunction]
#[pyo3(signature = (coords, time_index, matrices, covariances=None))]
fn rotate_cartesian_time_varying_numpy<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    time_index: PyReadonlyArray1<'py, i64>,
    matrices: PyReadonlyArray3<'py, f64>,
    covariances: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Option<Bound<'py, PyArray2<f64>>>)> {
    let coords_arr = coords.as_array();
    if coords_arr.ncols() != 6 {
        return Err(PyValueError::new_err("coords must have shape (N, 6)"));
    }
    let n = coords_arr.nrows();

    let mats_arr = matrices.as_array();
    if mats_arr.shape().len() != 3 || mats_arr.shape()[1] != 6 || mats_arr.shape()[2] != 6 {
        return Err(PyValueError::new_err("matrices must have shape (U, 6, 6)"));
    }

    let idx_arr = time_index.as_array();
    if idx_arr.len() != n {
        return Err(PyValueError::new_err(
            "time_index length must match coords rows",
        ));
    }
    let idx_slice = idx_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("time_index must be contiguous"))?;
    // i64 → usize with a non-negative check; errors here surface to
    // Python as ValueError rather than silently wrapping.
    let time_index_usize: Vec<usize> = idx_slice
        .iter()
        .map(|&v| {
            if v < 0 {
                Err(PyValueError::new_err("time_index must be non-negative"))
            } else {
                Ok(v as usize)
            }
        })
        .collect::<PyResult<_>>()?;

    let coords_flat = coords_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("coords must be contiguous"))?;
    let matrices_flat = mats_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("matrices must be contiguous"))?;

    let cov_view = covariances.as_ref().map(|arr| arr.as_array());
    let cov_empty: Vec<f64> = Vec::new();
    let cov_flat: &[f64] = if let Some(view) = cov_view.as_ref() {
        if view.nrows() != n || view.ncols() != 36 {
            return Err(PyValueError::new_err(
                "covariances must have shape (N, 36) with row-major 6x6 per row",
            ));
        }
        view.as_slice()
            .ok_or_else(|| PyValueError::new_err("covariances must be contiguous"))?
    } else {
        &cov_empty
    };

    let (coords_out_flat, cov_out_flat) = rotate_cartesian_time_varying_flat6(
        coords_flat,
        cov_flat,
        &time_index_usize,
        matrices_flat,
    )
    .map_err(PyValueError::new_err)?;

    let coords_shaped = ndarray::Array2::from_shape_vec((n, 6), coords_out_flat)
        .map_err(|e| PyValueError::new_err(format!("failed to shape coords output: {e}")))?;
    let cov_out = if cov_view.is_some() {
        let shaped = ndarray::Array2::from_shape_vec((n, 36), cov_out_flat).map_err(|e| {
            PyValueError::new_err(format!("failed to shape covariance output: {e}"))
        })?;
        Some(shaped.into_pyarray_bound(py))
    } else {
        None
    };
    Ok((coords_shaped.into_pyarray_bound(py), cov_out))
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

    Ok(out.into_pyarray_bound(py))
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
    Ok(out.into_pyarray_bound(py))
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
    Ok(out.into_pyarray_bound(py))
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
    Ok(out.into_pyarray_bound(py))
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
    Ok(out.into_pyarray_bound(py))
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
    Ok(out.into_pyarray_bound(py))
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
    Ok(out.into_pyarray_bound(py))
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
    Ok(out.into_pyarray_bound(py))
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
    Ok(out.into_pyarray_bound(py))
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
    Ok(shaped.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (r, v, dts, mus, max_iter=100, tol=1e-16))]
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
    Ok(shaped.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (r, v, dts, mus, max_iter=100, tol=1e-16))]
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
        coeffs.into_pyarray_bound(py),
        stumpff.into_pyarray_bound(py),
        ndarray::Array1::from_vec(chi_out).into_pyarray_bound(py),
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
    Ok((
        r_shaped.into_pyarray_bound(py),
        v_shaped.into_pyarray_bound(py),
    ))
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
    Ok(shaped.into_pyarray_bound(py))
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
    Ok(shaped.into_pyarray_bound(py))
}

#[pyfunction]
fn bound_longitude_residuals_numpy<'py>(
    py: Python<'py>,
    observed: PyReadonlyArray2<'py, f64>,
    residuals: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let obs = observed.as_array();
    let res = residuals.as_array();
    let n = obs.nrows();
    let d = obs.ncols();
    if res.shape() != [n, d] {
        return Err(PyValueError::new_err("residuals must match observed shape"));
    }
    let obs_slice = obs
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("observed must be contiguous"))?;
    let res_slice = res
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("residuals must be contiguous"))?;
    // Copy residuals into an owned mutable buffer for the in-place op.
    let mut buf = res_slice.to_vec();
    bound_longitude_residuals_flat(obs_slice, &mut buf, n, d);
    let arr = ndarray::Array2::from_shape_vec((n, d), buf)
        .map_err(|e| PyValueError::new_err(format!("shape: {e}")))?;
    Ok(arr.into_pyarray_bound(py))
}

#[pyfunction]
fn apply_cosine_latitude_correction_numpy<'py>(
    py: Python<'py>,
    lat: PyReadonlyArray1<'py, f64>,
    residuals: PyReadonlyArray2<'py, f64>,
    covariances: PyReadonlyArray3<'py, f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray3<f64>>)> {
    let lat_arr = lat.as_array();
    let res_arr = residuals.as_array();
    let cov_arr = covariances.as_array();
    let n = lat_arr.len();
    let d = res_arr.ncols();
    if res_arr.nrows() != n {
        return Err(PyValueError::new_err(
            "residuals row count must match lat length",
        ));
    }
    if cov_arr.shape() != [n, d, d] {
        return Err(PyValueError::new_err(format!(
            "covariances shape {:?} must be (N={n}, D={d}, D={d})",
            cov_arr.shape()
        )));
    }
    let lat_slice = lat_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("lat must be contiguous"))?;
    let res_slice = res_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("residuals must be contiguous"))?;
    let cov_slice = cov_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("covariances must be contiguous"))?;
    let mut res_buf = res_slice.to_vec();
    let mut cov_buf = cov_slice.to_vec();
    apply_cosine_latitude_correction_flat(lat_slice, &mut res_buf, &mut cov_buf, n, d);
    let res_out = ndarray::Array2::from_shape_vec((n, d), res_buf)
        .map_err(|e| PyValueError::new_err(format!("shape: {e}")))?;
    let cov_out = ndarray::Array3::from_shape_vec((n, d, d), cov_buf)
        .map_err(|e| PyValueError::new_err(format!("shape: {e}")))?;
    Ok((
        res_out.into_pyarray_bound(py),
        cov_out.into_pyarray_bound(py),
    ))
}

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
        ndarray::Array1::from_vec(h_hat).into_pyarray_bound(py),
        ndarray::Array1::from_vec(h_sigma).into_pyarray_bound(py),
        ndarray::Array1::from_vec(sigma_eff).into_pyarray_bound(py),
        ndarray::Array1::from_vec(chi2_red).into_pyarray_bound(py),
        ndarray::Array1::from_vec(n_used).into_pyarray_bound(py),
    ))
}

#[pyfunction]
fn weighted_mean_numpy<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray2<'py, f64>,
    w: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let s = samples.as_array();
    let w_arr = w.as_array();
    let n = s.nrows();
    let d = s.ncols();
    if w_arr.len() != n {
        return Err(PyValueError::new_err("W must have length N"));
    }
    let s_slice = s
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("samples must be contiguous"))?;
    let w_slice = w_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("W must be contiguous"))?;
    Ok(
        ndarray::Array1::from_vec(weighted_mean_flat(s_slice, w_slice, n, d))
            .into_pyarray_bound(py),
    )
}

#[pyfunction]
fn weighted_covariance_numpy<'py>(
    py: Python<'py>,
    mean: PyReadonlyArray1<'py, f64>,
    samples: PyReadonlyArray2<'py, f64>,
    w_cov: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let m = mean.as_array();
    let s = samples.as_array();
    let w_arr = w_cov.as_array();
    let d = m.len();
    let n = s.nrows();
    if s.ncols() != d {
        return Err(PyValueError::new_err(
            "samples must have D columns matching mean",
        ));
    }
    if w_arr.len() != n {
        return Err(PyValueError::new_err("W_cov must have length N"));
    }
    let m_slice = m
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("mean must be contiguous"))?;
    let s_slice = s
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("samples must be contiguous"))?;
    let w_slice = w_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("W_cov must be contiguous"))?;
    let cov = weighted_covariance_flat(m_slice, s_slice, w_slice, n, d);
    let shaped = ndarray::Array2::from_shape_vec((d, d), cov)
        .map_err(|e| PyValueError::new_err(format!("failed to shape: {e}")))?;
    Ok(shaped.into_pyarray_bound(py))
}

#[pyfunction]
fn calculate_chi2_numpy<'py>(
    py: Python<'py>,
    residuals: PyReadonlyArray2<'py, f64>,
    covariances: PyReadonlyArray3<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let r_arr = residuals.as_array();
    let c_arr = covariances.as_array();
    let n = r_arr.nrows();
    let d = r_arr.ncols();
    if c_arr.shape() != [n, d, d] {
        return Err(PyValueError::new_err(format!(
            "covariances shape {:?} must be (N={n}, D={d}, D={d})",
            c_arr.shape()
        )));
    }
    let r_slice = r_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("residuals must be contiguous"))?;
    let c_slice = c_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("covariances must be contiguous"))?;
    match calculate_chi2_flat(r_slice, c_slice, n, d) {
        Ok(out) => Ok(ndarray::Array1::from_vec(out).into_pyarray_bound(py)),
        Err(adam_core_rs_coords::chi2::Chi2Error::NanDiagonal { row, dim }) => {
            Err(PyValueError::new_err(format!(
                "Covariance matrix has NaN on diagonal (row={row}, dim={dim})."
            )))
        }
        Err(adam_core_rs_coords::chi2::Chi2Error::NotPositiveDefinite { row }) => {
            Err(PyValueError::new_err(format!(
                "Covariance matrix at row {row} is not positive definite."
            )))
        }
    }
}

#[pyfunction]
fn classify_orbits_numpy<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    e: PyReadonlyArray1<'py, f64>,
    q: PyReadonlyArray1<'py, f64>,
    q_apo: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let a_arr = a.as_array();
    let e_arr = e.as_array();
    let q_arr = q.as_array();
    let q_apo_arr = q_apo.as_array();
    let n = a_arr.len();
    if e_arr.len() != n || q_arr.len() != n || q_apo_arr.len() != n {
        return Err(PyValueError::new_err("a, e, q, Q must have equal length"));
    }
    let a_s = a_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("a must be contiguous"))?;
    let e_s = e_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("e must be contiguous"))?;
    let q_s = q_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("q must be contiguous"))?;
    let q_apo_s = q_apo_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Q must be contiguous"))?;
    let codes = classify_orbits_flat(a_s, e_s, q_s, q_apo_s);
    Ok(ndarray::Array1::from_vec(codes).into_pyarray_bound(py))
}

#[pyfunction]
fn tisserand_parameter_numpy<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    e: PyReadonlyArray1<'py, f64>,
    i_deg: PyReadonlyArray1<'py, f64>,
    ap: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a_arr = a.as_array();
    let e_arr = e.as_array();
    let i_arr = i_deg.as_array();
    if a_arr.len() != e_arr.len() || a_arr.len() != i_arr.len() {
        return Err(PyValueError::new_err("a, e, i_deg must have equal length"));
    }
    let a_slice = a_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("a must be contiguous"))?;
    let e_slice = e_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("e must be contiguous"))?;
    let i_slice = i_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("i_deg must be contiguous"))?;
    let out = tisserand_parameter_flat(a_slice, e_slice, i_slice, ap);
    Ok(ndarray::Array1::from_vec(out).into_pyarray_bound(py))
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
    Ok(shaped.into_pyarray_bound(py))
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
    Ok(shaped.into_pyarray_bound(py))
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
    Ok((
        states_shaped.into_pyarray_bound(py),
        cov_shaped.into_pyarray_bound(py),
    ))
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
        sph_shaped.into_pyarray_bound(py),
        lt_shaped.into_pyarray_bound(py),
        aberrated_shaped.into_pyarray_bound(py),
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
        sph_shaped.into_pyarray_bound(py),
        lt_shaped.into_pyarray_bound(py),
        aberrated_shaped.into_pyarray_bound(py),
        cov_shaped.into_pyarray_bound(py),
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
    let out = calculate_phase_angle_flat(obj_slice, obs_slice);
    Ok(ndarray::Array1::from_vec(out).into_pyarray_bound(py))
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
    let out = calculate_apparent_magnitude_v_flat(h_slice, obj_slice, obs_slice, g_slice);
    Ok(ndarray::Array1::from_vec(out).into_pyarray_bound(py))
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
    let (mag_out, alpha_out) =
        calculate_apparent_magnitude_v_and_phase_angle_flat(h_slice, obj_slice, obs_slice, g_slice);
    Ok((
        ndarray::Array1::from_vec(mag_out).into_pyarray_bound(py),
        ndarray::Array1::from_vec(alpha_out).into_pyarray_bound(py),
    ))
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
    let out = predict_magnitudes_bandpass_flat(
        h_slice, obj_slice, obs_slice, g_slice, tid_slice, dt_slice,
    );
    Ok(ndarray::Array1::from_vec(out).into_pyarray_bound(py))
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
        ab_shaped.into_pyarray_bound(py),
        ndarray::Array1::from_vec(lt).into_pyarray_bound(py),
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
        ndarray::Array1::from_vec(moids).into_pyarray_bound(py),
        ndarray::Array1::from_vec(dts).into_pyarray_bound(py),
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
    Ok((
        v1_shaped.into_pyarray_bound(py),
        v2_shaped.into_pyarray_bound(py),
    ))
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
        ndarray::Array1::from_vec(dep_idx).into_pyarray_bound(py),
        ndarray::Array1::from_vec(arr_idx).into_pyarray_bound(py),
        v1_shaped.into_pyarray_bound(py),
        v2_shaped.into_pyarray_bound(py),
    ))
}

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

#[pymodule]
fn _rust_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cartesian_to_spherical_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(spherical_to_cartesian_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(cartesian_to_geodetic_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(cartesian_to_keplerian_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(keplerian_to_cartesian_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(cartesian_to_cometary_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(cometary_to_cartesian_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(transform_coordinates_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(
        transform_coordinates_with_covariance_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(rotate_cartesian_time_varying_numpy, m)?)?;
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
    m.add_function(wrap_pyfunction!(tisserand_parameter_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(classify_orbits_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_chi2_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_mean_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_covariance_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(fit_absolute_magnitude_rows_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(fit_absolute_magnitude_grouped_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(bound_longitude_residuals_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(apply_cosine_latitude_correction_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_2body_along_arc_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_2body_arc_batch_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_2body_with_covariance_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(generate_ephemeris_2body_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(
        generate_ephemeris_2body_with_covariance_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(calculate_phase_angle_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_apparent_magnitude_v_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(
        calculate_apparent_magnitude_v_and_phase_angle_numpy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(predict_magnitudes_bandpass_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(izzo_lambert_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(porkchop_grid_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(add_light_time_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_moid_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_moid_batch_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calc_gibbs_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calc_herrick_gibbs_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calc_gauss_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(gauss_iod_orbits_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(gauss_iod_fused_numpy, m)?)?;
    spice::register(m)?;
    Ok(())
}
