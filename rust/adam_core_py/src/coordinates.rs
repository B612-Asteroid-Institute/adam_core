use adam_core_rs_coords::propagation::{
    CovariancePropagation, EpochPolicy, PropagationOptions, PropagationRequest, Propagator,
    TwoBodyPropagator, TwoBodyPropagatorConfig,
};
use adam_core_rs_coords::{
    apply_cosine_latitude_correction_flat, bound_longitude_residuals_flat, bound_longitude_value,
    calculate_chi2_flat, cartesian_to_cometary_flat6, cartesian_to_geodetic_flat6,
    cartesian_to_keplerian_flat6, cartesian_to_spherical_flat6, cartesian_to_spherical_row,
    classify_orbits_flat, cometary_to_cartesian_flat6, create_sampled_orbit_variants,
    fit_orbit_2body_least_squares, generate_ephemeris_2body_flat6, keplerian_to_cartesian_flat6,
    origin_mu_au3_day2, rotate_cartesian_time_varying_flat6,
    spherical_to_cartesian_flat6, spherical_to_cartesian_row, tisserand_parameter_flat,
    transform_values_flat6, transform_with_covariance_flat6, weighted_covariance_flat,
    weighted_mean_flat, ArrowSchemaExport, CoordinateBatch as DataCoordinateBatch, DataFrame,
    Epoch, Frame, IntoNestedRecordBatch, LeastSquaresConfig, ObserverBatch as DataObserverBatch,
    OrbitBatch as DataOrbitBatch, OrbitVariantBatch as DataOrbitVariantBatch,
    OrbitVariantSamplingMethod, Representation as CoordsRepresentation, TimeArray, TimeScale,
    TimeScaleProvider, TryFromNestedRecordBatch,
};
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use arrow_array::RecordBatch;
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::HashMap;

fn schema_metadata(schema: arrow_schema::Schema) -> (Vec<String>, HashMap<String, String>) {
    let fields = schema
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
    (fields, schema.metadata().clone())
}

#[pyfunction]
fn cartesian_coordinate_schema_metadata() -> (Vec<String>, HashMap<String, String>) {
    schema_metadata(DataCoordinateBatch::schema())
}

#[pyfunction]
fn orbit_schema_metadata() -> (Vec<String>, HashMap<String, String>) {
    schema_metadata(DataOrbitBatch::schema())
}

#[derive(Clone, Copy, PartialEq, Eq)]
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

fn covariance_transform_requires_mu(rep_in: Representation, rep_out: Representation) -> bool {
    matches!(rep_in, Representation::Keplerian | Representation::Cometary)
        || matches!(
            rep_out,
            Representation::Keplerian | Representation::Cometary
        )
}

fn covariance_transform_requires_t0(rep_in: Representation, rep_out: Representation) -> bool {
    matches!(rep_in, Representation::Cometary)
        || matches!(
            rep_out,
            Representation::Keplerian | Representation::Cometary
        )
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
        return Ok(shaped.into_pyarray(py));
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

    Ok(shaped.into_pyarray(py))
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
        return Ok(shaped.into_pyarray(py));
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

    Ok(shaped.into_pyarray(py))
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
        return Ok(shaped.into_pyarray(py));
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
    Ok(shaped.into_pyarray(py))
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
    Ok(shaped.into_pyarray(py))
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
    Ok(shaped.into_pyarray(py))
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
    Ok(shaped.into_pyarray(py))
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

    // Value composition (rep-in -> cart -> translate -> constant-frame
    // rotate -> rep-out) is the reusable coords primitive transform_values_flat6;
    // this wrapper only marshals numpy buffers and enforces the required
    // per-representation arguments.
    if rep_in == Representation::Geodetic {
        return Err(PyValueError::new_err(format!(
            "unsupported transform path: {representation_in} -> {representation_out}"
        )));
    }
    let needs_mu = matches!(rep_in, Representation::Keplerian | Representation::Cometary)
        || matches!(
            rep_out,
            Representation::Keplerian | Representation::Cometary
        );
    let needs_t0 = matches!(rep_in, Representation::Cometary)
        || matches!(
            rep_out,
            Representation::Keplerian | Representation::Cometary
        );

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
    } else if needs_mu {
        return Err(PyValueError::new_err("mu is required"));
    } else {
        mu_fallback = vec![0.0_f64; n];
        &mu_fallback
    };

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
    } else if needs_t0 {
        return Err(PyValueError::new_err("t0 is required"));
    } else {
        t0_fallback = vec![0.0_f64; n];
        &t0_fallback
    };

    let a_value = if rep_out == Representation::Geodetic {
        a.ok_or_else(|| PyValueError::new_err("a is required"))?
    } else {
        a.unwrap_or(0.0)
    };
    let f_value = if rep_out == Representation::Geodetic {
        f.ok_or_else(|| PyValueError::new_err("f is required"))?
    } else {
        f.unwrap_or(0.0)
    };

    let translation_view = translation_vectors.as_ref().map(|arr| arr.as_array());
    let translation_slice: Option<&[f64]> = if let Some(view) = translation_view.as_ref() {
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

    let (out_flat, ncols) = transform_values_flat6(
        flat,
        to_coords_rep(rep_in),
        to_coords_rep(rep_out),
        frame_in_value,
        frame_out_value,
        t0_slice,
        mu_slice,
        a_value,
        f_value,
        max_iter,
        tol,
        translation_slice,
    )
    .map_err(PyValueError::new_err)?;

    let shaped = ndarray::Array2::from_shape_vec((n, ncols), out_flat)
        .map_err(|e| PyValueError::new_err(format!("failed to shape output: {e}")))?;
    Ok(shaped.into_pyarray(py))
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

    let rep_in_parsed = parse_representation(representation_in)?;
    let rep_out_parsed = parse_representation(representation_out)?;
    if rep_in_parsed == Representation::Geodetic {
        return Err(PyValueError::new_err(
            "geodetic input is not supported by transform_coordinates_with_covariance_numpy",
        ));
    }
    if covariance_transform_requires_t0(rep_in_parsed, rep_out_parsed) && t0.is_none() {
        return Err(PyValueError::new_err(
            "t0 is required for Keplerian/Cometary covariance transforms",
        ));
    }
    if covariance_transform_requires_mu(rep_in_parsed, rep_out_parsed) && mu.is_none() {
        return Err(PyValueError::new_err(
            "mu is required for Keplerian/Cometary covariance transforms",
        ));
    }
    if rep_out_parsed == Representation::Geodetic && a.is_none() {
        return Err(PyValueError::new_err(
            "a is required for geodetic covariance transforms",
        ));
    }
    if rep_out_parsed == Representation::Geodetic && f.is_none() {
        return Err(PyValueError::new_err(
            "f is required for geodetic covariance transforms",
        ));
    }

    let rep_in = to_coords_rep(rep_in_parsed);
    let rep_out = to_coords_rep(rep_out_parsed);
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
    Ok((coords_shaped.into_pyarray(py), cov_shaped.into_pyarray(py)))
}

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
        Some(shaped.into_pyarray(py))
    } else {
        None
    };
    Ok((coords_shaped.into_pyarray(py), cov_out))
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
    Ok(arr.into_pyarray(py))
}

/// Column-only longitude wrap: reads column 1 of `observed` and `residuals`
/// (strided access allowed) and returns the wrapped longitude-residual
/// column. Avoids the 3x full-(N, D) buffer traffic of
/// `bound_longitude_residuals_numpy` when only the longitude column is
/// needed; the public Python wrapper assigns it back in place.
#[pyfunction]
fn bound_longitude_residual_column_numpy<'py>(
    py: Python<'py>,
    observed: PyReadonlyArray2<'py, f64>,
    residuals: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let obs = observed.as_array();
    let res = residuals.as_array();
    if res.shape() != obs.shape() {
        return Err(PyValueError::new_err("residuals must match observed shape"));
    }
    if obs.ncols() < 2 {
        return Err(PyValueError::new_err(
            "spherical residuals require at least 2 dimensions",
        ));
    }
    let out: Vec<f64> = obs
        .column(1)
        .iter()
        .zip(res.column(1).iter())
        .map(|(&lon_obs, &lon_resid)| bound_longitude_value(lon_obs, lon_resid))
        .collect();
    Ok(out.into_pyarray(py))
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
    Ok((res_out.into_pyarray(py), cov_out.into_pyarray(py)))
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
        Ok(out) => Ok(out.into_pyarray(py)),
        Err(adam_core_rs_coords::chi2::Chi2Error::NanDiagonal { row, dim }) => {
            let _ = (row, dim);
            Err(PyValueError::new_err(
                "Covariance matrix has NaNs on the diagonal.",
            ))
        }
        Err(adam_core_rs_coords::chi2::Chi2Error::NotPositiveDefinite { row }) => {
            Err(PyValueError::new_err(format!(
                "Covariance matrix at row {row} is not positive definite."
            )))
        }
    }
}

#[pyfunction]
fn compute_residuals_chi2_numpy<'py>(
    py: Python<'py>,
    observed: PyReadonlyArray2<'py, f64>,
    predicted: PyReadonlyArray2<'py, f64>,
    observed_cov: PyReadonlyArray3<'py, f64>,
    predicted_cov: PyReadonlyArray3<'py, f64>,
    is_spherical: bool,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
    bool,
)> {
    let obs = observed.as_array();
    let pred = predicted.as_array();
    let obs_c = observed_cov.as_array();
    let pred_c = predicted_cov.as_array();
    let n = obs.nrows();
    let d = obs.ncols();
    if pred.shape() != [n, d] {
        return Err(PyValueError::new_err(
            "predicted shape must match observed (N, D)",
        ));
    }
    if obs_c.shape() != [n, d, d] {
        return Err(PyValueError::new_err(format!(
            "observed_cov shape {:?} must be (N={n}, D={d}, D={d})",
            obs_c.shape()
        )));
    }
    if pred_c.shape() != [n, d, d] {
        return Err(PyValueError::new_err(format!(
            "predicted_cov shape {:?} must be (N={n}, D={d}, D={d})",
            pred_c.shape()
        )));
    }
    let obs_slice = obs
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("observed must be contiguous"))?;
    let pred_slice = pred
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("predicted must be contiguous"))?;
    let obs_cov_slice = obs_c
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("observed_cov must be contiguous"))?;
    let pred_cov_slice = pred_c
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("predicted_cov must be contiguous"))?;

    match adam_core_rs_coords::compute_residuals_chi2_flat(
        obs_slice,
        pred_slice,
        obs_cov_slice,
        pred_cov_slice,
        n,
        d,
        is_spherical,
    ) {
        Ok(out) => {
            let res_arr = ndarray::Array2::from_shape_vec((n, d), out.residuals)
                .map_err(|e| PyValueError::new_err(format!("shape: {e}")))?;
            let chi2_arr = ndarray::Array1::from_vec(out.chi2);
            let dof_arr = ndarray::Array1::from_vec(out.dof);
            Ok((
                res_arr.into_pyarray(py),
                chi2_arr.into_pyarray(py),
                dof_arr.into_pyarray(py),
                out.had_off_diagonal_nan,
            ))
        }
        Err(adam_core_rs_coords::ResidualsError::Chi2(
            adam_core_rs_coords::chi2::Chi2Error::NanDiagonal { .. },
        )) => Err(PyValueError::new_err(
            "Covariance matrix has NaNs on the diagonal.",
        )),
        Err(adam_core_rs_coords::ResidualsError::Chi2(
            adam_core_rs_coords::chi2::Chi2Error::NotPositiveDefinite { row },
        )) => Err(PyValueError::new_err(format!(
            "Covariance matrix at row {row} is not positive definite."
        ))),
        Err(adam_core_rs_coords::ResidualsError::InvalidShape(msg)) => {
            Err(PyValueError::new_err(msg))
        }
    }
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
    Ok(ndarray::Array1::from_vec(weighted_mean_flat(s_slice, w_slice, n, d)).into_pyarray(py))
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
    Ok(shaped.into_pyarray(py))
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
    Ok(ndarray::Array1::from_vec(codes).into_pyarray(py))
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
    Ok(ndarray::Array1::from_vec(out).into_pyarray(py))
}

fn read_orbit_ipc(bytes: &[u8]) -> PyResult<RecordBatch> {
    let mut reader = StreamReader::try_new(bytes, None)
        .map_err(|err| PyValueError::new_err(format!("invalid Arrow IPC stream: {err}")))?;
    let batch = reader
        .next()
        .ok_or_else(|| PyValueError::new_err("Arrow IPC stream contained no record batches"))?
        .map_err(|err| PyValueError::new_err(format!("failed to read Arrow IPC batch: {err}")))?;
    Ok(batch)
}

fn write_orbit_ipc(batch: &RecordBatch) -> PyResult<Vec<u8>> {
    let mut buffer = Vec::new();
    {
        let schema = batch.schema();
        let mut writer = StreamWriter::try_new(&mut buffer, &schema).map_err(|err| {
            PyValueError::new_err(format!("failed to start Arrow IPC writer: {err}"))
        })?;
        writer.write(batch).map_err(|err| {
            PyValueError::new_err(format!("failed to write Arrow IPC batch: {err}"))
        })?;
        writer.finish().map_err(|err| {
            PyValueError::new_err(format!("failed to finish Arrow IPC stream: {err}"))
        })?;
    }
    Ok(buffer)
}

/// W1 data-model bridge (bead personal-cmy.13, mechanism C): round-trip a quivr
/// `Orbits` table (Arrow IPC bytes in the nested quivr layout) through the
/// Rust-canonical `OrbitBatch` and back. Proves the single-crossing,
/// full-nested-schema bridge is lossless. The caller injects the
/// `adam_core_*` schema metadata (frame, time scale) Rust needs.
#[pyfunction]
fn orbits_nested_ipc_round_trip<'py>(
    py: Python<'py>,
    ipc_bytes: &Bound<'py, PyBytes>,
) -> PyResult<Bound<'py, PyBytes>> {
    let batch = read_orbit_ipc(ipc_bytes.as_bytes())?;
    let orbits = DataOrbitBatch::try_from_nested_record_batch(&batch)
        .map_err(|err| PyValueError::new_err(format!("failed to decode OrbitBatch: {err}")))?;
    let out = orbits
        .into_nested_record_batch()
        .map_err(|err| PyValueError::new_err(format!("failed to encode OrbitBatch: {err}")))?;
    let bytes = write_orbit_ipc(&out)?;
    Ok(PyBytes::new(py, &bytes))
}

/// Map the rotation-kernel `Frame` onto the typed-batch `Frame` used by
/// `OrbitBatch::rotate_frame`.
fn data_frame(frame: Frame) -> DataFrame {
    match frame {
        Frame::Ecliptic => DataFrame::Ecliptic,
        Frame::Equatorial => DataFrame::Equatorial,
        Frame::Itrf93 => DataFrame::Itrf93,
    }
}

/// W1 data-model workflow (bead personal-cmy.13): rotate a quivr `Orbits` table
/// (Arrow IPC, nested quivr layout) into `target_frame` entirely Rust-side in a
/// single crossing -- decode to `OrbitBatch`, rotate coordinates + covariance,
/// re-encode. Demonstrates a real Rust-native workflow over the bridge.
#[pyfunction]
fn orbits_rotate_frame_ipc<'py>(
    py: Python<'py>,
    ipc_bytes: &Bound<'py, PyBytes>,
    target_frame: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let target = data_frame(parse_frame(target_frame)?);
    let batch = read_orbit_ipc(ipc_bytes.as_bytes())?;
    let orbits = DataOrbitBatch::try_from_nested_record_batch(&batch)
        .map_err(|err| PyValueError::new_err(format!("failed to decode OrbitBatch: {err}")))?;
    let rotated = orbits
        .rotate_frame(target)
        .map_err(|err| PyValueError::new_err(format!("failed to rotate frame: {err}")))?;
    let out = rotated
        .into_nested_record_batch()
        .map_err(|err| PyValueError::new_err(format!("failed to encode OrbitBatch: {err}")))?;
    let bytes = write_orbit_ipc(&out)?;
    Ok(PyBytes::new(py, &bytes))
}

fn parse_variant_method(method: &str) -> PyResult<OrbitVariantSamplingMethod> {
    match method {
        "auto" => Ok(OrbitVariantSamplingMethod::Auto),
        "sigma-point" => Ok(OrbitVariantSamplingMethod::SigmaPoint),
        "monte-carlo" => Ok(OrbitVariantSamplingMethod::MonteCarlo),
        other => Err(PyValueError::new_err(format!(
            "variant method must be one of 'auto', 'sigma-point', or 'monte-carlo'; got {other:?}"
        ))),
    }
}

/// W1 data-model workflow (bead personal-cmy.13): sample covariance variants of
/// a quivr `Orbits` table (Arrow IPC) entirely Rust-side in a single crossing --
/// decode to `OrbitBatch`, run `VariantOrbits.create` sampling semantics, and
/// encode the resulting `OrbitVariantBatch` as quivr-`VariantOrbits` IPC.
///
/// Returns `(ipc_bytes, source_orbit_indices)` where
/// `source_orbit_indices[variant_row]` is the original orbit row index.
/// Physical parameters travel inside the IPC payload itself (the canonical
/// `OrbitVariantBatch` carries them since bead personal-cmy.13.2); the source
/// indices remain available for diagnostics and other per-orbit columns.
#[pyfunction]
#[pyo3(signature = (ipc_bytes, method, num_samples=10000, seed=None, alpha=1.0, beta=0.0, kappa=0.0))]
fn orbits_sample_variants_ipc<'py>(
    py: Python<'py>,
    ipc_bytes: &Bound<'py, PyBytes>,
    method: &str,
    num_samples: usize,
    seed: Option<u64>,
    alpha: f64,
    beta: f64,
    kappa: f64,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyArray1<i64>>)> {
    let batch = read_orbit_ipc(ipc_bytes.as_bytes())?;
    let orbits = DataOrbitBatch::try_from_nested_record_batch(&batch)
        .map_err(|err| PyValueError::new_err(format!("failed to decode OrbitBatch: {err}")))?;
    let samples = create_sampled_orbit_variants(
        &orbits,
        parse_variant_method(method)?,
        num_samples,
        seed,
        alpha,
        beta,
        kappa,
    )
    .map_err(|err| PyValueError::new_err(format!("failed to sample variants: {err}")))?;
    let source_indices: Vec<i64> = samples
        .source_orbit_indices
        .iter()
        .map(|&index| index as i64)
        .collect();
    let out = samples
        .variants
        .into_nested_record_batch()
        .map_err(|err| PyValueError::new_err(format!("failed to encode variants: {err}")))?;
    let bytes = write_orbit_ipc(&out)?;
    Ok((PyBytes::new(py, &bytes), source_indices.into_pyarray(py)))
}

/// W1 data-model workflow (bead personal-cmy.13): propagate a quivr `Orbits`
/// table (Arrow IPC) to a shared `target` epoch with 2-body dynamics entirely
/// Rust-side in a single crossing -- decode to `OrbitBatch`, propagate, re-encode.
#[pyfunction]
#[pyo3(signature = (ipc_bytes, target_days, target_nanos, target_scale, max_iter=100, tol=1e-14))]
fn orbits_propagate_2body_ipc<'py>(
    py: Python<'py>,
    ipc_bytes: &Bound<'py, PyBytes>,
    target_days: i64,
    target_nanos: i64,
    target_scale: &str,
    max_iter: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyBytes>> {
    let scale =
        TimeScale::parse(target_scale).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let batch = read_orbit_ipc(ipc_bytes.as_bytes())?;
    let orbits = DataOrbitBatch::try_from_nested_record_batch(&batch)
        .map_err(|err| PyValueError::new_err(format!("failed to decode OrbitBatch: {err}")))?;
    let propagated = orbits
        .propagate_2body_to(Epoch::new(target_days, target_nanos), scale, max_iter, tol)
        .map_err(|err| PyValueError::new_err(format!("failed to propagate: {err}")))?;
    let out = propagated
        .into_nested_record_batch()
        .map_err(|err| PyValueError::new_err(format!("failed to encode OrbitBatch: {err}")))?;
    let bytes = write_orbit_ipc(&out)?;
    Ok(PyBytes::new(py, &bytes))
}

/// Provider-owned time rescaling for the typed propagation adapter: delegates
/// to the ERFA-backed `TimeArray::rescale` service (UTC/TAI/TT/TDB supported;
/// UT1/GPS fail loudly pending provider contracts).
struct ErfaTimeProvider;

impl TimeScaleProvider for ErfaTimeProvider {
    fn rescale(
        &self,
        times: &TimeArray,
        new_scale: TimeScale,
    ) -> adam_core_rs_coords::types::SchemaResult<TimeArray> {
        times.rescale(new_scale)
    }
}

/// W12 typed propagation adapter (bead personal-cmy.15): decode a quivr
/// `Orbits` or `VariantOrbits` table (nested Arrow IPC) to the Rust-canonical
/// `OrbitBatch`/`OrbitVariantBatch`, run the typed `TwoBodyPropagator`
/// `PropagationRequest` pipeline (cross-product epoch policy, optional
/// linearized covariance transport), and re-encode as quivr IPC. Non-TDB
/// orbit epochs and target times rescale through the provider-owned ERFA
/// time service. Returns `(ipc_bytes, per_output_row_valid)`.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    ipc_bytes,
    is_variants,
    target_scale,
    target_days,
    target_nanos,
    covariance=false,
    max_iter=1000,
    tol=1e-14,
    chunk_size=None,
    thread_limit=None
))]
fn orbits_propagate_typed_ipc<'py>(
    py: Python<'py>,
    ipc_bytes: &Bound<'py, PyBytes>,
    is_variants: bool,
    target_scale: &str,
    target_days: Vec<i64>,
    target_nanos: Vec<i64>,
    covariance: bool,
    max_iter: usize,
    tol: f64,
    chunk_size: Option<usize>,
    thread_limit: Option<usize>,
) -> PyResult<(Bound<'py, PyBytes>, Vec<bool>)> {
    let scale =
        TimeScale::parse(target_scale).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let times = TimeArray::from_parts(scale, target_days, target_nanos)
        .map_err(|err| PyValueError::new_err(format!("invalid target times: {err}")))?;
    let options = PropagationOptions {
        chunk_size,
        thread_limit,
        epoch_policy: EpochPolicy::CrossProduct,
        covariance: if covariance {
            CovariancePropagation::Linearized
        } else {
            CovariancePropagation::None
        },
    };
    let propagator = TwoBodyPropagator::new(TwoBodyPropagatorConfig { max_iter, tol })
        .map_err(|err| PyValueError::new_err(format!("invalid propagator config: {err}")))?;
    let batch = read_orbit_ipc(ipc_bytes.as_bytes())?;
    let provider = ErfaTimeProvider;

    let (out, validity) = if is_variants {
        let variants =
            DataOrbitVariantBatch::try_from_nested_record_batch(&batch).map_err(|err| {
                PyValueError::new_err(format!("failed to decode OrbitVariantBatch: {err}"))
            })?;
        let request = PropagationRequest::new_variants(&variants, &times, options)
            .map_err(|err| PyValueError::new_err(format!("invalid propagation request: {err}")))?;
        let result = py
            .allow_threads(|| propagator.propagate(&request, &provider))
            .map_err(|err| PyValueError::new_err(format!("typed propagation failed: {err}")))?;
        let validity: Vec<bool> = (0..result.validity.len())
            .map(|index| result.validity.is_valid(index))
            .collect();
        let variants_out = result.variants.ok_or_else(|| {
            PyValueError::new_err("typed variant propagation returned no variants".to_string())
        })?;
        let out = variants_out
            .into_nested_record_batch()
            .map_err(|err| PyValueError::new_err(format!("failed to encode variants: {err}")))?;
        (out, validity)
    } else {
        let orbits = DataOrbitBatch::try_from_nested_record_batch(&batch)
            .map_err(|err| PyValueError::new_err(format!("failed to decode OrbitBatch: {err}")))?;
        let request = PropagationRequest::new(&orbits, &times, options)
            .map_err(|err| PyValueError::new_err(format!("invalid propagation request: {err}")))?;
        let result = py
            .allow_threads(|| propagator.propagate(&request, &provider))
            .map_err(|err| PyValueError::new_err(format!("typed propagation failed: {err}")))?;
        let validity: Vec<bool> = (0..result.validity.len())
            .map(|index| result.validity.is_valid(index))
            .collect();
        let out = result
            .orbits
            .into_nested_record_batch()
            .map_err(|err| PyValueError::new_err(format!("failed to encode OrbitBatch: {err}")))?;
        (out, validity)
    };
    let bytes = write_orbit_ipc(&out)?;
    Ok((PyBytes::new(py, &bytes), validity))
}

/// W1 data-model bridge (bead personal-cmy.13, mechanism A): zero-copy round-trip
/// of a quivr `Orbits` table through the Rust-canonical `OrbitBatch` using the
/// Arrow C Data Interface (no IPC serialize/deserialize copy). Accepts a pyarrow
/// `RecordBatch` and returns one.
#[pyfunction]
fn orbits_nested_round_trip_arrow<'py>(
    py: Python<'py>,
    batch: &Bound<'py, PyAny>,
) -> PyResult<PyObject> {
    let record_batch = RecordBatch::from_pyarrow_bound(batch)
        .map_err(|err| PyValueError::new_err(format!("invalid pyarrow RecordBatch: {err}")))?;
    let orbits = DataOrbitBatch::try_from_nested_record_batch(&record_batch)
        .map_err(|err| PyValueError::new_err(format!("failed to decode OrbitBatch: {err}")))?;
    let out = orbits
        .into_nested_record_batch()
        .map_err(|err| PyValueError::new_err(format!("failed to encode OrbitBatch: {err}")))?;
    out.to_pyarrow(py)
        .map_err(|err| PyValueError::new_err(format!("failed to export RecordBatch: {err}")))
}

/// W1 data-model bridge (bead personal-cmy.13 / OD slice 2): round-trip a quivr
/// `Observers` table (Arrow IPC, nested quivr layout) through the Rust-canonical
/// `ObserverBatch` and back. Establishes the observers transport for OD.
#[pyfunction]
fn observers_nested_ipc_round_trip<'py>(
    py: Python<'py>,
    ipc_bytes: &Bound<'py, PyBytes>,
) -> PyResult<Bound<'py, PyBytes>> {
    let batch = read_orbit_ipc(ipc_bytes.as_bytes())?;
    let observers = DataObserverBatch::try_from_nested_record_batch(&batch)
        .map_err(|err| PyValueError::new_err(format!("failed to decode ObserverBatch: {err}")))?;
    let out = observers
        .into_nested_record_batch()
        .map_err(|err| PyValueError::new_err(format!("failed to encode ObserverBatch: {err}")))?;
    let bytes = write_orbit_ipc(&out)?;
    Ok(PyBytes::new(py, &bytes))
}

/// Observation data-model bridge (bead personal-cmy.20): round-trip a quivr
/// observation table (`ADESObservations`, `PointSourceDetections`, `Exposures`,
/// `Associations`, `Photometry`, or `SourceCatalog`; Arrow IPC, nested quivr
/// layout) through its Rust-canonical batch and back, dispatching on the
/// `adam_core_schema` metadata key. Proves the observation transport is
/// lossless, including nullable Timestamp columns.
#[pyfunction]
fn observations_nested_ipc_round_trip<'py>(
    py: Python<'py>,
    ipc_bytes: &Bound<'py, PyBytes>,
) -> PyResult<Bound<'py, PyBytes>> {
    let batch = read_orbit_ipc(ipc_bytes.as_bytes())?;
    let out = adam_core_rs_coords::observations::round_trip_nested(&batch).map_err(|err| {
        PyValueError::new_err(format!("observation table round trip failed: {err}"))
    })?;
    let bytes = write_orbit_ipc(&out)?;
    Ok(PyBytes::new(py, &bytes))
}

/// ADES writer (bead personal-cmy.20 slice B): render a quivr
/// `ADESObservations` table (Arrow IPC, nested layout, obsTime already in utc)
/// to the ADES submission format byte-identically to the legacy Python
/// writer. `contexts` maps observatory codes to pre-rendered
/// `ObsContext.to_string()` blocks.
#[pyfunction]
fn ades_to_string_ipc<'py>(
    _py: Python<'py>,
    ipc_bytes: &Bound<'py, PyBytes>,
    contexts: std::collections::HashMap<String, String>,
    seconds_precision: i32,
    columns_precision: std::collections::HashMap<String, i32>,
) -> PyResult<String> {
    let batch = read_orbit_ipc(ipc_bytes.as_bytes())?;
    let observations =
        adam_core_rs_coords::observations::AdesObservationBatch::try_from_nested_record_batch(
            &batch,
        )
        .map_err(|err| {
            PyValueError::new_err(format!("failed to decode AdesObservationBatch: {err}"))
        })?;
    adam_core_rs_coords::ades_to_string(
        &observations,
        &contexts,
        seconds_precision,
        &columns_precision,
    )
    .map_err(|err| match err {
        // Preserve the exact legacy ValueError messages (missing context /
        // missing IDs) for drop-in dispatch from the public Python API.
        adam_core_rs_coords::SchemaError::InvalidRecordBatch(message) => {
            PyValueError::new_err(message)
        }
        other => PyValueError::new_err(other.to_string()),
    })
}

/// ObsContext renderer (bead personal-cmy.26): renders the
/// `dataclasses.asdict` JSON payload of an ObsContext into the legacy
/// `# section` / `! key value` header block.
#[pyfunction]
fn ades_obs_context_to_string(context_json: &str) -> PyResult<String> {
    adam_core_rs_coords::ades_io::obs_context_to_string(context_json).map_err(|err| match err {
        adam_core_rs_coords::SchemaError::InvalidRecordBatch(message) => {
            PyValueError::new_err(message)
        }
        other => PyValueError::new_err(other.to_string()),
    })
}

/// ObsContext metadata-section parser (bead personal-cmy.26): returns a JSON
/// array of `[mpc_code_or_null, context_dict]` pairs mirroring the legacy
/// `_parse_obs_contexts` structure.
#[pyfunction]
fn ades_parse_obs_contexts(ades_string: &str) -> PyResult<String> {
    adam_core_rs_coords::ades_io::ades_parse_obs_contexts(ades_string).map_err(|err| match err {
        adam_core_rs_coords::SchemaError::InvalidRecordBatch(message) => {
            PyValueError::new_err(message)
        }
        other => PyValueError::new_err(other.to_string()),
    })
}

/// OEM KVN writer (bead personal-cmy.28): writes a single-segment CCSDS OEM
/// file byte-identically to the Python `oem` package for adam-core's
/// structures. `covariances` is a list of (days, nanos, frame, 21
/// lower-triangle km values).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn oem_write_kvn<'py>(
    _py: Python<'py>,
    path: &str,
    header_json: &str,
    metadata_json: &str,
    time_scale: &str,
    days: numpy::PyReadonlyArray1<'py, i64>,
    nanos: numpy::PyReadonlyArray1<'py, i64>,
    states_km: numpy::PyReadonlyArray1<'py, f64>,
    covariances: Vec<(i64, i64, String, Vec<f64>)>,
) -> PyResult<()> {
    let scale = adam_core_rs_coords::TimeScale::parse(time_scale).map_err(time_value_error)?;
    let covariances: Vec<adam_core_rs_coords::OemCovarianceRecord> = covariances
        .into_iter()
        .map(|(days, nanos, frame, lower)| {
            let mut lower_triangle = [0.0f64; 21];
            if lower.len() != 21 {
                return Err(PyValueError::new_err(
                    "covariance lower triangle must have 21 values",
                ));
            }
            lower_triangle.copy_from_slice(&lower);
            Ok(adam_core_rs_coords::OemCovarianceRecord {
                days,
                nanos,
                frame,
                lower_triangle,
            })
        })
        .collect::<PyResult<_>>()?;
    adam_core_rs_coords::oem_write_kvn(
        std::path::Path::new(path),
        header_json,
        metadata_json,
        scale,
        days.as_slice()?,
        nanos.as_slice()?,
        states_km.as_slice()?,
        &covariances,
    )
    .map_err(time_value_error)
}

/// OEM KVN parser (bead personal-cmy.28): parse a KVN OEM file into a JSON
/// payload of header/segments with legacy-exact epoch integer splits.
#[pyfunction]
fn oem_parse_kvn(path: &str) -> PyResult<String> {
    adam_core_rs_coords::oem_parse_kvn(std::path::Path::new(path)).map_err(time_value_error)
}

/// OpenSpace Lua/dataclass renderer (bead personal-cmy.28): render a JSON
/// payload built by the thin Python dataclass wrappers byte-identically to the
/// legacy LuaDict implementation.
#[pyfunction]
fn openspace_lua_to_string(payload_json: &str, indent: usize) -> PyResult<String> {
    adam_core_rs_coords::openspace_lua_to_string(payload_json, indent).map_err(time_value_error)
}

/// OpenSpace asset initialization/deinitialization snippet renderer.
#[pyfunction]
fn openspace_create_initialization(assets: Vec<String>) -> String {
    adam_core_rs_coords::openspace_create_initialization(&assets)
}

/// W10 query parser: NEOCC OEF text -> JSON payload.
#[pyfunction]
fn query_neocc_parse_oef(data: &str) -> PyResult<String> {
    adam_core_rs_coords::neocc_parse_oef_json(data).map_err(time_value_error)
}

/// W10 query parser: SBDB JSON payload list -> normalized arrays JSON.
#[pyfunction]
fn query_sbdb_normalize_payloads(ids_json: &str, payloads_json: &str) -> PyResult<String> {
    adam_core_rs_coords::sbdb_normalize_payloads_json(ids_json, payloads_json)
        .map_err(time_value_error)
}

/// W10 query parser: Scout orbit rows -> normalized arrays JSON.
#[pyfunction]
fn query_scout_normalize_orbits(object_id: &str, rows_json: &str) -> PyResult<String> {
    adam_core_rs_coords::scout_normalize_orbits_json(object_id, rows_json).map_err(time_value_error)
}

/// W10 query parser: Horizons vector rows -> normalized arrays JSON.
#[pyfunction]
fn query_horizons_vectors_normalize(rows_json: &str) -> PyResult<String> {
    adam_core_rs_coords::horizons_vectors_normalize_json(rows_json).map_err(time_value_error)
}

/// W10 query parser: Horizons element rows -> normalized arrays JSON.
#[pyfunction]
fn query_horizons_elements_normalize(rows_json: &str, coordinate_type: &str) -> PyResult<String> {
    adam_core_rs_coords::horizons_elements_normalize_json(rows_json, coordinate_type)
        .map_err(time_value_error)
}

/// W10 query parser: Horizons ephemeris rows -> normalized arrays JSON.
#[pyfunction]
fn query_horizons_ephemeris_normalize(rows_json: &str) -> PyResult<String> {
    adam_core_rs_coords::horizons_ephemeris_normalize_json(rows_json).map_err(time_value_error)
}

/// MPC packed-date conversion (bead personal-cmy.26): packed epoch -> ISOT
/// string (TT scale), legacy `_unpack_mpc_date` semantics.
#[pyfunction]
fn unpack_mpc_date_isot(epoch_pf: &str) -> PyResult<String> {
    adam_core_rs_coords::mpc_designations::unpack_mpc_date_isot(epoch_pf)
        .map_err(mpc_designation_error)
}

/// ADES parser (bead personal-cmy.20 slice C): parse the observation blocks
/// of an ADES string into a quivr-compatible `ADESObservations` IPC payload
/// plus the list of unknown column names (for the caller to log). Context
/// metadata parsing stays Python-side.
#[pyfunction]
fn ades_string_to_observations_ipc<'py>(
    py: Python<'py>,
    ades_string: &str,
) -> PyResult<(Bound<'py, PyBytes>, Vec<String>)> {
    let (observations, unknown_columns) =
        adam_core_rs_coords::ades_string_to_observations(ades_string).map_err(|err| match err {
            adam_core_rs_coords::SchemaError::InvalidRecordBatch(message) => {
                PyValueError::new_err(message)
            }
            other => PyValueError::new_err(other.to_string()),
        })?;
    let batch = observations.into_nested_record_batch().map_err(|err| {
        PyValueError::new_err(format!("failed to encode AdesObservationBatch: {err}"))
    })?;
    let bytes = write_orbit_ipc(&batch)?;
    Ok((PyBytes::new(py, &bytes), unknown_columns))
}

fn time_value_error(err: adam_core_rs_coords::SchemaError) -> PyErr {
    match err {
        adam_core_rs_coords::SchemaError::InvalidRecordBatch(message) => {
            PyValueError::new_err(message)
        }
        other => PyValueError::new_err(other.to_string()),
    }
}

fn time_array_from(
    scale: adam_core_rs_coords::TimeScale,
    days: numpy::PyReadonlyArray1<'_, i64>,
    nanos: numpy::PyReadonlyArray1<'_, i64>,
) -> PyResult<adam_core_rs_coords::TimeArray> {
    adam_core_rs_coords::TimeArray::from_parts(
        scale,
        days.as_slice()?.to_vec(),
        nanos.as_slice()?.to_vec(),
    )
    .map_err(time_value_error)
}

fn time_array_out<'py>(
    py: Python<'py>,
    times: adam_core_rs_coords::TimeArray,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let days: Vec<i64> = times.epochs.iter().map(|epoch| epoch.days).collect();
    let nanos: Vec<i64> = times.epochs.iter().map(|epoch| epoch.nanos).collect();
    (PyArray1::from_vec(py, days), PyArray1::from_vec(py, nanos))
}

/// Timestamp op surface (bead personal-cmy.25): thin bindings over the
/// TimeArray arithmetic in adam_core_rs_coords (legacy-fixture gated).
#[pyfunction]
fn timestamp_add_nanos<'py>(
    py: Python<'py>,
    days: numpy::PyReadonlyArray1<'py, i64>,
    nanos: numpy::PyReadonlyArray1<'py, i64>,
    delta: numpy::PyReadonlyArray1<'py, i64>,
    check_range: bool,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    let times = time_array_from(adam_core_rs_coords::TimeScale::Tdb, days, nanos)?;
    let out = times
        .add_nanos_checked(delta.as_slice()?, check_range)
        .map_err(time_value_error)?;
    Ok(time_array_out(py, out))
}

#[pyfunction]
fn timestamp_add_days<'py>(
    py: Python<'py>,
    days: numpy::PyReadonlyArray1<'py, i64>,
    nanos: numpy::PyReadonlyArray1<'py, i64>,
    delta: numpy::PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    let times = time_array_from(adam_core_rs_coords::TimeScale::Tdb, days, nanos)?;
    let out = times
        .add_days(delta.as_slice()?)
        .map_err(time_value_error)?;
    Ok(time_array_out(py, out))
}

#[pyfunction]
fn timestamp_add_fractional_days<'py>(
    py: Python<'py>,
    days: numpy::PyReadonlyArray1<'py, i64>,
    nanos: numpy::PyReadonlyArray1<'py, i64>,
    fractional_days: numpy::PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    let times = time_array_from(adam_core_rs_coords::TimeScale::Tdb, days, nanos)?;
    let out = times
        .add_fractional_days(fractional_days.as_slice()?)
        .map_err(time_value_error)?;
    Ok(time_array_out(py, out))
}

#[pyfunction]
fn timestamp_difference<'py>(
    py: Python<'py>,
    days_a: numpy::PyReadonlyArray1<'py, i64>,
    nanos_a: numpy::PyReadonlyArray1<'py, i64>,
    days_b: numpy::PyReadonlyArray1<'py, i64>,
    nanos_b: numpy::PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    let a = time_array_from(adam_core_rs_coords::TimeScale::Tdb, days_a, nanos_a)?;
    let b = time_array_from(adam_core_rs_coords::TimeScale::Tdb, days_b, nanos_b)?;
    let (days, nanos) = a.difference(&b).map_err(time_value_error)?;
    Ok((PyArray1::from_vec(py, days), PyArray1::from_vec(py, nanos)))
}

#[pyfunction]
fn timestamp_mjd<'py>(
    py: Python<'py>,
    days: numpy::PyReadonlyArray1<'py, i64>,
    nanos: numpy::PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let times = time_array_from(adam_core_rs_coords::TimeScale::Tdb, days, nanos)?;
    Ok(PyArray1::from_vec(py, times.mjd_values()))
}

#[pyfunction]
fn timestamp_from_mjd<'py>(
    py: Python<'py>,
    mjd: numpy::PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    let out = adam_core_rs_coords::TimeArray::from_mjd(
        adam_core_rs_coords::TimeScale::Tdb,
        mjd.as_slice()?,
    )
    .map_err(time_value_error)?;
    Ok(time_array_out(py, out))
}

#[pyfunction]
fn timestamp_rescale<'py>(
    py: Python<'py>,
    days: numpy::PyReadonlyArray1<'py, i64>,
    nanos: numpy::PyReadonlyArray1<'py, i64>,
    from_scale: &str,
    to_scale: &str,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    let from_scale = adam_core_rs_coords::TimeScale::parse(from_scale).map_err(time_value_error)?;
    let to_scale = adam_core_rs_coords::TimeScale::parse(to_scale).map_err(time_value_error)?;
    let times = time_array_from(from_scale, days, nanos)?;
    let out = times.rescale(to_scale).map_err(time_value_error)?;
    Ok(time_array_out(py, out))
}

fn bandpass_value_error(err: adam_core_rs_coords::SchemaError) -> PyErr {
    match err {
        adam_core_rs_coords::SchemaError::InvalidRecordBatch(message) => {
            PyValueError::new_err(message)
        }
        other => PyValueError::new_err(other.to_string()),
    }
}

fn bandpass_data_for(
    data_dir: &str,
) -> PyResult<std::sync::Arc<adam_core_rs_coords::BandpassData>> {
    adam_core_rs_coords::bandpass_data(std::path::Path::new(data_dir)).map_err(bandpass_value_error)
}

/// Bandpass photometry runtime (bead personal-cmy.24): thin bindings over the
/// Rust-native vendored-data load + compute in adam_core_rs_coords::bandpasses.
#[pyfunction]
fn bandpasses_filter_ids(data_dir: &str) -> PyResult<Vec<String>> {
    Ok(bandpass_data_for(data_dir)?.filter_ids.clone())
}

#[pyfunction]
fn bandpasses_get_integrals(
    data_dir: &str,
    template_id: &str,
    filter_ids: Vec<String>,
) -> PyResult<Vec<f64>> {
    bandpass_data_for(data_dir)?
        .get_integrals(template_id, &filter_ids)
        .map_err(bandpass_value_error)
}

#[pyfunction]
fn bandpasses_compute_mix_integrals(
    data_dir: &str,
    weight_c: f64,
    weight_s: f64,
    filter_ids: Vec<String>,
) -> PyResult<Vec<f64>> {
    bandpass_data_for(data_dir)?
        .compute_mix_integrals(weight_c, weight_s, &filter_ids)
        .map_err(bandpass_value_error)
}

#[pyfunction]
#[pyo3(signature = (data_dir, template_id=None, mix=None))]
fn bandpasses_delta_table(
    data_dir: &str,
    template_id: Option<String>,
    mix: Option<(f64, f64)>,
) -> PyResult<Vec<f64>> {
    bandpass_data_for(data_dir)?
        .delta_table(template_id.as_deref(), mix)
        .map_err(bandpass_value_error)
}

#[pyfunction]
#[pyo3(signature = (data_dir, source_filter_id, target_filter_id, template_id=None, mix=None))]
fn bandpasses_delta_mag(
    data_dir: &str,
    source_filter_id: &str,
    target_filter_id: &str,
    template_id: Option<String>,
    mix: Option<(f64, f64)>,
) -> PyResult<f64> {
    bandpass_data_for(data_dir)?
        .delta_mag(
            template_id.as_deref(),
            mix,
            source_filter_id,
            target_filter_id,
        )
        .map_err(bandpass_value_error)
}

#[pyfunction]
#[pyo3(signature = (data_dir, source_filter_id, template_id=None, mix=None, target_filter_ids=None))]
fn bandpasses_color_terms(
    data_dir: &str,
    source_filter_id: &str,
    template_id: Option<String>,
    mix: Option<(f64, f64)>,
    target_filter_ids: Option<Vec<String>>,
) -> PyResult<Vec<(String, f64)>> {
    bandpass_data_for(data_dir)?
        .color_terms(
            template_id.as_deref(),
            mix,
            source_filter_id,
            target_filter_ids.as_deref(),
        )
        .map_err(bandpass_value_error)
}

#[pyfunction]
fn bandpasses_map_to_canonical(
    data_dir: &str,
    observatory_codes: Vec<String>,
    bands: Vec<String>,
    allow_fallback_filters: bool,
) -> PyResult<Vec<String>> {
    bandpass_data_for(data_dir)?
        .map_to_canonical_filter_bands(&observatory_codes, &bands, allow_fallback_filters)
        .map_err(bandpass_value_error)
}

#[pyfunction]
fn bandpasses_assert_filter_ids(data_dir: &str, filter_ids: Vec<String>) -> PyResult<()> {
    bandpass_data_for(data_dir)?
        .assert_filter_ids_have_curves(&filter_ids)
        .map_err(bandpass_value_error)
}

#[pyfunction]
fn bandpasses_register_custom_template(
    template_id: &str,
    wavelength_nm: Vec<f64>,
    reflectance: Vec<f64>,
) -> PyResult<()> {
    adam_core_rs_coords::register_custom_template(template_id, &wavelength_nm, &reflectance)
        .map_err(bandpass_value_error)
}

#[pyfunction]
fn bandpasses_clear_custom_templates() {
    adam_core_rs_coords::clear_custom_templates();
}

fn mpc_designation_error(err: adam_core_rs_coords::MpcDesignationError) -> PyErr {
    use adam_core_rs_coords::MpcDesignationError;
    match err {
        MpcDesignationError::Value(message) => PyValueError::new_err(message),
        MpcDesignationError::Key(key) => pyo3::exceptions::PyKeyError::new_err(key),
        MpcDesignationError::Index => {
            pyo3::exceptions::PyIndexError::new_err("string index out of range")
        }
    }
}

macro_rules! mpc_designation_fn {
    ($name:ident) => {
        /// MPC packed-designation helper (W11): legacy-exact Rust port of
        /// `adam_core.utils.mpc`, including exception types and messages.
        #[pyfunction]
        fn $name(designation: &str) -> PyResult<String> {
            adam_core_rs_coords::mpc_designations::$name(designation).map_err(mpc_designation_error)
        }
    };
}

mpc_designation_fn!(pack_numbered_designation);
mpc_designation_fn!(pack_provisional_designation);
mpc_designation_fn!(pack_survey_designation);
mpc_designation_fn!(pack_mpc_designation);
mpc_designation_fn!(unpack_numbered_designation);
mpc_designation_fn!(unpack_provisional_designation);
mpc_designation_fn!(unpack_survey_designation);
mpc_designation_fn!(unpack_mpc_designation);

/// W1 / OD slice 3: Rust-native OD residual evaluation over the bridge. Given
/// orbits (already at the observation times, 1:1 with observations), the observed
/// astrometry (`SphericalCoordinates`), and the observers, this composes the exact
/// kernels adam_core uses -- `generate_ephemeris_2body_flat6` (the 2-body
/// light-time ephemeris forward model) and `compute_residuals_chi2_flat` (spherical
/// residual + chi2) -- so results match `generate_ephemeris_2body` +
/// `Residuals.calculate`. Returns `(chi2 (N,), residuals (N, 6))`.
#[pyfunction]
#[pyo3(signature = (orbits_ipc, observed_ipc, observers_ipc, lt_tol=1e-10, max_iter=1000, tol=1e-15, stellar_aberration=false, max_lt_iter=10))]
#[allow(clippy::too_many_arguments)]
fn evaluate_residuals_2body_ipc<'py>(
    py: Python<'py>,
    orbits_ipc: &Bound<'py, PyBytes>,
    observed_ipc: &Bound<'py, PyBytes>,
    observers_ipc: &Bound<'py, PyBytes>,
    lt_tol: f64,
    max_iter: usize,
    tol: f64,
    stellar_aberration: bool,
    max_lt_iter: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let orbits =
        DataOrbitBatch::try_from_nested_record_batch(&read_orbit_ipc(orbits_ipc.as_bytes())?)
            .map_err(|err| PyValueError::new_err(format!("failed to decode OrbitBatch: {err}")))?;
    let observed = DataCoordinateBatch::try_from_nested_record_batch(&read_orbit_ipc(
        observed_ipc.as_bytes(),
    )?)
    .map_err(|err| {
        PyValueError::new_err(format!("failed to decode observed coordinates: {err}"))
    })?;
    let observers =
        DataObserverBatch::try_from_nested_record_batch(&read_orbit_ipc(observers_ipc.as_bytes())?)
            .map_err(|err| {
                PyValueError::new_err(format!("failed to decode ObserverBatch: {err}"))
            })?;

    let n = orbits.coordinates.len();
    if observed.len() != n || observers.coordinates.len() != n {
        return Err(PyValueError::new_err(
            "orbits, observed coordinates, and observers must have equal length (1:1)",
        ));
    }
    let orbit_values = orbits
        .coordinates
        .values
        .cartesian()
        .ok_or_else(|| PyValueError::new_err("orbits must be Cartesian"))?;
    let orbit_flat: Vec<f64> = orbit_values.iter().flatten().copied().collect();
    let observer_values = observers
        .coordinates
        .values
        .cartesian()
        .ok_or_else(|| PyValueError::new_err("observer coordinates must be Cartesian"))?;
    let observer_flat: Vec<f64> = observer_values.iter().flatten().copied().collect();
    let mus = orbits
        .coordinates
        .origins
        .origins
        .iter()
        .map(origin_mu_au3_day2)
        .collect::<Result<Vec<f64>, _>>()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    let (predicted, _light_time, _aberrated) = generate_ephemeris_2body_flat6(
        &orbit_flat,
        &observer_flat,
        &mus,
        lt_tol,
        max_iter,
        tol,
        stellar_aberration,
        max_lt_iter,
    );

    let observed_values = observed.values.raw_values();
    let observed_flat: Vec<f64> = observed_values.iter().flatten().copied().collect();
    let observed_cov = observed.covariance.as_ref().ok_or_else(|| {
        PyValueError::new_err("observed coordinates require covariance to compute chi2")
    })?;
    if observed_cov.dimension != 6 {
        return Err(PyValueError::new_err(
            "observed covariance must be 6x6 per row",
        ));
    }
    // The 2-body ephemeris model carries no covariance, matching
    // Residuals.calculate(observed, predicted) where the predicted ephemeris is
    // covariance-free (residual covariance = observed covariance).
    let predicted_cov = vec![0.0_f64; n * 36];
    let output = adam_core_rs_coords::compute_residuals_chi2_flat(
        &observed_flat,
        &predicted,
        &observed_cov.values_row_major,
        &predicted_cov,
        n,
        6,
        true,
    )
    .map_err(|err| PyValueError::new_err(format!("failed to compute residuals: {err:?}")))?;

    let chi2 = ndarray::Array1::from_vec(output.chi2);
    let residuals = ndarray::Array2::from_shape_vec((n, 6), output.residuals)
        .map_err(|err| PyValueError::new_err(format!("failed to shape residuals: {err}")))?;
    Ok((chi2.into_pyarray(py), residuals.into_pyarray(py)))
}

/// W6 / OD slice 4: Rust-native Gauss-Newton least-squares orbit fit over the
/// bridge. Differentially corrects a single orbit (at its epoch) against
/// astrometric observations, reusing the slice-3 evaluate as the inner residual.
/// Inputs are barycentric (SSB / ecliptic). Returns `(state (6,), covariance
/// (6, 6), chi2, iterations, converged)`.
#[pyfunction]
#[pyo3(signature = (orbit_ipc, observed_ipc, observers_ipc, xtol=1e-12, ftol=1e-12, max_iterations=100, lt_tol=1e-10, eph_max_iter=1000, eph_tol=1e-15, stellar_aberration=false, max_lt_iter=10))]
#[allow(clippy::too_many_arguments)]
fn fit_orbit_2body_least_squares_ipc<'py>(
    py: Python<'py>,
    orbit_ipc: &Bound<'py, PyBytes>,
    observed_ipc: &Bound<'py, PyBytes>,
    observers_ipc: &Bound<'py, PyBytes>,
    xtol: f64,
    ftol: f64,
    max_iterations: usize,
    lt_tol: f64,
    eph_max_iter: usize,
    eph_tol: f64,
    stellar_aberration: bool,
    max_lt_iter: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    f64,
    usize,
    bool,
)> {
    let orbits =
        DataOrbitBatch::try_from_nested_record_batch(&read_orbit_ipc(orbit_ipc.as_bytes())?)
            .map_err(|err| PyValueError::new_err(format!("failed to decode OrbitBatch: {err}")))?;
    if orbits.coordinates.len() != 1 {
        return Err(PyValueError::new_err(
            "least-squares fit requires exactly one orbit",
        ));
    }
    let observed = DataCoordinateBatch::try_from_nested_record_batch(&read_orbit_ipc(
        observed_ipc.as_bytes(),
    )?)
    .map_err(|err| {
        PyValueError::new_err(format!("failed to decode observed coordinates: {err}"))
    })?;
    let observers =
        DataObserverBatch::try_from_nested_record_batch(&read_orbit_ipc(observers_ipc.as_bytes())?)
            .map_err(|err| {
                PyValueError::new_err(format!("failed to decode ObserverBatch: {err}"))
            })?;
    let n = observed.len();
    if observers.coordinates.len() != n {
        return Err(PyValueError::new_err(
            "observed coordinates and observers must have equal length",
        ));
    }
    let orbit_values = orbits
        .coordinates
        .values
        .cartesian()
        .ok_or_else(|| PyValueError::new_err("orbit must be Cartesian"))?;
    let mut initial_state = [0.0_f64; 6];
    initial_state.copy_from_slice(&orbit_values[0]);
    let epoch = orbits
        .coordinates
        .times
        .as_ref()
        .ok_or_else(|| PyValueError::new_err("orbit requires an epoch"))?
        .epochs[0];
    let mu = origin_mu_au3_day2(&orbits.coordinates.origins.origins[0])
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let observed_flat: Vec<f64> = observed
        .values
        .raw_values()
        .iter()
        .flatten()
        .copied()
        .collect();
    let observed_cov = observed
        .covariance
        .as_ref()
        .ok_or_else(|| PyValueError::new_err("observed coordinates require covariance to fit"))?;
    let observer_values = observers
        .coordinates
        .values
        .cartesian()
        .ok_or_else(|| PyValueError::new_err("observer coordinates must be Cartesian"))?;
    let observer_flat: Vec<f64> = observer_values.iter().flatten().copied().collect();
    let obs_epochs = &observers
        .coordinates
        .times
        .as_ref()
        .ok_or_else(|| PyValueError::new_err("observers require epochs"))?
        .epochs;
    let config = LeastSquaresConfig {
        xtol,
        ftol,
        max_iterations,
        lt_tol,
        ephemeris_max_iter: eph_max_iter,
        ephemeris_tol: eph_tol,
        stellar_aberration,
        max_lt_iter,
    };
    let fit = fit_orbit_2body_least_squares(
        initial_state,
        epoch,
        mu,
        &observed_flat,
        &observed_cov.values_row_major,
        &observer_flat,
        obs_epochs,
        &config,
    );
    let state = ndarray::Array1::from_vec(fit.state.to_vec());
    let covariance = ndarray::Array2::from_shape_vec((6, 6), fit.covariance.to_vec())
        .map_err(|err| PyValueError::new_err(format!("failed to shape covariance: {err}")))?;
    Ok((
        state.into_pyarray(py),
        covariance.into_pyarray(py),
        fit.chi2,
        fit.iterations,
        fit.converged,
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cartesian_coordinate_schema_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(orbit_schema_metadata, m)?)?;
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
    m.add_function(wrap_pyfunction!(bound_longitude_residuals_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(bound_longitude_residual_column_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(apply_cosine_latitude_correction_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_chi2_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(compute_residuals_chi2_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_mean_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_covariance_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(classify_orbits_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(tisserand_parameter_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(orbits_nested_ipc_round_trip, m)?)?;
    m.add_function(wrap_pyfunction!(orbits_rotate_frame_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(orbits_sample_variants_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(orbits_propagate_typed_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(orbits_propagate_2body_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(orbits_nested_round_trip_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(observers_nested_ipc_round_trip, m)?)?;
    m.add_function(wrap_pyfunction!(observations_nested_ipc_round_trip, m)?)?;
    m.add_function(wrap_pyfunction!(ades_to_string_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(ades_string_to_observations_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(pack_numbered_designation, m)?)?;
    m.add_function(wrap_pyfunction!(pack_provisional_designation, m)?)?;
    m.add_function(wrap_pyfunction!(pack_survey_designation, m)?)?;
    m.add_function(wrap_pyfunction!(pack_mpc_designation, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_numbered_designation, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_provisional_designation, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_survey_designation, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_mpc_designation, m)?)?;
    m.add_function(wrap_pyfunction!(bandpasses_filter_ids, m)?)?;
    m.add_function(wrap_pyfunction!(bandpasses_get_integrals, m)?)?;
    m.add_function(wrap_pyfunction!(bandpasses_compute_mix_integrals, m)?)?;
    m.add_function(wrap_pyfunction!(bandpasses_delta_table, m)?)?;
    m.add_function(wrap_pyfunction!(bandpasses_delta_mag, m)?)?;
    m.add_function(wrap_pyfunction!(bandpasses_color_terms, m)?)?;
    m.add_function(wrap_pyfunction!(bandpasses_map_to_canonical, m)?)?;
    m.add_function(wrap_pyfunction!(bandpasses_assert_filter_ids, m)?)?;
    m.add_function(wrap_pyfunction!(bandpasses_register_custom_template, m)?)?;
    m.add_function(wrap_pyfunction!(bandpasses_clear_custom_templates, m)?)?;
    m.add_function(wrap_pyfunction!(timestamp_add_nanos, m)?)?;
    m.add_function(wrap_pyfunction!(timestamp_add_days, m)?)?;
    m.add_function(wrap_pyfunction!(timestamp_add_fractional_days, m)?)?;
    m.add_function(wrap_pyfunction!(timestamp_difference, m)?)?;
    m.add_function(wrap_pyfunction!(timestamp_mjd, m)?)?;
    m.add_function(wrap_pyfunction!(timestamp_from_mjd, m)?)?;
    m.add_function(wrap_pyfunction!(timestamp_rescale, m)?)?;
    m.add_function(wrap_pyfunction!(ades_obs_context_to_string, m)?)?;
    m.add_function(wrap_pyfunction!(ades_parse_obs_contexts, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_mpc_date_isot, m)?)?;
    m.add_function(wrap_pyfunction!(oem_write_kvn, m)?)?;
    m.add_function(wrap_pyfunction!(oem_parse_kvn, m)?)?;
    m.add_function(wrap_pyfunction!(openspace_lua_to_string, m)?)?;
    m.add_function(wrap_pyfunction!(openspace_create_initialization, m)?)?;
    m.add_function(wrap_pyfunction!(query_neocc_parse_oef, m)?)?;
    m.add_function(wrap_pyfunction!(query_sbdb_normalize_payloads, m)?)?;
    m.add_function(wrap_pyfunction!(query_scout_normalize_orbits, m)?)?;
    m.add_function(wrap_pyfunction!(query_horizons_vectors_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(query_horizons_elements_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(query_horizons_ephemeris_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_residuals_2body_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(fit_orbit_2body_least_squares_ipc, m)?)?;
    Ok(())
}
