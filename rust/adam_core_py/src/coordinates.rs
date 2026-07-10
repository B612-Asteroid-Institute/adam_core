use adam_core_rs_coords::propagation::{
    generate_ephemeris_barycentric, CovariancePropagation, EphemerisFailureCode, EphemerisOptions,
    EphemerisPhotometryOptions, EpochPolicy, PropagationDiagnostics, PropagationFailureCode,
    PropagationOptions, PropagationRequest, Propagator, TwoBodyPropagator, TwoBodyPropagatorConfig,
};
use adam_core_rs_coords::{
    apply_cosine_latitude_correction_flat, bound_longitude_residuals_flat, bound_longitude_value,
    calculate_chi2_flat, cartesian_to_cometary_flat6, cartesian_to_geodetic_flat6,
    cartesian_to_keplerian_flat6, cartesian_to_spherical_flat6, cartesian_to_spherical_row,
    classify_orbits_flat, cometary_to_cartesian_flat6, create_sampled_orbit_variants,
    fit_orbit_2body_least_squares, generate_ephemeris_2body_flat6, keplerian_to_cartesian_flat6,
    origin_mu_au3_day2, rotate_cartesian_time_varying_flat6, rotate_coordinates_to_frame,
    spherical_to_cartesian_flat6, spherical_to_cartesian_row, tisserand_parameter_flat,
    transform_values_flat6, transform_with_covariance_flat6, weighted_covariance_flat,
    weighted_mean_flat, ArrowSchemaExport, CoordinateBatch as DataCoordinateBatch,
    CoordinateRepresentation as DataRepresentation, CoordinateValues, CovarianceBatch,
    CovarianceUnits, DataFrame, Epoch, Frame, IntoNestedRecordBatch, LeastSquaresConfig,
    ObserverBatch as DataObserverBatch, OrbitBatch as DataOrbitBatch,
    OrbitVariantBatch as DataOrbitVariantBatch, OrbitVariantSamplingMethod, OriginArray, OriginId,
    Representation as CoordsRepresentation, TimeArray, TimeScale, TimeScaleProvider,
    TryFromNestedRecordBatch,
};
use adam_core_rs_spice::global_backend;
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use arrow_array::{Array, Int64Array, RecordBatch};
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

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

struct TypedPropagationBatchOutput {
    batch: RecordBatch,
    validity: Vec<bool>,
    diagnostics: PropagationDiagnostics,
}

fn propagate_typed_record_batch(
    batch: &RecordBatch,
    is_variants: bool,
    times: &TimeArray,
    options: PropagationOptions,
    propagator: &TwoBodyPropagator,
) -> Result<TypedPropagationBatchOutput, String> {
    if is_variants {
        let variants = DataOrbitVariantBatch::try_from_nested_record_batch(batch)
            .map_err(|err| format!("failed to decode OrbitVariantBatch: {err}"))?;
        let request = PropagationRequest::new_variants(&variants, times, options)
            .map_err(|err| format!("invalid propagation request: {err}"))?;
        let result = propagator
            .propagate(&request, &ErfaTimeProvider)
            .map_err(|err| format!("typed propagation failed: {err}"))?;
        let validity = (0..result.validity.len())
            .map(|index| result.validity.is_valid(index))
            .collect();
        let diagnostics = result.diagnostics;
        let batch = result
            .variants
            .ok_or_else(|| "typed variant propagation returned no variants".to_string())?
            .into_nested_record_batch()
            .map_err(|err| format!("failed to encode variants: {err}"))?;
        Ok(TypedPropagationBatchOutput {
            batch,
            validity,
            diagnostics,
        })
    } else {
        let orbits = DataOrbitBatch::try_from_nested_record_batch(batch)
            .map_err(|err| format!("failed to decode OrbitBatch: {err}"))?;
        let request = PropagationRequest::new(&orbits, times, options)
            .map_err(|err| format!("invalid propagation request: {err}"))?;
        let result = propagator
            .propagate(&request, &ErfaTimeProvider)
            .map_err(|err| format!("typed propagation failed: {err}"))?;
        let validity = (0..result.validity.len())
            .map(|index| result.validity.is_valid(index))
            .collect();
        let diagnostics = result.diagnostics;
        let batch = result
            .orbits
            .into_nested_record_batch()
            .map_err(|err| format!("failed to encode OrbitBatch: {err}"))?;
        Ok(TypedPropagationBatchOutput {
            batch,
            validity,
            diagnostics,
        })
    }
}

/// W12 typed propagation adapter (bead personal-cmy.15): decode a quivr
/// `Orbits` or `VariantOrbits` table (nested Arrow IPC) to the Rust-canonical
/// typed propagation pipeline. IPC remains only for cross-process/test workflows;
/// the in-process public facade uses the shared RecordBatch core below.
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
    let output = py
        .allow_threads(|| {
            propagate_typed_record_batch(&batch, is_variants, &times, options, &propagator)
        })
        .map_err(PyValueError::new_err)?;
    let bytes = write_orbit_ipc(&output.batch)?;
    Ok((PyBytes::new(py, &bytes), output.validity))
}

fn target_times_from_record_batch(batch: &RecordBatch) -> Result<TimeArray, String> {
    let scale = batch
        .schema()
        .metadata()
        .get("adam_core_time_scale")
        .ok_or_else(|| "target time RecordBatch is missing adam_core_time_scale".to_string())
        .and_then(|value| TimeScale::parse(value).map_err(|err| err.to_string()))?;
    let column = |name: &str| -> Result<&Int64Array, String> {
        batch
            .column_by_name(name)
            .ok_or_else(|| format!("target time RecordBatch is missing {name}"))?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| format!("target time {name} must be Int64"))
    };
    let days = column("days")?;
    let nanos = column("nanos")?;
    if days.len() != nanos.len() {
        return Err("target time days/nanos length mismatch".to_string());
    }
    let mut epochs = Vec::with_capacity(days.len());
    for row in 0..days.len() {
        if days.is_null(row) || nanos.is_null(row) {
            return Err(format!("target time row {row} is null"));
        }
        epochs.push(Epoch::new(days.value(row), nanos.value(row)));
    }
    TimeArray::new(scale, epochs).map_err(|err| err.to_string())
}

fn propagation_failure_reason(code: PropagationFailureCode) -> &'static str {
    match code {
        PropagationFailureCode::NonFiniteInputState => "non_finite_input_state",
        PropagationFailureCode::NonFiniteOutputState => "non_finite_output_state",
        PropagationFailureCode::NonFiniteCovariance => "non_finite_covariance",
        PropagationFailureCode::SolverZeroDerivative => "solver_zero_derivative",
        PropagationFailureCode::SolverMaxIterations => "solver_max_iterations",
        PropagationFailureCode::IntegratorFailure => "integrator_failure",
    }
}

/// Direct Rust implementation behind the Arrow-native public 2-body surface.
/// RecordBatch decode, epoch cross-product propagation, covariance transport,
/// physical-parameter repetition, and output table assembly all happen here.
fn propagate_orbits_record_batch(
    orbit_batch: &RecordBatch,
    target_time_batch: &RecordBatch,
    is_variants: bool,
    covariance: bool,
    max_iter: usize,
    tol: f64,
    chunk_size: Option<usize>,
    thread_limit: Option<usize>,
) -> Result<RecordBatch, String> {
    let target_times = target_times_from_record_batch(target_time_batch)?;
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
        .map_err(|err| format!("invalid propagator config: {err}"))?;
    let output = propagate_typed_record_batch(
        orbit_batch,
        is_variants,
        &target_times,
        options,
        &propagator,
    )?;

    // Preserve the public function's fail-fast contract for non-finite state
    // rows. Other solver diagnostics historically returned the final iterate.
    if let Some(failed) = output.diagnostics.failed_rows().find(|row| {
        matches!(
            row.failure_code,
            Some(
                PropagationFailureCode::NonFiniteInputState
                    | PropagationFailureCode::NonFiniteOutputState
            )
        )
    }) {
        let code = failed
            .failure_code
            .expect("filtered propagation failure has a code");
        return Err(format!(
            "propagation row failure: reason={}; output_row={}; input_orbit_index={}; input_time_index={}",
            propagation_failure_reason(code),
            failed.output_row,
            failed.input_orbit_index,
            failed.input_time_index,
        ));
    }

    Ok(output.batch)
}

/// Generic Arrow-native typed propagation for Orbits or VariantOrbits. Returns
/// the finished nested RecordBatch plus per-output-row validity.
#[pyfunction]
#[pyo3(signature = (orbit_batch, target_time_batch, is_variants=false, covariance=true, max_iter=1000, tol=1e-14, chunk_size=None, thread_limit=None))]
#[allow(clippy::too_many_arguments)]
fn propagate_orbits_typed_arrow<'py>(
    py: Python<'py>,
    orbit_batch: &Bound<'py, PyAny>,
    target_time_batch: &Bound<'py, PyAny>,
    is_variants: bool,
    covariance: bool,
    max_iter: usize,
    tol: f64,
    chunk_size: Option<usize>,
    thread_limit: Option<usize>,
) -> PyResult<(PyObject, Vec<bool>)> {
    let input = RecordBatch::from_pyarrow_bound(orbit_batch)
        .map_err(|err| PyValueError::new_err(format!("invalid orbit RecordBatch: {err}")))?;
    let targets = RecordBatch::from_pyarrow_bound(target_time_batch)
        .map_err(|err| PyValueError::new_err(format!("invalid target time RecordBatch: {err}")))?;
    let times = target_times_from_record_batch(&targets).map_err(PyValueError::new_err)?;
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
    let output = py
        .allow_threads(|| {
            propagate_typed_record_batch(&input, is_variants, &times, options, &propagator)
        })
        .map_err(PyValueError::new_err)?;
    let batch = output.batch.to_pyarrow(py).map_err(|err| {
        PyRuntimeError::new_err(format!("failed to export propagated RecordBatch: {err}"))
    })?;
    Ok((batch, output.validity))
}

/// Arrow-native public 2-body propagation: one Orbits RecordBatch and one
/// target-Timestamp RecordBatch enter Rust; one finished Orbits RecordBatch
/// leaves Rust.
#[pyfunction]
#[pyo3(signature = (orbit_batch, target_time_batch, is_variants=false, covariance=true, max_iter=1000, tol=1e-14, chunk_size=None, thread_limit=None))]
fn propagate_orbits_arrow<'py>(
    py: Python<'py>,
    orbit_batch: &Bound<'py, PyAny>,
    target_time_batch: &Bound<'py, PyAny>,
    is_variants: bool,
    covariance: bool,
    max_iter: usize,
    tol: f64,
    chunk_size: Option<usize>,
    thread_limit: Option<usize>,
) -> PyResult<PyObject> {
    let orbits = RecordBatch::from_pyarrow_bound(orbit_batch)
        .map_err(|err| PyValueError::new_err(format!("invalid Orbits RecordBatch: {err}")))?;
    let targets = RecordBatch::from_pyarrow_bound(target_time_batch)
        .map_err(|err| PyValueError::new_err(format!("invalid target time RecordBatch: {err}")))?;
    let output = py
        .allow_threads(|| {
            propagate_orbits_record_batch(
                &orbits,
                &targets,
                is_variants,
                covariance,
                max_iter,
                tol,
                chunk_size,
                thread_limit,
            )
        })
        .map_err(|err| {
            if err.starts_with("propagation row failure:") {
                PyRuntimeError::new_err(err)
            } else {
                PyValueError::new_err(err)
            }
        })?;
    output
        .to_pyarrow(py)
        .map_err(|err| PyRuntimeError::new_err(format!("failed to export RecordBatch: {err}")))
}

/// Rust-internal timer for the Arrow-native 2-body surface. PyArrow/PyO3
/// conversion happens once before every warmup and timed sample.
#[pyfunction]
#[pyo3(signature = (orbit_batch, target_time_batch, reps, trials, warmup_reps=1, max_iter=1000, tol=1e-14, chunk_size=None, thread_limit=None))]
#[allow(clippy::too_many_arguments)]
fn benchmark_propagate_orbits_arrow(
    orbit_batch: &Bound<'_, PyAny>,
    target_time_batch: &Bound<'_, PyAny>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
    max_iter: usize,
    tol: f64,
    chunk_size: Option<usize>,
    thread_limit: Option<usize>,
) -> PyResult<Vec<Vec<f64>>> {
    if reps == 0 || trials == 0 {
        return Err(PyValueError::new_err("reps and trials must be >= 1"));
    }
    let orbits = RecordBatch::from_pyarrow_bound(orbit_batch)
        .map_err(|err| PyValueError::new_err(format!("invalid Orbits RecordBatch: {err}")))?;
    let targets = RecordBatch::from_pyarrow_bound(target_time_batch)
        .map_err(|err| PyValueError::new_err(format!("invalid target time RecordBatch: {err}")))?;
    let run_once = || -> PyResult<()> {
        let output = propagate_orbits_record_batch(
            &orbits,
            &targets,
            false,
            true,
            max_iter,
            tol,
            chunk_size,
            thread_limit,
        )
        .map_err(PyRuntimeError::new_err)?;
        black_box(output);
        Ok(())
    };
    let mut sample_trials = Vec::with_capacity(trials);
    for _ in 0..trials {
        for _ in 0..warmup_reps {
            run_once()?;
        }
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let started = Instant::now();
            run_once()?;
            samples.push(started.elapsed().as_secs_f64());
        }
        sample_trials.push(samples);
    }
    Ok(sample_trials)
}

fn ephemeris_failure_reason(code: EphemerisFailureCode) -> &'static str {
    match code {
        EphemerisFailureCode::PropagationRowFailure => "propagation_row_failure",
        EphemerisFailureCode::NonFiniteObserverState => "non_finite_observer_state",
        EphemerisFailureCode::LightTimeNonConvergence => "non_finite_light_time",
        EphemerisFailureCode::NonFiniteEphemerisState => "non_finite_ephemeris_state",
        EphemerisFailureCode::NonFiniteAberratedState => "non_finite_aberrated_state",
    }
}

fn generate_ephemeris_record_batch(
    orbit_batch: &RecordBatch,
    observer_batch: &RecordBatch,
    lt_tol: f64,
    max_iter: usize,
    tol: f64,
    stellar_aberration: bool,
    predict_magnitudes: bool,
    predict_phase_angle: bool,
    chunk_size: Option<usize>,
    thread_limit: Option<usize>,
) -> Result<RecordBatch, String> {
    let mut orbits = DataOrbitBatch::try_from_nested_record_batch(orbit_batch)
        .map_err(|err| format!("failed to decode OrbitBatch: {err}"))?;
    let mut observers = DataObserverBatch::try_from_nested_record_batch(observer_batch)
        .map_err(|err| format!("failed to decode ObserverBatch: {err}"))?;
    if orbits.len() != observers.len() {
        return Err(format!(
            "orbits and observers must be pairwise: orbit_rows={}; observer_rows={}",
            orbits.len(),
            observers.len()
        ));
    }
    orbits.coordinates = rotate_coordinates_to_frame(&orbits.coordinates, DataFrame::Ecliptic)
        .map_err(|err| format!("failed to rotate orbit coordinates: {err}"))?;
    observers.coordinates =
        rotate_coordinates_to_frame(&observers.coordinates, DataFrame::Ecliptic)
            .map_err(|err| format!("failed to rotate observer coordinates: {err}"))?;
    orbits
        .validate()
        .map_err(|err| format!("invalid rotated OrbitBatch: {err}"))?;
    observers
        .validate()
        .map_err(|err| format!("invalid rotated ObserverBatch: {err}"))?;

    let output_time_scale = orbits
        .coordinates
        .times
        .as_ref()
        .ok_or_else(|| "orbit coordinates require times".to_string())?
        .scale;
    let compute_magnitudes = predict_magnitudes
        && orbits
            .physical_parameters
            .as_ref()
            .is_some_and(|parameters| {
                parameters
                    .h_v
                    .iter()
                    .zip(parameters.g.iter())
                    .any(|(h_v, g)| {
                        h_v.is_some_and(f64::is_finite) && g.is_some_and(f64::is_finite)
                    })
            });
    let (h_v, g) = if compute_magnitudes {
        let parameters = orbits
            .physical_parameters
            .as_ref()
            .expect("magnitude parameters were checked");
        (Some(parameters.h_v.clone()), Some(parameters.g.clone()))
    } else {
        (None, None)
    };
    let covariance = if orbits.coordinates.covariance.is_some() {
        CovariancePropagation::Linearized
    } else {
        CovariancePropagation::None
    };
    let options = EphemerisOptions {
        propagation: PropagationOptions {
            chunk_size,
            thread_limit,
            epoch_policy: EpochPolicy::Pairwise,
            covariance,
        },
        lt_tol,
        max_iter,
        tol,
        stellar_aberration,
        max_lt_iter: 10,
        output_time_scale,
        include_aberrated_coordinates: true,
        photometry: EphemerisPhotometryOptions {
            predict_magnitude_v: compute_magnitudes,
            predict_phase_angle,
            h_v,
            g,
        },
    };
    let propagator = TwoBodyPropagator::new(TwoBodyPropagatorConfig { max_iter, tol })
        .map_err(|err| format!("invalid propagator config: {err}"))?;
    let backend = global_backend()
        .lock()
        .map_err(|_| "SPICE backend lock is poisoned".to_string())?;
    let result = generate_ephemeris_barycentric(
        &propagator,
        &orbits,
        &observers,
        &options,
        &ErfaTimeProvider,
        &*backend,
    )
    .map_err(|err| format!("typed ephemeris generation failed: {err}"))?;
    drop(backend);

    if let Some(failed) = result.diagnostics.failed_rows().next() {
        return Err(format!(
            "ephemeris row failure: reason={}; output_row={}; input_orbit_index={}; observer_index={}",
            ephemeris_failure_reason(failed.failure_code.expect("failed row has a code")),
            failed.output_row,
            failed.input_orbit_index,
            failed.observer_index,
        ));
    }
    result
        .ephemeris
        .into_nested_record_batch()
        .map_err(|err| format!("failed to encode EphemerisBatch: {err}"))
}

/// Arrow-native public two-body ephemeris generation.
#[pyfunction]
#[pyo3(signature = (orbit_batch, observer_batch, lt_tol=1e-10, max_iter=1000, tol=1e-15, stellar_aberration=false, predict_magnitudes=true, predict_phase_angle=false, chunk_size=None, thread_limit=None))]
#[allow(clippy::too_many_arguments)]
fn generate_ephemeris_arrow<'py>(
    py: Python<'py>,
    orbit_batch: &Bound<'py, PyAny>,
    observer_batch: &Bound<'py, PyAny>,
    lt_tol: f64,
    max_iter: usize,
    tol: f64,
    stellar_aberration: bool,
    predict_magnitudes: bool,
    predict_phase_angle: bool,
    chunk_size: Option<usize>,
    thread_limit: Option<usize>,
) -> PyResult<PyObject> {
    let orbits = RecordBatch::from_pyarrow_bound(orbit_batch)
        .map_err(|err| PyValueError::new_err(format!("invalid Orbits RecordBatch: {err}")))?;
    let observers = RecordBatch::from_pyarrow_bound(observer_batch)
        .map_err(|err| PyValueError::new_err(format!("invalid Observers RecordBatch: {err}")))?;
    let output = py
        .allow_threads(|| {
            generate_ephemeris_record_batch(
                &orbits,
                &observers,
                lt_tol,
                max_iter,
                tol,
                stellar_aberration,
                predict_magnitudes,
                predict_phase_angle,
                chunk_size,
                thread_limit,
            )
        })
        .map_err(|err| {
            if err.starts_with("ephemeris row failure:") {
                PyRuntimeError::new_err(err)
            } else {
                PyValueError::new_err(err)
            }
        })?;
    output
        .to_pyarrow(py)
        .map_err(|err| PyRuntimeError::new_err(format!("failed to export RecordBatch: {err}")))
}

/// Rust-owned Instant timer for Arrow-native ephemeris generation.
#[pyfunction]
#[pyo3(signature = (orbit_batch, observer_batch, reps, trials, warmup_reps=1, lt_tol=1e-10, max_iter=1000, tol=1e-15, stellar_aberration=false, predict_magnitudes=true, predict_phase_angle=false, chunk_size=None, thread_limit=None))]
#[allow(clippy::too_many_arguments)]
fn benchmark_generate_ephemeris_arrow(
    orbit_batch: &Bound<'_, PyAny>,
    observer_batch: &Bound<'_, PyAny>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
    lt_tol: f64,
    max_iter: usize,
    tol: f64,
    stellar_aberration: bool,
    predict_magnitudes: bool,
    predict_phase_angle: bool,
    chunk_size: Option<usize>,
    thread_limit: Option<usize>,
) -> PyResult<Vec<Vec<f64>>> {
    if reps == 0 || trials == 0 {
        return Err(PyValueError::new_err("reps and trials must be >= 1"));
    }
    let orbits = RecordBatch::from_pyarrow_bound(orbit_batch)
        .map_err(|err| PyValueError::new_err(format!("invalid Orbits RecordBatch: {err}")))?;
    let observers = RecordBatch::from_pyarrow_bound(observer_batch)
        .map_err(|err| PyValueError::new_err(format!("invalid Observers RecordBatch: {err}")))?;
    let run_once = || -> PyResult<()> {
        let output = generate_ephemeris_record_batch(
            &orbits,
            &observers,
            lt_tol,
            max_iter,
            tol,
            stellar_aberration,
            predict_magnitudes,
            predict_phase_angle,
            chunk_size,
            thread_limit,
        )
        .map_err(PyRuntimeError::new_err)?;
        black_box(output);
        Ok(())
    };
    let mut sample_trials = Vec::with_capacity(trials);
    for _ in 0..trials {
        for _ in 0..warmup_reps {
            run_once()?;
        }
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let started = Instant::now();
            run_once()?;
            samples.push(started.elapsed().as_secs_f64());
        }
        sample_trials.push(samples);
    }
    Ok(sample_trials)
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

#[derive(Clone)]
struct TransformArrowConfig {
    representation_out: DataRepresentation,
    frame_out: DataFrame,
    target_origin: Option<OriginId>,
    a: f64,
    f: f64,
    max_iter: usize,
    tol: f64,
}

fn local_representation(rep: DataRepresentation) -> Representation {
    match rep {
        DataRepresentation::Cartesian => Representation::Cartesian,
        DataRepresentation::Spherical => Representation::Spherical,
        DataRepresentation::Geodetic => Representation::Geodetic,
        DataRepresentation::Keplerian => Representation::Keplerian,
        DataRepresentation::Cometary => Representation::Cometary,
    }
}

fn data_representation(rep: Representation) -> DataRepresentation {
    match rep {
        Representation::Cartesian => DataRepresentation::Cartesian,
        Representation::Spherical => DataRepresentation::Spherical,
        Representation::Geodetic => DataRepresentation::Geodetic,
        Representation::Keplerian => DataRepresentation::Keplerian,
        Representation::Cometary => DataRepresentation::Cometary,
    }
}

fn parse_data_frame(value: &str) -> PyResult<DataFrame> {
    match value {
        "ecliptic" => Ok(DataFrame::Ecliptic),
        "equatorial" => Ok(DataFrame::Equatorial),
        "itrf93" => Ok(DataFrame::Itrf93),
        other => Err(PyValueError::new_err(format!("unsupported frame: {other}"))),
    }
}

fn transformed_coordinate_values(
    flat: Vec<f64>,
    ncols: usize,
    rows: usize,
    representation: DataRepresentation,
) -> PyResult<CoordinateValues> {
    let values = if ncols == 6 {
        if flat.len() != rows * 6 {
            return Err(PyRuntimeError::new_err(format!(
                "coordinate transform returned {} values for {rows} rows",
                flat.len()
            )));
        }
        flat.chunks_exact(6)
            .map(|row| [row[0], row[1], row[2], row[3], row[4], row[5]])
            .collect()
    } else if representation == DataRepresentation::Keplerian && ncols == 13 {
        if flat.len() != rows * 13 {
            return Err(PyRuntimeError::new_err(format!(
                "Keplerian transform returned {} values for {rows} rows",
                flat.len()
            )));
        }
        flat.chunks_exact(13)
            .map(|row| [row[0], row[4], row[5], row[6], row[7], row[8]])
            .collect()
    } else {
        return Err(PyRuntimeError::new_err(format!(
            "unsupported coordinate transform output width {ncols}"
        )));
    };

    Ok(match representation {
        DataRepresentation::Cartesian => CoordinateValues::Cartesian(values),
        DataRepresentation::Spherical => CoordinateValues::Spherical(values),
        DataRepresentation::Keplerian => CoordinateValues::Keplerian(values),
        DataRepresentation::Cometary => CoordinateValues::Cometary(values),
        DataRepresentation::Geodetic => CoordinateValues::Geodetic(values),
    })
}

/// Decode one quivr-compatible coordinate RecordBatch, run the complete native
/// transform, and assemble the output RecordBatch without crossing Python.
fn transform_coordinates_record_batch(
    record_batch: &RecordBatch,
    config: &TransformArrowConfig,
) -> PyResult<Option<RecordBatch>> {
    let coordinates = DataCoordinateBatch::try_from_nested_record_batch(record_batch)
        .map_err(|err| PyValueError::new_err(format!("failed to decode coordinates: {err}")))?;
    let rows = coordinates.len();
    let representation_in = coordinates.representation();
    if representation_in == DataRepresentation::Geodetic {
        return Ok(None);
    }
    let times = coordinates.times.as_ref().ok_or_else(|| {
        PyValueError::new_err("coordinate transform requires non-null coordinate times")
    })?;
    let local_in = local_representation(representation_in);
    let local_out = local_representation(config.representation_out);
    let needs_mu = covariance_transform_requires_mu(local_in, local_out);
    let needs_t0 = covariance_transform_requires_t0(local_in, local_out);
    let mu = if needs_mu {
        coordinates
            .origins
            .origins
            .iter()
            .map(origin_mu_au3_day2)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| PyValueError::new_err(format!("failed to resolve origin mu: {err}")))?
    } else {
        vec![0.0; rows]
    };
    let t0 = if needs_t0 {
        times.mjd_values()
    } else {
        vec![0.0; rows]
    };
    let values_flat = coordinates
        .values
        .raw_values()
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    let covariance_flat = coordinates
        .covariance
        .as_ref()
        .map(|covariance| covariance.values_row_major.as_slice());

    let backend = global_backend()
        .lock()
        .map_err(|_| PyRuntimeError::new_err("SPICE backend lock is poisoned"))?;
    let output = backend
        .transform_coordinates(
            &values_flat,
            covariance_flat,
            to_coords_rep(local_in),
            to_coords_rep(local_out),
            coordinates.frame,
            config.frame_out,
            &coordinates.origins,
            config.target_origin.as_ref(),
            times,
            &t0,
            &mu,
            config.a,
            config.f,
            config.max_iter,
            config.tol,
        )
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    drop(backend);
    let Some(output) = output else {
        return Ok(None);
    };

    let values = transformed_coordinate_values(
        output.values,
        output.ncols,
        rows,
        config.representation_out,
    )?;
    let covariance = match output.covariance {
        Some(values) => {
            let mut covariance = CovarianceBatch::new(
                rows,
                6,
                values,
                CovarianceUnits::Coordinate(config.representation_out),
            )
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
            if let Some(validity) = coordinates
                .covariance
                .as_ref()
                .and_then(|input| input.row_validity.clone())
            {
                covariance = covariance
                    .with_row_validity(validity)
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
            }
            Some(covariance)
        }
        None => None,
    };
    let origins = config.target_origin.as_ref().map_or_else(
        || coordinates.origins.clone(),
        |target| OriginArray::repeat(target.clone(), rows),
    );
    let output = DataCoordinateBatch::new(
        values,
        config.frame_out,
        origins,
        coordinates.times.clone(),
        covariance,
    )
    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
    .into_nested_record_batch()
    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    Ok(Some(output))
}

/// Arrow-native `transform_coordinates`: one RecordBatch enters Rust and one
/// transformed RecordBatch leaves Rust. PyArrow conversion is outside the
/// Rust-owned implementation.
#[pyfunction]
#[pyo3(signature = (batch, representation_out, frame_out, target_origin=None, a=0.0, f=0.0, max_iter=100, tol=1e-15))]
fn transform_coordinates_arrow<'py>(
    py: Python<'py>,
    batch: &Bound<'py, PyAny>,
    representation_out: &str,
    frame_out: &str,
    target_origin: Option<&str>,
    a: f64,
    f: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<Option<PyObject>> {
    let record_batch = RecordBatch::from_pyarrow_bound(batch)
        .map_err(|err| PyValueError::new_err(format!("invalid pyarrow RecordBatch: {err}")))?;
    let config = TransformArrowConfig {
        representation_out: data_representation(parse_representation(representation_out)?),
        frame_out: parse_data_frame(frame_out)?,
        target_origin: target_origin.map(OriginId::from_code),
        a,
        f,
        max_iter,
        tol,
    };
    transform_coordinates_record_batch(&record_batch, &config)?
        .map(|output| {
            output.to_pyarrow(py).map_err(|err| {
                PyRuntimeError::new_err(format!("failed to export RecordBatch: {err}"))
            })
        })
        .transpose()
}

/// Benchmark the Arrow-native transform entirely inside Rust. PyArrow/PyO3
/// conversion and typed option parsing happen once before warmups and samples.
#[pyfunction]
#[pyo3(signature = (batches, representations_out, frames_out, target_origins, axes, flattenings, reps, trials, warmup_reps=1, max_iter=100, tol=1e-15))]
#[allow(clippy::too_many_arguments)]
fn benchmark_transform_coordinates_arrow(
    batches: &Bound<'_, PyList>,
    representations_out: Vec<String>,
    frames_out: Vec<String>,
    target_origins: Vec<Option<String>>,
    axes: Vec<f64>,
    flattenings: Vec<f64>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<Vec<Vec<f64>>> {
    if reps == 0 || trials == 0 {
        return Err(PyValueError::new_err("reps and trials must be >= 1"));
    }
    let record_batches = batches
        .iter()
        .map(|batch| {
            RecordBatch::from_pyarrow_bound(&batch)
                .map_err(|err| PyValueError::new_err(format!("invalid pyarrow RecordBatch: {err}")))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let case_count = record_batches.len();
    if [
        representations_out.len(),
        frames_out.len(),
        target_origins.len(),
        axes.len(),
        flattenings.len(),
    ]
    .into_iter()
    .any(|len| len != case_count)
    {
        return Err(PyValueError::new_err(
            "all transform benchmark option lists must match batches length",
        ));
    }
    let configs = representations_out
        .iter()
        .zip(frames_out.iter())
        .zip(target_origins.iter())
        .zip(axes.iter().copied())
        .zip(flattenings.iter().copied())
        .map(
            |((((representation_out, frame_out), target_origin), a), f)| {
                Ok(TransformArrowConfig {
                    representation_out: data_representation(parse_representation(
                        representation_out,
                    )?),
                    frame_out: parse_data_frame(frame_out)?,
                    target_origin: target_origin.as_deref().map(OriginId::from_code),
                    a,
                    f,
                    max_iter,
                    tol,
                })
            },
        )
        .collect::<PyResult<Vec<_>>>()?;
    let run_once = || -> PyResult<()> {
        for (batch, config) in record_batches.iter().zip(configs.iter()) {
            let output = transform_coordinates_record_batch(batch, config)?
                .ok_or_else(|| PyRuntimeError::new_err("native transform case fell back"))?;
            black_box(output);
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
            let started = Instant::now();
            run_once()?;
            samples.push(started.elapsed().as_secs_f64());
        }
        sample_trials.push(samples);
    }
    Ok(sample_trials)
}

/// Single-crossing perturber-MOID orchestrator over the global SPICE backend:
/// per perturber, spkez the perturber state relative to `origin_code` and run
/// the batched Rust MOID kernel against the primary Cartesian orbits. Returns
/// `(moids, dt_mins)` laid out perturber-major, orbit-minor (`p * n + i`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn calculate_perturber_moids_native<'py>(
    py: Python<'py>,
    primary: PyReadonlyArray2<'py, f64>,
    mus: PyReadonlyArray1<'py, f64>,
    time_scale: &str,
    time_days: PyReadonlyArray1<'py, i64>,
    time_nanos: PyReadonlyArray1<'py, i64>,
    perturber_codes: Vec<String>,
    frame: &str,
    origin_code: &str,
    max_iter: usize,
    xtol: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let arr = primary.as_array();
    if arr.ncols() != 6 {
        return Err(PyValueError::new_err("primary must have shape (N, 6)"));
    }
    let n = arr.nrows();
    let primary_flat = arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("primary must be contiguous"))?;

    let mus_view = mus.as_array();
    if mus_view.len() != n {
        return Err(PyValueError::new_err("mus must have length N"));
    }
    let mus_slice = mus_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("mus must be contiguous"))?;

    let frame_value = match frame {
        "ecliptic" => DataFrame::Ecliptic,
        "equatorial" => DataFrame::Equatorial,
        "itrf93" => DataFrame::Itrf93,
        other => return Err(PyValueError::new_err(format!("unsupported frame: {other}"))),
    };
    let origin = OriginId::from_code(origin_code);
    let perturbers: Vec<OriginId> = perturber_codes
        .iter()
        .map(|code| OriginId::from_code(code.clone()))
        .collect();

    let scale =
        TimeScale::parse(time_scale).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let days = time_days.as_slice()?;
    let nanos = time_nanos.as_slice()?;
    if days.len() != n || nanos.len() != n {
        return Err(PyValueError::new_err("times must have length N"));
    }
    let times = TimeArray::from_parts(scale, days.to_vec(), nanos.to_vec())
        .map_err(|err| PyValueError::new_err(format!("invalid times: {err}")))?;

    let backend = global_backend()
        .lock()
        .map_err(|_| PyRuntimeError::new_err("SPICE backend lock is poisoned"))?;
    let (moids, dt_mins) = backend
        .calculate_perturber_moids(
            primary_flat,
            mus_slice,
            &times,
            &perturbers,
            frame_value,
            &origin,
            max_iter,
            xtol,
        )
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    Ok((moids.into_pyarray(py), dt_mins.into_pyarray(py)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cartesian_coordinate_schema_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(transform_coordinates_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_transform_coordinates_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_perturber_moids_native, m)?)?;
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
    m.add_function(wrap_pyfunction!(propagate_orbits_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_orbits_typed_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_propagate_orbits_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(generate_ephemeris_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_generate_ephemeris_arrow, m)?)?;
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
