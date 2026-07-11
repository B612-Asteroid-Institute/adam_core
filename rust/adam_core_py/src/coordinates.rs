use adam_core_rs_coords::propagation::{
    generate_ephemeris_barycentric, CovariancePropagation, EphemerisFailureCode, EphemerisOptions,
    EphemerisPhotometryOptions, EpochPolicy, PropagationDiagnostics, PropagationFailureCode,
    PropagationOptions, PropagationRequest, Propagator, TwoBodyPropagator, TwoBodyPropagatorConfig,
};
use adam_core_rs_coords::{
    apply_cosine_latitude_correction_flat, bound_longitude_residuals_flat, bound_longitude_value,
    calculate_chi2_flat, cartesian_to_cometary_flat6, cartesian_to_geodetic_flat6,
    cartesian_to_keplerian_flat6, cartesian_to_spherical_flat6, cartesian_to_spherical_row,
    chi2_survival, classify_orbits_flat, cometary_to_cartesian_flat6, compute_residuals_chi2_flat,
    create_sampled_orbit_variants, fit_orbit_2body_least_squares, generate_ephemeris_2body_flat6,
    keplerian_to_cartesian_flat6, origin_mu_au3_day2, porkchop_grid_flat,
    rotate_cartesian_time_varying_flat6, rotate_coordinates_to_frame, spherical_to_cartesian_flat6,
    spherical_to_cartesian_row, tisserand_parameter_flat, transform_values_flat6,
    transform_with_covariance_flat6, weighted_covariance_flat, weighted_mean_flat,
    ArrowSchemaExport, CoordinateBatch as DataCoordinateBatch,
    CoordinateRepresentation as DataRepresentation, CoordinateValues, CovarianceBatch,
    CovarianceUnits, DataFrame, Epoch, Frame, IntoNestedRecordBatch, LeastSquaresConfig,
    ObserverBatch as DataObserverBatch, OrbitBatch as DataOrbitBatch,
    OrbitVariantBatch as DataOrbitVariantBatch, OrbitVariantSamplingMethod, OriginArray, OriginId,
    Representation as CoordsRepresentation, TimeArray, TimeScale, TimeScaleProvider,
    TryFromNestedRecordBatch,
};
use adam_core_rs_spice::global_backend;
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use arrow_array::{
    Array, Float64Array, Int64Array, LargeStringArray, RecordBatch, StructArray, UInt32Array,
};
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyReadwriteArray2,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use std::collections::{HashMap, HashSet};
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

/// In-place longitude wrap used by the public compatibility helper.
///
/// Mutating the caller-owned residual buffer preserves the legacy aliasing
/// contract while avoiding both an output allocation and a second strided
/// Python/NumPy assignment over the longitude column.
#[pyfunction]
fn bound_longitude_residual_column_in_place_numpy(
    observed: PyReadonlyArray2<'_, f64>,
    mut residuals: PyReadwriteArray2<'_, f64>,
) -> PyResult<()> {
    let obs = observed.as_array();
    let mut res = residuals.as_array_mut();
    if res.shape() != obs.shape() {
        return Err(PyValueError::new_err("residuals must match observed shape"));
    }
    if obs.ncols() < 2 {
        return Err(PyValueError::new_err(
            "spherical residuals require at least 2 dimensions",
        ));
    }
    let dimensions = obs.ncols();
    if let (Some(obs_slice), Some(res_slice)) = (obs.as_slice(), res.as_slice_mut()) {
        for (obs_row, res_row) in obs_slice
            .chunks_exact(dimensions)
            .zip(res_slice.chunks_exact_mut(dimensions))
        {
            res_row[1] = bound_longitude_value(obs_row[1], res_row[1]);
        }
    } else {
        for row in 0..obs.nrows() {
            res[[row, 1]] = bound_longitude_value(obs[[row, 1]], res[[row, 1]]);
        }
    }
    Ok(())
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

/// Arrow-native covariance-variant sampler behind `VariantOrbits.create`:
/// one Orbits RecordBatch enters Rust; the finished VariantOrbits nested
/// RecordBatch leaves Rust (sigma-point deterministic; Monte Carlo/auto use
/// the Rust-native RNG per decision 2026-07-03).
fn sample_orbit_variants_record_batch(
    orbit_batch: &RecordBatch,
    method: &str,
    num_samples: usize,
    seed: Option<u64>,
    alpha: f64,
    beta: f64,
    kappa: f64,
) -> Result<RecordBatch, String> {
    let orbits = DataOrbitBatch::try_from_nested_record_batch(orbit_batch)
        .map_err(|err| format!("failed to decode OrbitBatch: {err}"))?;
    let samples = create_sampled_orbit_variants(
        &orbits,
        parse_variant_method(method).map_err(|err| err.to_string())?,
        num_samples,
        seed,
        alpha,
        beta,
        kappa,
    )
    .map_err(|err| format!("failed to sample variants: {err}"))?;
    samples
        .variants
        .into_nested_record_batch()
        .map_err(|err| format!("failed to encode variants: {err}"))
}

/// Arrow-native public `VariantOrbits.create` surface.
#[pyfunction]
#[pyo3(signature = (orbit_batch, method, num_samples=10000, seed=None, alpha=1.0, beta=0.0, kappa=0.0))]
fn sample_orbit_variants_arrow<'py>(
    py: Python<'py>,
    orbit_batch: &Bound<'py, PyAny>,
    method: &str,
    num_samples: usize,
    seed: Option<u64>,
    alpha: f64,
    beta: f64,
    kappa: f64,
) -> PyResult<PyObject> {
    let orbits = RecordBatch::from_pyarrow_bound(orbit_batch)
        .map_err(|err| PyValueError::new_err(format!("invalid Orbits RecordBatch: {err}")))?;
    let output = py
        .allow_threads(|| {
            sample_orbit_variants_record_batch(
                &orbits,
                method,
                num_samples,
                seed,
                alpha,
                beta,
                kappa,
            )
        })
        .map_err(PyValueError::new_err)?;
    output
        .to_pyarrow(py)
        .map_err(|err| PyRuntimeError::new_err(format!("failed to export RecordBatch: {err}")))
}

/// Rust-owned Instant timer for the Arrow-native variant sampler.
#[pyfunction]
#[pyo3(signature = (orbit_batch, method, reps, trials, warmup_reps=1, num_samples=10000, seed=None, alpha=1.0, beta=0.0, kappa=0.0))]
#[allow(clippy::too_many_arguments)]
fn benchmark_sample_orbit_variants_arrow(
    orbit_batch: &Bound<'_, PyAny>,
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
    if reps == 0 || trials == 0 {
        return Err(PyValueError::new_err("reps and trials must be >= 1"));
    }
    let orbits = RecordBatch::from_pyarrow_bound(orbit_batch)
        .map_err(|err| PyValueError::new_err(format!("invalid Orbits RecordBatch: {err}")))?;
    let run_once = || -> PyResult<()> {
        let output = sample_orbit_variants_record_batch(
            &orbits,
            method,
            num_samples,
            seed,
            alpha,
            beta,
            kappa,
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

fn unpack_mpc_dates(
    values: &[String],
) -> Result<Vec<String>, adam_core_rs_coords::MpcDesignationError> {
    values
        .iter()
        .map(|value| adam_core_rs_coords::mpc_designations::unpack_mpc_date_isot(value))
        .collect()
}

#[pyfunction]
fn unpack_mpc_dates_isot(values: Vec<String>) -> PyResult<Vec<String>> {
    unpack_mpc_dates(&values).map_err(mpc_designation_error)
}

#[pyfunction]
#[pyo3(signature = (values, reps, trials, warmup_reps=1))]
fn benchmark_unpack_mpc_dates_isot(
    values: Vec<String>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    if reps == 0 || trials == 0 {
        return Err(PyValueError::new_err("reps and trials must be >= 1"));
    }
    let run_once = || unpack_mpc_dates(&values).map_err(mpc_designation_error);
    let mut sample_trials = Vec::with_capacity(trials);
    for _ in 0..trials {
        for _ in 0..warmup_reps {
            black_box(run_once()?);
        }
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let started = Instant::now();
            black_box(run_once()?);
            samples.push(started.elapsed().as_secs_f64());
        }
        sample_trials.push(samples);
    }
    Ok(sample_trials)
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

/// Rust-owned public table loader. Python receives Arrow batches and wraps
/// them directly as the declared quivr table; no Python file I/O occurs.
#[pyfunction]
fn bandpasses_load_table<'py>(
    py: Python<'py>,
    data_dir: &str,
    filename: &str,
) -> PyResult<Vec<PyObject>> {
    let allowed = [
        "bandpass_curves.parquet",
        "observatory_band_map.parquet",
        "asteroid_templates.parquet",
        "template_bandpass_integrals.parquet",
    ];
    if !allowed.contains(&filename) {
        return Err(PyValueError::new_err(format!(
            "unsupported bandpass table: {filename}"
        )));
    }
    adam_core_rs_coords::read_bandpass_parquet_batches(
        &std::path::Path::new(data_dir).join(filename),
    )
    .map_err(bandpass_value_error)?
    .into_iter()
    .map(|batch| {
        batch.to_pyarrow(py).map_err(|err| {
            PyRuntimeError::new_err(format!("failed to export bandpass RecordBatch: {err}"))
        })
    })
    .collect()
}

#[pyfunction]
#[pyo3(signature = (data_dir, filename, reps, trials, warmup_reps=1))]
fn benchmark_bandpasses_load_table(
    data_dir: &str,
    filename: &str,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    if reps == 0 || trials == 0 {
        return Err(PyValueError::new_err("reps and trials must be >= 1"));
    }
    let path = std::path::Path::new(data_dir).join(filename);
    let run_once =
        || adam_core_rs_coords::read_bandpass_parquet_batches(&path).map_err(bandpass_value_error);
    let mut sample_trials = Vec::with_capacity(trials);
    for _ in 0..trials {
        for _ in 0..warmup_reps {
            black_box(run_once()?);
        }
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let started = Instant::now();
            black_box(run_once()?);
            samples.push(started.elapsed().as_secs_f64());
        }
        sample_trials.push(samples);
    }
    Ok(sample_trials)
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

#[derive(Clone)]
struct NormalizedComposition {
    template_id: Option<String>,
    mix: Option<(f64, f64)>,
}

fn normalize_bandpass_composition(
    template_id: Option<String>,
    mix: Option<(f64, f64)>,
) -> PyResult<NormalizedComposition> {
    match (template_id, mix) {
        (Some(template_id), None) => {
            if template_id.is_empty() {
                return Err(PyValueError::new_err(
                    "composition template_id must be non-empty",
                ));
            }
            Ok(NormalizedComposition {
                template_id: Some(template_id),
                mix: None,
            })
        }
        (None, Some((weight_c, weight_s))) => {
            if !weight_c.is_finite() || !weight_s.is_finite() {
                return Err(PyValueError::new_err("composition weights must be finite"));
            }
            if weight_c < 0.0 || weight_s < 0.0 {
                return Err(PyValueError::new_err(
                    "composition weights must be non-negative",
                ));
            }
            let total = weight_c + weight_s;
            if total <= 0.0 {
                return Err(PyValueError::new_err(
                    "at least one composition weight must be > 0",
                ));
            }
            Ok(NormalizedComposition {
                template_id: None,
                mix: Some((weight_c / total, weight_s / total)),
            })
        }
        _ => Err(PyValueError::new_err(
            "exactly one of template_id or mix weights is required",
        )),
    }
}

#[pyfunction]
#[pyo3(signature = (template_id=None, mix=None))]
fn bandpasses_composition_key(
    template_id: Option<String>,
    mix: Option<(f64, f64)>,
) -> PyResult<(Option<String>, Option<(f64, f64)>)> {
    let normalized = normalize_bandpass_composition(template_id, mix)?;
    Ok((normalized.template_id, normalized.mix))
}

fn convert_magnitudes_for_bandpasses(
    data: &adam_core_rs_coords::BandpassData,
    magnitudes: &[f64],
    source_filter_ids: &[String],
    target_filter_ids: &[String],
    composition: &NormalizedComposition,
) -> PyResult<Vec<f64>> {
    if source_filter_ids.len() != magnitudes.len() || target_filter_ids.len() != magnitudes.len() {
        return Err(PyValueError::new_err(
            "source_filter_id/target_filter_id must match magnitude length",
        ));
    }
    data.assert_filter_ids_have_curves(source_filter_ids)
        .map_err(bandpass_value_error)?;
    data.assert_filter_ids_have_curves(target_filter_ids)
        .map_err(bandpass_value_error)?;
    let deltas = data
        .delta_table(composition.template_id.as_deref(), composition.mix)
        .map_err(bandpass_value_error)?;
    let filter_index: HashMap<&str, usize> = data
        .filter_ids
        .iter()
        .enumerate()
        .map(|(index, filter_id)| (filter_id.as_str(), index))
        .collect();
    Ok(magnitudes
        .iter()
        .zip(source_filter_ids)
        .zip(target_filter_ids)
        .map(|((&magnitude, source), target)| {
            magnitude
                + (deltas[filter_index[target.as_str()]] - deltas[filter_index[source.as_str()]])
        })
        .collect())
}

#[pyfunction]
#[pyo3(signature = (data_dir, magnitudes, source_filter_ids, target_filter_ids, template_id=None, mix=None))]
fn bandpasses_convert_magnitude<'py>(
    py: Python<'py>,
    data_dir: &str,
    magnitudes: PyReadonlyArray1<'py, f64>,
    source_filter_ids: Vec<String>,
    target_filter_ids: Vec<String>,
    template_id: Option<String>,
    mix: Option<(f64, f64)>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let composition = normalize_bandpass_composition(template_id, mix)?;
    let data = bandpass_data_for(data_dir)?;
    let output = convert_magnitudes_for_bandpasses(
        data.as_ref(),
        magnitudes.as_slice()?,
        &source_filter_ids,
        &target_filter_ids,
        &composition,
    )?;
    Ok(output.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (data_dir, magnitudes, source_filter_ids, target_filter_ids, template_id, mix, reps, trials, warmup_reps=1))]
#[allow(clippy::too_many_arguments)]
fn benchmark_bandpasses_convert_magnitude(
    data_dir: &str,
    magnitudes: PyReadonlyArray1<'_, f64>,
    source_filter_ids: Vec<String>,
    target_filter_ids: Vec<String>,
    template_id: Option<String>,
    mix: Option<(f64, f64)>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    if reps == 0 || trials == 0 {
        return Err(PyValueError::new_err("reps and trials must be >= 1"));
    }
    let data = bandpass_data_for(data_dir)?;
    let magnitudes = magnitudes.as_slice()?.to_vec();
    let composition = normalize_bandpass_composition(template_id, mix)?;
    let run_once = || {
        convert_magnitudes_for_bandpasses(
            &data,
            &magnitudes,
            &source_filter_ids,
            &target_filter_ids,
            &composition,
        )
    };
    let mut sample_trials = Vec::with_capacity(trials);
    for _ in 0..trials {
        for _ in 0..warmup_reps {
            black_box(run_once()?);
        }
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let started = Instant::now();
            black_box(run_once()?);
            samples.push(started.elapsed().as_secs_f64());
        }
        sample_trials.push(samples);
    }
    Ok(sample_trials)
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

fn epoch_plus_fractional_days(days: i64, nanos: i64, delta_days: f64) -> Epoch {
    let delta_nanos = (delta_days * 86_400_000_000_000.0_f64).round() as i64;
    Epoch::new(days, nanos + delta_nanos)
}

fn nested_time_struct_from_epochs(epochs: &[Epoch]) -> Result<arrow_array::StructArray, String> {
    arrow_array::StructArray::try_new(
        arrow_schema::Fields::from(vec![
            arrow_schema::Field::new("days", arrow_schema::DataType::Int64, true),
            arrow_schema::Field::new("nanos", arrow_schema::DataType::Int64, true),
        ]),
        vec![
            std::sync::Arc::new(Int64Array::from_iter_values(
                epochs.iter().map(|epoch| epoch.days),
            )) as arrow_array::ArrayRef,
            std::sync::Arc::new(Int64Array::from_iter_values(
                epochs.iter().map(|epoch| epoch.nanos),
            )) as arrow_array::ArrayRef,
        ],
        None,
    )
    .map_err(|err| format!("failed to build time struct: {err}"))
}

fn nested_origin_struct_from_codes<'a, I>(codes: I) -> Result<arrow_array::StructArray, String>
where
    I: Iterator<Item = &'a str>,
{
    arrow_array::StructArray::try_new(
        arrow_schema::Fields::from(vec![arrow_schema::Field::new(
            "code",
            arrow_schema::DataType::LargeUtf8,
            true,
        )]),
        vec![
            std::sync::Arc::new(arrow_array::LargeStringArray::from_iter_values(codes))
                as arrow_array::ArrayRef,
        ],
        None,
    )
    .map_err(|err| format!("failed to build origin struct: {err}"))
}

/// Flatten a coordinate covariance to row-major (rows_out, 36), broadcasting
/// a single source row when needed. `missing` fills absent/invalid rows;
/// `nan_to_zero` mirrors the legacy predicted-covariance NaN policy.
fn coordinate_covariance_flat(
    coordinates: &DataCoordinateBatch,
    rows_out: usize,
    missing: f64,
    nan_to_zero: bool,
) -> Vec<f64> {
    let source_rows = coordinates.len();
    let mut out = vec![missing; rows_out * 36];
    if let Some(covariance) = &coordinates.covariance {
        for out_row in 0..rows_out {
            let source_row = if source_rows == 1 { 0 } else { out_row };
            if covariance.is_row_valid(source_row) {
                let values = covariance.row_values(source_row);
                for (element, value) in values.iter().enumerate() {
                    out[out_row * 36 + element] = if nan_to_zero && value.is_nan() {
                        0.0
                    } else {
                        *value
                    };
                }
            }
        }
    }
    out
}

/// Arrow-native `Residuals.calculate` core: decode observed/predicted
/// coordinate RecordBatches, broadcast, run the fused residual/chi2 kernel,
/// evaluate chi-squared survival probabilities, and assemble the finished
/// Residuals RecordBatch. Returns the batch plus the off-diagonal-NaN flag.
fn residuals_calculate_record_batch(
    observed_batch: &RecordBatch,
    predicted_batch: &RecordBatch,
    use_predicted_covariance: bool,
) -> Result<(RecordBatch, bool), String> {
    let observed = DataCoordinateBatch::try_from_nested_record_batch(observed_batch)
        .map_err(|err| format!("failed to decode observed coordinates: {err}"))?;
    let predicted = DataCoordinateBatch::try_from_nested_record_batch(predicted_batch)
        .map_err(|err| format!("failed to decode predicted coordinates: {err}"))?;
    if observed.representation() != predicted.representation() {
        return Err("observed and predicted representations must match".to_string());
    }
    let rows = observed.len();
    let predicted_rows = predicted.len();
    if predicted_rows != rows && predicted_rows != 1 {
        return Err(format!(
            "predicted coordinates must have length 1 or match observed length ({rows}), got {predicted_rows}"
        ));
    }

    let observed_flat: Vec<f64> = observed
        .values
        .raw_values()
        .iter()
        .flatten()
        .copied()
        .collect();
    let predicted_raw = predicted.values.raw_values();
    let mut predicted_flat = Vec::with_capacity(rows * 6);
    for row in 0..rows {
        let source = if predicted_rows == 1 {
            &predicted_raw[0]
        } else {
            &predicted_raw[row]
        };
        predicted_flat.extend_from_slice(source);
    }
    let observed_covariance = coordinate_covariance_flat(&observed, rows, f64::NAN, false);
    let predicted_covariance = if use_predicted_covariance {
        coordinate_covariance_flat(&predicted, rows, 0.0, true)
    } else {
        vec![0.0; rows * 36]
    };

    let is_spherical = observed.representation() == DataRepresentation::Spherical;
    let output = compute_residuals_chi2_flat(
        &observed_flat,
        &predicted_flat,
        &observed_covariance,
        &predicted_covariance,
        rows,
        6,
        is_spherical,
    )
    .map_err(|err| format!("fused residuals kernel failed: {err:?}"))?;

    let probability: Vec<f64> = output
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

    let mut values_builder =
        arrow_array::builder::ListBuilder::new(arrow_array::builder::Float64Builder::new());
    for row in 0..rows {
        for column in 0..6 {
            values_builder
                .values()
                .append_value(output.residuals[row * 6 + column]);
        }
        values_builder.append(true);
    }
    let values = values_builder.finish();

    let fields = vec![
        arrow_schema::Field::new("values", values.data_type().clone(), true),
        arrow_schema::Field::new("chi2", arrow_schema::DataType::Float64, true),
        arrow_schema::Field::new("dof", arrow_schema::DataType::Int64, true),
        arrow_schema::Field::new("probability", arrow_schema::DataType::Float64, true),
    ];
    let arrays: Vec<arrow_array::ArrayRef> = vec![
        std::sync::Arc::new(values),
        std::sync::Arc::new(arrow_array::Float64Array::from(output.chi2)),
        std::sync::Arc::new(Int64Array::from(output.dof)),
        std::sync::Arc::new(arrow_array::Float64Array::from(probability)),
    ];
    let mut metadata = HashMap::new();
    metadata.insert(
        "adam_core_schema".to_string(),
        "Residuals.nested.quivr.v1".to_string(),
    );
    let batch = RecordBatch::try_new(
        std::sync::Arc::new(arrow_schema::Schema::new_with_metadata(fields, metadata)),
        arrays,
    )
    .map_err(|err| format!("failed to build Residuals RecordBatch: {err}"))?;
    Ok((batch, output.had_off_diagonal_nan))
}

/// Arrow-native public `Residuals.calculate` surface.
#[pyfunction]
#[pyo3(signature = (observed_batch, predicted_batch, use_predicted_covariance=true))]
fn residuals_calculate_arrow<'py>(
    py: Python<'py>,
    observed_batch: &Bound<'py, PyAny>,
    predicted_batch: &Bound<'py, PyAny>,
    use_predicted_covariance: bool,
) -> PyResult<(PyObject, bool)> {
    let observed = RecordBatch::from_pyarrow_bound(observed_batch).map_err(|err| {
        PyValueError::new_err(format!("invalid observed coordinates RecordBatch: {err}"))
    })?;
    let predicted = RecordBatch::from_pyarrow_bound(predicted_batch).map_err(|err| {
        PyValueError::new_err(format!("invalid predicted coordinates RecordBatch: {err}"))
    })?;
    let (batch, had_off_diagonal_nan) = py
        .allow_threads(|| {
            residuals_calculate_record_batch(&observed, &predicted, use_predicted_covariance)
        })
        .map_err(PyValueError::new_err)?;
    let exported = batch.to_pyarrow(py).map_err(|err| {
        PyRuntimeError::new_err(format!("failed to export Residuals RecordBatch: {err}"))
    })?;
    Ok((exported, had_off_diagonal_nan))
}

/// Rust-owned Instant timer for the Arrow-native `Residuals.calculate` surface.
#[pyfunction]
#[pyo3(signature = (observed_batch, predicted_batch, reps, trials, warmup_reps=1, use_predicted_covariance=true))]
fn benchmark_residuals_calculate_arrow(
    observed_batch: &Bound<'_, PyAny>,
    predicted_batch: &Bound<'_, PyAny>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
    use_predicted_covariance: bool,
) -> PyResult<Vec<Vec<f64>>> {
    if reps == 0 || trials == 0 {
        return Err(PyValueError::new_err("reps and trials must be >= 1"));
    }
    let observed = RecordBatch::from_pyarrow_bound(observed_batch).map_err(|err| {
        PyValueError::new_err(format!("invalid observed coordinates RecordBatch: {err}"))
    })?;
    let predicted = RecordBatch::from_pyarrow_bound(predicted_batch).map_err(|err| {
        PyValueError::new_err(format!("invalid predicted coordinates RecordBatch: {err}"))
    })?;
    let run_once = || -> PyResult<()> {
        let output =
            residuals_calculate_record_batch(&observed, &predicted, use_predicted_covariance)
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

/// Arrow-native perturber-MOID crossing: decode one Orbits RecordBatch, run
/// the SPICE-backed perturber loop + batched MOID kernel in Rust, and return
/// the finished PerturberMOIDs nested RecordBatch (perturber-major rows).
fn perturber_moids_record_batch(
    orbit_batch: &RecordBatch,
    perturber_codes: &[String],
    max_iter: usize,
    xtol: f64,
) -> Result<RecordBatch, String> {
    if perturber_codes.is_empty() {
        return Err("perturber MOIDs require at least one perturber".to_string());
    }
    let orbits = DataOrbitBatch::try_from_nested_record_batch(orbit_batch)
        .map_err(|err| format!("failed to decode OrbitBatch: {err}"))?;
    let rows = orbits.len();
    if rows == 0 {
        return Err("perturber MOIDs require at least one orbit".to_string());
    }
    let coordinates = &orbits.coordinates;
    let times = coordinates
        .times
        .as_ref()
        .ok_or_else(|| "orbit coordinates require times".to_string())?;
    let origin = coordinates.origins.origins[0].clone();
    if coordinates.origins.origins.iter().any(|o| o != &origin) {
        return Err("perturber MOIDs require a single orbit origin".to_string());
    }
    let states = coordinates
        .values
        .cartesian()
        .ok_or_else(|| "perturber MOIDs require Cartesian coordinates".to_string())?;
    let states_flat: Vec<f64> = states.iter().flatten().copied().collect();
    let mus = coordinates
        .origins
        .origins
        .iter()
        .map(origin_mu_au3_day2)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("failed to resolve origin mu: {err}"))?;
    let perturbers: Vec<OriginId> = perturber_codes
        .iter()
        .map(|code| OriginId::from_code(code.clone()))
        .collect();

    let backend = global_backend()
        .lock()
        .map_err(|_| "SPICE backend lock is poisoned".to_string())?;
    let (moids, dt_mins) = backend
        .calculate_perturber_moids(
            &states_flat,
            &mus,
            times,
            &perturbers,
            coordinates.frame,
            &origin,
            max_iter,
            xtol,
        )
        .map_err(|err| err.to_string())?;
    drop(backend);

    let output_rows = perturber_codes.len() * rows;
    let mut moid_epochs = Vec::with_capacity(output_rows);
    for chunk_start in (0..output_rows).step_by(rows) {
        for row in 0..rows {
            let epoch = times.epochs[row];
            moid_epochs.push(epoch_plus_fractional_days(
                epoch.days,
                epoch.nanos,
                dt_mins[chunk_start + row],
            ));
        }
    }
    let orbit_id_values: Vec<&str> = (0..perturber_codes.len())
        .flat_map(|_| orbits.orbit_id.iter().map(|orbit_id| orbit_id.0.as_str()))
        .collect();
    let orbit_ids =
        arrow_array::LargeStringArray::from_iter_values(orbit_id_values.iter().copied());
    let perturber_code_values: Vec<&str> = perturber_codes
        .iter()
        .flat_map(|code| std::iter::repeat_n(code.as_str(), rows))
        .collect();
    let perturber = nested_origin_struct_from_codes(perturber_code_values.iter().copied())?;
    let time = nested_time_struct_from_epochs(&moid_epochs)?;

    let fields = vec![
        arrow_schema::Field::new("orbit_id", arrow_schema::DataType::LargeUtf8, false),
        arrow_schema::Field::new("perturber", perturber.data_type().clone(), true),
        arrow_schema::Field::new("moid", arrow_schema::DataType::Float64, false),
        arrow_schema::Field::new("time", time.data_type().clone(), true),
    ];
    let arrays: Vec<arrow_array::ArrayRef> = vec![
        std::sync::Arc::new(orbit_ids),
        std::sync::Arc::new(perturber),
        std::sync::Arc::new(arrow_array::Float64Array::from(moids)),
        std::sync::Arc::new(time),
    ];
    let mut metadata = HashMap::new();
    metadata.insert(
        "adam_core_schema".to_string(),
        "PerturberMOIDs.nested.quivr.v1".to_string(),
    );
    metadata.insert(
        "adam_core_time_scale".to_string(),
        times.scale.as_str().to_string(),
    );
    RecordBatch::try_new(
        std::sync::Arc::new(arrow_schema::Schema::new_with_metadata(fields, metadata)),
        arrays,
    )
    .map_err(|err| format!("failed to build PerturberMOIDs RecordBatch: {err}"))
}

/// Arrow-native public perturber-MOID surface.
#[pyfunction]
#[pyo3(signature = (orbit_batch, perturber_codes, max_iter=100, xtol=1e-10))]
fn calculate_perturber_moids_arrow<'py>(
    py: Python<'py>,
    orbit_batch: &Bound<'py, PyAny>,
    perturber_codes: Vec<String>,
    max_iter: usize,
    xtol: f64,
) -> PyResult<PyObject> {
    let orbits = RecordBatch::from_pyarrow_bound(orbit_batch)
        .map_err(|err| PyValueError::new_err(format!("invalid Orbits RecordBatch: {err}")))?;
    let output = py
        .allow_threads(|| perturber_moids_record_batch(&orbits, &perturber_codes, max_iter, xtol))
        .map_err(PyValueError::new_err)?;
    output
        .to_pyarrow(py)
        .map_err(|err| PyRuntimeError::new_err(format!("failed to export RecordBatch: {err}")))
}

/// Rust-owned Instant timer for the Arrow-native perturber-MOID surface.
#[pyfunction]
#[pyo3(signature = (orbit_batch, perturber_codes, reps, trials, warmup_reps=1, max_iter=100, xtol=1e-10))]
#[allow(clippy::too_many_arguments)]
fn benchmark_calculate_perturber_moids_arrow(
    orbit_batch: &Bound<'_, PyAny>,
    perturber_codes: Vec<String>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
    max_iter: usize,
    xtol: f64,
) -> PyResult<Vec<Vec<f64>>> {
    if reps == 0 || trials == 0 {
        return Err(PyValueError::new_err("reps and trials must be >= 1"));
    }
    let orbits = RecordBatch::from_pyarrow_bound(orbit_batch)
        .map_err(|err| PyValueError::new_err(format!("invalid Orbits RecordBatch: {err}")))?;
    let run_once = || -> PyResult<()> {
        let output = perturber_moids_record_batch(&orbits, &perturber_codes, max_iter, xtol)
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

struct PorkchopSide {
    orbit_ids: Vec<String>,
    states_flat: Vec<f64>,
    mjds: Vec<f64>,
    epochs: Vec<Epoch>,
    scale: TimeScale,
}

fn porkchop_side(batch: &RecordBatch, label: &str) -> Result<PorkchopSide, String> {
    let orbits = DataOrbitBatch::try_from_nested_record_batch(batch)
        .map_err(|err| format!("failed to decode {label} OrbitBatch: {err}"))?;
    let coordinates = &orbits.coordinates;
    let times = coordinates
        .times
        .as_ref()
        .ok_or_else(|| format!("{label} orbit coordinates require times"))?;
    let states = coordinates
        .values
        .cartesian()
        .ok_or_else(|| format!("{label} orbits require Cartesian coordinates"))?;
    let rows = orbits.len();
    let mut order: Vec<usize> = (0..rows).collect();
    order.sort_by_key(|&row| (times.epochs[row].days, times.epochs[row].nanos));
    let mjd_values = times.mjd_values();
    Ok(PorkchopSide {
        orbit_ids: order
            .iter()
            .map(|&row| orbits.orbit_id[row].0.clone())
            .collect(),
        states_flat: order
            .iter()
            .flat_map(|&row| states[row].iter().copied())
            .collect(),
        mjds: order.iter().map(|&row| mjd_values[row]).collect(),
        epochs: order.iter().map(|&row| times.epochs[row]).collect(),
        scale: times.scale,
    })
}

fn porkchop_frame_and_origin(
    departure_batch: &RecordBatch,
    arrival_batch: &RecordBatch,
) -> Result<(DataFrame, OriginId), String> {
    let departure = DataOrbitBatch::try_from_nested_record_batch(departure_batch)
        .map_err(|err| format!("failed to decode departure OrbitBatch: {err}"))?;
    let arrival = DataOrbitBatch::try_from_nested_record_batch(arrival_batch)
        .map_err(|err| format!("failed to decode arrival OrbitBatch: {err}"))?;
    if departure.coordinates.frame != arrival.coordinates.frame {
        return Err("departure and arrival frames must be the same".to_string());
    }
    let origin = departure
        .coordinates
        .origins
        .origins
        .first()
        .cloned()
        .ok_or_else(|| "porkchop requires at least one departure orbit".to_string())?;
    if departure
        .coordinates
        .origins
        .origins
        .iter()
        .chain(arrival.coordinates.origins.origins.iter())
        .any(|o| o != &origin)
    {
        return Err("departure and arrival origins must be the same".to_string());
    }
    Ok((departure.coordinates.frame, origin))
}

/// Arrow-native porkchop crossing: decode departure/arrival Orbits
/// RecordBatches, run meshgrid + time filter + Rayon-batched Lambert in Rust,
/// and return the finished LambertSolutions nested RecordBatch.
#[allow(clippy::too_many_arguments)]
fn porkchop_record_batch(
    departure_batch: &RecordBatch,
    arrival_batch: &RecordBatch,
    propagation_origin: &str,
    prograde: bool,
    max_iter: u32,
    tol: f64,
) -> Result<RecordBatch, String> {
    let (frame, _shared_origin) = porkchop_frame_and_origin(departure_batch, arrival_batch)?;
    let departure = porkchop_side(departure_batch, "departure")?;
    let arrival = porkchop_side(arrival_batch, "arrival")?;
    let mu = origin_mu_au3_day2(&OriginId::from_code(propagation_origin))
        .map_err(|err| format!("failed to resolve propagation-origin mu: {err}"))?;

    let (dep_idx, arr_idx, v1_flat, v2_flat) = porkchop_grid_flat(
        &departure.states_flat,
        &departure.mjds,
        &arrival.states_flat,
        &arrival.mjds,
        mu,
        prograde,
        max_iter,
        tol,
        tol,
    );
    let rows = dep_idx.len();

    let state_column = |side: &PorkchopSide, indices: &[u32], component: usize| {
        arrow_array::Float64Array::from_iter_values(
            indices
                .iter()
                .map(|&index| side.states_flat[index as usize * 6 + component]),
        )
    };
    let solution_column = |flat: &[f64], component: usize| {
        arrow_array::Float64Array::from_iter_values((0..rows).map(|row| flat[row * 3 + component]))
    };
    let departure_epochs: Vec<Epoch> = dep_idx
        .iter()
        .map(|&index| departure.epochs[index as usize])
        .collect();
    let arrival_epochs: Vec<Epoch> = arr_idx
        .iter()
        .map(|&index| arrival.epochs[index as usize])
        .collect();

    let departure_time = nested_time_struct_from_epochs(&departure_epochs)?;
    let arrival_time = nested_time_struct_from_epochs(&arrival_epochs)?;
    let origin = nested_origin_struct_from_codes((0..rows).map(|_| propagation_origin))?;

    let mut fields = vec![
        arrow_schema::Field::new(
            "departure_body_id",
            arrow_schema::DataType::LargeUtf8,
            false,
        ),
        arrow_schema::Field::new("departure_time", departure_time.data_type().clone(), true),
    ];
    let mut arrays: Vec<arrow_array::ArrayRef> = vec![
        std::sync::Arc::new(arrow_array::LargeStringArray::from_iter_values(
            dep_idx
                .iter()
                .map(|&index| departure.orbit_ids[index as usize].as_str()),
        )),
        std::sync::Arc::new(departure_time),
    ];
    for (name, component) in [
        ("departure_body_x", 0),
        ("departure_body_y", 1),
        ("departure_body_z", 2),
        ("departure_body_vx", 3),
        ("departure_body_vy", 4),
        ("departure_body_vz", 5),
    ] {
        fields.push(arrow_schema::Field::new(
            name,
            arrow_schema::DataType::Float64,
            false,
        ));
        arrays.push(std::sync::Arc::new(state_column(
            &departure, &dep_idx, component,
        )));
    }
    fields.push(arrow_schema::Field::new(
        "arrival_body_id",
        arrow_schema::DataType::LargeUtf8,
        false,
    ));
    arrays.push(std::sync::Arc::new(
        arrow_array::LargeStringArray::from_iter_values(
            arr_idx
                .iter()
                .map(|&index| arrival.orbit_ids[index as usize].as_str()),
        ),
    ));
    fields.push(arrow_schema::Field::new(
        "arrival_time",
        arrival_time.data_type().clone(),
        true,
    ));
    arrays.push(std::sync::Arc::new(arrival_time));
    for (name, component) in [
        ("arrival_body_x", 0),
        ("arrival_body_y", 1),
        ("arrival_body_z", 2),
        ("arrival_body_vx", 3),
        ("arrival_body_vy", 4),
        ("arrival_body_vz", 5),
    ] {
        fields.push(arrow_schema::Field::new(
            name,
            arrow_schema::DataType::Float64,
            false,
        ));
        arrays.push(std::sync::Arc::new(state_column(
            &arrival, &arr_idx, component,
        )));
    }
    for (name, flat, component) in [
        ("solution_departure_vx", &v1_flat, 0),
        ("solution_departure_vy", &v1_flat, 1),
        ("solution_departure_vz", &v1_flat, 2),
        ("solution_arrival_vx", &v2_flat, 0),
        ("solution_arrival_vy", &v2_flat, 1),
        ("solution_arrival_vz", &v2_flat, 2),
    ] {
        fields.push(arrow_schema::Field::new(
            name,
            arrow_schema::DataType::Float64,
            false,
        ));
        arrays.push(std::sync::Arc::new(solution_column(flat, component)));
    }
    fields.push(arrow_schema::Field::new(
        "origin",
        origin.data_type().clone(),
        true,
    ));
    arrays.push(std::sync::Arc::new(origin));

    let mut metadata = HashMap::new();
    metadata.insert(
        "adam_core_schema".to_string(),
        "LambertSolutions.nested.quivr.v1".to_string(),
    );
    metadata.insert("adam_core_frame".to_string(), frame.as_str().to_string());
    metadata.insert(
        "adam_core_departure_time_scale".to_string(),
        departure.scale.as_str().to_string(),
    );
    metadata.insert(
        "adam_core_arrival_time_scale".to_string(),
        arrival.scale.as_str().to_string(),
    );
    RecordBatch::try_new(
        std::sync::Arc::new(arrow_schema::Schema::new_with_metadata(fields, metadata)),
        arrays,
    )
    .map_err(|err| format!("failed to build LambertSolutions RecordBatch: {err}"))
}

fn lambert_float_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a Float64Array, String> {
    batch
        .column_by_name(name)
        .ok_or_else(|| format!("LambertSolutions is missing {name}"))?
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| format!("LambertSolutions {name} must be Float64"))
}

fn lambert_string_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a LargeStringArray, String> {
    batch
        .column_by_name(name)
        .ok_or_else(|| format!("LambertSolutions is missing {name}"))?
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .ok_or_else(|| format!("LambertSolutions {name} must be LargeUtf8"))
}

fn lambert_time_column(
    batch: &RecordBatch,
    name: &str,
    scale: TimeScale,
) -> Result<TimeArray, String> {
    let values = batch
        .column_by_name(name)
        .ok_or_else(|| format!("LambertSolutions is missing {name}"))?
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| format!("LambertSolutions {name} must be a StructArray"))?;
    let days = values
        .column_by_name("days")
        .ok_or_else(|| format!("LambertSolutions {name} is missing days"))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| format!("LambertSolutions {name}.days must be Int64"))?;
    let nanos = values
        .column_by_name("nanos")
        .ok_or_else(|| format!("LambertSolutions {name} is missing nanos"))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| format!("LambertSolutions {name}.nanos must be Int64"))?;
    TimeArray::from_parts(
        scale,
        (0..days.len()).map(|row| days.value(row)).collect(),
        (0..nanos.len()).map(|row| nanos.value(row)).collect(),
    )
    .map_err(|err| err.to_string())
}

fn lambert_origin_codes(batch: &RecordBatch) -> Result<Vec<OriginId>, String> {
    let origin = batch
        .column_by_name("origin")
        .ok_or_else(|| "LambertSolutions is missing origin".to_string())?
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| "LambertSolutions origin must be a StructArray".to_string())?;
    let codes = origin
        .column_by_name("code")
        .ok_or_else(|| "LambertSolutions origin is missing code".to_string())?
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .ok_or_else(|| "LambertSolutions origin.code must be LargeUtf8".to_string())?;
    Ok((0..codes.len())
        .map(|row| OriginId::from_code(codes.value(row)))
        .collect())
}

/// Build one canonical nested OrbitBatch from a LambertSolutions batch.
fn lambert_solution_orbit_record_batch(
    batch: &RecordBatch,
    accessor: &str,
) -> Result<RecordBatch, String> {
    let schema = batch.schema();
    let metadata = schema.metadata();
    let frame = match metadata
        .get("adam_core_frame")
        .map(String::as_str)
        .unwrap_or("unspecified")
    {
        "ecliptic" => DataFrame::Ecliptic,
        "equatorial" => DataFrame::Equatorial,
        "itrf93" => DataFrame::Itrf93,
        "unspecified" => DataFrame::Unspecified,
        other => return Err(format!("unsupported LambertSolutions frame: {other}")),
    };
    let (time_name, scale_key, id_name, position_prefix, velocity_prefix, generated_prefix) =
        match accessor {
            "departure_body" => (
                "departure_time",
                "adam_core_departure_time_scale",
                Some("departure_body_id"),
                "departure_body_",
                "departure_body_",
                None,
            ),
            "arrival_body" => (
                "arrival_time",
                "adam_core_arrival_time_scale",
                Some("arrival_body_id"),
                "arrival_body_",
                "arrival_body_",
                None,
            ),
            "solution_departure" => (
                "departure_time",
                "adam_core_departure_time_scale",
                None,
                "departure_body_",
                "solution_departure_",
                Some("solution_departure_orbit_"),
            ),
            "solution_arrival" => (
                "arrival_time",
                "adam_core_arrival_time_scale",
                None,
                "arrival_body_",
                "solution_arrival_",
                Some("solution_arrival_orbit_"),
            ),
            other => return Err(format!("unknown LambertSolutions orbit accessor: {other}")),
        };
    let scale_name = metadata.get(scale_key).map(String::as_str).unwrap_or("tdb");
    let scale = TimeScale::parse(scale_name).map_err(|err| err.to_string())?;
    let times = lambert_time_column(batch, time_name, scale)?;
    let rows = batch.num_rows();
    let component =
        |prefix: &str, suffix: &str| lambert_float_column(batch, &format!("{prefix}{suffix}"));
    let x = component(position_prefix, "x")?;
    let y = component(position_prefix, "y")?;
    let z = component(position_prefix, "z")?;
    let vx = component(velocity_prefix, "vx")?;
    let vy = component(velocity_prefix, "vy")?;
    let vz = component(velocity_prefix, "vz")?;
    let values = (0..rows)
        .map(|row| {
            [
                x.value(row),
                y.value(row),
                z.value(row),
                vx.value(row),
                vy.value(row),
                vz.value(row),
            ]
        })
        .collect();
    let orbit_ids = match (id_name, generated_prefix) {
        (Some(name), None) => {
            let ids = lambert_string_column(batch, name)?;
            (0..rows)
                .map(|row| adam_core_rs_coords::OrbitId(ids.value(row).to_string()))
                .collect()
        }
        (None, Some(prefix)) => (0..rows)
            .map(|row| adam_core_rs_coords::OrbitId(format!("{prefix}{row}")))
            .collect(),
        _ => return Err("invalid LambertSolutions accessor configuration".to_string()),
    };
    let coordinates = DataCoordinateBatch::cartesian(
        values,
        frame,
        OriginArray::new(lambert_origin_codes(batch)?),
        Some(times),
        None,
    )
    .map_err(|err| err.to_string())?;
    DataOrbitBatch::new(orbit_ids, vec![None; rows], coordinates)
        .map_err(|err| err.to_string())?
        .into_nested_record_batch()
        .map_err(|err| err.to_string())
}

#[pyfunction]
fn lambert_solution_orbit_arrow<'py>(
    py: Python<'py>,
    batch: &Bound<'py, PyAny>,
    accessor: &str,
) -> PyResult<PyObject> {
    let batch = RecordBatch::from_pyarrow_bound(batch)
        .map_err(|err| PyValueError::new_err(format!("invalid LambertSolutions batch: {err}")))?;
    py.allow_threads(|| lambert_solution_orbit_record_batch(&batch, accessor))
        .map_err(PyValueError::new_err)?
        .to_pyarrow(py)
        .map_err(|err| PyRuntimeError::new_err(format!("failed to export OrbitBatch: {err}")))
}

#[pyfunction]
#[pyo3(signature = (batch, accessor, reps, trials, warmup_reps=1))]
fn benchmark_lambert_solution_orbit_arrow(
    batch: &Bound<'_, PyAny>,
    accessor: &str,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    if reps == 0 || trials == 0 {
        return Err(PyValueError::new_err("reps and trials must be >= 1"));
    }
    let batch = RecordBatch::from_pyarrow_bound(batch)
        .map_err(|err| PyValueError::new_err(format!("invalid LambertSolutions batch: {err}")))?;
    let run_once =
        || lambert_solution_orbit_record_batch(&batch, accessor).map_err(PyValueError::new_err);
    let mut sample_trials = Vec::with_capacity(trials);
    for _ in 0..trials {
        for _ in 0..warmup_reps {
            black_box(run_once()?);
        }
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let started = Instant::now();
            black_box(run_once()?);
            samples.push(started.elapsed().as_secs_f64());
        }
        sample_trials.push(samples);
    }
    Ok(sample_trials)
}

/// Arrow-native public porkchop surface.
#[pyfunction]
#[pyo3(signature = (departure_batch, arrival_batch, propagation_origin, prograde=true, max_iter=35, tol=1e-10))]
fn generate_porkchop_data_arrow<'py>(
    py: Python<'py>,
    departure_batch: &Bound<'py, PyAny>,
    arrival_batch: &Bound<'py, PyAny>,
    propagation_origin: &str,
    prograde: bool,
    max_iter: u32,
    tol: f64,
) -> PyResult<PyObject> {
    let departure = RecordBatch::from_pyarrow_bound(departure_batch).map_err(|err| {
        PyValueError::new_err(format!("invalid departure Orbits RecordBatch: {err}"))
    })?;
    let arrival = RecordBatch::from_pyarrow_bound(arrival_batch).map_err(|err| {
        PyValueError::new_err(format!("invalid arrival Orbits RecordBatch: {err}"))
    })?;
    let output = py
        .allow_threads(|| {
            porkchop_record_batch(
                &departure,
                &arrival,
                propagation_origin,
                prograde,
                max_iter,
                tol,
            )
        })
        .map_err(PyValueError::new_err)?;
    output
        .to_pyarrow(py)
        .map_err(|err| PyRuntimeError::new_err(format!("failed to export RecordBatch: {err}")))
}

/// Rust-owned Instant timer for the Arrow-native porkchop surface.
#[pyfunction]
#[pyo3(signature = (departure_batch, arrival_batch, propagation_origin, reps, trials, warmup_reps=1, prograde=true, max_iter=35, tol=1e-10))]
#[allow(clippy::too_many_arguments)]
fn benchmark_generate_porkchop_data_arrow(
    departure_batch: &Bound<'_, PyAny>,
    arrival_batch: &Bound<'_, PyAny>,
    propagation_origin: &str,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
    prograde: bool,
    max_iter: u32,
    tol: f64,
) -> PyResult<Vec<Vec<f64>>> {
    if reps == 0 || trials == 0 {
        return Err(PyValueError::new_err("reps and trials must be >= 1"));
    }
    let departure = RecordBatch::from_pyarrow_bound(departure_batch).map_err(|err| {
        PyValueError::new_err(format!("invalid departure Orbits RecordBatch: {err}"))
    })?;
    let arrival = RecordBatch::from_pyarrow_bound(arrival_batch).map_err(|err| {
        PyValueError::new_err(format!("invalid arrival Orbits RecordBatch: {err}"))
    })?;
    let run_once = || -> PyResult<()> {
        let output = porkchop_record_batch(
            &departure,
            &arrival,
            propagation_origin,
            prograde,
            max_iter,
            tol,
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

fn linkage_precision_nanos(precision: &str) -> Result<i64, String> {
    match precision {
        "ns" => Ok(1),
        "us" => Ok(1_000),
        "ms" => Ok(1_000_000),
        "s" => Ok(1_000_000_000),
        _ => Err(format!("Unsupported precision: {precision}")),
    }
}

fn rounded_time_columns(times: &TimeArray, precision: i64) -> (Vec<i64>, Vec<i64>) {
    let days = times.epochs.iter().map(|epoch| epoch.days).collect();
    let nanos = times
        .epochs
        .iter()
        .map(|epoch| epoch.nanos / precision * precision)
        .collect();
    (days, nanos)
}

struct EphemerisObserverLinkageKeys {
    left_days: Vec<i64>,
    left_nanos: Vec<i64>,
    right_days: Vec<i64>,
    right_nanos: Vec<i64>,
    observer_days: Vec<i64>,
    observer_nanos: Vec<i64>,
    expected_unique_observers: usize,
}

fn prepare_ephemeris_observer_linkage(
    ephemeris_batch: &RecordBatch,
    observer_batch: &RecordBatch,
    precision: &str,
) -> Result<EphemerisObserverLinkageKeys, String> {
    let ephemeris_coordinates = DataCoordinateBatch::try_from_nested_record_batch(ephemeris_batch)
        .map_err(|err| format!("failed to decode ephemeris coordinates: {err}"))?;
    let observers = DataObserverBatch::try_from_nested_record_batch(observer_batch)
        .map_err(|err| format!("failed to decode ObserverBatch: {err}"))?;
    let ephemeris_times = ephemeris_coordinates
        .times
        .as_ref()
        .ok_or_else(|| "ephemeris coordinates require times".to_string())?;
    let observer_times = observers
        .coordinates
        .times
        .as_ref()
        .ok_or_else(|| "observer coordinates require times".to_string())?
        .rescale(ephemeris_times.scale)
        .map_err(|err| format!("failed to rescale observer times: {err}"))?;
    let precision = linkage_precision_nanos(precision)?;
    let (left_days, left_nanos) = rounded_time_columns(ephemeris_times, precision);
    let (right_days, right_nanos) = rounded_time_columns(&observer_times, precision);
    let expected_unique_observers = observers
        .code
        .iter()
        .zip(observer_times.epochs.iter())
        .map(|(code, epoch)| (code.0.as_str(), epoch.days, epoch.nanos))
        .collect::<HashSet<_>>()
        .len();
    Ok(EphemerisObserverLinkageKeys {
        left_days,
        left_nanos,
        right_days,
        right_nanos,
        observer_days: observer_times
            .epochs
            .iter()
            .map(|epoch| epoch.days)
            .collect(),
        observer_nanos: observer_times
            .epochs
            .iter()
            .map(|epoch| epoch.nanos)
            .collect(),
        expected_unique_observers,
    })
}

#[pyfunction]
fn prepare_ephemeris_observer_linkage_arrow(
    ephemeris_batch: &Bound<'_, PyAny>,
    observer_batch: &Bound<'_, PyAny>,
    precision: &str,
) -> PyResult<(
    Vec<i64>,
    Vec<i64>,
    Vec<i64>,
    Vec<i64>,
    Vec<i64>,
    Vec<i64>,
    usize,
)> {
    let ephemeris = RecordBatch::from_pyarrow_bound(ephemeris_batch)
        .map_err(|err| PyValueError::new_err(format!("invalid Ephemeris RecordBatch: {err}")))?;
    let observers = RecordBatch::from_pyarrow_bound(observer_batch)
        .map_err(|err| PyValueError::new_err(format!("invalid Observer RecordBatch: {err}")))?;
    let keys = prepare_ephemeris_observer_linkage(&ephemeris, &observers, precision)
        .map_err(PyValueError::new_err)?;
    Ok((
        keys.left_days,
        keys.left_nanos,
        keys.right_days,
        keys.right_nanos,
        keys.observer_days,
        keys.observer_nanos,
        keys.expected_unique_observers,
    ))
}

#[pyfunction]
#[pyo3(signature = (ephemeris_batch, observer_batch, precision, reps, trials, warmup_reps=1))]
fn benchmark_prepare_ephemeris_observer_linkage_arrow(
    ephemeris_batch: &Bound<'_, PyAny>,
    observer_batch: &Bound<'_, PyAny>,
    precision: &str,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let ephemeris = RecordBatch::from_pyarrow_bound(ephemeris_batch)
        .map_err(|err| PyValueError::new_err(format!("invalid Ephemeris RecordBatch: {err}")))?;
    let observers = RecordBatch::from_pyarrow_bound(observer_batch)
        .map_err(|err| PyValueError::new_err(format!("invalid Observer RecordBatch: {err}")))?;
    benchmark_orbit_surface(reps, trials, warmup_reps, || {
        black_box(prepare_ephemeris_observer_linkage(
            &ephemeris, &observers, precision,
        )?);
        Ok(())
    })
}

fn take_record_batch_groups(
    batch: &RecordBatch,
    groups: Vec<(String, Vec<u32>)>,
) -> Result<Vec<(String, RecordBatch)>, String> {
    groups
        .into_iter()
        .map(|(id, rows)| {
            let indices = UInt32Array::from(rows);
            let columns = batch
                .columns()
                .iter()
                .map(|column| {
                    arrow::compute::take(column.as_ref(), &indices, None)
                        .map_err(|err| format!("failed to group column: {err}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let grouped = RecordBatch::try_new(batch.schema(), columns)
                .map_err(|err| format!("failed to assemble grouped batch: {err}"))?;
            Ok((id, grouped))
        })
        .collect()
}

fn first_appearance_groups(ids: Vec<String>) -> Result<Vec<(String, Vec<u32>)>, String> {
    let mut group_lookup = HashMap::<String, usize>::new();
    let mut groups = Vec::<(String, Vec<u32>)>::new();
    for (row, id) in ids.into_iter().enumerate() {
        let group = match group_lookup.get(&id) {
            Some(group) => *group,
            None => {
                let group = groups.len();
                group_lookup.insert(id.clone(), group);
                groups.push((id, Vec::new()));
                group
            }
        };
        groups[group]
            .1
            .push(u32::try_from(row).map_err(|_| "row index exceeds UInt32".to_string())?);
    }
    Ok(groups)
}

fn group_by_orbit_id_record_batch(
    batch: &RecordBatch,
) -> Result<Vec<(String, RecordBatch)>, String> {
    let orbits = DataOrbitBatch::try_from_nested_record_batch(batch)
        .map_err(|err| format!("failed to decode OrbitBatch: {err}"))?;
    let ids = orbits
        .orbit_id
        .iter()
        .map(|orbit_id| orbit_id.0.clone())
        .collect();
    take_record_batch_groups(batch, first_appearance_groups(ids)?)
}

/// Sorted-by-code grouping over the nested Observers RecordBatch, matching
/// the legacy `iterate_codes` (unique().sort()) iteration order.
fn group_observers_by_code_record_batch(
    batch: &RecordBatch,
) -> Result<Vec<(String, RecordBatch)>, String> {
    let observers = DataObserverBatch::try_from_nested_record_batch(batch)
        .map_err(|err| format!("failed to decode ObserverBatch: {err}"))?;
    let ids = observers.code.iter().map(|code| code.0.clone()).collect();
    let mut groups = first_appearance_groups(ids)?;
    groups.sort_by(|a, b| a.0.cmp(&b.0));
    take_record_batch_groups(batch, groups)
}

#[pyfunction]
fn group_observers_by_code_arrow<'py>(
    py: Python<'py>,
    observer_batch: &Bound<'py, PyAny>,
) -> PyResult<Vec<(String, PyObject)>> {
    let batch = RecordBatch::from_pyarrow_bound(observer_batch)
        .map_err(|err| PyValueError::new_err(format!("invalid Observers RecordBatch: {err}")))?;
    py.allow_threads(|| group_observers_by_code_record_batch(&batch))
        .map_err(PyValueError::new_err)?
        .into_iter()
        .map(|(code, grouped)| {
            grouped
                .to_pyarrow(py)
                .map(|value| (code, value))
                .map_err(|err| {
                    PyRuntimeError::new_err(format!(
                        "failed to export grouped Observers batch: {err}"
                    ))
                })
        })
        .collect()
}

fn cartesian_samples(values: &CoordinateValues, label: &str) -> Result<Vec<[f64; 6]>, String> {
    values
        .cartesian()
        .map(<[[f64; 6]]>::to_vec)
        .ok_or_else(|| format!("{label} coordinates must be Cartesian"))
}

fn variant_linkage_groups<'a>(
    variants: &'a DataOrbitVariantBatch,
    times: &TimeArray,
) -> HashMap<(&'a str, i64, i64), Vec<usize>> {
    let mut groups: HashMap<(&str, i64, i64), Vec<usize>> = HashMap::new();
    for (row, orbit_id) in variants.orbit_id.iter().enumerate() {
        let epoch = times.epochs[row];
        groups
            .entry((orbit_id.0.as_str(), epoch.days, epoch.nanos / 1_000_000))
            .or_default()
            .push(row);
    }
    groups
}

/// Rust-owned `VariantOrbits.collapse`: per-orbit weighted covariance from the
/// linked variants, with the mean state taken from the orbit rows.
fn collapse_variant_orbits_record_batch(
    orbit_batch: &RecordBatch,
    variant_batch: &RecordBatch,
) -> Result<RecordBatch, String> {
    let mut orbits = DataOrbitBatch::try_from_nested_record_batch(orbit_batch)
        .map_err(|err| format!("failed to decode OrbitBatch: {err}"))?;
    let variants = DataOrbitVariantBatch::try_from_nested_record_batch(variant_batch)
        .map_err(|err| format!("failed to decode OrbitVariantBatch: {err}"))?;
    let orbit_times = orbits
        .coordinates
        .times
        .clone()
        .ok_or_else(|| "orbit coordinates require times".to_string())?;
    let variant_times = variants
        .coordinates
        .times
        .as_ref()
        .ok_or_else(|| "variant coordinates require times".to_string())?;
    if orbit_times.scale != variant_times.scale {
        return Err("assertion failed: orbit and variant time scales must match".to_string());
    }
    let orbit_values = cartesian_samples(&orbits.coordinates.values, "orbit")?;
    let variant_values = cartesian_samples(&variants.coordinates.values, "variant")?;
    let groups = variant_linkage_groups(&variants, variant_times);

    let rows = orbits.len();
    let mut covariance_values = vec![0.0_f64; rows * 36];
    for row in 0..rows {
        let epoch = orbit_times.epochs[row];
        let key = (
            orbits.orbit_id[row].0.as_str(),
            epoch.days,
            epoch.nanos / 1_000_000,
        );
        let indices: &[usize] = groups.get(&key).map_or(&[], Vec::as_slice);
        let samples: Vec<f64> = indices
            .iter()
            .flat_map(|&index| variant_values[index].iter().copied())
            .collect();
        let weights: Vec<f64> = indices
            .iter()
            .map(|&index| variants.weights_cov[index].unwrap_or(f64::NAN))
            .collect();
        let covariance =
            weighted_covariance_flat(&orbit_values[row], &samples, &weights, indices.len(), 6);
        covariance_values[row * 36..(row + 1) * 36].copy_from_slice(&covariance);
    }
    let covariance = CovarianceBatch::new(
        rows,
        6,
        covariance_values,
        CovarianceUnits::Coordinate(DataRepresentation::Cartesian),
    )
    .map_err(|err| format!("failed to assemble covariance: {err}"))?;
    orbits.coordinates.covariance = Some(covariance);
    orbits
        .into_nested_record_batch()
        .map_err(|err| format!("failed to encode OrbitBatch: {err}"))
}

/// Rust-owned `VariantOrbits.collapse_by_object_id`: per-object uniform mean
/// and covariance grouped in first-appearance order.
fn collapse_variant_orbits_by_object_id_record_batch(
    variant_batch: &RecordBatch,
    orbit_ids: &[String],
) -> Result<RecordBatch, String> {
    let variants = DataOrbitVariantBatch::try_from_nested_record_batch(variant_batch)
        .map_err(|err| format!("failed to decode OrbitVariantBatch: {err}"))?;
    let times = variants
        .coordinates
        .times
        .as_ref()
        .ok_or_else(|| "variant coordinates require times".to_string())?;
    let variant_values = cartesian_samples(&variants.coordinates.values, "variant")?;

    let mut lookup: HashMap<Option<&str>, usize> = HashMap::new();
    let mut groups: Vec<(Option<String>, Vec<usize>)> = Vec::new();
    for (row, object_id) in variants.object_id.iter().enumerate() {
        let key = object_id.as_ref().map(|value| value.0.as_str());
        match lookup.get(&key) {
            Some(&group) => groups[group].1.push(row),
            None => {
                lookup.insert(key, groups.len());
                groups.push((object_id.as_ref().map(|value| value.0.clone()), vec![row]));
            }
        }
    }
    if groups.len() != orbit_ids.len() {
        return Err(format!(
            "expected {} collapsed orbit ids, got {}",
            groups.len(),
            orbit_ids.len()
        ));
    }

    let n_groups = groups.len();
    let mut means = Vec::with_capacity(n_groups);
    let mut covariance_values = vec![0.0_f64; n_groups * 36];
    let mut epochs = Vec::with_capacity(n_groups);
    let mut origins = Vec::with_capacity(n_groups);
    let mut object_ids = Vec::with_capacity(n_groups);
    let mut first_rows = Vec::with_capacity(n_groups);
    for (group, (object_id, rows)) in groups.into_iter().enumerate() {
        let unique_epochs: HashSet<Epoch> = rows.iter().map(|&row| times.epochs[row]).collect();
        if unique_epochs.len() != 1 {
            return Err(
                "assertion failed: all variants of an object must share a single epoch".to_string(),
            );
        }
        let unique_origins: HashSet<String> = rows
            .iter()
            .map(|&row| variants.coordinates.origins.origins[row].code())
            .collect();
        if unique_origins.len() != 1 {
            return Err(
                "assertion failed: all variants of an object must share a single origin"
                    .to_string(),
            );
        }
        let n = rows.len();
        let samples: Vec<f64> = rows
            .iter()
            .flat_map(|&row| variant_values[row].iter().copied())
            .collect();
        let uniform = vec![1.0 / n as f64; n];
        let mean = weighted_mean_flat(&samples, &uniform, n, 6);
        let covariance = weighted_covariance_flat(&mean, &samples, &uniform, n, 6);
        covariance_values[group * 36..(group + 1) * 36].copy_from_slice(&covariance);
        means.push([mean[0], mean[1], mean[2], mean[3], mean[4], mean[5]]);
        let first = rows[0];
        first_rows.push(first);
        epochs.push(times.epochs[first]);
        origins.push(variants.coordinates.origins.origins[first].clone());
        object_ids.push(object_id.map(adam_core_rs_coords::ObjectId));
    }

    let covariance = CovarianceBatch::new(
        n_groups,
        6,
        covariance_values,
        CovarianceUnits::Coordinate(DataRepresentation::Cartesian),
    )
    .map_err(|err| format!("failed to assemble covariance: {err}"))?;
    let coordinates = DataCoordinateBatch::cartesian(
        means,
        variants.coordinates.frame,
        OriginArray::new(origins),
        Some(
            TimeArray::new(times.scale, epochs)
                .map_err(|err| format!("failed to assemble times: {err}"))?,
        ),
        Some(covariance),
    )
    .map_err(|err| format!("failed to assemble coordinates: {err}"))?;
    let mut collapsed = DataOrbitBatch::new(
        orbit_ids
            .iter()
            .map(|orbit_id| adam_core_rs_coords::OrbitId(orbit_id.clone()))
            .collect(),
        object_ids,
        coordinates,
    )
    .map_err(|err| format!("failed to assemble OrbitBatch: {err}"))?;
    if let Some(physical_parameters) = &variants.physical_parameters {
        collapsed = collapsed
            .with_physical_parameters(physical_parameters.take(&first_rows))
            .map_err(|err| format!("failed to attach physical parameters: {err}"))?;
    }
    collapsed
        .into_nested_record_batch()
        .map_err(|err| format!("failed to encode OrbitBatch: {err}"))
}

#[pyfunction]
fn collapse_variant_orbits_arrow(
    py: Python<'_>,
    orbit_batch: &Bound<'_, PyAny>,
    variant_batch: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let orbits = RecordBatch::from_pyarrow_bound(orbit_batch)
        .map_err(|err| PyValueError::new_err(format!("invalid Orbits RecordBatch: {err}")))?;
    let variants = RecordBatch::from_pyarrow_bound(variant_batch).map_err(|err| {
        PyValueError::new_err(format!("invalid VariantOrbits RecordBatch: {err}"))
    })?;
    let output = py
        .allow_threads(|| collapse_variant_orbits_record_batch(&orbits, &variants))
        .map_err(PyValueError::new_err)?;
    output
        .to_pyarrow(py)
        .map_err(|err| PyRuntimeError::new_err(format!("failed to export RecordBatch: {err}")))
}

#[pyfunction]
fn collapse_variant_orbits_by_object_id_arrow(
    py: Python<'_>,
    variant_batch: &Bound<'_, PyAny>,
    orbit_ids: Vec<String>,
) -> PyResult<PyObject> {
    let variants = RecordBatch::from_pyarrow_bound(variant_batch).map_err(|err| {
        PyValueError::new_err(format!("invalid VariantOrbits RecordBatch: {err}"))
    })?;
    let output = py
        .allow_threads(|| collapse_variant_orbits_by_object_id_record_batch(&variants, &orbit_ids))
        .map_err(PyValueError::new_err)?;
    output
        .to_pyarrow(py)
        .map_err(|err| PyRuntimeError::new_err(format!("failed to export RecordBatch: {err}")))
}

#[pyfunction]
#[pyo3(signature = (orbit_batch, variant_batch, reps, trials, warmup_reps=1))]
fn benchmark_collapse_variant_orbits_arrow(
    orbit_batch: &Bound<'_, PyAny>,
    variant_batch: &Bound<'_, PyAny>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let orbits = RecordBatch::from_pyarrow_bound(orbit_batch)
        .map_err(|err| PyValueError::new_err(format!("invalid Orbits RecordBatch: {err}")))?;
    let variants = RecordBatch::from_pyarrow_bound(variant_batch).map_err(|err| {
        PyValueError::new_err(format!("invalid VariantOrbits RecordBatch: {err}"))
    })?;
    benchmark_orbit_surface(reps, trials, warmup_reps, || {
        black_box(collapse_variant_orbits_record_batch(&orbits, &variants)?);
        Ok(())
    })
}

#[pyfunction]
#[pyo3(signature = (variant_batch, orbit_ids, reps, trials, warmup_reps=1))]
fn benchmark_collapse_variant_orbits_by_object_id_arrow(
    variant_batch: &Bound<'_, PyAny>,
    orbit_ids: Vec<String>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let variants = RecordBatch::from_pyarrow_bound(variant_batch).map_err(|err| {
        PyValueError::new_err(format!("invalid VariantOrbits RecordBatch: {err}"))
    })?;
    benchmark_orbit_surface(reps, trials, warmup_reps, || {
        black_box(collapse_variant_orbits_by_object_id_record_batch(
            &variants, &orbit_ids,
        )?);
        Ok(())
    })
}

const ORBIT_CLASS_NAMES: [&str; 14] = [
    "AST", "AMO", "APO", "ATE", "CEN", "IEO", "IMB", "MBA", "MCA", "OMB", "TJN", "TNO", "PAA",
    "HYA",
];

fn dynamical_class_record_batch(batch: &RecordBatch) -> Result<Vec<String>, String> {
    let orbits = DataOrbitBatch::try_from_nested_record_batch(batch)
        .map_err(|err| format!("failed to decode OrbitBatch: {err}"))?;
    let times =
        orbits.coordinates.times.as_ref().ok_or_else(|| {
            "orbit dynamical classification requires coordinate times".to_string()
        })?;
    let t0 = times.mjd_values();
    let mu = orbits
        .coordinates
        .origins
        .origins
        .iter()
        .map(origin_mu_au3_day2)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("failed to resolve origin mu: {err}"))?;
    let values = orbits
        .coordinates
        .values
        .cartesian()
        .ok_or_else(|| "OrbitBatch coordinates must be Cartesian".to_string())?;
    let flat = values.iter().flatten().copied().collect::<Vec<_>>();
    let keplerian = cartesian_to_keplerian_flat6(&flat, &t0, &mu);
    let mut a = Vec::with_capacity(values.len());
    let mut e = Vec::with_capacity(values.len());
    let mut q = Vec::with_capacity(values.len());
    let mut q_apo = Vec::with_capacity(values.len());
    for row in keplerian.chunks_exact(13) {
        let semi_major_axis = row[0];
        let eccentricity = row[4];
        a.push(semi_major_axis);
        e.push(eccentricity);
        q.push(semi_major_axis * (1.0 - eccentricity));
        q_apo.push(semi_major_axis * (1.0 + eccentricity));
    }
    classify_orbits_flat(&a, &e, &q, &q_apo)
        .into_iter()
        .map(|code| {
            usize::try_from(code)
                .ok()
                .and_then(|index| ORBIT_CLASS_NAMES.get(index))
                .map(|name| (*name).to_string())
                .ok_or_else(|| format!("unknown orbit class code: {code}"))
        })
        .collect()
}

#[pyfunction]
fn group_by_orbit_id_arrow<'py>(
    py: Python<'py>,
    orbit_batch: &Bound<'py, PyAny>,
) -> PyResult<Vec<(String, PyObject)>> {
    let batch = RecordBatch::from_pyarrow_bound(orbit_batch)
        .map_err(|err| PyValueError::new_err(format!("invalid Orbits RecordBatch: {err}")))?;
    py.allow_threads(|| group_by_orbit_id_record_batch(&batch))
        .map_err(PyValueError::new_err)?
        .into_iter()
        .map(|(id, grouped)| {
            grouped
                .to_pyarrow(py)
                .map(|value| (id, value))
                .map_err(|err| {
                    PyRuntimeError::new_err(format!("failed to export grouped OrbitBatch: {err}"))
                })
        })
        .collect()
}

#[pyfunction]
fn dynamical_class_arrow(orbit_batch: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let batch = RecordBatch::from_pyarrow_bound(orbit_batch)
        .map_err(|err| PyValueError::new_err(format!("invalid Orbits RecordBatch: {err}")))?;
    dynamical_class_record_batch(&batch).map_err(PyValueError::new_err)
}

fn benchmark_orbit_surface<F>(
    reps: usize,
    trials: usize,
    warmup_reps: usize,
    mut run_once: F,
) -> PyResult<Vec<Vec<f64>>>
where
    F: FnMut() -> Result<(), String>,
{
    if reps == 0 || trials == 0 {
        return Err(PyValueError::new_err("reps and trials must be >= 1"));
    }
    let mut trial_samples = Vec::with_capacity(trials);
    for _ in 0..trials {
        for _ in 0..warmup_reps {
            run_once().map_err(PyValueError::new_err)?;
        }
        let mut samples = Vec::with_capacity(reps);
        for _ in 0..reps {
            let started = Instant::now();
            run_once().map_err(PyValueError::new_err)?;
            samples.push(started.elapsed().as_secs_f64());
        }
        trial_samples.push(samples);
    }
    Ok(trial_samples)
}

#[pyfunction]
#[pyo3(signature = (orbit_batch, reps, trials, warmup_reps=1))]
fn benchmark_group_by_orbit_id_arrow(
    orbit_batch: &Bound<'_, PyAny>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let batch = RecordBatch::from_pyarrow_bound(orbit_batch)
        .map_err(|err| PyValueError::new_err(format!("invalid Orbits RecordBatch: {err}")))?;
    benchmark_orbit_surface(reps, trials, warmup_reps, || {
        black_box(group_by_orbit_id_record_batch(&batch)?);
        Ok(())
    })
}

#[pyfunction]
#[pyo3(signature = (orbit_batch, reps, trials, warmup_reps=1))]
fn benchmark_dynamical_class_arrow(
    orbit_batch: &Bound<'_, PyAny>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let batch = RecordBatch::from_pyarrow_bound(orbit_batch)
        .map_err(|err| PyValueError::new_err(format!("invalid Orbits RecordBatch: {err}")))?;
    benchmark_orbit_surface(reps, trials, warmup_reps, || {
        black_box(dynamical_class_record_batch(&batch)?);
        Ok(())
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cartesian_coordinate_schema_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(
        prepare_ephemeris_observer_linkage_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        benchmark_prepare_ephemeris_observer_linkage_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(group_by_orbit_id_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(group_observers_by_code_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(collapse_variant_orbits_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(
        collapse_variant_orbits_by_object_id_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        benchmark_collapse_variant_orbits_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        benchmark_collapse_variant_orbits_by_object_id_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(dynamical_class_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_group_by_orbit_id_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_dynamical_class_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(transform_coordinates_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_transform_coordinates_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_perturber_moids_native, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_perturber_moids_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(
        benchmark_calculate_perturber_moids_arrow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(residuals_calculate_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_residuals_calculate_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(sample_orbit_variants_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_sample_orbit_variants_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(generate_porkchop_data_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(lambert_solution_orbit_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_lambert_solution_orbit_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_generate_porkchop_data_arrow, m)?)?;
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
    m.add_function(wrap_pyfunction!(
        bound_longitude_residual_column_in_place_numpy,
        m
    )?)?;
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
    m.add_function(wrap_pyfunction!(bandpasses_load_table, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_bandpasses_load_table, m)?)?;
    m.add_function(wrap_pyfunction!(bandpasses_filter_ids, m)?)?;
    m.add_function(wrap_pyfunction!(bandpasses_get_integrals, m)?)?;
    m.add_function(wrap_pyfunction!(bandpasses_compute_mix_integrals, m)?)?;
    m.add_function(wrap_pyfunction!(bandpasses_delta_table, m)?)?;
    m.add_function(wrap_pyfunction!(bandpasses_composition_key, m)?)?;
    m.add_function(wrap_pyfunction!(bandpasses_convert_magnitude, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_bandpasses_convert_magnitude, m)?)?;
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
    m.add_function(wrap_pyfunction!(unpack_mpc_dates_isot, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_unpack_mpc_dates_isot, m)?)?;
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
