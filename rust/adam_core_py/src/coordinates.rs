use adam_core_rs_coords::{
    apply_cosine_latitude_correction_flat, bound_longitude_residuals_flat, calculate_chi2_flat,
    cartesian_to_cometary_flat6, cartesian_to_geodetic_flat6, cartesian_to_keplerian_flat6,
    cartesian_to_spherical_flat6, cartesian_to_spherical_row, classify_orbits_flat,
    cometary_to_cartesian_flat6, keplerian_to_cartesian_flat6, rotate_cartesian_frame_flat6,
    rotate_cartesian_time_varying_flat6, spherical_to_cartesian_flat6, spherical_to_cartesian_row,
    tisserand_parameter_flat, transform_with_covariance_flat6, weighted_covariance_flat,
    weighted_mean_flat, ArrowSchemaExport, CoordinateBatch as DataCoordinateBatch, Frame,
    IntoNestedRecordBatch, OrbitBatch as DataOrbitBatch, Representation as CoordsRepresentation,
    TryFromNestedRecordBatch,
};
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
    Ok((
        coords_shaped.into_pyarray_bound(py),
        cov_shaped.into_pyarray_bound(py),
    ))
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
        Some(shaped.into_pyarray_bound(py))
    } else {
        None
    };
    Ok((coords_shaped.into_pyarray_bound(py), cov_out))
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
        Ok(out) => Ok(out.into_pyarray_bound(py)),
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
                res_arr.into_pyarray_bound(py),
                chi2_arr.into_pyarray_bound(py),
                dof_arr.into_pyarray_bound(py),
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
    Ok(PyBytes::new_bound(py, &bytes))
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
    m.add_function(wrap_pyfunction!(apply_cosine_latitude_correction_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_chi2_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(compute_residuals_chi2_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_mean_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_covariance_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(classify_orbits_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(tisserand_parameter_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(orbits_nested_ipc_round_trip, m)?)?;
    Ok(())
}
