//! numpy entry points for the local orbital frame kernels in
//! `adam_core_rs_coords::local_frames`: frame name parsing, Jacobians and
//! covariance rotation.

use adam_core_rs_coords::local_frames::{
    local_frame_covariances, local_frame_jacobians, LocalFrame,
};
use adam_core_rs_coords::SchemaError;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn schema_error(err: SchemaError) -> PyErr {
    PyValueError::new_err(match err {
        SchemaError::InvalidRecordBatch(message) => message,
        other => other.to_string(),
    })
}

fn rows6(values: &PyReadonlyArray2<'_, f64>, label: &str) -> PyResult<Vec<f64>> {
    let view = values.as_array();
    if view.ncols() != 6 {
        return Err(PyValueError::new_err(format!(
            "{label} must have shape (N, 6)"
        )));
    }
    view.as_slice()
        .map(<[f64]>::to_vec)
        .ok_or_else(|| PyValueError::new_err(format!("{label} must be contiguous")))
}

fn scalars(values: &PyReadonlyArray1<'_, f64>, label: &str) -> PyResult<Vec<f64>> {
    values
        .as_array()
        .as_slice()
        .map(<[f64]>::to_vec)
        .ok_or_else(|| PyValueError::new_err(format!("{label} must be contiguous")))
}

fn matrices6(values: &PyReadonlyArray3<'_, f64>, label: &str) -> PyResult<Vec<f64>> {
    let view = values.as_array();
    if view.shape()[1..] != [6, 6] {
        return Err(PyValueError::new_err(format!(
            "{label} must have shape (N, 6, 6)"
        )));
    }
    view.as_slice()
        .map(<[f64]>::to_vec)
        .ok_or_else(|| PyValueError::new_err(format!("{label} must be contiguous")))
}

fn array3<'py>(py: Python<'py>, values: Vec<f64>) -> PyResult<Bound<'py, PyArray3<f64>>> {
    ndarray::Array3::from_shape_vec((values.len() / 36, 6, 6), values)
        .map(|array| array.into_pyarray(py))
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

/// (N, 6, 6) Jacobians from inertial (N, 6) states to `frame`; `mu` (N,)
/// in AU^3/day^2 is only read for `_ROTATING` frames.
#[pyfunction]
fn local_frame_jacobians_numpy<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    mu: PyReadonlyArray1<'py, f64>,
    frame: &str,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let frame = LocalFrame::parse(frame).map_err(schema_error)?;
    let values = rows6(&values, "values")?;
    let mu = scalars(&mu, "mu")?;
    array3(
        py,
        local_frame_jacobians(&values, &mu, frame).map_err(schema_error)?,
    )
}

/// (N, 6, 6) covariances `J C J^T` in `frame`, from inertial (N, 6) states
/// and (N, 6, 6) covariances in AU and AU/day.
#[pyfunction]
fn local_frame_covariances_numpy<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    covariances: PyReadonlyArray3<'py, f64>,
    mu: PyReadonlyArray1<'py, f64>,
    frame: &str,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let frame = LocalFrame::parse(frame).map_err(schema_error)?;
    let values = rows6(&values, "values")?;
    let covariances = matrices6(&covariances, "covariances")?;
    let mu = scalars(&mu, "mu")?;
    array3(
        py,
        local_frame_covariances(&values, &covariances, &mu, frame).map_err(schema_error)?,
    )
}

/// Canonical registry name of a frame name or alias, e.g. `rtn` -> `RSW_INERTIAL`.
#[pyfunction]
fn local_frame_canonical_name(frame: &str) -> PyResult<String> {
    LocalFrame::parse(frame)
        .map(|frame| frame.canonical_name().to_string())
        .map_err(schema_error)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(local_frame_jacobians_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(local_frame_covariances_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(local_frame_canonical_name, m)?)?;
    Ok(())
}
