//! Shared support for the native orbit-determination drivers.
//!
//! The drivers never call Python for predictions. This module maps native
//! driver errors to Python exceptions at the single outer binding boundary.

use adam_core_rs_coords::propagation::PropagationError;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Map a native driver error at the single Python binding boundary.
pub(crate) fn od_error(err: PropagationError) -> PyErr {
    match err {
        PropagationError::InvalidRequest(_)
        | PropagationError::Schema(_)
        | PropagationError::MissingOrbitTimes
        | PropagationError::UnsupportedCovarianceMode(_) => PyValueError::new_err(err.to_string()),
        PropagationError::CmcCovarianceUnavailable(_)
        | PropagationError::GroupScientific { .. }
        | PropagationError::RowFailure(_)
        | PropagationError::Backend(_)
        | PropagationError::BackendProtocol(_)
        | PropagationError::ThreadPool(_) => PyRuntimeError::new_err(err.to_string()),
    }
}
