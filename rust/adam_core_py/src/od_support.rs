//! Shared support for the native orbit-determination drivers.
//!
//! The drivers never call Python for predictions.  This module only adapts
//! adam-core's process-global SPICE provider and maps native driver errors to
//! Python exceptions at the single outer binding boundary.

use adam_core_rs_coords::propagation::PropagationError;
use adam_core_rs_coords::types::{Frame, SchemaError, SchemaResult};
use adam_core_rs_coords::{OriginArray, OriginId, OriginTranslationProvider, TimeArray};
use adam_core_rs_spice::global_backend;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// `OriginTranslationProvider` over the process-global SPICE backend.  The
/// lock is held only for one translation call, never across propagation.
pub(crate) struct LockingSpiceTranslation;

impl OriginTranslationProvider for LockingSpiceTranslation {
    fn origin_translation_vectors(
        &self,
        origins: &OriginArray,
        target_origin: &OriginId,
        frame: Frame,
        times: &TimeArray,
    ) -> SchemaResult<Vec<[f64; 6]>> {
        let backend = global_backend().lock().map_err(|_| {
            SchemaError::InvalidRecordBatch("SPICE backend lock is poisoned".to_string())
        })?;
        OriginTranslationProvider::origin_translation_vectors(
            &*backend,
            origins,
            target_origin,
            frame,
            times,
        )
    }
}

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
