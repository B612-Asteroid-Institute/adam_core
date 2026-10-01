//! The callback route of the Rust orbit-determination drivers: any Python
//! propagator becomes a [`SphericalPredictor`] whose predictions come from
//! its own `generate_ephemeris`, so the Rust loops (fit, rejection, IOD,
//! full OD) drive it without a Python implementation of the loop. A
//! propagator with a native Rust backend keeps the one-crossing route (its
//! fused work-unit methods); this is the general route every propagator gets.
//!
//! Candidate orbits and observers cross as nested Arrow IPC; predicted
//! spherical coordinates come back as an `(M * N, 6)` NumPy array in
//! candidate-major, observer order (see
//! `adam_core.orbit_determination._native_callback`). A Python exception
//! raised by the propagator is kept and re-raised unchanged by the binding
//! once the driver has unwound.

use crate::coordinates::write_orbit_ipc;
use adam_core_rs_coords::propagation::{
    candidate_orbit_batch, OrbitGeometry, PropagationError, PropagationResultValue,
    SphericalPredictor,
};
use adam_core_rs_coords::types::{Frame, SchemaError, SchemaResult};
use adam_core_rs_coords::{
    IntoNestedRecordBatch, ObserverBatch, OriginArray, OriginId, OriginTranslationProvider,
    TimeArray,
};
use adam_core_rs_spice::global_backend;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Mutex;

const CALLBACK_MODULE: &str = "adam_core.orbit_determination._native_callback";

/// A Python propagator (anything with `generate_ephemeris(orbits, observers,
/// max_processes=1)`) as a [`SphericalPredictor`].
pub(crate) struct PyEphemerisPredictor {
    propagator: Py<PyAny>,
    helper: Py<PyAny>,
    error: Mutex<Option<PyErr>>,
}

impl PyEphemerisPredictor {
    pub(crate) fn new(py: Python<'_>, propagator: &Bound<'_, PyAny>) -> PyResult<Self> {
        let helper = py
            .import(CALLBACK_MODULE)?
            .getattr("predict_spherical")?
            .unbind();
        Ok(Self {
            propagator: propagator.clone().unbind(),
            helper,
            error: Mutex::new(None),
        })
    }

    /// The Python exception raised by the propagator during the last failed
    /// prediction, if any (cleared on every prediction).
    pub(crate) fn take_error(&self) -> Option<PyErr> {
        self.error.lock().ok().and_then(|mut slot| slot.take())
    }

    fn store_error(&self, err: PyErr) {
        if let Ok(mut slot) = self.error.lock() {
            *slot = Some(err);
        }
    }
}

fn ipc_bytes(batch: arrow_array::RecordBatch) -> PropagationResultValue<Vec<u8>> {
    write_orbit_ipc(&batch).map_err(|err| PropagationError::Backend(err.to_string()))
}

impl SphericalPredictor for PyEphemerisPredictor {
    fn predict_spherical(
        &self,
        states: &[[f64; 6]],
        geometry: &OrbitGeometry,
        observers: &ObserverBatch,
    ) -> PropagationResultValue<Result<Vec<f64>, String>> {
        if let Ok(mut slot) = self.error.lock() {
            slot.take();
        }
        let m = states.len();
        let n = observers.len();
        let orbits_ipc =
            ipc_bytes(candidate_orbit_batch(states, geometry)?.into_nested_record_batch()?)?;
        let observers_ipc = ipc_bytes(observers.clone().into_nested_record_batch()?)?;
        let outcome = Python::with_gil(|py| -> PyResult<Vec<f64>> {
            let result = self.helper.bind(py).call1((
                self.propagator.bind(py),
                PyBytes::new(py, &orbits_ipc),
                PyBytes::new(py, &observers_ipc),
                m,
                n,
            ))?;
            let array: PyReadonlyArray2<'_, f64> = result.extract()?;
            let view = array.as_array();
            if view.shape() != [m * n, 6] {
                return Err(PyValueError::new_err(format!(
                    "predict_spherical returned shape {:?}, expected ({}, 6)",
                    view.shape(),
                    m * n
                )));
            }
            Ok(view.iter().copied().collect())
        });
        match outcome {
            Ok(values) => Ok(Ok(values)),
            Err(err) => {
                let message = err.to_string();
                self.store_error(err);
                Err(PropagationError::Backend(format!(
                    "the Python propagator raised an exception: {message}"
                )))
            }
        }
    }
}

/// `OriginTranslationProvider` over the process-global SPICE backend that
/// takes the backend lock only for the duration of one call, so a Python
/// callback running between calls (which may itself use the backend) cannot
/// deadlock against a driver holding the lock.
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

/// Map a driver error onto the Python exception the facade raises: the
/// propagator's own exception when the callback raised one, `ValueError`
/// for invalid inputs, `RuntimeError` for backend failures.
pub(crate) fn od_error(err: PropagationError, predictor: Option<&PyEphemerisPredictor>) -> PyErr {
    if let Some(py_err) = predictor.and_then(PyEphemerisPredictor::take_error) {
        return py_err;
    }
    match err {
        PropagationError::InvalidRequest(_)
        | PropagationError::Schema(_)
        | PropagationError::MissingOrbitTimes
        | PropagationError::UnsupportedCovarianceMode(_) => PyValueError::new_err(err.to_string()),
        PropagationError::Backend(_)
        | PropagationError::BackendProtocol(_)
        | PropagationError::ThreadPool(_) => PyRuntimeError::new_err(err.to_string()),
    }
}
