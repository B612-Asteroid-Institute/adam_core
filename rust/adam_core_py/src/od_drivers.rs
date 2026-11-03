//! PyO3 surface of backend-generic Rust OD stages: low-level fit and rejection
//! kernels plus complete typed IOD/refinement/configured-pipeline routes over a
//! capsule-backed native propagator or adam-core's Rust two-body propagator.
//! Python objects are never called for predictions: a separately built
//! backend supplies only the propagation ABI and adam-core owns every OD
//! loop and public operation.
//!
//! Inputs cross as nested Arrow IPC (the `arrow_bridge` helpers), complete
//! typed stage configurations for configured workflows, and dictionaries only
//! for retained expert numerical kernels. Outputs are plain dictionaries of
//! scalars, lists, and NumPy arrays that the facade wraps into `FittedOrbits` /
//! `FittedOrbitMembers` without further computation. Driver errors map to
//! `ValueError` (invalid input) or `RuntimeError` (backend failure).

use crate::coordinates::{read_orbit_ipc, write_orbit_ipc};
use crate::od_config::{PyIodRunConfig, PyOdConfig, PyRefinementRunConfig};
use crate::od_support::od_error;
use adam_core::od::algorithms::{
    canonical_stage_settings, external_provider_metadata, two_body_provider_metadata,
    ErfaTimeProvider, GlobalSpiceTranslation as LockingSpiceTranslation,
};
use adam_core::od::{
    AssociatedObservations, AstrometryColumns, FittedOrbitMembersBatch, MemberedOrbits,
    OdBatchError, OdBatchResult, OdGroupResult, OdObservationBatch, OdPropagator,
    OdPropagatorSource, OdProvenance, OdRoute, ProviderMetadata,
};
use adam_core_rs_coords::od::algorithms::{
    prepare_observations, refine_with_predictor, run_od_with_predictors, PreparedObservations,
};
use adam_core_rs_coords::propagation::adapter::{
    cmc2003_fit_with, fit_orbit_whitened_with, iterative_fit_with, refine_with, run_iod_with,
    validate_fit_covariance_with, PropagatorPredictor, SphericalPredictor,
};
use adam_core_rs_coords::propagation::{
    AstrometrySnapshot, BackendIdentity, Cmc2003FitConfig, EphemerisOptions, ExternalPropagator,
    ExternalPropagatorV2, IodCandidate, IodCandidateFailure, IodResult, IterativeFitConfig,
    JacobianMethod, PropagationResultValue, RangingResult, RangingSample, RefinementConfig,
    TwoBodyPropagator, WhitenedFitConfig, WhitenedFitOutput, EXTERNAL_PROPAGATOR_CAPSULE,
};
use adam_core_rs_coords::{
    validate_loss, AssociationBatch, CoordinateBatch as DataCoordinateBatch, IntoNestedRecordBatch,
    ObserverBatch as DataObserverBatch, OdConfig, OrbitBatch as DataOrbitBatch, SourceCatalogBatch,
    TryFromNestedRecordBatch, TwoBodyModelConfig, CMC2003_APPARITION_GAP_DAYS, CMC2003_CHI2_FRAC,
    CMC2003_CHI2_RECOVER, CMC2003_CHI2_REJECT, CMC2003_MAX_ITERATIONS,
    CMC2003_MAX_REJECTED_FRACTION, CMC2003_PSD_FLOOR_FRAC,
};
use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyCapsule, PyDict, PyList, PyTuple};
use serde::Deserialize;
use std::hint::black_box;
use std::sync::Mutex;
use std::time::Instant;

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

struct OdObservations {
    observed: DataCoordinateBatch,
    observers: DataObserverBatch,
}

fn decode_observations(
    observed_ipc: &Bound<'_, PyBytes>,
    observers_ipc: &Bound<'_, PyBytes>,
) -> PyResult<OdObservations> {
    let observed = DataCoordinateBatch::try_from_nested_record_batch(&read_orbit_ipc(
        observed_ipc.as_bytes(),
    )?)
    .map_err(|err| value_error(format!("failed to decode observed coordinates: {err}")))?;
    let observers =
        DataObserverBatch::try_from_nested_record_batch(&read_orbit_ipc(observers_ipc.as_bytes())?)
            .map_err(|err| value_error(format!("failed to decode ObserverBatch: {err}")))?;
    if observers.len() != observed.len() {
        return Err(value_error(
            "observed coordinates and observers must have equal length",
        ));
    }
    Ok(OdObservations {
        observed,
        observers,
    })
}

fn decode_orbit(orbit_ipc: &Bound<'_, PyBytes>) -> PyResult<DataOrbitBatch> {
    DataOrbitBatch::try_from_nested_record_batch(&read_orbit_ipc(orbit_ipc.as_bytes())?)
        .map_err(|err| value_error(format!("failed to decode OrbitBatch: {err}")))
}

/// The whitened-fit options of every work unit, read from the
/// ``fit_settings`` dict with `fit_least_squares` / two-body defaults.
struct FitSettings {
    config: WhitenedFitConfig,
    options: EphemerisOptions,
    propagator: TwoBodyPropagator,
}

fn setting<'py, T: FromPyObject<'py>>(
    settings: Option<&Bound<'py, PyDict>>,
    key: &str,
    default: T,
) -> PyResult<T> {
    match settings {
        Some(dict) => match dict.get_item(key)? {
            Some(value) if !value.is_none() => value.extract::<T>(),
            _ => Ok(default),
        },
        None => Ok(default),
    }
}

/// The exact pre-0.7 defaults of the dict-driven public fit routes.
///
/// Their fit and propagation defaults came from [`WhitenedFitConfig`], while
/// their light-time/ephemeris defaults came from [`EphemerisOptions`]. Keep
/// that split explicit: [`TwoBodyModelConfig::default`] has different values.
fn dict_fit_defaults() -> WhitenedFitConfig {
    let mut defaults = WhitenedFitConfig::default();
    let ephemeris = EphemerisOptions::default();
    defaults.two_body.lt_tol = ephemeris.lt_tol;
    defaults.two_body.ephemeris_max_iter = ephemeris.max_iter;
    defaults.two_body.max_lt_iter = ephemeris.max_lt_iter;
    defaults
}

/// Build one typed fit configuration from dictionary overrides, then derive
/// all runtime settings through Core's canonical stage constructor. The legacy
/// stellar-aberration knob is the sole explicit post-construction difference.
fn parse_fit_settings(settings: Option<&Bound<'_, PyDict>>) -> PyResult<FitSettings> {
    let fit_defaults = dict_fit_defaults();
    let ephemeris_defaults = EphemerisOptions::default();
    let loss: String = setting(settings, "loss", fit_defaults.loss.as_str().to_string())?;
    let f_scale: f64 = setting(settings, "f_scale", fit_defaults.f_scale)?;
    let loss = validate_loss(&loss, f_scale).map_err(value_error)?;
    let jacobian: String = setting(
        settings,
        "jacobian",
        fit_defaults.jacobian.as_str().to_string(),
    )?;
    let jacobian = JacobianMethod::parse(&jacobian).map_err(value_error)?;
    let lt_tol: f64 = setting(settings, "lt_tol", fit_defaults.two_body.lt_tol)?;
    let eph_max_iter: usize = setting(
        settings,
        "eph_max_iter",
        fit_defaults.two_body.ephemeris_max_iter,
    )?;
    let eph_tol: f64 = setting(settings, "eph_tol", fit_defaults.two_body.ephemeris_tol)?;
    let stellar_aberration: bool = setting(
        settings,
        "stellar_aberration",
        ephemeris_defaults.stellar_aberration,
    )?;
    let max_lt_iter: usize = setting(settings, "max_lt_iter", fit_defaults.two_body.max_lt_iter)?;
    let prop_max_iter: usize = setting(
        settings,
        "prop_max_iter",
        fit_defaults.two_body.propagation_max_iter,
    )?;
    let prop_tol: f64 = setting(settings, "prop_tol", fit_defaults.two_body.propagation_tol)?;
    let config = WhitenedFitConfig {
        loss,
        f_scale,
        jacobian,
        validate_covariance: setting(
            settings,
            "validate_covariance",
            fit_defaults.validate_covariance,
        )?,
        xtol: setting(settings, "xtol", fit_defaults.xtol)?,
        ftol: setting(settings, "ftol", fit_defaults.ftol)?,
        gtol: setting(settings, "gtol", fit_defaults.gtol)?,
        max_iterations: setting(settings, "max_iterations", fit_defaults.max_iterations)?,
        max_nfev: setting(settings, "max_nfev", fit_defaults.max_nfev)?,
        two_body: TwoBodyModelConfig {
            propagation_max_iter: prop_max_iter,
            propagation_tol: prop_tol,
            lt_tol,
            ephemeris_max_iter: eph_max_iter,
            ephemeris_tol: eph_tol,
            max_lt_iter,
        },
    };
    let mut runtime = fit_settings_from_config(config)?;
    runtime.options.stellar_aberration = stellar_aberration;
    Ok(runtime)
}

/// Build the fixed ephemeris/provider adapter around an authoritative typed
/// fit configuration. Every configurable two-body tolerance/budget is copied
/// from that configuration; only non-configurable OD conventions remain fixed
/// here (UTC output, no stellar aberration, no propagated covariance).
fn fit_settings_from_config(config: WhitenedFitConfig) -> PyResult<FitSettings> {
    let runtime = canonical_stage_settings(config).map_err(batch_error)?;
    Ok(FitSettings {
        config,
        options: runtime.options,
        propagator: runtime.two_body,
    })
}

/// The refinement defaults of the Rust config types: ``fit``
/// (`WhitenedFitConfig::default()`), ``worst_residual`` (the loop settings of
/// `IterativeFitConfig::default()`), ``cmc2003`` (the loop settings of
/// `Cmc2003FitConfig::default()`, OrbFit's ``reject.def``) and ``refinement``
/// (the method of `RefinementConfig::default()`): the one source of the
/// Python signature defaults of the refinement entry points.
#[pyfunction]
pub fn od_defaults(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let fit = WhitenedFitConfig::default();
    let fit_dict = PyDict::new(py);
    fit_dict.set_item("loss", fit.loss.as_str())?;
    fit_dict.set_item("f_scale", fit.f_scale)?;
    fit_dict.set_item("jacobian", fit.jacobian.as_str())?;
    fit_dict.set_item("validate_covariance", fit.validate_covariance)?;
    fit_dict.set_item("xtol", fit.xtol)?;
    fit_dict.set_item("ftol", fit.ftol)?;
    fit_dict.set_item("gtol", fit.gtol)?;
    fit_dict.set_item("max_iterations", fit.max_iterations)?;
    fit_dict.set_item("max_nfev", fit.max_nfev)?;
    let iterative = IterativeFitConfig::default();
    let worst = PyDict::new(py);
    worst.set_item("rchi2_threshold", iterative.rchi2_threshold)?;
    worst.set_item("min_obs", iterative.min_obs)?;
    worst.set_item("min_arc_length", iterative.min_arc_length)?;
    worst.set_item(
        "contamination_percentage",
        iterative.contamination_percentage,
    )?;
    let cmc = Cmc2003FitConfig::default();
    let cmc_dict = PyDict::new(py);
    cmc_dict.set_item("chi2_reject", cmc.chi2_reject)?;
    cmc_dict.set_item("chi2_recover", cmc.chi2_recover)?;
    cmc_dict.set_item("chi2_frac", cmc.chi2_frac)?;
    cmc_dict.set_item("max_iterations", cmc.max_iterations)?;
    cmc_dict.set_item("max_rejected_fraction", cmc.max_rejected_fraction)?;
    cmc_dict.set_item("apparition_gap_days", cmc.apparition_gap_days)?;
    cmc_dict.set_item("psd_floor_frac", cmc.psd_floor_frac)?;
    let out = PyDict::new(py);
    out.set_item("fit", fit_dict)?;
    out.set_item("worst_residual", worst)?;
    out.set_item("cmc2003", cmc_dict)?;
    out.set_item(
        "refinement",
        match RefinementConfig::default() {
            RefinementConfig::WorstResidual(_) => "worst_residual",
            RefinementConfig::Cmc2003(_) => "cmc2003",
        },
    )?;
    Ok(out)
}

fn covariance_36(covariance: &PyReadonlyArray2<'_, f64>) -> PyResult<[f64; 36]> {
    let view = covariance.as_array();
    if view.shape() != [6, 6] {
        return Err(value_error("covariance must have shape (6, 6)"));
    }
    let mut out = [0.0_f64; 36];
    for (slot, value) in out.iter_mut().zip(view.iter()) {
        *slot = *value;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Running a driver over the chosen backend
// ---------------------------------------------------------------------------

fn validate_expected_identity(
    actual: &BackendIdentity,
    expected_name: &str,
    expected_version: &str,
    expected_two_body: bool,
) -> Result<(), String> {
    if actual.name == expected_name
        && actual.version == expected_version
        && actual.two_body_dynamics == expected_two_body
    {
        return Ok(());
    }
    Err(format!(
        concat!(
            "native propagator identity mismatch: declaration is {} {} ",
            "(two_body={}), capsule is {} {} (two_body={})"
        ),
        expected_name,
        expected_version,
        expected_two_body,
        actual.name,
        actual.version,
        actual.two_body_dynamics
    ))
}

/// A capsule-backed native propagator handed to a driver.  The retained
/// Python object owns the table and its context for the whole native call.
struct PySource {
    propagator: ExternalPropagator,
    _capsule: Py<PyAny>,
}

impl PySource {
    fn new(object: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Canonical batch workflows pass an outer zero-argument resolver so
        // Python-side backend classification and capsule construction both
        // remain deferred until Rust has completed all static validation and
        // prepared-input construction. Private low-level callers may still
        // pass a descriptor or raw capsule directly.
        let resolved = if object.is_callable() && object.downcast::<PyTuple>().is_err() {
            object.call0()?
        } else {
            object.clone()
        };
        let (capsule_object, expected_identity) =
            if let Ok(descriptor) = resolved.downcast::<PyTuple>() {
                if descriptor.len() != 4 {
                    return Err(PyValueError::new_err(
                        "native propagator descriptor must be (capsule, name, version, two_body)",
                    ));
                }
                (
                    descriptor.get_item(0)?,
                    Some((
                        descriptor.get_item(1)?.extract::<String>()?,
                        descriptor.get_item(2)?.extract::<String>()?,
                        descriptor.get_item(3)?.extract::<bool>()?,
                    )),
                )
            } else {
                // The raw capsule form remains available to the private low-level
                // bindings; adam-core's Python dispatch always supplies identity.
                (resolved, None)
            };
        let capsule_object = if capsule_object.is_callable() {
            capsule_object.call0()?
        } else {
            capsule_object
        };
        let capsule = capsule_object.downcast::<PyCapsule>().map_err(|_| {
            PyValueError::new_err(format!(
                "orbit determination requires a native {EXTERNAL_PROPAGATOR_CAPSULE:?} capsule"
            ))
        })?;
        let name = capsule
            .name()?
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        if name != EXTERNAL_PROPAGATOR_CAPSULE {
            return Err(PyValueError::new_err(format!(
                "expected a {EXTERNAL_PROPAGATOR_CAPSULE:?} capsule, got {name:?}"
            )));
        }
        let raw = capsule.pointer() as *const ExternalPropagatorV2;
        // SAFETY: a capsule of this name holds an `ExternalPropagatorV2`
        // whose context lives as long as the capsule, retained by `_capsule`.
        let propagator = unsafe { ExternalPropagator::from_raw(raw) }.map_err(od_error)?;
        if propagator.configuration_fingerprint().is_none()
            || propagator.data_fingerprint().is_none()
        {
            return Err(PyValueError::new_err(
                "native OD requires the ABI v2 provenance metadata suffix (configuration_fingerprint and data_fingerprint); this provider is propagation-compatible but not provenance-complete",
            ));
        }
        if let Some((expected_name, expected_version, expected_two_body)) = expected_identity {
            validate_expected_identity(
                propagator.identity(),
                &expected_name,
                &expected_version,
                expected_two_body,
            )
            .map_err(PyValueError::new_err)?;
        }
        Ok(Self {
            propagator,
            _capsule: capsule_object.unbind(),
        })
    }
}

fn provider_metadata_dict<'py>(
    py: Python<'py>,
    metadata: &ProviderMetadata,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("name", &metadata.name)?;
    out.set_item("version", &metadata.version)?;
    out.set_item("abi_version", metadata.abi_version)?;
    out.set_item(
        "identity_validated_by_capsule",
        metadata.identity_validated_by_capsule,
    )?;
    out.set_item("capabilities", metadata.capabilities)?;
    out.set_item("known_capabilities", metadata.known_capabilities)?;
    out.set_item("integration_time_scale", &metadata.integration_time_scale)?;
    out.set_item("two_body_dynamics", metadata.two_body_dynamics)?;
    out.set_item(
        "configuration_fingerprint",
        &metadata.configuration_fingerprint,
    )?;
    out.set_item("data_fingerprint", &metadata.data_fingerprint)?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (propagator=None, fit_settings=None))]
fn native_provider_metadata<'py>(
    py: Python<'py>,
    propagator: Option<&Bound<'py, PyAny>>,
    fit_settings: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let settings = parse_fit_settings(fit_settings)?;
    let source = propagator.map(PySource::new).transpose()?;
    let metadata = match &source {
        Some(source) => external_provider_metadata(&source.propagator).map_err(batch_error)?,
        None => two_body_provider_metadata(&settings.propagator),
    };
    provider_metadata_dict(py, &metadata)
}

/// [`PropagatorPredictor`] over an external propagator with the driver's
/// ephemeris settings and adam-core's time and SPICE providers.
fn external_predictor<'a>(
    propagator: &'a ExternalPropagator,
    settings: &'a FitSettings,
) -> PropagatorPredictor<'a, ExternalPropagator, LockingSpiceTranslation> {
    PropagatorPredictor {
        propagator,
        options: &settings.options,
        provider: &ErfaTimeProvider,
        translation_provider: &LockingSpiceTranslation,
    }
}

trait DriverError: Send {
    fn into_py_error(self) -> PyErr;
}

impl DriverError for adam_core_rs_coords::PropagationError {
    fn into_py_error(self) -> PyErr {
        od_error(self)
    }
}

impl DriverError for OdBatchError {
    fn into_py_error(self) -> PyErr {
        batch_error(self)
    }
}

/// Run `driver` with the GIL released over a capsule-backed native
/// propagator, or adam-core's two-body propagator when `None`.
fn run_with_predictor<R: Send, E: DriverError>(
    py: Python<'_>,
    propagator: Option<&Bound<'_, PyAny>>,
    settings: &FitSettings,
    driver: impl for<'p> FnOnce(&'p dyn SphericalPredictor, &'p ProviderMetadata) -> Result<R, E> + Send,
) -> PyResult<(R, ProviderMetadata)> {
    let source = propagator.map(PySource::new).transpose()?;
    let metadata = match &source {
        Some(source) => external_provider_metadata(&source.propagator).map_err(batch_error)?,
        None => two_body_provider_metadata(&settings.propagator),
    };
    let output = py
        .allow_threads(|| {
            if let Some(source) = &source {
                driver(&external_predictor(&source.propagator, settings), &metadata)
            } else {
                let predictor = PropagatorPredictor {
                    propagator: &settings.propagator,
                    options: &settings.options,
                    provider: &ErfaTimeProvider,
                    translation_provider: &LockingSpiceTranslation,
                };
                driver(&predictor, &metadata)
            }
        })
        .map_err(DriverError::into_py_error)?;
    Ok((output, metadata))
}

/// Lazy canonical provider source. The Python object is retained without
/// resolving its capsule until adam_core has completed static validation.
struct PyCanonicalProviderSource {
    propagator: Option<Py<PyAny>>,
    resolution_error: Mutex<Option<PyErr>>,
}

impl PyCanonicalProviderSource {
    fn new(propagator: Option<&Bound<'_, PyAny>>) -> Self {
        Self {
            propagator: propagator.map(|value| value.clone().unbind()),
            resolution_error: Mutex::new(None),
        }
    }

    fn take_resolution_error(&self) -> Option<PyErr> {
        self.resolution_error
            .lock()
            .expect("provider error lock")
            .take()
    }
}

impl OdPropagatorSource for PyCanonicalProviderSource {
    fn with_propagator<R>(
        &self,
        operation: impl for<'provider> FnOnce(
            &'provider OdPropagator<'provider>,
        ) -> Result<R, OdBatchError>,
    ) -> Result<R, OdBatchError> {
        let source = match Python::with_gil(|py| {
            self.propagator
                .as_ref()
                .map(|value| PySource::new(value.bind(py)))
                .transpose()
        }) {
            Ok(source) => source,
            Err(error) => {
                let message = error.to_string();
                *self.resolution_error.lock().expect("provider error lock") = Some(error);
                return Err(OdBatchError::invalid_provider_identity(message));
            }
        };
        if let Some(source) = &source {
            operation(&OdPropagator::external(&source.propagator))
        } else {
            operation(&OdPropagator::two_body())
        }
    }
}

fn run_canonical_workflow<R: Send>(
    py: Python<'_>,
    source: &PyCanonicalProviderSource,
    workflow: impl FnOnce() -> Result<R, OdBatchError> + Send,
) -> PyResult<R> {
    let result = py.allow_threads(workflow);
    if let Some(error) = source.take_resolution_error() {
        return Err(error);
    }
    result.map_err(batch_error)
}

// ---------------------------------------------------------------------------
// Output dicts
// ---------------------------------------------------------------------------

fn set_evaluation_fields(
    py: Python<'_>,
    out: &Bound<'_, PyDict>,
    evaluation: adam_core_rs_coords::propagation::FitEvaluation,
    n: usize,
) -> PyResult<()> {
    let residuals = ndarray::Array2::from_shape_vec((n, 6), evaluation.residuals)
        .map_err(|error| value_error(format!("failed to shape evaluation residuals: {error}")))?;
    out.set_item("residual_values", residuals.into_pyarray(py))?;
    out.set_item("residual_chi2", evaluation.chi2)?;
    out.set_item("residual_dof", evaluation.dof)?;
    out.set_item("residual_probability", evaluation.probability)?;
    out.set_item("chi2", evaluation.orbit_chi2)?;
    out.set_item("reduced_chi2", evaluation.reduced_chi2)?;
    out.set_item("arc_length", evaluation.arc_length)?;
    out.set_item("num_obs", evaluation.num_obs)?;
    out.set_item("outlier", evaluation.outlier)?;
    Ok(())
}

fn set_evaluation_result_fields(
    py: Python<'_>,
    out: &Bound<'_, PyDict>,
    evaluation: adam_core_rs_coords::propagation::FitEvaluation,
    n: usize,
    success: bool,
    status_code: i64,
) -> PyResult<()> {
    let solution = evaluation
        .outlier
        .iter()
        .map(|&value| !value)
        .collect::<Vec<_>>();
    set_evaluation_fields(py, out, evaluation, n)?;
    out.set_item("success", success)?;
    out.set_item("status_code", status_code)?;
    out.set_item("solution", solution)?;
    Ok(())
}

/// The dict contract of a fused whitened fit: solution, covariance, solver
/// diagnostics, weights and the evaluation over the full observation set.
fn fit_output_dict<'py>(
    py: Python<'py>,
    output: WhitenedFitOutput,
    n: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item(
        "state",
        ndarray::Array1::from_vec(output.state.to_vec()).into_pyarray(py),
    )?;
    let covariance = ndarray::Array2::from_shape_vec((6, 6), output.covariance.to_vec())
        .map_err(|err| value_error(format!("failed to shape covariance: {err}")))?;
    out.set_item("covariance", covariance.into_pyarray(py))?;
    out.set_item("iterations", output.iterations)?;
    out.set_item("optimizer_iterations", output.optimizer_iterations)?;
    out.set_item("converged", output.converged)?;
    out.set_item("cost", output.cost)?;
    out.set_item("fit_chi2", output.fit_chi2)?;
    out.set_item(
        "weights",
        ndarray::Array1::from_vec(output.weights).into_pyarray(py),
    )?;
    out.set_item(
        "residuals_whitened",
        ndarray::Array1::from_vec(output.residuals_whitened).into_pyarray(py),
    )?;
    set_evaluation_result_fields(
        py,
        &out,
        output.evaluation,
        n,
        output.converged,
        output.status_code,
    )?;
    out.set_item("warnings", output.warnings)?;
    Ok(out)
}

/// One scored IOD candidate: orbit, exact epoch, provenance (window,
/// triplet rows, root, velocity method), prefit and post-trial statistics.
/// The systematic-ranging sample of an IOD stage: the attributable, the
/// grid counts, and per admissible orbit its state, range, range rate, chi2
/// and weight (arrays), with the index of the best.
fn ranging_dict<'py>(py: Python<'py>, ranging: &RangingResult) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let a = &ranging.attributable;
    let attributable = PyDict::new(py);
    attributable.set_item("reference_row", a.reference_row)?;
    attributable.set_item("epoch_days", a.epoch.days)?;
    attributable.set_item("epoch_nanos", a.epoch.nanos)?;
    attributable.set_item("epoch_scale", a.epoch_scale.as_str())?;
    attributable.set_item("ra_deg", a.ra_deg)?;
    attributable.set_item("dec_deg", a.dec_deg)?;
    attributable.set_item("ra_rate_deg_per_day", a.ra_rate_deg_per_day)?;
    attributable.set_item("dec_rate_deg_per_day", a.dec_rate_deg_per_day)?;
    attributable.set_item("arc_days", a.arc_days)?;
    out.set_item("attributable", attributable)?;
    let states: Vec<f64> = ranging.samples.iter().flat_map(|s| s.state).collect();
    out.set_item(
        "states",
        ndarray::Array2::from_shape_vec((ranging.samples.len(), 6), states)
            .map_err(|err| PyValueError::new_err(err.to_string()))?
            .into_pyarray(py),
    )?;
    let column = |f: fn(&RangingSample) -> f64| ranging.samples.iter().map(f).collect::<Vec<f64>>();
    out.set_item("rho_au", column(|s| s.rho_au))?;
    out.set_item("rho_dot_au_per_day", column(|s| s.rho_dot_au_per_day))?;
    out.set_item("chi2", column(|s| s.chi2))?;
    out.set_item("weight", column(|s| s.weight))?;
    out.set_item("best", ranging.best)?;
    out.set_item("grid_points", ranging.grid_points)?;
    out.set_item("admissible_points", ranging.admissible_points)?;
    out.set_item("num_obs", ranging.num_obs)?;
    Ok(out)
}

fn candidate_failure_dict<'py>(
    py: Python<'py>,
    failure: &IodCandidateFailure,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("triplet", failure.triplet.to_vec())?;
    out.set_item("root_index", failure.root_index)?;
    out.set_item("message", &failure.message)?;
    Ok(out)
}

fn candidate_dict<'py>(py: Python<'py>, candidate: &IodCandidate) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item(
        "state",
        ndarray::Array1::from_vec(candidate.state.to_vec()).into_pyarray(py),
    )?;
    out.set_item("epoch_days", candidate.epoch.days)?;
    out.set_item("epoch_nanos", candidate.epoch.nanos)?;
    out.set_item("epoch_scale", candidate.epoch_scale.as_str())?;
    out.set_item("window", (candidate.window.0, candidate.window.1))?;
    out.set_item("triplet", candidate.triplet.to_vec())?;
    out.set_item("root_index", candidate.root_index)?;
    out.set_item("root_count", candidate.root_count)?;
    out.set_item("velocity_method", candidate.velocity_method.as_str())?;
    out.set_item("prefit_chi2", candidate.prefit_chi2)?;
    out.set_item("prefit_reduced_chi2", candidate.prefit_reduced_chi2)?;
    out.set_item("removed", candidate.removed.clone())?;
    out.set_item("chi2", candidate.chi2_total)?;
    out.set_item("reduced_chi2", candidate.reduced_chi2)?;
    out.set_item("num_obs", candidate.num_obs)?;
    out.set_item("arc_length", candidate.arc_length)?;
    out.set_item("postfit_reduced_chi2", candidate.postfit_reduced_chi2)?;
    out.set_item("whole_arc_score", candidate.whole_arc_score)?;
    out.set_item("source", candidate.source.as_str())?;
    out.set_item("converged", candidate.converged)?;
    out.set_item(
        "corrected_state",
        candidate
            .corrected_state
            .map(|state| ndarray::Array1::from_vec(state.to_vec()).into_pyarray(py)),
    )?;
    out.set_item("stages", candidate.stages)?;
    out.set_item("correction_note", candidate.correction_note.clone())?;
    out.set_item("accepted", candidate.accepted)?;
    Ok(out)
}

/// The dict contract of an IOD stage: ``found``, the accepted orbit's state,
/// exact epoch, statistics, per-observation residuals and flags (the legacy
/// one-orbit view), plus ``candidates`` (every scored candidate, see
/// `candidate_dict`), ``accepted`` (its index or None), the search counters
/// (``triplets_tried``, ``candidates_evaluated``, ``candidates_failed``,
/// ``corrections_run``), ``not_found_reason`` (None when found) and
/// ``ambiguous_with`` (indices of distinct, equally acceptable solutions).
fn iod_dict<'py>(py: Python<'py>, result: IodResult, n: usize) -> PyResult<Bound<'py, PyDict>> {
    let iod = result.to_output(n);
    let out = PyDict::new(py);
    out.set_item("found", iod.found)?;
    out.set_item("success", iod.found)?;
    out.set_item("status_code", i64::from(iod.found))?;
    out.set_item(
        "state",
        ndarray::Array1::from_vec(iod.state.to_vec()).into_pyarray(py),
    )?;
    // The epoch crosses exactly, as integer days and nanoseconds in
    // ``epoch_scale`` (None when no orbit was found).
    out.set_item("epoch_days", iod.epoch.map(|epoch| epoch.days))?;
    out.set_item("epoch_nanos", iod.epoch.map(|epoch| epoch.nanos))?;
    out.set_item("epoch_scale", iod.epoch_scale.as_str())?;
    out.set_item("arc_length", iod.arc_length)?;
    out.set_item("num_obs", iod.num_obs)?;
    out.set_item("chi2", iod.chi2_total)?;
    out.set_item("reduced_chi2", iod.reduced_chi2)?;
    if iod.found {
        let residuals = ndarray::Array2::from_shape_vec((n, 6), iod.residuals)
            .map_err(|err| value_error(format!("failed to shape IOD residuals: {err}")))?;
        out.set_item("residual_values", residuals.into_pyarray(py))?;
    } else {
        out.set_item("residual_values", py.None())?;
    }
    out.set_item("residual_chi2", iod.residual_chi2)?;
    out.set_item("residual_dof", iod.residual_dof)?;
    out.set_item("residual_probability", iod.residual_probability)?;
    out.set_item("solution", iod.solution)?;
    out.set_item("outlier", iod.outlier)?;
    out.set_item("triplets_tried", iod.triplets_tried)?;
    out.set_item("candidates_evaluated", iod.candidates_evaluated)?;
    out.set_item("candidates_failed", iod.candidates_failed)?;
    let candidate_failures = PyList::empty(py);
    for failure in &result.candidate_failures {
        candidate_failures.append(candidate_failure_dict(py, failure)?)?;
    }
    out.set_item("candidate_failures", candidate_failures)?;
    out.set_item("pair_visits", result.pair_visits)?;
    out.set_item("raw_pair_visits", result.raw_pair_visits)?;
    out.set_item("middle_visits", result.middle_visits)?;
    out.set_item("search_cap_exhausted", result.search_cap_exhausted)?;
    out.set_item("corrections_run", result.corrections_run)?;
    out.set_item("not_found_reason", iod.not_found_reason)?;
    out.set_item("accepted", result.accepted)?;
    out.set_item("ambiguous_with", result.ambiguous_with.clone())?;
    out.set_item(
        "ranging",
        result
            .ranging
            .as_ref()
            .map(|ranging| ranging_dict(py, ranging))
            .transpose()?,
    )?;
    let candidates = PyList::empty(py);
    for candidate in &result.candidates {
        candidates.append(candidate_dict(py, candidate)?)?;
    }
    out.set_item("candidates", candidates)?;
    Ok(out)
}
fn snapshot_dict<'py>(
    py: Python<'py>,
    snapshot: AstrometrySnapshot,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item(
        "lon",
        ndarray::Array1::from_vec(snapshot.lon).into_pyarray(py),
    )?;
    out.set_item(
        "lat",
        ndarray::Array1::from_vec(snapshot.lat).into_pyarray(py),
    )?;
    out.set_item(
        "sigma_lon",
        ndarray::Array1::from_vec(snapshot.sigma_lon).into_pyarray(py),
    )?;
    out.set_item(
        "sigma_lat",
        ndarray::Array1::from_vec(snapshot.sigma_lat).into_pyarray(py),
    )?;
    out.set_item(
        "variance_lon",
        ndarray::Array1::from_vec(snapshot.variance_lon).into_pyarray(py),
    )?;
    out.set_item(
        "variance_lat",
        ndarray::Array1::from_vec(snapshot.variance_lat).into_pyarray(py),
    )?;
    out.set_item(
        "cov_lonlat",
        ndarray::Array1::from_vec(snapshot.cov_lonlat).into_pyarray(py),
    )?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Work units
// ---------------------------------------------------------------------------

/// One-crossing whitened `fit_least_squares`: the analytic / central /
/// 2-point Jacobian, linear or Huber loss, the validated covariance and the
/// final evaluation. ``propagator`` is a retained native propagation-capsule
/// descriptor, or None for adam_core's two-body propagator.
#[pyfunction]
#[pyo3(signature = (propagator, orbit_ipc, observed_ipc, observers_ipc, ignore, fit_settings=None))]
fn fit_orbit_whitened_ipc<'py>(
    py: Python<'py>,
    propagator: Option<&Bound<'py, PyAny>>,
    orbit_ipc: &Bound<'py, PyBytes>,
    observed_ipc: &Bound<'py, PyBytes>,
    observers_ipc: &Bound<'py, PyBytes>,
    ignore: Vec<bool>,
    fit_settings: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let orbit = decode_orbit(orbit_ipc)?;
    let problem = decode_observations(observed_ipc, observers_ipc)?;
    let settings = parse_fit_settings(fit_settings)?;
    let n = problem.observed.len();
    let (output, provider) =
        run_with_predictor(py, propagator, &settings, |predictor, _metadata| {
            fit_orbit_whitened_with(
                predictor,
                &orbit,
                &problem.observed,
                &problem.observers,
                &ignore,
                &settings.config,
                &ErfaTimeProvider,
                &LockingSpiceTranslation,
            )
        })?;
    let out = fit_output_dict(py, output, n)?;
    out.set_item(
        "refinement_provider",
        provider_metadata_dict(py, &provider)?,
    )?;
    Ok(out)
}

/// `fit_orbit_whitened_ipc` over adam_core's two-body propagator.
#[pyfunction]
#[pyo3(signature = (orbit_ipc, observed_ipc, observers_ipc, ignore, fit_settings=None))]
fn fit_orbit_whitened_2body_ipc<'py>(
    py: Python<'py>,
    orbit_ipc: &Bound<'py, PyBytes>,
    observed_ipc: &Bound<'py, PyBytes>,
    observers_ipc: &Bound<'py, PyBytes>,
    ignore: Vec<bool>,
    fit_settings: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    fit_orbit_whitened_ipc(
        py,
        None,
        orbit_ipc,
        observed_ipc,
        observers_ipc,
        ignore,
        fit_settings,
    )
}

/// The weak-direction consistency check of `fit_least_squares` on an
/// externally supplied ``covariance`` of ``orbit`` (whose state is the
/// solution): the probe measurement ``delta_cost`` and the covariance
/// `fit_least_squares` would report for ``jacobian`` after validation.
#[pyfunction]
#[pyo3(signature = (propagator, orbit_ipc, observed_ipc, observers_ipc, ignore, covariance, jacobian="analytic", fit_settings=None))]
#[allow(clippy::too_many_arguments)]
fn validate_fit_covariance_ipc<'py>(
    py: Python<'py>,
    propagator: Option<&Bound<'py, PyAny>>,
    orbit_ipc: &Bound<'py, PyBytes>,
    observed_ipc: &Bound<'py, PyBytes>,
    observers_ipc: &Bound<'py, PyBytes>,
    ignore: Vec<bool>,
    covariance: PyReadonlyArray2<'py, f64>,
    jacobian: &str,
    fit_settings: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let orbit = decode_orbit(orbit_ipc)?;
    let problem = decode_observations(observed_ipc, observers_ipc)?;
    let settings = parse_fit_settings(fit_settings)?;
    let covariance = covariance_36(&covariance)?;
    let jacobian = JacobianMethod::parse(jacobian).map_err(value_error)?;
    let (output, provider) =
        run_with_predictor(py, propagator, &settings, |predictor, _metadata| {
            validate_fit_covariance_with(
                predictor,
                &orbit,
                &problem.observed,
                &problem.observers,
                &ignore,
                covariance,
                jacobian,
                &settings.config,
                &ErfaTimeProvider,
                &LockingSpiceTranslation,
            )
        })?;
    let out = PyDict::new(py);
    let validated = ndarray::Array2::from_shape_vec((6, 6), output.covariance.to_vec())
        .map_err(|err| value_error(format!("failed to shape covariance: {err}")))?;
    out.set_item("covariance", validated.into_pyarray(py))?;
    out.set_item("delta_cost", output.delta_cost)?;
    out.set_item("warnings", output.warnings)?;
    out.set_item(
        "refinement_provider",
        provider_metadata_dict(py, &provider)?,
    )?;
    Ok(out)
}

/// One-crossing `iterative_fit` (worst-residual rejection loop). Returns the
/// fused fit dict of the selected pass plus ``passes``.
#[pyfunction]
#[pyo3(signature = (
    propagator,
    orbit_ipc,
    observed_ipc,
    observers_ipc,
    rchi2_threshold=10.0,
    min_obs=6,
    min_arc_length=1.0,
    contamination_percentage=20.0,
    fit_settings=None
))]
#[allow(clippy::too_many_arguments)]
fn iterative_fit_ipc<'py>(
    py: Python<'py>,
    propagator: Option<&Bound<'py, PyAny>>,
    orbit_ipc: &Bound<'py, PyBytes>,
    observed_ipc: &Bound<'py, PyBytes>,
    observers_ipc: &Bound<'py, PyBytes>,
    rchi2_threshold: f64,
    min_obs: usize,
    min_arc_length: f64,
    contamination_percentage: f64,
    fit_settings: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let orbit = decode_orbit(orbit_ipc)?;
    let problem = decode_observations(observed_ipc, observers_ipc)?;
    let settings = parse_fit_settings(fit_settings)?;
    let config = IterativeFitConfig {
        rchi2_threshold,
        min_obs,
        min_arc_length,
        contamination_percentage,
        fit: settings.config,
    };
    let n = problem.observed.len();
    let (output, provider) =
        run_with_predictor(py, propagator, &settings, |predictor, _metadata| {
            iterative_fit_with(
                predictor,
                &orbit,
                &problem.observed,
                &problem.observers,
                &config,
                &ErfaTimeProvider,
                &LockingSpiceTranslation,
            )
        })?;
    let out = fit_output_dict(py, output.fit, n)?;
    out.set_item("passes", output.passes)?;
    out.set_item(
        "refinement_provider",
        provider_metadata_dict(py, &provider)?,
    )?;
    Ok(out)
}

/// `iterative_fit_ipc` over adam_core's two-body propagator.
#[pyfunction]
#[pyo3(signature = (
    orbit_ipc,
    observed_ipc,
    observers_ipc,
    rchi2_threshold=10.0,
    min_obs=6,
    min_arc_length=1.0,
    contamination_percentage=20.0,
    fit_settings=None
))]
#[allow(clippy::too_many_arguments)]
fn iterative_fit_2body_ipc<'py>(
    py: Python<'py>,
    orbit_ipc: &Bound<'py, PyBytes>,
    observed_ipc: &Bound<'py, PyBytes>,
    observers_ipc: &Bound<'py, PyBytes>,
    rchi2_threshold: f64,
    min_obs: usize,
    min_arc_length: f64,
    contamination_percentage: f64,
    fit_settings: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    iterative_fit_ipc(
        py,
        None,
        orbit_ipc,
        observed_ipc,
        observers_ipc,
        rchi2_threshold,
        min_obs,
        min_arc_length,
        contamination_percentage,
        fit_settings,
    )
}

/// One-crossing `cmc2003_fit_detailed`. Returns the fused fit dict of the
/// final pass plus ``n_iterations``, ``n_rejected``, ``n_recovered`` and the
/// sorted ``flags``.
#[pyfunction]
#[pyo3(signature = (
    propagator,
    orbit_ipc,
    observed_ipc,
    observers_ipc,
    chi2_reject=CMC2003_CHI2_REJECT,
    chi2_recover=CMC2003_CHI2_RECOVER,
    chi2_frac=CMC2003_CHI2_FRAC,
    max_iterations=CMC2003_MAX_ITERATIONS,
    max_rejected_fraction=CMC2003_MAX_REJECTED_FRACTION,
    apparition_gap_days=CMC2003_APPARITION_GAP_DAYS,
    psd_floor_frac=CMC2003_PSD_FLOOR_FRAC,
    fit_settings=None
))]
#[allow(clippy::too_many_arguments)]
fn cmc2003_fit_ipc<'py>(
    py: Python<'py>,
    propagator: Option<&Bound<'py, PyAny>>,
    orbit_ipc: &Bound<'py, PyBytes>,
    observed_ipc: &Bound<'py, PyBytes>,
    observers_ipc: &Bound<'py, PyBytes>,
    chi2_reject: f64,
    chi2_recover: f64,
    chi2_frac: f64,
    max_iterations: usize,
    max_rejected_fraction: f64,
    apparition_gap_days: f64,
    psd_floor_frac: f64,
    fit_settings: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let orbit = decode_orbit(orbit_ipc)?;
    let problem = decode_observations(observed_ipc, observers_ipc)?;
    let settings = parse_fit_settings(fit_settings)?;
    let config = Cmc2003FitConfig {
        chi2_reject,
        chi2_recover,
        chi2_frac,
        max_iterations,
        max_rejected_fraction,
        apparition_gap_days,
        psd_floor_frac,
        min_kept: Cmc2003FitConfig::default().min_kept,
        fit: settings.config,
    };
    let n = problem.observed.len();
    let (output, provider) =
        run_with_predictor(py, propagator, &settings, |predictor, _metadata| {
            cmc2003_fit_with(
                predictor,
                &orbit,
                &problem.observed,
                &problem.observers,
                &config,
                &ErfaTimeProvider,
                &LockingSpiceTranslation,
            )
        })?;
    let out = fit_output_dict(py, output.fit, n)?;
    out.set_item("n_iterations", output.n_iterations)?;
    out.set_item("n_rejected", output.n_rejected)?;
    out.set_item("n_recovered", output.n_recovered)?;
    out.set_item("flags", output.flags)?;
    out.set_item(
        "refinement_provider",
        provider_metadata_dict(py, &provider)?,
    )?;
    Ok(out)
}

/// Canonical refinement entry point. Every scientific refinement setting
/// comes from the Rust-owned typed stage configuration, not Python kwargs.
#[pyfunction]
#[pyo3(signature = (propagator, orbit_ipc, observed_ipc, observers_ipc, config))]
fn refine_config_ipc<'py>(
    py: Python<'py>,
    propagator: Option<&Bound<'py, PyAny>>,
    orbit_ipc: &Bound<'py, PyBytes>,
    observed_ipc: &Bound<'py, PyBytes>,
    observers_ipc: &Bound<'py, PyBytes>,
    config: PyRef<'py, PyRefinementRunConfig>,
) -> PyResult<Bound<'py, PyDict>> {
    let orbit = decode_orbit(orbit_ipc)?;
    let problem = decode_observations(observed_ipc, observers_ipc)?;
    let refinement = *config.inner.refinement();
    drop(config);
    let fit = match refinement {
        RefinementConfig::WorstResidual(loop_config) => loop_config.fit,
        RefinementConfig::Cmc2003(loop_config) => loop_config.fit,
    };
    let settings = fit_settings_from_config(fit)?;
    let n = problem.observed.len();
    let (output, provider) =
        run_with_predictor(py, propagator, &settings, |predictor, _metadata| {
            refine_with(
                predictor,
                &orbit,
                &problem.observed,
                &problem.observers,
                &refinement,
                &ErfaTimeProvider,
                &LockingSpiceTranslation,
            )
        })?;
    let out = fit_output_dict(py, output.fit, n)?;
    out.set_item("passes", output.passes)?;
    out.set_item("n_rejected", output.n_rejected)?;
    out.set_item("n_recovered", output.n_recovered)?;
    out.set_item("flags", output.flags)?;
    out.set_item(
        "refinement_provider",
        provider_metadata_dict(py, &provider)?,
    )?;
    Ok(out)
}

/// `cmc2003_fit_ipc` over adam_core's two-body propagator.
#[pyfunction]
#[pyo3(signature = (
    orbit_ipc,
    observed_ipc,
    observers_ipc,
    chi2_reject=CMC2003_CHI2_REJECT,
    chi2_recover=CMC2003_CHI2_RECOVER,
    chi2_frac=CMC2003_CHI2_FRAC,
    max_iterations=CMC2003_MAX_ITERATIONS,
    max_rejected_fraction=CMC2003_MAX_REJECTED_FRACTION,
    apparition_gap_days=CMC2003_APPARITION_GAP_DAYS,
    psd_floor_frac=CMC2003_PSD_FLOOR_FRAC,
    fit_settings=None
))]
#[allow(clippy::too_many_arguments)]
fn cmc2003_fit_2body_ipc<'py>(
    py: Python<'py>,
    orbit_ipc: &Bound<'py, PyBytes>,
    observed_ipc: &Bound<'py, PyBytes>,
    observers_ipc: &Bound<'py, PyBytes>,
    chi2_reject: f64,
    chi2_recover: f64,
    chi2_frac: f64,
    max_iterations: usize,
    max_rejected_fraction: f64,
    apparition_gap_days: f64,
    psd_floor_frac: f64,
    fit_settings: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    cmc2003_fit_ipc(
        py,
        None,
        orbit_ipc,
        observed_ipc,
        observers_ipc,
        chi2_reject,
        chi2_recover,
        chi2_frac,
        max_iterations,
        max_rejected_fraction,
        apparition_gap_days,
        psd_floor_frac,
        fit_settings,
    )
}
/// Canonical IOD entry point. Search, correction, ranging and correction-fit
/// settings come only from the Rust-owned typed stage configuration.
#[pyfunction]
#[pyo3(signature = (iod_propagator, observed_ipc, observers_ipc, config))]
fn run_iod_config_ipc<'py>(
    py: Python<'py>,
    iod_propagator: Option<&Bound<'py, PyAny>>,
    observed_ipc: &Bound<'py, PyBytes>,
    observers_ipc: &Bound<'py, PyBytes>,
    config: PyRef<'py, PyIodRunConfig>,
) -> PyResult<Bound<'py, PyDict>> {
    let problem = decode_observations(observed_ipc, observers_ipc)?;
    let iod_config = *config.inner.iod();
    let correction_fit = *config.inner.correction_fit();
    drop(config);
    let settings = fit_settings_from_config(correction_fit)?;
    let n = problem.observed.len();
    let (result, provider) =
        run_with_predictor(py, iod_propagator, &settings, |predictor, _metadata| {
            run_iod_with(
                predictor,
                &problem.observed,
                &problem.observers,
                &iod_config,
                &correction_fit,
                &ErfaTimeProvider,
                &LockingSpiceTranslation,
            )
        })?;
    let out = iod_dict(py, result, n)?;
    out.set_item("iod_provider", provider_metadata_dict(py, &provider)?)?;
    Ok(out)
}

/// Apply schema-validated observation-model specs in Core Rust and return
/// the complete transformed coordinate batch for a composed fitter route.
#[pyfunction]
#[pyo3(signature = (observed_ipc, obs_code, band, astcat, config))]
fn apply_observation_pipeline_ipc<'py>(
    py: Python<'py>,
    observed_ipc: &Bound<'py, PyBytes>,
    obs_code: Vec<String>,
    band: Vec<Option<String>>,
    astcat: Vec<Option<String>>,
    config: PyRef<'py, PyOdConfig>,
) -> PyResult<Bound<'py, PyDict>> {
    let observed = DataCoordinateBatch::try_from_nested_record_batch(&read_orbit_ipc(
        observed_ipc.as_bytes(),
    )?)
    .map_err(|err| value_error(format!("failed to decode observed coordinates: {err}")))?;
    let output = adam_core_rs_coords::od::apply_configured_observation_pipeline(
        &observed,
        &obs_code,
        &band,
        &astcat,
        config.inner.observation_pipeline(),
        &ErfaTimeProvider,
    )
    .map_err(od_error)?;
    let batch = output
        .observations
        .into_nested_record_batch()
        .map_err(|err| value_error(format!("failed to encode used coordinates: {err}")))?;
    let out = PyDict::new(py);
    out.set_item(
        "used_coordinates_ipc",
        PyBytes::new(py, &write_orbit_ipc(&batch)?),
    )?;
    out.set_item("original_astrometry", snapshot_dict(py, output.original)?)?;
    out.set_item("used_astrometry", snapshot_dict(py, output.used)?)?;
    out.set_item("pipeline_changed", output.models_changed)?;
    out.set_item("configuration_provenance", config.inner.provenance_json())?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Canonical batch workflows
// ---------------------------------------------------------------------------

fn decode_source_catalog(raw: &Bound<'_, PyBytes>) -> PyResult<SourceCatalogBatch> {
    SourceCatalogBatch::try_from_nested_record_batch(&read_orbit_ipc(raw.as_bytes())?)
        .map_err(|error| value_error(format!("failed to decode SourceCatalog: {error}")))
}

fn decode_association_batch(raw: &Bound<'_, PyBytes>) -> PyResult<AssociationBatch> {
    AssociationBatch::try_from_nested_record_batch(&read_orbit_ipc(raw.as_bytes())?)
        .map_err(|error| value_error(format!("failed to decode Associations: {error}")))
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct OrbitProvenanceInput {
    input_fingerprint: Option<String>,
    state_fingerprint: Option<String>,
    provenance_fingerprint: Option<String>,
    route: Option<String>,
    fitter: Option<String>,
    iod_backend: Option<String>,
    refinement_backend: Option<String>,
    evaluation_backend: Option<String>,
    iod_provider: Option<String>,
    refinement_provider: Option<String>,
    evaluation_provider: Option<String>,
    iod_diagnostics: Option<String>,
    refinement_diagnostics: Option<String>,
    evaluation_diagnostics: Option<String>,
    settings: Option<String>,
    configuration: Option<String>,
    iod_ambiguous: Option<bool>,
    iod_alternatives: Option<usize>,
    num_rejected: Option<usize>,
    rejected_fraction: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderMetadataInput {
    name: String,
    version: String,
    abi_version: u32,
    identity_validated_by_capsule: bool,
    capabilities: u64,
    known_capabilities: u64,
    integration_time_scale: String,
    two_body_dynamics: bool,
    configuration_fingerprint: String,
    data_fingerprint: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IodDiagnosticsInput {
    triplets_tried: usize,
    candidates_evaluated: usize,
    candidates_failed: usize,
    corrections_run: usize,
    search_cap_exhausted: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RefinementDiagnosticsInput {
    passes: usize,
    n_rejected: usize,
    n_recovered: usize,
    flags: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EvaluationDiagnosticsInput {
    num_obs: usize,
    arc_length: f64,
    chi2: f64,
    reduced_chi2: f64,
}

fn validate_backend_identity(
    row: usize,
    stage: &str,
    backend: Option<&str>,
    provider: Option<&ProviderMetadata>,
) -> PyResult<()> {
    match (backend, provider) {
        (Some(backend), Some(provider)) if backend == provider.backend_label() => Ok(()),
        (None, None) => Ok(()),
        _ => Err(value_error(format!(
            "orbit provenance at row {row} has an inconsistent {stage} backend identity relative to its consistency digest"
        ))),
    }
}

fn parse_provider_metadata(
    value: Option<String>,
    row: usize,
) -> PyResult<Option<ProviderMetadata>> {
    value
        .map(|value| {
            let decoded: ProviderMetadataInput = serde_json::from_str(&value).map_err(|error| {
                value_error(format!(
                    "invalid provider provenance at orbit row {row}: {error}"
                ))
            })?;
            Ok(ProviderMetadata {
                name: decoded.name,
                version: decoded.version,
                abi_version: decoded.abi_version,
                identity_validated_by_capsule: decoded.identity_validated_by_capsule,
                capabilities: decoded.capabilities,
                known_capabilities: decoded.known_capabilities,
                integration_time_scale: decoded.integration_time_scale,
                two_body_dynamics: decoded.two_body_dynamics,
                configuration_fingerprint: decoded.configuration_fingerprint,
                data_fingerprint: decoded.data_fingerprint,
            })
        })
        .transpose()
}

fn parse_typed_orbit_provenance(
    values: &[Option<String>],
    orbits: &DataOrbitBatch,
) -> PyResult<Vec<Option<OdProvenance>>> {
    if values.len() != orbits.len() {
        return Err(value_error(format!(
            "orbit provenance length {} does not match {} orbit rows",
            values.len(),
            orbits.len()
        )));
    }
    values
        .iter()
        .enumerate()
        .map(|(row, encoded)| {
            let encoded = encoded.as_ref().ok_or_else(|| {
                value_error(format!(
                    "canonical orbit row {row} has no complete Rust provenance"
                ))
            })?;
            let decoded: OrbitProvenanceInput = serde_json::from_str(encoded).map_err(|error| {
                value_error(format!("invalid orbit provenance at row {row}: {error}"))
            })?;
            if decoded.fitter.is_some() || decoded.settings.is_some() {
                return Err(value_error(format!(
                    "orbit provenance at row {row} contains unsupported executable settings"
                )));
            }
            let route = match decoded.route.as_deref() {
                Some("run_iod") => OdRoute::RunIod,
                Some("run_od") => OdRoute::RunOd,
                Some("refine") => OdRoute::Refine,
                Some("evaluate_orbits") => OdRoute::EvaluateOrbits,
                other => {
                    return Err(value_error(format!(
                        "orbit provenance at row {row} has unknown route {}",
                        other.unwrap_or("<missing>")
                    )))
                }
            };
            let configuration = decoded.configuration.ok_or_else(|| {
                value_error(format!(
                    "orbit provenance at row {row} has no configuration"
                ))
            })?;
            let configuration = adam_core::od::CanonicalConfiguration::from_value(
                serde_json::from_str(&configuration).map_err(|error| {
                    value_error(format!("invalid orbit configuration at row {row}: {error}"))
                })?,
            )
            .map_err(batch_error)?;
            let iod_diagnostics = decoded
                .iod_diagnostics
                .map(|value| serde_json::from_str::<IodDiagnosticsInput>(&value))
                .transpose()
                .map_err(|error| value_error(format!("invalid IOD diagnostics: {error}")))?
                .map(|value| adam_core::od::IodDiagnostics {
                    triplets_tried: value.triplets_tried,
                    candidates_evaluated: value.candidates_evaluated,
                    candidates_failed: value.candidates_failed,
                    corrections_run: value.corrections_run,
                    search_cap_exhausted: value.search_cap_exhausted,
                });
            let refinement_diagnostics = decoded
                .refinement_diagnostics
                .map(|value| serde_json::from_str::<RefinementDiagnosticsInput>(&value))
                .transpose()
                .map_err(|error| value_error(format!("invalid refinement diagnostics: {error}")))?
                .map(|value| adam_core::od::RefinementDiagnostics {
                    passes: value.passes,
                    n_rejected: value.n_rejected,
                    n_recovered: value.n_recovered,
                    flags: value.flags,
                });
            let evaluation_diagnostics = decoded
                .evaluation_diagnostics
                .map(|value| serde_json::from_str::<EvaluationDiagnosticsInput>(&value))
                .transpose()
                .map_err(|error| value_error(format!("invalid evaluation diagnostics: {error}")))?
                .map(|value| adam_core::od::EvaluationDiagnostics {
                    num_obs: value.num_obs,
                    arc_length: value.arc_length,
                    chi2: value.chi2,
                    reduced_chi2: value.reduced_chi2,
                });
            let iod_provider = parse_provider_metadata(decoded.iod_provider, row)?;
            let refinement_provider = parse_provider_metadata(decoded.refinement_provider, row)?;
            let evaluation_provider = parse_provider_metadata(decoded.evaluation_provider, row)?;
            validate_backend_identity(
                row,
                "IOD",
                decoded.iod_backend.as_deref(),
                iod_provider.as_ref(),
            )?;
            validate_backend_identity(
                row,
                "refinement",
                decoded.refinement_backend.as_deref(),
                refinement_provider.as_ref(),
            )?;
            validate_backend_identity(
                row,
                "evaluation",
                decoded.evaluation_backend.as_deref(),
                evaluation_provider.as_ref(),
            )?;
            let provenance = OdProvenance {
                input_fingerprint: decoded.input_fingerprint.unwrap_or_default(),
                state_fingerprint: decoded.state_fingerprint,
                provenance_fingerprint: decoded.provenance_fingerprint.unwrap_or_default(),
                route,
                iod_provider,
                refinement_provider,
                evaluation_provider,
                iod_diagnostics,
                refinement_diagnostics,
                evaluation_diagnostics,
                configuration,
                iod_ambiguous: decoded.iod_ambiguous,
                iod_alternatives: decoded.iod_alternatives,
                num_rejected: decoded.num_rejected,
                rejected_fraction: decoded.rejected_fraction,
            };
            provenance.validate().map_err(batch_error)?;
            Ok(Some(provenance))
        })
        .collect()
}

fn batch_error(error: OdBatchError) -> PyErr {
    if let Some(source) = error.source {
        od_error(source)
    } else {
        value_error(error.message)
    }
}

fn typed_failure_dict<'py>(
    py: Python<'py>,
    failure: &adam_core::od::OrbitDeterminationFailure,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("group_id", &failure.group_id)?;
    out.set_item("orbit_id", failure.orbit_id.as_deref())?;
    out.set_item("input_orbit_id", failure.input_orbit_id.as_deref())?;
    out.set_item("stage", failure.stage.as_str())?;
    out.set_item("code", failure.code.as_str())?;
    out.set_item("message", &failure.message)?;
    out.set_item("recoverable", failure.best_state_retained)?;
    out.set_item("num_obs", failure.num_obs)?;
    out.set_item("arc_length", failure.arc_length)?;
    out.set_item("triplets_tried", failure.triplets_tried)?;
    out.set_item("candidates_evaluated", failure.candidates_evaluated)?;
    out.set_item("candidates_failed", failure.candidates_failed)?;
    out.set_item("corrections_run", failure.corrections_run)?;
    out.set_item("passes", failure.passes)?;
    out.set_item("n_rejected", failure.n_rejected)?;
    out.set_item("n_recovered", failure.n_recovered)?;
    out.set_item("status_code", failure.status_code)?;
    out.set_item("best_state_retained", failure.best_state_retained)?;
    Ok(out)
}

fn typed_provenance_dict<'py>(
    py: Python<'py>,
    provenance: &OdProvenance,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("input_fingerprint", &provenance.input_fingerprint)?;
    out.set_item("state_fingerprint", provenance.state_fingerprint.as_deref())?;
    out.set_item("route", provenance.route.as_str())?;
    out.set_item("fitter", py.None())?;
    out.set_item("iod_backend", provenance.iod_backend())?;
    out.set_item("refinement_backend", provenance.refinement_backend())?;
    out.set_item("evaluation_backend", provenance.evaluation_backend())?;
    out.set_item(
        "iod_provider",
        provenance
            .iod_provider
            .as_ref()
            .map(ProviderMetadata::canonical_json),
    )?;
    out.set_item(
        "refinement_provider",
        provenance
            .refinement_provider
            .as_ref()
            .map(ProviderMetadata::canonical_json),
    )?;
    out.set_item(
        "evaluation_provider",
        provenance
            .evaluation_provider
            .as_ref()
            .map(ProviderMetadata::canonical_json),
    )?;
    out.set_item(
        "iod_diagnostics",
        provenance
            .iod_diagnostics
            .as_ref()
            .map(adam_core::od::IodDiagnostics::canonical_json),
    )?;
    out.set_item(
        "refinement_diagnostics",
        provenance
            .refinement_diagnostics
            .as_ref()
            .map(adam_core::od::RefinementDiagnostics::canonical_json),
    )?;
    out.set_item(
        "evaluation_diagnostics",
        provenance
            .evaluation_diagnostics
            .as_ref()
            .map(adam_core::od::EvaluationDiagnostics::canonical_json),
    )?;
    out.set_item("settings", py.None())?;
    out.set_item("configuration", provenance.configuration.canonical_json())?;
    out.set_item("iod_ambiguous", provenance.iod_ambiguous)?;
    out.set_item("iod_alternatives", provenance.iod_alternatives)?;
    out.set_item("num_rejected", provenance.num_rejected)?;
    out.set_item("rejected_fraction", provenance.rejected_fraction)?;
    out.set_item("provenance_fingerprint", &provenance.provenance_fingerprint)?;
    Ok(out)
}

fn typed_group_dict<'py>(py: Python<'py>, group: &OdGroupResult) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("group_id", &group.group_id)?;
    out.set_item("orbit_id", group.orbit_id())?;
    out.set_item("row_indices", &group.row_indices)?;
    out.set_item("member_indices", &group.member_indices)?;
    out.set_item(
        "original_astrometry",
        snapshot_dict(py, group.prepared.original.clone())?,
    )?;
    out.set_item(
        "used_astrometry",
        snapshot_dict(py, group.prepared.used.clone())?,
    )?;
    out.set_item(
        "member_input_fingerprints",
        group
            .members
            .iter()
            .map(|member| member.input_fingerprint.as_str())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "member_membership_fingerprints",
        group
            .members
            .iter()
            .map(|member| member.membership_fingerprint.as_str())
            .collect::<Vec<_>>(),
    )?;
    let orbit = group.orbit.as_ref();
    let refined = orbit.and_then(|record| record.refined.as_ref());
    let evaluation = orbit.and_then(|record| record.evaluation.as_ref());
    let iod = group.iod.as_ref();
    let output_kind = if refined.is_some() {
        "refined"
    } else if evaluation.is_some() {
        "evaluation"
    } else if orbit.is_some() && iod.is_some() {
        "iod"
    } else {
        "none"
    };
    out.set_item("output_kind", output_kind)?;
    out.set_item(
        "iod",
        iod.cloned()
            .map(|value| iod_dict(py, value, group.row_indices.len()))
            .transpose()?,
    )?;
    out.set_item(
        "refined",
        refined
            .cloned()
            .map(|value| {
                let passes = value.passes;
                let n_rejected = value.n_rejected;
                let n_recovered = value.n_recovered;
                let flags = value.flags.clone();
                let output = fit_output_dict(py, value.fit, group.row_indices.len())?;
                output.set_item("passes", passes)?;
                output.set_item("n_rejected", n_rejected)?;
                output.set_item("n_recovered", n_recovered)?;
                output.set_item("flags", flags)?;
                Ok::<_, PyErr>(output)
            })
            .transpose()?,
    )?;
    out.set_item(
        "evaluation",
        evaluation
            .cloned()
            .map(|value| evaluation_output_dict(py, value, group.row_indices.len()))
            .transpose()?,
    )?;
    out.set_item(
        "failure",
        group
            .failure
            .as_ref()
            .map(|failure| typed_failure_dict(py, failure))
            .transpose()?,
    )?;
    out.set_item(
        "provenance",
        orbit
            .map(|record| typed_provenance_dict(py, &record.provenance))
            .transpose()?,
    )?;
    if let Some(index) = group.input_orbit_index {
        out.set_item("orbit_index", index)?;
    }
    Ok(out)
}

fn typed_groups_list<'py>(py: Python<'py>, result: &OdBatchResult) -> PyResult<Bound<'py, PyList>> {
    let items = PyList::empty(py);
    for group in &result.groups {
        items.append(typed_group_dict(py, group)?)?;
    }
    Ok(items)
}

#[pyfunction]
#[pyo3(signature = (propagator, source_ipc, associations_ipc, observed_ipc, observers_ipc, config))]
fn run_iod_batch_ipc<'py>(
    py: Python<'py>,
    propagator: Option<&Bound<'py, PyAny>>,
    source_ipc: &Bound<'py, PyBytes>,
    associations_ipc: &Bound<'py, PyBytes>,
    observed_ipc: &Bound<'py, PyBytes>,
    observers_ipc: &Bound<'py, PyBytes>,
    config: PyRef<'py, PyOdConfig>,
) -> PyResult<Bound<'py, PyDict>> {
    let owned_config = config.inner.clone();
    drop(config);
    let source = decode_source_catalog(source_ipc)?;
    let associations = decode_association_batch(associations_ipc)?;
    let problem = decode_observations(observed_ipc, observers_ipc)?;
    let input = AssociatedObservations {
        observations: OdObservationBatch {
            source,
            observed: problem.observed,
            observers: problem.observers,
        },
        associations,
    };
    let provider_source = PyCanonicalProviderSource::new(propagator);
    let result = run_canonical_workflow(py, &provider_source, || {
        adam_core::od::run_iod(&input, &owned_config, &provider_source)
    })?;
    let out = PyDict::new(py);
    out.set_item("groups", typed_groups_list(py, &result)?)?;
    out.set_item(
        "provider",
        result
            .providers
            .iod
            .as_ref()
            .map(|metadata| provider_metadata_dict(py, metadata))
            .transpose()?,
    )?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (propagator, source_ipc, associations_ipc, observed_ipc, observers_ipc, config))]
fn run_od_batch_ipc<'py>(
    py: Python<'py>,
    propagator: Option<&Bound<'py, PyAny>>,
    source_ipc: &Bound<'py, PyBytes>,
    associations_ipc: &Bound<'py, PyBytes>,
    observed_ipc: &Bound<'py, PyBytes>,
    observers_ipc: &Bound<'py, PyBytes>,
    config: PyRef<'py, PyOdConfig>,
) -> PyResult<Bound<'py, PyDict>> {
    let owned_config = config.inner.clone();
    drop(config);
    let source = decode_source_catalog(source_ipc)?;
    let associations = decode_association_batch(associations_ipc)?;
    let problem = decode_observations(observed_ipc, observers_ipc)?;
    let input = AssociatedObservations {
        observations: OdObservationBatch {
            source,
            observed: problem.observed,
            observers: problem.observers,
        },
        associations,
    };
    let provider_source = PyCanonicalProviderSource::new(propagator);
    let result = run_canonical_workflow(py, &provider_source, || {
        adam_core::od::run_od(&input, &owned_config, &provider_source)
    })?;
    let out = PyDict::new(py);
    out.set_item("groups", typed_groups_list(py, &result)?)?;
    let providers = PyDict::new(py);
    providers.set_item(
        "iod",
        result
            .providers
            .iod
            .as_ref()
            .map(|metadata| provider_metadata_dict(py, metadata))
            .transpose()?,
    )?;
    providers.set_item(
        "refinement",
        result
            .providers
            .refinement
            .as_ref()
            .map(|metadata| provider_metadata_dict(py, metadata))
            .transpose()?,
    )?;
    out.set_item("provider", providers)?;
    Ok(out)
}

fn evaluation_output_dict<'py>(
    py: Python<'py>,
    evaluation: adam_core_rs_coords::propagation::FitEvaluation,
    n: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    set_evaluation_result_fields(py, &out, evaluation, n, true, 1)?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (
    propagator, orbit_ipc, source_ipc, observed_ipc, observers_ipc,
    member_orbit_ids, member_obs_ids, seed_iod_provenance,
    member_input_fingerprints, member_membership_fingerprints, original_lon, original_lat, original_sigma_lon, original_sigma_lat,
    original_variance_lon, original_variance_lat, original_cov_lonlat, used_lon, used_lat, used_sigma_lon, used_sigma_lat,
    used_variance_lon, used_variance_lat, used_cov_lonlat, config
))]
#[allow(clippy::too_many_arguments)]
fn refine_batch_ipc<'py>(
    py: Python<'py>,
    propagator: Option<&Bound<'py, PyAny>>,
    orbit_ipc: &Bound<'py, PyBytes>,
    source_ipc: &Bound<'py, PyBytes>,
    observed_ipc: &Bound<'py, PyBytes>,
    observers_ipc: &Bound<'py, PyBytes>,
    member_orbit_ids: Vec<String>,
    member_obs_ids: Vec<String>,
    seed_iod_provenance: Vec<Option<String>>,
    member_input_fingerprints: Vec<Option<String>>,
    member_membership_fingerprints: Vec<Option<String>>,
    original_lon: Vec<Option<f64>>,
    original_lat: Vec<Option<f64>>,
    original_sigma_lon: Vec<Option<f64>>,
    original_sigma_lat: Vec<Option<f64>>,
    original_variance_lon: Vec<Option<f64>>,
    original_variance_lat: Vec<Option<f64>>,
    original_cov_lonlat: Vec<Option<f64>>,
    used_lon: Vec<Option<f64>>,
    used_lat: Vec<Option<f64>>,
    used_sigma_lon: Vec<Option<f64>>,
    used_sigma_lat: Vec<Option<f64>>,
    used_variance_lon: Vec<Option<f64>>,
    used_variance_lat: Vec<Option<f64>>,
    used_cov_lonlat: Vec<Option<f64>>,
    config: PyRef<'py, PyRefinementRunConfig>,
) -> PyResult<Bound<'py, PyDict>> {
    let owned_config = config.inner;
    drop(config);
    let orbits = decode_orbit(orbit_ipc)?;
    let source = decode_source_catalog(source_ipc)?;
    let problem = decode_observations(observed_ipc, observers_ipc)?;
    let provenance = parse_typed_orbit_provenance(&seed_iod_provenance, &orbits)?;
    let member_count = member_orbit_ids.len();
    let input = MemberedOrbits {
        orbits,
        provenance,
        observations: OdObservationBatch {
            source,
            observed: problem.observed,
            observers: problem.observers,
        },
        members: FittedOrbitMembersBatch {
            orbit_id: member_orbit_ids,
            observation_id: member_obs_ids,
            outlier: vec![None; member_count],
            input_fingerprint: member_input_fingerprints,
            membership_fingerprint: member_membership_fingerprints,
            original_astrometry: AstrometryColumns {
                lon: original_lon,
                lat: original_lat,
                sigma_lon: original_sigma_lon,
                sigma_lat: original_sigma_lat,
                variance_lon: original_variance_lon,
                variance_lat: original_variance_lat,
                cov_lonlat: original_cov_lonlat,
            },
            used_astrometry: AstrometryColumns {
                lon: used_lon,
                lat: used_lat,
                sigma_lon: used_sigma_lon,
                sigma_lat: used_sigma_lat,
                variance_lon: used_variance_lon,
                variance_lat: used_variance_lat,
                cov_lonlat: used_cov_lonlat,
            },
        },
    };
    let provider_source = PyCanonicalProviderSource::new(propagator);
    let result = run_canonical_workflow(py, &provider_source, || {
        adam_core::od::refine(&input, &owned_config, &provider_source)
    })?;
    let out = PyDict::new(py);
    out.set_item("groups", typed_groups_list(py, &result)?)?;
    out.set_item(
        "provider",
        result
            .providers
            .refinement
            .as_ref()
            .map(|metadata| provider_metadata_dict(py, metadata))
            .transpose()?,
    )?;
    Ok(out)
}
#[pyfunction]
#[pyo3(signature = (
    propagator, orbit_ipc, source_ipc, observed_ipc, observers_ipc,
    member_orbit_ids, member_obs_ids, member_outlier,
    orbit_provenance, member_input_fingerprints, member_membership_fingerprints, original_lon, original_lat, original_sigma_lon, original_sigma_lat,
    original_variance_lon, original_variance_lat, original_cov_lonlat, used_lon, used_lat, used_sigma_lon, used_sigma_lat,
    used_variance_lon, used_variance_lat, used_cov_lonlat, config
))]
#[allow(clippy::too_many_arguments)]
fn evaluate_orbits_batch_ipc<'py>(
    py: Python<'py>,
    propagator: Option<&Bound<'py, PyAny>>,
    orbit_ipc: &Bound<'py, PyBytes>,
    source_ipc: &Bound<'py, PyBytes>,
    observed_ipc: &Bound<'py, PyBytes>,
    observers_ipc: &Bound<'py, PyBytes>,
    member_orbit_ids: Vec<String>,
    member_obs_ids: Vec<String>,
    member_outlier: Vec<Option<bool>>,
    orbit_provenance: Vec<Option<String>>,
    member_input_fingerprints: Vec<Option<String>>,
    member_membership_fingerprints: Vec<Option<String>>,
    original_lon: Vec<Option<f64>>,
    original_lat: Vec<Option<f64>>,
    original_sigma_lon: Vec<Option<f64>>,
    original_sigma_lat: Vec<Option<f64>>,
    original_variance_lon: Vec<Option<f64>>,
    original_variance_lat: Vec<Option<f64>>,
    original_cov_lonlat: Vec<Option<f64>>,
    used_lon: Vec<Option<f64>>,
    used_lat: Vec<Option<f64>>,
    used_sigma_lon: Vec<Option<f64>>,
    used_sigma_lat: Vec<Option<f64>>,
    used_variance_lon: Vec<Option<f64>>,
    used_variance_lat: Vec<Option<f64>>,
    used_cov_lonlat: Vec<Option<f64>>,
    config: PyRef<'py, PyOdConfig>,
) -> PyResult<Bound<'py, PyDict>> {
    let owned_config = config.inner.clone();
    drop(config);
    let orbits = decode_orbit(orbit_ipc)?;
    let source = decode_source_catalog(source_ipc)?;
    let problem = decode_observations(observed_ipc, observers_ipc)?;
    let provenance = parse_typed_orbit_provenance(&orbit_provenance, &orbits)?;
    let input = MemberedOrbits {
        orbits,
        provenance,
        observations: OdObservationBatch {
            source,
            observed: problem.observed,
            observers: problem.observers,
        },
        members: FittedOrbitMembersBatch {
            orbit_id: member_orbit_ids,
            observation_id: member_obs_ids,
            outlier: member_outlier,
            input_fingerprint: member_input_fingerprints,
            membership_fingerprint: member_membership_fingerprints,
            original_astrometry: AstrometryColumns {
                lon: original_lon,
                lat: original_lat,
                sigma_lon: original_sigma_lon,
                sigma_lat: original_sigma_lat,
                variance_lon: original_variance_lon,
                variance_lat: original_variance_lat,
                cov_lonlat: original_cov_lonlat,
            },
            used_astrometry: AstrometryColumns {
                lon: used_lon,
                lat: used_lat,
                sigma_lon: used_sigma_lon,
                sigma_lat: used_sigma_lat,
                variance_lon: used_variance_lon,
                variance_lat: used_variance_lat,
                cov_lonlat: used_cov_lonlat,
            },
        },
    };
    let provider_source = PyCanonicalProviderSource::new(propagator);
    let result = run_canonical_workflow(py, &provider_source, || {
        adam_core::od::evaluate_orbits(&input, &owned_config, &provider_source)
    })?;
    let out = PyDict::new(py);
    out.set_item("groups", typed_groups_list(py, &result)?)?;
    out.set_item(
        "provider",
        result
            .providers
            .evaluation
            .as_ref()
            .map(|metadata| provider_metadata_dict(py, metadata))
            .transpose()?,
    )?;
    Ok(out)
}

/// Native OD timing lanes supported by [`benchmark_od_fit_ipc`].
#[derive(Clone, Copy)]
enum NativeOdBenchmarkOperation {
    Refine,
    RunOd,
}

impl NativeOdBenchmarkOperation {
    fn parse(value: &str) -> PyResult<Self> {
        match value {
            "refine" => Ok(Self::Refine),
            "run_od" => Ok(Self::RunOd),
            _ => Err(value_error(
                "operation must be exactly 'refine' or 'run_od'",
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Refine => "refine",
            Self::RunOd => "run_od",
        }
    }
}

struct NativeOdBenchmarkResult {
    fit: WhitenedFitOutput,
    iod_found: Option<bool>,
    triplets_tried: Option<usize>,
    candidates_evaluated: Option<usize>,
    passes: usize,
    n_rejected: usize,
    n_recovered: usize,
}

#[allow(clippy::too_many_arguments)]
fn run_native_od_benchmark_once(
    operation: NativeOdBenchmarkOperation,
    predictor: &dyn SphericalPredictor,
    orbit: &DataOrbitBatch,
    problem: &OdObservations,
    prepared: &PreparedObservations,
    obs_code: &[String],
    band: &[Option<String>],
    astcat: &[Option<String>],
    config: &OdConfig,
) -> PropagationResultValue<NativeOdBenchmarkResult> {
    match operation {
        NativeOdBenchmarkOperation::Refine => {
            let refined = refine_with_predictor(
                predictor,
                orbit,
                prepared,
                &problem.observers,
                config,
                &ErfaTimeProvider,
                &LockingSpiceTranslation,
            )?;
            Ok(NativeOdBenchmarkResult {
                fit: refined.fit,
                iod_found: None,
                triplets_tried: None,
                candidates_evaluated: None,
                passes: refined.passes,
                n_rejected: refined.n_rejected,
                n_recovered: refined.n_recovered,
            })
        }
        NativeOdBenchmarkOperation::RunOd => {
            let output = run_od_with_predictors(
                predictor,
                predictor,
                &problem.observed,
                &problem.observers,
                obs_code,
                band,
                astcat,
                config,
                &ErfaTimeProvider,
                &LockingSpiceTranslation,
            )?;
            let iod_found = output.iod.found();
            let triplets_tried = output.iod.triplets_tried;
            let candidates_evaluated = output.iod.candidates_evaluated;
            let refined = output.refined.ok_or_else(|| {
                adam_core_rs_coords::propagation::PropagationError::InvalidRequest(
                    "benchmark run_od workload did not produce a refined orbit".to_string(),
                )
            })?;
            Ok(NativeOdBenchmarkResult {
                fit: refined.fit,
                iod_found: Some(iod_found),
                triplets_tried: Some(triplets_tried),
                candidates_evaluated: Some(candidates_evaluated),
                passes: refined.passes,
                n_rejected: refined.n_rejected,
                n_recovered: refined.n_recovered,
            })
        }
    }
}

/// Rust-owned `Instant` timing for Core OD fit stages.
///
/// Arrow decoding, Python extraction, configuration validation, and the
/// refinement lane's one-time observation preparation happen before the
/// correctness run and are excluded from every sample. `refine` times the
/// configured native refinement stage over an already prepared IOD handoff;
/// `run_od` times the complete configured prepare + IOD + refinement workflow.
/// The returned state/diagnostics come from an untimed preflight run so the
/// Python benchmark harness can assert correctness before reporting timings.
#[pyfunction]
#[pyo3(signature = (
    propagator,
    operation,
    orbit_ipc,
    observed_ipc,
    observers_ipc,
    obs_code,
    band,
    astcat,
    config,
    reps,
    trials,
    warmup_reps
))]
#[allow(clippy::too_many_arguments)]
fn benchmark_od_fit_ipc<'py>(
    py: Python<'py>,
    propagator: Option<&Bound<'py, PyAny>>,
    operation: &str,
    orbit_ipc: &Bound<'py, PyBytes>,
    observed_ipc: &Bound<'py, PyBytes>,
    observers_ipc: &Bound<'py, PyBytes>,
    obs_code: Vec<String>,
    band: Vec<Option<String>>,
    astcat: Vec<Option<String>>,
    config: PyRef<'py, PyOdConfig>,
    reps: usize,
    trials: usize,
    warmup_reps: usize,
) -> PyResult<Bound<'py, PyDict>> {
    if reps == 0 || trials == 0 {
        return Err(value_error("reps and trials must be >= 1"));
    }
    let operation = NativeOdBenchmarkOperation::parse(operation)?;
    let source = propagator.map(PySource::new).transpose()?;
    let provider_boundary = if source.is_some() {
        "external propagation ABI v2"
    } else {
        "direct typed TwoBodyPropagator"
    };
    let orbit = decode_orbit(orbit_ipc)?;
    let problem = decode_observations(observed_ipc, observers_ipc)?;
    config.inner.validate().map_err(value_error)?;
    let owned_config = config.inner.clone();
    let refinement = *owned_config.for_refinement().refinement();
    let refinement_fit = match refinement {
        RefinementConfig::WorstResidual(value) => value.fit,
        RefinementConfig::Cmc2003(value) => value.fit,
    };
    let settings = fit_settings_from_config(refinement_fit)?;
    drop(config);
    let n = problem.observed.len();
    if obs_code.len() != n || band.len() != n || astcat.len() != n {
        return Err(value_error(
            "observed, obs_code, band and astcat must have equal length",
        ));
    }
    let prepared = prepare_observations(
        &problem.observed,
        &obs_code,
        &band,
        &astcat,
        &owned_config,
        &ErfaTimeProvider,
    )
    .map_err(od_error)?;

    let (samples, correctness) = py
        .allow_threads(|| {
            let two_body = PropagatorPredictor {
                propagator: &settings.propagator,
                options: &settings.options,
                provider: &ErfaTimeProvider,
                translation_provider: &LockingSpiceTranslation,
            };
            let external = source
                .as_ref()
                .map(|source| external_predictor(&source.propagator, &settings));
            let predictor: &dyn SphericalPredictor = external
                .as_ref()
                .map_or(&two_body as &dyn SphericalPredictor, |value| value);
            let run_once = || {
                run_native_od_benchmark_once(
                    operation,
                    predictor,
                    &orbit,
                    &problem,
                    &prepared,
                    &obs_code,
                    &band,
                    &astcat,
                    &owned_config,
                )
            };
            let correctness = run_once()?;
            let nonfinite_state = correctness
                .fit
                .state
                .iter()
                .position(|value| !value.is_finite());
            let nonfinite_covariance = correctness
                .fit
                .covariance
                .iter()
                .position(|value| !value.is_finite());
            let expected_num_obs = n.checked_sub(correctness.n_rejected);
            if nonfinite_state.is_some()
                || nonfinite_covariance.is_some()
                || expected_num_obs != Some(correctness.fit.evaluation.num_obs)
            {
                return Err(adam_core_rs_coords::propagation::PropagationError::InvalidRequest(
                    format!(
                        "benchmark {} correctness run is invalid: nonfinite_state={nonfinite_state:?}, \
                         nonfinite_covariance={nonfinite_covariance:?}, num_obs={}, input_num_obs={n}, \
                         n_rejected={}, expected_num_obs={expected_num_obs:?}",
                        operation.as_str(),
                        correctness.fit.evaluation.num_obs,
                        correctness.n_rejected,
                    ),
                ));
            }
            let mut samples = Vec::with_capacity(trials);
            for _ in 0..trials {
                for _ in 0..warmup_reps {
                    black_box(run_once()?);
                }
                let mut trial = Vec::with_capacity(reps);
                for _ in 0..reps {
                    let started = Instant::now();
                    let result = run_once()?;
                    black_box(&result);
                    trial.push(started.elapsed().as_secs_f64());
                }
                samples.push(trial);
            }
            Ok((samples, correctness))
        })
        .map_err(od_error)?;

    let out = PyDict::new(py);
    out.set_item("operation", operation.as_str())?;
    out.set_item("timer", "std::time::Instant")?;
    out.set_item("provider_boundary", provider_boundary)?;
    out.set_item(
        "host_available_parallelism",
        std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get),
    )?;
    out.set_item("od_algorithm_threading", "serial")?;
    out.set_item("input_decoding_timed", false)?;
    out.set_item("settings_parsing_timed", false)?;
    out.set_item("samples_seconds", samples)?;
    out.set_item("state", correctness.fit.state.to_vec())?;
    out.set_item("iterations", correctness.fit.iterations)?;
    out.set_item("optimizer_iterations", correctness.fit.optimizer_iterations)?;
    out.set_item("converged", correctness.fit.converged)?;
    out.set_item("status_code", correctness.fit.status_code)?;
    out.set_item("reduced_chi2", correctness.fit.evaluation.reduced_chi2)?;
    out.set_item("num_obs", correctness.fit.evaluation.num_obs)?;
    out.set_item("iod_found", correctness.iod_found)?;
    out.set_item("triplets_tried", correctness.triplets_tried)?;
    out.set_item("candidates_evaluated", correctness.candidates_evaluated)?;
    out.set_item("passes", correctness.passes)?;
    out.set_item("n_rejected", correctness.n_rejected)?;
    out.set_item("n_recovered", correctness.n_recovered)?;
    Ok(out)
}

/// Canonical Cargo SemVer embedded in adam-core's native two-body provider.
/// Provider declarations compare this native spelling (including `-rc.N`),
/// never the Python distribution's independently normalized PEP 440 version.
#[pyfunction]
pub fn two_body_propagator_semver() -> &'static str {
    adam_core_rs_coords::propagation::two_body_reference::SEMVER
}

/// adam-core's two-body propagator as an external-propagator capsule: the
/// reference implementation of the provider ABI (and a test double of the
/// external route). The capsule owns the propagator; its pointer is the
/// table's address.
#[pyfunction]
pub fn two_body_propagator_capsule(py: Python<'_>) -> PyResult<Bound<'_, PyCapsule>> {
    let name = std::ffi::CString::new(EXTERNAL_PROPAGATOR_CAPSULE).expect("no NUL");
    PyCapsule::new(
        py,
        adam_core_rs_coords::propagation::two_body_reference::new(TwoBodyPropagator::default()),
        Some(name),
    )
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(two_body_propagator_semver, m)?)?;
    m.add_function(wrap_pyfunction!(two_body_propagator_capsule, m)?)?;
    m.add_function(wrap_pyfunction!(fit_orbit_whitened_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(fit_orbit_whitened_2body_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(validate_fit_covariance_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(iterative_fit_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(iterative_fit_2body_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(cmc2003_fit_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(cmc2003_fit_2body_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(refine_config_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(run_iod_config_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(run_iod_batch_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(run_od_batch_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(refine_batch_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_orbits_batch_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(native_provider_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(od_defaults, m)?)?;
    m.add_function(wrap_pyfunction!(apply_observation_pipeline_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_od_fit_ipc, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dict_fit_defaults_preserve_pre_0_7_ephemeris_values_bitwise() {
        let actual = dict_fit_defaults();
        let fit = WhitenedFitConfig::default();
        let ephemeris = EphemerisOptions::default();

        assert_eq!(actual.loss, fit.loss);
        assert_eq!(actual.f_scale.to_bits(), fit.f_scale.to_bits());
        assert_eq!(actual.jacobian, fit.jacobian);
        assert_eq!(actual.validate_covariance, fit.validate_covariance);
        assert_eq!(actual.xtol.to_bits(), fit.xtol.to_bits());
        assert_eq!(actual.ftol.to_bits(), fit.ftol.to_bits());
        assert_eq!(actual.gtol.to_bits(), fit.gtol.to_bits());
        assert_eq!(actual.max_iterations, fit.max_iterations);
        assert_eq!(actual.max_nfev, fit.max_nfev);
        assert_eq!(
            actual.two_body.propagation_max_iter,
            fit.two_body.propagation_max_iter
        );
        assert_eq!(
            actual.two_body.propagation_tol.to_bits(),
            fit.two_body.propagation_tol.to_bits()
        );
        assert_eq!(
            actual.two_body.ephemeris_tol.to_bits(),
            fit.two_body.ephemeris_tol.to_bits()
        );

        assert_eq!(actual.two_body.lt_tol.to_bits(), 1.0e-12_f64.to_bits());
        assert_eq!(actual.two_body.ephemeris_max_iter, 1_000);
        assert_eq!(actual.two_body.max_lt_iter, 10);
        assert_eq!(actual.two_body.lt_tol.to_bits(), ephemeris.lt_tol.to_bits());
        assert_eq!(actual.two_body.ephemeris_max_iter, ephemeris.max_iter);
        assert_eq!(actual.two_body.max_lt_iter, ephemeris.max_lt_iter);

        let runtime = canonical_stage_settings(actual).expect("legacy defaults are valid");
        assert_eq!(runtime.options.lt_tol.to_bits(), 1.0e-12_f64.to_bits());
        assert_eq!(runtime.options.max_iter, 1_000);
        assert_eq!(runtime.options.max_lt_iter, 10);
    }

    #[test]
    fn provider_identity_compares_canonical_cargo_prerelease_spelling_exactly() {
        let actual = BackendIdentity {
            name: "test.provider".to_string(),
            version: "1.2.3-rc.4".to_string(),
            two_body_dynamics: false,
        };
        assert!(validate_expected_identity(&actual, "test.provider", "1.2.3-rc.4", false).is_ok());
        let error =
            validate_expected_identity(&actual, "test.provider", "1.2.3rc4", false).unwrap_err();
        assert!(error.contains("1.2.3rc4"));
        assert!(error.contains("1.2.3-rc.4"));
    }
}
