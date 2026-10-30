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

use crate::coordinates::{read_orbit_ipc, write_orbit_ipc, ErfaTimeProvider};
use crate::od_config::{PyIodRunConfig, PyOdConfig, PyRefinementRunConfig};
use crate::od_support::{od_error, LockingSpiceTranslation};
use adam_core_rs_coords::propagation::adapter::{
    cmc2003_fit_with, fit_orbit_whitened_with, iterative_fit_with, refine_with, run_iod_with,
    validate_fit_covariance_with, PropagatorPredictor, SphericalPredictor,
};
use adam_core_rs_coords::propagation::{
    evaluate_orbit, AstrometrySnapshot, BackendIdentity, Cmc2003FitConfig, CovariancePropagation,
    EphemerisOptions, EpochPolicy, ExternalPropagator, ExternalPropagatorV2, IodCandidate,
    IodCandidateFailure, IodResult, IterativeFitConfig, JacobianMethod, PropagationOptions,
    PropagationResultValue, Propagator, RangingResult, RangingSample, RefinementConfig,
    TwoBodyPropagator, TwoBodyPropagatorConfig, WhitenedFitConfig, WhitenedFitOutput,
    EXTERNAL_PROPAGATOR_CAPSULE,
};
use adam_core_rs_coords::{
    validate_loss, AssociationBatch, CoordinateBatch as DataCoordinateBatch, IntoNestedRecordBatch,
    ObserverBatch as DataObserverBatch, OdConfig, OrbitBatch as DataOrbitBatch, SourceCatalogBatch,
    TimeScale, TryFromNestedRecordBatch, TwoBodyModelConfig, CMC2003_APPARITION_GAP_DAYS,
    CMC2003_CHI2_FRAC, CMC2003_CHI2_RECOVER, CMC2003_CHI2_REJECT, CMC2003_MAX_ITERATIONS,
    CMC2003_MAX_REJECTED_FRACTION, CMC2003_PSD_FLOOR_FRAC,
};
use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyCapsule, PyDict, PyList, PyTuple};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};

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

/// `fit_settings` over the Rust defaults: `WhitenedFitConfig::default()` for
/// the solver, `EphemerisOptions::default()` for the light-time /
/// aberration settings and `TwoBodyPropagatorConfig::default()` for the
/// Kepler solver. The dict only overrides them; no number lives here.
fn parse_fit_settings(settings: Option<&Bound<'_, PyDict>>) -> PyResult<FitSettings> {
    let fit_defaults = WhitenedFitConfig::default();
    let ephemeris_defaults = EphemerisOptions::default();
    let propagation_defaults = TwoBodyPropagatorConfig::default();
    let loss: String = setting(settings, "loss", fit_defaults.loss.as_str().to_string())?;
    let f_scale: f64 = setting(settings, "f_scale", fit_defaults.f_scale)?;
    let loss = validate_loss(&loss, f_scale).map_err(value_error)?;
    let jacobian: String = setting(
        settings,
        "jacobian",
        fit_defaults.jacobian.as_str().to_string(),
    )?;
    let jacobian = JacobianMethod::parse(&jacobian).map_err(value_error)?;
    let lt_tol: f64 = setting(settings, "lt_tol", ephemeris_defaults.lt_tol)?;
    let eph_max_iter: usize = setting(settings, "eph_max_iter", ephemeris_defaults.max_iter)?;
    let eph_tol: f64 = setting(settings, "eph_tol", ephemeris_defaults.tol)?;
    let stellar_aberration: bool = setting(
        settings,
        "stellar_aberration",
        ephemeris_defaults.stellar_aberration,
    )?;
    let max_lt_iter: usize = setting(settings, "max_lt_iter", ephemeris_defaults.max_lt_iter)?;
    let prop_max_iter: usize = setting(settings, "prop_max_iter", propagation_defaults.max_iter)?;
    let prop_tol: f64 = setting(settings, "prop_tol", propagation_defaults.tol)?;
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
    config.validate().map_err(value_error)?;
    let options = EphemerisOptions {
        propagation: PropagationOptions {
            chunk_size: None,
            thread_limit: None,
            epoch_policy: EpochPolicy::CrossProduct,
            covariance: CovariancePropagation::None,
        },
        lt_tol,
        max_iter: eph_max_iter,
        tol: eph_tol,
        stellar_aberration,
        max_lt_iter,
        output_time_scale: TimeScale::Utc,
        include_aberrated_coordinates: false,
        ..EphemerisOptions::default()
    };
    let propagator = TwoBodyPropagator::new(TwoBodyPropagatorConfig {
        max_iter: prop_max_iter,
        tol: prop_tol,
    })
    .map_err(|err| value_error(format!("invalid propagator config: {err}")))?;
    Ok(FitSettings {
        config,
        options,
        propagator,
    })
}

/// Build the fixed ephemeris/provider adapter around an authoritative typed
/// fit configuration. Every configurable two-body tolerance/budget is copied
/// from that configuration; only non-configurable OD conventions remain fixed
/// here (UTC output, no stellar aberration, no propagated covariance).
fn fit_settings_from_config(config: WhitenedFitConfig) -> PyResult<FitSettings> {
    config.validate().map_err(value_error)?;
    let mut settings = parse_fit_settings(None)?;
    let two_body = config.two_body;
    settings.options.lt_tol = two_body.lt_tol;
    settings.options.max_iter = two_body.ephemeris_max_iter;
    settings.options.tol = two_body.ephemeris_tol;
    settings.options.max_lt_iter = two_body.max_lt_iter;
    settings.propagator = TwoBodyPropagator::new(TwoBodyPropagatorConfig {
        max_iter: two_body.propagation_max_iter,
        tol: two_body.propagation_tol,
    })
    .map_err(|err| value_error(format!("invalid typed propagator config: {err}")))?;
    settings.config = config;
    Ok(settings)
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

#[derive(Clone)]
struct ProviderMetadata {
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

impl ProviderMetadata {
    fn external(propagator: &ExternalPropagator) -> Self {
        let identity = propagator.identity();
        Self {
            name: identity.name.clone(),
            version: identity.version.clone(),
            abi_version: propagator.abi_version(),
            identity_validated_by_capsule: true,
            capabilities: propagator.capabilities(),
            known_capabilities: propagator.known_capabilities(),
            integration_time_scale: propagator.integration_scale().as_str().to_string(),
            two_body_dynamics: identity.two_body_dynamics,
            configuration_fingerprint: propagator
                .configuration_fingerprint()
                .expect("PySource requires metadata")
                .to_string(),
            data_fingerprint: propagator
                .data_fingerprint()
                .expect("PySource requires metadata")
                .to_string(),
        }
    }

    fn two_body(settings: &FitSettings) -> Self {
        let identity = settings.propagator.backend();
        let mut capabilities =
            adam_core_rs_coords::propagation::EXTERNAL_PROPAGATOR_CAP_THREAD_SAFE;
        if settings.propagator.supports_stm() {
            capabilities |= adam_core_rs_coords::propagation::EXTERNAL_PROPAGATOR_CAP_STM;
        }
        if settings.propagator.supports_non_gravitational_parameters() {
            capabilities |=
                adam_core_rs_coords::propagation::EXTERNAL_PROPAGATOR_CAP_NON_GRAVITATIONAL;
        }
        Self {
            name: identity.name,
            version: identity.version,
            abi_version: adam_core_rs_coords::propagation::EXTERNAL_PROPAGATOR_ABI_VERSION,
            identity_validated_by_capsule: false,
            capabilities,
            known_capabilities: capabilities,
            integration_time_scale: settings
                .propagator
                .integration_time_scale()
                .as_str()
                .to_string(),
            two_body_dynamics: identity.two_body_dynamics,
            configuration_fingerprint:
                adam_core_rs_coords::propagation::two_body_reference::configuration_fingerprint(
                    settings.propagator.config(),
                ),
            data_fingerprint:
                adam_core_rs_coords::propagation::two_body_reference::DATA_FINGERPRINT.to_string(),
        }
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
    let metadata = source.as_ref().map_or_else(
        || ProviderMetadata::two_body(&settings),
        |source| ProviderMetadata::external(&source.propagator),
    );
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

/// Run `driver` with the GIL released over a capsule-backed native
/// propagator, or adam-core's two-body propagator when `None`.
fn run_with_predictor<R: Send>(
    py: Python<'_>,
    propagator: Option<&Bound<'_, PyAny>>,
    settings: &FitSettings,
    driver: impl for<'p> FnOnce(&'p dyn SphericalPredictor) -> PropagationResultValue<R> + Send,
) -> PyResult<(R, ProviderMetadata)> {
    let source = propagator.map(PySource::new).transpose()?;
    let metadata = source.as_ref().map_or_else(
        || ProviderMetadata::two_body(settings),
        |source| ProviderMetadata::external(&source.propagator),
    );
    let output = py
        .allow_threads(|| {
            if let Some(source) = &source {
                driver(&external_predictor(&source.propagator, settings))
            } else {
                let predictor = PropagatorPredictor {
                    propagator: &settings.propagator,
                    options: &settings.options,
                    provider: &ErfaTimeProvider,
                    translation_provider: &LockingSpiceTranslation,
                };
                driver(&predictor)
            }
        })
        .map_err(od_error)?;
    Ok((output, metadata))
}

fn run_with_stage_predictors<R: Send>(
    py: Python<'_>,
    propagator: Option<&Bound<'_, PyAny>>,
    iod_settings: &FitSettings,
    refinement_settings: &FitSettings,
    driver: impl for<'p> FnOnce(
            &'p dyn SphericalPredictor,
            &'p dyn SphericalPredictor,
        ) -> PropagationResultValue<R>
        + Send,
) -> PyResult<(R, ProviderMetadata, ProviderMetadata)> {
    let source = propagator.map(PySource::new).transpose()?;
    let (iod_metadata, refinement_metadata) = source.as_ref().map_or_else(
        || {
            (
                ProviderMetadata::two_body(iod_settings),
                ProviderMetadata::two_body(refinement_settings),
            )
        },
        |source| {
            let metadata = ProviderMetadata::external(&source.propagator);
            (metadata.clone(), metadata)
        },
    );
    let output = py
        .allow_threads(|| {
            if let Some(source) = &source {
                let iod = external_predictor(&source.propagator, iod_settings);
                let refinement = external_predictor(&source.propagator, refinement_settings);
                driver(&iod, &refinement)
            } else {
                let iod = PropagatorPredictor {
                    propagator: &iod_settings.propagator,
                    options: &iod_settings.options,
                    provider: &ErfaTimeProvider,
                    translation_provider: &LockingSpiceTranslation,
                };
                let refinement = PropagatorPredictor {
                    propagator: &refinement_settings.propagator,
                    options: &refinement_settings.options,
                    provider: &ErfaTimeProvider,
                    translation_provider: &LockingSpiceTranslation,
                };
                driver(&iod, &refinement)
            }
        })
        .map_err(od_error)?;
    Ok((output, iod_metadata, refinement_metadata))
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
    out.set_item("success", output.converged)?;
    out.set_item("status_code", output.status_code)?;
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
    let solution = output
        .evaluation
        .outlier
        .iter()
        .map(|&value| !value)
        .collect::<Vec<_>>();
    set_evaluation_fields(py, &out, output.evaluation, n)?;
    out.set_item("solution", solution)?;
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
    let (output, provider) = run_with_predictor(py, propagator, &settings, |predictor| {
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
    let (output, provider) = run_with_predictor(py, propagator, &settings, |predictor| {
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
    let (output, provider) = run_with_predictor(py, propagator, &settings, |predictor| {
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
    let (output, provider) = run_with_predictor(py, propagator, &settings, |predictor| {
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
    let (output, provider) = run_with_predictor(py, propagator, &settings, |predictor| {
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
    let (result, provider) = run_with_predictor(py, iod_propagator, &settings, |predictor| {
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

fn validate_catalog_problem(source: &SourceCatalogBatch, problem: &OdObservations) -> PyResult<()> {
    if source.len() != problem.observed.len() || source.len() != problem.observers.len() {
        return Err(value_error(format!(
            "SourceCatalog, observed coordinates, and observers must have equal length ({} != {} != {})",
            source.len(),
            problem.observed.len(),
            problem.observers.len()
        )));
    }
    for (row, (expected, actual)) in source
        .observatory_code
        .iter()
        .zip(problem.observers.code.iter())
        .enumerate()
    {
        if expected != &actual.0 {
            return Err(value_error(format!(
                "observer code mismatch at row {row}: SourceCatalog {expected:?}, observers {:?}",
                actual.0
            )));
        }
    }
    let observed_times = problem
        .observed
        .times
        .as_ref()
        .ok_or_else(|| value_error("orbit-determination observed coordinates require time"))?;
    if observed_times.scale != source.time.scale {
        return Err(value_error(format!(
            "SourceCatalog/observed time-scale mismatch: {} != {}",
            source.time.scale.as_str(),
            observed_times.scale.as_str()
        )));
    }
    let observer_times = problem
        .observers
        .coordinates
        .times
        .as_ref()
        .ok_or_else(|| value_error("orbit-determination observer coordinates require time"))?;
    if observer_times.scale != source.time.scale {
        return Err(value_error(format!(
            "SourceCatalog/observer time-scale mismatch: {} != {}",
            source.time.scale.as_str(),
            observer_times.scale.as_str()
        )));
    }
    for row in 0..source.len() {
        let observed_epoch = observed_times.epochs[row];
        if observed_epoch.days != source.time.days[row]
            || observed_epoch.nanos != source.time.nanos[row]
        {
            return Err(value_error(format!(
                "SourceCatalog/observed epoch mismatch at row {row}"
            )));
        }
        let observer_epoch = observer_times.epochs[row];
        if observer_epoch.days != source.time.days[row]
            || observer_epoch.nanos != source.time.nanos[row]
        {
            return Err(value_error(format!(
                "SourceCatalog/observer epoch mismatch at row {row}"
            )));
        }
    }
    Ok(())
}

fn checked_build_identity() -> PyResult<(&'static str, &'static str)> {
    let version = env!("CARGO_PKG_VERSION");
    let commit = env!("ADAM_CORE_BUILD_COMMIT");
    if commit.len() != 40
        || !commit
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(PyValueError::new_err(
            "canonical OD requires an exact clean adam-core build identity",
        ));
    }
    Ok((version, commit))
}

fn provider_json(metadata: &ProviderMetadata) -> String {
    json!({
        "name": &metadata.name,
        "version": &metadata.version,
        "abi_version": metadata.abi_version,
        "identity_validated_by_capsule": metadata.identity_validated_by_capsule,
        "capabilities": metadata.capabilities,
        "known_capabilities": metadata.known_capabilities,
        "integration_time_scale": &metadata.integration_time_scale,
        "two_body_dynamics": metadata.two_body_dynamics,
        "configuration_fingerprint": &metadata.configuration_fingerprint,
        "data_fingerprint": &metadata.data_fingerprint,
    })
    .to_string()
}

fn fingerprint_string(digest: &mut Sha256, value: &str) {
    digest.update((value.len() as u64).to_le_bytes());
    digest.update(value.as_bytes());
}

fn fingerprint_optional_string(digest: &mut Sha256, value: Option<&str>) {
    digest.update([u8::from(value.is_some())]);
    if let Some(value) = value {
        fingerprint_string(digest, value);
    }
}

fn fingerprint_optional_f64(digest: &mut Sha256, value: Option<f64>) {
    digest.update([u8::from(value.is_some())]);
    if let Some(value) = value {
        digest.update(value.to_bits().to_le_bytes());
    }
}

fn member_input_fingerprint(
    source: &SourceCatalogBatch,
    observers: &DataObserverBatch,
    source_row: usize,
    original: &AstrometrySnapshot,
    used: &AstrometrySnapshot,
    snapshot_row: usize,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"adam-core-od-member-input/v1\0");
    fingerprint_string(&mut digest, &source.id[source_row]);
    fingerprint_optional_string(&mut digest, source.exposure_id[source_row].as_deref());
    fingerprint_string(&mut digest, source.time.scale.as_str());
    digest.update(source.time.days[source_row].to_le_bytes());
    digest.update(source.time.nanos[source_row].to_le_bytes());
    for value in [source.ra[source_row], source.dec[source_row]] {
        digest.update(value.to_bits().to_le_bytes());
    }
    for value in [
        source.ra_sigma[source_row],
        source.dec_sigma[source_row],
        source.radec_corr[source_row],
        source.mag[source_row],
        source.mag_sigma[source_row],
        source.fwhm[source_row],
        source.a[source_row],
        source.a_sigma[source_row],
        source.b[source_row],
        source.b_sigma[source_row],
        source.pa[source_row],
        source.pa_sigma[source_row],
        source.exposure_duration[source_row],
        source.exposure_seeing[source_row],
        source.exposure_depth_5sigma[source_row],
    ] {
        fingerprint_optional_f64(&mut digest, value);
    }
    fingerprint_string(&mut digest, &source.observatory_code[source_row]);
    fingerprint_optional_string(&mut digest, source.filter[source_row].as_deref());
    let exposure_start_valid = source
        .exposure_start_time
        .validity
        .as_ref()
        .is_none_or(|validity| validity[source_row]);
    digest.update([u8::from(exposure_start_valid)]);
    if exposure_start_valid {
        fingerprint_string(&mut digest, source.exposure_start_time.scale.as_str());
        digest.update(source.exposure_start_time.days[source_row].to_le_bytes());
        digest.update(source.exposure_start_time.nanos[source_row].to_le_bytes());
    }
    fingerprint_optional_string(&mut digest, source.object_id[source_row].as_deref());
    fingerprint_string(&mut digest, &source.catalog_id[source_row]);
    fingerprint_optional_string(&mut digest, source.astcat[source_row].as_deref());
    fingerprint_optional_string(&mut digest, source.photcat[source_row].as_deref());

    fingerprint_string(&mut digest, &observers.code[source_row].0);
    fingerprint_string(&mut digest, observers.coordinates.frame.as_str());
    fingerprint_string(
        &mut digest,
        &observers.coordinates.origins.origins[source_row].code(),
    );
    for value in observers.coordinates.values.raw_values()[source_row] {
        digest.update(value.to_bits().to_le_bytes());
    }
    digest.update([u8::from(observers.coordinates.times.is_some())]);
    if let Some(times) = &observers.coordinates.times {
        fingerprint_string(&mut digest, times.scale.as_str());
        digest.update(times.epochs[source_row].days.to_le_bytes());
        digest.update(times.epochs[source_row].nanos.to_le_bytes());
    }
    digest.update([u8::from(observers.coordinates.covariance.is_some())]);
    if let Some(covariance) = &observers.coordinates.covariance {
        digest.update((covariance.row_dimension(source_row) as u64).to_le_bytes());
        digest.update([u8::from(covariance.is_row_valid(source_row))]);
        for value in covariance.row_values(source_row) {
            digest.update(value.to_bits().to_le_bytes());
        }
    }
    for value in [
        original.lon[snapshot_row],
        original.lat[snapshot_row],
        original.sigma_lon[snapshot_row],
        original.sigma_lat[snapshot_row],
        original.variance_lon[snapshot_row],
        original.variance_lat[snapshot_row],
        original.cov_lonlat[snapshot_row],
        used.lon[snapshot_row],
        used.lat[snapshot_row],
        used.sigma_lon[snapshot_row],
        used.sigma_lat[snapshot_row],
        used.variance_lon[snapshot_row],
        used.variance_lat[snapshot_row],
        used.cov_lonlat[snapshot_row],
    ] {
        digest.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn member_input_fingerprints(
    source: &SourceCatalogBatch,
    observers: &DataObserverBatch,
    rows: &[usize],
    original: &AstrometrySnapshot,
    used: &AstrometrySnapshot,
) -> Vec<String> {
    rows.iter()
        .enumerate()
        .map(|(snapshot_row, &source_row)| {
            member_input_fingerprint(source, observers, source_row, original, used, snapshot_row)
        })
        .collect()
}

fn membership_fingerprint(
    group_id: &str,
    orbit_id: &str,
    observation_id: &str,
    member_fingerprint: &str,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"adam-core-od-membership-pair/v1\0");
    fingerprint_string(&mut digest, group_id);
    fingerprint_string(&mut digest, orbit_id);
    fingerprint_string(&mut digest, observation_id);
    fingerprint_string(&mut digest, member_fingerprint);
    format!("{:x}", digest.finalize())
}

fn orbit_state_fingerprint(
    orbits: &DataOrbitBatch,
    row: usize,
    state_override: Option<&[f64; 6]>,
    covariance_override: Option<&[f64; 36]>,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"adam-core-od-orbit-state/v1\0");
    fingerprint_string(&mut digest, orbits.coordinates.frame.as_str());
    fingerprint_string(&mut digest, &orbits.coordinates.origins.origins[row].code());
    let state = state_override
        .copied()
        .unwrap_or(orbits.coordinates.values.raw_values()[row]);
    for value in state {
        digest.update(value.to_bits().to_le_bytes());
    }
    digest.update([u8::from(orbits.coordinates.times.is_some())]);
    if let Some(times) = &orbits.coordinates.times {
        fingerprint_string(&mut digest, times.scale.as_str());
        digest.update(times.epochs[row].days.to_le_bytes());
        digest.update(times.epochs[row].nanos.to_le_bytes());
    }
    let covariance = covariance_override.map(<[f64; 36]>::as_slice).or_else(|| {
        orbits
            .coordinates
            .covariance
            .as_ref()
            .map(|values| values.row_values(row))
    });
    digest.update([u8::from(covariance.is_some())]);
    if let Some(covariance) = covariance {
        digest.update((covariance.len() as u64).to_le_bytes());
        for &value in covariance {
            digest.update(value.to_bits().to_le_bytes());
        }
    }
    digest.update([u8::from(orbits.physical_parameters.is_some())]);
    if let Some(parameters) = &orbits.physical_parameters {
        for value in [
            parameters.h_v[row],
            parameters.h_v_sigma[row],
            parameters.g[row],
            parameters.g_sigma[row],
            parameters.sigma_eff[row],
            parameters.chi2_red[row],
        ] {
            fingerprint_optional_f64(&mut digest, value);
        }
    }
    digest.update([u8::from(orbits.non_gravitational_parameters.is_some())]);
    if let Some(parameters) = &orbits.non_gravitational_parameters {
        fingerprint_optional_string(&mut digest, parameters.source[row].as_deref());
        let values = parameters.row(row);
        for value in [
            values.a1, values.a2, values.a3, values.aln, values.nk, values.nm, values.nn, values.r0,
        ] {
            fingerprint_optional_f64(&mut digest, value);
        }
    }
    format!("{:x}", digest.finalize())
}

fn output_orbit_state_fingerprint(
    iod: Option<&IodResult>,
    refined: Option<&adam_core_rs_coords::propagation::RefinedOutput>,
    input_orbit: Option<(&DataOrbitBatch, usize)>,
) -> PyResult<Option<String>> {
    let override_state = refined.map(|value| &value.fit.state);
    let override_covariance = refined.map(|value| &value.fit.covariance);
    if let Some((orbits, row)) = input_orbit {
        return Ok(Some(orbit_state_fingerprint(
            orbits,
            row,
            override_state,
            override_covariance,
        )));
    }
    let Some(iod) = iod else {
        return Ok(None);
    };
    let Some(seed) = iod.seed_orbit().map_err(od_error)? else {
        return Ok(None);
    };
    Ok(Some(orbit_state_fingerprint(
        &seed,
        0,
        override_state,
        override_covariance,
    )))
}

fn group_input_fingerprint(
    group_id: &str,
    source: &SourceCatalogBatch,
    observers: &DataObserverBatch,
    rows: &[usize],
    original: &AstrometrySnapshot,
    used: &AstrometrySnapshot,
    input_orbit: Option<(&DataOrbitBatch, usize)>,
    member_outlier: Option<&[bool]>,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"adam-core-od-input/v1\0");
    fingerprint_string(&mut digest, group_id);
    if let Some((orbits, row)) = input_orbit {
        digest.update([1]);
        fingerprint_string(&mut digest, &orbits.orbit_id[row].0);
        fingerprint_optional_string(
            &mut digest,
            orbits.object_id[row].as_ref().map(|value| value.0.as_str()),
        );
        fingerprint_string(
            &mut digest,
            &orbit_state_fingerprint(orbits, row, None, None),
        );
    } else {
        digest.update([0]);
    }
    digest.update([u8::from(member_outlier.is_some())]);
    if let Some(member_outlier) = member_outlier {
        digest.update((member_outlier.len() as u64).to_le_bytes());
        for value in member_outlier {
            digest.update([u8::from(*value)]);
        }
    }
    for fingerprint in member_input_fingerprints(source, observers, rows, original, used) {
        fingerprint_string(&mut digest, &fingerprint);
    }
    format!("{:x}", digest.finalize())
}

fn canonical_orbit_id(group_id: &str, config_sha256: &str, input_fingerprint: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(b"adam-core-od-orbit/v1\0");
    fingerprint_string(&mut digest, group_id);
    fingerprint_string(&mut digest, config_sha256);
    fingerprint_string(&mut digest, input_fingerprint);
    format!("{:x}", digest.finalize())
}

fn configuration_identity(configuration: &str) -> Result<String, String> {
    let mut value: serde_json::Value = serde_json::from_str(configuration)
        .map_err(|error| format!("invalid orbit configuration: {error}"))?;
    let declared = value
        .get("configuration_sha256")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    if let Some(declared) = declared {
        if !is_sha256(&declared) {
            return Err("configuration_sha256 is not a lowercase SHA-256".to_string());
        }
        let object = value
            .as_object_mut()
            .ok_or_else(|| "configuration provenance is not an object".to_string())?;
        object.remove("configuration_sha256");
        // Profile identity records selection, not science, and is intentionally
        // excluded by OdConfig::configuration_sha256.
        object.remove("profile");
        let canonical = serde_json::to_vec(&value)
            .map_err(|error| format!("configuration is not serializable: {error}"))?;
        let actual = format!("{:x}", Sha256::digest(canonical));
        if declared != actual {
            return Err("configuration_sha256 does not match canonical configuration".to_string());
        }
        Ok(declared)
    } else {
        // Standalone stage configurations have no embedded full-OD science
        // digest, so bind their exact canonical JSON value.
        let canonical = serde_json::to_vec(&value)
            .map_err(|error| format!("configuration is not serializable: {error}"))?;
        Ok(format!("{:x}", Sha256::digest(canonical)))
    }
}

fn provenance_fingerprint(route: &str, configuration_sha256: &str, input: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(b"adam-core-od-provenance/v1\0");
    fingerprint_string(&mut digest, route);
    fingerprint_string(&mut digest, configuration_sha256);
    fingerprint_string(&mut digest, input);
    format!("{:x}", digest.finalize())
}

#[derive(Debug, Clone)]
struct BatchFailure {
    stage: &'static str,
    code: &'static str,
    message: String,
    num_obs: usize,
    arc_length: Option<f64>,
    triplets_tried: Option<usize>,
    candidates_evaluated: Option<usize>,
    candidates_failed: Option<usize>,
    corrections_run: Option<usize>,
    passes: Option<usize>,
    n_rejected: Option<usize>,
    n_recovered: Option<usize>,
    status_code: Option<i64>,
    best_state_retained: bool,
}

fn group_scientific_code(error: &adam_core_rs_coords::PropagationError) -> Option<&'static str> {
    match error {
        adam_core_rs_coords::PropagationError::GroupScientific { code, .. } => Some(code.as_str()),
        _ => None,
    }
}

fn local_scientific_error(error: &adam_core_rs_coords::PropagationError) -> bool {
    group_scientific_code(error).is_some()
}

#[derive(Debug, Clone, Deserialize)]
struct OrbitProvenanceInput {
    adam_core_version: Option<String>,
    adam_core_build: Option<String>,
    input_fingerprint: Option<String>,
    state_fingerprint: Option<String>,
    provenance_fingerprint: Option<String>,
    route: Option<String>,
    iod_backend: Option<String>,
    iod_provider: Option<String>,
    iod_diagnostics: Option<String>,
    configuration: Option<String>,
    iod_ambiguous: Option<bool>,
    iod_alternatives: Option<usize>,
}

#[derive(Debug, Clone)]
struct SeedIodProvenance {
    input_fingerprint: String,
    iod_backend: String,
    iod_provider: String,
    iod_diagnostics: String,
    configuration: String,
    iod_ambiguous: bool,
    iod_alternatives: usize,
}

fn validate_seed_source_fingerprint(
    seed: Option<&SeedIodProvenance>,
    orbits: &DataOrbitBatch,
    orbit_index: usize,
    source: &SourceCatalogBatch,
    observers: &DataObserverBatch,
    source_indices: &[usize],
    prepared: &adam_core_rs_coords::od::algorithms::PreparedObservations,
) -> PyResult<()> {
    let Some(seed) = seed else {
        return Ok(());
    };
    let group_id = orbits.object_id[orbit_index]
        .as_ref()
        .map(|value| value.0.as_str())
        .ok_or_else(|| value_error("composed IOD seed has no association group object_id"))?;
    let actual = group_input_fingerprint(
        group_id,
        source,
        observers,
        source_indices,
        &prepared.original,
        &prepared.used,
        None,
        None,
    );
    if actual != seed.input_fingerprint {
        return Err(value_error(format!(
            "current observations for orbit row {orbit_index} do not match the canonical IOD input fingerprint"
        )));
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn parse_orbit_provenance(
    values: &[Option<String>],
    orbits: &DataOrbitBatch,
    refinement_config_json: Option<&str>,
) -> PyResult<Vec<Option<SeedIodProvenance>>> {
    if values.len() != orbits.len() {
        return Err(value_error(format!(
            "orbit provenance length {} does not match {} orbit rows",
            values.len(),
            orbits.len()
        )));
    }
    let expected_refinement = refinement_config_json
        .map(serde_json::from_str::<serde_json::Value>)
        .transpose()
        .map_err(|error| {
            value_error(format!(
                "invalid canonical refinement configuration: {error}"
            ))
        })?;
    let (version, build) = checked_build_identity()?;
    values
        .iter()
        .enumerate()
        .map(|(row, encoded)| {
            let encoded = encoded.as_ref().ok_or_else(|| {
                value_error(format!(
                    "canonical orbit row {row} has no complete Rust provenance"
                ))
            })?;
            let provenance: OrbitProvenanceInput = serde_json::from_str(encoded).map_err(|error| {
                value_error(format!("invalid orbit provenance at row {row}: {error}"))
            })?;
            let input_fingerprint = provenance
                .input_fingerprint
                .as_deref()
                .filter(|value| is_sha256(value))
                .ok_or_else(|| value_error(format!("orbit provenance at row {row} has no valid input fingerprint")))?;
            let state_fingerprint = provenance
                .state_fingerprint
                .as_deref()
                .filter(|value| is_sha256(value))
                .ok_or_else(|| value_error(format!("orbit provenance at row {row} has no valid state fingerprint")))?;
            if provenance.adam_core_version.as_deref() != Some(version)
                || provenance.adam_core_build.as_deref() != Some(build)
            {
                return Err(value_error(format!(
                    "orbit provenance at row {row} has incompatible Core build identity"
                )));
            }
            let actual_state = orbit_state_fingerprint(orbits, row, None, None);
            if actual_state != state_fingerprint {
                return Err(value_error(format!(
                    "orbit row {row} does not match its Rust state fingerprint"
                )));
            }
            let route = provenance.route.as_deref().ok_or_else(|| {
                value_error(format!("orbit provenance at row {row} has no route"))
            })?;
            if !matches!(route, "run_iod" | "run_od" | "refine" | "evaluate_orbits") {
                return Err(value_error(format!(
                    "orbit provenance at row {row} has unknown route {route:?}"
                )));
            }
            let configuration = provenance.configuration.as_deref().ok_or_else(|| {
                value_error(format!("orbit provenance at row {row} has no configuration"))
            })?;
            let configuration_value: serde_json::Value = serde_json::from_str(configuration)
                .map_err(|error| value_error(format!("invalid orbit configuration at row {row}: {error}")))?;
            let configuration_sha = configuration_identity(configuration).map_err(|error| {
                value_error(format!("orbit configuration at row {row} is invalid: {error}"))
            })?;
            let declared_provenance_fingerprint = provenance
                .provenance_fingerprint
                .as_deref()
                .filter(|value| is_sha256(value))
                .ok_or_else(|| value_error(format!("orbit provenance at row {row} has no valid provenance fingerprint")))?;
            let expected_provenance_fingerprint =
                provenance_fingerprint(route, &configuration_sha, input_fingerprint);
            if declared_provenance_fingerprint != expected_provenance_fingerprint {
                return Err(value_error(format!(
                    "orbit provenance at row {row} does not bind its route, configuration, and input fingerprint"
                )));
            }
            if route != "run_iod" {
                return Ok(None);
            }
            if let Some(expected_refinement) = expected_refinement.as_ref() {
                if configuration_value.get("refinement") != Some(expected_refinement) {
                    return Err(value_error(format!(
                        "seed IOD refinement configuration at orbit row {row} does not match the requested refinement"
                    )));
                }
            }
            let config_sha = configuration_value
                .get("configuration_sha256")
                .and_then(serde_json::Value::as_str)
                .filter(|value| **value == configuration_sha)
                .ok_or_else(|| value_error(format!("seed IOD configuration at orbit row {row} has no valid configuration_sha256")))?;
            let object_id = orbits.object_id[row]
                .as_ref()
                .map(|value| value.0.as_str())
                .ok_or_else(|| value_error(format!("seed IOD orbit at row {row} requires its association group object_id")))?;
            if orbits.orbit_id[row].0 != canonical_orbit_id(object_id, config_sha, input_fingerprint) {
                return Err(value_error(format!(
                    "seed IOD orbit ID at row {row} is inconsistent with its canonical provenance"
                )));
            }
            let iod_provider = provenance.iod_provider.as_deref().ok_or_else(|| {
                value_error(format!("seed IOD provenance at row {row} has no provider"))
            })?;
            let provider: serde_json::Value = serde_json::from_str(iod_provider)
                .map_err(|error| value_error(format!("invalid seed IOD provider provenance at orbit row {row}: {error}")))?;
            for field in ["name", "version", "configuration_fingerprint", "data_fingerprint"] {
                if provider.get(field).and_then(serde_json::Value::as_str).is_none_or(str::is_empty) {
                    return Err(value_error(format!(
                        "seed IOD provider provenance at orbit row {row} has no {field}"
                    )));
                }
            }
            let iod_backend = provenance.iod_backend.filter(|value| !value.trim().is_empty())
                .ok_or_else(|| value_error(format!("seed IOD provenance at row {row} has no backend")))?;
            let iod_diagnostics = provenance.iod_diagnostics.filter(|value| !value.trim().is_empty())
                .ok_or_else(|| value_error(format!("seed IOD provenance at row {row} has no diagnostics")))?;
            Ok(Some(SeedIodProvenance {
                input_fingerprint: input_fingerprint.to_string(),
                iod_backend,
                iod_provider: iod_provider.to_string(),
                iod_diagnostics,
                configuration: configuration.to_string(),
                iod_ambiguous: provenance.iod_ambiguous.ok_or_else(|| value_error(format!("seed IOD provenance at row {row} has no ambiguity flag")))?,
                iod_alternatives: provenance.iod_alternatives.ok_or_else(|| value_error(format!("seed IOD provenance at row {row} has no alternative count")))?,
            }))
        })
        .collect()
}

struct BatchIodGroup {
    group_id: String,
    orbit_id: String,
    rows: Vec<usize>,
    prepared: adam_core_rs_coords::od::algorithms::PreparedObservations,
    iod: Option<IodResult>,
    failure: Option<BatchFailure>,
}

struct BatchOdGroup {
    group_id: String,
    orbit_id: String,
    rows: Vec<usize>,
    prepared: adam_core_rs_coords::od::algorithms::PreparedObservations,
    iod: Option<IodResult>,
    refined: Option<adam_core_rs_coords::propagation::RefinedOutput>,
    failure: Option<BatchFailure>,
}

struct BatchRefineGroup {
    orbit_id: String,
    orbit_index: usize,
    rows: Vec<usize>,
    member_rows: Vec<usize>,
    prepared: adam_core_rs_coords::od::algorithms::PreparedObservations,
    refined: Option<adam_core_rs_coords::propagation::RefinedOutput>,
    failure: Option<BatchFailure>,
}

struct BatchEvaluationGroup {
    orbit_id: String,
    orbit_index: usize,
    rows: Vec<usize>,
    member_rows: Vec<usize>,
    member_outlier: Vec<bool>,
    prepared: adam_core_rs_coords::od::algorithms::PreparedObservations,
    evaluation: Option<adam_core_rs_coords::propagation::FitEvaluation>,
    failure: Option<BatchFailure>,
}

#[allow(clippy::too_many_arguments)]
fn prepared_from_member_snapshots(
    observed: DataCoordinateBatch,
    member_indices: &[usize],
    original_lon: &[Option<f64>],
    original_lat: &[Option<f64>],
    original_sigma_lon: &[Option<f64>],
    original_sigma_lat: &[Option<f64>],
    original_variance_lon: &[Option<f64>],
    original_variance_lat: &[Option<f64>],
    original_cov_lonlat: &[Option<f64>],
    used_lon: &[Option<f64>],
    used_lat: &[Option<f64>],
    used_sigma_lon: &[Option<f64>],
    used_sigma_lat: &[Option<f64>],
    used_variance_lon: &[Option<f64>],
    used_variance_lat: &[Option<f64>],
    used_cov_lonlat: &[Option<f64>],
) -> PropagationResultValue<adam_core_rs_coords::od::algorithms::PreparedObservations> {
    let required = member_indices
        .iter()
        .copied()
        .max()
        .map_or(0, |value| value + 1);
    if [
        original_lon.len(),
        original_lat.len(),
        original_sigma_lon.len(),
        original_sigma_lat.len(),
        original_variance_lon.len(),
        original_variance_lat.len(),
        original_cov_lonlat.len(),
        used_lon.len(),
        used_lat.len(),
        used_sigma_lon.len(),
        used_sigma_lat.len(),
        used_variance_lon.len(),
        used_variance_lat.len(),
        used_cov_lonlat.len(),
    ]
    .iter()
    .any(|&len| len < required)
    {
        return Err(adam_core_rs_coords::PropagationError::InvalidRequest(
            "FittedOrbitMembers astrometry snapshot columns have inconsistent lengths".to_string(),
        ));
    }
    let spherical = observed.values.spherical().ok_or_else(|| {
        adam_core_rs_coords::PropagationError::InvalidRequest(
            "OD observations must be spherical".to_string(),
        )
    })?;
    let source_lon = spherical.iter().map(|value| value[1]).collect::<Vec<_>>();
    let source_lat = spherical.iter().map(|value| value[2]).collect::<Vec<_>>();
    let source_covariance = observed.covariance.as_ref().ok_or_else(|| {
        adam_core_rs_coords::PropagationError::InvalidRequest(
            "refinement observations require covariance".to_string(),
        )
    })?;
    if source_covariance.dimension != 6 {
        return Err(adam_core_rs_coords::PropagationError::InvalidRequest(
            "refinement requires six-dimensional spherical observation covariance".to_string(),
        ));
    }
    let source_original = AstrometrySnapshot::from_columns(
        &source_lon,
        &source_lat,
        &source_covariance.values_row_major,
    );
    let mut values = observed.values.raw_values().to_vec();
    let mut covariance = source_covariance.values_row_major.clone();
    let mut original = AstrometrySnapshot {
        lon: Vec::with_capacity(member_indices.len()),
        lat: Vec::with_capacity(member_indices.len()),
        sigma_lon: Vec::with_capacity(member_indices.len()),
        sigma_lat: Vec::with_capacity(member_indices.len()),
        variance_lon: Vec::with_capacity(member_indices.len()),
        variance_lat: Vec::with_capacity(member_indices.len()),
        cov_lonlat: Vec::with_capacity(member_indices.len()),
    };
    let mut lon = Vec::with_capacity(member_indices.len());
    let mut lat = Vec::with_capacity(member_indices.len());
    let mut models_changed = false;
    let equal = |left: f64, right: f64| {
        left.to_bits() == right.to_bits() || (left.is_nan() && right.is_nan())
    };
    for (row, &member) in member_indices.iter().enumerate() {
        let original_value = |column: &[Option<f64>]| {
            column[member].ok_or_else(|| {
                adam_core_rs_coords::PropagationError::InvalidRequest(
                    "refinement/evaluation requires original_astrometry for every declared member"
                        .to_string(),
                )
            })
        };
        let persisted_original = [
            original_value(original_lon)?,
            original_value(original_lat)?,
            original_value(original_sigma_lon)?,
            original_value(original_sigma_lat)?,
            original_value(original_variance_lon)?,
            original_value(original_variance_lat)?,
            original_value(original_cov_lonlat)?,
        ];
        let source_values = [
            source_original.lon[row],
            source_original.lat[row],
            source_original.sigma_lon[row],
            source_original.sigma_lat[row],
            source_original.variance_lon[row],
            source_original.variance_lat[row],
            source_original.cov_lonlat[row],
        ];
        if persisted_original
            .iter()
            .zip(source_values)
            .any(|(&persisted, source)| !equal(persisted, source))
        {
            return Err(adam_core_rs_coords::PropagationError::InvalidRequest(
                format!(
                    "observation source row {row} does not match persisted original_astrometry member row {member}"
                ),
            ));
        }
        original.lon.push(persisted_original[0]);
        original.lat.push(persisted_original[1]);
        original.sigma_lon.push(persisted_original[2]);
        original.sigma_lat.push(persisted_original[3]);
        original.variance_lon.push(persisted_original[4]);
        original.variance_lat.push(persisted_original[5]);
        original.cov_lonlat.push(persisted_original[6]);

        let member_value = |column: &[Option<f64>]| {
            column[member].ok_or_else(|| {
                adam_core_rs_coords::PropagationError::InvalidRequest(
                    "refinement/evaluation requires used_astrometry for every declared member"
                        .to_string(),
                )
            })
        };
        let used_lon = member_value(used_lon)?;
        let used_lat = member_value(used_lat)?;
        let sigma_lon = member_value(used_sigma_lon)?;
        let sigma_lat = member_value(used_sigma_lat)?;
        let variance_lon = member_value(used_variance_lon)?;
        let variance_lat = member_value(used_variance_lat)?;
        if !used_lon.is_finite()
            || !(0.0..360.0).contains(&used_lon)
            || !used_lat.is_finite()
            || !(-90.0..=90.0).contains(&used_lat)
            || !sigma_lon.is_finite()
            || !sigma_lat.is_finite()
            || !variance_lon.is_finite()
            || !variance_lat.is_finite()
            || sigma_lon <= 0.0
            || sigma_lat <= 0.0
            || variance_lon <= 0.0
            || variance_lat <= 0.0
            || !equal(sigma_lon, variance_lon.sqrt())
            || !equal(sigma_lat, variance_lat.sqrt())
        {
            return Err(adam_core_rs_coords::PropagationError::InvalidRequest(
                format!("member used_astrometry row {member} is not finite and positive"),
            ));
        }
        let cross = used_cov_lonlat[member].unwrap_or(f64::NAN);
        if cross.is_infinite()
            || (cross.is_finite() && variance_lon * variance_lat - cross.powi(2) <= 0.0)
        {
            return Err(adam_core_rs_coords::PropagationError::InvalidRequest(
                format!("member used_astrometry row {member} covariance is not positive definite"),
            ));
        }
        models_changed |= !equal(persisted_original[0], used_lon)
            || !equal(persisted_original[1], used_lat)
            || !equal(persisted_original[2], sigma_lon)
            || !equal(persisted_original[3], sigma_lat)
            || !equal(persisted_original[4], variance_lon)
            || !equal(persisted_original[5], variance_lat)
            || !equal(persisted_original[6], cross);
        values[row][1] = used_lon;
        values[row][2] = used_lat;
        let block = &mut covariance[row * 36..(row + 1) * 36];
        block[7] = variance_lon;
        block[14] = variance_lat;
        block[8] = cross;
        block[13] = cross;
        lon.push(used_lon);
        lat.push(used_lat);
    }
    let used = AstrometrySnapshot::from_columns(&lon, &lat, &covariance);
    let mut covariance_batch = adam_core_rs_coords::CovarianceBatch::new(
        member_indices.len(),
        source_covariance.dimension,
        covariance,
        source_covariance.units.clone(),
    )?;
    if let Some(validity) = &source_covariance.row_validity {
        covariance_batch = covariance_batch.with_row_validity(validity.clone())?;
    }
    let coordinates = DataCoordinateBatch::new(
        adam_core_rs_coords::CoordinateValues::Spherical(values),
        observed.frame,
        observed.origins,
        observed.times,
        Some(covariance_batch),
    )?;
    Ok(adam_core_rs_coords::propagation::AppliedObservationModels {
        original,
        used,
        models_changed,
        observations: coordinates,
    })
}

fn prepare_association_groups(
    groups: &[adam_core_rs_coords::od::AssociationGroup],
    source: &SourceCatalogBatch,
    problem: &OdObservations,
    config: &OdConfig,
) -> PyResult<
    Vec<(
        adam_core_rs_coords::od::algorithms::PreparedObservations,
        DataObserverBatch,
    )>,
> {
    groups
        .iter()
        .map(|group| {
            let observed = problem
                .observed
                .take(&group.source_indices)
                .map_err(|error| value_error(error.to_string()))?;
            let observers = problem
                .observers
                .take(&group.source_indices)
                .map_err(|error| value_error(error.to_string()))?;
            let obs_code = group
                .source_indices
                .iter()
                .map(|&row| source.observatory_code[row].clone())
                .collect::<Vec<_>>();
            let band = group
                .source_indices
                .iter()
                .map(|&row| source.filter[row].clone())
                .collect::<Vec<_>>();
            let astcat = group
                .source_indices
                .iter()
                .map(|&row| source.astcat[row].clone())
                .collect::<Vec<_>>();
            let prepared = adam_core_rs_coords::od::algorithms::prepare_observations(
                &observed,
                &obs_code,
                &band,
                &astcat,
                config,
                &ErfaTimeProvider,
            )
            .map_err(od_error)?;
            Ok((prepared, observers))
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn prepare_member_groups(
    groups: &[adam_core_rs_coords::od::OrbitMemberGroup],
    source: &SourceCatalogBatch,
    problem: &OdObservations,
    orbits: &DataOrbitBatch,
    seed_iod_provenance: Option<&[Option<SeedIodProvenance>]>,
    persisted_member_fingerprints: &[Option<String>],
    persisted_membership_fingerprints: &[Option<String>],
    original_lon: &[Option<f64>],
    original_lat: &[Option<f64>],
    original_sigma_lon: &[Option<f64>],
    original_sigma_lat: &[Option<f64>],
    original_variance_lon: &[Option<f64>],
    original_variance_lat: &[Option<f64>],
    original_cov_lonlat: &[Option<f64>],
    used_lon: &[Option<f64>],
    used_lat: &[Option<f64>],
    used_sigma_lon: &[Option<f64>],
    used_sigma_lat: &[Option<f64>],
    used_variance_lon: &[Option<f64>],
    used_variance_lat: &[Option<f64>],
    used_cov_lonlat: &[Option<f64>],
) -> PyResult<
    Vec<(
        adam_core_rs_coords::od::algorithms::PreparedObservations,
        DataObserverBatch,
    )>,
> {
    groups
        .iter()
        .map(|group| {
            let observed = problem
                .observed
                .take(&group.source_indices)
                .map_err(|error| value_error(error.to_string()))?;
            let observers = problem
                .observers
                .take(&group.source_indices)
                .map_err(|error| value_error(error.to_string()))?;
            let prepared = prepared_from_member_snapshots(
                observed,
                &group.member_indices,
                original_lon,
                original_lat,
                original_sigma_lon,
                original_sigma_lat,
                original_variance_lon,
                original_variance_lat,
                original_cov_lonlat,
                used_lon,
                used_lat,
                used_sigma_lon,
                used_sigma_lat,
                used_variance_lon,
                used_variance_lat,
                used_cov_lonlat,
            )
            .map_err(od_error)?;
            let actual_member_fingerprints = member_input_fingerprints(
                source,
                &problem.observers,
                &group.source_indices,
                &prepared.original,
                &prepared.used,
            );
            for (&member_row, actual) in group
                .member_indices
                .iter()
                .zip(actual_member_fingerprints.iter())
            {
                let persisted = persisted_member_fingerprints[member_row]
                    .as_deref()
                    .filter(|value| is_sha256(value))
                    .ok_or_else(|| {
                        value_error(format!(
                            "FittedOrbitMembers input_fingerprint is missing or invalid at row {member_row}"
                        ))
                    })?;
                if persisted != actual {
                    return Err(value_error(format!(
                        "current source/observer/prepared input does not match member fingerprint at row {member_row}"
                    )));
                }
            }
            let group_id = orbit_group_id(orbits, group.orbit_index, &group.orbit_id);
            for ((&source_row, &member_row), member_fingerprint) in group
                .source_indices
                .iter()
                .zip(group.member_indices.iter())
                .zip(actual_member_fingerprints.iter())
            {
                let actual = membership_fingerprint(
                    group_id,
                    &group.orbit_id,
                    &source.id[source_row],
                    member_fingerprint,
                );
                let persisted = persisted_membership_fingerprints[member_row]
                    .as_deref()
                    .filter(|value| is_sha256(value))
                    .ok_or_else(|| {
                        value_error(format!(
                            "FittedOrbitMembers membership_fingerprint is missing or invalid at row {member_row}"
                        ))
                    })?;
                if persisted != actual {
                    return Err(value_error(format!(
                        "current explicit membership does not match its Rust fingerprint at row {member_row}"
                    )));
                }
            }
            validate_seed_source_fingerprint(
                seed_iod_provenance.and_then(|values| values[group.orbit_index].as_ref()),
                orbits,
                group.orbit_index,
                source,
                &problem.observers,
                &group.source_indices,
                &prepared,
            )?;
            Ok((prepared, observers))
        })
        .collect()
}

fn iod_failure(result: &IodResult, n: usize) -> Option<BatchFailure> {
    let output = result.to_output(n);
    (!output.found).then(|| BatchFailure {
        stage: "iod",
        code: "no_solution",
        message: output
            .not_found_reason
            .unwrap_or_else(|| "IOD found no canonical state".to_string()),
        num_obs: n,
        arc_length: None,
        triplets_tried: Some(output.triplets_tried),
        candidates_evaluated: Some(output.candidates_evaluated),
        candidates_failed: Some(output.candidates_failed),
        corrections_run: Some(result.corrections_run),
        passes: None,
        n_rejected: None,
        n_recovered: None,
        status_code: None,
        best_state_retained: false,
    })
}

fn failure_dict<'py>(
    py: Python<'py>,
    group_id: &str,
    orbit_id: Option<&str>,
    input_orbit_id: Option<&str>,
    failure: &BatchFailure,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("group_id", group_id)?;
    out.set_item("orbit_id", orbit_id)?;
    out.set_item("input_orbit_id", input_orbit_id)?;
    out.set_item("stage", failure.stage)?;
    out.set_item("code", failure.code)?;
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

#[allow(clippy::too_many_arguments)]
fn provenance_dict<'py>(
    py: Python<'py>,
    route: &str,
    input_fingerprint: String,
    state_fingerprint: Option<String>,
    config_json: &str,
    iod_provider: Option<&ProviderMetadata>,
    refinement_provider: Option<&ProviderMetadata>,
    evaluation_provider: Option<&ProviderMetadata>,
    iod: Option<&IodResult>,
    refined: Option<&adam_core_rs_coords::propagation::RefinedOutput>,
    evaluation: Option<&adam_core_rs_coords::propagation::FitEvaluation>,
    seed_iod: Option<&SeedIodProvenance>,
) -> PyResult<Bound<'py, PyDict>> {
    let (version, build) = checked_build_identity()?;
    let effective_route = if seed_iod.is_some() { "run_od" } else { route };
    let effective_input_fingerprint = seed_iod
        .map(|seed| seed.input_fingerprint.as_str())
        .unwrap_or(input_fingerprint.as_str());
    let effective_config_json = seed_iod
        .map(|seed| seed.configuration.as_str())
        .unwrap_or(config_json);
    let configuration_sha = configuration_identity(effective_config_json)
        .map_err(|error| value_error(format!("cannot fingerprint orbit provenance: {error}")))?;
    let out = PyDict::new(py);
    out.set_item("adam_core_version", version)?;
    out.set_item("adam_core_build", build)?;
    out.set_item("input_fingerprint", effective_input_fingerprint)?;
    out.set_item("state_fingerprint", state_fingerprint)?;
    out.set_item(
        "provenance_fingerprint",
        provenance_fingerprint(
            effective_route,
            &configuration_sha,
            effective_input_fingerprint,
        ),
    )?;
    out.set_item("route", effective_route)?;
    out.set_item("fitter", py.None())?;
    let backend = |provider: &ProviderMetadata| format!("{} {}", provider.name, provider.version);
    out.set_item(
        "iod_backend",
        seed_iod
            .map(|seed| seed.iod_backend.clone())
            .or_else(|| iod_provider.map(backend)),
    )?;
    out.set_item("refinement_backend", refinement_provider.map(backend))?;
    out.set_item("evaluation_backend", evaluation_provider.map(backend))?;
    out.set_item(
        "iod_provider",
        seed_iod
            .map(|seed| seed.iod_provider.clone())
            .or_else(|| iod_provider.map(provider_json)),
    )?;
    out.set_item(
        "refinement_provider",
        refinement_provider.map(provider_json),
    )?;
    out.set_item(
        "evaluation_provider",
        evaluation_provider.map(provider_json),
    )?;
    let iod_diagnostics = seed_iod
        .map(|seed| seed.iod_diagnostics.clone())
        .or_else(|| {
            iod.map(|result| {
                let output = result.to_output(0);
                json!({
                    "triplets_tried": output.triplets_tried,
                    "candidates_evaluated": output.candidates_evaluated,
                    "candidates_failed": output.candidates_failed,
                    "corrections_run": result.corrections_run,
                    "search_cap_exhausted": result.search_cap_exhausted,
                })
                .to_string()
            })
        });
    let refinement_diagnostics = refined.map(|result| {
        json!({
            "passes": result.passes,
            "n_rejected": result.n_rejected,
            "n_recovered": result.n_recovered,
            "flags": &result.flags,
        })
        .to_string()
    });
    let evaluation_diagnostics = evaluation.map(|result| {
        json!({
            "num_obs": result.num_obs,
            "arc_length": result.arc_length,
            "chi2": result.orbit_chi2,
            "reduced_chi2": result.reduced_chi2,
        })
        .to_string()
    });
    out.set_item("iod_diagnostics", iod_diagnostics)?;
    out.set_item("refinement_diagnostics", refinement_diagnostics)?;
    out.set_item("evaluation_diagnostics", evaluation_diagnostics)?;
    out.set_item("settings", py.None())?;
    out.set_item("configuration", effective_config_json)?;
    out.set_item(
        "iod_ambiguous",
        seed_iod
            .map(|seed| seed.iod_ambiguous)
            .or_else(|| iod.map(|result| !result.ambiguous_with.is_empty())),
    )?;
    out.set_item(
        "iod_alternatives",
        seed_iod
            .map(|seed| seed.iod_alternatives)
            .or_else(|| iod.map(|result| result.ambiguous_with.len())),
    )?;
    out.set_item("num_rejected", refined.map(|result| result.n_rejected))?;
    out.set_item(
        "rejected_fraction",
        refined.map(|result| {
            if result.fit.evaluation.chi2.is_empty() {
                0.0
            } else {
                result.n_rejected as f64 / result.fit.evaluation.chi2.len() as f64
            }
        }),
    )?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn base_group_dict<'py>(
    py: Python<'py>,
    group_id: &str,
    orbit_id: &str,
    rows: &[usize],
    prepared: &adam_core_rs_coords::od::algorithms::PreparedObservations,
    iod_provider: Option<&ProviderMetadata>,
    refinement_provider: Option<&ProviderMetadata>,
    evaluation_provider: Option<&ProviderMetadata>,
    config_json: &str,
    route: &str,
    iod: Option<&IodResult>,
    refined: Option<&adam_core_rs_coords::propagation::RefinedOutput>,
    evaluation: Option<&adam_core_rs_coords::propagation::FitEvaluation>,
    seed_iod: Option<&SeedIodProvenance>,
    input_orbit: Option<(&DataOrbitBatch, usize)>,
    member_outlier: Option<&[bool]>,
    source: &SourceCatalogBatch,
    observers: &DataObserverBatch,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("group_id", group_id)?;
    out.set_item("orbit_id", orbit_id)?;
    out.set_item("row_indices", rows)?;
    out.set_item(
        "original_astrometry",
        snapshot_dict(py, prepared.original.clone())?,
    )?;
    out.set_item("used_astrometry", snapshot_dict(py, prepared.used.clone())?)?;
    let member_fingerprints =
        member_input_fingerprints(source, observers, rows, &prepared.original, &prepared.used);
    let membership_fingerprints = rows
        .iter()
        .zip(member_fingerprints.iter())
        .map(|(&row, fingerprint)| {
            membership_fingerprint(group_id, orbit_id, &source.id[row], fingerprint)
        })
        .collect::<Vec<_>>();
    out.set_item("member_input_fingerprints", member_fingerprints)?;
    out.set_item("member_membership_fingerprints", membership_fingerprints)?;
    out.set_item(
        "provenance",
        provenance_dict(
            py,
            route,
            group_input_fingerprint(
                group_id,
                source,
                observers,
                rows,
                &prepared.original,
                &prepared.used,
                input_orbit,
                member_outlier,
            ),
            output_orbit_state_fingerprint(iod, refined, input_orbit)?,
            config_json,
            iod_provider,
            refinement_provider,
            evaluation_provider,
            iod,
            refined,
            evaluation,
            seed_iod,
        )?,
    )?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn group_dict<'py>(
    py: Python<'py>,
    group_id: &str,
    orbit_id: &str,
    rows: &[usize],
    prepared: &adam_core_rs_coords::od::algorithms::PreparedObservations,
    iod_provider: Option<&ProviderMetadata>,
    refinement_provider: Option<&ProviderMetadata>,
    config_json: &str,
    route: &str,
    iod: Option<IodResult>,
    refined: Option<adam_core_rs_coords::propagation::RefinedOutput>,
    failure: Option<BatchFailure>,
    seed_iod: Option<&SeedIodProvenance>,
    input_orbit: Option<(&DataOrbitBatch, usize)>,
    source: &SourceCatalogBatch,
    observers: &DataObserverBatch,
) -> PyResult<Bound<'py, PyDict>> {
    let out = base_group_dict(
        py,
        group_id,
        orbit_id,
        rows,
        prepared,
        iod_provider,
        refinement_provider,
        None,
        config_json,
        route,
        iod.as_ref(),
        refined.as_ref(),
        None,
        seed_iod,
        input_orbit,
        None,
        source,
        observers,
    )?;
    let has_state = iod.as_ref().is_some_and(|result| result.accepted.is_some());
    let has_refined = refined.is_some();
    let refinement_failed_without_state = route == "run_od"
        && failure
            .as_ref()
            .is_some_and(|failure| failure.stage == "refinement")
        && !has_refined;
    let output_kind = if has_refined {
        "refined"
    } else if has_state && !refinement_failed_without_state {
        "iod"
    } else {
        "none"
    };
    out.set_item("output_kind", output_kind)?;
    out.set_item(
        "iod",
        iod.map(|result| iod_dict(py, result, rows.len()))
            .transpose()?,
    )?;
    out.set_item(
        "refined",
        refined
            .map(|result| {
                let output = fit_output_dict(py, result.fit, rows.len())?;
                output.set_item("passes", result.passes)?;
                output.set_item("n_rejected", result.n_rejected)?;
                output.set_item("n_recovered", result.n_recovered)?;
                output.set_item("flags", result.flags)?;
                Ok::<_, PyErr>(output)
            })
            .transpose()?,
    )?;
    out.set_item(
        "failure",
        failure
            .as_ref()
            .map(|failure| {
                failure_dict(
                    py,
                    group_id,
                    (output_kind != "none").then_some(orbit_id),
                    input_orbit
                        .map(|(orbits, row)| orbits.orbit_id[row].0.as_str())
                        .or_else(|| {
                            (route == "run_od" && failure.stage == "refinement").then_some(orbit_id)
                        }),
                    failure,
                )
            })
            .transpose()?,
    )?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (source_ipc, associations_ipc, observed_ipc, observers_ipc, config))]
fn iod_od_batch_preflight_ipc(
    source_ipc: &Bound<'_, PyBytes>,
    associations_ipc: &Bound<'_, PyBytes>,
    observed_ipc: &Bound<'_, PyBytes>,
    observers_ipc: &Bound<'_, PyBytes>,
    config: PyRef<'_, PyOdConfig>,
) -> PyResult<usize> {
    config.inner.validate().map_err(value_error)?;
    checked_build_identity()?;
    let source = decode_source_catalog(source_ipc)?;
    let associations = decode_association_batch(associations_ipc)?;
    let groups = adam_core_rs_coords::od::plan_associations(&source.id, &associations)
        .map_err(value_error)?;
    if groups.is_empty() {
        return Ok(0);
    }
    let problem = decode_observations(observed_ipc, observers_ipc)?;
    validate_catalog_problem(&source, &problem)?;
    Ok(groups.len())
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
    config.inner.validate().map_err(value_error)?;
    checked_build_identity()?;
    let source = decode_source_catalog(source_ipc)?;
    let associations = decode_association_batch(associations_ipc)?;
    let groups = adam_core_rs_coords::od::plan_associations(&source.id, &associations)
        .map_err(value_error)?;
    if groups.is_empty() {
        let out = PyDict::new(py);
        out.set_item("groups", PyList::empty(py))?;
        out.set_item("provider", py.None())?;
        return Ok(out);
    }
    let problem = decode_observations(observed_ipc, observers_ipc)?;
    validate_catalog_problem(&source, &problem)?;
    let config_sha = config.inner.configuration_sha256();
    let config_json = config.inner.provenance_json();
    let owned_config = config.inner.clone();
    drop(config);
    let prepared_groups = prepare_association_groups(&groups, &source, &problem, &owned_config)?;

    let correction_fit = *owned_config.for_iod().correction_fit();
    let settings = fit_settings_from_config(correction_fit)?;
    let (results, metadata) = run_with_predictor(py, propagator, &settings, |predictor| {
        let mut results = Vec::with_capacity(groups.len());
        for (group, (prepared, observers)) in groups.iter().zip(prepared_groups) {
            let input_fingerprint = group_input_fingerprint(
                &group.group_id,
                &source,
                &problem.observers,
                &group.source_indices,
                &prepared.original,
                &prepared.used,
                None,
                None,
            );
            let orbit_id = canonical_orbit_id(&group.group_id, &config_sha, &input_fingerprint);
            match adam_core_rs_coords::od::algorithms::run_iod_with_predictor(
                predictor,
                &prepared,
                &observers,
                &owned_config,
                &ErfaTimeProvider,
                &LockingSpiceTranslation,
            ) {
                Ok(iod) => {
                    let failure = iod_failure(&iod, group.source_indices.len());
                    results.push(BatchIodGroup {
                        group_id: group.group_id.clone(),
                        orbit_id,
                        rows: group.source_indices.clone(),
                        prepared,
                        iod: Some(iod),
                        failure,
                    });
                }
                Err(error) if local_scientific_error(&error) => results.push(BatchIodGroup {
                    group_id: group.group_id.clone(),
                    orbit_id,
                    rows: group.source_indices.clone(),
                    prepared,
                    iod: None,
                    failure: Some(BatchFailure {
                        stage: "iod",
                        code: group_scientific_code(&error)
                            .expect("guarded group scientific error"),
                        message: error.to_string(),
                        num_obs: group.source_indices.len(),
                        arc_length: None,
                        triplets_tried: None,
                        candidates_evaluated: None,
                        candidates_failed: None,
                        corrections_run: None,
                        passes: None,
                        n_rejected: None,
                        n_recovered: None,
                        status_code: None,
                        best_state_retained: false,
                    }),
                }),
                Err(error) => return Err(error),
            }
        }
        Ok(results)
    })?;

    let items = PyList::empty(py);
    for result in results {
        items.append(group_dict(
            py,
            &result.group_id,
            &result.orbit_id,
            &result.rows,
            &result.prepared,
            Some(&metadata),
            None,
            &config_json,
            "run_iod",
            result.iod,
            None,
            result.failure,
            None,
            None,
            &source,
            &problem.observers,
        )?)?;
    }
    let out = PyDict::new(py);
    out.set_item("groups", items)?;
    out.set_item("provider", provider_metadata_dict(py, &metadata)?)?;
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
    config.inner.validate().map_err(value_error)?;
    checked_build_identity()?;
    let source = decode_source_catalog(source_ipc)?;
    let associations = decode_association_batch(associations_ipc)?;
    let groups = adam_core_rs_coords::od::plan_associations(&source.id, &associations)
        .map_err(value_error)?;
    if groups.is_empty() {
        let out = PyDict::new(py);
        out.set_item("groups", PyList::empty(py))?;
        out.set_item("provider", py.None())?;
        return Ok(out);
    }
    let problem = decode_observations(observed_ipc, observers_ipc)?;
    validate_catalog_problem(&source, &problem)?;
    let config_sha = config.inner.configuration_sha256();
    let config_json = config.inner.provenance_json();
    let owned_config = config.inner.clone();
    drop(config);
    let prepared_groups = prepare_association_groups(&groups, &source, &problem, &owned_config)?;
    let correction_fit = *owned_config.for_iod().correction_fit();
    let iod_settings = fit_settings_from_config(correction_fit)?;
    let refinement = *owned_config.for_refinement().refinement();
    let refinement_fit = match refinement {
        RefinementConfig::WorstResidual(value) => value.fit,
        RefinementConfig::Cmc2003(value) => value.fit,
    };
    let refinement_settings = fit_settings_from_config(refinement_fit)?;
    let (results, iod_metadata, refinement_metadata) = run_with_stage_predictors(
        py,
        propagator,
        &iod_settings,
        &refinement_settings,
        |iod_predictor, refinement_predictor| {
            let mut results = Vec::with_capacity(groups.len());
            for (group, (prepared, observers)) in groups.iter().zip(prepared_groups) {
                let input_fingerprint = group_input_fingerprint(
                    &group.group_id,
                    &source,
                    &problem.observers,
                    &group.source_indices,
                    &prepared.original,
                    &prepared.used,
                    None,
                    None,
                );
                let orbit_id = canonical_orbit_id(&group.group_id, &config_sha, &input_fingerprint);
                let iod = match adam_core_rs_coords::od::algorithms::run_iod_with_predictor(
                    iod_predictor,
                    &prepared,
                    &observers,
                    &owned_config,
                    &ErfaTimeProvider,
                    &LockingSpiceTranslation,
                ) {
                    Ok(iod) => iod,
                    Err(error) if local_scientific_error(&error) => {
                        results.push(BatchOdGroup {
                            group_id: group.group_id.clone(),
                            orbit_id,
                            rows: group.source_indices.clone(),
                            prepared,
                            iod: None,
                            refined: None,
                            failure: Some(BatchFailure {
                                stage: "iod",
                                code: group_scientific_code(&error)
                                    .expect("guarded group scientific error"),
                                message: error.to_string(),
                                num_obs: group.source_indices.len(),
                                arc_length: None,
                                triplets_tried: None,
                                candidates_evaluated: None,
                                candidates_failed: None,
                                corrections_run: None,
                                passes: None,
                                n_rejected: None,
                                n_recovered: None,
                                status_code: None,
                                best_state_retained: false,
                            }),
                        });
                        continue;
                    }
                    Err(error) => return Err(error),
                };
                if let Some(failure) = iod_failure(&iod, group.source_indices.len()) {
                    results.push(BatchOdGroup {
                        group_id: group.group_id.clone(),
                        orbit_id,
                        rows: group.source_indices.clone(),
                        prepared,
                        iod: Some(iod),
                        refined: None,
                        failure: Some(failure),
                    });
                    continue;
                }
                let seed = iod
                    .seed_orbit()?
                    .expect("found IOD result has a seed orbit");
                match adam_core_rs_coords::od::algorithms::refine_with_predictor(
                    refinement_predictor,
                    &seed,
                    &prepared,
                    &observers,
                    &owned_config,
                    &ErfaTimeProvider,
                    &LockingSpiceTranslation,
                ) {
                    Ok(refined) => {
                        let failure = (!refined.fit.converged).then(|| BatchFailure {
                            stage: "refinement",
                            code: "not_converged",
                            message: format!(
                                "refinement did not converge (status_code={})",
                                refined.fit.status_code
                            ),
                            num_obs: refined.fit.evaluation.num_obs,
                            arc_length: Some(refined.fit.evaluation.arc_length),
                            triplets_tried: None,
                            candidates_evaluated: None,
                            candidates_failed: None,
                            corrections_run: None,
                            passes: Some(refined.passes),
                            n_rejected: Some(refined.n_rejected),
                            n_recovered: Some(refined.n_recovered),
                            status_code: Some(refined.fit.status_code),
                            best_state_retained: true,
                        });
                        results.push(BatchOdGroup {
                            group_id: group.group_id.clone(),
                            orbit_id,
                            rows: group.source_indices.clone(),
                            prepared,
                            iod: Some(iod),
                            refined: Some(refined),
                            failure,
                        });
                    }
                    Err(error) if local_scientific_error(&error) => results.push(BatchOdGroup {
                        group_id: group.group_id.clone(),
                        orbit_id,
                        rows: group.source_indices.clone(),
                        prepared,
                        iod: Some(iod),
                        refined: None,
                        failure: Some(BatchFailure {
                            stage: "refinement",
                            code: group_scientific_code(&error)
                                .expect("guarded group scientific error"),
                            message: error.to_string(),
                            num_obs: group.source_indices.len(),
                            arc_length: None,
                            triplets_tried: None,
                            candidates_evaluated: None,
                            candidates_failed: None,
                            corrections_run: None,
                            passes: None,
                            n_rejected: None,
                            n_recovered: None,
                            status_code: None,
                            best_state_retained: false,
                        }),
                    }),
                    Err(error) => return Err(error),
                }
            }
            Ok(results)
        },
    )?;

    let items = PyList::empty(py);
    for result in results {
        let refinement_attempted = result.refined.is_some()
            || result
                .failure
                .as_ref()
                .is_some_and(|failure| failure.stage == "refinement");
        items.append(group_dict(
            py,
            &result.group_id,
            &result.orbit_id,
            &result.rows,
            &result.prepared,
            Some(&iod_metadata),
            refinement_attempted.then_some(&refinement_metadata),
            &config_json,
            "run_od",
            result.iod,
            result.refined,
            result.failure,
            None,
            None,
            &source,
            &problem.observers,
        )?)?;
    }
    let out = PyDict::new(py);
    out.set_item("groups", items)?;
    let providers = PyDict::new(py);
    providers.set_item("iod", provider_metadata_dict(py, &iod_metadata)?)?;
    providers.set_item(
        "refinement",
        provider_metadata_dict(py, &refinement_metadata)?,
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
    let solution = evaluation
        .outlier
        .iter()
        .map(|&value| !value)
        .collect::<Vec<_>>();
    set_evaluation_fields(py, &out, evaluation, n)?;
    out.set_item("success", true)?;
    out.set_item("status_code", 1_i64)?;
    out.set_item("solution", solution)?;
    Ok(out)
}

fn orbit_group_id<'a>(orbits: &'a DataOrbitBatch, row: usize, fallback: &'a str) -> &'a str {
    orbits.object_id[row]
        .as_ref()
        .map(|value| value.0.as_str())
        .unwrap_or(fallback)
}

fn evaluation_group_dict<'py>(
    py: Python<'py>,
    result: BatchEvaluationGroup,
    provider: &ProviderMetadata,
    config_json: &str,
    source: &SourceCatalogBatch,
    observers: &DataObserverBatch,
    orbits: &DataOrbitBatch,
) -> PyResult<Bound<'py, PyDict>> {
    let group_id = orbit_group_id(orbits, result.orbit_index, &result.orbit_id);
    let has_evaluation = result.evaluation.is_some();
    let out = base_group_dict(
        py,
        group_id,
        &result.orbit_id,
        &result.rows,
        &result.prepared,
        None,
        None,
        Some(provider),
        config_json,
        "evaluate_orbits",
        None,
        None,
        result.evaluation.as_ref(),
        None,
        Some((orbits, result.orbit_index)),
        Some(&result.member_outlier),
        source,
        observers,
    )?;
    out.set_item("orbit_index", result.orbit_index)?;
    out.set_item("member_indices", &result.member_rows)?;
    out.set_item(
        "output_kind",
        if has_evaluation { "evaluation" } else { "none" },
    )?;
    out.set_item(
        "evaluation",
        result
            .evaluation
            .map(|value| evaluation_output_dict(py, value, result.rows.len()))
            .transpose()?,
    )?;
    out.set_item(
        "failure",
        result
            .failure
            .as_ref()
            .map(|failure| {
                failure_dict(
                    py,
                    group_id,
                    has_evaluation.then_some(result.orbit_id.as_str()),
                    Some(&result.orbit_id),
                    failure,
                )
            })
            .transpose()?,
    )?;
    Ok(out)
}

fn validate_member_snapshot_lengths(
    member_count: usize,
    columns: &[(&str, usize)],
) -> PyResult<()> {
    for &(name, len) in columns {
        if len != member_count {
            return Err(value_error(format!(
                "FittedOrbitMembers {name} length {len} does not match {member_count} rows"
            )));
        }
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (
    orbit_ipc, source_ipc, observed_ipc, observers_ipc,
    member_orbit_ids, member_obs_ids, seed_iod_provenance,
    member_input_fingerprints, member_membership_fingerprints, original_lon, original_lat, original_sigma_lon, original_sigma_lat,
    original_variance_lon, original_variance_lat, original_cov_lonlat, used_lon, used_lat, used_sigma_lon, used_sigma_lat,
    used_variance_lon, used_variance_lat, used_cov_lonlat, config
))]
#[allow(clippy::too_many_arguments)]
fn refine_batch_preflight_ipc(
    orbit_ipc: &Bound<'_, PyBytes>,
    source_ipc: &Bound<'_, PyBytes>,
    observed_ipc: &Bound<'_, PyBytes>,
    observers_ipc: &Bound<'_, PyBytes>,
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
    config: PyRef<'_, PyRefinementRunConfig>,
) -> PyResult<usize> {
    config.inner.validate().map_err(value_error)?;
    checked_build_identity()?;
    let orbits = decode_orbit(orbit_ipc)?;
    let source = decode_source_catalog(source_ipc)?;
    validate_member_snapshot_lengths(
        member_orbit_ids.len(),
        &[
            ("obs_id", member_obs_ids.len()),
            ("input_fingerprint", member_input_fingerprints.len()),
            (
                "membership_fingerprint",
                member_membership_fingerprints.len(),
            ),
            ("original_astrometry.lon", original_lon.len()),
            ("original_astrometry.lat", original_lat.len()),
            ("original_astrometry.sigma_lon", original_sigma_lon.len()),
            ("original_astrometry.sigma_lat", original_sigma_lat.len()),
            (
                "original_astrometry.variance_lon",
                original_variance_lon.len(),
            ),
            (
                "original_astrometry.variance_lat",
                original_variance_lat.len(),
            ),
            ("original_astrometry.cov_lonlat", original_cov_lonlat.len()),
            ("used_astrometry.lon", used_lon.len()),
            ("used_astrometry.lat", used_lat.len()),
            ("used_astrometry.sigma_lon", used_sigma_lon.len()),
            ("used_astrometry.sigma_lat", used_sigma_lat.len()),
            ("used_astrometry.variance_lon", used_variance_lon.len()),
            ("used_astrometry.variance_lat", used_variance_lat.len()),
            ("used_astrometry.cov_lonlat", used_cov_lonlat.len()),
        ],
    )?;
    let groups = adam_core_rs_coords::od::plan_orbit_members(
        &orbits,
        &source.id,
        &member_orbit_ids,
        &member_obs_ids,
    )
    .map_err(value_error)?;
    if groups.is_empty() {
        return Ok(0);
    }
    let problem = decode_observations(observed_ipc, observers_ipc)?;
    validate_catalog_problem(&source, &problem)?;
    let seed_iod_provenance = parse_orbit_provenance(
        &seed_iod_provenance,
        &orbits,
        Some(&config.inner.canonical_json()),
    )?;
    prepare_member_groups(
        &groups,
        &source,
        &problem,
        &orbits,
        Some(&seed_iod_provenance),
        &member_input_fingerprints,
        &member_membership_fingerprints,
        &original_lon,
        &original_lat,
        &original_sigma_lon,
        &original_sigma_lat,
        &original_variance_lon,
        &original_variance_lat,
        &original_cov_lonlat,
        &used_lon,
        &used_lat,
        &used_sigma_lon,
        &used_sigma_lat,
        &used_variance_lon,
        &used_variance_lat,
        &used_cov_lonlat,
    )?;
    Ok(groups.len())
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
    config.inner.validate().map_err(value_error)?;
    checked_build_identity()?;
    let orbits = decode_orbit(orbit_ipc)?;
    let source = decode_source_catalog(source_ipc)?;
    validate_member_snapshot_lengths(
        member_orbit_ids.len(),
        &[
            ("obs_id", member_obs_ids.len()),
            ("input_fingerprint", member_input_fingerprints.len()),
            (
                "membership_fingerprint",
                member_membership_fingerprints.len(),
            ),
            ("original_astrometry.lon", original_lon.len()),
            ("original_astrometry.lat", original_lat.len()),
            ("original_astrometry.sigma_lon", original_sigma_lon.len()),
            ("original_astrometry.sigma_lat", original_sigma_lat.len()),
            (
                "original_astrometry.variance_lon",
                original_variance_lon.len(),
            ),
            (
                "original_astrometry.variance_lat",
                original_variance_lat.len(),
            ),
            ("original_astrometry.cov_lonlat", original_cov_lonlat.len()),
            ("used_astrometry.lon", used_lon.len()),
            ("used_astrometry.lat", used_lat.len()),
            ("used_astrometry.sigma_lon", used_sigma_lon.len()),
            ("used_astrometry.sigma_lat", used_sigma_lat.len()),
            ("used_astrometry.variance_lon", used_variance_lon.len()),
            ("used_astrometry.variance_lat", used_variance_lat.len()),
            ("used_astrometry.cov_lonlat", used_cov_lonlat.len()),
        ],
    )?;
    let groups = adam_core_rs_coords::od::plan_orbit_members(
        &orbits,
        &source.id,
        &member_orbit_ids,
        &member_obs_ids,
    )
    .map_err(value_error)?;
    if groups.is_empty() {
        let out = PyDict::new(py);
        out.set_item("groups", PyList::empty(py))?;
        out.set_item("provider", py.None())?;
        return Ok(out);
    }
    let problem = decode_observations(observed_ipc, observers_ipc)?;
    validate_catalog_problem(&source, &problem)?;
    let refinement = *config.inner.refinement();
    let config_json = config.inner.canonical_json();
    let seed_iod_provenance =
        parse_orbit_provenance(&seed_iod_provenance, &orbits, Some(&config_json))?;
    drop(config);
    let prepared_groups = prepare_member_groups(
        &groups,
        &source,
        &problem,
        &orbits,
        Some(&seed_iod_provenance),
        &member_input_fingerprints,
        &member_membership_fingerprints,
        &original_lon,
        &original_lat,
        &original_sigma_lon,
        &original_sigma_lat,
        &original_variance_lon,
        &original_variance_lat,
        &original_cov_lonlat,
        &used_lon,
        &used_lat,
        &used_sigma_lon,
        &used_sigma_lat,
        &used_variance_lon,
        &used_variance_lat,
        &used_cov_lonlat,
    )?;
    let fit = match refinement {
        RefinementConfig::WorstResidual(value) => value.fit,
        RefinementConfig::Cmc2003(value) => value.fit,
    };
    let settings = fit_settings_from_config(fit)?;
    let (results, metadata) = run_with_predictor(py, propagator, &settings, |predictor| {
        let mut results = Vec::with_capacity(groups.len());
        for (group, (prepared, observers)) in groups.iter().zip(&prepared_groups) {
            let seed = orbits.take(&[group.orbit_index])?;
            match refine_with(
                predictor,
                &seed,
                &prepared.observations,
                observers,
                &refinement,
                &ErfaTimeProvider,
                &LockingSpiceTranslation,
            ) {
                Ok(refined) => {
                    let failure = (!refined.fit.converged).then(|| BatchFailure {
                        stage: "refinement",
                        code: "not_converged",
                        message: format!(
                            "refinement did not converge (status_code={})",
                            refined.fit.status_code
                        ),
                        num_obs: refined.fit.evaluation.num_obs,
                        arc_length: Some(refined.fit.evaluation.arc_length),
                        triplets_tried: None,
                        candidates_evaluated: None,
                        candidates_failed: None,
                        corrections_run: None,
                        passes: Some(refined.passes),
                        n_rejected: Some(refined.n_rejected),
                        n_recovered: Some(refined.n_recovered),
                        status_code: Some(refined.fit.status_code),
                        best_state_retained: true,
                    });
                    results.push(BatchRefineGroup {
                        orbit_id: group.orbit_id.clone(),
                        orbit_index: group.orbit_index,
                        rows: group.source_indices.clone(),
                        member_rows: group.member_indices.clone(),
                        prepared: prepared.clone(),
                        refined: Some(refined),
                        failure,
                    });
                }
                Err(error) if local_scientific_error(&error) => {
                    results.push(BatchRefineGroup {
                        orbit_id: group.orbit_id.clone(),
                        orbit_index: group.orbit_index,
                        rows: group.source_indices.clone(),
                        member_rows: group.member_indices.clone(),
                        prepared: prepared.clone(),
                        refined: None,
                        failure: Some(BatchFailure {
                            stage: "refinement",
                            code: group_scientific_code(&error)
                                .expect("guarded group scientific error"),
                            message: error.to_string(),
                            num_obs: group.source_indices.len(),
                            arc_length: None,
                            triplets_tried: None,
                            candidates_evaluated: None,
                            candidates_failed: None,
                            corrections_run: None,
                            passes: None,
                            n_rejected: None,
                            n_recovered: None,
                            status_code: None,
                            best_state_retained: false,
                        }),
                    });
                }
                Err(error) => return Err(error),
            }
        }
        Ok(results)
    })?;
    let items = PyList::empty(py);
    for result in results {
        let group_id = orbit_group_id(&orbits, result.orbit_index, &result.orbit_id);
        let group = group_dict(
            py,
            group_id,
            &result.orbit_id,
            &result.rows,
            &result.prepared,
            None,
            Some(&metadata),
            &config_json,
            "refine",
            None,
            result.refined,
            result.failure,
            seed_iod_provenance[result.orbit_index].as_ref(),
            Some((&orbits, result.orbit_index)),
            &source,
            &problem.observers,
        )?;
        group.set_item("orbit_index", result.orbit_index)?;
        group.set_item("member_indices", &result.member_rows)?;
        items.append(group)?;
    }
    let out = PyDict::new(py);
    out.set_item("groups", items)?;
    out.set_item("provider", provider_metadata_dict(py, &metadata)?)?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (
    orbit_ipc, source_ipc, observed_ipc, observers_ipc,
    member_orbit_ids, member_obs_ids, member_outlier,
    orbit_provenance, member_input_fingerprints, member_membership_fingerprints, original_lon, original_lat, original_sigma_lon, original_sigma_lat,
    original_variance_lon, original_variance_lat, original_cov_lonlat, used_lon, used_lat, used_sigma_lon, used_sigma_lat,
    used_variance_lon, used_variance_lat, used_cov_lonlat, config
))]
#[allow(clippy::too_many_arguments)]
fn evaluate_orbits_batch_preflight_ipc(
    orbit_ipc: &Bound<'_, PyBytes>,
    source_ipc: &Bound<'_, PyBytes>,
    observed_ipc: &Bound<'_, PyBytes>,
    observers_ipc: &Bound<'_, PyBytes>,
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
    config: PyRef<'_, PyOdConfig>,
) -> PyResult<usize> {
    config.inner.validate().map_err(value_error)?;
    checked_build_identity()?;
    let orbits = decode_orbit(orbit_ipc)?;
    let source = decode_source_catalog(source_ipc)?;
    validate_member_snapshot_lengths(
        member_orbit_ids.len(),
        &[
            ("obs_id", member_obs_ids.len()),
            ("outlier", member_outlier.len()),
            ("input_fingerprint", member_input_fingerprints.len()),
            (
                "membership_fingerprint",
                member_membership_fingerprints.len(),
            ),
            ("original_astrometry.lon", original_lon.len()),
            ("original_astrometry.lat", original_lat.len()),
            ("original_astrometry.sigma_lon", original_sigma_lon.len()),
            ("original_astrometry.sigma_lat", original_sigma_lat.len()),
            (
                "original_astrometry.variance_lon",
                original_variance_lon.len(),
            ),
            (
                "original_astrometry.variance_lat",
                original_variance_lat.len(),
            ),
            ("original_astrometry.cov_lonlat", original_cov_lonlat.len()),
            ("used_astrometry.lon", used_lon.len()),
            ("used_astrometry.lat", used_lat.len()),
            ("used_astrometry.sigma_lon", used_sigma_lon.len()),
            ("used_astrometry.sigma_lat", used_sigma_lat.len()),
            ("used_astrometry.variance_lon", used_variance_lon.len()),
            ("used_astrometry.variance_lat", used_variance_lat.len()),
            ("used_astrometry.cov_lonlat", used_cov_lonlat.len()),
        ],
    )?;
    let groups = adam_core_rs_coords::od::plan_orbit_members(
        &orbits,
        &source.id,
        &member_orbit_ids,
        &member_obs_ids,
    )
    .map_err(value_error)?;
    if groups.is_empty() {
        return Ok(0);
    }
    let problem = decode_observations(observed_ipc, observers_ipc)?;
    validate_catalog_problem(&source, &problem)?;
    if let Some(row) = member_outlier.iter().position(Option::is_none) {
        return Err(value_error(format!(
            "FittedOrbitMembers outlier is null at row {row}"
        )));
    }
    parse_orbit_provenance(&orbit_provenance, &orbits, None)?;
    prepare_member_groups(
        &groups,
        &source,
        &problem,
        &orbits,
        None,
        &member_input_fingerprints,
        &member_membership_fingerprints,
        &original_lon,
        &original_lat,
        &original_sigma_lon,
        &original_sigma_lat,
        &original_variance_lon,
        &original_variance_lat,
        &original_cov_lonlat,
        &used_lon,
        &used_lat,
        &used_sigma_lon,
        &used_sigma_lat,
        &used_variance_lon,
        &used_variance_lat,
        &used_cov_lonlat,
    )?;
    Ok(groups.len())
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
    config.inner.validate().map_err(value_error)?;
    checked_build_identity()?;
    let orbits = decode_orbit(orbit_ipc)?;
    let source = decode_source_catalog(source_ipc)?;
    validate_member_snapshot_lengths(
        member_orbit_ids.len(),
        &[
            ("obs_id", member_obs_ids.len()),
            ("outlier", member_outlier.len()),
            ("input_fingerprint", member_input_fingerprints.len()),
            (
                "membership_fingerprint",
                member_membership_fingerprints.len(),
            ),
            ("original_astrometry.lon", original_lon.len()),
            ("original_astrometry.lat", original_lat.len()),
            ("original_astrometry.sigma_lon", original_sigma_lon.len()),
            ("original_astrometry.sigma_lat", original_sigma_lat.len()),
            (
                "original_astrometry.variance_lon",
                original_variance_lon.len(),
            ),
            (
                "original_astrometry.variance_lat",
                original_variance_lat.len(),
            ),
            ("original_astrometry.cov_lonlat", original_cov_lonlat.len()),
            ("used_astrometry.lon", used_lon.len()),
            ("used_astrometry.lat", used_lat.len()),
            ("used_astrometry.sigma_lon", used_sigma_lon.len()),
            ("used_astrometry.sigma_lat", used_sigma_lat.len()),
            ("used_astrometry.variance_lon", used_variance_lon.len()),
            ("used_astrometry.variance_lat", used_variance_lat.len()),
            ("used_astrometry.cov_lonlat", used_cov_lonlat.len()),
        ],
    )?;
    let groups = adam_core_rs_coords::od::plan_orbit_members(
        &orbits,
        &source.id,
        &member_orbit_ids,
        &member_obs_ids,
    )
    .map_err(value_error)?;
    if groups.is_empty() {
        let out = PyDict::new(py);
        out.set_item("groups", PyList::empty(py))?;
        out.set_item("provider", py.None())?;
        return Ok(out);
    }
    let problem = decode_observations(observed_ipc, observers_ipc)?;
    validate_catalog_problem(&source, &problem)?;
    if let Some(row) = member_outlier.iter().position(Option::is_none) {
        return Err(value_error(format!(
            "FittedOrbitMembers outlier is null at row {row}"
        )));
    }
    parse_orbit_provenance(&orbit_provenance, &orbits, None)?;
    let parameters = config.inner.for_evaluation().covariance_dimension() as i64;
    let config_json = config.inner.provenance_json();
    let refinement = *config.inner.for_refinement().refinement();
    let evaluation_fit = match refinement {
        RefinementConfig::WorstResidual(value) => value.fit,
        RefinementConfig::Cmc2003(value) => value.fit,
    };
    drop(config);
    let prepared_groups = prepare_member_groups(
        &groups,
        &source,
        &problem,
        &orbits,
        None,
        &member_input_fingerprints,
        &member_membership_fingerprints,
        &original_lon,
        &original_lat,
        &original_sigma_lon,
        &original_sigma_lat,
        &original_variance_lon,
        &original_variance_lat,
        &original_cov_lonlat,
        &used_lon,
        &used_lat,
        &used_sigma_lon,
        &used_sigma_lat,
        &used_variance_lon,
        &used_variance_lat,
        &used_cov_lonlat,
    )?;
    let settings = fit_settings_from_config(evaluation_fit)?;
    let (results, metadata) = run_with_predictor(py, propagator, &settings, |predictor| {
        let mut results = Vec::with_capacity(groups.len());
        for (group, (prepared, observers)) in groups.iter().zip(&prepared_groups) {
            let orbit = orbits.take(&[group.orbit_index])?;
            let state = orbit
                .coordinates
                .values
                .cartesian()
                .expect("validated orbit coordinates are Cartesian")[0];
            let ignore = group
                .member_indices
                .iter()
                .map(|&row| member_outlier[row].unwrap_or(false))
                .collect::<Vec<_>>();
            match evaluate_orbit(
                predictor,
                state,
                &orbit,
                &prepared.observations,
                observers,
                &ignore,
                parameters,
            ) {
                Ok(evaluation) => results.push(BatchEvaluationGroup {
                    orbit_id: group.orbit_id.clone(),
                    orbit_index: group.orbit_index,
                    rows: group.source_indices.clone(),
                    member_rows: group.member_indices.clone(),
                    member_outlier: ignore,
                    prepared: prepared.clone(),
                    evaluation: Some(evaluation),
                    failure: None,
                }),
                Err(error) if local_scientific_error(&error) => {
                    results.push(BatchEvaluationGroup {
                        orbit_id: group.orbit_id.clone(),
                        orbit_index: group.orbit_index,
                        rows: group.source_indices.clone(),
                        member_rows: group.member_indices.clone(),
                        member_outlier: ignore,
                        prepared: prepared.clone(),
                        evaluation: None,
                        failure: Some(BatchFailure {
                            stage: "evaluation",
                            code: group_scientific_code(&error)
                                .expect("guarded group scientific error"),
                            message: error.to_string(),
                            num_obs: group.source_indices.len(),
                            arc_length: None,
                            triplets_tried: None,
                            candidates_evaluated: None,
                            candidates_failed: None,
                            corrections_run: None,
                            passes: None,
                            n_rejected: None,
                            n_recovered: None,
                            status_code: None,
                            best_state_retained: false,
                        }),
                    });
                }
                Err(error) => return Err(error),
            }
        }
        Ok(results)
    })?;
    let items = PyList::empty(py);
    for result in results {
        items.append(evaluation_group_dict(
            py,
            result,
            &metadata,
            &config_json,
            &source,
            &problem.observers,
            &orbits,
        )?)?;
    }
    let out = PyDict::new(py);
    out.set_item("groups", items)?;
    out.set_item("provider", provider_metadata_dict(py, &metadata)?)?;
    Ok(out)
}

#[cfg(test)]
mod batch_contract_tests {
    use super::*;
    use adam_core_rs_coords::propagation::GroupScientificFailureCode;

    #[test]
    fn only_typed_group_scientific_errors_are_local() {
        let local = adam_core_rs_coords::PropagationError::GroupScientific {
            code: GroupScientificFailureCode::RankDeficient,
            message: "test".to_string(),
        };
        assert!(local_scientific_error(&local));
        assert_eq!(group_scientific_code(&local), Some("rank_deficient"));
        assert!(!local_scientific_error(
            &adam_core_rs_coords::PropagationError::RowFailure("coverage".to_string())
        ));
        assert!(!local_scientific_error(
            &adam_core_rs_coords::PropagationError::Backend("global".to_string())
        ));
    }
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
    m.add_function(wrap_pyfunction!(iod_od_batch_preflight_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(run_iod_batch_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(run_od_batch_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(refine_batch_preflight_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(refine_batch_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_orbits_batch_preflight_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_orbits_batch_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(native_provider_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(od_defaults, m)?)?;
    m.add_function(wrap_pyfunction!(apply_observation_pipeline_ipc, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
