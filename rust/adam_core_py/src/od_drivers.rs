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
    AstrometrySnapshot, BackendIdentity, Cmc2003FitConfig, CovariancePropagation, EphemerisOptions,
    EpochPolicy, ExternalPropagator, ExternalPropagatorV2, IodCandidate, IodCandidateFailure,
    IodResult, IterativeFitConfig, JacobianMethod, PropagationOptions, PropagationResultValue,
    Propagator, RangingResult, RangingSample, RefinementConfig, TwoBodyPropagator,
    TwoBodyPropagatorConfig, WhitenedFitConfig, WhitenedFitOutput, EXTERNAL_PROPAGATOR_CAPSULE,
};
use adam_core_rs_coords::{
    validate_loss, CoordinateBatch as DataCoordinateBatch, IntoNestedRecordBatch,
    ObserverBatch as DataObserverBatch, OrbitBatch as DataOrbitBatch, TimeScale,
    TryFromNestedRecordBatch, TwoBodyModelConfig, CMC2003_APPARITION_GAP_DAYS, CMC2003_CHI2_FRAC,
    CMC2003_CHI2_RECOVER, CMC2003_CHI2_REJECT, CMC2003_MAX_ITERATIONS,
    CMC2003_MAX_REJECTED_FRACTION, CMC2003_PSD_FLOOR_FRAC,
};
use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyCapsule, PyDict, PyList, PyTuple};

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
        let (capsule_object, expected_identity) =
            if let Ok(descriptor) = object.downcast::<PyTuple>() {
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
                (object.clone(), None)
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
            _capsule: object.clone().unbind(),
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
// ---------------------------------------------------------------------------
// Output dicts
// ---------------------------------------------------------------------------

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
    let residuals = ndarray::Array2::from_shape_vec((n, 6), output.evaluation.residuals)
        .map_err(|err| value_error(format!("failed to shape residuals: {err}")))?;
    out.set_item("residual_values", residuals.into_pyarray(py))?;
    out.set_item("residual_chi2", output.evaluation.chi2)?;
    out.set_item("residual_dof", output.evaluation.dof)?;
    out.set_item("residual_probability", output.evaluation.probability)?;
    out.set_item("chi2", output.evaluation.orbit_chi2)?;
    out.set_item("reduced_chi2", output.evaluation.reduced_chi2)?;
    out.set_item("arc_length", output.evaluation.arc_length)?;
    out.set_item("num_obs", output.evaluation.num_obs)?;
    out.set_item("outlier", output.evaluation.outlier)?;
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
