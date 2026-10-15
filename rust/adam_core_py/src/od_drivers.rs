//! PyO3 surface of the backend-generic Rust OD drivers: the one-crossing
//! work units (`fit_least_squares`, `iterative_fit`, `cmc2003_fit`, `iod`,
//! `full_od`, `run_od`, the covariance probe) over either
//!
//! * a Python propagator, wrapped as a callback [`SphericalPredictor`]
//!   (`od_callback`): the Rust loop drives the propagator's own
//!   `generate_ephemeris`, so the Python facade needs no loop of its own; or
//! * adam_core's Rust two-body propagator (``propagator=None``): the fused
//!   in-tree route, the same contract a Rust-backed plugin (adam-assist)
//!   exposes as methods.
//!
//! Inputs cross as nested Arrow IPC (the `arrow_bridge` helpers) plus dicts of
//! settings (`fit_settings`, `iod_settings`, `refinement`) and the models'
//! `_native_specs()` dicts; outputs are plain dicts of scalars, lists and
//! NumPy arrays that the facade wraps into `FittedOrbits` /
//! `FittedOrbitMembers` without further computation. Driver errors map to
//! `ValueError` (invalid input) or `RuntimeError` (backend failure); an
//! exception raised by a Python propagator is re-raised unchanged.

use crate::coordinates::{read_orbit_ipc, ErfaTimeProvider};
use crate::observation_uncertainty::{bias_table, efcc18_table, veres_lookup};
use crate::od_callback::{od_error, LockingSpiceTranslation, PyEphemerisPredictor};
use adam_core_rs_coords::observation_uncertainty::{
    Efcc18DebiasModel, EmpiricalCovarianceMode, EmpiricalCovarianceModel, IdentityModel,
    NightBatchDeweightingModel, ObservationUncertaintyModel, PerformanceWeightedModel,
    SigmaFloorModel,
};
use adam_core_rs_coords::propagation::{
    cmc2003_fit_with, fit_orbit_whitened_with, full_od_with, iterative_fit_with, run_iod_with,
    run_od_with, validate_fit_covariance_with, AstrometrySnapshot, Cmc2003FitConfig,
    CorrectionConfig, CovariancePropagation, EphemerisOptions, EpochPolicy, FullOdConfig,
    FullOdOutput, IodAcceptance, IodCandidate, IodConfig, IodResult, IterativeFitConfig,
    JacobianMethod, ObservationSelectionMethod, PrefitScope, PropagationOptions,
    PropagationResultValue, PropagatorPredictor, RefinementConfig, SpanSearchConfig,
    SphericalPredictor, TwoBodyPropagator, TwoBodyPropagatorConfig, VelocityMethod,
    WhitenedFitConfig, WhitenedFitOutput,
};
use adam_core_rs_coords::veres2017::{SigmaFillModel, VeresFloorModel, VeresReplaceModel};
use adam_core_rs_coords::{
    validate_loss, CoordinateBatch as DataCoordinateBatch, LossType,
    ObserverBatch as DataObserverBatch, OrbitBatch as DataOrbitBatch, TimeScale,
    TryFromNestedRecordBatch, TwoBodyModelConfig, CMC2003_APPARITION_GAP_DAYS, CMC2003_CHI2_FRAC,
    CMC2003_CHI2_RECOVER, CMC2003_CHI2_REJECT, CMC2003_MAX_ITERATIONS,
    CMC2003_MAX_REJECTED_FRACTION, CMC2003_PSD_FLOOR_FRAC, HUBER_F_SCALE_DEFAULT,
};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::sync::Arc;

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

/// `IodConfig` from the ``iod_settings`` dict over `IodConfig::default()`.
fn parse_iod_settings(settings: Option<&Bound<'_, PyDict>>) -> PyResult<IodConfig> {
    parse_iod_settings_over(settings, IodConfig::default())
}

/// The pre-a6w.4 IOD defaults (combination triplets, prefit acceptance) of
/// the deprecated `iod_fit_ipc`.
fn legacy_iod_defaults() -> IodConfig {
    IodConfig {
        observation_selection_method: ObservationSelectionMethod::Combinations,
        acceptance: IodAcceptance::Prefit,
        ..IodConfig::default()
    }
}

/// `IodConfig::default()` (with `HUBER_F_SCALE_DEFAULT` as
/// ``huber_f_scale``) under the keys of the ``iod_settings``,
/// ``span_settings`` and ``correction_settings`` dicts: the one source of the
/// Python signature defaults of `run_iod` and `NativeOrbitFitter`.
#[pyfunction]
pub fn iod_defaults(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let config = IodConfig::default();
    let out = PyDict::new(py);
    out.set_item("min_obs", config.min_obs)?;
    out.set_item("min_arc_length", config.min_arc_length)?;
    out.set_item("contamination_percentage", config.contamination_percentage)?;
    out.set_item("rchi2_threshold", config.rchi2_threshold)?;
    out.set_item(
        "observation_selection_method",
        config.observation_selection_method.as_str(),
    )?;
    out.set_item("light_time", config.light_time)?;
    out.set_item("max_triplets", config.max_triplets)?;
    out.set_item("velocity_method", config.velocity_method.as_str())?;
    out.set_item("acceptance", config.acceptance.as_str())?;
    out.set_item("span_target_days", config.spans.target_span_days)?;
    out.set_item("span_min_days", config.spans.min_span_days)?;
    out.set_item("span_max_days", config.spans.max_span_days)?;
    out.set_item(
        "span_dense_window_days",
        config.spans.dense_window_days.to_vec(),
    )?;
    out.set_item("span_edge_window_days", config.spans.edge_window_days)?;
    out.set_item("span_max_triplets", config.spans.max_triplets)?;
    out.set_item("max_corrections", config.correction.max_corrections)?;
    out.set_item("max_converged", config.correction.max_converged)?;
    out.set_item("correction_growth", config.correction.growth)?;
    out.set_item("min_stage_obs", config.correction.min_stage_obs)?;
    out.set_item("max_stages", config.correction.max_stages)?;
    out.set_item("stage_loss", config.correction.stage_loss.as_str())?;
    out.set_item("prefit_scope", config.correction.prefit_scope.as_str())?;
    out.set_item("huber_f_scale", HUBER_F_SCALE_DEFAULT)?;
    Ok(out)
}

/// `IodConfig` from the ``iod_settings`` dict over `defaults`.
fn parse_iod_settings_over(
    settings: Option<&Bound<'_, PyDict>>,
    defaults: IodConfig,
) -> PyResult<IodConfig> {
    let observation_selection_method =
        match setting::<Option<String>>(settings, "observation_selection_method", None)? {
            Some(method) => ObservationSelectionMethod::parse(&method).map_err(value_error)?,
            None => defaults.observation_selection_method,
        };
    Ok(IodConfig {
        min_obs: setting(settings, "min_obs", defaults.min_obs)?,
        min_arc_length: setting(settings, "min_arc_length", defaults.min_arc_length)?,
        contamination_percentage: setting(
            settings,
            "contamination_percentage",
            defaults.contamination_percentage,
        )?,
        rchi2_threshold: setting(settings, "rchi2_threshold", defaults.rchi2_threshold)?,
        observation_selection_method,
        light_time: setting(settings, "light_time", defaults.light_time)?,
        mu: setting(settings, "mu", defaults.mu)?,
        speed_of_light: setting(settings, "speed_of_light", defaults.speed_of_light)?,
        max_triplets: setting(settings, "max_triplets", defaults.max_triplets)?,
        velocity_method: match setting::<Option<String>>(settings, "velocity_method", None)? {
            Some(method) => VelocityMethod::parse(&method).map_err(value_error)?,
            None => defaults.velocity_method,
        },
        spans: SpanSearchConfig {
            target_span_days: setting(
                settings,
                "span_target_days",
                defaults.spans.target_span_days,
            )?,
            min_span_days: setting(settings, "span_min_days", defaults.spans.min_span_days)?,
            max_span_days: setting(settings, "span_max_days", defaults.spans.max_span_days)?,
            dense_window_days: setting(
                settings,
                "span_dense_window_days",
                defaults.spans.dense_window_days,
            )?,
            edge_window_days: setting(
                settings,
                "span_edge_window_days",
                defaults.spans.edge_window_days,
            )?,
            max_triplets: setting(settings, "span_max_triplets", defaults.spans.max_triplets)?,
        },
        acceptance: match setting::<Option<String>>(settings, "acceptance", None)? {
            Some(value) => IodAcceptance::parse(&value).map_err(value_error)?,
            None => defaults.acceptance,
        },
        correction: CorrectionConfig {
            max_corrections: setting(
                settings,
                "max_corrections",
                defaults.correction.max_corrections,
            )?,
            max_converged: setting(settings, "max_converged", defaults.correction.max_converged)?,
            growth: setting(settings, "correction_growth", defaults.correction.growth)?,
            min_stage_obs: setting(settings, "min_stage_obs", defaults.correction.min_stage_obs)?,
            max_stages: setting(settings, "max_stages", defaults.correction.max_stages)?,
            stage_loss: match setting::<Option<String>>(settings, "stage_loss", None)? {
                Some(value) => LossType::parse(&value).map_err(value_error)?,
                None => defaults.correction.stage_loss,
            },
            prefit_scope: match setting::<Option<String>>(settings, "prefit_scope", None)? {
                Some(value) => PrefitScope::parse(&value).map_err(value_error)?,
                None => defaults.correction.prefit_scope,
            },
        },
    })
}

/// `RefinementConfig` from the ``refinement`` dict: ``method`` is
/// ``"cmc2003"`` (the `RefinementConfig::default()` loop) or
/// ``"worst_residual"``, each over that loop's `Default` settings.
fn parse_refinement(
    settings: Option<&Bound<'_, PyDict>>,
    fit: WhitenedFitConfig,
) -> PyResult<RefinementConfig> {
    let method: String = setting(settings, "method", "cmc2003".to_string())?;
    match method.as_str() {
        "cmc2003" => {
            let defaults = Cmc2003FitConfig::default();
            Ok(RefinementConfig::Cmc2003(Cmc2003FitConfig {
                chi2_reject: setting(settings, "chi2_reject", defaults.chi2_reject)?,
                chi2_recover: setting(settings, "chi2_recover", defaults.chi2_recover)?,
                chi2_frac: setting(settings, "chi2_frac", defaults.chi2_frac)?,
                max_iterations: setting(settings, "max_iterations", defaults.max_iterations)?,
                max_rejected_fraction: setting(
                    settings,
                    "max_rejected_fraction",
                    defaults.max_rejected_fraction,
                )?,
                apparition_gap_days: setting(
                    settings,
                    "apparition_gap_days",
                    defaults.apparition_gap_days,
                )?,
                psd_floor_frac: setting(settings, "psd_floor_frac", defaults.psd_floor_frac)?,
                fit,
            }))
        }
        "worst_residual" => {
            let defaults = IterativeFitConfig::default();
            Ok(RefinementConfig::WorstResidual(IterativeFitConfig {
                rchi2_threshold: setting(settings, "rchi2_threshold", defaults.rchi2_threshold)?,
                min_obs: setting(settings, "min_obs", defaults.min_obs)?,
                min_arc_length: setting(settings, "min_arc_length", defaults.min_arc_length)?,
                contamination_percentage: setting(
                    settings,
                    "contamination_percentage",
                    defaults.contamination_percentage,
                )?,
                fit,
            }))
        }
        other => Err(value_error(format!(
            "refinement method must be 'cmc2003' or 'worst_residual'; got {other:?}"
        ))),
    }
}

fn required<'py, T: FromPyObject<'py>>(spec: &Bound<'py, PyDict>, key: &str) -> PyResult<T> {
    spec.get_item(key)?
        .ok_or_else(|| value_error(format!("model spec is missing {key:?}")))?
        .extract::<T>()
}

/// One observation model from its ``_native_spec()`` dict.
fn model_from_spec(spec: &Bound<'_, PyDict>) -> PyResult<Box<dyn ObservationUncertaintyModel>> {
    let model: String = required(spec, "model")?;
    match model.as_str() {
        "identity" => Ok(Box::new(IdentityModel)),
        "empirical_covariance" | "performance_weighted" | "sigma_floor" => {
            let table = bias_table(
                required(spec, "table_obs_code")?,
                required(spec, "table_band")?,
                required::<PyReadonlyArray1<'_, f64>>(spec, "bias_ra_arcsec")?,
                required::<PyReadonlyArray1<'_, f64>>(spec, "bias_dec_arcsec")?,
                required::<PyReadonlyArray1<'_, f64>>(spec, "resid_var_ra")?,
                required::<PyReadonlyArray1<'_, f64>>(spec, "resid_var_dec")?,
                required::<PyReadonlyArray1<'_, f64>>(spec, "resid_cov_ra_dec")?,
                required::<PyReadonlyArray1<'_, f64>>(spec, "resid_cov_n")?,
                required::<PyReadonlyArray1<'_, f64>>(spec, "chi2_per_obs")?,
            )?;
            let min_resid_cov_n: f64 = required(spec, "min_resid_cov_n")?;
            Ok(match model.as_str() {
                "empirical_covariance" => {
                    let mode: String = required(spec, "mode")?;
                    Box::new(EmpiricalCovarianceModel {
                        table,
                        mode: EmpiricalCovarianceMode::parse(&mode).map_err(value_error)?,
                        min_resid_cov_n,
                    })
                }
                "performance_weighted" => Box::new(PerformanceWeightedModel {
                    table,
                    min_resid_cov_n,
                }),
                _ => Box::new(SigmaFloorModel {
                    table,
                    min_resid_cov_n,
                }),
            })
        }
        "night_batch" => Ok(Box::new(
            NightBatchDeweightingModel::new(required(spec, "cap")?).map_err(value_error)?,
        )),
        "efcc18" => Ok(Box::new(Efcc18DebiasModel {
            bias_table: Arc::new(efcc18_table(&required::<PyReadonlyArray3<'_, f32>>(
                spec,
                "bias_table",
            )?)?),
            exclude_astcats: required(spec, "exclude_astcats")?,
        })),
        "veres" => {
            let lookup = veres_lookup(
                required(spec, "table_obs_code")?,
                required(spec, "table_astcat")?,
                &required::<PyReadonlyArray1<'_, f64>>(spec, "sigma_ra_arcsec")?,
                &required::<PyReadonlyArray1<'_, f64>>(spec, "sigma_dec_arcsec")?,
                required(spec, "fallback_sigma_arcsec")?,
            )?;
            let kind: String = required(spec, "kind")?;
            Ok(match kind.as_str() {
                "floor" => Box::new(VeresFloorModel {
                    lookup,
                    fill_missing: required(spec, "fill_missing")?,
                }),
                "replace" => Box::new(VeresReplaceModel { lookup }),
                "fill" => Box::new(SigmaFillModel { lookup }),
                other => {
                    return Err(value_error(format!(
                        "veres model kind must be 'floor', 'replace' or 'fill'; got {other:?}"
                    )))
                }
            })
        }
        other => Err(value_error(format!(
            "unknown observation model spec {other:?}"
        ))),
    }
}

fn models_from_specs(
    specs: Option<&Bound<'_, PyList>>,
) -> PyResult<Vec<Box<dyn ObservationUncertaintyModel>>> {
    match specs {
        None => Ok(Vec::new()),
        Some(list) => list
            .iter()
            .map(|item| model_from_spec(item.downcast::<PyDict>()?))
            .collect(),
    }
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

/// Run `driver` with the GIL released, over a Python propagator wrapped as a
/// callback predictor (`propagator` given) or adam_core's two-body
/// propagator (`None`). Errors map to Python exceptions; an exception raised
/// inside the callback is re-raised unchanged.
fn run_with_predictor<R: Send>(
    py: Python<'_>,
    propagator: Option<&Bound<'_, PyAny>>,
    settings: &FitSettings,
    driver: impl for<'p> FnOnce(&'p dyn SphericalPredictor) -> PropagationResultValue<R> + Send,
) -> PyResult<R> {
    match propagator {
        Some(object) => {
            let predictor = PyEphemerisPredictor::new(py, object)?;
            py.allow_threads(|| driver(&predictor))
                .map_err(|err| od_error(err, Some(&predictor)))
        }
        None => py
            .allow_threads(|| {
                let predictor = PropagatorPredictor {
                    propagator: &settings.propagator,
                    options: &settings.options,
                    provider: &ErfaTimeProvider,
                    translation_provider: &LockingSpiceTranslation,
                };
                driver(&predictor)
            })
            .map_err(|err| od_error(err, None)),
    }
}

/// [`run_with_predictor`] for a two-stage driver: `iod_propagator` scores the
/// IOD candidates and `propagator` refines (each `None` = adam_core's two-body
/// propagator; `iod_propagator=None` alone = the refinement propagator). A
/// Python exception raised by either propagator is re-raised unchanged.
fn run_with_predictors<R: Send>(
    py: Python<'_>,
    iod_propagator: Option<&Bound<'_, PyAny>>,
    propagator: Option<&Bound<'_, PyAny>>,
    settings: &FitSettings,
    driver: impl for<'p> FnOnce(
            &'p dyn SphericalPredictor,
            &'p dyn SphericalPredictor,
        ) -> PropagationResultValue<R>
        + Send,
) -> PyResult<R> {
    let iod_callback = iod_propagator
        .map(|object| PyEphemerisPredictor::new(py, object))
        .transpose()?;
    let refine_callback = propagator
        .map(|object| PyEphemerisPredictor::new(py, object))
        .transpose()?;
    let result = py.allow_threads(|| {
        let two_body = PropagatorPredictor {
            propagator: &settings.propagator,
            options: &settings.options,
            provider: &ErfaTimeProvider,
            translation_provider: &LockingSpiceTranslation,
        };
        let refine: &dyn SphericalPredictor = match &refine_callback {
            Some(callback) => callback,
            None => &two_body,
        };
        let iod: &dyn SphericalPredictor = match &iod_callback {
            Some(callback) => callback,
            None => refine,
        };
        driver(iod, refine)
    });
    result.map_err(|err| {
        if let Some(py_err) = iod_callback
            .as_ref()
            .and_then(PyEphemerisPredictor::take_error)
        {
            return py_err;
        }
        od_error(err, refine_callback.as_ref())
    })
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
    out.set_item("corrections_run", result.corrections_run)?;
    out.set_item("not_found_reason", iod.not_found_reason)?;
    out.set_item("accepted", result.accepted)?;
    out.set_item("ambiguous_with", result.ambiguous_with.clone())?;
    let candidates = PyList::empty(py);
    for candidate in &result.candidates {
        candidates.append(candidate_dict(py, candidate)?)?;
    }
    out.set_item("candidates", candidates)?;
    Ok(out)
}

/// The dict contract of a fused full OD: ``found``, the ``iod`` decision
/// and, when found, the refined ``fit`` dict with the loop diagnostics.
fn full_od_dict<'py>(
    py: Python<'py>,
    output: FullOdOutput,
    n: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("found", output.iod.found())?;
    out.set_item("iod", iod_dict(py, output.iod, n)?)?;
    match output.refined {
        Some(refined) => {
            let fit = fit_output_dict(py, refined.fit, n)?;
            fit.set_item("passes", refined.passes)?;
            fit.set_item("n_rejected", refined.n_rejected)?;
            fit.set_item("n_recovered", refined.n_recovered)?;
            fit.set_item("flags", refined.flags)?;
            out.set_item("fit", fit)?;
        }
        None => out.set_item("fit", py.None())?,
    }
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
/// fused final evaluation. ``propagator`` is any Python propagator (callback
/// route) or None for adam_core's two-body propagator.
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
    let output = run_with_predictor(py, propagator, &settings, |predictor| {
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
    fit_output_dict(py, output, n)
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
    let output = run_with_predictor(py, propagator, &settings, |predictor| {
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
    let output = run_with_predictor(py, propagator, &settings, |predictor| {
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
        fit: settings.config,
    };
    let n = problem.observed.len();
    let output = run_with_predictor(py, propagator, &settings, |predictor| {
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

/// One-crossing IOD stage (`run_iod`): triplet selection, Gauss candidates,
/// acceptance against ``rchi2_threshold`` with outlier trials, returning the
/// typed candidate set (see `iod_dict`). ``iod_propagator`` scores the
/// candidates (None = adam_core's two-body propagator). ``fit_settings``
/// carries only the ephemeris / two-body options here.
#[pyfunction]
#[pyo3(signature = (iod_propagator, observed_ipc, observers_ipc, iod_settings=None, fit_settings=None))]
fn run_iod_ipc<'py>(
    py: Python<'py>,
    iod_propagator: Option<&Bound<'py, PyAny>>,
    observed_ipc: &Bound<'py, PyBytes>,
    observers_ipc: &Bound<'py, PyBytes>,
    iod_settings: Option<&Bound<'py, PyDict>>,
    fit_settings: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    run_iod_impl(
        py,
        iod_propagator,
        observed_ipc,
        observers_ipc,
        parse_iod_settings(iod_settings)?,
        fit_settings,
    )
}

fn run_iod_impl<'py>(
    py: Python<'py>,
    iod_propagator: Option<&Bound<'py, PyAny>>,
    observed_ipc: &Bound<'py, PyBytes>,
    observers_ipc: &Bound<'py, PyBytes>,
    config: IodConfig,
    fit_settings: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let problem = decode_observations(observed_ipc, observers_ipc)?;
    let settings = parse_fit_settings(fit_settings)?;
    let n = problem.observed.len();
    let result = run_with_predictor(py, iod_propagator, &settings, |predictor| {
        run_iod_with(
            predictor,
            &problem.observed,
            &problem.observers,
            &config,
            &settings.config,
            &ErfaTimeProvider,
            &LockingSpiceTranslation,
        )
    })?;
    iod_dict(py, result, n)
}

/// `run_iod_ipc` over adam_core's two-body propagator.
#[pyfunction]
#[pyo3(signature = (observed_ipc, observers_ipc, iod_settings=None, fit_settings=None))]
fn run_iod_2body_ipc<'py>(
    py: Python<'py>,
    observed_ipc: &Bound<'py, PyBytes>,
    observers_ipc: &Bound<'py, PyBytes>,
    iod_settings: Option<&Bound<'py, PyDict>>,
    fit_settings: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    run_iod_ipc(
        py,
        None,
        observed_ipc,
        observers_ipc,
        iod_settings,
        fit_settings,
    )
}

/// The pre-candidate-set name of `run_iod_ipc`; same dict, but settings
/// default to the legacy search (combination triplets, prefit acceptance)
/// (deprecated).
#[pyfunction]
#[pyo3(signature = (propagator, observed_ipc, observers_ipc, iod_settings=None, fit_settings=None))]
fn iod_fit_ipc<'py>(
    py: Python<'py>,
    propagator: Option<&Bound<'py, PyAny>>,
    observed_ipc: &Bound<'py, PyBytes>,
    observers_ipc: &Bound<'py, PyBytes>,
    iod_settings: Option<&Bound<'py, PyDict>>,
    fit_settings: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    run_iod_impl(
        py,
        propagator,
        observed_ipc,
        observers_ipc,
        parse_iod_settings_over(iod_settings, legacy_iod_defaults())?,
        fit_settings,
    )
}

/// One-crossing `NativeOrbitFitter.full_od`: Gauss IOD (``iod_settings``)
/// on ``iod_propagator`` (None = ``propagator``) followed by the
/// ``refinement`` loop with ``fit_settings`` on ``propagator``.
#[pyfunction]
#[pyo3(signature = (propagator, observed_ipc, observers_ipc, iod_settings=None, refinement=None, fit_settings=None, iod_propagator=None))]
#[allow(clippy::too_many_arguments)]
fn full_od_ipc<'py>(
    py: Python<'py>,
    propagator: Option<&Bound<'py, PyAny>>,
    observed_ipc: &Bound<'py, PyBytes>,
    observers_ipc: &Bound<'py, PyBytes>,
    iod_settings: Option<&Bound<'py, PyDict>>,
    refinement: Option<&Bound<'py, PyDict>>,
    fit_settings: Option<&Bound<'py, PyDict>>,
    iod_propagator: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let problem = decode_observations(observed_ipc, observers_ipc)?;
    let settings = parse_fit_settings(fit_settings)?;
    let config = FullOdConfig {
        iod: parse_iod_settings(iod_settings)?,
        refinement: parse_refinement(refinement, settings.config)?,
    };
    let n = problem.observed.len();
    let output = run_with_predictors(
        py,
        iod_propagator,
        propagator,
        &settings,
        |iod_predictor, predictor| {
            full_od_with(
                iod_predictor,
                predictor,
                &problem.observed,
                &problem.observers,
                &config,
                &ErfaTimeProvider,
                &LockingSpiceTranslation,
            )
        },
    )?;
    full_od_dict(py, output, n)
}

/// `full_od_ipc` over adam_core's two-body propagator (``iod_propagator``
/// may still name a Python propagator for the IOD stage).
#[pyfunction]
#[pyo3(signature = (observed_ipc, observers_ipc, iod_settings=None, refinement=None, fit_settings=None, iod_propagator=None))]
fn full_od_2body_ipc<'py>(
    py: Python<'py>,
    observed_ipc: &Bound<'py, PyBytes>,
    observers_ipc: &Bound<'py, PyBytes>,
    iod_settings: Option<&Bound<'py, PyDict>>,
    refinement: Option<&Bound<'py, PyDict>>,
    fit_settings: Option<&Bound<'py, PyDict>>,
    iod_propagator: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    full_od_ipc(
        py,
        None,
        observed_ipc,
        observers_ipc,
        iod_settings,
        refinement,
        fit_settings,
        iod_propagator,
    )
}

/// One-crossing `run_od`: the observation models (``models`` = list of
/// ``_native_spec()`` dicts, applied in order) on the original observations,
/// the full OD on the used observations, and the original / used astrometry
/// snapshots for the members' provenance.
#[pyfunction]
#[pyo3(signature = (
    propagator,
    observed_ipc,
    observers_ipc,
    obs_code,
    band,
    astcat,
    models=None,
    iod_settings=None,
    refinement=None,
    fit_settings=None,
    iod_propagator=None
))]
#[allow(clippy::too_many_arguments)]
fn run_od_ipc<'py>(
    py: Python<'py>,
    propagator: Option<&Bound<'py, PyAny>>,
    observed_ipc: &Bound<'py, PyBytes>,
    observers_ipc: &Bound<'py, PyBytes>,
    obs_code: Vec<String>,
    band: Vec<Option<String>>,
    astcat: Vec<Option<String>>,
    models: Option<&Bound<'py, PyList>>,
    iod_settings: Option<&Bound<'py, PyDict>>,
    refinement: Option<&Bound<'py, PyDict>>,
    fit_settings: Option<&Bound<'py, PyDict>>,
    iod_propagator: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let problem = decode_observations(observed_ipc, observers_ipc)?;
    let settings = parse_fit_settings(fit_settings)?;
    let config = FullOdConfig {
        iod: parse_iod_settings(iod_settings)?,
        refinement: parse_refinement(refinement, settings.config)?,
    };
    let models = models_from_specs(models)?;
    let n = problem.observed.len();
    let output = run_with_predictors(
        py,
        iod_propagator,
        propagator,
        &settings,
        move |iod_predictor, predictor| {
            run_od_with(
                iod_predictor,
                predictor,
                &problem.observed,
                &problem.observers,
                &obs_code,
                &band,
                &astcat,
                models,
                &config,
                &ErfaTimeProvider,
                &LockingSpiceTranslation,
            )
        },
    )?;
    let out = full_od_dict(py, output.full_od, n)?;
    out.set_item("original_astrometry", snapshot_dict(py, output.original)?)?;
    out.set_item("used_astrometry", snapshot_dict(py, output.used)?)?;
    out.set_item("models_changed", output.models_changed)?;
    Ok(out)
}

/// `run_od_ipc` over adam_core's two-body propagator.
#[pyfunction]
#[pyo3(signature = (
    observed_ipc,
    observers_ipc,
    obs_code,
    band,
    astcat,
    models=None,
    iod_settings=None,
    refinement=None,
    fit_settings=None,
    iod_propagator=None
))]
#[allow(clippy::too_many_arguments)]
fn run_od_2body_ipc<'py>(
    py: Python<'py>,
    observed_ipc: &Bound<'py, PyBytes>,
    observers_ipc: &Bound<'py, PyBytes>,
    obs_code: Vec<String>,
    band: Vec<Option<String>>,
    astcat: Vec<Option<String>>,
    models: Option<&Bound<'py, PyList>>,
    iod_settings: Option<&Bound<'py, PyDict>>,
    refinement: Option<&Bound<'py, PyDict>>,
    fit_settings: Option<&Bound<'py, PyDict>>,
    iod_propagator: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    run_od_ipc(
        py,
        None,
        observed_ipc,
        observers_ipc,
        obs_code,
        band,
        astcat,
        models,
        iod_settings,
        refinement,
        fit_settings,
        iod_propagator,
    )
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_orbit_whitened_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(fit_orbit_whitened_2body_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(validate_fit_covariance_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(iterative_fit_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(iterative_fit_2body_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(cmc2003_fit_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(cmc2003_fit_2body_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(iod_fit_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(run_iod_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(iod_defaults, m)?)?;
    m.add_function(wrap_pyfunction!(run_iod_2body_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(full_od_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(full_od_2body_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(run_od_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(run_od_2body_ipc, m)?)?;
    Ok(())
}
