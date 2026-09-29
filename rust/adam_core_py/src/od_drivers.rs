//! PyO3 surface of the backend-generic Rust OD drivers over adam_core's own
//! two-body propagator: the one-crossing work units that a Rust-backed
//! propagator plugin (adam-assist) exposes as methods are exposed here for the
//! built-in two-body backend, so the Python veneer's fused dispatch has an
//! in-tree implementation and the parity suite can exercise it.
//!
//! Inputs cross as nested Arrow IPC (the `arrow_bridge` helpers) plus a
//! ``fit_settings`` dict of the whitened-fit options; outputs are plain dicts
//! of scalars, lists and NumPy arrays that the veneer wraps into
//! `FittedOrbits` / `FittedOrbitMembers` without further computation.

use crate::coordinates::{read_orbit_ipc, ErfaTimeProvider};
use adam_core_rs_coords::propagation::{
    cmc2003_fit_barycentric, fit_orbit_whitened_barycentric, iterative_fit_barycentric,
    Cmc2003FitConfig, CovariancePropagation, EphemerisOptions, EpochPolicy, IterativeFitConfig,
    JacobianMethod, PropagationOptions, TwoBodyPropagator, TwoBodyPropagatorConfig,
    WhitenedFitConfig, WhitenedFitOutput,
};
use adam_core_rs_coords::{
    validate_loss, CoordinateBatch as DataCoordinateBatch, ObserverBatch as DataObserverBatch,
    OrbitBatch as DataOrbitBatch, TimeScale, TryFromNestedRecordBatch, TwoBodyModelConfig,
    CMC2003_APPARITION_GAP_DAYS, CMC2003_CHI2_FRAC, CMC2003_CHI2_RECOVER, CMC2003_CHI2_REJECT,
    CMC2003_MAX_ITERATIONS, CMC2003_MAX_REJECTED_FRACTION, CMC2003_PSD_FLOOR_FRAC,
};
use adam_core_rs_spice::global_backend;
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

struct OdProblem {
    orbit: DataOrbitBatch,
    observed: DataCoordinateBatch,
    observers: DataObserverBatch,
}

fn decode_problem(
    orbit_ipc: &Bound<'_, PyBytes>,
    observed_ipc: &Bound<'_, PyBytes>,
    observers_ipc: &Bound<'_, PyBytes>,
) -> PyResult<OdProblem> {
    let orbit =
        DataOrbitBatch::try_from_nested_record_batch(&read_orbit_ipc(orbit_ipc.as_bytes())?)
            .map_err(|err| value_error(format!("failed to decode OrbitBatch: {err}")))?;
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
    Ok(OdProblem {
        orbit,
        observed,
        observers,
    })
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

fn parse_fit_settings(settings: Option<&Bound<'_, PyDict>>) -> PyResult<FitSettings> {
    let loss: String = setting(settings, "loss", "linear".to_string())?;
    let f_scale: f64 = setting(settings, "f_scale", 1.345)?;
    let loss = validate_loss(&loss, f_scale).map_err(value_error)?;
    let jacobian: String = setting(settings, "jacobian", "analytic".to_string())?;
    let jacobian = JacobianMethod::parse(&jacobian).map_err(value_error)?;
    let lt_tol: f64 = setting(settings, "lt_tol", 1.0e-12)?;
    let eph_max_iter: usize = setting(settings, "eph_max_iter", 1000)?;
    let eph_tol: f64 = setting(settings, "eph_tol", 1.0e-15)?;
    let stellar_aberration: bool = setting(settings, "stellar_aberration", false)?;
    let max_lt_iter: usize = setting(settings, "max_lt_iter", 10)?;
    let prop_max_iter: usize = setting(settings, "prop_max_iter", 1000)?;
    let prop_tol: f64 = setting(settings, "prop_tol", 1e-14)?;
    let config = WhitenedFitConfig {
        loss,
        f_scale,
        jacobian,
        validate_covariance: setting(settings, "validate_covariance", true)?,
        xtol: setting(settings, "xtol", 1e-12)?,
        ftol: setting(settings, "ftol", 1e-12)?,
        gtol: setting(settings, "gtol", 1e-12)?,
        max_iterations: setting(settings, "max_iterations", 100)?,
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

/// One-crossing whitened `fit_least_squares` over the two-body backend: the
/// analytic / central / 2-point Jacobian, linear or Huber loss, the validated
/// covariance and the fused final evaluation. Returns the dict contract of a
/// propagator's ``fit_least_squares_whitened`` work unit.
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
    let problem = decode_problem(orbit_ipc, observed_ipc, observers_ipc)?;
    let settings = parse_fit_settings(fit_settings)?;
    let n = problem.observed.len();
    let output = py
        .allow_threads(|| {
            let backend = global_backend()
                .lock()
                .map_err(|_| "SPICE backend lock is poisoned".to_string())?;
            fit_orbit_whitened_barycentric(
                &settings.propagator,
                &problem.orbit,
                &problem.observed,
                &problem.observers,
                &ignore,
                &settings.config,
                &settings.options,
                &ErfaTimeProvider,
                &*backend,
            )
            .map_err(|err| err.to_string())
        })
        .map_err(value_error)?;
    fit_output_dict(py, output, n)
}

/// One-crossing `iterative_fit` (worst-residual rejection loop) over the
/// two-body backend. Returns the fused fit dict of the selected pass plus
/// ``passes``.
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
    let problem = decode_problem(orbit_ipc, observed_ipc, observers_ipc)?;
    let settings = parse_fit_settings(fit_settings)?;
    let config = IterativeFitConfig {
        rchi2_threshold,
        min_obs,
        min_arc_length,
        contamination_percentage,
        fit: settings.config,
    };
    let n = problem.observed.len();
    let output = py
        .allow_threads(|| {
            let backend = global_backend()
                .lock()
                .map_err(|_| "SPICE backend lock is poisoned".to_string())?;
            iterative_fit_barycentric(
                &settings.propagator,
                &problem.orbit,
                &problem.observed,
                &problem.observers,
                &config,
                &settings.options,
                &ErfaTimeProvider,
                &*backend,
            )
            .map_err(|err| err.to_string())
        })
        .map_err(value_error)?;
    let out = fit_output_dict(py, output.fit, n)?;
    out.set_item("passes", output.passes)?;
    Ok(out)
}

/// One-crossing `cmc2003_fit_detailed` over the two-body backend. Returns the
/// fused fit dict of the final pass plus ``n_iterations``, ``n_rejected``,
/// ``n_recovered`` and the sorted ``flags``.
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
    let problem = decode_problem(orbit_ipc, observed_ipc, observers_ipc)?;
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
    let output = py
        .allow_threads(|| {
            let backend = global_backend()
                .lock()
                .map_err(|_| "SPICE backend lock is poisoned".to_string())?;
            cmc2003_fit_barycentric(
                &settings.propagator,
                &problem.orbit,
                &problem.observed,
                &problem.observers,
                &config,
                &settings.options,
                &ErfaTimeProvider,
                &*backend,
            )
            .map_err(|err| err.to_string())
        })
        .map_err(value_error)?;
    let out = fit_output_dict(py, output.fit, n)?;
    out.set_item("n_iterations", output.n_iterations)?;
    out.set_item("n_rejected", output.n_rejected)?;
    out.set_item("n_recovered", output.n_recovered)?;
    out.set_item("flags", output.flags)?;
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_orbit_whitened_2body_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(iterative_fit_2body_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(cmc2003_fit_2body_ipc, m)?)?;
    Ok(())
}
