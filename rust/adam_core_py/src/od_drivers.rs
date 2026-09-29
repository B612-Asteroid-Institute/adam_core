//! PyO3 surface of the backend-generic Rust OD drivers over adam_core's own
//! two-body propagator: the one-crossing work units that a Rust-backed
//! propagator plugin (adam-assist) exposes as methods are exposed here for the
//! built-in two-body backend, so the Python veneer's fused dispatch has an
//! in-tree implementation and the parity suite can exercise it.
//!
//! Inputs cross as nested Arrow IPC (the `arrow_bridge` helpers); outputs are
//! plain dicts of scalars, lists and NumPy arrays that the veneer wraps into
//! `FittedOrbits` / `FittedOrbitMembers` without further computation.

use crate::coordinates::{read_orbit_ipc, ErfaTimeProvider};
use adam_core_rs_coords::propagation::{
    fit_orbit_whitened_barycentric, CovariancePropagation, EphemerisOptions, EpochPolicy,
    JacobianMethod, PropagationOptions, TwoBodyPropagator, TwoBodyPropagatorConfig,
    WhitenedFitConfig,
};
use adam_core_rs_coords::{
    validate_loss, CoordinateBatch as DataCoordinateBatch, ObserverBatch as DataObserverBatch,
    OrbitBatch as DataOrbitBatch, TimeScale, TryFromNestedRecordBatch, TwoBodyModelConfig,
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
    Ok(OdProblem {
        orbit,
        observed,
        observers,
    })
}

#[allow(clippy::too_many_arguments)]
fn ephemeris_options(
    lt_tol: f64,
    eph_max_iter: usize,
    eph_tol: f64,
    stellar_aberration: bool,
    max_lt_iter: usize,
) -> EphemerisOptions {
    EphemerisOptions {
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
    }
}

/// One-crossing whitened `fit_least_squares` over the two-body backend: the
/// analytic / central / 2-point Jacobian, linear or Huber loss, the validated
/// covariance and the fused final evaluation. Returns the dict contract of a
/// propagator's ``fit_least_squares_whitened`` work unit.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    orbit_ipc,
    observed_ipc,
    observers_ipc,
    ignore,
    loss="linear",
    f_scale=1.345,
    jacobian="analytic",
    validate_covariance=true,
    xtol=1e-12,
    ftol=1e-12,
    gtol=1e-12,
    max_iterations=100,
    lt_tol=1.0e-12,
    eph_max_iter=1000,
    eph_tol=1.0e-15,
    stellar_aberration=false,
    max_lt_iter=10,
    prop_max_iter=1000,
    prop_tol=1e-14
))]
fn fit_orbit_whitened_2body_ipc<'py>(
    py: Python<'py>,
    orbit_ipc: &Bound<'py, PyBytes>,
    observed_ipc: &Bound<'py, PyBytes>,
    observers_ipc: &Bound<'py, PyBytes>,
    ignore: Vec<bool>,
    loss: &str,
    f_scale: f64,
    jacobian: &str,
    validate_covariance: bool,
    xtol: f64,
    ftol: f64,
    gtol: f64,
    max_iterations: usize,
    lt_tol: f64,
    eph_max_iter: usize,
    eph_tol: f64,
    stellar_aberration: bool,
    max_lt_iter: usize,
    prop_max_iter: usize,
    prop_tol: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let problem = decode_problem(orbit_ipc, observed_ipc, observers_ipc)?;
    let loss = validate_loss(loss, f_scale).map_err(value_error)?;
    let jacobian = JacobianMethod::parse(jacobian).map_err(value_error)?;
    let config = WhitenedFitConfig {
        loss,
        f_scale,
        jacobian,
        validate_covariance,
        xtol,
        ftol,
        gtol,
        max_iterations,
        two_body: TwoBodyModelConfig {
            propagation_max_iter: prop_max_iter,
            propagation_tol: prop_tol,
            lt_tol,
            ephemeris_max_iter: eph_max_iter,
            ephemeris_tol: eph_tol,
            max_lt_iter,
        },
    };
    let options = ephemeris_options(
        lt_tol,
        eph_max_iter,
        eph_tol,
        stellar_aberration,
        max_lt_iter,
    );
    let propagator = TwoBodyPropagator::new(TwoBodyPropagatorConfig {
        max_iter: prop_max_iter,
        tol: prop_tol,
    })
    .map_err(|err| value_error(format!("invalid propagator config: {err}")))?;

    let output = py
        .allow_threads(|| {
            let backend = global_backend()
                .lock()
                .map_err(|_| "SPICE backend lock is poisoned".to_string())?;
            fit_orbit_whitened_barycentric(
                &propagator,
                &problem.orbit,
                &problem.observed,
                &problem.observers,
                &ignore,
                &config,
                &options,
                &ErfaTimeProvider,
                &*backend,
            )
            .map_err(|err| err.to_string())
        })
        .map_err(value_error)?;

    let n = problem.observed.len();
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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_orbit_whitened_2body_ipc, m)?)?;
    Ok(())
}
