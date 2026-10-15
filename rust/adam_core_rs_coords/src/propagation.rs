//! Typed propagation abstractions for standalone `adam-core-rs` workflows.
//!
//! RM-STANDALONE-006 starts the propagation layer inside `adam_core_rs_coords`
//! under the modules-first policy. The public contracts use Rust-native
//! `OrbitBatch`/`TimeArray` types, require an explicit `TimeScaleProvider`, and
//! keep per-row numerical failures in `Validity` plus diagnostics while reserving
//! `Err` for request/setup errors.

mod diagnostics;
mod ephemeris;
mod od;
mod od_ranging;
mod od_refine;
mod pipeline;
mod request;
#[cfg(test)]
mod tests;
mod two_body;

pub use diagnostics::{
    PropagationConvergence, PropagationConvergenceStatus, PropagationDiagnostics,
    PropagationFailureCode, RowOutput,
};
pub use ephemeris::{
    generate_ephemeris, generate_ephemeris_barycentric, generate_ephemeris_translated,
    EphemerisDiagnostics, EphemerisFailureCode, EphemerisOptions, EphemerisPhotometryOptions,
    EphemerisResult, EphemerisRowDiagnostic,
};
pub use od::{
    candidate_orbit_batch, evaluate_orbit, evaluate_orbit_barycentric,
    fit_orbit_least_squares_barycentric, fit_orbit_least_squares_evaluated_barycentric, iod_fit,
    iod_fit_barycentric, iod_fit_linkages_barycentric, od_fit_barycentric, run_iod,
    vallado_least_squares_barycentric, CorrectionConfig, EvaluatedLeastSquaresFit, FitEvaluation,
    IodAcceptance, IodCandidate, IodConfig, IodOutput, IodResult, ObservationSelectionMethod,
    OdConfig, OdMethod, OdOutput, OrbitGeometry, PrefitScope, PropagatorPredictor,
    SpanSearchConfig, SphericalPredictor, ValladoConfig, ValladoIteration, ValladoResult,
    ValladoStatus, VelocityMethod, INVALID_LIGHT_TIME_MESSAGE, IOD_DEFAULT_MU,
    IOD_DEFAULT_SPEED_OF_LIGHT,
};
pub use od_ranging::{
    admissible_rho_dot_interval, attributable, ranging_state, systematic_ranging_with,
    Attributable, RangingConfig, RangingResult, RangingSample, RANGING_EARTH_MU,
};
pub use od_refine::{
    cmc2003_fit_barycentric, cmc2003_fit_with, fit_orbit_whitened_barycentric,
    fit_orbit_whitened_with, full_od_barycentric, full_od_with, iterative_fit_barycentric,
    iterative_fit_with, max_outliers, refine_barycentric, refine_with, run_iod_barycentric,
    run_iod_corrected, run_iod_with, run_od_barycentric, run_od_with, symmetric_eigen_6x6,
    validate_fit_covariance_with, AstrometrySnapshot, Cmc2003FitConfig, Cmc2003FitOutput,
    CovarianceValidation, FullOdConfig, FullOdOutput, IterativeFitConfig, IterativeFitOutput,
    JacobianMethod, RefinedOutput, RefinementConfig, RunOdOutput, WhitenedFitConfig,
    WhitenedFitOutput, DELTA_CHI2_WINDOW,
};
pub use pipeline::{OrbitRow, PropagationResult};
pub use request::{
    CovariancePropagation, EpochOrder, EpochPolicy, PropagationInput, PropagationOptions,
    PropagationRequest,
};
pub use two_body::{TwoBodyPropagator, TwoBodyPropagatorConfig, TwoBodyShard};

use crate::types::time::TimeScaleProvider;
use crate::{Epoch, SchemaError, TimeArray, TimeScale};
use std::fmt;

pub type PropagationResultValue<T> = std::result::Result<T, PropagationError>;

#[derive(Debug, Clone, PartialEq)]
pub enum PropagationError {
    Schema(SchemaError),
    InvalidRequest(String),
    MissingOrbitTimes,
    UnsupportedCovarianceMode(CovariancePropagation),
    Backend(String),
    BackendProtocol(String),
    ThreadPool(String),
}

impl fmt::Display for PropagationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Schema(err) => write!(f, "schema error: {err}"),
            Self::InvalidRequest(message) => write!(f, "invalid propagation request: {message}"),
            Self::MissingOrbitTimes => write!(f, "propagation requires per-orbit coordinate times"),
            Self::UnsupportedCovarianceMode(mode) => {
                write!(f, "unsupported covariance propagation mode: {mode:?}")
            }
            Self::Backend(message) => write!(f, "propagation backend error: {message}"),
            Self::BackendProtocol(message) => {
                write!(f, "propagation backend protocol error: {message}")
            }
            Self::ThreadPool(message) => {
                write!(f, "failed to build propagation thread pool: {message}")
            }
        }
    }
}

impl std::error::Error for PropagationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Schema(err) => Some(err),
            _ => None,
        }
    }
}

impl From<SchemaError> for PropagationError {
    fn from(value: SchemaError) -> Self {
        Self::Schema(value)
    }
}

/// Who a [`Propagator`] is: recorded in orbit-determination provenance and
/// used to decide which derivatives are exact for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendIdentity {
    /// Stable backend name (e.g. `adam_core.two_body`, `adam_assist`).
    pub name: String,
    /// Backend version (its crate or package version).
    pub version: String,
    /// Whether the backend integrates exactly the Kepler two-body problem, so
    /// the analytic two-body Jacobian is its exact derivative.
    pub two_body_dynamics: bool,
}

pub trait Propagator: Sync {
    type Shard: PropagatorShard;

    /// The backend's identity. The default names the implementing type with
    /// an unknown version and no two-body declaration; backends override it.
    fn backend(&self) -> BackendIdentity {
        BackendIdentity {
            name: std::any::type_name::<Self>().to_string(),
            version: "unknown".to_string(),
            two_body_dynamics: false,
        }
    }

    fn integration_time_scale(&self) -> TimeScale;
    fn supports(&self, mode: CovariancePropagation) -> bool;
    /// Called once per Rayon task/chunk in the default typed dispatch path.
    /// Stateful backends can own mutable per-worker/per-chunk integration state
    /// inside the shard without requiring `Sync` on that state.
    fn create_shard(&self) -> Self::Shard;
    fn propagate(
        &self,
        request: &PropagationRequest<'_>,
        provider: &dyn TimeScaleProvider,
    ) -> PropagationResultValue<PropagationResult>;
}

pub trait PropagatorShard: Send {
    fn propagate_one(
        &mut self,
        orbit: OrbitRow<'_>,
        times: &[Epoch],
    ) -> PropagationResultValue<RowOutput>;
}

fn rescale_for_propagation(
    times: &TimeArray,
    scale: TimeScale,
    provider: &dyn TimeScaleProvider,
    field: &str,
) -> PropagationResultValue<TimeArray> {
    let rescaled = times.rescale_with_provider(scale, provider)?;
    rescaled.validate()?;
    if rescaled.scale != scale {
        return Err(PropagationError::InvalidRequest(format!(
            "{field} provider returned {} instead of required {}",
            rescaled.scale.as_str(),
            scale.as_str()
        )));
    }
    if rescaled.len() != times.len() {
        return Err(PropagationError::InvalidRequest(format!(
            "{field} provider changed length from {} to {}",
            times.len(),
            rescaled.len()
        )));
    }
    Ok(rescaled)
}

fn run_with_thread_limit<T, F>(thread_limit: Option<usize>, f: F) -> PropagationResultValue<T>
where
    T: Send,
    F: FnOnce() -> PropagationResultValue<T> + Send,
{
    match thread_limit {
        Some(threads) => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|err| PropagationError::ThreadPool(err.to_string()))?;
            pool.install(f)
        }
        None => f(),
    }
}
