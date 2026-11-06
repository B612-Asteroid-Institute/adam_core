//! Typed propagation abstractions for standalone `adam-core-rs` workflows.
//!
//! RM-STANDALONE-006 starts the propagation layer inside `adam_core_rs_coords`
//! under the modules-first policy. The public contracts use Rust-native
//! `OrbitBatch`/`TimeArray` types, require an explicit `TimeScaleProvider`, and
//! keep per-row numerical failures in `Validity` plus diagnostics while reserving
//! `Err` for request/setup errors.

mod diagnostics;
mod ephemeris;
mod external;
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
pub use external::{
    two_body_reference, ExternalContext, ExternalPropagator, ExternalPropagatorV2,
    OwnedExternalPropagator, PropagateRequestV2, PropagateStmRequestV2, PropagateStmV2,
    PropagateV2, EXTERNAL_NON_GRAVITATIONAL_WIDTH, EXTERNAL_PHYSICAL_PARAMETERS_WIDTH,
    EXTERNAL_PROPAGATION_FAILURE_INTEGRATOR, EXTERNAL_PROPAGATION_FAILURE_NONE,
    EXTERNAL_PROPAGATION_FAILURE_NON_FINITE_COVARIANCE,
    EXTERNAL_PROPAGATION_FAILURE_NON_FINITE_INPUT_STATE,
    EXTERNAL_PROPAGATION_FAILURE_NON_FINITE_OUTPUT_STATE,
    EXTERNAL_PROPAGATION_FAILURE_SOLVER_MAX_ITERATIONS,
    EXTERNAL_PROPAGATION_FAILURE_SOLVER_ZERO_DERIVATIVE, EXTERNAL_PROPAGATOR_ABI_VERSION,
    EXTERNAL_PROPAGATOR_CAPSULE, EXTERNAL_PROPAGATOR_CAP_NON_GRAVITATIONAL,
    EXTERNAL_PROPAGATOR_CAP_STM, EXTERNAL_PROPAGATOR_CAP_THREAD_SAFE,
    EXTERNAL_PROPAGATOR_KNOWN_CAPABILITIES, EXTERNAL_PROPAGATOR_RESERVED_WORDS,
    EXTERNAL_PROPAGATOR_STATUS_ERROR, EXTERNAL_PROPAGATOR_STATUS_INVALID,
    EXTERNAL_PROPAGATOR_STATUS_OK, EXTERNAL_PROPAGATOR_STATUS_PANIC,
    EXTERNAL_PROPAGATOR_V2_LEGACY_FULL_SIZE, EXTERNAL_PROPAGATOR_V2_METADATA_SIZE,
    EXTERNAL_PROPAGATOR_V2_MIN_SIZE, EXTERNAL_PROPAGATOR_V2_STM_SIZE,
    PROPAGATE_REQUEST_V2_MIN_SIZE, PROPAGATE_STM_REQUEST_V2_MIN_SIZE,
};
pub(crate) use od::validate_od_covariance_dimension;
#[allow(deprecated)]
pub use od::{
    candidate_orbit_batch, evaluate_orbit, evaluate_orbit_barycentric,
    fit_orbit_least_squares_barycentric, fit_orbit_least_squares_evaluated_barycentric, iod_fit,
    iod_fit_barycentric, vallado_least_squares_barycentric, CorrectionConfig,
    EvaluatedLeastSquaresFit, FitEvaluation, IodAcceptance, IodCandidate, IodCandidateFailure,
    IodConfig, IodOutput, IodResult, ObservationSelectionMethod, OrbitGeometry, PrefitScope,
    SpanSearchConfig, ValladoConfig, ValladoIteration, ValladoResult, ValladoStatus,
    VelocityMethod, INVALID_LIGHT_TIME_MESSAGE, IOD_DEFAULT_MU, IOD_DEFAULT_SPEED_OF_LIGHT,
};
pub use od_ranging::{
    admissible_rho_dot_interval, attributable, ranging_state, Attributable, RangingConfig,
    RangingResult, RangingSample, RANGING_EARTH_MU,
};
pub use od_refine::{
    apply_configured_observation_pipeline, cmc2003_fit_barycentric, fit_orbit_whitened_barycentric,
    iterative_fit_barycentric, max_outliers, symmetric_eigen_6x6, AppliedObservationModels,
    AstrometrySnapshot, Cmc2003FitConfig, Cmc2003FitOutput, CovarianceValidation,
    IterativeFitConfig, IterativeFitOutput, JacobianMethod, RefinedOutput, RefinementConfig,
    WhitenedFitConfig, WhitenedFitOutput, DELTA_CHI2_WINDOW,
};
pub use pipeline::{OrbitRow, PropagationResult};

/// Predictor-level focused OD kernels over the narrow [`od::SphericalPredictor`]
/// interface. Canonical configured composition lives under
/// `crate::od::algorithms`; these focused kernels are not alternate public
/// batch orchestration.
pub mod adapter {
    pub use super::od::{run_iod, PropagatorPredictor, SphericalPredictor};
    pub use super::od_ranging::systematic_ranging_with;
    pub use super::od_refine::{
        cmc2003_fit_with, fit_orbit_whitened_with, iterative_fit_with, refine_with,
        run_iod_corrected, run_iod_with, validate_fit_covariance_with,
    };
}
pub use request::{
    CovariancePropagation, EpochOrder, EpochPolicy, PropagationInput, PropagationOptions,
    PropagationRequest,
};
pub use two_body::{TwoBodyPropagator, TwoBodyPropagatorConfig, TwoBodyShard};

use crate::types::time::TimeScaleProvider;
use crate::{Epoch, SchemaError, TimeArray, TimeScale};
use std::fmt;

pub type PropagationResultValue<T> = std::result::Result<T, PropagationError>;

/// Stable typed classifications for group-local numerical/scientific outcomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupScientificFailureCode {
    RankDeficient,
    NoIncludedObservations,
    NoFitPass,
}

impl GroupScientificFailureCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RankDeficient => "rank_deficient",
            Self::NoIncludedObservations => "no_included_observations",
            Self::NoFitPass => "no_fit_pass",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PropagationError {
    Schema(SchemaError),
    InvalidRequest(String),
    /// CMC2003 cannot calibrate expected residuals without a validated covariance.
    CmcCovarianceUnavailable(String),
    MissingOrbitTimes,
    UnsupportedCovarianceMode(CovariancePropagation),
    /// An explicitly classified group-local numerical/scientific outcome.
    GroupScientific {
        code: GroupScientificFailureCode,
        message: String,
    },
    /// Candidate-local propagation failure reported in-band by a successful backend request.
    /// This is not safe to continue at the group-workflow boundary because a
    /// provider may use it for global coverage/resource failures.
    RowFailure(String),
    /// Fatal provider operation failure (including a caught provider panic).
    Backend(String),
    /// Fatal ABI/protocol failure.
    BackendProtocol(String),
    ThreadPool(String),
}

impl fmt::Display for PropagationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Schema(err) => write!(f, "schema error: {err}"),
            Self::InvalidRequest(message) => write!(f, "invalid propagation request: {message}"),
            Self::CmcCovarianceUnavailable(message) => {
                write!(f, "CMC2003 expected covariance unavailable: {message}")
            }
            Self::MissingOrbitTimes => write!(f, "propagation requires per-orbit coordinate times"),
            Self::UnsupportedCovarianceMode(mode) => {
                write!(f, "unsupported covariance propagation mode: {mode:?}")
            }
            Self::GroupScientific { code, message } => {
                write!(f, "group scientific failure ({}): {message}", code.as_str())
            }
            Self::RowFailure(message) => write!(f, "propagation row failure: {message}"),
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
    /// Stable backend name (for example `adam_core.two_body` or `external.n_body`).
    pub name: String,
    /// Backend version (its crate or package version).
    pub version: String,
    /// Whether the backend integrates exactly the Kepler two-body problem, so
    /// the analytic two-body Jacobian is its exact derivative.
    pub two_body_dynamics: bool,
}

/// One propagated row with its state-transition matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StmRow {
    /// Heliocentric ecliptic state at the target time.
    pub state: [f64; 6],
    /// Row-major `d state(t) / d state(epoch)`.
    pub stm: [f64; 36],
    pub valid: bool,
}

pub trait Propagator: Sync {
    type Shard: PropagatorShard;

    /// Whether [`Propagator::propagate_stm`] is implemented: the backend
    /// supplies exact derivatives of its own dynamics, which orbit
    /// determination then uses (`JacobianMethod::Propagator`).
    fn supports_stm(&self) -> bool {
        false
    }

    /// Whether propagation accepts per-orbit non-gravitational parameters
    /// (A1/A2/A3 and their Marsden-law metadata) without dropping them.
    fn supports_non_gravitational_parameters(&self) -> bool {
        false
    }

    /// One heliocentric (SUN) ecliptic state at `epoch_tdb`, propagated to
    /// each of `times_tdb` (TDB) with its state-transition matrix. Optional
    /// non-gravitational parameters are held fixed while differentiating the
    /// six Cartesian state components.
    fn propagate_stm(
        &self,
        _state: [f64; 6],
        _non_gravitational_parameters: Option<crate::NonGravitationalParametersRow>,
        _epoch_tdb: crate::Epoch,
        _times_tdb: &[crate::Epoch],
    ) -> PropagationResultValue<Vec<StmRow>> {
        Err(PropagationError::InvalidRequest(
            "this propagator provides no state-transition matrices".to_string(),
        ))
    }

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
