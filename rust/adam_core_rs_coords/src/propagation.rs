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
    evaluate_orbit_barycentric, fit_orbit_least_squares_barycentric,
    fit_orbit_least_squares_evaluated_barycentric, iod_fit_barycentric,
    iod_fit_linkages_barycentric, od_fit_barycentric, vallado_least_squares_barycentric,
    EvaluatedLeastSquaresFit, FitEvaluation, IodConfig, IodOutput, ObservationSelectionMethod,
    OdConfig, OdMethod, OdOutput, ValladoConfig, ValladoIteration, ValladoResult, ValladoStatus,
    INVALID_LIGHT_TIME_MESSAGE,
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

pub trait Propagator: Sync {
    type Shard: PropagatorShard;

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
