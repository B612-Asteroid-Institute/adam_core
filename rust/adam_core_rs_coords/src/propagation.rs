//! Typed propagation abstractions for standalone `adam-core-rs` workflows.
//!
//! RM-STANDALONE-006 starts the propagation layer inside `adam_core_rs_coords`
//! under the modules-first policy. The public contracts use Rust-native
//! `OrbitBatch`/`TimeArray` types, require an explicit `TimeScaleProvider`, and
//! keep per-row numerical failures in `Validity` plus diagnostics while reserving
//! `Err` for request/setup errors.

use crate::propagate::{propagate_2body_along_arc, propagate_2body_with_covariance_row};
use crate::types::origin_mu_au3_day2;
use crate::types::time::TimeScaleProvider;
use crate::{
    CoordinateBatch, CovarianceBatch, CovarianceUnits, Epoch, ObjectId, OrbitBatch, OrbitId,
    OriginArray, OriginId, SchemaError, TimeArray, TimeScale, Validity,
};
use rayon::prelude::*;
use std::fmt;

const DEFAULT_TWO_BODY_MAX_ITER: usize = 1000;
const DEFAULT_TWO_BODY_TOL: f64 = 1.0e-14;

pub type PropagationResultValue<T> = std::result::Result<T, PropagationError>;

#[derive(Debug, Clone, PartialEq)]
pub enum PropagationError {
    Schema(SchemaError),
    InvalidRequest(String),
    MissingOrbitTimes,
    UnsupportedCovarianceMode(CovariancePropagation),
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpochPolicy {
    /// Each orbit propagates from its own `coordinates.times` row to every
    /// requested epoch. Output rows are orbit-major and preserve caller epoch
    /// order within each orbit.
    CrossProduct,
    /// Requires `len(orbits) == len(times)`; output has one row per orbit.
    Pairwise,
    /// Reserved for a future compact per-orbit epoch subset representation.
    PerOrbit { indices: Box<[u32]> },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CovariancePropagation {
    None,
    Linearized,
    Monte { samples: usize, seed: u64 },
    SigmaPoint { alpha: f64, beta: f64, kappa: f64 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct PropagationOptions {
    pub chunk_size: Option<usize>,
    pub thread_limit: Option<usize>,
    pub epoch_policy: EpochPolicy,
    pub covariance: CovariancePropagation,
}

impl Default for PropagationOptions {
    fn default() -> Self {
        Self {
            chunk_size: None,
            thread_limit: None,
            epoch_policy: EpochPolicy::CrossProduct,
            covariance: CovariancePropagation::Linearized,
        }
    }
}

impl PropagationOptions {
    pub fn validate(&self) -> PropagationResultValue<()> {
        if self.chunk_size == Some(0) {
            return Err(PropagationError::InvalidRequest(
                "chunk_size must be positive when provided".to_string(),
            ));
        }
        if self.thread_limit == Some(0) {
            return Err(PropagationError::InvalidRequest(
                "thread_limit must be positive when provided".to_string(),
            ));
        }
        match self.covariance {
            CovariancePropagation::Monte { samples: 0, .. } => {
                return Err(PropagationError::InvalidRequest(
                    "Monte covariance propagation requires samples > 0".to_string(),
                ));
            }
            CovariancePropagation::SigmaPoint { alpha, beta, kappa } => {
                if !alpha.is_finite() || !beta.is_finite() || !kappa.is_finite() {
                    return Err(PropagationError::InvalidRequest(
                        "sigma-point covariance parameters must be finite".to_string(),
                    ));
                }
                if alpha <= 0.0 {
                    return Err(PropagationError::InvalidRequest(
                        "sigma-point alpha must be positive".to_string(),
                    ));
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// Chronological permutation metadata for requested epochs.
///
/// `sorted_to_input[sorted_position] = caller_epoch_index` and
/// `input_to_sorted[caller_epoch_index] = sorted_position`. Propagators may use
/// this to process epochs chronologically while still returning caller-order
/// rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpochOrder {
    pub sorted_to_input: Vec<usize>,
    pub input_to_sorted: Vec<usize>,
    pub is_chronological: bool,
}

impl EpochOrder {
    pub fn from_times(times: &TimeArray) -> Self {
        let mut sorted_to_input = (0..times.len()).collect::<Vec<_>>();
        sorted_to_input.sort_by(|&left, &right| {
            times.epochs[left]
                .mjd()
                .partial_cmp(&times.epochs[right].mjd())
                .expect("epoch MJD values are finite")
        });
        let mut input_to_sorted = vec![0; times.len()];
        for (sorted_position, &input_index) in sorted_to_input.iter().enumerate() {
            input_to_sorted[input_index] = sorted_position;
        }
        let is_chronological = sorted_to_input.iter().copied().eq(0..times.len());
        Self {
            sorted_to_input,
            input_to_sorted,
            is_chronological,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PropagationRequest<'a> {
    pub orbits: &'a OrbitBatch,
    pub times: &'a TimeArray,
    pub options: PropagationOptions,
    pub epoch_order: EpochOrder,
}

impl<'a> PropagationRequest<'a> {
    pub fn new(
        orbits: &'a OrbitBatch,
        times: &'a TimeArray,
        options: PropagationOptions,
    ) -> PropagationResultValue<Self> {
        orbits.validate()?;
        times.validate()?;
        options.validate()?;
        if orbits.coordinates.times.is_none() {
            return Err(PropagationError::MissingOrbitTimes);
        }
        match &options.epoch_policy {
            EpochPolicy::CrossProduct => {}
            EpochPolicy::Pairwise => {
                if orbits.len() != times.len() {
                    return Err(PropagationError::InvalidRequest(format!(
                        "pairwise propagation requires len(orbits) == len(times); got {} and {}",
                        orbits.len(),
                        times.len()
                    )));
                }
            }
            EpochPolicy::PerOrbit { .. } => {
                return Err(PropagationError::InvalidRequest(
                    "PerOrbit epoch policy is reserved for a future compact representation"
                        .to_string(),
                ));
            }
        }
        Ok(Self {
            orbits,
            times,
            options,
            epoch_order: EpochOrder::from_times(times),
        })
    }

    pub fn output_len(&self) -> usize {
        match self.options.epoch_policy {
            EpochPolicy::CrossProduct => self.orbits.len() * self.times.len(),
            EpochPolicy::Pairwise => self.orbits.len(),
            EpochPolicy::PerOrbit { .. } => 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropagationConvergenceStatus {
    Converged,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropagationConvergence {
    pub output_row: usize,
    pub input_orbit_index: usize,
    pub input_time_index: usize,
    pub status: PropagationConvergenceStatus,
    pub message: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropagationDiagnostics {
    pub convergence: Vec<PropagationConvergence>,
    pub epoch_order: EpochOrder,
}

impl PropagationDiagnostics {
    pub fn failed_rows(&self) -> impl Iterator<Item = &PropagationConvergence> {
        self.convergence
            .iter()
            .filter(|row| row.status == PropagationConvergenceStatus::Failed)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PropagationResult {
    pub orbits: OrbitBatch,
    pub times: TimeArray,
    pub validity: Validity,
    pub diagnostics: PropagationDiagnostics,
}

#[derive(Debug, Clone, Copy)]
pub struct OrbitRow<'a> {
    pub index: usize,
    pub orbit_id: &'a OrbitId,
    pub object_id: Option<&'a ObjectId>,
    pub state: [f64; 6],
    pub origin: &'a OriginId,
    pub mu: f64,
    pub time: Epoch,
    pub covariance: Option<&'a [f64]>,
    pub covariance_valid: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RowOutput {
    pub states: Vec<[f64; 6]>,
    pub covariance: Option<Vec<[f64; 36]>>,
    pub covariance_validity: Option<Vec<bool>>,
    pub validity: Vec<bool>,
    pub messages: Vec<Option<String>>,
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoBodyPropagatorConfig {
    pub max_iter: usize,
    pub tol: f64,
}

impl Default for TwoBodyPropagatorConfig {
    fn default() -> Self {
        Self {
            max_iter: DEFAULT_TWO_BODY_MAX_ITER,
            tol: DEFAULT_TWO_BODY_TOL,
        }
    }
}

impl TwoBodyPropagatorConfig {
    pub fn validate(self) -> PropagationResultValue<()> {
        if self.max_iter == 0 {
            return Err(PropagationError::InvalidRequest(
                "TwoBodyPropagatorConfig.max_iter must be positive".to_string(),
            ));
        }
        if !self.tol.is_finite() || self.tol <= 0.0 {
            return Err(PropagationError::InvalidRequest(
                "TwoBodyPropagatorConfig.tol must be finite and positive".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TwoBodyPropagator {
    config: TwoBodyPropagatorConfig,
}

impl TwoBodyPropagator {
    pub fn new(config: TwoBodyPropagatorConfig) -> PropagationResultValue<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn config(&self) -> TwoBodyPropagatorConfig {
        self.config
    }

    fn propagate_blocks(
        &self,
        request: &PropagationRequest<'_>,
        orbit_times: &TimeArray,
        target_times: &TimeArray,
        mus: &[f64],
    ) -> PropagationResultValue<Vec<OrbitBlock>> {
        let states = request
            .orbits
            .coordinates
            .values
            .cartesian()
            .ok_or_else(|| {
                PropagationError::InvalidRequest(
                    "TwoBodyPropagator requires Cartesian orbit coordinates".to_string(),
                )
            })?;
        let covariance = request.orbits.coordinates.covariance.as_ref();
        let include_covariance =
            request.options.covariance == CovariancePropagation::Linearized && covariance.is_some();
        let orbit_indices = (0..request.orbits.len()).collect::<Vec<_>>();
        let chunk_size = normalized_chunk_size(request.options.chunk_size, request.orbits.len());
        let policy = request.options.epoch_policy.clone();

        let chunk_results = orbit_indices
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut shard = self.create_shard();
                let mut blocks = Vec::with_capacity(chunk.len());
                for &orbit_index in chunk {
                    let row = orbit_row(
                        request.orbits,
                        states,
                        orbit_times,
                        covariance,
                        mus,
                        orbit_index,
                        include_covariance,
                    );
                    let time_indices =
                        time_indices_for_policy(&policy, orbit_index, target_times.len());
                    let times = time_indices
                        .iter()
                        .map(|&time_index| target_times.epochs[time_index])
                        .collect::<Vec<_>>();
                    let output = shard.propagate_one(row, &times)?;
                    if output.states.len() != time_indices.len()
                        || output.validity.len() != time_indices.len()
                        || output.messages.len() != time_indices.len()
                    {
                        return Err(PropagationError::InvalidRequest(
                            "propagator shard returned inconsistent row lengths".to_string(),
                        ));
                    }
                    if let Some(covariance_rows) = &output.covariance {
                        if covariance_rows.len() != time_indices.len() {
                            return Err(PropagationError::InvalidRequest(
                                "propagator shard returned inconsistent covariance length"
                                    .to_string(),
                            ));
                        }
                    }
                    if let Some(covariance_validity) = &output.covariance_validity {
                        if covariance_validity.len() != time_indices.len() {
                            return Err(PropagationError::InvalidRequest(
                                "propagator shard returned inconsistent covariance validity length"
                                    .to_string(),
                            ));
                        }
                    }
                    blocks.push(OrbitBlock {
                        orbit_index,
                        time_indices,
                        states: output.states,
                        covariance: output.covariance,
                        covariance_validity: output.covariance_validity,
                        validity: output.validity,
                        messages: output.messages,
                    });
                }
                Ok(blocks)
            })
            .collect::<Vec<_>>();

        let mut blocks = Vec::with_capacity(request.orbits.len());
        for chunk in chunk_results {
            blocks.extend(chunk?);
        }
        Ok(blocks)
    }
}

impl Propagator for TwoBodyPropagator {
    type Shard = TwoBodyShard;

    fn integration_time_scale(&self) -> TimeScale {
        TimeScale::Tdb
    }

    fn supports(&self, mode: CovariancePropagation) -> bool {
        matches!(
            mode,
            CovariancePropagation::None | CovariancePropagation::Linearized
        )
    }

    fn create_shard(&self) -> Self::Shard {
        TwoBodyShard {
            config: self.config,
        }
    }

    fn propagate(
        &self,
        request: &PropagationRequest<'_>,
        provider: &dyn TimeScaleProvider,
    ) -> PropagationResultValue<PropagationResult> {
        request.orbits.validate()?;
        request.times.validate()?;
        request.options.validate()?;
        if !self.supports(request.options.covariance) {
            return Err(PropagationError::UnsupportedCovarianceMode(
                request.options.covariance,
            ));
        }
        let input_orbit_times = request
            .orbits
            .coordinates
            .times
            .as_ref()
            .ok_or(PropagationError::MissingOrbitTimes)?;
        let integration_scale = self.integration_time_scale();
        let orbit_times = rescale_for_propagation(
            input_orbit_times,
            integration_scale,
            provider,
            "orbit coordinate times",
        )?;
        let target_times =
            rescale_for_propagation(request.times, integration_scale, provider, "target times")?;
        let mus = request
            .orbits
            .coordinates
            .origins
            .origins
            .iter()
            .map(origin_mu_au3_day2)
            .collect::<Result<Vec<_>, _>>()?;

        run_with_thread_limit(request.options.thread_limit, || {
            let blocks = self.propagate_blocks(request, &orbit_times, &target_times, &mus)?;
            assemble_result(request, &target_times, blocks)
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoBodyShard {
    config: TwoBodyPropagatorConfig,
}

impl PropagatorShard for TwoBodyShard {
    fn propagate_one(
        &mut self,
        orbit: OrbitRow<'_>,
        times: &[Epoch],
    ) -> PropagationResultValue<RowOutput> {
        let dts = times
            .iter()
            .map(|epoch| epoch.mjd() - orbit.time.mjd())
            .collect::<Vec<_>>();

        let (states, covariance, covariance_validity) = match orbit.covariance {
            Some(covariance) if orbit.covariance_valid => {
                let input_covariance = covariance_36(covariance)?;
                let input_has_nan = input_covariance.iter().any(|value| value.is_nan());
                let mut states = Vec::with_capacity(dts.len());
                let mut covariance_rows = Vec::with_capacity(dts.len());
                let mut covariance_validity = Vec::with_capacity(dts.len());
                for &dt in &dts {
                    let (state, covariance_row) = propagate_2body_with_covariance_row(
                        orbit.state,
                        input_covariance,
                        dt,
                        orbit.mu,
                        self.config.max_iter,
                        self.config.tol,
                    );
                    states.push(state);
                    covariance_rows.push(covariance_row);
                    covariance_validity.push(true);
                }
                let validity = states
                    .iter()
                    .zip(covariance_rows.iter())
                    .map(|(state, covariance_row)| {
                        state_is_finite(state)
                            && (input_has_nan
                                || covariance_row.iter().all(|value| value.is_finite()))
                    })
                    .collect::<Vec<_>>();
                let messages = validity_messages(&validity);
                return Ok(RowOutput {
                    states,
                    covariance: Some(covariance_rows),
                    covariance_validity: Some(covariance_validity),
                    validity,
                    messages,
                });
            }
            Some(_) => {
                let states = propagate_2body_along_arc(
                    orbit.state,
                    &dts,
                    orbit.mu,
                    self.config.max_iter,
                    self.config.tol,
                );
                let covariance_rows = vec![[0.0_f64; 36]; dts.len()];
                let covariance_validity = vec![false; dts.len()];
                (states, Some(covariance_rows), Some(covariance_validity))
            }
            None => {
                let states = propagate_2body_along_arc(
                    orbit.state,
                    &dts,
                    orbit.mu,
                    self.config.max_iter,
                    self.config.tol,
                );
                (states, None, None)
            }
        };

        let validity = states.iter().map(state_is_finite).collect::<Vec<_>>();
        let messages = validity_messages(&validity);
        Ok(RowOutput {
            states,
            covariance,
            covariance_validity,
            validity,
            messages,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
struct OrbitBlock {
    orbit_index: usize,
    time_indices: Vec<usize>,
    states: Vec<[f64; 6]>,
    covariance: Option<Vec<[f64; 36]>>,
    covariance_validity: Option<Vec<bool>>,
    validity: Vec<bool>,
    messages: Vec<Option<String>>,
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

fn normalized_chunk_size(chunk_size: Option<usize>, rows: usize) -> usize {
    chunk_size.unwrap_or(1).max(1).min(rows.max(1))
}

fn time_indices_for_policy(
    policy: &EpochPolicy,
    orbit_index: usize,
    times_len: usize,
) -> Vec<usize> {
    match policy {
        EpochPolicy::CrossProduct => (0..times_len).collect(),
        EpochPolicy::Pairwise => vec![orbit_index],
        EpochPolicy::PerOrbit { .. } => Vec::new(),
    }
}

fn orbit_row<'a>(
    orbits: &'a OrbitBatch,
    states: &'a [[f64; 6]],
    orbit_times: &'a TimeArray,
    covariance: Option<&'a CovarianceBatch>,
    mus: &[f64],
    orbit_index: usize,
    include_covariance: bool,
) -> OrbitRow<'a> {
    let covariance_values = if include_covariance {
        covariance.map(|covariance| covariance.row_values(orbit_index))
    } else {
        None
    };
    let covariance_valid = covariance
        .as_ref()
        .is_none_or(|covariance| covariance.is_row_valid(orbit_index));
    OrbitRow {
        index: orbit_index,
        orbit_id: &orbits.orbit_id[orbit_index],
        object_id: orbits.object_id[orbit_index].as_ref(),
        state: states[orbit_index],
        origin: &orbits.coordinates.origins.origins[orbit_index],
        mu: mus[orbit_index],
        time: orbit_times.epochs[orbit_index],
        covariance: covariance_values,
        covariance_valid,
    }
}

fn assemble_result(
    request: &PropagationRequest<'_>,
    target_times: &TimeArray,
    blocks: Vec<OrbitBlock>,
) -> PropagationResultValue<PropagationResult> {
    let output_rows = blocks.iter().map(|block| block.states.len()).sum::<usize>();
    let mut orbit_ids = Vec::with_capacity(output_rows);
    let mut object_ids = Vec::with_capacity(output_rows);
    let mut states = Vec::with_capacity(output_rows);
    let mut origins = Vec::with_capacity(output_rows);
    let mut epochs = Vec::with_capacity(output_rows);
    let mut validity = Vec::with_capacity(output_rows);
    let mut diagnostics = Vec::with_capacity(output_rows);

    let output_has_covariance = request.options.covariance == CovariancePropagation::Linearized
        && request.orbits.coordinates.covariance.is_some();
    let mut covariance_values = if output_has_covariance {
        Some(Vec::with_capacity(output_rows * 36))
    } else {
        None
    };
    let mut covariance_validity = if output_has_covariance {
        Some(Vec::with_capacity(output_rows))
    } else {
        None
    };

    for block in blocks {
        for row_offset in 0..block.states.len() {
            let output_row = states.len();
            let time_index = block.time_indices[row_offset];
            let row_valid = block.validity[row_offset];
            orbit_ids.push(request.orbits.orbit_id[block.orbit_index].clone());
            object_ids.push(request.orbits.object_id[block.orbit_index].clone());
            states.push(block.states[row_offset]);
            origins.push(request.orbits.coordinates.origins.origins[block.orbit_index].clone());
            epochs.push(target_times.epochs[time_index]);
            validity.push(row_valid);
            diagnostics.push(PropagationConvergence {
                output_row,
                input_orbit_index: block.orbit_index,
                input_time_index: time_index,
                status: if row_valid {
                    PropagationConvergenceStatus::Converged
                } else {
                    PropagationConvergenceStatus::Failed
                },
                message: block.messages[row_offset].clone(),
            });
            if let Some(values) = &mut covariance_values {
                let covariance_rows = block.covariance.as_ref().ok_or_else(|| {
                    PropagationError::InvalidRequest(
                        "missing covariance rows in propagator output".to_string(),
                    )
                })?;
                values.extend_from_slice(&covariance_rows[row_offset]);
            }
            if let Some(validity_values) = &mut covariance_validity {
                let row_validity = block.covariance_validity.as_ref().ok_or_else(|| {
                    PropagationError::InvalidRequest(
                        "missing covariance validity in propagator output".to_string(),
                    )
                })?;
                validity_values.push(row_validity[row_offset]);
            }
        }
    }

    let times = TimeArray::new(target_times.scale, epochs)?;
    let covariance = build_output_covariance(
        request.orbits.coordinates.covariance.as_ref(),
        covariance_values,
        covariance_validity,
        output_rows,
    )?;
    let coordinates = CoordinateBatch::cartesian(
        states,
        request.orbits.coordinates.frame,
        OriginArray::new(origins),
        Some(times.clone()),
        covariance,
    )?;
    let orbits = OrbitBatch::new(orbit_ids, object_ids, coordinates)?;
    Ok(PropagationResult {
        orbits,
        times,
        validity: Validity::from_bools(&validity),
        diagnostics: PropagationDiagnostics {
            convergence: diagnostics,
            epoch_order: request.epoch_order.clone(),
        },
    })
}

fn build_output_covariance(
    input: Option<&CovarianceBatch>,
    values: Option<Vec<f64>>,
    row_validity: Option<Vec<bool>>,
    rows: usize,
) -> PropagationResultValue<Option<CovarianceBatch>> {
    let Some(input_covariance) = input else {
        return Ok(None);
    };
    let Some(values) = values else {
        return Ok(None);
    };
    let units = match &input_covariance.units {
        CovarianceUnits::Coordinate(representation) => CovarianceUnits::Coordinate(*representation),
        CovarianceUnits::ObservationAngular2D => CovarianceUnits::ObservationAngular2D,
        CovarianceUnits::Photometry1D => CovarianceUnits::Photometry1D,
        CovarianceUnits::Custom(units) => CovarianceUnits::Custom(units.clone()),
    };
    let covariance = CovarianceBatch::new(rows, 6, values, units)?;
    match row_validity {
        Some(validity) if validity.iter().any(|value| !*value) => Ok(Some(
            covariance.with_row_validity(Validity::from_bools(&validity))?,
        )),
        _ => Ok(Some(covariance)),
    }
}

fn covariance_36(values: &[f64]) -> PropagationResultValue<&[f64; 36]> {
    values.try_into().map_err(|_| {
        PropagationError::InvalidRequest(format!(
            "expected 36 covariance values for a Cartesian row, got {}",
            values.len()
        ))
    })
}

fn state_is_finite(state: &[f64; 6]) -> bool {
    state.iter().all(|value| value.is_finite())
}

fn validity_messages(validity: &[bool]) -> Vec<Option<String>> {
    validity
        .iter()
        .map(|valid| {
            if *valid {
                None
            } else {
                Some("two-body propagation produced a non-finite row".to_string())
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Frame, SchemaResult};
    use crate::{CoordinateRepresentation, CovarianceUnits, NANOS_PER_DAY};

    struct NoopProvider;

    impl TimeScaleProvider for NoopProvider {
        fn rescale(&self, _times: &TimeArray, _new_scale: TimeScale) -> SchemaResult<TimeArray> {
            Err(SchemaError::InvalidRecordBatch(
                "test provider should not be called".to_string(),
            ))
        }
    }

    struct WrongScaleProvider;

    impl TimeScaleProvider for WrongScaleProvider {
        fn rescale(&self, times: &TimeArray, _new_scale: TimeScale) -> SchemaResult<TimeArray> {
            TimeArray::new(TimeScale::Utc, times.epochs.clone())
        }
    }

    fn tdb_times() -> TimeArray {
        TimeArray::from_parts(
            TimeScale::Tdb,
            vec![60_000, 60_010],
            vec![0, NANOS_PER_DAY / 2],
        )
        .unwrap()
    }

    fn ut1_times() -> TimeArray {
        TimeArray::from_parts(TimeScale::Ut1, vec![60_000, 60_010], vec![0, 0]).unwrap()
    }

    fn sample_orbits(covariance: Option<CovarianceBatch>) -> OrbitBatch {
        let times =
            TimeArray::from_parts(TimeScale::Tdb, vec![60_000, 60_001], vec![0, 0]).unwrap();
        let coordinates = CoordinateBatch::cartesian(
            vec![
                [1.0, 0.2, 0.1, 0.001, 0.015, 0.0005],
                [1.2, -0.1, 0.05, -0.002, 0.014, 0.0002],
            ],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::Named("SUN".to_string()), 2),
            Some(times),
            covariance,
        )
        .unwrap();
        OrbitBatch::new(
            vec![
                OrbitId("orbit-a".to_string()),
                OrbitId("orbit-b".to_string()),
            ],
            vec![Some(ObjectId("object-a".to_string())), None],
            coordinates,
        )
        .unwrap()
    }

    #[test]
    fn request_records_epoch_permutation() {
        let orbits = sample_orbits(None);
        let times =
            TimeArray::from_parts(TimeScale::Tdb, vec![60_010, 60_000], vec![0, 0]).unwrap();
        let request =
            PropagationRequest::new(&orbits, &times, PropagationOptions::default()).unwrap();
        assert!(!request.epoch_order.is_chronological);
        assert_eq!(request.epoch_order.sorted_to_input, vec![1, 0]);
        assert_eq!(request.epoch_order.input_to_sorted, vec![1, 0]);
        assert_eq!(request.output_len(), 4);
    }

    #[test]
    fn pairwise_policy_requires_matching_lengths() {
        let orbits = sample_orbits(None);
        let times = TimeArray::from_parts(TimeScale::Tdb, vec![60_010], vec![0]).unwrap();
        let options = PropagationOptions {
            epoch_policy: EpochPolicy::Pairwise,
            ..PropagationOptions::default()
        };
        let err = PropagationRequest::new(&orbits, &times, options).unwrap_err();
        assert!(matches!(err, PropagationError::InvalidRequest(_)));
    }

    #[test]
    fn two_body_cross_product_matches_low_level_kernel() {
        let orbits = sample_orbits(None);
        let times = tdb_times();
        let options = PropagationOptions {
            covariance: CovariancePropagation::None,
            chunk_size: Some(1),
            thread_limit: Some(1),
            ..PropagationOptions::default()
        };
        let request = PropagationRequest::new(&orbits, &times, options).unwrap();
        let propagator = TwoBodyPropagator::default();
        let result = propagator.propagate(&request, &NoopProvider).unwrap();
        assert_eq!(result.orbits.len(), 4);
        assert_eq!(result.times.scale, TimeScale::Tdb);
        assert_eq!(result.orbits.orbit_id[0].0, "orbit-a");
        assert_eq!(result.orbits.orbit_id[1].0, "orbit-a");
        assert_eq!(result.orbits.orbit_id[2].0, "orbit-b");
        assert_eq!(
            result.orbits.coordinates.times.as_ref().unwrap(),
            &result.times
        );
        for row in 0..result.validity.len() {
            assert!(result.validity.is_valid(row));
        }

        let states = result.orbits.coordinates.values.cartesian().unwrap();
        let mu = origin_mu_au3_day2(&OriginId::Named("SUN".to_string())).unwrap();
        let expected = propagate_2body_along_arc(
            [1.0, 0.2, 0.1, 0.001, 0.015, 0.0005],
            &[0.0, 10.5],
            mu,
            DEFAULT_TWO_BODY_MAX_ITER,
            DEFAULT_TWO_BODY_TOL,
        );
        for (actual, expected) in states.iter().take(2).zip(expected.iter()) {
            for column in 0..6 {
                assert!((actual[column] - expected[column]).abs() < 1.0e-13);
            }
        }
    }

    #[test]
    fn two_body_pairwise_outputs_one_row_per_orbit() {
        let orbits = sample_orbits(None);
        let times = tdb_times();
        let options = PropagationOptions {
            epoch_policy: EpochPolicy::Pairwise,
            covariance: CovariancePropagation::None,
            ..PropagationOptions::default()
        };
        let request = PropagationRequest::new(&orbits, &times, options).unwrap();
        let result = TwoBodyPropagator::default()
            .propagate(&request, &NoopProvider)
            .unwrap();
        assert_eq!(result.orbits.len(), 2);
        assert_eq!(result.orbits.orbit_id[0].0, "orbit-a");
        assert_eq!(result.orbits.orbit_id[1].0, "orbit-b");
        assert_eq!(result.times.epochs[0], times.epochs[0]);
        assert_eq!(result.times.epochs[1], times.epochs[1]);
    }

    #[test]
    fn covariance_linearized_propagates_and_preserves_null_rows() {
        let mut covariance_values = vec![0.0; 72];
        for row in 0..2 {
            for diag in 0..6 {
                covariance_values[row * 36 + diag * 6 + diag] = 1.0e-12;
            }
        }
        let covariance = CovarianceBatch::new(
            2,
            6,
            covariance_values,
            CovarianceUnits::Coordinate(CoordinateRepresentation::Cartesian),
        )
        .unwrap()
        .with_row_validity(Validity::from_bools(&[true, false]))
        .unwrap();
        let orbits = sample_orbits(Some(covariance));
        let times = tdb_times();
        let request =
            PropagationRequest::new(&orbits, &times, PropagationOptions::default()).unwrap();
        let result = TwoBodyPropagator::default()
            .propagate(&request, &NoopProvider)
            .unwrap();
        let output_covariance = result.orbits.coordinates.covariance.as_ref().unwrap();
        assert_eq!(output_covariance.rows, 4);
        assert!(output_covariance.is_row_valid(0));
        assert!(output_covariance.is_row_valid(1));
        assert!(!output_covariance.is_row_valid(2));
        assert!(!output_covariance.is_row_valid(3));
        assert!(output_covariance
            .row_values(0)
            .iter()
            .any(|value| *value != 0.0));
    }

    #[test]
    fn unsupported_covariance_mode_fails_as_setup_error() {
        let orbits = sample_orbits(None);
        let times = tdb_times();
        let options = PropagationOptions {
            covariance: CovariancePropagation::Monte {
                samples: 16,
                seed: 7,
            },
            ..PropagationOptions::default()
        };
        let request = PropagationRequest::new(&orbits, &times, options).unwrap();
        let propagator = TwoBodyPropagator::default();
        assert!(!propagator.supports(request.options.covariance));
        let err = propagator.propagate(&request, &NoopProvider).unwrap_err();
        assert!(matches!(
            err,
            PropagationError::UnsupportedCovarianceMode(_)
        ));
    }

    #[test]
    fn nonfinite_state_is_a_row_failure_not_setup_error() {
        let times = TimeArray::from_parts(TimeScale::Tdb, vec![60_000], vec![0]).unwrap();
        let coordinates = CoordinateBatch::cartesian(
            vec![[f64::NAN, 0.2, 0.1, 0.001, 0.015, 0.0005]],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::Named("SUN".to_string()), 1),
            Some(times.clone()),
            None,
        )
        .unwrap();
        let orbits =
            OrbitBatch::new(vec![OrbitId("bad".to_string())], vec![None], coordinates).unwrap();
        let request =
            PropagationRequest::new(&orbits, &times, PropagationOptions::default()).unwrap();
        let result = TwoBodyPropagator::default()
            .propagate(&request, &NoopProvider)
            .unwrap();
        assert!(!result.validity.is_valid(0));
        let failures = result.diagnostics.failed_rows().collect::<Vec<_>>();
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].input_orbit_index, 0);
    }

    #[test]
    fn provider_required_time_scale_errors_are_setup_errors() {
        let orbits = sample_orbits(None);
        let options = PropagationOptions {
            covariance: CovariancePropagation::None,
            ..PropagationOptions::default()
        };
        let times = ut1_times();
        let request = PropagationRequest::new(&orbits, &times, options).unwrap();
        let err = TwoBodyPropagator::default()
            .propagate(&request, &NoopProvider)
            .unwrap_err();
        assert!(matches!(err, PropagationError::Schema(_)));
    }

    #[test]
    fn provider_must_return_integration_scale() {
        let orbits = sample_orbits(None);
        let options = PropagationOptions {
            covariance: CovariancePropagation::None,
            ..PropagationOptions::default()
        };
        let times = ut1_times();
        let request = PropagationRequest::new(&orbits, &times, options).unwrap();
        let err = TwoBodyPropagator::default()
            .propagate(&request, &WrongScaleProvider)
            .unwrap_err();
        assert!(
            matches!(err, PropagationError::InvalidRequest(message) if message.contains("required tdb"))
        );
    }

    #[test]
    fn unsupported_origin_fails_loudly() {
        let times = TimeArray::from_parts(TimeScale::Tdb, vec![60_000], vec![0]).unwrap();
        let coordinates = CoordinateBatch::cartesian(
            vec![[1.0, 0.2, 0.1, 0.001, 0.015, 0.0005]],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::Named("UNKNOWN".to_string()), 1),
            Some(times.clone()),
            None,
        )
        .unwrap();
        let orbits = OrbitBatch::new(
            vec![OrbitId("bad-origin".to_string())],
            vec![None],
            coordinates,
        )
        .unwrap();
        let request =
            PropagationRequest::new(&orbits, &times, PropagationOptions::default()).unwrap();
        let err = TwoBodyPropagator::default()
            .propagate(&request, &NoopProvider)
            .unwrap_err();
        assert!(
            matches!(err, PropagationError::Schema(SchemaError::UnsupportedOrigin(origin)) if origin == "UNKNOWN")
        );
    }
}
