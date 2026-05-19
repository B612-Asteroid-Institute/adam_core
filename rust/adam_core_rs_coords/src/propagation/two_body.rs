use super::diagnostics::{failure_message, failure_messages, PropagationFailureCode, RowOutput};
use super::pipeline::{
    assemble_result, normalized_chunk_size, orbit_row, time_indices_for_policy, OrbitBlock,
    OrbitRow, PropagationResult,
};
use super::request::{CovariancePropagation, PropagationRequest};
use super::{
    rescale_for_propagation, run_with_thread_limit, PropagationError, PropagationResultValue,
    Propagator, PropagatorShard,
};
use crate::propagate::{
    propagate_2body_along_arc_with_diagnostics,
    propagate_2body_with_covariance_row_with_diagnostics, ChiConvergence, ChiConvergenceStatus,
};
use crate::types::origin_mu_au3_day2;
use crate::types::time::TimeScaleProvider;
use crate::{Epoch, TimeArray, TimeScale};
use rayon::prelude::*;

pub(crate) const DEFAULT_TWO_BODY_MAX_ITER: usize = 1000;
pub(crate) const DEFAULT_TWO_BODY_TOL: f64 = 1.0e-14;

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
        let input_coordinates = request.input.coordinates();
        let states = input_coordinates.values.cartesian().ok_or_else(|| {
            PropagationError::InvalidRequest(
                "TwoBodyPropagator requires Cartesian orbit coordinates".to_string(),
            )
        })?;
        let covariance = input_coordinates.covariance.as_ref();
        let include_covariance =
            request.options.covariance == CovariancePropagation::Linearized && covariance.is_some();
        let orbit_indices = (0..request.input.len()).collect::<Vec<_>>();
        let chunk_size = normalized_chunk_size(request.options.chunk_size, request.input.len());
        let policy = request.options.epoch_policy.clone();

        let chunk_results = orbit_indices
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut shard = self.create_shard();
                let mut blocks = Vec::with_capacity(chunk.len());
                for &orbit_index in chunk {
                    let row = orbit_row(
                        request.input,
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
                        || output.iterations.len() != time_indices.len()
                        || output.failure_codes.len() != time_indices.len()
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
                        backend: output.backend,
                        iterations: output.iterations,
                        failure_codes: output.failure_codes,
                    });
                }
                Ok(blocks)
            })
            .collect::<Vec<_>>();

        let mut blocks = Vec::with_capacity(request.input.len());
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
        request.input.validate()?;
        request.times.validate()?;
        request.options.validate()?;
        if !self.supports(request.options.covariance) {
            return Err(PropagationError::UnsupportedCovarianceMode(
                request.options.covariance,
            ));
        }
        let input_orbit_times = request
            .input
            .coordinates()
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
            .input
            .coordinates()
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
        let covariance_requested = orbit.covariance.is_some();

        if !state_is_finite(&orbit.state) || !orbit.mu.is_finite() {
            return Ok(row_failure_output(
                dts.len(),
                covariance_requested,
                PropagationFailureCode::NonFiniteInputState,
            ));
        }

        let (states, covariance, covariance_validity, convergence, input_has_nan_covariance) =
            match orbit.covariance {
                Some(covariance) if orbit.covariance_valid => {
                    let input_covariance = covariance_36(covariance)?;
                    let input_has_nan = input_covariance.iter().any(|value| value.is_nan());
                    let mut states = Vec::with_capacity(dts.len());
                    let mut covariance_rows = Vec::with_capacity(dts.len());
                    let mut covariance_validity = Vec::with_capacity(dts.len());
                    let mut convergence = Vec::with_capacity(dts.len());
                    for &dt in &dts {
                        let (state, covariance_row, row_convergence) =
                            propagate_2body_with_covariance_row_with_diagnostics(
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
                        convergence.push(row_convergence);
                    }
                    (
                        states,
                        Some(covariance_rows),
                        Some(covariance_validity),
                        convergence,
                        input_has_nan,
                    )
                }
                Some(_) => {
                    let (states, convergence) = propagate_2body_along_arc_with_diagnostics(
                        orbit.state,
                        &dts,
                        orbit.mu,
                        self.config.max_iter,
                        self.config.tol,
                    );
                    let covariance_rows = vec![[0.0_f64; 36]; dts.len()];
                    let covariance_validity = vec![false; dts.len()];
                    (
                        states,
                        Some(covariance_rows),
                        Some(covariance_validity),
                        convergence,
                        false,
                    )
                }
                None => {
                    let (states, convergence) = propagate_2body_along_arc_with_diagnostics(
                        orbit.state,
                        &dts,
                        orbit.mu,
                        self.config.max_iter,
                        self.config.tol,
                    );
                    (states, None, None, convergence, false)
                }
            };

        let failure_codes = states
            .iter()
            .enumerate()
            .map(|(index, state)| {
                let covariance_row = covariance.as_ref().map(|rows| &rows[index]);
                two_body_failure_code(
                    state,
                    covariance_row,
                    input_has_nan_covariance,
                    convergence[index],
                )
            })
            .collect::<Vec<_>>();
        let validity = failure_codes
            .iter()
            .map(Option::is_none)
            .collect::<Vec<_>>();
        let messages = failure_messages(&failure_codes);
        let iterations = convergence
            .iter()
            .map(|row| Some(row.iterations))
            .collect::<Vec<_>>();
        Ok(RowOutput {
            states,
            covariance,
            covariance_validity,
            validity,
            messages,
            backend: two_body_backend(),
            iterations,
            failure_codes,
        })
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

fn two_body_backend() -> Option<String> {
    Some("two_body".to_string())
}

fn row_failure_output(
    rows: usize,
    covariance_requested: bool,
    failure_code: PropagationFailureCode,
) -> RowOutput {
    RowOutput {
        states: vec![[f64::NAN; 6]; rows],
        covariance: covariance_requested.then(|| vec![[0.0_f64; 36]; rows]),
        covariance_validity: covariance_requested.then(|| vec![false; rows]),
        validity: vec![false; rows],
        messages: vec![Some(failure_message(failure_code)); rows],
        backend: two_body_backend(),
        iterations: vec![None; rows],
        failure_codes: vec![Some(failure_code); rows],
    }
}

fn two_body_failure_code(
    state: &[f64; 6],
    covariance: Option<&[f64; 36]>,
    input_has_nan_covariance: bool,
    convergence: ChiConvergence,
) -> Option<PropagationFailureCode> {
    match convergence.status {
        ChiConvergenceStatus::Converged => {}
        ChiConvergenceStatus::ZeroDerivative => {
            return Some(PropagationFailureCode::SolverZeroDerivative);
        }
        ChiConvergenceStatus::MaxIterations => {
            return Some(PropagationFailureCode::SolverMaxIterations);
        }
    }
    if !state_is_finite(state) {
        return Some(PropagationFailureCode::NonFiniteOutputState);
    }
    if let Some(covariance) = covariance {
        if !input_has_nan_covariance && covariance.iter().any(|value| !value.is_finite()) {
            return Some(PropagationFailureCode::NonFiniteCovariance);
        }
    }
    None
}
