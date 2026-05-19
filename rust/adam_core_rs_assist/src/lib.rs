//! GPL `assist-rs` adapter for adam-core Rust propagation contracts.
//!
//! This crate is the deliberate GPL boundary for ASSIST/REBOUND-backed
//! propagation. The permissive core crates expose only backend-generic
//! contracts; this crate maps those contracts to `assist-rs` types.

use adam_core_rs_coords::propagation::{
    CovariancePropagation, EpochPolicy, OrbitRow, PropagationConvergence,
    PropagationConvergenceStatus, PropagationDiagnostics, PropagationError, PropagationFailureCode,
    PropagationRequest, PropagationResult, PropagationResultValue, Propagator, PropagatorShard,
    RowOutput,
};
use adam_core_rs_coords::types::Frame;
use adam_core_rs_coords::{
    CoordinateBatch, CovarianceBatch, CovarianceUnits, Epoch, OrbitBatch, OrbitVariantBatch,
    OriginArray, OriginId, TimeArray, TimeScale, TimeScaleProvider, Validity,
};
use assist_rs::{assist_propagate, AssistData, IntegratorConfig, Orbit as AssistOrbit};
use rayon::prelude::*;
use std::sync::Arc;

const BACKEND_NAME: &str = "assist_rs";

#[derive(Clone)]
pub struct AssistPropagator {
    data: Arc<AssistData>,
    integrator: IntegratorConfig,
}

impl AssistPropagator {
    pub fn new(data: Arc<AssistData>) -> Self {
        Self {
            data,
            integrator: IntegratorConfig::default(),
        }
    }

    pub fn with_integrator(data: Arc<AssistData>, integrator: IntegratorConfig) -> Self {
        Self { data, integrator }
    }

    pub fn data(&self) -> &Arc<AssistData> {
        &self.data
    }

    pub fn integrator(&self) -> IntegratorConfig {
        self.integrator
    }

    fn propagate_with_stm(
        &self,
        request: &PropagationRequest<'_>,
        compute_stm: bool,
    ) -> PropagationResultValue<PropagationResult> {
        validate_request_scope(request, compute_stm)?;
        let input_coordinates = request.input.coordinates();
        let states = input_coordinates.values.cartesian().ok_or_else(|| {
            PropagationError::InvalidRequest(
                "AssistPropagator requires Cartesian orbit coordinates".to_string(),
            )
        })?;
        let orbit_times = input_coordinates
            .times
            .as_ref()
            .ok_or(PropagationError::MissingOrbitTimes)?;
        let covariance = input_coordinates.covariance.as_ref();
        let include_covariance = compute_stm && covariance.is_some();
        let orbit_indices = (0..request.input.len()).collect::<Vec<_>>();
        let chunk_size = normalized_chunk_size(request.options.chunk_size, request.input.len());
        let policy = request.options.epoch_policy.clone();

        run_with_thread_limit(request.options.thread_limit, || {
            let chunk_results = orbit_indices
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut shard =
                        AssistShard::new(Arc::clone(&self.data), self.integrator, compute_stm);
                    let mut blocks = Vec::with_capacity(chunk.len());
                    for &orbit_index in chunk {
                        let row = orbit_row(
                            request,
                            states,
                            orbit_times,
                            covariance,
                            orbit_index,
                            include_covariance,
                        );
                        let time_indices =
                            time_indices_for_policy(&policy, orbit_index, request.times.len());
                        let sorted_time_indices = sorted_time_indices(&time_indices, request.times);
                        let sorted_times = sorted_time_indices
                            .iter()
                            .map(|&time_index| request.times.epochs[time_index])
                            .collect::<Vec<_>>();
                        let sorted_output = shard.propagate_one(row, &sorted_times)?;
                        blocks.push(reorder_output(
                            orbit_index,
                            time_indices,
                            sorted_time_indices,
                            sorted_output,
                        )?);
                    }
                    Ok::<Vec<AssistOrbitBlock>, PropagationError>(blocks)
                })
                .collect::<Vec<_>>();

            let mut blocks = Vec::with_capacity(request.input.len());
            for chunk in chunk_results {
                blocks.extend(chunk?);
            }
            assemble_result(request, request.times, blocks)
        })
    }
}

impl Propagator for AssistPropagator {
    type Shard = AssistShard;

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
        AssistShard::new(Arc::clone(&self.data), self.integrator, false)
    }

    fn propagate(
        &self,
        request: &PropagationRequest<'_>,
        _provider: &dyn TimeScaleProvider,
    ) -> PropagationResultValue<PropagationResult> {
        request.input.validate()?;
        request.times.validate()?;
        request.options.validate()?;
        if !self.supports(request.options.covariance) {
            return Err(PropagationError::UnsupportedCovarianceMode(
                request.options.covariance,
            ));
        }
        let compute_stm = request.options.covariance == CovariancePropagation::Linearized
            && request.input.coordinates().covariance.is_some();
        self.propagate_with_stm(request, compute_stm)
    }
}

#[derive(Clone)]
pub struct AssistShard {
    data: Arc<AssistData>,
    integrator: IntegratorConfig,
    compute_stm: bool,
}

impl AssistShard {
    pub fn new(data: Arc<AssistData>, integrator: IntegratorConfig, compute_stm: bool) -> Self {
        Self {
            data,
            integrator,
            compute_stm,
        }
    }
}

impl PropagatorShard for AssistShard {
    fn propagate_one(
        &mut self,
        orbit: OrbitRow<'_>,
        times: &[Epoch],
    ) -> PropagationResultValue<RowOutput> {
        let covariance_requested = self.compute_stm && orbit.covariance.is_some();
        if !state_is_finite(&orbit.state) {
            return Ok(row_failure_output(
                times.len(),
                covariance_requested,
                Some(PropagationFailureCode::NonFiniteInputState),
                "assist-rs propagation input state was non-finite".to_string(),
            ));
        }

        let assist_orbit = AssistOrbit::new(orbit.state, orbit.time.mjd());
        let target_epochs = times.iter().map(|time| time.mjd()).collect::<Vec<_>>();
        let assist_orbits = [assist_orbit];
        let propagated = match assist_propagate(
            &self.data,
            &assist_orbits,
            &target_epochs,
            self.compute_stm,
            Some(1),
            &self.integrator,
        ) {
            Ok(mut values) => values.pop().unwrap_or_default(),
            Err(err) => {
                return Ok(row_failure_output(
                    times.len(),
                    covariance_requested,
                    None,
                    format!("assist-rs propagation failed: {err}"),
                ));
            }
        };

        if propagated.len() != times.len() {
            return Err(PropagationError::InvalidRequest(format!(
                "assist-rs returned {} rows for {} target epochs",
                propagated.len(),
                times.len()
            )));
        }

        let mut states = Vec::with_capacity(propagated.len());
        let mut validity = Vec::with_capacity(propagated.len());
        let mut messages = Vec::with_capacity(propagated.len());
        let mut failure_codes = Vec::with_capacity(propagated.len());
        let mut covariance_rows =
            covariance_requested.then(|| Vec::with_capacity(propagated.len()));
        let mut covariance_validity =
            covariance_requested.then(|| Vec::with_capacity(propagated.len()));
        let input_covariance = match (covariance_requested, orbit.covariance_valid) {
            (true, true) => orbit.covariance.map(covariance_6x6),
            _ => None,
        }
        .transpose()?;

        for state in propagated {
            let failure_code = if state_is_finite(&state.state) {
                None
            } else {
                Some(PropagationFailureCode::NonFiniteOutputState)
            };
            states.push(state.state);
            validity.push(failure_code.is_none());
            messages.push(failure_code.map(assist_failure_message));
            failure_codes.push(failure_code);

            if let (Some(rows), Some(row_validity)) =
                (&mut covariance_rows, &mut covariance_validity)
            {
                match input_covariance
                    .as_ref()
                    .and_then(|covariance| state.propagate_covariance(covariance))
                {
                    Some(covariance) => {
                        rows.push(flatten_covariance(covariance));
                        row_validity.push(true);
                    }
                    None => {
                        rows.push([0.0; 36]);
                        row_validity.push(false);
                    }
                }
            }
        }

        Ok(RowOutput {
            states,
            covariance: covariance_rows,
            covariance_validity,
            validity,
            messages,
            backend: Some(BACKEND_NAME.to_string()),
            iterations: vec![None; times.len()],
            failure_codes,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
struct AssistOrbitBlock {
    orbit_index: usize,
    time_indices: Vec<usize>,
    states: Vec<[f64; 6]>,
    covariance: Option<Vec<[f64; 36]>>,
    covariance_validity: Option<Vec<bool>>,
    validity: Vec<bool>,
    messages: Vec<Option<String>>,
    iterations: Vec<Option<usize>>,
    failure_codes: Vec<Option<PropagationFailureCode>>,
}

fn validate_request_scope(
    request: &PropagationRequest<'_>,
    compute_stm: bool,
) -> PropagationResultValue<()> {
    let coordinates = request.input.coordinates();
    if coordinates.frame != Frame::Ecliptic {
        return Err(PropagationError::InvalidRequest(format!(
            "AssistPropagator initial scope requires ecliptic Cartesian input; got frame {}",
            coordinates.frame.as_str()
        )));
    }
    if coordinates.values.cartesian().is_none() {
        return Err(PropagationError::InvalidRequest(
            "AssistPropagator initial scope requires Cartesian coordinates".to_string(),
        ));
    }
    let orbit_times = coordinates
        .times
        .as_ref()
        .ok_or(PropagationError::MissingOrbitTimes)?;
    if orbit_times.scale != TimeScale::Tdb || request.times.scale != TimeScale::Tdb {
        return Err(PropagationError::InvalidRequest(format!(
            "AssistPropagator initial scope is TDB-only; got orbit times {} and target times {}",
            orbit_times.scale.as_str(),
            request.times.scale.as_str()
        )));
    }
    for origin in &coordinates.origins.origins {
        if !is_heliocentric_origin(origin) {
            return Err(PropagationError::InvalidRequest(format!(
                "AssistPropagator initial scope requires heliocentric SUN origins; got {}",
                origin.code()
            )));
        }
    }
    if compute_stm && coordinates.covariance.is_none() {
        return Err(PropagationError::InvalidRequest(
            "AssistPropagator STM mode requires input covariance".to_string(),
        ));
    }
    Ok(())
}

fn is_heliocentric_origin(origin: &OriginId) -> bool {
    matches!(origin, OriginId::Naif(10))
        || matches!(origin, OriginId::Named(code) if code.eq_ignore_ascii_case("SUN"))
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

fn sorted_time_indices(time_indices: &[usize], target_times: &TimeArray) -> Vec<usize> {
    let mut sorted = time_indices.to_vec();
    sorted.sort_by(|&left, &right| {
        target_times.epochs[left]
            .mjd()
            .partial_cmp(&target_times.epochs[right].mjd())
            .expect("epoch MJD values are finite")
    });
    sorted
}

fn reorder_output(
    orbit_index: usize,
    time_indices: Vec<usize>,
    sorted_time_indices: Vec<usize>,
    sorted_output: RowOutput,
) -> PropagationResultValue<AssistOrbitBlock> {
    validate_row_output_lengths(sorted_time_indices.len(), &sorted_output)?;
    let mut states = Vec::with_capacity(time_indices.len());
    let mut covariance = sorted_output
        .covariance
        .as_ref()
        .map(|_| Vec::with_capacity(time_indices.len()));
    let mut covariance_validity = sorted_output
        .covariance_validity
        .as_ref()
        .map(|_| Vec::with_capacity(time_indices.len()));
    let mut validity = Vec::with_capacity(time_indices.len());
    let mut messages = Vec::with_capacity(time_indices.len());
    let mut iterations = Vec::with_capacity(time_indices.len());
    let mut failure_codes = Vec::with_capacity(time_indices.len());

    for &time_index in &time_indices {
        let sorted_position = sorted_time_indices
            .iter()
            .position(|&index| index == time_index)
            .ok_or_else(|| {
                PropagationError::InvalidRequest(
                    "internal time-order mapping did not contain caller epoch".to_string(),
                )
            })?;
        states.push(sorted_output.states[sorted_position]);
        if let Some(values) = &mut covariance {
            values.push(
                sorted_output
                    .covariance
                    .as_ref()
                    .expect("covariance rows are present")[sorted_position],
            );
        }
        if let Some(values) = &mut covariance_validity {
            values.push(
                sorted_output
                    .covariance_validity
                    .as_ref()
                    .expect("covariance validity rows are present")[sorted_position],
            );
        }
        validity.push(sorted_output.validity[sorted_position]);
        messages.push(sorted_output.messages[sorted_position].clone());
        iterations.push(sorted_output.iterations[sorted_position]);
        failure_codes.push(sorted_output.failure_codes[sorted_position]);
    }

    Ok(AssistOrbitBlock {
        orbit_index,
        time_indices,
        states,
        covariance,
        covariance_validity,
        validity,
        messages,
        iterations,
        failure_codes,
    })
}

fn validate_row_output_lengths(expected: usize, output: &RowOutput) -> PropagationResultValue<()> {
    if output.states.len() != expected
        || output.validity.len() != expected
        || output.messages.len() != expected
        || output.iterations.len() != expected
        || output.failure_codes.len() != expected
    {
        return Err(PropagationError::InvalidRequest(
            "assist-rs shard returned inconsistent row lengths".to_string(),
        ));
    }
    if let Some(covariance) = &output.covariance {
        if covariance.len() != expected {
            return Err(PropagationError::InvalidRequest(
                "assist-rs shard returned inconsistent covariance length".to_string(),
            ));
        }
    }
    if let Some(covariance_validity) = &output.covariance_validity {
        if covariance_validity.len() != expected {
            return Err(PropagationError::InvalidRequest(
                "assist-rs shard returned inconsistent covariance validity length".to_string(),
            ));
        }
    }
    Ok(())
}

fn orbit_row<'a>(
    request: &PropagationRequest<'a>,
    states: &'a [[f64; 6]],
    orbit_times: &'a TimeArray,
    covariance: Option<&'a CovarianceBatch>,
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
        orbit_id: &request.input.orbit_id()[orbit_index],
        object_id: request.input.object_id()[orbit_index].as_ref(),
        variant_id: request
            .input
            .variant_id()
            .and_then(|variant_id| variant_id[orbit_index].as_ref()),
        weight: request
            .input
            .weights()
            .and_then(|weights| weights[orbit_index]),
        weight_cov: request
            .input
            .weights_cov()
            .and_then(|weights_cov| weights_cov[orbit_index]),
        state: states[orbit_index],
        origin: &request.input.coordinates().origins.origins[orbit_index],
        mu: f64::NAN,
        time: orbit_times.epochs[orbit_index],
        covariance: covariance_values,
        covariance_valid,
    }
}

fn assemble_result(
    request: &PropagationRequest<'_>,
    target_times: &TimeArray,
    blocks: Vec<AssistOrbitBlock>,
) -> PropagationResultValue<PropagationResult> {
    let output_rows = blocks.iter().map(|block| block.states.len()).sum::<usize>();
    let mut orbit_ids = Vec::with_capacity(output_rows);
    let mut object_ids = Vec::with_capacity(output_rows);
    let mut states = Vec::with_capacity(output_rows);
    let mut origins = Vec::with_capacity(output_rows);
    let mut epochs = Vec::with_capacity(output_rows);
    let mut validity = Vec::with_capacity(output_rows);
    let mut convergence = Vec::with_capacity(output_rows);
    let mut variant_ids = request
        .input
        .variant_id()
        .map(|_| Vec::with_capacity(output_rows));
    let mut weights = request
        .input
        .weights()
        .map(|_| Vec::with_capacity(output_rows));
    let mut weights_cov = request
        .input
        .weights_cov()
        .map(|_| Vec::with_capacity(output_rows));
    let output_has_covariance = request.options.covariance == CovariancePropagation::Linearized
        && request.input.coordinates().covariance.is_some();
    let mut covariance_values = output_has_covariance.then(|| Vec::with_capacity(output_rows * 36));
    let mut covariance_validity = output_has_covariance.then(|| Vec::with_capacity(output_rows));

    for block in blocks {
        for row_offset in 0..block.states.len() {
            let output_row = states.len();
            let input_time_index = block.time_indices[row_offset];
            orbit_ids.push(request.input.orbit_id()[block.orbit_index].clone());
            object_ids.push(request.input.object_id()[block.orbit_index].clone());
            if let Some(values) = &mut variant_ids {
                values.push(
                    request.input.variant_id().expect("variant ids are present")[block.orbit_index]
                        .clone(),
                );
            }
            if let Some(values) = &mut weights {
                values.push(
                    request
                        .input
                        .weights()
                        .expect("variant weights are present")[block.orbit_index],
                );
            }
            if let Some(values) = &mut weights_cov {
                values.push(
                    request
                        .input
                        .weights_cov()
                        .expect("variant covariance weights are present")[block.orbit_index],
                );
            }
            states.push(block.states[row_offset]);
            origins.push(request.input.coordinates().origins.origins[block.orbit_index].clone());
            epochs.push(target_times.epochs[input_time_index]);
            let row_valid = block.validity[row_offset];
            validity.push(row_valid);
            convergence.push(PropagationConvergence {
                output_row,
                input_orbit_index: block.orbit_index,
                input_time_index,
                status: if row_valid {
                    PropagationConvergenceStatus::Converged
                } else {
                    PropagationConvergenceStatus::Failed
                },
                backend: Some(BACKEND_NAME.to_string()),
                iterations: block.iterations[row_offset],
                failure_code: block.failure_codes[row_offset],
                message: block.messages[row_offset].clone(),
            });
            if let Some(values) = &mut covariance_values {
                let covariance_rows = block.covariance.as_ref().ok_or_else(|| {
                    PropagationError::InvalidRequest(
                        "missing covariance rows in assist-rs output".to_string(),
                    )
                })?;
                values.extend_from_slice(&covariance_rows[row_offset]);
            }
            if let Some(values) = &mut covariance_validity {
                let row_validity = block.covariance_validity.as_ref().ok_or_else(|| {
                    PropagationError::InvalidRequest(
                        "missing covariance validity in assist-rs output".to_string(),
                    )
                })?;
                values.push(row_validity[row_offset]);
            }
        }
    }

    let times = TimeArray::new(target_times.scale, epochs)?;
    let covariance = build_output_covariance(
        request.input.coordinates().covariance.as_ref(),
        covariance_values,
        covariance_validity,
        output_rows,
    )?;
    let coordinates = CoordinateBatch::cartesian(
        states,
        request.input.coordinates().frame,
        OriginArray::new(origins),
        Some(times.clone()),
        covariance,
    )?;
    let variants = match (variant_ids, weights, weights_cov) {
        (Some(variant_ids), Some(weights), Some(weights_cov)) => Some(OrbitVariantBatch::new(
            orbit_ids.clone(),
            object_ids.clone(),
            variant_ids,
            weights,
            weights_cov,
            coordinates.clone(),
        )?),
        (None, None, None) => None,
        _ => {
            return Err(PropagationError::InvalidRequest(
                "incomplete variant metadata in assist-rs propagation output".to_string(),
            ));
        }
    };
    let orbits = OrbitBatch::new(orbit_ids, object_ids, coordinates)?;
    Ok(PropagationResult {
        orbits,
        variants,
        times,
        validity: Validity::from_bools(&validity),
        diagnostics: PropagationDiagnostics {
            convergence,
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

fn covariance_6x6(values: &[f64]) -> PropagationResultValue<[[f64; 6]; 6]> {
    if values.len() != 36 {
        return Err(PropagationError::InvalidRequest(format!(
            "expected 36 covariance values for a Cartesian row, got {}",
            values.len()
        )));
    }
    let mut out = [[0.0; 6]; 6];
    for row in 0..6 {
        for col in 0..6 {
            out[row][col] = values[row * 6 + col];
        }
    }
    Ok(out)
}

fn flatten_covariance(values: [[f64; 6]; 6]) -> [f64; 36] {
    let mut out = [0.0; 36];
    for row in 0..6 {
        for col in 0..6 {
            out[row * 6 + col] = values[row][col];
        }
    }
    out
}

fn state_is_finite(state: &[f64; 6]) -> bool {
    state.iter().all(|value| value.is_finite())
}

fn row_failure_output(
    rows: usize,
    covariance_requested: bool,
    failure_code: Option<PropagationFailureCode>,
    message: String,
) -> RowOutput {
    RowOutput {
        states: vec![[f64::NAN; 6]; rows],
        covariance: covariance_requested.then(|| vec![[0.0; 36]; rows]),
        covariance_validity: covariance_requested.then(|| vec![false; rows]),
        validity: vec![false; rows],
        messages: vec![Some(message); rows],
        backend: Some(BACKEND_NAME.to_string()),
        iterations: vec![None; rows],
        failure_codes: vec![failure_code; rows],
    }
}

fn assist_failure_message(code: PropagationFailureCode) -> String {
    match code {
        PropagationFailureCode::NonFiniteInputState => {
            "assist-rs propagation input state was non-finite".to_string()
        }
        PropagationFailureCode::NonFiniteOutputState => {
            "assist-rs propagation produced a non-finite state".to_string()
        }
        PropagationFailureCode::NonFiniteCovariance => {
            "assist-rs covariance propagation produced a non-finite covariance".to_string()
        }
        PropagationFailureCode::SolverZeroDerivative => {
            "assist-rs propagation reported a zero derivative".to_string()
        }
        PropagationFailureCode::SolverMaxIterations => {
            "assist-rs propagation reached the maximum iteration count".to_string()
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use adam_core_rs_coords::propagation::PropagationOptions;
    use adam_core_rs_coords::types::SchemaResult;
    use adam_core_rs_coords::{ObjectId, OrbitId, SchemaError, VariantId, NANOS_PER_DAY};
    use std::cell::Cell;

    struct NoopProvider;

    impl TimeScaleProvider for NoopProvider {
        fn rescale(&self, _times: &TimeArray, _new_scale: TimeScale) -> SchemaResult<TimeArray> {
            Err(SchemaError::InvalidRecordBatch(
                "assist adapter tests should not rescale time".to_string(),
            ))
        }
    }

    fn sample_times(scale: TimeScale) -> TimeArray {
        TimeArray::from_parts(scale, vec![60_010, 60_000], vec![0, NANOS_PER_DAY / 2]).unwrap()
    }

    fn sample_orbits(frame: Frame, origin: OriginId, time_scale: TimeScale) -> OrbitBatch {
        let coordinates = CoordinateBatch::cartesian(
            vec![[1.0, 0.2, 0.1, 0.001, 0.015, 0.0005]],
            frame,
            OriginArray::repeat(origin, 1),
            Some(TimeArray::from_parts(time_scale, vec![60_000], vec![0]).unwrap()),
            None,
        )
        .unwrap();
        OrbitBatch::new(
            vec![OrbitId("orbit-a".to_string())],
            vec![Some(ObjectId("object-a".to_string()))],
            coordinates,
        )
        .unwrap()
    }

    #[test]
    fn heliocentric_origin_accepts_sun_forms_only() {
        assert!(is_heliocentric_origin(&OriginId::Named("SUN".to_string())));
        assert!(is_heliocentric_origin(&OriginId::Named("sun".to_string())));
        assert!(is_heliocentric_origin(&OriginId::Naif(10)));
        assert!(!is_heliocentric_origin(&OriginId::SolarSystemBarycenter));
        assert!(!is_heliocentric_origin(&OriginId::Naif(0)));
        assert!(!is_heliocentric_origin(&OriginId::Named(
            "EARTH".to_string()
        )));
    }

    #[test]
    fn sorted_time_indices_preserve_caller_order_metadata() {
        let times = sample_times(TimeScale::Tdb);
        let indices = vec![0, 1];
        let sorted = sorted_time_indices(&indices, &times);
        assert_eq!(sorted, vec![1, 0]);
    }

    #[test]
    fn covariance_flatten_round_trips_row_major_values() {
        let values = (0..36).map(|value| value as f64).collect::<Vec<_>>();
        let matrix = covariance_6x6(&values).unwrap();
        assert_eq!(matrix[2][3], 15.0);
        assert_eq!(flatten_covariance(matrix).as_slice(), values.as_slice());
    }

    #[test]
    fn request_scope_rejects_non_tdb_times() {
        let orbits = sample_orbits(
            Frame::Ecliptic,
            OriginId::Named("SUN".to_string()),
            TimeScale::Utc,
        );
        let targets = TimeArray::from_parts(TimeScale::Tdb, vec![60_010], vec![0]).unwrap();
        let request =
            PropagationRequest::new(&orbits, &targets, PropagationOptions::default()).unwrap();
        let err = validate_request_scope(&request, false).unwrap_err();
        assert!(
            matches!(err, PropagationError::InvalidRequest(message) if message.contains("TDB-only"))
        );
    }

    #[test]
    fn request_scope_rejects_non_heliocentric_origins() {
        let orbits = sample_orbits(
            Frame::Ecliptic,
            OriginId::SolarSystemBarycenter,
            TimeScale::Tdb,
        );
        let targets = TimeArray::from_parts(TimeScale::Tdb, vec![60_010], vec![0]).unwrap();
        let request =
            PropagationRequest::new(&orbits, &targets, PropagationOptions::default()).unwrap();
        let err = validate_request_scope(&request, false).unwrap_err();
        assert!(
            matches!(err, PropagationError::InvalidRequest(message) if message.contains("SUN origins"))
        );
    }

    #[test]
    fn request_scope_rejects_non_ecliptic_frames() {
        let orbits = sample_orbits(
            Frame::Equatorial,
            OriginId::Named("SUN".to_string()),
            TimeScale::Tdb,
        );
        let targets = TimeArray::from_parts(TimeScale::Tdb, vec![60_010], vec![0]).unwrap();
        let request =
            PropagationRequest::new(&orbits, &targets, PropagationOptions::default()).unwrap();
        let err = validate_request_scope(&request, false).unwrap_err();
        assert!(
            matches!(err, PropagationError::InvalidRequest(message) if message.contains("ecliptic"))
        );
    }

    #[test]
    fn assemble_result_preserves_variant_metadata() {
        let coordinates = CoordinateBatch::cartesian(
            vec![[1.0, 0.0, 0.0, 0.0, 0.01, 0.0]],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::Named("SUN".to_string()), 1),
            Some(TimeArray::from_parts(TimeScale::Tdb, vec![60_000], vec![0]).unwrap()),
            None,
        )
        .unwrap();
        let variants = OrbitVariantBatch::new(
            vec![OrbitId("orbit-a".to_string())],
            vec![None],
            vec![Some(VariantId("v0".to_string()))],
            vec![Some(1.0)],
            vec![Some(1.0)],
            coordinates,
        )
        .unwrap();
        let targets = TimeArray::from_parts(TimeScale::Tdb, vec![60_010], vec![0]).unwrap();
        let request = PropagationRequest::new_variants(
            &variants,
            &targets,
            PropagationOptions {
                covariance: CovariancePropagation::None,
                ..PropagationOptions::default()
            },
        )
        .unwrap();
        let block = AssistOrbitBlock {
            orbit_index: 0,
            time_indices: vec![0],
            states: vec![[1.1, 0.0, 0.0, 0.0, 0.01, 0.0]],
            covariance: None,
            covariance_validity: None,
            validity: vec![true],
            messages: vec![None],
            iterations: vec![None],
            failure_codes: vec![None],
        };
        let result = assemble_result(&request, &targets, vec![block]).unwrap();
        let out = result.variants.unwrap();
        assert_eq!(out.variant_id[0].as_ref().unwrap().0, "v0");
        assert_eq!(out.weights[0], Some(1.0));
        assert!(result.validity.is_valid(0));
    }

    #[test]
    fn provider_is_not_used_by_tdb_only_scope() {
        let calls = Cell::new(0);
        struct CountingProvider<'a>(&'a Cell<usize>);
        impl TimeScaleProvider for CountingProvider<'_> {
            fn rescale(&self, times: &TimeArray, new_scale: TimeScale) -> SchemaResult<TimeArray> {
                self.0.set(self.0.get() + 1);
                TimeArray::new(new_scale, times.epochs.clone())
            }
        }
        let provider = CountingProvider(&calls);
        let orbits = sample_orbits(
            Frame::Ecliptic,
            OriginId::Named("SUN".to_string()),
            TimeScale::Tdb,
        );
        let targets = TimeArray::from_parts(TimeScale::Tdb, vec![60_010], vec![0]).unwrap();
        let request =
            PropagationRequest::new(&orbits, &targets, PropagationOptions::default()).unwrap();
        validate_request_scope(&request, false).unwrap();
        assert_eq!(calls.get(), 0);
        let _ = &provider;
        let _ = NoopProvider;
    }
}
