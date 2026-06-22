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
    generate_ephemeris_translated, rotate_ecliptic_to_equatorial6, rotate_equatorial_to_ecliptic6,
    CoordinateBatch, CovarianceBatch, CovarianceUnits, EphemerisOptions, EphemerisResult, Epoch,
    ObserverBatch, OrbitBatch, OrbitVariantBatch, OriginArray, OriginId, OriginTranslationProvider,
    TimeArray, TimeScale, TimeScaleProvider, Validity, NANOS_PER_DAY,
};
use assist_rs::ffi;
use assist_rs::{
    assist_propagate, libassist_sys, librebound_sys, AssistData, AssistSim, Error as AssistError,
    Ias15AdaptiveMode, IntegratorConfig, Orbit as AssistOrbit, Simulation,
};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::sync::Arc;

#[cfg(feature = "python")]
mod python;

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
            integrator: python_default_integrator(),
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
        orbit_times_tdb: &TimeArray,
        target_times_tdb: &TimeArray,
        target_output_times: &TimeArray,
        compute_stm: bool,
    ) -> PropagationResultValue<PropagationResult> {
        validate_request_scope(request, compute_stm)?;
        let input_coordinates = request.input.coordinates();
        let states = input_coordinates.values.cartesian().ok_or_else(|| {
            PropagationError::InvalidRequest(
                "AssistPropagator requires Cartesian orbit coordinates".to_string(),
            )
        })?;
        let covariance = input_coordinates.covariance.as_ref();
        let include_covariance = compute_stm && covariance.is_some();
        let normalized_states = normalize_input_states(
            self.data.as_ref(),
            states,
            input_coordinates.frame,
            &input_coordinates.origins,
            orbit_times_tdb,
        )?;
        let orbit_indices = (0..request.input.len()).collect::<Vec<_>>();
        let policy = request.options.epoch_policy.clone();
        let requested_chunk_size = request.options.chunk_size;

        run_with_thread_limit(request.options.thread_limit, || {
            let chunk_size = normalized_chunk_size(requested_chunk_size, request.input.len());
            let context = AssistChunkContext {
                data: &self.data,
                integrator: self.integrator,
                compute_stm,
                request,
                normalized_states: &normalized_states,
                orbit_times_tdb,
                target_times_tdb,
                covariance,
                include_covariance,
                policy: &policy,
                frame: input_coordinates.frame,
                origins: &input_coordinates.origins,
            };
            let chunk_results = orbit_indices
                .par_chunks(chunk_size)
                .map(|chunk| propagate_assist_chunk(&context, chunk))
                .collect::<Vec<_>>();

            let mut blocks = Vec::with_capacity(request.input.len());
            for chunk in chunk_results {
                blocks.extend(chunk?);
            }
            assemble_result(request, target_output_times, blocks)
        })
    }
}

impl AssistPropagator {
    /// Generate ephemerides natively over this ASSIST propagator, translating
    /// the observers to the orbits' origin and ecliptic frame via
    /// `translation_provider` (for example an
    /// `adam_core_rs_spice::AdamCoreSpiceBackend`). The orbits propagate in
    /// their own origin, so this is the Rust-native ASSIST ephemeris path that
    /// replaces the Python observer-state / `get_perturber_state` glue.
    pub fn generate_ephemeris<T>(
        &self,
        orbits: &OrbitBatch,
        observers: &ObserverBatch,
        options: &EphemerisOptions,
        time_provider: &dyn TimeScaleProvider,
        translation_provider: &T,
    ) -> PropagationResultValue<EphemerisResult>
    where
        T: OriginTranslationProvider + ?Sized,
    {
        generate_ephemeris_translated(
            self,
            orbits,
            observers,
            options,
            time_provider,
            translation_provider,
        )
    }
}

impl Propagator for AssistPropagator {
    type Shard = AssistShard;

    fn integration_time_scale(&self) -> TimeScale {
        TimeScale::Tdb
    }

    // RM-STANDALONE-007B decision (2026-06-21): `CovariancePropagation::Linearized`
    // is a deliberately separate, lower-level Rust-typed-trait surface that transports
    // covariance with the ASSIST state-transition matrix (first-order linearization).
    // It is intentionally NOT the public `adam_assist` covariance behavior: the Python
    // public path samples variants and collapses them, so the public PyO3 boundary
    // (`python.rs`) mirrors that sampled/collapse semantics and forces
    // `CovariancePropagation::None`. Keep Linearized reachable only through the typed
    // Rust trait for callers that explicitly want STM transport; do not wire it into
    // the drop-in public covariance API without an explicit, separately-labeled surface.
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
        let orbit_times_tdb =
            rescale_for_assist(input_orbit_times, provider, "orbit coordinate times")?;
        let target_times_tdb = rescale_for_assist(request.times, provider, "target times")?;
        let target_output_times = restore_from_assist_scale(
            &target_times_tdb,
            request.times.scale,
            provider,
            "target times",
        )?;
        let compute_stm = request.options.covariance == CovariancePropagation::Linearized
            && request.input.coordinates().covariance.is_some();
        self.propagate_with_stm(
            request,
            &orbit_times_tdb,
            &target_times_tdb,
            &target_output_times,
            compute_stm,
        )
    }
}

struct AssistChunkContext<'a, 'request> {
    data: &'a Arc<AssistData>,
    integrator: IntegratorConfig,
    compute_stm: bool,
    request: &'a PropagationRequest<'request>,
    normalized_states: &'a [[f64; 6]],
    orbit_times_tdb: &'a TimeArray,
    target_times_tdb: &'a TimeArray,
    covariance: Option<&'a CovarianceBatch>,
    include_covariance: bool,
    policy: &'a EpochPolicy,
    frame: Frame,
    origins: &'a OriginArray,
}

#[derive(Debug, Clone)]
struct SameEpochStateGroup {
    epoch: Epoch,
    time_indices: Vec<usize>,
    sorted_time_indices: Vec<usize>,
    orbit_indices: Vec<usize>,
}

fn propagate_assist_chunk(
    context: &AssistChunkContext<'_, '_>,
    chunk: &[usize],
) -> PropagationResultValue<Vec<AssistOrbitBlock>> {
    if context.include_covariance {
        return propagate_assist_chunk_one_by_one(context, chunk);
    }
    propagate_state_only_chunk(context, chunk)
}

fn propagate_assist_chunk_one_by_one(
    context: &AssistChunkContext<'_, '_>,
    chunk: &[usize],
) -> PropagationResultValue<Vec<AssistOrbitBlock>> {
    let mut shard = AssistShard::new(
        Arc::clone(context.data),
        context.integrator,
        context.compute_stm,
    );
    let mut blocks = Vec::with_capacity(chunk.len());
    for &orbit_index in chunk {
        let row = orbit_row(
            context.request,
            context.normalized_states,
            context.orbit_times_tdb,
            context.covariance,
            orbit_index,
            context.include_covariance,
        );
        let time_indices =
            time_indices_for_policy(context.policy, orbit_index, context.target_times_tdb.len())?;
        let sorted_time_indices = sorted_time_indices(&time_indices, context.target_times_tdb);
        let sorted_times = sorted_time_indices
            .iter()
            .map(|&time_index| context.target_times_tdb.epochs[time_index])
            .collect::<Vec<_>>();
        let sorted_output = shard.propagate_one(row, &sorted_times)?;
        let mut block = reorder_output(
            orbit_index,
            sorted_time_indices.clone(),
            sorted_time_indices,
            sorted_output,
        )?;
        restore_output_block(
            context.data.as_ref(),
            context.frame,
            &context.origins.origins[orbit_index],
            context.target_times_tdb,
            &mut block,
        )?;
        blocks.push(block);
    }
    Ok(blocks)
}

fn propagate_state_only_chunk(
    context: &AssistChunkContext<'_, '_>,
    chunk: &[usize],
) -> PropagationResultValue<Vec<AssistOrbitBlock>> {
    let mut blocks = Vec::with_capacity(chunk.len());
    let mut groups: Vec<SameEpochStateGroup> = Vec::new();

    for &orbit_index in chunk {
        let time_indices =
            time_indices_for_policy(context.policy, orbit_index, context.target_times_tdb.len())?;
        let sorted_time_indices = sorted_time_indices(&time_indices, context.target_times_tdb);
        if !state_is_finite(&context.normalized_states[orbit_index]) {
            blocks.push(state_only_failure_block(
                orbit_index,
                sorted_time_indices,
                Some(PropagationFailureCode::NonFiniteInputState),
                "assist-rs propagation input state was non-finite".to_string(),
            ));
            continue;
        }
        let epoch = context.orbit_times_tdb.epochs[orbit_index];
        if let Some(group) = groups.iter_mut().find(|group| {
            group.epoch == epoch
                && group.time_indices == time_indices
                && group.sorted_time_indices == sorted_time_indices
        }) {
            group.orbit_indices.push(orbit_index);
        } else {
            groups.push(SameEpochStateGroup {
                epoch,
                time_indices,
                sorted_time_indices,
                orbit_indices: vec![orbit_index],
            });
        }
    }

    for group in groups {
        blocks.extend(propagate_same_epoch_state_group(context, &group)?);
    }

    Ok(blocks)
}

fn propagate_same_epoch_state_group(
    context: &AssistChunkContext<'_, '_>,
    group: &SameEpochStateGroup,
) -> PropagationResultValue<Vec<AssistOrbitBlock>> {
    let sorted_times = group
        .sorted_time_indices
        .iter()
        .map(|&time_index| context.target_times_tdb.epochs[time_index])
        .collect::<Vec<_>>();
    let states = group
        .orbit_indices
        .iter()
        .map(|&orbit_index| context.normalized_states[orbit_index])
        .collect::<Vec<_>>();

    match propagate_same_epoch_states(
        context.data.as_ref(),
        context.integrator,
        &states,
        group.epoch,
        &sorted_times,
    ) {
        Ok(group_states) => {
            let mut blocks = Vec::with_capacity(group.orbit_indices.len());
            for (group_row, orbit_index) in group.orbit_indices.iter().copied().enumerate() {
                let sorted_output = state_only_output_from_states(group_states[group_row].clone());
                let mut block = reorder_output(
                    orbit_index,
                    group.sorted_time_indices.clone(),
                    group.sorted_time_indices.clone(),
                    sorted_output,
                )?;
                restore_output_block(
                    context.data.as_ref(),
                    context.frame,
                    &context.origins.origins[orbit_index],
                    context.target_times_tdb,
                    &mut block,
                )?;
                blocks.push(block);
            }
            Ok(blocks)
        }
        Err(err) => {
            let (failure_code, message) = classify_assist_error(err)?;
            Ok(group
                .orbit_indices
                .iter()
                .map(|&orbit_index| {
                    state_only_failure_block(
                        orbit_index,
                        group.sorted_time_indices.clone(),
                        Some(failure_code),
                        message.clone(),
                    )
                })
                .collect())
        }
    }
}

fn propagate_same_epoch_states(
    data: &AssistData,
    integrator: IntegratorConfig,
    states: &[[f64; 6]],
    epoch: Epoch,
    target_times: &[Epoch],
) -> Result<Vec<Vec<[f64; 6]>>, AssistError> {
    if states.is_empty() {
        return Ok(Vec::new());
    }
    let ephem = &data.ephem;
    let t0 = mjd_tdb_to_assist_time(epoch.mjd(), ephem.jd_ref());
    let sun0 = particle_state(ephem.get_body_state(ffi::ASSIST_BODY_SUN, t0)?);

    let mut sim = Simulation::new()?;
    sim.set_t(t0);
    apply_integrator_config(&mut sim, integrator);
    let mut asim = AssistSim::new(sim, ephem)?;
    asim.set_forces(ffi::ASSIST_FORCES_DEFAULT);

    for state in states {
        let helio_eq = rotate_ecliptic_to_equatorial6(state);
        let bary_eq = add_state(helio_eq, sun0);
        asim.sim_mut().add_test_particle(
            bary_eq[0], bary_eq[1], bary_eq[2], bary_eq[3], bary_eq[4], bary_eq[5],
        );
    }

    let mut states_by_orbit = vec![Vec::with_capacity(target_times.len()); states.len()];
    for target_time in target_times {
        let t_target = mjd_tdb_to_assist_time(target_time.mjd(), ephem.jd_ref());
        asim.integrate(t_target)?;
        let particles = asim.sim().particles();
        if particles.len() < states.len() {
            return Err(AssistError::Other(
                "No particles after integration".to_string(),
            ));
        }
        let sun_t = particle_state(ephem.get_body_state(ffi::ASSIST_BODY_SUN, t_target)?);
        for (row, particle) in particles.iter().take(states.len()).enumerate() {
            let bary_eq = [
                particle.x,
                particle.y,
                particle.z,
                particle.vx,
                particle.vy,
                particle.vz,
            ];
            states_by_orbit[row].push(rotate_equatorial_to_ecliptic6(&subtract_state(
                bary_eq, sun_t,
            )));
        }
    }

    Ok(states_by_orbit)
}

fn particle_state(particle: ffi::reb_particle) -> [f64; 6] {
    [
        particle.x,
        particle.y,
        particle.z,
        particle.vx,
        particle.vy,
        particle.vz,
    ]
}

fn apply_integrator_config(sim: &mut Simulation, integrator: IntegratorConfig) {
    if let Some(dt) = integrator.initial_dt {
        sim.set_dt(dt);
    }
    if let Some(epsilon) = integrator.epsilon {
        sim.set_ias15_epsilon(epsilon);
    }
    if let Some(min_dt) = integrator.min_dt {
        sim.set_ias15_min_dt(min_dt);
    }
    if let Some(mode) = integrator.adaptive_mode {
        sim.set_ias15_adaptive_mode(mode);
    }
}

fn state_only_output_from_states(states: Vec<[f64; 6]>) -> RowOutput {
    let mut validity = Vec::with_capacity(states.len());
    let mut messages = Vec::with_capacity(states.len());
    let mut failure_codes = Vec::with_capacity(states.len());
    for state in &states {
        let failure_code = if state_is_finite(state) {
            None
        } else {
            Some(PropagationFailureCode::NonFiniteOutputState)
        };
        validity.push(failure_code.is_none());
        messages.push(failure_code.map(assist_failure_message));
        failure_codes.push(failure_code);
    }
    RowOutput {
        states,
        covariance: None,
        covariance_validity: None,
        validity,
        messages,
        backend: Some(BACKEND_NAME.to_string()),
        iterations: vec![None; failure_codes.len()],
        failure_codes,
    }
}

fn state_only_failure_block(
    orbit_index: usize,
    time_indices: Vec<usize>,
    failure_code: Option<PropagationFailureCode>,
    message: String,
) -> AssistOrbitBlock {
    let rows = time_indices.len();
    AssistOrbitBlock {
        orbit_index,
        time_indices,
        states: vec![[f64::NAN; 6]; rows],
        covariance: None,
        covariance_validity: None,
        validity: vec![false; rows],
        messages: vec![Some(message); rows],
        iterations: vec![None; rows],
        failure_codes: vec![failure_code; rows],
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
                let (failure_code, message) = classify_assist_error(err)?;
                return Ok(row_failure_output(
                    times.len(),
                    covariance_requested,
                    Some(failure_code),
                    message,
                ));
            }
        };

        if propagated.len() != times.len() {
            return Err(PropagationError::BackendProtocol(format!(
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

fn python_default_integrator() -> IntegratorConfig {
    IntegratorConfig {
        initial_dt: Some(1.0e-6),
        min_dt: Some(1.0e-9),
        adaptive_mode: Some(Ias15AdaptiveMode::Global),
        epsilon: Some(1.0e-6),
    }
}

fn rescale_for_assist(
    times: &TimeArray,
    provider: &dyn TimeScaleProvider,
    field: &str,
) -> PropagationResultValue<TimeArray> {
    let rescaled = times.rescale_with_provider(TimeScale::Tdb, provider)?;
    rescaled.validate()?;
    if rescaled.scale != TimeScale::Tdb {
        return Err(PropagationError::InvalidRequest(format!(
            "{field} provider returned {} instead of required tdb",
            rescaled.scale.as_str()
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

fn restore_from_assist_scale(
    times_tdb: &TimeArray,
    target_scale: TimeScale,
    provider: &dyn TimeScaleProvider,
    field: &str,
) -> PropagationResultValue<TimeArray> {
    let backend_times_tdb = python_backend_output_times(times_tdb)?;
    let restored = backend_times_tdb.rescale_with_provider(target_scale, provider)?;
    restored.validate()?;
    if restored.scale != target_scale {
        return Err(PropagationError::InvalidRequest(format!(
            "{field} provider returned {} instead of requested {}",
            restored.scale.as_str(),
            target_scale.as_str()
        )));
    }
    if restored.len() != times_tdb.len() {
        return Err(PropagationError::InvalidRequest(format!(
            "{field} provider changed length from {} to {} while restoring public output scale",
            times_tdb.len(),
            restored.len()
        )));
    }
    Ok(restored)
}

fn python_backend_output_times(times_tdb: &TimeArray) -> PropagationResultValue<TimeArray> {
    // Python adam-assist rebuilds propagated timestamps from f64 Julian dates
    // returned to the public table. Preserve that caller-visible nanosecond
    // quantization so UTC outputs match the frozen public-semantics fixture.
    TimeArray::new(
        TimeScale::Tdb,
        times_tdb
            .epochs
            .iter()
            .map(|epoch| {
                let jd = epoch.mjd() + 2_400_000.5;
                epoch_from_mjd(jd - 2_400_000.5)
            })
            .collect(),
    )
    .map_err(PropagationError::from)
}

fn epoch_from_mjd(mjd: f64) -> Epoch {
    let days = mjd.floor() as i64;
    let nanos = ((mjd - days as f64) * NANOS_PER_DAY as f64).round() as i64;
    Epoch::new(days, nanos)
}

fn validate_request_scope(
    request: &PropagationRequest<'_>,
    compute_stm: bool,
) -> PropagationResultValue<()> {
    let coordinates = request.input.coordinates();
    if !matches!(coordinates.frame, Frame::Ecliptic | Frame::Equatorial) {
        return Err(PropagationError::InvalidRequest(format!(
            "AssistPropagator requires ecliptic or equatorial Cartesian input; got frame {}",
            coordinates.frame.as_str()
        )));
    }
    if coordinates.values.cartesian().is_none() {
        return Err(PropagationError::InvalidRequest(
            "AssistPropagator requires Cartesian coordinates".to_string(),
        ));
    }
    coordinates
        .times
        .as_ref()
        .ok_or(PropagationError::MissingOrbitTimes)?;
    for origin in &coordinates.origins.origins {
        if !is_supported_public_origin(origin) {
            return Err(PropagationError::InvalidRequest(format!(
                "AssistPropagator currently supports SUN and SOLAR_SYSTEM_BARYCENTER origins; got {}",
                origin.code()
            )));
        }
    }
    if compute_stm && !is_native_assist_scope(coordinates.frame, &coordinates.origins) {
        return Err(PropagationError::InvalidRequest(
            "AssistPropagator linearized covariance currently requires native SUN/ecliptic input; public covariance frame/origin transforms are not implemented yet"
                .to_string(),
        ));
    }
    Ok(())
}

fn is_native_assist_scope(frame: Frame, origins: &OriginArray) -> bool {
    frame == Frame::Ecliptic && origins.origins.iter().all(is_sun_origin)
}

fn is_supported_public_origin(origin: &OriginId) -> bool {
    is_sun_origin(origin) || is_solar_system_barycenter_origin(origin)
}

fn is_sun_origin(origin: &OriginId) -> bool {
    matches!(origin, OriginId::Naif(10))
        || matches!(origin, OriginId::Named(code) if code.eq_ignore_ascii_case("SUN"))
}

fn is_solar_system_barycenter_origin(origin: &OriginId) -> bool {
    matches!(origin, OriginId::SolarSystemBarycenter | OriginId::Naif(0))
        || matches!(origin, OriginId::Named(code) if code.eq_ignore_ascii_case("SOLAR_SYSTEM_BARYCENTER") || code.eq_ignore_ascii_case("SSB"))
}

fn normalize_input_states(
    data: &AssistData,
    states: &[[f64; 6]],
    frame: Frame,
    origins: &OriginArray,
    orbit_times_tdb: &TimeArray,
) -> PropagationResultValue<Vec<[f64; 6]>> {
    states
        .iter()
        .copied()
        .enumerate()
        .map(|(row, state)| {
            let origin = &origins.origins[row];
            let sun_bary_eq = if is_solar_system_barycenter_origin(origin) {
                sun_barycentric_equatorial(data, orbit_times_tdb.epochs[row])?
            } else {
                [0.0; 6]
            };
            normalize_state_to_assist_frame(state, frame, origin, sun_bary_eq)
        })
        .collect()
}

fn restore_output_block(
    data: &AssistData,
    frame: Frame,
    origin: &OriginId,
    target_times_tdb: &TimeArray,
    block: &mut AssistOrbitBlock,
) -> PropagationResultValue<()> {
    for (row_offset, state) in block.states.iter_mut().enumerate() {
        let time_index = block.time_indices[row_offset];
        let sun_bary_eq = if is_solar_system_barycenter_origin(origin) {
            sun_barycentric_equatorial(data, target_times_tdb.epochs[time_index])?
        } else {
            [0.0; 6]
        };
        *state = restore_state_from_assist_frame(*state, frame, origin, sun_bary_eq)?;
    }
    Ok(())
}

fn normalize_state_to_assist_frame(
    state: [f64; 6],
    frame: Frame,
    origin: &OriginId,
    sun_bary_eq: [f64; 6],
) -> PropagationResultValue<[f64; 6]> {
    let state_eq = state_to_equatorial(state, frame)?;
    let helio_eq = if is_sun_origin(origin) {
        state_eq
    } else if is_solar_system_barycenter_origin(origin) {
        subtract_state(state_eq, sun_bary_eq)
    } else {
        return Err(PropagationError::InvalidRequest(format!(
            "unsupported ASSIST input origin {}",
            origin.code()
        )));
    };
    Ok(rotate_equatorial_to_ecliptic6(&helio_eq))
}

fn restore_state_from_assist_frame(
    state: [f64; 6],
    frame: Frame,
    origin: &OriginId,
    sun_bary_eq: [f64; 6],
) -> PropagationResultValue<[f64; 6]> {
    let helio_eq = rotate_ecliptic_to_equatorial6(&state);
    let state_eq = if is_sun_origin(origin) {
        helio_eq
    } else if is_solar_system_barycenter_origin(origin) {
        add_state(helio_eq, sun_bary_eq)
    } else {
        return Err(PropagationError::InvalidRequest(format!(
            "unsupported ASSIST output origin {}",
            origin.code()
        )));
    };
    state_from_equatorial(state_eq, frame)
}

fn state_to_equatorial(state: [f64; 6], frame: Frame) -> PropagationResultValue<[f64; 6]> {
    match frame {
        Frame::Equatorial => Ok(state),
        Frame::Ecliptic => Ok(rotate_ecliptic_to_equatorial6(&state)),
        _ => Err(PropagationError::InvalidRequest(format!(
            "AssistPropagator does not support frame {}",
            frame.as_str()
        ))),
    }
}

fn state_from_equatorial(state: [f64; 6], frame: Frame) -> PropagationResultValue<[f64; 6]> {
    match frame {
        Frame::Equatorial => Ok(state),
        Frame::Ecliptic => Ok(rotate_equatorial_to_ecliptic6(&state)),
        _ => Err(PropagationError::InvalidRequest(format!(
            "AssistPropagator does not support frame {}",
            frame.as_str()
        ))),
    }
}

fn sun_barycentric_equatorial(
    data: &AssistData,
    epoch_tdb: Epoch,
) -> PropagationResultValue<[f64; 6]> {
    let t = mjd_tdb_to_assist_time(epoch_tdb.mjd(), data.ephem.jd_ref());
    let sun = data
        .ephem
        .get_body_state(ffi::ASSIST_BODY_SUN, t)
        .map_err(|err| {
            PropagationError::Backend(format!("assist-rs Sun state lookup failed: {err}"))
        })?;
    Ok([sun.x, sun.y, sun.z, sun.vx, sun.vy, sun.vz])
}

fn mjd_tdb_to_assist_time(mjd_tdb: f64, jd_ref: f64) -> f64 {
    (mjd_tdb + 2_400_000.5) - jd_ref
}

fn add_state(left: [f64; 6], right: [f64; 6]) -> [f64; 6] {
    [
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
        left[3] + right[3],
        left[4] + right[4],
        left[5] + right[5],
    ]
}

fn subtract_state(left: [f64; 6], right: [f64; 6]) -> [f64; 6] {
    [
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
        left[3] - right[3],
        left[4] - right[4],
        left[5] - right[5],
    ]
}

fn normalized_chunk_size(chunk_size: Option<usize>, rows: usize) -> usize {
    let rows = rows.max(1);
    match chunk_size {
        Some(size) => size.max(1).min(rows),
        None => rows.div_ceil(rayon::current_num_threads().max(1)).max(1),
    }
}

fn time_indices_for_policy(
    policy: &EpochPolicy,
    orbit_index: usize,
    times_len: usize,
) -> PropagationResultValue<Vec<usize>> {
    match policy {
        EpochPolicy::CrossProduct => Ok((0..times_len).collect()),
        EpochPolicy::Pairwise => Ok(vec![orbit_index]),
        EpochPolicy::PerOrbit { .. } => Err(PropagationError::InvalidRequest(
            "PerOrbit epoch policy is not implemented by AssistPropagator".to_string(),
        )),
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

fn compare_blocks_for_public_output(
    request: &PropagationRequest<'_>,
    left: &AssistOrbitBlock,
    right: &AssistOrbitBlock,
) -> Ordering {
    let left_orbit = request.input.orbit_id()[left.orbit_index].0.as_str();
    let right_orbit = request.input.orbit_id()[right.orbit_index].0.as_str();
    let orbit_order = left_orbit.cmp(right_orbit);
    if orbit_order != Ordering::Equal {
        return orbit_order;
    }
    let variant_id = request.input.variant_id();
    let left_variant = variant_id
        .and_then(|values| values[left.orbit_index].as_ref())
        .map(|value| value.0.as_str());
    let right_variant = variant_id
        .and_then(|values| values[right.orbit_index].as_ref())
        .map(|value| value.0.as_str());
    left_variant.cmp(&right_variant)
}

fn assemble_result(
    request: &PropagationRequest<'_>,
    target_times: &TimeArray,
    blocks: Vec<AssistOrbitBlock>,
) -> PropagationResultValue<PropagationResult> {
    let mut blocks = blocks;
    blocks.sort_by(|left, right| compare_blocks_for_public_output(request, left, right));
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

fn classify_assist_error(
    err: AssistError,
) -> PropagationResultValue<(PropagationFailureCode, String)> {
    let message = format!("assist-rs propagation failed: {err}");
    match err {
        AssistError::Sys(libassist_sys::Error::Reb(
            librebound_sys::Error::NoParticles
            | librebound_sys::Error::CloseEncounter
            | librebound_sys::Error::Escape
            | librebound_sys::Error::Collision
            | librebound_sys::Error::IntegrationFailed(_),
        ))
        | AssistError::LightTimeConvergence(_) => {
            Ok((PropagationFailureCode::IntegratorFailure, message))
        }
        AssistError::Sys(libassist_sys::Error::EphemerisError(_))
        | AssistError::Sys(libassist_sys::Error::Other(_))
        | AssistError::Sys(libassist_sys::Error::Reb(librebound_sys::Error::Other(_)))
        | AssistError::InvalidBody(_)
        | AssistError::InvalidObservatory(_)
        | AssistError::MissingEarthOrientation(_)
        | AssistError::Io(_)
        | AssistError::Other(_) => Err(PropagationError::Backend(message)),
    }
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
        PropagationFailureCode::IntegratorFailure => {
            "assist-rs propagation reported an integration failure".to_string()
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
    use adam_core_rs_coords::{
        EphemerisOptions, ObjectId, ObservatoryCode, OrbitId, SchemaError, VariantId,
    };
    use serde_json::{json, Value};

    // Local measured residuals against the frozen Python fixture on 2026-05-20:
    // max position 2.220446049250313e-16 AU (33 micrometers), max velocity
    // 1.0408340855860843e-17 AU/day (18 picometers/s). These assertions keep
    // comfortable cross-build headroom while staying far tighter than the prior
    // smoke envelope.
    const ASSIST_FIXTURE_POSITION_TOLERANCE_AU: f64 = 1.0e-14;
    const ASSIST_FIXTURE_VELOCITY_TOLERANCE_AU_PER_DAY: f64 = 1.0e-15;
    const AU_METERS: f64 = 149_597_870_700.0;
    const SECONDS_PER_DAY: f64 = 86_400.0;

    #[derive(Debug, Clone)]
    struct FixtureResidual {
        case_id: String,
        rows: usize,
        max_position_abs_au: f64,
        max_velocity_abs_au_per_day: f64,
        max_time_abs_ns: i64,
    }

    impl FixtureResidual {
        fn max_position_abs_m(&self) -> f64 {
            self.max_position_abs_au * AU_METERS
        }

        fn max_velocity_abs_m_per_s(&self) -> f64 {
            self.max_velocity_abs_au_per_day * AU_METERS / SECONDS_PER_DAY
        }
    }

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

    fn assert_state_close(actual: &[f64; 6], expected: &[f64; 6]) {
        for (left, right) in actual.iter().zip(expected.iter()) {
            assert!((left - right).abs() < 1.0e-14, "{actual:?} != {expected:?}");
        }
    }

    #[test]
    fn python_public_integrator_defaults_are_used() {
        let config = python_default_integrator();
        assert_eq!(config.initial_dt, Some(1.0e-6));
        assert_eq!(config.min_dt, Some(1.0e-9));
        assert_eq!(config.adaptive_mode, Some(Ias15AdaptiveMode::Global));
        assert_eq!(config.epsilon, Some(1.0e-6));
    }

    #[test]
    fn public_origin_helpers_accept_sun_and_ssb_only() {
        assert!(is_sun_origin(&OriginId::Named("SUN".to_string())));
        assert!(is_sun_origin(&OriginId::Named("sun".to_string())));
        assert!(is_sun_origin(&OriginId::Naif(10)));
        assert!(!is_sun_origin(&OriginId::SolarSystemBarycenter));
        assert!(is_solar_system_barycenter_origin(
            &OriginId::SolarSystemBarycenter
        ));
        assert!(is_solar_system_barycenter_origin(&OriginId::Naif(0)));
        assert!(is_solar_system_barycenter_origin(&OriginId::Named(
            "SOLAR_SYSTEM_BARYCENTER".to_string()
        )));
        assert!(is_solar_system_barycenter_origin(&OriginId::Named(
            "ssb".to_string()
        )));
        assert!(!is_supported_public_origin(&OriginId::Named(
            "EARTH".to_string()
        )));
    }

    #[test]
    fn native_sun_ecliptic_normalization_is_identity() {
        let state = [1.0, 0.2, -0.1, 0.001, 0.015, -0.0005];
        let origin = OriginId::Named("SUN".to_string());
        let normalized =
            normalize_state_to_assist_frame(state, Frame::Ecliptic, &origin, [0.0; 6]).unwrap();
        let restored =
            restore_state_from_assist_frame(normalized, Frame::Ecliptic, &origin, [0.0; 6])
                .unwrap();
        assert_state_close(&normalized, &state);
        assert_state_close(&restored, &state);
    }

    #[test]
    fn ssb_equatorial_normalization_restores_public_state() {
        let helio_ecliptic = [1.0, 0.2, -0.1, 0.001, 0.015, -0.0005];
        let sun_bary_eq = [0.004, -0.002, 0.001, 0.00001, -0.00002, 0.00003];
        let origin = OriginId::SolarSystemBarycenter;
        let public_state = restore_state_from_assist_frame(
            helio_ecliptic,
            Frame::Equatorial,
            &origin,
            sun_bary_eq,
        )
        .unwrap();
        let normalized =
            normalize_state_to_assist_frame(public_state, Frame::Equatorial, &origin, sun_bary_eq)
                .unwrap();
        assert_state_close(&normalized, &helio_ecliptic);
    }

    #[test]
    fn rescale_for_assist_accepts_utc_without_provider() {
        let utc = TimeArray::from_parts(TimeScale::Utc, vec![60_000], vec![0]).unwrap();
        let tdb = rescale_for_assist(&utc, &NoopProvider, "target times").unwrap();
        assert_eq!(tdb.scale, TimeScale::Tdb);
        assert_eq!(tdb.len(), utc.len());
    }

    #[test]
    fn request_scope_accepts_public_time_origin_and_frame_semantics() {
        let orbits = sample_orbits(
            Frame::Equatorial,
            OriginId::SolarSystemBarycenter,
            TimeScale::Utc,
        );
        let targets = TimeArray::from_parts(TimeScale::Utc, vec![60_010], vec![0]).unwrap();
        let request = PropagationRequest::new(
            &orbits,
            &targets,
            PropagationOptions {
                covariance: CovariancePropagation::None,
                ..PropagationOptions::default()
            },
        )
        .unwrap();
        validate_request_scope(&request, false).unwrap();
    }

    #[test]
    fn request_scope_rejects_unsupported_origins() {
        let orbits = sample_orbits(
            Frame::Ecliptic,
            OriginId::Named("EARTH".to_string()),
            TimeScale::Tdb,
        );
        let targets = TimeArray::from_parts(TimeScale::Tdb, vec![60_010], vec![0]).unwrap();
        let request = PropagationRequest::new(
            &orbits,
            &targets,
            PropagationOptions {
                covariance: CovariancePropagation::None,
                ..PropagationOptions::default()
            },
        )
        .unwrap();
        let err = validate_request_scope(&request, false).unwrap_err();
        assert!(
            matches!(err, PropagationError::InvalidRequest(message) if message.contains("SUN and SOLAR_SYSTEM_BARYCENTER"))
        );
    }

    #[test]
    fn request_scope_rejects_unsupported_frames() {
        let orbits = sample_orbits(
            Frame::Itrf93,
            OriginId::Named("SUN".to_string()),
            TimeScale::Tdb,
        );
        let targets = TimeArray::from_parts(TimeScale::Tdb, vec![60_010], vec![0]).unwrap();
        let request = PropagationRequest::new(
            &orbits,
            &targets,
            PropagationOptions {
                covariance: CovariancePropagation::None,
                ..PropagationOptions::default()
            },
        )
        .unwrap();
        let err = validate_request_scope(&request, false).unwrap_err();
        assert!(
            matches!(err, PropagationError::InvalidRequest(message) if message.contains("ecliptic or equatorial"))
        );
    }

    #[test]
    fn sorted_time_indices_order_by_mjd() {
        let times = sample_times(TimeScale::Tdb);
        let indices = vec![0, 1];
        let sorted = sorted_time_indices(&indices, &times);
        assert_eq!(sorted, vec![1, 0]);
    }

    #[test]
    fn restored_assist_output_times_match_python_jd_roundtrip() {
        let utc = TimeArray::from_parts(TimeScale::Utc, vec![60_001, 60_002], vec![0, 0]).unwrap();
        let tdb = rescale_for_assist(&utc, &NoopProvider, "target times").unwrap();
        let restored =
            restore_from_assist_scale(&tdb, TimeScale::Utc, &NoopProvider, "target times").unwrap();
        assert_eq!(
            restored.epochs,
            vec![Epoch::new(60_001, 17_512), Epoch::new(60_002, 380)]
        );
    }

    #[test]
    fn assemble_result_sorts_public_output_by_orbit_variant_and_time() {
        let coordinates = CoordinateBatch::cartesian(
            vec![
                [1.0, 0.0, 0.0, 0.0, 0.01, 0.0],
                [2.0, 0.0, 0.0, 0.0, 0.02, 0.0],
            ],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::Named("SUN".to_string()), 2),
            Some(TimeArray::from_parts(TimeScale::Tdb, vec![60_000, 60_000], vec![0, 0]).unwrap()),
            None,
        )
        .unwrap();
        let variants = OrbitVariantBatch::new(
            vec![
                OrbitId("orbit-b".to_string()),
                OrbitId("orbit-a".to_string()),
            ],
            vec![None, None],
            vec![
                Some(VariantId("v1".to_string())),
                Some(VariantId("v0".to_string())),
            ],
            vec![Some(1.0), Some(2.0)],
            vec![Some(1.0), Some(2.0)],
            coordinates,
        )
        .unwrap();
        let targets =
            TimeArray::from_parts(TimeScale::Tdb, vec![60_002, 60_001], vec![0, 0]).unwrap();
        let request = PropagationRequest::new_variants(
            &variants,
            &targets,
            PropagationOptions {
                covariance: CovariancePropagation::None,
                ..PropagationOptions::default()
            },
        )
        .unwrap();
        let block_b = AssistOrbitBlock {
            orbit_index: 0,
            time_indices: vec![1, 0],
            states: vec![
                [2.1, 0.0, 0.0, 0.0, 0.02, 0.0],
                [2.2, 0.0, 0.0, 0.0, 0.02, 0.0],
            ],
            covariance: None,
            covariance_validity: None,
            validity: vec![true, true],
            messages: vec![None, None],
            iterations: vec![None, None],
            failure_codes: vec![None, None],
        };
        let block_a = AssistOrbitBlock {
            orbit_index: 1,
            time_indices: vec![1, 0],
            states: vec![
                [1.1, 0.0, 0.0, 0.0, 0.01, 0.0],
                [1.2, 0.0, 0.0, 0.0, 0.01, 0.0],
            ],
            covariance: None,
            covariance_validity: None,
            validity: vec![true, true],
            messages: vec![None, None],
            iterations: vec![None, None],
            failure_codes: vec![None, None],
        };
        let result = assemble_result(&request, &targets, vec![block_b, block_a]).unwrap();
        let out = result.variants.unwrap();
        assert_eq!(
            out.orbit_id
                .iter()
                .map(|value| value.0.as_str())
                .collect::<Vec<_>>(),
            vec!["orbit-a", "orbit-a", "orbit-b", "orbit-b"]
        );
        assert_eq!(
            out.variant_id
                .iter()
                .map(|value| value.as_ref().map(|id| id.0.as_str()))
                .collect::<Vec<_>>(),
            vec![Some("v0"), Some("v0"), Some("v1"), Some("v1")]
        );
        assert_eq!(
            out.coordinates
                .times
                .as_ref()
                .unwrap()
                .epochs
                .iter()
                .map(|epoch| epoch.days)
                .collect::<Vec<_>>(),
            vec![60_001, 60_002, 60_001, 60_002]
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
    fn default_chunk_size_uses_rows_per_rayon_thread() {
        let rows: usize = 25;
        let expected = rows.div_ceil(rayon::current_num_threads().max(1));
        assert_eq!(normalized_chunk_size(None, rows), expected.max(1));
        assert_eq!(normalized_chunk_size(Some(0), rows), 1);
        assert_eq!(normalized_chunk_size(Some(100), rows), rows);
    }

    #[test]
    fn per_orbit_epoch_policy_fails_loudly() {
        let err = time_indices_for_policy(
            &EpochPolicy::PerOrbit {
                indices: vec![0].into_boxed_slice(),
            },
            0,
            1,
        )
        .unwrap_err();
        assert!(
            matches!(err, PropagationError::InvalidRequest(message) if message.contains("PerOrbit"))
        );
    }

    #[test]
    fn assist_setup_errors_fail_loudly() {
        let err =
            classify_assist_error(AssistError::Other("missing kernels".to_string())).unwrap_err();
        assert!(
            matches!(err, PropagationError::Backend(message) if message.contains("missing kernels"))
        );
    }

    #[test]
    fn assist_integration_errors_are_row_failures() {
        let (code, message) = classify_assist_error(AssistError::Sys(libassist_sys::Error::Reb(
            librebound_sys::Error::Collision,
        )))
        .unwrap();
        assert_eq!(code, PropagationFailureCode::IntegratorFailure);
        assert!(message.contains("collision"));
    }

    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../migration/artifacts/assist_public_semantics_fixture_2026-05-20.json"
        ))
        .expect("fixture JSON must parse")
    }

    fn live_propagator_from_env() -> AssistPropagator {
        let planets = std::env::var("ADAM_CORE_RS_ASSIST_PLANETS_PATH")
            .expect("set ADAM_CORE_RS_ASSIST_PLANETS_PATH to the DE440 BSP path");
        let asteroids = std::env::var("ADAM_CORE_RS_ASSIST_ASTEROIDS_PATH")
            .expect("set ADAM_CORE_RS_ASSIST_ASTEROIDS_PATH to the DE441-n16 BSP path");
        let ephem = assist_rs::Ephemeris::from_paths(
            std::path::Path::new(&planets),
            std::path::Path::new(&asteroids),
        )
        .expect("failed to load ASSIST ephemeris kernels");
        AssistPropagator::new(std::sync::Arc::new(AssistData::new(ephem)))
    }

    fn json_array<'a>(value: &'a Value, field: &str) -> &'a Vec<Value> {
        value
            .get(field)
            .and_then(Value::as_array)
            .unwrap_or_else(|| panic!("fixture field {field} must be an array"))
    }

    fn json_string_array(value: &Value, field: &str) -> Vec<String> {
        json_array(value, field)
            .iter()
            .map(|item| {
                item.as_str()
                    .expect("fixture value must be a string")
                    .to_string()
            })
            .collect()
    }

    fn json_optional_string_array(value: &Value, field: &str) -> Vec<Option<String>> {
        json_array(value, field)
            .iter()
            .map(|item| item.as_str().map(str::to_string))
            .collect()
    }

    fn json_optional_f64_array(value: &Value, field: &str) -> Vec<Option<f64>> {
        json_array(value, field).iter().map(Value::as_f64).collect()
    }

    fn json_time_array(value: &Value) -> TimeArray {
        let scale = TimeScale::parse(
            value
                .get("scale")
                .and_then(Value::as_str)
                .expect("fixture time must have scale"),
        )
        .unwrap();
        let days = json_array(value, "days")
            .iter()
            .map(|item| item.as_i64().expect("fixture day must be i64"))
            .collect::<Vec<_>>();
        let nanos = json_array(value, "nanos")
            .iter()
            .map(|item| item.as_i64().expect("fixture nanos must be i64"))
            .collect::<Vec<_>>();
        TimeArray::from_parts(scale, days, nanos).unwrap()
    }

    fn json_states(value: &Value) -> Vec<[f64; 6]> {
        json_array(value, "values")
            .iter()
            .map(|row| {
                let row = row.as_array().expect("state row must be an array");
                assert_eq!(row.len(), 6);
                [
                    row[0].as_f64().unwrap(),
                    row[1].as_f64().unwrap(),
                    row[2].as_f64().unwrap(),
                    row[3].as_f64().unwrap(),
                    row[4].as_f64().unwrap(),
                    row[5].as_f64().unwrap(),
                ]
            })
            .collect()
    }

    fn json_cartesian_coordinates(value: &Value) -> CoordinateBatch {
        let frame = Frame::parse(
            value
                .get("frame")
                .and_then(Value::as_str)
                .expect("coordinates must have frame"),
        )
        .unwrap();
        let origins = OriginArray::new(
            json_string_array(value, "origin_codes")
                .into_iter()
                .map(OriginId::from_code)
                .collect(),
        );
        CoordinateBatch::cartesian(
            json_states(value),
            frame,
            origins,
            Some(json_time_array(
                value.get("time").expect("coordinates must have time"),
            )),
            None,
        )
        .unwrap()
    }

    fn json_orbits(value: &Value) -> OrbitBatch {
        OrbitBatch::new(
            json_string_array(value, "orbit_id")
                .into_iter()
                .map(OrbitId)
                .collect(),
            json_optional_string_array(value, "object_id")
                .into_iter()
                .map(|item| item.map(ObjectId))
                .collect(),
            json_cartesian_coordinates(value.get("coordinates").unwrap()),
        )
        .unwrap()
    }

    fn json_variants(value: &Value) -> OrbitVariantBatch {
        OrbitVariantBatch::new(
            json_string_array(value, "orbit_id")
                .into_iter()
                .map(OrbitId)
                .collect(),
            json_optional_string_array(value, "object_id")
                .into_iter()
                .map(|item| item.map(ObjectId))
                .collect(),
            json_optional_string_array(value, "variant_id")
                .into_iter()
                .map(|item| item.map(VariantId))
                .collect(),
            json_optional_f64_array(value, "weights"),
            json_optional_f64_array(value, "weights_cov"),
            json_cartesian_coordinates(value.get("coordinates").unwrap()),
        )
        .unwrap()
    }

    fn assert_string_column(actual: &[String], expected: &[String], label: &str) {
        assert_eq!(actual, expected, "{label}");
    }

    fn assert_object_ids(actual: &[Option<ObjectId>], expected: &[Option<String>]) {
        let actual = actual
            .iter()
            .map(|item| item.as_ref().map(|value| value.0.clone()))
            .collect::<Vec<_>>();
        assert_eq!(&actual, expected);
    }

    fn assert_variant_ids(actual: &[Option<VariantId>], expected: &[Option<String>]) {
        let actual = actual
            .iter()
            .map(|item| item.as_ref().map(|value| value.0.clone()))
            .collect::<Vec<_>>();
        assert_eq!(&actual, expected);
    }

    fn assert_coordinates_match_fixture(
        actual: &CoordinateBatch,
        expected: &Value,
        case_id: &str,
    ) -> FixtureResidual {
        let expected_coordinates = json_cartesian_coordinates(expected);
        assert_eq!(actual.frame, expected_coordinates.frame, "{case_id} frame");
        assert_eq!(
            actual
                .origins
                .origins
                .iter()
                .map(OriginId::code)
                .collect::<Vec<_>>(),
            expected_coordinates
                .origins
                .origins
                .iter()
                .map(OriginId::code)
                .collect::<Vec<_>>(),
            "{case_id} origins",
        );
        let actual_times = actual
            .times
            .as_ref()
            .expect("actual output must have times");
        let expected_times = expected_coordinates
            .times
            .as_ref()
            .expect("expected output must have times");
        assert_eq!(actual_times, expected_times, "{case_id} times");
        let actual_states = actual.values.cartesian().expect("actual must be Cartesian");
        let expected_states = expected_coordinates
            .values
            .cartesian()
            .expect("expected must be Cartesian");
        assert_eq!(actual_states.len(), expected_states.len(), "{case_id} rows");
        let mut max_position_abs: f64 = 0.0;
        let mut max_velocity_abs: f64 = 0.0;
        for (actual, expected) in actual_states.iter().zip(expected_states.iter()) {
            for axis in 0..3 {
                max_position_abs = max_position_abs.max((actual[axis] - expected[axis]).abs());
            }
            for axis in 3..6 {
                max_velocity_abs = max_velocity_abs.max((actual[axis] - expected[axis]).abs());
            }
        }
        FixtureResidual {
            case_id: case_id.to_string(),
            rows: actual_states.len(),
            max_position_abs_au: max_position_abs,
            max_velocity_abs_au_per_day: max_velocity_abs,
            max_time_abs_ns: max_time_abs_ns(actual_times, expected_times),
        }
    }

    fn max_time_abs_ns(actual: &TimeArray, expected: &TimeArray) -> i64 {
        assert_eq!(actual.scale, expected.scale);
        actual
            .epochs
            .iter()
            .zip(expected.epochs.iter())
            .map(|(actual, expected)| {
                ((actual.days - expected.days) as i128 * NANOS_PER_DAY as i128
                    + (actual.nanos - expected.nanos) as i128)
                    .abs()
            })
            .max()
            .unwrap_or(0)
            .try_into()
            .expect("fixture time residual must fit i64 nanoseconds")
    }

    fn assert_residuals_within_acceptance(residuals: &[FixtureResidual]) {
        let max_position_abs = residuals
            .iter()
            .map(|residual| residual.max_position_abs_au)
            .fold(0.0_f64, f64::max);
        let max_velocity_abs = residuals
            .iter()
            .map(|residual| residual.max_velocity_abs_au_per_day)
            .fold(0.0_f64, f64::max);
        assert!(
            max_position_abs <= ASSIST_FIXTURE_POSITION_TOLERANCE_AU,
            "position max abs {max_position_abs:e} AU exceeds measured public-semantics fixture tolerance {ASSIST_FIXTURE_POSITION_TOLERANCE_AU:e} AU"
        );
        assert!(
            max_velocity_abs <= ASSIST_FIXTURE_VELOCITY_TOLERANCE_AU_PER_DAY,
            "velocity max abs {max_velocity_abs:e} AU/day exceeds measured public-semantics fixture tolerance {ASSIST_FIXTURE_VELOCITY_TOLERANCE_AU_PER_DAY:e} AU/day"
        );
    }

    fn write_residual_artifact(fixture: &Value, residuals: &[FixtureResidual]) {
        let Ok(path) = std::env::var("ADAM_CORE_RS_ASSIST_RESIDUALS_PATH") else {
            return;
        };
        let path = std::path::Path::new(&path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("failed to create residual artifact directory");
        }
        let max_position_abs_au = residuals
            .iter()
            .map(|residual| residual.max_position_abs_au)
            .fold(0.0_f64, f64::max);
        let max_velocity_abs_au_per_day = residuals
            .iter()
            .map(|residual| residual.max_velocity_abs_au_per_day)
            .fold(0.0_f64, f64::max);
        let max_time_abs_ns = residuals
            .iter()
            .map(|residual| residual.max_time_abs_ns)
            .max()
            .unwrap_or(0);
        let cases = residuals
            .iter()
            .map(|residual| {
                json!({
                    "case_id": residual.case_id.as_str(),
                    "rows": residual.rows,
                    "max_position_abs_au": residual.max_position_abs_au,
                    "max_position_abs_m": residual.max_position_abs_m(),
                    "max_velocity_abs_au_per_day": residual.max_velocity_abs_au_per_day,
                    "max_velocity_abs_m_per_s": residual.max_velocity_abs_m_per_s(),
                    "max_time_abs_ns": residual.max_time_abs_ns,
                })
            })
            .collect::<Vec<_>>();
        let artifact = json!({
            "artifact_schema_version": 1,
            "fixture_id": fixture.get("fixture_id").cloned().unwrap_or(Value::Null),
            "acceptance_target": fixture.get("acceptance_target").cloned().unwrap_or(Value::Null),
            "source_fixture_schema_version": fixture.get("fixture_schema_version").cloned().unwrap_or(Value::Null),
            "packages": fixture.get("packages").cloned().unwrap_or(Value::Null),
            "kernels": fixture.get("kernels").cloned().unwrap_or(Value::Null),
            "thresholds": {
                "position_abs_au": ASSIST_FIXTURE_POSITION_TOLERANCE_AU,
                "position_abs_m": ASSIST_FIXTURE_POSITION_TOLERANCE_AU * AU_METERS,
                "velocity_abs_au_per_day": ASSIST_FIXTURE_VELOCITY_TOLERANCE_AU_PER_DAY,
                "velocity_abs_m_per_s": ASSIST_FIXTURE_VELOCITY_TOLERANCE_AU_PER_DAY * AU_METERS / SECONDS_PER_DAY,
            },
            "global_max": {
                "position_abs_au": max_position_abs_au,
                "position_abs_m": max_position_abs_au * AU_METERS,
                "velocity_abs_au_per_day": max_velocity_abs_au_per_day,
                "velocity_abs_m_per_s": max_velocity_abs_au_per_day * AU_METERS / SECONDS_PER_DAY,
                "time_abs_ns": max_time_abs_ns,
            },
            "cases": cases,
        });
        std::fs::write(
            path,
            format!(
                "{}\n",
                serde_json::to_string_pretty(&artifact).expect("residual artifact must serialize")
            ),
        )
        .expect("failed to write residual artifact");
        println!(
            "wrote ASSIST residual artifact {}; max position {max_position_abs_au:e} AU ({:.6e} m), max velocity {max_velocity_abs_au_per_day:e} AU/day ({:.6e} m/s), max time {max_time_abs_ns} ns",
            path.display(),
            max_position_abs_au * AU_METERS,
            max_velocity_abs_au_per_day * AU_METERS / SECONDS_PER_DAY,
        );
    }

    fn assert_orbits_match_fixture(
        actual: &OrbitBatch,
        expected: &Value,
        case_id: &str,
    ) -> FixtureResidual {
        let actual_orbit_ids = actual
            .orbit_id
            .iter()
            .map(|item| item.0.clone())
            .collect::<Vec<_>>();
        assert_string_column(
            &actual_orbit_ids,
            &json_string_array(expected, "orbit_id"),
            "orbit_id",
        );
        assert_object_ids(
            &actual.object_id,
            &json_optional_string_array(expected, "object_id"),
        );
        assert_coordinates_match_fixture(
            &actual.coordinates,
            expected.get("coordinates").unwrap(),
            case_id,
        )
    }

    fn assert_variants_match_fixture(
        actual: &OrbitVariantBatch,
        expected: &Value,
        case_id: &str,
    ) -> FixtureResidual {
        let actual_orbit_ids = actual
            .orbit_id
            .iter()
            .map(|item| item.0.clone())
            .collect::<Vec<_>>();
        assert_string_column(
            &actual_orbit_ids,
            &json_string_array(expected, "orbit_id"),
            "orbit_id",
        );
        assert_object_ids(
            &actual.object_id,
            &json_optional_string_array(expected, "object_id"),
        );
        assert_variant_ids(
            &actual.variant_id,
            &json_optional_string_array(expected, "variant_id"),
        );
        assert_eq!(actual.weights, json_optional_f64_array(expected, "weights"));
        assert_eq!(
            actual.weights_cov,
            json_optional_f64_array(expected, "weights_cov")
        );
        assert_coordinates_match_fixture(
            &actual.coordinates,
            expected.get("coordinates").unwrap(),
            case_id,
        )
    }

    #[test]
    #[ignore = "requires ASSIST kernel env vars and compares against the frozen Python public-semantics fixture"]
    fn live_assist_matches_public_semantics_fixture_propagation_cases() {
        let propagator = live_propagator_from_env();
        let fixture = fixture();
        let cases = fixture
            .get("propagation_cases")
            .and_then(Value::as_array)
            .expect("fixture must contain propagation cases");
        let mut residuals = Vec::with_capacity(cases.len());
        for case in cases {
            let case_id = case
                .get("case_id")
                .and_then(Value::as_str)
                .expect("case must have id");
            let input = case.get("input_orbits").unwrap();
            let targets = json_time_array(case.get("target_times").unwrap());
            let options = PropagationOptions {
                covariance: CovariancePropagation::None,
                thread_limit: Some(1),
                chunk_size: Some(1),
                ..PropagationOptions::default()
            };
            match input
                .get("table_type")
                .and_then(Value::as_str)
                .expect("input must have table_type")
            {
                "Orbits" => {
                    let orbits = json_orbits(input);
                    let request = PropagationRequest::new(&orbits, &targets, options).unwrap();
                    let result = propagator.propagate(&request, &NoopProvider).unwrap();
                    residuals.push(assert_orbits_match_fixture(
                        &result.orbits,
                        case.get("output_orbits").unwrap(),
                        case_id,
                    ));
                }
                "VariantOrbits" => {
                    let variants = json_variants(input);
                    let request =
                        PropagationRequest::new_variants(&variants, &targets, options).unwrap();
                    let result = propagator.propagate(&request, &NoopProvider).unwrap();
                    let actual = result
                        .variants
                        .as_ref()
                        .expect("variant propagation should return variant batch");
                    residuals.push(assert_variants_match_fixture(
                        actual,
                        case.get("output_orbits").unwrap(),
                        case_id,
                    ));
                }
                other => panic!("unexpected fixture table_type {other}"),
            }
        }
        write_residual_artifact(&fixture, &residuals);
        assert_residuals_within_acceptance(&residuals);
    }

    #[test]
    #[ignore = "requires ADAM_CORE_RS_ASSIST_PLANETS_PATH and ADAM_CORE_RS_ASSIST_ASTEROIDS_PATH"]
    fn live_assist_propagates_with_env_kernels() {
        let propagator = live_propagator_from_env();
        let orbits = sample_orbits(
            Frame::Ecliptic,
            OriginId::Named("SUN".to_string()),
            TimeScale::Tdb,
        );
        let targets =
            TimeArray::from_parts(TimeScale::Tdb, vec![60_001, 60_002], vec![0, 0]).unwrap();
        let request = PropagationRequest::new(
            &orbits,
            &targets,
            PropagationOptions {
                covariance: CovariancePropagation::None,
                thread_limit: Some(1),
                chunk_size: Some(1),
                ..PropagationOptions::default()
            },
        )
        .unwrap();
        let result = propagator.propagate(&request, &NoopProvider).unwrap();
        assert_eq!(result.orbits.len(), 2);
        assert!(result.validity.is_valid(0));
        assert!(result.validity.is_valid(1));
    }

    #[test]
    #[ignore = "requires ADAM_CORE_RS_ASSIST_PLANETS_PATH and ADAM_CORE_RS_ASSIST_ASTEROIDS_PATH"]
    fn live_assist_generate_ephemeris_translates_observers_via_spice() {
        let propagator = live_propagator_from_env();
        let planets = std::env::var("ADAM_CORE_RS_ASSIST_PLANETS_PATH").unwrap();
        let mut spice = adam_core_rs_spice::AdamCoreSpiceBackend::new();
        spice
            .furnsh(std::path::Path::new(&planets))
            .expect("furnsh planets kernel");

        let orbits = sample_orbits(
            Frame::Ecliptic,
            OriginId::Named("SUN".to_string()),
            TimeScale::Tdb,
        );
        let observer_times =
            TimeArray::from_parts(TimeScale::Tdb, vec![60_001, 60_002], vec![0, 0]).unwrap();
        // Observers expressed at the solar-system barycenter; the SPICE-backed
        // translation provider shifts them to the orbits' SUN origin natively.
        let observer_coordinates = CoordinateBatch::cartesian(
            vec![[0.0; 6], [0.0; 6]],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::SolarSystemBarycenter, 2),
            Some(observer_times),
            None,
        )
        .unwrap();
        let observers = ObserverBatch::new(
            vec![
                ObservatoryCode("SSB".to_string()),
                ObservatoryCode("SSB".to_string()),
            ],
            observer_coordinates,
        )
        .unwrap();

        let result = propagator
            .generate_ephemeris(
                &orbits,
                &observers,
                &EphemerisOptions::default(),
                &NoopProvider,
                &spice,
            )
            .unwrap();
        assert_eq!(result.ephemeris.len(), 2);
        assert!(result
            .ephemeris
            .coordinates
            .values
            .spherical()
            .expect("ephemeris is spherical")
            .iter()
            .all(|row| row.iter().all(|value| value.is_finite())));
    }
}
