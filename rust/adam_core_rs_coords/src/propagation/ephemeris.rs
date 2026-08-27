use super::request::{CovariancePropagation, EpochPolicy, PropagationOptions, PropagationRequest};
use super::{PropagationError, PropagationResultValue, Propagator};
use crate::ephemeris::{
    generate_ephemeris_2body_flat6, generate_ephemeris_2body_with_covariance_flat6,
};
use crate::photometry::{
    calculate_apparent_magnitude_v_and_phase_angle_flat, calculate_apparent_magnitude_v_flat,
    calculate_phase_angle_flat,
};
use crate::translation::{
    deduplicated_origin_translation_vectors, normalize_coordinates_to, OriginTranslationProvider,
};
use crate::types::time::TimeScaleProvider;
use crate::types::{
    origin_mu_au3_day2, CoordinateBatch, CoordinateRepresentation, CovarianceBatch,
    CovarianceUnits, EphemerisBatch, Frame, ObserverBatch, OriginArray, OriginId, TimeArray,
    TimeScale, Validity, NANOS_PER_DAY,
};
use crate::{Epoch, OrbitBatch};

#[derive(Debug, Clone, PartialEq)]
pub struct EphemerisOptions {
    pub propagation: PropagationOptions,
    pub lt_tol: f64,
    pub max_iter: usize,
    pub tol: f64,
    pub stellar_aberration: bool,
    pub max_lt_iter: usize,
    pub output_time_scale: TimeScale,
    pub include_aberrated_coordinates: bool,
    pub photometry: EphemerisPhotometryOptions,
}

impl Default for EphemerisOptions {
    fn default() -> Self {
        Self {
            propagation: PropagationOptions {
                covariance: CovariancePropagation::None,
                ..PropagationOptions::default()
            },
            lt_tol: 1.0e-12,
            max_iter: 1_000,
            tol: 1.0e-15,
            stellar_aberration: false,
            max_lt_iter: 10,
            output_time_scale: TimeScale::Utc,
            include_aberrated_coordinates: true,
            photometry: EphemerisPhotometryOptions::default(),
        }
    }
}

impl EphemerisOptions {
    pub fn validate(&self, orbit_rows: usize) -> PropagationResultValue<()> {
        self.propagation.validate()?;
        if matches!(self.propagation.epoch_policy, EpochPolicy::PerOrbit { .. }) {
            return Err(PropagationError::InvalidRequest(
                "typed ephemeris generation does not support PerOrbit propagation".to_string(),
            ));
        }
        if !matches!(
            self.propagation.covariance,
            CovariancePropagation::None | CovariancePropagation::Linearized
        ) {
            return Err(PropagationError::InvalidRequest(
                "typed ephemeris generation supports covariance=None or linearized only"
                    .to_string(),
            ));
        }
        if !self.lt_tol.is_finite() || self.lt_tol <= 0.0 {
            return Err(PropagationError::InvalidRequest(
                "EphemerisOptions.lt_tol must be finite and positive".to_string(),
            ));
        }
        if self.max_iter == 0 {
            return Err(PropagationError::InvalidRequest(
                "EphemerisOptions.max_iter must be positive".to_string(),
            ));
        }
        if !self.tol.is_finite() || self.tol <= 0.0 {
            return Err(PropagationError::InvalidRequest(
                "EphemerisOptions.tol must be finite and positive".to_string(),
            ));
        }
        if self.max_lt_iter == 0 {
            return Err(PropagationError::InvalidRequest(
                "EphemerisOptions.max_lt_iter must be positive".to_string(),
            ));
        }
        self.photometry.validate(orbit_rows)
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct EphemerisPhotometryOptions {
    pub predict_magnitude_v: bool,
    pub predict_phase_angle: bool,
    pub h_v: Option<Vec<Option<f64>>>,
    pub g: Option<Vec<Option<f64>>>,
}

impl EphemerisPhotometryOptions {
    fn validate(&self, orbit_rows: usize) -> PropagationResultValue<()> {
        if self.predict_magnitude_v && (self.h_v.is_none() || self.g.is_none()) {
            return Err(PropagationError::InvalidRequest(
                "predict_magnitude_v requires per-orbit H_v and G values".to_string(),
            ));
        }
        validate_optional_photometry_column("h_v", self.h_v.as_ref(), orbit_rows)?;
        validate_optional_photometry_column("g", self.g.as_ref(), orbit_rows)?;
        Ok(())
    }

    fn any_requested(&self) -> bool {
        self.predict_magnitude_v || self.predict_phase_angle
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EphemerisFailureCode {
    PropagationRowFailure,
    NonFiniteObserverState,
    LightTimeNonConvergence,
    NonFiniteEphemerisState,
    NonFiniteAberratedState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EphemerisRowDiagnostic {
    pub output_row: usize,
    pub input_orbit_index: usize,
    pub observer_index: usize,
    pub failure_code: Option<EphemerisFailureCode>,
    pub message: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EphemerisDiagnostics {
    pub rows: Vec<EphemerisRowDiagnostic>,
}

impl EphemerisDiagnostics {
    pub fn failed_rows(&self) -> impl Iterator<Item = &EphemerisRowDiagnostic> {
        self.rows.iter().filter(|row| row.failure_code.is_some())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EphemerisResult {
    pub ephemeris: EphemerisBatch,
    pub diagnostics: EphemerisDiagnostics,
}

pub fn generate_ephemeris<P: Propagator>(
    propagator: &P,
    orbits: &OrbitBatch,
    observers: &ObserverBatch,
    options: &EphemerisOptions,
    provider: &dyn TimeScaleProvider,
) -> PropagationResultValue<EphemerisResult> {
    generate_ephemeris_impl(propagator, orbits, observers, options, provider, None)
}

/// Like [`generate_ephemeris`], but performs the light-time geometry in the
/// solar-system-barycentric (SSB) frame: the propagated object and observer
/// are translated to SSB via `translation_provider`, and the SSB gravitational
/// parameter is used for the light-time 2-body sub-step. This matches the public
/// Python ephemeris composition, whose barycentric normalization precedes both
/// origin-mu lookup and `add_light_time`. Aberrated coordinates are emitted in
/// SSB.
pub fn generate_ephemeris_barycentric<P, T>(
    propagator: &P,
    orbits: &OrbitBatch,
    observers: &ObserverBatch,
    options: &EphemerisOptions,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<EphemerisResult>
where
    P: Propagator,
    T: OriginTranslationProvider,
{
    generate_ephemeris_impl(
        propagator,
        orbits,
        observers,
        options,
        provider,
        Some(translation_provider as &dyn OriginTranslationProvider),
    )
}

fn generate_ephemeris_impl<P: Propagator>(
    propagator: &P,
    orbits: &OrbitBatch,
    observers: &ObserverBatch,
    options: &EphemerisOptions,
    provider: &dyn TimeScaleProvider,
    barycentric: Option<&dyn OriginTranslationProvider>,
) -> PropagationResultValue<EphemerisResult> {
    orbits.validate()?;
    observers.validate()?;
    options.validate(orbits.len())?;
    validate_ephemeris_input_contract(orbits, observers, options, barycentric.is_none())?;

    let observer_times = observers.coordinates.times.as_ref().ok_or_else(|| {
        PropagationError::InvalidRequest("observer coordinates require times".to_string())
    })?;
    let propagation_request =
        PropagationRequest::new(orbits, observer_times, options.propagation.clone())?;
    let pairwise = options.propagation.epoch_policy == EpochPolicy::Pairwise;
    let (propagated_orbits, propagation_validity) = if pairwise {
        // The public pairwise contract receives orbits already propagated to
        // the paired observer epochs. PropagationRequest above owns length and
        // time-scale validation; avoid an unnecessary zero-dt solver pass.
        (orbits.clone(), Validity::all_valid(orbits.len()))
    } else {
        let propagation = propagator.propagate(&propagation_request, provider)?;
        (propagation.orbits, propagation.validity)
    };
    let output_rows = propagated_orbits.len();
    let observer_rows = observers.len();
    let propagated_covariance = propagated_orbits.coordinates.covariance.as_ref();
    let transport_covariance = options.propagation.covariance == CovariancePropagation::Linearized
        && propagated_covariance.is_some();

    let observer_output_times = rescale_output_times(
        observer_times,
        options.output_time_scale,
        provider,
        "observer output times",
    )?;
    let propagated_times = propagated_orbits
        .coordinates
        .times
        .as_ref()
        .ok_or_else(|| {
            PropagationError::InvalidRequest(
                "propagator output is missing coordinate times".to_string(),
            )
        })?;
    let propagated_states = propagated_orbits
        .coordinates
        .values
        .cartesian()
        .ok_or_else(|| {
            PropagationError::InvalidRequest(
                "propagator output must be Cartesian for ephemeris generation".to_string(),
            )
        })?;
    let observer_states = observers.coordinates.values.cartesian().ok_or_else(|| {
        PropagationError::InvalidRequest(
            "observer coordinates must be Cartesian for ephemeris generation".to_string(),
        )
    })?;

    // Barycentric geometry: translate the propagated object and the observer to
    // SSB so the light-time 2-body sub-step uses barycentric velocity (Python
    // parity). The SSB gravitational parameter is selected below.
    let barycentric_orbit_offsets = match barycentric {
        Some(translation_provider) => Some(deduplicated_origin_translation_vectors(
            translation_provider,
            &propagated_orbits.coordinates.origins,
            &OriginId::SolarSystemBarycenter,
            Frame::Ecliptic,
            propagated_times,
        )?),
        None => None,
    };
    let barycentric_observer_offsets = match barycentric {
        Some(_)
            if propagated_orbits.coordinates.origins == observers.coordinates.origins
                && propagated_times == observer_times =>
        {
            barycentric_orbit_offsets.clone()
        }
        Some(translation_provider) => Some(deduplicated_origin_translation_vectors(
            translation_provider,
            &observers.coordinates.origins,
            &OriginId::SolarSystemBarycenter,
            Frame::Ecliptic,
            observer_times,
        )?),
        None => None,
    };

    let mut spherical_states = Vec::with_capacity(output_rows);
    let mut spherical_covariance_values =
        transport_covariance.then(|| Vec::with_capacity(output_rows * 36));
    let mut spherical_covariance_validity =
        transport_covariance.then(|| Vec::with_capacity(output_rows));
    let mut light_time_days = Vec::with_capacity(output_rows);
    let mut aberrated_states = Vec::with_capacity(output_rows);
    let mut aberrated_epochs = Vec::with_capacity(output_rows);
    let mut coordinate_epochs = Vec::with_capacity(output_rows);
    let mut spherical_origins = Vec::with_capacity(output_rows);
    let mut aberrated_origins = Vec::with_capacity(output_rows);
    let mut validity = Vec::with_capacity(output_rows);
    let mut diagnostics = Vec::with_capacity(output_rows);
    let mut object_pos_flat = Vec::with_capacity(output_rows * 3);
    let mut observer_pos_flat = Vec::with_capacity(output_rows * 3);
    let mut h_v_rows = Vec::with_capacity(output_rows);
    let mut g_rows = Vec::with_capacity(output_rows);

    let mut geometry_objects_flat = Vec::with_capacity(output_rows * 6);
    let mut geometry_observers_flat = Vec::with_capacity(output_rows * 6);
    let mut mus = Vec::with_capacity(output_rows);
    let mut input_failures = Vec::with_capacity(output_rows);
    let mut covariance_row_validity = Vec::with_capacity(output_rows);
    let mut covariance_input_flat =
        transport_covariance.then(|| Vec::with_capacity(output_rows * 36));

    for (output_row, propagated_state) in propagated_states.iter().copied().enumerate() {
        let observer_index = if pairwise {
            output_row
        } else {
            output_row % observer_rows
        };
        let observer_state = observer_states[observer_index];
        let propagated_origin = &propagated_orbits.coordinates.origins.origins[output_row];
        let mut row_failure = if propagation_validity.is_valid(output_row) {
            None
        } else {
            Some(EphemerisFailureCode::PropagationRowFailure)
        };
        if row_failure.is_none() && !state_is_finite(&observer_state) {
            row_failure = Some(EphemerisFailureCode::NonFiniteObserverState);
        }
        input_failures.push(row_failure);

        let mut geometry_object = propagated_state;
        let mut geometry_observer = observer_state;
        if let (Some(orbit_offsets), Some(observer_offsets)) =
            (&barycentric_orbit_offsets, &barycentric_observer_offsets)
        {
            for axis in 0..6 {
                geometry_object[axis] += orbit_offsets[output_row][axis];
                geometry_observer[axis] += observer_offsets[observer_index][axis];
            }
        }
        geometry_objects_flat.extend_from_slice(&geometry_object);
        geometry_observers_flat.extend_from_slice(&geometry_observer);
        let light_time_origin = if barycentric.is_some() {
            &OriginId::SolarSystemBarycenter
        } else {
            propagated_origin
        };
        mus.push(origin_mu_au3_day2(light_time_origin)?);

        let covariance_valid =
            propagated_covariance.is_some_and(|covariance| covariance.is_row_valid(output_row));
        covariance_row_validity.push(covariance_valid);
        if let Some(values) = &mut covariance_input_flat {
            if covariance_valid {
                values.extend_from_slice(
                    propagated_covariance
                        .expect("transported covariance is present")
                        .row_values(output_row),
                );
            } else {
                values.extend_from_slice(&[f64::NAN; 36]);
            }
        }
    }

    let (spherical_flat, light_time_output, aberrated_flat, covariance_output) =
        match covariance_input_flat {
            Some(covariance_input) => {
                let (spherical, light_time, aberrated, covariance) =
                    generate_ephemeris_2body_with_covariance_flat6(
                        &geometry_objects_flat,
                        &covariance_input,
                        &geometry_observers_flat,
                        &mus,
                        options.lt_tol,
                        options.max_iter,
                        options.tol,
                        options.stellar_aberration,
                        options.max_lt_iter,
                    );
                (spherical, light_time, aberrated, Some(covariance))
            }
            None => {
                let (spherical, light_time, aberrated) = generate_ephemeris_2body_flat6(
                    &geometry_objects_flat,
                    &geometry_observers_flat,
                    &mus,
                    options.lt_tol,
                    options.max_iter,
                    options.tol,
                    options.stellar_aberration,
                    options.max_lt_iter,
                );
                (spherical, light_time, aberrated, None)
            }
        };

    for output_row in 0..output_rows {
        let input_orbit_index = if pairwise {
            output_row
        } else {
            output_row / observer_rows
        };
        let observer_index = if pairwise {
            output_row
        } else {
            output_row % observer_rows
        };
        let spherical = std::array::from_fn(|axis| spherical_flat[output_row * 6 + axis]);
        let light_time = light_time_output[output_row];
        let aberrated = std::array::from_fn(|axis| aberrated_flat[output_row * 6 + axis]);
        let propagated_origin = &propagated_orbits.coordinates.origins.origins[output_row];
        let observer_origin = &observers.coordinates.origins.origins[observer_index];
        let observer_time = observer_output_times.epochs[observer_index];
        let propagated_time = propagated_times.epochs[output_row];
        let observer_state = observer_states[observer_index];
        let mut row_failure = input_failures[output_row];
        if row_failure.is_none() {
            row_failure = ephemeris_failure_code(&spherical, light_time, &aberrated);
        }

        let valid = row_failure.is_none();
        if let Some(values) = &mut spherical_covariance_values {
            let covariance = covariance_output
                .as_ref()
                .expect("covariance output is present");
            values.extend_from_slice(&covariance[output_row * 36..(output_row + 1) * 36]);
        }
        if let Some(covariance_validity) = &mut spherical_covariance_validity {
            covariance_validity.push(valid && covariance_row_validity[output_row]);
        }
        let emission_epoch = if light_time.is_finite() {
            epoch_add_fractional_days(propagated_time, -light_time)
        } else {
            propagated_time
        };

        spherical_states.push(spherical);
        light_time_days.push(light_time);
        aberrated_states.push(aberrated);
        aberrated_epochs.push(emission_epoch);
        coordinate_epochs.push(observer_time);
        spherical_origins.push(OriginId::Named(observers.code[observer_index].0.clone()));
        aberrated_origins.push(if barycentric.is_some() {
            OriginId::SolarSystemBarycenter
        } else {
            propagated_origin.clone()
        });
        validity.push(valid);
        diagnostics.push(EphemerisRowDiagnostic {
            output_row,
            input_orbit_index,
            observer_index,
            failure_code: row_failure,
            message: row_failure.map(ephemeris_failure_message),
        });

        object_pos_flat.extend_from_slice(&aberrated[..3]);
        observer_pos_flat.extend_from_slice(&observer_state[..3]);
        h_v_rows.push(photometry_value_for_row(
            options.photometry.h_v.as_ref(),
            input_orbit_index,
        ));
        g_rows.push(photometry_value_for_row(
            options.photometry.g.as_ref(),
            input_orbit_index,
        ));

        if barycentric.is_none() && propagated_origin != observer_origin {
            return Err(PropagationError::InvalidRequest(
                "internal ephemeris origin mismatch after request validation".to_string(),
            ));
        }
    }

    // H-G photometry must use HELIOCENTRIC positions (Python parity:
    // `attach_magnitude_or_phase` transforms the aberrated emission-time
    // coordinates and the observers to origin=SUN before the kernels). In the
    // barycentric workflow the aberrated states are SSB and observers keep
    // their input origin, so translate both to SUN here; the SUN-origin-only
    // non-barycentric path is already heliocentric.
    if let (Some(translation_provider), true) = (barycentric, options.photometry.any_requested()) {
        let sun = OriginId::Naif(10);
        let emission_times = TimeArray::new(propagated_times.scale, aberrated_epochs.clone())
            .map_err(|err| {
                PropagationError::InvalidRequest(format!(
                    "invalid emission epochs for photometry: {err}"
                ))
            })?;
        let ssb_origins = OriginArray::repeat(OriginId::SolarSystemBarycenter, output_rows);
        let object_to_sun = translation_provider.origin_translation_vectors(
            &ssb_origins,
            &sun,
            Frame::Ecliptic,
            &emission_times,
        )?;
        let observer_to_sun = translation_provider.origin_translation_vectors(
            &observers.coordinates.origins,
            &sun,
            Frame::Ecliptic,
            observer_times,
        )?;
        for row in 0..output_rows {
            let observer_index = row % observer_rows;
            for axis in 0..3 {
                object_pos_flat[row * 3 + axis] += object_to_sun[row][axis];
                observer_pos_flat[row * 3 + axis] += observer_to_sun[observer_index][axis];
            }
        }
    }

    let (predicted_magnitude_v, alpha_deg) = compute_photometry(
        &options.photometry,
        &h_v_rows,
        &object_pos_flat,
        &observer_pos_flat,
        &g_rows,
    );

    let coordinates = build_ephemeris_coordinates(
        spherical_states,
        options.output_time_scale,
        coordinate_epochs,
        spherical_origins,
        spherical_covariance_values,
        spherical_covariance_validity,
    )?;
    let aberrated_coordinates = build_aberrated_coordinates(
        options,
        aberrated_states,
        propagated_times.scale,
        aberrated_epochs,
        aberrated_origins,
    )?;
    let ephemeris = EphemerisBatch::new(
        propagated_orbits.orbit_id,
        propagated_orbits.object_id,
        coordinates,
        predicted_magnitude_v,
        alpha_deg,
        light_time_days,
        aberrated_coordinates,
        Validity::from_bools(&validity),
    )?;
    Ok(EphemerisResult {
        ephemeris,
        diagnostics: EphemerisDiagnostics { rows: diagnostics },
    })
}

fn compute_photometry(
    options: &EphemerisPhotometryOptions,
    h_v_rows: &[f64],
    object_pos_flat: &[f64],
    observer_pos_flat: &[f64],
    g_rows: &[f64],
) -> (Option<Vec<f64>>, Option<Vec<f64>>) {
    match (options.predict_magnitude_v, options.predict_phase_angle) {
        (true, true) => {
            let (magnitude, alpha) = calculate_apparent_magnitude_v_and_phase_angle_flat(
                h_v_rows,
                object_pos_flat,
                observer_pos_flat,
                g_rows,
            );
            (Some(magnitude), Some(alpha))
        }
        (true, false) => (
            Some(calculate_apparent_magnitude_v_flat(
                h_v_rows,
                object_pos_flat,
                observer_pos_flat,
                g_rows,
            )),
            None,
        ),
        (false, true) => (
            None,
            Some(calculate_phase_angle_flat(
                object_pos_flat,
                observer_pos_flat,
            )),
        ),
        (false, false) => (None, None),
    }
}

fn build_ephemeris_coordinates(
    spherical_states: Vec<[f64; 6]>,
    scale: TimeScale,
    epochs: Vec<Epoch>,
    origins: Vec<OriginId>,
    covariance_values: Option<Vec<f64>>,
    covariance_validity: Option<Vec<bool>>,
) -> PropagationResultValue<CoordinateBatch> {
    let covariance = match covariance_values {
        Some(values) => {
            let rows = spherical_states.len();
            let covariance = CovarianceBatch::new(
                rows,
                6,
                values,
                CovarianceUnits::Coordinate(CoordinateRepresentation::Spherical),
            )?;
            Some(match covariance_validity {
                Some(validity) => covariance.with_row_validity(Validity::from_bools(&validity))?,
                None => covariance,
            })
        }
        None => None,
    };
    CoordinateBatch::spherical(
        spherical_states,
        Frame::Equatorial,
        OriginArray::new(origins),
        Some(TimeArray::new(scale, epochs)?),
        covariance,
    )
    .map_err(PropagationError::from)
}

fn build_aberrated_coordinates(
    options: &EphemerisOptions,
    aberrated_states: Vec<[f64; 6]>,
    scale: TimeScale,
    epochs: Vec<Epoch>,
    origins: Vec<OriginId>,
) -> PropagationResultValue<Option<CoordinateBatch>> {
    if !options.include_aberrated_coordinates {
        return Ok(None);
    }
    CoordinateBatch::cartesian(
        aberrated_states,
        Frame::Ecliptic,
        OriginArray::new(origins),
        Some(TimeArray::new(scale, epochs)?),
        None,
    )
    .map(Some)
    .map_err(PropagationError::from)
}

/// Generate ephemerides while translating the observers to the orbits' origin
/// and the ecliptic frame natively, via `translation_provider`.
///
/// This is the Rust-native replacement for the Python observer-state /
/// `get_perturber_state` glue: the orbits are propagated in their own origin
/// (so 2-body central-body dynamics are unaffected), the observers are rotated
/// to ecliptic and shifted to the orbits' origin, and then [`generate_ephemeris`]
/// runs with its same-origin ecliptic contract satisfied by construction.
///
/// Orbits must already be ecliptic and share a single origin (rotate them first
/// if needed); only the observer normalization needs the SPICE-backed provider.
pub fn generate_ephemeris_translated<P, T>(
    propagator: &P,
    orbits: &OrbitBatch,
    observers: &ObserverBatch,
    options: &EphemerisOptions,
    time_provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<EphemerisResult>
where
    P: Propagator,
    T: OriginTranslationProvider + ?Sized,
{
    if orbits.coordinates.frame != Frame::Ecliptic {
        return Err(PropagationError::InvalidRequest(
            "generate_ephemeris_translated requires ecliptic orbit coordinates".to_string(),
        ));
    }
    let target_origin = orbits
        .coordinates
        .origins
        .origins
        .first()
        .cloned()
        .ok_or_else(|| {
            PropagationError::InvalidRequest(
                "ephemeris generation requires at least one orbit".to_string(),
            )
        })?;
    if orbits
        .coordinates
        .origins
        .origins
        .iter()
        .any(|origin| origin != &target_origin)
    {
        return Err(PropagationError::InvalidRequest(
            "generate_ephemeris_translated requires orbits to share a single origin".to_string(),
        ));
    }

    let observer_coordinates = normalize_coordinates_to(
        &observers.coordinates,
        &target_origin,
        Frame::Ecliptic,
        translation_provider,
    )?;
    let translated_observers = ObserverBatch::new(observers.code.clone(), observer_coordinates)?;
    generate_ephemeris(
        propagator,
        orbits,
        &translated_observers,
        options,
        time_provider,
    )
}

fn validate_ephemeris_input_contract(
    orbits: &OrbitBatch,
    observers: &ObserverBatch,
    options: &EphemerisOptions,
    require_same_origin: bool,
) -> PropagationResultValue<()> {
    if orbits.coordinates.frame != Frame::Ecliptic || observers.coordinates.frame != Frame::Ecliptic
    {
        return Err(PropagationError::InvalidRequest(
            "typed ephemeris generation currently requires ecliptic Cartesian orbit and observer coordinates"
                .to_string(),
        ));
    }
    if require_same_origin
        && !origins_share_single_origin(&orbits.coordinates.origins, &observers.coordinates.origins)
    {
        return Err(PropagationError::InvalidRequest(
            "typed ephemeris generation requires orbit and observer coordinates in the same origin until origin-state translation is wired"
                .to_string(),
        ));
    }
    if require_same_origin
        && options.photometry.any_requested()
        && !orbits
            .coordinates
            .origins
            .origins
            .first()
            .is_some_and(is_sun_origin)
    {
        return Err(PropagationError::InvalidRequest(
            "typed ephemeris photometry requires heliocentric SUN-origin coordinates until origin-state translation is wired"
                .to_string(),
        ));
    }
    Ok(())
}

fn origins_share_single_origin(left: &OriginArray, right: &OriginArray) -> bool {
    let Some(first) = left.origins.first().or_else(|| right.origins.first()) else {
        return true;
    };
    left.origins.iter().all(|origin| origin == first)
        && right.origins.iter().all(|origin| origin == first)
}

fn is_sun_origin(origin: &OriginId) -> bool {
    match origin {
        OriginId::Naif(10) => true,
        OriginId::Named(code) => code == "SUN",
        _ => false,
    }
}

fn rescale_output_times(
    times: &TimeArray,
    scale: TimeScale,
    provider: &dyn TimeScaleProvider,
    field: &str,
) -> PropagationResultValue<TimeArray> {
    let out = times.rescale_with_provider(scale, provider)?;
    out.validate()?;
    if out.scale != scale {
        return Err(PropagationError::InvalidRequest(format!(
            "{field} provider returned {} instead of required {}",
            out.scale.as_str(),
            scale.as_str()
        )));
    }
    if out.len() != times.len() {
        return Err(PropagationError::InvalidRequest(format!(
            "{field} provider changed length from {} to {}",
            times.len(),
            out.len()
        )));
    }
    Ok(out)
}

fn validate_optional_photometry_column(
    field: &str,
    values: Option<&Vec<Option<f64>>>,
    expected: usize,
) -> PropagationResultValue<()> {
    let Some(values) = values else {
        return Ok(());
    };
    if values.len() != expected {
        return Err(PropagationError::InvalidRequest(format!(
            "{field} must have one value per orbit; expected {expected}, got {}",
            values.len()
        )));
    }
    for (row, value) in values.iter().enumerate() {
        if value.is_some_and(|value| !value.is_finite()) {
            return Err(PropagationError::InvalidRequest(format!(
                "{field} row {row} must be finite when present"
            )));
        }
    }
    Ok(())
}

fn photometry_value_for_row(values: Option<&Vec<Option<f64>>>, orbit_index: usize) -> f64 {
    values
        .and_then(|values| values[orbit_index])
        .unwrap_or(f64::NAN)
}

fn state_is_finite(state: &[f64; 6]) -> bool {
    state.iter().all(|value| value.is_finite())
}

fn ephemeris_failure_code(
    spherical: &[f64; 6],
    light_time: f64,
    aberrated: &[f64; 6],
) -> Option<EphemerisFailureCode> {
    if !light_time.is_finite() {
        return Some(EphemerisFailureCode::LightTimeNonConvergence);
    }
    if !state_is_finite(spherical) {
        return Some(EphemerisFailureCode::NonFiniteEphemerisState);
    }
    if !state_is_finite(aberrated) {
        return Some(EphemerisFailureCode::NonFiniteAberratedState);
    }
    None
}

fn ephemeris_failure_message(code: EphemerisFailureCode) -> String {
    match code {
        EphemerisFailureCode::PropagationRowFailure => {
            "initial propagation failed for this ephemeris row".to_string()
        }
        EphemerisFailureCode::NonFiniteObserverState => {
            "observer state was non-finite for this ephemeris row".to_string()
        }
        EphemerisFailureCode::LightTimeNonConvergence => {
            "light-time correction failed to converge for this ephemeris row".to_string()
        }
        EphemerisFailureCode::NonFiniteEphemerisState => {
            "ephemeris spherical state was non-finite for this row".to_string()
        }
        EphemerisFailureCode::NonFiniteAberratedState => {
            "aberrated Cartesian state was non-finite for this row".to_string()
        }
    }
}

fn epoch_add_fractional_days(epoch: Epoch, delta_days: f64) -> Epoch {
    let delta_nanos = (delta_days * NANOS_PER_DAY as f64).round() as i64;
    Epoch::new(epoch.days, epoch.nanos + delta_nanos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ephemeris::generate_ephemeris_2body_row;
    use crate::propagation::{TwoBodyPropagator, TwoBodyPropagatorConfig};
    use crate::types::{SchemaError, SchemaResult};
    use crate::{ObjectId, ObservatoryCode, OrbitId};

    struct NoopProvider;

    impl TimeScaleProvider for NoopProvider {
        fn rescale(&self, _times: &TimeArray, _new_scale: TimeScale) -> SchemaResult<TimeArray> {
            Err(SchemaError::InvalidRecordBatch(
                "test provider should not be called".to_string(),
            ))
        }
    }

    fn sample_times(rows: usize) -> TimeArray {
        TimeArray::new(
            TimeScale::Tdb,
            (0..rows)
                .map(|row| Epoch::new(60_000 + row as i64, 0))
                .collect(),
        )
        .unwrap()
    }

    fn sample_orbits() -> OrbitBatch {
        let coordinates = CoordinateBatch::cartesian(
            vec![
                [1.0, 0.2, 0.1, 0.001, 0.015, 0.0005],
                [1.2, -0.1, 0.05, -0.002, 0.014, 0.0002],
            ],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::Named("SUN".to_string()), 2),
            Some(sample_times(2)),
            None,
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

    fn sample_observers() -> ObserverBatch {
        let coordinates = CoordinateBatch::cartesian(
            vec![
                [0.9, -0.2, 0.0, 0.002, 0.017, 0.0],
                [1.1, 0.1, 0.02, -0.001, 0.016, 0.0001],
            ],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::Named("SUN".to_string()), 2),
            Some(sample_times(2)),
            None,
        )
        .unwrap();
        ObserverBatch::new(
            vec![
                ObservatoryCode("500".to_string()),
                ObservatoryCode("X05".to_string()),
            ],
            coordinates,
        )
        .unwrap()
    }

    #[test]
    fn translated_ephemeris_matches_manually_translated_observers() {
        use crate::translation::OriginTranslationProvider;

        struct StubOriginProvider {
            vectors: Vec<[f64; 6]>,
        }
        impl OriginTranslationProvider for StubOriginProvider {
            fn origin_translation_vectors(
                &self,
                origins: &OriginArray,
                _target_origin: &OriginId,
                _frame: Frame,
                times: &TimeArray,
            ) -> SchemaResult<Vec<[f64; 6]>> {
                assert_eq!(origins.len(), times.len());
                Ok(self.vectors.clone())
            }
        }

        let orbits = sample_orbits();
        let observer_states = vec![
            [0.9, -0.2, 0.0, 0.002, 0.017, 0.0],
            [1.1, 0.1, 0.02, -0.001, 0.016, 0.0001],
        ];
        let codes = vec![
            ObservatoryCode("500".to_string()),
            ObservatoryCode("X05".to_string()),
        ];
        let ssb_observers = ObserverBatch::new(
            codes.clone(),
            CoordinateBatch::cartesian(
                observer_states.clone(),
                Frame::Ecliptic,
                OriginArray::repeat(OriginId::SolarSystemBarycenter, 2),
                Some(sample_times(2)),
                None,
            )
            .unwrap(),
        )
        .unwrap();
        let vectors = vec![
            [0.005, -0.003, 0.001, 1.0e-6, -2.0e-6, 3.0e-7],
            [0.004, -0.002, 0.0015, 2.0e-6, -1.0e-6, 1.0e-7],
        ];
        let provider = StubOriginProvider {
            vectors: vectors.clone(),
        };
        let options = EphemerisOptions::default();

        let translated = generate_ephemeris_translated(
            &TwoBodyPropagator::default(),
            &orbits,
            &ssb_observers,
            &options,
            &NoopProvider,
            &provider,
        )
        .unwrap();

        let manual_states: Vec<[f64; 6]> = observer_states
            .iter()
            .zip(vectors.iter())
            .map(|(state, vector)| {
                let mut out = *state;
                for index in 0..6 {
                    out[index] += vector[index];
                }
                out
            })
            .collect();
        let sun_observers = ObserverBatch::new(
            codes,
            CoordinateBatch::cartesian(
                manual_states,
                Frame::Ecliptic,
                OriginArray::repeat(OriginId::Named("SUN".to_string()), 2),
                Some(sample_times(2)),
                None,
            )
            .unwrap(),
        )
        .unwrap();
        let expected = generate_ephemeris(
            &TwoBodyPropagator::default(),
            &orbits,
            &sun_observers,
            &options,
            &NoopProvider,
        )
        .unwrap();

        assert_eq!(translated, expected);
    }

    #[test]
    fn typed_generate_ephemeris_matches_low_level_two_body_rows() {
        let orbits = sample_orbits();
        let observers = sample_observers();
        let options = EphemerisOptions {
            output_time_scale: TimeScale::Tdb,
            photometry: EphemerisPhotometryOptions {
                predict_magnitude_v: true,
                predict_phase_angle: true,
                h_v: Some(vec![Some(18.0), Some(19.0)]),
                g: Some(vec![Some(0.15), Some(0.2)]),
            },
            ..EphemerisOptions::default()
        };
        let result = generate_ephemeris(
            &TwoBodyPropagator::default(),
            &orbits,
            &observers,
            &options,
            &NoopProvider,
        )
        .unwrap();

        assert_eq!(result.ephemeris.len(), 4);
        assert_eq!(result.ephemeris.coordinates.frame, Frame::Equatorial);
        assert_eq!(
            result.ephemeris.coordinates.times.as_ref().unwrap().scale,
            TimeScale::Tdb
        );
        assert_eq!(
            result.ephemeris.coordinates.origins.origins[0].code(),
            "500"
        );
        assert_eq!(
            result.ephemeris.coordinates.origins.origins[1].code(),
            "X05"
        );
        assert!(result
            .ephemeris
            .predicted_magnitude_v
            .as_ref()
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
        assert!(result
            .ephemeris
            .alpha_deg
            .as_ref()
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
        for row in 0..result.ephemeris.validity.len() {
            assert!(result.ephemeris.validity.is_valid(row));
        }

        let propagated_request = PropagationRequest::new(
            &orbits,
            observers.coordinates.times.as_ref().unwrap(),
            options.propagation.clone(),
        )
        .unwrap();
        let propagated = TwoBodyPropagator::default()
            .propagate(&propagated_request, &NoopProvider)
            .unwrap();
        let propagated_states = propagated.orbits.coordinates.values.cartesian().unwrap();
        let observer_states = observers.coordinates.values.cartesian().unwrap();
        let spherical = result.ephemeris.coordinates.values.spherical().unwrap();
        let aberrated = result
            .ephemeris
            .aberrated_coordinates
            .as_ref()
            .unwrap()
            .values
            .cartesian()
            .unwrap();
        let mu = origin_mu_au3_day2(&OriginId::Named("SUN".to_string())).unwrap();
        for row in 0..result.ephemeris.len() {
            let observer_index = row % observers.len();
            let (expected_spherical, expected_lt, expected_aberrated) =
                generate_ephemeris_2body_row::<f64>(
                    propagated_states[row],
                    observer_states[observer_index],
                    mu,
                    options.lt_tol,
                    options.max_iter,
                    options.tol,
                    options.stellar_aberration,
                    options.max_lt_iter,
                );
            for column in 0..6 {
                assert!(
                    (spherical[row][column] - expected_spherical[column]).abs() < 1.0e-13,
                    "spherical row {row} column {column}: {} vs {}",
                    spherical[row][column],
                    expected_spherical[column]
                );
                assert!(
                    (aberrated[row][column] - expected_aberrated[column]).abs() < 1.0e-13,
                    "aberrated row {row} column {column}: {} vs {}",
                    aberrated[row][column],
                    expected_aberrated[column]
                );
            }
            assert!((result.ephemeris.light_time_days[row] - expected_lt).abs() < 1.0e-14);
        }
    }

    #[test]
    fn typed_generate_ephemeris_records_observer_row_failures() {
        let orbits = sample_orbits();
        let mut observers = sample_observers();
        let states = observers
            .coordinates
            .values
            .cartesian()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(row, state)| {
                let mut state = *state;
                if row == 1 {
                    state[0] = f64::NAN;
                }
                state
            })
            .collect::<Vec<_>>();
        observers.coordinates = CoordinateBatch::cartesian(
            states,
            Frame::Ecliptic,
            observers.coordinates.origins.clone(),
            observers.coordinates.times.clone(),
            None,
        )
        .unwrap();

        let options = EphemerisOptions {
            output_time_scale: TimeScale::Tdb,
            ..EphemerisOptions::default()
        };
        let result = generate_ephemeris(
            &TwoBodyPropagator::default(),
            &orbits,
            &observers,
            &options,
            &NoopProvider,
        )
        .unwrap();

        assert!(result.ephemeris.validity.is_valid(0));
        assert!(!result.ephemeris.validity.is_valid(1));
        assert!(result.ephemeris.validity.is_valid(2));
        assert!(!result.ephemeris.validity.is_valid(3));
        let failures = result.diagnostics.failed_rows().collect::<Vec<_>>();
        assert_eq!(failures.len(), 2);
        assert!(failures
            .iter()
            .all(|row| { row.failure_code == Some(EphemerisFailureCode::NonFiniteObserverState) }));
    }

    #[test]
    fn typed_generate_ephemeris_rejects_origin_translation_gap() {
        let orbits = sample_orbits();
        let mut observers = sample_observers();
        observers.coordinates.origins = OriginArray::repeat(OriginId::SolarSystemBarycenter, 2);
        let err = generate_ephemeris(
            &TwoBodyPropagator::default(),
            &orbits,
            &observers,
            &EphemerisOptions::default(),
            &NoopProvider,
        )
        .unwrap_err();
        assert!(
            matches!(err, PropagationError::InvalidRequest(message) if message.contains("same origin"))
        );
    }

    #[test]
    fn typed_generate_ephemeris_photometry_requires_sun_origin() {
        let mut orbits = sample_orbits();
        let mut observers = sample_observers();
        orbits.coordinates.origins = OriginArray::repeat(OriginId::SolarSystemBarycenter, 2);
        observers.coordinates.origins = OriginArray::repeat(OriginId::SolarSystemBarycenter, 2);
        let options = EphemerisOptions {
            photometry: EphemerisPhotometryOptions {
                predict_phase_angle: true,
                ..EphemerisPhotometryOptions::default()
            },
            ..EphemerisOptions::default()
        };
        let err = generate_ephemeris(
            &TwoBodyPropagator::default(),
            &orbits,
            &observers,
            &options,
            &NoopProvider,
        )
        .unwrap_err();
        assert!(
            matches!(err, PropagationError::InvalidRequest(message) if message.contains("SUN-origin"))
        );
    }

    #[test]
    fn typed_generate_ephemeris_surfaces_light_time_nonconvergence_as_row_failure() {
        let orbits = sample_orbits();
        let observers = sample_observers();
        let options = EphemerisOptions {
            output_time_scale: TimeScale::Tdb,
            max_lt_iter: 1,
            ..EphemerisOptions::default()
        };
        let result = generate_ephemeris(
            &TwoBodyPropagator::new(TwoBodyPropagatorConfig::default()).unwrap(),
            &orbits,
            &observers,
            &options,
            &NoopProvider,
        )
        .unwrap();
        assert!(!result.ephemeris.validity.is_valid(0));
        let failures = result.diagnostics.failed_rows().collect::<Vec<_>>();
        assert!(!failures.is_empty());
        assert!(failures.iter().all(|row| {
            row.failure_code == Some(EphemerisFailureCode::LightTimeNonConvergence)
        }));
    }
}
