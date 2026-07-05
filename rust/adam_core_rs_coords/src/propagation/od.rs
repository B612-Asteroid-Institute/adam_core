//! Backend-generic least-squares orbit determination (bead personal-cmy.7).
//!
//! The Gauss-Newton driver lives in the permissive core, generic over the
//! [`Propagator`] trait: candidate states are predicted through the shared
//! backend-generic ephemeris workflow, so each iteration costs one base
//! prediction plus one batched 6-state Jacobian prediction. GPL backends
//! (the adam-assist equivalent) only plug in their `Propagator`
//! implementation; no OD logic lives behind the GPL boundary.

use super::ephemeris::{generate_ephemeris_barycentric, EphemerisOptions};
use super::{PropagationError, PropagationResultValue, Propagator};
use crate::orbit_least_squares::{
    fit_orbit_least_squares_with_predictor, LeastSquaresConfig, LeastSquaresFit,
};
use crate::translation::OriginTranslationProvider;
use crate::types::time::TimeScaleProvider;
use crate::{CoordinateBatch, ObjectId, ObserverBatch, OrbitBatch, OrbitId};

/// Differentially correct a single orbit against astrometric observations
/// using Gauss-Newton least squares, with predictions generated through the
/// backend-generic barycentric ephemeris workflow.
///
/// `orbit` must contain exactly one row (Cartesian, with an epoch);
/// `observed` are spherical coordinates with per-row covariance aligned
/// row-wise with `observers`. The fitted state and `inv(JᵀJ)` covariance are
/// expressed in the input orbit's frame/origin.
#[allow(clippy::too_many_arguments)]
pub fn fit_orbit_least_squares_barycentric<P, T>(
    propagator: &P,
    orbit: &OrbitBatch,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    options: &EphemerisOptions,
    config: &LeastSquaresConfig,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<LeastSquaresFit>
where
    P: Propagator,
    T: OriginTranslationProvider,
{
    if orbit.len() != 1 {
        return Err(PropagationError::InvalidRequest(
            "least-squares OD corrects exactly one orbit".to_string(),
        ));
    }
    let n = observers.len();
    if observed.len() != n {
        return Err(PropagationError::InvalidRequest(format!(
            "observed rows ({}) must match observer rows ({n})",
            observed.len()
        )));
    }
    let observed_values = observed.values.spherical().ok_or_else(|| {
        PropagationError::InvalidRequest(
            "least-squares OD requires spherical observed coordinates".to_string(),
        )
    })?;
    let observed_flat: Vec<f64> = observed_values
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    let observed_cov = observed
        .covariance
        .as_ref()
        .ok_or_else(|| {
            PropagationError::InvalidRequest(
                "least-squares OD requires observed covariance".to_string(),
            )
        })?
        .values_row_major
        .clone();
    let initial_state = orbit.coordinates.values.cartesian().ok_or_else(|| {
        PropagationError::InvalidRequest(
            "least-squares OD requires Cartesian orbit coordinates".to_string(),
        )
    })?[0];
    let times = orbit.coordinates.times.as_ref().ok_or_else(|| {
        PropagationError::InvalidRequest("least-squares OD requires an orbit epoch".to_string())
    })?;
    let epoch = times.epochs[0];
    let scale = times.scale;
    let frame = orbit.coordinates.frame;
    let origin = orbit.coordinates.origins.origins[0].clone();

    let predict = |states: &[[f64; 6]]| -> Result<Vec<f64>, String> {
        let candidates = build_candidate_batch(states, epoch, scale, frame, &origin)
            .map_err(|err| err.to_string())?;
        let result = generate_ephemeris_barycentric(
            propagator,
            &candidates,
            observers,
            options,
            provider,
            translation_provider,
        )
        .map_err(|err| err.to_string())?;
        for row in 0..result.ephemeris.len() {
            if !result.ephemeris.validity.is_valid(row) {
                return Err(format!(
                    "ephemeris row {row} failed during least-squares prediction"
                ));
            }
        }
        let values = result
            .ephemeris
            .coordinates
            .values
            .spherical()
            .ok_or_else(|| "ephemeris output was not spherical".to_string())?;
        Ok(values.iter().flat_map(|row| row.iter().copied()).collect())
    };

    fit_orbit_least_squares_with_predictor(
        initial_state,
        &observed_flat,
        &observed_cov,
        n,
        config,
        predict,
    )
    .map_err(PropagationError::Backend)
}

fn build_candidate_batch(
    states: &[[f64; 6]],
    epoch: crate::Epoch,
    scale: crate::TimeScale,
    frame: crate::types::Frame,
    origin: &crate::OriginId,
) -> Result<OrbitBatch, crate::SchemaError> {
    let coordinates = CoordinateBatch::cartesian(
        states.to_vec(),
        frame,
        crate::OriginArray::repeat(origin.clone(), states.len()),
        Some(crate::TimeArray::new(scale, vec![epoch; states.len()])?),
        None,
    )?;
    OrbitBatch::new(
        (0..states.len())
            .map(|index| OrbitId(format!("lsq-candidate-{index}")))
            .collect(),
        vec![None::<ObjectId>; states.len()],
        coordinates,
    )
}
