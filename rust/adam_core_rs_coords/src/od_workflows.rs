//! Focused Rust OD workflow composition over the closed [`crate::od`] config.
//!
//! These predictor-level functions are building blocks for the forthcoming
//! canonical batch surface. They are not a grouping or failure-envelope API.

use crate::od_config::OdConfig;
use crate::propagation::adapter::{refine_with, run_iod_with, SphericalPredictor};
use crate::propagation::{
    apply_configured_observation_pipeline, AppliedObservationModels, IodResult, RefinedOutput,
};
use crate::translation::OriginTranslationProvider;
use crate::types::time::TimeScaleProvider;
use crate::{CoordinateBatch, ObserverBatch, OrbitBatch, PropagationResultValue};

/// Closed-pipeline preparation result. It retains supplied/used astrometry and
/// the exact coordinate batch consumed by all subsequent stages.
pub type PreparedObservations = AppliedObservationModels;

#[allow(clippy::too_many_arguments)]
pub fn prepare_observations(
    observed: &CoordinateBatch,
    obs_code: &[String],
    band: &[Option<String>],
    astcat: &[Option<String>],
    config: &OdConfig,
    provider: &dyn TimeScaleProvider,
) -> PropagationResultValue<PreparedObservations> {
    config
        .validate()
        .map_err(crate::PropagationError::InvalidRequest)?;
    apply_configured_observation_pipeline(
        observed,
        obs_code,
        band,
        astcat,
        config.observation_pipeline(),
        provider,
    )
}

pub fn run_iod_with_predictor<T: OriginTranslationProvider>(
    predictor: &dyn SphericalPredictor,
    prepared: &PreparedObservations,
    observers: &ObserverBatch,
    config: &OdConfig,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<IodResult> {
    let iod = config.for_iod();
    run_iod_with(
        predictor,
        &prepared.observations,
        observers,
        iod.iod(),
        iod.correction_fit(),
        provider,
        translation_provider,
    )
}

pub fn refine_with_predictor<T: OriginTranslationProvider>(
    predictor: &dyn SphericalPredictor,
    seed: &OrbitBatch,
    prepared: &PreparedObservations,
    observers: &ObserverBatch,
    config: &OdConfig,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<RefinedOutput> {
    let refinement = config.for_refinement();
    refine_with(
        predictor,
        seed,
        &prepared.observations,
        observers,
        refinement.refinement(),
        provider,
        translation_provider,
    )
}

#[derive(Debug, Clone)]
pub struct ConfiguredRunOdOutput {
    pub prepared: PreparedObservations,
    pub iod: IodResult,
    pub refined: Option<RefinedOutput>,
}

/// Exact configured composition: prepare once, IOD, then refine the accepted
/// IOD seed against the same prepared coordinates.
#[allow(clippy::too_many_arguments)]
pub fn run_od_with_predictors<T: OriginTranslationProvider>(
    iod_predictor: &dyn SphericalPredictor,
    refinement_predictor: &dyn SphericalPredictor,
    observed: &CoordinateBatch,
    observers: &ObserverBatch,
    obs_code: &[String],
    band: &[Option<String>],
    astcat: &[Option<String>],
    config: &OdConfig,
    provider: &dyn TimeScaleProvider,
    translation_provider: &T,
) -> PropagationResultValue<ConfiguredRunOdOutput> {
    let prepared = prepare_observations(observed, obs_code, band, astcat, config, provider)?;
    let iod = run_iod_with_predictor(
        iod_predictor,
        &prepared,
        observers,
        config,
        provider,
        translation_provider,
    )?;
    let refined = match iod.seed_orbit()? {
        Some(seed) => Some(refine_with_predictor(
            refinement_predictor,
            &seed,
            &prepared,
            observers,
            config,
            provider,
            translation_provider,
        )?),
        None => None,
    };
    Ok(ConfiguredRunOdOutput {
        prepared,
        iod,
        refined,
    })
}
