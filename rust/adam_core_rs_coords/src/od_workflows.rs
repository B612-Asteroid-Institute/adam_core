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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::od_config::{
        EvaluationConfig, IodRunConfig, ObservationPipeline, ObservationTransform,
        RefinementRunConfig, ResourceBindings,
    };
    use crate::propagation::adapter::PropagatorPredictor;
    use crate::propagation::{
        generate_ephemeris_barycentric, BackendIdentity, CovariancePropagation, EphemerisOptions,
        EpochPolicy, IodConfig, OrbitGeometry, PropagationOptions, StmRow, TwoBodyPropagator,
        WhitenedFitConfig,
    };
    use crate::types::{Frame, SchemaResult};
    use crate::{
        CoordinateRepresentation, CoordinateValues, CovarianceBatch, CovarianceUnits, Epoch,
        NonGravitationalParametersRow, ObjectId, ObservatoryCode, OrbitId, OriginArray, OriginId,
        TimeArray, TimeScale,
    };

    const TRUTH_STATE: [f64; 6] = [1.2, 0.1, 0.05, -0.002, 0.016, 0.001];
    const NUM_OBS: usize = 12;
    const SIGMA_DEG: f64 = 0.5 / 3600.0;
    const NANOS_PER_DAY: f64 = 86_400_000_000_000.0;

    struct IdentityTimeProvider;

    impl TimeScaleProvider for IdentityTimeProvider {
        fn rescale(&self, times: &TimeArray, new_scale: TimeScale) -> SchemaResult<TimeArray> {
            TimeArray::new(new_scale, times.epochs.clone())
        }
    }

    struct ZeroTranslationProvider;

    impl OriginTranslationProvider for ZeroTranslationProvider {
        fn origin_translation_vectors(
            &self,
            origins: &OriginArray,
            _target_origin: &OriginId,
            _frame: Frame,
            _times: &TimeArray,
        ) -> SchemaResult<Vec<[f64; 6]>> {
            Ok(vec![[0.0; 6]; origins.len()])
        }
    }

    fn ephemeris_options() -> EphemerisOptions {
        EphemerisOptions {
            propagation: PropagationOptions {
                chunk_size: None,
                thread_limit: None,
                epoch_policy: EpochPolicy::CrossProduct,
                covariance: CovariancePropagation::None,
            },
            output_time_scale: TimeScale::Tdb,
            ..EphemerisOptions::default()
        }
    }

    fn orbit(state: [f64; 6]) -> OrbitBatch {
        let coordinates = CoordinateBatch::cartesian(
            vec![state],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::Named("SUN".to_string()), 1),
            Some(TimeArray::new(TimeScale::Tdb, vec![Epoch::new(60_000, 0)]).unwrap()),
            None,
        )
        .unwrap();
        OrbitBatch::new(
            vec![OrbitId("test-orbit".to_string())],
            vec![Some(ObjectId("test-object".to_string()))],
            coordinates,
        )
        .unwrap()
    }

    fn dense_observers() -> ObserverBatch {
        // Three observations per civil night, then a three-day jump. The
        // 0.3-day offset keeps TDB->UTC conversion away from midnight, while
        // the four-night layout retains a nine-day IOD arc.
        let offsets: Vec<f64> = (0..NUM_OBS)
            .map(|row| 0.3 + 3.0 * (row / 3) as f64 + 0.2 * (row % 3) as f64)
            .collect();
        let times = TimeArray::new(
            TimeScale::Tdb,
            offsets
                .iter()
                .map(|offset| Epoch::new(60_002, (offset * NANOS_PER_DAY) as i64))
                .collect(),
        )
        .unwrap();
        let states = offsets
            .iter()
            .map(|offset| {
                let theta = 0.35 + 0.01725 * offset;
                [
                    theta.cos(),
                    theta.sin(),
                    0.0,
                    -0.01725 * theta.sin(),
                    0.01725 * theta.cos(),
                    0.0,
                ]
            })
            .collect();
        ObserverBatch::new(
            (0..NUM_OBS)
                .map(|row| ObservatoryCode(format!("T{row:02}")))
                .collect(),
            CoordinateBatch::cartesian(
                states,
                Frame::Ecliptic,
                OriginArray::repeat(OriginId::Named("SUN".to_string()), NUM_OBS),
                Some(times),
                None,
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn synthetic_observations() -> CoordinateBatch {
        let predicted = generate_ephemeris_barycentric(
            &TwoBodyPropagator::default(),
            &orbit(TRUTH_STATE),
            &dense_observers(),
            &ephemeris_options(),
            &IdentityTimeProvider,
            &ZeroTranslationProvider,
        )
        .unwrap()
        .ephemeris
        .coordinates;
        let mut values = predicted.values.raw_values().to_vec();
        let mut seed = 12345_u64;
        let mut noise = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.6 * SIGMA_DEG
        };
        for value in &mut values {
            let cos_lat = value[2].to_radians().cos();
            value[1] += noise() / cos_lat;
            value[2] += noise();
            value[0] = f64::NAN;
            value[3] = f64::NAN;
            value[4] = f64::NAN;
            value[5] = f64::NAN;
        }
        let mut covariance = vec![f64::NAN; NUM_OBS * 36];
        for row in 0..NUM_OBS {
            covariance[row * 36 + 7] = SIGMA_DEG * SIGMA_DEG;
            covariance[row * 36 + 14] = SIGMA_DEG * SIGMA_DEG;
        }
        CoordinateBatch::new(
            CoordinateValues::Spherical(values),
            predicted.frame,
            predicted.origins,
            predicted.times,
            Some(
                CovarianceBatch::new(
                    NUM_OBS,
                    6,
                    covariance,
                    CovarianceUnits::Coordinate(CoordinateRepresentation::Spherical),
                )
                .unwrap(),
            ),
        )
        .unwrap()
    }

    fn pipeline_config(caps: &[usize]) -> OdConfig {
        let definitions = caps
            .iter()
            .map(
                |&effective_observation_cap| ObservationTransform::NightlyBatchDeweighting {
                    effective_observation_cap,
                },
            )
            .collect();
        let pipeline =
            ObservationPipeline::bind_explicit(definitions, &ResourceBindings::new()).unwrap();
        let iod = IodRunConfig::new(
            IodConfig {
                max_triplets: Some(3),
                rchi2_threshold: 50.0,
                ..IodConfig::default()
            },
            WhitenedFitConfig::default(),
        )
        .unwrap();
        OdConfig::explicit(
            pipeline,
            iod,
            RefinementRunConfig::core_v1(),
            EvaluationConfig::six_dimensional(),
        )
        .unwrap()
    }

    fn metadata() -> (Vec<String>, Vec<Option<String>>, Vec<Option<String>>) {
        (
            vec!["500".to_string(); NUM_OBS],
            vec![None; NUM_OBS],
            vec![None; NUM_OBS],
        )
    }

    fn assert_float_bits(left: &[f64], right: &[f64]) {
        assert_eq!(left.len(), right.len());
        for (left, right) in left.iter().zip(right) {
            assert_eq!(left.to_bits(), right.to_bits());
        }
    }

    fn assert_coordinate_value_bits(left: &CoordinateValues, right: &CoordinateValues) {
        assert_eq!(left.raw_values().len(), right.raw_values().len());
        for (left, right) in left.raw_values().iter().zip(right.raw_values()) {
            assert_float_bits(left, right);
        }
    }

    fn assert_prepared_bits(left: &PreparedObservations, right: &PreparedObservations) {
        assert_coordinate_value_bits(&left.observations.values, &right.observations.values);
        assert_float_bits(
            &left
                .observations
                .covariance
                .as_ref()
                .unwrap()
                .values_row_major,
            &right
                .observations
                .covariance
                .as_ref()
                .unwrap()
                .values_row_major,
        );
        assert_float_bits(&left.original.lon, &right.original.lon);
        assert_float_bits(&left.original.lat, &right.original.lat);
        assert_float_bits(&left.original.sigma_lon, &right.original.sigma_lon);
        assert_float_bits(&left.original.sigma_lat, &right.original.sigma_lat);
        assert_float_bits(&left.original.cov_lonlat, &right.original.cov_lonlat);
        assert_float_bits(&left.used.lon, &right.used.lon);
        assert_float_bits(&left.used.lat, &right.used.lat);
        assert_float_bits(&left.used.sigma_lon, &right.used.sigma_lon);
        assert_float_bits(&left.used.sigma_lat, &right.used.sigma_lat);
        assert_float_bits(&left.used.cov_lonlat, &right.used.cov_lonlat);
        assert_eq!(left.models_changed, right.models_changed);
    }

    struct StageCountingPredictor<'a> {
        stage: &'static str,
        inner: &'a dyn SphericalPredictor,
        calls: std::cell::Cell<usize>,
        states: std::cell::Cell<usize>,
    }

    impl StageCountingPredictor<'_> {
        fn reset(&self) {
            self.calls.set(0);
            self.states.set(0);
        }
    }

    impl SphericalPredictor for StageCountingPredictor<'_> {
        fn predict_spherical(
            &self,
            states: &[[f64; 6]],
            geometry: &OrbitGeometry,
            observers: &ObserverBatch,
        ) -> PropagationResultValue<Result<Vec<f64>, String>> {
            self.calls.set(self.calls.get() + 1);
            self.states.set(self.states.get() + states.len());
            self.inner.predict_spherical(states, geometry, observers)
        }

        fn supports_stm(&self) -> bool {
            self.inner.supports_stm()
        }

        fn propagate_stm(
            &self,
            state: [f64; 6],
            non_gravitational_parameters: Option<NonGravitationalParametersRow>,
            epoch_tdb: Epoch,
            times_tdb: &[Epoch],
        ) -> PropagationResultValue<Vec<StmRow>> {
            self.inner
                .propagate_stm(state, non_gravitational_parameters, epoch_tdb, times_tdb)
        }

        fn backend(&self) -> BackendIdentity {
            let mut identity = self.inner.backend();
            identity.name = format!("test.stage.{}", self.stage);
            identity
        }
    }

    #[test]
    fn prepare_observations_executes_repeated_pipeline_and_retains_snapshots() {
        let observed = synthetic_observations();
        let (obs_code, band, astcat) = metadata();
        let once = prepare_observations(
            &observed,
            &obs_code,
            &band,
            &astcat,
            &pipeline_config(&[2]),
            &IdentityTimeProvider,
        )
        .unwrap();
        let repeated_config = pipeline_config(&[2, 1]);
        let repeated = prepare_observations(
            &observed,
            &obs_code,
            &band,
            &astcat,
            &repeated_config,
            &IdentityTimeProvider,
        )
        .unwrap();

        assert!(once.models_changed && repeated.models_changed);
        assert_eq!(
            repeated_config.observation_pipeline().transform_kinds(),
            vec!["nightly_batch_deweighting", "nightly_batch_deweighting"]
        );
        let provenance: serde_json::Value =
            serde_json::from_str(&repeated_config.provenance_json()).unwrap();
        assert_eq!(
            provenance["transforms"][0]["effective_observation_cap"].as_u64(),
            Some(2)
        );
        assert_eq!(
            provenance["transforms"][1]["effective_observation_cap"].as_u64(),
            Some(1)
        );
        assert_float_bits(&once.original.lon, &repeated.original.lon);
        assert_float_bits(&once.original.lat, &repeated.original.lat);
        assert_float_bits(&repeated.original.lon, &repeated.used.lon);
        assert_float_bits(&repeated.original.lat, &repeated.used.lat);

        let original_variance = observed.covariance.as_ref().unwrap().values_row_major[7];
        let once_variance = once
            .observations
            .covariance
            .as_ref()
            .unwrap()
            .values_row_major[7];
        let repeated_variance = repeated
            .observations
            .covariance
            .as_ref()
            .unwrap()
            .values_row_major[7];
        // The first night has three observations: cap 2 scales by 3/2 and
        // the repeated cap 1 then scales the already-used block by 3.
        assert_eq!(once_variance.to_bits(), (original_variance * 1.5).to_bits());
        assert_eq!(
            repeated_variance.to_bits(),
            (original_variance * 1.5 * 3.0).to_bits()
        );
        assert_ne!(once_variance.to_bits(), repeated_variance.to_bits());
    }

    #[test]
    fn run_od_prepares_once_and_exactly_composes_isolated_stage_predictors() {
        let observed = synthetic_observations();
        let observers = dense_observers();
        let (obs_code, band, astcat) = metadata();
        let config = pipeline_config(&[2, 1]);
        let propagator = TwoBodyPropagator::default();
        let options = ephemeris_options();
        let inner = PropagatorPredictor {
            propagator: &propagator,
            options: &options,
            provider: &IdentityTimeProvider,
            translation_provider: &ZeroTranslationProvider,
        };
        let iod_predictor = StageCountingPredictor {
            stage: "iod",
            inner: &inner,
            calls: std::cell::Cell::new(0),
            states: std::cell::Cell::new(0),
        };
        let refinement_predictor = StageCountingPredictor {
            stage: "refinement",
            inner: &inner,
            calls: std::cell::Cell::new(0),
            states: std::cell::Cell::new(0),
        };

        let prepared = prepare_observations(
            &observed,
            &obs_code,
            &band,
            &astcat,
            &config,
            &IdentityTimeProvider,
        )
        .unwrap();
        let iod = run_iod_with_predictor(
            &iod_predictor,
            &prepared,
            &observers,
            &config,
            &IdentityTimeProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert!(iod_predictor.calls.get() > 0);
        assert_eq!(refinement_predictor.calls.get(), 0);
        let seed = iod
            .seed_orbit()
            .unwrap()
            .expect("configured IOD found a seed");
        let iod_evidence = (iod_predictor.calls.get(), iod_predictor.states.get());
        let refined = refine_with_predictor(
            &refinement_predictor,
            &seed,
            &prepared,
            &observers,
            &config,
            &IdentityTimeProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        assert_eq!(
            (iod_predictor.calls.get(), iod_predictor.states.get()),
            iod_evidence
        );
        let refinement_evidence = (
            refinement_predictor.calls.get(),
            refinement_predictor.states.get(),
        );
        assert!(refinement_evidence.0 > 0);
        assert_eq!(iod_predictor.backend().name, "test.stage.iod");
        assert_eq!(refinement_predictor.backend().name, "test.stage.refinement");

        iod_predictor.reset();
        refinement_predictor.reset();
        let composed = run_od_with_predictors(
            &iod_predictor,
            &refinement_predictor,
            &observed,
            &observers,
            &obs_code,
            &band,
            &astcat,
            &config,
            &IdentityTimeProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        let composed_refined = composed.refined.as_ref().expect("refinement ran");

        assert!(composed_refined.fit.converged);
        assert_prepared_bits(&composed.prepared, &prepared);
        assert_eq!(composed.iod.accepted, iod.accepted);
        assert_eq!(composed.iod.triplets_tried, iod.triplets_tried);
        assert_eq!(composed.iod.candidates.len(), iod.candidates.len());
        assert_eq!(
            composed.iod.accepted_candidate().unwrap().corrected_state,
            iod.accepted_candidate().unwrap().corrected_state
        );
        assert_float_bits(&composed_refined.fit.state, &refined.fit.state);
        assert_float_bits(&composed_refined.fit.covariance, &refined.fit.covariance);
        assert_float_bits(&composed_refined.fit.weights, &refined.fit.weights);
        assert_eq!(
            composed_refined.fit.evaluation.outlier,
            refined.fit.evaluation.outlier
        );
        assert_eq!(composed_refined.n_rejected, refined.n_rejected);
        assert_eq!(composed_refined.n_recovered, refined.n_recovered);
        assert_eq!(
            (iod_predictor.calls.get(), iod_predictor.states.get()),
            iod_evidence
        );
        assert_eq!(
            (
                refinement_predictor.calls.get(),
                refinement_predictor.states.get(),
            ),
            refinement_evidence
        );
    }
}
