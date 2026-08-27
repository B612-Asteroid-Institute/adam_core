use super::two_body::{DEFAULT_TWO_BODY_MAX_ITER, DEFAULT_TWO_BODY_TOL};
use super::*;
use crate::propagate::propagate_2body_along_arc;
use crate::types::{Frame, SchemaResult};
use crate::{
    origin_mu_au3_day2, CoordinateBatch, CoordinateRepresentation, CovarianceBatch,
    CovarianceUnits, ObjectId, OrbitBatch, OrbitId, OrbitVariantBatch, OriginArray, OriginId,
    SchemaError, TimeArray, TimeScale, TimeScaleProvider, Validity, VariantId, NANOS_PER_DAY,
};
use std::cell::Cell;

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

struct PassthroughUt1Provider {
    calls: Cell<usize>,
}

impl PassthroughUt1Provider {
    fn new() -> Self {
        Self {
            calls: Cell::new(0),
        }
    }
}

impl TimeScaleProvider for PassthroughUt1Provider {
    fn rescale(&self, times: &TimeArray, new_scale: TimeScale) -> SchemaResult<TimeArray> {
        self.calls.set(self.calls.get() + 1);
        if times.scale != TimeScale::Ut1 || new_scale != TimeScale::Tdb {
            return Err(SchemaError::InvalidTimeScale(format!(
                "test provider only handles ut1->tdb, got {}->{}",
                times.scale.as_str(),
                new_scale.as_str()
            )));
        }
        TimeArray::new(TimeScale::Tdb, times.epochs.clone())
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

fn sample_orbit_times() -> TimeArray {
    TimeArray::from_parts(TimeScale::Tdb, vec![60_000, 60_001], vec![0, 0]).unwrap()
}

fn sample_orbits(covariance: Option<CovarianceBatch>) -> OrbitBatch {
    sample_orbits_with_times(covariance, sample_orbit_times())
}

fn sample_orbits_with_times(covariance: Option<CovarianceBatch>, times: TimeArray) -> OrbitBatch {
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

fn assert_cartesian_states_close(left: &OrbitBatch, right: &OrbitBatch, tolerance: f64) {
    let left_states = left.coordinates.values.cartesian().unwrap();
    let right_states = right.coordinates.values.cartesian().unwrap();
    assert_eq!(left_states.len(), right_states.len());
    for (left_row, right_row) in left_states.iter().zip(right_states.iter()) {
        for column in 0..6 {
            assert!((left_row[column] - right_row[column]).abs() <= tolerance);
        }
    }
}

fn sample_variants() -> OrbitVariantBatch {
    let orbits = sample_orbits(None);
    OrbitVariantBatch::new(
        orbits.orbit_id,
        orbits.object_id,
        vec![
            Some(VariantId("variant-a".to_string())),
            Some(VariantId("variant-b".to_string())),
        ],
        vec![Some(0.7), Some(0.3)],
        vec![Some(0.49), Some(0.09)],
        orbits.coordinates,
    )
    .unwrap()
}

#[test]
fn request_records_epoch_permutation() {
    let orbits = sample_orbits(None);
    let times = TimeArray::from_parts(TimeScale::Tdb, vec![60_010, 60_000], vec![0, 0]).unwrap();
    let request = PropagationRequest::new(&orbits, &times, PropagationOptions::default()).unwrap();
    assert!(!request.epoch_order.is_chronological);
    assert_eq!(request.epoch_order.sorted_to_input, vec![1, 0]);
    assert_eq!(request.epoch_order.input_to_sorted, vec![1, 0]);
    assert_eq!(request.output_len(), 4);
}

#[test]
fn request_accepts_variant_metadata() {
    let variants = sample_variants();
    let times = tdb_times();
    let request =
        PropagationRequest::new_variants(&variants, &times, PropagationOptions::default()).unwrap();
    assert_eq!(request.input.len(), 2);
    assert_eq!(request.output_len(), 4);
    assert_eq!(
        request.input.variant_id().unwrap()[0].as_ref().unwrap().0,
        "variant-a"
    );
    assert_eq!(request.input.weights().unwrap()[1], Some(0.3));
    assert_eq!(request.input.weights_cov().unwrap()[0], Some(0.49));
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
fn two_body_records_backend_iterations_for_successful_rows() {
    let orbits = sample_orbits(None);
    let times = tdb_times();
    let options = PropagationOptions {
        covariance: CovariancePropagation::None,
        thread_limit: Some(1),
        ..PropagationOptions::default()
    };
    let request = PropagationRequest::new(&orbits, &times, options).unwrap();
    let result = TwoBodyPropagator::default()
        .propagate(&request, &NoopProvider)
        .unwrap();

    assert_eq!(result.diagnostics.convergence.len(), 4);
    for row in &result.diagnostics.convergence {
        assert_eq!(row.backend.as_deref(), Some("two_body"));
        assert!(row.iterations.is_some_and(|iterations| iterations > 0));
        assert_eq!(row.failure_code, None);
        assert_eq!(row.status, PropagationConvergenceStatus::Converged);
        assert!(row.message.is_none());
    }
}

#[test]
fn two_body_solver_nonconvergence_is_a_row_failure() {
    let orbits = sample_orbits(None);
    let times = TimeArray::from_parts(TimeScale::Tdb, vec![60_042], vec![0]).unwrap();
    let options = PropagationOptions {
        covariance: CovariancePropagation::None,
        thread_limit: Some(1),
        ..PropagationOptions::default()
    };
    let request = PropagationRequest::new(&orbits, &times, options).unwrap();
    let propagator = TwoBodyPropagator::new(TwoBodyPropagatorConfig {
        max_iter: 1,
        tol: f64::MIN_POSITIVE,
    })
    .unwrap();
    let result = propagator.propagate(&request, &NoopProvider).unwrap();

    let failures = result.diagnostics.failed_rows().collect::<Vec<_>>();
    assert!(!failures.is_empty());
    for failure in failures {
        assert_eq!(failure.backend.as_deref(), Some("two_body"));
        assert_eq!(
            failure.failure_code,
            Some(PropagationFailureCode::SolverMaxIterations)
        );
        assert!(failure.iterations.is_some_and(|iterations| iterations > 0));
        assert!(failure
            .message
            .as_deref()
            .unwrap()
            .contains("maximum iteration"));
        assert!(!result.validity.is_valid(failure.output_row));
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
fn two_body_accepts_utc_inputs_via_rust_time_rescale() {
    let tdb_orbits = sample_orbits(None);
    let tdb_times = tdb_times();
    let options = PropagationOptions {
        covariance: CovariancePropagation::None,
        thread_limit: Some(1),
        ..PropagationOptions::default()
    };
    let tdb_request = PropagationRequest::new(&tdb_orbits, &tdb_times, options.clone()).unwrap();
    let tdb_result = TwoBodyPropagator::default()
        .propagate(&tdb_request, &NoopProvider)
        .unwrap();

    let utc_orbit_times = tdb_orbits
        .coordinates
        .times
        .as_ref()
        .unwrap()
        .rescale(TimeScale::Utc)
        .unwrap();
    let utc_orbits = sample_orbits_with_times(None, utc_orbit_times);
    let utc_times = tdb_times.rescale(TimeScale::Utc).unwrap();
    let utc_request = PropagationRequest::new(&utc_orbits, &utc_times, options).unwrap();
    let utc_result = TwoBodyPropagator::default()
        .propagate(&utc_request, &NoopProvider)
        .unwrap();

    assert_eq!(utc_result.times.scale, TimeScale::Tdb);
    assert_cartesian_states_close(&tdb_result.orbits, &utc_result.orbits, 1.0e-13);
}

#[test]
fn two_body_uses_provider_for_ut1_inputs() {
    let tdb_orbits = sample_orbits(None);
    let tdb_times = tdb_times();
    let options = PropagationOptions {
        covariance: CovariancePropagation::None,
        thread_limit: Some(1),
        ..PropagationOptions::default()
    };
    let tdb_request = PropagationRequest::new(&tdb_orbits, &tdb_times, options.clone()).unwrap();
    let tdb_result = TwoBodyPropagator::default()
        .propagate(&tdb_request, &NoopProvider)
        .unwrap();

    let ut1_orbit_times = TimeArray::new(
        TimeScale::Ut1,
        tdb_orbits
            .coordinates
            .times
            .as_ref()
            .unwrap()
            .epochs
            .clone(),
    )
    .unwrap();
    let ut1_orbits = sample_orbits_with_times(None, ut1_orbit_times);
    let ut1_times = TimeArray::new(TimeScale::Ut1, tdb_times.epochs.clone()).unwrap();
    let ut1_request = PropagationRequest::new(&ut1_orbits, &ut1_times, options).unwrap();
    let provider = PassthroughUt1Provider::new();
    let ut1_result = TwoBodyPropagator::default()
        .propagate(&ut1_request, &provider)
        .unwrap();

    assert_eq!(provider.calls.get(), 2);
    assert_eq!(ut1_result.times.scale, TimeScale::Tdb);
    assert_cartesian_states_close(&tdb_result.orbits, &ut1_result.orbits, 1.0e-15);
}

#[test]
fn two_body_preserves_variant_metadata() {
    let variants = sample_variants();
    let times = tdb_times();
    let options = PropagationOptions {
        covariance: CovariancePropagation::None,
        chunk_size: Some(1),
        thread_limit: Some(1),
        ..PropagationOptions::default()
    };
    let request = PropagationRequest::new_variants(&variants, &times, options).unwrap();
    let result = TwoBodyPropagator::default()
        .propagate(&request, &NoopProvider)
        .unwrap();
    let output_variants = result.variants.as_ref().unwrap();
    assert_eq!(output_variants.len(), 4);
    assert_eq!(result.orbits, output_variants.to_orbit_batch().unwrap());
    assert_eq!(
        output_variants.variant_id[0].as_ref().unwrap().0,
        "variant-a"
    );
    assert_eq!(
        output_variants.variant_id[1].as_ref().unwrap().0,
        "variant-a"
    );
    assert_eq!(
        output_variants.variant_id[2].as_ref().unwrap().0,
        "variant-b"
    );
    assert_eq!(
        output_variants.variant_id[3].as_ref().unwrap().0,
        "variant-b"
    );
    assert_eq!(
        output_variants.weights,
        vec![Some(0.7), Some(0.7), Some(0.3), Some(0.3)]
    );
    assert_eq!(
        output_variants.weights_cov,
        vec![Some(0.49), Some(0.49), Some(0.09), Some(0.09)]
    );
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
    let request = PropagationRequest::new(&orbits, &times, PropagationOptions::default()).unwrap();
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
    let request = PropagationRequest::new(&orbits, &times, PropagationOptions::default()).unwrap();
    let result = TwoBodyPropagator::default()
        .propagate(&request, &NoopProvider)
        .unwrap();
    assert!(!result.validity.is_valid(0));
    let failures = result.diagnostics.failed_rows().collect::<Vec<_>>();
    assert_eq!(failures.len(), 1);
    assert_eq!(failures[0].input_orbit_index, 0);
    assert_eq!(failures[0].backend.as_deref(), Some("two_body"));
    assert_eq!(failures[0].iterations, None);
    assert_eq!(
        failures[0].failure_code,
        Some(PropagationFailureCode::NonFiniteInputState)
    );
    assert!(failures[0]
        .message
        .as_deref()
        .unwrap()
        .contains("input state"));
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
    let request = PropagationRequest::new(&orbits, &times, PropagationOptions::default()).unwrap();
    let err = TwoBodyPropagator::default()
        .propagate(&request, &NoopProvider)
        .unwrap_err();
    assert!(
        matches!(err, PropagationError::Schema(SchemaError::UnsupportedOrigin(origin)) if origin == "UNKNOWN")
    );
}
