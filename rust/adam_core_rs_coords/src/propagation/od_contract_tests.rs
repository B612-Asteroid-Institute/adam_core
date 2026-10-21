//! Independent review tests for a2d33b3a (included into od.rs via #[path]).
use super::*;
use crate::propagation::{
    CovariancePropagation, EphemerisOptions, EpochPolicy, PropagationOptions, TwoBodyPropagator,
};
use crate::types::{Frame, SchemaError, SchemaResult};
use crate::{CovarianceUnits, Epoch, ObservatoryCode, OriginId, TimeScale};
use std::time::Instant;

struct NoopProvider;
impl TimeScaleProvider for NoopProvider {
    fn rescale(&self, _times: &TimeArray, _new_scale: TimeScale) -> SchemaResult<TimeArray> {
        Err(SchemaError::InvalidRecordBatch("no provider".to_string()))
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

const TRUTH: [f64; 6] = [1.2, 0.1, 0.05, -0.002, 0.016, 0.001];

fn options() -> EphemerisOptions {
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
    OrbitBatch::new(
        vec![OrbitId("o".to_string())],
        vec![None],
        CoordinateBatch::cartesian(
            vec![state],
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::from_code("SUN"), 1),
            Some(TimeArray::new(TimeScale::Tdb, vec![Epoch::new(60_000, 0)]).unwrap()),
            None,
        )
        .unwrap(),
    )
    .unwrap()
}

/// Observers at `epochs` (TDB days, fractional) with station codes.
fn observers_at(epochs: &[f64], codes: &[&str]) -> ObserverBatch {
    let n = epochs.len();
    let times = TimeArray::new(
        TimeScale::Tdb,
        epochs
            .iter()
            .map(|&t| {
                let d = t.floor();
                Epoch::new(d as i64, ((t - d) * 86_400e9).round() as i64)
            })
            .collect(),
    )
    .unwrap();
    let states: Vec<[f64; 6]> = epochs
        .iter()
        .map(|&t| {
            let theta = 0.35 + 0.0172 * (t - 60_002.0);
            [
                theta.cos(),
                theta.sin(),
                0.0,
                -0.0172 * theta.sin(),
                0.0172 * theta.cos(),
                0.0,
            ]
        })
        .collect();
    ObserverBatch::new(
        codes
            .iter()
            .map(|c| ObservatoryCode(c.to_string()))
            .collect(),
        CoordinateBatch::cartesian(
            states,
            Frame::Ecliptic,
            OriginArray::repeat(OriginId::from_code("SUN"), n),
            Some(times),
            None,
        )
        .unwrap(),
    )
    .unwrap()
}

fn observe(observers: &ObserverBatch, sigma_deg: &[f64]) -> CoordinateBatch {
    let n = observers.len();
    let predicted = generate_ephemeris_barycentric(
        &TwoBodyPropagator::default(),
        &orbit(TRUTH),
        observers,
        &options(),
        &NoopProvider,
        &ZeroTranslationProvider,
    )
    .unwrap()
    .ephemeris
    .coordinates;
    let mut cov = vec![0.0; n * 36];
    for row in 0..n {
        cov[row * 36 + 7] = sigma_deg[row] * sigma_deg[row];
        cov[row * 36 + 14] = sigma_deg[row] * sigma_deg[row];
    }
    let values: Vec<[f64; 6]> = predicted
        .values
        .raw_values()
        .iter()
        .map(|v| [f64::NAN, v[1], v[2], f64::NAN, f64::NAN, f64::NAN])
        .collect();
    CoordinateBatch::new(
        crate::CoordinateValues::Spherical(values),
        predicted.frame,
        predicted.origins.clone(),
        predicted.times.clone(),
        Some(
            CovarianceBatch::new(
                n,
                6,
                cov,
                CovarianceUnits::Coordinate(crate::CoordinateRepresentation::Spherical),
            )
            .unwrap(),
        ),
    )
    .unwrap()
}

fn predictor<'a>(
    p: &'a TwoBodyPropagator,
    o: &'a EphemerisOptions,
) -> PropagatorPredictor<'a, TwoBodyPropagator, ZeroTranslationProvider> {
    PropagatorPredictor {
        propagator: p,
        options: o,
        provider: &NoopProvider,
        translation_provider: &ZeroTranslationProvider,
    }
}

fn xorshift_perms(n: usize, count: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut s = seed;
    (0..count)
        .map(|_| {
            let mut p: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                p.swap(i, (s % (i as u64 + 1)) as usize);
            }
            p
        })
        .collect()
}

fn row_sig(o: &CoordinateBatch, b: &ObserverBatch, row: usize) -> String {
    format!(
        "{:?}|{:?}|{:?}|{}|{:?}",
        o.values.raw_values()[row]
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        o.covariance
            .as_ref()
            .unwrap()
            .row_values(row)
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        o.times.as_ref().unwrap().epochs[row],
        b.code[row].0,
        b.coordinates.values.raw_values()[row]
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
    )
}

/// Arc with exact duplicates and same-epoch/same-station rows differing
/// only in covariance (sigma) — the cases where caller order could leak.
fn tie_heavy_arc() -> (CoordinateBatch, ObserverBatch) {
    let mut epochs = Vec::new();
    let mut codes = Vec::new();
    let mut sig = Vec::new();
    for k in 0..8 {
        let t = 60_002.0 + 3.0 * k as f64;
        for dup in 0..3 {
            epochs.push(t);
            codes.push(if k % 2 == 0 { "X05" } else { "I41" });
            sig.push(if dup == 2 { 2.0 / 3600.0 } else { 1.0 / 3600.0 });
        }
    }
    let obs = observers_at(&epochs, &codes);
    let o = observe(&obs, &sig);
    (o, obs)
}

fn result_signature(r: &IodResult, o: &CoordinateBatch, b: &ObserverBatch) -> Vec<String> {
    let mut out = vec![format!(
        "accepted={:?} tried={} eval={} failed={} reason={:?}",
        r.accepted,
        r.triplets_tried,
        r.candidates_evaluated,
        r.candidates_failed,
        r.not_found_reason
    )];
    for c in &r.candidates {
        out.push(format!(
            "{:?} root={} state={:?} chi2={} rchi2={} acc={} rows=[{}|{}|{}] removed={:?}",
            c.epoch,
            c.root_index,
            c.state.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            c.prefit_chi2.to_bits(),
            c.reduced_chi2.to_bits(),
            c.accepted,
            row_sig(o, b, c.triplet[0]),
            row_sig(o, b, c.triplet[1]),
            row_sig(o, b, c.triplet[2]),
            c.removed
                .iter()
                .map(|&r| row_sig(o, b, r))
                .collect::<Vec<_>>(),
        ));
    }
    out
}

#[test]
fn review_iod_is_bitwise_shuffle_invariant_on_tie_heavy_arc() {
    let (o, b) = tie_heavy_arc();
    let p = TwoBodyPropagator::default();
    let opt = options();
    let pred = predictor(&p, &opt);
    for method in [
        ObservationSelectionMethod::Spans,
        ObservationSelectionMethod::Combinations,
    ] {
        let config = IodConfig {
            acceptance: IodAcceptance::Prefit,
            observation_selection_method: method,
            rchi2_threshold: 1e4,
            contamination_percentage: 20.0,
            velocity_method: VelocityMethod::Auto,
            ..IodConfig::default()
        };
        let base = run_iod(&pred, &o, &b, &config).unwrap();
        let base_sig = result_signature(&base, &o, &b);
        assert!(base.found(), "{base_sig:?}");
        for perm in xorshift_perms(o.len(), 24, 0x1234_5678_9abc_def1) {
            let so = take_coordinate_batch(&o, &perm).unwrap();
            let sb = take_observer_batch(&b, &perm).unwrap();
            let r = run_iod(&pred, &so, &sb, &config).unwrap();
            assert_eq!(
                result_signature(&r, &so, &sb),
                base_sig,
                "{method:?} {perm:?}"
            );
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Layout {
    ThreeNights,
    TenNights,
    SingleEpoch,
    TwoEpochs,
    Unique,
}

fn layout_times(layout: Layout, n: usize) -> Vec<f64> {
    match layout {
        Layout::ThreeNights | Layout::TenNights => {
            let k = if matches!(layout, Layout::ThreeNights) {
                3
            } else {
                10
            };
            let per = n / k;
            let mut t = Vec::with_capacity(n);
            for night in 0..k {
                for j in 0..per {
                    t.push(60_000.0 + night as f64 + 0.25 * j as f64 / per as f64);
                }
            }
            t
        }
        Layout::SingleEpoch => vec![60_000.0; n],
        Layout::TwoEpochs => {
            let mut t = vec![60_000.0; n / 2];
            t.extend(vec![60_001.0; n - n / 2]);
            t
        }
        Layout::Unique => (0..n)
            .map(|k| 60_000.0 + 30.0 * k as f64 / n as f64)
            .collect(),
    }
}

#[test]
#[ignore]
fn review_triplet_work_scaling() {
    let config = SpanSearchConfig::default();
    for layout in [
        Layout::ThreeNights,
        Layout::TenNights,
        Layout::SingleEpoch,
        Layout::TwoEpochs,
        Layout::Unique,
    ] {
        for n in [1_000_usize, 10_000, 100_000, 1_000_000] {
            let times = layout_times(layout, n);
            let start = Instant::now();
            let nights = observation_nights(&times, None);
            let mut stats = TripletSearchStats::default();
            let spans = span_triplets_with_limits(
                &times, &nights, &config, 32, 4096, 1_048_576, 1_048_576, &mut stats,
            );
            let span_elapsed = start.elapsed();
            assert!(stats.pair_visits <= 4096);
            let start = Instant::now();
            let mut cstats = TripletSearchStats::default();
            let comb = combination_triplets_with_stats(&times, 32, 4096, 16384, &mut cstats);
            let comb_elapsed = start.elapsed();
            assert!(cstats.pair_visits <= 4096 && cstats.middle_visits <= 16384);
            eprintln!(
                "REVIEW-SCALING layout={layout:?} n={n} spans: triplets={} pair_visits={} middle_lookups={} t={:?} | combinations: triplets={} pair_visits={} middle_visits={} t={:?}",
                spans.len(), stats.pair_visits, stats.middle_visits, span_elapsed,
                comb.len(), cstats.pair_visits, cstats.middle_visits, comb_elapsed
            );
        }
    }
}

/// Quality impact of the shared pair-visit cap: distinct triplets (night
/// pairs) found with the default cap vs without it on dense multi-night arcs.
#[test]
#[ignore]
fn review_triplet_cap_quality() {
    let config = SpanSearchConfig::default();
    for nights_count in [10_usize, 20, 30] {
        for per_night in [16_usize, 64, 128, 256, 512] {
            let mut times = Vec::new();
            for night in 0..nights_count {
                for j in 0..per_night {
                    times.push(60_000.0 + night as f64 + 0.25 * j as f64 / per_night as f64);
                }
            }
            let nights = observation_nights(&times, None);
            let mut s1 = TripletSearchStats::default();
            let capped = span_triplets_with_limits(
                &times, &nights, &config, 32, 4096, 1_048_576, 1_048_576, &mut s1,
            );
            let mut s2 = TripletSearchStats::default();
            let start = Instant::now();
            let uncapped = span_triplets_with_limits(
                &times,
                &nights,
                &config,
                32,
                usize::MAX,
                usize::MAX,
                usize::MAX,
                &mut s2,
            );
            let windows = |v: &Vec<([usize; 3], (usize, usize))>| {
                let mut w: Vec<(usize, usize)> = v.iter().map(|x| x.1).collect();
                w.dedup();
                w.len()
            };
            eprintln!(
                "REVIEW-QUALITY nights={nights_count} per_night={per_night} n={} capped: triplets={} windows={} visits={} | uncapped: triplets={} windows={} visits={} t={:?}",
                times.len(), capped.len(), windows(&capped), s1.pair_visits,
                uncapped.len(), windows(&uncapped), s2.pair_visits, start.elapsed()
            );
        }
    }
}

/// End-to-end run_iod work with many exact duplicates / no valid triplets.
#[test]
#[ignore]
fn review_run_iod_end_to_end_scaling() {
    let p = TwoBodyPropagator::default();
    let opt = options();
    let pred = predictor(&p, &opt);
    for k in [1_usize, 10, 100, 1000, 5000] {
        let mut epochs = Vec::new();
        let mut codes = Vec::new();
        for row in 0..8 {
            for _ in 0..k {
                epochs.push(60_002.0 + 3.0 * row as f64);
                codes.push("X05");
            }
        }
        let b = observers_at(&epochs, &codes);
        let o = observe(&b, &vec![1.0 / 3600.0; epochs.len()]);
        for method in [
            ObservationSelectionMethod::Spans,
            ObservationSelectionMethod::Combinations,
        ] {
            let config = IodConfig {
                acceptance: IodAcceptance::Prefit,
                observation_selection_method: method,
                rchi2_threshold: 1e4,
                ..IodConfig::default()
            };
            let start = Instant::now();
            let r = run_iod(&pred, &o, &b, &config).unwrap();
            eprintln!(
                "REVIEW-E2E dup n={} {method:?} found={} tried={} eval={} t={:?}",
                o.len(),
                r.found(),
                r.triplets_tried,
                r.candidates_evaluated,
                start.elapsed()
            );
        }
        // No valid triplet: only two distinct epochs.
        let mut e2 = vec![60_002.0; 4 * k];
        e2.extend(vec![60_005.0; 4 * k]);
        let b2 = observers_at(&e2, &vec!["X05"; e2.len()]);
        let o2 = observe(&b2, &vec![1.0 / 3600.0; e2.len()]);
        let config = IodConfig {
            acceptance: IodAcceptance::Prefit,
            observation_selection_method: ObservationSelectionMethod::Combinations,
            ..IodConfig::default()
        };
        let start = Instant::now();
        let r = run_iod(&pred, &o2, &b2, &config).unwrap();
        eprintln!(
            "REVIEW-E2E two-epochs n={} found={} reason={:?} t={:?}",
            o2.len(),
            r.found(),
            r.not_found_reason,
            start.elapsed()
        );
    }
}

#[test]
fn fewer_than_three_distinct_epochs_is_rank_deficient_before_search() {
    let epochs = [60_002.0, 60_002.0, 60_005.0, 60_005.0, 60_002.0, 60_005.0];
    let observers = observers_at(&epochs, &["X05"; 6]);
    let observed = observe(&observers, &[1.0 / 3600.0; 6]);
    let propagator = TwoBodyPropagator::default();
    let options = options();
    let predictor = predictor(&propagator, &options);
    let result = run_iod(
        &predictor,
        &observed,
        &observers,
        &IodConfig {
            min_obs: 3,
            min_arc_length: 0.0,
            acceptance: IodAcceptance::Prefit,
            ..IodConfig::default()
        },
    )
    .unwrap();
    assert!(!result.found());
    assert!(result
        .not_found_reason
        .as_deref()
        .unwrap()
        .contains("2 distinct observation epochs"));
    assert_eq!(result.triplets_tried, 0);
    assert_eq!(result.candidates_evaluated, 0);
}

/// Validation must be eager and consistent for every OD input defect, on
/// empty and non-empty batches.
#[test]
fn review_input_validation_matrix() {
    let (o, b) = tie_heavy_arc();
    let p = TwoBodyPropagator::default();
    let opt = options();
    let pred = predictor(&p, &opt);
    let config = IodConfig {
        acceptance: IodAcceptance::Prefit,
        ..IodConfig::default()
    };
    let mut failures = Vec::new();
    let cases: Vec<(&str, usize, f64)> = vec![
        ("lon nan", 1, f64::NAN),
        ("lon inf", 1, f64::INFINITY),
        ("lon -inf", 1, f64::NEG_INFINITY),
        ("lon 360", 1, 360.0),
        ("lon -1e-12", 1, -1e-12),
        ("lat nan", 2, f64::NAN),
        ("lat 90.0000001", 2, 90.000_000_1),
        ("lat -inf", 2, f64::NEG_INFINITY),
        ("rho inf", 0, f64::INFINITY),
        ("vlat -inf", 5, f64::NEG_INFINITY),
    ];
    for (name, col, value) in cases {
        let mut bad = o.clone();
        if let crate::CoordinateValues::Spherical(v) = &mut bad.values {
            v[5][col] = value;
        }
        match run_iod(&pred, &bad, &b, &config) {
            Err(PropagationError::InvalidRequest(_)) => {}
            other => failures.push(format!("{name}: {:?}", other.map(|r| r.not_found_reason))),
        }
    }
    for (name, idx, value) in [
        ("var0", 7, 0.0),
        ("var neg", 14, -1e-9),
        ("var nan", 7, f64::NAN),
        ("var inf", 14, f64::INFINITY),
        ("cov offdiag nan", 8, f64::NAN),
        ("cov not PD", 8, 1.0),
    ] {
        let mut bad = o.clone();
        bad.covariance.as_mut().unwrap().values_row_major[5 * 36 + idx] = value;
        if idx == 8 && value.is_finite() {
            bad.covariance.as_mut().unwrap().values_row_major[5 * 36 + 13] = value;
        }
        match run_iod(&pred, &bad, &b, &config) {
            Err(PropagationError::InvalidRequest(_)) => {}
            other => failures.push(format!("{name}: {:?}", other.map(|r| r.not_found_reason))),
        }
    }
    for (name, idx, value) in [("obs x nan", 0, f64::NAN), ("obs vz inf", 5, f64::INFINITY)] {
        let mut bad = b.clone();
        if let crate::CoordinateValues::Cartesian(v) = &mut bad.coordinates.values {
            v[5][idx] = value;
        }
        match run_iod(&pred, &o, &bad, &config) {
            Err(PropagationError::InvalidRequest(_)) => {}
            other => failures.push(format!("{name}: {:?}", other.map(|r| r.not_found_reason))),
        }
    }
    // Fewer than min_obs rows must still validate content eagerly.
    let few: Vec<usize> = vec![0, 1];
    let mut so = take_coordinate_batch(&o, &few).unwrap();
    let sb = take_observer_batch(&b, &few).unwrap();
    if let crate::CoordinateValues::Spherical(v) = &mut so.values {
        v[0][1] = f64::NAN;
    }
    match run_iod(&pred, &so, &sb, &config) {
        Err(PropagationError::InvalidRequest(_)) => {}
        other => failures.push(format!(
            "below-min_obs NaN: {:?}",
            other.map(|r| r.not_found_reason)
        )),
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Prefit outlier trial: keep-mask rebuilt with `removed.contains` for every
/// count => O(n * m^2) per candidate (m = contamination * n), plus a
/// legacy triplet budget 3 * (m + 1) that also grows with n.
#[test]
#[ignore]
fn review_prefit_outlier_trial_scaling() {
    let p = TwoBodyPropagator::default();
    let opt = options();
    let pred = predictor(&p, &opt);
    for (k, max_triplets) in [
        (32_usize, Some(1)),
        (64, Some(1)),
        (125, Some(1)),
        (250, Some(1)),
        (500, Some(1)),
        (32, None),
        (64, None),
        (125, None),
    ] {
        let mut epochs = Vec::new();
        for row in 0..8 {
            for _ in 0..k {
                epochs.push(60_002.0 + 3.0 * row as f64);
            }
        }
        let b = observers_at(&epochs, &vec!["X05"; epochs.len()]);
        let o = observe(&b, &vec![1.0 / 3600.0; epochs.len()]);
        let config = IodConfig {
            acceptance: IodAcceptance::Prefit,
            observation_selection_method: ObservationSelectionMethod::Combinations,
            rchi2_threshold: 1e-300,
            min_arc_length: 0.0,
            contamination_percentage: 20.0,
            max_triplets,
            ..IodConfig::default()
        };
        let start = Instant::now();
        let r = run_iod(&pred, &o, &b, &config).unwrap();
        eprintln!(
            "REVIEW-PREFIT-OUTLIER n={} max_triplets={max_triplets:?} found={} tried={} eval={} t={:?}",
            o.len(), r.found(), r.triplets_tried, r.candidates_evaluated, start.elapsed()
        );
    }
}

/// Default corrected acceptance end-to-end on duplicate-heavy and
/// no-valid-triplet arcs.
#[test]
#[ignore]
fn review_corrected_end_to_end_scaling() {
    use super::super::od_refine::{run_iod_corrected, WhitenedFitConfig};
    let p = TwoBodyPropagator::default();
    let opt = options();
    let pred = predictor(&p, &opt);
    for k in [1_usize, 10, 100, 1000, 4000] {
        let mut epochs = Vec::new();
        for row in 0..8 {
            for _ in 0..k {
                epochs.push(60_002.0 + 3.0 * row as f64);
            }
        }
        let b = observers_at(&epochs, &vec!["X05"; epochs.len()]);
        let o = observe(&b, &vec![1.0 / 3600.0; epochs.len()]);
        for method in [
            ObservationSelectionMethod::Spans,
            ObservationSelectionMethod::Combinations,
        ] {
            let config = IodConfig {
                observation_selection_method: method,
                ..IodConfig::default()
            };
            let start = Instant::now();
            let r = run_iod_corrected(
                &pred,
                &o,
                &b,
                &config,
                &WhitenedFitConfig::default(),
                &NoopProvider,
                &ZeroTranslationProvider,
            )
            .unwrap();
            eprintln!("REVIEW-CORRECTED dup n={} {method:?} found={} tried={} eval={} corrections={} t={:?}", o.len(), r.found(), r.triplets_tried, r.candidates_evaluated, r.corrections_run, start.elapsed());
        }
        let mut e2 = vec![60_002.0; 4 * k];
        e2.extend(vec![60_003.0; 4 * k]);
        let b2 = observers_at(&e2, &vec!["X05"; e2.len()]);
        let o2 = observe(&b2, &vec![1.0 / 3600.0; e2.len()]);
        let start = Instant::now();
        let r = run_iod_corrected(
            &pred,
            &o2,
            &b2,
            &IodConfig::default(),
            &WhitenedFitConfig::default(),
            &NoopProvider,
            &ZeroTranslationProvider,
        )
        .unwrap();
        eprintln!(
            "REVIEW-CORRECTED two-epochs n={} found={} ranging={} reason={:?} t={:?}",
            o2.len(),
            r.found(),
            r.ranging.is_some(),
            r.not_found_reason,
            start.elapsed()
        );
    }
}
