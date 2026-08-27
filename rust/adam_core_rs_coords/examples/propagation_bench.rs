use adam_core_rs_coords::types::{Frame, SchemaResult};
use adam_core_rs_coords::{
    generate_ephemeris, generate_ephemeris_2body_row, origin_mu_au3_day2,
    propagate_2body_along_arc, CoordinateBatch, CovariancePropagation, EphemerisOptions,
    EphemerisPhotometryOptions, Epoch, EpochPolicy, ObjectId, ObservatoryCode, ObserverBatch,
    OrbitBatch, OrbitId, OrbitVariantBatch, OriginArray, OriginId, PropagationOptions,
    PropagationRequest, Propagator, SchemaError, TimeArray, TimeScale, TimeScaleProvider,
    TwoBodyPropagator, VariantId,
};
use std::hint::black_box;
use std::time::Instant;

const ROWS: usize = 1_000;
const TIMES: usize = 20;
const REPEATS: usize = 11;
const WARMUP: usize = 3;
const MAX_ITER: usize = 1_000;
const TOL: f64 = 1.0e-14;
const COMPARISON_SCOPE: &str = "rust_internal_not_python_jax_quivr";

struct NoopProvider;

impl TimeScaleProvider for NoopProvider {
    fn rescale(&self, _times: &TimeArray, _new_scale: TimeScale) -> SchemaResult<TimeArray> {
        Err(SchemaError::InvalidRecordBatch(
            "benchmark uses TDB inputs and should not call the provider".to_string(),
        ))
    }
}

#[derive(Clone, Copy)]
struct Summary {
    p50: f64,
    p95: f64,
}

fn main() {
    // Diagnostic benchmark only: compares typed Rust propagation overhead and
    // Rayon pool modes against the raw serial Rust two-body kernel. It is not a
    // Python/quivr/JAX comparison and should not be cited as such.
    let orbits = build_orbits(ROWS);
    let variants = build_variants(&orbits);
    let times = build_target_times(TIMES);
    let observers = build_observers(&times);
    let per_call_pool_options = PropagationOptions {
        chunk_size: Some(64),
        thread_limit: Some(1),
        epoch_policy: EpochPolicy::CrossProduct,
        covariance: CovariancePropagation::None,
    };
    let global_pool_options = PropagationOptions {
        thread_limit: None,
        ..per_call_pool_options.clone()
    };
    let propagator = TwoBodyPropagator::default();
    let provider = NoopProvider;

    let raw = measure(|| raw_kernel_rows(&orbits, &times), REPEATS, WARMUP);
    let typed_orbits_per_call_pool = measure(
        || {
            propagate_orbits(
                &propagator,
                &provider,
                &orbits,
                &times,
                &per_call_pool_options,
            )
        },
        REPEATS,
        WARMUP,
    );
    let typed_variants_per_call_pool = measure(
        || {
            propagate_variants(
                &propagator,
                &provider,
                &variants,
                &times,
                &per_call_pool_options,
            )
        },
        REPEATS,
        WARMUP,
    );
    let typed_orbits_global_pool = measure(
        || {
            propagate_orbits(
                &propagator,
                &provider,
                &orbits,
                &times,
                &global_pool_options,
            )
        },
        REPEATS,
        WARMUP,
    );
    let typed_variants_global_pool = measure(
        || {
            propagate_variants(
                &propagator,
                &provider,
                &variants,
                &times,
                &global_pool_options,
            )
        },
        REPEATS,
        WARMUP,
    );
    let ephemeris_options = EphemerisOptions {
        propagation: global_pool_options.clone(),
        output_time_scale: TimeScale::Tdb,
        photometry: EphemerisPhotometryOptions {
            predict_magnitude_v: true,
            predict_phase_angle: true,
            h_v: Some(vec![Some(18.0); ROWS]),
            g: Some(vec![Some(0.15); ROWS]),
        },
        ..EphemerisOptions::default()
    };
    let raw_ephemeris = measure(|| raw_ephemeris_rows(&orbits, &observers), REPEATS, WARMUP);
    let typed_ephemeris_global_pool = measure(
        || {
            typed_ephemeris_rows(
                &propagator,
                &provider,
                &orbits,
                &observers,
                &ephemeris_options,
            )
        },
        REPEATS,
        WARMUP,
    );

    let rayon_threads = rayon::current_num_threads();
    println!("surface,comparison_scope,baseline,thread_mode,rows,times,repeats,rayon_threads,seconds_p50,seconds_p95,ratio_p50_vs_baseline");
    print_summary(
        "raw_kernel_serial",
        "raw_kernel_serial",
        "serial",
        raw,
        raw.p50,
        rayon_threads,
    );
    print_summary(
        "typed_orbits_per_call_pool_1_thread",
        "raw_kernel_serial",
        "per_call_rayon_pool_1_thread",
        typed_orbits_per_call_pool,
        raw.p50,
        rayon_threads,
    );
    print_summary(
        "typed_variants_per_call_pool_1_thread",
        "raw_kernel_serial",
        "per_call_rayon_pool_1_thread",
        typed_variants_per_call_pool,
        raw.p50,
        rayon_threads,
    );
    print_summary(
        "typed_orbits_global_pool",
        "raw_kernel_serial",
        "default_global_rayon_pool",
        typed_orbits_global_pool,
        raw.p50,
        rayon_threads,
    );
    print_summary(
        "typed_variants_global_pool",
        "raw_kernel_serial",
        "default_global_rayon_pool",
        typed_variants_global_pool,
        raw.p50,
        rayon_threads,
    );
    print_summary(
        "raw_ephemeris_kernel_serial",
        "raw_ephemeris_kernel_serial",
        "serial",
        raw_ephemeris,
        raw_ephemeris.p50,
        rayon_threads,
    );
    print_summary(
        "typed_ephemeris_global_pool",
        "raw_ephemeris_kernel_serial",
        "default_global_rayon_pool",
        typed_ephemeris_global_pool,
        raw_ephemeris.p50,
        rayon_threads,
    );
}

fn propagate_orbits(
    propagator: &TwoBodyPropagator,
    provider: &NoopProvider,
    orbits: &OrbitBatch,
    times: &TimeArray,
    options: &PropagationOptions,
) -> usize {
    let request = PropagationRequest::new(orbits, times, options.clone()).unwrap();
    propagator
        .propagate(&request, provider)
        .unwrap()
        .orbits
        .len()
}

fn propagate_variants(
    propagator: &TwoBodyPropagator,
    provider: &NoopProvider,
    variants: &OrbitVariantBatch,
    times: &TimeArray,
    options: &PropagationOptions,
) -> usize {
    let request = PropagationRequest::new_variants(variants, times, options.clone()).unwrap();
    propagator
        .propagate(&request, provider)
        .unwrap()
        .variants
        .as_ref()
        .unwrap()
        .len()
}

fn typed_ephemeris_rows(
    propagator: &TwoBodyPropagator,
    provider: &NoopProvider,
    orbits: &OrbitBatch,
    observers: &ObserverBatch,
    options: &EphemerisOptions,
) -> usize {
    generate_ephemeris(propagator, orbits, observers, options, provider)
        .unwrap()
        .ephemeris
        .len()
}

fn print_summary(
    surface: &str,
    baseline: &str,
    thread_mode: &str,
    summary: Summary,
    raw_p50: f64,
    rayon_threads: usize,
) {
    println!(
        "{surface},{COMPARISON_SCOPE},{baseline},{thread_mode},{ROWS},{TIMES},{REPEATS},{rayon_threads},{:.9},{:.9},{:.3}",
        summary.p50,
        summary.p95,
        summary.p50 / raw_p50
    );
}

fn measure<F>(mut run_once: F, repeats: usize, warmup: usize) -> Summary
where
    F: FnMut() -> usize,
{
    for _ in 0..warmup {
        black_box(run_once());
    }
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        black_box(run_once());
        samples.push(start.elapsed().as_secs_f64());
    }
    samples.sort_by(|left, right| left.partial_cmp(right).unwrap());
    Summary {
        p50: percentile(&samples, 0.50),
        p95: percentile(&samples, 0.95),
    }
}

fn percentile(sorted_samples: &[f64], percentile: f64) -> f64 {
    let last = sorted_samples.len().saturating_sub(1);
    let index = ((last as f64) * percentile).round() as usize;
    sorted_samples[index]
}

fn build_orbits(rows: usize) -> OrbitBatch {
    let states = (0..rows)
        .map(|row| {
            let offset = row as f64 * 1.0e-6;
            [
                1.0 + offset,
                0.2 - offset,
                0.1 + 0.5 * offset,
                0.001,
                0.015 + 0.1 * offset,
                0.0005,
            ]
        })
        .collect::<Vec<_>>();
    let times = TimeArray::new(
        TimeScale::Tdb,
        (0..rows)
            .map(|row| Epoch::new(60_000 + (row % 7) as i64, 0))
            .collect(),
    )
    .unwrap();
    let coordinates = CoordinateBatch::cartesian(
        states,
        Frame::Ecliptic,
        OriginArray::repeat(OriginId::Named("SUN".to_string()), rows),
        Some(times),
        None,
    )
    .unwrap();
    OrbitBatch::new(
        (0..rows)
            .map(|row| OrbitId(format!("orbit-{row}")))
            .collect(),
        (0..rows)
            .map(|row| Some(ObjectId(format!("object-{}", row / 4))))
            .collect(),
        coordinates,
    )
    .unwrap()
}

fn build_variants(orbits: &OrbitBatch) -> OrbitVariantBatch {
    let rows = orbits.len();
    OrbitVariantBatch::new(
        orbits.orbit_id.clone(),
        orbits.object_id.clone(),
        (0..rows)
            .map(|row| Some(VariantId(format!("variant-{}", row % 16))))
            .collect(),
        vec![Some(1.0 / rows as f64); rows],
        vec![Some(1.0 / (rows as f64 * rows as f64)); rows],
        orbits.coordinates.clone(),
    )
    .unwrap()
}

fn build_target_times(rows: usize) -> TimeArray {
    TimeArray::new(
        TimeScale::Tdb,
        (0..rows)
            .map(|row| Epoch::new(60_000 + row as i64, 0))
            .collect(),
    )
    .unwrap()
}

fn build_observers(times: &TimeArray) -> ObserverBatch {
    let rows = times.len();
    let states = (0..rows)
        .map(|row| {
            let offset = row as f64 * 1.0e-5;
            [
                0.95 + offset,
                -0.2 + 0.5 * offset,
                0.02,
                0.002,
                0.016,
                0.0001,
            ]
        })
        .collect::<Vec<_>>();
    let coordinates = CoordinateBatch::cartesian(
        states,
        Frame::Ecliptic,
        OriginArray::repeat(OriginId::Named("SUN".to_string()), rows),
        Some(times.clone()),
        None,
    )
    .unwrap();
    ObserverBatch::new(
        (0..rows)
            .map(|row| ObservatoryCode(format!("obs-{row}")))
            .collect(),
        coordinates,
    )
    .unwrap()
}

fn raw_kernel_rows(orbits: &OrbitBatch, times: &TimeArray) -> usize {
    let states = orbits.coordinates.values.cartesian().unwrap();
    let orbit_times = orbits.coordinates.times.as_ref().unwrap();
    let mu = origin_mu_au3_day2(&OriginId::Named("SUN".to_string())).unwrap();
    let mut rows = 0;
    for (orbit_index, state) in states.iter().enumerate() {
        let dts = times
            .epochs
            .iter()
            .map(|epoch| epoch.mjd() - orbit_times.epochs[orbit_index].mjd())
            .collect::<Vec<_>>();
        let propagated = propagate_2body_along_arc(*state, &dts, mu, MAX_ITER, TOL);
        rows += propagated.len();
        black_box(propagated);
    }
    rows
}

fn raw_ephemeris_rows(orbits: &OrbitBatch, observers: &ObserverBatch) -> usize {
    let states = orbits.coordinates.values.cartesian().unwrap();
    let orbit_times = orbits.coordinates.times.as_ref().unwrap();
    let observer_states = observers.coordinates.values.cartesian().unwrap();
    let observer_times = observers.coordinates.times.as_ref().unwrap();
    let mu = origin_mu_au3_day2(&OriginId::Named("SUN".to_string())).unwrap();
    let mut rows = 0;
    for (orbit_index, state) in states.iter().enumerate() {
        let dts = observer_times
            .epochs
            .iter()
            .map(|epoch| epoch.mjd() - orbit_times.epochs[orbit_index].mjd())
            .collect::<Vec<_>>();
        let propagated = propagate_2body_along_arc(*state, &dts, mu, MAX_ITER, TOL);
        for (propagated_state, observer_state) in propagated.iter().zip(observer_states.iter()) {
            let row = generate_ephemeris_2body_row::<f64>(
                *propagated_state,
                *observer_state,
                mu,
                1.0e-12,
                MAX_ITER,
                1.0e-15,
                false,
                10,
            );
            black_box(row);
            rows += 1;
        }
    }
    rows
}
