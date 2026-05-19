use adam_core_rs_coords::types::{Frame, SchemaResult};
use adam_core_rs_coords::{
    origin_mu_au3_day2, propagate_2body_along_arc, CoordinateBatch, CovariancePropagation, Epoch,
    EpochPolicy, ObjectId, OrbitBatch, OrbitId, OrbitVariantBatch, OriginArray, OriginId,
    PropagationOptions, PropagationRequest, Propagator, SchemaError, TimeArray, TimeScale,
    TimeScaleProvider, TwoBodyPropagator, VariantId,
};
use std::hint::black_box;
use std::time::Instant;

const ROWS: usize = 1_000;
const TIMES: usize = 20;
const REPEATS: usize = 11;
const WARMUP: usize = 3;
const MAX_ITER: usize = 1_000;
const TOL: f64 = 1.0e-14;

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
    let orbits = build_orbits(ROWS);
    let variants = build_variants(&orbits);
    let times = build_target_times(TIMES);
    let options = PropagationOptions {
        chunk_size: Some(64),
        thread_limit: Some(1),
        epoch_policy: EpochPolicy::CrossProduct,
        covariance: CovariancePropagation::None,
    };
    let propagator = TwoBodyPropagator::default();
    let provider = NoopProvider;

    let raw = measure(|| raw_kernel_rows(&orbits, &times), REPEATS, WARMUP);
    let typed_orbits = measure(
        || {
            let request = PropagationRequest::new(&orbits, &times, options.clone()).unwrap();
            propagator
                .propagate(&request, &provider)
                .unwrap()
                .orbits
                .len()
        },
        REPEATS,
        WARMUP,
    );
    let typed_variants = measure(
        || {
            let request =
                PropagationRequest::new_variants(&variants, &times, options.clone()).unwrap();
            propagator
                .propagate(&request, &provider)
                .unwrap()
                .variants
                .as_ref()
                .unwrap()
                .len()
        },
        REPEATS,
        WARMUP,
    );

    println!("surface,rows,times,repeats,seconds_p50,seconds_p95,ratio_p50_vs_raw");
    print_summary("raw_kernel_serial", raw, raw.p50);
    print_summary("typed_orbits_single_thread", typed_orbits, raw.p50);
    print_summary("typed_variants_single_thread", typed_variants, raw.p50);
}

fn print_summary(surface: &str, summary: Summary, raw_p50: f64) {
    println!(
        "{surface},{ROWS},{TIMES},{REPEATS},{:.9},{:.9},{:.3}",
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
