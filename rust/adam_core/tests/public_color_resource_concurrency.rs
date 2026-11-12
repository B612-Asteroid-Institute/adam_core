use std::sync::{Arc, Barrier};

use adam_core::coords::{
    CoordinateBatch, DataFrame, ObjectId, OrbitBatch, OrbitId, OriginArray, OriginId,
    SourceCatalogBatch, TimeArray, TimeColumn, TimeScale,
};
use adam_core::kernel_data::Resolver;
use adam_core::{estimate_colors, ColorEstimationResources, ColorFitStatus, ColorPhaseModel};

fn one_source_catalog() -> SourceCatalogBatch {
    SourceCatalogBatch {
        id: vec!["source-1".to_string()],
        exposure_id: vec![None],
        time: TimeColumn::new(TimeScale::Tdb, vec![60_000], vec![0]),
        ra: vec![0.0],
        dec: vec![0.0],
        ra_sigma: vec![None],
        dec_sigma: vec![None],
        radec_corr: vec![None],
        mag: vec![Some(20.0)],
        mag_sigma: vec![Some(0.1)],
        fwhm: vec![None],
        a: vec![None],
        a_sigma: vec![None],
        b: vec![None],
        b_sigma: vec![None],
        pa: vec![None],
        pa_sigma: vec![None],
        observatory_code: vec!["500".to_string()],
        filter: vec![Some("g".to_string())],
        exposure_start_time: TimeColumn::new(TimeScale::Tdb, vec![60_000], vec![0]),
        exposure_duration: vec![None],
        exposure_seeing: vec![None],
        exposure_depth_5sigma: vec![None],
        object_id: vec![Some("object-1".to_string())],
        catalog_id: vec!["catalog-1".to_string()],
        astcat: vec![None],
        photcat: vec![None],
    }
}

fn one_orbit() -> OrbitBatch {
    let coordinates = CoordinateBatch::cartesian(
        vec![[1.0, 0.0, 0.0, 0.0, 0.017_202_098_95, 0.0]],
        DataFrame::Ecliptic,
        OriginArray::repeat(OriginId::from_code("SUN"), 1),
        Some(TimeArray::from_parts(TimeScale::Tdb, vec![60_000], vec![0]).unwrap()),
        None,
    )
    .unwrap();
    OrbitBatch::new(
        vec![OrbitId("orbit-1".to_string())],
        vec![Some(ObjectId("object-1".to_string()))],
        coordinates,
    )
    .unwrap()
}

#[test]
#[ignore = "requires resolvable real observer-geometry kernels and MPC obscodes (installed Python, cache, or network)"]
fn identical_concurrent_first_resource_initialization_waits_and_succeeds() {
    let observations = one_source_catalog();
    let orbits = one_orbit();
    let resolver = Resolver::from_env();
    let kernel_paths = resolver
        .observer_geometry_kernel_paths()
        .expect("real observer-geometry kernels must resolve");
    let obscodes = resolver
        .obscodes_json()
        .expect("real MPC obscodes must resolve");
    let resources = ColorEstimationResources::bundled(kernel_paths.clone(), obscodes);
    let barrier = Arc::new(Barrier::new(8));

    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..8 {
            let barrier = Arc::clone(&barrier);
            let observations = &observations;
            let orbits = &orbits;
            let resources = &resources;
            handles.push(scope.spawn(move || {
                barrier.wait();
                let result = estimate_colors(
                    observations,
                    orbits,
                    ColorPhaseModel::C1C2,
                    false,
                    None,
                    false,
                    None,
                    resources,
                )
                .unwrap();
                assert_eq!(result.len(), 1);
                assert_eq!(result.rows[0].status, ColorFitStatus::Ok);
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
    });

    // Once a nonempty table is installed, later JSON is intentionally ignored.
    let later = ColorEstimationResources::bundled(
        kernel_paths,
        "not valid JSON and must not be parsed".to_string(),
    );
    let result = estimate_colors(
        &observations,
        &orbits,
        ColorPhaseModel::C1C2,
        false,
        None,
        false,
        None,
        &later,
    )
    .unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result.rows[0].status, ColorFitStatus::Ok);
}
