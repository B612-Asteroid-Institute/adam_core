use std::path::Path;
use std::sync::{Arc, Barrier};

use adam_core::coords::{
    CoordinateBatch, DataFrame, OrbitBatch, OriginArray, OriginId, SourceCatalogBatch, TimeArray,
    TimeColumn, TimeScale,
};
use adam_core::{estimate_colors, ColorEstimationResources, ColorFitStatus, ColorPhaseModel};

fn one_source_catalog() -> SourceCatalogBatch {
    SourceCatalogBatch {
        id: vec!["source-1".to_string()],
        exposure_id: vec![None],
        time: TimeColumn::new(TimeScale::Utc, vec![60_000], vec![0]),
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
        exposure_start_time: TimeColumn::new(TimeScale::Utc, vec![60_000], vec![0]),
        exposure_duration: vec![None],
        exposure_seeing: vec![None],
        exposure_depth_5sigma: vec![None],
        object_id: vec![Some("object-1".to_string())],
        catalog_id: vec!["catalog-1".to_string()],
        astcat: vec![None],
        photcat: vec![None],
    }
}

fn empty_orbits() -> OrbitBatch {
    let coordinates = CoordinateBatch::cartesian(
        vec![],
        DataFrame::Ecliptic,
        OriginArray::repeat(OriginId::from_code("SUN"), 0),
        Some(TimeArray::new(TimeScale::Tdb, vec![]).unwrap()),
        None,
    )
    .unwrap();
    OrbitBatch::new(vec![], vec![], coordinates).unwrap()
}

#[test]
fn identical_concurrent_first_resource_initialization_waits_and_succeeds() {
    let observations = one_source_catalog();
    let orbits = empty_orbits();
    let kernel_paths =
        vec![Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/minimal_lsk.tls")];
    let obscodes = r#"{"500":{"Longitude":0.0,"cos":0.0,"sin":0.0,"Name":"Geocentric"}}"#;
    let resources =
        ColorEstimationResources::bundled(kernel_paths.clone(), obscodes.to_string());
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
                assert_eq!(result.rows[0].status, ColorFitStatus::NoOrbit);
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
    assert_eq!(result.rows[0].status, ColorFitStatus::NoOrbit);
}
