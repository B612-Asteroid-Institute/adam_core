use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::OnceLock;

use adam_core::coords::{
    CoordinateBatch, DataFrame, ObjectId, OrbitBatch, OrbitId, OriginArray, OriginId,
    SourceCatalogBatch, TimeArray, TimeColumn, TimeScale,
};
use adam_core::kernel_data::Resolver;
use adam_core::{
    estimate_colors, ColorComposition, ColorEstimationError, ColorEstimationResources,
    ColorFitStatus, ColorPhaseModel,
};

fn empty_source_catalog() -> SourceCatalogBatch {
    let time = TimeColumn::new(TimeScale::Utc, vec![], vec![]);
    SourceCatalogBatch {
        id: vec![],
        exposure_id: vec![],
        time: time.clone(),
        ra: vec![],
        dec: vec![],
        ra_sigma: vec![],
        dec_sigma: vec![],
        radec_corr: vec![],
        mag: vec![],
        mag_sigma: vec![],
        fwhm: vec![],
        a: vec![],
        a_sigma: vec![],
        b: vec![],
        b_sigma: vec![],
        pa: vec![],
        pa_sigma: vec![],
        observatory_code: vec![],
        filter: vec![],
        exposure_start_time: time,
        exposure_duration: vec![],
        exposure_seeing: vec![],
        exposure_depth_5sigma: vec![],
        object_id: vec![],
        catalog_id: vec![],
        astcat: vec![],
        photcat: vec![],
    }
}

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

fn source_catalog_for_objects(object_ids: &[&str]) -> SourceCatalogBatch {
    let n = object_ids.len();
    SourceCatalogBatch {
        id: (0..n).map(|row| format!("order-source-{row}")).collect(),
        exposure_id: vec![None; n],
        time: TimeColumn::new(TimeScale::Tdb, vec![60_000; n], vec![0; n]),
        ra: vec![0.0; n],
        dec: vec![0.0; n],
        ra_sigma: vec![None; n],
        dec_sigma: vec![None; n],
        radec_corr: vec![None; n],
        mag: vec![Some(20.0); n],
        mag_sigma: vec![Some(0.1); n],
        fwhm: vec![None; n],
        a: vec![None; n],
        a_sigma: vec![None; n],
        b: vec![None; n],
        b_sigma: vec![None; n],
        pa: vec![None; n],
        pa_sigma: vec![None; n],
        observatory_code: vec!["SUN".to_string(); n],
        filter: vec![Some("LSST_g".to_string()); n],
        exposure_start_time: TimeColumn::new(TimeScale::Tdb, vec![60_000; n], vec![0; n]),
        exposure_duration: vec![None; n],
        exposure_seeing: vec![None; n],
        exposure_depth_5sigma: vec![None; n],
        object_id: object_ids
            .iter()
            .map(|object_id| Some((*object_id).to_string()))
            .collect(),
        catalog_id: vec!["catalog-1".to_string(); n],
        astcat: vec![None; n],
        photcat: vec![None; n],
    }
}

fn fit_source_catalog(observatory_code: &str) -> SourceCatalogBatch {
    let n = 8;
    let filters = ["LSST_g", "LSST_r"]
        .into_iter()
        .flat_map(|filter| std::iter::repeat_n(Some(filter.to_string()), 4))
        .collect();
    let magnitudes = [20.0, 19.5]
        .into_iter()
        .flat_map(|magnitude| std::iter::repeat_n(Some(magnitude), 4))
        .collect();
    SourceCatalogBatch {
        id: (0..n).map(|row| format!("source-{row}")).collect(),
        exposure_id: vec![None; n],
        time: TimeColumn::new(TimeScale::Tdb, vec![60_000; n], vec![0; n]),
        ra: vec![0.0; n],
        dec: vec![0.0; n],
        ra_sigma: vec![None; n],
        dec_sigma: vec![None; n],
        radec_corr: vec![None; n],
        mag: magnitudes,
        mag_sigma: vec![Some(0.1); n],
        fwhm: vec![None; n],
        a: vec![None; n],
        a_sigma: vec![None; n],
        b: vec![None; n],
        b_sigma: vec![None; n],
        pa: vec![None; n],
        pa_sigma: vec![None; n],
        observatory_code: vec![observatory_code.to_string(); n],
        filter: filters,
        exposure_start_time: TimeColumn::new(TimeScale::Tdb, vec![60_000; n], vec![0; n]),
        exposure_duration: vec![None; n],
        exposure_seeing: vec![None; n],
        exposure_depth_5sigma: vec![None; n],
        object_id: vec![Some("object-1".to_string()); n],
        catalog_id: vec!["catalog-1".to_string(); n],
        astcat: vec![None; n],
        photcat: vec![None; n],
    }
}

fn one_orbit(origin: OriginId) -> OrbitBatch {
    let coordinates = CoordinateBatch::cartesian(
        vec![[1.0, 0.0, 0.0, 0.0, 0.017_202_098_95, 0.0]],
        DataFrame::Ecliptic,
        OriginArray::repeat(origin, 1),
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

fn no_geometry_resources() -> ColorEstimationResources {
    ColorEstimationResources::bundled(vec![], String::new())
}

static REAL_GEOMETRY_RESOURCES: OnceLock<ColorEstimationResources> = OnceLock::new();

fn real_geometry_resources() -> &'static ColorEstimationResources {
    REAL_GEOMETRY_RESOURCES.get_or_init(|| {
        ColorEstimationResources::resolve(&Resolver::from_env())
            .expect("observer-geometry kernels and MPC obscodes must resolve")
    })
}

#[test]
fn public_estimate_colors_owns_high_level_projection_contract() {
    let observations = empty_source_catalog();
    let orbits = empty_orbits();
    let resources = ColorEstimationResources::new(PathBuf::from("."), vec![], String::new());

    let error = estimate_colors(
        &observations,
        &orbits,
        ColorPhaseModel::C1C2,
        true,
        Some(ColorComposition::Template("NEO".to_string())),
        false,
        None,
        &resources,
    )
    .unwrap_err();

    assert!(matches!(
        error,
        ColorEstimationError::InvalidConfiguration(_)
    ));
    assert_eq!(
        error.to_string(),
        "composition is accepted only with an explicit source-to-target projection"
    );
}

#[test]
fn public_estimate_colors_returns_typed_empty_orbit_result() {
    let observations = one_source_catalog();
    let orbits = empty_orbits();
    let resources = no_geometry_resources();
    let debug = format!("{resources:?}");
    assert!(debug.contains("observer_geometry_kernel_paths"));
    assert!(debug.contains("mpc_obscodes_json_bytes"));

    let result = estimate_colors(
        &observations,
        &orbits,
        ColorPhaseModel::C1C2,
        false,
        None,
        false,
        None,
        &resources,
    )
    .unwrap();

    assert_eq!(result.len(), 1);
    assert_eq!(result.rows[0].object_id, "object-1");
    assert_eq!(result.rows[0].status, ColorFitStatus::NoOrbit);
    assert_eq!(result.rows[0].status.as_str(), "no_orbit");
    assert_eq!(result.rows[0].status.to_string(), "no_orbit");
    assert_eq!(result.rows[0].num_obs, Some(1));
    assert_eq!(result.to_record_batch().unwrap().num_rows(), 1);
}

#[test]
fn public_estimate_colors_preserves_nonlexical_first_appearance_order() {
    let observations = source_catalog_for_objects(&["zeta", "alpha", "zeta", "mu"]);
    let resources = no_geometry_resources();

    let result = estimate_colors(
        &observations,
        &empty_orbits(),
        ColorPhaseModel::C1C2,
        false,
        None,
        false,
        None,
        &resources,
    )
    .unwrap();

    assert_eq!(
        result
            .rows
            .iter()
            .map(|row| row.object_id.as_str())
            .collect::<Vec<_>>(),
        ["zeta", "alpha", "mu"]
    );
    assert!(result
        .rows
        .iter()
        .all(|row| row.status == ColorFitStatus::NoOrbit));
    assert_eq!(result.rows[0].num_obs, Some(2));
}

#[test]
fn public_estimate_colors_requires_observer_geometry_resources_only_when_needed() {
    let error = estimate_colors(
        &fit_source_catalog("SUN"),
        &one_orbit(OriginId::from_code("SUN")),
        ColorPhaseModel::C1C2,
        false,
        None,
        false,
        None,
        &no_geometry_resources(),
    )
    .unwrap_err();

    assert_eq!(
        error.to_string(),
        "color estimation requires observer-geometry SPICE kernel paths"
    );
}

#[test]
#[ignore = "requires resolvable real observer-geometry kernels and MPC obscodes (installed Python, cache, or network)"]
fn public_estimate_colors_real_fit_matches_frozen_sun_observer_golden() {
    let observations = fit_source_catalog("SUN");
    let orbits = one_orbit(OriginId::from_code("SUN"));
    let resources = real_geometry_resources();

    let result = estimate_colors(
        &observations,
        &orbits,
        ColorPhaseModel::C1C2,
        false,
        None,
        false,
        None,
        resources,
    )
    .unwrap();
    let row = &result.rows[0];

    assert_eq!(row.status, ColorFitStatus::Ok);
    assert_eq!(
        row.filter_ids.as_deref(),
        Some(&["LSST_g".to_string(), "LSST_r".to_string()][..])
    );
    assert_eq!(row.num_used_by_filter.as_deref(), Some(&[4, 4][..]));
    assert_eq!(row.num_obs, Some(8));
    assert_eq!(row.num_invalid, Some(0));
    assert_eq!(row.num_unsupported_filter, Some(0));
    assert_eq!(row.num_clipped, Some(0));
    assert_eq!(row.num_used, Some(8));
    let magnitudes = row.absolute_magnitudes.as_ref().unwrap();
    assert!((magnitudes[0] - 20.0).abs() < 1.0e-10);
    assert!((magnitudes[1] - 19.5).abs() < 1.0e-10);
    assert!((row.colors.as_ref().unwrap()[0] - 0.5).abs() < 1.0e-10);
}

#[test]
#[ignore = "requires resolvable real observer-geometry kernels and MPC obscodes (installed Python, cache, or network)"]
fn public_estimate_colors_pins_clean_unknown_observatory_error() {
    let unknown_code = estimate_colors(
        &fit_source_catalog("ZZZ"),
        &one_orbit(OriginId::from_code("SUN")),
        ColorPhaseModel::C1C2,
        false,
        None,
        false,
        None,
        real_geometry_resources(),
    )
    .unwrap();
    assert_eq!(unknown_code.rows[0].status, ColorFitStatus::Failed);
    assert_eq!(
        unknown_code.rows[0].error.as_deref(),
        Some("ZZZ is not a valid MPC observatory code and was not found in SPICE kernels.")
    );
}

#[test]
fn public_estimate_colors_rejects_non_heliocentric_before_resources() {
    let non_heliocentric = estimate_colors(
        &fit_source_catalog("SUN"),
        &one_orbit(OriginId::from_code("EARTH")),
        ColorPhaseModel::C1C2,
        false,
        None,
        false,
        None,
        &no_geometry_resources(),
    )
    .unwrap();
    assert_eq!(non_heliocentric.rows[0].status, ColorFitStatus::Failed);
    assert_eq!(
        non_heliocentric.rows[0].error.as_deref(),
        Some("color estimation requires heliocentric (SUN-origin) orbit coordinates")
    );
}

#[test]
fn public_estimate_colors_preserves_time_conversion_context() {
    let mut observations = fit_source_catalog("SUN");
    observations.time = TimeColumn::new(TimeScale::Utc, vec![-3_000_000; 8], vec![0; 8]);
    let resources = no_geometry_resources();

    let error = estimate_colors(
        &observations,
        &one_orbit(OriginId::from_code("SUN")),
        ColorPhaseModel::C1C2,
        false,
        None,
        false,
        None,
        &resources,
    )
    .unwrap_err();

    assert_eq!(
        error.to_string(),
        "cannot convert observation times to TDB: invalid record batch: eraUtctai failed with ERFA status -1"
    );

    let mut ancient_orbit = one_orbit(OriginId::from_code("SUN"));
    ancient_orbit.coordinates.times =
        Some(TimeArray::from_parts(TimeScale::Utc, vec![-3_000_000], vec![0]).unwrap());
    let error = estimate_colors(
        &fit_source_catalog("SUN"),
        &ancient_orbit,
        ColorPhaseModel::C1C2,
        false,
        None,
        false,
        None,
        &resources,
    )
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "cannot convert orbit epochs to TDB: invalid record batch: eraUtctai failed with ERFA status -1"
    );
}

#[test]
fn public_estimate_colors_signature_accepts_python_aligned_options() {
    let observations = empty_source_catalog();
    let orbits = empty_orbits();
    let resources = ColorEstimationResources::bundled(vec![], String::new());
    let mapping = BTreeMap::from([("LSST_g".to_string(), "LSST_g".to_string())]);

    let result = estimate_colors(
        &observations,
        &orbits,
        ColorPhaseModel::Hg12Star,
        false,
        Some(ColorComposition::Mix(0.7, 0.3)),
        true,
        Some(mapping),
        &resources,
    )
    .unwrap();

    // Valid configuration and empty input do not acquire observer resources;
    // no Python or private IPC type participates in the call.
    assert!(result.is_empty());
}
