//! Standalone use of the high-level public color-estimation API.
//!
//! This example is compile-checked but not run by doctests because resolving
//! default kernels may download the pinned kernel wheels on first use.

use std::error::Error;

use adam_core::coords::{
    CoordinateBatch, DataFrame, ObjectId, OrbitBatch, OrbitId, OriginArray, OriginId,
    SourceCatalogBatch, TimeArray, TimeColumn, TimeScale,
};
use adam_core::kernel_data::Resolver;
use adam_core::{
    estimate_colors, ColorFitStatus, ColorPhaseModel, ResolvedColorEstimationResources,
};

fn synthetic_observations() -> SourceCatalogBatch {
    let n = 8;
    SourceCatalogBatch {
        id: (0..n).map(|row| format!("source-{row}")).collect(),
        exposure_id: vec![None; n],
        time: TimeColumn::new(TimeScale::Tdb, vec![60_000; n], vec![0; n]),
        ra: vec![0.0; n],
        dec: vec![0.0; n],
        ra_sigma: vec![None; n],
        dec_sigma: vec![None; n],
        radec_corr: vec![None; n],
        mag: [20.0, 19.5]
            .into_iter()
            .flat_map(|magnitude| std::iter::repeat_n(Some(magnitude), 4))
            .collect(),
        mag_sigma: vec![Some(0.1); n],
        fwhm: vec![None; n],
        a: vec![None; n],
        a_sigma: vec![None; n],
        b: vec![None; n],
        b_sigma: vec![None; n],
        pa: vec![None; n],
        pa_sigma: vec![None; n],
        observatory_code: vec!["SUN".to_string(); n],
        filter: ["LSST_g", "LSST_r"]
            .into_iter()
            .flat_map(|filter| std::iter::repeat_n(Some(filter.to_string()), 4))
            .collect(),
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

fn synthetic_orbit() -> Result<OrbitBatch, Box<dyn Error>> {
    let coordinates = CoordinateBatch::cartesian(
        vec![[1.0, 0.0, 0.0, 0.0, 0.017_202_098_95, 0.0]],
        DataFrame::Ecliptic,
        OriginArray::repeat(OriginId::from_code("SUN"), 1),
        Some(TimeArray::from_parts(
            TimeScale::Tdb,
            vec![60_000],
            vec![0],
        )?),
        None,
    )?;
    Ok(OrbitBatch::new(
        vec![OrbitId("orbit-1".to_string())],
        vec![Some(ObjectId("object-1".to_string()))],
        coordinates,
    )?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let resolver = Resolver::from_env();
    let resolved = ResolvedColorEstimationResources::resolve(&resolver)?;
    let resources = resolved.as_resources();

    let fits = estimate_colors(
        &synthetic_observations(),
        &synthetic_orbit()?,
        ColorPhaseModel::C1C2,
        false,
        None,
        false,
        None,
        &resources,
    )?;

    assert_eq!(fits.rows[0].status, ColorFitStatus::Ok);
    println!(
        "{}: {} = {:.3}",
        fits.rows[0].object_id,
        fits.rows[0].color_lhs_filter_ids.as_ref().unwrap()[0],
        fits.rows[0].colors.as_ref().unwrap()[0]
    );
    Ok(())
}
