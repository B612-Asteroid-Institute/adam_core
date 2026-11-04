//! Standalone Rust use of all four canonical Core 0.7 OD workflows.
//!
//! The executable uses empty typed batches so it needs no Python, IPC, SPICE
//! fixture, or resource file. Replace `empty_observations()` and
//! `empty_membered_orbits()` with application data for scientific execution.

use adam_core::coords::propagation::ExternalPropagator;
use adam_core::coords::{
    AssociationBatch, CoordinateBatch, DataFrame, ObserverBatch, OrbitBatch, OriginArray,
    SourceCatalogBatch, TimeColumn, TimeScale,
};
use adam_core::od::{
    AssociatedObservations, AstrometryColumns, EvaluationConfig, FittedOrbitMembersBatch,
    IodRunConfig, IodStageConfig, MemberedOrbits, ObservationPipeline, OdConfig, OdConfigError,
    OdObservationBatch, OdPropagator, RefinementRunConfig, ResourceBindings, ResourceSlot,
};
use std::path::Path;

fn empty_source() -> SourceCatalogBatch {
    let time = TimeColumn {
        scale: TimeScale::Tdb,
        days: Vec::new(),
        nanos: Vec::new(),
        validity: None,
    };
    SourceCatalogBatch {
        id: Vec::new(),
        exposure_id: Vec::new(),
        time: time.clone(),
        ra: Vec::new(),
        dec: Vec::new(),
        ra_sigma: Vec::new(),
        dec_sigma: Vec::new(),
        radec_corr: Vec::new(),
        mag: Vec::new(),
        mag_sigma: Vec::new(),
        fwhm: Vec::new(),
        a: Vec::new(),
        a_sigma: Vec::new(),
        b: Vec::new(),
        b_sigma: Vec::new(),
        pa: Vec::new(),
        pa_sigma: Vec::new(),
        observatory_code: Vec::new(),
        filter: Vec::new(),
        exposure_start_time: time,
        exposure_duration: Vec::new(),
        exposure_seeing: Vec::new(),
        exposure_depth_5sigma: Vec::new(),
        object_id: Vec::new(),
        catalog_id: Vec::new(),
        astcat: Vec::new(),
        photcat: Vec::new(),
    }
}

fn empty_observations() -> Result<OdObservationBatch, Box<dyn std::error::Error>> {
    let observed = CoordinateBatch::spherical(
        Vec::new(),
        DataFrame::Equatorial,
        OriginArray::new(Vec::new()),
        None,
        None,
    )?;
    let observer_coordinates = CoordinateBatch::cartesian(
        Vec::new(),
        DataFrame::Ecliptic,
        OriginArray::new(Vec::new()),
        None,
        None,
    )?;
    Ok(OdObservationBatch {
        source: empty_source(),
        observed,
        observers: ObserverBatch::new(Vec::new(), observer_coordinates)?,
    })
}

fn empty_membered_orbits() -> Result<MemberedOrbits, Box<dyn std::error::Error>> {
    let coordinates = CoordinateBatch::cartesian(
        Vec::new(),
        DataFrame::Ecliptic,
        OriginArray::new(Vec::new()),
        None,
        None,
    )?;
    Ok(MemberedOrbits {
        orbits: OrbitBatch::new(Vec::new(), Vec::new(), coordinates)?,
        observations: empty_observations()?,
        members: FittedOrbitMembersBatch {
            orbit_id: Vec::new(),
            observation_id: Vec::new(),
            outlier: Vec::new(),
            input_fingerprint: Vec::new(),
            membership_fingerprint: Vec::new(),
            original_astrometry: AstrometryColumns::default(),
            used_astrometry: AstrometryColumns::default(),
        },
        provenance: Vec::new(),
    })
}

/// Public profiles are path-free; standalone Rust binds each semantic slot to
/// an explicit local path. No discovery, download, or fallback occurs.
#[allow(dead_code)]
fn public_profile_config(bias_dat: &Path) -> Result<OdConfig, OdConfigError> {
    let resources = ResourceBindings::new().bind(ResourceSlot::Efcc18CatalogBiasTable, bias_dat);
    adam_core::od::profiles::CORE_OD_V1.bind(&resources)
}

/// An application that directly links a propagation implementation wraps the
/// validated ABI-v2 backend explicitly. The four workflow calls do not change.
#[allow(dead_code)]
fn external_provider(external: &ExternalPropagator) -> OdPropagator<'_> {
    OdPropagator::external(external)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pipeline = ObservationPipeline::as_supplied();
    let iod_stage = IodStageConfig::core_v1();
    let refinement_config = RefinementRunConfig::core_v1();
    let iod_config = IodRunConfig::explicit(pipeline.clone(), iod_stage.clone())?;
    let od_config = OdConfig::explicit(pipeline, iod_stage, refinement_config)?;
    let evaluation_config = EvaluationConfig::core_v1();
    let provider = OdPropagator::two_body();

    let associated = AssociatedObservations {
        observations: empty_observations()?,
        associations: AssociationBatch {
            detection_id: Vec::new(),
            object_id: Vec::new(),
        },
    };
    let membered = empty_membered_orbits()?;

    // The root and module paths deliberately expose the same implementations.
    let iod = adam_core::run_iod(&associated, &iod_config, &provider)?;
    let full = adam_core::od::run_od(&associated, &od_config, &provider)?;
    let refined = adam_core::refine(&membered, &refinement_config, &provider)?;
    let evaluated = adam_core::od::evaluate_orbits(&membered, &evaluation_config, &provider)?;

    // Borrow immutable products when the result remains live.
    let _: &[_] = iod.fitted_orbits();
    let _: &[_] = iod.fitted_orbit_members();
    let _: &[_] = iod.failures();

    // Or consume one result and move out all three products together.
    let (orbits, members, failures) = full.into_products();
    assert!(orbits.is_empty() && members.is_empty() && failures.is_empty());
    assert!(refined.fitted_orbits().is_empty());
    assert!(evaluated.fitted_orbits().is_empty());
    Ok(())
}
