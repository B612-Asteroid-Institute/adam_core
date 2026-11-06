//! Complete standalone Rust scenarios for the four canonical Core 0.7 OD workflows.
//!
//! The example constructs one- and multi-hypothesis batches, preserves overlap
//! and source order, observes an isolated no-solution group between successful
//! groups, composes IOD with refinement, compares that composition with
//! `run_od`, evaluates a full group and a validated member subset, and relabels
//! an orbit together with its member foreign keys.

use adam_core::coords::propagation::{
    generate_ephemeris_barycentric, CovariancePropagation, EphemerisOptions, EpochPolicy,
    IodConfig, PropagationOptions, TwoBodyPropagator,
};
use adam_core::coords::translation::OriginTranslationProvider;
use adam_core::coords::{
    AssociationBatch, CoordinateBatch, CoordinateRepresentation, CoordinateValues, CovarianceBatch,
    CovarianceUnits, DataFrame, Epoch, ObjectId, ObservatoryCode, ObserverBatch, OrbitBatch,
    OrbitId, OriginArray, OriginId, SchemaError, SourceCatalogBatch, TimeArray, TimeColumn,
    TimeScale, TimeScaleProvider,
};
use adam_core::od::{
    AssociatedObservations, AstrometryColumns, EvaluationConfig, FittedOrbitMembersBatch,
    IodStageConfig, MemberedOrbits, ObservationPipeline, OdConfig, OdConfigError,
    OdGroupFailureCode, OdObservationBatch, OdPropagator, RefinementRunConfig, ResourceBindings,
    ResourceSlot, WhitenedFitConfig,
};
use adam_core::spice::{global_backend, SpkWriter, Type9Segment};
use std::error::Error;
use std::path::{Path, PathBuf};

const ROWS: usize = 12;

struct FurnishedKernel(PathBuf);

impl FurnishedKernel {
    fn zero_sun() -> Self {
        let path = std::env::temp_dir().join(format!(
            "adam-core-od-documentation-zero-sun-{}.bsp",
            std::process::id()
        ));
        let start_et = (59_000.0 - 51_544.5) * 86_400.0;
        let end_et = (61_000.0 - 51_544.5) * 86_400.0;
        let mut writer = SpkWriter::new_spk("adam-core-od-documentation");
        writer
            .add_type9(Type9Segment {
                target: 10,
                center: 0,
                frame_id: 1,
                start_et,
                end_et,
                segment_id: "ZERO SUN".to_string(),
                degree: 1,
                states: vec![0.0; 12],
                epochs: vec![start_et, end_et],
            })
            .expect("valid zero-Sun segment");
        writer.write(&path).expect("write temporary SPK");
        global_backend()
            .lock()
            .expect("SPICE backend lock")
            .furnsh(&path)
            .expect("furnish temporary SPK");
        Self(path)
    }
}

impl Drop for FurnishedKernel {
    fn drop(&mut self) {
        if let Ok(mut backend) = global_backend().lock() {
            backend.unload(&self.0);
        }
        let _ = std::fs::remove_file(&self.0);
    }
}

struct IdentityTimeProvider;

impl TimeScaleProvider for IdentityTimeProvider {
    fn rescale(&self, times: &TimeArray, new_scale: TimeScale) -> Result<TimeArray, SchemaError> {
        TimeArray::new(new_scale, times.epochs.clone())
    }
}

struct ZeroTranslationProvider;

impl OriginTranslationProvider for ZeroTranslationProvider {
    fn origin_translation_vectors(
        &self,
        origins: &OriginArray,
        _target_origin: &OriginId,
        _frame: DataFrame,
        _times: &TimeArray,
    ) -> Result<Vec<[f64; 6]>, SchemaError> {
        Ok(vec![[0.0; 6]; origins.len()])
    }
}

fn one_row_orbit() -> OrbitBatch {
    let mut orbit = OrbitBatch::new(
        vec![OrbitId("truth".to_string())],
        vec![Some(ObjectId("truth-group".to_string()))],
        CoordinateBatch::cartesian(
            vec![[1.2, 0.1, 0.05, -0.002, 0.016, 0.001]],
            DataFrame::Ecliptic,
            OriginArray::repeat(OriginId::from_code("SUN"), 1),
            Some(TimeArray::new(TimeScale::Tdb, vec![Epoch::new(60_000, 0)]).expect("truth epoch")),
            None,
        )
        .expect("truth coordinates"),
    )
    .expect("truth orbit");
    // Keep this assignment visible: applications may also populate typed
    // physical and non-gravitational parameter batches on `OrbitBatch`.
    orbit.coordinates.values =
        CoordinateValues::Cartesian(vec![[1.2, 0.1, 0.05, -0.002, 0.016, 0.001]]);
    orbit
}

fn scientific_observations() -> OdObservationBatch {
    const NANOS_PER_DAY: f64 = 86_400_000_000_000.0;
    let offsets = (0..ROWS)
        .map(|row| 0.3 + 3.0 * (row / 3) as f64 + 0.2 * (row % 3) as f64)
        .collect::<Vec<_>>();
    let times = TimeArray::new(
        TimeScale::Tdb,
        offsets
            .iter()
            .map(|offset| Epoch::new(60_002, (offset * NANOS_PER_DAY) as i64))
            .collect(),
    )
    .expect("observation times");
    let observers = ObserverBatch::new(
        (0..ROWS)
            .map(|row| ObservatoryCode(format!("T{row:02}")))
            .collect(),
        CoordinateBatch::cartesian(
            offsets
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
                .collect(),
            DataFrame::Ecliptic,
            OriginArray::repeat(OriginId::from_code("SUN"), ROWS),
            Some(times.clone()),
            None,
        )
        .expect("observer coordinates"),
    )
    .expect("observers");
    let predicted = generate_ephemeris_barycentric(
        &TwoBodyPropagator::default(),
        &one_row_orbit(),
        &observers,
        &EphemerisOptions {
            propagation: PropagationOptions {
                chunk_size: None,
                thread_limit: None,
                epoch_policy: EpochPolicy::CrossProduct,
                covariance: CovariancePropagation::None,
            },
            output_time_scale: TimeScale::Tdb,
            ..EphemerisOptions::default()
        },
        &IdentityTimeProvider,
        &ZeroTranslationProvider,
    )
    .expect("synthetic ephemeris")
    .ephemeris
    .coordinates;
    let mut values = predicted.values.raw_values().to_vec();
    for value in &mut values {
        value[0] = f64::NAN;
        value[3] = f64::NAN;
        value[4] = f64::NAN;
        value[5] = f64::NAN;
    }
    let sigma = 0.5 / 3600.0;
    let mut covariance = vec![f64::NAN; ROWS * 36];
    for row in 0..ROWS {
        covariance[row * 36 + 7] = sigma * sigma;
        covariance[row * 36 + 14] = sigma * sigma;
    }
    let observed = CoordinateBatch::new(
        CoordinateValues::Spherical(values.clone()),
        predicted.frame,
        predicted.origins,
        predicted.times.clone(),
        Some(
            CovarianceBatch::new(
                ROWS,
                6,
                covariance,
                CovarianceUnits::Coordinate(CoordinateRepresentation::Spherical),
            )
            .expect("observation covariance"),
        ),
    )
    .expect("observed coordinates");
    let epochs = predicted.times.expect("predicted times").epochs;
    let time = TimeColumn {
        scale: TimeScale::Tdb,
        days: epochs.iter().map(|epoch| epoch.days).collect(),
        nanos: epochs.iter().map(|epoch| epoch.nanos).collect(),
        validity: None,
    };
    OdObservationBatch {
        source: SourceCatalogBatch {
            id: (0..ROWS).map(|row| format!("science-{row}")).collect(),
            exposure_id: vec![None; ROWS],
            time: time.clone(),
            ra: values.iter().map(|value| value[1]).collect(),
            dec: values.iter().map(|value| value[2]).collect(),
            ra_sigma: vec![Some(sigma); ROWS],
            dec_sigma: vec![Some(sigma); ROWS],
            radec_corr: vec![Some(0.0); ROWS],
            mag: vec![None; ROWS],
            mag_sigma: vec![None; ROWS],
            fwhm: vec![None; ROWS],
            a: vec![None; ROWS],
            a_sigma: vec![None; ROWS],
            b: vec![None; ROWS],
            b_sigma: vec![None; ROWS],
            pa: vec![None; ROWS],
            pa_sigma: vec![None; ROWS],
            observatory_code: observers.code.iter().map(|code| code.0.clone()).collect(),
            filter: vec![None; ROWS],
            exposure_start_time: time,
            exposure_duration: vec![None; ROWS],
            exposure_seeing: vec![None; ROWS],
            exposure_depth_5sigma: vec![None; ROWS],
            object_id: vec![None; ROWS],
            catalog_id: vec!["example".to_string(); ROWS],
            astcat: vec![None; ROWS],
            photcat: vec![None; ROWS],
        },
        observed,
        observers,
    }
}

fn explicit_config() -> Result<OdConfig, Box<dyn Error>> {
    let iod = IodStageConfig::new(
        IodConfig {
            max_triplets: Some(3),
            rchi2_threshold: 50.0,
            ..IodConfig::default()
        },
        WhitenedFitConfig::default(),
    )
    .map_err(std::io::Error::other)?;
    Ok(OdConfig::explicit(
        ObservationPipeline::as_supplied(),
        iod,
        RefinementRunConfig::core_v1(),
    )?)
}

fn single_group(observations: OdObservationBatch) -> AssociatedObservations {
    AssociatedObservations {
        associations: AssociationBatch {
            detection_id: observations.source.id.clone(),
            object_id: vec![Some("one-hypothesis".to_string()); ROWS],
        },
        observations,
    }
}

fn overlapping_groups(observations: OdObservationBatch) -> AssociatedObservations {
    let mut detection_id = observations
        .source
        .id
        .iter()
        .rev()
        .cloned()
        .collect::<Vec<_>>();
    let mut object_id = vec![Some("first-success".to_string()); ROWS];
    // This insufficient two-row group is isolated between two complete groups.
    detection_id.extend(observations.source.id[0..2].iter().cloned());
    object_id.extend(vec![Some("too-short".to_string()); 2]);
    detection_id.extend(observations.source.id.iter().cloned());
    object_id.extend(vec![Some("second-success".to_string()); ROWS]);
    AssociatedObservations {
        observations,
        associations: AssociationBatch {
            detection_id,
            object_id,
        },
    }
}

fn select<T: Clone>(values: &[T], indices: &[usize]) -> Vec<T> {
    indices.iter().map(|&index| values[index].clone()).collect()
}

fn astrometry_subset(values: &AstrometryColumns, indices: &[usize]) -> AstrometryColumns {
    AstrometryColumns {
        lon: select(&values.lon, indices),
        lat: select(&values.lat, indices),
        sigma_lon: select(&values.sigma_lon, indices),
        sigma_lat: select(&values.sigma_lat, indices),
        variance_lon: select(&values.variance_lon, indices),
        variance_lat: select(&values.variance_lat, indices),
        cov_lonlat: select(&values.cov_lonlat, indices),
    }
}

fn member_subset(input: &MemberedOrbits, indices: &[usize]) -> MemberedOrbits {
    let members = &input.members;
    MemberedOrbits {
        orbits: input.orbits.clone(),
        observations: input.observations.clone(),
        provenance: input.provenance.clone(),
        members: FittedOrbitMembersBatch {
            orbit_id: select(&members.orbit_id, indices),
            observation_id: select(&members.observation_id, indices),
            outlier: select(&members.outlier, indices),
            input_fingerprint: select(&members.input_fingerprint, indices),
            membership_fingerprint: select(&members.membership_fingerprint, indices),
            original_astrometry: astrometry_subset(&members.original_astrometry, indices),
            used_astrometry: astrometry_subset(&members.used_astrometry, indices),
        },
    }
}

/// Public profiles are path-free; standalone Rust binds every semantic slot to
/// an explicit local path. No package discovery, download, cache, or fallback occurs.
#[allow(dead_code)]
fn public_profile_config(bias_dat: &Path) -> Result<OdConfig, OdConfigError> {
    let resources = ResourceBindings::new().bind(ResourceSlot::Efcc18CatalogBiasTable, bias_dat);
    adam_core::od::profiles::CORE_OD_V1.bind(&resources)
}

fn main() -> Result<(), Box<dyn Error>> {
    let _kernel = FurnishedKernel::zero_sun();
    let observations = scientific_observations();
    let config = explicit_config()?;
    let provider = OdPropagator::two_body();

    // One explicit hypothesis.
    let one = single_group(observations.clone());
    let one_iod = adam_core::run_iod(&one, &config.for_iod(), &provider)?;
    assert_eq!(one_iod.fitted_orbits().len(), 1);
    assert!(one_iod.failures().is_empty());

    // Three first-appearance groups with overlap and one isolated scientific
    // failure. Reversed association rows are restored to source-table order.
    let multiple = overlapping_groups(observations.clone());
    let multi_iod = adam_core::od::run_iod(&multiple, &config.for_iod(), &provider)?;
    assert_eq!(
        multi_iod
            .fitted_orbits()
            .iter()
            .map(|record| record.group_id.as_str())
            .collect::<Vec<_>>(),
        ["first-success", "second-success"]
    );
    assert_eq!(multi_iod.failures().len(), 1);
    assert_eq!(multi_iod.failures()[0].group_id, "too-short");
    assert!(matches!(
        &multi_iod.failures()[0].code,
        OdGroupFailureCode::NoSolution | OdGroupFailureCode::Scientific(_)
    ));
    for members in multi_iod.fitted_orbit_members().chunks(ROWS) {
        assert_eq!(
            members
                .iter()
                .map(|member| member.source_index)
                .collect::<Vec<_>>(),
            (0..ROWS).collect::<Vec<_>>()
        );
    }

    // Persist the complete IOD handoff and refine without reapplying the
    // observation pipeline. The fused and explicit compositions agree.
    let handoff = one_iod
        .to_membered_orbits(&observations)?
        .expect("successful IOD handoff");
    let composed = adam_core::refine(&handoff, &config.for_refinement(), &provider)?;
    let direct = adam_core::run_od(&one, &config, &provider)?;
    assert_eq!(direct.fitted_orbits().len(), composed.fitted_orbits().len());
    for (fused, staged) in direct.fitted_orbits().iter().zip(composed.fitted_orbits()) {
        assert_eq!(fused.orbit, staged.orbit);
        assert_eq!(
            fused.provenance.configuration.canonical_json(),
            staged.provenance.configuration.canonical_json()
        );
    }

    // Evaluate the complete group and a validated six-member subset.
    let evaluation = EvaluationConfig::core_v1();
    let full_evaluation = adam_core::evaluate_orbits(&handoff, &evaluation, &provider)?;
    assert!(!full_evaluation.is_empty());
    let subset = member_subset(&handoff, &[0, 1, 2, 3, 4, 5]);
    let subset_evaluation = adam_core::od::evaluate_orbits(&subset, &evaluation, &provider)?;
    assert!(!subset_evaluation.is_empty());

    // `orbit_id` is relational: edit the orbit label and every member foreign
    // key together. Immutable object/group identities and fingerprints remain.
    let mut relabeled = handoff.clone();
    let old_label = relabeled.orbits.orbit_id[0].0.clone();
    let new_label = "analyst-label".to_string();
    relabeled.orbits.orbit_id[0].0.clone_from(&new_label);
    for label in &mut relabeled.members.orbit_id {
        if *label == old_label {
            label.clone_from(&new_label);
        }
    }
    let relabeled_evaluation = adam_core::evaluate_orbits(&relabeled, &evaluation, &provider)?;
    assert_eq!(
        relabeled_evaluation.fitted_orbits()[0].orbit.orbit_id[0].0,
        new_label
    );

    // Borrow immutable products or consume all three vectors together.
    let _: &[_] = direct.fitted_orbits();
    let _: &[_] = direct.fitted_orbit_members();
    let _: &[_] = direct.failures();
    let (_orbits, _members, _failures) = direct.into_products();
    Ok(())
}
