fn canonical_workflow_signatures_do_not_drift() {
    use adam_core::od::*;
    type Batch = Result<OdBatchResult, OdBatchError>;
    let _: fn(&AssociatedObservations, &IodRunConfig, &OdPropagator<'_>) -> Batch =
        adam_core::run_iod;
    let _: fn(&AssociatedObservations, &OdConfig, &OdPropagator<'_>) -> Batch = adam_core::run_od;
    let _: fn(&MemberedOrbits, &RefinementRunConfig, &OdPropagator<'_>) -> Batch =
        adam_core::refine;
    let _: fn(&MemberedOrbits, &EvaluationConfig, &OdPropagator<'_>) -> Batch =
        adam_core::evaluate_orbits;
    let _: fn(&AssociatedObservations, &IodRunConfig, &OdPropagator<'_>) -> Batch =
        adam_core::od::run_iod;
    let _: fn(&AssociatedObservations, &OdConfig, &OdPropagator<'_>) -> Batch =
        adam_core::od::run_od;
    let _: fn(&MemberedOrbits, &RefinementRunConfig, &OdPropagator<'_>) -> Batch =
        adam_core::od::refine;
    let _: fn(&MemberedOrbits, &EvaluationConfig, &OdPropagator<'_>) -> Batch =
        adam_core::od::evaluate_orbits;
    let _ = OdPropagator::two_body();
    let _: fn(
        &adam_core::coords::propagation::ExternalPropagator,
    ) -> Result<ProviderMetadata, OdBatchError> = ProviderMetadata::from_external_propagator;
}

fn advanced_algorithm_exports_are_deliberate() {
    use adam_core::od::algorithms::{
        prepare_observations, refine_prepared_orbit, solve_prepared_iod, PreparedObservations,
    };

    let _ = prepare_observations;
    let _ = solve_prepared_iod::<adam_core::spice::AdamCoreSpiceBackend>;
    let _ = refine_prepared_orbit::<adam_core::spice::AdamCoreSpiceBackend>;
    let _: Option<PreparedObservations> = None;
}

fn batch_products_are_immutable_and_losslessly_consumable(result: &adam_core::od::OdBatchResult) {
    let _: &[adam_core::od::FittedOrbitRecord] = result.fitted_orbits();
    let _: &[adam_core::od::FittedOrbitMemberRecord] = result.fitted_orbit_members();
    let _: &[adam_core::od::OrbitDeterminationFailure] = result.failures();
    let owned: (
        Vec<adam_core::od::FittedOrbitRecord>,
        Vec<adam_core::od::FittedOrbitMemberRecord>,
        Vec<adam_core::od::OrbitDeterminationFailure>,
    ) = result.clone().into_products();
    let _ = owned;
}

#[test]
fn canonical_workflows_have_deliberate_root_and_module_exports() {
    canonical_workflow_signatures_do_not_drift();
    advanced_algorithm_exports_are_deliberate();
    let _: fn(&adam_core::od::OdBatchResult) =
        batch_products_are_immutable_and_losslessly_consumable;

    assert_eq!(adam_core::od::OdRoute::RunIod.as_str(), "run_iod");
    assert_eq!(adam_core::od::OdRoute::RunOd.as_str(), "run_od");
    assert_eq!(adam_core::od::OdRoute::Refine.as_str(), "refine");
    assert_eq!(
        adam_core::od::OdRoute::EvaluateOrbits.as_str(),
        "evaluate_orbits"
    );
    assert!(std::mem::size_of::<adam_core::od::OdConfig>() > 0);
    assert!(std::mem::size_of::<adam_core::od::IodRunConfig>() > 0);
    assert!(std::mem::size_of::<adam_core::od::IodStageConfig>() > 0);
    assert!(std::mem::size_of::<adam_core::od::RefinementRunConfig>() > 0);
    assert!(std::mem::size_of::<adam_core::od::EvaluationConfig>() > 0);
    assert!(std::mem::size_of::<adam_core::od::profiles::ProfileIdentity>() > 0);
    assert!(!adam_core::build_identity().version.is_empty());
}

#[test]
fn optional_observation_policies_are_typed_public_rust_configuration() {
    use adam_core::od::profiles::{V2_FULL_RESOURCE, V2_SIGMA_FILL_RESOURCE};
    use adam_core::od::{
        ObservationPipeline, ObservationTransform, ResourceBindings, ResourceSlot,
    };

    let error = ObservationPipeline::bind_explicit(
        vec![
            ObservationTransform::Vfc2017AstrometricUncertaintyFloor {
                resource: V2_SIGMA_FILL_RESOURCE,
                fill_missing: false,
            },
            ObservationTransform::Vfc2017AstrometricUncertaintyReplace {
                resource: V2_SIGMA_FILL_RESOURCE,
            },
            ObservationTransform::PerformanceWeightedCovarianceScaling {
                resource: V2_FULL_RESOURCE,
                minimum_residual_count: 30,
            },
            ObservationTransform::MeasuredBiasAstrometricUncertaintyFloor {
                resource: V2_FULL_RESOURCE,
                minimum_residual_count: 30,
            },
        ],
        &ResourceBindings::new(),
    )
    .expect_err("resource transforms require exact bindings");
    let slots = error
        .resource_problems()
        .iter()
        .map(|problem| problem.slot)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        slots,
        std::collections::BTreeSet::from([
            ResourceSlot::StationCatalogUncertaintyTable,
            ResourceSlot::StationBandResidualCovarianceTable,
        ])
    );
}

#[test]
fn canonical_workflows_keep_meaningful_rustdoc_and_examples() {
    let core = include_str!("../src/od.rs");
    for (signature, headline, route_terms) in [
        (
            "pub fn run_iod(",
            "/// Run canonical batch initial orbit determination (IOD).",
            ["IodRunConfig", "first appearance", "no-solution"],
        ),
        (
            "pub fn run_od(",
            "/// Run canonical full OD: one preparation, IOD, then refinement.",
            [
                "OdConfig",
                "run_od == run_iod + refine",
                "applies the configured observation pipeline",
            ],
        ),
        (
            "pub fn refine(",
            "/// Refine explicit orbit/member groups using their persisted prepared handoff.",
            [
                "RefinementRunConfig",
                "original/used astrometry",
                "mutable relational data",
            ],
        ),
        (
            "pub fn evaluate_orbits(",
            "/// Evaluate explicit orbit/member groups without changing their states.",
            ["EvaluationConfig", "validated explicit", "does not refit"],
        ),
    ] {
        let end = core.find(signature).expect("canonical workflow signature");
        let start = core[..end]
            .rfind(headline)
            .expect("canonical workflow Rustdoc headline");
        let docs = &core[start..end];
        let normalized = docs
            .lines()
            .map(|line| line.trim_start_matches("///").trim())
            .collect::<Vec<_>>()
            .join(" ");
        for required in ["# Example", "Empty input", "provider", "provenance"] {
            assert!(
                normalized.contains(required),
                "{signature} lacks {required}"
            );
        }
        for required in route_terms {
            assert!(
                normalized.contains(required),
                "{signature} lacks {required}"
            );
        }
    }
}

#[test]
fn advanced_algorithms_are_focused_and_have_no_full_workflow_export() {
    let core = include_str!("../src/od.rs");
    let coords = include_str!("../../adam_core_rs_coords/src/lib.rs");
    let algorithms = include_str!("../../adam_core_rs_coords/src/od_algorithms.rs");
    let core_algorithms = core
        .split("pub mod algorithms {")
        .nth(1)
        .expect("public algorithms module")
        .split("mod stage_support {")
        .next()
        .expect("advanced algorithms boundary");

    for required in [
        "prepare_observations",
        "solve_prepared_iod",
        "refine_prepared_orbit",
        "PreparedObservations",
    ] {
        assert!(
            core_algorithms.contains(required),
            "missing advanced API export: {required}"
        );
    }
    for required in [
        "custom native Rust predictors",
        "do not perform canonical grouping",
    ] {
        assert!(
            core.contains(required),
            "missing advanced API contract: {required}"
        );
    }
    assert_eq!(core_algorithms.matches("pub use ").count(), 1);
    assert_eq!(algorithms.matches("\npub fn ").count(), 3);
    assert_eq!(algorithms.matches("\npub type ").count(), 1);
    for internal in [
        "canonical_stage_settings",
        "ErfaTimeProvider",
        "GlobalSpiceTranslation",
        "two_body_provider_metadata",
        "from_external_propagator",
    ] {
        assert!(
            !core_algorithms.contains(internal),
            "internal stage support remains in public algorithms: {internal}"
        );
    }
    for removed in [
        "run_iod_with_predictor",
        "refine_with_predictor",
        "run_od_with_predictors",
        "ConfiguredRunOdOutput",
        "external_provider_metadata",
    ] {
        assert!(
            !core.contains(removed),
            "obsolete Core OD export remains: {removed}"
        );
        assert!(
            !coords.contains(removed),
            "obsolete coords OD export remains: {removed}"
        );
        assert!(
            !algorithms.contains(removed),
            "obsolete algorithm item remains: {removed}"
        );
    }
}

#[test]
fn external_provider_metadata_is_type_owned_and_metadata_only() {
    let core = include_str!("../src/od.rs");
    let signature = "pub fn from_external_propagator(";
    let end = core.find(signature).expect("public metadata constructor");
    let start = core[..end]
        .rfind("/// Read and validate ABI-v2 identity")
        .expect("metadata constructor Rustdoc");
    let docs = &core[start..end];
    for required in [
        "without running OD",
        "does not",
        "call propagation",
        "preflight",
    ] {
        assert!(
            docs.contains(required),
            "external metadata contract lacks {required}"
        );
    }
    assert_eq!(
        core.matches("pub fn from_external_propagator(").count(),
        1,
        "external metadata validation must have one public implementation"
    );
}

#[test]
fn ordinary_public_od_items_keep_rustdoc() {
    let core = include_str!("../src/od.rs");
    let before_python_view = core
        .split("/// Binding-only borrowed view")
        .next()
        .expect("ordinary prefix");
    let provider_marker = "/// Provider identities used by a canonical batch invocation.";
    let after_python_view = &core[core.find(provider_marker).expect("ordinary result suffix")..];
    let after_python_view = after_python_view
        .split("#[doc(hidden)]\npub mod python_bindings")
        .next()
        .expect("ordinary workflow suffix");
    let ordinary = format!("{before_python_view}\n{after_python_view}");
    let mut documented = false;
    let mut public_enum_depth: Option<usize> = None;
    let mut depth = 0usize;

    for (index, line) in ordinary.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("///") {
            documented = true;
            continue;
        }
        if trimmed.starts_with("pub use ") {
            documented = false;
            continue;
        }
        if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with("#") {
            continue;
        }

        let is_public_declaration = [
            "pub struct ",
            "pub enum ",
            "pub fn ",
            "pub const ",
            "pub type ",
            "pub mod ",
        ]
        .iter()
        .any(|prefix| trimmed.starts_with(prefix));
        let is_public_field = trimmed.starts_with("pub ")
            && trimmed.contains(':')
            && trimmed.ends_with(',')
            && !trimmed.starts_with("pub(crate)");
        if is_public_declaration || is_public_field {
            assert!(
                documented,
                "public OD item lacks Rustdoc at line {}: {trimmed}",
                index + 1
            );
        }

        if trimmed.starts_with("pub enum ") {
            public_enum_depth = Some(depth + trimmed.matches('{').count());
        } else if public_enum_depth.is_some_and(|enum_depth| depth == enum_depth)
            && trimmed.ends_with(',')
            && !trimmed.starts_with("pub ")
        {
            assert!(
                documented,
                "public OD enum variant lacks Rustdoc at line {}: {trimmed}",
                index + 1
            );
        }

        depth += trimmed.matches('{').count();
        depth = depth.saturating_sub(trimmed.matches('}').count());
        if public_enum_depth.is_some_and(|enum_depth| depth < enum_depth) {
            public_enum_depth = None;
        }
        documented = false;
    }
}

#[test]
fn removed_public_preparation_seams_are_not_reintroduced() {
    let core = include_str!("../src/od.rs");
    for forbidden in [
        "pub enum BatchPreparation",
        "pub struct PreparedIod",
        "pub struct PreparedOd",
        "pub struct PreparedRefinement",
        "pub struct PreparedEvaluation",
        "pub fn prepare_iod",
        "pub fn prepare_od",
        "pub fn prepare_refinement",
        "pub fn prepare_evaluation",
        "pub fn execute",
        "pub fn prepare_associated_batch",
        "pub fn run_iod_from_prepared",
        "pub fn run_od_from_prepared",
        "pub fn prepare_refinement_batch",
        "pub fn prepare_evaluation_batch",
        "pub fn refine_from_prepared",
        "pub fn evaluate_orbits_from_prepared",
        "pub struct OrbitIdRename",
        "pub fn renamed_orbit_ids",
        "CoreBuildIdentity",
        "InvalidBuildIdentity",
        "Core identities differ",
        "pub trait OdPropagatorSource",
        "pub fitted_orbits: Vec",
        "pub fitted_orbit_members: Vec",
        "pub failures: Vec",
        "pub struct OdGroupResult",
        "pub fn groups(",
        "pub fn group_orbit(",
        "pub fn group_members(",
        "pub fn group_failure(",
    ] {
        assert!(!core.contains(forbidden), "reintroduced seam: {forbidden}");
    }
}
