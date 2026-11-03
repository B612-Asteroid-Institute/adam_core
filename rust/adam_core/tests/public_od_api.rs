#[allow(dead_code)]
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
}

#[allow(dead_code)]
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
