#[allow(dead_code)]
fn canonical_workflows_are_callable() {
    let _ = adam_core::run_iod;
    let _ = adam_core::run_od;
    let _ = adam_core::refine;
    let _ = adam_core::evaluate_orbits;

    let _ = adam_core::od::run_iod;
    let _ = adam_core::od::run_od;
    let _ = adam_core::od::refine;
    let _ = adam_core::od::evaluate_orbits;
    let _ = adam_core::od::OdPropagator::two_body();
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
    assert!(std::mem::size_of::<adam_core::od::RefinementRunConfig>() > 0);
    assert!(std::mem::size_of::<adam_core::od::EvaluationConfig>() > 0);
}

#[test]
fn removed_public_preparation_seams_are_not_reintroduced() {
    let core = include_str!("../src/od.rs");
    for forbidden in [
        "pub fn prepare_associated_batch",
        "pub fn run_iod_from_prepared",
        "pub fn run_od_from_prepared",
        "pub fn prepare_refinement_batch",
        "pub fn prepare_evaluation_batch",
        "pub fn refine_from_prepared",
        "pub fn evaluate_orbits_from_prepared",
        "#[doc(hidden)]",
    ] {
        assert!(!core.contains(forbidden), "reintroduced seam: {forbidden}");
    }
}
