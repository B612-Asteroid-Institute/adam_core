use adam_core::coords::translation::OriginTranslationProvider;

#[allow(dead_code)]
fn canonical_workflows_are_callable<T: OriginTranslationProvider>() {
    let _ = adam_core::run_iod::<T>;
    let _ = adam_core::run_od::<T>;
    let _ = adam_core::refine::<T>;
    let _ = adam_core::evaluate_orbits::<T>;

    let _ = adam_core::od::run_iod::<T>;
    let _ = adam_core::od::run_od::<T>;
    let _ = adam_core::od::refine::<T>;
    let _ = adam_core::od::evaluate_orbits::<T>;
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
fn pyo3_batch_bindings_delegate_without_a_second_orchestrator() {
    let binding = include_str!("../../adam_core_py/src/od_drivers.rs");
    for delegated in [
        "adam_core::od::run_iod_from_prepared",
        "adam_core::od::run_od_from_prepared",
        "adam_core::od::refine_from_prepared",
        "adam_core::od::evaluate_orbits_from_prepared",
    ] {
        assert!(
            binding.contains(delegated),
            "missing delegation: {delegated}"
        );
    }
    for forbidden in [
        "fn group_input_fingerprint(",
        "fn prepare_association_groups(",
        "fn prepare_member_groups(",
        "struct BatchIodGroup",
        "struct BatchOdGroup",
        "struct BatchRefineGroup",
        "struct BatchEvaluationGroup",
    ] {
        assert!(
            !binding.contains(forbidden),
            "PyO3 regained OD orchestration authority: {forbidden}"
        );
    }
}
