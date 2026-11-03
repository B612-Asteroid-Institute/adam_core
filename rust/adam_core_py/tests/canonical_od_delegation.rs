fn function_section<'a>(source: &'a str, name: &str, next: &str) -> &'a str {
    let start = source
        .find(name)
        .unwrap_or_else(|| panic!("missing binding function {name}"));
    let end = source[start..]
        .find(next)
        .map(|offset| start + offset)
        .unwrap_or(source.len());
    &source[start..end]
}

#[test]
fn canonical_bindings_only_decode_prepare_delegate_and_marshal() {
    let source = include_str!("../src/od_drivers.rs");
    let cases = [
        ("fn run_iod_batch_ipc", "fn run_od_batch_ipc", "prepare_iod"),
        (
            "fn run_od_batch_ipc",
            "fn evaluation_output_dict",
            "prepare_od",
        ),
        (
            "fn refine_batch_ipc",
            "fn evaluate_orbits_batch_ipc",
            "prepare_refinement",
        ),
        (
            "fn evaluate_orbits_batch_ipc",
            "/// Native OD timing lanes",
            "prepare_evaluation",
        ),
    ];
    for (name, next, preparation) in cases {
        let section = function_section(source, name, next);
        assert!(section.contains(&format!("adam_core::od::{preparation}")));
        assert!(section.contains("run_with_canonical_provider"));
        assert!(section.contains("prepared.execute(provider)"));
        for forbidden in [
            "PredictionService",
            "OdExecutionContext",
            "fit_settings_from_config",
            "group_input_fingerprint",
            "ProviderMetadata {",
        ] {
            assert!(
                !section.contains(forbidden),
                "{name} regained scientific orchestration through {forbidden}"
            );
        }
    }
}
