# Core 0.7 OD test migration inventory

Baseline: `d21c45649be37a218a7acccba78cc7c888501c48`
Inventory scope: Python test functions under `src/adam_core/orbit_determination/tests` that existed at the baseline and are absent by name on the correction line.
Missing-by-name count: **273** across **23** files.

This is an accountability inventory, not a claim that names are the contract. Every missing function is listed below. `RUST-PORTED` means the scientific invariant moved to a named native test area; `RETIRED` is limited to intentionally removed executable Python model/fitter/attestation APIs; `DEFERRED` identifies batch-native assertions owned by Bead `personal-eack9`. The correction line retains the focused numerical suite only as internal qualification and does not restore a Python authoritative workflow.

Current native evidence anchors:

- `rust/adam_core_rs_coords/src/observation_uncertainty.rs`
- `rust/adam_core_rs_coords/src/veres2017.rs`
- `rust/adam_core_rs_coords/src/efcc18.rs`
- `rust/adam_core_rs_coords/src/od_config.rs`
- `rust/adam_core_rs_coords/src/propagation/od.rs`
- `rust/adam_core_rs_coords/src/propagation/od_refine.rs`
- `src/adam_core/orbit_determination/tests/test_iod*.py`
- `src/adam_core/orbit_determination/tests/test_cmc2003_rejection.py`
- `src/adam_core/orbit_determination/tests/test_evaluate.py`
- `src/adam_core/orbit_determination/tests/test_least_squares.py`

## `test_cmc2003_rejection.py` — REPLACED

Executable fitter/default dispatch retired; rejection rules and native loop are covered by `test_cmc2003_rejection.py` plus Rust `od_refine` tests.

Baseline-only function names (4):
- `test_cmc2003_dispatch_forwards_loss_and_kwargs`
- `test_cmc2003_end_to_end_through_refine_fit`
- `test_default_is_cmc2003`
- `test_unknown_scheme_raises`

## `test_efcc18_debias_model.py` — RUST-PORTED

Executable model class retired; correction, wrapping, epoch, pass-through, and covariance invariants are covered in Rust `efcc18` and `observation_uncertainty` tests, with resource parsing in `od_config`.

Baseline-only function names (12):
- `test_composes_with_covariance_models`
- `test_constructor_validation`
- `test_covariance_unchanged_including_nan_pattern`
- `test_empty_observations`
- `test_epoch_matters`
- `test_exclude_astcats`
- `test_injected_correction_round_trip`
- `test_mixed_rows_only_covered_move`
- `test_pole_and_nonfinite_rows_pass_through`
- `test_ra_wraps_into_range`
- `test_uncovered_and_null_catalogs_pass_through`
- `test_zero_correction_returns_input`

## `test_evaluate.py` — PARTIAL/DEFERRED

Numerical statistics and frozen reference parity remain in the current file. Legacy empty/error public-workflow behavior is superseded by the canonical batch contract in `personal-eack9`.

Baseline-only function names (2):
- `test_evaluate_orbits_empty_legacy_error`
- `test_evaluate_orbits_frozen_legacy_parity`

## `test_huber_loss.py` — REPLACED

Mutable fitter class/pickling surface retired; Huber behavior and explicit fit settings are covered by Rust solver/config tests.

Baseline-only function names (2):
- `test_default_fitter_is_linear`
- `test_native_fitter_forwards_loss_and_pickles`

## `test_iod.py` — PARTIAL/DEFERRED

Gauss/Gibbs/numerical/capsule behavior remains in current Python and Rust tests. Fused linkage/batch entry points are superseded by `personal-eack9`.

Baseline-only function names (5):
- `test_initial_orbit_determination_fused_batch_deduplicates`
- `test_initial_orbit_determination_fused_empty`
- `test_iod`
- `test_iod_assist_capsule_matches_explicit_prefit_stage`
- `test_iod_fused_matches_pinned_legacy_fixture`

## `test_least_squares.py` — PORTED

Adjustment and zero-component regressions are retained through the test-only native refinement bridge; the removed public least-squares alias is not restored.

Baseline-only function names (3):
- `test_least_squares_adjusts`
- `test_least_squares_is_refine_without_rejection`
- `test_least_squares_zero_base`

## `test_legacy_deprecations.py` — RETIRED

Removed APIs have no compatibility aliases or deprecation shims by design.

Baseline-only function names (3):
- `test_differential_correction_warns_before_doing_anything`
- `test_least_squares_warns_on_construction`
- `test_od_warns_before_doing_anything`

## `test_native_dispatch.py` — REPLACED

Python fitter/model dispatch and algorithm work-unit seams were removed. Capsule ABI/capability/identity validation is covered by native propagator ABI tests and Rust provider prevalidation.

Baseline-only function names (17):
- `test_compatible_fitters_keep_fused_paths`
- `test_custom_fitter_hooks_are_called`
- `test_custom_fitter_runs_before_native_settings_validation`
- `test_dispatch_requires_a_declared_backend`
- `test_distinct_iod_and_refinement_capsules_are_obtained_and_forwarded`
- `test_external_propagator_capsule_runs_od_natively`
- `test_further_custom_model_override_is_refused`
- `test_instance_model_override_is_refused`
- `test_live_custom_model_behavior_is_refused`
- `test_native_backend_rejects_algorithm_work_units`
- `test_native_backend_rejects_empty_identity`
- `test_only_exact_shipped_models_keep_fused_run_od`
- `test_overlapping_entry_points_are_deprecated_aliases`
- `test_overridden_dynamics_fail_without_a_native_capsule`
- `test_provider_identity_uses_native_semver_with_or_without_version_module`
- `test_python_propagators_are_refused`
- `test_run_od_model_science_ignores_all_python_model_execution_surfaces`

## `test_native_orbit_fitter.py` — RETIRED

`NativeOrbitFitter` and the Python fitter plugin hierarchy are deliberately removed.

Baseline-only function names (6):
- `test_native_orbit_fitter_full_od`
- `test_native_orbit_fitter_is_orbit_fitter`
- `test_native_orbit_fitter_pickle_roundtrip`
- `test_native_orbit_fitter_propagator_kwargs_are_not_shared`
- `test_native_orbit_fitter_refine_fit`
- `test_native_orbit_fitter_takes_a_propagator_instance`

## `test_observation_uncertainty.py` — RUST-PORTED

Executable Python models are retired. Empirical covariance, station/band selection, floor/fill, nightly deweighting, ordering, validation, and position invariants are covered by Rust `observation_uncertainty`, `veres2017`, and `od_config` tests.

Baseline-only function names (40):
- `test_absent_station_passes_through`
- `test_add_mode_at_equator`
- `test_add_mode_high_declination`
- `test_all_models_preserve_positions`
- `test_applies_models_in_sequence`
- `test_assert_positions_unchanged_detects_length_change`
- `test_assert_positions_unchanged_detects_shift`
- `test_band_specific_row_preferred`
- `test_batch_above_cap_is_deweighted`
- `test_batch_at_or_below_cap_passes_through`
- `test_batches_are_per_station_and_night`
- `test_bias_below_baseline_is_noop`
- `test_chi2_below_one_clamps_to_unchanged`
- `test_compatible_types_cast`
- `test_cross_term_scales_with_the_block`
- `test_default_table_resolves_from_the_package`
- `test_duplicate_station_rows_raise`
- `test_empty_composite_raises`
- `test_empty_observations`
- `test_empty_observations_pass_through`
- `test_extra_columns_dropped`
- `test_floors_each_axis_independently`
- `test_high_declination_ra_floor`
- `test_identity_composition_is_identity`
- `test_invalid_cap_raises`
- `test_invalid_mode_raises`
- `test_low_resid_cov_n_passes_through`
- `test_missing_column_raises`
- `test_missing_package_is_named`
- `test_nan_base_cross_term_treated_as_zero`
- `test_negative_variance_raises`
- `test_non_model_component_raises`
- `test_noop`
- `test_null_obs_code_raises`
- `test_null_resid_cov_n_passes_through`
- `test_optional_table_validated`
- `test_replace_mode`
- `test_scales_sigmas_by_sqrt_chi2`
- `test_uncastable_column_raises`
- `test_valid_table_passes`

## `test_observatory_bias_model_wiring.py` — RETIRED

The `observatory_bias_model=` flag is removed without an alias; transforms now exist only in explicit Rust-owned `ObservationPipeline` declarations.

Baseline-only function names (11):
- `test_applies_model_and_does_not_forward_it`
- `test_default_none_leaves_observations_unchanged`
- `test_differential_correction`
- `test_entry_point_accepts_observatory_bias_model`
- `test_evaluate_orbits`
- `test_fit_least_squares`
- `test_fitter_interface_does_not_take_the_flag`
- `test_initial_orbit_determination`
- `test_iod`
- `test_od`
- `test_upstream_application_reaches_both_stages`

## `test_od.py` — PARTIAL/DEFERRED

Focused Python orchestration aliases are retired. Numerical refinement/rejection remains covered natively; canonical batch OD is `personal-eack9`.

Baseline-only function names (3):
- `test_od`
- `test_od_is_refine_with_worst_residual_rejection`
- `test_od_method_validation_matches_legacy`

## `test_od_observations.py` — RUST-PORTED

Catalog metadata preservation is tested by pipeline/native table tests; executable uncertainty-model routing is retired.

Baseline-only function names (1):
- `test_astcat_survives_uncertainty_models`

## `test_od_ops_parity.py` — DEFERRED

Legacy Python sort/select/linkage orchestration is superseded by Rust-owned deterministic batch grouping in `personal-eack9`.

Baseline-only function names (6):
- `test_select_observations_matches_legacy`
- `test_select_observations_rejects_unknown_method`
- `test_select_observations_too_few_returns_empty`
- `test_sort_by_id_and_time_handles_chunked_observations`
- `test_sort_by_id_and_time_matches_legacy_ordering`
- `test_sort_by_id_and_time_missing_observation_errors`

## `test_od_orchestration.py` — RETIRED/DEFERRED

Python model orchestration and Python canonical provenance are deliberately removed. Original/used astrometry and pipeline semantics are Rust-owned; canonical batch result/provenance assertions move to `personal-eack9`.

Baseline-only function names (34):
- `test_direct_fitter_path_leaves_provenance_null`
- `test_duplicate_observation_ids_raise`
- `test_efcc18_composite_records_shifted_positions_and_inflated_sigma`
- `test_empirical_covariance_inflates_used_sigma_and_keeps_positions`
- `test_empirical_covariance_records_used_sigma_and_is_consumed_by_the_fit`
- `test_empty_observations_return_empty_tables`
- `test_from_spherical_empty`
- `test_from_spherical_missing_covariance_terms_are_nan`
- `test_from_spherical_snapshots_position_and_uncertainty`
- `test_identity_model_matches_no_model_fit`
- `test_identity_model_records_used_equal_to_original`
- `test_legacy_members_parquet_loads_with_null_provenance`
- `test_live_python_model_is_refused_before_core_stage_composition`
- `test_member_with_unknown_obs_id_raises`
- `test_members_with_and_without_provenance_concatenate`
- `test_members_without_provenance_have_null_columns`
- `test_model_dropping_observations_raises`
- `test_model_reordering_observations_raises`
- `test_models_and_cmc2003_refinement`
- `test_native_run_od_validates_full_settings_before_empty_exit`
- `test_night_batch_deweighting_records_scaled_sigma`
- `test_no_models_and_worst_residual_refinement`
- `test_no_models_records_originals_and_keeps_fitter_flags`
- `test_none_and_empty_sequence_return_input`
- `test_object_id_is_passed_to_backend_and_stamped_when_null`
- `test_object_id_none_is_null_on_both_routes`
- `test_provenance_joins_by_obs_id_when_fitter_reorders_members`
- `test_run_od_explicit_default_models_missing_bias_dat_is_file_not_found`
- `test_run_od_explicit_default_models_name_the_missing_data_package`
- `test_run_od_explicit_default_models_report_every_missing_input_at_once`
- `test_run_od_is_exported_from_the_package`
- `test_run_od_rejects_unknown_models_string`
- `test_run_od_with_the_default_stack`
- `test_sequence_is_applied_in_order_like_a_composite`

## `test_od_python_baseline_parity.py` — RUST-PORTED

Python-model baseline fixtures are replaced by direct Rust kernel, transform, resource, CMC, whitening, and configuration tests; no executable Python-model parity surface remains.

Baseline-only function names (8):
- `test_astcat_round_trip_and_covariance`
- `test_cmc2003_decision_kernels`
- `test_debias_model_positions`
- `test_every_model_matches_and_preserves_positions`
- `test_linear_huber_ignore_and_cmc2003`
- `test_run_od_composite_provenance`
- `test_tiles_and_corrections_on_published_table`
- `test_whitening_jacobian_and_robust_helpers`

## `test_orbit_fitter.py` — RETIRED

The abstract Python fitter plugin contract is removed.

Baseline-only function names (2):
- `test_initial_fit_only_fitter_stays_instantiable`
- `test_orbit_fitter_still_requires_initial_fit`

## `test_provenance.py` — RETIRED/DEFERRED

The rejected Python attestation/model-descriptor design is removed. Rust config/resource fingerprints are tested now; complete batch provider/input/build/result provenance belongs to `personal-eack9`.

Baseline-only function names (67):
- `test_ambiguous_iod_survives_refinement`
- `test_backend_identity_names_declared_and_python_backends`
- `test_bias_module_delegate_mutation_is_refused_before_execution`
- `test_bias_row_columns_monkeypatch_is_custom_on_real_route`
- `test_builtin_behavior_overrides_and_nested_overrides_require_contract`
- `test_builtin_descriptor_uses_exact_build_and_not_import_history`
- `test_canonical_configuration_is_injective_and_user_mappings_cannot_forge_tags`
- `test_common_configuration_values_are_portable_and_distinct`
- `test_complete_provenance_keeps_distinct_native_stage_providers`
- `test_composed_guard_rejects_model_state_changed_after_snapshot`
- `test_composed_numpy_delegate_replacement_is_ignored_without_call`
- `test_composite_native_model_specs_delegate_mutation_is_refused`
- `test_contracted_moduleless_model_is_rejected`
- `test_custom_contract_is_validated_exactly`
- `test_custom_declaration_canonicalization_is_sensitive`
- `test_custom_descriptor_is_stable_across_hash_seed_and_import_order`
- `test_custom_fingerprint_is_the_implementation_identity`
- `test_custom_non_string_module_metadata_is_never_formatted`
- `test_custom_subclass_and_instance_apply_override_require_contract`
- `test_default_constructor_mutation_is_refused_before_constructor`
- `test_default_loader_selector_mutation_is_refused_before_loading`
- `test_distribution_identity_probes_fail_closed_before_all_work`
- `test_documented_custom_model_migration_policy_is_executable`
- `test_exact_class_special_lookup_hooks_are_refused_before_execution`
- `test_extra_hostile_instance_field_is_refused_before_value_lookup`
- `test_genuine_installed_class_and_apply_monkeypatches_are_refused`
- `test_hostile_model_sequence_is_rejected_without_iteration`
- `test_hostile_registered_field_values_are_refused_without_hooks`
- `test_inherited_abc_lookup_mutation_is_refused_without_execution`
- `test_input_fingerprint_hashes_caller_and_used_astrometry_independently`
- `test_local_contracted_class_is_rejected_before_all_work`
- `test_main_dynamic_nonexistent_contract_is_rejected_before_all_work`
- `test_model_descriptor_canonicalizes_colliding_sets_and_mapping_order`
- `test_model_descriptor_hashes_numpy_configuration_and_sets`
- `test_model_descriptor_hashes_table_content`
- `test_model_descriptor_rejects_unsupported_and_cyclic_configuration`
- `test_mutable_abc_metaclass_dispatch_is_attested_without_execution`
- `test_native_kernel_delegate_mutation_is_refused_before_execution`
- `test_native_model_spec_snapshot_has_no_mutable_model_references`
- `test_native_settings_are_called_once_and_same_snapshot_is_recorded_and_used`
- `test_nested_custom_lookup_is_refused_without_any_hook`
- `test_nested_lookup_special_resolution_hook_is_refused_before_execution`
- `test_other_entry_points_leave_provenance_null`
- `test_preflight_describes_contract_before_apply_or_fit`
- `test_preflight_failure_happens_before_fitter_work`
- `test_preflight_rejects_native_python_version_mismatch_before_model_work`
- `test_provenance_keeps_exact_native_identity_not_later_declarations`
- `test_provenance_preflight_refuses_non_reproducible_builds`
- `test_provider_cannot_replace_a_separate_orchestration_native_alias`
- `test_provider_mutation_is_bound_by_final_model_and_native_spec_snapshot`
- `test_provider_native_spec_builder_replacement_is_ignored_without_call`
- `test_provider_time_native_replacement_is_refused_before_fake_native_work`
- `test_public_pathlib_class_identity_is_normalized_across_private_relocation`
- `test_public_run_od_recursively_preflights_nested_custom_models`
- `test_python_callable_configuration_requires_explicit_portable_data`
- `test_real_git_source_checkout_detector_is_fail_closed`
- `test_registered_instance_data_cannot_be_shadowed_on_the_class`
- `test_registered_instance_shadows_and_nested_subclass_are_custom`
- `test_replaced_native_build_function_is_refused_without_call`
- `test_replaced_native_package_module_is_refused_without_call`
- `test_run_od_records_provenance`
- `test_runtime_untracked_source_guard_precedes_models`
- `test_sigma_fill_model_constant_monkeypatch_is_custom_on_real_route`
- `test_uniform_provider_rejects_disagreement_and_partial_omission`
- `test_unsupported_dynamic_model_has_documented_type_error`
- `test_veres_lookup_payload_monkeypatch_is_custom_on_real_route`
- `test_veres_registered_class_mutations_are_custom`

## `test_reconstruction_contracts.py` — REPLACED/DEFERRED

Source/build identity remains covered by release identity gates; model-descriptor assertions are retired; provider/input/composed-route provenance moves to `personal-eack9`.

Baseline-only function names (9):
- `test_build_identity_is_exact_clean_head`
- `test_efcc_descriptor_content_version_exclusions`
- `test_explicit_stage_composition_route_completes`
- `test_fused_and_composed_share_native_model_science_and_provenance`
- `test_input_fingerprint_is_layout_canonical`
- `test_input_fingerprint_is_sensitive`
- `test_model_descriptor_numpy_variants`
- `test_provider_metadata_comes_from_capsule_not_python_declaration`
- `test_two_distinct_epoch_arcs_are_refused_on_all_routes`

## `test_run_iod.py` — PARTIAL/DEFERRED

Focused numeric IOD settings/capsule coverage remains. Linkage and public stage-composition APIs are superseded by canonical batch workflows in `personal-eack9`.

Baseline-only function names (11):
- `test_fitter_forwards_the_velocity_method`
- `test_fitter_initial_fit_runs_iod_on_the_configured_class`
- `test_initial_orbit_determination_accepts_an_unmatched_linkage`
- `test_initial_orbit_determination_runs_run_iod_per_linkage`
- `test_iod_is_the_accepted_orbit_of_run_iod`
- `test_linkage_api_validates_settings_before_empty_exit`
- `test_native_iod_rejects_invalid_settings_centrally`
- `test_refinement_signature_defaults_are_the_rust_defaults`
- `test_run_iod_capsule_matches_in_tree_two_body_driver`
- `test_run_iod_then_refine_is_the_stage_contract`
- `test_signature_defaults_are_the_rust_defaults`

## `test_run_od_null_covariance.py` — RUST-PORTED/DEFERRED

Missing covariance fails closed in the native solver and fill behavior is covered by Rust `veres2017`; canonical configured workflow parity moves to `personal-eack9`.

Baseline-only function names (3):
- `test_fused_and_composed_native_model_routes_agree_with_filled_sigmas`
- `test_sigma_fill_supplies_the_missing_covariance`
- `test_without_a_filling_model_the_fit_is_refused`

## `test_solver_budgets.py` — RUST-PORTED

Both limits are fields of strict Rust-owned fit configuration and covered by round-trip/validation tests.

Baseline-only function names (1):
- `test_fitter_serializes_both_solver_limits`

## `test_veres2017.py` — RUST-PORTED

Executable Veres model/loaders are retired; lookup precedence, fill/floor/replace, validation, and cross-term behavior are covered by Rust `veres2017` and real-resource `od_config` tests.

Baseline-only function names (23):
- `test_composes_after_efcc18_style_position_models`
- `test_composes_with_other_models`
- `test_cross_term_is_preserved`
- `test_default_table_without_the_package_raises_import_error`
- `test_duplicate_rows_raise`
- `test_fills_only_missing_axes_and_keeps_reported`
- `test_floors_small_sigmas_and_keeps_large_ones`
- `test_global_row_beats_fallback_and_is_unique`
- `test_known_working_table_entries`
- `test_lookup_order`
- `test_lookup_order_station_catalog_station_catalog_global`
- `test_missing_column_raises`
- `test_missing_sigma_left_unless_fill_missing`
- `test_model_defaults_resolve_the_package_tables`
- `test_no_fallback_returns_none`
- `test_non_positive_sigma_raises`
- `test_nothing_missing_returns_same_object`
- `test_null_astcat_is_a_station_or_global_row`
- `test_package_working_table_still_validates`
- `test_replaces_both_axes_and_zeroes_cross_term`
- `test_unknown_catalog_uses_fallback_or_passes_through`
- `test_unresolvable_without_global_or_fallback_stays_nan`
- `test_working_table_schema_and_sanity`
