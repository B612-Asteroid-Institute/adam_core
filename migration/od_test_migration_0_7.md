# Core 0.7 OD assertion migration crosswalk

Baseline: `d21c45649be37a218a7acccba78cc7c888501c48`
Scope: Python test functions under `src/adam_core/orbit_determination/tests` that exist at the baseline and remain absent by name on this correction line.
Remaining absent-by-name count: **267** across **23** files.

Every row is an assertion-level disposition. `RETIRED-API` marks a specifically removed executable interface; `RETIRED-SCHEMA` marks an intentionally unsupported legacy serialized schema. `SUCCESSOR-personal-7olgj` assigns resource-bound behavior to that separately tracked dependency. `REPLACED-eack9` is used only when the evidence column names an exact maintained replacement test; otherwise the row remains deferred or retired. Resource-gated deferral names the missing binding explicitly. No row claims a broad file was “Rust-ported.”

## Restored numerical fixture assertions (no longer absent by name)

- `test_iod.py::test_iod_fused_matches_pinned_legacy_fixture` — **RESTORED**: `test_iod.py::test_iod_fused_matches_pinned_legacy_fixture` now replays the 2026-07-12 ASSIST fixture through `run_prefit_iod_qualified`.
- `test_od_python_baseline_parity.py::test_tiles_and_corrections_on_published_table` — **RESTORED**: the same test now replays published-table EFCC18 tiles and corrections.
- `test_od_python_baseline_parity.py::test_debias_model_positions` — **RESTORED**: the same test now executes real-table positions through the bound declarative Rust pipeline.
- `test_od_python_baseline_parity.py::test_whitening_jacobian_and_robust_helpers` — **RESTORED**: the same test now replays all retained lower-level kernel values.
- `test_od_python_baseline_parity.py::test_cmc2003_decision_kernels` — **RESTORED**: the same test now replays the native decision-kernel products.
- `test_od_python_baseline_parity.py::test_linear_huber_ignore_and_cmc2003` — **RESTORED**: the same test now pins fitted state/covariance/statistics/weights/outliers/iterations.

## Remaining baseline-only Python assertions

### `test_cmc2003_rejection.py`

| Baseline function | Disposition | Exact replacement or technical reason |
|---|---|---|
| `test_cmc2003_dispatch_forwards_loss_and_kwargs` | **RETIRED-API** | Python fitter/default-scheme dispatch was removed; Rust `od_config::tests::stage_json_round_trips_strictly_and_rejects_unknown_or_missing_fields` validates typed refinement policy. |
| `test_cmc2003_end_to_end_through_refine_fit` | **REPLACED** | `test_cmc2003_rejection.py::TestCmc2003FitDetailed::test_gross_outliers_are_rejected` and Rust `cmc2003_rejects_gross_outliers_and_recovers_truth`. |
| `test_default_is_cmc2003` | **RETIRED-API** | Python fitter/default-scheme dispatch was removed; Rust `od_config::tests::stage_json_round_trips_strictly_and_rejects_unknown_or_missing_fields` validates typed refinement policy. |
| `test_unknown_scheme_raises` | **RETIRED-API** | Python fitter/default-scheme dispatch was removed; Rust `od_config::tests::stage_json_round_trips_strictly_and_rejects_unknown_or_missing_fields` validates typed refinement policy. |

### `test_efcc18_debias_model.py`

| Baseline function | Disposition | Exact replacement or technical reason |
|---|---|---|
| `test_composes_with_covariance_models` | **REPLACED/RETIRED** | Python `CompositeModel` is retired; `od_config::tests::pipeline_executes_non_commuting_transforms_in_declaration_order` pins native execution order. |
| `test_constructor_validation` | **REPLACED/RETIRED** | `od_config::tests::invalid_transform_is_configuration_error_not_resource_error`. |
| `test_covariance_unchanged_including_nan_pattern` | **REPLACED/RETIRED** | `observation_uncertainty::tests::efcc18_model_moves_positions_and_keeps_covariance` and the restored real-table pipeline replay. |
| `test_empty_observations` | **REPLACED-eack9** | `test_empty_batches_are_rust_owned_and_never_bind_provider` is the exact maintained replacement. |
| `test_epoch_matters` | **REPLACED/RETIRED** | `efcc18::tests::corrections_read_the_right_cell_and_apply_proper_motion`. |
| `test_exclude_astcats` | **REPLACED/RETIRED** | `test_od_python_baseline_parity.py::TestEfcc18::test_debias_model_positions` (excluded and unexcluded bound configs). |
| `test_injected_correction_round_trip` | **REPLACED/RETIRED** | `efcc18::tests::corrections_read_the_right_cell_and_apply_proper_motion`. |
| `test_mixed_rows_only_covered_move` | **REPLACED/RETIRED** | `efcc18::tests::debias_positions_wraps_and_skips_poles`. |
| `test_pole_and_nonfinite_rows_pass_through` | **REPLACED/RETIRED** | `efcc18::tests::debias_positions_wraps_and_skips_poles`. |
| `test_ra_wraps_into_range` | **REPLACED/RETIRED** | `efcc18::tests::ra_wrap_is_harmless` and `::debias_positions_wraps_and_skips_poles`. |
| `test_uncovered_and_null_catalogs_pass_through` | **REPLACED/RETIRED** | `efcc18::tests::debias_positions_wraps_and_skips_poles`. |
| `test_zero_correction_returns_input` | **REPLACED/RETIRED** | The removed Python object-identity return contract is retired; Rust correction values/pass-through are pinned by `debias_positions_wraps_and_skips_poles`. |

### `test_evaluate.py`

| Baseline function | Disposition | Exact replacement or technical reason |
|---|---|---|
| `test_evaluate_orbits_empty_legacy_error` | **REPLACED-eack9** | `test_empty_batches_are_rust_owned_and_never_bind_provider` is the exact maintained replacement. |
| `test_evaluate_orbits_frozen_legacy_parity` | **REPLACED** | `test_evaluate.py::test_evaluate_orbits_frozen_reference_parity` replays the retained fixture. |

### `test_huber_loss.py`

| Baseline function | Disposition | Exact replacement or technical reason |
|---|---|---|
| `test_default_fitter_is_linear` | **RETIRED-API** | Mutable/pickleable Python fitter defaults were removed; Rust `linear_predictor_huber_bounds_influence_reports_exact_weights_and_robust_covariance` and `od_config::tests::stage_json_round_trips_strictly_and_rejects_unknown_or_missing_fields` pin behavior/settings. |
| `test_native_fitter_forwards_loss_and_pickles` | **RETIRED-API** | Mutable/pickleable Python fitter defaults were removed; Rust `linear_predictor_huber_bounds_influence_reports_exact_weights_and_robust_covariance` and `od_config::tests::stage_json_round_trips_strictly_and_rejects_unknown_or_missing_fields` pin behavior/settings. |

### `test_iod.py`

| Baseline function | Disposition | Exact replacement or technical reason |
|---|---|---|
| `test_initial_orbit_determination_fused_batch_deduplicates` | **REPLACED-eack9** | `test_overlap_first_appearance_source_order_and_no_iod_placeholder` and `test_malformed_associations_abort_before_factory` pin deterministic grouping and explicit duplicate-pair rejection. |
| `test_initial_orbit_determination_fused_empty` | **REPLACED-eack9** | `test_empty_batches_are_rust_owned_and_never_bind_provider` is the exact maintained replacement. |
| `test_iod` | **REPLACED** | `test_iod.py::test_focused_iod_returns_candidate_report` covers the focused external-capsule success products. |
| `test_iod_assist_capsule_matches_explicit_prefit_stage` | **REPLACED** | `test_iod.py::test_iod_fused_matches_pinned_legacy_fixture` now exercises the explicit prefit bridge and pins every product. |

### `test_least_squares.py`

| Baseline function | Disposition | Exact replacement or technical reason |
|---|---|---|
| `test_least_squares_adjusts` | **REPLACED** | `test_least_squares.py::test_refinement_adjusts_and_improves_chi2`. |
| `test_least_squares_is_refine_without_rejection` | **RETIRED-API** | The removed `LeastSquares` alias/equivalence surface has no compatibility alias; refinement numerics remain in `test_refinement_adjusts_and_improves_chi2`. |
| `test_least_squares_zero_base` | **REPLACED** | `test_least_squares.py::test_refinement_handles_zero_state_component`. |

### `test_legacy_deprecations.py`

| Baseline function | Disposition | Exact replacement or technical reason |
|---|---|---|
| `test_differential_correction_warns_before_doing_anything` | **RETIRED-API** | `test_differential_correction_warns_before_doing_anything` exercised a removed Python fitter/model/deprecation interface; Core 0.7 intentionally provides no alias or executable callback surface. |
| `test_least_squares_warns_on_construction` | **RETIRED-API** | `test_least_squares_warns_on_construction` exercised a removed Python fitter/model/deprecation interface; Core 0.7 intentionally provides no alias or executable callback surface. |
| `test_od_warns_before_doing_anything` | **RETIRED-API** | `test_od_warns_before_doing_anything` exercised a removed Python fitter/model/deprecation interface; Core 0.7 intentionally provides no alias or executable callback surface. |

### `test_native_dispatch.py`

| Baseline function | Disposition | Exact replacement or technical reason |
|---|---|---|
| `test_compatible_fitters_keep_fused_paths` | **RETIRED-API** | This assertion targeted removed Python fitter/model dispatch, override, alias, or live-behavior hooks; only the closed typed Rust config/capsule path remains. |
| `test_custom_fitter_hooks_are_called` | **RETIRED-API** | This assertion targeted removed Python fitter/model dispatch, override, alias, or live-behavior hooks; only the closed typed Rust config/capsule path remains. |
| `test_custom_fitter_runs_before_native_settings_validation` | **RETIRED-API** | This assertion targeted removed Python fitter/model dispatch, override, alias, or live-behavior hooks; only the closed typed Rust config/capsule path remains. |
| `test_dispatch_requires_a_declared_backend` | **REPLACED** | Rust `propagation::external::tests::exact_abi_version_and_frozen_prefixes_are_enforced`, `::provenance_suffix_is_atomic_strict_and_legacy_propagation_compatible`, and `od_refine::tests::external_propagators_run_the_od_stages_like_native_ones`. |
| `test_distinct_iod_and_refinement_capsules_are_obtained_and_forwarded` | **REPLACED** | Rust `propagation::external::tests::exact_abi_version_and_frozen_prefixes_are_enforced`, `::provenance_suffix_is_atomic_strict_and_legacy_propagation_compatible`, and `od_refine::tests::external_propagators_run_the_od_stages_like_native_ones`. |
| `test_external_propagator_capsule_runs_od_natively` | **REPLACED** | Rust `propagation::external::tests::exact_abi_version_and_frozen_prefixes_are_enforced`, `::provenance_suffix_is_atomic_strict_and_legacy_propagation_compatible`, and `od_refine::tests::external_propagators_run_the_od_stages_like_native_ones`. |
| `test_further_custom_model_override_is_refused` | **RETIRED-API** | This assertion targeted removed Python fitter/model dispatch, override, alias, or live-behavior hooks; only the closed typed Rust config/capsule path remains. |
| `test_instance_model_override_is_refused` | **RETIRED-API** | This assertion targeted removed Python fitter/model dispatch, override, alias, or live-behavior hooks; only the closed typed Rust config/capsule path remains. |
| `test_live_custom_model_behavior_is_refused` | **RETIRED-API** | This assertion targeted removed Python fitter/model dispatch, override, alias, or live-behavior hooks; only the closed typed Rust config/capsule path remains. |
| `test_native_backend_rejects_algorithm_work_units` | **REPLACED** | Rust `propagation::external::tests::exact_abi_version_and_frozen_prefixes_are_enforced`, `::provenance_suffix_is_atomic_strict_and_legacy_propagation_compatible`, and `od_refine::tests::external_propagators_run_the_od_stages_like_native_ones`. |
| `test_native_backend_rejects_empty_identity` | **REPLACED** | Rust `propagation::external::tests::exact_abi_version_and_frozen_prefixes_are_enforced`, `::provenance_suffix_is_atomic_strict_and_legacy_propagation_compatible`, and `od_refine::tests::external_propagators_run_the_od_stages_like_native_ones`. |
| `test_only_exact_shipped_models_keep_fused_run_od` | **RETIRED-API** | This assertion targeted removed Python fitter/model dispatch, override, alias, or live-behavior hooks; only the closed typed Rust config/capsule path remains. |
| `test_overlapping_entry_points_are_deprecated_aliases` | **RETIRED-API** | This assertion targeted removed Python fitter/model dispatch, override, alias, or live-behavior hooks; only the closed typed Rust config/capsule path remains. |
| `test_overridden_dynamics_fail_without_a_native_capsule` | **REPLACED** | Rust `propagation::external::tests::exact_abi_version_and_frozen_prefixes_are_enforced`, `::provenance_suffix_is_atomic_strict_and_legacy_propagation_compatible`, and `od_refine::tests::external_propagators_run_the_od_stages_like_native_ones`. |
| `test_provider_identity_uses_native_semver_with_or_without_version_module` | **REPLACED** | Rust `propagation::external::tests::exact_abi_version_and_frozen_prefixes_are_enforced`, `::provenance_suffix_is_atomic_strict_and_legacy_propagation_compatible`, and `od_refine::tests::external_propagators_run_the_od_stages_like_native_ones`. |
| `test_python_propagators_are_refused` | **REPLACED** | Rust `propagation::external::tests::exact_abi_version_and_frozen_prefixes_are_enforced`, `::provenance_suffix_is_atomic_strict_and_legacy_propagation_compatible`, and `od_refine::tests::external_propagators_run_the_od_stages_like_native_ones`. |
| `test_run_od_model_science_ignores_all_python_model_execution_surfaces` | **RETIRED-API** | This assertion targeted removed Python fitter/model dispatch, override, alias, or live-behavior hooks; only the closed typed Rust config/capsule path remains. |

### `test_native_orbit_fitter.py`

| Baseline function | Disposition | Exact replacement or technical reason |
|---|---|---|
| `test_native_orbit_fitter_full_od` | **RETIRED-API** | `test_native_orbit_fitter_full_od` exercised a removed Python fitter/model/deprecation interface; Core 0.7 intentionally provides no alias or executable callback surface. |
| `test_native_orbit_fitter_is_orbit_fitter` | **RETIRED-API** | `test_native_orbit_fitter_is_orbit_fitter` exercised a removed Python fitter/model/deprecation interface; Core 0.7 intentionally provides no alias or executable callback surface. |
| `test_native_orbit_fitter_pickle_roundtrip` | **RETIRED-API** | `test_native_orbit_fitter_pickle_roundtrip` exercised a removed Python fitter/model/deprecation interface; Core 0.7 intentionally provides no alias or executable callback surface. |
| `test_native_orbit_fitter_propagator_kwargs_are_not_shared` | **RETIRED-API** | `test_native_orbit_fitter_propagator_kwargs_are_not_shared` exercised a removed Python fitter/model/deprecation interface; Core 0.7 intentionally provides no alias or executable callback surface. |
| `test_native_orbit_fitter_refine_fit` | **RETIRED-API** | `test_native_orbit_fitter_refine_fit` exercised a removed Python fitter/model/deprecation interface; Core 0.7 intentionally provides no alias or executable callback surface. |
| `test_native_orbit_fitter_takes_a_propagator_instance` | **RETIRED-API** | `test_native_orbit_fitter_takes_a_propagator_instance` exercised a removed Python fitter/model/deprecation interface; Core 0.7 intentionally provides no alias or executable callback surface. |

### `test_observation_uncertainty.py`

| Baseline function | Disposition | Exact replacement or technical reason |
|---|---|---|
| `test_absent_station_passes_through` | **REPLACED** | `observation_uncertainty::tests::empirical_replace_mode_nan_cross_term_and_pass_through_rules`. |
| `test_add_mode_at_equator` | **REPLACED** | `observation_uncertainty::tests::empirical_add_mode_at_equator_and_high_declination`. |
| `test_add_mode_high_declination` | **REPLACED** | `observation_uncertainty::tests::empirical_add_mode_at_equator_and_high_declination`. |
| `test_all_models_preserve_positions` | **REPLACED** | Exact position invariants are asserted by Rust `efcc18_model_moves_positions_and_keeps_covariance`, `night_batch_deweighting_is_per_station_and_night`, and Veres fill/floor/replace tests. |
| `test_applies_models_in_sequence` | **REPLACED** | `od_config::tests::pipeline_executes_non_commuting_transforms_in_declaration_order` pins execution order; `od_workflows::tests::prepare_observations_executes_repeated_pipeline_and_retains_snapshots` pins repeated execution. |
| `test_assert_positions_unchanged_detects_length_change` | **RETIRED-API** | Python model/composite/identity helper behavior was removed; no callback compatibility surface remains. |
| `test_assert_positions_unchanged_detects_shift` | **RETIRED-API** | Python model/composite/identity helper behavior was removed; no callback compatibility surface remains. |
| `test_band_specific_row_preferred` | **REPLACED** | `observation_uncertainty::tests::band_specific_row_is_preferred`. |
| `test_batch_above_cap_is_deweighted` | **REPLACED** | `observation_uncertainty::tests::night_batch_deweighting_is_per_station_and_night`. |
| `test_batch_at_or_below_cap_passes_through` | **REPLACED** | `observation_uncertainty::tests::night_batch_deweighting_is_per_station_and_night`. |
| `test_batches_are_per_station_and_night` | **REPLACED** | `observation_uncertainty::tests::night_batch_deweighting_is_per_station_and_night`. |
| `test_bias_below_baseline_is_noop` | **RETIRED-API** | Performance-weighted and generic sigma-floor executable models are outside the closed transform set; their old helper semantics are not a Core 0.7 pipeline contract. |
| `test_chi2_below_one_clamps_to_unchanged` | **RETIRED-API** | Performance-weighted and generic sigma-floor executable models are outside the closed transform set; their old helper semantics are not a Core 0.7 pipeline contract. |
| `test_compatible_types_cast` | **RETIRED-API** | Permissive Python bias-table casting was removed; content-pinned resource tables now require the exact Rust-owned Arrow schema. |
| `test_cross_term_scales_with_the_block` | **REPLACED** | `observation_uncertainty::tests::night_batch_deweighting_is_per_station_and_night`. |
| `test_default_table_resolves_from_the_package` | **RETIRED-API** | Runtime data-package discovery/fallback was removed; callers bind semantic resource slots explicitly. |
| `test_duplicate_station_rows_raise` | **REPLACED** | Rust `od_config::tests::{invalid_transform_is_configuration_error_not_resource_error,separately_packaged_reference_resources_parse}` and strict resource schema parsing replace Python table/model validation for the closed transform set. |
| `test_empty_composite_raises` | **RETIRED-API** | Python model/composite/identity helper behavior was removed; no callback compatibility surface remains. |
| `test_empty_observations` | **REPLACED-eack9** | `test_empty_batches_are_rust_owned_and_never_bind_provider` is the exact maintained replacement. |
| `test_empty_observations_pass_through` | **RETIRED-API** | The executable Python model identity/pass-through method is removed; canonical empty-group behavior is implemented by the canonical batch-native workflow layer. |
| `test_extra_columns_dropped` | **RETIRED-API** | Permissive Python table normalization was removed; bound resources now require the exact Rust-owned schema and do not silently drop fields. |
| `test_floors_each_axis_independently` | **RETIRED-API** | Performance-weighted and generic sigma-floor executable models are outside the closed transform set; their old helper semantics are not a Core 0.7 pipeline contract. |
| `test_high_declination_ra_floor` | **RETIRED-API** | The generic Python `SigmaFloorModel` is outside the closed transform set; Veres-specific high-declination floor math remains covered by `veres2017::tests::floor_model_floors_per_axis_with_cos_dec_and_fill_missing`. |
| `test_identity_composition_is_identity` | **RETIRED-API** | Python model/composite/identity helper behavior was removed; no callback compatibility surface remains. |
| `test_invalid_cap_raises` | **REPLACED** | Rust `od_config::tests::{invalid_transform_is_configuration_error_not_resource_error,separately_packaged_reference_resources_parse}` and strict resource schema parsing replace Python table/model validation for the closed transform set. |
| `test_invalid_mode_raises` | **REPLACED** | Rust `od_config::tests::{invalid_transform_is_configuration_error_not_resource_error,separately_packaged_reference_resources_parse}` and strict resource schema parsing replace Python table/model validation for the closed transform set. |
| `test_low_resid_cov_n_passes_through` | **REPLACED** | `observation_uncertainty::tests::empirical_replace_mode_nan_cross_term_and_pass_through_rules`. |
| `test_missing_column_raises` | **REPLACED** | Rust `od_config::tests::{invalid_transform_is_configuration_error_not_resource_error,separately_packaged_reference_resources_parse}` and strict resource schema parsing replace Python table/model validation for the closed transform set. |
| `test_missing_package_is_named` | **RETIRED-API** | Runtime data-package discovery/fallback was removed; callers bind semantic resource slots explicitly. |
| `test_nan_base_cross_term_treated_as_zero` | **REPLACED** | `observation_uncertainty::tests::empirical_replace_mode_nan_cross_term_and_pass_through_rules`. |
| `test_negative_variance_raises` | **REPLACED** | Rust `od_config::tests::{invalid_transform_is_configuration_error_not_resource_error,separately_packaged_reference_resources_parse}` and strict resource schema parsing replace Python table/model validation for the closed transform set. |
| `test_non_model_component_raises` | **RETIRED-API** | Python model/composite/identity helper behavior was removed; no callback compatibility surface remains. |
| `test_noop` | **RETIRED-API** | Python model/composite/identity helper behavior was removed; no callback compatibility surface remains. |
| `test_null_obs_code_raises` | **REPLACED** | Rust `od_config::tests::{invalid_transform_is_configuration_error_not_resource_error,separately_packaged_reference_resources_parse}` and strict resource schema parsing replace Python table/model validation for the closed transform set. |
| `test_null_resid_cov_n_passes_through` | **REPLACED** | `observation_uncertainty::tests::empirical_replace_mode_nan_cross_term_and_pass_through_rules`. |
| `test_optional_table_validated` | **REPLACED** | Rust `od_config::tests::{invalid_transform_is_configuration_error_not_resource_error,separately_packaged_reference_resources_parse}` and strict resource schema parsing replace Python table/model validation for the closed transform set. |
| `test_replace_mode` | **REPLACED** | `observation_uncertainty::tests::empirical_replace_mode_nan_cross_term_and_pass_through_rules`. |
| `test_scales_sigmas_by_sqrt_chi2` | **RETIRED-API** | Performance-weighted and generic sigma-floor executable models are outside the closed transform set; their old helper semantics are not a Core 0.7 pipeline contract. |
| `test_uncastable_column_raises` | **REPLACED** | Rust `od_config::tests::{invalid_transform_is_configuration_error_not_resource_error,separately_packaged_reference_resources_parse}` and strict resource schema parsing replace Python table/model validation for the closed transform set. |
| `test_valid_table_passes` | **REPLACED** | Rust `od_config::tests::{invalid_transform_is_configuration_error_not_resource_error,separately_packaged_reference_resources_parse}` and strict resource schema parsing replace Python table/model validation for the closed transform set. |

### `test_observatory_bias_model_wiring.py`

| Baseline function | Disposition | Exact replacement or technical reason |
|---|---|---|
| `test_applies_model_and_does_not_forward_it` | **RETIRED-API** | `test_applies_model_and_does_not_forward_it` exercised a removed Python fitter/model/deprecation interface; Core 0.7 intentionally provides no alias or executable callback surface. |
| `test_default_none_leaves_observations_unchanged` | **RETIRED-API** | `test_default_none_leaves_observations_unchanged` exercised a removed Python fitter/model/deprecation interface; Core 0.7 intentionally provides no alias or executable callback surface. |
| `test_differential_correction` | **RETIRED-API** | `test_differential_correction` exercised a removed Python fitter/model/deprecation interface; Core 0.7 intentionally provides no alias or executable callback surface. |
| `test_entry_point_accepts_observatory_bias_model` | **RETIRED-API** | `test_entry_point_accepts_observatory_bias_model` exercised a removed Python fitter/model/deprecation interface; Core 0.7 intentionally provides no alias or executable callback surface. |
| `test_evaluate_orbits` | **RETIRED-API** | `test_evaluate_orbits` exercised a removed Python fitter/model/deprecation interface; Core 0.7 intentionally provides no alias or executable callback surface. |
| `test_fit_least_squares` | **RETIRED-API** | `test_fit_least_squares` exercised a removed Python fitter/model/deprecation interface; Core 0.7 intentionally provides no alias or executable callback surface. |
| `test_fitter_interface_does_not_take_the_flag` | **RETIRED-API** | `test_fitter_interface_does_not_take_the_flag` exercised a removed Python fitter/model/deprecation interface; Core 0.7 intentionally provides no alias or executable callback surface. |
| `test_initial_orbit_determination` | **RETIRED-API** | `test_initial_orbit_determination` exercised a removed Python fitter/model/deprecation interface; Core 0.7 intentionally provides no alias or executable callback surface. |
| `test_iod` | **RETIRED-API** | `test_iod` exercised a removed Python fitter/model/deprecation interface; Core 0.7 intentionally provides no alias or executable callback surface. |
| `test_od` | **RETIRED-API** | `test_od` exercised a removed Python fitter/model/deprecation interface; Core 0.7 intentionally provides no alias or executable callback surface. |
| `test_upstream_application_reaches_both_stages` | **RETIRED-API** | `test_upstream_application_reaches_both_stages` exercised a removed Python fitter/model/deprecation interface; Core 0.7 intentionally provides no alias or executable callback surface. |

### `test_od.py`

| Baseline function | Disposition | Exact replacement or technical reason |
|---|---|---|
| `test_od` | **DEFERRED/RETIRED** | The old one-group `od()` orchestration/method-validation API is retired; canonical batch `run_od` composition/failures are covered by the canonical batch-native workflow layer. Numerical composition is pinned by `od_workflows::tests::run_od_prepares_once_and_exactly_composes_isolated_stage_predictors`. |
| `test_od_is_refine_with_worst_residual_rejection` | **DEFERRED/RETIRED** | The old one-group `od()` orchestration/method-validation API is retired; canonical batch `run_od` composition/failures are covered by the canonical batch-native workflow layer. Numerical composition is pinned by `od_workflows::tests::run_od_prepares_once_and_exactly_composes_isolated_stage_predictors`. |
| `test_od_method_validation_matches_legacy` | **DEFERRED/RETIRED** | The old one-group `od()` orchestration/method-validation API is retired; canonical batch `run_od` composition/failures are covered by the canonical batch-native workflow layer. Numerical composition is pinned by `od_workflows::tests::run_od_prepares_once_and_exactly_composes_isolated_stage_predictors`. |

### `test_od_observations.py`

| Baseline function | Disposition | Exact replacement or technical reason |
|---|---|---|
| `test_astcat_survives_uncertainty_models` | **REPLACED** | `test_od_python_baseline_parity.py::TestEfcc18::test_debias_model_positions` runs catalog metadata through the declarative native pipeline; `od_workflows::tests::prepare_observations_executes_repeated_pipeline_and_retains_snapshots` pins retention. |

### `test_od_ops_parity.py`

| Baseline function | Disposition | Exact replacement or technical reason |
|---|---|---|
| `test_select_observations_matches_legacy` | **RETIRED-API** | The public linkage observation-selection helper was removed; canonical membership is explicit through `Associations` / `FittedOrbitMembers`. |
| `test_select_observations_rejects_unknown_method` | **RETIRED-API** | The string-selected linkage observation-selection method API was removed without an alias. |
| `test_select_observations_too_few_returns_empty` | **RETIRED-API** | The linkage selection helper was removed; typed IOD/refinement failures now represent insufficient scientific membership. |
| `test_sort_by_id_and_time_handles_chunked_observations` | **RETIRED-API** | The public linkage sort helper was removed; Rust planners consume explicit IDs and preserve source-row order independent of Arrow chunk layout. |
| `test_sort_by_id_and_time_matches_legacy_ordering` | **RETIRED-API** | The public linkage sort helper was removed; `test_overlap_first_appearance_source_order_and_no_iod_placeholder` pins canonical order. |
| `test_sort_by_id_and_time_missing_observation_errors` | **RETIRED-API** | The public linkage sort helper was removed; `test_malformed_associations_abort_before_factory` pins unknown-observation refusal. |

### `test_od_orchestration.py`

| Baseline function | Disposition | Exact replacement or technical reason |
|---|---|---|
| `test_direct_fitter_path_leaves_provenance_null` | **RETIRED-API** | Lower-level focused fitters are expert algorithms, not canonical provenance-producing workflows; canonical provenance is emitted only by the four batch APIs. |
| `test_duplicate_observation_ids_raise` | **REPLACED-eack9** | `test_malformed_associations_abort_before_factory` is the exact maintained replacement. |
| `test_efcc18_composite_records_shifted_positions_and_inflated_sigma` | **REPLACED/DEFERRED** | Rust `od_workflows::tests::prepare_observations_executes_repeated_pipeline_and_retains_snapshots` pins numerical original/used retention; Python member-table assembly is implemented by the canonical batch-native workflow layer. |
| `test_empirical_covariance_inflates_used_sigma_and_keeps_positions` | **REPLACED/DEFERRED** | Rust `od_workflows::tests::prepare_observations_executes_repeated_pipeline_and_retains_snapshots` pins numerical original/used retention; Python member-table assembly is implemented by the canonical batch-native workflow layer. |
| `test_empirical_covariance_records_used_sigma_and_is_consumed_by_the_fit` | **REPLACED/DEFERRED** | Rust `od_workflows::tests::prepare_observations_executes_repeated_pipeline_and_retains_snapshots` pins numerical original/used retention; Python member-table assembly is implemented by the canonical batch-native workflow layer. |
| `test_empty_observations_return_empty_tables` | **REPLACED-eack9** | `test_empty_batches_are_rust_owned_and_never_bind_provider` is the exact maintained replacement. |
| `test_from_spherical_empty` | **REPLACED/DEFERRED** | Rust `od_workflows::tests::prepare_observations_executes_repeated_pipeline_and_retains_snapshots` pins numerical original/used retention; Python member-table assembly is implemented by the canonical batch-native workflow layer. |
| `test_from_spherical_missing_covariance_terms_are_nan` | **REPLACED/DEFERRED** | Rust `od_workflows::tests::prepare_observations_executes_repeated_pipeline_and_retains_snapshots` pins numerical original/used retention; Python member-table assembly is implemented by the canonical batch-native workflow layer. |
| `test_from_spherical_snapshots_position_and_uncertainty` | **REPLACED/DEFERRED** | Rust `od_workflows::tests::prepare_observations_executes_repeated_pipeline_and_retains_snapshots` pins numerical original/used retention; Python member-table assembly is implemented by the canonical batch-native workflow layer. |
| `test_identity_model_matches_no_model_fit` | **RETIRED-API** | Legacy `models=` strings/sequences and executable Python model behavior were removed without aliases. |
| `test_identity_model_records_used_equal_to_original` | **REPLACED/DEFERRED** | Rust `od_workflows::tests::prepare_observations_executes_repeated_pipeline_and_retains_snapshots` pins numerical original/used retention; Python member-table assembly is implemented by the canonical batch-native workflow layer. |
| `test_legacy_members_parquet_loads_with_null_provenance` | **RETIRED-SCHEMA** | Core 0.7 intentionally breaks the legacy member parquet schema: canonical refine/evaluate require exact variance, input, and membership fingerprints. |
| `test_live_python_model_is_refused_before_core_stage_composition` | **RETIRED-API** | Legacy `models=` strings/sequences and executable Python model behavior were removed without aliases. |
| `test_member_with_unknown_obs_id_raises` | **REPLACED-eack9** | `test_malformed_members_abort_before_factory` is the exact maintained replacement. |
| `test_members_with_and_without_provenance_concatenate` | **RETIRED-SCHEMA** | Mixing legacy and canonical member schemas is retired; canonical members require complete lossless handoff identity. |
| `test_members_without_provenance_have_null_columns` | **RETIRED-SCHEMA** | Legacy null-provenance member rows are not valid canonical refine/evaluate inputs in Core 0.7. |
| `test_model_dropping_observations_raises` | **RETIRED-API** | Row-dropping was possible only through removed executable Python callbacks; every closed Rust transform preserves row count by construction. |
| `test_model_reordering_observations_raises` | **RETIRED-API** | Row-reordering was possible only through removed executable Python callbacks; every closed Rust transform edits the caller-order batch in place. |
| `test_models_and_cmc2003_refinement` | **REPLACED/DEFERRED** | `od_config::tests::pipeline_executes_non_commuting_transforms_in_declaration_order` pins transform order and `od_workflows::tests::run_od_prepares_once_and_exactly_composes_isolated_stage_predictors` pins configured CMC composition; canonical batch results/failures are covered by the canonical batch-native workflow layer. |
| `test_native_run_od_validates_full_settings_before_empty_exit` | **REPLACED/DEFERRED** | `od_config::tests::stage_json_round_trips_strictly_and_rejects_unknown_or_missing_fields` pins strict complete settings; canonical empty-batch preflight order is covered by the canonical batch-native workflow layer. |
| `test_night_batch_deweighting_records_scaled_sigma` | **REPLACED/DEFERRED** | `od_workflows::tests::prepare_observations_executes_repeated_pipeline_and_retains_snapshots` pins numerical nightly scaling, repetition, and used snapshots; member assembly is covered by the canonical batch-native workflow layer. |
| `test_no_models_and_worst_residual_refinement` | **REPLACED/DEFERRED** | Rust `od_refine::tests::iterative_fit_rejects_the_worst_observation_until_converged` pins worst-residual refinement; canonical batch composition is covered by the canonical batch-native workflow layer. |
| `test_no_models_records_originals_and_keeps_fitter_flags` | **REPLACED/DEFERRED** | Rust `od_workflows::tests::prepare_observations_executes_repeated_pipeline_and_retains_snapshots` pins numerical original/used retention; Python member-table assembly is implemented by the canonical batch-native workflow layer. |
| `test_none_and_empty_sequence_return_input` | **RETIRED-API** | Legacy `models=` strings/sequences and executable Python model behavior were removed without aliases. |
| `test_object_id_is_passed_to_backend_and_stamped_when_null` | **REPLACED-eack9** | `test_overlap_first_appearance_source_order_and_no_iod_placeholder` pins explicit association group IDs and source-order membership; canonical groups never invent a null ID. |
| `test_object_id_none_is_null_on_both_routes` | **RETIRED-API** | Null linkage/object hypotheses are retired; `Associations.object_id` is the required non-null canonical group identity. |
| `test_provenance_joins_by_obs_id_when_fitter_reorders_members` | **REPLACED-eack9** | `test_overlap_first_appearance_source_order_and_no_iod_placeholder` and `test_membership_and_seed_group_fingerprints_are_enforced_before_factory` pin ID-based member identity and source-row order. |
| `test_run_od_explicit_default_models_missing_bias_dat_is_file_not_found` | **RETIRED-API** | Legacy `models=` strings/sequences and executable Python model behavior were removed without aliases. |
| `test_run_od_explicit_default_models_name_the_missing_data_package` | **RETIRED-API** | Legacy `models=` strings/sequences and executable Python model behavior were removed without aliases. |
| `test_run_od_explicit_default_models_report_every_missing_input_at_once` | **RETIRED-API** | Legacy `models=` strings/sequences and executable Python model behavior were removed without aliases. |
| `test_run_od_is_exported_from_the_package` | **REPLACED-eack9** | `test_primary_batch_names_are_exported_without_one_group_aliases` is the exact maintained replacement. |
| `test_run_od_rejects_unknown_models_string` | **RETIRED-API** | Legacy `models=` strings/sequences and executable Python model behavior were removed without aliases. |
| `test_run_od_with_the_default_stack` | **REPLACED/DEFERRED** | Real EFCC execution is pinned by `TestEfcc18::test_debias_model_positions`, non-commuting fill/deweight order by `od_config::tests::pipeline_executes_non_commuting_transforms_in_declaration_order`, and configured stage composition by the direct `od_workflows` composition test; canonical batch results are covered by the canonical batch-native workflow layer. |
| `test_sequence_is_applied_in_order_like_a_composite` | **REPLACED/DEFERRED** | `od_config::tests::pipeline_executes_non_commuting_transforms_in_declaration_order` fails under reversed execution; the `od_workflows` preparation test separately pins repeated nightly execution. Python composite callbacks are retired. |

### `test_od_python_baseline_parity.py`

| Baseline function | Disposition | Exact replacement or technical reason |
|---|---|---|
| `test_astcat_round_trip_and_covariance` | **REPLACED** | `test_od_observations.py::test_from_ades_retains_astcat` and `::test_from_ades_uncertainty_conversion` use the same ADES rows/formulas. |
| `test_every_model_matches_and_preserves_positions` | **RETIRED/REPLACED** | Closed transforms are covered by exact Rust tests `empirical_add_mode_at_equator_and_high_declination`, `empirical_replace_mode_nan_cross_term_and_pass_through_rules`, `night_batch_deweighting_is_per_station_and_night`, and Veres fill/floor/replace tests; identity/performance/sigma-floor/legacy-Veres executable Python models are outside the closed pipeline and retired. |
| `test_run_od_composite_provenance` | **REPLACED-eack9** | `test_profile_shaped_deweighting_is_exact_across_fused_and_composed_routes` and `test_route_and_configuration_provenance_are_cryptographically_bound` pin transformed member/provenance composition. |

### `test_orbit_fitter.py`

| Baseline function | Disposition | Exact replacement or technical reason |
|---|---|---|
| `test_initial_fit_only_fitter_stays_instantiable` | **RETIRED-API** | `test_initial_fit_only_fitter_stays_instantiable` exercised a removed Python fitter/model/deprecation interface; Core 0.7 intentionally provides no alias or executable callback surface. |
| `test_orbit_fitter_still_requires_initial_fit` | **RETIRED-API** | `test_orbit_fitter_still_requires_initial_fit` exercised a removed Python fitter/model/deprecation interface; Core 0.7 intentionally provides no alias or executable callback surface. |

### `test_provenance.py`

| Baseline function | Disposition | Exact replacement or technical reason |
|---|---|---|
| `test_ambiguous_iod_survives_refinement` | **DEFERRED/REPLACED** | Runtime build/provider/input/result provenance is covered by the canonical batch-native workflow layer; path-free config/resource identity is already pinned by `od_config::tests::complete_config_fingerprint_is_deterministic_and_path_free` and external ABI identity tests. |
| `test_backend_identity_names_declared_and_python_backends` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_bias_module_delegate_mutation_is_refused_before_execution` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_bias_row_columns_monkeypatch_is_custom_on_real_route` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_builtin_behavior_overrides_and_nested_overrides_require_contract` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_builtin_descriptor_uses_exact_build_and_not_import_history` | **DEFERRED/REPLACED** | Runtime build/provider/input/result provenance is covered by the canonical batch-native workflow layer; path-free config/resource identity is already pinned by `od_config::tests::complete_config_fingerprint_is_deterministic_and_path_free` and external ABI identity tests. |
| `test_canonical_configuration_is_injective_and_user_mappings_cannot_forge_tags` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_common_configuration_values_are_portable_and_distinct` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_complete_provenance_keeps_distinct_native_stage_providers` | **DEFERRED/REPLACED** | Runtime build/provider/input/result provenance is covered by the canonical batch-native workflow layer; path-free config/resource identity is already pinned by `od_config::tests::complete_config_fingerprint_is_deterministic_and_path_free` and external ABI identity tests. |
| `test_composed_guard_rejects_model_state_changed_after_snapshot` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_composed_numpy_delegate_replacement_is_ignored_without_call` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_composite_native_model_specs_delegate_mutation_is_refused` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_contracted_moduleless_model_is_rejected` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_custom_contract_is_validated_exactly` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_custom_declaration_canonicalization_is_sensitive` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_custom_descriptor_is_stable_across_hash_seed_and_import_order` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_custom_fingerprint_is_the_implementation_identity` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_custom_non_string_module_metadata_is_never_formatted` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_custom_subclass_and_instance_apply_override_require_contract` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_default_constructor_mutation_is_refused_before_constructor` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_default_loader_selector_mutation_is_refused_before_loading` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_distribution_identity_probes_fail_closed_before_all_work` | **DEFERRED/REPLACED** | Runtime build/provider/input/result provenance is covered by the canonical batch-native workflow layer; path-free config/resource identity is already pinned by `od_config::tests::complete_config_fingerprint_is_deterministic_and_path_free` and external ABI identity tests. |
| `test_documented_custom_model_migration_policy_is_executable` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_exact_class_special_lookup_hooks_are_refused_before_execution` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_extra_hostile_instance_field_is_refused_before_value_lookup` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_genuine_installed_class_and_apply_monkeypatches_are_refused` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_hostile_model_sequence_is_rejected_without_iteration` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_hostile_registered_field_values_are_refused_without_hooks` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_inherited_abc_lookup_mutation_is_refused_without_execution` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_input_fingerprint_hashes_caller_and_used_astrometry_independently` | **DEFERRED/REPLACED** | Runtime build/provider/input/result provenance is covered by the canonical batch-native workflow layer; path-free config/resource identity is already pinned by `od_config::tests::complete_config_fingerprint_is_deterministic_and_path_free` and external ABI identity tests. |
| `test_local_contracted_class_is_rejected_before_all_work` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_main_dynamic_nonexistent_contract_is_rejected_before_all_work` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_model_descriptor_canonicalizes_colliding_sets_and_mapping_order` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_model_descriptor_hashes_numpy_configuration_and_sets` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_model_descriptor_hashes_table_content` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_model_descriptor_rejects_unsupported_and_cyclic_configuration` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_mutable_abc_metaclass_dispatch_is_attested_without_execution` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_native_kernel_delegate_mutation_is_refused_before_execution` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_native_model_spec_snapshot_has_no_mutable_model_references` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_native_settings_are_called_once_and_same_snapshot_is_recorded_and_used` | **DEFERRED/REPLACED** | Runtime build/provider/input/result provenance is covered by the canonical batch-native workflow layer; path-free config/resource identity is already pinned by `od_config::tests::complete_config_fingerprint_is_deterministic_and_path_free` and external ABI identity tests. |
| `test_nested_custom_lookup_is_refused_without_any_hook` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_nested_lookup_special_resolution_hook_is_refused_before_execution` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_other_entry_points_leave_provenance_null` | **DEFERRED/REPLACED** | Runtime build/provider/input/result provenance is covered by the canonical batch-native workflow layer; path-free config/resource identity is already pinned by `od_config::tests::complete_config_fingerprint_is_deterministic_and_path_free` and external ABI identity tests. |
| `test_preflight_describes_contract_before_apply_or_fit` | **DEFERRED/REPLACED** | Runtime build/provider/input/result provenance is covered by the canonical batch-native workflow layer; path-free config/resource identity is already pinned by `od_config::tests::complete_config_fingerprint_is_deterministic_and_path_free` and external ABI identity tests. |
| `test_preflight_failure_happens_before_fitter_work` | **DEFERRED/REPLACED** | Runtime build/provider/input/result provenance is covered by the canonical batch-native workflow layer; path-free config/resource identity is already pinned by `od_config::tests::complete_config_fingerprint_is_deterministic_and_path_free` and external ABI identity tests. |
| `test_preflight_rejects_native_python_version_mismatch_before_model_work` | **DEFERRED/REPLACED** | Runtime build/provider/input/result provenance is covered by the canonical batch-native workflow layer; path-free config/resource identity is already pinned by `od_config::tests::complete_config_fingerprint_is_deterministic_and_path_free` and external ABI identity tests. |
| `test_provenance_keeps_exact_native_identity_not_later_declarations` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_provenance_preflight_refuses_non_reproducible_builds` | **DEFERRED/REPLACED** | Runtime build/provider/input/result provenance is covered by the canonical batch-native workflow layer; path-free config/resource identity is already pinned by `od_config::tests::complete_config_fingerprint_is_deterministic_and_path_free` and external ABI identity tests. |
| `test_provider_cannot_replace_a_separate_orchestration_native_alias` | **DEFERRED/REPLACED** | Runtime build/provider/input/result provenance is covered by the canonical batch-native workflow layer; path-free config/resource identity is already pinned by `od_config::tests::complete_config_fingerprint_is_deterministic_and_path_free` and external ABI identity tests. |
| `test_provider_mutation_is_bound_by_final_model_and_native_spec_snapshot` | **DEFERRED/REPLACED** | Runtime build/provider/input/result provenance is covered by the canonical batch-native workflow layer; path-free config/resource identity is already pinned by `od_config::tests::complete_config_fingerprint_is_deterministic_and_path_free` and external ABI identity tests. |
| `test_provider_native_spec_builder_replacement_is_ignored_without_call` | **DEFERRED/REPLACED** | Runtime build/provider/input/result provenance is covered by the canonical batch-native workflow layer; path-free config/resource identity is already pinned by `od_config::tests::complete_config_fingerprint_is_deterministic_and_path_free` and external ABI identity tests. |
| `test_provider_time_native_replacement_is_refused_before_fake_native_work` | **DEFERRED/REPLACED** | Runtime build/provider/input/result provenance is covered by the canonical batch-native workflow layer; path-free config/resource identity is already pinned by `od_config::tests::complete_config_fingerprint_is_deterministic_and_path_free` and external ABI identity tests. |
| `test_public_pathlib_class_identity_is_normalized_across_private_relocation` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_public_run_od_recursively_preflights_nested_custom_models` | **DEFERRED/REPLACED** | Runtime build/provider/input/result provenance is covered by the canonical batch-native workflow layer; path-free config/resource identity is already pinned by `od_config::tests::complete_config_fingerprint_is_deterministic_and_path_free` and external ABI identity tests. |
| `test_python_callable_configuration_requires_explicit_portable_data` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_real_git_source_checkout_detector_is_fail_closed` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_registered_instance_data_cannot_be_shadowed_on_the_class` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_registered_instance_shadows_and_nested_subclass_are_custom` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_replaced_native_build_function_is_refused_without_call` | **DEFERRED/REPLACED** | Runtime build/provider/input/result provenance is covered by the canonical batch-native workflow layer; path-free config/resource identity is already pinned by `od_config::tests::complete_config_fingerprint_is_deterministic_and_path_free` and external ABI identity tests. |
| `test_replaced_native_package_module_is_refused_without_call` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_run_od_records_provenance` | **DEFERRED/REPLACED** | Runtime build/provider/input/result provenance is covered by the canonical batch-native workflow layer; path-free config/resource identity is already pinned by `od_config::tests::complete_config_fingerprint_is_deterministic_and_path_free` and external ABI identity tests. |
| `test_runtime_untracked_source_guard_precedes_models` | **DEFERRED/REPLACED** | Runtime build/provider/input/result provenance is covered by the canonical batch-native workflow layer; path-free config/resource identity is already pinned by `od_config::tests::complete_config_fingerprint_is_deterministic_and_path_free` and external ABI identity tests. |
| `test_sigma_fill_model_constant_monkeypatch_is_custom_on_real_route` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_uniform_provider_rejects_disagreement_and_partial_omission` | **DEFERRED/REPLACED** | Runtime build/provider/input/result provenance is covered by the canonical batch-native workflow layer; path-free config/resource identity is already pinned by `od_config::tests::complete_config_fingerprint_is_deterministic_and_path_free` and external ABI identity tests. |
| `test_unsupported_dynamic_model_has_documented_type_error` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_veres_lookup_payload_monkeypatch_is_custom_on_real_route` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |
| `test_veres_registered_class_mutations_are_custom` | **RETIRED-API** | This assertion attested mutable/executable Python model classes, hooks, delegates, mappings, descriptors, or custom contracts. That design was removed; the closed Rust `ObservationTransform` enum has no equivalent attack surface. |

### `test_reconstruction_contracts.py`

| Baseline function | Disposition | Exact replacement or technical reason |
|---|---|---|
| `test_build_identity_is_exact_clean_head` | **REPLACED-eack9** | `test_native_build_identity_is_exact_clean_head` is the maintained runtime assertion; release inventory separately enforces clean artifact source. |
| `test_efcc_descriptor_content_version_exclusions` | **RETIRED-API** | Python model-descriptor identity was removed; bound Rust resource/config fingerprints replace it (`od_config::tests::complete_config_fingerprint_is_deterministic_and_path_free`). |
| `test_explicit_stage_composition_route_completes` | **REPLACED-eack9** | `test_run_od_is_exact_run_iod_then_refine` and `test_run_od_failure_is_exact_run_iod_then_refine` is the exact maintained replacement. |
| `test_fused_and_composed_share_native_model_science_and_provenance` | **REPLACED-eack9** | `test_run_od_is_exact_run_iod_then_refine` and `test_provider_factory_counts_and_external_exact_composition` is the exact maintained replacement. |
| `test_input_fingerprint_is_layout_canonical` | **REPLACED-eack9** | `test_input_fingerprint_is_independent_of_arrow_chunk_layout` is the exact maintained replacement. |
| `test_input_fingerprint_is_sensitive` | **REPLACED-eack9** | `test_member_source_identity_mutations_fail_before_provider_binding` and `test_orbit_state_and_provenance_mutations_abort_before_factory` is the exact maintained replacement. |
| `test_model_descriptor_numpy_variants` | **RETIRED-API** | Python model-descriptor identity was removed; bound Rust resource/config fingerprints replace it (`od_config::tests::complete_config_fingerprint_is_deterministic_and_path_free`). |
| `test_provider_metadata_comes_from_capsule_not_python_declaration` | **REPLACED** | `test_jacobian_declaration.py::test_capsule_identity_prevents_two_body_spoofing` proves capsule identity is authoritative; `test_explicit_evaluation_subset_and_stage_provenance` pins emitted capsule metadata. |
| `test_two_distinct_epoch_arcs_are_refused_on_all_routes` | **REPLACED/RETIRED-API** | Rust `od_contract_tests::fewer_than_three_distinct_epochs_is_rank_deficient_before_search`, `::corrected_candidate_pool_rejects_two_epochs_before_search`, and `od_refine::tests::fit_rejects_fewer_than_three_distinct_epochs_before_prediction` pin eager single-group IOD/refinement refusal; only removed Python route aliases are retired. |

### `test_run_iod.py`

| Baseline function | Disposition | Exact replacement or technical reason |
|---|---|---|
| `test_fitter_forwards_the_velocity_method` | **RETIRED-API** | Python fitter dispatch/default forwarding was removed; typed Rust `IodRunConfig` is the only policy path. |
| `test_fitter_initial_fit_runs_iod_on_the_configured_class` | **RETIRED-API** | Python fitter dispatch/default forwarding was removed; typed Rust `IodRunConfig` is the only policy path. |
| `test_initial_orbit_determination_accepts_an_unmatched_linkage` | **RETIRED-API** | Public linkage orchestration was removed; canonical grouping uses `Associations` and `test_overlap_first_appearance_source_order_and_no_iod_placeholder`. |
| `test_initial_orbit_determination_runs_run_iod_per_linkage` | **RETIRED-API** | Public linkage orchestration was removed; canonical grouping uses `Associations` and `test_overlap_first_appearance_source_order_and_no_iod_placeholder`. |
| `test_iod_is_the_accepted_orbit_of_run_iod` | **REPLACED** | `test_iod.py::test_iod_fused_matches_pinned_legacy_fixture` and current `test_run_iod.py` corrected/prefit acceptance tests pin the accepted candidate policy. |
| `test_linkage_api_validates_settings_before_empty_exit` | **RETIRED-API** | Public linkage orchestration was removed; canonical grouping uses `Associations` and `test_overlap_first_appearance_source_order_and_no_iod_placeholder`. |
| `test_native_iod_rejects_invalid_settings_centrally` | **REPLACED** | `test_run_iod.py::test_run_iod_validates_all_settings_before_empty_exit` and `od_config::tests::stage_json_round_trips_strictly_and_rejects_unknown_or_missing_fields`. |
| `test_refinement_signature_defaults_are_the_rust_defaults` | **REPLACED** | `test_final_config.py::test_complete_stage_configs_are_explicitly_constructible` and `::test_core_constants_match_d21_effective_values_exactly`; high-level APIs have no implicit defaults. |
| `test_run_iod_capsule_matches_in_tree_two_body_driver` | **REPLACED** | Rust `od_refine::tests::external_propagators_run_the_od_stages_like_native_ones`. |
| `test_run_iod_then_refine_is_the_stage_contract` | **REPLACED** | `od_workflows::tests::run_od_prepares_once_and_exactly_composes_isolated_stage_predictors`. |
| `test_signature_defaults_are_the_rust_defaults` | **REPLACED** | `test_final_config.py::test_complete_stage_configs_are_explicitly_constructible` and `::test_core_constants_match_d21_effective_values_exactly`; high-level APIs have no implicit defaults. |

### `test_run_od_null_covariance.py`

| Baseline function | Disposition | Exact replacement or technical reason |
|---|---|---|
| `test_fused_and_composed_native_model_routes_agree_with_filled_sigmas` | **SUCCESSOR-personal-7olgj** | Exact fill-resource wiring requires the separately tracked resource-loader task; eack9 pins the same lossless transformed-variance handoff with profile-shaped `NightlyBatchDeweighting`. |
| `test_sigma_fill_supplies_the_missing_covariance` | **DEFERRED-resource/eack9** | End-to-end fill→OD needs the separately supplied pinned station uncertainty resource; binding failure is covered by `::test_fill_transform_requires_explicit_validated_resource`, final configured workflow parity by the canonical workflow plus resource wiring. |
| `test_without_a_filling_model_the_fit_is_refused` | **REPLACED** | `test_run_od_null_covariance.py::test_without_a_fill_transform_the_fit_is_refused`. |

### `test_solver_budgets.py`

| Baseline function | Disposition | Exact replacement or technical reason |
|---|---|---|
| `test_fitter_serializes_both_solver_limits` | **REPLACED** | Rust `od_refine::tests::iteration_budget_is_independent_of_evaluation_budget` and `::evaluation_budget_bounds_every_residual_evaluation`, plus `od_config::tests::stage_json_round_trips_strictly_and_rejects_unknown_or_missing_fields`. |

### `test_veres2017.py`

| Baseline function | Disposition | Exact replacement or technical reason |
|---|---|---|
| `test_composes_after_efcc18_style_position_models` | **RETIRED/REPLACED** | Executable Python model composition is retired; native declaration-order execution is pinned by `od_config::tests::pipeline_executes_non_commuting_transforms_in_declaration_order`, with EFCC positions covered by the real-table replay. |
| `test_composes_with_other_models` | **RETIRED/REPLACED** | Executable Python model composition is retired; synthetic fill followed by nightly deweighting is pinned as a non-commuting native sequence by `od_config::tests::pipeline_executes_non_commuting_transforms_in_declaration_order`. |
| `test_cross_term_is_preserved` | **REPLACED/RETIRED** | `veres2017::tests::floor_model_floors_per_axis_with_cos_dec_and_fill_missing`. |
| `test_default_table_without_the_package_raises_import_error` | **DEFERRED/RETIRED** | Runtime package discovery and legacy default tables are retired; exact separately supplied resource parsing is the ignored acceptance test `od_config::tests::separately_packaged_reference_resources_parse` pending resource wiring. |
| `test_duplicate_rows_raise` | **REPLACED/RETIRED** | `veres2017::tests::lookup_order_and_validation`. |
| `test_fills_only_missing_axes_and_keeps_reported` | **REPLACED/RETIRED** | `veres2017::tests::fill_model_fills_only_missing_axes_and_zeroes_cross_term`. |
| `test_floors_small_sigmas_and_keeps_large_ones` | **REPLACED/RETIRED** | `veres2017::tests::floor_model_floors_per_axis_with_cos_dec_and_fill_missing`. |
| `test_global_row_beats_fallback_and_is_unique` | **REPLACED/RETIRED** | `veres2017::tests::generic_rows_lookup_order_station_catalog_station_catalog_global`. |
| `test_known_working_table_entries` | **DEFERRED/RETIRED** | Runtime package discovery and legacy default tables are retired; exact separately supplied resource parsing is the ignored acceptance test `od_config::tests::separately_packaged_reference_resources_parse` pending resource wiring. |
| `test_lookup_order` | **REPLACED/RETIRED** | `veres2017::tests::lookup_order_and_validation`. |
| `test_lookup_order_station_catalog_station_catalog_global` | **REPLACED/RETIRED** | `veres2017::tests::generic_rows_lookup_order_station_catalog_station_catalog_global`. |
| `test_missing_column_raises` | **REPLACED/RETIRED** | `od_config::tests::separately_packaged_reference_resources_parse` plus strict `sigma_schema` validation (resource-gated). |
| `test_missing_sigma_left_unless_fill_missing` | **REPLACED/RETIRED** | `veres2017::tests::floor_model_floors_per_axis_with_cos_dec_and_fill_missing`. |
| `test_model_defaults_resolve_the_package_tables` | **DEFERRED/RETIRED** | Runtime package discovery and legacy default tables are retired; exact separately supplied resource parsing is the ignored acceptance test `od_config::tests::separately_packaged_reference_resources_parse` pending resource wiring. |
| `test_no_fallback_returns_none` | **REPLACED/RETIRED** | `veres2017::tests::lookup_order_and_validation`. |
| `test_non_positive_sigma_raises` | **REPLACED/RETIRED** | `veres2017::tests::lookup_order_and_validation`. |
| `test_nothing_missing_returns_same_object` | **REPLACED/RETIRED** | The executable Python identity/object-reuse contract is retired; Rust `fill_model_fills_only_missing_axes_and_zeroes_cross_term` pins unchanged values. |
| `test_null_astcat_is_a_station_or_global_row` | **REPLACED/RETIRED** | `veres2017::tests::generic_rows_lookup_order_station_catalog_station_catalog_global`. |
| `test_package_working_table_still_validates` | **DEFERRED/RETIRED** | Runtime package discovery and legacy default tables are retired; exact separately supplied resource parsing is the ignored acceptance test `od_config::tests::separately_packaged_reference_resources_parse` pending resource wiring. |
| `test_replaces_both_axes_and_zeroes_cross_term` | **REPLACED/RETIRED** | `veres2017::tests::replace_model_overwrites_and_zeroes_cross_term`. |
| `test_unknown_catalog_uses_fallback_or_passes_through` | **REPLACED/RETIRED** | `veres2017::tests::lookup_order_and_validation`. |
| `test_unresolvable_without_global_or_fallback_stays_nan` | **REPLACED/RETIRED** | `veres2017::tests::lookup_order_and_validation`. |
| `test_working_table_schema_and_sanity` | **DEFERRED/RETIRED** | Runtime package discovery and legacy default tables are retired; exact separately supplied resource parsing is the ignored acceptance test `od_config::tests::separately_packaged_reference_resources_parse` pending resource wiring. |

## Deleted d21 Rust `run_od` assertions

| Deleted d21 Rust assertion | Disposition | Exact replacement or deferral |
|---|---|---|
| `run_od_applies_models_and_records_provenance` | **REPLACED-eack9** | `test_profile_shaped_deweighting_is_exact_across_fused_and_composed_routes` and `test_route_and_configuration_provenance_are_cryptographically_bound` pin transform execution, exact handoff, and canonical provenance. |
| `run_od_lets_a_sigma_model_fill_a_missing_covariance` | **SUCCESSOR-personal-7olgj** | End-to-end fill requires the separately tracked explicit resource-loader/binding task; no runtime fallback or synthetic Python model is permitted. |
| `run_od_without_covariance_or_a_filling_model_is_refused` | **REPLACED** | `test_run_od_null_covariance.py::test_without_a_fill_transform_the_fit_is_refused`. |
| The deleted `run_od_with` block inside `iod_refinement_full_od_and_run_od_use_only_their_configured_stage_predictors` | **REPLACED** | `od_workflows::tests::run_od_prepares_once_and_exactly_composes_isolated_stage_predictors` directly exercises the public configured predictor seam, exact prepared-row IOD→refine composition, prepare-once behavior, and stage isolation. |

## Mechanical verification

The crosswalk is generated from AST function names at the exact baseline and current tree. The restored section is excluded from the absent set; every remaining `(file, function)` appears exactly once above. The correction report records the independent count/check command.
