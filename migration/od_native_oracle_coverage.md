# Native replacements for retired Python-callback OD oracles

The propagation-only contract cannot execute the former fake
`LinearPropagator.generate_ephemeris` inside an adam-core Rust workflow. Commit
`22de8c7a` initially left 23 tests collected but unconditionally skipped through
`make_linear_problem`; that is not coverage. The dead callback fixtures and
those tests are removed. Their closed-form responsibilities are active in the
native tests below (all in
`rust/adam_core_rs_coords/src/propagation/od_refine.rs`) and in the capsule
parity tests in the Python OD suite.

| Retired Python oracle (parameter case where applicable) | Active native equivalent(s) |
|---|---|
| `TestSyntheticLinearCovariance::test_covariance_matches_weighted_least_squares` | `linear_predictor_recovers_exact_wls_covariance_and_unit_quadratic_probe` |
| `TestSyntheticLinearCovariance::test_probe_measures_unity_on_quadratic_surface` | `linear_predictor_recovers_exact_wls_covariance_and_unit_quadratic_probe` |
| `TestValidatedCovariance::test_trustworthy_covariance_passes_silently` | `linear_predictor_recovers_exact_wls_covariance_and_unit_quadratic_probe` |
| `TestValidatedCovariance::test_conditioning_warning_for_fd_covariance` | `linear_predictor_preserves_probe_warning_branches_and_analytic_fallback` |
| `TestValidatedCovariance::test_analytic_failure_falls_back_to_central_difference` | `linear_predictor_preserves_probe_warning_branches_and_analytic_fallback` |
| `TestLinearProblemWithOutlier::test_huber_bounds_the_outlier_influence` | `linear_predictor_huber_covariance_removes_tail_and_probe_uses_robust_cost` |
| `TestLinearProblemWithOutlier::test_huber_members_report_weights_not_rejections` | `linear_predictor_huber_covariance_removes_tail_and_probe_uses_robust_cost` and active Python `TestTwoBodyWithOutlier::test_outlier_is_downweighted_not_rejected` |
| `TestLinearProblemWithOutlier::test_huber_covariance_drops_the_outlier_information` | `linear_predictor_huber_covariance_removes_tail_and_probe_uses_robust_cost` |
| `TestLinearProblemWithOutlier::test_probe_uses_robust_cost` | `linear_predictor_huber_covariance_removes_tail_and_probe_uses_robust_cost` |
| `TestFullPredictionFinalization::test_covariance_uses_full_predictor_independently_of_solver[False-analytic]` | `linear_predictor_cmc_finalization_uses_full_non_keplerian_sensitivities` |
| `TestFullPredictionFinalization::test_covariance_uses_full_predictor_independently_of_solver[False-central]` | `linear_predictor_cmc_finalization_uses_full_non_keplerian_sensitivities` |
| `TestFullPredictionFinalization::test_covariance_uses_full_predictor_independently_of_solver[False-2-point]` | `linear_predictor_cmc_finalization_uses_full_non_keplerian_sensitivities` |
| `TestFullPredictionFinalization::test_covariance_uses_full_predictor_independently_of_solver[True-analytic]` | `linear_predictor_cmc_finalization_uses_full_non_keplerian_sensitivities` |
| `TestFullPredictionFinalization::test_covariance_uses_full_predictor_independently_of_solver[True-central]` | `linear_predictor_cmc_finalization_uses_full_non_keplerian_sensitivities` |
| `TestFullPredictionFinalization::test_covariance_uses_full_predictor_independently_of_solver[True-2-point]` | `linear_predictor_cmc_finalization_uses_full_non_keplerian_sensitivities` |
| `TestFullPredictionFinalization::test_near_threshold_selection_and_final_refit_use_consistent_derivatives[6.0-1-analytic]` | `linear_predictor_cmc_near_threshold_recovery_and_final_mask_covariance` |
| `TestFullPredictionFinalization::test_near_threshold_selection_and_final_refit_use_consistent_derivatives[6.0-1-central]` | `linear_predictor_cmc_near_threshold_recovery_and_final_mask_covariance` |
| `TestFullPredictionFinalization::test_near_threshold_selection_and_final_refit_use_consistent_derivatives[6.0-1-2-point]` | `linear_predictor_cmc_near_threshold_recovery_and_final_mask_covariance` |
| `TestFullPredictionFinalization::test_near_threshold_selection_and_final_refit_use_consistent_derivatives[10.0-2-analytic]` | `linear_predictor_cmc_near_threshold_recovery_and_final_mask_covariance` |
| `TestFullPredictionFinalization::test_near_threshold_selection_and_final_refit_use_consistent_derivatives[10.0-2-central]` | `linear_predictor_cmc_near_threshold_recovery_and_final_mask_covariance` |
| `TestFullPredictionFinalization::test_near_threshold_selection_and_final_refit_use_consistent_derivatives[10.0-2-2-point]` | `linear_predictor_cmc_near_threshold_recovery_and_final_mask_covariance` |
| `TestFullPredictionFinalization::test_full_model_fit_rejects_outlier_and_recovers_known_state` | `linear_predictor_cmc_rejects_outlier_and_recovers_known_state_and_covariance` |
| `TestFullPredictionFinalization::test_central_step_size_stability_against_closed_form_covariance` | `linear_predictor_cmc_finalization_uses_full_non_keplerian_sensitivities` |

Additional active boundary coverage is
`external_propagators_run_the_od_stages_like_native_ones` (native Rust vs the
external capsule), while Python
`test_run_iod_capsule_matches_in_tree_two_body_driver` compares the public
capsule route with the distinct built-in two-body driver. Solver-budget tests
assert exact evaluation/iteration limits rather than repeat-run determinism. The closed-form native tests above additionally pin
warning counts, final-refit accounting, robust-cost probing, and covariance
formation from the final selected mask. Stage routing is covered by
`iod_refinement_full_od_and_run_od_use_only_their_configured_stage_predictors`,
which gives IOD and refinement separate counting predictors and verifies the
configured triplet budget.
