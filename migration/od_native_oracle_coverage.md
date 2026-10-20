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
| `TestSyntheticLinearCovariance::test_covariance_matches_weighted_least_squares` | `analytic_fit_recovers_truth_with_validated_covariance` |
| `TestSyntheticLinearCovariance::test_probe_measures_unity_on_quadratic_surface` | `fabricated_covariance_fails_the_probe`, `analytic_fit_recovers_truth_with_validated_covariance` |
| `TestValidatedCovariance::test_trustworthy_covariance_passes_silently` | `analytic_fit_recovers_truth_with_validated_covariance` |
| `TestValidatedCovariance::test_conditioning_warning_for_fd_covariance` | `fabricated_covariance_fails_the_probe` |
| `TestValidatedCovariance::test_analytic_failure_falls_back_to_central_difference` | `non_finite_analytic_jacobian_continues_with_central_differences`, `fabricated_covariance_fails_the_probe` |
| `TestLinearProblemWithOutlier::test_huber_bounds_the_outlier_influence` | `huber_loss_downweights_a_gross_outlier` |
| `TestLinearProblemWithOutlier::test_huber_members_report_weights_not_rejections` | `huber_loss_downweights_a_gross_outlier` and active Python `TestTwoBodyWithOutlier::test_outlier_is_downweighted_not_rejected` |
| `TestLinearProblemWithOutlier::test_huber_covariance_drops_the_outlier_information` | `huber_loss_downweights_a_gross_outlier`, `analytic_fit_recovers_truth_with_validated_covariance` |
| `TestLinearProblemWithOutlier::test_probe_uses_robust_cost` | `huber_loss_downweights_a_gross_outlier`, `fabricated_covariance_fails_the_probe` |
| `TestFullPredictionFinalization::test_covariance_uses_full_predictor_independently_of_solver[False-analytic]` | `analytic_fit_recovers_truth_with_validated_covariance`, `cmc2003_finalization_uses_one_full_predictor_jacobian_outside_solver_budget` |
| `TestFullPredictionFinalization::test_covariance_uses_full_predictor_independently_of_solver[False-central]` | `finite_difference_jacobians_reach_the_same_minimum`, `cmc2003_finalization_uses_one_full_predictor_jacobian_outside_solver_budget` |
| `TestFullPredictionFinalization::test_covariance_uses_full_predictor_independently_of_solver[False-2-point]` | `finite_difference_jacobians_reach_the_same_minimum`, `cmc2003_finalization_uses_one_full_predictor_jacobian_outside_solver_budget` |
| `TestFullPredictionFinalization::test_covariance_uses_full_predictor_independently_of_solver[True-analytic]` | `analytic_fit_recovers_truth_with_validated_covariance`, `fabricated_covariance_fails_the_probe` |
| `TestFullPredictionFinalization::test_covariance_uses_full_predictor_independently_of_solver[True-central]` | `finite_difference_jacobians_reach_the_same_minimum`, `fabricated_covariance_fails_the_probe` |
| `TestFullPredictionFinalization::test_covariance_uses_full_predictor_independently_of_solver[True-2-point]` | `finite_difference_jacobians_reach_the_same_minimum`, `fabricated_covariance_fails_the_probe` |
| `TestFullPredictionFinalization::test_near_threshold_selection_and_final_refit_use_consistent_derivatives[6.0-1-analytic]` | `cmc2003_rejects_gross_outliers_and_recovers_truth`, `cmc2003_finalization_uses_one_full_predictor_jacobian_outside_solver_budget` |
| `TestFullPredictionFinalization::test_near_threshold_selection_and_final_refit_use_consistent_derivatives[6.0-1-central]` | `cmc2003_rejects_gross_outliers_and_recovers_truth`, `finite_difference_jacobians_reach_the_same_minimum` |
| `TestFullPredictionFinalization::test_near_threshold_selection_and_final_refit_use_consistent_derivatives[6.0-1-2-point]` | `cmc2003_rejects_gross_outliers_and_recovers_truth`, `finite_difference_jacobians_reach_the_same_minimum` |
| `TestFullPredictionFinalization::test_near_threshold_selection_and_final_refit_use_consistent_derivatives[10.0-2-analytic]` | `cmc2003_rejects_gross_outliers_and_recovers_truth`, `cmc2003_max_iterations_refits_on_the_final_selection` |
| `TestFullPredictionFinalization::test_near_threshold_selection_and_final_refit_use_consistent_derivatives[10.0-2-central]` | `cmc2003_rejects_gross_outliers_and_recovers_truth`, `finite_difference_jacobians_reach_the_same_minimum` |
| `TestFullPredictionFinalization::test_near_threshold_selection_and_final_refit_use_consistent_derivatives[10.0-2-2-point]` | `cmc2003_rejects_gross_outliers_and_recovers_truth`, `finite_difference_jacobians_reach_the_same_minimum` |
| `TestFullPredictionFinalization::test_full_model_fit_rejects_outlier_and_recovers_known_state` | `cmc2003_rejects_gross_outliers_and_recovers_truth` |
| `TestFullPredictionFinalization::test_central_step_size_stability_against_closed_form_covariance` | `finite_difference_jacobians_reach_the_same_minimum`, `analytic_fit_recovers_truth_with_validated_covariance` |

Additional active boundary coverage is
`external_propagators_run_the_od_stages_like_native_ones` (native Rust vs the
external capsule), while Python
`test_run_iod_capsule_matches_in_tree_two_body_driver` compares the public
capsule route with the distinct built-in two-body driver. Solver-budget tests
assert exact evaluation/iteration limits rather than repeat-run determinism.
