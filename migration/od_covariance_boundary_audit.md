# OD covariance boundary audit

Correction baseline: `18e230cd9bb9772ff7d40d915fd179fe54a3d8b0`.

The policy is that `CoordinateCovariances.to_matrix()` is Python
representation/I/O plumbing only. Covariance selection, validation, model
application, whitening, rejection, and fitting arithmetic used by OD belong in
Rust.

## Final production boundaries

| Call site | Final classification |
|---|---|
| `evaluate.py` | Representation marshalling into one Rust evaluation call. Rust owns residual and covariance arithmetic. |
| `algorithms/fitting.py` | Representation marshalling for retained expert numerical kernels. Rust owns covariance validation, Cholesky whitening, solver and fit-covariance work. |
| `fitted_orbits.py` | Output-table assembly from a native result covariance. Native code computes the uncertainty square roots. |
| `workflows.py` and `_pipeline.py` | Arrow IPC marshalling only. The configured observation pipeline and every canonical OD stage execute in Rust. |

There is no `LeastSquares` compatibility facade and no private
`_legacy_angular_only_covariance` setting. Unsupported range-bearing
observations and unsupported covariance dimensions fail instead of being
projected through an untested fallback.

## Dimension-first complete-OD contract

The canonical Rust pipeline validates an observation covariance's storage
dimension before snapshots, transforms, fixed-stride indexing, or lon/lat
variance interpretation. Missing covariance remains permitted at this first
check because an explicitly configured fill transform may supply it; any
present covariance must have dimension six.

`od::algorithms::prepare_observations` and `run_od_with_predictors` both enter
that guarded pipeline. The PyO3 pipeline and batch IPC routes use the same
implementation, including association-group preparation. Lower-level
refinement independently reaches `observed_covariance_flat`, which applies the
same typed dimension guard before returning covariance values.

Maintained regressions freeze the contract:

- Rust `complete_od_rejects_covariance_dimension_before_pipeline_interpretation`
  covers public `prepare_observations` and `run_od_with_predictors` with both
  explicit as-supplied and model-changing pipelines.
- Python `test_batch_ipc_rejects_unsupported_covariance_dimension_first`
  covers uniform 9D and mixed 6D/9D Arrow covariance rows through the canonical
  native batch route and requires the typed dimension error rather than the
  missing-variance error.
- Existing exact-composition tests continue to prove that `run_od` prepares
  once and is bitwise equal to `run_iod` followed by `refine`, including
  persisted original/used astrometry and covariance.

This is the canonical adaptation of the dimension-first correctness behavior
accepted in `bf9f6e9c`: the former one-group entry guards now live at the shared
Rust-owned preparation boundary used by the final batch architecture.
