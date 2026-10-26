# OD `CoordinateCovariances.to_matrix()` boundary audit

Audit baseline: `cdaa4d0d87758829b61f8b449061336d44a31870`.

The policy is that `CoordinateCovariances.to_matrix()` is Python
representation/I/O plumbing only. Covariance selection, masking, validation,
whitening, and arithmetic used by OD belong in Rust.

## Production OD call sites

| Call site | Classification |
|---|---|
| `evaluate.py` | Boundary serialization of observed/predicted covariance into one fused Rust evaluation call. All residual/covariance arithmetic is native. |
| `differential_correction.py` | Boundary serialization into the Rust Cholesky/whitening kernel. Factorization and validation are native. |
| `fitted_orbits.py` | Output-table assembly: reads the native result covariance to snapshot lon/lat uncertainties; square roots run in Rust. It is not part of fitting or model execution. |
| `observation_uncertainty.py` and `veres2017.py` | Boundary serialization for Rust model kernels in the rejected composed implementation. The concurrent native-model successor replaces the blessed composed/fused execution with one Rust application stage; this task deliberately does not edit those files. |
| `least_squares.py` (before this change) | **Violation:** `_angular_only` allocated a Python matrix, filled it with NaNs, copied the angular block, and rebuilt covariance before native refinement. Removed. The wrapper sends typed observations unchanged and a private compatibility setting tells the PyO3/Rust driver to retain the angular 2×2 block before Rust validation/whitening. |

The deprecated `LeastSquares` wrapper now passes typed observations to
`refine(..., rejection="none")` unchanged. Native selection clears non-angular
coordinate values, converts either 6D or 9D covariance rows to the legacy 6D
angular-only representation, ignores finite or invalid irrelevant blocks exactly
as before, and leaves relevant angular values unchanged for fail-closed Rust
whitening validation. Exact-equivalence, 9D, hostile irrelevant-value,
invalid-angular-value, and monkeypatched `CoordinateCovariances` regressions
prove that no Python covariance materialization remains in this compatibility
path.

All typed complete-OD entries (`full_od_barycentric`, `full_od_with`,
`run_od_barycentric`, and `run_od_with`) validate observation-covariance storage
dimension as their first operation. Thus public `run_od` rejects all-9D and mixed
6D/9D Arrow inputs with the accurate typed dimension error before snapshots,
observation models, fixed-6 indexing, or missing-variance interpretation. The
ordinary refinement boundary retains the same validation through
`observed_covariance_flat`.

## Final-line deprecated compatibility coverage (N1)

`_legacy_angular_only_covariance` is a one-release private bridge for deprecated
`LeastSquares.least_squares` only. The shared PyO3 settings parser carries the
flag, but only `fit_orbit_whitened_ipc` consumes it; complete IOD/OD entries
ignore it and still reject 9D or mixed observation covariance. A direct final-line
`run_od_ipc` regression freezes that non-expansion. New code must use supported
6D observation covariance rather than adopting this deprecated flag.
