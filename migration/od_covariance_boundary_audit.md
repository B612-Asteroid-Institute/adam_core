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
`refine(..., rejection="none")` unchanged. Native selection clears non-angular coordinate values, converts either 6D or 9D
covariance rows to the legacy 6D angular-only representation, ignores finite or
invalid irrelevant blocks exactly as before, and leaves relevant
angular values unchanged for fail-closed Rust whitening validation. Ordinary OD
rejects unsupported 9D observation covariance with a typed `ValueError` instead
of reaching fixed-stride indexing or a Rust assertion. Exact-equivalence, 9D,
hostile irrelevant-value, invalid-angular-value, and monkeypatched
`CoordinateCovariances` regressions prove that no Python covariance
materialization remains in this compatibility path.
