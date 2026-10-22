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
| `least_squares.py` (before this change) | **Violation:** `_angular_only` allocated a Python matrix, filled it with NaNs, copied the angular block, and rebuilt covariance before native refinement. Removed. The native fitter already consumes only the angular 2x2 block and rejects/ignores other dimensions by contract. |

The deprecated `LeastSquares` wrapper now passes the typed observations to
`refine(..., rejection="none")` unchanged. Its exact-equivalence test and a
regression that makes `CoordinateCovariances.to_matrix` raise prove that no
Python covariance materialization remains in this compatibility path.
