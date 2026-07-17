# Dynamics, missions, OD, IOD, and impacts public-surface disposition

Updated 2026-07-17 against upstream main
`9b756803ab3afbe11e33df9e57d30a28e7976b92`. The complete symbol inventory is
`manifest.json`; the 44-row parity registry is only a benchmark subset.

## Scalar and batch dynamics

| Surface group | Disposition | Evidence |
|---|---|---|
| mean motion, C3, Tisserand, Barker/Stumpff/Lagrange/Kepler helpers | direct Rust scalar/vector kernels with thin Python type/error veneers | fixed/fuzz parity and Rust `Instant` timing |
| Lambert and MOID (single, batch, perturbers) | one Rust crossing; warning/error/order semantics preserved | zero-vector warning, random/fixed parity, native timing |
| two-body propagation, arc batches, covariance propagation, ephemeris generation, light time | one Rust/Arrow crossing for public defaults | random/fixed parity, covariance fixtures, scaling and native timing |
| supplied abstract/custom propagator or optional SPK propagation | explicit provider boundary | provider/fallback tests |
| porkchop/C3 mission grids and mission departure preparation | fused Rust computation crossing; Python wraps tables and supplied providers | fixture parity, scaling, native timing |

## Orbit determination and IOD

| Surface group | Disposition | Evidence |
|---|---|---|
| Gibbs, Herrick-Gibbs, Gauss roots/candidates | direct Rust kernels; candidate priority/order preserved; a supplied central-body `mu` consistently governs geometry and velocity rather than reproducing the legacy split-`mu` bug | fixed/random parity, custom-`mu` regression, and timing |
| least-squares fitter, differential correction, `evaluate_orbits` | fused backend-generic Rust work units; Python preserves public table/errors and unsupported-provider fallback | latest-oracle fixtures, ignored-observation/order/statistics tests, timing |
| `iod_worker`, linkage IOD, and `initial_orbit_determination` | fused Rust orchestration through the selected backend; Python supplies nondeterministic IDs and fallback for unsupported providers | full-linkage fixture, root/order tests, ASSIST integration, timing |
| top-level OD batch and scheduling parameters | Rust/ASSIST scheduling; historical Ray parameters are signature-compatible no-ops | serial/parallel parity and no-Ray import tests |

## Impacts and associations

| Surface group | Disposition | Evidence |
|---|---|---|
| impact detection, probability reduction, Mahalanobis distance, linkage collapse | one Rust crossing per public work unit | deterministic fixtures, covariance/statistical tests, native timing |
| observation/exposure/source association and ADES preparation used by OD | one Arrow crossing; Rust owns matching/grouping/product assembly | product fixtures, ordering/null/error tests, timing |

## Timing and cache policy

Every qualifying operation has a Rust-owned timing adapter using
`std::time::Instant`; Python/PyO3/PyArrow conversion is outside samples.
Public performance promotion is controlled by legacy/current Python timings;
native Rust is diagnostic. Observer, perturber, SPICE, and translation compute
rows clear semantic result caches before each warmup and timed sample. Cache-hit
identities are separate and never used as compute evidence.

## Closure

No adam-core-owned numerical orchestration gap remains in these domains.
Abstract propagators, explicitly supplied backends, and optional SPK propagation
are deliberate provider boundaries rather than default Python implementations.
