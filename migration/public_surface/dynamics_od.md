# Dynamics, missions, OD, IOD, and impacts public-surface disposition

Updated 2026-08-12 against frozen updated-upstream main
`757c09fca86adf9e3d5899952db3d379e09413f6`. The complete 629-symbol inventory
and all frozen-upstream-only retirement classifications are in `manifest.json`;
the 44-row parity registry is only a benchmark subset.

## Scalar and batch dynamics

| Surface group | Disposition | Evidence |
|---|---|---|
| mean motion, C3, Tisserand, Barker/Stumpff/Lagrange/Kepler helpers | direct Rust scalar/vector kernels with thin Python type/error veneers | fixed/fuzz parity and Rust `Instant` timing |
| Lambert and MOID (single, batch, perturbers) | one Rust crossing; warning/error/order semantics preserved | zero-vector warning, random/fixed parity, native timing |
| two-body propagation, arc batches, covariance propagation, ephemeris generation, light time | one Rust/Arrow crossing for public defaults | random/fixed parity, covariance fixtures, scaling and native timing |
| supplied abstract/custom propagator or optional SPK propagation | explicit provider boundary | provider/fallback tests |
| porkchop/C3 mission grids and mission departure preparation | fused Rust computation crossing; Python wraps tables and supplied providers | fixture parity, scaling, native timing |
| historical Ray workers and scalar orchestration helpers (`*_worker_ray`, `moid_worker`, `lambert_worker`, scalar MOID internals) | intentionally retired implementation details; fused Rust batch entrypoints own the same useful workflows and `max_processes` remains signature-compatible | all retired symbol IDs are recorded in the manifest; no-Ray imports, public workflow parity, and native timing |

## Orbit determination and IOD

| Surface group | Disposition | Evidence |
|---|---|---|
| Gibbs, Herrick-Gibbs, Gauss roots/candidates | direct Rust kernels; candidate priority/order preserved; a supplied central-body `mu` consistently governs geometry and velocity rather than reproducing the legacy split-`mu` bug | fixed/random parity, custom-`mu` regression, and timing |
| least-squares fitter, differential correction, `evaluate_orbits` | adam-core-owned backend-generic Rust algorithms; Python preserves public tables/errors; native providers supply propagation capsules only and unsupported providers fail clearly | active Rust solver/covariance tests, ignored-observation/order/statistics tests, timing |
| `iod_worker`, linkage IOD, and `initial_orbit_determination` | adam-core-owned Rust linkage orchestration through explicit stage propagation capsules; Python supplies public table IDs only | full-linkage fixture, root/order tests, native capsule integration, timing |
| top-level OD batch and scheduling parameters | adam-core Rust scheduling; historical Ray parameters are signature-compatible no-ops; `iod_worker_remote` and `od_worker_remote` are retired | serial/parallel parity and no-Ray import tests |

## Fit-time observation models and robust differential correction

Added 2026-09-16 (branch `od-obs-uncertainty-rust`, reimplementing the Python
`kk/obs-uncertainty-interface` branch at `d56114ac`). Rust owns every
per-observation arithmetic; Python owns the quivr tables, schema validation and
the fitter orchestration that drives a supplied `Propagator`.

| Surface group | Disposition | Evidence |
|---|---|---|
| `ObservationUncertaintyModel` interpreters (`EmpiricalCovarianceModel`, `PerformanceWeightedModel`, `SigmaFloorModel`, `NightBatchDeweightingModel`, `CompositeModel`, `IdentityModel`) | one Rust crossing per `apply` (`bias_table_model_apply`, `night_batch_deweighting_model_apply`); `validate_bias_table` is pyarrow schema work | Rust unit tests, Python unit tests, Python-baseline parity fixture (positions provably unchanged) |
| `EFCC18DebiasModel`, VFC2017 `VeresFloorModel` / `VeresReplaceModel` / `SigmaFillModel`, `load_sigma_table` | Rust-owned lookup and arithmetic (`efcc18_debias_model_apply`, `veres_model_apply`, `veres_sigma_lookup`); the sigma tables are data of the private `adam-observatory-uncertainties` package resolved by soft import (nothing bundled) | RING tile-index parity vs JPL `tiles.dat`, healpy oracle test, Python-baseline parity fixture |
| `OrbitDeterminationObservations.from_ades` / `astcat` | Rust `ades_angular_covariance` kernel; table assembly and `Observers.from_codes` crossing in Python | unit tests, parity fixture |
| `fit_least_squares` (whitened residuals, dynamics-consistent Jacobians, Huber loss, weak-direction probe), `iterative_fit` | complete adam-core Rust solvers; concrete Rust propagators compose generically and Python providers cross only the v2 propagation capsule | active native solver/covariance/Huber tests and Python public-API tests |
| `cmc2003_fit`, `cmc2003_fit_detailed` | complete adam-core Rust reject/re-include loop over core-owned refits | hand-built rule tests, outlier-injection integration tests, native loop tests |
| `run_od`, `apply_observation_models`, `attach_observation_provenance`, `ObservationAstrometry`, `OrbitFitter.refine_fit` / `full_od`, `NativeOrbitFitter` | adam-core owns model/fitter orchestration and native OD drivers; the `OrbitFitter` plugin boundary remains public, while propagation providers expose no OD work units | orchestration tests, provenance tests, end-to-end observation-model tests |

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
`approxLangrangeCoeffs` is an intentionally retired Gauss implementation helper;
the public Gauss solve is Rust-backed. The former generic propagator composition
helpers (`propagation_worker*`, `ephemeris_worker_ray`, and
`attach_magnitude_or_phase`) are likewise retired because concrete backends own
those complete crossings. Abstract propagators, explicitly supplied backends,
and optional SPK propagation are deliberate provider boundaries rather than
default Python implementations.
