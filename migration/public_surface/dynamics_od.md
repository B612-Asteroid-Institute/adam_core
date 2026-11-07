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
| historical Ray workers and scalar orchestration helpers (`*_worker_ray`, `moid_worker`, `lambert_worker`, scalar MOID internals) | intentionally retired implementation details; fused Rust batch entrypoints own the same useful workflows; obsolete MOID process/chunk and solver controls are removed rather than shimmed | all retired symbol IDs are recorded in the manifest; no-Ray imports, public workflow parity, API-contract tests, and native timing |

## Orbit determination and IOD

| Surface group | Disposition | Evidence |
|---|---|---|
| Gibbs, Herrick-Gibbs, Gauss roots/candidates | direct Rust kernels; candidate priority/order preserved; a supplied central-body `mu` consistently governs geometry and velocity rather than reproducing the legacy split-`mu` bug | fixed/random parity, custom-`mu` regression, and timing |
| least-squares fitter, differential correction, `evaluate_orbits` | adam-core-owned backend-generic Rust algorithms; Python preserves public tables/errors; native providers supply propagation capsules only and unsupported providers fail clearly | active Rust solver/covariance tests, ignored-observation/order/statistics tests, timing |
| `iod_worker`, linkage IOD, and `initial_orbit_determination` | adam-core-owned Rust linkage orchestration through explicit stage propagation capsules; Python supplies public table IDs only | full-linkage fixture, root/order tests, native capsule integration, timing |
| top-level OD batch and scheduling parameters | adam-core Rust scheduling; historical Ray parameters are signature-compatible no-ops; `iod_worker_remote` and `od_worker_remote` are retired | serial/parallel parity and no-Ray import tests |

## Core 0.7 OD configuration and observation pipeline

The September 2026 executable-model/plugin design is retired. Core Rust now
owns a closed ordered `ObservationPipeline`, complete stage configuration,
resource validation/parsing, fitting, rejection, covariance, and predictor-level
configured composition. Python exposes immutable declarations, binds explicit
local paths by semantic slot, marshals tables, and assembles result tables.

| Surface group | Disposition | Evidence |
|---|---|---|
| `FillMissingAstrometricUncertainty`, `Efcc18CatalogBiasCorrection`, `StationBandResidualCovarianceAdjustment`, `NightlyBatchDeweighting` | closed Rust variants; order and repetition preserved; no callbacks or transform plugin ABI | native transform/resource tests and exact resource parity |
| `station_catalog_uncertainty_table`, `efcc18_catalog_bias_table`, `station_band_residual_covariance_table` | explicit content-pinned semantic slots; no Core package discovery, fallback, or runtime download | aggregate hash/schema/row/structure validation with path-free provenance |
| `OdConfig`, stage configs, `CORE_OD_V1`, `ASTEROID_INSTITUTE_OD_V1` | immutable Rust definitions mirrored in Python; profiles never select propagators | strict native round-trip, profile/explicit parity, fingerprint tests |
| `fit_least_squares`, `iterative_fit`, `cmc2003_fit_detailed` | Core-owned Rust numerical algorithms; propagation providers expose only the negotiated capsule | native solver/covariance/rejection tests |
| canonical batch `run_iod`, `run_od`, `refine`, `evaluate_orbits` | Rust-owned deterministic Associations/FittedOrbitMembers grouping and typed failures; Python only marshals and assembles the three direct result tables | native planner/workflow, public-surface, and migration contract tests |

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

The generated manifest remains authoritative: canonical `refine` and
`evaluate_orbits` are classified `mixed_rust_python_gap` because Python still
assembles Rust-decided typed columns into quivr tables and native timing is not
yet represented as a direct Arrow result. This document does not claim that gap
is closed. `approxLangrangeCoeffs` is an intentionally retired Gauss
implementation helper;
the public Gauss solve is Rust-backed. The former generic propagator composition
helpers (`propagation_worker*`, `ephemeris_worker_ray`, and
`attach_magnitude_or_phase`) are likewise retired because concrete backends own
those complete crossings. Abstract propagators, explicitly supplied backends,
and optional SPK propagation are deliberate provider boundaries rather than
default Python implementations.
