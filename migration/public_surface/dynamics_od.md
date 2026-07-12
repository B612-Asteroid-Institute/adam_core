# Dynamics, propagation, missions, impacts, OD, and IOD public-surface audit

**Audit date:** 2026-07-10  
**Parent bead:** `personal-cmy.37.3`  
**Audited tree:** `src/adam_core/{dynamics,propagator,missions,orbit_determination}` at `f27370b2`

## Scope and accounting rules

This is a source inventory, not the selected parity registry. It includes every non-underscore module-level function/class, custom public class method, package re-export, and relevant constructor/protocol behavior in the named packages. Public names that are only reachable from their defining module are still counted. Private helpers and tests are not counted. Plotting/display helpers are listed but are exempt from Rust implementation, parity, and native timing.

Status vocabulary:

- **Rust-1x**: compatible Python veneer around one Rust call/Arrow round trip; Rust owns the substantive work.
- **Rust kernel**: substantive calculation is Rust, but small Python input/output shaping remains.
- **Mixed/Python**: substantive public behavior or orchestration remains in Python.
- **Contract**: abstract interface/type alias; implementation evidence belongs to concrete backends.
- **Data model**: declarative quivr/dataclass/error shape. Generic inherited quivr operations (`from_kwargs`, `from_pyarrow`, `empty`, slicing, selection, serialization, etc.) are infrastructure, not silently claimed as adam-core Rust work.
- **Parity selected** means randomized baseline-main parity exists in `migration/parity`; ordinary unit tests do not count.
- **Native timed** means `std::time::Instant` surrounds a direct Rust call. Python/PyO3 timings do not count.

The selected registry currently covers only 19 rows in these domains (including four raw-kernel rows and two covariance lanes that are not separate Python APIs). It is therefore not complete-public-surface governance.

## Summary

| Domain | Current conclusion |
|---|---|
| Dynamics / two-body | Production propagation, ephemeris, MOID, Lambert, and most scalar kernels are Rust-backed. Only a subset has parity, and direct-Rust timing is much narrower. `calculate_c3` and impact post-processing remain Python. |
| Propagator contracts | The base classes are now contracts rather than Python composition. Concrete parity/timing must be proven by each backend; adam-assist is the currently relevant backend. Two compatibility utilities still perform Python table orchestration. |
| Missions | `generate_porkchop_data` is Rust-1x and governed. `LambertSolutions` accessors, departure-direction conversion, and body preparation/propagation remain Python. Plot/color helpers are exempt. |
| Impacts | Collision detection is only an abstract backend contract. Variant creation, collision dispatch, probability reduction, linking, and Mahalanobis post-processing remain Python and have no selected parity/native timing. |
| OD / IOD | Gibbs/Herrick-Gibbs/Gauss and `gaussIOD` are Rust kernels; full OD/IOD orchestration is still Python/Ray/SciPy with repeated backend crossings. A conditional native least-squares backend hook exists, but the public orchestration and fallback are not Rust-1x. |
| Direct-Rust timing | Present only for `propagate_2body`, both ephemeris lanes, `calculate_perturber_moids`, `generate_porkchop_data`, and `gaussIOD`. Every other qualifying Rust-backed API is missing a native adapter. |

## Dynamics public inventory

| Public surface | Implementation | Selected parity | Direct-Rust timing | Disposition / gap |
|---|---|---:|---:|---|
| `dynamics.generate_ephemeris_2body` | Rust-1x Arrow facade | yes (state + covariance lanes) | yes | Complete for current contract. |
| `dynamics.propagate_2body` | Rust-1x Arrow facade | yes | yes | Complete for current contract. |
| `DynamicsNumericalError(stage, reason, context)` and `__str__` | Python error/value veneer | no | N/A | Retain; behavior needs compatibility tests, not native timing. |
| `add_light_time` | Rust kernel | yes | no | Native timing gap. |
| `add_stellar_aberration` | Rust kernel | no | no | Parity + native timing gap. |
| `solve_barker` | Rust kernel | no | no | Parity + native timing gap. |
| `ChiDiagnostics(...)` | Python frozen diagnostic value | no | N/A | Retain as veneer. |
| `calc_chi` | Rust kernel | no | no | Parity + native timing gap. |
| `calc_chi_diagnostics` | one Rust chi call plus Python norms/diagnostic assembly | no | no | Acceptable 1x veneer; parity + native timing gap. |
| `calc_period` | Rust kernel | no | no | Parity + native timing gap. |
| `calc_periapsis_distance` | Rust kernel | no | no | Parity + native timing gap. |
| `calc_apoapsis_distance` | Rust kernel | no | no | Parity + native timing gap. |
| `calc_semi_major_axis` | Rust kernel | no | no | Parity + native timing gap. |
| `calc_semi_latus_rectum` | Rust kernel | no | no | Parity + native timing gap. |
| `calc_mean_motion` | Rust kernel | yes | no | Native timing gap. |
| `calc_mean_anomaly` | Rust kernel | no | no | Parity + native timing gap. |
| `solve_kepler` | Rust kernel | no | no | Parity + native timing gap. |
| `calc_lagrange_coefficients` | Rust kernel | no | no | Parity + native timing gap. |
| `apply_lagrange_coefficients` | Rust kernel | no | no | Parity + native timing gap. |
| `solve_lambert` | Rust kernel | yes | no | Native timing gap. |
| `calculate_c3` | NumPy subtraction/norm | covered only inside porkchop facade | no | Direct implementation + standalone parity/timing gap. |
| `PerturberMOIDs(...)` | quivr schema; generic inherited table API | indirectly | N/A | Data-model classification; no custom methods. |
| `calculate_moid` | Rust kernel with Python `Timestamp` assembly | yes | no | Native timing gap. |
| `calculate_perturber_moids` | Rust-1x per public call after SPICE setup | yes | yes | Complete for current contract. |
| `calc_stumpff` | Rust kernel | no | no | Parity + native timing gap. |
| `calc_tisserand_parameter` | Rust kernel | yes as `dynamics.tisserand_parameter` | no | Native timing gap; registry ID should preserve the actual public name in future complete inventory. |
| `ImpactProbabilities(...)` | quivr schema | no | N/A | Data-model classification. |
| `CollisionConditions(...)`, `.default()` | quivr schema; Python default-row assembly | no | no | Default constructor parity/implementation gap is grouped with impacts. |
| `CollisionEvent(...)`, `.preview()` | quivr schema; plotting preview | no | N/A | Schema retained; `preview` is plotting-exempt. |
| `ImpactMixin.detect_collisions` | abstract single-crossing contract | frozen legacy/downstream parity | Rust-owned downstream timing | Complete as a provider contract: `adam_assist.ASSISTPropagator` delegates directly to compiled `NativeAssistPropagator.detect_collisions`; impact artifacts and the 26-lane ASSIST timing governance cover the concrete backend. |
| `calculate_impacts` | Python variant sampling + backend collision call | no | no | Not 1x; implementation/parity/native-timing gap. |
| `calculate_impact_probabilities` | Python loops, Arrow selection, NumPy reductions, row assembly | no | no | Implementation/parity/native-timing gap. |
| `link_impacting_variants` | Python quivr linkage assembly | no | no | Implementation/parity/native-timing gap. |
| `calculate_mahalanobis_distance` | Python composition over residuals + NumPy sqrt | no | no | Implementation/parity/native-timing gap. |
| `prepare_propagated_variants`, `generate_impact_visualization_data`, `create_sphere`, `add_earth`, `add_moon`, `plot_impact_simulation`, `plot_risk_corridor` | plotting/display | no | N/A | Explicitly exempt. |

`dynamics.__init__` re-exports only `generate_ephemeris_2body`, `propagate_2body`, and `DynamicsNumericalError`; the remaining names are public at their module paths.

## Propagator contracts and utilities

| Public surface | Implementation | Parity / native timing | Disposition / gap |
|---|---|---|---|
| `EphemerisMixin()` / `generate_ephemeris(...)` | abstract backend contract | frozen legacy/downstream parity plus Rust-owned timing | Complete as a provider contract: `adam_assist.ASSISTPropagator` delegates directly to compiled `NativeAssistPropagator.generate_ephemeris`; governed fixtures cover mixed observers and UTC output. |
| `Propagator()` / `propagate_orbits(...)` | abstract backend contract | frozen legacy/downstream parity plus Rust-owned timing | Complete as a provider contract: all four governed propagation cases delegate to compiled `NativeAssistPropagator.propagate_orbits`. |
| `TimestampType`, `OrbitType`, `EphemerisType`, `ObserverType` | typing aliases | N/A | Retain. Package exports omit `TimestampType` and `ObserverType`, but they remain public from `propagator.types`/`propagator.propagator`. |
| `ensure_input_time_scale` | one-crossing veneer over Rust `TimeArray` rescaling followed by quivr wrapping | frozen pinned-main parity and `benchmark_ensure_input_time_scale` | Complete for Rust-supported scales; the public Timestamp UT1 provider boundary remains Astropy-owned by design. |
| `ensure_input_origin_and_frame` | one Arrow crossing owns first-appearance mixed-origin grouping, result-row membership/order, frame/origin transforms, and coordinate assembly | frozen pinned-main parity and `benchmark_ensure_input_origin_and_frame_arrow` | Complete; Python only wraps the returned coordinate batch and row indices. |

The base-class constructors have no custom behavior. Abstract method signatures, accepted covariance methods, stable ordering, and one-crossing semantics are part of the public contract even though no computation can be timed on an abstract class.

## Missions public inventory

| Public surface | Implementation | Selected parity | Direct-Rust timing | Disposition / gap |
|---|---|---:|---:|---|
| `LambertSolutions(...)` | quivr schema | via generator only | N/A | Data-model classification. |
| `.departure_body_orbit()` | Python table reconstruction | no | no | Implementation/parity/native-timing gap. |
| `.arrival_body_orbit()` | Python table reconstruction | no | no | Same. |
| `.solution_departure_orbit()` | Python IDs + table reconstruction | no | no | Same. |
| `.solution_arrival_orbit()` | Python IDs + table reconstruction | no | no | Same. |
| `.c3_departure()` / `.c3_arrival()` | Python extraction + `calculate_c3` | only indirectly | no | Direct implementation + standalone parity/timing gap. |
| `.vinf_departure()` / `.vinf_arrival()` | Python extraction + NumPy norm | only indirectly | no | Direct implementation + standalone parity/timing gap. |
| `.time_of_flight()` | Python timestamp conversion/subtraction | only indirectly | no | Direct implementation + standalone parity/timing gap. |
| `departure_spherical_coordinates` | Rust-1x Arrow facade owns normalization, Cartesian-to-spherical/frame transform, metadata, and output assembly; Python preserves assertion/warning compatibility and wraps the batch | frozen pinned-main parity plus behavior suite | yes | Complete; pinned cases cover Earth/ecliptic and Mars/equatorial vectors, values, time, origin, and frame metadata. |
| `prepare_and_propagate_orbits` | Rust-1x major-body product; for `Orbits`, one Rust preprocessing crossing followed by the explicit `Propagator.propagate_orbits` provider boundary | frozen pinned-main parity for both branches | yes for the complete major-body product; concrete provider timing governs the provider branch | Complete under the provider-boundary policy. Rust owns scale conversion, NumPy-compatible grid semantics, input transform, SPICE state lookup, units, and batch assembly; Python only dispatches the declared body/provider union and wraps products. |
| `generate_porkchop_data` | Rust-1x Arrow facade | yes | yes | Complete for current contract. |
| `generate_saturated_colorscale`, `generate_perceptual_colorscale`, `plot_porkchop_plotly` | plotting/display | no | N/A | Explicitly exempt. |

`missions.__init__` exports nothing; all names are public from `missions.porkchop`.

## OD data models and utility surfaces

All five quivr classes below have public declarative constructors and inherited generic table operations. Those inherited operations are classified as quivr infrastructure; custom adam-core behavior is inventoried separately.

| Public surface | Implementation | Selected parity | Direct-Rust timing | Disposition / gap |
|---|---|---:|---:|---|
| `OrbitDeterminationPhotometry(...)` | quivr schema | no | N/A | Data model. |
| `OrbitDeterminationObservations(...)` | quivr schema | no | N/A | Data model. |
| `FittedOrbits(...)` | quivr schema | no | N/A | Data model. |
| `FittedOrbits.to_orbits()` | Python table reconstruction | no | no | Implementation/parity/native-timing gap. |
| `FittedOrbitMembers(...)` | quivr schema | no | N/A | Data model. |
| `assign_duplicate_observations` | Python/Arrow loops and filtering | no | no | Implementation/parity/native-timing gap. |
| `drop_duplicate_orbits` | Python quivr/Arrow composition | no | no | Implementation/parity/native-timing gap. |
| `calculate_max_outliers` | scalar Python policy | no | no | Implementation/parity/native-timing gap (small but public and non-plotting). |
| `remove_lowest_probability_observation` | Python/Arrow reduction and filtering | no | no | Implementation/parity/native-timing gap. |

## OD and IOD kernels and orchestration

| Public surface | Implementation | Selected parity | Direct-Rust timing | Disposition / gap |
|---|---|---:|---:|---|
| `calcGibbs` | Rust kernel | yes | no | Native timing gap. |
| `calcHerrickGibbs` | Rust kernel | yes | no | Native timing gap. |
| `calcGauss` | Rust kernel | yes | no | Native timing gap. |
| `gaussIOD` | Rust-1x Arrow facade | yes | yes | Complete in the governed shared-root regime; unconstrained multi-root subset equivalence remains explicitly excluded by current tolerance policy. |
| `residual_function` | Python orbit assembly + backend ephemeris + residual reduction | no | no | Multi-crossing implementation/parity/native-timing gap. |
| `fit_least_squares` | conditional backend-native hook, otherwise SciPy/Python iterative fallback; Python evaluation/assembly on both paths | no | no | Public API is not wholly Rust-1x; finish native orchestration and govern both dispatch semantics and results. |
| `evaluate_orbits` | explicit `Propagator.generate_ephemeris` provider boundary followed by one Rust crossing for stable order validation, residuals, ignore masks, statistics, arc length, and member indexing; Python only wraps the returned arrays as quivr tables | yes | yes | Complete. Frozen pinned-main fixtures cover normal, ignored, empty, and malformed-order cases; reordered equal-length provider output now raises the documented stable-order error intentionally. |
| `LeastSquares(use_central_difference)` | Python stateful algorithm class | no | no | Constructor semantics need parity with a Rust-owned implementation. |
| `LeastSquares.least_squares` | repeated backend calls + Python finite differences/normal equations | no | no | Multi-crossing implementation/parity/native-timing gap. |
| `OrbitFitter()` / `initial_fit(...)` | abstract contract | downstream only | downstream only | Govern concrete backend implementations. |
| `OrbitFitter.__getstate__` / `__setstate__` | public serialization protocol that raises until overridden | unit behavior only | N/A | Retain contract behavior; no native timing qualification. |
| `od_worker` | Python per-orbit indexing/loop and calls to `od` | no | no | Orchestration gap. |
| `od` | Python/SciPy iterative finite differences, repeated ephemeris crossings, outlier policy | no | no | Core OD Rust-1x gap. |
| `differential_correction` | Python/Ray chunking, object-store handling, worker dispatch, concatenation | no | no | Top-level OD Rust-1x/parity/native-timing gap. |
| `sort_by_id_and_time` | Python/Arrow joins/sorts | no | no | IOD utility implementation/parity/native-timing gap. |
| `select_observations` | Python combinations/percentiles/sorting | no | no | IOD utility implementation/parity/native-timing gap. |
| `iod_worker` | Python per-linkage loop/indexing and calls to `iod` | no | no | Orchestration gap. |
| `iod` | Python candidate selection, Rust `gaussIOD`, repeated backend ephemeris calls, residual/outlier acceptance | no | no | Core IOD remains multi-crossing Python; implementation/parity/native-timing gap. |
| `initial_orbit_determination` | Python/Ray chunking, object-store handling, workers, deduplication, sorting | no | no | Top-level IOD Rust-1x/parity/native-timing gap. |

Package-level `orbit_determination` re-exports `fit_least_squares`, `OrbitDeterminationObservations`, `evaluate_orbits`, `FittedOrbitMembers`, `FittedOrbits`, `drop_duplicate_orbits`, `gaussIOD`, `calcGibbs`, `calcHerrickGibbs`, `initial_orbit_determination`, `iod`, `select_observations`, `sort_by_id_and_time`, `calculate_max_outliers`, and `remove_lowest_probability_observation`. Other names remain public at module paths. `differential_correction` and `initial_orbit_determination` are the intended top-level orchestration APIs (`__all__` in their modules).

## Direct-Rust timing status

The current native adapter map in `migration/parity/_native_rust_runner.py` contains only:

- `dynamics.propagate_2body`
- `dynamics.generate_ephemeris_2body` (also used for the covariance lane)
- `dynamics.calculate_perturber_moids`
- `dynamics.generate_porkchop_data`
- `orbit_determination.gaussIOD`
- `evaluate_orbits` through `benchmark_evaluate_orbits_numpy` (outside the selected 44-API registry)
- `ensure_input_time_scale` and `ensure_input_origin_and_frame` through their dedicated Rust-owned benchmarks (outside the selected registry)
- all concrete `adam_assist.ASSISTPropagator` propagation, ephemeris, covariance, and collision lanes through the downstream 26-lane Rust timing contract

Consequently, current report rows for `add_light_time`, `calc_mean_motion`, `calculate_moid`, `solve_lambert`, Tisserand, raw propagation/MOID/porkchop kernels, `calcGibbs`, `calcHerrickGibbs`, and `calcGauss` explicitly show missing native samples. Rust-backed public helpers absent from the selected registry have no native-timing row at all. Direct timing for Python-owned OD/IOD/impact/mission utilities is blocked until a qualifying direct Rust entrypoint exists.

## Required follow-up beads

Children of `personal-cmy.37.3` cover every non-plotting gap above, grouped only where surfaces share one implementation boundary:

| Child | Boundary |
|---|---|
| `personal-cmy.37.3.1` | `calculate_c3` implementation plus missing scalar dynamics parity/native timing |
| `personal-cmy.37.3.2` | Impact defaults, orchestration, probability reduction, linkage, and Mahalanobis distance |
| `personal-cmy.37.3.3` | Propagator normalization utilities and concrete backend contract evidence (complete: one-crossing utilities, frozen parity, native timing, and compiled ASSIST delegation evidence) |
| `personal-cmy.37.3.4` | All custom `LambertSolutions` methods |
| `personal-cmy.37.3.5` | Mission departure-direction and body preparation/propagation orchestration (complete: Rust-owned products around the explicit propagator boundary, frozen parity, and native timing) |
| `personal-cmy.37.3.6` | Fitted-orbit conversion/deduplication and outlier utilities |
| `personal-cmy.37.3.7` | `evaluate_orbits` one-crossing orchestration (complete: frozen parity and Rust-owned timing) |
| `personal-cmy.37.3.8` | `residual_function` and `fit_least_squares` native orchestration |
| `personal-cmy.37.3.9` | `LeastSquares` public algorithm |
| `personal-cmy.37.3.10` | `od_worker`, `od`, and `differential_correction` |
| `personal-cmy.37.3.11` | IOD sorting and observation-selection utilities |
| `personal-cmy.37.3.12` | `iod_worker`, `iod`, and `initial_orbit_determination` |
| `personal-cmy.37.3.13` | Missing direct-Rust timing for already governed dynamics/OD kernels |

The child descriptions carry the exhaustive API lists and acceptance details. No shared parity-registry refactor is requested or authorized by this audit. Inventory review should precede any registry-governance redesign.
