# OD observation-uncertainty features on the Rust adam-core: handoff / PR description

Date: 2026-09-16
Branch: `od-obs-uncertainty-rust` (off `origin/main` @ `8fb48119`)
Bead: `od_experiments_setup-uzs`

## What this branch is

A **reimplementation, not a merge**, of the 19 Python OD commits on
`kk/obs-uncertainty-interface` (`d56114ac`, 423 commits behind `main`, merge
base `e087943e`) onto the Rust `main`, on top of its existing OD core. The
Python branch is the behavioral and API baseline; the Rust migration
(`adam_core_rust_migration_review_handoff_2026-04-27.md`) is the style and
code baseline. A Python caller sees the same public surface as on the Python
branch; every per-observation arithmetic runs in Rust.

## Rust OD core map: what `main` already had

| Layer | Already on `main` (`8fb48119`) | Added here |
|---|---|---|
| Data model (`adam_core_rs_coords::types`) | `CoordinateBatch` (spherical / Cartesian + covariance + times + origins), `CovarianceBatch`, `OrbitBatch`, `ObserverBatch`, `EphemerisBatch`, `TimeArray` / `Epoch`, nested quivr Arrow codecs | `OrbitDeterminationAstrometry` column view (lon, lat, `(N, 36)` covariance, station, band, astcat, UTC MJD, TDB JD) that the models edit in place |
| Observations | `AdesObservationBatch` (with `ast_cat`), PSD / exposures / associations / source-catalog batches, ADES and Obs80 parsers | `ades_angular_covariance_flat` (ADES rms -> `(N, 6, 6)` lon/lat covariance) |
| HEALPix | `healpix::ang2pix` — faithful healpix_cxx port, **RING and NEST**, nside validation | `efcc18::ra_dec_to_healpix` (nside 64, RING, the legacy theta/phi convention) + the JPL `tiles.dat` parity tests |
| Residuals | `compute_residuals_chi2_flat`, `chi2_survival`, `bound_longitude_value`, cos-lat correction, Cholesky chi2 | `observation_whitening_matrices`, `whiten_residual_pairs` (2N whitened components) |
| Fitter | `orbit_least_squares::fit_orbit_least_squares_with_predictor` — Gauss-Newton, **forward-difference** Jacobian, `inv(JᵀJ)`; `propagation::od` drivers (`fit_orbit_least_squares_evaluated_barycentric`, legacy `od_fit_barycentric`, Vallado, IOD) generic over the Rust `Propagator` trait | `whitened_2body_model_angles` / `whitened_2body_jacobian`: exact Jacobian of the whitened 2-body model over `Dual<6>` (the autodiff crate already made `propagate_2body_row` and `generate_ephemeris_2body_row` generic over `Scalar`); `robust_cost` / `robust_weights` / `robust_jacobian_scale` (Huber, scipy convention) |
| Outliers | worst-observation policy, `calculate_max_outliers` | `cmc2003_apparitions`, `cmc2003_expected_residual_chi2` (`I ∓ J C Jᵀ` with eigenvalue floor), `cmc2003_select` |
| Observation models | — | `ObservationUncertaintyModel` trait + `IdentityModel`, `EmpiricalCovarianceModel`, `PerformanceWeightedModel`, `SigmaFloorModel`, `NightBatchDeweightingModel`, `Efcc18DebiasModel`, `CompositeModel`; `BiasTable` lookup; `efcc18` parse / lookup / corrections / debias; `veres2017` generic `VeresSigmaLookup`, `VeresFloorModel`, `VeresReplaceModel`, `SigmaFillModel` (tables come from the data package) |
| PyO3 (`adam_core_py`) | `od_ops.rs`, `orbit_determination.rs` (Gauss IOD), `coordinate_ops::evaluate_orbits_numpy`, Arrow propagate / ephemeris | `observation_uncertainty.rs` (13 functions), `differential_correction.rs` (11 functions); registered in `lib.rs`, guarded in `_rust/api.py` |

New Rust files: `rust/adam_core_rs_coords/src/{efcc18,observation_uncertainty,veres2017,cmc2003,differential_correction}.rs`,
`rust/adam_core_py/src/{observation_uncertainty,differential_correction}.rs`. Existing files touched: `ades_io.rs` (one added kernel), both `lib.rs`.

## Python <-> Rust boundary

Rust owns every per-observation arithmetic and lookup and, since 2026-09-29
/ 2026-10-01 (see the follow-ups below), the orchestration too; Python owns
quivr tables, pyarrow schema validation, the `OrbitFitter` plugin boundary
and table assembly. The table below is the original (2026-09-16) boundary;
the "Stays Python" entries for the fit and the loops are superseded:

| Python surface (same names as the Python branch) | Rust kernel(s) behind it | Stays Python (why) |
|---|---|---|
| `OrbitDeterminationObservations.astcat`, `from_ades` | `ades_angular_covariance_numpy` | table assembly, `Observers.from_codes` (its own Rust crossing) |
| `EmpiricalCovarianceModel` / `PerformanceWeightedModel` / `SigmaFloorModel` | `bias_table_model_apply_numpy` | `validate_bias_table` (pyarrow cast), quivr `set_column` only when Rust reports a change |
| `NightBatchDeweightingModel` | `night_batch_deweighting_model_apply_numpy` | — |
| `EFCC18DebiasModel`, `observations.efcc18.*` | `efcc18_parse_bias_dat`, `efcc18_ra_dec_to_healpix_numpy`, `efcc18_catalog_columns_numpy`, `efcc18_corrections_numpy`, `efcc18_debias_model_apply_numpy`, `healpix_ang2pix_lonlat_numpy` | locating / downloading / checksumming / caching `bias.dat` (I/O) |
| `VeresFloorModel` / `VeresReplaceModel` / `SigmaFillModel` / `VeresSigmaLookup` / `load_sigma_table` | `veres_sigma_lookup_numpy`, `veres_model_apply_numpy` | `validate_veres_sigma_table` (pyarrow cast); default tables resolved by importing the private `observatory_uncertainties` data package (no table bundled, 2026-09-27) |
| `fit_least_squares` (whitened residuals, `jacobian="analytic"`, `loss="huber"`, weak-direction probe), `iterative_fit`, `residual_function` | `observation_whitening_matrices_numpy`, `whiten_residual_pairs_numpy`, `whitened_2body_jacobian_numpy`, `robust_*_numpy`, `validate_robust_loss` | `scipy.optimize.least_squares` (the grid-validated optimizer), N-body residuals through `propagator.generate_ephemeris`, the probe / central-difference fallback (they call the propagator) |
| `cmc2003_fit`, `cmc2003_fit_detailed` | `cmc2003_apparitions_numpy`, `cmc2003_expected_residual_chi2_numpy`, `cmc2003_select_numpy` | the refit loop (calls `fit_least_squares`) |
| `run_od`, `apply_observation_models`, `attach_observation_provenance`, `ObservationAstrometry`, `FittedOrbitMembers.{weight, original_astrometry, used_astrometry, astcat}` | `sqrt_values_numpy` | pyarrow `index_in` / `take` joins; drives any `OrbitFitter` (adam_fo overrides `full_od`) |
| `OrbitFitter.refine_fit` / `full_od`, `NativeOrbitFitter` (`loss`, `outlier_rejection="cmc2003"`) | via the functions above | plugin interface (flag-free by design) |
| `observatory_bias_model=` on `evaluate_orbits`, `fit_least_squares`, `iterative_fit`, `iod`, `initial_orbit_determination`, `od`, `differential_correction` | — | applied once at entry, before the Rust-backed pipelines |

## Deliberate divergences from the two baselines

* **`fit_least_squares` default path.** The branch's `jacobian="analytic"` /
  `validate_covariance=True` defaults are kept (they are the covariance fix).
  `main`'s fused Rust Gauss-Newton dispatch (`propagator.fit_least_squares_evaluated`
  / `fit_least_squares`) is therefore reached only with `jacobian="2-point"`,
  `loss="linear"` and no scipy kwargs: it uses the same forward-difference
  Jacobian whose covariance the analytic path exists to correct. Members from
  that path now also carry `weight` (1 / 0), and with `validate_covariance=True`
  (the default) the native covariance is put through the same weak-direction
  probe (three extra residual evaluations; a `RuntimeWarning` on failure, the
  fit unchanged) — pass `validate_covariance=False` for the single-crossing
  behaviour of `main`.
* **`od.differential_correction` is not deprecated** (the Python branch
  deprecated it in favor of `iterative_fit`); on `main` it is a Rust-backed
  supported entry point. It gained `observatory_bias_model` like the others.
* **`OrbitFitter.initial_fit` keeps `main`'s `reference_orbit` warm-start
  kwarg** (absent on the Python branch); `NativeOrbitFitter` honours it by
  evaluating the seed instead of running Gauss IOD.
* **No `OrbitDeterminationObservations` Rust batch type / nested Arrow codec.**
  The models cross on the numpy boundary (like `evaluate_orbits_numpy` and the
  rest of `od_ops.rs`), because `run_od` must drive Python `OrbitFitter`
  plugins anyway. An Arrow-native `OrbitDeterminationObservationBatch` is the
  natural next step if a Rust-side `run_od` is ever wanted.
* **healpy is no longer needed** for EFCC18 (the Rust `ang2pix` port is used);
  it stays an optional oracle in tests.
* Python-side test helpers use `main`'s `Propagator` contract
  (`propagate_orbits` + `generate_ephemeris` abstract, no Python
  `EphemerisMixin` composition); the synthetic two-body propagator strips orbit
  covariance before propagating, as a backend's `generate_ephemeris(covariance=False)` does.

## 2026-09-23 follow-up: OD defaults handoff

The Python baseline moved to `kk/obs-uncertainty-interface` @ `23d8bfc6`
(`bc727239` SigmaFillModel + station-only / global sigma-table rows, `23d8bfc6`
test follow-up); both are ported here: `VeresSigmaRow.astcat` is `Option`,
`VeresSigmaLookup` resolves (station, catalog) → station → catalog → global row
→ fallback, `SigmaFillModel` (Rust `veres2017::SigmaFillModel`, PyO3 model
`"fill"`) fills only missing per-axis variances and zeroes the cross-term where
it filled. The Python test module is the baseline's file verbatim.

The study's defaults handoff
(`adam_od_experiments/docs/OD_DEFAULTS_HANDOFF_rust_migration.md`) prescribes
the shipped stack: `SigmaFillModel(load_sigma_fill_table())` → `EFCC18DebiasModel`
(RING) → `EmpiricalCovarianceModel(bias_table, mode="add")` →
`NightBatchDeweightingModel(cap=4)` with `NativeOrbitFitter(outlier_rejection=
"cmc2003")`; not defaults: Huber, any Veres model, `SigmaFloorModel`,
`PerformanceWeightedModel`, `cap=1`. That stack is documented in
`docs/source/use_cases/orbit_determination.rst`. Class-level defaults were NOT
flipped, mirroring the Python baseline: `NativeOrbitFitter` still defaults to
`outlier_rejection="worst_residual"`, `NightBatchDeweightingModel` already
defaults to `cap=4`, `EmpiricalCovarianceModel` to `mode="add"`, and
`SigmaFillModel(sigma_table=None)` now (2026-09-27) resolves `v2_sigma_fill`
by importing the private `observatory_uncertainties` data package, and the
Veres models resolve its `veres2017_working` table; adam_core bundles no sigma
table and still does not depend on the package (ImportError names it when
missing). (Superseded 2026-09-28: the fitter default was flipped, see below.)

## 2026-09-28 follow-up: the defaults are the signature defaults (PR review)

Kathleen's decision: no configuration object, no factory. `NativeOrbitFitter()`
defaults to `outlier_rejection="cmc2003"`; `run_od()` called without `models`
builds the default stack inline (`SigmaFillModel()` → `EFCC18DebiasModel()` →
`EmpiricalCovarianceModel()` → `NightBatchDeweightingModel()`), each model's
no-argument constructor being its default; `models=None` opts out. The
bias-table models resolve `v2_full` from the data package when built without
a table (`load_bias_table`, like `load_sigma_table`); adam_core does not
depend on the package and the ImportError / FileNotFoundError names the
missing data. The lower-level entry points keep `observatory_bias_model=None`.
The defaults summary, including which default models import data packages,
is the `run_od` docstring (Notes), mirrored by the docs table "The shipped
defaults" and PR #217's "Defaults" section.

## 2026-09-29 follow-up: orchestration moved into Rust (Alec's review)

Alec: Python should be a surface-level interface, not do orchestration or
looping. Following main's own pattern (backend-generic drivers in
`propagation/od.rs` generic over the Rust `Propagator` trait, exposed by
Rust-backed propagators as one-crossing methods that the veneer dispatches
to), `propagation/od_refine.rs` now holds:

| Work unit | Rust driver | Propagator method | Two-body binding |
|---|---|---|---|
| `fit_least_squares` (whitened, analytic/central/2-point Jacobian, linear/Huber, validated covariance, fused evaluation) | `fit_orbit_whitened_barycentric` | `fit_least_squares_whitened(orbit, observations, ignore_mask, fit_settings=)` | `fit_orbit_whitened_2body_ipc` |
| `iterative_fit` (worst-residual loop) | `iterative_fit_barycentric` | `iterative_fit(orbit, observations, ..., fit_settings=)` | `iterative_fit_2body_ipc` |
| `cmc2003_fit_detailed` (CMC2003 loop) | `cmc2003_fit_barycentric` | `cmc2003_fit(orbit, observations, ..., fit_settings=)` | `cmc2003_fit_2body_ipc` |
| `NativeOrbitFitter.full_od` (Gauss IOD decision loop → refinement) | `full_od_barycentric` | `full_od(observations, iod_settings=, refinement=, fit_settings=)` | `full_od_2body_ipc` |
| `run_od` (models → full OD → provenance snapshots) | `run_od_barycentric` | `run_od(observations, models=[specs], ...)` | `run_od_2body_ipc` |

The solver is Levenberg-Marquardt with Marquardt scaling and scipy's
ftol/xtol/gtol rules (IRLS weights for Huber; scipy's curvature scaling only
for the covariance): same minimum to solver tolerance, not bit-identical
iterates. Python keeps: ids, the `OrbitFitter` plugin boundary, table
loading (data packages), table assembly, and the fallback loops for
propagators without the work units (a user-defined Python model keeps the
Python model composition). adam-assist implements the five methods in a
follow-up by calling the drivers (it already does so for
`fit_least_squares_evaluated`, `od_fit`, `initial_orbit_determination`).

Parity (Python parity tests on the two-body fixture): fused vs scipy fit
state 1e-9 / covariance 1e-6 from a shared start; loops: identical outlier
sets, weights and flags; composed `run_od`: state 1e-7 (different IOD seeds
along the flat along-track valley), provenance snapshots 1e-12. Rust unit
tests cover truth recovery, finite-difference agreement, ignore masks, Huber
downweighting, the probe fallback, both loops, full OD and run_od.

## 2026-10-01 follow-up: Python is a facade (the callback route)

Alec, in person: every high-level OD function must be callable from Rust
(precovery must not go through Python), Rust first with a Python facade, and
both should keep the right error types. The 2026-09-29 drivers covered the
Rust-callable part for Rust propagators; this slice finishes the facade:

* `SphericalPredictor` (`propagation/od.rs`): the one operation the drivers
  need from a backend (predicted spherical coordinates for candidate states).
  `PropagatorPredictor` implements it for any Rust `Propagator`; the drivers
  are now `*_with(predictor, ...)` with the `*_barycentric` signatures kept as
  wrappers. `iod_fit` and `evaluate_orbit` are predictor-based too.
* `rust/adam_core_py/src/od_callback.rs`: `PyEphemerisPredictor` wraps any
  Python propagator (anything with `generate_ephemeris`) as a predictor; the
  Rust loop calls it with the GIL re-acquired per prediction (candidates and
  observers cross as nested Arrow IPC, predictions return as an `(M·N, 6)`
  array, `adam_core.orbit_determination._native_callback`). The SPICE backend
  lock is taken per translation call (`LockingSpiceTranslation`), never
  across a callback. A Python exception raised inside the callback is stored
  and re-raised unchanged; other driver errors map to `ValueError`
  (invalid input) / `RuntimeError` (backend failure).
* Bindings `fit_orbit_whitened_ipc`, `iterative_fit_ipc`, `cmc2003_fit_ipc`,
  `iod_fit_ipc`, `full_od_ipc`, `run_od_ipc`, `validate_fit_covariance_ipc`
  take the propagator object (`None` = two-body); the `*_2body_ipc` names
  remain as the fused in-tree route.
* Python: `fit_least_squares` (scipy optimizer deleted; `xtol`/`ftol`/`gtol`/
  `max_nfev` only, others `ValueError`), `iterative_fit`, `cmc2003_fit_detailed`
  and `iod` are convert → call → convert, dispatching to the propagator's
  fused work unit when present and to the callback route otherwise; the
  probe helpers `_weak_direction_delta_chi2` / `_validated_covariance` keep
  their signatures over the Rust probe. `NativeOrbitFitter.full_od` is the
  fused `full_od` or `initial_fit` + `refine_fit` (two native crossings);
  `run_od` is the fused `run_od` or models + `full_od` + provenance join.
  Duck-typed test propagators that only implement `generate_ephemeris` keep
  working (the callback calls once per candidate when a propagator cannot
  batch).
* Still Python, deliberately: `od.differential_correction` and the public
  `LeastSquares` (main's legacy loops with their own fused routes), the
  multi-linkage Ray driver `initial_orbit_determination` (it calls `iod`
  per linkage, now native), model table loading, ids and table assembly.
* Not yet: Rust-side defaults for `IodConfig` / `FullOdConfig` and table
  readers for the data packages (a Rust caller reconstructs the shipped
  configuration by hand); adam-assist's five fused methods (separate PR,
  needs a core release that contains the drivers and a pin bump from rc.5).

## 2026-10-07 follow-up: Arrow slice-offset bug mitigated

Found 2026-10-01 while routing OD through the callback: `propagate_2body(
orbits[1:2], times)` propagated row 0. The Python `RecordBatch` carries the
slice offset correctly; the Rust nested-schema decoders (`OrbitBatch`,
`ObserverBatch`, ...) read struct children from the start of the shared
buffers. Flat primitive columns (the day/nanos time batches, the coordinate
value columns of `transform_coordinates`) honour offsets, and adam-assist
marshals through numpy, so ASSIST propagation / ephemeris / fused OD were never
exposed. Fix: `adam_core._rust.arrow.contiguous_record_batch` rebuilds every
offset array with `pa.concat_arrays` and is now the input half of every
table-shaped crossing (`arrow_bridge._to_record_batch`,
`transform._coordinate_record_batch`, `Trajectory._native_batch`,
`LambertSolutions._record_batch`). The Rust decoders are unchanged; a Rust-side
fix (apply `ArrayData::offset()` in the nested decode) would make the Python
step redundant. The former strict-xfail pins are now regression tests.

## Correctness gates

* **HEALPix RING order.** `efcc18::tests::ring_order_matches_jpl_tiles_dat`
  pins the 8 published `tiles.dat` anchors (and asserts the nested scheme does
  not reproduce them); `ring_order_matches_full_jpl_tiles_dat_when_available`
  reproduced **all 49152** tile centres of JPL's `tiles.dat` (run locally with
  `ADAM_CORE_EFCC18_TILES_DAT`). The Python test `test_efcc18.py` adds a
  20 000-point healpy ring oracle when healpy is installed.
* **Behavioral parity to the grid-validated Python.** The Python branch was run
  in its own venv on synthetic inputs; inputs and products are pinned in
  `migration/artifacts/od_python_baseline_parity_fixture_2026-09-16.json` and
  replayed by `test_od_python_baseline_parity.py` (169 checks, 0 mismatches):
  `from_ades` positions / covariance exact; EFCC18 tiles and corrections on the
  published `bias.dat` exact; every sigma-interpreter covariance block exact
  with positions provably unchanged; whitening factors 1e-9, analytic Jacobian
  6e-13 relative; fitted states 2e-11 relative, covariances 1.3e-10 relative,
  Huber weights 8e-9; CMC2003 rejection set identical; `run_od` used
  astrometry exact, fitted orbit at the truth epoch to 8e-12 AU.
* **Position rules.** EFCC18 moves lon/lat and never the covariance; every
  other model leaves lon/lat bit-identical (asserted in Rust and Python tests
  and in the parity fixture).

## Validation run (2026-09-16)

* `cargo +1.87.0 fmt --all --check`, `cargo +1.87.0 clippy --workspace --all-targets -- -D warnings`, `cargo +1.87.0 test --workspace` (CI-pinned toolchain): green; 232 coords-crate tests incl. 27 new.
* `pytest src/adam_core/orbit_determination src/adam_core/observations/tests/test_efcc18.py ...`: 245 passed, 14 skipped (adam_assist / OORB / pin data not installed).
* Governance: `migration/public_surface/manifest.json` regenerated (707 symbols, 0 unreviewed); `ruff` / `black` / `isort` clean.
* Left as on `main`: `mypy --strict` reports the pre-existing `Module "adam_core" has no attribute "_rust_native"` pattern; two clippy `needless_late_init` findings in unrelated files fire only under a newer stable clippy (not 1.87.0).

## Not ported / follow-ups

* (Done 2026-09-29 / 2026-10-01.) The whitened / Huber fit, the loops and
  the full OD are Rust drivers; the scipy path is gone.
* No Rust-owned `bias.dat` acquisition (the kernel-data crate pattern would fit).
* Benchmark (`benchmark_*`) twins and `_rust/status.py` registry rows were not
  added for the new kernels (they are new surface, not migrations of legacy
  APIs).
