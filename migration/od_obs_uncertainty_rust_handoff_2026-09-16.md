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

## 2026-10-12 follow-up: review round 2, phase 1

Bead `od_experiments_setup-5jq`, plan `migration/pr217_review2_plan_2026-10-12.md`
section 3 phase 1, brief `migration/agent_briefs/phase1_rust_2026-10-12.md`.
Four correctness fixes, no contract change (public signatures and existing
dict keys unchanged; keys and `IodOutput` fields were added).

1. **CMC2003 statistic never makes a bad row look acceptable**
   (`cmc2003.rs` `cmc2003_expected_residual_chi2`, `cmc2003_select`). A
   non-finite statistic (NaN residual, NaN Jacobian rows, overflow) is
   `+inf` with `NonFiniteChi2`; a singular expected covariance is `+inf`
   with `SingularResidualCovariance` unless both residual components are
   exactly zero (then `0`). Both used to be `0.0`, below `chi2_recover`.
   `cmc2003_select`: when the worst selected chi2 is infinite the batch
   threshold is the infinite value itself (`inf * chi2_frac` would be NaN
   for `chi2_frac == 0`), so that pass rejects exactly the infinite rows;
   sorting already used `total_cmp`; the minimum-sample, rejected-fraction
   and apparition guards are unchanged and still apply to infinite rows
   (an infinite row that is the last of its apparition is kept and
   flagged, as before). Rule documented in the Rust doc comments and the
   `rejection.py` module / `_expected_residual_chi2` docstrings (the rst
   use case lists only the constants, so it was left alone). Tests:
   `cmc2003::tests::{non_finite_statistic_is_infinite_rejected_and_never_recovered,
   infinite_worst_row_with_zero_chi2_frac_is_still_rejected,
   singular_expected_covariance_rejects_nonzero_and_keeps_zero_residuals}`;
   `tests/test_cmc2003_guards.py` (kernel pins + `cmc2003_fit_detailed` on
   the two-body callback route with one NaN-position observation: flagged,
   rejected, truth recovered).

2. **The covariance probe does not swallow backend errors** (`od_refine.rs`
   `validated_covariance`, `validate_fit_covariance_with`). The outer
   `PropagationError` of `weak_direction_delta_cost` now propagates with `?`
   (previously `.unwrap_or_else(|err| Err(err.to_string()))` turned it into
   a warning and returned the unvalidated covariance; on the callback route
   the next prediction then cleared the stored Python exception). The
   in-band `Err(String)` outcomes (no positive eigenvalue, rejected trial
   prediction, non-finite probe residuals) remain warnings. The CMC
   finalisation path reaches the probe through `fit_orbit_whitened_impl`
   and already propagated with `?`. Tests:
   `od_refine::tests::probe_backend_error_propagates_and_in_band_failure_warns`
   (a `SphericalPredictor` double failing only on the two-candidate probe
   batch: `fit_orbit_whitened_with` and `validate_fit_covariance_with`
   return `Err`, the in-band variant warns); `tests/test_probe_errors.py`
   (a `TwoBodyPropagator` subclass raising only on the probe batch:
   `fit_least_squares` raises that exception type, no warning; a subclass
   raising on its second call, i.e. a solver step, does the same; a clean
   control fit). Deviation from the brief: the Python probe double keys on
   the probe's two-candidate batch rather than "its second call", because
   on the analytic route the second prediction is a solver trial step (that
   case is covered by the second test and already propagated).

3. **Fused `run_od` runs the models before requiring a covariance**
   (`od_refine.rs` `run_od_with`, `od.rs` `observed_covariance_flat`).
   `run_od_with` called `observed_covariance_flat` (which errors on a
   `None` covariance) before the models ran, so `SigmaFillModel` could not
   fill it on the fused route while the composed route applies the models
   first. Now a `None` covariance becomes an `n x 36` NaN-filled covariance
   for the models (units: spherical coordinate covariance, no row
   validity); after the models every observation must have finite,
   positive lon/lat variances, else `InvalidRequest` whose message ends
   with `MISSING_COVARIANCE_REMEDY` (names `SigmaFillModel`). The same
   remedy text replaced the bare "orbit determination requires observed
   covariance" of `observed_covariance_flat`, which is what the composed
   route (`apply_observation_models` -> `NativeOrbitFitter.full_od` ->
   `iod` -> `iod_fit` -> `evaluate_orbit`) raised as `ValueError` for the
   same input (verified before the change: both routes raised that exact
   message), so both routes now raise `ValueError` naming the model without
   touching the coordinator-owned Python files. Tests:
   `od_refine::tests::{run_od_lets_a_sigma_model_fill_a_missing_covariance,
   run_od_without_covariance_or_a_filling_model_is_refused}`;
   `tests/test_run_od_null_covariance.py` (fused and composed routes with
   `CoordinateCovariances.nulls` observations: `SigmaFillModel` with a
   global row yields a fit with the filled sigmas recorded as used
   astrometry, no filling model raises `ValueError` matching
   `SigmaFillModel`, both routes agree on the fit).

4. **IOD: deterministic order, no early exit, failures stay local**
   (`od.rs` `iod_fit`, `od_drivers.rs` `iod_dict`, `iod.py`
   `tables_from_iod_output`). `iod_fit` now sorts the observations by
   epoch (integer `Epoch` days, then nanos), station code, then input row
   (`iod_observation_order`), runs the unchanged search
   (`iod_fit_sorted`: same triplet ordering, same `3 * (max_outliers + 1)`
   budget) on the sorted view (`take_coordinate_batch` /
   `take_observer_batch`) and maps `residuals`, `residual_chi2`,
   `residual_dof`, `residual_probability`, `solution` and `outlier` back to
   the input order (`restore_input_order`); `iod_fit_linkages_barycentric`
   inherits this. The `max_outliers == 0` early return is gone: zero
   contamination only skips the outlier trials. A candidate whose
   `evaluate_orbit` fails with `Backend` / `BackendProtocol` /
   `ThreadPool` (including the in-band per-row failure, which
   `evaluate_orbit` maps to `Backend`) is counted in `candidates_failed`
   and skipped; `InvalidRequest` / `Schema` propagate. `IodOutput` gained
   `triplets_tried`, `candidates_evaluated`, `candidates_failed`,
   `not_found_reason` (`IodOutput::not_found(n)` kept; `not_found_because`
   and `with_search_counters` added); every not-found path now states a
   reason (fewer than `min_obs`, no distinct-epoch triplet, every candidate
   prediction failed + last message, or "N candidates from M triplets
   scored, none within rchi2_threshold" with the best reduced chi2). The
   dict keys of the same names were added to `iod_dict` (so `iod_fit_ipc`
   and the `iod` entry of `full_od` / `run_od` outputs carry them) and
   `tables_from_iod_output` logs them at debug level when not found. On the
   callback route a Python exception raised for one candidate is now a
   skipped candidate (its stored exception is dropped once the search
   moves on); if every candidate failed the not-found reason carries the
   last message rather than raising. Tests (Rust, `od::tests`):
   `iod_observation_order_sorts_by_epoch_then_code_then_row`,
   `iod_fit_accepts_the_noise_free_arc_in_input_order`,
   `iod_fit_is_invariant_to_input_order` (shuffled input, both
   contaminations: identical state and member arrays in input order),
   `iod_fit_zero_contamination_continues_past_a_rejected_triplet` (a
   predictor displacing every finite candidate of the first triplet by half
   a degree), `iod_fit_skips_a_candidate_whose_prediction_fails` (each
   skippable error kind plus the in-band failure),
   `iod_fit_reports_total_prediction_failure_as_not_found`,
   `iod_fit_propagates_caller_errors_and_explains_rejections`. Python:
   `tests/test_iod_determinism.py` (`iod()` on `TwoBodyPropagator` with a
   shuffled copy: same state, epoch, chi2, and the same `solution` /
   `outlier` flags per observation id, at 0 % and 20 % contamination).
   Not done, by design: the plan's "Python `iod()` restores the legacy
   time/code sort as belt-and-braces" (not in this brief; the Rust sort
   makes it redundant). Follow-up for the coordinator-owned
   `native_orbit_fitter.py`: `tables_from_native_full_od` returns empty
   tables on an unfound IOD without passing through
   `tables_from_iod_output`, so the fused `full_od` / `run_od` routes do
   not log `not_found_reason` yet (the dict carries it).

5. **IOD: search budget and lazy triplet enumeration** (coordinator,
   after the agent's items; review items "avoid materializing all C(n, 3)
   combinations" and "triplet search and outlier allowance should remain
   separate concepts", plus the regression cases the review listed that
   were still missing: duplicate epochs, multiple roots).
   `IodConfig.max_triplets: Option<usize>` (`None` = legacy
   `3 * (max_outliers + 1)`, so no default changed; `Some(0)` is
   `InvalidRequest`), parsed from the `max_triplets` key of `iod_settings`
   in `od_drivers.rs`; not exposed on the Python signatures (the facade
   consolidation decides that). `select_observation_triplets_bounded` /
   `combination_triplets` (`od.rs`): (first, last) pairs are drawn in
   non-increasing arc order from a frontier heap seeded with the full-arc
   pair (sorted times, so moving the first later or the last earlier never
   lengthens the arc); only the pairs sharing the current arc have their
   middles enumerated and sorted by midpoint offset then index; the walk
   stops at the budget. The order is exactly the legacy one (pinned against
   the materialized sort on cadences with tied arcs and coincident epochs:
   `combination_triplets_match_the_materialized_legacy_order`); 3000
   observations with a budget of 24 run instantly
   (`combination_triplets_do_not_materialize_all_combinations`). Other
   tests: `iod_fit_search_budget_is_independent_of_contamination`,
   `iod_fit_never_pairs_duplicate_epochs_in_a_triplet` (duplicated last
   row, also with the twins swapped in the input),
   `iod_fit_scores_every_root_of_a_triplet` (the fixture's triplets all
   have two finite Gauss roots; the first made hopeless, the second is
   scored before another triplet). Python (`tests/test_iod_determinism.py`):
   `iod()` with a duplicated epoch, `iod_fit_ipc` with `max_triplets`.
   `tables_from_native_full_od` now logs `not_found_reason` at debug level
   (the follow-up noted under item 4).

6. **Fused routes honor the configured IOD propagator; fallback and
   tie rules pinned** (coordinator). Review items "the configured IOD
   propagator and settings are not consistently honored", "fallback
   behavior differs depending on whether the initial batch attempt or a
   later prediction fails" and the rejection section's "deterministic for
   ... residual ranking, ties". `NativeOrbitFitter.fused_propagator_matches`
   (`isinstance(propagator, propagator_class) and not propagator_kwargs`)
   gates both `NativeOrbitFitter.full_od` and `run_od`'s `_fused_run_od`;
   a mismatch runs the composition (IOD on `propagator_class`, as the
   docstrings say). All native test doubles subclass the test
   `TwoBodyPropagator`, so the existing fused tests keep their route. Test:
   `test_od_orchestration.py::test_fused_route_requires_the_configured_iod_propagator`
   (counting native propagator: fused used for the matching class, not for
   another class or extra kwargs; results found either way). The callback
   fallback was NOT changed: one rule is now pinned by
   `test_native_callback.py::test_callback_fallback_rule_is_the_same_for_every_prediction`
   (an exception from `generate_ephemeris` propagates unchanged and is never
   retried per candidate; a batch returning too few rows is predicted one
   candidate per call; a per-candidate result with the wrong row count is a
   `ValueError`). A symmetric "retry per candidate on a raising batch" was
   tried and rejected: it silently absorbed the probe-batch exception that
   item 2 makes propagate. Worst-residual ties
   (`od_refine.rs::lowest_probability_row`: lowest probability, then larger
   residual norm, then first candidate row; NaN probability is an error) are
   pinned by `worst_residual_selection_breaks_ties_deterministically`.
   Not addressed, by instruction (contract or policy decisions): explicit
   backend identity for dispatch, removal of the callback route, declared
   Jacobian compatibility per backend, typed stage handoffs, defaults in
   Rust `Default` impls, provenance records, API consolidation.

Gates run before each of the four commits (`cargo +1.87.0`): `fmt --all
--check`, `clippy --workspace --all-targets -- -D warnings`, `test
--workspace` (coords crate 255 -> 265 tests, all green); `maturin develop
--release`; `pytest src/adam_core/orbit_determination src/adam_core/dynamics
src/adam_core/observations` minus the three adam_assist-importing OD files:
579 -> 589 passed, 14 skipped, plus the two pre-existing collection errors
of this venv (`dynamics/tests/test_impact_viz_data.py` needs geopandas,
`dynamics/tests/test_lambert.py` needs adam_assist); `ruff` / `black` /
`isort` clean. No public signature changed, so
`migration/public_surface/manifest.json` was not regenerated (its `line`
fields for the touched `iod.py` / `rejection.py` symbols are a few lines
stale).

## 2026-10-13 follow-up: corrector rewritten (bead od_experiments_setup-a6w.1)

Commit `d450e1b3`. Answers the review's "one consistent meaning for convergence
and step acceptance, iteration and evaluation limits, singular fits, and
iterations / success / status_code" and the three solver defects the
walk-forward and consistency runs exposed.

* `solve` (`od_refine.rs`) is the trust-region Gauss-Newton method of scipy's
  `least_squares(method="trf", x_scale="jac")` without bounds (the Python
  baseline's solver): exact trust-region step on the scaled normal equations
  (Moré–Sorensen iteration on the 6x6 eigen-decomposition), radius driven by
  the actual-to-predicted reduction ratio, scipy's ftol / xtol / gtol rules
  with gtol on the column-scaled gradient. Replaces the Levenberg-Marquardt
  loop that accepted any decrease with `lambda /= 3` and no trust region.
* A non-finite Jacobian / gradient / normal matrix ends the fit with
  `status_code = -2`, `success = False`, iterate untouched (was: NaN ignored
  by the gradient norm, status 1 "converged").
* `max_nfev` is scipy's `nfev` (initial evaluation plus trial steps); the
  finite-difference Jacobian evaluations stay in the `iterations` column but
  are not budget, as in the baseline and the study harness's `max_nfev=200`.
* The contract (objective, derivatives and their exactness, scaling, step and
  acceptance, every status code, limits, covariance) lives on
  `WhitenedFitConfig` and in the `fit_least_squares` Notes.
* Validation against the recorded cases (astroforge_OD_consistency, ASSIST
  0.4.1, this commit installed with `scripts/parity/reinstall_adam_core_from_head.sh`):

  | case | before | after |
  |---|---|---|
  | 2000 FE8 seed (`results/parity/lm_repro`), analytic Jacobian | rchi2 1514, status 1, success | rchi2 2.316 in 7 evaluations, status 2 |
  | 2004 UE step 39 -> 40 (363 obs), analytic Jacobian | rchi2 1.3658 (scipy 1.3301) | 1.33014 = scipy's fixed point |
  | same, central differences | 1.317 (recorded "true minimum") | 1.31184 |

  With an inexact (two-body) Jacobian on ASSIST dynamics the solver converges
  to that Jacobian's fixed point, as scipy does; the analytic-Jacobian
  warning (`Propagator.two_body_dynamics`) says so, and `jacobian="central"`
  is exact for any propagator.
* Tests: trust-region step, NaN Jacobian, rejected finite trials end in xtol,
  central differences within default limits, budgets rewritten to trial
  semantics (Rust + `test_solver_budgets.py`), scipy trf parity
  (`test_corrector_contract.py`). Pins relaxed with their reason: forward
  differences 1e-8 (O(h) fixed point), Huber weight 1e-5 (robust cost equal
  to scipy to 1e-9, end state 1e-11 au apart on the flat valley).

## 2026-10-13 follow-up: typed IOD stage (bead od_experiments_setup-a6w.2)

The review's "typed batch between IOD and refinement" and "explicit
propagators per stage", without changing the search policy (beads a6w.3 /
a6w.4 do that).

* Rust `run_iod(iod_predictor, observed, observers, &IodConfig) -> IodResult`
  (`od.rs`): `IodResult { candidates: Vec<IodCandidate>, accepted, epoch_scale,
  triplets_tried, candidates_evaluated, candidates_failed, corrections_run,
  not_found_reason }`; `IodCandidate` carries state, exact `Epoch`, window
  rows, triplet rows, root index / count, `VelocityMethod`, prefit and
  post-trial statistics, removed rows, acceptance, and the full
  `FitEvaluation` for the accepted candidate only (dense arcs would otherwise
  retain one residual table per candidate). `restore_input_order` maps every
  candidate back to the caller's rows. `iod_fit` = `run_iod(..).to_output(n)`
  (deprecated one-orbit view; the linkage batch driver still uses it).
* `full_od_with(iod_predictor, predictor, ...)`, `run_od_with(iod_predictor,
  predictor, ...)`, `full_od_barycentric<I, R, T>` / `run_od_barycentric<I, R, T>`:
  IOD scores on the first, refinement runs on the second; `FullOdOutput.iod`
  is the `IodResult`.
* Bindings (`od_drivers.rs`): `run_with_predictors` (each of the two may be a
  Python propagator through `PyEphemerisPredictor` or adam_core's two-body;
  `iod_propagator=None` means the refining one; a Python exception from either
  is re-raised unchanged); `run_iod_ipc` / `run_iod_2body_ipc`; `iod_fit_ipc`
  aliases `run_iod_ipc`; `full_od_ipc` / `run_od_ipc` and the `_2body` twins
  take `iod_propagator=None`. The IOD dict keeps the accepted-orbit keys and
  adds `candidates` (list of candidate dicts), `accepted`, `corrections_run`.
* Python: `run_iod(observations, propagator, *, ...) -> (FittedOrbits,
  FittedOrbitMembers, IodReport)` in `iod.py` (frozen dataclasses `IodReport`
  / `IodCandidate`; `report_from_iod_output`), dispatching to a propagator's
  fused `run_iod` work unit or the callback binding; `iod()` and
  `NativeOrbitFitter.initial_fit` wrap it (the per-object
  `initial_orbit_determination` shortcut is gone from `iod()`, see CHANGELOG);
  `NativeOrbitFitter.iod_propagator_for(propagator)` gives fused `full_od` /
  `run_od` the configured IOD propagator when it is not the refining one, and
  the fused route no longer requires `fused_propagator_matches`.
* Tests: `test_run_iod.py` (legacy `iod()` parity with `run_iod`, fused vs
  callback candidate sets, empty input, two distinct Python propagators through
  `full_od_ipc` / `run_od_ipc`, `initial_fit` on the configured class),
  `test_iod_determinism.py` (candidate report contents and counters; candidate
  set invariant to row order), `test_od_orchestration.py` (fused route runs IOD
  on the configured propagator and still takes the fused route).

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
