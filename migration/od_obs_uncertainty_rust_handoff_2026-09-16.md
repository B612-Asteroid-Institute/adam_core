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
| Observation models | — | `ObservationUncertaintyModel` trait + `IdentityModel`, `EmpiricalCovarianceModel`, `PerformanceWeightedModel`, `SigmaFloorModel`, `NightBatchDeweightingModel`, `Efcc18DebiasModel`, `CompositeModel`; `BiasTable` lookup; `efcc18` parse / lookup / corrections / debias; `veres2017` bundled table, `VeresSigmaLookup`, `VeresFloorModel`, `VeresReplaceModel` |
| PyO3 (`adam_core_py`) | `od_ops.rs`, `orbit_determination.rs` (Gauss IOD), `coordinate_ops::evaluate_orbits_numpy`, Arrow propagate / ephemeris | `observation_uncertainty.rs` (13 functions), `differential_correction.rs` (11 functions); registered in `lib.rs`, guarded in `_rust/api.py` |

New Rust files: `rust/adam_core_rs_coords/src/{efcc18,observation_uncertainty,veres2017,cmc2003,differential_correction}.rs`,
`rust/adam_core_py/src/{observation_uncertainty,differential_correction}.rs`. Existing files touched: `ades_io.rs` (one added kernel), both `lib.rs`.

## Python <-> Rust boundary

Rust owns every per-observation arithmetic and lookup; Python owns quivr
tables, pyarrow schema validation, the `OrbitFitter` plugin boundary and the
scipy optimizer driving a user-supplied `Propagator`:

| Python surface (same names as the Python branch) | Rust kernel(s) behind it | Stays Python (why) |
|---|---|---|
| `OrbitDeterminationObservations.astcat`, `from_ades` | `ades_angular_covariance_numpy` | table assembly, `Observers.from_codes` (its own Rust crossing) |
| `EmpiricalCovarianceModel` / `PerformanceWeightedModel` / `SigmaFloorModel` | `bias_table_model_apply_numpy` | `validate_bias_table` (pyarrow cast), quivr `set_column` only when Rust reports a change |
| `NightBatchDeweightingModel` | `night_batch_deweighting_model_apply_numpy` | — |
| `EFCC18DebiasModel`, `observations.efcc18.*` | `efcc18_parse_bias_dat`, `efcc18_ra_dec_to_healpix_numpy`, `efcc18_catalog_columns_numpy`, `efcc18_corrections_numpy`, `efcc18_debias_model_apply_numpy`, `healpix_ang2pix_lonlat_numpy` | locating / downloading / checksumming / caching `bias.dat` (I/O) |
| `VeresFloorModel` / `VeresReplaceModel` / `VeresSigmaLookup` / `veres2017_sigma_table` | `veres2017_sigma_table_columns`, `veres_sigma_lookup_numpy`, `veres_model_apply_numpy` | `validate_veres_sigma_table` (pyarrow cast) |
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
  that path now also carry `weight` (1 / 0).
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

* No Rust-native optimizer for the whitened / Huber fit: the grid-validated
  scipy trust-region path is preserved for parity; a Rust IRLS Gauss-Newton on
  `fit_orbit_least_squares_with_predictor` would let the adam-assist backend
  fit with the analytic Jacobian in one crossing.
* No Rust-owned `bias.dat` acquisition (the kernel-data crate pattern would fit).
* Benchmark (`benchmark_*`) twins and `_rust/status.py` registry rows were not
  added for the new kernels (they are new surface, not migrations of legacy
  APIs).
