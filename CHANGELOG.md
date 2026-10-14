# Changelog

This file contains notable changes in adam-core

## [Unreleased]

### Added

- Fit-time observation handling for orbit determination, reimplemented in Rust
  from the Python `kk/obs-uncertainty-interface` branch (`d56114ac`) on top of
  the Rust OD core: `run_od` orchestration recording original vs. used
  astrometry on `FittedOrbitMembers` (`ObservationAstrometry`, `weight`,
  `astcat`); the `ObservationUncertaintyModel` interpreters
  (`EmpiricalCovarianceModel`, `PerformanceWeightedModel`, `SigmaFloorModel`,
  `NightBatchDeweightingModel`, `CompositeModel`, `IdentityModel`) driven by the
  `BIAS_TABLE_SCHEMA` observatory bias table; EFCC18 star-catalog debiasing
  (`adam_core.observations.efcc18`, `EFCC18DebiasModel`) reading JPL's
  `bias.dat` in HEALPix RING order through the Rust `ang2pix` port; the
  station/catalog sigma table (`VERES2017_SIGMA_TABLE_SCHEMA`, now allowing
  station-only and global rows) with its interpreters `SigmaFillModel` (fills
  only missing sigmas; the Asteroid Institute default with the `v2_sigma_fill`
  table of the private data package), `VeresFloorModel` and
  `VeresReplaceModel` (its legacy `veres2017_working` table); adam_core bundles
  no sigma table and resolves these defaults by importing the
  `observatory_uncertainties` data package on demand (`load_sigma_table`),
  raising `ImportError` when it is absent; the
  `astcat` column and `from_ades` converter on `OrbitDeterminationObservations`;
  `OrbitFitter.refine_fit` / `full_od` and `NativeOrbitFitter`; and the
  `observatory_bias_model` parameter on every module-level OD entry point.
- Rust orchestration of orbit determination: the whitened / robust
  differential correction, the worst-residual and CMC2003 rejection loops,
  the full OD (Gauss IOD decision loop followed by refinement) and `run_od`
  (observation models, full OD, provenance snapshots) are backend-generic Rust
  drivers over the `Propagator` trait (`adam_core_rs_coords::propagation::
  {fit_orbit_whitened_barycentric, iterative_fit_barycentric,
  cmc2003_fit_barycentric, full_od_barycentric, run_od_barycentric}`). A
  Rust-backed propagator exposes them as one-crossing methods
  (`fit_least_squares_whitened`, `iterative_fit`, `cmc2003_fit`, `full_od`,
  `run_od`) that `fit_least_squares`, `iterative_fit`, `cmc2003_fit_detailed`,
  `NativeOrbitFitter.full_od` and `run_od` dispatch to, wrapping the returned
  dicts into tables; the in-tree two-body implementations are the
  `_rust_native.*_2body_ipc` functions. Every other propagator (ASSIST until
  adam-assist adds the methods, PYOORB, user classes, duck-typed test
  doubles) runs the same Rust loops through the callback route: the driver
  calls the propagator's own `generate_ephemeris` whenever it needs
  predictions (`adam_core.orbit_determination._native_callback`), so no
  orbit-determination loop remains in Python. `fit_least_squares` no longer
  forwards keyword arguments to `scipy.optimize.least_squares` (only `xtol`,
  `ftol`, `gtol`, `max_nfev` are accepted; others raise `ValueError`) and
  `iod` runs the Rust decision loop for every propagator (`iterate` is
  accepted and ignored). A Python propagator's exception is re-raised
  unchanged; driver errors are `ValueError` / `RuntimeError`. Observation
  models provide `_native_specs()` so a model stack crosses once.
- Fixed: a sliced quivr table (`orbits[i:i+1]`, Arrow offset > 0) handed to
  the typed Rust record-batch crossings (`propagate_2body`,
  `generate_ephemeris_2body`, the trajectory and porkchop batches) was read as
  the parent table's first rows, because the Rust nested-schema decoders read
  struct children from the start of the shared buffers. The Python bridge now
  materializes offset arrays (`adam_core._rust.arrow.contiguous_record_batch`)
  before every such crossing; `take`, `apply_mask` and the IPC path were
  already correct. Regression tests in
  `dynamics/tests/test_dynamics_table_slices.py`.
- The shipped defaults are the signature defaults: `NativeOrbitFitter` now
  defaults to `outlier_rejection="cmc2003"` (was `"worst_residual"`), and
  `run_od` called without `models` applies the default observation-model
  stack written out in its body (`SigmaFillModel()` -> `EFCC18DebiasModel()`
  -> `EmpiricalCovarianceModel()` -> `NightBatchDeweightingModel()`) and
  explained in its docstring; pass `models=None` to fit the observations as
  supplied. `EmpiricalCovarianceModel`, `PerformanceWeightedModel` and
  `SigmaFloorModel` built without a table load `v2_full` from the
  `adam-observatory-uncertainties` data package (`load_bias_table`), like
  `SigmaFillModel` loads `v2_sigma_fill`; adam_core still does not depend on
  the package and names it in the `ImportError` when it is missing.
- Whitened-residual differential correction: `fit_least_squares` now minimizes
  the 2N whitened (lon, lat) residual components with an exact 2-body
  Jacobian from the Rust forward-mode autodiff kernels, validates the
  covariance along its weakest direction (falling back to central differences),
  supports Huber's M-estimator (`loss="huber"`), and reports per-observation
  weights; `iterative_fit` wraps it with outlier rejection. Carpino, Milani &
  Chesley (2003) rejection with re-inclusion (`cmc2003_fit`,
  `cmc2003_fit_detailed`) runs its decision kernels in Rust.

- Native CPython 3.11-3.13 wheels for manylinux 2.17 x86-64/AArch64 and
  macOS Apple silicon/Intel, with clean-room artifact acceptance and build-once
  trusted-publishing automation. Windows is deferred while upstream ASSIST
  requires POSIX memory mapping.
- Rust-native rotation-period estimation, grouped detection workflows, and
  best-apparition selection, including Rust-owned timing samples.
- Automatic kernel-data resolution for pure-Rust consumers: explicit override,
  installed Python package, cache, then checksummed wheel download.
- Upstream `main` at `9b756803` is integrated: MPC Obs80 parsing, strict Scout
  `file=mpc` snapshots and lifecycle errors, and validity-bounded Trajectory
  methods are Rust-owned behind compatible Python schemas.

### Added

- The typed IOD stage: Rust `run_iod` (and `run_iod_barycentric`) returns an
  `IodResult`, every scored Gauss candidate (`IodCandidate`: state, exact
  epoch, search window, triplet rows, root index and count, velocity method,
  prefit and post-trial statistics, outlier rows, acceptance) plus the
  accepted index, the search counters and the not-found reason; the
  bindings `run_iod_ipc` / `run_iod_2body_ipc` and the IOD dicts carry the
  same (`candidates`, `accepted`, `corrections_run`); Python `run_iod`
  returns the accepted orbit, its members and an `IodReport`. The search
  policy is unchanged. `iod_fit` / `iod_fit_ipc` remain as the one-orbit view
  and are deprecated in favour of `run_iod`.
- The span-bounded IOD triplet search, `observation_selection_method="spans"`
  (Rust `SpanSearchConfig`, settings keys `span_*`): windows of the arc in
  turn (densest 10 d, densest 30 d, last 30 d, first 30 d, whole arc), and
  within each a lazy walk of (first, last) pairs by distance of their span
  from a 6-day target, one triplet per pair with the middle nearest the
  midpoint and at most one per pair of nights, across windows, budget 32 triplets unless
  `max_triplets` is given; never materializes all pairs. Values are
  provisional (bead a6w.9). The default stays `"combinations"` until the
  post-correction acceptance of bead a6w.4.
- `velocity_method` for IOD (`run_iod`, `iod`, `NativeOrbitFitter`,
  `IodConfig`): `"gauss"`, `"gibbs"` (default, unchanged), `"herrick+gibbs"`
  or `"auto"` (Herrick-Gibbs when consecutive Gauss position vectors are
  within 5 degrees, Vallado 2013, Gibbs otherwise); each candidate records
  the method used. Rust `gauss_iod_candidates` reports it per root.
- The IOD propagator is explicit at every stage boundary: `full_od_with`,
  `run_od_with` and their barycentric drivers take an IOD predictor and a
  refinement predictor; the `full_od` / `run_od` bindings (and their
  `_2body` twins) accept `iod_propagator`; `NativeOrbitFitter.iod_propagator_for`
  hands fused work units the configured `propagator_class(**propagator_kwargs)`
  when it differs from the refining propagator, so the fused route now runs
  for every configuration and matches the composition on IOD backend.

### Changed

- IOD default policy (bead a6w.4): a Gauss orbit is a seed for differential
  corrections, not the answer. `IodAcceptance::Corrected` (Python
  `acceptance="corrected"`, `NativeOrbitFitter(iod_acceptance=...)`) scores
  every candidate of the search, then corrects up to `max_corrections` (8)
  of them in increasing prefit reduced chi2: first on the observations the
  triplet spans (grown to at least 6), then on spans grown fourfold, jumping
  apparition gaps to the nearest night, up to the whole arc, every stage
  within `rchi2_threshold`; no stage before the whole arc rejects anything,
  and the whole-arc stage gets the CMC2003 loop when its plain fit is above
  the threshold. It stops after `max_converged` (2) candidates reach the
  whole arc and accepts the one with the lowest whole-arc reduced chi2
  (index order on ties). The accepted orbit is that whole-arc fit, rejected
  rows reported as outliers. Rust
  `run_iod_with` / `run_iod_corrected`; `full_od` / `run_od` use it.
  `IodConfig::default()`, `run_od_ipc` / `full_od_ipc` / `run_iod_ipc`,
  `run_iod` and `NativeOrbitFitter` now default to `"spans"` with corrected
  acceptance; `iod()` and the deprecated `iod_fit_ipc` keep the legacy
  `"combinations"` search with the prefit gate. Values provisional (a6w.9).
- `iod()` and `NativeOrbitFitter.initial_fit` are thin wrappers over
  `run_iod`; the per-object shortcut to a propagator's compiled
  `initial_orbit_determination` work unit is gone from `iod()` (the linkage
  batch `initial_orbit_determination` still uses it), so a propagator without
  the fused `run_iod` work unit runs the core search through its own
  `generate_ephemeris` and gets the deterministic ordering, continuation and
  budget fixes of this branch.
- The native differential-correction solver is now the trust-region
  Gauss-Newton method of `scipy.optimize.least_squares(method="trf",
  x_scale="jac")` without bounds (exact trust-region step on the scaled
  normal equations, radius driven by the actual-to-predicted reduction
  ratio, `gtol` on the column-scaled gradient) instead of a
  Levenberg-Marquardt loop that accepted any decrease; the solver contract
  (objective, scaling, step acceptance, every `status_code`, limits,
  covariance) is written once on `WhitenedFitConfig` and in the
  `fit_least_squares` Notes. `max_nfev` now bounds scipy's `nfev` (the
  initial evaluation plus trial steps); finite-difference Jacobian
  evaluations are reported in `iterations` but are no longer budget, as in
  the Python baseline.
- The IOD epoch crosses the Rust boundary exactly: `IodOutput.epoch`
  (integer days and nanoseconds, `None` unless found) and
  `IodOutput.epoch_scale` replace the float `epoch_mjd`, the refinement seed
  of `full_od` sits at that exact epoch, `FullOdOutput.epoch_scale` is gone,
  and the IOD dicts carry `epoch_days` / `epoch_nanos` / `epoch_scale`.
- Rust `Default` impls for `IodConfig`, `RefinementConfig` (CMC2003) and
  `FullOdConfig`; the binding parsers (`fit_settings`, `iod_settings`,
  `refinement`) read every numerical default from the Rust configuration
  types instead of repeating the numbers.
- `Propagator.two_body_dynamics` (class attribute, default False) declares
  exact Kepler two-body dynamics; `fit_least_squares`, `iterative_fit`,
  `cmc2003_fit_detailed`, `NativeOrbitFitter.full_od` and `run_od` warn
  (`UserWarning`) when `jacobian="analytic"` is paired with a propagator
  that does not declare it, on the fused and the callback route alike.
- `run_od(models="default")` checks every default model's data before
  building any of them and raises one error naming all missing inputs
  (`ImportError` when the `adam-observatory-uncertainties` package is
  missing, `FileNotFoundError` when only EFCC18's `bias.dat` is).
- `fit_least_squares` defaults to `jacobian="analytic"`; the fused Rust
  Gauss-Newton work units of a propagator (`fit_least_squares_evaluated` /
  `fit_least_squares`) are reached with `jacobian="2-point"` and the linear
  loss, matching their forward-difference covariance. The default therefore
  runs the scipy solver with one N-body ephemeris per residual evaluation,
  plus the two-evaluation covariance probe and, when it fails, a
  twelve-evaluation central-difference fallback, on every rejection pass,
  instead of a single native crossing; choose `jacobian="2-point"` where the
  legacy covariance is acceptable and throughput matters. With
  `validate_covariance=True` (the default) the fused path now runs the same
  weak-direction probe on the native covariance and warns when it fails.
- `OrbitFitter.refine_fit` joins the fitter interface with a default that
  raises `NotImplementedError`, so fitters implementing only `initial_fit`
  (for example `adam_fo` releases predating this method) stay instantiable and
  usable for `initial_fit`; `full_od` and `run_od` require an override.
- The compiled Rust extension is now required. Public Python functions remain
  compatibility veneers while numerical, table, product, query, and
  orchestration work executes in Rust.
- Astropy, Astroquery, Healpy, and plotting stacks are explicit optional extras;
  ordinary time conversion, SBDB access, HEALPix operations, and geographic
  timezone lookup no longer import them in the default runtime.
- ASSIST orchestration belongs to `adam-assist`, which consumes
  `libassist-sys` and `librebound-sys` directly. No `assist-rs` v2 facade is
  published.
- `coordinates.residuals.calculate_chi2` now documents and enforces the
  covariance contract required by valid covariance matrices: inputs must be
  symmetric positive definite. Singular or indefinite matrices raise
  `ValueError`; NaN diagonal entries still raise, and NaN off-diagonal entries
  are treated as zero with a warning for legacy compatibility.
- `gaussIOD(mu=...)` now uses the supplied central-body gravitational parameter
  consistently for candidate geometry and velocity. The legacy implementation
  incorrectly reverted to the solar constant inside its velocity helpers.

### Deprecated

- `orbit_determination.od.od`, `orbit_determination.od.differential_correction`
  and `orbit_determination.least_squares.LeastSquares`, the legacy
  finite-difference loops (the former still solving with SciPy), warn with
  `DeprecationWarning` on use; `fit_least_squares`, `iterative_fit`,
  `cmc2003_fit_detailed` and `run_od` run the Rust solver.

### Fixed

- The solver reported `success=True` (`status_code=1`) when the Jacobian or
  gradient was non-finite, because the gradient norm ignored NaN; a
  non-finite derivative now ends the fit with `status_code=-2` and
  `success=False`, the iterate untouched.
- `iterative_fit` returns the lowest reduced chi2 among the passes whose fit
  succeeded, falling back to the lowest overall only when no pass converged;
  previously a failed first pass was kept over a later converged fit.
- `NativeOrbitFitter` copies `propagator_kwargs` instead of sharing one
  mutable default dictionary across instances.
- `run_od` returns the same `object_id` column on the fused and the composed
  route: the caller's `object_id` on every row, null when none was given. The
  fused route previously leaked the opaque backend identifier into it.
- IOD triplet search: the `Combinations` selection is enumerated lazily in
  the legacy priority order (never the full `C(n, 3)` list), the search
  budget is its own setting (`IodConfig.max_triplets` / the `max_triplets`
  key of `iod_settings`; unset keeps the legacy `3 * (max_outliers + 1)`
  budget), so the contamination percentage only sets the outlier allowance;
  a duplicated epoch is used at most once per triplet and every finite Gauss
  root of a triplet is scored before the next triplet; the fused `full_od` /
  `run_od` routes log the IOD not-found reason.
- The fused `full_od` / `run_od` work units are only used when the
  refinement propagator is the fitter's configured IOD propagator
  (`NativeOrbitFitter.fused_propagator_matches`: an instance of
  `propagator_class` with no `propagator_kwargs`); otherwise the composition
  runs IOD on the configured class as documented, instead of silently
  substituting the refinement propagator.
- CMC2003: a row whose rejection statistic cannot be established (NaN residual or Jacobian, singular expected residual covariance with a nonzero residual) now scores `+inf` with the `non_finite_chi2` / `singular_residual_covariance` flag, so it is rejected when selected and never re-included; previously it scored `0` and looked acceptable.
- The weak-direction covariance probe no longer converts a backend error raised while predicting the probe displacement into a "covariance could not be validated" warning: it propagates like any other prediction failure (on the callback route the Python exception reaches the caller unchanged); only the in-band probe outcomes (no positive eigenvalue, rejected trial prediction, non-finite probe residuals) remain warnings.
- The fused `run_od` route (a Rust-backed propagator's `run_od` work unit) runs the observation models before requiring an observed covariance, so observations without reported uncertainties can be filled by a `SigmaFillModel` exactly as on the composed route; after the models every observation must carry finite, positive lon/lat variances, else `ValueError` names `SigmaFillModel` as the remedy (the same message every other OD entry point now gives for a missing covariance).
- Gauss IOD (`iod`, `NativeOrbitFitter.initial_fit` / `full_od`, `run_od`, the linkage batch) is deterministic in the caller's row order: the Rust driver sorts observations by epoch, station code and input row before selecting triplets and maps every member array back to the input order. A zero contamination percentage only disables outlier removal (a rejected candidate is followed by the next candidate and triplet instead of ending the search), a candidate whose prediction fails is counted and skipped instead of aborting the search (`InvalidRequest` / schema errors still propagate), and the IOD dict carries `triplets_tried`, `candidates_evaluated`, `candidates_failed` and `not_found_reason`, logged at debug level when no orbit is accepted.

### Compatibility

- Python 3.10 and older are no longer supported; the native release supports
  Python 3.11 through 3.13.
- Private JAX Jacobian and light-time helpers are removed. Public compatibility
  entrypoints use Rust kernels, and JAX is not a default runtime dependency.
- Ray-backed orchestration symbols are removed; historical scheduling keyword
  arguments remain accepted where needed for call compatibility.
- Optional Astropy/UT1, Astroquery monkeypatch, Healpy, plotting, external
  propagator, and provider integrations remain explicit boundaries.

## [0.2.4] - 2024-09-20

### Added

- `Observers.from_codes()` method allows creation of observers from equal length arrays of codes and times, to be treated as pairs.
- `SourceCatalog` class has been added to better represent data coming directly from source catalogs. 