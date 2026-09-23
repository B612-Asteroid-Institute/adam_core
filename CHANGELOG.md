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
  `bias.dat` in HEALPix RING order through the Rust `ang2pix` port; the VFC2017
  station/catalog sigma table (`VeresFloorModel`, `VeresReplaceModel`); the
  `astcat` column and `from_ades` converter on `OrbitDeterminationObservations`;
  `OrbitFitter.refine_fit` / `full_od` and `NativeOrbitFitter`; and the
  `observatory_bias_model` parameter on every module-level OD entry point.
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

### Changed

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

### Fixed

- `iterative_fit` returns the lowest reduced chi2 among the passes whose fit
  succeeded, falling back to the lowest overall only when no pass converged;
  previously a failed first pass was kept over a later converged fit.
- `NativeOrbitFitter` copies `propagator_kwargs` instead of sharing one
  mutable default dictionary across instances.

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