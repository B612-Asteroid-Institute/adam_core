# Changelog

This file contains notable changes in adam-core

## [Unreleased]

### Added

- Rust-native per-row `OrbitBatch` assessment of strict 6D/9D covariance
  semantics and canonical non-gravitational/Marsden consistency, without
  mutating or repairing supplied inputs. Consumers can call the same canonical
  covariance-row assessment directly.
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
- Standards-complete MPC minor-planet, comet, and satellite designation codecs,
  plus strict classification into ADES `permID`, `provID`, and `trkSub` fields.

### Changed

- Non-gravitational assessment now reports explicit A-coefficient and covariance
  parameterization facts, with separate nominal and covariance reasons for
  requiring a Marsden law, instead of ambiguous force-activation terminology.
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
- ADES `rmsRA` and `rmsDec` accept the representable positive range from
  `0.00001` through values below `99999.5`, reducing fixed-point precision only
  as needed to satisfy the seven-character PSV field width.

### Compatibility

- Python 3.10 and older are no longer supported; the native release supports
  Python 3.11 through 3.13.
- Private JAX Jacobian and light-time helpers are removed. Public compatibility
  entrypoints use Rust kernels, and JAX is not a default runtime dependency.
- Ray-backed orchestration symbols are removed; historical scheduling keyword
  arguments remain accepted where needed for call compatibility.
- Optional Astropy/UT1, Astroquery monkeypatch, Healpy, plotting, external
  propagator, and provider integrations remain explicit boundaries.
- MPC designation codecs now require canonical, standards-defined spellings,
  use the MPC A-prefix form for pre-1925 minor-planet provisionals, and raise
  `ValueError` for malformed or ambiguous forms that legacy helpers accepted.
- The minimum representable ADES `rmsRA` and `rmsDec` value is now `0.00001`;
  smaller positive values cannot be represented in PosDecimalTypeW7.

## [0.2.4] - 2024-09-20

### Added

- `Observers.from_codes()` method allows creation of observers from equal length arrays of codes and times, to be treated as pairs.
- `SourceCatalog` class has been added to better represent data coming directly from source catalogs. 