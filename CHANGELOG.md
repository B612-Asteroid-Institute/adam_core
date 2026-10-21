# Changelog

This file contains notable changes in adam-core

## [Unreleased]

### Changed

- `orbit_to_oem(version="2.0")` now writes a validated OEM 2.0 file like `version="3.0"` does for 3.0: `REF_FRAME = ICRF`, 16 significant digits (CCSDS 502.0-B-2 6.5.5), millisecond epochs, comments and covariance blocks in RSW, RTN or TNW. The default `version=None` keeps the legacy output unchanged, and the SANA orbit-relative frames such as `VNC_ROTATING` remain a 3.0 provision. `covariance_label` writes a table label (RSW, RTN, TNW) over a SANA frame of the same family, for instance `TNW_ROTATING` content under `TNW`, with a COMMENT naming the content.

## [0.6.1] - 2026-10-16

### Fixed

- Restored fresh Cargo resolution for Rust 1.87 consumers after all `zip`
  releases compatible with that MSRV were yanked. Kernel-data wheel extraction
  now uses `rawzip` with the existing pure-Rust `flate2` backend while
  retaining Stored and Deflate support, Zip64 parsing, CRC/size verification,
  and atomic cache publication.

### Compatibility

- No numerical algorithm, scientific model, covariance behavior, `SourceCatalog`
  schema, OEM behavior, or public Python/Rust API changed. The Python package
  and all six public Rust crates advance together to `0.6.1` so exact internal
  crate requirements remain coherent.

## [0.6.0] - 2026-10-15

### Added

- `orbit_to_oem` writes OEM 3.0 (`version="3.0"`) with labels, comments and local orbital frame covariance blocks (`covariance_frame`), and writes the state block of 9x9 covariances.
- `LocalFrameCovariances` and `local_frame_jacobians` rotate covariances into RSW, TNW and VNC frames; `orbit_from_oem` reads ICRF, J2000 and GCRF.
- Nullable `astcat` and `photcat` provenance columns on the Python
  `SourceCatalog` schema and Rust `SourceCatalogBatch`, preserving exact source
  strings and nulls through Arrow/Parquet and observation IPC.

### Changed

- OEM 2.0 output remains the default and byte-compatible for existing 6x6
  covariance inputs; when a 9x9 covariance is supplied, its Cartesian 6x6
  state block is emitted.
- OEM readers accept the equivalent `ICRF`, `J2000`, and `GCRF` frame labels.

### Compatibility

- The Python and Rust provenance fields are additive and row-nullable. Current
  Python `SourceCatalog.from_pyarrow`/`from_parquet` readers and the current Rust
  observation IPC dispatcher upgrade legacy 25-column
  `SourceCatalog` artifacts that lack `astcat` and `photcat` by adding typed
  null `LargeString` fields. The Python shim also accepts a legacy Arrow table
  missing only one of the two fields. It does not customize generic Quivr
  dataframe readers or relax unrelated schema/type validation.
- No compatibility claim is made for older adam-core readers consuming the new
  27-column schema. Adding fields to the public, exhaustively constructible Rust
  `SourceCatalogBatch` is semver-breaking for Rust consumers.
- The Rust OEM surface is intentionally source-breaking: the old
  `oem_write_kvn` export is removed; `oem_to_kvn` accepts comments and decimal
  precision; and `oem_render_orbits_kvn` / `oem_write_orbits_kvn` accept
  `OemWriteOptions` while reporting the number of rounded OEM epochs.
- The private CPython native symbol `_rust_native.oem_write_kvn` is removed;
  supported Python callers continue through `orbit_to_oem` and `orbit_from_oem`.
- These Rust API changes require the Python package and all six public Rust
  crates to advance together to `0.6.0`. Package publication remains a separate
  release decision.

## [0.5.8] - 2026-10-05

### Added

- Standards-complete MPC minor-planet, comet, and satellite designation codecs,
  plus strict classification into ADES `permID`, `provID`, and `trkSub` fields.
- ATLAS Teide (`R17`) station and observatory-band mappings.

### Changed

- ADES `rmsRA` and `rmsDec` accept the representable positive range from
  `0.00001` through values below `99999.5`, reducing fixed-point precision only
  as needed to satisfy the seven-character PSV field width.

### Compatibility

- MPC designation codecs now require canonical, standards-defined spellings,
  use the MPC A-prefix form for pre-1925 minor-planet provisionals, and raise
  `ValueError` for malformed or ambiguous forms that legacy helpers accepted.
- The minimum representable ADES `rmsRA` and `rmsDec` value is now `0.00001`;
  smaller positive values cannot be represented in PosDecimalTypeW7.

## [0.5.7] - 2026-08-31

### Added

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

### Compatibility

- Python 3.10 and older are no longer supported; the native release supports
  Python 3.11 through 3.13.
- Private JAX Jacobian and light-time helpers are removed. Public compatibility
  entrypoints use Rust kernels, and JAX is not a default runtime dependency.
- Ray-backed orchestration symbols are removed; historical scheduling keyword
  arguments remain accepted where needed for call compatibility.
- Optional Astropy/UT1, Astroquery monkeypatch, Healpy, plotting, external
  propagator, and provider integrations remain explicit boundaries.

### Known limitations

- Near-parabolic and hyperbolic cometary conversion retains the inherited
  legacy defect tracked by issue #210; its post-stable correction is not part
  of this migration release.

## [0.2.4] - 2024-09-20

### Added

- `Observers.from_codes()` method allows creation of observers from equal length arrays of codes and times, to be treated as pairs.
- `SourceCatalog` class has been added to better represent data coming directly from source catalogs. 