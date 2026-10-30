# Changelog

This file contains notable changes in adam-core

## [Unreleased]

### Added

- A Rust-authoritative, complete OD configuration boundary: ``OdConfig``,
  stage-specific ``IodRunConfig``, ``RefinementRunConfig``, and
  ``EvaluationConfig``, ordered ``ObservationPipeline`` declarations, strict
  semantic resource bindings, immutable path-free profiles, and canonical
  configuration fingerprints/provenance.
- Closed native observation transforms for filling missing astrometric
  uncertainty, EFCC18 catalog-bias correction, station/band residual covariance
  adjustment, and nightly batch deweighting. Resource content identity, size,
  schema, and structure are validated before a bound pipeline can execute.
- Predictor-level Rust composition under ``adam_core_rs_coords::od::algorithms``:
  ``prepare_observations``, ``run_iod_with_predictor``,
  ``refine_with_predictor``, and ``run_od_with_predictors``. The configured
  composition prepares one coordinate batch and gives that same batch to IOD
  and refinement.
- Rust-owned bounded deterministic IOD/search/correction/ranging, whitened and
  robust differential correction, covariance validation, worst-residual and
  CMC2003 rejection, and propagation-only native provider ABI v2.
- Original/used astrometry, physical parameters, and non-gravitational
  parameters in fitted-result schemas where applicable. Older serialized
  fitted-orbit tables are upgraded with nullable fields by the current reader.

### Changed

- Python OD code is a typed declaration, resource-binding, Arrow-marshalling,
  and result-assembly boundary. Numerical expert functions live under
  ``adam_core.orbit_determination.algorithms``; propagators contribute only
  their negotiated native propagation capsule and do not own fitting,
  rejection, covariance, transforms, grouping, or OD policy.
- ``CORE_OD_V1`` and ``ASTEROID_INSTITUTE_OD_V1`` expand in Rust and require
  explicit local paths for their semantic resource slots. Core performs no
  runtime download, data-package discovery, or fallback to another profile.
- The accepted d21 numerical policy is represented explicitly, including
  light-time tolerance ``1e-12``, ephemeris maximum iterations ``1000``, and
  light-time maximum iterations ``10``. Fit evaluation and solver budgets are
  independent, IOD traversal is bounded, and provider/protocol/global failures
  remain fail-closed.
- Native provider ABI v2 uses an append-only negotiated layout, explicit
  identity and capability metadata, stable per-row failure codes, optional
  dynamics-consistent STMs, and complete physical/non-gravitational parameter
  transport. The provisional v1 layout is rejected.

### Removed

- The executable Python observation-model hierarchy, composite model callbacks,
  Python fitter/plugin hierarchy, model-attestation machinery, implicit
  ``models=`` selection, observatory-bias keyword routing, and public one-group
  configured workflow aliases. There are no compatibility aliases for these
  APIs.
- Python prediction callbacks and provider-owned OD work units. Unsupported
  Python-only propagators now fail with migration guidance instead of being
  invoked per prediction.

### Compatibility

- Core 0.7's canonical batch ``run_iod``, ``run_od``, ``refine``, and
  ``evaluate_orbits`` workflows are forthcoming. This foundation deliberately
  does not expose an interim Python configured workflow; the batch layer will
  own deterministic grouping, memberships, typed per-group failures, complete
  preflight ordering, and runtime provenance.
- These changes are intentionally source-breaking for the removed Python and
  Rust OD surfaces. They target adam-core 0.7.0; no merge, publication, release,
  or cross-package compatibility claim is implied by this unreleased entry.

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
