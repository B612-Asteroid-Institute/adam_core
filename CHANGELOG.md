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
  uncertainty; per-axis VFC2017 uncertainty floor (with explicit missing-value
  behavior) and replacement; EFCC18 catalog-bias correction; station/band
  residual covariance adjustment; chi2-per-observation performance weighting;
  measured-bias uncertainty flooring; and nightly batch deweighting. VFC2017
  floor preserves supplied cross terms, while replacement sets both angular
  cross entries to +0.0 because its lookup has no correlation. Resource content
  identity, size, schema, and structure are validated before a bound pipeline
  can execute.
- Advanced one-group Rust numerical kernels under
  ``adam_core::od::algorithms``: ``prepare_observations``,
  ``solve_prepared_iod``, and ``refine_prepared_orbit``. They support custom
  native predictors and focused research while the four canonical workflows
  remain the only public batching, provenance, continuation, and failure-policy
  routes.
- Rust-owned bounded deterministic IOD/search/correction/ranging, whitened and
  robust differential correction, covariance validation, worst-residual and
  CMC2003 rejection, and propagation-only native provider ABI v2.
- Original/used astrometry with exact diagonal variances, complete per-member
  source/observer and membership fingerprints, exact orbit-state fingerprints,
  and an unkeyed validated consistency digest over route, complete configuration,
  input identity, and propagated IOD metadata, plus physical parameters and
  non-gravitational parameters in fitted-result schemas where applicable.
  Older serialized tables upgrade with nullable fields but canonical
  refinement/evaluation fail closed until complete identity is present.
- Canonical batch ``run_iod``, ``run_od``, ``refine``, and
  ``evaluate_orbits`` workflows with Rust-owned deterministic grouping,
  complete preflight, typed per-group failures, explicit memberships, and
  canonical path-free runtime provenance. Rust decides output disposition,
  success/status, solution/outlier flags, state retention, and typed failure
  classification. Failure rows distinguish emitted ``orbit_id`` from
  ``input_orbit_id``; Python only assembles the returned typed columns. Every
  workflow returns exactly
  ``FittedOrbits``, ``FittedOrbitMembers``, and
  ``OrbitDeterminationFailures``.

### Changed

- Removed the redundant private legacy differential-correction and full-OD
  implementations and their parity-oracle tests. Independent ``run_iod`` and
  ``refine`` APIs and prepared-stage kernels remain authoritative; canonical
  workflow numerical behavior and provider/request contracts are unchanged.
- Optional observation policies use typed VFC2017, performance-weighted, and
  measured-bias-floor declarations in an ordered ``ObservationPipeline``;
  deliberate no-transformation identity uses ``ObservationPipeline::as_supplied()``
  / ``ObservationPipeline.as_supplied()``. Resource paths remain separate exact
  ``ResourceBindings`` inputs and are excluded from configuration identity and
  provenance.
- Native Python bindings now use the PyO3/NumPy 0.28 and Arrow/Parquet 58
  generation, extending standard wheels, CI, and release validation through
  CPython 3.14 on every existing manylinux and macOS target. Optional JAX wheel
  availability remains outside the standard support gate. The already-published
  0.7.0 artifacts are immutable, so these Python changes require a separately
  authorized post-0.7 release. ``adam_core_rs_coords`` directly exposes Arrow
  types and ``adam_core`` re-exports that API; the Arrow 58 source break and the
  project's exact internal version policy require an aligned 0.8.0 line for all
  six public Rust crates. Manifests intentionally remain at 0.7.0 in this
  candidate pending separate version-bump and publication authorization.
- MOID APIs now expose only algorithm-owned controls: Python
  ``calculate_perturber_moids`` no longer accepts ignored ``chunk_size`` or
  ``max_processes``, and NumPy/PyO3/Rust/SPICE/benchmark MOID entry points no
  longer accept ignored ``max_iter`` or ``xtol``. The canonical Rust scalar and
  batch functions return typed errors; the old NaN compatibility surface and
  helper exports were removed.
- SPICE kernel paths now cross Python in one typed initialization call. Rust
  owns path validation, parsing, load order, deduplication, and process-global
  ``KernelPool`` state. Perturber-MOID calls require this backend to be
  initialized explicitly rather than invoking Python-owned setup internally.
  The no-op ``force`` controls were removed from ``setup_SPICE`` and
  ``setup_mpc_obscodes``.
- Canonical Rust and Python OD batches now execute independent association or
  orbit/member groups concurrently on Rayon's shared pool. Indexed collection
  preserves first-appearance product and failure order, exact ``run_od``
  composition, and invocation-wide fatal-error behavior; inner propagation and
  coordinate kernels share the same pool in ordinary Rust composition rather
  than creating per-group thread pools. A fatal group lowers a stage-local
  cancellation bound without interrupting already-running work; lower indices
  remain eligible to establish earlier precedence, and every stage is a barrier.
  Each non-empty canonical invocation acquires one shared read guard on the
  initialized process-global SPICE backend before group scheduling; workers read
  that backend directly, and ``run_od`` holds the same guard across IOD and
  refinement. Other scientific readers remain concurrent. Kernel mutation is an
  initialization-only operation and Python mutators fail clearly while readers
  are active rather than queueing a writer; repeated initialization with exact
  paths that are already registered remains an idempotent read-side no-op. SPICE
  kernel-pool resolution also
  stays serial when invoked from a Rayon worker to protect non-OD callers that
  hold an exclusive lock across a query.
- The public Rust ``global_backend()`` synchronization type intentionally changes
  from ``&Mutex<AdamCoreSpiceBackend>`` to
  ``&RwLock<AdamCoreSpiceBackend>``. Rust consumers migrate immutable queries
  from ``.lock()`` to ``.read()`` and initialization/kernel mutation to
  ``.write()``. The bounded SPKEZ result cache now uses a narrow interior mutex,
  so cache use does not require exclusive ownership of the kernel pool or maps.
- Python OD code is a typed declaration, resource-binding, Arrow-marshalling,
  and result-assembly boundary. Numerical expert functions live under
  ``adam_core.orbit_determination.algorithms``; propagators contribute only
  their negotiated native propagation capsule and do not own fitting,
  rejection, covariance, transforms, grouping, or OD policy.
- ``CORE_OD_V1`` and ``ASTEROID_INSTITUTE_OD_V1`` expand in Rust and require
  explicit local paths for their semantic resource slots. Profile execution
  performs no runtime acquisition or fallback. Python depends exactly on
  ``jpl-debias-2018==2023.3.1.1`` and the public helper resolves only its
  exported ``bias_dat`` path; Rust still validates the exact file identity and
  structure. Standalone Rust remains explicit-path-only.
- The unreleased ``ASTEROID_INSTITUTE_OD_V1`` definition now pins the current
  space-observatory-inclusive station tables (1,126 residual-covariance rows
  and 2,480 uncertainty-fill rows). ``CORE_OD_V1`` is unchanged. Deployment
  and distribution details for separately supplied licensed resources remain
  outside adam-core.
- ``orbit_to_oem(version="2.0")`` writes validated OEM 2.0 with ICRF, 16
  significant digits, millisecond epochs, comments, and covariance blocks in
  RSW/RTN/TNW. ``version=None`` retains legacy output; SANA orbit-relative
  frames such as VNC_ROTATING remain an OEM 3.0 provision.
- The accepted d21 numerical policy is represented explicitly, including
  light-time tolerance ``1e-12``, ephemeris maximum iterations ``1000``, and
  light-time maximum iterations ``10``. Fit evaluation and solver budgets are
  independent, IOD traversal is bounded, and provider/protocol/global failures
  remain fail-closed.
- Historical IOD-policy evidence used ASSIST propagation with MPC observations
  on 2026-10-14: the selected policy seeded 116 of 122 first solutions across
  126 walk-forward objects, versus 16 for the legacy combinations/prefit gate
  and 114 for the comparison ladder. The backend name records study context;
  the policy and public workflows remain provider-generic.
- Native provider ABI v2 uses an append-only negotiated layout, explicit
  identity and capability metadata, stable per-row failure codes, optional
  dynamics-consistent STMs, and complete physical/non-gravitational parameter
  transport. The provisional v1 layout is rejected.
- **Breaking Rust API targeted for 0.8.0:** ``adam_core::od::algorithms`` is
  limited to ``prepare_observations``, ``solve_prepared_iod``,
  ``refine_prepared_orbit``, and ``PreparedObservations``. This additional
  cleanup corrects a stale compatibility implication without changing the
  published 0.7.0 release or artifact history:
  - ``run_iod_with_predictor`` becomes ``solve_prepared_iod`` and
    ``refine_with_predictor`` becomes ``refine_prepared_orbit``;
  - ``run_od_with_predictors`` and ``ConfiguredRunOdOutput`` are removed in
    favor of canonical ``run_od``;
  - ``external_provider_metadata`` becomes
    ``ProviderMetadata::from_external_propagator`` for metadata-only ABI-v2
    identity/provenance validation;
  - the public ``ErfaTimeProvider`` helper is replaced by a caller-owned
    ``TimeScaleProvider`` that may delegate to ``TimeArray::rescale``;
  - the public ``GlobalSpiceTranslation`` wrapper is replaced by a direct
    ``global_backend().read()`` guard passed as the translation provider; and
  - ``CanonicalStageSettings``, ``canonical_stage_settings``, and
    ``two_body_provider_metadata`` are internal. Canonical callers use typed
    workflow configuration and ``OdBatchResult::providers()``; advanced callers
    own their predictor options and provider services.
- **Breaking Rust API targeted for 0.8.0:** the exhaustive public
  ``ObservationTransform`` enum adds
  ``Vfc2017AstrometricUncertaintyFloor``,
  ``Vfc2017AstrometricUncertaintyReplace``,
  ``PerformanceWeightedCovarianceScaling``, and
  ``MeasuredBiasAstrometricUncertaintyFloor``. Downstream exhaustive matches
  written against 0.7.0 must handle these variants when migrating to 0.8.0.
- **Breaking Rust API:** ``PropagationError`` adds ``RowFailure`` and
  ``CmcCovarianceUnavailable``. Exhaustive downstream matches must handle
  candidate-local numerical failures and fail-closed CMC covariance refusal.
  Fatal provider and ABI/protocol failures remain separate variants.

### Fixed

- MOID uses the pure-Rust global Baluev-Mikryukov/DISTLINK critical-point
  solver through every scalar, batch, and SPICE-backed path, with deterministic
  witnesses and row-specific typed failures.
- Canonical OD configuration JSON now preserves every finite ``f64`` bit
  across Python construction, Rust parsing, persisted provenance, and later
  validation. Configuration identity is validated before observation-pipeline
  or provider work, and composed results validate the complete seed
  configuration/profile and every propagated IOD metadata field through an
  unkeyed consistency digest. Validated seed configuration JSON is re-emitted
  canonically before composed provenance, so equivalent whitespace and finite
  float spellings are representation-only.
- SPICE state lookup now resolves each epoch across all furnished kernels
  through spicekit's ``KernelPool`` rather than requiring one file to cover a
  complete batch.
- Weak-direction covariance validation now uses a second 0.1-sigma probe after
  a failed 1-sigma probe. Approximately quadratic scaling identifies curvature;
  a step-independent response is reported as evaluation noise with a provider
  integration-tolerance hint. Backend prediction errors still propagate, and
  the diagnostic tests fit-surface consistency rather than uncertainty realism.
- A sliced Quivr table (for example ``orbits[i:i+1]`` with a nonzero Arrow
  offset) handed to a typed Rust record-batch crossing could be read from the
  parent table's first rows. The Python bridge now materializes offset arrays
  with ``adam_core._rust.arrow.contiguous_record_batch`` before propagation,
  ephemeris, trajectory, and porkchop crossings. ``take``, ``apply_mask``, and
  IPC inputs were already correct; sliced-table regressions live in
  ``dynamics/tests/test_dynamics_table_slices.py``.

### Removed

- The executable Python observation-model hierarchy, composite model callbacks,
  Python fitter/plugin hierarchy, model-attestation machinery, implicit
  ``models=`` selection, observatory-bias keyword routing, and public one-group
  configured workflow aliases. There are no compatibility aliases for these
  APIs.
- Python prediction callbacks and provider-owned OD work units. Unsupported
  Python-only propagators now fail with migration guidance instead of being
  invoked per prediction.

### Migration from pre-0.7 OD APIs

- Replace ``initial_orbit_determination`` / ``iod`` with ``run_iod``;
  ``od`` with ``run_od``; and ``differential_correction`` with ``refine``.
  Replace direct residual evaluation with ``evaluate_orbits``. Python calls are
  association/member-backed batch functions returning exactly
  ``(FittedOrbits, FittedOrbitMembers, OrbitDeterminationFailures)``. Standalone
  Rust calls the same orchestration through ``adam_core::{run_iod, run_od,
  refine, evaluate_orbits}`` (or ``adam_core::od``), receives ``OdBatchResult``,
  and borrows its three immutable slices or consumes ``into_products()``.
- Split declaration from execution scope: ``IodStageConfig`` contains only the
  complete IOD/search/correction/ranging declaration; bound ``IodRunConfig``
  additionally owns the ordered pipeline, verified resources, and optional
  profile identity. ``EvaluationConfig`` is independent and self-contained;
  passing a full ``OdConfig`` to IOD or evaluation is no longer accepted.
- Select every pipeline, semantic resource, stage config, and propagation-only
  provider explicitly. Python ``None`` explicitly selects Core's built-in
  two-body provider; Rust uses ``OdPropagator::two_body()``. Python callbacks,
  provider-owned OD, profile-selected propagators, and resource fallback are
  removed.
- Replace ``NativeOrbitFitter``, ``OrbitFitter``, ``LeastSquares``, executable
  observation models, ``CompositeModel``, ``models=``, and
  ``observatory_bias_model=`` with a bound ``OdConfig`` or one of its stage
  projections. Removed APIs have no compatibility aliases.
- Construct raw IOD/OD input as ``SourceCatalog`` plus explicit
  ``Associations``. Supply ``FittedOrbitMembers`` explicitly to refinement and
  evaluation. One object uses the same batch route; linkage sorting and
  alternate one-group orchestration are not public 0.7 APIs.
- Bind every semantic resource slot explicitly. Use
  ``core_od_v1_resource_bindings()`` for the public profile's installed EFCC18
  path, or pass explicit ``ResourceBindings``. Standalone Rust always receives
  paths directly. There is no runtime acquisition or resource/profile fallback.
- Use ``ObservationPipeline.as_supplied()`` only for deliberate external
  preprocessing; predefined profiles are ineligible for that provenance.
  Transform declaration order and repetition are significant.
- Legacy fitted-member artifacts without complete original/used astrometry,
  memberships, source fingerprints, configuration, and orbit-state identity
  cannot be refined or evaluated canonically. Rerun IOD/OD to create the 0.7
  handoff rather than translating incomplete provenance.
- ``orbit_id`` is mutable relational data. Relabel by directly editing the
  orbit row and all matching member foreign keys together. Rename helpers are
  removed; scientific/member fingerprints do not depend on that label.
- Remove uses of model/callback/plugin APIs, public preparation helpers,
  ``OdPropagatorSource``, raw group handles, rename APIs, and OD build-identity
  fields/checks. Core build identity remains artifact diagnostics only and is
  not a scientific provenance or compatibility dimension.
- The public Rust crates compile and link without Python or PyO3. Canonical
  standalone Rust OD workflows do not import Python, use PyO3/IPC, or discover
  OD semantic-resource packages; callers pass explicit ``ResourceBindings``.
  This OD rule does not change the separate optional Python-package probe in
  ``adam_core::kernel_data`` for installed NAIF kernel-data distributions.
- Tests should compare NaN-bearing results by canonical IPC bytes/bit patterns,
  assert typed failure columns and original group IDs, and verify exact
  ``run_od == run_iod + refine`` composition instead of Python callback order.

### Compatibility

- Core 0.7's canonical OD workflows require explicit configuration. The
  built-in two-body route needs no capsule; an external provider must supply
  the complete propagation-only ABI-v2 identity and provenance fingerprints.
  Invalid input/configuration/resource/provider contracts fail atomically;
  safely isolated scientific failures continue as typed group failures.
- The removed legacy Python OD modules and Rust OD surfaces shipped as
  source-breaking changes in immutable adam-core 0.7.0. This unreleased entry
  does not rewrite that release or its artifact history.
- Only the later Rust OD API cleanup described by the dedicated **Breaking Rust
  API targeted for 0.8.0** bullet targets the intended 0.8.0 boundary. No merge,
  publication, release, or cross-package compatibility claim is implied by that
  unreleased cleanup.

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
