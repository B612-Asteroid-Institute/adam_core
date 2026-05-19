# Standalone Rust Surface Roadmap (2026-05-15)

Status: long-term capability map for making `adam-core-rs` capable of covering the full `adam_core` functional surface without requiring the Python interface.

Related artifacts:

- Full public-ish inventory: `migration/artifacts/adam_core_public_surface_inventory_2026-05-15.json` (`507` entries)
- Standalone surface status registry: `migration/standalone_rust_surface_status.json`
- Rust-native data model RFC: `migration/rust_native_data_model_rfc_2026-05-15.md`
- Gap report: `migration/adam_core_rust_surface_gap_report_2026-05-15.md`
- Existing review backlog: `migration/review_task_backlog_2026-04-28.md`
- Current migration tracker: `migration/TODO.md`

## Premise

The previous migration framing asked: “Which surfaces should be ported for the current Python/PyO3 interface?” The long-term framing is different:

> `adam-core-rs` should become a standalone Rust library that can handle the whole adam_core domain, with Python becoming an adapter rather than the owner of core behavior.

That changes the interpretation of “keep Python.” For the current Python API, many table/orchestration/network surfaces are poor speedup targets. For standalone Rust, they are capability gaps. The correct implementation direction is **not** a line-by-line port of PyArrow/quivr classes. It is a Rust-native domain model plus adapters for Arrow/Python where needed.

## Design principles

1. **Rust owns domain semantics; Python adapts.** Long-term Python modules should wrap Rust-native types and workflows, not reimplement them.
2. **Use typed Rust data contracts before porting high-level workflows.** OD, propagation, impact, observations, and exports should not be ported on top of ad hoc `&[f64]` buffers alone.
3. **Separate core computation from adapters.** Core crates should not depend on PyO3. Arrow, Python, network, and plotting should be optional features or adapter crates.
4. **Fail loudly on unsupported semantics.** Do not add silent fallbacks while replacing Python behavior.
5. **Keep scientific parity explicit.** Time scales, SPICE frame/body resolution, n-body dynamics, OD root selection, covariance propagation, and photometry validity policies need written parity contracts before implementation changes.
6. **Port workflows by stable boundaries, not helper-by-helper.** Small helpers should exist in Rust because they compose inside Rust workflows, but user-visible milestones should be high-level capabilities.

## Current Rust baseline

Current workspace members:

| Crate | Current role | Long-term direction |
|---|---|---|
| `adam_core_rs_autodiff` | `Dual<N>` and scalar trait for covariance/Jacobian kernels | Keep as low-level math/autodiff crate. Consider extending only when real workflows need more AD features. |
| `adam_core_rs_coords` | Coordinates, dynamics kernels, propagation, ephemeris, Lambert, MOID, photometry, weighted stats | Keep as the numerical-kernel base initially. Split into domain crates only after Rust-native data contracts stabilize. |
| `adam_core_rs_spice` | Direct Rust-to-Rust `spicekit` backend for SPK/PCK/text-kernel loading and frame/body lookup | Promote from backend wrapper to first-class ephemeris/frame service used by Rust workflows. |
| `adam_core_rs_orbit_determination` | Gibbs/Herrick-Gibbs/Gauss/GaussIOD kernels | Expand after propagation and data-model foundations exist. |
| `adam_core_py` | PyO3 boundary for current Python package | Treat as adapter only; do not put domain behavior here. |

## Prospective module/layer architecture (modules first)

Default implementation posture: **build new standalone-Rust work as modules inside the current crates first**. Create new crates only after a boundary has repeated use, a stable public API, and a clear dependency direction. The 2026-04-16 cleanup collapsed premature `adam_core_rs_math` / `adam_core_rs_dynamics` crates into `adam_core_rs_coords`; this roadmap should not recreate that churn.

The names below are layer labels and possible future crate names, not a crate-creation task list. A future crate split should be an explicit workstream decision with validation, not an automatic first implementation step.

```text
adam_core_rs_core          # errors, units, IDs, constants, validation helpers
adam_core_rs_time          # Epoch/TimeArray, scale conversion, leap-second policy
adam_core_rs_types         # coordinates, covariance, orbits, observers, observations, ephemerides
adam_core_rs_spice         # kernel store, frames, origins, body lookup, SPICE state providers
adam_core_rs_coords        # representation transforms, frame transforms, covariance propagation
adam_core_rs_dynamics      # 2-body, Lambert, MOID, Tisserand, porkchop, aberration
adam_core_rs_propagation   # Propagator trait, 2-body backend, safe ASSIST/n-body wrapper, ephemeris generation
adam_core_rs_od            # IOD, OD, least squares, outliers, evaluation, diagnostics
adam_core_rs_photometry    # H-G/H-G12/absolute magnitude, bandpass registry, magnitude predictions
adam_core_rs_observations  # detections, exposures, associations, ADES/MPC transforms
adam_core_rs_products      # OEM, SPK, OpenSpace, report/export products
adam_core_rs_catalogs      # optional network clients for Horizons/Scout/SBDB/NEOCC/source catalogs
adam_core_rs_arrow         # optional Arrow import/export adapters
adam_core_py               # Python adapter over Rust core + Arrow adapters
```

Recommended dependency direction:

```text
core → time → types
core + time + spice → coords
coords + spice → dynamics
coords + dynamics + spice + time + types → propagation
propagation + observations + photometry → od / impacts / products
catalogs and Python adapters sit at the edge
```

## Entire surface-area map

The exact 507-entry list is in `migration/artifacts/adam_core_public_surface_inventory_2026-05-15.json`. This table maps every module family from that inventory into a long-term Rust workstream.

| Workstream | Python modules covered | Inventory count | Implementation direction |
|---|---|---:|---|
| **W1 Core coordinate/time/orbit data model** | `time.time`; `coordinates.cartesian`, `coordinates.spherical`, `coordinates.keplerian`, `coordinates.cometary`, `coordinates.geodetics`, `coordinates.origin`, `coordinates.covariances`; `orbits.orbits`, `orbits.variants`, `orbits.classification`, `orbits.physical_parameters`; `observers.observers`, `observers.state` | 204 | Rust-native typed batches for epochs, origins, frames, coordinate representations, covariance blocks, orbit rows, observer rows, and metadata. Provide Arrow/Python adapters, not PyArrow-shaped internals. |
| **W2 Coordinate transforms, units, residuals, variants** | `coordinates.transform`, `coordinates.units`, `coordinates.residuals`, `coordinates.variants` | 34 | Lift existing Rust kernels onto Rust-native coordinate batches. Add Rust-native unit conversions, residual arrays, covariance sampling, and variant generation. Keep SciPy/NumPy choices only in Python adapter until Rust-native linalg/parity is implemented. |
| **W3 SPICE, frames, origins, observers** | `utils.spice_backend`, `utils.spice`, `observers.utils` | 18 | Make `adam_core_rs_spice` a reusable Rust service: kernel store, frame graph, body/frame/name lookup, origin-state cache, observer-state generation, fixed-kernel test fixtures. |
| **W4 Dynamics and mission design** | `dynamics.kepler`, `dynamics.barker`, `dynamics.stumpff`, `dynamics.chi`, `dynamics.lagrange`, `dynamics.propagation`, `dynamics.ephemeris`, `dynamics.aberrations`, `dynamics.lambert`, `dynamics.moid`, `dynamics.tisserand`, `dynamics.impacts`, `dynamics.exceptions`, `dynamics._rust_compat`, `missions.porkchop` | 60 | Govern existing Rust helpers, then expose typed Rust APIs for two-body propagation, ephemerides, Lambert/porkchop, MOID, Tisserand, aberration, and impacts. `dynamics._rust_compat` is counted only for inventory/accounting; it is adapter shim glue to retire, not a Rust port target. Make the ASSIST-compatible n-body adapter line the strategic propagation blocker; `assist-rs` is the intended GPL-licensed ASSIST/REBOUND Rust harness, and permissive core crates should depend only on backend-generic contracts. |
| **W5 Propagator abstraction and execution model** | `propagator.propagator`, `propagator.utils`, `parallel`, `ray_cluster`, `utils.chunking`, `utils.bounded_lru` | 33 | After the Rust propagator/n-body backend exists, replace Python subclass/Ray orchestration with Rust `Propagator` traits and Rayon chunking. Do not remove current Ray dispatch for ASSIST-touching surfaces before RM-FUTURE-002 / RM-STANDALONE-007 lands and RM-WD3-001 step 3 is revisited with data. |
| **W6 Orbit determination workflows** | `orbit_determination.gibbs`, `orbit_determination.herrick_gibbs`, `orbit_determination.gauss`, `orbit_determination.iod`, `orbit_determination.od`, `orbit_determination.least_squares`, `orbit_determination.evaluate`, `orbit_determination.outliers`, `orbit_determination.differential_correction`, `orbit_determination.orbit_fitter`, `orbit_determination.fitted_orbits` | 28 | Port after Rust data model + propagator trait exist. Build typed residual/fit/diagnostic outputs, deterministic convergence policies, root-selection policy, and covariance handling. |
| **W7 Photometry and bandpasses** | `photometry.magnitude`, `photometry.magnitude_common`, `photometry.absolute_magnitude`, `photometry.bandpasses.api`, `photometry.bandpasses.tables`, `photometry.bandpasses.vendor`, `photometry.bandpasses.constants`, `observations.photometry` | 35 | Existing numeric magnitude kernels are Rust. Add Rust bandpass registry/data model, filter conversions, grouped absolute-magnitude workflows, and observation/exposure joins once observation data model exists. |
| **W8 Observations, exposures, catalogs, associations** | `observations.ades`, `observations.detections`, `observations.exposures`, `observations.associations`, `observations.source_catalog` | 29 | Rust-native observation/exposure/association batches, ADES/MPC import/export data contracts, source-catalog schemas, and validation. Network retrieval can remain optional. |
| **W9 Orbit products, I/O, and visualization products** | `orbits.spice_kernel`, `orbits.oem_io`, `orbits.ephemeris`, `orbits.openspace.assets`, `orbits.openspace.renderable`, `orbits.openspace.translation`, `orbits.openspace.lua`, `orbits.plots`, `dynamics.plots`, `utils.plots.logos` | 43 | Implement product emitters as optional Rust exporters over Rust-native orbit/ephemeris batches. Plotting should produce data/specs or use optional crates; do not make plotting a core dependency. |
| **W10 Query/network clients** | `orbits.query.horizons`, `orbits.query.scout`, `orbits.query.sbdb`, `orbits.query.neocc` | 11 | Optional `catalogs`/`net` feature using `reqwest`/`serde`, isolated from core numerical crates. Preserve explicit error/timeout semantics. |
| **W11 Utility/string/helper surfaces** | `utils.mpc`, `utils.helpers.orbits`, `utils.helpers.observations` | 12 | Port small helpers when needed by Rust workflows. Use strong parsed types for MPC designations and observation IDs rather than only string functions. |
| **W12 Python compatibility and packaging** | `adam_core_py`; Python modules that become adapters | n/a | Keep Python API stable by round-tripping Rust-native batches through Arrow/PyO3. Python remains a consumer, not the source of truth. Adapter work is in addition to the 507 public-ish Python entries and should be sized from `API_MIGRATIONS`, `_rust/api.py`, and `adam_core_py` bindings. |

## Implementation directions by workstream

### W0 — Governance and roadmap control

Goal: keep the long-term port auditable while not destabilizing the current green migration branch.

Directions:

- Add a standalone-Rust status registry separate from the current Python-interface `API_MIGRATIONS` table.
- Track three states per surface family: `python-adapter-only`, `rust-native-kernel`, `rust-native-workflow`.
- Add dependency metadata: data model, time, SPICE, propagation, observation model, and network requirement.
- Keep the current canonical parity/speed gates for Python-facing migrated APIs.
- Add Rust-native validation suites for APIs that have no Python speed relevance but are required for standalone completeness.
- Track Python/PyO3 adapter work separately from the 507-entry inventory; use the current migrated API/binding count as the adapter sizing basis.

First deliverable:

- `migration/standalone_rust_surface_status.json` or a typed Rust/Python-generated report derived from the 507-entry inventory and this roadmap.

### W1 — Rust-native data model

Goal: create the foundation that lets high-level workflows move to Rust without copying PyArrow/quivr semantics.

Directions:

- Use strongly typed batches with explicit lengths and validity policies:
  - `TimeArray`, `OriginArray`, `Frame`, `CoordinateRep`, `CoordinateBatch`, `CovarianceBatch`, `OrbitBatch`, `ObserverBatch`, `ObservationBatch`, `ExposureBatch`, `EphemerisBatch`, `ResidualBatch`, `FitResultBatch`.
- Prefer structure-of-arrays storage for large vectorized workloads; expose row views for scalar algorithms.
- Store covariance as fixed-size row blocks (`[f64; 36]` initially; consider symmetric `[f64; 21]` later only if it does not complicate interop).
- Keep metadata/provenance explicit rather than piggybacking on Arrow schema fields.
- Add optional Arrow adapters after core types exist:
  - `TryFrom<&arrow_array::RecordBatch>` into Rust batches;
  - `Into<RecordBatch>` for Python/quivr compatibility;
  - schema-version checks that fail loudly.

Implementation order:

1. Core enums and IDs: `Frame`, `OriginCode`, `TimeScale`, `CoordinateRepresentation`, `PhotometricFilter`, `OrbitClass`.
2. Numeric batches for coordinates + covariance.
3. Orbit/observer/observation/exposure batches.
4. Arrow import/export adapters.
5. Python wrappers that delegate construction/validation to Rust where feasible.

### W1T — Time model (part of W1)

Goal: replace Python/Astropy ownership of `Timestamp` semantics with an explicit Rust time model, while preserving the current behavior as the oracle.

Current baseline:

- `Timestamp.rescale` uses Python `erfa` for UTC↔TAI leap-second handling, an exact TAI↔TT constant, a project-local TT↔TDB approximation, and Astropy/IERS for UT1.
- `Timestamp.et()` / `_jd_tdb_to_et` are pure arithmetic (`(MJD_TDB - 51544.5) * 86400`) and already bit-match `sp.str2et("JD ... TDB")` for fixed fixtures.
- `adam_core_rs_spice` retains parsed LSK/FK text-kernel content, but current timestamp scale conversion policy remains ERFA-based for leap seconds unless explicitly changed.

Directions:

- Define the exact supported time scales: at minimum TDB, TT, TAI, UTC, GPS if currently exposed.
- Keep MJD/JD representation policy explicit. Prefer an internal high-precision split representation (`day: i64`, `fraction: f64`) if f64 MJD roundoff becomes visible in parity.
- Evaluate a narrower set of implementation strategies against the existing baseline before porting conversions:
  - FFI/wrapper around ERFA/SOFA if licensing/distribution is acceptable;
  - Rust-native library such as `hifitime` only if it matches Astropy/ERFA across leap-second and TDB cases;
  - Rust workflows that are initially TDB-only while Python adapters keep ERFA rescaling;
  - spicekit LSK only for ET/TDB and kernel metadata, not as a full Astropy replacement unless parity proves it.
- Create fixed fixtures across every post-1999 leap second and representative pre/post J2000 epochs.
- Preserve the current pure arithmetic identity for MJD_TDB ↔ ET where applicable.

Acceptance criteria:

- `Timestamp.rescale` parity fixtures pass against the Python/Astropy current behavior.
- Observer/SPICE paths using ET do not drift across leap-second boundaries.
- Unsupported scales fail loudly, not silently approximate.

### W2 — Coordinates, covariance, units, variants

Goal: expose the already-migrated numerical coordinate kernels through typed Rust-native APIs.

Directions:

- Wrap existing transform kernels with `CoordinateBatch` inputs/outputs.
- Keep representation/frame/origin transforms as a single high-level Rust call path.
- Add covariance propagation as a first-class batch operation, including NaN policy.
- Port coordinate variants/sigma-point generation natively only after data-model and linalg choices are settled:
  - use `faer` or another Rust linalg backend for matrix square roots/eigendecomposition if parity and performance justify replacing SciPy `sqrtm`;
  - otherwise keep Python adapter-specific behavior while Rust workflows use a simpler validated Rust covariance sampler.
- Add unit conversions as tiny Rust helpers because they compose in Rust workflows, even if they do not matter for Python speed.

Acceptance criteria:

- Rust-native transform/covariance APIs can execute without PyO3/NumPy.
- Arrow/Python adapter outputs remain compatible with current coordinate tables.

### W3 — SPICE, frames, origins, observers

Goal: make frame/origin/body services usable directly from Rust workflows.

Status 2026-05-15: RM-STANDALONE-005 promoted `AdamCoreSpiceBackend` beyond string/low-level SPICE wrappers. The crate now has typed Rust service methods for data-model frames/origins/times, AU/AU-day state batches, origin translation vectors, frame transform matrices, and Earth-fixed ground-observer state generation. MPC observatory-table ownership remains a later data-ingest/model task; the current Rust API accepts already-resolved parallax coefficients.

Directions:

- Promote `AdamCoreSpiceBackend` into a Rust service API with:
  - kernel registry and last-loaded-wins semantics;
  - SPK/PCK/text-kernel dispatch;
  - body name/code resolution;
  - frame association and frame graph lookup;
  - batched `spkez`, `pxform`, and `sxform`;
  - origin translation vectors in AU/AU-day for coordinate workflows.
- Add Rust-native observer state generation:
  - geodetic → ITRF state;
  - ITRF → inertial frame with PCK `sxform`;
  - observatory code lookup and MPC observer-table handling if needed by standalone workflows.
- Keep CSpice parity ownership in `spicekit`/`spicekit-bench`, but adam-core should own fixture coverage for its units, defaults, and kernel lifecycle.

Acceptance criteria:

- Fixed-kernel Rust fixtures cover default kernels, user-loaded SPKs, ITRF93 rotations, LSK/FK retention, body alias resolution, and origin translations.
- Rust workflows no longer need Python `utils.spice` or observer helpers.

### W4/W5 — Dynamics, propagation, ephemerides, impacts

Goal: make propagation and ephemeris generation a Rust-owned workflow, while keeping the execution-model changes gated on the Rust n-body backend.

Status 2026-05-19: RM-STANDALONE-006 has a typed `Propagator`/`TwoBodyPropagator` surface with Rust-side provider, diagnostics, variant, and Arrow coverage. RM-STANDALONE-006A split the implementation into focused `propagation/` submodules. The diagnostic `propagation_bench` is Rust-internal only: raw serial Rust kernel vs typed Rust/pool modes, not Python/quivr/JAX evidence. Python/quivr end-to-end typed propagation parity is tracked as W12 adapter work until a typed PyO3 adapter exists.

Directions:

- Keep current 2-body kernels as the first `Propagator` implementation.
- Use `Propagator` consistently as the trait name. Under the existing modules-first policy, RM-STANDALONE-006 may start in a `propagation` module inside current crates; split to a future `adam_core_rs_propagation` crate only after the dependency boundary is stable and explicitly approved.
- Define a Rust trait around typed request/response objects rather than a minimal state-only method. The trait owns only propagation; ephemeris generation is a backend-agnostic workflow over a `Propagator` so light-time, aberration, rotation, and photometry semantics live in one place for 2-body and n-body backends.
- RM-STANDALONE-006 should land any missing prerequisite typed data-model pieces that the trait references (`EpochPolicy`, `CovariancePropagation`, `PropagationDiagnostics`, `EphemerisOptions`, `ObserverBatch`, `EphemerisBatch`, and row views as needed) before wiring high-level workflows.

```rust
pub enum EpochPolicy {
    CrossProduct,
    Pairwise,
    PerOrbit { indices: Box<[u32]> },
}

pub enum CovariancePropagation {
    None,
    Linearized,
    Monte { samples: usize, seed: u64 },
    SigmaPoint { alpha: f64, beta: f64, kappa: f64 },
}

pub struct PropagationOptions {
    pub chunk_size: Option<usize>,
    pub thread_limit: Option<usize>,
    pub epoch_policy: EpochPolicy,
    pub covariance: CovariancePropagation,
}

pub struct PropagationRequest<'a> {
    pub orbits: &'a OrbitBatch,
    pub times: &'a TimeArray,
    pub options: PropagationOptions,
}

pub struct PropagationResult {
    pub orbits: OrbitBatch,
    pub times: TimeArray,
    pub validity: Validity,
    pub diagnostics: PropagationDiagnostics,
}

pub trait Propagator: Sync {
    type Shard: PropagatorShard;

    fn integration_time_scale(&self) -> TimeScale;
    fn supports(&self, mode: CovariancePropagation) -> bool;
    /// Called once per Rayon worker; shards are Send, not Sync, and own mutable per-worker backend state.
    fn create_shard(&self) -> Self::Shard;
    fn propagate(
        &self,
        request: &PropagationRequest<'_>,
        provider: &dyn TimeScaleProvider,
    ) -> Result<PropagationResult>;
}

pub trait PropagatorShard: Send {
    fn propagate_one(&mut self, orbit: OrbitRow<'_>, times: &[Epoch]) -> Result<RowOutput>;
}

pub fn generate_ephemeris<P: Propagator>(
    propagator: &P,
    orbits: &OrbitBatch,
    observers: &ObserverBatch,
    options: &EphemerisOptions,
    spice: &dyn OriginStateProvider,
    provider: &dyn TimeScaleProvider,
) -> Result<EphemerisBatch>;
```

- Do not add a separate `request.covariances`; covariance lives on `OrbitBatch.coordinates.covariance`, while `CovariancePropagation` controls whether and how it is propagated.
- `PropagationRequest::new` should validate or normalize time ordering and return enough permutation metadata to restore caller row order; backends should not each own sorting policy.
- `TimeScaleProvider` is mandatory on propagation calls. Inputs must convert to `integration_time_scale()` or fail loudly, and `PropagationResult.times` must state the output scale by contract rather than by accident.
- Use per-thread shards for Rayon dispatch. `TwoBodyPropagator` can return a zero-sized shard; any ASSIST-compatible backend should return a shard that owns its own C simulation state.
- Reserve `Err` for setup/request errors. Per-row solver/integrator failures should be represented through `Validity` and diagnostics.
- Add `TwoBodyPropagator` first by lifting existing kernels.
- Add an ASSIST-compatible n-body backend by adapting `assist-rs` from a GPL-licensed harness crate/package to the core `Propagator` trait:
  - keep permissive core crates free of direct `assist-rs`/ASSIST/REBOUND dependencies;
  - use `assist-rs` `AssistData`, `Orbit`, propagation, STM/covariance, observatory, and ephemeris support where it matches the core contracts;
  - preserve backend pluggability by mapping `assist-rs` types/errors into `PropagationRequest`/`PropagationResult`, `Validity`, and diagnostics at the harness boundary.
- RM-STANDALONE-007A is decided: the GPL boundary is a separate `assist-rs` harness/adapter, mirroring the current Python `adam-assist` separation.
- Rebuild impact/collision helpers on top of the same propagator trait.
- Keep current Python Ray dispatch for ASSIST-touching surfaces until the Rust n-body backend lands; only then revisit RM-WD3-001 step 3 and replace in-process Ray defaults with data.

Acceptance criteria:

- 2-body Rust workflow reproduces current `dynamics.propagate_2body` behavior without Python.
- Backend-agnostic Rust `generate_ephemeris` reproduces current `dynamics.generate_ephemeris_2body` core behavior without Python for normalized same-origin ecliptic Cartesian inputs; origin/frame translation gaps fail loudly until the service/provider boundary is wired.
- Time-scale provider integration is tested: UTC/TT inputs integrate in the backend scale and return explicitly scaled output times.
- Per-row failure granularity is tested through `Validity` and diagnostics.
- n-body Rust workflow reproduces representative ASSIST-backed Python outputs and performance profiles using fixture-driven tests by default, with live `assist-rs` integration gated behind an explicit GPL harness feature/package boundary.
- Ray default changes for propagation/OD/impact paths happen only after RM-FUTURE-002 / RM-STANDALONE-007 and an updated parallel profile.

### W6 — Orbit determination workflows

Goal: port IOD/OD/LSQ after propagation and typed data exist.

Directions:

- Keep existing Rust Gibbs/Herrick/Gauss/GaussIOD kernels.
- Build Rust-native workflow types:
  - `IODProblem`, `IODSolutionSet`, `ODProblem`, `LeastSquaresOptions`, `FitDiagnostics`, `FittedOrbitBatch`.
- Inject a `Propagator` trait object/generic into OD and LSQ so 2-body and n-body use the same orchestration.
- Reuse `adam_core_rs_autodiff::Dual` for OD partials/Jacobians where forward-mode AD is appropriate instead of designing a parallel AD layer.
- Preserve deterministic convergence and diagnostic semantics:
  - iteration counts;
  - chi2/reduced-chi2;
  - residual batches;
  - covariance outputs;
  - outlier removal decisions.
- Make Gauss root-selection policy explicit and fixture-governed before broadening random input domains.

Acceptance criteria:

- Rust LSQ end-to-end fixture matches current Python workflow outputs and diagnostics.
- Rust OD/IOD workflows can run without constructing Python quivr tables.
- Performance summaries distinguish inner microbenchmarks from end-to-end wall-clock.

### W7 — Photometry and bandpass workflows

Goal: move from Rust numeric kernels to Rust-native photometry workflows.

Directions:

- Keep existing H-G phase/magnitude kernels.
- Add Rust data types for filters, bandpass metadata, exposures, and detection photometry.
- Decide whether bandpass/vendor data should be compiled into the crate, loaded from package data, or supplied by users.
- Port absolute-magnitude group workflows after observations/exposures data model exists.
- Keep validity policies explicit for invalid geometry, missing filter conversions, and NaN inputs.

Acceptance criteria:

- `predict_magnitudes` and absolute-magnitude estimators can run over Rust-native orbit/ephemeris/exposure/detection batches.
- Python bandpass API becomes an adapter over the Rust registry or remains clearly out-of-core if product scope says so.

### W8 — Observations, exposures, associations, catalogs

Goal: make the data that drives OD/photometry available natively in Rust.

Directions:

- Define `ObservationBatch`, `DetectionBatch`, `ExposureBatch`, `AssociationBatch`, and `SourceCatalogBatch`.
- Implement ADES/MPC parsing/serialization as optional but first-class Rust data ingest if standalone means end-to-end survey workflows.
- Keep crossmatch/association operations vectorized and typed.
- Treat network download/query as optional edge functionality; parsing and validation should not require network features.

Acceptance criteria:

- OD/photometry workflows consume Rust observation/exposure batches.
- ADES/MPC fixture round trips match current Python behavior where applicable.

### W9 — Orbit products, file formats, and visualization outputs

Goal: produce the same outputs without Python orchestration when required by standalone product scope.

Directions:

- SPK writing already exists through spicekit; expose it through Rust-native orbit/ephemeris batches.
- Add OEM read/write over Rust orbit/ephemeris types.
- Add OpenSpace renderable/Lua/export builders as optional text-product generators.
- Plotting should be optional and should preferably produce data/specification outputs rather than make heavy plotting dependencies part of core.

Acceptance criteria:

- SPK/OEM/OpenSpace output fixtures match Python output semantically.
- Product exporters are feature-gated outside numerical core.

### W10 — Query and network clients

Goal: decide whether standalone means numerical/workflow standalone only, or also network-client standalone.

Directions:

- If network clients are required, implement an optional `net` feature or companion crate with `reqwest` + `serde`.
- Keep query response parsing separate from HTTP transport so fixture tests do not need network.
- Preserve current timeout/error behavior explicitly.

Acceptance criteria:

- Horizons/Scout/SBDB/NEOCC parsers have local fixtures.
- Live network tests, if any, are opt-in and not required for normal correctness CI.

### W11 — Utility surfaces and execution model

Goal: avoid carrying Python infrastructure assumptions into Rust.

Directions:

- Replace `parallel`/`ray_cluster` defaults with Rust Rayon for in-process workloads only after the corresponding Rust-native compute backend exists; until then, keep the current RM-WD3 deferred posture for ASSIST-touching surfaces.
- Keep a distributed execution abstraction only if product workflows require multi-machine execution.
- Port MPC/string helpers into typed parsers where they are part of observation/catalog ingestion.
- Keep cache/chunk helpers private to the crates that need them.

Acceptance criteria:

- Core workflows do not depend on Python Ray, process pickling, or Python-side chunking once their Rust-native compute backend has landed.
- Rust APIs expose deterministic chunking and thread controls only where needed.

### W12 — Python and Arrow adapters

Goal: preserve the existing Python user interface while moving ownership to Rust.

Directions:

- Python classes should become thin wrappers around Rust-native batches plus Arrow adapters.
- Adapter responsibilities:
  - Python import compatibility;
  - quivr/PyArrow table conversion;
  - NumPy buffer conversion where still needed;
  - user-facing exceptions mapped from Rust errors.
- Add explicit typed propagation adapter parity once a PyO3 adapter exists: Python/quivr `Orbits`/variants should map into the Rust-canonical `OrbitBatch`/`OrbitVariantBatch` and `PropagationRequest` contracts, including provider-owned non-TDB rescaling.
- Avoid putting new domain behavior in `adam_core_py`.

Acceptance criteria:

- Existing Python tests continue to pass as workflows migrate underneath.
- Rust-native tests can run without importing Python.

## Suggested milestone plan

Sizing legend: **S** = design/prototype-sized, **M** = multi-surface implementation, **L** = strategic project likely requiring multiple phases.

### Milestone A — Standalone governance and typed skeleton (S)

Scope:

- Add standalone status registry.
- Define core Rust enums/errors and first typed coordinate/time/orbit batches.
- Add Arrow adapter prototype for one coordinate batch.

Exit criteria:

- Rust-only tests cover construction, validation, Arrow round-trip, and Python adapter smoke for one batch type.

### Milestone B — Time/SPICE foundations (M)

Scope:

- Implement Rust time strategy/parity fixtures.
- Promote `adam_core_rs_spice` service API for frames/origins/observer states. **Complete in RM-STANDALONE-005.**

Exit criteria:

- Fixed leap-second and SPICE fixture suite passes without Python.

### Milestone C — Rust-native coordinates/orbits (M)

Scope:

- Lift existing coordinate transform/covariance kernels onto typed batches.
- Port units, residual helpers, and Rust-native covariance/variant workflows as needed.

Exit criteria:

- Coordinate workflows can run end-to-end in Rust and round-trip through Python/Arrow.

### Milestone D — Propagation and ephemeris trait (L)

Scope:

- Define the `Propagator` trait and backend-agnostic `generate_ephemeris` workflow.
- Implement typed `TwoBodyPropagator`.
- Integrate an ASSIST-compatible n-body backend by adapting GPL-licensed `assist-rs` to the core `Propagator` contracts, or explicitly choose an equivalent Rust n-body path that preserves ASSIST parity.

Exit criteria:

- 2-body and representative n-body propagation/ephemeris fixtures pass.
- Parallelism lives in Rust for in-process workloads.

### Milestone E — OD/IOD/LSQ workflows (L)

Scope:

- Port IOD/OD/LSQ orchestration to Rust-native batches.
- Preserve diagnostics, covariance, outlier, and convergence semantics.

Exit criteria:

- End-to-end OD/IOD/LSQ fixtures pass without Python table construction.

### Milestone F — Observations, photometry, products (L)

Scope:

- Port observation/exposure/association/bandpass data workflows.
- Add native photometry workflows over Rust batches.
- Add SPK/OEM/OpenSpace/ADES product emitters as optional features.

Exit criteria:

- End-to-end observation → OD/photometry → product-output fixtures pass.

### Milestone G — Optional catalog/network/product edges (M)

Scope:

- Add optional network clients if standalone product scope requires them.
- Add optional plotting/export integrations.

Exit criteria:

- Network and visualization are feature-gated and fixture-tested without live services by default.

## Near-term task proposals

| Task | Size | Purpose | Depends on | Notes |
|---|---:|---|---|---|
| **RM-STANDALONE-001** | S | Create standalone surface status registry from the 507-entry inventory. | Existing inventory | Complete in `migration/standalone_rust_surface_status.json`; no runtime code changes. |
| **RM-STANDALONE-002** | M | Define Rust-native data model RFC for time, coordinates, covariances, orbits, observers, observations, exposures, and fitted results. | RM-STANDALONE-001 + product-scope decisions above | Complete in `migration/rust_native_data_model_rfc_2026-05-15.md`; covers covariance, variants, chunking, provenance, and adapter boundaries. |
| **RM-STANDALONE-003** | M | Prototype `CoordinateBatch` + `OrbitBatch` Rust types and Arrow adapters. | RM-STANDALONE-002 | Complete in `adam_core_rs_coords::types` with first flat Cartesian/Orbit Arrow contracts and PyO3 schema-metadata fixture hooks. |
| **RM-STANDALONE-004** | M | Time-scale strategy spike against the current ERFA baseline. | RM-STANDALONE-002/003 | Complete in `migration/time_scale_strategy_spike_2026-05-15.md` with leap-second fixture artifact and TDB→ET Rust arithmetic helper; first ERFA FFI implementation follows as RM-STANDALONE-004A. |
| **RM-STANDALONE-004A** | M | ERFA/liberfa FFI implementation for UTC↔TAI conversion. | RM-STANDALONE-004 | Complete in `adam_core_rs_coords::types::time` via `erfars`; fixture-backed `TimeArray::rescale` covers UTC/TAI/TT/TDB and rejects UT1/GPS. |
| **RM-STANDALONE-004B** | M | Saturate Rust time-rescale parity against existing Python `Timestamp` rescale tests. | RM-STANDALONE-004A | Complete: the fixture now mirrors the full scale-pair correctness matrix including UT1, Python verifies it against `Timestamp`, and Rust verifies it through an explicit provider boundary while provider-less UT1/GPS fail loudly. |
| **RM-STANDALONE-005** | M | Rust SPICE service API for origin/frame/observer states. | Existing `adam_core_rs_spice` | Complete: typed service methods build on direct `spicekit` plus Rust-native `TimeArray`/`CoordinateBatch` contracts. |
| **RM-STANDALONE-006** | L | Typed `Propagator` trait and `TwoBodyPropagator` implementation. | RM-STANDALONE-003/004B/005 | Complete through the first typed ephemeris workflow: core trait/backend work, covariance, variants, Rayon controls, Rust-side provider/Arrow/diagnostics validation, benchmark framing, module split, and backend-agnostic `generate_ephemeris<P: Propagator>` are in place. |
| **RM-STANDALONE-006E-PY / W12** | M | Python/quivr end-to-end typed propagation adapter parity. | Typed PyO3 propagation adapter | Separate adapter task: verify quivr/Arrow mapping and provider-owned non-TDB rescaling against the Rust-canonical propagation contracts once the adapter exists. |
| **RM-STANDALONE-006G** | L | Backend-agnostic typed `generate_ephemeris<P: Propagator>` workflow. | RM-STANDALONE-006 + observer/ephemeris typed batches | Complete for normalized same-origin ecliptic Cartesian orbit/observer inputs: `ObserverBatch`/`EphemerisBatch` contracts, `EphemerisOptions`, initial propagation via `Propagator`, shared light-time/stellar-aberration/rotation/photometry semantics, explicit output time scaling, row validity, diagnostics, and local benchmark smoke. Origin/frame translation remains fail-loud until a higher-level origin-state provider is wired at the adapter/service boundary. |
| **RM-STANDALONE-007A** | S | ASSIST/REBOUND GPL packaging-boundary decision. | Before RM-STANDALONE-007 code | Complete: use `assist-rs` (GPL-3.0) as the ASSIST/REBOUND Rust harness and adapt it to adam-core's core `Propagator` contracts from a GPL crate/package boundary mirroring `adam-assist`. |
| **RM-STANDALONE-007** | L | `assist-rs` GPL harness `Propagator` adapter spike and n-body parity fixture plan. | RM-STANDALONE-006 + RM-STANDALONE-007A | In progress: first excluded GPL crate skeleton (`rust/adam_core_rs_assist`) maps a normalized TDB-only heliocentric ecliptic Cartesian typed propagation path to `assist-rs`, including optional STM covariance transport and row diagnostics, without rewiring Python/Ray defaults. That normalized path is spike-only: parity and benchmark acceptance must target Python `adam_assist.ASSISTPropagator` public semantics, including caller-facing origin/frame/time behavior and identical kernel files. Remaining spike work is ASSIST/Python parity fixtures, live data-loading validation, public-semantics expansion, richer error classification, private-`assist-rs` build-story resolution, kernel identity checks, and the ephemeris-fast-path decision. |
| **RM-STANDALONE-008** | L | Rust-native OD/LSQ design over typed propagator and observation batches. | RM-STANDALONE-003/006/007 | Design before implementation; reuse `adam_core_rs_autodiff::Dual` where appropriate. |
| **RM-STANDALONE-009** | M | Observation/exposure/bandpass Rust model and parsers. | RM-STANDALONE-003 | Unblocks photometry workflows. |
| **RM-STANDALONE-010** | S | Product/export scope decision: SPK/OEM/OpenSpace/ADES/network/plotting required or optional. | RM-STANDALONE-001 | Prevents core crate dependency bloat. |

## Product-scope decisions

Resolved on 2026-05-15:

1. **Schema ownership:** Rust schemas are canonical; Python/quivr adapts through Arrow/PyO3 adapters.
2. **Time strategy:** start with FFI to ERFA/SOFA; outline and evaluate a Rust-native replacement as the next step.
3. **N-body backend:** ASSIST-compatible propagation remains preferred through `assist-rs` (GPL-3.0), with a GPL harness/adapter implementing adam-core's backend-generic `Propagator` contracts while permissive core crates remain pluggable and `assist-rs`-free.
4. **Network clients:** include network clients in first standalone scope, preferably as optional net/companion functionality with fixture-testable parsers and opt-in live tests.
5. **Plotting:** include plotting/visualization products in first standalone scope, but expect a new data/spec-oriented approach rather than direct Python plotting parity.
6. **File/product formats:** keep SPK/OEM/ADES/OpenSpace/product exporters in first standalone planning; exact first-release cut can be decided under `RM-STANDALONE-010`.

## Validation model

For each workstream:

- **Rust-native unit tests:** validate Rust types and algorithms without Python.
- **Fixture parity tests:** compare against current Python behavior, fixed CSPICE/ERFA fixtures, or spicekit-bench where appropriate.
- **Adapter tests:** verify Python/Arrow wrappers preserve current schemas and exceptions.
- **Performance tests:** only where performance is a stated goal; do not block standalone capability on Python-call speed.
- **End-to-end workflow tests:** required before claiming standalone coverage for propagation, OD, photometry, impacts, or product generation.

## Summary recommendation

The long-term track should remain a small but explicit standalone foundation project, not another isolated kernel-port sequence:

1. generate a standalone surface status registry from the 507-entry inventory — **complete**;
2. write the Rust-native data model RFC using the resolved scope decisions above — **complete**;
3. prototype coordinate/orbit/time batch types plus Arrow adapters — **complete**;
4. decide the time-scale implementation strategy against the current ERFA/TDB→ET baseline — **complete**;
5. implement the first ERFA/liberfa UTC↔TAI service behind the fixture — **complete**;
6. saturate Rust time-rescale parity against the existing Python `Timestamp` rescale tests — **complete**;
7. extend the Rust SPICE service API — **complete**;
8. then tackle the RM-FUTURE-002 / RM-STANDALONE-007 `assist-rs` GPL harness adapter integration — **next**.

Once those foundations exist, the high-level workflow ports become tractable and testable instead of being a piecemeal translation of Python orchestration.
