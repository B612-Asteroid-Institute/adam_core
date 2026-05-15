# RFC: Rust-Native Data Model for Standalone adam-core-rs (2026-05-15)

Status: RM-STANDALONE-002 design artifact.

Related artifacts:

- `migration/standalone_rust_surface_roadmap_2026-05-15.md`
- `migration/standalone_rust_surface_status.json`
- `migration/artifacts/adam_core_public_surface_inventory_2026-05-15.json`
- `migration/adam_core_rust_surface_gap_report_2026-05-15.md`

## 1. Purpose

This RFC defines the first Rust-native data contracts needed for `adam-core-rs` to become a standalone library. The goal is not to mirror PyArrow/quivr classes line-by-line. The goal is to make Rust the owner of core domain schemas and workflows, with Python/quivr becoming an adapter layer.

User decisions incorporated:

- Rust schemas are canonical; Python/quivr adapts through Arrow/PyO3 adapters.
- Time conversion starts with FFI to ERFA/SOFA; a Rust-native replacement should be outlined/evaluated as a follow-up.
- `assist-rs` is the preferred n-body backend, but propagation must be backend-pluggable.
- Network clients are in first standalone scope, preferably optional/feature-gated.
- Plotting/visualization products are in first standalone scope, but likely as data/spec output rather than a direct Python plotting port.

## 2. Non-goals

- Do not create many new crates immediately. Implement modules inside current crates first; split crates only after stable boundaries emerge.
- Do not rewrite numerical kernels in this RFC. Existing Rust kernels remain valid.
- Do not replace the current Python API in one step. Python compatibility remains an adapter requirement.
- Do not invent silent fallbacks for unsupported semantics.
- Do not decide final plotting libraries or live-network behavior here; define the data contracts they consume/produce.

## 3. Baseline facts to preserve

### Current time model

- Python `Timestamp` stores a single `scale` plus `days: i64` and `nanos: i64` since MJD epoch.
- `Timestamp.rescale` uses Python `erfa` for UTC↔TAI↔TT↔TDB.
- `Timestamp.et()` / `_jd_tdb_to_et` are pure arithmetic: `(MJD_TDB - 51544.5) * 86400`.
- The TDB→ET arithmetic already bit-matches `sp.str2et("JD ... TDB")` for fixed fixtures.

### Current coordinate conventions

- Distances are AU.
- Velocities are AU/day.
- Public coordinate angles are degrees.
- Spherical angular rates are degrees/day.
- Coordinate covariance values use the same units/order as their owning coordinate representation.
- Coordinate frame is currently a scalar attribute (`ecliptic`, `equatorial`, `itrf93`, or `unspecified`).
- Origin is currently a per-row table of string codes.

### Current Rust governance

- `API_MIGRATIONS` currently tracks 42 Rust-governed API IDs.
- Existing Rust kernels should be lifted behind typed Rust batches instead of bypassed or duplicated.

## 4. Design principles

1. **Canonical Rust schemas.** Rust defines the semantic schema; Arrow/quivr is an interchange adapter.
2. **Structure-of-arrays for batches.** Prefer columnar storage for large workloads, with row views for scalar algorithms.
3. **Explicit validity.** Do not use NaN as the only missing-value representation. Preserve NaN as a numeric value when science semantics require it.
4. **Typed units and domains.** Units and angle conventions should be encoded in field names/types and schema metadata.
5. **Backend pluggability.** Propagation and time services should be trait-backed where multiple implementations are expected.
6. **No fallback ambiguity.** Unsupported frames, scales, origins, or file/network semantics should return typed errors.
7. **Adapter isolation.** PyO3, NumPy, quivr, and Arrow conversion code should live at the edge.

## 5. Proposed module placement

Start as modules inside existing crates:

| Module | Initial crate | Later split candidate |
|---|---|---|
| `types::ids`, `types::units`, `types::schema` | `adam_core_rs_coords` or new internal module | `adam_core_rs_core` |
| `types::time` | `adam_core_rs_coords` initially, using an ERFA/SOFA FFI module | `adam_core_rs_time` |
| `types::coordinates`, `types::covariance` | `adam_core_rs_coords` | `adam_core_rs_types` / `adam_core_rs_coords` |
| `types::orbits`, `types::observers` | `adam_core_rs_coords` + `adam_core_rs_spice` adapters | `adam_core_rs_types` |
| `types::observations`, `types::photometry` | `adam_core_rs_coords` initially | `adam_core_rs_observations` / `adam_core_rs_photometry` |
| `types::fitting` | `adam_core_rs_orbit_determination` | `adam_core_rs_od` |
| `arrow_adapters` | optional module/feature | `adam_core_rs_arrow` |

Crate splitting should require an explicit follow-up review.

## 6. Core primitive types

### 6.1 IDs

Use typed wrappers instead of unstructured strings at workflow boundaries:

```rust
pub struct OrbitId(pub String);
pub struct ObjectId(pub String);
pub struct ObservationId(pub String);
pub struct ExposureId(pub String);
pub struct CatalogId(pub String);
pub struct VariantId(pub String);
pub struct ObservatoryCode(pub String);
pub struct PhotometricFilter(pub String);
```

Initial implementation can store `Vec<String>` or `Vec<Option<String>>`. Later optimization can dictionary-encode repeated IDs without changing the semantic type.

### 6.2 Time scales and epochs

```rust
pub enum TimeScale {
    Tai,
    Tdb,
    Tt,
    Utc,
    Gps,
}

pub struct Epoch {
    pub days: i64,
    pub nanos: i64,
}

pub struct TimeArray {
    pub scale: TimeScale,
    pub epochs: Vec<Epoch>,
}
```

Invariants:

- `0 <= nanos < 86_400_000_000_000` after normalization.
- A `TimeArray` has exactly one scale.
- Cross-scale comparison must rescale explicitly.
- ET conversion must rescale to TDB then use the current arithmetic baseline.
- UTC↔TAI↔TT↔TDB rescaling initially delegates to ERFA/SOFA FFI.

Follow-up replacement path:

1. Lock down ERFA/SOFA fixture coverage across leap seconds and TDB cases.
2. Evaluate Rust-native candidates against those fixtures.
3. Accept a Rust-native implementation only if it matches the fixture policy or if a deliberate science-policy change is approved.

### 6.3 Validity

Use an explicit validity bitmap for nullable arrays:

```rust
pub struct Validity {
    // bit i = row i is valid
    pub bits: Vec<u64>,
}
```

Guidelines:

- Numeric NaN is a valid numeric payload unless validity says the row/field is absent.
- Arrow nulls map to invalid validity bits.
- Existing all-NaN covariance-row policies remain numeric payload policies, not null policies.

### 6.4 Units and frames

```rust
pub enum Frame {
    Ecliptic,
    Equatorial,
    Itrf93,
    Unspecified,
    Spice(String),
}

pub enum CoordinateRepresentation {
    Cartesian,
    Spherical,
    Keplerian,
    Cometary,
    Geodetic,
}

pub enum OriginId {
    SolarSystemBarycenter,
    Naif(i32),
    Named(String),
    Observatory(ObservatoryCode),
}
```

Initial canonical units should match the current public API:

| Quantity | Unit |
|---|---|
| distance | AU |
| velocity | AU/day |
| time | MJD day+nanos plus scale |
| Cartesian position | AU |
| Cartesian velocity | AU/day |
| Keplerian/cometary angles | degrees |
| Spherical longitude/latitude | degrees |
| Spherical angular rates | degrees/day |
| Geodetic longitude/latitude | degrees |
| Magnitudes | mag |
| RA/Dec uncertainties from observation files | preserve source units at parse boundary, normalize in typed observation covariance |

Internal kernels may convert to radians, but typed batch fields should preserve the canonical public units unless a future policy explicitly changes the user-facing schema.

## 7. Covariance model

Current `CoordinateCovariances` is a nullable LargeList of length-36 rows. Standalone Rust needs a general covariance model for both 6-D coordinates and lower-dimensional observations.

```rust
pub struct CovarianceBatch {
    pub dimension: usize,
    pub values_row_major: Vec<f64>, // len = rows * dimension * dimension
    pub row_validity: Option<Validity>,
    pub units: CovarianceUnits,
}

pub enum CovarianceUnits {
    Coordinate(CoordinateRepresentation),
    ObservationAngular2D,
    Photometry1D,
    Custom(Vec<String>),
}
```

Invariants:

- `dimension > 0`.
- `values_row_major.len() == rows * dimension * dimension`.
- Matrices are stored row-major.
- Symmetry is a validation property, not a storage compression assumption.
- Invalid covariance rows represent missing covariance.
- Valid rows may contain NaNs where existing numerical compatibility requires them.

Initial prototype guidance:

- RM-STANDALONE-003 may implement only `dimension = 6` first for coordinates.
- The RFC still reserves the generic shape so observations do not require a second covariance type.

## 8. Coordinate batches

```rust
pub enum CoordinateValues {
    Cartesian(Vec<[f64; 6]>),   // x, y, z, vx, vy, vz
    Spherical(Vec<[f64; 6]>),   // rho, lon_deg, lat_deg, vrho, vlon_deg_day, vlat_deg_day
    Keplerian(Vec<[f64; 6]>),   // a, e, i_deg, raan_deg, ap_deg, M_deg
    Cometary(Vec<[f64; 6]>),    // q, e, i_deg, raan_deg, ap_deg, tp_mjd_tdb
    Geodetic(Vec<[f64; 6]>),    // alt, lon_deg, lat_deg, vup, veast, vnorth
}

pub struct CoordinateBatch {
    pub values: CoordinateValues,
    pub frame: Frame,
    pub origins: OriginArray,
    pub times: Option<TimeArray>,
    pub covariance: Option<CovarianceBatch>,
}

pub struct OriginArray {
    pub origins: Vec<OriginId>,
}
```

Design decisions:

- `frame` is batch-scalar initially, matching the current quivr attribute. Mixed-frame data should be split or rejected.
- `origins` are per-row because current origin is a per-row column and mixed-origin transforms are real future requirements.
- `times`, when present, must have `len == CoordinateBatch::len()`.
- `covariance`, when present, must have `rows == CoordinateBatch::len()` and `dimension == 6`.
- Coordinate values are fixed 6-wide per representation even when some columns are not meaningful for a particular workflow.

Required methods:

```rust
impl CoordinateBatch {
    pub fn len(&self) -> usize;
    pub fn representation(&self) -> CoordinateRepresentation;
    pub fn validate(&self) -> Result<(), SchemaError>;
    pub fn row(&self, index: usize) -> CoordinateRow<'_>;
}
```

Compatibility policies:

- Current covariance-transform NaN row policy must be preserved.
- Cartesian↔Cartesian frame-only Python performance decisions do not determine Rust-native API design.
- Unsupported frame/origin combinations should error with typed reasons.

## 9. Orbit and physical-parameter batches

```rust
pub struct PhysicalParametersBatch {
    pub h_v: Vec<f64>,
    pub h_v_sigma: Vec<f64>,
    pub g: Vec<f64>,
    pub g_sigma: Vec<f64>,
    pub sigma_eff: Vec<f64>,
    pub chi2_red: Vec<f64>,
    pub validity: PhysicalParameterValidity,
}

pub struct OrbitBatch {
    pub orbit_id: Vec<OrbitId>,
    pub object_id: Vec<Option<ObjectId>>,
    pub coordinates: CoordinateBatch, // Cartesian for current Orbits compatibility
    pub physical_parameters: Option<PhysicalParametersBatch>,
}
```

Invariants:

- `coordinates.representation() == Cartesian` for `OrbitBatch` compatibility with current `Orbits`.
- Element/orbital representations should use `CoordinateBatch` directly or a future `ElementBatch`, not overload `OrbitBatch`.
- `orbit_id.len() == coordinates.len()`.
- `physical_parameters`, when present, has matching length.

## 10. Observer batches

```rust
pub struct ObserverBatch {
    pub code: Vec<ObservatoryCode>,
    pub coordinates: CoordinateBatch, // Cartesian observer state
}
```

Invariants:

- `coordinates.representation() == Cartesian`.
- `coordinates.len() == code.len()`.
- Observer generation should be owned by `adam_core_rs_spice` service APIs, including geodetic observatory positions, PCK rotations, and origin/frame transforms.

## 11. Observation and exposure batches

The standalone model should distinguish raw parsed observations from normalized OD-ready observations.

### 11.1 ExposureBatch

```rust
pub struct ExposureBatch {
    pub exposure_id: Vec<ExposureId>,
    pub start_time: TimeArray,
    pub duration_seconds: Vec<f64>,
    pub filter: Vec<PhotometricFilter>,
    pub observatory_code: Vec<ObservatoryCode>,
    pub seeing_arcsec: Vec<f64>,
    pub depth_5sigma_mag: Vec<f64>,
    pub validity: ExposureValidity,
}
```

Validation:

- `duration_seconds >= 0`.
- `start_time.len() == exposure_id.len()`.
- Optional seeing/depth use explicit validity.

### 11.2 DetectionBatch

```rust
pub struct DetectionBatch {
    pub observation_id: Vec<ObservationId>,
    pub exposure_id: Vec<Option<ExposureId>>,
    pub time: TimeArray,
    pub ra_deg: Vec<f64>,
    pub dec_deg: Vec<f64>,
    pub angular_covariance: Option<CovarianceBatch>, // dimension 2
    pub photometry: Option<PhotometryBatch>,
}
```

Validation:

- `0 <= ra_deg <= 360`.
- `-90 <= dec_deg <= 90`.
- Angular covariance, when present, is dimension 2 and uses normalized angular units.

### 11.3 ADESObservationBatch

ADES should remain a first-class parse/serialize batch because ADES is a product/data-interchange format, not just a Python convenience.

```rust
pub struct AdesObservationBatch {
    pub perm_id: Vec<Option<ObjectId>>,
    pub prov_id: Vec<Option<String>>,
    pub tracklet_sub_id: Vec<Option<String>>,
    pub obs_sub_id: Vec<Option<ObservationId>>,
    pub obs_time: TimeArray,
    pub ra_deg: Vec<f64>,
    pub dec_deg: Vec<f64>,
    pub rms_ra_cos_dec_arcsec: Vec<f64>,
    pub rms_dec_arcsec: Vec<f64>,
    pub rms_corr: Vec<f64>,
    pub photometry: Option<PhotometryBatch>,
    pub station: Vec<ObservatoryCode>,
    pub mode: Vec<String>,
    pub astrometric_catalog: Vec<String>,
    pub photometric_catalog: Vec<Option<String>>,
    pub remarks: Vec<Option<String>>,
    pub validity: AdesValidity,
}
```

ADES parser/serializer tests should fixture round-trip representative observations and preserve formatting precision policy separately from normalized OD covariance policy.

## 12. Photometry and bandpass batches

```rust
pub struct PhotometryBatch {
    pub mag: Vec<f64>,
    pub mag_sigma: Vec<f64>,
    pub filter: Vec<Option<PhotometricFilter>>,
    pub validity: PhotometryValidity,
}

pub struct BandpassRegistry {
    // Implementation may be static data, package data, or user-supplied.
}
```

Policies:

- Current H-G numerical kernels remain valid.
- Bandpass conversions and vendor tables should be Rust-owned for standalone workflows.
- Missing filter conversions should be typed errors unless a workflow explicitly allows missing output.

## 13. Ephemeris and residual batches

```rust
pub struct EphemerisBatch {
    pub orbit_id: Vec<OrbitId>,
    pub object_id: Vec<Option<ObjectId>>,
    pub coordinates: CoordinateBatch, // Spherical observed coordinates
    pub predicted_magnitude_v: Vec<f64>,
    pub alpha_deg: Vec<f64>,
    pub light_time_days: Vec<f64>,
    pub aberrated_coordinates: Option<CoordinateBatch>, // Cartesian
    pub validity: EphemerisValidity,
}

pub struct ResidualBatch {
    pub values: Vec<Vec<f64>>, // or a fixed-width storage once dimensions are known
    pub dimension: usize,
    pub chi2: Vec<f64>,
    pub dof: Vec<i64>,
    pub probability: Vec<f64>,
    pub validity: ResidualValidity,
}
```

Near-term implementation should prefer fixed-width residual storage when the dimension is known (`2` for RA/Dec, `6` for coordinate residuals) and use generic storage only at adapter boundaries.

## 14. Variants and sigma-point propagation

```rust
pub struct OrbitVariantBatch {
    pub orbit_id: Vec<OrbitId>,
    pub object_id: Vec<Option<ObjectId>>,
    pub variant_id: Vec<Option<VariantId>>,
    pub weights: Vec<f64>,
    pub weights_cov: Vec<f64>,
    pub coordinates: CoordinateBatch,
}
```

Requirements:

- Variant generation must preserve existing Monte Carlo seed behavior when adapting from Python.
- Sigma-point generation needs an explicit Rust linalg policy before replacing SciPy `sqrtm`.
- Propagator requests must be able to carry variants/covariances so OD/impact workflows do not need a parallel API later.

## 15. Fitted orbit and OD result batches

```rust
pub struct FittedOrbitBatch {
    pub orbit_id: Vec<OrbitId>,
    pub object_id: Vec<Option<ObjectId>>,
    pub coordinates: CoordinateBatch,
    pub arc_length_days: Vec<f64>,
    pub num_obs: Vec<i64>,
    pub chi2: Vec<f64>,
    pub reduced_chi2: Vec<f64>,
    pub iterations: Vec<i64>,
    pub success: Vec<bool>,
    pub status_code: Vec<i64>,
    pub validity: FittedOrbitValidity,
}

pub struct FittedOrbitMemberBatch {
    pub orbit_id: Vec<OrbitId>,
    pub observation_id: Vec<ObservationId>,
    pub residuals: Option<ResidualBatch>,
    pub solution: Vec<bool>,
    pub outlier: Vec<bool>,
    pub validity: FittedOrbitMemberValidity,
}
```

OD workflow types should build on these batches:

- `IODProblem`
- `IODSolutionSet`
- `ODProblem`
- `LeastSquaresOptions`
- `FitDiagnostics`

Convergence diagnostics, outlier decisions, root-selection policy, and covariance propagation must remain explicit test surfaces.

## 16. Propagation request/response data

The data model must support the pluggable propagator design before `assist-rs` is wired.

```rust
pub struct PropagationOptions {
    pub chunk_size: Option<usize>,
    pub thread_limit: Option<usize>,
}

pub struct PropagationRequest<'a> {
    pub orbits: &'a OrbitBatch,
    pub times: &'a TimeArray,
    pub covariances: Option<&'a CovarianceBatch>,
    pub variants: Option<&'a OrbitVariantBatch>,
    pub options: PropagationOptions,
}

pub struct PropagationResult {
    pub orbits: OrbitBatch,
    pub covariances: Option<CovarianceBatch>,
    pub variants: Option<OrbitVariantBatch>,
}

pub trait PropagatorBackend {
    fn propagate(&self, request: &PropagationRequest<'_>) -> Result<PropagationResult>;
    fn generate_ephemeris(
        &self,
        orbits: &OrbitBatch,
        observers: &ObserverBatch,
        options: &EphemerisOptions,
    ) -> Result<EphemerisBatch>;
}
```

The first backend should wrap existing 2-body Rust kernels. The n-body backend should target `assist-rs`, but the trait must not encode `assist-rs`-specific assumptions.

## 17. Network and plotting/product scope

### 17.1 Network clients

Network clients are in first standalone scope, but should sit at the edge:

- Use an optional `net` feature or companion module.
- Separate HTTP transport from response parsers.
- Fixture-test parsers without live network.
- Keep live service tests opt-in.
- Feed parsed results into canonical Rust batches, not one-off structs.

### 17.2 Plotting and visualization

Plotting is in first standalone scope, but should not make core numerical crates depend on plotting libraries.

Preferred approach:

- Rust products emit data/specs for plots or visualization tools.
- OpenSpace exporters generate typed product records and text output from Rust batches.
- Python plotting can remain an adapter over product specs.
- Direct parity with Python Plotly figures is not required unless product scope later demands it.

## 18. Arrow and Python adapter contracts

Arrow adapters should be generated from Rust schema definitions where possible.

```rust
pub trait ArrowSchemaExport {
    fn schema() -> arrow_schema::Schema;
}

pub trait TryFromRecordBatch: Sized {
    fn try_from_record_batch(batch: &arrow_array::RecordBatch) -> Result<Self>;
}

pub trait IntoRecordBatch {
    fn into_record_batch(self) -> Result<arrow_array::RecordBatch>;
}
```

Adapter rules:

- Arrow schema metadata should include `adam_core_schema`, semantic version, units, frame/origin/time-scale metadata where scalar.
- Quivr adapters must preserve current column names when serving the existing Python API.
- Rust-native errors map to Python exceptions at the PyO3 boundary.
- Adapter validation should be strict by default; schema mismatches fail loudly.

## 19. Error model

Define typed error domains early:

```rust
pub enum SchemaError {
    LengthMismatch { field: &'static str, expected: usize, actual: usize },
    InvalidTimeScale(String),
    UnsupportedFrame(String),
    UnsupportedOrigin(String),
    InvalidCovarianceShape { rows: usize, dimension: usize, values: usize },
    InvalidUnit { field: &'static str, unit: String },
    MissingRequiredField(&'static str),
}
```

Downstream crates can wrap these errors but should not replace them with unstructured strings.

## 20. Validation plan

### RM-STANDALONE-002 validation

- This RFC must be internally consistent with `standalone_rust_surface_status.json` and `TODO.md`.
- Docs-only validation is `git diff --check` plus any lightweight JSON/RFC consistency checks.

### RM-STANDALONE-003 prototype validation

- Rust unit tests for `TimeArray`, `CoordinateBatch`, `CovarianceBatch`, and `OrbitBatch` construction/validation.
- Arrow round-trip tests for one Cartesian `CoordinateBatch` and one `OrbitBatch` fixture.
- Python adapter smoke that recreates current `CartesianCoordinates`/`Orbits` tables from Rust schema output.
- Fixture for null covariance vs numeric all-NaN covariance row.

### RM-STANDALONE-004 time validation

- ERFA/SOFA FFI parity against current Python `Timestamp.rescale` across UTC↔TAI↔TT↔TDB.
- Fixed leap-second matrix covering every post-1999 leap second.
- TDB→ET fixture preserving current pure arithmetic identity.
- Document Rust-native replacement candidates and why they pass/fail parity.

### Later workflow validation

- Coordinate transform/covariance parity through typed batches.
- Observer/SPICE fixtures through Rust service APIs.
- 2-body propagator typed fixtures.
- `assist-rs` n-body representative fixtures.
- OD/IOD/LSQ end-to-end fixtures without Python table construction.
- Network parser fixtures without live HTTP.
- Product/export fixtures for SPK, OEM, ADES, and OpenSpace outputs.

## 21. Rollout plan

1. **RM-STANDALONE-003:** implement prototype `TimeArray`, `CoordinateBatch`, `CovarianceBatch`, `OrbitBatch`, and Arrow adapters as modules in current crates.
2. **RM-STANDALONE-004:** implement or prototype ERFA/SOFA FFI time conversion and document Rust-native replacement candidates.
3. **RM-STANDALONE-005:** promote `adam_core_rs_spice` service APIs for origins, frames, and observers.
4. **RM-STANDALONE-006:** define `PropagatorBackend` and lift existing 2-body kernels behind it.
5. **RM-STANDALONE-007:** integrate `assist-rs` behind the trait while keeping backend pluggability.
6. **RM-STANDALONE-008+**: port OD/IOD/LSQ and observation/photometry/product workflows on top of typed batches.

## 22. Acceptance criteria for this RFC

- Names and describes the canonical Rust batch types needed for standalone workflows.
- Records Rust-schema ownership and Python/quivr adapter posture.
- Records ERFA/SOFA FFI as the initial time strategy with a Rust-native follow-up path.
- Keeps `assist-rs` preferred but backend-pluggable.
- Covers covariance, variants, chunking, provenance, Arrow/Python adapters, network clients, and plotting/product outputs.
- Defines concrete next implementation tasks and validation gates.
