# Public-surface audit: photometry, bandpasses, and constants

Audit date: 2026-07-11  
Parent bead: `personal-cmy.37.6`  
Scope: every public class, function, constant, and relevant inherited operation under `src/adam_core/photometry` plus `src/adam_core/constants.py`.

## Rules

Only plotting/display may remain Python. Generic quivr/dataclass behavior and static constants are classified separately rather than counted as Rust migration credit. Each adam-core-owned non-plotting operation must execute in Rust or be a thin one-crossing compatibility veneer, with frozen updated-upstream parity and qualifying Rust-owned timing where deterministic computation is material.

## Data models and constants

`GroupedPhysicalParameters`, `BandpassCurves`, `ObservatoryBandMap`, `AsteroidTemplates`, `TemplateBandpassIntegrals`, `RotationPeriodObservations`, `RotationPeriodResult`, and `GroupedRotationPeriodResults` are quivr schemas. Their columns, generated constructors, and inherited table operations are external/generic. `TemplateSpec` and the private rotation-fit dataclasses are compatibility data containers. `TEMPLATE_SPECS`, `ATLAS_MPC_CODES`, `HARMONIC_FACTORS`, rotation confidence thresholds, `KM_P_AU`, `S_P_DAY`, `DE44X_CONSTANTS`, and `DE44X` are static compatibility data. These require export/value tests but not native timing and do not count as migrated algorithms.

## Governed numeric magnitude surface

| Public API | Current ownership | Parity | Native timing | Disposition |
|---|---|---:|---:|---|
| `calculate_phase_angle` | one Rust NumPy crossing after compatible validation/extraction | selected | measured | Acceptable veneer |
| `calculate_apparent_magnitude_v` | one Rust NumPy crossing after compatible validation/broadcast | selected | measured | Acceptable veneer |
| `calculate_apparent_magnitude_v_and_phase_angle` | one Rust NumPy crossing | selected | measured | Acceptable veneer |
| `predict_magnitudes` | ordinary exposure path crosses once for midpoint observer states, filter indexing, reference/composition conversion, geometry validation, and prediction; explicit observer overrides retain a compatible provider fallback | selected facade and complete-path tests | complete Rust `Instant` sample | Complete |
| `convert_magnitude` | one Arrow-to-Rust crossing for validation, lookup, composition, and arithmetic; object-string conversion no longer allocates Python `list[str]` rows | selected | measured | Complete |
| `estimate_absolute_magnitude_v_from_detections` | one ordinary-path crossing owns exposure-ID join, midpoint observer states, band mapping, prediction, filtering, fit, nulls, and Arrow `PhysicalParameters` assembly | selected facade, complete join, and Arrow-schema tests | complete Rust `Instant` sample | Complete |
| `estimate_absolute_magnitude_v_from_detections_grouped` | one ordinary-path crossing additionally owns null-ID filtering, lexical grouping/order, degenerate-group removal, and nested Arrow assembly | selected facade, grouped ordering, and nested-schema tests | complete Rust `Instant` sample | Complete |
| `estimate_colors` | public `adam_core::estimate_colors` owns the complete high-level typed workflow; Python makes one private IPC crossing that only decodes, calls that public API once, and encodes its typed `ColorFitBatch` | direct public-Rust API/adapter-structure tests plus frozen outputs for six fixtures across all three phase laws and generic-filter, projection, status/count, and exact-schema facade tests | scaling reviewed externally; no in-tree timing surface | Complete |

The raw six photometry kernel lanes remain independently timed. A complete-facade adapter additionally times prediction, single fit, and grouped fit with Rust-owned observer generation and Arrow assembly inside each `Instant` sample and PyO3 conversion outside it.

## Rust-native color-estimation contract (PR #202 integration)

`estimate_colors` is a one-row-per-object API over native `SourceCatalog` and `Orbits`, exposed under the same public name in Python and the publishable `adam_core` Rust crate. Public Rust accepts typed `SourceCatalogBatch`/`OrbitBatch`, aligned scientific options, and explicit runtime resources, then returns a typed `ColorFitBatch`. Its production Python facade performs only schema definition and IPC/configuration marshalling, makes exactly one private IPC call, decodes exactly one batch, verifies that the raw `RecordBatch` schema exactly equals `ColorFit.schema` including metadata, and wraps it with the shared quivr bridge. The private PyO3 adapter only decodes transport values, calls public `adam_core::estimate_colors` exactly once, and encodes its typed result. Public Rust owns setup policy, grouping, orbit handling, geometry/propagation integration, canonical filter resolution, validity classification, optional projection and composition conversion, fitting, clipping, rank/null-space identifiability, covariance, pair colors, statuses/counts, and high-level result assembly.

A nonempty orbit table is validated call-wide for Cartesian representation, ecliptic frame, and convertible epochs. An empty orbit table bypasses only those inapplicable validations and emits one isolated `no_orbit` row for every attributable observation object. Matching-orbit heliocentric origin and active non-gravitational parameters are per-object validations that isolate `failed` rows. Call-level time/schema errors preserve their operation context. Per-object errors deliberately use clean Rust-native messages without Python exception-class prefixes, and invalid observatory codes do not repeat a wrapper code.

Rust exposes `ColorFitStatus` as a documented non-exhaustive enum and serializes it through stable `as_str`/`Display` values. `ColorEstimationError` is likewise non-exhaustive. Published `adam_core_rs_coords` artifacts compile all five canonical bandpass Parquet tables into the library; `bundled_bandpass_data()` parses and process-caches them from `include_bytes!` without a runtime crate-source path. `ResolvedColorEstimationResources` combines that embedded bundle with `adam_core_rs_kernel_data::Resolver` for default kernels and MPC obscodes, while explicit custom directories remain path-backed. Resources populate process-global state: kernels accumulate in load order, the first loaded nonempty obscodes table wins, and each estimate retains the SPICE read guard. Identical concurrent first initialization is serialized and idempotent, while later mutation must be completed before concurrent science. UTC rescaling must be warmed on one thread before concurrent UTC calls until the separately tracked ERFA first-use race is corrected.

### Filter resolution and eligibility

Strict and non-strict modes share the magnitude resolver's precedence: canonical filter ID, exact `(observatory_code, reported_band)` mapping, then optional generic SDSS/PS1 fallback. Color estimation differs at the failure boundary: null, unresolved, and strict-disabled fallback filters are skipped into `num_unsupported_filter` rather than raising for the batch.

Native mode is the default and fits every finite-valid observation in each resolved canonical filter as a separate intercept. It replaces the retired legacy solver's same-letter `g/i/r/u` pooling and allow-list. The retention denominator is exactly the valid, mapped native rows. If that set is empty, status is `no_usable_photometry`.

Projection mode receives an explicit canonical source-to-target map and a required `BandpassComposition`. Only sources in that map are eligible; Rust performs the composition conversion and pools rows after projection. Unprojected source filters count as unsupported, and the retention denominator is exactly the projected eligible rows. There is no implicit uncorrected pooling. LSST `u/g/r/i` is an optional explicit projection executed in the same fused native call, never a kernel assumption or Python transformation.

For every object that reaches row classification, invalid magnitude/uncertainty takes precedence over unsupported filter, and the aggregate partition remains exact:

`num_obs = num_invalid + num_unsupported_filter + num_clipped + num_used`.

### Native `ColorFit` list schema and numeric policy

The following top-level columns are nullable Arrow `large_list` fields whose item field uses the stated exact primitive type:

- `filter_ids: large_list<item: large_string not null>`;
- `absolute_magnitudes`, `absolute_magnitude_sigmas`, and `magnitude_covariance: large_list<item: float64 not null>`;
- `num_used_by_filter: large_list<item: int64 not null>`;
- `color_lhs_filter_ids`, `color_rhs_filter_ids: large_list<item: large_string not null>`;
- `colors`, `color_sigmas: large_list<item: float64 not null>`.

List elements themselves are never Arrow null. String and integer elements are present; floating-point non-identifiability is represented by NaN. Top-level null lists encode status as specified below. The raw Rust `RecordBatch` must match the complete final `ColorFit.schema` field order, primitive/list types, and top-level/item nullability before quivr wrapping.

`filter_ids` contains only final active filters with at least one used row; fully clipped filters are absent. IDs are sorted by deterministic byte-wise/code-point lexical canonical-ID order. Magnitudes, magnitude sigmas, and used counts align element-for-element with those IDs, and `sum(num_used_by_filter) == num_used`. An individually non-fitable H and its sigma are NaN.

For `N = len(filter_ids)`, `magnitude_covariance` is the row-major flattened `N x N` H sub-block. Rust masks the full row and column of every individually non-fitable H to NaN in this exposed sub-block. When `dof <= 0`, the entire exposed covariance and every magnitude/phase/color sigma are NaN.

Rust enumerates every pair `(i, j)` with `i < j` in the canonical-ID order. The lhs ID, rhs ID, color, and color-sigma lists all have length `N*(N-1)/2` and align element-for-element. IDs—not conventional band names—define the sign: `color = raw H_lhs - raw H_rhs`. Each contrast has its own null-space fitability test and uses the raw solution plus unmasked full-parameter covariance, so a finite color and sigma may be emitted even when both exposed individual H values are NaN. A non-fitable contrast is represented by NaN floating elements, not null list elements.

### Status/list contract

- `ok`: `filter_ids`, `absolute_magnitudes`, `absolute_magnitude_sigmas`, `num_used_by_filter`, `magnitude_covariance`, and all four pair lists are non-null. With one active filter, all four pair lists are non-null empty lists. `force_g_bounds=False` always follows ordinary `ok` behavior even when the fitted phase parameter is outside nominal bounds.
- `phase_param_out_of_bounds` with forced bounds: retain non-null `filter_ids` and `num_used_by_filter` diagnostics, but set magnitude, magnitude-sigma, covariance, and every pair list field to top-level null.
- `no_usable_photometry`, `too_many_outliers`, `no_orbit`, `multiple_orbits`, and `failed`: every native-product list field is top-level null.

Forced bounds are `G12* in [0, 1]` for `HG12star` and `G in [-0.255, 1]` for `HG`; `c1c2` has no forced-bound status. Status is always present and error is nullable. Every attributable terminal row retains `num_obs`. `no_orbit`, `multiple_orbits`, and failures before classification have null classification counts; classified rows retain the exact count partition, and `num_used` on an attempted-fit failure means rows admitted to that fit. Phase and fit diagnostics remain only when the corresponding fit ran. The fixed `g/i/r/u` magnitude, sigma, filter, color, and per-band count columns have been removed. Native and optional projected outputs use the same generic list schema.

Rust tests—not Python computation—establish native-mode numerical parity where applicable and enforce native/projection ordering, null/NaN, covariance, contrast-fitability, status, retention, and count invariants. Public facade tests cover all six fixtures and three phase laws, one-call marshalling, generic filters, complete explicit projection (including identity edges), strict fallback accounting, multi-object order/failures, UTC/frame/non-gravitational behavior, exact schema equality, list-child nullability, and one-batch IPC rejection.

## Rotation-period surface

The rotation integration added 31 public symbols and 12 public constants, bringing the pre-`9b756803` manifest to 576 symbols / 66 constants. The Obs80/Scout/Trajectory integration then reached 595 symbols / 67 constants; after the current upstream reconciliation, the authoritative complete manifest is 629 symbols / 78 constants. `estimate_rotation_period` now makes one native crossing: Rust owns time-scale/light-time conversion, distance reduction, frequency-grid construction, clipped Fourier fits, order selection, F-distribution thresholds, alias clustering, session-offset selection, confidence classification, and result diagnostics. `exact_evaluation_backend="numpy"` and `"jax"` remain accepted compatibility selectors but both use the default-artifact Rust engine; JAX is not imported.

`estimate_rotation_period_best_apparition` performs grouping, per-apparition solves, error candidates, ranking, and flagging in one Rust crossing. `estimate_rotation_period_from_detections_grouped` performs lexical grouping and all per-object solves in one grouped Rust estimator crossing after the Rust observation-construction boundary. The ordinary observation builder owns exposure alignment, midpoint observer states, ecliptic geometry, phase angle, UTC observing-night/session IDs, validation, and TDB output in Rust. A whole-call compatibility path remains only when `Exposures.observers` is explicitly overridden or an unsupported coordinate frame requires provider composition.

The public validation helpers `relative_error_pct`, `harmonic_adjusted_error_pct`, `alias_bucket`, `within_tolerance`, and `near_day_alias` are thin Rust veneers. Fixed synthetic/LCDB fixtures, grouped ordering/failure tests, best-apparition tests, NumPy/JAX-selector compatibility, and Rust-owned `Instant` samples cover the rotation surface.

## Runtime bandpass APIs

The following already call Rust once for their substantive runtime algorithm, with Python argument/result adaptation: `map_to_canonical_filter_bands`, `assert_filter_ids_have_curves`, `get_integrals`, `compute_mix_integrals`, `bandpass_delta_mag`, `bandpass_color_terms`, and `register_custom_template`. Their compatible exception, cache, custom-registry, and numeric semantics still need complete manifest entries and facade parity.

The four public loaders retain bounded Python cache policy but perform package-data Parquet I/O in one Rust crossing and directly wrap Rust Arrow batches. Filter discovery, mapping, integral lookup/mixing, delta/color terms, magnitude conversion, custom-template registration/clearing, validation, and the process-wide registry are Rust-owned. Private composition/cache helpers are bounded compatibility policy over those Rust operations, not independent public algorithms.

## Vendoring/product builders

`build_bandpass_curves`, `build_observatory_band_map`, `build_asteroid_templates`, `build_solar_spectrum`, and `build_template_bandpass_integrals` are one-crossing Rust product veneers. Rust owns SVO and STScI HTTP, VOTable TABLEDATA and FITS BINTABLE parsing, curve normalization, template math, Parquet input loading, interpolation/integration, Arrow assembly, and atomic publication. Python only converts specs and wraps returned Arrow batches. Recorded SVO/FITS responses, frozen canonical Parquet products, malformed/duplicate/empty inputs, missing-directory atomicity, and Rust-owned `Instant` timing are gated under `personal-cmy.37.6.2`.

## Plotting

No photometry API in this audited tree is plotting/display. Consequently there are no Python implementation waivers in this domain.

## Closure

All adam-core-owned non-plotting photometry operations are Rust-backed. Public Python code is limited to validation/broadcast compatibility, bounded cache policy, quivr wrapping, and explicit observer-provider fallback; static constants and generic schemas remain classified rather than counted as migration credit.
