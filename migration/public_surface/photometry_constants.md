# Public-surface audit: photometry, bandpasses, and constants

Audit date: 2026-07-11  
Parent bead: `personal-cmy.37.6`  
Scope: every public class, function, constant, and relevant inherited operation under `src/adam_core/photometry` plus `src/adam_core/constants.py`.

## Rules

Only plotting/display may remain Python. Generic quivr/dataclass behavior and static constants are classified separately rather than counted as Rust migration credit. Each adam-core-owned non-plotting operation must execute in Rust or be a thin one-crossing compatibility veneer, with pinned-legacy parity and qualifying Rust-owned timing where deterministic computation is material.

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

The raw six photometry kernel lanes remain independently timed. A complete-facade adapter additionally times prediction, single fit, and grouped fit with Rust-owned observer generation and Arrow assembly inside each `Instant` sample and PyO3 conversion outside it.

## Rotation-period surface

The rotation integration added 31 public symbols and 12 public constants, bringing the pre-`9b756803` manifest to 576 symbols / 66 constants. After the Obs80/Scout/Trajectory integration, the authoritative complete manifest is 595 symbols / 67 constants. `estimate_rotation_period` now makes one native crossing: Rust owns time-scale/light-time conversion, distance reduction, frequency-grid construction, clipped Fourier fits, order selection, F-distribution thresholds, alias clustering, session-offset selection, confidence classification, and result diagnostics. `exact_evaluation_backend="numpy"` and `"jax"` remain accepted compatibility selectors but both use the default-artifact Rust engine; JAX is not imported.

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
