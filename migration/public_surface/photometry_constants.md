# Public-surface audit: photometry, bandpasses, and constants

Audit date: 2026-07-11  
Parent bead: `personal-cmy.37.6`  
Scope: every public class, function, constant, and relevant inherited operation under `src/adam_core/photometry` plus `src/adam_core/constants.py`.

## Rules

Only plotting/display may remain Python. Generic quivr/dataclass behavior and static constants are classified separately rather than counted as Rust migration credit. Each adam-core-owned non-plotting operation must execute in Rust or be a thin one-crossing compatibility veneer, with pinned-legacy parity and qualifying Rust-owned timing where deterministic computation is material.

## Data models and constants

`GroupedPhysicalParameters`, `BandpassCurves`, `ObservatoryBandMap`, `AsteroidTemplates`, and `TemplateBandpassIntegrals` are quivr schemas. Their columns, generated constructors, and inherited table operations are external/generic. `TemplateSpec` is a frozen compatibility dataclass. `TEMPLATE_SPECS`, `ATLAS_MPC_CODES`, `KM_P_AU`, `S_P_DAY`, `DE44X_CONSTANTS`, and `DE44X` are static compatibility data. These require export/value tests but not native timing and do not count as migrated algorithms.

## Governed numeric magnitude surface

| Public API | Current ownership | Parity | Native timing | Disposition |
|---|---|---:|---:|---|
| `calculate_phase_angle` | one Rust NumPy crossing after compatible validation/extraction | selected | measured | Acceptable veneer |
| `calculate_apparent_magnitude_v` | one Rust NumPy crossing after compatible validation/broadcast | selected | measured | Acceptable veneer |
| `calculate_apparent_magnitude_v_and_phase_angle` | one Rust NumPy crossing | selected | measured | Acceptable veneer |
| `predict_magnitudes` | **mixed**: filter validation, `Exposures.observers` multi-step workflow, mappings, and table extraction precede one Rust kernel | selected kernel/facade lane, but complete orchestration is not one crossing | measured kernel only | Rust facade gap |
| `convert_magnitude` | Python/PyArrow mapping and NumPy arithmetic over Rust-owned tables | ordinary tests; native missing | missing | Rust gap |
| `estimate_absolute_magnitude_v_from_detections` | **mixed** exposure join, band mapping, observer prediction, filtering, Rust row fit, and Python result assembly | selected row kernel only | row kernel measured | Rust facade gap |
| `estimate_absolute_magnitude_v_from_detections_grouped` | **mixed** Python join/sort/group/output assembly around Rust grouped fit | selected grouped kernel only | grouped kernel measured | Rust facade gap |

The six selected photometry kernel lanes now have Rust-owned `Instant` adapters. That evidence does not make the larger prediction/fitting facades one-crossing operations.

## Runtime bandpass APIs

The following already call Rust once for their substantive runtime algorithm, with Python argument/result adaptation: `map_to_canonical_filter_bands`, `assert_filter_ids_have_curves`, `get_integrals`, `compute_mix_integrals`, `bandpass_delta_mag`, `bandpass_color_terms`, and `register_custom_template`. Their compatible exception, cache, custom-registry, and numeric semantics still need complete manifest entries and facade parity.

Remaining Python/mixed runtime surfaces:

- `load_bandpass_curves`, `load_observatory_band_map`, `load_asteroid_templates`, and `load_template_integrals` perform package-data discovery and quivr Parquet loading in Python.
- `bandpass_filter_id_table` performs Python package-data lookup, Arrow/table mapping, and caching around Rust data access.
- `bandpass_integrals_for_composition`, `bandpass_composition_key`, `bandpass_delta_table_for_composition`, and its cached variant retain Python branching, normalization, cache policy, and output adaptation.

Children of `personal-cmy.37.6` cover end-to-end prediction/fitting, magnitude/composition helpers, and Rust-owned bandpass loading/registry behavior.

## Vendoring/product builders

`build_bandpass_curves`, `build_observatory_band_map`, `build_asteroid_templates`, `build_solar_spectrum`, and `build_template_bandpass_integrals` remain Python implementations. They own HTTP, VOTable/FITS parsing, normalization, interpolation/integration, loops, quivr assembly, and Parquet writes. They are product/data-generation APIs, not plotting, so they are not exempt. A dedicated child requires Rust-owned end-to-end workflows, recorded service fixtures, frozen product parity, atomic file integration, errors, and typed results.

## Plotting

No photometry API in this audited tree is plotting/display. Consequently there are no Python implementation waivers in this domain.

## Closure

Photometry is numerically Rust-backed but not public-surface complete. Closure requires the four implementation children under `personal-cmy.37.6`, facade-level parity beyond raw kernels, explicit generic/static classifications, and Rust-Instant timing for every newly Rust-owned deterministic operation.
