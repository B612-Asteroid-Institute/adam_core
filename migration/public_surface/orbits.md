# `adam_core.orbits` public-surface audit

Audit date: 2026-07-10  
Scope bead: `personal-cmy.37.1`  
Audited tree: every non-test Python module below `src/adam_core/orbits`, including APIs which are public by Python naming convention but are not re-exported by a package `__init__`. This inventory was produced from the source modules, not from the selected 44-API parity registry.

## Classification and evidence legend

Implementation:

- **Rust/Arrow facade** — one public Python-to-Rust crossing; Rust owns the domain operation and finished Arrow result.
- **Rust/NumPy facade** — one crossing to a flat Rust kernel; Python only extracts inputs/maps the result.
- **Mixed** — a Rust kernel/parser/writer exists, but adam-core-owned control flow, grouping, conversion, or result assembly remains in Python, or the operation makes multiple crossings.
- **Python** — adam-core-owned implementation has no Rust domain implementation.
- **External/generic** — quivr, PyArrow, dataclass, Enum, HTTP client, plotting, or other unchanged infrastructure; it is not evidence that the orbit domain operation was migrated.

Evidence:

- **Parity: pass** means a direct legacy/frozen-fixture comparison exists for the named behavior.
- **Parity: partial** means only a kernel, parser, representative fixture, or comparison against another current implementation is covered.
- **Parity: unit-only/absent** is not legacy parity.
- **Native timing: measured** means `std::time::Instant` is inside Rust around the qualifying operation. Python wall-clock timing is not native timing.
- **Native timing: missing** means the operation qualifies once migrated but has no such measurement.
- **Native timing: N/Q** means it is schema/transport/external I/O/generic infrastructure or plotting rather than qualifying Rust-owned domain work.

## Package exports and immediate compatibility finding

`adam_core.orbits.__all__` declares `Ephemeris`, `Orbits`, `VariantOrbits`, and `VariantEphemeris`, but `VariantEphemeris` is not imported into the package namespace. `from adam_core.orbits import VariantEphemeris` therefore does not satisfy the declared surface. `query.__init__` exports `query_horizons`, `query_neocc`, `query_sbdb`, `query_sbdb_new`, and `query_scout`, but not the public `query_horizons_ephemeris`, `get_scout_objects`, or `scout_orbits_to_variant_orbits`. `openspace.__init__` exports only the two `create_renderable_*` helpers. These differences are documented rather than repaired in this audit.

## Table schemas, constructors, columns, and inherited quivr operations

The following table classes are public data models. Their declared columns are public descriptors/properties, and their inherited constructor/table operations are part of the usable surface.

| Class | Public columns/properties |
|---|---|
| `Orbits` | `orbit_id`, `object_id`, `coordinates`, `physical_parameters` |
| `VariantOrbits` | `orbit_id`, `object_id`, `variant_id`, `weights`, `weights_cov`, `coordinates`, `physical_parameters` |
| `Ephemeris` | `orbit_id`, `object_id`, `coordinates`, `predicted_magnitude_v`, `alpha`, `light_time`, `aberrated_coordinates` |
| `VariantEphemeris` | `orbit_id`, `object_id`, `variant_id`, `weights`, `weights_cov`, `coordinates`, `aberrated_coordinates`, `predicted_magnitude_v`, `alpha`, `light_time` |
| `PhysicalParameters` | `H_v`, `H_v_sigma`, `G`, `G_sigma`, `sigma_eff`, `chi2_red` |
| `ScoutObjectSummary` | `unc`, `lastRun`, `dec`, `H`, `moid`, `geocentricScore`, `ra`, `rating`, `tisserandScore`, `uncP1`, `ieoScore`, `rate`, `rmsN`, `Vmag`, `neoScore`, `nObs`, `objectName`, `phaScore`, `tEphem`, `arc`, `caDist`, `elong`, `vInf`, `neo1kmScore` |
| `ScoutOrbit` | `idx`, `epoch`, `ec`, `qr`, `tp`, `om`, `w`, `inc`, `H`, `dca`, `tca`, `moid`, `vinf`, `geoEcc`, `impFlag` |

Relevant operations inherited by all seven `qv.Table` subclasses were checked against installed quivr 0.8.1:

- construction/embedding: the generated keyword constructor, `from_kwargs`, `from_pyarrow`, `from_parquet`, `from_feather`, `from_csv`, `from_dataframe`, `from_flat_dataframe`, `empty`, `nulls`, and `as_column`;
- access/introspection/validation: `table`, `schema`, column descriptors, attributes, `attributes`, `column`, `chunk_counts`, `flattened_table`, `fragmented`, `invalid_mask`, `null_mask`, `is_valid`, `validate`, `__len__`, row slicing/indexing with `__getitem__`, and row iteration with `__iter__`;
- relational/table manipulation: `select`, `apply_mask`, `where`, `take`, `sort_by`, `set_column`, `with_table`, `unique_indices`, `drop_duplicates`, and `separate_invalid`;
- serialization/conversion: `to_csv`, `to_parquet`, `to_feather`, `to_dataframe`, `to_structarray`, and direct Arrow access through `table`.

Status: **External/generic; parity and native timing N/Q.** These unchanged quivr operations must remain compatible, but they are not adam-core-owned Rust migration credit. The nested Orbit/Variant Rust codecs and bridge tests demonstrate schema transport (including covariance, time metadata, origin, and physical parameters); they do not replace or benchmark generic quivr operations. `PhysicalParameters` currently adds no adam-core methods beyond its schema, so there is no separate domain algorithm to port.

## Core orbit and ephemeris classes

| Public API | Implementation status | Parity status | Native timing | Disposition |
|---|---|---|---|---|
| `Orbits(...)` / inherited constructors and columns | External/generic quivr; nested Rust `OrbitBatch` codec exists | Bridge round-trip pass; constructor semantics unit-only | N/Q | Keep as compatible data veneer |
| `Orbits.group_by_orbit_id()` | **Python** generator: `unique`, PyArrow mask, `apply_mask` per group | Unit behavior is exercised indirectly; no legacy parity lane | missing | Gap |
| `Orbits.dynamical_class()` | **Mixed**: Python coordinate conversion followed by separate `calc_orbit_class` Rust/NumPy crossing | Classification kernel random parity passes; complete method only unit-covered | missing | Gap; fuse table conversion/classification/result in one crossing |
| `Orbits.preview(propagator)` | Python Plotly display wrapper | unit/legacy parity absent | N/Q | Plotting waiver |
| `Ephemeris(...)` / columns | External/generic quivr | schema/unit coverage | N/Q | Compatible data veneer |
| `Ephemeris.link_to_observers(observers, precision="ns")` | **Python** rescale/round/link/warning composition | Current unit tests cover precisions and warning, not pinned legacy parity | missing | Gap |
| `VariantOrbits(...)` / columns | External/generic quivr; Rust variant codec exists | bridge schema tests pass | N/Q | Compatible data veneer |
| `VariantOrbits.create(orbits, method="auto", num_samples=10000, alpha=1, beta=0, kappa=0, seed=None)` | **Rust/Arrow facade**; one RecordBatch crossing, Rust sampling and finished table assembly | pass for sigma points; Monte Carlo statistical contract and intentional RNG difference documented/tested | **measured** (`sample_orbit_variants_record_batch`) | Complete |
| `VariantOrbits.link_to_orbits(orbits)` | **Python** `MultiKeyLinkage` composition | no direct legacy parity | missing | Gap |
| `VariantOrbits.collapse(orbits)` | **Python** row loop, linkage selection, weighted covariance, repeated table assembly | unit-only | missing | Gap |
| `VariantOrbits.collapse_by_object_id()` | **Python** grouping, averaging, covariance and table assembly | unit-only | missing | Gap |
| `VariantEphemeris(...)` / columns | External/generic quivr | schema/unit coverage | N/Q | Compatible data veneer; package export is broken |
| `VariantEphemeris.link_to_ephemeris(ephemeris)` | **Python** `MultiKeyLinkage` composition | no direct legacy parity | missing | Gap |
| `VariantEphemeris.collapse(ephemeris)` | **Python** row loop and two covariance paths | unit-only | missing | Gap |
| `VariantEphemeris.collapse_by_object_id(aberration_mode="recompute", group_chunk_size=200000)` | **Mixed**: extensive Python sort/group/weighted circular statistics/table composition; only light-time subkernel is Rust | unit tests cover grouping, longitude wrap, weights and aberration modes; no legacy parity | missing | Gap |
| `VariantEphemeris.collapse_sigma_points_orbit_major(n_times, n_variants=13)` | **Python/NumPy** specialized reshape/einsum/table assembly | partial: compared only with current generic Python collapse | missing | Gap |

## Classification

| Public API | Implementation status | Parity status | Native timing | Disposition |
|---|---|---|---|---|
| `classification.calc_orbit_class(elements)` | **Rust/NumPy facade**: Python extracts `a/e/q/Q`, one Rust rules call, Python maps integer codes to labels | random-fuzz kernel parity pass and public Keplerian/Cometary unit tests | **missing**; current evidence explicitly says no Rust-internal adapter (`personal-98v.1`) | Gap is native timing only; implementation shape is acceptable |
| `classification.CLASS_CODE_TO_NAME` | Public module constant by naming convention | exact mapping exercised by classification tests | N/Q | Retain compatibility |

## Arrow bridge

These names are public by Python naming convention even though `arrow_bridge` is migration infrastructure rather than a package export.

| Public API | Status |
|---|---|
| `orbits_to_ipc`, `orbits_from_ipc`, `variants_to_ipc`, `variants_from_ipc`, `observers_to_ipc`, `observers_from_ipc`, `coordinates_to_ipc` | Python metadata/IPC transport veneers. Exact nested round-trip tests exist. Native timing N/Q: serialization is the boundary, not a domain workload. |
| `orbits_to_record_batch`, `orbits_from_record_batch`, `variants_to_record_batch`, `variants_from_record_batch`, `observers_to_record_batch` | Python Arrow C Data Interface veneers. Exact round-trip/typed workflow tests exist. Native timing N/Q. |
| `round_trip_orbits`, `round_trip_observers`, `round_trip_orbits_zero_copy` | One-crossing Rust codec diagnostics with exact round-trip tests. These are diagnostic transport APIs, not legacy domain APIs; native timing N/Q. |

Private candidate functions (`_evaluate_residuals_2body_ipc_candidate`, `_fit_orbit_least_squares_2body_candidate`, `_rotate_orbits_frame_ipc_candidate`, `_sample_orbit_variants_arrow`, `_propagate_orbits_typed_arrow`, `_propagate_orbits_2body_ipc_candidate`) were inspected to establish the public facades' backend status but are excluded from the public inventory because of their leading underscore.

## SPICE kernel helpers

| Public API | Implementation status | Parity status | Native timing | Disposition |
|---|---|---|---|---|
| `fit_chebyshev(coordinates, window_start, window_end, degree, mid_time=None, half_interval=None)` | one Rust IPC crossing; Rust owns selection, units, basis, and SVD minimum-norm fit | frozen pinned-legacy NumPy coefficients for over- and underdetermined windows | shared Rust fit/product timing | Complete veneer |
| `orbits_to_spk(orbits, output_file, start_time, end_time, propagator=None, max_processes=None, step_days=0.25, target_id_start=1000000, window_days=32, comment=..., kernel_type="w03")` | one fused Rust product crossing without provider; optional propagation is one declared provider call followed by the same crossing | Type 3 ASSIST and Type 9 no-provider readback, mappings/order/errors, atomicity, and >25-segment DAF chain | Rust `Instant` p50 lanes | Complete veneer/provider boundary |
| `write_spkw03_segment(propagated_orbit, handle, target_id, start_time, end_time, window_seconds=86400, cheby_degree=15)` | one Rust IPC crossing into native segment fit/writer | direct shim plus product Type 3 readback | shared native timing | Complete veneer |
| `write_spkw09_segment(propagated_orbit, handle, target_id, start_time, end_time)` | one Rust IPC crossing into native state/time conversion and segment writer | direct shim and exact sampled-state readback | shared native timing | Complete veneer |
| `DEFAULT_KERNELS`, `J2000_TDB_JD` | Public constants | N/A | N/Q | Retain |

## OEM helpers

| Public API | Implementation status | Parity status | Native timing | Disposition |
|---|---|---|---|---|
| `orbit_to_oem(orbits, output_file, originator="ADAM CORE USER")` | **Mixed**: Python validation, transform, sort, metadata/covariance assembly; one Rust KVN writer call | strong frozen-legacy writer fixture plus behavioral tests | missing | Gap: move adam-core-owned preparation into one typed Rust crossing |
| `orbit_to_oem_propagated(orbits, output_file, times, propagator_klass, originator=...)` | **Mixed**: Python propagator lifecycle plus same Python preparation/Rust writer | behavioral tests; writer core inherits fixture parity, complete facade parity partial | missing | Gap |
| `orbit_from_oem(input_file)` | **Mixed**: Rust KVN parse, then Python per-state/per-covariance loops, unit conversions and table assembly | strong frozen-legacy parser fixture and round trip | missing | Gap: Rust should produce finished typed table/RecordBatch |
| `OEM_VERSION`, `REF_FRAME_VALUES`, `CCSDS_CENTER_NAME_VALUES` | Public constants | fixture-covered where relevant | N/Q | Retain |

Private frame/center mapping and `_write_oem_kvn` helpers are implementation details, but their remaining Python composition is included in the public facade classifications above.

## OpenSpace and Lua helpers

### Data types and constructors

Public dataclass constructors/properties:

- `Gui(*, name, path)`; `Asset(*, identifier, parent, gui, renderable=None, transform=None)`.
- `Resource(*, path)`.
- `Renderable(*, type, dim_in_atmosphere=None, enabled=None, opacity=None, render_bin_mode=None, tag=None)`.
- `RenderableOrbitalKepler` adds `color`, `format`, `path`, `segment_quality`, `type`, `contiguous_mode`, `enable_max_size`, `enable_outline`, `max_size`, `outline_color`, `outline_width`, `point_size_exponent`, `rendering`, `render_size`, `start_render_idx`, `trail_fade`, `trail_width`.
- `RenderableTrailOrbit` adds `color`, `period`, `resolution`, `translation`, `type`, `enable_fade`, `line_fade_amount`, `line_length`, `line_width`, `point_size`, `rendering`.
- `RenderableTrailTrajectory` adds `color`, `end_time`, `start_time`, `translation`, `type`, `accurate_trail_positions`, `enable_fade`, `enable_sweep_chunking`, `line_fade_amount`, `line_length`, `line_width`, `point_size`, `rendering`, `sample_interval`, `show_full_trail`, `sweep_chunk_size`, `time_stamp_subsample_factor`.
- `Translation(*, type)`; `Transform(*, translation)`; `KeplerTranslation` adds `argument_of_periapsis`, `ascending_node`, `eccentricity`, `epoch`, `inclination`, `mean_anomaly`, `period`, `semi_major_axis`, `type`; `SpiceTranslation` adds `observer`, `target`, `fixed_date`, `frame`, `type`.

Public enums and members:

- `RenderBinMode`: `BACKGROUND`, `OPAQUE`, `PREDEFERREDTRANSPARENT`, `OVERLAY`, `POSTDEFERREDTRANSPARENT`, `STICKER`.
- `RenderableOrbitalKeplerFormat`: `TLE`, `OMM`, `SBDB`.
- `RenderableOrbitalKeplerRendering`: `TRAIL`, `POINT`, `POINTS_TRAILS`.
- `RenderableTrailRendering`: `LINES`, `POINTS`, `LINES_POINTS`.

The dataclass/Enum construction and field access are **External/generic, parity/native timing N/Q** and are acceptable Python compatibility veneers. `LuaDict.to_string(indent=0)` and `Resource.to_string(indent=0)` serialize one prepared payload in one Rust call; representative frozen OpenSpace text fixtures pass, native timing is N/Q for this small formatting boundary. `LuaDict.to_pascal_case(s)` is now a direct Rust veneer with frozen legacy cases. Inherited `to_string` is public on every LuaDict subclass.

### Functions

| Public API | Implementation status | Parity status | Native timing | Disposition |
|---|---|---|---|---|
| `orbits_to_sbdb_file(orbits, path)` | one fused Rust transform/epoch/CSV/write crossing | frozen legacy panels: TDB/UTC, heliocentric and SSB/equatorial, quoting/float edge cases; exact current fallback bytes | Rust `Instant` p50 lanes | Complete veneer |
| `create_initialization(assets)` | one-call Rust string formatter | frozen text fixture pass | N/Q (formatting helper) | Complete veneer |
| `create_renderable_orbital_kepler(...)` | one fused Rust transform/CSV/model/render/staged multi-file write crossing | frozen whole-directory all-option fixture plus exact current fallback bytes | Rust `Instant` p50 lanes | Complete veneer |
| `create_renderable_trail_orbit(...)` | one fused Rust transform/per-orbit model/render/staged write crossing for Kepler and SPICE modes | frozen whole-directory Kepler/Spice fixtures, nullable IDs, snippets/errors, exact current fallback bytes | Rust `Instant` p50 lanes | Complete veneer |

The two asset creators' complete signatures are the source signatures; every keyword listed there is public and must remain compatible. They are grouped as single operations rather than pretending each keyword is a separate API.

## Network query helpers

External services remain integration boundaries, but adam-core-owned request construction, retries, chunking, response normalization, and typed result assembly must be Rust-owned. Only plotting/display is exempt from implementation migration.

| Public API | Implementation status | Parity status | Native timing | Disposition |
|---|---|---|---|---|
| `query_horizons_ephemeris(object_ids, observers)` | one-crossing Rust HTTP/protocol/parser/typed Ephemeris product | recorded response and live integration | deterministic processing measured | Complete |
| `query_horizons(object_ids, times, coordinate_type="cartesian", location="@sun", aberrations="geometric", id_type=None)` | one-crossing Rust HTTP, 50-epoch chunking, element conversion, sort, typed Orbits product | all three representations recorded + live chunk parity | deterministic processing measured | Complete |
| `query_neocc(object_ids, orbit_type="ke", orbit_epoch="present-day")` | one-crossing Rust HTTP/OEF/covariance/physical/Arrow product | recorded complete products and errors | measured | Complete |
| `query_sbdb(ids)` | one-crossing direct Rust SBDB client and typed product | recorded legacy-equivalent products | measured | Complete |
| `query_sbdb_new(ids, *, max_concurrent_requests=1, timeout_s=60, max_attempts=5, allow_missing=False, orbit_id_from_input=False)` | same Rust client with timeout/retry/fair-use/missing options | recorded option/error/physical parity | measured | Complete |
| `NotFoundError(message, object_id)`; properties `message`, `object_id`; `__str__` | Python compatibility exception over Rust missing-object signal | unit behavior | N/Q | Compatible veneer |
| `get_scout_objects()` | one-crossing Rust HTTP/summary Arrow product | recorded/live | measured | Complete |
| `scout_orbits_to_variant_orbits(object_id, scout_orbits)` | one-crossing Rust conversion and VariantOrbits assembly | recorded/synthetic | measured with Scout product | Complete |
| `query_scout(ids)` | one-crossing Rust HTTP fan-out/conversion/typed product | recorded/live | measured | Complete |
| `ScoutObjectSummary(...)`, `ScoutOrbit(...)` and columns | External/generic quivr schemas | schema tests | N/Q | Compatible data veneers |

## Plotting-only waiver

The following are inventoried but explicitly excluded from child migration beads:

- `plots.plot_orbit(orbit, propagator, start_time=None, logo=True)` — Python/Plotly display composition.
- `plots.ellipsoid(center, radii, rotation, num_points=100)` — Python/NumPy plot geometry helper.
- `plots.add_observation_plot(fig, observed, radius_mult)` — Python/Plotly display mutation.
- `Orbits.preview(propagator)` — display convenience wrapper.

Parity and native timing are N/Q under the accepted plotting/display waiver. Computational kernels reused outside display would need a separate audit if promoted to a non-plotting API.

## Gap-to-bead coverage

Concrete implementation beads cover every non-plotting gap identified above:

1. `personal-cmy.37.1.4`: core `Orbits` grouping/fused dynamical classification and classification native timing;
2. `personal-cmy.37.1.2`: all VariantOrbits/VariantEphemeris linkage and collapse operations;
3. `personal-cmy.37.1.3`: `Ephemeris.link_to_observers`;
4. `personal-cmy.37.4.4`: OEM read/write/propagated facades;
6. `personal-cmy.37.4.1`: Horizons facades;
7. `personal-cmy.37.4.2`: NEOCC, SBDB legacy/new, Scout facades, and exception semantics;
8. `personal-cmy.37.1.1`: package export compatibility.

`personal-cmy.37.4.5` closed the previously listed OpenSpace product and
`LuaDict.to_pascal_case` gaps.

Each implementation child must add direct pinned-legacy parity for its complete public facade and qualifying Rust-internal timing, not merely a Python benchmark or a test against another current Python path. Generic quivr/dataclass/Enum/transport operations and plotting are deliberately classified rather than silently counted as migrated.
