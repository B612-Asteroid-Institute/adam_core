# Public-surface audit: observations and source catalogs

Audit date: 2026-07-11  
Parent bead: `personal-cmy.37.7`  
Scope: every public class, function, custom method, and relevant inherited operation under `src/adam_core/observations`.

## Rules

Only plotting/display may remain Python. Unchanged quivr constructors, descriptors, slicing, selection, and serialization are **external/generic** and do not earn Rust migration credit. A completed adam-core domain operation must execute directly in Rust or use a compatible one-crossing Python veneer. Qualifying deterministic work requires pinned-legacy parity and Rust-owned `std::time::Instant` timing.

## Generic data models

`ADESObservations`, `Associations`, `PointSourceDetections`, `Exposures`, `Photometry`, and `SourceCatalog` are quivr tables. Their declared columns, generated constructors, and inherited table methods are generic quivr infrastructure. `ObservatoryObsContext`, `SubmitterObsContext`, `TelescopeObsContext`, `SoftwareObsContext`, and `ObsContext` are compatibility dataclasses; generated field access is generic, while their adam-core validation and serialization contracts remain governed. `STRING25` and `STRING100` are compatibility constants.

## ADES and Arrow transport

| Public API | Current ownership | Parity/timing | Disposition |
|---|---|---|---|
| `ObsContext.to_string` | one Rust string-rendering crossing after Python dataclass assembly | frozen fixture coverage; formatting timing N/Q | Compatible veneer |
| `ADES_to_string` | one fused Rust crossing owns UTC rescale, all context/observation rendering, validation, and final assembly | frozen writer fixture plus Rust-Instant fused timing | Compatible veneer |
| `ADES_string_to_tables` | one fused Rust crossing owns observation/context parsing, warning inputs, and nested Arrow encoding; Python reconstructs declared compatibility objects | frozen parser fixture plus Rust-Instant fused timing | Compatible veneer |
| `observations_to_ipc`, `observations_from_ipc`, `round_trip_observations` | migration transport/codec diagnostics | exact round trips; native timing N/Q | Keep classified, not domain migration credit |

## Associations, detections, and exposures

| Public API | Current ownership | Parity/timing | Disposition |
|---|---|---|---|
| `Associations.group_by_object` | one Rust grouping/IPC crossing (non-null first-appearance groups, null group last, original row order) | ordering unit parity plus Rust-Instant timing | Compatible veneer |
| `Associations.link_to_detections` | one-line key-column selection over external quivr `Linkage` machinery | linkage unit tests | Classified generic quivr boundary (no adam-core computation) |
| `PointSourceDetections.group_by_exposure` | one Rust grouping/IPC crossing, including the legacy null-ID empty-group semantics | ordering unit parity plus Rust-Instant timing | Compatible veneer |
| `PointSourceDetections.healpixels` | Rust `ang2pix` port of healpix_cxx (nest+ring, lonlat, near-pole branches, nside validation and exact healpy error) | exact healpy-oracle parity across random/structured/near-pole samples and both schemes; Rust-Instant timing | Rust-owned |
| `PointSourceDetections.group_by_healpixel` | one Rust crossing owns pixel assignment plus ascending-pixel grouping; numpy int64 keys preserved | healpy-oracle parity plus grouping unit tests and Rust-Instant timing | Compatible veneer |
| `PointSourceDetections.link_to_exposures` | one-line key-column selection over external quivr `Linkage` machinery | linkage unit tests | Classified generic quivr boundary (no adam-core computation) |
| `Exposures.group_by_observatory_code` | one Rust typed grouping/IPC crossing; Python yields compatibility tables | ordering/schema unit parity plus Rust-Instant timing | Compatible veneer |
| `Exposures.midpoint` | one Rust crossing owns half-even nanosecond conversion and epoch carry | exact midpoint unit parity plus Rust-Instant timing | Compatible veneer |
| `Exposures.observers` | one Arrow crossing owns midpoint epochs, code grouping, SPICE state dispatch, frame/origin handling, and nested observer assembly | observer fixture ULP parity plus Rust-Instant fused timing | Compatible veneer |

These gaps are covered by children of `personal-cmy.37.7` for observation grouping/linkage/HEALPix and exposure-derived workflows.

## `SourceCatalog` custom surface

- `detections`, `associations`, and `photometry`: computation-free declarative column re-projections through generic validated quivr constructors. There is no kernel to migrate or time; classified as generic-projection compatible veneers (no Rust migration credit claimed).
- `exposures`: the projection/validation stays the generic constructor (preserving the legacy validate-before-dedupe order); the keep-first dedupe by exposure ID is one Rust crossing (`exposures_drop_duplicate_ids_ipc`) with Rust-Instant timing.
- `coordinates`: arcsecond-to-degree conversion and NaN-filled (N, 6, 6) covariance assembly are one Rust crossing in legacy IEEE order (`radec_covariance_matrices_numpy`) with Rust-Instant timing; nulls arrive as NaN exactly like the legacy NumPy conversion. The surrounding `SphericalCoordinates.from_kwargs` (null rho/velocity columns) is the generic constructor.
- `observers`: `exposure_midpoint=True` with null-free duration/start columns uses the fused Rust midpoint+observer crossing (shared with `Exposures.observers`, Rust-Instant timed); coverage failures and null-bearing inputs fall back to the exact legacy composition, whose `Observers.from_codes` is itself a single Rust crossing. `exposure_midpoint=False` is a zero-computation veneer over the Rust-owned `Observers.from_codes`. Fused-vs-legacy equality is asserted in tests.
- `healpixels`: dispatches to the Rust healpix_cxx port with exact healpy-oracle parity (fixture pixel values retained).

Row order, nulls, dedupe order, covariance units, midpoint arithmetic, schemas, metadata, and exceptions are preserved; deviations would fail the retained legacy-valued unit tests.

## Plotting and inherited operations

This domain defines no plotting APIs. Therefore no adam-core-owned method above is exempt. Generic inherited quivr operations remain compatible external infrastructure and are explicitly excluded from migration credit, parity microbenchmarks, and native timing.

## Closure

The observations domain is complete. The ADES product task (`personal-cmy.37.4.3`) and all three observation-domain implementation children (`personal-cmy.37.7.1`, `.37.7.2`, `.37.7.3`) are closed: every deterministic grouping, projection, HEALPix, time, and observer workflow executes in Rust behind a one-crossing veneer with parity coverage (frozen legacy fixtures, healpy-oracle equality, legacy-valued unit tests, fused-vs-legacy assertions) and Rust-Instant timing. The remaining Python is limited to declared compatibility wrapping, computation-free generic quivr projections/constructors, generic `Linkage` key selection, and explicitly retained legacy fallbacks for space-based/unknown observatory codes and null-bearing inputs.
