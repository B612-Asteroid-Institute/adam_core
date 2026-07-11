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
| `Associations.group_by_object` | Python/PyArrow generator, including null-last policy | unit-only; native missing | Rust gap |
| `Associations.link_to_detections` | Python quivr linkage assembly | unit-only; native missing | Rust gap |
| `PointSourceDetections.group_by_exposure` | Python sort/filter generator | unit-only; native missing | Rust gap |
| `PointSourceDetections.healpixels` | Python call into healpy | unit-only; native missing | Rust-owned replacement required |
| `PointSourceDetections.group_by_healpixel` | Python NumPy/PyArrow grouping around healpy | unit-only; native missing | Rust gap |
| `PointSourceDetections.link_to_exposures` | Python quivr linkage assembly | unit-only; native missing | Rust gap |
| `Exposures.group_by_observatory_code` | one Rust typed grouping/IPC crossing; Python yields compatibility tables | ordering/schema unit parity plus Rust-Instant timing | Compatible veneer |
| `Exposures.midpoint` | one Rust crossing owns half-even nanosecond conversion and epoch carry | exact midpoint unit parity plus Rust-Instant timing | Compatible veneer |
| `Exposures.observers` | one Arrow crossing owns midpoint epochs, code grouping, SPICE state dispatch, frame/origin handling, and nested observer assembly | observer fixture ULP parity plus Rust-Instant fused timing | Compatible veneer |

These gaps are covered by children of `personal-cmy.37.7` for observation grouping/linkage/HEALPix and exposure-derived workflows.

## `SourceCatalog` custom surface

Every custom projection remains Python-owned or mixed:

- `detections`, `associations`, and `photometry` reconstruct output tables in Python.
- `exposures` reconstructs and deduplicates in Python.
- `coordinates` builds six-dimensional covariance matrices, converts arcseconds to degrees, and assembles `SphericalCoordinates` in NumPy/Python.
- `observers` owns midpoint branching/time arithmetic around `Observers.from_codes`.
- `healpixels` calls healpy after Arrow-to-NumPy conversion.

None has a selected pinned-legacy facade lane or qualifying Rust-owned timer. A child of `personal-cmy.37.7` requires typed one-crossing Rust projections preserving row order, nulls, deduplication, covariance units, midpoint behavior, schemas, metadata, exceptions, and HEALPix nest/ring semantics.

## Plotting and inherited operations

This domain defines no plotting APIs. Therefore no adam-core-owned method above is exempt. Generic inherited quivr operations remain compatible external infrastructure and are explicitly excluded from migration credit, parity microbenchmarks, and native timing.

## Closure

The observations domain is not complete. Closure requires the ADES product task plus all three observation-domain implementation children, direct pinned-legacy facade parity, and Rust-Instant timing for deterministic grouping, projection, HEALPix, time, and observer workflows.
