# Public surface audit: I/O, queries, products, and utilities

Audit date: 2026-07-11

Parent bead: `personal-cmy.37.4`

Audited revision: `f27370b2`

## Decision applied

The acceptance boundary is stricter than the older “intentional I/O boundary”
recorded in `migration/remaining_rust_surface.md`: **only plotting/display may
remain implemented in Python**. Network latency, third-party services, file I/O,
and Python-framework integration are not exemptions. An adam-core-owned public
operation must either:

1. be implemented directly in Rust; or
2. be a compatibility veneer that makes one top-level crossing into a
   Rust-owned operation and only adapts the returned Rust/Arrow value to the
   historical Python name.

A Rust parser below Python HTTP loops, or a Rust text renderer below Python
product assembly and file writes, is partial migration rather than closure.
External work still needs integration gates even when microbenchmarking the
network or filesystem would be meaningless. Native timing applies to the
Rust-owned deterministic portion; live-I/O gates establish protocol and product
correctness.

This audit does not edit the shared parity/status registries. It creates scoped
implementation beads and leaves registry changes to their governance bead.

## Scope and exclusions

Included:

- `adam_core.orbits.query` public clients and module-public parser/converter
  surfaces;
- ADES PSV serialization and context models;
- OEM, SPK, and OpenSpace product APIs;
- MPC designation/date utilities;
- generic chunk/cache/parallel utilities and Ray initialization;
- public SPICE utility/backend lifecycle and state-query facades.

Excluded as domain-owned elsewhere where obvious:

- orbit table behavior unrelated to acquisition or products;
- coordinate representations/transforms except where they are orchestration
  inside an audited public product;
- dynamics/propagator algorithms except where an audited product currently
  invokes them (recorded as an explicit dependency, not silently audited here);
- observation/source-catalog, photometry, and bandpass-vendoring behavior not
  part of ADES;
- inherited quivr/PyArrow methods, which are generic infrastructure rather than
  adam-core-owned APIs;
- test-fixture helpers under `utils.helpers`;
- plotting modules and `utils.plots`, the sole accepted Python implementation
  category.

Private `_iterate_chunks` and `_iterate_chunk_indices` are noted because public
query paths depend on them, but they are not counted as public API commitments.

## Findings at a glance

| Family | Current Rust ownership | Closure under the one-crossing rule |
|---|---|---|
| Horizons | row normalization only | **Gap:** astroquery construction, HTTP, chunking, dataframe assembly/sort, repeated JSON crossings, and output table construction are Python |
| NEOCC | OEF parser only | **Gap:** requests loop, policy/error handling, per-object assembly, coordinate conversion, and concatenation are Python |
| Scout | orbit-row normalization only | **Gap:** both HTTP entrypoints, summary parsing, table construction, per-object loop, conversion, and concatenation are Python |
| SBDB | direct-payload normalization only | **Gap:** legacy astroquery client is Python; new client owns sessions, retries, backoff, concurrency/fair-use, filtering, JSON crossings, and table construction in Python |
| ADES PSV | observation/context parse and rendering kernels | **Gap:** public writer performs time rescale, per-context Rust calls, Arrow bridge, then another Rust call; parser reconstructs Python context objects after a second Rust call |
| OEM | KVN parser/writer and file read/write | **Gap:** validation, transforms, metadata, unit/covariance conversion, row loops, and Orbits reconstruction remain Python; propagated writer also invokes Python propagation |
| OpenSpace | Lua node and initialization text rendering | **Gap:** orbit transform, model graph, CSV generation, loops, path handling, and all asset file orchestration remain Python |
| SPK | low-level Rust DAF writer | **Gap:** propagation dispatch, transform, grouping, Chebyshev fit/windows, segment preparation, and final orchestration remain Python |
| MPC | eight scalar pack/unpack functions and batched packed-date decode | Designation APIs and `convert_mpc_packed_dates` satisfy one Rust crossing; Astropy `Time` construction is the external compatibility boundary |
| SPICE backend | kernel readers/writers and low-level backend methods | Low-level methods are thin; **gap:** high-level setup/data discovery, obscodes file read, Python cache/dedup, time/frame/unit conversion, and typed table assembly |
| Chunk/LRU helpers | retired public-ish names | Unused numeric chunking module removed; LRU functions renamed private and retained only as the documented Python container cache-policy boundary around Rust semantic calls; private OD/query iterators remain tracked by their fused-workflow beads |
| Parallel/Ray | none | **Gap:** arbitrary Python callable/ObjectRef orchestration cannot be treated as a permanent exception; migrate callers to fused Rayon operations and retire, or define a Rust-owned replacement |

The previous fixture files for ADES, OEM, OpenSpace, and MPC are useful parity
evidence, but they do not prove a public API is one crossing. Likewise,
`rust/adam_core_rs_coords/src/query.rs` proves deterministic parser coverage,
not Rust ownership of a query client.

## Detailed public surface inventory

### Query clients and parsers

#### Horizons

Public package export:

- `query_horizons`

Additional module-public operation:

- `query_horizons_ephemeris`

The `_get_horizons_vectors`, `_get_horizons_elements`, and
`_get_horizons_ephemeris` helpers are private but define the service boundary.
They instantiate `astroquery.jplhorizons.Horizons`, issue requests, convert to
pandas, and impose ordering in Python. The public methods then serialize records
to JSON, cross into Rust normalization, and reconstruct quivr coordinate/orbit
or ephemeris tables in Python. `query_horizons` also chunks and concatenates in
Python. This is not a one-crossing veneer.

Required endpoint: Rust owns request construction/protocol, service limits,
chunking, response parsing, stable ordering, coordinate selection, and typed
Arrow output. Recorded service responses should be the deterministic oracle;
opt-in live tests cover service compatibility and errors.

#### NEOCC

Public package export: `query_neocc`. `_parse_oef` is private but Rust-backed.
The live requests loop, designation normalization, epoch option mapping, service
errors, physical-parameter/table assembly, coordinate conversion, and
concatenation remain Python. The HTTP/file-like download is a real migration
gap, not an intentional boundary.

#### Scout

Public package export: `query_scout`. Module-public surfaces also include
`get_scout_objects`, `scout_orbits_to_variant_orbits`, `ScoutObjectSummary`, and
`ScoutOrbit`. Rust normalizes orbit rows, but requests, summary JSON parsing,
Arrow construction, per-object fan-out, conversions, and concatenation remain
Python. Both summary and sampled-orbit endpoints need a Rust client and typed
output contract.

#### SBDB

Public package exports: `query_sbdb` and `query_sbdb_new`; `NotFoundError` is a
public exception by module access. `query_sbdb` still delegates the entire
client to astroquery. `query_sbdb_new` is more explicit but remains Python-owned:
thread-local sessions, request parameters, timeout/retry/backoff, HTTP status
classification, concurrency and fair-use warning, allow-missing filtering,
and output assembly all occur outside Rust. Rust only normalizes already-fetched
payloads.

The two names should converge on one Rust client policy while preserving their
published compatibility differences until deprecation is deliberate. JPL's
one-in-flight fair-use default must be a tested Rust policy, not merely a Python
warning.

### ADES PSV

Public observation-package names include `ADESObservations`, the five context
dataclasses, and `ObsContext`; module-public operations are `ADES_to_string`,
`ADES_string_to_tables`, and `ObsContext.to_string`.

Substantial deterministic behavior is already Rust-owned, including PSV
observation parse/render and context parse/render. The public writer is still
multi-crossing: Python rescales the table, calls `ObsContext.to_string` once per
context, builds an IPC payload, and then invokes the Rust writer. The parser
calls Rust for observations, calls Rust again for contexts, and builds nested
Python dataclasses. Therefore the public endpoints are not yet one crossing.

Closure requires Rust context models and one writer/parser entrypoint each.
Python dataclass/quivr objects may remain compatibility representations at the
outer boundary, but may not orchestrate repeated native calls. File convenience
APIs, if added, must also perform file I/O in Rust.

### OEM products

Public module functions:

- `orbit_to_oem`
- `orbit_to_oem_propagated`
- `orbit_from_oem`

Rust already writes and reads KVN files. `orbit_to_oem` still validates and
transforms the orbit, builds metadata, converts units/covariances row by row,
and only then calls Rust. `orbit_from_oem` calls the Rust parser once but loops
through segments/states/covariances and reconstructs the complete table in
Python. Both are real product-level gaps.

`orbit_to_oem_propagated` additionally calls a propagator. Product closure must
not hide that Python call: either use a Rust propagator endpoint in one Rust
workflow or remain explicitly blocked on the propagator-domain bead.

### OpenSpace products

Package exports:

- `create_renderable_orbital_kepler`
- `create_renderable_trail_orbit`

Module-public supporting surface includes `orbits_to_sbdb_file`,
`create_initialization`, `LuaDict.to_string`, `Resource.to_string`, and the
`Gui`, `Asset`, translation, renderable, resource, and rendering enum models.

Only final Lua node rendering and initialization snippets are Rust-owned.
Transforms, epoch and orbital-value construction, dataclass graph assembly,
CSV serialization through pandas, directory creation, file writing/appending,
per-orbit loops, SPICE resource paths, and load/unload snippets are Python.
OpenSpace is a serialized visualization product, not plotting/display code, so
it is not covered by the plotting exception.

The Rust endpoint should own the complete multi-file product and use atomic
commit semantics to avoid partial `.csv`/`.asset` sets.

### SPK products

Public-ish module functions are `fit_chebyshev`, `orbits_to_spk`,
`write_spkw03_segment`, and `write_spkw09_segment`. The native DAF writer and
atomic final write are already strong foundations. The public product still
performs time-grid creation, optional Python propagation, coordinate transform,
orbit grouping, sorting, ID assignment, Chebyshev matrix/least-squares fitting,
windowing, unit conversion, and segment calls in Python.

`orbits_to_spk` needs one Rust workflow crossing. Optional propagation is an
explicit dependency on a Rust propagator; the no-propagator path can close
independently. Compatibility segment helpers should become one-crossing shims
or be deliberately retired if they were never intended public.

### MPC serialization utilities

The scalar designation operations are direct one-call veneers over Rust:

- `pack_numbered_designation`
- `pack_provisional_designation`
- `pack_survey_designation`
- `pack_mpc_designation`
- `unpack_numbered_designation`
- `unpack_provisional_designation`
- `unpack_survey_designation`
- `unpack_mpc_designation`

`convert_mpc_packed_dates` now decodes the complete input batch through
`unpack_mpc_dates_isot` in one Rust crossing and constructs the externally owned
Astropy `Time` compatibility object once. Rust-Instant timing covers the shared
batched decoder with Astropy/PyO3 conversion excluded.

### SPICE utility APIs

Public `adam_core.utils` exports `setup_SPICE` and `get_perturber_state`.
Additional module-public lifecycle/state APIs include `clear_spkez_cache`,
`setup_mpc_obscodes`, `list_registered_kernels`, `register_spice_kernel`,
`unregister_spice_kernel`, and `get_spice_body_state`. `RustBackend` exposes
kernel lifecycle, name lookup, transforms, body-state batches, and observer
state batches.

Most `RustBackend` methods are legitimate thin low-level veneers. High-level
utilities are not: Python discovers package data, loops over kernel loads,
reads obscodes JSON, owns an OrderedDict cache and reverse-state logic,
deduplicates epochs, maps frames, converts time/units, and constructs coordinate
tables. These operations must be fused into Rust public endpoints. Fixed-kernel
integration tests should validate adam-core wiring while leaving CSPICE oracle
breadth to spicekit.

### Generic utilities and execution

The public-ish `bounded_lru_get`/`bounded_lru_put` names and unused
`pad_to_fixed_size`/`process_in_chunks` module have been retired. One shared
underscore-private bounded-LRU helper remains solely as the explicit Python
container cache-policy boundary around Rust semantic state calls; it is not an
adam-core public API. `_iterate_chunks` and `_iterate_chunk_indices` remain
private implementation details of OD/IOD and Horizons and are eliminated with
their fused workflow beads rather than promoted as standalone public APIs.

`adam_core.parallel` publicly names `ParallelBackend`, `SequentialBackend`,
`RayBackend`, `get_backend`, and `resolve_max_processes`; their methods expose
Python callables and Ray ObjectRefs. `initialize_use_ray` is also public by
module and documentation. Arbitrary Python callable scheduling is not a useful
line-by-line Rust port. Closure should instead move adam-core workflows to
fused Rayon entrypoints and retire/deprecate the generic Ray surface. If a
compatibility API is retained, its contract and execution must be Rust-owned;
Ray attachment cannot remain an untracked exception.

## Governance requirements

A completed child must provide evidence appropriate to its boundary:

- **Queries:** recorded HTTP fixtures covering success, missing objects,
  malformed responses, timeout/retry/status behavior, order, and empty input;
  opt-in live service checks; no network time in a speed claim.
- **ADES/OEM/OpenSpace/SPK/MPC:** frozen legacy bytes/values, round trips, error
  semantics, empty/multiple-record cases, and direct Rust tests.
- **File products:** temporary-directory integration, overwrite/atomicity/error
  behavior, and verification that no partial product survives failure.
- **Utilities/concurrency:** semantic parity plus thread/process integration;
  Rust-owned `Instant` timing only where retained deterministic computation is
  material.
- **Public veneers:** instrumentation or source tests proving one top-level
  Rust call, not merely the presence of a Rust subroutine.

Plotting is the only accepted Python implementation waiver. Any other retained
Python behavior must be classified as an outer compatibility adaptation with no
algorithm, HTTP, file, scheduling, grouping, parsing, or product orchestration.

## Child beads

| Bead | Deliverable |
|---|---|
| `personal-cmy.37.4.1` | Rust Horizons HTTP/protocol/chunking and typed output |
| `personal-cmy.37.4.2` | Rust NEOCC, Scout, and SBDB clients end to end |
| `personal-cmy.37.4.3` | Single-crossing ADES contexts plus PSV serialization |
| `personal-cmy.37.4.4` | Single-crossing OEM read/write products |
| `personal-cmy.37.4.5` | Rust-owned OpenSpace multi-file products |
| `personal-cmy.37.4.6` | Rust-owned SPK fitting/segment/product workflow |
| `personal-cmy.37.4.7` | Batched MPC dates and port-or-retire generic utilities |
| `personal-cmy.37.4.8` | Replace or retire Python Ray/parallel orchestration |
| `personal-cmy.37.4.9` | One-crossing SPICE setup/cache/state utility facades |
| `personal-cmy.37.4.10` | Query/product integration and native-timing governance |

These are children of this audit epic, not entries in a second markdown TODO
system. The table is an audit index; status and dependencies live in Beads.
