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
| Horizons | complete one-crossing Rust HTTP products | Rust owns API URLs, HTTP, 50-epoch chunking, CSV protocol parsing, target-name compatibility, ordering, element conversion, nulls, and nested Orbits/Ephemeris Arrow assembly; Python only projects time/code columns and wraps the returned batch |
| NEOCC | OEF parser only | **Gap:** requests loop, policy/error handling, per-object assembly, coordinate conversion, and concatenation are Python |
| Scout | orbit-row normalization only | **Gap:** both HTTP entrypoints, summary parsing, table construction, per-object loop, conversion, and concatenation are Python |
| SBDB | direct-payload normalization only | **Gap:** legacy astroquery client is Python; new client owns sessions, retries, backoff, concurrency/fair-use, filtering, JSON crossings, and table construction in Python |
| ADES PSV | fused Rust writer/parser plus observation/context kernels | Public writer/parser each satisfy one crossing; Python only reconstructs compatibility dataclasses/quivr objects |
| OEM | fused Rust product writer/reader plus KVN engine | Writer and reader each satisfy one crossing (ecliptic rotation, sort, metadata, unit/covariance conversion, orbit-id and table assembly in Rust); ITRF93 pre-transform stays on the Rust `transform_coordinates` crossing; propagated writer remains a declared propagator-provider boundary |
| OpenSpace | fused SBDB CSV and orbital/trail asset products plus Lua/initialization rendering | Public products each satisfy one crossing: Rust owns transform, epochs/periods, model graph, CSV/Lua rendering, per-orbit loops, SPICE snippets, staged writes, and atomic publication; Python retains enum/default compatibility and an uncovered-case fallback |
| SPK | fused Rust fitting, Type 3/9 segment, multi-summary DAF, and product workflow | No-propagator products satisfy one crossing; optional propagation is the declared provider boundary followed by the same one product crossing. Rust owns transform/group/sort/IDs/windows/fits/units/segments and atomic output |
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

Both exports are one-crossing veneers over the Rust Horizons client. Rust constructs the `horizons.api` protocol for vectors, elements, and observer ephemerides; rescale/sorts epochs; applies the 50-epoch request limit; performs HTTP; parses CSV service tables; preserves astroquery's observable 32-character target-name truncation; converts Keplerian/cometary elements to Cartesian with the canonical solar parameter; orders rows; and emits exact nested quivr-compatible Arrow. Verbatim vectors/elements/ephemerides recordings gate offline parity, a live integration gate checks the external service, and Rust-owned timing measures deterministic response processing only.

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

PSV observation/context parse and rendering are Rust-owned. The public writer
now performs UTC rescaling, all context JSON rendering, observation rendering,
and output assembly in one `ades_to_string_fused_ipc` crossing. The public
parser performs observation parsing, context parsing, nested Arrow encoding,
and unknown-column collection in one `ades_string_to_tables_fused_ipc`
crossing. Python only reconstructs the declared `ObsContext` dataclasses and
`ADESObservations` quivr compatibility object. Frozen byte/error parity and
Rust-Instant timing cover both fused operations.

### OEM products

Public module functions:

- `orbit_to_oem`
- `orbit_to_oem_propagated`
- `orbit_from_oem`

`orbit_to_oem` keeps the legacy Python assertions, single-time warning, and
nondeterministic CREATION_DATE input, then performs everything else in one
`oem_write_orbits_kvn` crossing: ecliptic->equatorial rotation, stable time
sort, metadata/frame/center mapping with exact legacy errors, AU->km state
and covariance conversion in legacy IEEE order, `np.tril_indices` extraction,
KVN rendering, and the file write. ITRF93 input pre-transforms on the
Rust-owned `transform_coordinates` crossing (SPICE/time-dependent), then
writes through the same fused crossing. `orbit_from_oem` is one
`oem_read_orbits_ipc` crossing owning parsing, frame/center mapping with
exact legacy errors, km->AU conversion, last-match-wins covariance joins,
legacy per-state orbit ids, and nested Orbits assembly; empty files return
`Orbits.empty()`, and the rare mixed-frame/scale multi-segment case falls
back to the retained legacy composition so quivr surfaces its own behavior.
Frozen legacy fixtures cover writer bytes and full parsed-table equality;
both fused operations have Rust-Instant timing.

`orbit_to_oem_propagated` calls the propagator (declared provider boundary)
and then uses the same fused writer crossing.

### OpenSpace products

Package exports:

- `create_renderable_orbital_kepler`
- `create_renderable_trail_orbit`

Module-public supporting surface includes `orbits_to_sbdb_file`,
`create_initialization`, `LuaDict.to_string`, `Resource.to_string`, and the
`Gui`, `Asset`, translation, renderable, resource, and rendering enum models.

`orbits_to_sbdb_file`, `create_renderable_orbital_kepler`, and
`create_renderable_trail_orbit` now each enter one fused Rust workflow. Rust
owns the heliocentric/ecliptic Kepler transform, TDB epoch and period
construction, pandas-compatible CSV bytes, Lua model ordering/rendering,
nullable object-ID fallback, Kepler and SPICE translation branches, resource
paths/load-unload snippets, per-orbit loops, staged file writes, and atomic
publication. Python only converts public enum/default values into one option
payload and retains the legacy composition for explicitly uncovered/error
fallbacks. Frozen whole-directory legacy fixtures cover both asset products,
including all options and SPICE mode; Rust-owned `Instant` timing covers the
complete native products. `LuaDict.to_pascal_case`, inherited `to_string`, and
`create_initialization` are likewise direct Rust formatting veneers.

### SPK products

Public module functions are `fit_chebyshev`, `orbits_to_spk`,
`write_spkw03_segment`, and `write_spkw09_segment`. All are direct Rust
crossings. Rust owns inclusive window selection, Chebyshev basis construction,
over/underdetermined SVD minimum-norm fitting, AU/day conversion order, SSB/J2000
normalization, stable orbit grouping and epoch sorting, target IDs, Type 3/9
segment preparation, multi-record DAF summary/name chaining and address
relocation, and atomic output. Frozen legacy NumPy fit coefficients cover both
least-squares shapes; Type 3 (ASSIST provider) and Type 9 (no provider) readback,
late segments beyond the former 25-summary limit, errors, mappings, public
segment shims, and Rust-owned timing are gated.

With `propagator=None`, `orbits_to_spk` is one product crossing. Optional
propagation remains an explicit propagator-provider boundary: Python constructs
the provider's requested time grid and calls it once, then passes the sampled
`Orbits` through the same fused Rust product crossing. The compatibility
`comment` remains accepted and intentionally unserialized, matching the current
native writer contract.

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

Most `RustBackend` methods are legitimate thin low-level veneers.
`get_perturber_state` is now one fused `perturber_states_arrow` crossing: Rust
owns the TDB rescale, integer-epoch dedup, the bounded forward/reverse
epoch-state cache (the Rust mirror of the legacy `_SpkezCacheKey` semantics;
retention order is the only, values-neutral difference), batched SPK lookup
with the legacy ET arithmetic, the legacy divide-by-unit conversion order, and
nested `CartesianCoordinates` assembly. `get_spice_body_state` is one fused
`spice_body_states_arrow` crossing over the shared `state_batch` core (legacy
`/KM_P_AU` then `*S_P_DAY` order). Both veneers keep the legacy frame
validation error, `setup_SPICE()` kernel lifecycle, and a retained legacy
composition as the failure fallback so `NotCovered`/wrapped-`ValueError`
contracts stay byte-identical; fused-vs-legacy bit-exact equality is asserted
in tests, and both crossings have Rust-Instant timing with the cache cleared
before every sample. `clear_spkez_cache` clears the retained Python cache and
the Rust-side cache together. `setup_SPICE`/`setup_mpc_obscodes` remain
Python package-data discovery loops over idempotent Rust loads (kernel
lifecycle explicitly preserved; Rust-side lazy init tracked under
personal-cmy.36.1, data provenance under personal-3uy). Fixed-kernel
integration tests validate adam-core wiring while leaving CSPICE oracle
breadth to spicekit.

### Generic utilities and execution

The public-ish `bounded_lru_get`/`bounded_lru_put` names and unused
`pad_to_fixed_size`/`process_in_chunks` module have been retired. One shared
underscore-private bounded-LRU helper remains solely as the explicit Python
container cache-policy boundary around Rust semantic state calls; it is not an
adam-core public API. `_iterate_chunks` and `_iterate_chunk_indices` remain
private implementation details of OD/IOD and Horizons and are eliminated with
their fused workflow beads rather than promoted as standalone public APIs.

The generic `adam_core.parallel` and `adam_core.ray_cluster` modules have been
retired together with the direct Ray dependency. Their Python-callable and
ObjectRef contracts were not meaningful Rust APIs and are not retained as
shims. Production propagation and ephemeris already execute in fused Rayon
entrypoints; ASSIST-backed OD and IOD now execute in backend-generic Rust work
units. Their historical `max_processes` parameters remain accepted and ignored
for signature compatibility. Non-native provider fallbacks are deterministic
serial compositions and no longer initialize an object store or scheduler.

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
| `personal-cmy.37.4.8` | Replace or retire Python Ray/parallel orchestration (complete: dependency/modules removed; workflows use Rust/Rayon or serial provider fallback) |
| `personal-cmy.37.4.9` | One-crossing SPICE setup/cache/state utility facades |
| `personal-cmy.37.4.10` | Query/product integration and native-timing governance |

These are children of this audit epic, not entries in a second markdown TODO
system. The table is an audit index; status and dependencies live in Beads.
