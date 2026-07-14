# Orbit, ephemeris, OEM, SPK, and query public-surface disposition

Updated 2026-07-14 against upstream main
`936cc636096fcfefcee3e1310c21528444f39546`. `manifest.json` is the
symbol-level inventory; this is its grouped implementation disposition.

## Tables and grouping

| Surface group | Disposition | Evidence |
|---|---|---|
| Orbits/Ephemeris/Variant tables, columns, constructors, projection, linkage objects | generic quivr/PyArrow data veneer; UUIDs remain Python inputs where nondeterministic identity is required | schema/linkage/round-trip tests |
| `Orbits.group_by_orbit_id` | one Arrow crossing; Rust owns stable grouping and output batches | grouping parity and benchmark coverage |
| `Orbits.dynamical_class` / `calc_orbit_class` | one Arrow/numeric crossing; Rust owns conversion inputs, classification, and stable labels | random parity, method tests, native timing |
| `Ephemeris.link_to_observers` and variant linkage methods | generic quivr linkage after Rust-backed time coercion; no adam-core numerical algorithm | precision/warning/linkage tests |
| `VariantOrbits.create`, `collapse`, `collapse_by_object_id` | one Arrow crossing; Rust owns sampling/collapse grouping, means, covariances, validation, and finished batches; Python supplies RNG/UUID compatibility inputs | statistical/fixed parity and native timing |
| VariantEphemeris `collapse`, grouped collapse, and orbit-major sigma-point collapse | one Rust compute crossing; Python wraps quivr outputs and optionally invokes the explicit aberration provider | circular-longitude, grouping, weights, aberration-mode, and timing tests |
| `preview` | plotting boundary | plotting exemption |

## OEM and SPK products

| Surface group | Disposition | Evidence |
|---|---|---|
| OEM read/write | fused Rust parser/writer and typed batch assembly; Python preserves validation order, current creation timestamp input, logging, and output type | frozen latest-oracle fixtures, malformed input, metadata, covariance, byte/round-trip tests |
| propagated OEM | propagation is an explicit supplied-provider boundary; finished OEM preparation/write is the same fused Rust path | provider and product tests |
| SPK Type 3/9 creation | Rust owns interpolation/fitting, DAF records, multi-summary chaining, atomic publication, and bytes | over/underdetermined fixtures, multi-record and SPICE-open tests, timing |
| SPK creation with explicit propagator | supplied propagation remains a provider boundary; SPK product construction stays Rust-owned | fallback/provider tests |

## Query and product APIs

| Surface group | Disposition | Evidence |
|---|---|---|
| Horizons | one Rust HTTP/parse/typed-product crossing with Python compatibility exceptions | recorded chunk/error/order tests and live smoke |
| NEOCC, Scout, and SBDB | one Rust HTTP/parse/normalization/Arrow crossing | recorded retry/missing/schema tests, live smoke, clean-room acceptance |
| historical `SBDB.query` monkeypatch target | explicit lazy compatibility seam; no default Astroquery import | monkeypatch and optional-layering tests |
| OpenSpace CSV/Lua/assets/SPICE snippets | one Rust product crossing with atomic publication | frozen bytes, malformed inputs, rollback tests, timing |
| physical-parameter tables/constants | static data/data veneer; fitting and numerical transforms use Rust paths | schema and fit tests |

## Closure

All adam-core-owned non-plotting orbit/product algorithms are Rust-backed.
Remaining Python code is table wrapping, nondeterministic identity/timestamp
input, generic linkage, or an explicitly supplied propagator/network
compatibility provider. There are no untracked implementation gaps in this
domain.
