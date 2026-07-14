# Coordinates, time, origins, and observers public-surface disposition

Updated 2026-07-14 against upstream main
`936cc636096fcfefcee3e1310c21528444f39546`. The symbol-level authority is
`manifest.json`; this document supplies the grouped disposition referenced by
those rows.

## Disposition vocabulary

- **one crossing**: Python validates/coerces compatibility inputs, calls one
  Rust operation, and wraps its output.
- **data veneer**: quivr/PyArrow schema, column projection, construction, or
  conversion with no adam-core algorithm.
- **provider boundary**: explicitly selected third-party/network/kernel
  behavior; it is not a hidden default backend.
- **control veneer**: cache clearing, logging, or iteration with no numerical
  domain work.

## Coordinate tables and transforms

| Surface group | Disposition | Evidence |
|---|---|---|
| Cartesian/Cometary/Keplerian/Spherical/Geodetic constructors, columns, `values`, `r`, `v`, sigma projections, and identity constructors | data veneer | schema, null, dtype, and round-trip suites |
| vector magnitudes/unit vectors/angular momentum, derived `a/q/Q/p/P/n`, unit conversion, RIC matrices, spherical unit normalization | one crossing into Rust numeric kernels; Python returns legacy NumPy/table types | focused property parity and broad coordinate tests |
| Cartesian `rotate` and `translate` | one crossing; Rust owns state/covariance arithmetic and IEEE propagation | fixed/random covariance and transform parity |
| all representation conversions and `transform_coordinates` branches | one crossing per public conversion; Rust owns state/covariance transforms, time-dependent rotations, origin translation, and mixed-origin dispatch | random fuzz, branch fixtures, SPICE tests, native timing |
| unsupported/custom frame or explicitly supplied state provider | provider boundary with compatible fallback/error | provider/fallback tests |
| Google Maps URL | plotting/external-URL boundary | plotting exemption |

## Covariances, variants, residuals, and units

| Surface group | Disposition | Evidence |
|---|---|---|
| covariance matrix/sigma/null accessors | data veneer | schema/null tests |
| sigma expansion, PSD repair, random and sigma-point sampling, weighted mean/covariance, sampling transforms, coordinate-variant creation | one crossing into Rust computation, with Python RNG/UUID inputs supplied where compatibility requires nondeterminism | fixed/fuzz/statistical parity and Rust-owned timing |
| `Residuals.calculate`, chi-square, longitude bounding, cosine-latitude correction, reduced chi-square | one crossing for built-in coordinate types; explicitly custom coordinate objects use the compatibility provider path | parity registry, custom-coordinate tests, native timing |
| AU/km and AU/day/km/s scalar/vector/covariance conversions | one crossing or constant-factor veneer over Rust; operation order is fixed by compatibility tests | exact/fixed tests |

## Origin and time

| Surface group | Disposition | Evidence |
|---|---|---|
| origin enums/constants/columns and static gravitational parameters | static data or data veneer | exact constant/schema tests |
| origin conversion, `mu`, and barycenter gravitational sums | one crossing / thin scalar veneer over Rust-owned constants | origin parity tests |
| Timestamp columns, projections, keys/signatures, ordering/link construction | data/control veneer | timestamp/table tests |
| MJD/ISO constructors and renderers, arithmetic, rounding, equality, reductions, unique, and ordinary TAI/TT/UTC/TDB rescaling | one crossing into Rust integer-time/leap-second kernels; null/table wrapping stays Python | frozen fixtures, leap seconds, half-even rounding, fuzz tests, native timing |
| observing-night calculation and geographic timezone lookup | one crossing; Rust owns timezone polygons and DST-compatible IANA resolution | timezone/DST/clean-room tests |
| `rescale_astropy`, Astropy object conversion, and UT1/IERS | explicit lazy `adam_core[astropy]` provider boundary | optional-layering and provider tests |
| `Timestamp.link` | generic quivr linkage after Rust-backed coercion/rounding | linkage tests |

## Observers and SPICE state

| Surface group | Disposition | Evidence |
|---|---|---|
| observatory constants/tables and columns | installed data/static veneer | package-data governance |
| parallax coefficient geodesy and timezone | one crossing into Rust | fixed alias/geodesy/timezone tests and timing |
| `Observers.from_codes` / string `from_code` for MPC, space, and custom kernels | one Arrow crossing; Rust owns normalization, state lookup, rotation, ordering, and finished table assembly | mixed-observer parity, cache-cleared compute benchmarks, clean-room smoke |
| explicit `OriginCodes` state providers and unsupported custom providers | provider boundary | fallback tests |
| observer iteration and cache-clear APIs | control veneer | cache-governance tests |
| SPICE state/frame helpers | direct Rust kernels; Python only resolves installed paths and preserves exceptions | SPICE backend/kernel tests and compute/cache benchmark identities |

## Closure

No adam-core-owned numerical gap remains in this domain. Generic quivr/table
mechanics, plotting/URLs, cache controls, and explicitly requested Astropy or
custom providers are the only non-Rust computation boundaries. Native timing is
recorded at the qualifying Rust operation rather than at every alias/property;
the selected 44-row registry is benchmark evidence, not the symbol inventory.
