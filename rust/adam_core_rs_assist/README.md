# adam_core_rs_assist

GPL-3.0 adapter crate for wiring `assist-rs` into the permissive `adam_core_rs_coords` propagation contracts.

This package is intentionally excluded from the default adam-core Rust workspace so the permissive core crates do not take a GPL dependency. Build and test it explicitly when working on the ASSIST-backed adapter. Because `assist-rs` is a private SSH dependency, local builds may need `CARGO_NET_GIT_FETCH_WITH_CLI=true` so Cargo uses the developer's Git/SSH agent.

Current scope:

- Propagation requests for Cartesian `Orbits` / `VariantOrbits` using Python `adam_assist.ASSISTPropagator` public semantics for the first supported slice.
- Caller-facing `SUN` / `SOLAR_SYSTEM_BARYCENTER`, `ecliptic` / `equatorial`, and TDB/UTC propagation input/target handling, internally normalized to ASSIST/`assist-rs` and restored on output.
- ASSIST propagation through `assist-rs` using the currently exposed `assist_propagate` path.
- Optional 6×6 linear covariance transport through ASSIST STM output for native SUN/ecliptic cases only; public origin/frame covariance transforms are not implemented yet.
- Optional `python` feature exposing an experimental GPL Python package, `adam_assist_rust`, for apples-to-apples Python-callable benchmarks against `adam_assist`; build extension wheels with the `extension-module` feature as used by `pdm run assist-rust-develop`. This is not yet a drop-in replacement package; ephemeris, collision, and covariance-public-API parity remain open.

RM-STANDALONE-007B benchmark acceptance targets Python `adam_assist.ASSISTPropagator` public semantics and must compare against the same DE440 + DE441-n16 kernel files before making any speed claim. External builds of this GPL adapter remain blocked on access to the private `assist-rs` dependency unless/until that repository becomes publicly reachable or an approved distribution path exists.
