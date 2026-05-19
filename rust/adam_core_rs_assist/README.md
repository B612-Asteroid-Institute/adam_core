# adam_core_rs_assist

GPL-3.0 adapter crate for wiring `assist-rs` into the permissive `adam_core_rs_coords` propagation contracts.

This package is intentionally excluded from the default adam-core Rust workspace so the permissive core crates do not take a GPL dependency. Build and test it explicitly when working on the ASSIST-backed adapter. Because `assist-rs` is a private SSH dependency, local builds may need `CARGO_NET_GIT_FETCH_WITH_CLI=true` so Cargo uses the developer's Git/SSH agent.

Initial scope:

- TDB-only propagation requests.
- Heliocentric ecliptic Cartesian input/output (`SUN` / `NAIF:10` origins only).
- Gravity-only ASSIST propagation through `assist-rs`.
- Optional 6×6 linear covariance transport through ASSIST STM output.
