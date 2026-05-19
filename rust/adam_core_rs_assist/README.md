# adam_core_rs_assist

GPL-3.0 adapter crate for wiring `assist-rs` into the permissive `adam_core_rs_coords` propagation contracts.

This package is intentionally excluded from the default adam-core Rust workspace so the permissive core crates do not take a GPL dependency. Build and test it explicitly when working on the ASSIST-backed adapter. Because `assist-rs` is a private SSH dependency, local builds may need `CARGO_NET_GIT_FETCH_WITH_CLI=true` so Cargo uses the developer's Git/SSH agent.

Current spike scope:

- TDB-only propagation requests.
- Heliocentric ecliptic Cartesian input/output (`SUN` / `NAIF:10` origins only).
- ASSIST propagation through `assist-rs` using the currently exposed `assist_propagate` path.
- Optional 6×6 linear covariance transport through ASSIST STM output.

This normalized scope is **not** the acceptance target for RM-STANDALONE-007B. The parity/benchmark milestone must match Python `adam_assist.ASSISTPropagator` public semantics: accept the same caller-facing origins, frames, and time scales; normalize internally to the ASSIST/`assist-rs` contract; restore output origin/frame/time semantics; and compare against the same DE440 + DE441-n16 kernel files before making any speed claim. External builds of this GPL adapter remain blocked on access to the private `assist-rs` dependency unless/until that repository becomes publicly reachable or an approved distribution path exists.
