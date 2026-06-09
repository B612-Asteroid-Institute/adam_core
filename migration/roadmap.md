# ADAM Core Rust Migration Roadmap

Last updated: 2026-04-15

## Program Goal

Achieve API-majority Rust-backed implementation in `adam_core` while preserving operational stability and scientific parity.

## Milestone Plan

## Milestone 1: Foundation + First Kernels (completed)

Scope:
- Rust workspace and PyO3/maturin integration.
- First NumPy-boundary Rust kernels:
  - `coordinates.cartesian_to_spherical`
  - `dynamics.calc_mean_motion`
- CI smoke + nightly benchmark + public dependent-package smoke wiring.
- Governance artifacts (`src/adam_core/_rust/status.py`, `waivers.yaml`, contracts docs).

Exit Criteria:
- Rust workspace checks enforced in CI.
- PR wheel artifact available.
- Local full-loop command available.
- Core parity tests pass for migrated APIs.

## Milestone 2: Coordinate/Dynamics Expansion (current)

Scope:
- Add 2-4 additional coordinate/dynamics kernels.
- Randomized parity test suite for all migrated kernels.
- Benchmark comparisons with p50/p95 reporting and strict default-gate enforcement.
- Establish non-violable high-level execution rule: no Python<->Rust ping-pong inside a migrated high-level call path.

Exit Criteria:
- At least 4 APIs in `dual` or `rust-default` state.
- No unowned or stale waiver records.
- High-level transform migration design approved with single-crossing execution contract.

## Milestone 3: First Arrow-Boundary Pilot (completed)

Scope:
- Migrate one table-centric API with Arrow boundary contract.
- Add explicit null/schema parity validation.

Exit Criteria:
- Arrow pilot documented and benchmarked.
- Stable adapter contract published.

## Milestone 4: Orbit-Determination Entry (current)

Scope:
- Migrate first orbit-determination kernel with explicit boundary contract.
- Add candidate-specific parity and benchmark gates.

Exit Criteria:
- One orbit-determination API in `dual` mode with complete contract/tests.
- Rust default decision recorded via perf gate or waiver.

## Milestone 5: Rust-Default Promotion Wave

Scope:
- Promote selected APIs to `rust-default` where perf and parity gates are met.
- Maintain selective fallback only via approved waivers.

Exit Criteria:
- Demonstrated >=20% p50/p95 speedup for each `rust-default` API (target rust_over_legacy <= 0.8333).
- Public dependent-package smoke remains green.
- At least one high-level orchestrator (`transform_coordinates`) runs end-to-end in Rust with single-crossing contract test coverage.

## Ongoing Governance

- Update `src/adam_core/_rust/status.py` (the single registry consumed at runtime and by migration scripts) in every migration PR.
- Use `migration/waivers.yaml` for each retained fallback path with owner and review date.
- Re-review waivers every 2-week milestone.
