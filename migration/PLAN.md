# ADAM Core Rust Transition Plan (Canonical)

Source: user-approved plan in this thread (2026-04-14)
Status: authoritative

## Summary
- Goal: migrate `adam-core` to an API-majority Rust-backed implementation while keeping a stable Python package and shipping incrementally.
- Execution model: strangler pattern, 2-week milestones, 1-2 engineers, production-safe progressive cutover.
- Workspace strategy: new clean sibling checkout from `adam-core` main; prior BVH experiment used as reference only (limited targeted reuse if explicitly approved).
- Core decisions:
  - Cargo workspace multi-crate
  - PyO3 + typed Python wrappers
  - maturin wheel builds
  - Hybrid data boundary (kernel-type rule): NumPy for dense numeric kernels; Arrow for table/schema/null-sensitive APIs
  - Rust default only after parity/perf gates
  - Selective long-lived fallback only via waiver registry

## Implementation Changes
- Milestone 1 (first 2 weeks):
  - Create new checkout and scaffold Rust workspace + Python binding crate.
  - Integrate maturin packaging and CI smoke checks.
  - Land first 2 Rust-backed kernels (one coordinate, one dynamics) with Python wrappers.
  - Implement boundary adapters for both NumPy and Arrow paths.

- Kernel migration order:
  1. Coordinates + dynamics kernels first.
  2. Orbit-determination kernels next where perf/parity support it.
  3. Targeted BVH/geometry clean-room Rust ports using prior branch as reference.
  4. Continue until >50% public API entrypoints are Rust-backed.

- Boundary policy (updated):
  - NumPy boundary: vectorized numeric kernels with contiguous array inputs/outputs.
  - Arrow boundary: table-centric workflows needing schema fidelity, nullable semantics, or quivr alignment.
  - Each migrated API must declare its boundary contract explicitly in adapter docs/tests.
  - High-level entrypoint rule (non-violable): once a high-level API crosses Python -> Rust, its internal execution graph must remain in Rust until final return to Python (no repeated Python<->Rust ping-pong within the same call).

- Fallback governance:
  - Waiver file per retained legacy path: reason, missing criterion, owner, review date.
  - Re-reviewed every milestone; no implicit permanent fallback.

- Downstream integration gates:
  - Tier-1 compatibility checks: `adam-assist`, `adam-api`, `adam_etl`.
  - PR smoke + nightly full compatibility/perf suites.

## Public APIs / Interfaces / Types
- Internal Rust binding namespace (e.g., `adam_core._rust`) behind existing public APIs.
- Per-API contract required:
  - boundary type (NumPy/Arrow)
  - shapes/dtypes or schema/null expectations
  - Python↔Rust error mapping
  - execution-boundary policy (`single-crossing` required for high-level entrypoints such as `transform_coordinates`)
- Migration status metadata per API: `legacy`, `dual`, `rust-default`, with waiver reference when applicable.

## Migration Surface Scope Policy (Atomic Entry Points)
- Default migration unit is the highest-level atomic function entrypoint that is directly used by `adam-core` call sites or downstream users.
- Do not migrate internal-only helper layers by default when they are only dimensionality/validation plumbing behind an already-migrated public atomic function.
- If a function is not called elsewhere in `adam-core` and is not a known downstream-facing surface, keep it internal to the parent implementation and prioritize moving more of that work under the parent Rust-backed entrypoint.
- Helper-level Rust ports are allowed only when they materially improve measurable performance or correctness and do not create duplicate public migration surfaces.
- For high-level orchestration APIs (for example `transform_coordinates`), migration is only considered complete when dispatch + transform chain runs end-to-end in Rust behind one Python boundary crossing.

## Test Plan
- Correctness: existing tests pass plus function-specific tolerance checks against legacy/JAX baselines.
- Full-suite requirement: run the standard `pytest --benchmark-skip -m 'not profile'` suite in a Rust-enabled environment, and fail the run if the Rust extension is unavailable.
- Performance gate: require >=20% p50/p95 speedup before switching default to Rust.
- High-level boundary-crossing gate: migrated high-level APIs must prove single Python->Rust entry and single Rust->Python return for the full call path.
- CI policy: PR smoke (build/import/tests + lightweight perf), nightly full benchmarks and trend reports.
- Packaging: Linux/macOS wheels for x86_64 + aarch64 validated in clean envs and dependent-project installs.

## Assumptions and Defaults
- Success metric is API-majority in Rust, not LOC.
- Minor breaking API changes are allowed when justified, but wrapper continuity is preferred.
- JAX path remains where Rust is not yet faster/better, governed by explicit waivers.
- Hybrid boundary choice is a deliberate optimization decision per API, not one global default.

## Engineering Requirements Contract (AGENTS Alignment)
- This migration is explicitly bound to `/Users/aleck/Code/AGENTS.md`.
- Control flow must stay simple and decomposed: prefer guard clauses, small focused functions, and clear early returns.
- Prefer functional implementations over new classes unless avoiding classes is clearly awkward.
- Reuse existing entrypoints before adding new ones; avoid duplicate APIs and unnecessary abstractions.
- Do not add fallback behavior unless explicitly requested; prefer loud failures when contract conditions are violated.
- Performance-sensitive Python paths must stay vectorized/batched (avoid per-item loops when array-oriented options exist).
- Keep testing/scripts disciplined: quick checks via inline terminal execution, and avoid leaving throwaway top-level scripts.
- Python code added in migration scope must include type hints and prefer strong concrete types over generic containers.
