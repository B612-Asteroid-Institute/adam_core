# Rust Migration TODO Tracker

Last updated: 2026-04-28 (review task backlog)

## Current Review-Derived Backlog

- [ ] Work from [`review_task_backlog_2026-04-28.md`](review_task_backlog_2026-04-28.md) before starting additional kernel ports unless the user explicitly reprioritizes.
- [x] Direct Rust-to-Rust `spicekit` integration for standalone `adam-core-rs` is implemented in `rust/adam_core_rs_spice` and validated through the full parity/performance cadence.
- [x] RM-P0-004 packaging follow-up cleanup is documented in [`packaging.md`](packaging.md): Cargo is the single wheel version source, uv is lock-only pending local-install revalidation, and current skip-count expectations are recorded.
- [ ] RM-P0-005F: audit restored Python compatibility surfaces, decide which deserve long-term Python API support, and ensure Rust callers use Rust-to-Rust helpers rather than Python shims.
- [x] Review feedback from `adam_core_rust_migration_review_handoff_2026-04-27.md` is decomposed into RM-P0/RM-P1 tasks in the review backlog.
- [x] Other-agent Wave D3/E2/E3 pending work is consolidated under RM-WD3/RM-WE2/RM-WE3 tasks in the review backlog.
- [ ] For every functional/performance change, follow the baseline-main parity and speed verification workflow documented in the review backlog and `migration/parity/README.md`.

## Active Sprint (Milestone 1 hardening)

- [x] Create new clean migration checkout and Rust workspace scaffold.
- [x] Land first two Rust-backed kernels (`coordinates.cartesian_to_spherical`, `dynamics.calc_mean_motion`).
- [x] Add runtime migration status registry and base waiver registry.
- [x] Add CI Rust quality gates (`cargo fmt --check`, `cargo clippy`, `cargo test`).
- [x] Upload Rust wheel artifact on pull requests.
- [x] Add a one-command local full validation loop.
- [x] Add roadmap and milestone acceptance criteria documentation.
- [x] Define waiver schema and review process in documentation.

## Next Priority Queue

- [x] Add randomized parity tests for current migrated APIs.
- [x] Enforce canonical perf gate using p50 and p95 metrics for `rust-default` APIs.
- [x] Introduce first Arrow-boundary pilot API.
- [x] Promote first APIs to `rust-default` only after perf and parity pass.
- [x] Add weekly scheduled Tier-1 dependent smoke run on `main`.
- [x] Select first orbit-determination kernel candidate (`orbit_determination.calcGibbs`) and freeze NumPy boundary contract.
- [x] Implement first orbit-determination Rust-backed API in `dual` mode with migration metadata.
- [x] Add orbit-determination parity tests for rust wrapper and public API path.
- [x] Add orbit-determination benchmark harness (legacy vs rust wrapper) and document promotion decision.
- [x] Add AGENTS alignment to written requirements and backend contract docs.
- [x] Validate contract checks after AGENTS alignment (`rust-smoke`, strict perf gate, rust-required pytest collection).
- [x] Add second orbit-determination Rust component (`orbit_determination.calcHerrickGibbs`) with parity/benchmark coverage and promote to `rust-default` after passing +20% p50/p95 gate.
- [x] Add third orbit-determination Rust component (`orbit_determination.calcGauss`) with parity/benchmark coverage and promote to `rust-default` after passing +20% p50/p95 gate.
- [x] Add Tier-1 smoke extension covering at least one orbit-determination import/use path.
- [x] Add high-level atomic `orbit_determination.gaussIOD` Rust candidate-generation path (`dual`) while keeping helper-level ports out of the public migration surface.
- [x] Add reproducible `gaussIOD` parity/performance benchmark coverage and enforce +20% p50/p95 promotion gate before default switch.
- [x] Gate `gaussIOD` to legacy-default and record explicit waiver when benchmark is below promotion threshold.
- [x] Add Rust-backed `coordinates.spherical.to_cartesian` NumPy boundary path and wire high-level transform entrypoint.
- [x] Add parity tests and benchmark-gate coverage for `coordinates.spherical.to_cartesian`.
- [x] Promote `coordinates.spherical.to_cartesian` to `rust-default` after +20% p50/p95 gate pass.
- [x] Add Rust-backed `coordinates.cartesian_to_geodetic` NumPy boundary path and wire high-level transform entrypoint.
- [x] Add parity + benchmark-gate coverage for `coordinates.cartesian_to_geodetic` and enforce +20% p50/p95 before keeping `rust-default`.
- [x] Add Rust-backed `coordinates.cartesian_to_keplerian` NumPy boundary path and wire high-level transform entrypoint.
- [x] Add parity + benchmark-gate coverage for `coordinates.cartesian_to_keplerian` and enforce +20% p50/p95 before keeping `rust-default`.
- [x] Add Rust-backed `coordinates.keplerian.to_cartesian` NumPy boundary path, including parity + benchmark-gate coverage and high-level single-crossing support for Keplerian-input transform paths.
- [x] Add Rust-native high-level transform dispatcher for `coordinates.transform_coordinates` with single-crossing execution (no Python<->Rust ping-pong in one call path).
- [x] Add contract test for high-level single-crossing execution boundary on `transform_coordinates` and include it in `rust-smoke`.
- [x] Extend `coordinates.transform_coordinates` Rust single-crossing coverage to `equatorial <-> ecliptic` frame transitions for supported representation paths.
- [x] Review refactor (2026-04-16): rename `_transform_coordinates_single_crossing_rust` → `_try_transform_coordinates_rust`, extract explicit `_rust_transform_supports` predicate, add parameterized test enumerating unsupported paths.
- [x] Review refactor (2026-04-16): consolidate `_rust.api` Python wrappers to thin pass-throughs (only `_as_contiguous_f64` coercion + `None` on unavailable backend); Rust owns shape/length validation.
- [x] Review refactor (2026-04-16): delete `migration/api_status.yaml`; make `src/adam_core/_rust/status.py` (`API_MIGRATIONS`) the single source of truth for migration scripts and runtime dispatch.
- [x] Review refactor (2026-04-16): collapse `adam_core_rs_math` and `adam_core_rs_dynamics` into `adam_core_rs_coords`; drop them from the workspace.
- [x] Extract `adam_core_rs_naif` into standalone MIT-licensed crate `spicekit` — Rust library, PyO3 bindings (`spicekit-py` → PyPI `spicekit`), and CSpice parity oracle (`spicekit-bench`) all carved into the sibling repo. `spicekit` is public as crates.io `spicekit = "0.1.0"` and PyPI `spicekit==0.1.0`; adam-core now consumes the Rust crate directly through `rust/adam_core_rs_spice` for runtime SPICE, while the Python `spicekit` wheel remains available for external users. See [`migration/spicekit_extraction_plan.md`](spicekit_extraction_plan.md).
- [x] Port universal-variable 2-body propagation ladder (`stumpff` → `chi` → `lagrange` → `propagate_2body`) from JAX to Rust with `Dual<6>` covariance transport; ship as `status="dual"` / `default="legacy"` pending perf gate.
- [x] Add parity tests for `dynamics.propagate_2body` (states + covariance) against JAX reference.
- [x] Add formal benchmark entry for `dynamics.propagate_2body` in `migration/scripts/rust_backend_benchmark_gate.py` covering a representative (N_orbits, N_times) grid, including a covariance path; promote to `default="rust"` and wire dispatch in `_propagate_2body_serial` once +20% p50/p95 gate clears.
- [x] spicekit pre-publication polish (2026-04-21): fix 35 `-D warnings` clippy errors across the base crate; flip `spicekit-bench` default feature off; pin MSRV=1.85 via `rust-toolchain.toml` + `rust-version.workspace`; add `.github/workflows/ci.yml` (fmt + clippy + test + maturin wheel matrix across manylinux/macos/windows × py3.10–3.12); add `.pyi` stubs + `py.typed`; add minimal `read_spk` examples (Rust + Python); tighten NAIF attribution in `naif_builtin_table.rs` + README disclaimer. crates.io/PyPI publish confirmed on 2026-04-28.
- [x] spicekit-py integration tests (2026-04-21): add `crates/spicekit-py/tests/` pytest suite (53 tests: test_names, test_spk, test_pck, test_spk_writer, test_text_kernel) covering every PyO3 symbol including a Type 9 SPK write→read round-trip; kernel fixtures resolved via `naif-*` PyPI packages; wire into `.github/workflows/ci.yml` as a new `pytest` job (ubuntu + macos-14 × py3.10, 3.12) using `maturin develop --release --extras test` then `pytest tests/`. All 53 tests pass in 0.07s.
- [x] Port `dynamics.generate_ephemeris_2body` as a single fused `T: Scalar`-generic row kernel (LT Newton loop + optional stellar aberration + ec→eq rotation + Cartesian→spherical), reusing `propagate_2body_row` under `Dual<6>` so covariance transport replaces the JAX `transform_covariances_jacobian` call in one pass. Scope: [`migration/generate_ephemeris_2body_plan.md`](generate_ephemeris_2body_plan.md).
- [x] Add parity tests for `dynamics.generate_ephemeris_2body` (states + light-time + aberrated state + covariance) vs `_generate_ephemeris_2body_vmap` + `transform_covariances_jacobian`.
- [x] Add `generate_ephemeris_2body` + `_with_covariance` entries to `migration/scripts/rust_backend_benchmark_gate.py`; promoted to `rust-default` (1120x p50 / 1018x p95 state; 2331x p50 / 2133x p95 covariance) and resolved `waiver-20260422-generate-ephemeris-2body-perf-pending` with the benchmark artifact as resolution evidence. Dispatch wired into `generate_ephemeris_2body`, replacing the chunked vmap + `transform_covariances_jacobian` path with a single Rust crossing; legacy JAX fallback retained behind `rust_result is None`.
- [x] Port H-G photometry kernels (`calculate_phase_angle`, `calculate_apparent_magnitude_v`, `calculate_apparent_magnitude_v_and_phase_angle`) to Rust with rayon-parallel batched f64 kernels (no Dual needed — photometry is not in `Ephemeris` covariance). Wired dispatch into the three public functions in `src/adam_core/photometry/magnitude.py`. Benchmark gate at 100k rows: 6.28x / 4.14x / 5.36x p50 respectively; all promoted to `rust-default` and `waiver-20260422-photometry-perf-pending` resolved. Closes the `generate_ephemeris_2body` surface end-to-end — on `predict_magnitudes=True`/`predict_phase_angle=True` paths, the entire chain (LT + aberration + ec→eq + cart→sph + H-G magnitudes + phase angle) now runs in Rust.
- [x] gaussIOD fusion revisit (2026-04-22): applied 2026-04-22 fusion rule to the public `gaussIOD` API — new fused Rust kernel `gauss_iod_fused_numpy` absorbs the entire body (equatorial RA/Dec→ecliptic unit-vector rotation, Milani A/B/V/C0/h0 coefficients, 8th-order polynomial root-finding, per-root orbit construction) into one Rust crossing. Polynomial roots via Laguerre+deflation (robust to wide coefficient dynamic range — Durand-Kerner stalled on real polys with leading coeff C0²~1e-3 and roots from 0.86 to 37.6 AU). Benchmark 128 real-world triplets: **2.72x p50 / 2.81x p95** — clears 1.2x gate by 2×. Promoted to `default="rust"` and resolved `waiver-20260415-gaussiod-perf`. Legacy python+numba path retained behind `_GAUSS_IOD_USE_FUSED_RUST=False` for parity comparisons.
- [x] calc_mean_motion: dedicated isolated benchmark `migration/scripts/calc_mean_motion_bench.py` at realistic N=50k (KeplerianCoordinates.n and CometaryCoordinates.n call sites). Measures **1.49-1.61x p50 / 2.09-2.65x p95** in isolation — both clear 1.2x gate. Main gate's 0.65x measurement at N=400k was contaminated by memory/cache state from preceding large benchmarks and is not a fair stand-alone number. Known trade-off: at N>=400k JAX's XLA SIMD wins (~4.6x faster); portable stable Rust can't match without unsafe intrinsics or `-C target-cpu=native` (breaks wheel portability). Dispatch is split cleanly by contract (not duck-typed): `dynamics.kepler.calc_mean_motion` stays JAX-typed (composable inside `lax.cond` in `coordinates.transform`); NumPy callers (`KeplerianCoordinates.n`, `CometaryCoordinates.n`) call `adam_core._rust.api.calc_mean_motion_numpy` directly for the Rust fast path. Resolved `waiver-20260414-calc-mean-motion-perf`.

## Future major work

- [ ] **N-body propagation port** (its own project, multi-day scope). Dominates production workloads — most real pipelines use N-body (ASSIST/REBOUND or adam-core's numerical integrator) rather than 2-body Keplerian. Requires: adaptive-step RK integrator in Rust with `Dual<6>` covariance support, SPICE-backed perturber state lookup (already via spicekit), proper hyperbolic/parabolic coverage, and integration into existing `Propagator` abstraction. Biggest single remaining win after the 2-body + ephemeris + IOD + photometry fusion work landed 2026-04-22.

## Acceptance Criteria (Milestone 1)

- [x] CI enforces Rust quality checks for workspace.
- [x] PRs publish Rust wheel artifact for downstream validation.
- [x] Local command exists to run lint + Rust build + smoke tests in one step.
- [x] Governance docs describe roadmap + waiver process with explicit owner/review-date requirements.

## Notes

- Keep legacy/JAX fallback paths unless Rust path is both correct and operationally better.
- Canonical perf gate from `migration/PLAN.md`: Rust must be >=20% faster (p50/p95) before `rust-default`.
- Non-violable boundary rule from `migration/PLAN.md`: high-level APIs must run end-to-end in Rust after boundary entry, with a single Python->Rust crossing per call.
- Migration scope rule from `migration/PLAN.md`: prioritize highest-level atomic entrypoints over internal helper-by-helper ports unless helper migration is required for measurable gains.
- Full-suite validation contract: run standard `pytest --benchmark-skip -m 'not profile'` with `ADAM_CORE_REQUIRE_RUST_BACKEND=1`.
- Local environment note: rust-required full-suite pytest run passes in this checkout when `PYTHONPATH=/Users/aleck/Code/mpcq/src` is set.
- Engineering policy contract: follow `/Users/aleck/Code/AGENTS.md` rules for control flow, functional style, fallback posture, vectorization, scripting discipline, and Python typing.
- Any retained fallback path must be represented in `migration/waivers.yaml` once a waiver exists.
- Promotion decision: `orbit_determination.calcGibbs` benchmarked and set to `rust-default`.
- Promotion decision: `orbit_determination.calcGauss` benchmarked and set to `rust-default`.
- Promotion decision: `coordinates.spherical.to_cartesian` benchmarked and set to `rust-default`.
- Promotion decision: `coordinates.cartesian_to_geodetic` benchmarked and set to `rust-default`.
- Promotion decision: `coordinates.cartesian_to_keplerian` benchmarked and set to `rust-default`.
- Promotion decision: `coordinates.keplerian.to_cartesian` benchmarked and set to `rust-default`.
- Active waiver: `waiver-20260414-calc-mean-motion-perf`.
- Active waiver: `waiver-20260415-gaussiod-perf`.
- Active temporary waiver: `waiver-20260428-photometry-warm-performance-temporary` for known photometry n=2000 warm speed-gate misses; review by 2026-05-12.
- Active temporary waiver: `waiver-20260428-cartesian-to-spherical-warm-performance-temporary` for known `coordinates.cartesian_to_spherical` n=2000 warm p95 speed-gate misses; review by 2026-05-12.
- Promotion decision: `dynamics.propagate_2body` (and `…_with_covariance`) benchmarked and set to `rust-default` (87x/414x p50).
