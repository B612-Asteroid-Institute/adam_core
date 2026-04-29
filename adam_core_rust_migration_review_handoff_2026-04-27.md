# adam-core Rust Migration Review Handoff

Date: 2026-04-27
Last updated: 2026-04-29
Reviewer: Codex
Migration checkout: `/Users/aleck/Code/adam-core-rust-migration`
Baseline checkout: `/Users/aleck/Code/adam-core`

## Read This First: Current Reviewer State On 2026-04-29

This document began as a 2026-04-27 static critique. The original critique is
kept below for provenance, but several blockers listed there have since been
closed. Reviewers should treat this section and the 2026-04-29 addendum at the
end of the file as the current state.

Current migration checkout state:

- Path: `/Users/aleck/Code/adam-core-rust-migration`
- Branch: `rust-migration-waves-d-e`
- HEAD: `0bf2e3cd` (`Retire live-legacy benchmark governance`)
- Working tree after the last task commit contains only uncommitted grounding
  files: `decisions.md` and `journal.md`. They are intentionally not committed.
- Baseline oracle remains the sibling checkout `/Users/aleck/Code/adam-core`
  installed in `.legacy-venv` for parity and speed comparisons.

Current milestone posture:

- All RM-P0 stabilization blockers in
  `migration/review_task_backlog_2026-04-28.md` are complete.
- The branch is ready for review of the P0 hardening work.
- The migration is not "finished": P1 governance/coverage tasks and Wave
  D3/E2/E3 performance work remain open.
- Next open task is RM-P1-008: make `src/adam_core/_rust/status.py` a more
  trustworthy registry that distinguishes public-rust-default, rust-only,
  raw-kernel-only, partial coverage, and randomized-fuzz exclusions.

Current validation evidence from the latest completed task, RM-P0-007:

- `pdm run script-preflight`: passed.
- `pdm run rust-latency-gate`: passed on rerun. First run had one transient
  p95 microbenchmark outlier on `propagate_2body_with_covariance`; rerun was
  green and all ratios were within the Rust-only regression thresholds.
- `pdm run rust-quality`: passed (`cargo fmt --all --check`,
  `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test --workspace`).
- `pdm run test-rust-full`: passed when run with escalated permissions:
  `723 passed, 144 skipped, 2 deselected, 56 warnings`.
- `pdm run rust-parity-main`: passed. All 22 wired APIs passed randomized
  fuzz parity against baseline main.
- `pdm run rust-parity-speed-cold`: passed with existing temporary photometry
  warm-speed waivers only.
- Canonical tables were regenerated:
  `migration/artifacts/parity_report.md` and
  `migration/artifacts/parity_table_rca.json`.
- `git diff --check`: passed.

Validation caveats:

- A non-escalated `pdm run test-rust-full` failed only because the tool sandbox
  denied Ray/psutil macOS `sysctl` process inspection and DNS/network access
  for JPL/Horizons-backed tests. The escalated rerun passed.
- `pdm run docs-check` could not run in the active environment because Sphinx
  is not installed. Installing docs dependencies with `pdm install -G docs`
  wanted to refresh a stale lockfile; dependency files were intentionally left
  untouched.

Active waivers still requiring reviewer attention:

- `waiver-20260428-photometry-warm-performance-temporary`: four photometry APIs
  miss or skim the n=2000 warm p50/p95 speed policy, while cold-start speedups
  remain around 29-31x. Review by 2026-05-12.
- `waiver-20260428-cartesian-to-spherical-warm-performance-temporary` remains
  recorded for prior repeated warm p95 instability, although the latest
  cold/warm artifact raw-passed this API. Review by 2026-05-12.

Disposition of post-handoff reviewer feedback:

- The claim that `calc_chi_numpy` and `calc_lagrange_coefficients_numpy` still
  default to `tol=1e-16` is stale. Current PyO3 signatures, `_rust.api`, and
  Python compatibility wrappers use `tol=1e-15`, implemented in `f2334505`.
- The `calc_chi` compatibility wrapper does bypass the warm-started
  `OrbitConstants` arc optimization, but that was intentionally documented
  rather than hidden behind compat routing. It is not a cause of the photometry
  warm waivers. Revisit only if direct downstream use of
  `adam_core.dynamics.chi.calc_chi` is measured as a hot loop.
- The docs-check/lockfile gap is real and is now an explicit hard requirement
  in RM-P1-010.
- The "rerun once" handling of the Rust-only latency gate is a real
  statistical-discipline gap and is now tracked as RM-P1-018.
- RM-P1-015 should be folded into RM-P1-008 because RM-P1-008 already owns
  registry-level `gaussIOD` randomized-fuzz exclusion visibility.

Current parity/reporting coverage:

- 22 of 25 declared APIs are wired directly into randomized fuzz generators.
- 2 orchestration APIs are covered indirectly by underlying kernel parity:
  `dynamics.calculate_perturber_moids` and `dynamics.generate_porkchop_data`.
- `orbit_determination.gaussIOD` remains intentionally unwired from randomized
  fuzz because Rust Laguerre+deflation and legacy `np.roots` can find different
  root subsets on random triplets. Fixed-fixture/manual parity remains a
  separate follow-up.

## Executive Summary

Historical 2026-04-27 assessment follows. It is useful context, but it is not
the current blocker list. See the section above and the final 2026-04-29
addendum for the current review posture.

The Rust migration contains substantial technical work and many promising ports, but it is not merge-ready. The largest risks are integration and governance risks, not isolated Rust numerical kernels:

- `spicekit` is treated as a hard runtime dependency but is not declared in Python packaging metadata.
- CI/PDM scripts reference deleted or missing files and will fail as written.
- Public Python module compatibility has been broken by deleting modules that still exist on baseline main.
- Runtime behavior is inconsistent: decisions say Rust is mandatory, but `_rust/api.py` still exposes nullable fallback wrappers and production code uses `assert` as a runtime guard.
- The migration checkout has not integrated current baseline main, including the docs/RTD/CI overhaul in baseline commit `22a1efa3`.
- Some parity/performance reporting overstates public API coverage because raw Rust kernels are measured where public dispatch intentionally takes a different path.
- Benchmark governance still contains stale live-legacy tooling even though the journal correctly documents that live legacy benchmarks became contaminated by Rust-backed fallthroughs.

The branch should go through a stabilization pass before approval. The stabilization should focus on packaging, CI, public API compatibility, runtime failure semantics, baseline rebase, and status/gate accuracy.

## Scope And Constraints

The user requested a thorough critique of the adam-core Rust migration compared with baseline main, including journal, decisions, and implementation review. The user initially requested no source changes. This file was created later by explicit request as a handoff artifact.

This review is static/inspection-based. I did not run the full build or test suite because those commands can update build caches/artifacts and the original review request explicitly said not to change anything. I did inspect files, git state, diffs, line references, scripts, workflow metadata, and project journal/decisions.

## Checkouts Reviewed

### Migration Checkout

Path: `/Users/aleck/Code/adam-core-rust-migration`

Observed state during review:

- Branch: `main`
- HEAD: `f556b5e7a87324d38e9d11953bb724eb56032968`
- `origin/main`: `f556b5e7a87324d38e9d11953bb724eb56032968`
- `origin` remote points to local `/Users/aleck/Code/adam-core`
- Working tree is heavily dirty relative to migration HEAD.
- Tracked diff size observed: 60 tracked files changed, 2639 insertions, 6023 deletions.
- Notable untracked migration artifacts/files include `rust/`, `migration/`, `src/adam_core/_rust/`, root `Cargo.toml`, `Cargo.lock`, `uv.lock`, `conftest.py`, docs/reference additions, and workflow changes.
- Ignored build/cache artifacts include `target/` and Python `__pycache__/` directories.

Static hygiene:

- `git diff --check HEAD --` was clean at review time.

### Baseline Checkout

Path: `/Users/aleck/Code/adam-core`

Observed state during review:

- Branch: `main`
- HEAD: `22a1efa3979cad5651e3f0765b2536983be6ab99`
- Commit summary: `Docs: RTD-first narrative overhaul and real-world use-case coverage (#194)`
- Baseline is one commit ahead of the migration base.
- Baseline checkout had only unrelated untracked local files such as `.dockerignore`, `.gcloudignore`, grounding files, docs build artifacts, and screenshots.

Important baseline change not integrated by migration:

- Commit `22a1efa3` changed 55 files, including `.github/workflows/pip-build-lint-test-coverage.yml`, `.gitignore`, `.readthedocs.yaml`, many `docs/source/*` files, `pyproject.toml`, `src/adam_core/dynamics/plots.py`, and `test_impact_viz_data.py`.
- Baseline `docs/source/index.rst` is a substantially rewritten RTD landing page and toctree with `Reference <reference/index>`.
- Migration `docs/source/index.rst` appears based on the older docs structure and only adds `reference/rust_backend_contracts`.

## Grounding From decisions.md And journal.md

Important decisions that should govern the migration:

- 2026-04-16: Execute migration in sibling checkout `adam-core-rust-migration`.
- 2026-04-16: Python package surface remains stable while implementation moves to Rust via PyO3/maturin.
- 2026-04-16: Rust implementations must clear parity and performance gates before default cutover.
- 2026-04-16: Default switch requires at least +20% p50/p95 speedup versus legacy.
- 2026-04-16: Existing pytest unit/integration suites must run through Python bindings with Rust backend enabled, not only custom parity tests.
- 2026-04-16 review period: temporary legacy fallback was accepted to keep legacy vs Rust runnable side by side, but this was explicitly temporary.
- 2026-04-17: `coordinates.transform_coordinates` promoted to rust-default for supported representation/frame workloads.
- 2026-04-17: Pure Cartesian to Cartesian frame-only rotations intentionally remain outside the Rust dispatcher because the legacy cached matrix path was faster at that time.
- 2026-04-20: Production SPICE access must route through `utils/spice_backend.py`; production modules should not import `spiceypy` directly.
- 2026-04-20: `spicekit` is the extracted pure-Rust SPICE/NAIF package; adam-core consumes it rather than owning all low-level NAIF reader code.
- 2026-04-20: `adam_core_py` Rust crate depends on `spicekit` via a git-pinned dependency, with an open item that private `spicekit` can break CI unless public or a deploy key is configured.
- 2026-04-21: `sp.prop2b` oracle tests were replaced with self-consistency roundtrip plus energy/angular-momentum conservation tests. The decision records a tolerance of 100 m position / 1 mm/s velocity over a 10,000-day roundtrip.
- 2026-04-22: No rustless adam-core environment should remain; Rust extension is mandatory at import time. JAX fallbacks may remain only as explicit parity references, not production safety nets.
- 2026-04-23: Live `rust_backend_benchmark_gate.py` legacy comparisons became contaminated because the supposed legacy path internally fell through to Rust-backed methods. The journal recommends freezing historical snapshots and switching to Rust-only latency baselines.
- 2026-04-25: Constitutional parity harness was created against baseline `/Users/aleck/Code/adam-core` pinned to upstream main `22a1efa3`; 21 APIs passed randomized fuzz at that time.
- 2026-04-25: Photometry warm speedups remained below/near the 1.2x gate for four APIs, while cold Rust performance was much faster due avoiding JAX import/JIT costs. Decision was deferred: use SIMD math, accept warm parity, or add waivers.
- 2026-04-25: `gaussIOD` randomized parity was intentionally unwired because Rust and legacy root solvers find different subsets of polynomial roots on random triplets.
- 2026-04-25: Remaining gaps included `calculate_perturber_moids`, `generate_porkchop_data`, and harder `transform_coordinates` cases such as ITRF93/origin translation.
- 2026-04-27: `calculate_chi2`, `weighted_mean`, and `weighted_covariance` were ported. Journal claims 784 pass / 23 skip in a full sweep. One least-squares test flaked once under suite pollution and passed cleanly on retry.

## Files And Areas Inspected

Key files inspected in the migration checkout:

- `/Users/aleck/Code/adam-core-rust-migration/decisions.md`
- `/Users/aleck/Code/adam-core-rust-migration/journal.md`
- `/Users/aleck/Code/adam-core-rust-migration/pyproject.toml`
- `/Users/aleck/Code/adam-core-rust-migration/uv.lock`
- `/Users/aleck/Code/adam-core-rust-migration/conftest.py`
- `/Users/aleck/Code/adam-core-rust-migration/.gitignore`
- `/Users/aleck/Code/adam-core-rust-migration/.github/workflows/pip-build-lint-test-coverage.yml`
- `/Users/aleck/Code/adam-core-rust-migration/.github/workflows/publish.yml`
- `/Users/aleck/Code/adam-core-rust-migration/docs/source/index.rst`
- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/_rust/api.py`
- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/_rust/status.py`
- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/utils/spice_backend.py`
- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/orbits/spice_kernel.py`
- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/coordinates/transform.py`
- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/coordinates/cartesian.py`
- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/coordinates/covariances.py`
- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/coordinates/residuals.py`
- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/dynamics/propagation.py`
- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/dynamics/tests/test_propagation.py`
- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/photometry/magnitude.py`
- `/Users/aleck/Code/adam-core-rust-migration/migration/parity/_inputs.py`
- `/Users/aleck/Code/adam-core-rust-migration/migration/parity/_rust_runner.py`
- `/Users/aleck/Code/adam-core-rust-migration/migration/parity/parity_main.py`
- `/Users/aleck/Code/adam-core-rust-migration/migration/parity/parity_speed.py`
- `/Users/aleck/Code/adam-core-rust-migration/migration/waivers.yaml`
- `/Users/aleck/Code/adam-core-rust-migration/rust/adam_core_py/src/lib.rs`
- `/Users/aleck/Code/adam-core-rust-migration/rust/adam_core_rs_coords/src/chi2.rs`
- `/Users/aleck/Code/adam-core-rust-migration/rust/adam_core_rs_coords/src/weighted.rs`

Key baseline files inspected:

- `/Users/aleck/Code/adam-core/docs/source/index.rst`
- `/Users/aleck/Code/adam-core/pyproject.toml`
- `/Users/aleck/Code/adam-core/.github/workflows/pip-build-lint-test-coverage.yml`
- Public modules that were deleted in migration but exist in baseline, including dynamics and coordinate helper modules listed below.

## Findings

### 1. Blocker: `spicekit` Is A Hard Runtime Dependency But Is Not Declared

Evidence:

- `pyproject.toml` dependencies do not include `spicekit`.
- `pyproject.toml` removed baseline `spiceypy` from runtime dependencies.
- `uv.lock` still includes `spiceypy` and does not show `spicekit`; `uv.lock` itself is untracked in the migration checkout.
- `_rust/api.py` imports `spicekit` dynamically and sets `SPICEKIT_AVAILABLE` based on import success.
- `utils/spice_backend.py` raises if `SPICEKIT_AVAILABLE` is false.
- `orbits/spice_kernel.py` creates a NAIF SPK writer through `naif_spk_writer`; if unavailable, SPK writing fails.

Line references:

- `/Users/aleck/Code/adam-core-rust-migration/pyproject.toml`: dependencies around lines 27-53. No `spicekit` is present.
- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/_rust/api.py`: optional `spicekit` import around lines 29-35.
- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/_rust/api.py`: NAIF wrapper functions return `None` when `SPICEKIT_AVAILABLE` is false around lines 802-880.
- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/utils/spice_backend.py`: `RustBackend.__init__` raises if `SPICEKIT_AVAILABLE` is false around lines 98-103.
- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/utils/spice_backend.py`: `get_backend()` raises if `SPICEKIT_AVAILABLE` is false around lines 372-388.
- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/orbits/spice_kernel.py`: `naif_spk_writer("adam-core")` is used around lines 129-134.

Impact:

- A clean install of adam-core from this branch can import/build without installing `spicekit`, then fail at runtime for SPICE operations.
- A published wheel would not carry its true runtime dependency.
- CI and user environments may accidentally pass only because `spicekit` is installed locally from adjacent development work.
- This conflicts with the architectural decision that `spicekit` is the extracted dependency and adam-core consumes it explicitly.

Recommended action:

- Decide whether `spicekit` is a Python package dependency, a Rust crate only, or both.
- If Python `spicekit` is required by `_rust/api.py`, declare it in `pyproject.toml` and lock it.
- If the package is not yet public, do not pretend this branch is releasable; use a private index/deploy key strategy or keep the branch blocked until publication.
- Add a clean-install CI job or script that creates a fresh environment and imports/runs SPICE backend operations without relying on local editable packages.

### 2. Blocker: CI And PDM Scripts Are Stale And Reference Missing Files

Evidence:

- `pyproject.toml` defines `rust-smoke`, `rust-parity-randomized`, and `rust-od-benchmark` scripts that point at missing files.
- Workflow `.github/workflows/pip-build-lint-test-coverage.yml` invokes those scripts.
- Missing targets observed:
  - `src/adam_core/tests/test_rust_orbit_determination.py`
  - `src/adam_core/tests/test_rust_parity_randomized.py`
  - `src/adam_core/dynamics/tests/test_kepler.py`
  - `migration/scripts/rust_orbit_determination_benchmark.py`
- Existing related targets observed:
  - `src/adam_core/tests/test_rust_backends.py`
  - `src/adam_core/coordinates/tests/test_transforms_spherical.py`
  - `src/adam_core/coordinates/tests/test_transforms_keplerian.py`
  - `migration/scripts/rust_backend_benchmark_gate.py`
  - `migration/parity/parity_main.py`
  - `migration/parity/parity_speed.py`

Line references:

- `/Users/aleck/Code/adam-core-rust-migration/pyproject.toml`: `rust-smoke` references missing tests around lines 149-152.
- `/Users/aleck/Code/adam-core-rust-migration/pyproject.toml`: `rust-parity-randomized` references a missing test around line 153.
- `/Users/aleck/Code/adam-core-rust-migration/pyproject.toml`: `rust-od-benchmark` references a missing script around line 155.
- `/Users/aleck/Code/adam-core-rust-migration/.github/workflows/pip-build-lint-test-coverage.yml`: rust smoke/parity/perf steps invoke these scripts around lines 133-138.

Impact:

- CI will fail before reaching meaningful Rust correctness checks.
- Local developer commands advertised by the project are not trustworthy.
- This undermines the stated validation contract that existing pytest suites plus Rust-enabled tests are the acceptance criteria.

Recommended action:

- Replace stale script targets with the current `migration/parity/parity_main.py` and `migration/parity/parity_speed.py` flow, or recreate the missing tests/scripts intentionally.
- Ensure every PDM script referenced by CI exists and runs from a clean checkout.
- Add a quick script self-check in CI or preflight to catch missing script target paths.

### 3. High: Public Python Package Surface Was Broken By Deleted Modules

Evidence:

The migration deletes modules that exist on baseline main and are importable public package paths. Examples include:

- `/Users/aleck/Code/adam-core/src/adam_core/dynamics/aberrations.py`
- `/Users/aleck/Code/adam-core/src/adam_core/dynamics/barker.py`
- `/Users/aleck/Code/adam-core/src/adam_core/dynamics/chi.py`
- `/Users/aleck/Code/adam-core/src/adam_core/dynamics/kepler.py`
- `/Users/aleck/Code/adam-core/src/adam_core/dynamics/lagrange.py`
- `/Users/aleck/Code/adam-core/src/adam_core/dynamics/stumpff.py`
- `/Users/aleck/Code/adam-core/src/adam_core/coordinates/jacobian.py`

Journal context:

- An earlier journal entry indicated `dynamics/aberrations.py` and related chain remained alive by necessity for n-body propagator behavior.
- A later journal entry says a deletion wave removed the old chain after porting light-time.

Impact:

- External users importing these modules will break even if the top-level workflows still pass.
- This violates the explicit migration decision to keep the Python package surface stable.
- This is a compatibility issue separate from whether the internal implementation moved to Rust successfully.

Recommended action:

- Restore deleted public modules as compatibility shims.
- Re-export Rust-backed implementations or thin wrappers from those module paths.
- Add import-compatibility tests comparing baseline public module paths with migration public module paths.
- If any public path is intentionally removed, require an explicit deprecation/removal decision and release-note entry, not an incidental migration deletion.

### 4. High: Runtime Contract Is Inconsistent And Uses `assert` For Production Failure Handling

Evidence:

- Decisions say no rustless adam-core environment should remain and Rust extension is mandatory at import time.
- `_rust/api.py` still documents review-period behavior where wrappers return `None` when Rust extension is unavailable.
- `_rust/api.py` catches any exception importing `_rust_native` and sets `RUST_BACKEND_AVAILABLE = False`.
- Many production modules call Rust wrappers, then use `assert out is not None`.
- `conftest.py` only enforces Rust backend availability when `ADAM_CORE_REQUIRE_RUST_BACKEND` is set; default tests do not necessarily enforce the no-rustless rule.

Line references:

- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/_rust/api.py`: review-period nullable fallback docstring around lines 1-13.
- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/_rust/api.py`: import catch and `RUST_BACKEND_AVAILABLE = False` around lines 21-27.
- `/Users/aleck/Code/adam-core-rust-migration/conftest.py`: backend requirement is env-gated around lines 8-24.
- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/photometry/magnitude.py`: production `assert out is not None` around line 152 and similar asserts around lines 276, 322, and 446.
- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/coordinates/residuals.py`: `calculate_chi2` uses `assert out is not None` around line 448.
- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/dynamics/propagation.py`: 2-body propagation paths assert Rust outputs around lines 208 and 214.
- Similar production asserts were observed in `coordinates/covariances.py`, `coordinates/cometary.py`, `coordinates/spherical.py`, `coordinates/transform.py`, `coordinates/geodetics.py`, `coordinates/keplerian.py`, `orbit_determination/gibbs.py`, `orbit_determination/herrick_gibbs.py`, `orbit_determination/gauss.py`, `missions/porkchop.py`, `propagator/propagator.py`, `orbits/variants.py`, `orbits/classification.py`, `dynamics/lambert.py`, `dynamics/moid.py`, and `dynamics/propagation.py`.

Impact:

- `assert` statements are removed under `python -O`; production behavior changes under optimization.
- Missing Rust can become obscure errors such as `np.asarray(None)` behavior or unpacking failures instead of clear backend initialization errors.
- The code simultaneously claims to support nullable fallback wrappers and to have no rustless production environment.
- Catching all exceptions when importing `_rust_native` can hide real ABI/import bugs and convert them into late runtime `None` behavior.

Recommended action:

- Pick one runtime contract and enforce it consistently.
- If Rust is mandatory, `_rust/api.py` should fail loudly at import time or expose a `_require_rust()` helper that raises a specific exception.
- Do not use `assert` for runtime validation of backend availability or FFI results.
- Replace nullable return wrappers with explicit exceptions unless a given function is genuinely optional and documented as such.
- Make CI run with backend requirement enabled by default, not only when an environment variable is manually set.

### 5. High: Migration Has Not Integrated Current Baseline Main

Evidence:

- Baseline main checkout is at `22a1efa3979cad5651e3f0765b2536983be6ab99`.
- Migration checkout HEAD/base is `f556b5e7a87324d38e9d11953bb724eb56032968`.
- Baseline commit `22a1efa3` is a docs/RTD-first overhaul and touches 55 files.
- Migration docs and CI appear based on the pre-`22a1efa3` structure.
- Baseline added docs reference structure and docs build/check tooling; migration does not reflect those changes.

Line references:

- `/Users/aleck/Code/adam-core/docs/source/index.rst`: baseline RTD landing page and reference toctree, including `Reference <reference/index>` around line 122.
- `/Users/aleck/Code/adam-core-rust-migration/docs/source/index.rst`: migration old index with `reference/rust_backend_contracts` insertion around line 20.
- `/Users/aleck/Code/adam-core-rust-migration/pyproject.toml`: optional dependencies around lines 55-64 do not include the baseline docs extras from `22a1efa3`.
- `/Users/aleck/Code/adam-core-rust-migration/.github/workflows/pip-build-lint-test-coverage.yml`: lacks the baseline docs CI job added by the docs overhaul.

Impact:

- Even if Rust migration tests pass, the branch would regress baseline docs, docs CI, and associated project metadata.
- Reviewers cannot cleanly distinguish Rust migration changes from stale baseline drift.
- The docs contract for new Rust backend reference material is not integrated into the current docs architecture.

Recommended action:

- Rebase or merge current baseline main before final review.
- Re-apply Rust docs additions into the new baseline docs/reference structure.
- Preserve baseline docs extras and docs CI unless there is a separate approved decision to remove them.

### 6. High: `coordinates.transform_coordinates` Parity Coverage Does Not Match Public Dispatch Behavior

Evidence:

- `migration/parity/_inputs.py` tests cartesian-to-cartesian frame rotation by feeding raw NumPy arrays to the Rust kernel.
- `migration/parity/_rust_runner.py` calls `_rust.api.transform_coordinates_numpy` directly.
- Public `transform_coordinates` support logic in `coordinates/transform.py` explicitly excludes Cartesian to Cartesian representation output.
- Journal entry 2026-04-25 says the cart-to-cart frame rotation parity gap was closed using raw Rust and reports a large speedup.
- Decision 2026-04-17 says pure Cartesian-to-Cartesian frame-only rotations remain on the legacy path because that was faster under the public dispatcher at that time.

Line references:

- `/Users/aleck/Code/adam-core-rust-migration/migration/parity/_inputs.py`: cart-to-cart parity sample around lines 82-100.
- `/Users/aleck/Code/adam-core-rust-migration/migration/parity/_rust_runner.py`: direct raw Rust transform call around lines 31-45.
- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/coordinates/transform.py`: support predicate around lines 204-270.
- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/coordinates/transform.py`: explicit Cartesian output exclusion around lines 256-263.
- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/_rust/status.py`: `coordinates.transform_coordinates` registry entry around lines 75-80.

Impact:

- The parity report can be technically correct for the raw kernel while misleading for the public API.
- Status `default="rust"` for `coordinates.transform_coordinates` is too coarse if important public call shapes intentionally bypass the raw Rust dispatcher.
- Reviewers may believe a public dispatch path has parity/performance coverage when only a lower-level kernel path was tested.

Recommended action:

- Split status/coverage into raw-kernel coverage and public-dispatch coverage.
- Add public API parity tests for the actual `transform_coordinates(...)` dispatcher path.
- If Cartesian-to-Cartesian is intentionally routed through `CartesianCoordinates.rotate`, document that as a separate Rust-backed public path rather than claiming raw dispatcher coverage.
- Update `status.py` to encode unsupported or intentionally excluded cases.

### 7. Medium/High: Benchmark Governance Still Points At Stale Live-Legacy Gates

Evidence:

- Journal correctly documents that the live `rust_backend_benchmark_gate.py` became contaminated because the supposed legacy path now calls Rust-backed methods internally.
- `.gitignore` marks `migration/artifacts/rust_benchmark_gate.json` as stale and superseded.
- `pyproject.toml` still has `rust-perf-gate` pointing to `migration/scripts/rust_backend_benchmark_gate.py` and outputting the stale artifact.
- Workflow uploads or expects `migration/artifacts/rust_benchmark_gate.json`.

Line references:

- `/Users/aleck/Code/adam-core-rust-migration/pyproject.toml`: `rust-perf-gate` around line 154.
- `/Users/aleck/Code/adam-core-rust-migration/.gitignore`: stale benchmark artifact note around lines 123-127.
- `/Users/aleck/Code/adam-core-rust-migration/.github/workflows/pip-build-lint-test-coverage.yml`: benchmark gate and upload behavior around lines 138-144.

Impact:

- CI may enforce an invalid performance comparison.
- CI may fail on stale artifact behavior.
- Performance governance is split between journal-described current strategy and committed script metadata.
- The +20% gate can be satisfied, failed, or bypassed for reasons unrelated to true Rust-vs-baseline performance.

Recommended action:

- Remove or retire `rust_backend_benchmark_gate.py` from active CI unless it is rewritten to avoid contaminated legacy paths.
- Promote `migration/parity/parity_speed.py` or a Rust-latency-baseline script as the active gate.
- Ensure artifacts uploaded by CI match the current benchmark strategy.
- Explicitly record waivers for APIs that pass cold latency but fail warm +20% p50/p95, especially photometry.

### 8. Medium: Propagation Tests Lost An Independent Oracle

Evidence:

- Baseline tests used `spiceypy.sp.prop2b` as an independent math oracle.
- Migration replaced this with Rust-backed self-consistency via forward/backward roundtrip plus conservation of energy and angular momentum.
- The invariants are useful but are not equivalent to an independent implementation oracle.
- The current test comment allows 100 m / 1 m/s, while the decision records 100 m / 1 mm/s.

Line references:

- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/dynamics/tests/test_propagation.py`: `_propagate_2body_single` calls Rust wrapper around lines 21-33.
- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/dynamics/tests/test_propagation.py`: roundtrip/invariant test helper around lines 46-79.
- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/dynamics/tests/test_propagation.py`: comment and thresholds around lines 68-72.

Impact:

- Shared implementation bugs can survive if forward and backward propagation share the same bug.
- Sign/unit mistakes may preserve some invariants while still being wrong relative to an external oracle.
- The tolerance mismatch between decision and test weakens confidence in the documented validation contract.

Recommended action:

- Add at least one independent oracle fixture that does not call the same Rust implementation.
- If `sp.prop2b` is intentionally removed because `spicekit` has no equivalent, store fixed expected vectors from a trusted oracle as test fixtures.
- Reconcile the velocity tolerance: either tighten to 1 mm/s as recorded or update the decision with rationale for 1 m/s.

### 9. Medium: Migration Status Registry Is Not A Trustworthy Source Of Truth

Evidence:

- `src/adam_core/_rust/status.py` describes itself as the single source of truth for migration state.
- Every listed API has `status="dual"` and `default="rust"`.
- Several legacy paths have been removed or are no longer true production alternatives.
- Some entries are raw kernel status rather than public dispatch status.
- Waiver state appears disconnected from the active registry/status story.

Line references:

- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/_rust/status.py`: module docstring around lines 1-5.
- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/_rust/status.py`: status fields around lines 21-28.
- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/_rust/status.py`: all API entries around lines 31-186.

Impact:

- Governance reports can be inaccurate.
- A reviewer cannot tell which APIs are truly dual, rust-only, raw-kernel-only, or public-rust-default.
- Waivers and unsupported subcases become easy to lose.

Recommended action:

- Extend statuses to include at least `legacy`, `dual`, `rust-only`, `raw-kernel-only`, and `public-rust-default` or equivalent.
- Track subcase exclusions for broad APIs like `transform_coordinates`.
- Make status generation fail if it claims dual support but the legacy implementation/module is gone.

### 10. Medium: Runtime Dependency Cleanup Is Incomplete

Evidence:

- `pyproject.toml` still includes heavy dependencies such as `jax`, `jaxlib`, and `numba` in runtime dependencies.
- The migration removed many JAX production paths and describes retained JAX code as parity references.
- If these dependencies are no longer production runtime requirements, they should not remain in core runtime dependencies.
- If they are still required, the remaining production callsites should be explicitly documented.

Line references:

- `/Users/aleck/Code/adam-core-rust-migration/pyproject.toml`: runtime dependencies around lines 27-53.

Impact:

- Users pay installation cost for unused heavy dependencies.
- The packaging story is unclear: `spicekit` is omitted while potentially obsolete dependencies remain.
- It is harder to validate that the Rust migration actually reduced runtime dependency surface.

Recommended action:

- Audit all production imports of `jax`, `jaxlib`, and `numba`.
- Move parity/test-only dependencies into optional/test dependency groups.
- Keep runtime dependencies minimal and aligned with the no-rustless contract.

### 11. Medium: `calculate_chi2` Introduces A Public Behavior Change For Non-SPD Covariances

Evidence:

- Baseline used `np.linalg.inv` and would accept some invertible indefinite covariance matrices.
- Rust implementation uses Cholesky decomposition and rejects non-positive-definite covariance matrices.
- This is numerically and scientifically defensible for covariance matrices, but it is still a public behavior change if callers previously passed invertible non-SPD matrices.

Line references:

- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/coordinates/residuals.py`: warning and validation around lines 429-440.
- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/coordinates/residuals.py`: Rust call and assert around lines 442-449.
- `/Users/aleck/Code/adam-core-rust-migration/rust/adam_core_rs_coords/src/chi2.rs`: Cholesky/positive-definite rejection around lines 99-100.

Impact:

- Some existing caller inputs may now raise where they previously returned a value.
- This is probably the right mathematical behavior, but it needs an explicit test and release note.

Recommended action:

- Document that covariance matrices must be symmetric positive definite.
- Add tests for non-SPD input behavior.
- Prefer explicit Python-side error message before entering Rust if that makes user diagnostics clearer.

### 12. Medium: Publish/Wheel Build Story Needs Clarification

Evidence:

- `pyproject.toml` uses `maturin` as the build backend.
- `publish.yml` still runs `pdm build`, then `pdm run rust-build` separately.
- Under a maturin build backend, the relation between `pdm build`, `maturin build`, generated wheels, and uploaded artifacts needs to be explicit.

Line references:

- `/Users/aleck/Code/adam-core-rust-migration/pyproject.toml`: build system and `tool.maturin` configuration around lines 66-75.
- `/Users/aleck/Code/adam-core-rust-migration/.github/workflows/publish.yml`: build sequence around lines 25-30.

Impact:

- Release automation may build the wrong artifact, duplicate artifacts, or fail to include the native extension as expected.
- This is especially important because the migration declares Rust mandatory.

Recommended action:

- Define one authoritative wheel build path.
- Test a built wheel in a clean environment and import `adam_core._rust_native`.
- Ensure the publish workflow uploads only artifacts known to contain the Rust extension.

### 13. Medium: Photometry Warm Performance Gate Is Unresolved

Evidence:

- Journal records four photometry APIs at warm parity or below the +20% gate after chunked Rayon fixes.
- Cold performance strongly favors Rust due avoiding JAX import/JIT cost.
- The decision was deferred among SIMD math, accepting warm parity, or waivers.

Impact:

- The original migration gate says Rust defaults require at least 1.2x p50/p95 speedup.
- If photometry remains Rust-default without waivers, governance is inconsistent.

Recommended action:

- Either implement a SIMD/libm strategy that clears warm gates, or record explicit waivers with rationale based on cold-start/real-world CLI usage.
- Ensure the active benchmark gate encodes the chosen policy rather than relying on journal narrative.

### 14. Medium: `gaussIOD` Randomized Parity Is Intentionally Unwired But Needs Registry-Level Visibility

Evidence:

- Journal records that `gaussIOD` runner adapters exist but randomized fuzz is unwired because Rust Laguerre/deflation and legacy `np.roots`/LAPACK find different root subsets on roughly 15% of random triplets.
- `status.py` still marks `orbit_determination.gaussIOD` as `status="dual"`, `default="rust"`.

Line references:

- `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/_rust/status.py`: `gaussIOD` entry around lines 172-178.

Impact:

- Governance can report randomized coverage that is not actually enforced.
- The known root-subset mismatch is important enough that it should be visible in status or waivers, not only journal prose.

Recommended action:

- Add a status/waiver note for `gaussIOD` randomized parity exclusion.
- Separate fixed-fixture parity from randomized fuzz parity in reports.

### 15. Low/Medium: Rust/PyO3 Binding Layer Is Large And Will Become Hard To Maintain

Evidence:

- `rust/adam_core_py/src/lib.rs` contains many PyO3 wrappers in a single large file.
- Recent wrappers such as weighted mean/covariance and chi-square do good validation, but the file is accumulating many unrelated APIs.

Line references:

- `/Users/aleck/Code/adam-core-rust-migration/rust/adam_core_py/src/lib.rs`: weighted wrappers around lines 751-805.
- `/Users/aleck/Code/adam-core-rust-migration/rust/adam_core_py/src/lib.rs`: chi-square wrapper around lines 807-842.
- `/Users/aleck/Code/adam-core-rust-migration/rust/adam_core_py/src/lib.rs`: propagation arc wrappers around lines 897-974.

Impact:

- Not a merge blocker by itself.
- Long-term maintainability and reviewability will degrade as more APIs are added.

Recommended action:

- After stabilization, split PyO3 bindings into modules by domain: coordinates, dynamics, photometry, orbit determination, SPICE.
- Keep validation at the PyO3 boundary and avoid duplicating shape rules across Python and Rust.

## Positive Technical Notes

The review should not be read as dismissing the implementation. There is strong technical work here:

- The migration includes substantial Rust kernel coverage across coordinates, photometry, propagation, ephemeris, orbit determination, Lambert/MOID, covariance, residuals, and SPICE-related paths.
- The journal shows real investigation discipline: benchmark contamination was identified, photometry Rayon overhead was diagnosed, hyperbolic chi solver regression was root-caused, and gaussIOD randomized mismatch was analyzed rather than hidden by tolerance bumps.
- The `spice_backend.py` direction is architecturally sound. Centralizing SPICE access behind a backend abstraction is preferable to scattered direct `spiceypy` imports.
- PyO3 wrappers often validate shapes/dimensions before entering Rust, which is the right boundary pattern.
- Recent `calculate_chi2` implementation is mathematically better than explicit inverse for SPD covariance matrices.
- `weighted_mean` and `weighted_covariance` are good candidates for Rust: small, deterministic, and easy to parity-test.
- Arc propagation work appears targeted at real OD inner-loop cost patterns rather than only synthetic batch throughput.
- `git diff --check` being clean is a good baseline hygiene signal.

## Risk Areas To Re-Review After Fixes

After the stabilization pass, a follow-up reviewer should re-check:

- Clean install from scratch, including `spicekit` availability.
- Built wheel import of `adam_core._rust_native`.
- All PDM scripts referenced by workflows.
- Full CI workflow behavior after baseline rebase.
- Public import compatibility against baseline module paths.
- Status registry accuracy against actual dispatch behavior.
- Performance gates after stale benchmark script removal.
- `python -O` behavior or equivalent validation that no production logic relies on `assert`.
- Docs build under the baseline RTD structure.
- Parity reports for public APIs, not only raw kernels.

## Recommended Stabilization Checklist

1. Declare/package `spicekit` correctly and validate clean install.
2. Rebase or merge current baseline main at `22a1efa3` or newer.
3. Preserve baseline docs/RTD/CI changes and re-home Rust docs into the new reference structure.
4. Remove or fix stale PDM scripts and workflow steps.
5. Replace nullable Rust wrappers and production `assert`s with explicit fail-fast exceptions.
6. Restore deleted public modules as compatibility shims or document intentional removals through a proper deprecation process.
7. Update `status.py` so it reflects actual public dispatch, raw kernel coverage, rust-only APIs, dual APIs, waivers, and unsupported subcases.
8. Retire contaminated live-legacy benchmark gate from active CI.
9. Promote the current parity/speed harness or Rust latency baseline strategy into CI.
10. Add independent oracle fixtures for propagation, or fixed trusted vectors if no runtime oracle remains.
11. Resolve photometry warm-gate policy with SIMD work or explicit waivers.
12. Add clean wheel/release validation under the maturin build backend.
13. Audit runtime dependencies and move parity/test-only dependencies out of core runtime dependencies.
14. Add public import compatibility tests comparing baseline-known module paths.
15. Re-run full rust-required pytest, cargo test, cargo fmt, cargo clippy, parity fuzz, and speed gates from a clean checkout.

## Bottom Line

The migration is technically promising but operationally inconsistent. It should not be approved or merged until the packaging, CI, baseline integration, public API compatibility, runtime contract, and governance-reporting issues are fixed. Once those are addressed, the Rust kernel work itself looks worth continuing and likely close to a viable migration path.

## Addendum: Claude Progress Transcript Reviewed On 2026-04-28

Source reviewed: `/Users/aleck/Code/adam-core/rust-migration-from-claude.txt`

Transcript size: 12,544 lines.

Purpose of this addendum: the original review above was written before reading the Claude progress transcript. The transcript contains later state and several corrections. This addendum should be read together with the original review. Where there is a conflict, this addendum reflects the newer inspected state as of 2026-04-28.

### Current Git State After Reading Transcript

The original review described the migration checkout as dirty on `main` at `f556b5e7`. That was true for the earlier inspection, but it is now stale.

Current migration checkout observed on 2026-04-28:

- Path: `/Users/aleck/Code/adam-core-rust-migration`
- Branch: `rust-migration-waves-d-e`
- HEAD: `74b8e02b579b8d1516532f50b39f3231644abd0c`
- Commit subject: `Rust migration: waves D + E1 + E2 — kernels, parity gate, perf wins`
- Working tree: clean before this addendum was appended.
- Commit size: 145 files changed, 39,856 insertions, 6,143 deletions.
- Migration repo `origin` currently resolves to local `/Users/aleck/Code/adam-core`, not GitHub, despite the transcript saying `origin` was set to GitHub near the end.
- Baseline checkout `/Users/aleck/Code/adam-core` is still on `main` at `22a1efa3979cad5651e3f0765b2536983be6ab99` and has a local branch plus remote-tracking branch named `rust-migration-waves-d-e`.
- Baseline checkout `origin` is `git@github.com:B612-Asteroid-Institute/adam_core.git`.

Interpretation:

- The work is no longer just dirty local changes; it has been collapsed into one large commit on `rust-migration-waves-d-e`.
- The transcript claims the branch was pushed to GitHub. Locally, the baseline checkout has `remotes/origin/rust-migration-waves-d-e`, which is consistent with that claim, but this addendum did not perform a network `ls-remote` verification.
- The migration checkout's current remote being local is a discrepancy against the transcript and should be verified before any follow-up push/PR work.

### Progress Captured In The Transcript

The transcript records a large amount of implementation, validation, and governance work after the earlier migration journal entries. Key progress:

1. Constitutional parity harness was built and run.

   Components added under `migration/parity/`:

   - `__init__.py`
   - `README.md`
   - `tolerances.py`
   - `_legacy_runner.py`
   - `_oracle.py`
   - `_inputs.py`
   - `_rust_runner.py`
   - `parity_fuzz.py`
   - `parity_speed.py`
   - `parity_main.py`

   Behavior recorded in transcript:

   - Legacy oracle runs in `.legacy-venv` against upstream JAX/numba baseline.
   - Rust runner calls the migration's `_rust.api` wrappers.
   - Randomized fuzz samples realistic asteroid/comet orbit inputs.
   - `parity_main.py` orchestrates parity and speed gates and emits JSON artifacts.
   - Initial all-API fuzz sweep covered 21 APIs × 8 seeds × 128 rows and passed after tolerance fixes.

2. Tolerance fixes and root-cause notes were added.

   Transcript records:

   - `coordinates.cartesian_to_keplerian` worst absolute difference was about `2.8e-10`, not `2e-7`; tolerance tightened to `1e-9`.
   - `coordinates.cartesian_to_cometary` received the same tightening.
   - `photometry.calculate_phase_angle` and fused magnitude+phase phase-angle tolerance moved from `1e-11` to `1e-10` degrees due a documented small-angle `atan2` ULP ceiling.
   - Propagation covariance tolerance remained looser because 6x6 covariance propagation accumulates floating-point drift through matrix products.

3. Live upstream/legacy bugs or mismatches were discovered while building the oracle.

   Transcript records three important findings:

   - Upstream `_keplerian_to_cartesian_p_vmap` expects semi-latus rectum `p`, not semi-major axis `a`; parity runner was corrected to use `_keplerian_to_cartesian_a_vmap`.
   - Cometary vmaps live in `coordinates.transform`, not `coordinates.cometary`.
   - Upstream `calcGauss` numba path has a strict signature mismatch: it calls `approxLangrangeCoeffs(r2_mag, t12)` with two args while the numba signature expects three. The legacy runner re-implements the relevant path inline with explicit `mu`.

4. Hyperbolic universal-Kepler divergence was fixed.

   Transcript records Task #138:

   - Rust `calc_chi` was reverted from Laguerre's method back to plain Newton-Raphson to match JAX reference behavior.
   - Failure mode: backward hyperbolic propagation through perihelion could diverge catastrophically under Laguerre branch selection.
   - Concrete case: 1I/'Oumuamua 10,000-day forward/backward roundtrip reportedly diverged to about `2.8e+20 AU` under Laguerre and returned to about `6e-13 AU` under Newton.
   - Hyperbolic propagation test caps were restored from 5,000 days to 10,000 days.
   - Validation recorded: cargo tests passed; dynamics/coordinates/propagator/OD pytest subset passed with 295 passed / 23 skipped; dynamics parity fuzz remained within prior worst-case levels.

5. Cold and warm speed gates were separated.

   Transcript records user pushback asking for both cold and warm comparisons and ensuring separate installations were used.

   Implemented behavior:

   - Warm timings run repeated calls inside already-started processes.
   - Cold timings spawn fresh subprocesses and include import/first-call cost.
   - Rust process uses migration `.venv`.
   - Legacy process uses `.legacy-venv` pinned to upstream/baseline.
   - `PYTHONPATH` is cleared to prevent cross-install leakage.

   Important result:

   - Warm gate: most APIs beat the 1.2x threshold, but four photometry APIs were only about `0.95x` to `1.12x` warm because XLA/Accelerate has SIMD-vectorized transcendentals while Rust stdlib transcendentals are scalar.
   - Cold gate: Rust was far faster across most APIs, at least `18x` in the journal/transcript summary, except `calc_mean_motion`, which was about `0.95x` cold because legacy does not pay a JAX import for that one-line operation.
   - This does not remove the photometry warm-gate policy gap; it clarifies it.

6. Photometry Rayon scheduling overhead was fixed.

   Transcript records:

   - Initial photometry Rust kernels used per-row Rayon tasks (`par_iter_mut().enumerate()` style), which made small elementwise transcendental kernels slower.
   - Kernels were changed to chunked parallelism, `par_chunks_mut(PHOT_CHUNK=1024)`.
   - This improved scaling but did not fully beat warm XLA/Accelerate transcendental performance.
   - SIMD options were considered; `sleef-sys` was called old/unmaintained, while `pulp` lacks transcendental functions. Hand-rolled polynomial approximations were considered multi-day/high-validation-risk work.

7. `transform_coordinates` raw cart-to-cart gap was closed at the kernel level.

   Transcript records:

   - Raw Rust `transform_coordinates_numpy` was fuzzed against a legacy public API oracle for cartesian ecliptic-to-equatorial frame rotations.
   - Reported parity: 4 seeds × 128 rows passed, worst absolute diff about `3.5e-18`, worst relative diff about `9.2e-14`.
   - Reported speed at n=2000: about `29.0x` p50 / `28.2x` p95.

   Review interpretation remains unchanged:

   - This is strong raw-kernel evidence.
   - It does not fully invalidate the earlier critique that public `transform_coordinates` dispatch accounting is too coarse, because the public dispatcher still has explicit exclusions/subcases.

8. Arc propagation API was added and wired into production dispatch.

   Transcript records:

   - New API: `propagate_2body_along_arc_numpy(orbit, dts, mu, max_iter, tol)` for single-orbit-many-times patterns.
   - New batch API: `propagate_2body_arc_batch_numpy`.
   - Rust computes orbit constants once and warm-starts Newton iterations along sorted `dt` values.
   - Dispatch heuristic routes single-orbit/many-time cases through the arc path for moderate `n_times` values, while large batches keep the parallel cold-start path.
   - Reported microbenchmarks: n=10 single-orbit arc around `10.6x` faster than cold batch; n=100 around `3.4x`; n=1000 and above parallel batch wins.
   - Production `_run_2body_propagate` dispatch was updated.
   - Full pytest sweep recorded: 784 passed / 23 skipped / 0 failed.

9. Reporter scripts and RCA artifacts were added.

   Transcript records additional scripts/artifacts:

   - `migration/scripts/parity_table.py`
   - `migration/scripts/perf_scaling_table.py`
   - `migration/scripts/perf_e1_e2_kernels.py`
   - `migration/artifacts/parity_table.json`
   - `migration/artifacts/parity_table_rca.json`
   - `migration/artifacts/perf_scaling_table.json`
   - `migration/artifacts/perf_e1_e2_kernels.json`

   Reported coverage after RCA:

   - 22 APIs measured directly.
   - 2 orchestration-implied APIs: `calculate_perturber_moids` and `generate_porkchop_data`.
   - 1 unwired randomized-fuzz API: `gaussIOD`, due intrinsic root-subset divergence between Rust Laguerre/deflation and legacy `np.roots`/LAPACK.
   - Claimed coverage: 24 of 25 declared APIs with parity, 96%; `gaussIOD` has runner adapters retained for fixed-fixture use.

10. RCA verdicts were generated for parity differences.

   Transcript records this breakdown across measured APIs:

   - 3 exact zero-diff bit-parity cases.
   - 13 last-bit / <=2 ULP bit-parity cases.
   - 5 equally accurate cases where both legacy and Rust pass standard candles and differ only by expected transcendental/angle ULP behavior.
   - 5 more-accurate Rust covariance cases, where Rust Dual<6> AD reportedly stays finite on stiff inputs where JAX `jacfwd` can overflow.
   - 0 less-accurate cases.

   Standard candles mentioned:

   - Keplerian period and derived quantities against JPL/Horizons-like truth.
   - Propagation roundtrip tests.
   - Ephemeris against Horizons at about `1e-10 deg` / `0.36 mas` tolerance.
   - Rust covariance autodiff tests.

   Review interpretation:

   - This improves confidence in numerical ports.
   - It does not remove the need for governance/status accuracy, public-dispatch coverage clarity, or clean CI packaging.

11. Surface-area audit was performed.

   Transcript records an audit of non-test Python source files:

   - Total counted Python sources: 100.
   - Bit-identical to upstream legacy: 66.
   - Altered or new in migration: 34.
   - Files already using Rust kernels: 19.
   - Altered without Rust use: 15.
   - Meaningful remaining Rust-port work estimated: about 17 files, including about 12 untouched hot-path files and about 5 altered-but-still-Python orchestration files.

   High-priority remaining areas listed:

   - `coordinates/variants.py`
   - `orbit_determination/differential_correction.py`
   - `orbit_determination/evaluate.py`
   - `orbit_determination/outliers.py`
   - `orbit_determination/least_squares.py`
   - `orbit_determination/od.py`
   - `orbit_determination/iod.py`
   - `dynamics/impacts.py`
   - `coordinates/residuals.py` outer orchestration

   Low-value or non-Rust areas listed:

   - `__init__.py` wiring
   - quivr schemas
   - I/O and external clients
   - plotting/viz
   - generic utilities
   - time module backed by ERFA/C
   - SPICE plumbing that already calls Rust-backed `spicekit`

12. Wave E1/E2 small-kernel ports were added.

   Transcript records these ports:

   - `dynamics/tisserand.py::calc_tisserand_parameter` to Rust.
   - `orbits/classification.py` orbital classification rules to Rust.
   - `coordinates/residuals.py::calculate_chi2` to Rust Cholesky solve.
   - `photometry/absolute_magnitude.py::_fit_absolute_magnitude_rows` single/grouped kernels to Rust.
   - `coordinates/residuals.py::bound_longitude_residuals` to Rust.
   - `coordinates/residuals.py::apply_cosine_latitude_correction` to Rust.

   Validation reported across these waves:

   - Tisserand: 4/4 pytest and cargo tests passed.
   - Classification: 39/39 pytest and cargo tests passed.
   - Chi2: residual and OD test subsets passed; one least-squares flake was observed once and attributed to pre-existing order-dependent FP drift.
   - Absolute magnitude: photometry tests passed and full sweep recorded 784/23/0.
   - Spherical residual helpers: coordinates/photometry subsets passed, cargo tests passed.

13. Weighted mean/covariance were initially ported, then reverted from production dispatch.

   This is a correction to the original review above.

   Original review said `weighted_mean` and `weighted_covariance` now dispatch to Rust. That was true of an intermediate state captured in the journal, but it is false in current commit `74b8e02b`.

   Current observed implementation:

   - `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/coordinates/covariances.py`: `weighted_mean` returns `np.dot(W, samples)`.
   - `/Users/aleck/Code/adam-core-rust-migration/src/adam_core/coordinates/covariances.py`: `weighted_covariance` returns `(W_cov * residual.T) @ residual`.
   - The comments explicitly state these dispatch to NumPy/BLAS, not Rust, because Apple Accelerate/OpenBLAS beats attempted pure-Rust/faer/hand-rolled loops.
   - Rust weighted kernels still exist in `rust/adam_core_rs_coords/src/weighted.rs` and PyO3 wrappers still exist, but they are no longer production dispatch for these functions.

   Impact on original critique:

   - Remove `weighted_mean`/`weighted_covariance` from the list of production Rust-only `assert` examples.
   - Keep the broader runtime-contract critique, because many other production sites still use nullable wrappers plus `assert`.
   - Treat weighted mean/cov as a positive example of reversing a Rust port when BLAS is better.

14. `faer` was added for future linear algebra but is not production-critical yet.

   Transcript records:

   - `faer 0.22` was added to the workspace.
   - It did not beat BLAS for the weighted mean/covariance case.
   - It remains available for future Cholesky/SymmetricEigen/SVD work, especially sigma-point sampling and least-squares kernel fusion.

15. Final E1/E2 performance table conclusion.

   Transcript records final conclusion after reverting weighted mean/covariance:

   - `calc_tisserand_parameter`: loses at small N, wins at scale.
   - `orbits.classification`: loses at small N, wins at scale.
   - `calculate_chi2`: loses at small N, wins strongly at scale.
   - `_fit_abs_mag_rows`: wins at all sizes after quickselect median fix.
   - `bound_longitude_residuals`: loses at small N, wins at scale.
   - `apply_cosine_latitude_correction`: loses at small N, wins at scale.
   - `weighted_mean`: reverted to NumPy/BLAS.
   - `weighted_covariance`: reverted to NumPy/BLAS.

   Review interpretation:

   - The Rust migration is now more honest about small-N PyO3 overhead.
   - The next performance lever is fusion of higher-level kernels, especially `Residuals.calculate`, not one-function-at-a-time PyO3 crossings.

16. Large single commit and push attempt.

   Transcript records:

   - User asked for one big commit rather than seven logical commits.
   - Branch created: `rust-migration-waves-d-e`.
   - Commit made: `74b8e02b Rust migration: waves D + E1 + E2 — kernels, parity gate, perf wins`.
   - First push went to local `/Users/aleck/Code/adam-core` because migration origin was a local filesystem clone.
   - Transcript then records setting origin to GitHub and pushing, with GitHub branch and PR creation URLs printed.

   Current observed discrepancy:

   - Migration checkout origin is currently back to `/Users/aleck/Code/adam-core`, not GitHub.
   - Baseline checkout has GitHub origin and has both local and remote-tracking `rust-migration-waves-d-e` branches.

   Follow-up:

   - Before opening a PR or pushing more changes, explicitly verify remotes and branch tracking from both checkouts.

### How The Transcript Changes The Original Review

The transcript improves the implementation story in several ways:

- The work is committed, not an uncommitted dirty tree.
- Parity governance is more developed than the original review implied: there are fuzz, speed, scaling, and RCA reporters with JSON artifacts.
- The numerical accuracy story is stronger: several loose tolerances have root-cause notes and standard-candle arguments.
- Hyperbolic propagation divergence was diagnosed and fixed.
- Weighted mean/covariance were reverted from production Rust dispatch after data showed BLAS was better.
- There is a concrete surface-area audit and future work plan.

The transcript does not remove these original blockers:

- `spicekit` is still a runtime requirement but is not declared in `pyproject.toml` as a Python dependency.
- `pyproject.toml` still contains stale script targets: `rust-parity-randomized` references a missing `src/adam_core/tests/test_rust_parity_randomized.py`, and `rust-od-benchmark` references a missing `migration/scripts/rust_orbit_determination_benchmark.py`.
- Workflow `.github/workflows/pip-build-lint-test-coverage.yml` still invokes those stale scripts.
- Public compatibility issue remains: deleted modules such as `dynamics/kepler.py`, `dynamics/lagrange.py`, `dynamics/stumpff.py`, `dynamics/aberrations.py`, `dynamics/barker.py`, `dynamics/chi.py`, and `coordinates/jacobian.py` still break baseline import paths.
- Nullable `_rust/api.py` wrappers plus production `assert` checks still conflict with the no-rustless production decision.
- Migration still has not integrated baseline main commit `22a1efa3` docs/RTD/CI overhaul cleanly.
- `status.py` still risks overstating public dispatch coverage and does not encode enough subcase/waiver detail.
- The stale live-legacy `rust_backend_benchmark_gate.py` still exists and remains referenced by active scripts, despite newer parity/speed/RCA tooling being better aligned with the documented benchmark strategy.
- Photometry warm-gate policy remains unresolved: accept cold-start wins, add waivers, or invest in SIMD/transcendental math.
- Propagation oracle concern is reduced but not eliminated: roundtrip and standard-candle checks are strong, but replacing independent `prop2b` oracle with self-consistency should remain explicit and justified.

### Updated Priority For The Next Agent

The next agent should not start by writing more Rust kernels. The highest-leverage work is integration hardening:

1. Verify branch/remotes and decide whether the authoritative branch is the migration checkout branch, the baseline checkout branch, or GitHub `rust-migration-waves-d-e`.
2. Fix `pyproject.toml` script targets and workflow references so CI can run the current parity/speed scripts instead of missing files.
3. Declare or otherwise package `spicekit` correctly for clean Python installs.
4. Restore public compatibility shims for deleted baseline modules, or record explicit breaking-change decisions.
5. Replace production `assert out is not None` patterns with explicit runtime errors or mandatory import-time Rust failure.
6. Rebase/merge baseline main `22a1efa3` docs/RTD/CI changes into the branch.
7. Update `status.py` to reflect measured direct coverage, orchestration-implied coverage, raw-kernel coverage, public-dispatch subcases, and `gaussIOD` randomized-fuzz exclusion.
8. Retire `rust_backend_benchmark_gate.py` from active CI or clearly mark it historical; promote `parity_main.py`, `parity_speed.py`, `parity_table.py`, and `perf_scaling_table.py` as current governance.
9. Only after the above, continue Wave E3 kernel fusion for `Residuals.calculate` / LSQ inner-loop performance.

### Updated Bottom Line

The Claude transcript materially improves confidence in the Rust numerical work and shows stronger engineering discipline than the original static review alone could see. It also confirms the migration has moved from dirty worktree to a large committed branch.

The approval answer is still the same: do not merge yet. The remaining blockers are packaging, CI correctness, baseline integration, public API compatibility, runtime contract cleanup, and governance/status accuracy. The next review should focus on those integration blockers before approving more kernel work.

## Addendum: Published spicekit Dependency Update On 2026-04-28

`spicekit` is now public as both a Rust crate and a Python package:

- crates.io: `spicekit = "0.1.0"`
- PyPI: `spicekit==0.1.0`

adam-core no longer directly depends on the Rust `spicekit` crate after
the `spicekit-py` carve-out; the NAIF PyO3 bindings live in the
published Python package. The concrete adam-core packaging fix is
therefore to depend on `spicekit>=0.1.0` in `pyproject.toml`, not to add
an unused Rust dependency to `rust/adam_core_py/Cargo.toml`.

This resolves the original review's highest-priority `spicekit` Python
packaging blocker once the lockfiles are refreshed and clean-install CI
confirms the published wheel imports correctly. The remaining blockers
from the review still stand: stale CI/PDM script references, deleted
public compatibility modules, nullable/assert runtime contract,
baseline docs/RTD drift, and status/governance accuracy.

## Addendum: Review Task Backlog Created On 2026-04-28

The follow-up task list now lives in:

- `/Users/aleck/Code/adam-core-rust-migration/migration/review_task_backlog_2026-04-28.md`

Important clarification: the previous addendum's "do not add a Rust
`spicekit` dependency to `adam_core_py`" guidance was correct for the
then-current Python-package-only carve-out. It is not sufficient if
`adam-core-rs` is intended to ship as a standalone Rust library.

The new RM-P0-001 task explicitly tracks direct Rust-to-Rust
`adam-core-rs` -> `spicekit` integration, with Python wrappers becoming
thin consumers of adam-core's Rust backend rather than importing Python
`spicekit` objects for adam-core SPICE operations.

## Addendum: Current Review Handoff After RM-P0 Completion On 2026-04-29

This addendum supersedes the unresolved-blocker list in the original
2026-04-27 review and the 2026-04-28 addenda. It should be the starting point
for the next reviewer.

### Current Branch And Commit State

- Migration checkout: `/Users/aleck/Code/adam-core-rust-migration`
- Branch: `rust-migration-waves-d-e`
- Current HEAD: `0bf2e3cd` (`Retire live-legacy benchmark governance`)
- Recent task commits, newest first:
  - `0bf2e3cd` Retire live-legacy benchmark governance
  - `da362611` Enforce mandatory Rust backend contract
  - `1125d7a2` Remove reference-only compatibility shims
  - `f2334505` Address compatibility wrapper review flags
  - `4ccc547b` Make compatibility helpers Rust-backed
  - `02481391` Track Rust-backed compatibility wrapper cleanup
  - `f9b0f25b` Audit Rust migration public compatibility surfaces
  - `f3e71ecf` Complete Rust migration P0 validation chunk
- Working tree after `0bf2e3cd` contains only `decisions.md` and `journal.md`
  modifications. These are local grounding files and should not be committed
  unless the user explicitly asks.

### What Changed Since The Original Review

The original review identified seven practical P0 blockers. Their current state:

| Original blocker | Current status |
|---|---|
| `spicekit` packaging/runtime ambiguity | Resolved for adam-core architecture: Python production no longer imports Python `spicekit`; `rust/adam_core_rs_spice` depends directly on public crates.io `spicekit = "0.1"`, and Python uses adam-core's Rust backend. |
| Stale PDM/CI scripts | Resolved by `script-preflight`, current `rust-parity-*` scripts, wheel inspection, and CI artifact-path checks. |
| Deleted public Python modules | Resolved for supported surfaces: public helper modules are restored as thin Rust-backed wrappers where retained; private/reference-only shims were removed intentionally and documented. |
| Nullable `_rust.api` wrappers / production `assert` guards | Resolved by mandatory Rust backend contract. `adam_core` imports `_rust`; `_rust/api.py` eagerly imports `_rust_native` and validates required symbols. Missing/stale native extension fails loudly. |
| Baseline docs/RTD drift | Partially addressed: Rust docs now live under `docs/source/reference/`, but `pdm run docs-check` remains blocked locally by missing Sphinx/stale lockfile behavior. RM-P1-010 remains open for final docs re-home/build validation. |
| Status/governance overstatement | Improved by parity table RCA and benchmark governance docs, but not complete. RM-P1-008 remains the next open task to make `status.py` encode richer coverage/status taxonomy. |
| Contaminated live-legacy benchmark governance | Resolved by RM-P0-007. Active speed comparisons are baseline-main subprocess parity speed for fair APIs and Rust-only latency regression for post-legacy APIs. Historical live-legacy artifacts are preserved under `migration/artifacts/history/`. |

### Current Architecture Decisions To Preserve

- The Rust extension is mandatory. There is no supported rustless production
  environment.
- Parity and performance comparisons use the sibling baseline-main checkout at
  `/Users/aleck/Code/adam-core` through `.legacy-venv`, not in-process JAX
  fallbacks inside the migration package.
- Python public helper APIs that remain supported should be thin Rust-backed
  wrappers. Rust code should call Rust directly, not route back through Python.
- Private compatibility shims and reference-only JAX helpers may be removed when
  they are not useful standalone utilities and not known downstream
  dependencies.
- `adam-core-rs` should call public Rust `spicekit` directly; Python
  `spicekit` is not a production boundary for adam-core Rust operations.
- New functional/performance changes should be validated with the standard
  cadence before moving to the next task.

### Validation Cadence To Keep Using

For each unique migration task, continue running:

```bash
pdm run script-preflight
pdm run rust-quality
pdm run test-rust-full
pdm run rust-parity-main
pdm run rust-parity-speed-cold
pdm run python -m migration.scripts.parity_table \
  --parity-artifact migration/artifacts/parity_gate.json \
  --speed-artifact migration/artifacts/parity_speed_cold_warm.json \
  --json-output migration/artifacts/parity_table_rca.json \
  --markdown-output migration/artifacts/parity_report.md
git diff --check
```

Important execution note: in this environment, `pdm run test-rust-full` may need
escalated permissions because Ray uses psutil/macOS `sysctl`, and some tests hit
network-backed JPL/Horizons/SPK paths. A non-escalated failure in those areas is
not automatically a code regression; rerun with the proper permissions before
triage.

### Latest Validation Results

Latest complete task validation was RM-P0-007, commit `0bf2e3cd`:

- Targeted Python checks: `py_compile`, `ruff`, and `black --check` passed for
  touched governance/parity scripts.
- `pdm run script-preflight`: passed (`27 PDM scripts`, `4 workflows`).
- `pdm run rust-latency-gate`: passed on rerun. The first run had one transient
  p95 outlier on `propagate_2body_with_covariance`; rerun showed worst p50
  regression ratio `1.293x` (`predict_magnitudes`) and worst p95 ratio
  `1.256x` (`solve_lambert`), both inside the default thresholds of `1.75x`
  p50 and `2.50x` p95.
- `pdm run rust-quality`: passed. Rust workspace tests passed for autodiff,
  coords, orbit determination, and SPICE crates.
- `pdm run test-rust-full`: passed with escalated permissions,
  `723 passed, 144 skipped, 2 deselected, 56 warnings`.
- `pdm run rust-parity-main`: passed. All 22 wired APIs passed randomized fuzz
  parity and the warm speed gate passed with existing waivers only.
- `pdm run rust-parity-speed-cold`: passed. Existing photometry warm-speed
  waivers were applied; no new waiver was introduced.
- Canonical parity/performance artifacts regenerated:
  - `migration/artifacts/parity_gate.json`
  - `migration/artifacts/parity_speed_cold_warm.json`
  - `migration/artifacts/parity_report.md`
  - `migration/artifacts/parity_table_rca.json`
- `git diff --check`: passed.

Docs validation caveat:

- `pdm run docs-check` could not run because `sphinx-build` is not installed in
  the active environment. Attempting `pdm install -G docs` wanted to refresh the
  stale `pdm.lock` hash and wrote outside the normal sandbox log path, so docs
  dependency state was left untouched. This is tracked as part of RM-P1-010.

### Current Parity And Performance Coverage

Current canonical report: `migration/artifacts/parity_report.md`.

Coverage summary:

- 22 of 25 declared APIs are wired directly in randomized-fuzz `GENERATORS`.
- 2 additional orchestration APIs are covered indirectly by underlying kernel
  parity:
  - `dynamics.calculate_perturber_moids`
  - `dynamics.generate_porkchop_data`
- 1 declared API remains intentionally unwired from randomized fuzz:
  - `orbit_determination.gaussIOD`

Reason for `gaussIOD` exclusion:

- Rust Laguerre+deflation and legacy `np.roots`/LAPACK can find different
  subsets of the 8th-order polynomial roots on random triplets. This is an
  algorithmic root-selection mismatch, not a direct kernel drift. Fixed-fixture
  parity/manual checks are the right next representation.

Performance state:

- Warm speed gates are green after applying existing temporary waivers.
- Cold speedups remain strong for Rust-backed paths because the baseline-main
  side pays fresh Python/JAX import/JIT costs.
- Photometry warm speed remains the known unresolved policy issue.
- `coordinates.cartesian_to_spherical` currently raw-passes but still has a
  temporary waiver because earlier p95 misses were reproducible enough to need a
  policy/SIMD decision rather than relying on one green run.

### Active Waivers And Review Dates

Active temporary waivers:

- `waiver-20260428-photometry-warm-performance-temporary`
  - Applies to four photometry APIs in the n=2000 warm speed gate.
  - Latest cold/warm artifact still shows waived warm misses or near-misses,
    with cold speedups around `29x` to `31x`.
  - Review by 2026-05-12.
- `waiver-20260428-cartesian-to-spherical-warm-performance-temporary`
  - Applies to prior repeated p95-only warm instability for
    `coordinates.cartesian_to_spherical`.
  - Latest artifact raw-passes, but the waiver remains because this needs a
    durable policy or SIMD/transcendental resolution.
  - Review by 2026-05-12.

### Current Governance And Packaging State

Packaging:

- Supported wheel path is `pdm run wheel-build` followed by
  `pdm run wheel-inspect`.
- Cargo package version in `rust/adam_core_py/Cargo.toml` is the wheel/runtime
  version source.
- `src/adam_core/_version.py` is generated during the wheel-version step.
- Clean pip install/import/native SPICE smoke passed during RM-P0-004.
- `uv` remains lock-only/local-install caveat: local uv `0.6.16` previously
  selected or retained PyPI `adam_core 0.5.5` without `_rust_native` for direct
  local wheel/develop tests. Do not use `maturin develop --uv` or direct
  `uv pip install dist/*.whl` as authoritative until revalidated on the intended
  uv version.

Benchmark governance:

- Active baseline-main parity/speed gates live under `migration/parity/`.
- Active Rust-only regression gate is
  `migration/scripts/rust_backend_benchmark_gate.py` writing
  `migration/artifacts/rust_latency_current.json`.
- Historical rust-vs-legacy evidence lives under `migration/artifacts/history/`.
- Current CI artifact contract uploads `rust-latency-current` from
  `migration/artifacts/rust_latency_current.json`.
- `migration/scripts/check_pdm_ci_scripts.py` rejects stale live-legacy flags and
  old artifact paths such as `--max-rust-over-legacy` and
  `migration/artifacts/rust_benchmark_gate.json`.

### Open Task List For The Next Agent

Next recommended order:

1. RM-P1-008 + RM-P1-015: make `status.py` a trustworthy registry and encode
   `gaussIOD` randomized-fuzz exclusion visibility in the same change.
   - Add richer taxonomy for public-rust-default, rust-only, raw-kernel-only,
     dual, partial coverage, and exclusions.
   - Encode `coordinates.transform_coordinates` subcases.
   - Encode `gaussIOD` randomized-fuzz exclusion.
   - Fail governance generation if a row claims dual support after the legacy
     implementation is gone.
2. RM-P1-009: add public dispatch parity for `coordinates.transform_coordinates`.
   - Raw Rust kernel parity is not enough for quivr/coordinate-object public
     dispatch.
   - Cover the intentional Cartesian-to-Cartesian frame-only exclusion if it is
     still retained.
3. RM-P1-010: re-home/finalize Rust docs in the current RTD structure, resolve
   lockfile/docs-dependency drift, and make `pdm run docs-check` pass
   locally/CI.
   - This owns the missing Sphinx/docs dependency issue and stale lockfile
     behavior disclosed in RM-P0-007.
4. RM-P1-011: audit runtime dependencies.
   - Production imports of `jax`, `jaxlib`, `numba`, `spiceypy`, and Python
     `spicekit` should be justified or moved to optional/test groups.
5. RM-P1-012: restore or replace an independent propagation oracle.
   - The baseline-main `.legacy-venv` oracle is adequate for migration parity
     today, but fixed trusted vectors or another independent propagation
     reference reduce single-oracle risk.
6. RM-P1-014 and RM-P1-014A: resolve temporary warm-performance waivers before
   the 2026-05-12 review date.
   - Decide SIMD/transcendental investment, cold-start waiver, or selective
     dispatch/revert policy.
7. RM-P1-013 / RM-WE2-001: document and test `calculate_chi2` SPD covariance
   contract.
   - Rust Cholesky rejects non-SPD covariance matrices. This is likely correct,
     but it is a public behavior change compared with `np.linalg.inv` accepting
     some merely invertible matrices.
8. RM-P1-018: harden Rust-only latency-gate statistical policy.
   - Define rerun policy, sample aggregation, and artifact requirements for
     pass-after-rerun cases.
9. RM-P1-016: split `rust/adam_core_py/src/lib.rs` into domain modules after the
   registry/status cleanup.
10. RM-P1-017: final clean validation pass before asking for broad merge review.
11. Wave work after governance cleanup:
    - RM-WD3-001 parallel backend abstraction for remaining Ray/rayon/sequential
      policy surfaces.
    - RM-WE2-002 fused `Residuals.calculate`.
    - RM-WE2-003 variants and covariance sampling linalg.
    - RM-WE2-004 OD evaluation/outlier helpers.
    - RM-WE3-001 least-squares inner-loop fusion.
    - RM-WE3-002 quivr-bound constitutional gaps.

### Reviewer Focus Areas

A reviewer should focus on these files and contracts first:

- `src/adam_core/_rust/status.py`: current registry is still too coarse.
- `migration/parity/README.md`, `migration/benchmark_governance.md`, and
  `docs/source/reference/rust_benchmark_governance.rst`: ensure benchmark
  governance wording is precise and does not overclaim live legacy speedups.
- `migration/artifacts/parity_report.md`: verify tolerance rationale and RCA are
  sufficient for each widened tolerance.
- `src/adam_core/dynamics/_rust_compat.py` and restored compatibility modules:
  verify retained public helper APIs are thin Rust-backed wrappers and not
  production hot-path detours through Python.
- `src/adam_core/_rust/api.py`: verify mandatory native-extension contract and
  required-symbol list are complete enough to catch stale wheels.
- `.github/workflows/*` and `pyproject.toml`: verify wheel-build/inspect,
  script-preflight, and artifact upload paths match the documented contracts.
- `migration/packaging.md`: review Cargo-version source and uv local-install
  caveat.

### Bottom Line For Review

The branch has moved from "not merge-ready P0 blockers" to "P0 stabilization
complete; ready for focused review." Do not evaluate the original 2026-04-27
blocker list as current without checking the status table above. The remaining
work is now governance fidelity, public-dispatch coverage, dependency/docs
cleanup, and performance-policy decisions before additional large kernel waves.
