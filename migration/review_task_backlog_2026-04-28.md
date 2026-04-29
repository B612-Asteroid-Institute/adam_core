# Rust Migration Review Task Backlog

Created: 2026-04-28

Purpose: handoff task list for the adam-core Rust migration after reviewing:

- `adam_core_rust_migration_review_handoff_2026-04-27.md`
- `/Users/aleck/Code/adam-core/rust-migration-from-claude.txt`
- `decisions.md`
- `journal.md`
- `migration/TODO.md`
- `migration/parity/README.md`
- `migration/artifacts/history/README.md`

Do not treat this as a wishlist. These are the concrete open tasks that must be either completed, explicitly waived, or intentionally deferred with owner/rationale before the migration branch is merge-ready.

## Current Operating Rule

Do not start more Rust kernel work before the integration hardening tasks unless the user explicitly reprioritizes. The numerical Rust work is strong enough to continue, but the branch is not merge-safe until packaging, CI, public compatibility, runtime contract, baseline integration, and governance are repaired.

## Baseline And Verification Discipline

Every functional or performance-significant change must be verified against the baseline main implementation as it existed before the Rust migration behavior changed.

Baseline setup:

- Baseline checkout: `/Users/aleck/Code/adam-core`
- Migration checkout: `/Users/aleck/Code/adam-core-rust-migration`
- Legacy oracle venv: `.legacy-venv` inside the migration checkout
- The parity harness assumes baseline `adam_core` is installed in `.legacy-venv` from `/Users/aleck/Code/adam-core`.

For every new Rust-default API or public dispatch change:

1. Add or update `src/adam_core/_rust/status.py`.
2. Add or update `migration/parity/tolerances.py`.
3. Add or update `migration/parity/_inputs.py`.
4. Add or update `migration/parity/_rust_runner.py`.
5. Add or update `migration/parity/_legacy_runner.py`.
6. Run the single-API parity and speed gate before deleting or contaminating the legacy reference path.
7. Preserve fair historical speed data if the legacy path is about to be removed.
8. If live legacy measurement is already contaminated, cite the frozen historical snapshot or mark the surface un-comparable; do not invent a live Rust-vs-legacy number.

Current commands:

```bash
.venv/bin/python -m migration.parity.parity_main --apis <api_id> --speed-n 2000
.venv/bin/python -m migration.parity.parity_fuzz --seeds 8 --n 128 --output migration/artifacts/parity_fuzz.json
.venv/bin/python -m migration.parity.parity_speed --n 2000 --reps 7 --output migration/artifacts/parity_speed.json
.venv/bin/python -m migration.parity.parity_speed --n 2000 --reps 21 --warmup 3 --cold --output migration/artifacts/parity_speed_cold_warm.json
```

Current quality gate commands:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
ADAM_CORE_REQUIRE_RUST_BACKEND=1 pdm run pytest --benchmark-skip -m 'not profile'
```

Rules:

- The default promotion gate remains at least 1.2x speedup at p50 and p95 unless the user approves a specific waiver.
- Cold-start wins may justify a waiver, but the waiver must be explicit and tied to workload reality.
- `migration/scripts/rust_backend_benchmark_gate.py` must not be used as a live Rust-vs-legacy gate unless it is proven to avoid Rust-backed fallthrough contamination. In the post-legacy world it is only acceptable as Rust-only latency regression tracking against `migration/artifacts/rust_latency_baseline.json`.
- Raw Rust kernel parity does not prove public API parity unless the public dispatcher actually calls that raw kernel for the tested shape.
- If a legacy oracle is removed, fixed trusted vectors or frozen historical artifacts must be created first.

## P0 Merge Blockers

### RM-P0-001: Direct Rust `spicekit` Integration For Standalone `adam-core-rs`

Status: complete; direct Rust-to-Rust `spicekit` integration validated (2026-04-28)

Reason: the current SPICE path is Python adam-core -> Python `RustBackend` -> Python package `spicekit` -> PyO3 -> Rust `spicekit`. That is Rust-backed for heavy work, but it is not direct Rust-to-Rust and is not suitable as the final architecture for a separately shipped `adam-core-rs`.

Scope:

- Add a direct Cargo dependency on public `spicekit = "0.1"` in the Rust crate that owns adam-core SPICE behavior.
- Introduce a Rust-side adam-core SPICE backend module or crate that owns kernel registration, SPK/PCK readers, text-kernel body bindings, last-loaded-wins semantics, `NotCovered` behavior, and batched `spkez`, `pxform`, `sxform`, and `bodn2c` access.
- Expose a Rust public API usable by standalone Rust consumers without Python.
- Expose thin PyO3 wrappers from `adam_core_py` over the adam-core Rust backend.
- Rewire Python `utils/spice_backend.py` so adam-core calls its own native extension for SPICE instead of importing Python `spicekit` objects.
- Decide whether the Python `spicekit>=0.1.0` dependency remains needed for any adam-core Python runtime behavior after this refactor. If it is no longer needed, remove it from runtime dependencies and validate clean install.

Acceptance:

- A Rust consumer can call adam-core SPICE functions through Rust APIs with no Python interpreter.
- adam-core Python SPICE operations work without importing the Python `spicekit` package unless an explicit retained dependency decision says otherwise.
- Existing `src/adam_core/utils/tests/test_spice_backend.py` coverage passes.
- CSpice parity remains owned by the `spicekit` repo's `spicekit-bench`; adam-core tests validate adam-core wiring and semantics.
- Clean wheel install exercises SPICE backend operations from adam-core without relying on editable sibling checkouts.

Verification:

```bash
cargo check --workspace
cargo test --workspace
ADAM_CORE_REQUIRE_RUST_BACKEND=1 pdm run pytest --benchmark-skip src/adam_core/utils/tests/test_spice_backend.py src/adam_core/utils/tests/test_spice.py src/adam_core/orbits/tests/test_spice_kernel.py
```

2026-04-28 implementation notes:

- Added `rust/adam_core_rs_spice`, an adam-core Rust crate that depends
  directly on public crates.io `spicekit = "0.1"`.
- Moved adam-core's SPICE backend state into Rust:
  `AdamCoreSpiceBackend` owns kernel registration, DAF idword dispatch,
  SPK/PCK reader lists, text-kernel body bindings, last-loaded-wins name
  resolution, and batched `spkez`, `pxform`, `sxform`, and `bodn2c`.
- Exposed the backend plus compatibility `NaifSpk`, `NaifPck`,
  `NaifSpkWriter`, and `naif_*` wrappers through `adam_core_py`.
- Rewrote `src/adam_core/utils/spice_backend.py` so Python owns only a
  process-local native backend object, a lock, and `NotCovered`
  exception mapping. It no longer imports Python `spicekit` objects.
- Removed Python runtime dependency `spicekit>=0.1.0` from
  `pyproject.toml` and `uv.lock`; adam-core runtime SPICE now reaches
  spicekit through Cargo, not through the Python package boundary.
- Direct smoke verification loaded DE440, resolved `EARTH`, executed
  `spkez_batch`, and confirmed Python `spicekit` was not imported.
- Targeted SPICE validation passed after rebuilding the extension:
  `19 passed` across `test_spice_backend.py`, `test_spice.py`, and
  `test_spice_kernel.py`.
- Full validation passed after rebuilding the editable extension:
  `cargo check --workspace`, `cargo fmt --all --check`,
  `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test --workspace`, Python ruff/black/py_compile checks on the
  touched Python files, and `pdm run test-rust-full`
  (`708 passed, 144 skipped, 2 deselected`).
- Baseline-main parity/performance validation passed against the
  `.legacy-venv` oracle installed from `/Users/aleck/Code/adam-core`:
  `migration.parity.parity_main --speed-n 2000` wrote
  `migration/artifacts/parity_gate.json` with all 22 wired APIs passing
  fuzz parity and warm speed passing after applying only the known
  temporary photometry warm-performance waiver.
- Full cold/warm performance artifact regenerated with
  `migration.parity.parity_speed --n 2000 --reps 7 --cold`; final
  `migration/artifacts/parity_speed_cold_warm.json` has
  `all_passed=true`, no unwaived failed speed rows, and the four expected
  photometry waiver rows. Canonical review tables were regenerated in
  `migration/artifacts/parity_report.md` and
  `migration/artifacts/parity_table_rca.json`.
- One cold/warm timing run observed a transient scheduler outlier on
  `orbit_determination.calcGauss` (`2.29x` p50 but `0.36x` p95 due a
  single ~20 ms Rust sample). An isolated 21-rep calcGauss run was stable
  at `2.42x` p50 / `2.40x` p95, and the repeated full cold/warm artifact
  passed at `2.36x` p50 / `2.40x` p95 / `28.60x` cold.

### RM-P0-002: Verify Authoritative Branch, Remotes, And Baseline Main

Status: complete; parity/performance gate green with temporary photometry warm-performance waiver (2026-04-28)

Reason: the transcript records a push to GitHub, but local migration checkout origin was observed pointing at the sibling `/Users/aleck/Code/adam-core` at review time. Baseline main integration is also stale around baseline commit `22a1efa3` or newer.

Scope:

- Verify remotes and branch tracking in both checkouts.
- Decide the authoritative branch: migration checkout branch, baseline checkout branch, or GitHub `rust-migration-waves-d-e`.
- Rebase or merge current baseline main into the migration branch.
- Record the exact baseline commit used by the parity oracle after updating.

Acceptance:

- `git remote -v`, `git branch -vv`, and `git status` are unambiguous in both checkouts.
- Migration branch contains current baseline docs, CI, metadata, and source changes unless explicitly waived.
- The parity harness points at the intended baseline checkout and commit.

2026-04-28 execution notes:

- Baseline checkout `/Users/aleck/Code/adam-core` was verified on `main` at `22a1efa3979cad5651e3f0765b2536983be6ab99`.
- Migration checkout `/Users/aleck/Code/adam-core-rust-migration` was verified on `rust-migration-waves-d-e`.
- Merged `origin/main` into the migration branch as commit `795dda758d495342e343ea828dc47924029361b8`.
- Resolved baseline docs/RTD conflicts by keeping the new generated `docs/source/reference/` tree and moving `rust_backend_contracts` under `docs/source/reference/index.rst`.
- Preserved Rust migration ignores and, at that point, the public
  `spicekit>=0.1.0` Python dependency introduced during the packaging
  carve-out. RM-P0-001 later removed that Python runtime dependency in
  favor of the direct Cargo `spicekit` dependency in `adam_core_rs_spice`.
- Validation repaired unrelated/pre-existing quality issues exposed by the required gate: Rust files needed `cargo fmt`, several Wave E2 Rust files failed `clippy -D warnings`, and the inherited Python 3.13 fixture test collection path imported `mpcq` despite the dependency being marker-gated to `<3.13`.
- Green checks after repairs: `pdm run rust-quality`; `pdm run test-rust-full` with `ADAM_CORE_REQUIRE_RUST_BACKEND=1` (`708 passed, 144 skipped, 2 deselected`).
- Parity fuzz gate passed for all 22 wired APIs.
- Required performance gate was red because warm photometry timings missed the 1.2x p50/p95 policy threshold. This is the known unresolved photometry warm-policy issue, not a baseline merge parity failure. The user approved a temporary waiver on 2026-04-28; enforcement is tracked as `waiver-20260428-photometry-warm-performance-temporary`. Artifacts: `migration/artifacts/parity_gate.json` and `migration/artifacts/parity_speed_cold_warm.json`.
- After waiver enforcement was wired into `migration.parity.parity_speed`, `migration.parity.parity_main` exits successfully with all 22 APIs passing parity and any remaining photometry warm misses marked as waived while preserving `raw_passed=false` in the JSON artifact.

### RM-P0-003: Repair Stale PDM Scripts And CI Workflow References

Status: complete; stale PDM/CI references repaired and validated (2026-04-28)

Reason: `pyproject.toml` and `.github/workflows/pip-build-lint-test-coverage.yml` still reference missing or stale Rust migration scripts/tests.

Known stale or suspect targets:

- `src/adam_core/tests/test_rust_orbit_determination.py`
- `src/adam_core/tests/test_rust_parity_randomized.py`
- `src/adam_core/dynamics/tests/test_kepler.py`
- `migration/scripts/rust_orbit_determination_benchmark.py`
- `migration/scripts/rust_backend_benchmark_gate.py` when invoked with old live-legacy flags
- `migration/artifacts/rust_benchmark_gate.json`

Scope:

- Replace stale PDM script targets with current `migration/parity/parity_main.py`, `migration/parity/parity_speed.py`, and current smoke tests.
- Update workflow steps and uploaded artifact names/paths.
- Add a small preflight that fails if a PDM script references a missing file.

Acceptance:

- Every PDM script referenced by CI exists and runs from a clean checkout.
- CI uses the current parity/speed/Rust-latency governance path, not deleted randomized test files or deleted OD benchmark scripts.
- `pdm run dev-loop` no longer calls missing paths.

2026-04-28 execution notes:

- Added `migration/scripts/check_pdm_ci_scripts.py` and PDM script
  `script-preflight`. The preflight parses `[tool.pdm.scripts]`, checks
  GitHub workflow `pdm run ...` references against defined PDM scripts,
  rejects known stale deleted-test/deleted-script/old-live-legacy-gate
  strings, and checks referenced repo paths exist while ignoring known
  output directories.
- Replaced stale `rust-smoke` targets with existing current smoke tests:
  `test_rust_backends.py`, SPICE backend/API/kernel tests,
  `test_rust_covariance_autodiff.py`, and OD `test_iod.py`, with
  `ADAM_CORE_REQUIRE_RUST_BACKEND=1`.
- Replaced deleted `rust-parity-randomized` and `rust-od-benchmark`
  script usage with current parity scripts:
  `rust-parity-fuzz`, `rust-parity-main`, `rust-parity-speed`, and
  `rust-parity-speed-cold`.
- Repointed `rust-perf-gate` to the current Rust-only latency regression
  gate via `rust-latency-gate`; removed old `--max-rust-over-legacy`
  usage and the obsolete `migration/artifacts/rust_benchmark_gate.json`
  artifact path from CI.
- Updated `.github/workflows/pip-build-lint-test-coverage.yml` to run
  `pdm run script-preflight`, stop calling deleted randomized parity
  scripts, run `pdm run rust-latency-gate`, and upload
  `migration/artifacts/rust_latency_current.json` as
  `rust-latency-current`.
- Validation passed: `pdm run script-preflight`; ruff/black/py_compile
  on the preflight script; `pdm run rust-smoke` (`72 passed`);
  `pdm run rust-latency-gate`; compatibility alias
  `pdm run rust-perf-gate`; `pdm run rust-quality`;
  `pdm run test-rust-full` (`708 passed, 144 skipped, 2 deselected`);
  baseline-main
  `pdm run rust-parity-main`; cold/warm
  `pdm run rust-parity-speed-cold`; regenerated canonical
  `migration/artifacts/parity_report.md` and
  `migration/artifacts/parity_table_rca.json`; `git diff --check`.

### RM-P0-004: Clarify Packaging, Clean Install, And Wheel/Publish Path

Status: completed 2026-04-28

Reason: the branch uses maturin, Rust is mandatory, and the publish workflow must build the correct native-extension wheel. RM-P0-001 removed the Python `spicekit` runtime dependency from adam-core by routing adam-core's SPICE backend through native Rust-to-Rust `spicekit`, so this task focuses on the adam-core wheel itself.

Scope:

- Define one authoritative wheel build path for adam-core.
- Align `pdm build`, `maturin build`, `rust-build`, and `.github/workflows/publish.yml`.
- Validate that built wheels contain and import `adam_core._rust_native`.
- Validate clean install in a fresh environment with no editable sibling packages.
- Reconcile `uv.lock`, PDM lock behavior, and the existing `adam-assist` self-dependency lock issue.

Acceptance:

- A fresh environment can install the built wheel and import `adam_core`.
- `adam_core._rust_native` imports from the wheel.
- A minimal SPICE backend operation works from the installed wheel.
- Publish workflow uploads only artifacts known to contain the native extension.

2026-04-28 execution notes:

- Defined `pdm run wheel-build` as the authoritative wheel path:
  generate `src/adam_core/_version.py` from
  `rust/adam_core_py/Cargo.toml`, then run
  `pdm build --no-sdist --dest dist`.
- Added `migration/scripts/write_maturin_version.py` because maturin uses
  the Cargo package version when `[project].version` is dynamic. Set the
  PyO3 package version to `0.5.6`, matching the latest repo tag available
  in this checkout.
- Added `migration/scripts/check_wheel_artifacts.py` and
  `pdm run wheel-inspect`. The check fails on non-wheel publish artifacts,
  verifies the wheel name/version metadata, requires
  `adam_core/_version.py`, requires runtime version to match wheel
  metadata, and requires an `adam_core._rust_native` extension file.
- Added explicit maturin wheel inclusion for `adam_core/_version.py`
  because the generated file is gitignored and was otherwise missing from
  the native wheel.
- Rewired `rust-build` to call `wheel-build`; rewired publish, CI, and
  tier-1 dependent smoke workflows to build `dist/*.whl`, inspect the wheel,
  and upload/install only the inspected wheel artifact.
- Repaired `rust-develop` by removing `maturin develop --uv` and ensuring
  pip is available before `maturin develop`. In this local environment,
  `maturin develop --uv` and direct `uv pip install` of the local wheel
  selected or retained PyPI `adam_core 0.5.5`, which lacks
  `_rust_native`. Standard pip install of the built wheel installed the
  correct local native wheel.
- Added a narrow uv override/source for `adam-core` so the lock can resolve
  adam-assist against this workspace instead of the published adam-core
  version range embedded in adam-assist metadata. This is not a full PDM to
  uv migration; local uv remains unsafe as the authoritative install path
  until its group-sync/local-wheel behavior is validated on the intended
  current release.
- Validation passed: `pdm run wheel-build`; `pdm run wheel-inspect`;
  clean temporary `python3.13 -m venv` plus `python -m pip install
  dist/adam_core-0.5.6-...whl`; import of installed
  `adam_core._rust_native`; minimal native SPICE backend `furnsh` +
  `spkez_batch`; verified Python `spicekit` was not imported during that
  smoke; `pdm run script-preflight`; ruff/black/py_compile on packaging
  scripts; TOML parsing for `pyproject.toml`, `uv.lock`, and
  `rust/adam_core_py/Cargo.toml`; `uv lock --check`; `cargo check
  --workspace`; `pdm run rust-build`; `pdm run rust-quality`;
  `pdm run test-rust-full` (`708 passed, 144 skipped, 2 deselected`);
  baseline-main `pdm run rust-parity-main`; cold/warm
  `pdm run rust-parity-speed-cold`; regenerated
  `migration/artifacts/parity_report.md` and
  `migration/artifacts/parity_table_rca.json`; `git diff --check`.

### RM-P0-004F: Packaging Follow-Up Cleanup

Status: completed 2026-04-28

Reason: RM-P0-004 review found three non-blocking but real handoff risks:
version-source confusion, uv local-install ambiguity, and unexplained
full-suite skip-count drift relative to earlier sessions.

Scope:

- Consolidate version-source documentation so future releases do not confuse
  PDM SCM metadata with maturin's Cargo-driven wheel metadata.
- Document the current uv posture: lock resolution is expected to pass, but
  PDM plus standard pip remain the authoritative build/install path until uv
  local-wheel behavior is validated on the intended current uv release.
- Confirm `dev-loop` no longer references retired parity scripts.
- Explain the current `708 passed, 144 skipped, 2 deselected` validation
  shape so it is not mistaken for an RM-P0-004 coverage regression.

Acceptance:

- `pyproject.toml` has one declared wheel version source.
- Release-tag/version mismatches fail loudly.
- Packaging handoff notes state the supported build/install commands and uv
  caveat.
- `pdm run script-preflight` remains green.

2026-04-28 execution notes:

- Removed the unused `[tool.pdm.version] source = "scm"` block from
  `pyproject.toml`. The native wheel version source is now explicitly
  `rust/adam_core_py/Cargo.toml` `[package].version`, which is the value
  maturin uses for wheel metadata.
- Extended `migration/scripts/write_maturin_version.py` to validate exact
  `vX.Y.Z` release tags against the Cargo package version before writing
  `src/adam_core/_version.py`.
- Added `migration/packaging.md` with the supported PDM/maturin wheel path,
  version-source contract, uv caveat, and current skip-count interpretation.
- Confirmed current `dev-loop` uses live scripts (`rust-parity-main`,
  `rust-parity-speed-cold`, and `rust-latency-gate`) and
  `pdm run script-preflight` catches missing `pdm run ...` references.
- `pdm run test-rust-full -- -rs` confirmed the current skip shape:
  139 benchmark skips from `--benchmark-skip`, two explicit Lambert skips,
  and three optional PYOORB skips; the two deselections are profile-marked
  tests excluded by `-m 'not profile'`.

### RM-P0-005: Restore Public Python Module Compatibility Or Document Breaking Changes

Status: completed 2026-04-28

Reason: migration deleted importable modules that exist on baseline main, which violates the stable Python package surface decision unless explicitly approved as breaking changes.

Baseline module paths needing compatibility review:

- `adam_core.dynamics.aberrations`
- `adam_core.dynamics.barker`
- `adam_core.dynamics.chi`
- `adam_core.dynamics.kepler`
- `adam_core.dynamics.lagrange`
- `adam_core.dynamics.stumpff`
- `adam_core.coordinates.jacobian`

Scope:

- Restore deleted modules as compatibility shims where possible.
- Re-export Rust-backed or replacement implementations from those module paths.
- Add import-compatibility tests comparing baseline-known public module paths to migration module paths.
- If removal is intentional, record an explicit breaking-change decision and release-note entry.

Acceptance:

- Importing baseline public module paths does not fail unless there is an explicit approved breaking-change record.
- Compatibility tests are part of CI.

2026-04-28 execution notes:

- Restored importable baseline module paths in the migration checkout:
  `adam_core.dynamics.aberrations`, `adam_core.dynamics.barker`,
  `adam_core.dynamics.chi`, `adam_core.dynamics.kepler`,
  `adam_core.dynamics.lagrange`, `adam_core.dynamics.stumpff`, and
  `adam_core.coordinates.jacobian`.
- `dynamics.aberrations` is a compatibility shim over the mandatory Rust
  light-time kernel for `_add_light_time`, `_add_light_time_vmap`, and
  `add_light_time`.
- Restored scalar/universal-variable helper modules from baseline for public
  import compatibility. These are not wired into the current production
  propagation/ephemeris paths, which remain Rust-backed. RM-P0-005G later
  converted the supported restored helper implementations themselves to thin
  Rust-backed wrappers.
- The focused calc-mean-motion benchmark now uses the separate baseline-main
  oracle instead of importing an in-checkout JAX reference alias.
- Restored `coordinates.jacobian.calc_jacobian` for compatibility and tests;
  production covariance transforms still use the Rust forward-mode AD path.
- Added `src/adam_core/tests/test_public_module_compatibility.py` to guard
  the restored module imports and smoke-test representative helpers.
- Validation passed: `pdm run pytest -q
  src/adam_core/tests/test_public_module_compatibility.py` (`11 passed`);
  targeted compatibility/nearby surface suite
  (`test_public_module_compatibility.py`, coordinate Keplerian/Cometary
  tests, dynamics propagation/ephemeris tests) passed (`91 passed`) when
  run outside the sandbox because local Ray tests need macOS process-list
  access; ruff/black/py_compile passed for the restored modules and test.
- Full rust-required suite after the compatibility restoration and coordinate
  kernel stabilization passed: `719 passed, 144 skipped, 2 deselected,
  56 warnings`.
- Baseline-main parity/performance cadence passed after this task:
  `pdm run rust-parity-main` passed all 22 fuzzed APIs and warm speed with
  only the known photometry waivers displayed; `pdm run
  rust-parity-speed-cold` passed with `all_passed=true`; canonical parity
  tables were regenerated in `migration/artifacts/parity_report.md` and
  `migration/artifacts/parity_table_rca.json`.
- Validation exposed a repeated `coordinates.cartesian_to_spherical` warm
  p95 instability in the full cold/warm artifact. The Rust kernel was
  stabilized by chunking Rayon work at row-block granularity instead of
  dispatching one Rayon task per row for spherical conversions. The final
  cold/warm artifact raw-passed the coordinate row (`1.67x` p50, `1.57x`
  p95, `18.77x` cold), but a temporary waiver remains attached and
  RM-P1-014A tracks the permanent policy decision.
- Non-timing validation passed after the final artifacts: `pdm run
  script-preflight`; `pdm run rust-quality`; `git diff --check`.

### RM-P0-005F: Audit Retained Python Public Surfaces And Rust-Native Ownership

Status: completed 2026-04-29

Reason: RM-P0-005 restored baseline Python module paths to avoid an accidental
breaking change, but restoring an import path is not the same as deciding it
deserves indefinite Python-level API support. The long-term architecture is
`adam-core-rs` calling Rust helpers directly; Rust code must not route back
through Python compatibility shims for orbital math helpers.

Initial callsite audit:

- Clear current multi-caller library families: `add_light_time` and
  `calc_mean_motion`. These already have Rust kernels used by production
  library paths.
- Internal restored-helper chains: `solve_barker`, `calc_chi`, and
  `calc_stumpff` are used inside restored Kepler/Lagrange/chi helper modules
  but are not broad production entrypoints.
- Mostly downstream/public compatibility surfaces today:
  `calc_period`, `calc_periapsis_distance`, `calc_apoapsis_distance`,
  `calc_semi_major_axis`, `calc_semi_latus_rectum`, `calc_mean_anomaly`,
  `solve_kepler`, `calc_lagrange_coefficients`,
  `apply_lagrange_coefficients`, `add_stellar_aberration`, and
  `calc_jacobian`.

Scope:

- Produce an explicit inventory for every restored symbol under
  `adam_core.dynamics.{aberrations,barker,chi,kepler,lagrange,stumpff}` and
  `adam_core.coordinates.jacobian`.
- Classify each symbol as one of: keep as supported Python public API, deprecate
  and remove in a planned breaking release, or retain only as a test/parity
  reference.
- For every retained Python public API, decide whether it should be a thin
  Rust-backed wrapper, a documented pure-Python/NumPy compatibility helper, or
  a JAX-only reference excluded from production use.
- Add Rust-native equivalents or expose existing Rust helpers for retained
  orbital math surfaces where Rust callers would otherwise need to re-enter
  Python.
- Verify that `adam-core-rs` and Rust kernels call Rust functions directly for
  Kepler/Stumpff/chi/lagrange/light-time math and never depend on Python module
  compatibility shims.
- Update docs/release notes with the support/deprecation decision for each
  restored surface.

Acceptance:

- A maintained/deprecated/reference-only decision is recorded for every restored
  symbol.
- Rust-to-Rust call paths exist for any retained helper needed by Rust
  algorithms.
- No Rust implementation depends on Python compatibility modules for orbital
  helper math.
- Public import compatibility tests remain green for supported/deprecated
  surfaces, and any planned removals have explicit release-note coverage.

2026-04-29 execution notes:

- Added `docs/source/reference/rust_public_compatibility.rst` and linked it
  from the reference toctree. The page records every restored symbol, its
  classification, its Python implementation posture, the Rust-native path
  where one exists, and release-note language for the deprecated/private
  light-time shims and reference-only `coordinates.jacobian.calc_jacobian`.
- Classification outcome: `_add_light_time` and `_add_light_time_vmap` are
  deprecated/private compatibility shims; `coordinates.jacobian.calc_jacobian`
  is reference-only; all other restored helper symbols remain supported Python
  APIs or supported diagnostics, with the caveat that production paths must use
  Rust-native entrypoints directly.
- Exposed Rust-to-Rust helper paths for retained orbital/helper surfaces:
  `calc_period`, `calc_periapsis_distance`, `calc_apoapsis_distance`,
  `calc_semi_major_axis`, `calc_semi_latus_rectum`, `calc_mean_motion`,
  `calc_mean_anomaly`, `solve_barker`, `solve_kepler_true_anomaly`,
  `apply_lagrange_coefficients`, and `apply_stellar_aberration_row`.
  Existing public Rust paths already covered `calc_stumpff`, `calc_chi`,
  `calc_chi_with_init`, `calc_lagrange_coefficients`, `add_light_time_row`,
  and `add_light_time_batch_flat`.
- Extended `test_public_module_compatibility.py` to assert that every restored
  symbol is documented and to statically reject production imports of the
  restored compatibility modules outside the modules themselves and tests. This
  preserves import compatibility while preventing future Python production code
  from re-routing Rust-owned orbital helper math through compatibility shims.
- Validation passed: targeted compatibility test (`13 passed`);
  ruff/black for the updated test; `cargo test -p adam_core_rs_coords`
  (`57 passed`); `pdm run rust-develop`; `pdm run script-preflight`;
  `pdm run rust-quality`; full rust-required pytest (`721 passed,
  144 skipped, 2 deselected, 56 warnings`); baseline-main `pdm run
  rust-parity-main` (22/22 fuzzed APIs and warm speed pass); cold/warm
  `pdm run rust-parity-speed-cold` (`all_passed=true`); canonical parity
  tables regenerated in `migration/artifacts/parity_report.md` and
  `migration/artifacts/parity_table_rca.json`; `git diff --check`.

### RM-P0-005G: Make Supported Python Compatibility Helpers Rust-Backed

Status: complete (2026-04-29)

Reason: RM-P0-005F exposed Rust-native helper functions for the retained
orbital helper surfaces, but several supported Python wrappers still execute
local JAX/NumPy compatibility implementations. If a symbol is a supported
Python public API in this migration branch, it should be a thin wrapper over
the Rust implementation unless there is an explicit composability reason to
keep a separate Python/JAX implementation.

Scope:

- Add PyO3/native bindings and `adam_core._rust.api` wrappers for supported
  helper surfaces that currently remain Python/JAX implementations:
  `calc_stumpff`, `calc_chi`, `calc_lagrange_coefficients`,
  `apply_lagrange_coefficients`, `solve_barker`, `solve_kepler`,
  Kepler scalar helpers, and `add_stellar_aberration`.
- Prefer batched/vectorized NumPy-boundary wrappers where callers may pass
  arrays. Avoid replacing vectorized Python/JAX helpers with per-element
  Python loops over scalar Rust calls.
- Keep JAX compatibility only under a clearly private name if a current
  migration script or test still needs in-checkout JAX composability. Otherwise
  rely on the separate baseline-main checkout for legacy parity references.
- Update `docs/source/reference/rust_public_compatibility.rst` so the Python
  implementation column no longer claims these supported APIs are
  JAX-compatible local implementations once they delegate to Rust.
- Run the full baseline-main parity/performance cadence after changing wrappers.

Acceptance:

- Every supported restored Python API either delegates to Rust or has an
  explicit documented exception.
- No supported restored helper performs duplicate JAX/NumPy orbital math in
  production package code without a concrete local reason.
- Public compatibility tests still pass and cover representative scalar and
  batched inputs where applicable.

Completion notes:

- Added PyO3 and `adam_core._rust.api` wrappers for Kepler scalar helpers,
  `solve_barker`, `solve_kepler`, `calc_stumpff`, `calc_chi`,
  `calc_lagrange_coefficients`, `apply_lagrange_coefficients`, and
  `add_stellar_aberration`.
- Rewrote the supported restored Python modules to delegate to Rust and fail
  loudly if `_rust_native` is unavailable. The reference-only
  `coordinates.jacobian.calc_jacobian` remains a separate RM-P0-005H concern.
- Updated the focused `calc_mean_motion` benchmark script to use the
  baseline-main oracle instead of importing an in-checkout JAX alias.
- Added guardrails that reject JAX imports in supported restored helper
  modules, while preserving the documented reference-only jacobian exception.
- Validation: targeted public compatibility tests `15 passed`; `cargo check -p
  adam_core_py`; `cargo test -p adam_core_rs_coords` `57 passed`;
  `pdm run rust-quality`; full rust-required suite `723 passed / 144 skipped /
  2 deselected / 56 warnings`; `pdm run rust-parity-main` passed 22/22 measured
  APIs; `pdm run rust-parity-speed-cold` passed with the existing temporary
  photometry waiver; parity report artifacts regenerated.

2026-04-29 reviewer follow-up:

- Aligned the supported `calc_chi` and `calc_lagrange_coefficients`
  compatibility defaults to `tol=1e-15`, matching the production propagation
  default and avoiding unnecessarily strict default iteration.
- Removed stale JAX cache-clearing from the ephemeris profile test rather than
  treating it as a local parity oracle.
- Added smoke coverage for Kepler infinity contracts (`calc_period(a < 0)` and
  `calc_apoapsis_distance(e >= 1)`).
- Documented that the `calc_chi` compatibility wrapper is not the warm-started
  single-orbit/many-dt hot path; callers should use the production propagation
  path for that workload.
- Clarified that broadcasted inputs are intentionally copied to contiguous
  float64 arrays at the PyO3/FFI boundary.

### RM-P0-005H: Remove Deprecated Private Shims And In-Repo Reference-Only JAX Helpers

Status: complete (2026-04-29)

Reason: The migration checkout now uses a separate baseline-main checkout and
`.legacy-venv` for parity and benchmark references. Keeping duplicate
reference-only JAX implementations inside the migrated package increases
surface area and contradicts the mandatory-Rust direction unless there is a
specific local diagnostic or downstream-compatibility need.

Scope:

- Audit `_add_light_time`, `_add_light_time_vmap`, and
  `coordinates.jacobian.calc_jacobian` for in-repo usage and plausible
  downstream utility.
- Remove private/deprecated shims when they are not useful standalone utilities
  and not required for known consumers. If removal is a breaking change for a
  baseline-imported symbol, document it explicitly in release notes.
- Remove or relocate any JAX-only reference helpers from package modules when
  the baseline-main oracle already supplies the legacy implementation for
  parity/benchmark comparisons.
- Adjust public compatibility tests from "must import" to "documented removal"
  for any symbol intentionally removed.

Acceptance:

- No in-package JAX reference implementation remains solely for parity with
  baseline main.
- Any removed private/reference-only symbol has explicit release-note coverage.
- Existing production and parity/performance gates remain green.

Completion notes:

- Audited `_add_light_time`, `_add_light_time_vmap`, and
  `coordinates.jacobian.calc_jacobian`. No production in-repo callers were
  found. The only remaining `_add_light_time_vmap` reference is the legacy
  parity runner importing from the separate baseline-main checkout, so it is not
  a migration-package dependency.
- Removed the private light-time shims from
  `adam_core.dynamics.aberrations`; retained the supported Rust-backed
  `add_light_time` and `add_stellar_aberration` APIs.
- Deleted `src/adam_core/coordinates/jacobian.py`, removing the in-package JAX
  reference-only jacfwd/vmap helper. Legacy JAX behavior remains available via
  the baseline-main oracle checkout used by parity harnesses.
- Updated `docs/source/reference/rust_public_compatibility.rst` release-note
  coverage and inventory so removed private/reference-only symbols are
  documented as absent with replacements/rationale.
- Updated public compatibility tests to assert supported symbols remain
  importable while removed private/reference-only symbols are documented and
  absent.

### RM-P0-006: Enforce One Runtime Contract For Rust Backend Availability

Status: complete (2026-04-29)

Reason: decisions say there is no rustless adam-core environment, but `_rust/api.py` still catches backend import exceptions, wrappers return `None`, and production code uses `assert out is not None`.

Scope:

- Choose and enforce one contract: Rust mandatory at import time, or optional backend with explicit exceptions. Current decisions prefer mandatory Rust.
- Remove broad exception swallowing around `_rust_native` import unless it is test-only and explicit.
- Replace production `assert out is not None` with explicit runtime errors or make wrappers raise directly.
- Make backend availability checks CI-default, not only manually enabled through `ADAM_CORE_REQUIRE_RUST_BACKEND`.
- Audit production asserts in coordinates, dynamics, photometry, OD, missions, propagator, variants, classification, MOID, Lambert, porkchop, and residuals.

Acceptance:

- `python -O` cannot change production behavior.
- Missing or broken Rust extension fails loudly with a specific error.
- No production path converts missing Rust into late `None` behavior.

Completion notes:

- `_rust/api.py` now imports `adam_core._rust_native` eagerly and raises a
  specific `ImportError` if the native extension is missing.
- `_rust/api.py` validates all required native symbols at import time and raises
  a specific `ImportError` listing missing symbols if the installed extension is
  stale or incomplete.
- Native wrapper functions no longer return `None` for backend unavailability.
  They return concrete native results or propagate native exceptions. The
  remaining `None` returns in NAIF name helpers represent unresolved NAIF names,
  not backend absence.
- Removed production `assert rust_* is not None` checks that could disappear
  under `python -O`; wrapper-level import/symbol validation now owns backend
  availability.
- Removed `ADAM_CORE_REQUIRE_RUST_BACKEND=1` from the default PDM Rust smoke and
  full-suite scripts. The compiled extension is now required by import behavior
  rather than an opt-in environment flag.
- Updated `docs/source/reference/rust_backend_contracts.rst` and current
  migration notes to document the mandatory Rust backend contract.

### RM-P0-007: Retire Contaminated Live-Legacy Benchmark Governance From Active CI

Status: complete (2026-04-29)

Reason: the journal correctly records that some live legacy paths became contaminated by Rust-backed fallthroughs. Active script metadata still points at stale live-legacy concepts.

Scope:

- Decide the active performance gate: current parity speed harness against baseline subprocess for still-fair APIs, and Rust-only latency regression for post-legacy APIs.
- Remove or rewrite old `--max-rust-over-legacy` uses.
- Ensure uploaded CI artifacts match current files.
- Preserve frozen history in `migration/artifacts/history/`.

Acceptance:

- CI cannot claim a Rust-vs-legacy speedup from a path that calls Rust on both sides.
- Current artifacts are named and uploaded consistently.
- Governance docs explain when to use parity speed vs Rust-latency baseline.

Completion notes:

- Added `migration/benchmark_governance.md` and
  `docs/source/reference/rust_benchmark_governance.rst` to document the active
  split: baseline-main parity/speed for fair APIs wired through
  `migration/parity/`, and Rust-only latency regression for post-legacy APIs.
- Updated `migration/parity/README.md`, `migration/parity/__init__.py`,
  `parity_main.py`, and `parity_speed.py` wording so the active speed gate is
  described as Rust vs baseline-main subprocess timing, not current-branch
  live-legacy fallback timing.
- Deleted the broken active one-off
  `migration/scripts/ephemeris_wide_observer_bench.py`, whose legacy JAX helper
  imports no longer exist after the Rust-only latency gate rewrite.
- Preserved dated one-off historical outputs under
  `migration/artifacts/history/`:
  `ephemeris_wide_observer_bench_2026-04-22.json` and
  `rust_orbit_determination_benchmark_2026-04-22.json`.
- Strengthened `migration/scripts/check_pdm_ci_scripts.py` so preflight rejects
  stale live-legacy artifacts in active paths and enforces that
  `rust-latency-gate` writes `migration/artifacts/rust_latency_current.json`
  while CI uploads it as `rust-latency-current`.
- Validation passed: targeted compile/ruff/black; `pdm run script-preflight`;
  `pdm run rust-latency-gate` after one transient microbenchmark p95 outlier
  rerun; `pdm run rust-quality`; escalated `pdm run test-rust-full`
  (`723 passed, 144 skipped, 2 deselected`); `pdm run rust-parity-main`;
  `pdm run rust-parity-speed-cold`; regenerated canonical parity/performance
  tables; `git diff --check`.
  A non-escalated `pdm run test-rust-full` failed only from sandbox restrictions
  (`psutil`/Ray macOS `sysctl` denial and DNS/network denial); the escalated
  rerun passed.
  `pdm run docs-check` could not run because the active environment lacks
  Sphinx and `pdm install -G docs` wanted to refresh a stale lockfile; no
  dependency files were changed.

## P1 Stabilization Tasks

Recommended current order after RM-P1-009 completion:

1. RM-P1-010, including lockfile/docs-dependency cleanup and a passing
   `pdm run docs-check`.
2. RM-P1-011.
3. RM-P1-012.
4. RM-P1-014 and RM-P1-014A before the 2026-05-12 waiver review date.
5. RM-P1-013 / RM-WE2-001.
6. RM-P1-018.
7. RM-P1-016 and RM-P1-017, then Wave D3/E2/E3 implementation work.

Reviewer-feedback disposition:

- The reported `calc_chi_numpy` / `calc_lagrange_coefficients_numpy`
  `tol=1e-16` issue is stale. RM-P0-005G reviewer follow-up already aligned
  PyO3 signatures, `_rust.api`, and Python compatibility wrappers to
  `tol=1e-15`.
- The `calc_chi` compatibility wrapper still does not internally route
  single-orbit/many-dt batches to the warm-started arc kernel. This was
  deliberately handled as documentation, not a hidden routing optimization.
  It is unrelated to the photometry warm-performance waiver. Revisit only if
  direct downstream use of `adam_core.dynamics.chi.calc_chi` shows it is a hot
  loop.
- The RM-P0-007 latency-gate rerun is accepted as task validation, but the
  statistical policy for the Rust-only latency gate is now tracked explicitly
  in RM-P1-018.

### RM-P1-008: Make `status.py` A Trustworthy Registry

Status: complete (2026-04-29); RM-P1-015 folded into this task.

Scope:

- Extend status taxonomy beyond `legacy`, `dual`, and `rust-default`.
- Represent `rust-only`, `raw-kernel-only`, `public-rust-default`, and subcase exclusions.
- Encode broad API subcases, especially `coordinates.transform_coordinates`.
- Encode `gaussIOD` randomized-fuzz exclusion. This folds RM-P1-015 into
  RM-P1-008; do not do the same registry/exclusion work twice.
- Fail governance generation if a row claims dual support but the legacy implementation is gone.

Acceptance:

- A reviewer can tell which APIs are public-rust-default, raw-kernel-only, rust-only, dual, waived, or partially covered.

Completion notes:

- `src/adam_core/_rust/status.py` now uses an explicit registry taxonomy:
  `public-rust-default`, `raw-kernel-only`, `orchestration-rust-default`,
  `rust-only`, `dual`, and `legacy`, with typed boundary, default-backend, and
  parity-coverage metadata.
- Registry rows encode direct randomized-fuzz coverage, targeted-test-only
  coverage, orchestration-implied coverage, and randomized-fuzz exclusions.
- `orbit_determination.gaussIOD` is explicitly
  `random-fuzz-excluded`, with the root-subset divergence rationale recorded
  in both registry metadata and the canonical parity table output.
- `coordinates.transform_coordinates` is marked as partial randomized-fuzz
  coverage with supported and excluded subcases called out; RM-P1-009 later
  moved the randomized parity case to the public dispatcher boundary.
- Wave D3/E2/E3 helper surfaces such as `calculate_chi2`, residual helpers,
  MOID/porkchop helpers, Tisserand, absolute-magnitude fits, weighted stats,
  and transform/rotation raw kernels are represented as targeted-test coverage
  instead of being overclaimed as randomized-fuzz coverage.
- `dynamics.add_light_time` was added to the registry so the fuzzed API set and
  status registry agree.
- `validate_api_migrations()` now fails import/report generation on duplicate
  IDs, invalid enum values, dual rows without a current legacy implementation,
  Rust-default rows without a Rust module, randomized-fuzz exclusions without
  excluded subcases, and coverage states that require explanatory notes.
- `migration/scripts/parity_table.py`,
  `migration/scripts/rust_migration_state_report.py`, and
  `migration/scripts/rust_backend_benchmark_gate.py` now consume the registry
  metadata rather than inferring coverage from tolerance rationale text.
- Added `src/adam_core/tests/test_rust_migration_status.py` guardrails that
  validate registry invariants, fuzz-generator alignment, tolerance-manifest
  alignment, `gaussIOD` exclusion visibility, transform partial-coverage
  visibility, and latency-gate scope.

Validation:

- `pdm run script-preflight`: passed.
- `pdm run rust-latency-gate`: passed; compared 22 latency-gate APIs against
  the Rust-only latency baseline.
- `pdm run rust-quality`: passed.
- `pdm run test-rust-full`: passed with escalated permissions:
  `730 passed, 144 skipped, 2 deselected, 56 warnings`.
- A non-escalated full-suite run failed only from sandbox limits
  (`psutil`/Ray macOS `sysctl` denial and DNS/network denial). The first
  escalated rerun then exposed two deterministic-test hygiene failures in
  observer tests caused by selecting an arbitrary `OBSERVATORY_CODES` set
  member whose MPC parallax coefficients were `NaN`; the tests now choose a
  sorted valid Earth-based MPC code.
- `pdm run rust-parity-main`: passed; all 22 direct fuzz APIs passed
  randomized parity against the baseline-main oracle with only existing
  photometry warm waivers in speed output.
- `pdm run rust-parity-speed-cold`: passed with existing temporary photometry
  warm waivers only.
- Canonical tables regenerated:
  `migration/artifacts/parity_report.md` and
  `migration/artifacts/parity_table_rca.json`.

### RM-P1-009: Add Public Dispatch Parity For `coordinates.transform_coordinates`

Status: complete (2026-04-29)

Reason: raw Rust kernel parity does not prove the public dispatcher path, especially for Cartesian-to-Cartesian frame-only paths intentionally excluded from the raw Rust dispatcher.

Scope:

- Add public API parity tests for actual `transform_coordinates(...)` call shapes.
- Separate raw NumPy kernel tests from public quivr/coordinate-object dispatch tests.
- Document the cartesian-to-cartesian exclusion as intentional if retained.

Acceptance:

- Parity reports do not overstate public dispatch coverage.

Completion notes:

- The canonical `coordinates.transform_coordinates` baseline-main parity case
  now builds `CartesianCoordinates` quivr objects in both the migration and
  baseline-main subprocesses and calls public `transform_coordinates(...)`.
  It no longer compares the raw `transform_coordinates_numpy` kernel directly
  while labeling the row as the public API.
- The fuzzed public case is `CartesianCoordinates` ecliptic -> equatorial into
  `SphericalCoordinates`, which exercises the migration-side fused Rust
  frame+representation path and the baseline-main public dispatcher.
- The registry remains partial by design. It now lists the public dispatcher
  subcase as covered and calls out remaining exclusions:
  Cartesian-to-Cartesian frame-only fallthrough, ITRF93/time-varying rotations,
  origin translation/user-kernel SPICE coverage, and remaining non-Cartesian
  representation combinations.
- `migration/parity/tolerances.py` rationale now describes the public
  dispatcher case and the expected last-ulp rotation/transcendental drift.

Validation:

- Targeted diagnostic before broad validation:
  `pdm run python -m migration.parity.parity_main --apis coordinates.transform_coordinates --fuzz-seeds 4 --fuzz-n 64 --speed-n 500 --speed-reps 5 --speed-warmup 1`
  passed with `4/4` fuzz seeds, worst_abs `1.421e-14`, worst_rel `6.151e-15`,
  and speed `2.03x` p50 / `2.16x` p95.
- Full task validation passed: targeted ruff/black/status-registry tests;
  `pdm run script-preflight`; `pdm run rust-quality`; escalated
  `pdm run test-rust-full` (`730 passed, 144 skipped, 2 deselected,
  56 warnings`); `pdm run rust-parity-main` (all 22 direct fuzz APIs passed,
  public transform dispatch speed `2.92x` p50 / `3.01x` p95);
  `pdm run rust-parity-speed-cold` (public transform dispatch `3.24x` warm p50
  / `3.08x` warm p95 / `1.48x` cold; existing photometry warm waivers only);
  regenerated canonical parity/RCA tables; `git diff --check`.
- A non-escalated `pdm run test-rust-full` failed only from sandbox-denied
  Ray/psutil macOS `sysctl` process inspection and DNS/network access; the
  escalated rerun passed.

### RM-P1-010: Re-Home Rust Docs Into Current Baseline RTD Structure

Status: open

Scope:

- Preserve baseline docs extras and docs CI.
- Move Rust backend docs into the new baseline `docs/source/reference/` structure.
- Resolve lockfile/docs-dependency drift without smuggling unrelated
  dependency changes into another task.
- Ensure `pdm run docs-check` builds locally and in CI.

Acceptance:

- `pdm run docs-check` passes in a clean/current dev environment.
- Docs build under the baseline RTD structure.
- Rust migration docs are discoverable and not bolted onto stale docs layout.

### RM-P1-011: Audit Runtime Dependencies

Status: open

Scope:

- Audit production imports of `jax`, `jaxlib`, `numba`, `spiceypy`, and Python `spicekit`.
- Move parity/test-only dependencies into optional/test groups.
- Keep production dependencies aligned with mandatory Rust and direct Rust `spicekit` decisions.

Acceptance:

- Runtime dependencies are minimal and justified by production imports.

### RM-P1-012: Restore Or Replace Independent Propagation Oracle

Status: open

Reason: replacing `sp.prop2b` with forward/backward self-consistency is useful but not an independent oracle.

Scope:

- Add fixed trusted propagation vectors from an independent implementation, or restore a non-production oracle path in tests.
- Reconcile velocity tolerance mismatch: decision says 1 mm/s, test comment allowed 1 m/s at review time.

Acceptance:

- Propagation tests include at least one independent expected-output check that does not call the same Rust implementation under test.

### RM-P1-013: Document And Test `calculate_chi2` SPD Covariance Contract

Status: open

Reason: Rust Cholesky rejects non-SPD covariance matrices that baseline `np.linalg.inv` may have accepted if merely invertible. This is probably correct, but it is a public behavior change.

Scope:

- Document covariance matrices must be symmetric positive definite.
- Add tests for non-SPD behavior and diagnostic errors.
- Keep NaN policy explicit: NaN diagonal raises, NaN off-diagonal maps to zero to match legacy behavior.

Acceptance:

- Users get a clear error for non-SPD covariance input.
- The behavior change is documented and release-noted if needed.

### RM-P1-014: Resolve Photometry Warm Performance Gate Policy

Status: temporary waiver active; permanent resolution still open

Scope:

- Decide among SIMD/transcendental investment, cold-start waiver, or reverting/selective dispatch.
- If waiving, add explicit waivers with workload rationale and review date.
- If implementing SIMD, define crate/library choice and ULP validation plan.

Acceptance:

- Photometry Rust-default status is justified by measured warm gates or explicit waiver.

2026-04-28 note:

- Temporary waiver `waiver-20260428-photometry-warm-performance-temporary` is active for the four photometry APIs in the constitutional speed gate. This is not a permanent resolution; review by 2026-05-12.

### RM-P1-015: Make `gaussIOD` Randomized Parity Exclusion Visible

Status: complete (2026-04-29); folded into RM-P1-008

Scope:

- Add registry/waiver visibility for the known randomized root-subset mismatch.
- Separate fixed-fixture parity from randomized fuzz parity in reports.
- Keep runner adapters for fixed-fixture/manual parity.

Acceptance:

- Governance output cannot imply randomized `gaussIOD` parity is enforced when it is intentionally unwired.

### RM-P1-014A: Resolve Cartesian-To-Spherical Warm Performance Policy

Status: temporary waiver active; permanent resolution still open

Reason: RM-P0-005 validation exposed a repeated p95-only warm speed miss for
`coordinates.cartesian_to_spherical` in the full n=2000 cold/warm artifact.
Parity passes and cold-start speed is roughly 19x faster than the baseline JAX
path, but warm steady-state speed is near the 1.2x threshold on Apple Silicon.
The kernel does scalar `sqrt`/`atan2`/`asin` math, where XLA can use optimized
vectorized transcendentals. A chunked-Rayon fix reduced per-row dispatch
overhead but did not consistently clear the warm p95 gate.

Scope:

- Decide among SIMD/transcendental investment, cold-start waiver, or
  reverting/selective dispatch for this small warm workload.
- Keep raw p50/p95/cold timings visible in parity artifacts while the temporary
  waiver is active.
- If implementing SIMD, define crate/library choice and ULP validation plan.

Acceptance:

- `coordinates.cartesian_to_spherical` Rust-default status is justified by
  measured warm gates or explicit waiver.

2026-04-28 note:

- Temporary waiver
  `waiver-20260428-cartesian-to-spherical-warm-performance-temporary` is active;
  review by 2026-05-12.
- After chunking spherical conversion kernels, the latest full cold/warm
  artifact raw-passed this API at `1.67x` p50, `1.57x` p95, and `18.77x`
  cold. The waiver remains active because earlier full-artifact p95 misses
  were reproducible enough that this still needs a permanent policy or SIMD
  resolution rather than relying on one green run.

### RM-P1-016: Split Large PyO3 Binding File After Stabilization

Status: open

Scope:

- Split `rust/adam_core_py/src/lib.rs` into domain modules: coordinates, dynamics, photometry, orbit determination, SPICE.
- Keep shape validation at the PyO3 boundary.
- Avoid duplicating shape rules across Python and Rust.

Acceptance:

- Binding layer is maintainable and reviewable before the next large wave.

### RM-P1-017: Final Clean Validation Pass

Status: open

Scope:

- Run full Rust quality checks.
- Run full rust-required pytest.
- Run parity fuzz and speed gates.
- Run clean-install/wheel validation.
- Run docs build.
- Run public import compatibility tests.

Acceptance:

- Final review can start from green, current artifacts rather than journal claims.

### RM-P1-018: Harden Rust-Only Latency Gate Statistical Policy

Status: open

Reason: RM-P0-007 validation accepted `rust-latency-gate` after one rerun
cleared a transient p95 microbenchmark outlier. That is defensible for a
documentation/governance task, but it is not a durable statistical policy for
the active post-legacy performance regression signal.

Scope:

- Define the allowed rerun policy for `pdm run rust-latency-gate`.
- Increase repeats or add repeated-trial aggregation if needed for microsecond
  APIs.
- Preserve raw samples and failed attempts when a rerun changes the outcome.
- Document when a p95 miss is considered scheduler noise versus a real
  regression requiring action.

Acceptance:

- A reviewer can reproduce the latency-gate decision policy without relying on
  chat-time judgment.
- CI/local artifacts expose enough raw data to audit pass-after-rerun cases.

## Other Agent Backlog And Wave Status

### Completed Or Do Not Redo Without New Evidence

- Task #135: `.legacy-venv` baseline oracle setup.
- Task #136: `migration/parity/` constitutional harness.
- Task #137: cartesian-to-keplerian/cometary tolerance tightened by data.
- Task #138: hyperbolic universal-Kepler divergence fixed by reverting to Newton after Laguerre regression.
- Task #140: `propagate_2body_along_arc` consumed in `_run_2body_propagate` for OD inner-loop single-orbit/many-dt patterns.
- Wave E1: tisserand, classification, and absolute-magnitude fitting kernels completed.
- Wave E2 completed pieces: `calculate_chi2`, `bound_longitude_residuals`, `apply_cosine_latitude_correction`.
- Weighted mean/covariance: attempted and reverted to NumPy/BLAS in production because BLAS wins; Rust kernels remain but should not be re-promoted without new SIMD evidence.

### RM-WD3-001: Wave D3 Parallel Backend Abstraction

Status: open

Source task: #134 from transcript.

Reason: Ray has been removed from MOID and porkchop but remains in propagator/OD paths where n-body ASSIST or propagator-bound work dominates. The code needs an explicit abstraction for choosing rayon, Ray, or sequential execution rather than ad hoc per-module choices.

Known remaining Ray surfaces:

- `src/adam_core/propagator/propagator.py`
- `src/adam_core/orbit_determination/od.py`
- `src/adam_core/orbit_determination/iod.py`
- `src/adam_core/dynamics/impacts.py`
- `src/adam_core/dynamics/ephemeris.py`

Scope:

- Define a `parallel_backend` policy/abstraction.
- Make Ray usage explicit and centralized.
- Preserve behavior for ASSIST/n-body propagator-bound work.
- Remove module-local Ray scaffolding where rayon or sequential execution is now sufficient.

Verification:

- Compare wall-time and behavior against baseline main for representative OD/IOD/propagator workloads.
- Run full rust-required tests and targeted OD/IOD/propagator tests.

### RM-WE2-001: Wave E2 `calculate_chi2` Follow-Up

Status: partially complete

Completed:

- Rust Cholesky kernel is implemented and production-dispatched.
- Cargo, residuals, OD, and full pytest sweeps were recorded green.

Open:

- RM-P1-013 documentation/tests for SPD behavior.
- Include `calculate_chi2` in current parity/speed governance if not already represented after status registry cleanup.

### RM-WE2-002: Fuse `Residuals.calculate`

Status: open

Reason: small-N PyO3 overhead dominates because `Residuals.calculate` currently composes multiple Rust kernels and Python/quivr steps. OD inner loops care about N around 10-100, not only large-N throughput.

Scope:

- Design a fused Rust/PyO3 call that performs observed-predicted residuals, longitude wrapping, cosine-latitude correction, chi2, and any required probability fields in one crossing where practical.
- Keep quivr table assembly in Python only if it is not performance-critical or cannot be cleanly represented.
- Preserve current `compute_residuals_ndarray` fast path for OD callers that only need residual columns.

Verification:

- Parity against baseline `Residuals.calculate`.
- OD least-squares test coverage.
- Bench both small-N OD workloads and large-N batch workloads.

### RM-WE2-003: Variants And Covariance Sampling Linear Algebra

Status: open

Source task: #148 follow-up.

Scope:

- Decide and document the linalg crate strategy for Cholesky, symmetric eigen, and SVD needs. `faer 0.22` is already present but not production-critical.
- Port or refactor `sample_covariance_sigma_points`, `sample_covariance_random`, and `VariantOrbits.create` hot paths where data shows wins.
- Avoid re-promoting weighted mean/covariance unless a new SIMD strategy beats BLAS.

Verification:

- Parity for sigma-point and Monte Carlo sampling shape/statistical contracts.
- Performance at realistic variant counts.

### RM-WE2-004: OD Evaluation And Outlier Helpers

Status: open

Scope:

- Review `orbit_determination/evaluate.py` and `orbit_determination/outliers.py` for small pure-numeric kernels.
- Do not port orchestration blindly; focus on fused hot paths around residual/chi2/outlier scoring.

Verification:

- Parity against baseline evaluation outputs.
- Bench in realistic OD loops, not isolated toy calls only.

### RM-WE3-001: Least-Squares Inner-Loop Fusion

Status: open

Reason: next high-leverage performance lever after Wave E2 is fusing LSQ inner-loop linear algebra and residual scoring to avoid PyO3 and quivr overhead at small N.

Scope:

- Investigate ATWA/normal-equation solve path in `orbit_determination/least_squares.py`.
- Use `faer` or another approved linalg strategy for Cholesky/QR as appropriate.
- Preserve numerical diagnostics and convergence behavior.

Verification:

- Baseline parity on deterministic OD cases.
- Existing least-squares tests, including order-dependent flake awareness.
- Performance comparison on representative OD fits.

### RM-WE3-002: Quivr-Bound Constitutional Gaps

Status: open

Source task: #139.

Scope:

- Close or explicitly track remaining parity harness gaps that require Python+pyarrow/quivr round trips:
  - `calculate_perturber_moids`
  - `generate_porkchop_data`
  - hard `transform_coordinates` cases such as ITRF93/origin-translation public dispatch
  - `gaussIOD` fixed-fixture parity path versus randomized exclusion
- Extend harness style where NumPy-boundary subprocess handoff is insufficient.

Verification:

- Reports distinguish raw kernel coverage, public dispatch coverage, and orchestration coverage.

### RM-WE4-001: I/O And External Client Cold Paths

Status: deferred

Scope:

- Only optimize if profiling shows a bottleneck.
- Keep prior qv.concatenate cleanup pattern: accumulate lists and concatenate once.
- Avoid new entrypoints or scripts for cold convenience paths.

### RM-WE5-001: Schema-Level Work

Status: deferred

Scope:

- Low priority unless schema construction becomes a measured bottleneck.
- Do not port quivr schemas to Rust without a broader Arrow-native design.

### RM-WE6-001: Time/ERFA Exploration

Status: exploratory

Scope:

- Time handling is backed by ERFA/C; treat as a separate investigation, not a Rust migration default target.

### RM-FUTURE-001: N-Body Propagation Port

Status: future major project

Scope:

- Rust adaptive-step n-body integrator with covariance support.
- SPICE-backed perturber lookup.
- Integration with existing `Propagator` abstraction.
- This is a multi-day or multi-week project and should not be mixed into stabilization work.

## Final Agent Instruction

Before changing implementation code, read this file plus `decisions.md` and `journal.md`. If the next task touches performance, establish the baseline comparison method first. If the next task deletes or replaces legacy behavior, preserve the oracle or historical artifact first.
