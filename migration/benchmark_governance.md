# Rust Benchmark Governance

This document separates active migration gates from frozen historical
rust-vs-legacy evidence. The core rule is: do not claim a live
Rust-vs-legacy speedup from any path where the "legacy" side can call Rust.

## Active Gates

### Baseline-main parity and speed

Use `pdm run rust-parity-main` and `pdm run rust-parity-speed-cold` for APIs
that are wired into `migration/parity/` and still have a fair baseline-main
oracle in `/Users/aleck/Code/adam-core/.legacy-venv`.

These gates run the current Rust path in-process and the baseline-main
implementation in a separate Python subprocess. They are valid because the
legacy side is not imported from the migration checkout and cannot fall
through to current-branch Rust implementations.

Baseline speed artifacts now carry named size lanes:

- `tiny-n`: a quick one-off/small-call lane (`n≈10`) that answers whether the
  Rust boundary is fast for tiny public calls. p50 is enforced at the standard
  1.2× threshold; p95 is reported but not enforced because microsecond-scale
  p95 is scheduler-jitter dominated under the canonical multi-thread policy.
- `small-n`: the historical `n=2000` p50/p95 promotion gate. Enforced at the
  standard 1.2× p50/p95 threshold.
- `large-n`: API-shaped larger workloads. Structured workload metadata records
  axes such as rows, orbits × epochs, orbits × observers, and OD triplet counts
  instead of only a flat free-form size string. Enforced at the standard
  1.2× p50/p95 threshold; substantially larger speedups are preferred.

Per the 2026-05-04 user decisions captured in `decisions.md`, no large-n
performance waivers are acceptable. Small-n and large-n must meet ≥1.2×
p50/p95; tiny-n must meet ≥1.3× p50 while p95 remains diagnostic per the
2026-05-07 tiny-lane jitter decision. Underlying performance misses must be
optimized, not waived; transient timing rerouting via short-lived waivers is
only acceptable while an active optimization is in flight.

All Rust-vs-baseline governance lanes default to **multi-thread** warm timing:
Rust Rayon and the legacy NumPy/JAX/XLA/BLAS pools both run uncapped, giving a
production-realistic best-effort comparison. The single-thread Rust-vs-baseline
mode is retained only as a diagnostic because legacy JAX/XLA cannot be reliably
capped on macOS Apple Silicon. Rust-only latency regression detection remains
single-thread by default because it has no legacy comparison and capped Rayon
improves stability.

The parity speed gate also uses a built-in source-governed timing trial count
(`CANONICAL_SPEED_TRIALS`) for every API/lane. Each trial records the configured
reps/warmup distribution; pass/fail uses median-of-trial p50/p95 aggregates and
the JSON artifacts retain the raw sample matrix plus per-trial percentiles. The
trial count is intentionally not exposed as a CLI flag for canonical scripts.

Canonical warm performance lanes measure **non-cached semantic computation**.
Before every warmup and timed legacy/current-Python invocation, the harness
clears adam_core's observer-state, origin-translation, and SPKEZ result caches
outside the measured interval. Imports, loaded SPICE kernels/readers, JIT state,
and thread pools remain warm. This prevents repeated identical benchmark inputs
from becoming a one-sided memoization comparison. There is no separate
cache-hit performance lane; production caches remain enabled outside the
benchmark harness.

Artifacts distinguish **legacy adam_core**, **current through Python**, and
**native Rust**. Gate thresholds apply to legacy/current-through-Python because
that is the compatible user path. Native Rust is measured only by a Rust-owned
`std::time::Instant` loop around direct Rust calls; Python/PyO3 may launch the
loop but must be outside every sample. PyO3 crossing latency is not native-Rust
latency. Missing Rust-internal adapters remain null with a reason/TODO—never
replace them with crossing or Python timings.

Baseline-main legacy timings are serialized in
`migration/artifacts/parity_legacy_speed_baseline.json` and reused by the
canonical speed gates. The cache is keyed by API, lane, structured workload
shape, reps/warmup, built-in timing-trial count, seed, thread mode, baseline
checkout identity, and benchmark process version/source hash. Refresh it
intentionally after adding benchmark
APIs, changing workload shapes, changing the timing process, or updating the
baseline checkout.

Artifacts:

- `migration/artifacts/parity_legacy_speed_baseline.json`
- `migration/artifacts/parity_gate.json`
- `migration/artifacts/parity_speed_cold_warm.json`
- `migration/artifacts/parity_report.md`
- `migration/artifacts/parity_table_rca.json`

### Rust-only latency regression

Use `pdm run rust-latency-gate` for post-legacy APIs and local performance
regression tracking. GitHub Actions uses `pdm run rust-latency-gate-ci`, which
keeps the same measurement harness but compares against the Ubuntu runner
baseline because latency baselines are machine/OS specific. Both gates measure
current Rust latency only and compare it with a committed Rust-only baseline.
The default harness runs
a built-in source-governed number of independent timing trials per API and
compares the median of the per-trial p50/p95 estimates, while preserving all raw
samples in the current artifact. This absorbs isolated scheduler outliers
without requiring ad hoc manual reruns.
The default pass/fail mode is single-process/single-thread so the committed
baseline and current run are apples-to-apples; production/library Rust still
uses normal multi-thread Rayon / OpenMP / BLAS / JAX behavior outside the
benchmark harness. Multi-thread measurements (also written as `--threads
multi-thread`; `native` is accepted as a deprecated alias) are allowed only
as explicitly labeled diagnostics with separate baseline/output paths. Trial
count is not user-configurable; diagnostic changes to trial policy require a
source edit and a clearly labeled artifact.

Artifacts:

- `migration/artifacts/rust_latency_baseline.json` — local/default reference baseline.
- `migration/artifacts/rust_latency_baseline_github_ubuntu.json` — GitHub Actions Ubuntu reference baseline.
- `migration/artifacts/rust_latency_current.json`

The CI artifact name for the current run is `rust-latency-current`, and its
path must stay `migration/artifacts/rust_latency_current.json`; the upload step
runs with `always()` so failed latency gates still preserve the current-run
artifact for triage.

## Not Allowed In Active CI

- Do not use current-branch Python fallback paths as a "legacy" timing source.
- Do not reintroduce `--max-rust-over-legacy` into PDM scripts or workflows.
- Do not upload or cite `migration/artifacts/rust_benchmark_gate.json` as a
  current artifact. That live Rust-vs-legacy gate was retired.
- Do not keep one-off live-legacy benchmark scripts under `migration/scripts/`
  after their legacy helper imports are deleted. Preserve their JSON output
  under `migration/artifacts/history/` instead.

## Historical Artifacts

Historical rust-vs-legacy speedups are preserved under
`migration/artifacts/history/`. They are valid only as dated migration
evidence, not as current CI gates.

The authoritative historical snapshot for broad API speedups is
`migration/artifacts/history/rust_vs_legacy_final_snapshot_2026-04-23.json`,
with caveats documented in the sibling `README.md`.

Additional one-off artifacts in that directory may support a specific
promotion or waiver decision. They should keep dated filenames and must not be
overwritten by active runs.

## New Or Changed APIs

Before deleting or bypassing a legacy implementation, capture one of:

- a baseline-main parity/speed artifact through `migration/parity/`;
- fixed trusted vectors with documented tolerances; or
- a dated historical benchmark artifact under `migration/artifacts/history/`.

After the legacy path is gone or contaminated, use Rust-only latency tracking
for regression detection. Do not recreate a "legacy" column from the migrated
package.
