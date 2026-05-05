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
  Rust boundary is fast for tiny public calls. Enforced at the standard
  1.2× p50/p95 threshold.
- `small-n`: the historical `n=2000` p50/p95 promotion gate. Enforced at the
  standard 1.2× p50/p95 threshold.
- `large-n`: API-shaped larger workloads. Structured workload metadata records
  axes such as rows, orbits × epochs, orbits × observers, and OD triplet counts
  instead of only a flat free-form size string. Enforced at the standard
  1.2× p50/p95 threshold; substantially larger speedups are preferred.

Per the 2026-05-04 user decisions captured in `decisions.md`, no large-n
performance waivers are acceptable. Every lane must meet ≥1.2× p50/p95.
Underlying performance misses must be optimized, not waived; transient timing
rerouting via short-lived waivers is only acceptable while an active
optimization is in flight.

All governance lanes default to single-process/single-thread warm timing. Native
or multithreaded scaling measurements are allowed only as separately labeled
artifacts and must not be used as default pass/fail evidence unless explicitly
approved.

Baseline-main legacy timings are serialized in
`migration/artifacts/parity_legacy_speed_baseline.json` and reused by the
canonical speed gates. The cache is keyed by API, lane, structured workload
shape, reps/warmup, seed, thread mode, baseline checkout identity, and benchmark
process version/source hash. Refresh it intentionally after adding benchmark
APIs, changing workload shapes, changing the timing process, or updating the
baseline checkout.

Artifacts:

- `migration/artifacts/parity_legacy_speed_baseline.json`
- `migration/artifacts/parity_gate.json`
- `migration/artifacts/parity_speed_cold_warm.json`
- `migration/artifacts/parity_report.md`
- `migration/artifacts/parity_table_rca.json`

### Rust-only latency regression

Use `pdm run rust-latency-gate` for post-legacy APIs and active CI
performance regression tracking. This gate measures current Rust latency only
and compares it with the committed Rust-only baseline. The default harness runs
three independent timing trials per API and compares the median of the per-trial
p50/p95 estimates, while preserving all raw samples in the current artifact.
This absorbs isolated scheduler outliers without requiring ad hoc manual reruns.
The default pass/fail mode is single-process/single-thread so the committed
baseline and current run are apples-to-apples; production/library Rust still
uses normal multi-thread Rayon / OpenMP / BLAS / JAX behavior outside the
benchmark harness. Multi-thread measurements (also written as `--threads
multi-thread`; `native` is accepted as a deprecated alias) are allowed only
as explicitly labeled diagnostics with separate
baseline/output paths. Use `--trials 1` only for explicitly labeled diagnostics.

Artifacts:

- `migration/artifacts/rust_latency_baseline.json`
- `migration/artifacts/rust_latency_current.json`

The CI artifact name for the current run is `rust-latency-current`, and its
path must stay `migration/artifacts/rust_latency_current.json`.

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
