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

Artifacts:

- `migration/artifacts/parity_gate.json`
- `migration/artifacts/parity_speed_cold_warm.json`
- `migration/artifacts/parity_report.md`
- `migration/artifacts/parity_table_rca.json`

### Rust-only latency regression

Use `pdm run rust-latency-gate` for post-legacy APIs and active CI
performance regression tracking. This gate measures current Rust latency only
and compares it with the committed Rust-only baseline.

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
