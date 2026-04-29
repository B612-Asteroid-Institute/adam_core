Rust Benchmark Governance
=========================

Active migration gates are separated from frozen historical
Rust-vs-legacy evidence. The rule is simple: do not claim a live
Rust-vs-legacy speedup from any path where the "legacy" side can call Rust.

Active Gates
------------

Baseline-main parity and speed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pdm run rust-parity-main`` and ``pdm run rust-parity-speed-cold`` compare the
current Rust path against the upstream ``main`` checkout installed in
``/Users/aleck/Code/adam-core/.legacy-venv``. This is valid for APIs wired into
``migration/parity/`` because the oracle runs in a separate Python subprocess
and does not import the migration checkout.

These gates write:

* ``migration/artifacts/parity_gate.json``
* ``migration/artifacts/parity_speed_cold_warm.json``
* ``migration/artifacts/parity_report.md``
* ``migration/artifacts/parity_table_rca.json``

Rust-only latency regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pdm run rust-latency-gate`` is the active CI performance regression gate for
post-legacy APIs. It measures current Rust latency only and compares it with
the committed Rust-only baseline:

* ``migration/artifacts/rust_latency_baseline.json``
* ``migration/artifacts/rust_latency_current.json``

CI uploads the current run as artifact ``rust-latency-current`` from path
``migration/artifacts/rust_latency_current.json``.

Retired Paths
-------------

The old live Rust-vs-legacy benchmark gate is retired. PDM scripts and
workflows must not use ``--max-rust-over-legacy`` or upload
``migration/artifacts/rust_benchmark_gate.json`` as a current artifact.

Historical Rust-vs-legacy evidence lives under ``migration/artifacts/history/``
and is not overwritten by active runs. The broad final snapshot is
``migration/artifacts/history/rust_vs_legacy_final_snapshot_2026-04-23.json``;
one-off historical artifacts in the same directory are dated and may support
specific promotion or waiver decisions.

New Or Changed APIs
-------------------

Before deleting or bypassing a legacy implementation, capture either a
baseline-main parity/speed artifact, fixed trusted vectors with documented
tolerances, or a dated historical benchmark artifact. After the legacy path is
gone or contaminated, track performance with Rust-only latency regression
rather than recreating a "legacy" column from the migrated package.
