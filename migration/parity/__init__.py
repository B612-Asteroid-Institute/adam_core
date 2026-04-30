"""Baseline-main parity + speedup gate for the Rust migration.

This package owns the two baseline-main guarantees for APIs wired into the
parity harness:

1. **Bit/ULP-level parity** between current Rust output and the upstream
   baseline-main implementation, sampled across a randomized input space.
2. **Speedup floor**: each measured API must be >= 1.2x p50 and p95 latency
   vs the baseline-main implementation unless an explicit waiver is attached.

The legacy oracle is the upstream-pinned ``adam_core`` install in the
sibling ``.legacy-venv`` (see migration/parity/_oracle.py). We invoke
the oracle via subprocess because both repositories export the same
``adam_core`` package name and cannot coexist in one venv.

Do not use this package to time current-branch Python fallbacks as "legacy".
For post-legacy performance regression tracking, use
``migration/scripts/rust_backend_benchmark_gate.py``.
"""
