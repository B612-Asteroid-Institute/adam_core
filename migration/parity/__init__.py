"""Constitutional parity + speedup gate for the Rust migration.

This package owns the two non-negotiable constitutional guarantees:

1. **Bit/ULP-level parity** between every rust-default API and the
   upstream JAX/numba legacy implementation, sampled across a randomized
   input space (fuzz).
2. **Speedup floor**: every rust-default API (and the orchestration
   functions that depend on them) must measure ≥ 1.2× p50 and p95
   latency vs the legacy implementation on identical workloads.

The legacy oracle is the upstream-pinned ``adam_core`` install in the
sibling ``.legacy-venv`` (see migration/parity/_oracle.py). We invoke
the oracle via subprocess because both repositories export the same
``adam_core`` package name and cannot coexist in one venv.
"""
