"""Shared helpers for two-runtime adam_assist performance comparisons.

Pairs with :class:`migration.parity._assist_oracle.LegacyAssistPropagator`: the
legacy adam_assist call is timed *inside* the isolated ``.legacy-assist-venv``
(the proxy ``time_*`` methods), and the Rust ``adam_assist`` call is timed
locally with :func:`time_rust`. :func:`percentiles` computes matching p50/p95
for both sides so speedups are apples-to-apples.
"""

from __future__ import annotations

import gc
import time
from typing import Any, Callable

import numpy as np

TWO_RUNTIME_COMPARISON_MODE = "gpl_rust_assist_backend_vs_legacy_python_adam_assist"
PERFORMANCE_COLUMNS = {
    "legacy_adam_core": "legacy adam_assist over pinned legacy adam_core",
    "current_python": "current adam_assist public Python method",
    "native_rust": "direct Rust call timed inside Rust with std::time::Instant",
    "gate": "legacy/current_python; native_rust is diagnostic",
}
NATIVE_RUST_TODO = "personal-98v.1"


def percentiles(samples: list[float]) -> tuple[float, float]:
    """Return (p50, p95) seconds for a list of per-rep timings."""
    arr = np.asarray(samples, dtype=np.float64)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 95))


def performance_timing_payload(
    legacy_samples: list[float],
    current_python_samples: list[float],
    native_rust_samples: list[float],
    *,
    native_operation: str,
) -> dict[str, Any]:
    """Canonical three-column payload with Rust-owned native samples."""
    legacy_p50, legacy_p95 = percentiles(legacy_samples)
    current_p50, current_p95 = percentiles(current_python_samples)
    native_p50, native_p95 = percentiles(native_rust_samples)
    legacy = {"values": legacy_samples, "p50": legacy_p50, "p95": legacy_p95}
    current = {
        "values": current_python_samples,
        "p50": current_p50,
        "p95": current_p95,
    }
    return {
        # Historical payloads retain the raw vectors. Explicit three-column
        # aliases carry summaries + pointers instead of duplicating matrices.
        "python": legacy,
        "legacy_adam_core": {
            "p50": legacy_p50,
            "p95": legacy_p95,
            "samples_alias": "python.values",
        },
        "rust": current,
        "current_python": {
            "p50": current_p50,
            "p95": current_p95,
            "samples_alias": "rust.values",
        },
        "native_rust": {
            "status": "measured",
            "operation": native_operation,
            "timer": "std::time::Instant",
            "values": native_rust_samples,
            "p50": native_p50,
            "p95": native_p95,
        },
        "speedup": {
            "p50_python_over_rust": legacy_p50 / current_p50,
            "p95_python_over_rust": legacy_p95 / current_p95,
            "p50_legacy_over_current_python": legacy_p50 / current_p50,
            "p95_legacy_over_current_python": legacy_p95 / current_p95,
        },
    }


def time_native_rust(
    propagator: Any, *, repeats: int, warmups: int
) -> tuple[str, list[float]]:
    """Read Rust-Instant samples for the most recently prepared operation."""
    operation, trials = propagator.benchmark_last_native(
        repeats, trials=1, warmup_reps=warmups
    )
    return operation, list(trials[0])


def time_rust(
    call: Callable[[], Any], *, repeats: int, warmups: int
) -> tuple[list[float], Any]:
    """Time a local Rust call; return (per-rep seconds, last result).

    Mirrors the legacy proxy ``time_*`` loop (one priming call + warmups +
    ``gc.collect()`` per rep) so both runtimes are measured comparably.
    """
    result = call()
    for _ in range(warmups):
        result = call()
    samples: list[float] = []
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter()
        result = call()
        samples.append(time.perf_counter() - started)
    return samples, result
