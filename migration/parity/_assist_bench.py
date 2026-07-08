"""Shared helpers for two-runtime adam_assist performance comparisons.

Pairs with :class:`migration.parity._assist_oracle.LegacyAssistPropagator`: the
legacy adam_assist call is timed *inside* the isolated ``.legacy-assist-venv``
(the proxy ``time_*`` methods), and the Rust ``adam_assist_rust`` call is timed
locally with :func:`time_rust`. :func:`percentiles` computes matching p50/p95
for both sides so speedups are apples-to-apples.
"""

from __future__ import annotations

import gc
import time
from typing import Any, Callable

import numpy as np

TWO_RUNTIME_COMPARISON_MODE = "gpl_rust_assist_backend_vs_legacy_python_adam_assist"


def percentiles(samples: list[float]) -> tuple[float, float]:
    """Return (p50, p95) seconds for a list of per-rep timings."""
    arr = np.asarray(samples, dtype=np.float64)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 95))


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
