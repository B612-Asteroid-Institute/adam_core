"""Focused benchmark for `dynamics.calc_mean_motion`.

The main gate's `calc_mean_motion` measurement is contaminated by state
from the many large memory-hungry benchmarks that run before it in the
same process — this kernel is memory-bandwidth-bound on its inputs
(4 flops per 24 bytes of memory traffic), so L3 cache and allocator
state from earlier benchmarks artificially inflate its per-call time.

This standalone script runs the measurement with a clean process state.
Reports legacy-JAX vs rust p50/p95 speedup at a size that matches
production call sites (KeplerianCoordinates.n and CometaryCoordinates.n
typically evaluate on 100..100k orbits per batch).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from adam_core._rust import RUST_BACKEND_AVAILABLE
from adam_core._rust import api as rust_api
from adam_core.dynamics.kepler import _calc_mean_motion_jax


REPEATS = 9  # extra repeats to dampen outliers
N = 50_000


def _timed(fn) -> np.ndarray:
    fn()  # prime (full-size JIT trace + rayon pool wake)
    samples = np.empty(REPEATS, dtype=np.float64)
    for i in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        samples[i] = time.perf_counter() - t0
    return samples


def main() -> None:
    if not RUST_BACKEND_AVAILABLE:
        raise SystemExit("Rust backend unavailable; run `maturin develop` first.")

    rng = np.random.default_rng(20260414)
    a = np.ascontiguousarray(0.5 + 4.0 * rng.random(N), dtype=np.float64)
    mu = np.ascontiguousarray(2.959122e-4 * np.ones(N), dtype=np.float64)

    # Pre-warm on sub-slice so JAX JIT is compiled before the timed prime.
    _ = np.asarray(_calc_mean_motion_jax(a[:1024], mu[:1024]), dtype=np.float64)
    _ = rust_api.calc_mean_motion_numpy(a[:1024], mu[:1024])

    legacy_ts = _timed(lambda: np.asarray(_calc_mean_motion_jax(a, mu), dtype=np.float64))
    rust_ts = _timed(lambda: np.asarray(rust_api.calc_mean_motion_numpy(a, mu), dtype=np.float64))

    # Parity check (bit-exact except for FP reordering; atol=1e-13 matches
    # the main gate's convention).
    legacy_out = np.asarray(_calc_mean_motion_jax(a, mu), dtype=np.float64)
    rust_out = np.asarray(rust_api.calc_mean_motion_numpy(a, mu), dtype=np.float64)
    np.testing.assert_allclose(rust_out, legacy_out, rtol=0.0, atol=1e-13)

    report = {
        "api": "dynamics.calc_mean_motion",
        "n": N,
        "repeats": REPEATS,
        "legacy_seconds_p50": float(np.median(legacy_ts)),
        "legacy_seconds_p95": float(np.percentile(legacy_ts, 95)),
        "rust_seconds_p50": float(np.median(rust_ts)),
        "rust_seconds_p95": float(np.percentile(rust_ts, 95)),
        "speedup_p50": float(np.median(legacy_ts) / np.median(rust_ts)),
        "speedup_p95": float(
            np.percentile(legacy_ts, 95) / np.percentile(rust_ts, 95)
        ),
        "legacy_samples_seconds": legacy_ts.tolist(),
        "rust_samples_seconds": rust_ts.tolist(),
    }
    out_path = Path("migration/artifacts/calc_mean_motion_bench.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
