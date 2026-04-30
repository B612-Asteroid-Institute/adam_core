"""Focused benchmark for ``dynamics.calc_mean_motion``.

The main gate's calc_mean_motion measurement can be contaminated by state from
larger memory-heavy benchmarks that run before it. This standalone script uses
our baseline-main legacy oracle for the Python reference and the current Rust
backend for the migrated implementation.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from adam_core._rust import api as rust_api
from migration.parity._oracle import parity, time_legacy

REPEATS = 9
N = 50_000


def _timed(fn: Callable[[], object]) -> np.ndarray:
    fn()
    samples = np.empty(REPEATS, dtype=np.float64)
    for i in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        samples[i] = time.perf_counter() - t0
    return samples


def main() -> None:
    rng = np.random.default_rng(20260414)
    a = np.ascontiguousarray(0.5 + 4.0 * rng.random(N), dtype=np.float64)
    mu = np.ascontiguousarray(2.959122e-4 * np.ones(N), dtype=np.float64)

    _ = rust_api.calc_mean_motion_numpy(a[:1024], mu[:1024])

    legacy_ts = np.asarray(
        time_legacy("dynamics.calc_mean_motion", reps=REPEATS, warmup=1, a=a, mu=mu),
        dtype=np.float64,
    )
    rust_ts = _timed(
        lambda: np.asarray(rust_api.calc_mean_motion_numpy(a, mu), dtype=np.float64)
    )

    legacy_out = np.asarray(parity("dynamics.calc_mean_motion", a=a, mu=mu)["out"])
    rust_out = np.asarray(rust_api.calc_mean_motion_numpy(a, mu), dtype=np.float64)
    np.testing.assert_allclose(rust_out, legacy_out, rtol=0.0, atol=1e-13)

    report = {
        "api": "dynamics.calc_mean_motion",
        "n": N,
        "repeats": REPEATS,
        "reference": "baseline-main legacy oracle",
        "legacy_seconds_p50": float(np.median(legacy_ts)),
        "legacy_seconds_p95": float(np.percentile(legacy_ts, 95)),
        "rust_seconds_p50": float(np.median(rust_ts)),
        "rust_seconds_p95": float(np.percentile(rust_ts, 95)),
        "speedup_p50": float(np.median(legacy_ts) / np.median(rust_ts)),
        "speedup_p95": float(np.percentile(legacy_ts, 95) / np.percentile(rust_ts, 95)),
        "legacy_samples_seconds": legacy_ts.tolist(),
        "rust_samples_seconds": rust_ts.tolist(),
    }
    out_path = Path("migration/artifacts/calc_mean_motion_bench.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
