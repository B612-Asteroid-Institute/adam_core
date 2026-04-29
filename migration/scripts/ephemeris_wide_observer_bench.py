"""Ad-hoc ephemeris benchmark: few orbits × many observers.

This is a one-off measurement at a workload shape the main gate doesn't
cover: 100 orbits × 100_000 observation epochs = 10_000_000 rows for the
state path. Memory for the cov path at 10M is ~6 GB (input + output 6x6
per row) so cov runs at 100 × 10_000 = 1_000_000 rows.

Reuses the builders and JAX fallback helpers from the gate harness so
the measurement is apples-to-apples with the standing benchmark.

Usage:
    python migration/scripts/ephemeris_wide_observer_bench.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adam_core._rust import api as rust_api  # noqa: E402
from rust_backend_benchmark_gate import (  # noqa: E402
    _build_ephemeris_cov_inputs,
    _build_ephemeris_inputs,
    _ephemeris_jax_chunked,
    _ephemeris_jax_cov,
)

REPEATS = 5


def _timed(fn):
    # Prime once on full-size input before taking timed repeats (matches
    # the main gate's _timed_runs policy).
    out = fn()
    times = np.empty(REPEATS, dtype=np.float64)
    for i in range(REPEATS):
        t0 = time.perf_counter()
        out = fn()
        times[i] = time.perf_counter() - t0
    return times, out


def _summary(legacy_ts: np.ndarray, rust_ts: np.ndarray) -> dict:
    return {
        "legacy_p50_s": float(np.median(legacy_ts)),
        "legacy_p95_s": float(np.percentile(legacy_ts, 95)),
        "rust_p50_s": float(np.median(rust_ts)),
        "rust_p95_s": float(np.percentile(rust_ts, 95)),
        "speedup_p50": float(np.median(legacy_ts) / np.median(rust_ts)),
        "speedup_p95": float(np.percentile(legacy_ts, 95) / np.percentile(rust_ts, 95)),
    }


def main() -> None:
    # State: 100 orbits x 100_000 observations = 10M rows.
    print("Building state inputs (100 x 100_000 = 10_000_000 rows)...", flush=True)
    eph_orbits, eph_times, eph_observers, eph_mus = _build_ephemeris_inputs(
        100, 100_000
    )
    print(f"  state rows: {eph_orbits.shape[0]:,}", flush=True)

    # Cov: 100 x 10_000 = 1M rows (cov arrays are 36-wide so memory
    # scales 6x faster than state).
    print("Building cov inputs (100 x 10_000 = 1_000_000 rows)...", flush=True)
    (
        eph_cov_orbits,
        eph_cov_cov_in,
        eph_cov_times,
        eph_cov_observers,
        eph_cov_mus,
    ) = _build_ephemeris_cov_inputs(100, 10_000)
    print(f"  cov rows: {eph_cov_orbits.shape[0]:,}", flush=True)

    # Warm-up at full size (JAX jits shape-specifically + rayon pool startup).
    print("Warming JAX state path...", flush=True)
    _ = _ephemeris_jax_chunked(eph_orbits, eph_times, eph_observers, eph_mus)
    print("Warming Rust state path...", flush=True)
    _ = rust_api.generate_ephemeris_2body_numpy(eph_orbits, eph_observers, eph_mus)
    print("Warming JAX cov path...", flush=True)
    _ = _ephemeris_jax_cov(
        eph_cov_orbits, eph_cov_cov_in, eph_cov_times, eph_cov_observers, eph_cov_mus
    )
    print("Warming Rust cov path...", flush=True)
    _ = rust_api.generate_ephemeris_2body_with_covariance_numpy(
        eph_cov_orbits, eph_cov_cov_in, eph_cov_observers, eph_cov_mus
    )

    print(f"Timing state path ({REPEATS} repeats each)...", flush=True)
    legacy_state_ts, legacy_state_out = _timed(
        lambda: _ephemeris_jax_chunked(eph_orbits, eph_times, eph_observers, eph_mus)
    )
    rust_state_ts, rust_state_out = _timed(
        lambda: np.asarray(
            rust_api.generate_ephemeris_2body_numpy(eph_orbits, eph_observers, eph_mus)[
                0
            ],
            dtype=np.float64,
        )
    )
    np.testing.assert_allclose(rust_state_out, legacy_state_out, rtol=1e-11, atol=1e-9)

    print(f"Timing cov path ({REPEATS} repeats each)...", flush=True)
    legacy_cov_ts, legacy_cov_out = _timed(
        lambda: _ephemeris_jax_cov(
            eph_cov_orbits,
            eph_cov_cov_in,
            eph_cov_times,
            eph_cov_observers,
            eph_cov_mus,
        )
    )

    def _rust_cov():
        _, _, _, cov = rust_api.generate_ephemeris_2body_with_covariance_numpy(
            eph_cov_orbits, eph_cov_cov_in, eph_cov_observers, eph_cov_mus
        )
        return np.asarray(cov, dtype=np.float64).reshape(-1, 6, 6)

    rust_cov_ts, rust_cov_out = _timed(_rust_cov)
    np.testing.assert_allclose(rust_cov_out, legacy_cov_out, rtol=1e-5, atol=1e-14)

    report = {
        "grid_state": "100 orbits x 100000 observers = 10,000,000 rows",
        "grid_cov": "100 orbits x 10000 observers = 1,000,000 rows",
        "repeats": REPEATS,
        "generate_ephemeris_2body": _summary(legacy_state_ts, rust_state_ts),
        "generate_ephemeris_2body_with_covariance": _summary(
            legacy_cov_ts, rust_cov_ts
        ),
    }
    out_path = Path("migration/artifacts/ephemeris_wide_observer_bench.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
