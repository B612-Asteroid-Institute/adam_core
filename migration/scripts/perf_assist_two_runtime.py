"""Two-runtime performance comparison for adam_assist (legacy Python vs Rust).

Mirrors the adam_core speed gate's two-runtime model. The legacy,
composition-based ``adam_assist.ASSISTPropagator`` is timed *inside* the
isolated ``.legacy-assist-venv`` (the timing loop runs in that subprocess, so
per-rep seconds exclude subprocess spawn + Arrow-IPC transfer), and the Rust
``adam_assist_rust.ASSISTPropagator`` is timed locally in this runtime. Prints a
p50 / p95 / speedup table per lane.

Run:  .venv/bin/python -m migration.scripts.perf_assist_two_runtime
Requires the dedicated legacy runtime (.legacy-assist-venv); see
migration/parity/README.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adam_assist_rust import ASSISTPropagator as RustASSISTPropagator  # noqa: E402
from adam_core.observers import Observers  # noqa: E402
from adam_core.time import Timestamp  # noqa: E402
from adam_core.utils.helpers.orbits import make_real_orbits  # noqa: E402
from migration.parity._assist_oracle import (  # noqa: E402
    LEGACY_ASSIST_VENV_PYTHON,
    time_legacy,
)
from migration.parity._assist_serde import table_to_ipc  # noqa: E402

REPS = 5
WARMUP = 1
EPOCH_MJD = 60000.0


def _pctl(xs: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


def _time_local(fn, reps: int = REPS, warmup: int = WARMUP) -> list[float]:
    for _ in range(warmup):
        fn()
    out: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def _common_epoch_orbits(n: int):
    """n real orbits pinned to a common epoch (short, comparable integration)."""
    orbits = make_real_orbits(n)
    epoch = Timestamp.from_mjd(np.full(len(orbits), EPOCH_MJD), scale="tdb")
    return orbits.set_column("coordinates.time", epoch)


def main() -> int:
    if not LEGACY_ASSIST_VENV_PYTHON.exists():
        print("legacy .legacy-assist-venv not built; skipping perf comparison")
        return 0

    rust = RustASSISTPropagator()
    rows: list[tuple[str, list[float], list[float]]] = []

    targets = Timestamp.from_mjd([EPOCH_MJD + 0.5, EPOCH_MJD + 1.0], scale="tdb")
    for label, n in [("propagate 1x2", 1), ("propagate 20x2", 20)]:
        orbits = _common_epoch_orbits(n)
        legacy = time_legacy(
            "propagate_orbits",
            orbits=table_to_ipc(orbits),
            orbits_cls="Orbits",
            times=table_to_ipc(targets),
            kwargs={"max_processes": 1},
            reps=REPS,
            warmup=WARMUP,
        )
        local = _time_local(
            lambda o=orbits: rust.propagate_orbits(o, targets, max_processes=1)
        )
        rows.append((label, legacy, local))

    orbits = _common_epoch_orbits(5)
    eph_times = Timestamp.from_mjd([EPOCH_MJD + 0.5, EPOCH_MJD + 1.0], scale="utc")
    observers = Observers.from_code("X05", eph_times)
    eph_kwargs = dict(
        max_processes=1, predict_magnitudes=False, predict_phase_angle=False
    )
    legacy = time_legacy(
        "generate_ephemeris",
        orbits=table_to_ipc(orbits),
        orbits_cls="Orbits",
        observers=table_to_ipc(observers),
        kwargs=eph_kwargs,
        reps=REPS,
        warmup=WARMUP,
    )
    local = _time_local(
        lambda: rust.generate_ephemeris(orbits, observers, **eph_kwargs)
    )
    rows.append(("ephemeris 5x2obs", legacy, local))

    header = (
        f"{'lane':18} {'legacy p50':>12} {'rust p50':>11} {'x p50':>7} "
        f"{'legacy p95':>12} {'rust p95':>11} {'x p95':>7}"
    )
    print(f"\nadam_assist two-runtime perf (legacy Python ASSIST vs Rust), reps={REPS}")
    print(header)
    print("-" * len(header))
    for label, legacy_t, local_t in rows:
        lp50, rp50 = _pctl(legacy_t, 50), _pctl(local_t, 50)
        lp95, rp95 = _pctl(legacy_t, 95), _pctl(local_t, 95)
        print(
            f"{label:18} {lp50 * 1e3:>10.2f}ms {rp50 * 1e3:>9.2f}ms {lp50 / rp50:>6.2f}x "
            f"{lp95 * 1e3:>10.2f}ms {rp95 * 1e3:>9.2f}ms {lp95 / rp95:>6.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
