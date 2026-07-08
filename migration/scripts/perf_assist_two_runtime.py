"""Two-runtime performance smoke comparison for adam_assist (legacy vs Rust).

A quick, dependency-light check that the two-runtime perf plumbing works and
that the Rust backend beats legacy Python ASSIST on representative lanes. The
full, artifact-writing benchmarks live in
``migration/scripts/benchmark_assist_{public_semantics,covariance,impacts}.py``
(also two-runtime). The legacy ``adam_assist.ASSISTPropagator`` is timed inside
the isolated ``.legacy-assist-venv``; the Rust propagator is timed locally.

Run:  .venv/bin/python -m migration.scripts.perf_assist_two_runtime
Requires the dedicated legacy runtime (.legacy-assist-venv); see
migration/parity/README.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adam_assist_rust import ASSISTPropagator as RustASSISTPropagator  # noqa: E402
from adam_core.observers import Observers  # noqa: E402
from adam_core.time import Timestamp  # noqa: E402
from adam_core.utils.helpers.orbits import make_real_orbits  # noqa: E402
from migration.parity._assist_bench import percentiles, time_rust  # noqa: E402
from migration.parity._assist_oracle import (  # noqa: E402
    LEGACY_ASSIST_VENV_PYTHON,
    LegacyAssistPropagator,
)

REPEATS = 5
WARMUPS = 1
EPOCH_MJD = 60000.0


def _common_epoch_orbits(n: int):
    """n real orbits pinned to a common epoch (short, comparable integration)."""
    orbits = make_real_orbits(n)
    epoch = Timestamp.from_mjd(np.full(len(orbits), EPOCH_MJD), scale="tdb")
    return orbits.set_column("coordinates.time", epoch)


def main() -> int:
    if not LEGACY_ASSIST_VENV_PYTHON.exists():
        print("legacy .legacy-assist-venv not built; skipping perf comparison")
        return 0

    legacy = LegacyAssistPropagator()
    rust = RustASSISTPropagator()
    rows: list[tuple[str, list[float], list[float]]] = []

    targets = Timestamp.from_mjd([EPOCH_MJD + 0.5, EPOCH_MJD + 1.0], scale="tdb")
    for label, n in [("propagate 1x2", 1), ("propagate 20x2", 20)]:
        orbits = _common_epoch_orbits(n)
        legacy_s = legacy.time_propagate_orbits(
            orbits, targets, repeats=REPEATS, warmups=WARMUPS, max_processes=1
        )
        rust_s, _ = time_rust(
            lambda o=orbits: rust.propagate_orbits(o, targets, max_processes=1),
            repeats=REPEATS,
            warmups=WARMUPS,
        )
        rows.append((label, legacy_s, rust_s))

    orbits = _common_epoch_orbits(5)
    eph_times = Timestamp.from_mjd([EPOCH_MJD + 0.5, EPOCH_MJD + 1.0], scale="utc")
    observers = Observers.from_code("X05", eph_times)
    eph_kwargs = dict(
        max_processes=1, predict_magnitudes=False, predict_phase_angle=False
    )
    legacy_s = legacy.time_generate_ephemeris(
        orbits, observers, repeats=REPEATS, warmups=WARMUPS, **eph_kwargs
    )
    rust_s, _ = time_rust(
        lambda: rust.generate_ephemeris(orbits, observers, **eph_kwargs),
        repeats=REPEATS,
        warmups=WARMUPS,
    )
    rows.append(("ephemeris 5x2obs", legacy_s, rust_s))

    header = (
        f"{'lane':18} {'legacy p50':>12} {'rust p50':>11} {'x p50':>7} "
        f"{'legacy p95':>12} {'rust p95':>11} {'x p95':>7}"
    )
    print(
        f"\nadam_assist two-runtime perf (legacy Python ASSIST vs Rust), reps={REPEATS}"
    )
    print(header)
    print("-" * len(header))
    for label, legacy_s, rust_s in rows:
        lp50, lp95 = percentiles(legacy_s)
        rp50, rp95 = percentiles(rust_s)
        print(
            f"{label:18} {lp50 * 1e3:>10.2f}ms {rp50 * 1e3:>9.2f}ms {lp50 / rp50:>6.2f}x "
            f"{lp95 * 1e3:>10.2f}ms {rp95 * 1e3:>9.2f}ms {lp95 / rp95:>6.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
