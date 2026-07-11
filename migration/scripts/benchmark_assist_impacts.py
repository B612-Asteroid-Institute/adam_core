"""Two-runtime collision/impact-detection benchmark (bead personal-cmy.9).

Times legacy Python ``adam_assist.ASSISTPropagator.detect_collisions`` inside
``.legacy-assist-venv`` against local Rust-backed
``adam_assist.ASSISTPropagator.detect_collisions`` on identical same-epoch
workloads mixing deep Earth impactors (which exercise the stopping-condition
removal path) with safe heliocentric orbits (which exercise the long stepping
tail). Writes a JSON artifact next to the other assist public-semantics
benchmarks.

Parity is asserted per lane before timings are recorded: identical impact
sets and identical survivor sets. Impact-time agreement is reported (max
abs diff, days) rather than asserted at a threshold; the two libassist C
builds differ, so raw step sequences are not bit-identical (see
test_adam_assist_impacts.py for the gated tolerances).

Run with: pdm run assist-impacts-benchmark
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
from adam_assist import ASSISTPropagator as RustASSISTPropagator
from adam_core.coordinates import CartesianCoordinates, Origin
from adam_core.coordinates.origin import OriginCodes
from adam_core.dynamics.impacts import EARTH_RADIUS_KM, CollisionConditions
from adam_core.orbits import Orbits
from adam_core.time import Timestamp
from adam_core.utils.spice import get_perturber_state

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from migration.parity._assist_bench import (  # noqa: E402
    NATIVE_RUST_TODO,
    PERFORMANCE_COLUMNS,
    TWO_RUNTIME_COMPARISON_MODE,
    percentiles,
    time_rust,
)
from migration.parity._assist_oracle import (  # noqa: E402
    LEGACY_ASSIST_VENV_PYTHON,
    LegacyAssistPropagator,
)

EPOCH_MJD = 60000.0
NUM_DAYS = 30
REPEATS = 3
WARMUPS = 1
LANES = (10, 50, 200)
IMPACTOR_FRACTION = 0.2
ARTIFACT = Path("migration/artifacts/assist_impacts_benchmark_2026-07-03.json")


def _machine() -> dict[str, str]:
    try:
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        cpu = platform.processor()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_brand": cpu,
    }


def _lane_orbits(
    n: int, rng: np.random.Generator, *, impactor_fraction: float
) -> Orbits:
    """``n`` same-epoch orbits: a configurable fraction of deep radial-infall
    Earth impactors (offsets 30,000-90,000 km, Earth-matched velocity) and the
    rest safe orbits displaced 0.05-0.30 AU with perturbed velocities."""
    epoch = Timestamp.from_mjd([EPOCH_MJD], scale="tdb")
    earth = get_perturber_state(
        OriginCodes.EARTH, epoch, frame="ecliptic", origin=OriginCodes.SUN
    ).values[0]

    n_impactors = max(1, int(round(n * impactor_fraction)))
    values = np.tile(earth, (n, 1))
    unit = rng.normal(size=(n, 3))
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    # Impactors: small radial offsets, same velocity -> infall within ~1 day.
    values[:n_impactors, :3] += unit[:n_impactors] * rng.uniform(
        2.0e-4, 6.0e-4, size=(n_impactors, 1)
    )
    # Safe rows: large offsets and mildly perturbed velocities.
    values[n_impactors:, :3] += unit[n_impactors:] * rng.uniform(
        0.05, 0.30, size=(n - n_impactors, 1)
    )
    values[n_impactors:, 3:] += rng.normal(scale=2.0e-3, size=(n - n_impactors, 3))

    return Orbits.from_kwargs(
        orbit_id=[f"orbit_{i:04d}" for i in range(n)],
        object_id=["impactor" if i < n_impactors else "safe" for i in range(n)],
        coordinates=CartesianCoordinates.from_kwargs(
            x=values[:, 0],
            y=values[:, 1],
            z=values[:, 2],
            vx=values[:, 3],
            vy=values[:, 4],
            vz=values[:, 5],
            time=Timestamp.from_mjd([EPOCH_MJD] * n, scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN"] * n),
            frame="ecliptic",
        ),
    )


def _conditions() -> CollisionConditions:
    return CollisionConditions.from_kwargs(
        condition_id=["Earth"],
        collision_object=Origin.from_kwargs(code=["EARTH"]),
        collision_distance=[EARTH_RADIUS_KM],
        stopping_condition=[True],
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ARTIFACT)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--warmups", type=int, default=WARMUPS)
    parser.add_argument("--lanes", type=int, nargs="+", default=list(LANES))
    parser.add_argument("--num-days", type=int, default=NUM_DAYS)
    parser.add_argument("--impactor-fraction", type=float, default=IMPACTOR_FRACTION)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.repeats < 3:
        raise ValueError("--repeats must be at least 3 to report p50/p95")
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")
    if any(n < 1 for n in args.lanes):
        raise ValueError("--lanes entries must be positive")
    if not 0.0 < args.impactor_fraction <= 1.0:
        raise ValueError("--impactor-fraction must be in (0, 1]")
    if not LEGACY_ASSIST_VENV_PYTHON.exists():
        print(
            "legacy adam_assist runtime (.legacy-assist-venv) not built; "
            "see migration/parity/README"
        )
        return 1
    rng = np.random.default_rng(20260703)
    legacy_propagator = LegacyAssistPropagator()
    rust_propagator = RustASSISTPropagator()
    conditions = _conditions()

    lanes = []
    for n in args.lanes:
        orbits = _lane_orbits(n, rng, impactor_fraction=args.impactor_fraction)
        # Legacy timing runs inside the isolated .legacy-assist-venv runtime;
        # the parity output comes from a separate (cached) legacy call.
        py_samples = legacy_propagator.time_detect_collisions(
            orbits,
            args.num_days,
            conditions,
            repeats=args.repeats,
            warmups=args.warmups,
        )
        py_results, py_events = legacy_propagator.detect_collisions(
            orbits, args.num_days, conditions
        )
        rust_samples, (rust_results, rust_events) = time_rust(
            lambda o=orbits: rust_propagator.detect_collisions(
                o, args.num_days, conditions
            ),
            repeats=args.repeats,
            warmups=args.warmups,
        )

        py_impacted = sorted(py_events.orbit_id.to_pylist())
        rust_impacted = sorted(rust_events.orbit_id.to_pylist())
        assert py_impacted == rust_impacted, (
            f"lane n={n}: impact sets diverged "
            f"({len(py_impacted)} python vs {len(rust_impacted)} rust)"
        )
        assert sorted(py_results.orbit_id.to_pylist()) == sorted(
            rust_results.orbit_id.to_pylist()
        ), f"lane n={n}: result sets diverged"
        py_times = np.sort(
            py_events.coordinates.time.mjd().to_numpy(zero_copy_only=False)
        )
        rust_times = np.sort(
            rust_events.coordinates.time.mjd().to_numpy(zero_copy_only=False)
        )
        max_impact_time_diff = (
            float(np.abs(py_times - rust_times).max()) if len(py_times) else 0.0
        )

        py_p50, py_p95 = percentiles(py_samples)
        rust_p50, rust_p95 = percentiles(rust_samples)
        lane = {
            "n_orbits": n,
            "num_days": args.num_days,
            "n_impacts": len(py_impacted),
            "python_samples_s": py_samples,
            "rust_samples_s": rust_samples,
            "python_p50_s": py_p50,
            "python_p95_s": py_p95,
            "legacy_adam_core_p50_s": py_p50,
            "legacy_adam_core_p95_s": py_p95,
            "legacy_adam_core_samples_alias": "python_samples_s",
            "rust_p50_s": rust_p50,
            "rust_p95_s": rust_p95,
            "current_python_p50_s": rust_p50,
            "current_python_p95_s": rust_p95,
            "current_python_samples_alias": "rust_samples_s",
            "native_rust_status": "unavailable",
            "native_rust_samples_s": [],
            "native_rust_p50_s": None,
            "native_rust_p95_s": None,
            "native_rust_unavailable_reason": (
                "no Rust-internal Instant adapter; a Python->PyO3 call is not "
                "accepted as native-Rust timing"
            ),
            "native_rust_todo": NATIVE_RUST_TODO,
            "speedup_p50": py_p50 / rust_p50,
            "speedup_p95": py_p95 / rust_p95,
            "max_impact_time_diff_days": max_impact_time_diff,
        }
        lanes.append(lane)
        print(
            f"n={n:4d} impacts={lane['n_impacts']:3d} "
            f"python p50={py_p50:8.3f}s rust p50={rust_p50:8.3f}s "
            f"speedup p50={lane['speedup_p50']:6.2f}x p95={lane['speedup_p95']:6.2f}x "
            f"impact-time maxdiff={max_impact_time_diff:.2e} d"
        )

    artifact = {
        "schema_version": 3,
        "comparison_mode": TWO_RUNTIME_COMPARISON_MODE,
        "performance_columns": PERFORMANCE_COLUMNS,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": (
            "Two-runtime: legacy adam_assist.detect_collisions timed inside the "
            "isolated .legacy-assist-venv vs Rust adam_assist.detect_collisions "
            "timed locally, on identical same-epoch impactor/safe workloads; parity "
            "of impact and survivor sets asserted per lane before timing."
        ),
        "machine": _machine(),
        "packages": {
            "adam_assist": package_version("adam-assist"),
            "adam_core": package_version("adam-core"),
        },
        "repeats": args.repeats,
        "warmups": args.warmups,
        "impactor_fraction": args.impactor_fraction,
        "lanes": lanes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
