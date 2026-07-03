"""Apples-to-apples collision/impact-detection benchmark (bead personal-cmy.9).

Times Python ``adam_assist.ASSISTPropagator._detect_collisions`` against the
Rust-backed ``adam_assist_rust.ASSISTPropagator._detect_collisions`` on
identical same-epoch workloads mixing deep Earth impactors (which exercise the
stopping-condition removal path) with safe heliocentric orbits (which exercise
the long stepping tail). Writes a JSON artifact next to the other assist
public-semantics benchmarks.

Parity is asserted per lane before timings are recorded: identical impact
sets and identical survivor sets. Impact-time agreement is reported (max
abs diff, days) rather than asserted at a threshold; the two libassist C
builds differ, so raw step sequences are not bit-identical (see
test_adam_assist_rust_impacts.py for the gated tolerances).

Run with: pdm run assist-impacts-benchmark
"""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import time
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
from adam_assist import ASSISTPropagator as PythonASSISTPropagator
from adam_core.coordinates import CartesianCoordinates, Origin
from adam_core.coordinates.origin import OriginCodes
from adam_core.dynamics.impacts import EARTH_RADIUS_KM, CollisionConditions
from adam_core.orbits import Orbits
from adam_core.time import Timestamp
from adam_core.utils.spice import get_perturber_state

from adam_assist_rust import ASSISTPropagator as RustASSISTPropagator

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


def _lane_orbits(n: int, rng: np.random.Generator) -> Orbits:
    """``n`` same-epoch orbits: IMPACTOR_FRACTION deep radial-infall Earth
    impactors (offsets 30,000-90,000 km, Earth-matched velocity) and the rest
    safe orbits displaced 0.05-0.30 AU with perturbed velocities."""
    epoch = Timestamp.from_mjd([EPOCH_MJD], scale="tdb")
    earth = get_perturber_state(
        OriginCodes.EARTH, epoch, frame="ecliptic", origin=OriginCodes.SUN
    ).values[0]

    n_impactors = max(1, int(round(n * IMPACTOR_FRACTION)))
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


def _time_detect(propagator, orbits, conditions) -> tuple[list[float], object, object]:
    samples: list[float] = []
    results = events = None
    for _ in range(WARMUPS):
        propagator._detect_collisions(orbits, NUM_DAYS, conditions)
    for _ in range(REPEATS):
        start = time.perf_counter()
        results, events = propagator._detect_collisions(orbits, NUM_DAYS, conditions)
        samples.append(time.perf_counter() - start)
    return samples, results, events


def _p50_p95(samples: list[float]) -> tuple[float, float]:
    ordered = sorted(samples)
    return (
        statistics.median(ordered),
        float(np.percentile(np.asarray(ordered), 95)),
    )


def main() -> None:
    rng = np.random.default_rng(20260703)
    python_propagator = PythonASSISTPropagator()
    rust_propagator = RustASSISTPropagator()
    conditions = _conditions()

    lanes = []
    for n in LANES:
        orbits = _lane_orbits(n, rng)
        py_samples, py_results, py_events = _time_detect(
            python_propagator, orbits, conditions
        )
        rust_samples, rust_results, rust_events = _time_detect(
            rust_propagator, orbits, conditions
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

        py_p50, py_p95 = _p50_p95(py_samples)
        rust_p50, rust_p95 = _p50_p95(rust_samples)
        lane = {
            "n_orbits": n,
            "num_days": NUM_DAYS,
            "n_impacts": len(py_impacted),
            "python_samples_s": py_samples,
            "rust_samples_s": rust_samples,
            "python_p50_s": py_p50,
            "python_p95_s": py_p95,
            "rust_p50_s": rust_p50,
            "rust_p95_s": rust_p95,
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
        "schema_version": 1,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": (
            "Python adam_assist vs Rust adam_assist_rust _detect_collisions on "
            "identical same-epoch impactor/safe workloads; parity of impact and "
            "survivor sets asserted per lane before timing."
        ),
        "machine": _machine(),
        "packages": {
            "adam_assist": package_version("adam-assist"),
            "adam_assist_rust": package_version("adam-assist-rust"),
            "adam_core": package_version("adam-core"),
        },
        "repeats": REPEATS,
        "warmups": WARMUPS,
        "impactor_fraction": IMPACTOR_FRACTION,
        "lanes": lanes,
    }
    ARTIFACT.write_text(json.dumps(artifact, indent=2))
    print(f"wrote {ARTIFACT}")


if __name__ == "__main__":
    main()
