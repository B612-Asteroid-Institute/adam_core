"""Generate pinned-main fixtures for propagator compatibility utilities.

Run with the pristine legacy environment and pinned checkout::

    PYTHONPATH=/Users/aleck/Code/adam-core-legacy-main/src \
      .legacy-venv/bin/python migration/scripts/generate_propagator_utils_fixture.py
"""

import json
from pathlib import Path

from adam_core.coordinates import CartesianCoordinates, Origin
from adam_core.orbits import Orbits
from adam_core.propagator.utils import (
    ensure_input_origin_and_frame,
    ensure_input_time_scale,
)
from adam_core.time import Timestamp
from adam_core.utils.spice import setup_SPICE

OUT = Path("migration/artifacts/propagator_utils_fixture_2026-07-12.json")


def make_cases():
    input_times = Timestamp.from_mjd([60000.0, 60001.0, 60002.0], scale="tdb")
    inputs = Orbits.from_kwargs(
        orbit_id=["a", "b", "c"],
        object_id=["A", "B", "C"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0, 2.0, 3.0],
            y=[0.1, 0.2, 0.3],
            z=[-0.1, -0.2, -0.3],
            vx=[0.01, 0.02, 0.03],
            vy=[0.001, 0.002, 0.003],
            vz=[-0.001, -0.002, -0.003],
            time=input_times,
            origin=Origin.from_kwargs(code=["EARTH", "SUN", "MOON"]),
            frame="equatorial",
        ),
    )
    result_times = Timestamp.from_mjd([60002.0, 60000.0, 60001.0, 60000.5], scale="tdb")
    results = Orbits.from_kwargs(
        orbit_id=["c", "a", "b", "a"],
        object_id=["C", "A", "B", "A"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[3.0, 1.0, 2.0, 1.5],
            y=[0.3, 0.1, 0.2, 0.15],
            z=[-0.3, -0.1, -0.2, -0.15],
            vx=[0.03, 0.01, 0.02, 0.015],
            vy=[0.003, 0.001, 0.002, 0.0015],
            vz=[-0.003, -0.001, -0.002, -0.0015],
            time=result_times,
            origin=Origin.from_kwargs(code=["SOLAR_SYSTEM_BARYCENTER"] * 4),
            frame="ecliptic",
        ),
    )
    return inputs, results


def main():
    setup_SPICE()
    inputs, results = make_cases()
    utc_results = results.set_column(
        "coordinates.time", results.coordinates.time.rescale("utc")
    )
    time_output = ensure_input_time_scale(utc_results, inputs.coordinates.time)
    origin_output = ensure_input_origin_and_frame(inputs, results)
    payload = {
        "legacy_commit": "4c1fbc4cd1a67b1e8527f20dce0b853b9a4022ac",
        "time_scale": {
            "days": time_output.coordinates.time.days.to_pylist(),
            "nanos": time_output.coordinates.time.nanos.to_pylist(),
            "scale": time_output.coordinates.time.scale,
        },
        "origin_and_frame": {
            "orbit_id": origin_output.orbit_id.to_pylist(),
            "object_id": origin_output.object_id.to_pylist(),
            "values": origin_output.coordinates.values.tolist(),
            "days": origin_output.coordinates.time.days.to_pylist(),
            "nanos": origin_output.coordinates.time.nanos.to_pylist(),
            "scale": origin_output.coordinates.time.scale,
            "origins": origin_output.coordinates.origin.code.to_pylist(),
            "frame": origin_output.coordinates.frame,
        },
    }
    OUT.write_text(json.dumps(payload, indent=1, allow_nan=True) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
