"""Generate pinned-main parity cases for departure_spherical_coordinates."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np

# The pristine legacy oracle intentionally omits plotting extras. This fixture
# exercises no plotting path, so provide an inert import shim rather than
# mutating that environment.
if importlib.util.find_spec("plotly") is None:
    plotly = types.ModuleType("plotly")
    graph_objects = types.ModuleType("plotly.graph_objects")
    plotly.graph_objects = graph_objects
    sys.modules["plotly"] = plotly
    sys.modules["plotly.graph_objects"] = graph_objects

from adam_core.coordinates import CartesianCoordinates, Origin
from adam_core.coordinates.origin import OriginCodes
from adam_core.missions.porkchop import (
    departure_spherical_coordinates,
    prepare_and_propagate_orbits,
)
from adam_core.orbits import Orbits
from adam_core.time import Timestamp
from adam_core.utils.spice import setup_SPICE

OUT = Path("migration/artifacts/departure_spherical_fixture_2026-07-12.json")


class EchoPropagator:
    """Provider-boundary fixture: repeat the Rust-prepared input at target times."""

    def propagate_orbits(self, body, times, max_processes=1):
        state = body.coordinates.values[0]
        rows = len(times)
        coordinates = CartesianCoordinates.from_kwargs(
            x=np.full(rows, state[0]),
            y=np.full(rows, state[1]),
            z=np.full(rows, state[2]),
            vx=np.full(rows, state[3]),
            vy=np.full(rows, state[4]),
            vz=np.full(rows, state[5]),
            time=times,
            origin=Origin.from_kwargs(
                code=[body.coordinates.origin.code[0].as_py()] * rows
            ),
            frame=body.coordinates.frame,
        )
        return Orbits.from_kwargs(
            orbit_id=[body.orbit_id[0].as_py()] * rows,
            object_id=[body.object_id[0].as_py()] * rows,
            coordinates=coordinates,
        )


def orbit_payload(orbits):
    return {
        "orbit_id": orbits.orbit_id.to_pylist(),
        "object_id": orbits.object_id.to_pylist(),
        "values": orbits.coordinates.values.tolist(),
        "days": orbits.coordinates.time.days.to_pylist(),
        "nanos": orbits.coordinates.time.nanos.to_pylist(),
        "scale": orbits.coordinates.time.scale,
        "origins": orbits.coordinates.origin.code.to_pylist(),
        "frame": orbits.coordinates.frame,
    }


def main():
    setup_SPICE()
    times = Timestamp.from_mjd([60000.0, 60000.5, 60001.25], scale="tdb")
    vx = np.array([1.0, -2.0, 0.25])
    vy = np.array([2.0, 0.5, -1.5])
    vz = np.array([0.5, -1.0, 3.0])
    cases = []
    for origin, frame in [
        (OriginCodes.EARTH, "ecliptic"),
        (OriginCodes.MARS, "equatorial"),
    ]:
        output = departure_spherical_coordinates(origin, times, frame, vx, vy, vz)
        cases.append(
            {
                "origin": origin.name,
                "frame_in": frame,
                "values": output.values.tolist(),
                "days": output.time.days.to_pylist(),
                "nanos": output.time.nanos.to_pylist(),
                "scale": output.time.scale,
                "origins": output.origin.code.to_pylist(),
                "frame": output.frame,
            }
        )
    start = Timestamp.from_mjd([60000.0], scale="utc")
    end = Timestamp.from_mjd([60002.0], scale="utc")
    major = prepare_and_propagate_orbits(
        OriginCodes.EARTH,
        start,
        end,
        propagation_origin=OriginCodes.SUN,
        step_size=0.5,
    )
    body = Orbits.from_kwargs(
        orbit_id=["provider-body"],
        object_id=["provider-body"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0],
            y=[0.1],
            z=[-0.05],
            vx=[0.001],
            vy=[0.017],
            vz=[0.0002],
            time=Timestamp.from_mjd([60000.0], scale="tdb"),
            origin=Origin.from_kwargs(code=["EARTH"]),
            frame="equatorial",
        ),
    )
    provider = prepare_and_propagate_orbits(
        body,
        Timestamp.from_mjd([60000.0], scale="tdb"),
        Timestamp.from_mjd([60002.0], scale="tdb"),
        propagation_origin=OriginCodes.SUN,
        step_size=0.5,
        propagator_class=EchoPropagator,
        max_processes=1,
    )
    OUT.write_text(
        json.dumps(
            {
                "legacy_commit": "4c1fbc4cd1a67b1e8527f20dce0b853b9a4022ac",
                "vx": vx.tolist(),
                "vy": vy.tolist(),
                "vz": vz.tolist(),
                "cases": cases,
                "prepare_major_body": orbit_payload(major),
                "prepare_provider_body": orbit_payload(provider),
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
