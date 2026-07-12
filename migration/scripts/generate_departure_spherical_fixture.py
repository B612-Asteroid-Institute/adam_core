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

from adam_core.coordinates.origin import OriginCodes
from adam_core.missions.porkchop import departure_spherical_coordinates
from adam_core.time import Timestamp
from adam_core.utils.spice import setup_SPICE

OUT = Path("migration/artifacts/departure_spherical_fixture_2026-07-12.json")


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
    OUT.write_text(
        json.dumps(
            {
                "legacy_commit": "4c1fbc4cd1a67b1e8527f20dce0b853b9a4022ac",
                "vx": vx.tolist(),
                "vy": vy.tolist(),
                "vz": vz.tolist(),
                "cases": cases,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
