"""Generate frozen legacy NumPy-lstsq fixtures for public fit_chebyshev."""

import json
from pathlib import Path

import numpy as np

from adam_core.coordinates import CartesianCoordinates
from adam_core.coordinates.origin import Origin
from adam_core.orbits.spice_kernel import fit_chebyshev
from adam_core.time import Timestamp

OUT = Path("migration/artifacts/spk_fit_fixture_2026-07-12.json")


def main():
    count = 21
    days = np.arange(60000, 60000 + count, dtype=np.int64)
    t = np.arange(count, dtype=np.float64)
    coordinates = CartesianCoordinates.from_kwargs(
        x=1.0 + 0.01 * t + 0.0001 * t**2,
        y=-0.5 + 0.02 * t - 0.0002 * t**2,
        z=0.1 * np.sin(t / 3),
        vx=0.001 + 1e-5 * t,
        vy=-0.002 + 2e-5 * t,
        vz=0.0005 * np.cos(t / 4),
        time=Timestamp.from_kwargs(
            days=days, nanos=np.zeros(count, dtype=np.int64), scale="tdb"
        ),
        origin=Origin.from_kwargs(code=["SOLAR_SYSTEM_BARYCENTER"] * count),
        frame="equatorial",
    )
    et = coordinates.time.et().to_numpy(zero_copy_only=False)
    cases = []
    for name, first, last, degree in [
        ("overdetermined_degree3", 2, 18, 3),
        ("underdetermined_degree15", 0, 10, 15),
    ]:
        start, end = float(et[first]), float(et[last])
        coefficients, actual_mid, actual_half = fit_chebyshev(
            coordinates, start, end, degree
        )
        cases.append(
            {
                "name": name,
                "window_start": start,
                "window_end": end,
                "degree": degree,
                "mid_time": None,
                "half_interval": None,
                "coefficients": coefficients.tolist(),
                "actual_mid": actual_mid,
                "actual_half": actual_half,
            }
        )
    OUT.write_text(
        json.dumps(
            {
                "days": days.tolist(),
                "values": coordinates.values.tolist(),
                "cases": cases,
            },
            indent=1,
        )
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
