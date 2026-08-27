"""Generate the frozen legacy fixture for ``orbits_to_sbdb_file`` (bead
personal-cmy.37.4.5): deterministic orbit panels -> exact CSV bytes written by
the untouched legacy checkout (astropy epoch strings + pandas ``to_csv``).

Run with the pinned legacy interpreter from the repo root:

    .legacy-venv/bin/python migration/scripts/generate_openspace_sbdb_fixture.py

Panels deliberately exercise: exponent-notation float reprs (e < 1e-4),
integral and fractional MJD epochs (including 1-nanosecond fractions), a
comma-bearing orbit id (pandas QUOTE_MINIMAL), non-heliocentric origins and
an equatorial input frame (origin translation + rotation), and a UTC-scale
input (TDB rescale). Also records a Python ``repr(float)`` oracle table for
the Rust ``py_float_repr`` unit tests.
"""

import json
import tempfile
from pathlib import Path

from adam_core.coordinates import CartesianCoordinates, KeplerianCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.coordinates.transform import transform_coordinates
from adam_core.orbits import Orbits
from adam_core.orbits.openspace.assets import orbits_to_sbdb_file
from adam_core.time import Timestamp

OUT = Path("migration/artifacts/openspace_sbdb_fixture_2026-07-12.json")

FLOAT_REPR_VALUES = [
    0.0,
    -0.0,
    1.0,
    -0.5,
    0.1,
    123456.789,
    60000.5,
    10000.0,
    0.0001,
    1e-05,
    5e-05,
    2.5e-10,
    5e-324,
    1e15,
    9999999999999998.0,
    1e16,
    1.2345678901234567e20,
    -3.75e18,
    6.02e23,
    1.7976931348623157e308,
    3.141592653589793,
    2.2250738585072014e-308,
]


def flat(orbits: Orbits) -> dict:
    c = orbits.coordinates
    return {
        "orbit_id": orbits.orbit_id.to_pylist(),
        "object_id": orbits.object_id.to_pylist(),
        "x": c.x.to_pylist(),
        "y": c.y.to_pylist(),
        "z": c.z.to_pylist(),
        "vx": c.vx.to_pylist(),
        "vy": c.vy.to_pylist(),
        "vz": c.vz.to_pylist(),
        "days": c.time.days.to_pylist(),
        "nanos": c.time.nanos.to_pylist(),
        "scale": c.time.scale,
        "frame": c.frame,
        "origin": c.origin.code.to_pylist(),
    }


def write_csv(orbits: Orbits) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "panel.csv"
        orbits_to_sbdb_file(orbits, str(path))
        return path.read_text()


def main() -> None:
    n = 8
    keplerian = KeplerianCoordinates.from_kwargs(
        a=[1.2, 2.5, 0.9, 5.5, 1.0000001, 3.3, 2.1, 40.0],
        e=[5e-05, 0.1, 0.25, 0.7, 0.02, 0.05, 0.5, 0.9],
        i=[5.0, 10.5, 2.25, 30.0, 0.001, 15.75, 8.5, 170.0],
        raan=[10.0, 80.0, 350.0, 120.5, 0.125, 200.25, 45.0, 300.0],
        ap=[20.0, 90.0, 340.0, 60.75, 0.25, 180.5, 270.0, 10.0],
        # M avoids exactly 180.0: the legacy checkout's cartesian->keplerian
        # returns NaN mean anomaly at an exact half revolution (a legacy
        # transform defect already governed by the transform parity RCA);
        # the CSV fixture pins the CSV product, not that defect.
        M=[0.25, 45.5, 180.25, 359.9, 90.0, 270.25, 30.0, 120.0],
        time=Timestamp.from_kwargs(
            days=[60000, 60000, 60123, 60123, 59870, 59870, 60250, 60250],
            nanos=[
                0,
                43_200_000_000_000,
                123_456_789,
                86_399_000_000_000,
                0,
                1,
                500_000_000,
                60_000_000_000_000,
            ],
            scale="tdb",
        ),
        origin=Origin.from_kwargs(code=["SUN"] * n),
        frame="ecliptic",
    )
    cartesian = transform_coordinates(
        keplerian, representation_out=CartesianCoordinates
    )
    orbit_ids = [
        "orbit_00",
        "orbit_01",
        "orbit_02",
        "orbit,03",  # exercises pandas QUOTE_MINIMAL
        "orbit 04",
        "orbit_05",
        "orbit_06",
        "orbit_07",
    ]
    object_ids = [f"object_{i:02d}" for i in range(n)]

    panel_a = Orbits.from_kwargs(
        orbit_id=orbit_ids, object_id=object_ids, coordinates=cartesian
    )

    cartesian_b = transform_coordinates(
        cartesian,
        frame_out="equatorial",
        origin_out=OriginCodes.SOLAR_SYSTEM_BARYCENTER,
    )
    panel_b = Orbits.from_kwargs(
        orbit_id=orbit_ids, object_id=object_ids, coordinates=cartesian_b
    )

    cartesian_c = CartesianCoordinates.from_kwargs(
        x=cartesian.x,
        y=cartesian.y,
        z=cartesian.z,
        vx=cartesian.vx,
        vy=cartesian.vy,
        vz=cartesian.vz,
        time=Timestamp.from_kwargs(
            days=cartesian.time.days,
            nanos=cartesian.time.nanos,
            scale="utc",
        ),
        origin=cartesian.origin,
        frame=cartesian.frame,
    )
    panel_c = Orbits.from_kwargs(
        orbit_id=orbit_ids, object_id=object_ids, coordinates=cartesian_c
    )

    fixture = {
        "panels": [
            {"name": name, "orbits": flat(panel), "csv": write_csv(panel)}
            for name, panel in [
                ("heliocentric_ecliptic_tdb", panel_a),
                ("ssb_equatorial_tdb", panel_b),
                ("heliocentric_ecliptic_utc", panel_c),
            ]
        ],
        "float_repr_oracle": [
            {"value_hex": value.hex(), "repr": repr(value)}
            for value in FLOAT_REPR_VALUES
        ],
    }
    OUT.write_text(json.dumps(fixture, indent=1))
    print(f"wrote {OUT} ({len(fixture['panels'])} panels)")


if __name__ == "__main__":
    main()
