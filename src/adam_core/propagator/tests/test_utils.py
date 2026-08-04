import numpy as np

from adam_core.coordinates import CartesianCoordinates, Origin
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

from ..utils import ensure_input_origin_and_frame, ensure_input_time_scale


def test_ensure_input_time_scale():
    # Test that ensure_input_time_scale works
    times = Timestamp.from_mjd(np.array([1.0, 2.0, 3.0]), scale="tdb")
    orbits = Orbits.from_kwargs(
        coordinates=CartesianCoordinates.from_kwargs(
            x=np.array([1.0, 2.0, 3.0]),
            y=np.array([1.0, 2.0, 3.0]),
            z=np.array([1.0, 2.0, 3.0]),
            vx=np.array([1.0, 2.0, 3.0]),
            vy=np.array([1.0, 2.0, 3.0]),
            vz=np.array([1.0, 2.0, 3.0]),
            time=times,
        ),
    )

    results = orbits.set_column(
        "coordinates.time", orbits.coordinates.time.rescale("utc")
    )
    assert results.coordinates.time.scale == "utc"

    results = ensure_input_time_scale(orbits, times)
    assert results.coordinates.time.scale == "tdb"


def test_ensure_input_origin_and_frame():
    # Test that ensure_input_origin_and_frame works
    times = Timestamp.from_mjd(np.array([60000.0, 60001.0, 60002.0]), scale="tdb")
    orbits = Orbits.from_kwargs(
        coordinates=CartesianCoordinates.from_kwargs(
            x=np.array([1.0, 2.0, 3.0]),
            y=np.array([1.0, 2.0, 3.0]),
            z=np.array([1.0, 2.0, 3.0]),
            vx=np.array([1.0, 2.0, 3.0]),
            vy=np.array([1.0, 2.0, 3.0]),
            vz=np.array([1.0, 2.0, 3.0]),
            time=times,
            origin=Origin.from_kwargs(code=["EARTH", "SUN", "MOON"]),
            frame="equatorial",
        ),
    )

    results = Orbits.from_kwargs(
        orbit_id=orbits.orbit_id,
        coordinates=CartesianCoordinates.from_kwargs(
            x=orbits.coordinates.x,
            y=orbits.coordinates.y,
            z=orbits.coordinates.z,
            vx=orbits.coordinates.vx,
            vy=orbits.coordinates.vy,
            vz=orbits.coordinates.vz,
            time=orbits.coordinates.time,
            origin=Origin.from_kwargs(
                code=["SOLAR_SYSTEM_BARYCENTER", "SUN", "SOLAR_SYSTEM_BARYCENTER"]
            ),
            frame="ecliptic",
        ),
    )

    results = ensure_input_origin_and_frame(orbits, results)
    assert results.coordinates.origin.code.to_pylist() == ["EARTH", "SUN", "MOON"]
    assert results.coordinates.frame == "equatorial"
