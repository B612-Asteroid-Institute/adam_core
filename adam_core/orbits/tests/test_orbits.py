from astropy.time import Time

from ...coordinates import CartesianCoordinates
from ...coordinates.origin import Origin
from ...coordinates.times import Times
from ...utils.helpers import orbits as orbits_helpers
from ..orbits import Orbits


def test_orbits__init__():
    coordinates = CartesianCoordinates.from_kwargs(
        x=[0.5],
        y=[0.5],
        z=[0.004],
        vx=[0.005],
        vy=[-0.005],
        vz=[0.0002],
        times=Times.from_astropy(Time([59000.0], scale="tdb", format="mjd")),
        origin=Origin.from_kwargs(code=["SUN"]),
        frame="ecliptic",
    )

    orbits = Orbits.from_kwargs(
        coordinates=coordinates, orbit_ids=["1"], object_ids=["Test Orbit"]
    )

    assert orbits.orbit_ids[0].as_py() == "1"
    assert orbits.object_ids[0].as_py() == "Test Orbit"
    assert orbits.coordinates.times.jd1.to_numpy()[0] == 2459000.0
    assert orbits.coordinates.times.jd2.to_numpy()[0] == 0.5
    assert orbits.coordinates.times.scale == "tdb"
    assert orbits.coordinates.x.to_numpy()[0] == 0.5
    assert orbits.coordinates.y.to_numpy()[0] == 0.5
    assert orbits.coordinates.z.to_numpy()[0] == 0.004
    assert orbits.coordinates.vx.to_numpy()[0] == 0.005
    assert orbits.coordinates.vy.to_numpy()[0] == -0.005
    assert orbits.coordinates.vz.to_numpy()[0] == 0.0002


def test_orbit_iteration():
    orbits = orbits_helpers.make_simple_orbits(num_orbits=10)
    for o in orbits:
        assert len(o) == 1
