from ...coordinates import CartesianCoordinates
from ...coordinates.origin import Origin
from ...time import Timestamp
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
        time=Timestamp.from_mjd([59000.0], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN"]),
        frame="ecliptic",
    )

    orbits = Orbits.from_kwargs(
        coordinates=coordinates, orbit_id=["1"], object_id=["Test Orbit"]
    )

    assert orbits.orbit_id[0].as_py() == "1"
    assert orbits.object_id[0].as_py() == "Test Orbit"
    assert orbits.coordinates.time.days.to_numpy()[0] == 59000
    assert orbits.coordinates.time.nanos.to_numpy()[0] == 0
    assert orbits.coordinates.time.scale == "tdb"
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
