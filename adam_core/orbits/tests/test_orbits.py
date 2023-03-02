from astropy.time import Time

from ...coordinates import KeplerianCoordinates
from ...utils.helpers import orbits as orbits_helpers
from ..orbits import Orbits


def test_orbits__init__():
    keplerian_elements = KeplerianCoordinates(
        times=Time(59000.0, scale="tdb", format="mjd"),
        a=1.0,
        e=0.002,
        i=10.0,
        raan=50.0,
        ap=20.0,
        M=30.0,
    )

    orbits = Orbits(keplerian_elements, orbit_ids=["1"], object_ids=["Test Orbit"])

    assert orbits.orbit_ids[0] == "1"
    assert orbits.object_ids[0] == "Test Orbit"
    assert orbits.keplerian.times[0].mjd == 59000.0
    assert orbits.keplerian.times[0].scale == "tdb"
    assert orbits.keplerian.a[0] == 1.0
    assert orbits.keplerian.e[0] == 0.002
    assert orbits.keplerian.i[0] == 10.0
    assert orbits.keplerian.raan[0] == 50.0
    assert orbits.keplerian.ap[0] == 20.0
    assert orbits.keplerian.M[0] == 30.0


def test_orbits__eq__():
    keplerian_elements1 = KeplerianCoordinates(
        times=Time(59000.0, scale="tdb", format="mjd"),
        a=1.0,
        e=0.002,
        i=10.0,
        raan=50.0,
        ap=20.0,
        M=30.0,
    )
    keplerian_elements2 = KeplerianCoordinates(
        times=Time(59000.0, scale="tdb", format="mjd"),
        a=1.0,
        e=0.002,
        i=10.0,
        raan=50.0,
        ap=20.0,
        M=30.0,
    )

    orbits1 = Orbits(keplerian_elements1, orbit_ids=["1"], object_ids=["Test Orbit"])
    orbits2 = Orbits(keplerian_elements2, orbit_ids=["1"], object_ids=["Test Orbit"])

    assert orbits1 == orbits2

    keplerian_elements3 = KeplerianCoordinates(
        times=Time(59000.0, scale="tdb", format="mjd"),
        a=0.0,
        e=0.002,
        i=10.0,
        raan=50.0,
        ap=20.0,
        M=30.0,
    )

    orbits3 = Orbits(keplerian_elements3, orbit_ids=["1"], object_ids=["Test Orbit"])
    assert orbits1 != orbits3


def test_orbit_iteration():
    orbits = orbits_helpers.make_simple_orbits(num_orbits=10)
    for o in orbits:
        assert len(o) == 1
