from astropy.time import Time

from ...coordinates import KeplerianCoordinates
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
