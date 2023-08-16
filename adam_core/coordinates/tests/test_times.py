import numpy as np
import pytest
from astropy.time import Time

from ..times import Times


@pytest.mark.parametrize("scale", ["tdb", "utc", "tt"])
def test_Times_to_from_astropy_roundtrip(scale):
    # Test that we can round trip to and from astropy Time objects
    times_astropy = Time(
        np.linspace(50000.0, 60000.0, 1000),
        format="mjd",
        scale=scale,
    )
    times = Times.from_astropy(times_astropy)
    assert times.scale == scale

    times_astropy_2 = times.to_astropy()
    np.testing.assert_equal(times_astropy_2.mjd, times_astropy.mjd)


def test_Times_to_scale():
    # Test that we can convert between time scales correctly
    times_tdb = Time(np.arange(59000, 60000), scale="tdb", format="mjd")
    times_tdb = Times.from_astropy(times_tdb)
    assert times_tdb.scale == "tdb"

    times_utc = times_tdb.to_scale("utc")
    assert times_utc.scale == "utc"
    np.testing.assert_equal(times_utc.to_astropy().mjd, times_tdb.to_astropy().utc.mjd)
