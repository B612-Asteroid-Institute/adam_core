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
