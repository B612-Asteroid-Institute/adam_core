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


def test_Times_from_jd():
    # Test that we can create a Times object correctly from julian dates and a time scale
    times_jd = np.linspace(2450000.5, 2460000.5, 1000, dtype=np.double)
    times = Times.from_jd(times_jd, "utc")

    assert times.scale == "utc"
    np.testing.assert_equal(times.jd().to_numpy(), times_jd)


def test_Times_from_mjd():
    # Test that we can create a Times object correctly from modified julian dates and a time scale
    times_mjd = np.linspace(50000.0, 60000.0, 1000, dtype=np.double)
    times = Times.from_mjd(times_mjd, "utc")

    # Surprised we can't do better here
    assert times.scale == "utc"
    np.testing.assert_allclose(times.mjd().to_numpy(), times_mjd, atol=0, rtol=1e-9)


def test_Times_unique():
    # Test that we can get unique times
    times_jd = np.array([2450000.5, 2450000.5, 2450001.5])
    times = Times.from_jd(times_jd, "utc")

    times_unique = times.unique()
    assert len(times_unique) == 2
    np.testing.assert_equal(times_unique.jd().to_numpy(), np.unique(times_jd))
