import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from ...time import Timestamp
from ..observers import OBSERVATORY_PARALLAX_COEFFICIENTS, Observers


@pytest.fixture
def codes_times() -> tuple[pa.Array, Timestamp]:
    codes = pa.array(
        ["500", "X05", "I41", "X05", "I41", "W84", "500"],
    )

    times = Timestamp.from_kwargs(
        days=[59000, 59001, 59002, 59003, 59004, 59005, 59006],
        nanos=[0, 0, 0, 0, 0, 0, 0],
        scale="tdb",
    )
    return codes, times


def test_Observers_from_codes(codes_times) -> None:
    # Test that observers from code returns the correct number of observers
    # and in the order that they were requested
    codes, times = codes_times

    observers = Observers.from_codes(codes, times)
    assert len(observers) == 7
    assert pc.all(pc.equal(observers.code, codes)).as_py()
    assert pc.all(pc.equal(observers.coordinates.time.days, times.days)).as_py()
    assert pc.all(pc.equal(observers.coordinates.time.nanos, times.nanos)).as_py()


def test_Observers_from_codes_non_pyarrow(codes_times) -> None:
    # Test that observers from code returns the correct number of observers
    # and in the order that they were requested
    codes, times = codes_times

    observers = Observers.from_codes(codes.to_numpy(zero_copy_only=False), times)
    assert len(observers) == 7
    assert pc.all(pc.equal(observers.code, codes)).as_py()
    assert pc.all(pc.equal(observers.coordinates.time.days, times.days)).as_py()
    assert pc.all(pc.equal(observers.coordinates.time.nanos, times.nanos)).as_py()

    observers = Observers.from_codes(codes.to_pylist(), times)
    assert len(observers) == 7
    assert pc.all(pc.equal(observers.code, codes)).as_py()
    assert pc.all(pc.equal(observers.coordinates.time.days, times.days)).as_py()
    assert pc.all(pc.equal(observers.coordinates.time.nanos, times.nanos)).as_py()


def test_Observers_from_codes_raises(codes_times) -> None:
    # Test that observers from code raises an error if the codes and times
    # are not the same length
    codes, times = codes_times

    with pytest.raises(ValueError, match="codes and times must have the same length."):
        Observers.from_codes(codes[:3], times)
    with pytest.raises(ValueError, match="codes and times must have the same length."):
        Observers.from_codes(codes, times[:3])


def test_ObservatoryParallaxCoefficients_lon_lat() -> None:
    # Data taken from: https://en.wikipedia.org/wiki/List_of_observatory_codes
    # From: https://geohack.toolforge.org/geohack.php?pagename=Zwicky_Transient_Facility&params=33.35731_N_116.85981_W_
    I41 = OBSERVATORY_PARALLAX_COEFFICIENTS.select("code", "I41")
    lon, lat = I41.lon_lat()
    np.testing.assert_allclose(lon[0], -116.85981, atol=1e-4, rtol=0)
    np.testing.assert_allclose(lat[0], 33.35731, atol=1e-4, rtol=0)

    # From: https://geohack.toolforge.org/geohack.php?pagename=Vera_C._Rubin_Observatory&params=30_14_40.7_S_70_44_57.9_W_region:CL-CO_type:landmark
    X05 = OBSERVATORY_PARALLAX_COEFFICIENTS.select("code", "X05")
    lon, lat = X05.lon_lat()
    np.testing.assert_allclose(lon[0], -70.749417, atol=1e-4, rtol=0)
    np.testing.assert_allclose(lat[0], -30.244639, atol=1e-4, rtol=0)

    # From: https://geohack.toolforge.org/geohack.php?pagename=V%C3%ADctor_M._Blanco_Telescope&params=30.16967_S_70.80653_W_
    W84 = OBSERVATORY_PARALLAX_COEFFICIENTS.select("code", "W84")
    lon, lat = W84.lon_lat()
    np.testing.assert_allclose(lon[0], -70.80653, atol=1e-3, rtol=0)
    np.testing.assert_allclose(lat[0], -30.169679, atol=1e-3, rtol=0)

    # From: https://geohack.toolforge.org/geohack.php?params=32_22_48.6_S_20_48_38.1_E
    M22 = OBSERVATORY_PARALLAX_COEFFICIENTS.select("code", "M22")
    lon, lat = M22.lon_lat()
    np.testing.assert_allclose(lon[0], 20.810583, atol=1e-4, rtol=0)
    np.testing.assert_allclose(lat[0], -32.380167, atol=1e-4, rtol=0)

    # From: https://geohack.toolforge.org/geohack.php?params=20_42_26.04_N_156_15_21.28_W
    F51 = OBSERVATORY_PARALLAX_COEFFICIENTS.select("code", "F51")
    lon, lat = F51.lon_lat()
    np.testing.assert_allclose(lon[0], -156.255911, atol=1e-4, rtol=0)
    np.testing.assert_allclose(lat[0], 20.707233, atol=1e-4, rtol=0)


def test_ObservatoryParallaxCoeffiecients_timezone() -> None:
    I41 = OBSERVATORY_PARALLAX_COEFFICIENTS.select("code", "I41")
    assert I41.timezone() == "America/Los_Angeles"

    X05 = OBSERVATORY_PARALLAX_COEFFICIENTS.select("code", "X05")
    assert X05.timezone() == "America/Santiago"

    W84 = OBSERVATORY_PARALLAX_COEFFICIENTS.select("code", "W84")
    assert W84.timezone() == "America/Santiago"

    M22 = OBSERVATORY_PARALLAX_COEFFICIENTS.select("code", "M22")
    assert M22.timezone() == "Africa/Johannesburg"

    F51 = OBSERVATORY_PARALLAX_COEFFICIENTS.select("code", "F51")
    assert F51.timezone() == "Pacific/Honolulu"
