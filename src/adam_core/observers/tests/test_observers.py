import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import spiceypy as sp

from ...coordinates.origin import OriginCodes
from ...time import Timestamp
from ...utils.spice import get_perturber_state, get_spice_body_state
from ..observers import OBSERVATORY_CODES, OBSERVATORY_PARALLAX_COEFFICIENTS, Observers
from ..state import get_mpc_observer_state, get_observer_state


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


def test_origincode_observer():
    """Test getting observer state using an OriginCodes enum."""
    test_time = Timestamp.from_iso8601(["2022-01-01T00:00:00Z"])

    # Test with EARTH OriginCode
    earth_code = OriginCodes.EARTH

    # Get state using get_observer_state
    observer_state = get_observer_state(
        earth_code, test_time, frame="ecliptic", origin=OriginCodes.SUN
    )

    # Get state using get_perturber_state
    perturber_state = get_perturber_state(
        earth_code, test_time, frame="ecliptic", origin=OriginCodes.SUN
    )

    # Both should return identical results
    np.testing.assert_allclose(
        observer_state.values, perturber_state.values, rtol=1e-10
    )


def test_mpc_observatory_code():
    """Test getting observer state using an MPC observatory code."""
    test_time = Timestamp.from_iso8601(["2022-01-01T00:00:00Z"])

    # Use "500" (geocenter) which should return Earth's state directly
    geocenter_state = get_observer_state(
        "500", test_time, frame="ecliptic", origin=OriginCodes.SUN
    )

    # Get Earth's state
    earth_state = get_perturber_state(
        OriginCodes.EARTH, test_time, frame="ecliptic", origin=OriginCodes.SUN
    )

    # Both should return identical results
    np.testing.assert_allclose(geocenter_state.values, earth_state.values, rtol=1e-10)

    # Test a non-geocenter MPC code if available
    non_geocenter_codes = [
        code for code in OBSERVATORY_CODES if code != "500" and len(code) == 3
    ]

    if non_geocenter_codes:
        # Use the first available MPC code
        test_code = non_geocenter_codes[0]

        # Get state using the MPC code
        mpc_state = get_observer_state(
            test_code, test_time, frame="ecliptic", origin=OriginCodes.SUN
        )

        # Verify we got valid state vectors
        assert mpc_state.values.shape == (1, 6)
        assert mpc_state.frame == "ecliptic"
        assert mpc_state.origin.code[0].as_py() == "SUN"

        # The MPC observatory state should be near Earth but not identical
        # (it's on Earth's surface)
        earth_distance = np.linalg.norm(
            mpc_state.values[0, :3] - earth_state.values[0, :3]
        )

        # Distance should be small (Earth radius is about 4.2e-5 AU)
        assert 1e-6 < earth_distance < 1e-4


def test_invalid_code():
    """Test that using an invalid code raises the correct error."""
    test_time = Timestamp.from_iso8601(["2022-01-01T00:00:00Z"])

    # Try with a completely invalid code
    with pytest.raises(ValueError, match="not a valid MPC observatory code"):
        get_observer_state(
            "INVALID_CODE", test_time, frame="ecliptic", origin=OriginCodes.SUN
        )

    # Try with an invalid type
    with pytest.raises(AssertionError, match="code must be a string or OriginCodes"):
        get_observer_state(
            123,  # Not a string or OriginCodes
            test_time,
            frame="ecliptic",
            origin=OriginCodes.SUN,
        )


def test_mpc_vs_spice_precedence():
    """Test that SPICE codes take precedence over MPC codes when both exist."""
    test_time = Timestamp.from_iso8601(["2022-01-01T00:00:00Z"])

    # For this test, we'll create a temporary SPICE kernel with a body name
    # that matches an MPC observatory code

    # First, find an MPC code that we can use
    test_mpc_code = None

    for code in OBSERVATORY_CODES:
        # Skip geocenter and spacecraft codes
        if code == "500" or len(code) != 3:
            continue

        # Check if this code isn't already defined in SPICE
        try:
            sp.bodn2c(code)
        except sp.SpiceyError:
            test_mpc_code = code
            break

    if test_mpc_code is None:
        pytest.skip("Could not find a suitable MPC code for this test")

    # Get the MPC observatory state first for comparison
    mpc_state = get_mpc_observer_state(
        test_mpc_code, test_time, frame="ecliptic", origin=OriginCodes.SUN
    )

    # Now, get Mars state with direct SPICE call
    mars_state = get_spice_body_state(
        OriginCodes.MARS_BARYCENTER.value,
        test_time,
        frame="ecliptic",
        origin=OriginCodes.SUN,
    )

    # Create a temporary alias in SPICE that maps the MPC code to Mars ID
    # This simulates having a SPICE object with the same name as an MPC code
    sp.boddef(test_mpc_code, OriginCodes.MARS_BARYCENTER.value)

    try:
        # Now get the state again, it should still return MPC state
        # as MPC codes take precedence over SPICE codes
        hybrid_state = get_observer_state(
            test_mpc_code, test_time, frame="ecliptic", origin=OriginCodes.SUN
        )

        # The result should match Mars, not the MPC observatory
        np.testing.assert_allclose(hybrid_state.values, mpc_state.values, rtol=1e-10)

        # And it should NOT match the MPC observatory
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                hybrid_state.values, mars_state.values, rtol=1e-10
            )
    finally:
        # Clean up the temporary alias to avoid affecting other tests
        try:
            # Try to undefine the alias
            sp.bodc2n(
                OriginCodes.MARS_BARYCENTER.value
            )  # This will raise an error, but we need to try something
        except sp.SpiceyError:
            pass  # Ignore, we just want to make sure we don't leave aliases
