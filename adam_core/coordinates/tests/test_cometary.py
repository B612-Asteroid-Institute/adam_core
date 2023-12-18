import numpy as np
import numpy.testing as npt
import pytest

from ...time import Timestamp
from ..cometary import CometaryCoordinates
from ..origin import Origin


@pytest.fixture
def cometary_elements(orbital_elements):
    return CometaryCoordinates.from_kwargs(
        q=orbital_elements["q"],
        e=orbital_elements["e"],
        i=orbital_elements["incl"],
        raan=orbital_elements["Omega"],
        ap=orbital_elements["w"],
        tp=orbital_elements["tp_mjd"],
        time=Timestamp.from_mjd(orbital_elements["mjd_tdb"].values, scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN"] * len(orbital_elements)),
        frame="ecliptic",
    )


def test_CometaryCoordinates_period(cometary_elements, orbital_elements):
    # Test that the period calculated from Cometary elements matches the
    # period in the test data.
    P_desired = orbital_elements["P"]
    # For parabolic/hyperbolic orbits, orbital period is infinite
    # Our test data from JPL Horizons uses a value of 1e99 for infinite
    P_desired = np.where(P_desired > 1e99, np.inf, P_desired)
    P_actual = cometary_elements.P
    npt.assert_allclose(P_actual, P_desired, rtol=0.0, atol=1e-10)


def test_CometaryCoordinates_period_setter(cometary_elements):
    # Assert that we cannot set the period
    with pytest.raises(ValueError, match="Cannot set period (P)*"):
        cometary_elements.P = np.full(len(cometary_elements), np.nan)


def test_CometaryCoordinates_period_deleter(cometary_elements):
    # Assert that we cannot delete the period
    with pytest.raises(ValueError, match="Cannot delete period (P)*"):
        del cometary_elements.P


def test_CometaryCoordinates_apoapsis_distance(cometary_elements, orbital_elements):
    # Test that the apoapsis distance calculated from Cometary elements
    # matches the apoapsis distance in the test data.
    Q_desired = orbital_elements["Q"]
    # For parabolic/hyperbolic orbits, apoapsis distance is infinite
    # Our test data from JPL Horizons uses a value of 1e99 for infinite
    Q_desired = np.where(Q_desired > 1e99, np.inf, Q_desired)
    Q_actual = cometary_elements.Q
    npt.assert_allclose(Q_actual, Q_desired, rtol=0.0, atol=1e-12)


def test_CometaryCoordinates_apoapsis_distance_setter(cometary_elements):
    # Assert that we cannot set the apoapsis distance
    with pytest.raises(ValueError, match="Cannot set apoapsis distance (Q)*"):
        cometary_elements.Q = np.full(len(cometary_elements), np.nan)


def test_CometaryCoordinates_apoapsis_distance_deleter(cometary_elements):
    # Assert that we cannot delete the apoapsis distance
    with pytest.raises(ValueError, match="Cannot delete apoapsis distance (Q)*"):
        del cometary_elements.Q


def test_CometaryCoordinates_semi_major_axis(cometary_elements, orbital_elements):
    # Test that the semi-major axis calculated from Cometary elements
    # matches the semi-major axis in the test data.
    a_desired = orbital_elements["a"]
    a_actual = cometary_elements.a
    npt.assert_allclose(a_actual, a_desired, rtol=0.0, atol=1e-12)


def test_CometaryCoordinates_semi_major_axis_setter(cometary_elements):
    # Assert that we cannot set the semi-major axis
    with pytest.raises(ValueError, match="Cannot set semi-major axis (a)*"):
        cometary_elements.a = np.full(len(cometary_elements), np.nan)


def test_CometaryCoordinates_semi_major_axis_deleter(cometary_elements):
    # Assert that we cannot delete the semi-major axis
    with pytest.raises(ValueError, match="Cannot delete semi-major axis (a)*"):
        del cometary_elements.a


def test_CometaryCoordinates_semi_latus_rectum(cometary_elements, orbital_elements):
    # Test that the semi-latus rectum calculated from Cometary elements
    # matches the semi-latus rectum in the test data.
    a = orbital_elements["a"].values
    e = orbital_elements["e"].values
    p_desired = (1 - e**2) * a
    p_actual = cometary_elements.p
    npt.assert_allclose(p_actual, p_desired, rtol=0.0, atol=1e-12)


def test_CometaryCoordinates_semi_latus_rectum_setter(cometary_elements):
    # Assert that we cannot set the semi-latus rectum
    with pytest.raises(ValueError, match="Cannot set semi-latus rectum (p)*"):
        cometary_elements.p = np.full(len(cometary_elements), np.nan)


def test_CometaryCoordinates_semi_latus_rectum_deleter(cometary_elements):
    # Assert that we cannot delete the semi-latus rectum
    with pytest.raises(ValueError, match="Cannot delete semi-latus rectum (p)*"):
        del cometary_elements.p


def test_CometaryCoordinates_mean_motion(cometary_elements, orbital_elements):
    # Test that the mean motion calculated from Cometary elements
    # matches the mean motion in the test data.
    n_desired = orbital_elements["n"]
    n_actual = cometary_elements.n
    npt.assert_allclose(n_actual, n_desired, rtol=0.0, atol=1e-12)


def test_CometaryCoordinates_mean_motion_setter(cometary_elements):
    # Assert that we cannot set the mean motion
    with pytest.raises(ValueError, match="Cannot set mean motion (n)*"):
        cometary_elements.n = np.full(len(cometary_elements), np.nan)


def test_CometaryCoordinates_mean_motion_deleter(cometary_elements):
    # Assert that we cannot delete the mean motion
    with pytest.raises(ValueError, match="Cannot delete mean motion (n)*"):
        del cometary_elements.n
