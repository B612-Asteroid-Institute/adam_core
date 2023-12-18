import numpy as np
import numpy.testing as npt
import pytest

from ...time import Timestamp
from ..keplerian import KeplerianCoordinates
from ..origin import Origin


@pytest.fixture
def keplerian_elements(orbital_elements):
    return KeplerianCoordinates.from_kwargs(
        a=orbital_elements["a"],
        e=orbital_elements["e"],
        i=orbital_elements["incl"],
        raan=orbital_elements["Omega"],
        ap=orbital_elements["w"],
        M=orbital_elements["M"],
        time=Timestamp.from_mjd(orbital_elements["mjd_tdb"].values, scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN"] * len(orbital_elements)),
        frame="ecliptic",
    )


def test_KeplerianCoordinates_period(keplerian_elements, orbital_elements):
    # Test that the period calculated from Keplerian elements matches the
    # period in the test data.
    P_desired = orbital_elements["P"]
    # For parabolic/hyperbolic orbits, orbital period is infinite
    # Our test data from JPL Horizons uses a value of 1e99 for infinite
    P_desired = np.where(P_desired > 1e99, np.inf, P_desired)
    P_actual = keplerian_elements.P
    npt.assert_allclose(P_actual, P_desired, rtol=0.0, atol=1e-10)


def test_KeplerianCoordinates_period_setter(keplerian_elements):
    # Assert that we cannot set the period
    with pytest.raises(ValueError, match="Cannot set period (P)*"):
        keplerian_elements.P = np.full(len(keplerian_elements), np.nan)


def test_KeplerianCoordinates_period_deleter(keplerian_elements):
    # Assert that we cannot delete the period
    with pytest.raises(ValueError, match="Cannot delete period (P)*"):
        del keplerian_elements.P


def test_KeplerianCoordinates_perapsis_distance(keplerian_elements, orbital_elements):
    # Test that the periapsis distance calculated from Keplerian elements
    # matches the periapsis distance in the test data.
    q_desired = orbital_elements["q"]
    q_actual = keplerian_elements.q
    npt.assert_allclose(q_actual, q_desired, rtol=0.0, atol=1e-12)


def test_KeplerianCoordinates_perapsis_distance_setter(keplerian_elements):
    # Assert that we cannot set the periapsis distance
    with pytest.raises(ValueError, match="Cannot set periapsis distance (q)*"):
        keplerian_elements.q = np.full(len(keplerian_elements), np.nan)


def test_KeplerianCoordinates_perapsis_distance_deleter(keplerian_elements):
    # Assert that we cannot delete the periapsis distance
    with pytest.raises(ValueError, match="Cannot delete periapsis distance (q)*"):
        del keplerian_elements.q


def test_KeplerianCoordinates_apoapsis_distance(keplerian_elements, orbital_elements):
    # Test that the apoapsis distance calculated from Keplerian elements
    # matches the apoapsis distance in the test data.
    Q_desired = orbital_elements["Q"]
    # For parabolic/hyperbolic orbits, apoapsis distance is infinite
    # Our test data from JPL Horizons uses a value of 1e99 for infinite
    Q_desired = np.where(Q_desired > 1e99, np.inf, Q_desired)
    Q_actual = keplerian_elements.Q
    npt.assert_allclose(Q_actual, Q_desired, rtol=0.0, atol=1e-12)


def test_KeplerianCoordinates_apoapsis_distance_setter(keplerian_elements):
    # Assert that we cannot set the apoapsis distance
    with pytest.raises(ValueError, match="Cannot set apoapsis distance (Q)*"):
        keplerian_elements.Q = np.full(len(keplerian_elements), np.nan)


def test_KeplerianCoordinates_apoapsis_distance_deleter(keplerian_elements):
    # Assert that we cannot delete the apoapsis distance
    with pytest.raises(ValueError, match="Cannot delete apoapsis distance (Q)*"):
        del keplerian_elements.Q


def test_KeplerianCoordinates_semi_latus_rectum(keplerian_elements, orbital_elements):
    # Test that the semi-latus rectum calculated from Keplerian elements
    # matches the semi-latus rectum in the test data.
    a = orbital_elements["a"].values
    e = orbital_elements["e"].values
    p_desired = (1 - e**2) * a
    p_actual = keplerian_elements.p
    npt.assert_allclose(p_actual, p_desired, rtol=0.0, atol=1e-12)


def test_KeplerianCoordinates_semi_latus_rectum_setter(keplerian_elements):
    # Assert that we cannot set the semi-latus rectum
    with pytest.raises(ValueError, match="Cannot set semi-latus rectum (p)*"):
        keplerian_elements.p = np.full(len(keplerian_elements), np.nan)


def test_KeplerianCoordinates_semi_latus_rectum_deleter(keplerian_elements):
    # Assert that we cannot delete the semi-latus rectum
    with pytest.raises(ValueError, match="Cannot delete semi-latus rectum (p)*"):
        del keplerian_elements.p


def test_KeplerianCoordinates_mean_motion(keplerian_elements, orbital_elements):
    # Test that the mean motion calculated from Keplerian elements
    # matches the mean motion in the test data.
    n_desired = orbital_elements["n"]
    n_actual = keplerian_elements.n
    npt.assert_allclose(n_actual, n_desired, rtol=0.0, atol=1e-12)


def test_KeplerianCoordinates_mean_motion_setter(keplerian_elements):
    # Assert that we cannot set the mean motion
    with pytest.raises(ValueError, match="Cannot set mean motion (n)*"):
        keplerian_elements.n = np.full(len(keplerian_elements), np.nan)


def test_KeplerianCoordinates_mean_motion_deleter(keplerian_elements):
    # Assert that we cannot delete the mean motion
    with pytest.raises(ValueError, match="Cannot delete mean motion (n)*"):
        del keplerian_elements.n
