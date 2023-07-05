import numpy as np
import numpy.testing as npt

from ..kepler import (
    _calc_elliptical_anomalies,
    _calc_hyperbolic_anomalies,
    _calc_parabolic_anomalies,
    calc_mean_anomaly,
    solve_kepler,
)

RELATIVE_TOLERANCE = 0.0
ABSOLUTE_TOLERANCE = 1e-15

# --- Tests last updated: 2023-06-14


def test_calc_mean_anomaly_elliptical():
    # Test mean anomaly calculations for elliptical orbits
    # At the two apses, mean anomaly and true anomaly
    # should be equal.

    # Periapsis point
    nu = 0.0
    e = 0.5
    M_desired = nu
    M_actual = calc_mean_anomaly(nu, e)
    npt.assert_allclose(
        M_actual, M_desired, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE
    )

    # Apoapsis point
    nu = np.pi
    e = 0.5
    M_desired = nu
    M_actual = calc_mean_anomaly(nu, e)
    npt.assert_allclose(
        M_actual, M_desired, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE
    )

    # Periapsis point (at 360 degrees)
    nu = 2 * np.pi
    e = 0.5
    M_desired = 0.0
    M_actual = calc_mean_anomaly(nu, e)
    npt.assert_allclose(
        M_actual, M_desired, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE
    )


def test__calc_elliptical_anomalies():
    # Test eccentric anomaly and mean anomaly calculations
    # for elliptical orbits. At the two apses, mean anomaly,
    # eccentric anomaly and true anomaly should be equal.

    # Periapsis point
    nu = 0.0
    e = 0.5
    M_desired = nu
    E_desired = nu
    E_actual, M_actual = _calc_elliptical_anomalies(nu, e)
    npt.assert_allclose(
        (E_actual, M_actual),
        (E_desired, M_desired),
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    )

    # Apoapsis point
    nu = np.pi
    e = 0.5
    M_desired = nu
    E_desired = nu
    E_actual, M_actual = _calc_elliptical_anomalies(nu, e)
    npt.assert_allclose(
        (E_actual, M_actual),
        (E_desired, M_desired),
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    )

    # Periapsis point (at 360 degrees)
    nu = 2 * np.pi
    e = 0.5
    M_desired = 0.0
    E_desired = 0.0
    E_actual, M_actual = _calc_elliptical_anomalies(nu, e)
    npt.assert_allclose(
        (E_actual, M_actual),
        (E_desired, M_desired),
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    )


def test__calc_parabolic_anomalies():
    # Test eccentric anomaly and mean anomaly calculations
    # for parabolic orbits.  At the only valid apsis, mean anomaly,
    # eccentric anomaly and true anomaly should be equal.

    # Periapsis point
    nu = 0.0
    e = 1.0
    M_desired = nu
    E_desired = nu
    E_actual, M_actual = _calc_parabolic_anomalies(nu, e)
    npt.assert_allclose(
        (E_actual, M_actual),
        (E_desired, M_desired),
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    )


def test__calc_hyperbolic_anomalies():
    # Test eccentric anomaly and mean anomaly calculations
    # for hyperbolic orbits. At the only valid apsis, mean anomaly,
    # eccentric anomaly and true anomaly should be equal.

    # Periapsis point
    nu = 0.0
    e = 1.5
    M_desired = nu
    E_desired = nu
    E_actual, M_actual = _calc_hyperbolic_anomalies(nu, e)
    npt.assert_allclose(
        (E_actual, M_actual),
        (E_desired, M_desired),
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    )


def test_solve_kepler_elliptical():
    # Test true anomaly  calculations for elliptical orbits.
    # At the two apses, true anomaly and mean anomaly
    # should be equal.

    # Periapsis point
    e = 0.5
    M = 0.0
    nu_desired = 0.0
    nu_actual = solve_kepler(e, M)
    npt.assert_allclose(
        nu_actual, nu_desired, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE
    )

    # Apoapsis point
    e = 0.5
    M = np.pi
    nu_desired = np.pi
    nu_actual = solve_kepler(e, M)
    npt.assert_allclose(
        nu_actual, nu_desired, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE
    )

    # Periapsis point (at 360 degrees)
    e = 0.5
    M = 2 * np.pi
    nu_desired = 0.0
    nu_actual = solve_kepler(e, M)
    npt.assert_allclose(
        nu_actual, nu_desired, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE
    )


def test_solve_kepler_parabolic():
    # Test true anomaly calculations for parabolic orbits.
    # At the only valid apsis, true anomaly and mean anomaly
    # should be equal.

    # Periapsis point
    e = 1.0
    M = 0.0
    nu_desired = 0.0
    nu_actual = solve_kepler(e, M)
    npt.assert_allclose(
        nu_actual, nu_desired, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE
    )


def test_solve_kepler_hyperbolic():
    # Test true anomaly calculations for hyperbolic orbits.
    # At the only valid apsis, true anomaly and mean anomaly
    # should be equal.

    # Periapsis point
    e = 1.5
    M = 0.0
    nu_desired = 0.0
    nu_actual = solve_kepler(e, M)
    npt.assert_allclose(
        nu_actual, nu_desired, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE
    )


def test_solve_kepler_elliptical_orbits(orbital_elements):
    # Test true anomaly calculations for elliptical orbits.
    # Limit orbits to those with eccentricity < 1.0
    orbital_elements = orbital_elements[orbital_elements["e"] < 1.0]
    nu_desired = orbital_elements["nu"]
    M = orbital_elements["M"].values
    e = orbital_elements["e"].values

    for e_i, M_i, nu_desired_i in zip(e, M, nu_desired):
        nu_actual = np.degrees(solve_kepler(e_i, np.radians(M_i)))
        npt.assert_allclose(nu_actual, nu_desired_i, rtol=0.0, atol=1e-12)


def test_solve_kepler_hyperbolic_orbits(orbital_elements):
    # Test true anomaly calculations for elliptical orbits.
    # Limit orbits to those with eccentricity > 1.0
    orbital_elements = orbital_elements[orbital_elements["e"] > 1.0]
    nu_desired = orbital_elements["nu"]
    M = orbital_elements["M"].values
    e = orbital_elements["e"].values

    for e_i, M_i, nu_desired_i in zip(e, M, nu_desired):
        nu_actual = np.degrees(solve_kepler(e_i, np.radians(M_i)))
        npt.assert_allclose(nu_actual, nu_desired_i, rtol=0.0, atol=1e-12)
