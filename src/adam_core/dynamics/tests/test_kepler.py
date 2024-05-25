import numpy as np
import numpy.testing as npt

from ...coordinates.origin import Origin
from ..kepler import (
    _calc_elliptical_anomalies,
    _calc_hyperbolic_anomalies,
    _calc_parabolic_anomalies,
    calc_apoapsis_distance,
    calc_mean_anomaly,
    calc_mean_motion,
    calc_periapsis_distance,
    calc_period,
    calc_semi_latus_rectum,
    calc_semi_major_axis,
    solve_kepler,
)

RELATIVE_TOLERANCE = 0.0
ABSOLUTE_TOLERANCE = 1e-15

# --- Tests last updated: 2023-12-18


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


def test_calc_period(orbital_elements):
    # Test period calculations for elliptical orbits
    P_desired = orbital_elements["P"]
    # For parabolic/hyperbolic orbits, orbital period is infinite
    # Our test data from JPL Horizons uses a value of 1e99 for infinite
    P_desired = np.where(P_desired > 1e99, np.inf, P_desired)
    a = orbital_elements["a"].values
    origin = Origin.from_kwargs(
        code=np.full(len(orbital_elements), "SUN", dtype="object")
    )
    mu = origin.mu()

    P_actual = calc_period(a, mu)
    npt.assert_allclose(P_actual, P_desired, rtol=0.0, atol=1e-10)


def test_calc_periapsis_distance(orbital_elements):
    # Test periapse distance calculations
    q_desired = orbital_elements["q"]
    a = orbital_elements["a"].values
    e = orbital_elements["e"].values

    q_actual = calc_periapsis_distance(a, e)
    npt.assert_allclose(q_actual, q_desired, rtol=0.0, atol=1e-12)


def test_calc_apoapsis_distance(orbital_elements):
    # Test apoapsis distance calculations
    Q_desired = orbital_elements["Q"]
    # For parabolic/hyperbolic orbits, apoapsis distance is infinite
    # Our test data from JPL Horizons uses a value of 1e99 for infinite
    Q_desired = np.where(Q_desired > 1e99, np.inf, Q_desired)
    a = orbital_elements["a"].values
    e = orbital_elements["e"].values

    Q_actual = calc_apoapsis_distance(a, e)
    npt.assert_allclose(Q_actual, Q_desired, rtol=0.0, atol=1e-12)


def test_calc_semi_major_axis(orbital_elements):
    # Test semi-major axis calculations
    a_desired = orbital_elements["a"]
    q = orbital_elements["q"].values
    e = orbital_elements["e"].values

    a_actual = calc_semi_major_axis(q, e)
    npt.assert_allclose(a_actual, a_desired, rtol=0.0, atol=1e-12)


def test_calc_semi_latus_rectum(orbital_elements):
    # Test semi-latus rectum calculations
    a = orbital_elements["a"].values
    e = orbital_elements["e"].values
    p_desired = (1 - e**2) * a

    p_actual = calc_semi_latus_rectum(a, e)
    npt.assert_allclose(p_actual, p_desired, rtol=0.0, atol=1e-12)


def test_calc_mean_motion(orbital_elements):
    # Test mean motion calculations
    n_desired = orbital_elements["n"]
    a = orbital_elements["a"].values
    origin = Origin.from_kwargs(
        code=np.full(len(orbital_elements), "SUN", dtype="object")
    )
    mu = origin.mu()

    n_actual = np.degrees(calc_mean_motion(a, mu))
    npt.assert_allclose(n_actual, n_desired, rtol=0.0, atol=1e-12)
