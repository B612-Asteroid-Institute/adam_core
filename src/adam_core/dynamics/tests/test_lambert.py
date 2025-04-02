import numpy as np
import numpy.testing as npt
import pyarrow.compute as pc

from adam_core.constants import KM_P_AU, S_P_DAY
from adam_core.coordinates.origin import OriginCodes, OriginGravitationalParameters
from adam_core.dynamics.lambert import calculate_c3, solve_lambert
from adam_core.orbits.query.horizons import query_horizons
from adam_core.time import Timestamp
from adam_core.utils.spice import get_perturber_state

# Test cases from Vallado's "Fundamentals of Astrodynamics and Applications"
# and other well-known problems

def test_lambert_solver_lamberthub():

    # Initial conditions
    mu_sun = OriginGravitationalParameters.SUN
    r1 = np.array([0.159321004, 0.579266185, 0.052359607])  # [AU]
    r2 = np.array([0.057594337, 0.605750797, 0.068345246])  # [AU]
    tof = 0.010794065 * 365.25  # Convert years to days

    # Solving the problem
    v1, v2 = solve_lambert(r1, r2, tof, mu=mu_sun, tol=1e-10, max_iter=100000)

    # Expected final results (in AU/year)
    expected_v1 = np.array([-9.303603251, 3.018641330, 1.536362143])
    # Convert expected velocities from AU/year to AU/day
    expected_v1 = expected_v1 / 365.25

    # Assert the results
    np.testing.assert_allclose(v1, expected_v1, atol=1e-6, rtol=1e-6)


def test_lambert_mars_2020():
    """
    Test the Lambert solver for the Mars 2020 mission.

    Values from https://www.ulalaunch.com/docs/default-source/launch-booklets/mars2020_mobrochure_200717.pdf

    C3: 14.49 km2/s2
    Launch date July 30, 2020
    Mars arrival Feb. 18, 2021
    """
    mu_sun = OriginGravitationalParameters.SUN
    launch_date = Timestamp.from_iso8601(["2020-07-30T00:00:00"], scale="utc")
    arrival_date = Timestamp.from_iso8601(["2021-02-18T00:00:00"], scale="utc")
    earth_departure = get_perturber_state(OriginCodes.EARTH, launch_date, frame="ecliptic", origin=OriginCodes.SUN)
    mars_arrival = get_perturber_state(OriginCodes.MARS_BARYCENTER, arrival_date, frame="ecliptic", origin=OriginCodes.SUN)

    tof = pc.subtract(mars_arrival.time.mjd(), earth_departure.time.mjd())[0].as_py()
    v1, v2 = solve_lambert(earth_departure.r, mars_arrival.r, tof, mu=mu_sun, prograde=True, tol=1e-10, max_iter=100000)
    c3 = calculate_c3(v1, earth_departure.v)
    c3_km2_s2 = c3 * KM_P_AU**2 / S_P_DAY**2
    
    reported_c3 = 14.49
    assert np.isclose(c3_km2_s2, reported_c3, rtol=1e-6)
    

