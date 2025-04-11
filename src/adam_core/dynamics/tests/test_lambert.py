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
    earth_departure = get_perturber_state(
        OriginCodes.EARTH, launch_date, frame="ecliptic", origin=OriginCodes.SUN
    )
    mars_arrival = get_perturber_state(
        OriginCodes.MARS_BARYCENTER,
        arrival_date,
        frame="ecliptic",
        origin=OriginCodes.SUN,
    )

    tof = pc.subtract(mars_arrival.time.mjd(), earth_departure.time.mjd())[0].as_py()
    v1, v2 = solve_lambert(
        earth_departure.r,
        mars_arrival.r,
        tof,
        mu=mu_sun,
        prograde=True,
        tol=1e-10,
        max_iter=100000,
    )
    c3 = calculate_c3(v1, earth_departure.v)
    c3_km2_s2 = c3 * KM_P_AU**2 / S_P_DAY**2

    reported_c3 = 14.49
    assert np.isclose(c3_km2_s2, reported_c3, rtol=1e-6)


def test_lambert_see_mars_examples():
    """
    Test the lambert solver for examples provided by SEE
    """
    # Unused values from Mike here
    # ra = [63.1079, 9.4392, 80.9863, 56.4845, 102.1723, 99.326, 133.1727, 122.2525]
    # dec = [6.3005, 23.3053, 16.7756, 45.2431, 18.4345, 50.0726, 30.4812, 35.2493]
    # arrival_dec = [
    #     32.5661,
    #     -2.8719,
    #     18.2404,
    #     -20.364,
    #     5.9451,
    #     -29.925,
    #     17.3844,
    #     -22.9037,
    # ]
    # transfer_angle = [
    #     224.0892,
    #     203,
    #     220.0415,
    #     143.7046,
    #     211.0862,
    #     150.5542,
    #     205.2572,
    #     171.4079,
    # ]


    departure_dates = Timestamp.from_iso8601(
        [
            "2020-08-16T09:28:32Z",
            "2020-07-30T00:00:00Z",
            "2022-09-18T06:08:21Z",
            "2022-09-09T19:26:20Z",
            "2024-10-04T03:25:24Z",
            "2024-10-12T04:14:09Z",
            "2026-11-02T05:07:04Z",
            "2026-11-13T17:22:23Z",
        ],
        scale="utc",
    )

    arrival_dates = Timestamp.from_iso8601(
        [
            "2021-09-22T14:54:29Z",
            "2021-02-18T00:00:00Z",
            "2023-10-08T09:59:35Z",
            "2023-04-02T15:16:41Z",
            "2025-09-09T00:45:07Z",
            "2025-05-17T14:47:19Z",
            "2027-09-10T06:32:34Z",
            "2027-07-28T11:53:47Z",
        ],
        scale="utc",
    )

    tof = pc.subtract(arrival_dates.mjd(), departure_dates.mjd())

    provided_c3 = [16.6274, 14.4561, 13.8492, 18.5724, 11.1263, 17.8031, 9.3115, 11.7556]
    provided_arrival_vinf = [3.6671, 2.56, 3.149, 3.5665, 2.4925, 4.2497, 2.5679, 3.1699]

    # These are the tof values provided by Mike,
    # but we get more precise values calculating them
    # from departure and arrival dates. Let's test with both and see
    # which produces better results
    transfer_time = [
        402.226,
        143.1797,  # This tof seems wrong, as Feb 18, 2021 to July 30, 2020 is > 143 days
        385.161,
        204.827,
        339.889,
        217.44,
        312.059,
        256.772,
    ]

    # Test that our tof and given are within 1e-3
    # assert np.testing.assert_allclose(tof.to_numpy(), transfer_time, atol=1e-2)

    # Test that our arrival vinf and c3 are close to the given values
    earth_departure_positions = get_perturber_state(
        OriginCodes.EARTH, departure_dates, frame="ecliptic", origin=OriginCodes.SUN
    )
    mars_arrival_positions = get_perturber_state(
        OriginCodes.MARS_BARYCENTER, arrival_dates, frame="ecliptic", origin=OriginCodes.SUN
    )

    mu_sun = OriginGravitationalParameters.SUN

    v1, v2 = solve_lambert(
        earth_departure_positions.r,
        mars_arrival_positions.r,
        tof,
        mu=mu_sun,
        prograde=True,
    )

    c3 = calculate_c3(v1, earth_departure_positions.v)
    c3_km2_s2 = c3 * KM_P_AU**2 / S_P_DAY**2

    np.testing.assert_allclose(c3_km2_s2, provided_c3, atol=1e-3)

    calculated_arrival_vinf = np.linalg.norm(v2 - mars_arrival_positions.v, axis=1)
    calculated_arrival_vinf_km_s = calculated_arrival_vinf * KM_P_AU / S_P_DAY
    np.testing.assert_allclose(calculated_arrival_vinf_km_s, provided_arrival_vinf, atol=1e-3)
