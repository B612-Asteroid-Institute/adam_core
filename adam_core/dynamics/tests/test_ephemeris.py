import numpy as np
from astropy import units as u
from astropy.time import Time

from ...observers import Observers
from ..ephemeris import generate_ephemeris_2body


def test_generate_ephemeris_2body(propagated_orbits, ephemeris):
    # For our catalog of test orbits, generate ephemeris using Horizons generated state vectors
    # and compare the results to the Horizons generated ephemeris
    orbit_ids = propagated_orbits.orbit_id.unique().to_numpy(zero_copy_only=False)
    for orbit_id in orbit_ids:

        propagated_orbit = propagated_orbits.select("orbit_id", orbit_id)
        ephemeris_orbit = ephemeris[ephemeris["orbit_id"].isin([orbit_id])]

        observers = Observers.from_code(
            "X05",
            Time(ephemeris_orbit["mjd_utc"].values, format="mjd", scale="utc"),
        )

        ephemeris_orbit_2body = generate_ephemeris_2body(propagated_orbit, observers)

        # Extract only the ephemeris table
        ephemeris_orbit_2body = ephemeris_orbit_2body.left_table

        # Calculate the offset in light travel time
        light_time_diff = np.abs(
            ephemeris_orbit_2body.light_time.to_numpy()
            - (ephemeris_orbit["lighttime"].values * 60 / 86400)
        ) * (u.d).to(u.microsecond)

        # Assert that the light time is less than 100 microseconds
        np.testing.assert_array_less(light_time_diff, 100)

        # Calculate the difference in RA and Dec
        angular_diff = np.abs(
            ephemeris_orbit_2body.coordinates.values[:, 1:3]
            - ephemeris_orbit[["RA", "DEC"]].values
        ) * (u.deg).to(u.milliarcsecond)

        # Topocentric difference
        range_diff = np.abs(
            ephemeris_orbit_2body.coordinates.values[:, 0]
            - ephemeris_orbit["delta"].values
        ) * (u.au).to(u.m)

        if orbit_id == "00014":
            # Assert that the position is less than 50 milliarcsecond
            # Orbit 14 is 6522 Aci (1991 NQ) an inner main-belt asteroid
            np.testing.assert_array_less(angular_diff, 50)

            # Assert that the range is less than 5000 m
            np.testing.assert_array_less(range_diff, 5000)
        elif orbit_id == "00019":
            # Assert that the position is less than 10 milliarcsecond
            # Orbit 19 is 911 Agamemnon (A919 FB), a Jupiter Trojan, which is interesting
            # since Orbit 20 is also a Jupiter Trojan.
            np.testing.assert_array_less(angular_diff, 10)

            # Assert that the range is less than 12000 m
            np.testing.assert_array_less(range_diff, 12000)
        else:
            # Assert that the position is less than 1 milliarcsecond
            np.testing.assert_array_less(angular_diff, 1)

            # Assert that the range is less than 200 m
            np.testing.assert_array_less(range_diff, 200)
