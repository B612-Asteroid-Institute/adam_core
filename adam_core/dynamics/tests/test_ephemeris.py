import numpy as np
import quivr as qv
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

        observers_list = []
        for observatory_code in ephemeris_orbit["observatory_code"].unique():
            observatory_mask = ephemeris_orbit["observatory_code"] == observatory_code
            observer_i = Observers.from_code(
                observatory_code,
                Time(
                    ephemeris_orbit[observatory_mask]["mjd_utc"].values,
                    format="mjd",
                    scale="utc",
                ),
            )
            observers_list.append(observer_i)

        observers = qv.concatenate(observers_list)

        ephemeris_orbit_2body = generate_ephemeris_2body(propagated_orbit, observers)

        # Extract only the ephemeris table
        ephemeris_orbit_2body = ephemeris_orbit_2body.left_table

        # Calculate the offset in light travel time
        light_time_diff = np.abs(
            ephemeris_orbit_2body.light_time.to_numpy()
            - (ephemeris_orbit["lighttime"].values * 60 / 86400)
        ) * (u.d).to(u.microsecond)

        # Assert that the light time is less than 10 microseconds
        np.testing.assert_array_less(light_time_diff, 10)

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

        # Orbit 00014 is 6522 Aci (1991 NQ) an inner main-belt asteroid
        # with a diameter of about 5 km. What's interesting here is that of the 90
        # ephemeris positions only 2 show a range difference of more than 15 m
        # (many are less than 10 m). Their values are 92.17 and 177.24 m.
        # These are the same observations that have offsets in RA and Dec greater
        # than 1 milliarcsecond. The maximum offset for those observations not
        # corresponding to the ones with a discrepant range offset is
        # only 0.013 milliarcseconds!
        if orbit_id == "00014":
            # Assert that the position is less than 2 milliarcsecond
            np.testing.assert_array_less(angular_diff, 2)

            # Assert that the range is less than 200 m
            np.testing.assert_array_less(range_diff, 200)

        # Orbit 00019 is 1143 Odysseus (1930 BH), a Jupiter Trojan with a
        # diameter of about 115 km. What's interesting here is that of the 90
        # ephemeris positions only 2 show a range difference of more than 10 m.
        # Their values are 231.79 and 468.26 m.  These are the same observations
        # that have offsets in RA and Dec greater than 0.5 milliarcsecond.
        # However, the maximum offset for those observations not corresponding to the
        # ones with a discrepant range offset is only 0.004 milliarcseconds!
        elif orbit_id == "00019":
            # Assert that the position is less than 1 milliarcsecond
            np.testing.assert_array_less(angular_diff, 1)

            # Assert that the range is less than 500 m
            np.testing.assert_array_less(range_diff, 500)

        # Orbit 00000 is 594913 'Aylo'chaxnim (2020 AV2) (an orbit completely
        # interior to Venus' orbit). It would not surprise me that there are
        # some dynamics that we are not modeling fully for this orbit in terms
        # of light travel time and/or the ephemeris generation.
        elif orbit_id == "00000":
            # Assert that the position is less than 1 milliarcsecond
            np.testing.assert_array_less(angular_diff, 1)

            # Assert that the range is less than 200 m
            np.testing.assert_array_less(range_diff, 200)

        # These tolerance apply to the rest of the orbits (25/28) and
        # show excellent results. RA, DEC to within 0.1 milliarcsecond
        # and range to within 20 m.
        else:
            # Assert that the position is less than 0.1 milliarcsecond
            np.testing.assert_array_less(angular_diff, 0.1)

            # Assert that the range is less than 20 m
            np.testing.assert_array_less(range_diff, 20)
