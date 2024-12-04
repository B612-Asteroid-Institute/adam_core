import cProfile

import jax
import numpy as np
import pyarrow.compute as pc
import pytest
import quivr as qv
from astropy import units as u

from ...observers import Observers
from ...time import Timestamp
from ..ephemeris import generate_ephemeris_2body
from ..propagation import propagate_2body

OBJECT_IDS = [
    "594913 'Aylo'chaxnim (2020 AV2)",
    "163693 Atira (2003 CP20)",
    "(2010 TK7)",
    "3753 Cruithne (1986 TO)",
    "54509 YORP (2000 PH5)",
    "2063 Bacchus (1977 HB)",
    "1221 Amor (1932 EA1)",
    "433 Eros (A898 PA)",
    "3908 Nyx (1980 PA)",
    "434 Hungaria (A898 RB)",
    "1876 Napolitania (1970 BA)",
    "2001 Einstein (1973 EB)",
    "2 Pallas (A802 FA)",
    "6 Hebe (A847 NA)",
    "6522 Aci (1991 NQ)",
    "10297 Lynnejones (1988 RJ13)",
    "17032 Edlu (1999 FM9)",
    "202930 Ivezic (1998 SG172)",
    "911 Agamemnon (A919 FB)",
    "1143 Odysseus (1930 BH)",
    "1172 Aneas (1930 UA)",
    "3317 Paris (1984 KF)",
    "5145 Pholus (1992 AD)",
    "5335 Damocles (1991 DA)",
    "15760 Albion (1992 QB1)",
    "15788 (1993 SB)",
    "15789 (1993 SC)",
    "1I/'Oumuamua (A/2017 U1)",
]

TOLERANCES = {
    # range (m), angular difference (mas), light_time difference (microseconds)
    # Default range is about consistent with our ability to get
    # observatory state vector positions compared to Horizons
    "default": (20, 0.1, 10),
    # 594913 'Aylo'chaxnim (2020 AV2) : Vatira
    # Range difference for first 30 observations between
    # 32.8 m and 80.4 m, for last 60 observations range is
    # between 0.13 m and 26.3 m
    "594913 'Aylo'chaxnim (2020 AV2)": (100, 1, 10),
    # 3753 Cruithne (1986 TO) : NEO (ATEN)
    # Range anywhere between 0.01 m and 40.5 m throughout
    # the 90 observations
    "3753 Cruithne (1986 TO)": (50, 1, 10),
    # 6522 Aci (1991 NQ) : Inner Main Belt asteroid
    # 89/90 RA, Decs have a mean offset of 0.007 mas
    # 1/90 RA, Decs has an offset of 1.24 mas
    # 88/90 ranges have a mean offset of 10.6 m
    # 2/90 ranges are 89.5, 174.3 m
    "6522 Aci (1991 NQ)": (200, 2, 10),
    # 1143 Odysseus (1930 BH) : Jupiter Trojan
    # 89/90 light times have a mean offset of 0.14 microseconds
    # 1/90 light times has an offset of 38.4 microseconds
    # 89/90 RA, Decs have a mean offset of 0.0056 mas
    # 1/90 RA, Decs has an offset of 8.9 mas
    # 89/90 ranges have a mean offset of 6.9 m
    # 1/90 ranges has an offset of 11575 m? ...
    "1143 Odysseus (1930 BH)": (11600, 10, 50),
}


@pytest.mark.parametrize("object_id", OBJECT_IDS)
def test_generate_ephemeris_2body(object_id, propagated_orbits, ephemeris):
    # For our catalog of test orbits, generate ephemeris using Horizons generated state vectors
    # and compare the results to the Horizons generated ephemeris
    propagated_orbit = propagated_orbits.select("object_id", object_id)
    ephemeris_orbit = ephemeris[ephemeris["targetname"].isin([object_id])]

    observers_list = []
    for observatory_code in ephemeris_orbit["observatory_code"].unique():
        observatory_mask = ephemeris_orbit["observatory_code"] == observatory_code
        observer_i = Observers.from_code(
            observatory_code,
            Timestamp.from_mjd(
                ephemeris_orbit[observatory_mask]["mjd_utc"].values,
                scale="utc",
            ),
        )
        observers_list.append(observer_i)

    observers = qv.concatenate(observers_list)

    ephemeris_orbit_2body = generate_ephemeris_2body(propagated_orbit, observers)

    # Get the tolerances for this orbit
    if object_id in TOLERANCES:
        range_tolerance, angular_tolerance, light_time_tolerance = TOLERANCES[object_id]
    else:
        range_tolerance, angular_tolerance, light_time_tolerance = TOLERANCES["default"]

    # Calculate the offset in light travel time
    light_time_diff = np.abs(
        ephemeris_orbit_2body.light_time.to_numpy()
        - (ephemeris_orbit["lighttime"].values * 60 / 86400)
    ) * (u.d).to(u.microsecond)

    # Assert that the light time is less than the tolerance
    np.testing.assert_array_less(light_time_diff, light_time_tolerance)

    # Calculate the difference in RA and Dec
    angular_diff = np.abs(
        ephemeris_orbit_2body.coordinates.values[:, 1:3]
        - ephemeris_orbit[["RA", "DEC"]].values
    ) * (u.deg).to(u.milliarcsecond)

    # Topocentric difference
    range_diff = np.abs(
        ephemeris_orbit_2body.coordinates.values[:, 0] - ephemeris_orbit["delta"].values
    ) * (u.au).to(u.m)

    # Assert that the position is less than the tolerance
    np.testing.assert_array_less(angular_diff, angular_tolerance)

    # Assert that the range is less than the tolerance
    np.testing.assert_array_less(range_diff, range_tolerance)

    # Ephemeris 2-body does not propagate the covariance or provide the
    # aberrated state so the aberrated coordinates should be empty
    assert ephemeris_orbit_2body.aberrated_coordinates.frame == "unspecified"
    assert pc.all(pc.is_null(ephemeris_orbit_2body.aberrated_coordinates.x)).as_py()
    assert pc.all(pc.is_null(ephemeris_orbit_2body.aberrated_coordinates.y)).as_py()
    assert pc.all(pc.is_null(ephemeris_orbit_2body.aberrated_coordinates.z)).as_py()
    assert pc.all(pc.is_null(ephemeris_orbit_2body.aberrated_coordinates.vx)).as_py()
    assert pc.all(pc.is_null(ephemeris_orbit_2body.aberrated_coordinates.vy)).as_py()
    assert pc.all(pc.is_null(ephemeris_orbit_2body.aberrated_coordinates.vz)).as_py()


@pytest.mark.profile
def test_profile_generate_ephemeris_2body_matrix(propagated_orbits, tmp_path):
    """Profile the generate_ephemeris_2body function with different combinations of orbits,
    observers and times. Results are saved to a stats file that can be visualized with snakeviz.
    """
    # Clear the jax cache
    jax.clear_caches()
    # Create profiler
    profiler = cProfile.Profile(subcalls=True, builtins=True)
    profiler.bias = 0

    n_entries = [1, 10, 100, 1000]

    # create 1000 times, observers, and propagate orbits to those times
    times = Timestamp.from_mjd(
        np.arange(60000, 60000 + 1000, 1),
        scale="tdb",
    )
    observers = Observers.from_code(
        "X05",
        times=times,
    )
    propagated_orbits = propagate_2body(
        propagated_orbits[0],
        times,
    )

    def to_profile():
        for n_entries_i in n_entries:
            generate_ephemeris_2body(
                propagated_orbits[:n_entries_i],
                observers[:n_entries_i],
            )

    # Run profiling
    profiler.enable()
    to_profile()
    profiler.disable()

    # Save and print results
    stats_file = tmp_path / "ephemeris_profile.prof"
    profiler.dump_stats(stats_file)
    print(f"Run 'snakeviz {stats_file}' to view the profile results.")
