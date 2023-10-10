from importlib.resources import files
from typing import List

import numpy as np
import pandas as pd
import pyarrow as pa
import quivr as qv

from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_core.orbits.query import query_horizons, query_horizons_ephemeris, query_sbdb
from adam_core.orbits.query.horizons import (
    _get_horizons_elements,
    _get_horizons_vectors,
)
from adam_core.time import Timestamp


def _get_orbital_elements(
    object_ids: List[str], time: Timestamp, location: str, refplane: str = "ecliptic"
) -> pd.DataFrame:
    """
    Get orbital elements as Cartesian, Cometary, and Keplerian representations from JPL Horizons.

    Parameters
    ----------
    object_id : List[str]
        Object IDs to
    epoch : Timestamp
        Epoch at which to query orbital elements.
    location : str
        Location of the observer (in this case typically "@sun" or "@ssb"
        for barycentric or heliocentric elements, respectively).

    Returns
    -------
    horizons_elements : `~pandas.DataFrame`
        DataFrame containing orbital elements in different representations for each object ID.
    """
    vectors_df = _get_horizons_vectors(
        object_ids,
        time,
        location=location,
        id_type="smallbody",
        aberrations="geometric",
        refplane=refplane,
    )
    vectors_df = vectors_df[
        ["targetname", "datetime_jd", "x", "y", "z", "vx", "vy", "vz"]
    ]

    elements_df = _get_horizons_elements(
        object_ids,
        time,
        location=location,
        id_type="smallbody",
        refplane=refplane,
    )
    elements_df = elements_df[
        [
            "targetname",
            "datetime_jd",
            "a",
            "q",
            "Q",
            "e",
            "incl",
            "Omega",
            "w",
            "M",
            "nu",
            "n",
            "Tp_jd",
            "P",
        ]
    ]

    horizons_elements = vectors_df.merge(elements_df, on=["targetname", "datetime_jd"])

    horizons_elements.insert(
        1, "mjd_tdb", Timestamp.from_jd(horizons_elements["datetime_jd"]).mjd()
    )
    horizons_elements.insert(
        len(horizons_elements.columns),
        "tp_mjd",
        Timestamp.from_jd(horizons_elements["Tp_jd"]).mjd(),
    )
    horizons_elements.drop(columns=["datetime_jd", "Tp_jd"], inplace=True)
    return horizons_elements


if __name__ == "__main__":
    # Read sample object IDs from the included objects.csv file
    objects_file = files("adam_core.utils.helpers.data").joinpath("objects.csv")
    objects_df = pd.read_csv(objects_file)

    # Query for the orbital elements from SBDB and save to a file
    object_ids = objects_df["object_id"].values

    orbits = query_sbdb(object_ids)

    # Rename 'Oumuamua to match the name in the Horizons database (1I/'Oumuamua (A/2017 U1))
    object_ids = orbits.object_id.to_numpy(zero_copy_only=False)
    orbits = orbits.set_column(
        "object_id",
        np.where(
            object_ids == "'Oumuamua (A/2017 U1)",
            "1I/'Oumuamua (A/2017 U1)",
            object_ids,
        ),
    )

    orbits.to_parquet(files("adam_core.utils.helpers.data").joinpath("orbits.parquet"))

    # Query for the orbital elements in different representations from JPL Horizons and save to a file
    for frame in ["ecliptic", "equatorial"]:
        if frame == "equatorial":
            refplane = "earth"
            frame_out = "eq"
        else:
            refplane = "ecliptic"
            frame_out = "ec"

        for origin in ["@sun", "@ssb"]:

            horizons_elements_dfs = []
            for orbit in orbits:
                prov_designation = (
                    orbit.object_id.to_numpy(zero_copy_only=False)[0]
                    .split("(")
                    .pop()
                    .split(")")[0]
                )

                horizons_elements_df = _get_orbital_elements(
                    [prov_designation],
                    orbit.coordinates.time,
                    location=origin,
                    refplane=refplane,
                )
                horizons_elements_dfs.append(horizons_elements_df)

            horizons_elements_df = pd.concat(horizons_elements_dfs)

            if origin == "@sun":
                horizons_elements_df.to_csv(
                    files("adam_core.utils.helpers.data").joinpath(
                        f"elements_sun_{frame_out}.csv"
                    ),
                    index=False,
                )
            else:
                horizons_elements_df.to_csv(
                    files("adam_core.utils.helpers.data").joinpath(
                        f"elements_ssb_{frame_out}.csv"
                    ),
                    index=False,
                )

    # Lets query for propagated Horizons state vectors
    propagated_orbits_list = []
    ephemeris_dfs = []

    # Create an array of delta times to propagate the orbits and
    # get ephemerides relative to the epoch at which the orbits are defined
    total_days = 60
    half_arc = total_days / 2
    # Make it so there are 3 observations every 2 days
    dts = np.arange(-half_arc, half_arc, 2)
    dts = np.concatenate([dts, dts + 1 / 48, dts + 1 / 24])
    dts.sort()
    num_dts = len(dts)

    for i, orbit in enumerate(orbits):
        # Get the provisional designation
        prov_designation = (
            orbit.object_id.to_numpy(zero_copy_only=False)[0]
            .split("(")
            .pop()
            .split(")")[0]
        )

        # Extract times from orbit and propagate +/- 30 days
        times = orbit.coordinates.time.add_fractional_days(dts)

        # Query for propagated Horizons state vectors
        propagated_orbit = query_horizons([prov_designation], times)
        propagated_orbit = propagated_orbit.set_column(
            "orbit_id", pa.array([f"{i:05d}" for _ in range(len(times))])
        )

        # Define two observers one at the Rubin Observatory and one at CTIO
        observer_X05 = Observers.from_code("X05", times[: num_dts // 2])
        observer_W84 = Observers.from_code("W84", times[num_dts // 2 :])
        observers = qv.concatenate([observer_X05, observer_W84])

        # Query for ephemeris at the same times
        ephemeris = query_horizons_ephemeris([prov_designation], observers)
        ephemeris["orbit_id"] = [f"{i:05d}" for _ in range(len(times))]

        propagated_orbits_list.append(propagated_orbit)
        ephemeris_dfs.append(ephemeris)

    propagated_orbits: Orbits = qv.concatenate(propagated_orbits_list)
    propagated_orbits.to_parquet(
        files("adam_core.utils.helpers.data").joinpath("propagated_orbits.parquet"),
    )

    ephemeris_df = pd.concat(ephemeris_dfs, ignore_index=True)
    ephemeris_df.to_csv(
        files("adam_core.utils.helpers.data").joinpath("ephemeris.csv"), index=False
    )
