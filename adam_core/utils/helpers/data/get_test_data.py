from importlib.resources import files
from typing import List

import pandas as pd
from astropy.time import Time

from adam_core.orbits.query import query_sbdb
from adam_core.orbits.query.horizons import (
    _get_horizons_elements,
    _get_horizons_vectors,
)


def _get_orbital_elements(
    object_ids: List[str], time: Time, location: str
) -> pd.DataFrame:
    """
    Get orbital elements as Cartesian, Cometary, and Keplerian representations from JPL Horizons.

    Parameters
    ----------
    object_id : List[str]
        Object IDs to query.
    epoch : `~astropy.time.core.Time`
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
    )
    vectors_df = vectors_df[
        ["targetname", "datetime_jd", "x", "y", "z", "vx", "vy", "vz"]
    ]

    elements_df = _get_horizons_elements(
        object_ids, time, location=location, id_type="smallbody"
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
        1, "mjd_tdb", Time(horizons_elements["datetime_jd"], format="jd").mjd
    )
    horizons_elements.insert(
        len(horizons_elements.columns),
        "tp_mjd",
        Time(horizons_elements["Tp_jd"], format="jd").mjd,
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
    orbits_df = orbits.to_dataframe()
    orbits_df.to_csv(
        files("adam_core.utils.helpers.data").joinpath("orbits.csv"), index=False
    )

    # Query for the orbital elements in different representations from JPL Horizons and save to a file
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
                [prov_designation], orbit.coordinates.time.to_astropy(), location=origin
            )
            horizons_elements_dfs.append(horizons_elements_df)

        horizons_elements_df = pd.concat(horizons_elements_dfs)

        if origin == "@sun":
            horizons_elements_df.to_csv(
                files("adam_core.utils.helpers.data").joinpath("elements_sun.csv"),
                index=False,
            )
        else:
            horizons_elements_df.to_csv(
                files("adam_core.utils.helpers.data").joinpath("elements_ssb.csv"),
                index=False,
            )
