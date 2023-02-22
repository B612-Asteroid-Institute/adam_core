from importlib.resources import files
from typing import Optional

import numpy as np
import pandas as pd
from astropy.time import Time

from ...coordinates import KeplerianCoordinates
from ...orbits import Orbits


def make_real_orbits(num_orbits: Optional[int] = None) -> Orbits:
    """
    Returns an `~adam_core.orbits.orbits.Orbits` object with real orbits drawn
    from our list of sample objects.

    Parameters
    ----------
    num_orbits : optional, int
        The number of orbits to return, which must be less than or equal to
        the number of sample objects (26).

    Returns
    -------
    orbits : `~adam_core.orbits.orbits.Orbits`
        Orbits object containing the sample orbits.
    """
    orbits_file = files("adam_core.utils.helpers.data").joinpath("sample_orbits.csv")
    df = pd.read_csv(orbits_file, index_col=False)

    if num_orbits is None:
        num_orbits = len(df)

    if num_orbits > len(df):
        raise ValueError(
            f"num_orbits must be less than or equal to the number of sample orbits ({len(df)})."
        )

    return Orbits.from_df(df.iloc[:num_orbits])


def make_simple_orbits(num_orbits: int = 10) -> Orbits:
    """
    Returns an `~adam_core.orbits.orbits.Orbits` object with simple orbits.

    Parameters
    ----------
    num_orbits : int, optional
        The number of orbits to return.
        Default is 10.

    Returns
    -------
    orbits : `~adam_core.orbits.orbits.Orbits`
        Orbits object containing the sample orbits.
    """

    data = {
        "a": np.linspace(1, 10, num_orbits),
        "e": np.linspace(0, 2, num_orbits),
        "i": np.linspace(0, 180, num_orbits),
        "raan": np.linspace(0, 360, num_orbits),
        "ap": np.linspace(0, 360, num_orbits),
        "M": np.linspace(0, 360, num_orbits),
    }
    for dim, value in data.copy().items():
        data[f"sigma_{dim}"] = np.round(0.01 * value, 4)
        data[dim] = np.round(value, 3)

    data["times"] = Time(
        np.round(np.linspace(59000.0, 59000.0 + num_orbits, num_orbits), 3),
        scale="tdb",
        format="mjd",
    )

    coords = KeplerianCoordinates(**data)
    object_ids = [f"Object {i:03d}" for i in range(num_orbits)]
    orbit_ids = [f"Orbit {i:03d}" for i in range(num_orbits)]

    orbits = Orbits(
        coords,
        orbit_ids=orbit_ids,
        object_ids=object_ids,
    )
    return orbits
