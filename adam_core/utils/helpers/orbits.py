from importlib.resources import files
from typing import Optional

import numpy as np
import pandas as pd
from astropy.time import Time

from ...coordinates.covariances import CoordinateCovariances
from ...coordinates.keplerian import KeplerianCoordinates
from ...coordinates.origin import Origin
from ...coordinates.times import Times
from ...orbits import Orbits


def make_real_orbits(num_orbits: Optional[int] = None) -> Orbits:
    """
    Returns an `~adam_core.orbits.orbits.Orbits` object with real orbits drawn
    from our list of sample objects.

    Parameters
    ----------
    num_orbits : optional, int
        The number of orbits to return, which must be less than or equal to
        the number of sample objects (27).

    Returns
    -------
    orbits : `~adam_core.orbits.orbits.Orbits`
        Orbits object containing the sample orbits.
    """
    orbits_file = files("adam_core.utils.helpers.data").joinpath("sample_orbits.csv")
    df = pd.read_csv(orbits_file, index_col=False, float_precision="round_trip")
    df["orbit_ids"] = np.array([None for _ in range(len(df))], dtype=object)

    if num_orbits is None:
        num_orbits = len(df)

    if num_orbits > len(df):
        raise ValueError(
            f"num_orbits must be less than or equal to the number of sample orbits ({len(df)})."
        )

    return Orbits.from_dataframe(df, "ecliptic")[:num_orbits]


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
    # Hyperbolic orbits have negative semi-major axes
    data["a"] = np.where(data["e"] > 1.0, -data["a"], data["a"])

    sigmas = np.zeros((num_orbits, 6))
    for i, dim in enumerate(["a", "e", "i", "raan", "ap", "M"]):
        sigmas[:, i] = np.round(0.01 * data[dim], 4)

    data["covariances"] = CoordinateCovariances.from_sigmas(sigmas)
    data["times"] = Times.from_astropy(
        Time(
            np.round(np.linspace(59000.0, 59000.0 + num_orbits, num_orbits), 3),
            scale="tdb",
            format="mjd",
        )
    )
    data["origin"] = Origin.from_kwargs(code=["SUN" for i in range(num_orbits)])
    data["frame"] = "ecliptic"

    coords = KeplerianCoordinates.from_kwargs(**data)
    object_ids = [f"Object {i:03d}" for i in range(num_orbits)]
    orbit_ids = [f"Orbit {i:03d}" for i in range(num_orbits)]

    orbits = Orbits.from_kwargs(
        coordinates=coords.to_cartesian(),
        orbit_ids=orbit_ids,
        object_ids=object_ids,
    )
    return orbits
