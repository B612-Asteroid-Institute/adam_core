from importlib.resources import files
from typing import Optional

import pandas as pd

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
