from importlib.resources import files

import pandas as pd
import pytest

from adam_core.orbits import Orbits


@pytest.fixture
def orbital_elements():
    orbital_elements_file = files("adam_core.utils.helpers.data").joinpath(
        "elements_sun_ec.csv"
    )
    df = pd.read_csv(
        orbital_elements_file, index_col=False, float_precision="round_trip"
    )
    return df


@pytest.fixture
def orbital_elements_barycentric():
    orbital_elements_file = files("adam_core.utils.helpers.data").joinpath(
        "elements_ssb_ec.csv"
    )
    df = pd.read_csv(
        orbital_elements_file, index_col=False, float_precision="round_trip"
    )
    return df


@pytest.fixture
def propagated_orbits():
    return Orbits.from_parquet(
        files("adam_core.utils.helpers.data").joinpath("propagated_orbits.parquet")
    )


@pytest.fixture
def ephemeris():
    ephemeris_file = files("adam_core.utils.helpers.data").joinpath("ephemeris.csv")
    df = pd.read_csv(
        ephemeris_file,
        index_col=False,
        float_precision="round_trip",
        dtype={"orbit_id": str},
    )
    return df
