from importlib.resources import files

import pandas as pd
import pytest


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
def orbital_elements_equatorial():
    orbital_elements_file = files("adam_core.utils.helpers.data").joinpath(
        "elements_sun_eq.csv"
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
def orbital_elements_barycentric_equatorial():
    orbital_elements_file = files("adam_core.utils.helpers.data").joinpath(
        "elements_ssb_eq.csv"
    )
    df = pd.read_csv(
        orbital_elements_file, index_col=False, float_precision="round_trip"
    )
    return df
