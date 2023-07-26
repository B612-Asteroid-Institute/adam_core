from importlib.resources import files

import pandas as pd
import pytest


@pytest.fixture
def orbital_elements():
    orbital_elements_file = files("adam_core.utils.helpers.data").joinpath(
        "elements_sun.csv"
    )
    df = pd.read_csv(
        orbital_elements_file, index_col=False, float_precision="round_trip"
    )
    return df
