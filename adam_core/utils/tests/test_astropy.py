import numpy as np
import pytest
from astropy.time import Time

from ..astropy import _check_times


def test__check_times():
    # Create an array of epochs
    times = np.linspace(59580, 59590, 100)

    # Test that an error is raised when times are not
    # an astropy time object
    with pytest.raises(TypeError):
        _check_times(times, "test")

    # Test that _check_times passes when an astropy time object is
    # given as intended
    Time(times, format="mjd", scale="utc")

    return
