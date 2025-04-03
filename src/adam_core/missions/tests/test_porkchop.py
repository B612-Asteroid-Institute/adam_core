import numpy as np
import pytest

from adam_core.coordinates.origin import OriginCodes
from adam_core.time import Timestamp

from ..porkchop import generate_porkchop_data


def test_generate_porkchop_data_origins():
    # Test with different origins
    earliest_launch = Timestamp.from_mjd([60000], scale="tdb")
    maximum_arrival = Timestamp.from_mjd([60100], scale="tdb")
    
    # Test with Sun as origin
    results_sun = generate_porkchop_data(
        departure_body=OriginCodes.EARTH,
        arrival_body=OriginCodes.MARS_BARYCENTER,
        earliest_launch_time=earliest_launch,
        maximum_arrival_time=maximum_arrival,
        propagation_origin=OriginCodes.SUN,
        step_size=5.0,  # Larger step size for faster test
    )
    
    
    # Verify that results are generated
    assert len(results_sun) > 0
    
    # Check that origins match what we specified
    assert results_sun.origin.as_OriginCodes() == OriginCodes.SUN

    # Check that time of flight is valid (positive)
    assert np.all(results_sun.time_of_flight() > 0)
    
    # Check that C3 values are computed
    assert np.all(~np.isnan(results_sun.c3()))
    


