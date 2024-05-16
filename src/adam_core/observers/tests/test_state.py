import os

import numpy as np
import pytest

from ...constants import KM_P_AU, S_P_DAY
from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import OriginCodes
from ...coordinates.residuals import Residuals
from ...observers import get_observer_state
from ...time import Timestamp
from ...utils import get_perturber_state

DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")


def test_get_observer_state_OriginCodes():
    # Test that get_observer_state works using OriginCodes instead of MPC
    # observatory codes
    codes = [
        OriginCodes.SUN,
        OriginCodes.MERCURY,
        OriginCodes.VENUS,
        OriginCodes.EARTH,
        OriginCodes.MARS_BARYCENTER,
        OriginCodes.JUPITER_BARYCENTER,
    ]
    for code in codes:

        times = Timestamp.from_mjd(np.arange(59000, 60000), scale="tdb")
        perturber_state = get_perturber_state(
            code, times, frame="ecliptic", origin=OriginCodes.SOLAR_SYSTEM_BARYCENTER
        )
        observer_state = get_observer_state(
            code, times, frame="ecliptic", origin=OriginCodes.SOLAR_SYSTEM_BARYCENTER
        )
        assert perturber_state == observer_state


expected_precision = {
    # Units of m and mm/s
    "X05": {"r": 15, "v": 2},
    "I41": {"r": 10, "v": 2},
    "W84": {"r": 10, "v": 2},
    "F51": {"r": 15, "v": 2},
    "000": {"r": 10, "v": 2},
    "500": {"r": 0.10, "v": 0.01},
}


@pytest.mark.parametrize("code", expected_precision.keys())
@pytest.mark.parametrize("origin", ["sun", "ssb"])
def test_get_observer_state(code, origin):
    # Test that we can get the observer state for X05 to within 15 m and 2 mm/s
    # using the MPC extended observatory codes file and SPICE kernels
    states_expected = CartesianCoordinates.from_parquet(
        os.path.join(DATA_DIR, f"{code}_{origin}.parquet"),
    )

    if origin == "sun":
        origin_code = OriginCodes.SUN
    else:
        origin_code = OriginCodes.SOLAR_SYSTEM_BARYCENTER

    # Get the observer state using adam_core
    states = get_observer_state(
        code,
        states_expected.time,
        frame=states_expected.frame,
        origin=origin_code,
    )

    # Calculate residuals and extract values
    residuals = Residuals.calculate(states, states_expected)
    residual_values = np.stack(residuals.values.to_numpy(zero_copy_only=False))

    # Calculate the offset in position in meters
    r_offset = np.linalg.norm(residual_values[:, :3], axis=1) * KM_P_AU * 1000
    # Calculate the offset in velocity in mm/s
    v_offset = (
        np.linalg.norm(residual_values[:, 3:], axis=1) * KM_P_AU / S_P_DAY * 1000000
    )
    np.testing.assert_array_less(r_offset, expected_precision[code]["r"])
    np.testing.assert_array_less(v_offset, expected_precision[code]["v"])


def test_get_observer_state_raises():
    # Test that when we ask for a space-based observatory we raise an error
    code = "C51"
    with pytest.raises(ValueError):
        # Get the observer state using adam_core
        get_observer_state(
            code,
            Timestamp.from_mjd([59000, 60000], scale="tdb"),
        )
