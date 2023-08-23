import os

import numpy as np
import pandas as pd
import pytest
from astropy.time import Time

from ...constants import KM_P_AU, S_P_DAY
from ...coordinates import Residuals
from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import OriginCodes
from ...observers import get_observer_state
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

        times = Time(np.arange(59000, 60000), format="mjd", scale="tdb")
        perturber_state = get_perturber_state(
            code, times, frame="ecliptic", origin=OriginCodes.SOLAR_SYSTEM_BARYCENTER
        )
        observer_state = get_observer_state(
            code, times, frame="ecliptic", origin=OriginCodes.SOLAR_SYSTEM_BARYCENTER
        )
        assert perturber_state == observer_state


def test_get_observer_state_X05():
    # Test that we can get the observer state for X05 to within 15 m and 2 mm/s
    # using the MPC extended observatory codes file and SPICE kernels
    code = "X05"
    for origin in ["sun", "ssb"]:
        states_df = pd.read_csv(
            os.path.join(DATA_DIR, f"{code}_{origin}.csv"),
            index_col=False,
            float_precision="round_trip",
        )
        states_expected = CartesianCoordinates.from_dataframe(states_df, "ecliptic")

        if origin == "sun":
            origin_code = OriginCodes.SUN
        else:
            origin_code = OriginCodes.SOLAR_SYSTEM_BARYCENTER

        # Get the observer state using adam_core
        states = get_observer_state(
            code,
            states_expected.time.to_astropy(),
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

        np.testing.assert_array_less(r_offset, 15)  # Positions agree to within 15 m
        np.testing.assert_array_less(v_offset, 2)  # Velocities agree to within 2 mm/s


def test_get_observer_state_I41():
    # Test that we can get the observer state for I41 to within 10 m and 1 mm/s
    # using the MPC extended observatory codes file and SPICE kernels
    code = "I41"
    for origin in ["sun", "ssb"]:
        states_df = pd.read_csv(
            os.path.join(DATA_DIR, f"{code}_{origin}.csv"),
            index_col=False,
            float_precision="round_trip",
        )
        states_expected = CartesianCoordinates.from_dataframe(states_df, "ecliptic")

        if origin == "sun":
            origin_code = OriginCodes.SUN
        else:
            origin_code = OriginCodes.SOLAR_SYSTEM_BARYCENTER

        # Get the observer state using adam_core
        states = get_observer_state(
            code,
            states_expected.time.to_astropy(),
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

        np.testing.assert_array_less(r_offset, 10)  # Positions agree to within 10 m
        np.testing.assert_array_less(v_offset, 2)  # Velocities agree to within 2 mm/s


def test_get_observer_state_W84():
    # Test that we can get the observer state for W84 to within 10 m and 1 mm/s
    # using the MPC extended observatory codes file and SPICE kernels
    code = "W84"
    for origin in ["sun", "ssb"]:
        states_df = pd.read_csv(
            os.path.join(DATA_DIR, f"{code}_{origin}.csv"),
            index_col=False,
            float_precision="round_trip",
        )
        states_expected = CartesianCoordinates.from_dataframe(states_df, "ecliptic")

        if origin == "sun":
            origin_code = OriginCodes.SUN
        else:
            origin_code = OriginCodes.SOLAR_SYSTEM_BARYCENTER

        # Get the observer state using adam_core
        states = get_observer_state(
            code,
            states_expected.time.to_astropy(),
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

        np.testing.assert_array_less(r_offset, 10)  # Positions agree to within 10 m
        np.testing.assert_array_less(v_offset, 2)  # Velocities agree to within 2 mm/s


def test_get_observer_state_000():
    # Test that we can get the observer state for 000 to within 10 m and 1 mm/s
    # using the MPC extended observatory codes file and SPICE kernels
    code = "000"
    for origin in ["sun", "ssb"]:
        states_df = pd.read_csv(
            os.path.join(DATA_DIR, f"{code}_{origin}.csv"),
            index_col=False,
            float_precision="round_trip",
        )
        states_expected = CartesianCoordinates.from_dataframe(states_df, "ecliptic")

        if origin == "sun":
            origin_code = OriginCodes.SUN
        else:
            origin_code = OriginCodes.SOLAR_SYSTEM_BARYCENTER

        # Get the observer state using adam_core
        states = get_observer_state(
            code,
            states_expected.time.to_astropy(),
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

        np.testing.assert_array_less(r_offset, 10)  # Positions agree to within 10 m
        np.testing.assert_array_less(v_offset, 2)  # Velocities agree to within 2 mm/s


def test_get_observer_state_500():
    # Test that we can get the observer state for 500 to within 10 cm and 0.01 mm/s
    # using the MPC extended observatory codes file and SPICE kernels
    code = "500"
    for origin in ["sun", "ssb"]:
        states_df = pd.read_csv(
            os.path.join(DATA_DIR, f"{code}_{origin}.csv"),
            index_col=False,
            float_precision="round_trip",
        )
        states_expected = CartesianCoordinates.from_dataframe(states_df, "ecliptic")

        if origin == "sun":
            origin_code = OriginCodes.SUN
        else:
            origin_code = OriginCodes.SOLAR_SYSTEM_BARYCENTER

        # Get the observer state using adam_core
        states = get_observer_state(
            code,
            states_expected.time.to_astropy(),
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

        np.testing.assert_array_less(r_offset, 0.10)  # Positions agree to within 10 cm
        np.testing.assert_array_less(
            v_offset, 0.01
        )  # Velocities agree to within 0.01 mm/s


def test_get_observer_state_raises():
    # Test that when we ask for a space-based observatory we raise an error
    code = "C51"
    with pytest.raises(ValueError):
        # Get the observer state using adam_core
        get_observer_state(
            code,
            Time([59000, 59001], format="mjd", scale="tdb"),
        )
