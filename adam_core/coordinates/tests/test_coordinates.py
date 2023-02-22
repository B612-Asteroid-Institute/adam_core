from typing import Optional, Union

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.time import Time

from ..coordinates import Coordinates

DUMMY_COLS = {}
DUMMY_UNITS = {}
for i in ["a", "b", "c"]:
    DUMMY_COLS[i] = i
DUMMY_UNITS["a"] = u.au
DUMMY_UNITS["b"] = u.au
DUMMY_UNITS["c"] = u.au


class DummyCoordinates(Coordinates):
    def __init__(
        self,
        a: Optional[Union[int, float, np.ndarray]] = None,
        b: Optional[Union[int, float, np.ndarray]] = None,
        c: Optional[Union[int, float, np.ndarray]] = None,
        times: Optional[Time] = None,
        covariances: Optional[np.ndarray] = None,
        sigma_a: Optional[np.ndarray] = None,
        sigma_b: Optional[np.ndarray] = None,
        sigma_c: Optional[np.ndarray] = None,
        origin: str = "heliocenter",
        frame: str = "ecliptic",
        names: dict = DUMMY_COLS,
        units: dict = DUMMY_UNITS,
    ):
        sigmas = (sigma_a, sigma_b, sigma_c)
        Coordinates.__init__(
            self,
            a=a,
            b=b,
            c=c,
            covariances=covariances,
            sigmas=sigmas,
            times=times,
            origin=origin,
            frame=frame,
            names=names,
            units=units,
        )
        return


def test_Coordinates_to_df_uncertainties():
    # Test that to_df() returns the correct dataframe for coordinates
    # without any errors
    coordinates = DummyCoordinates(
        a=np.array([1, 2, 3]),
        b=np.array([4, 5, 6]),
        c=np.array([7, 8, 9]),
        times=Time(np.array([59000.0, 59001.0, 59002.0]), format="mjd", scale="utc"),
    )

    df = coordinates.to_df()
    df_desired = pd.DataFrame(
        {
            "mjd_utc": [59000.0, 59001.0, 59002.0],
            "a": np.array([1, 2, 3], dtype=np.float64),
            "b": np.array([4, 5, 6], dtype=np.float64),
            "c": np.array([7, 8, 9], dtype=np.float64),
            "origin": ["heliocenter", "heliocenter", "heliocenter"],
            "frame": ["ecliptic", "ecliptic", "ecliptic"],
        }
    )
    pd.testing.assert_frame_equal(df, df_desired, rtol=0, atol=1e-16)

    # Test that to_df() returns the correct dataframe for coordinates
    # with errors
    sigma_a = np.array([0.1, 0.2, 0.3])
    sigma_b = np.array([0.4, 0.5, 0.6])
    sigma_c = np.array([0.7, 0.8, 0.9])
    coordinates = DummyCoordinates(
        a=np.array([1, 2, 3]),
        b=np.array([4, 5, 6]),
        c=np.array([7, 8, 9]),
        sigma_a=sigma_a,
        sigma_b=sigma_b,
        sigma_c=sigma_c,
        times=Time(np.array([59000.0, 59001.0, 59002.0]), format="mjd", scale="utc"),
    )

    cov_a_a = sigma_a**2
    cov_b_b = sigma_b**2
    cov_c_c = sigma_c**2
    df = coordinates.to_df()
    df_desired = pd.DataFrame(
        {
            "mjd_utc": [59000.0, 59001.0, 59002.0],
            "a": np.array([1, 2, 3], dtype=np.float64),
            "b": np.array([4, 5, 6], dtype=np.float64),
            "c": np.array([7, 8, 9], dtype=np.float64),
            "sigma_a": sigma_a,
            "sigma_b": sigma_b,
            "sigma_c": sigma_c,
            "cov_a_a": cov_a_a,
            "cov_b_a": np.zeros(3),
            "cov_b_b": cov_b_b,
            "cov_c_a": np.zeros(3),
            "cov_c_b": np.zeros(3),
            "cov_c_c": cov_c_c,
            "origin": ["heliocenter", "heliocenter", "heliocenter"],
            "frame": ["ecliptic", "ecliptic", "ecliptic"],
        }
    )
    pd.testing.assert_frame_equal(df, df_desired, rtol=0, atol=1e-16)

    # Test that to_df() returns the correct dataframe for coordinates
    # with errors but with the user asking for no sigmas
    sigma_a = np.array([0.1, 0.2, 0.3])
    sigma_b = np.array([0.4, 0.5, 0.6])
    sigma_c = np.array([0.7, 0.8, 0.9])
    coordinates = DummyCoordinates(
        a=np.array([1, 2, 3]),
        b=np.array([4, 5, 6]),
        c=np.array([7, 8, 9]),
        sigma_a=sigma_a,
        sigma_b=sigma_b,
        sigma_c=sigma_c,
        times=Time(np.array([59000.0, 59001.0, 59002.0]), format="mjd", scale="utc"),
    )

    cov_a_a = sigma_a**2
    cov_b_b = sigma_b**2
    cov_c_c = sigma_c**2
    df = coordinates.to_df(sigmas=False)
    df_desired = pd.DataFrame(
        {
            "mjd_utc": [59000.0, 59001.0, 59002.0],
            "a": np.array([1, 2, 3], dtype=np.float64),
            "b": np.array([4, 5, 6], dtype=np.float64),
            "c": np.array([7, 8, 9], dtype=np.float64),
            "cov_a_a": cov_a_a,
            "cov_b_a": np.zeros(3),
            "cov_b_b": cov_b_b,
            "cov_c_a": np.zeros(3),
            "cov_c_b": np.zeros(3),
            "cov_c_c": cov_c_c,
            "origin": ["heliocenter", "heliocenter", "heliocenter"],
            "frame": ["ecliptic", "ecliptic", "ecliptic"],
        }
    )
    pd.testing.assert_frame_equal(df, df_desired, rtol=0, atol=1e-16)

    # Test that to_df() returns the correct dataframe for coordinates
    # with errors but with the user asking for no sigmas and no covariances
    sigma_a = np.array([0.1, 0.2, 0.3])
    sigma_b = np.array([0.4, 0.5, 0.6])
    sigma_c = np.array([0.7, 0.8, 0.9])
    coordinates = DummyCoordinates(
        a=np.array([1, 2, 3]),
        b=np.array([4, 5, 6]),
        c=np.array([7, 8, 9]),
        sigma_a=sigma_a,
        sigma_b=sigma_b,
        sigma_c=sigma_c,
        times=Time(np.array([59000.0, 59001.0, 59002.0]), format="mjd", scale="utc"),
    )

    df = coordinates.to_df(sigmas=False, covariances=False)
    df_desired = pd.DataFrame(
        {
            "mjd_utc": [59000.0, 59001.0, 59002.0],
            "a": np.array([1, 2, 3], dtype=np.float64),
            "b": np.array([4, 5, 6], dtype=np.float64),
            "c": np.array([7, 8, 9], dtype=np.float64),
            "origin": ["heliocenter", "heliocenter", "heliocenter"],
            "frame": ["ecliptic", "ecliptic", "ecliptic"],
        }
    )
    pd.testing.assert_frame_equal(df, df_desired, rtol=0, atol=1e-16)


def test_Coordinates_to_df_raises():
    # Test that to_df() raises an error if the sigmas and covariances
    # are incorrectly typed
    sigma_a = np.array([0.1, 0.2, 0.3])
    sigma_b = np.array([0.4, 0.5, 0.6])
    sigma_c = np.array([0.7, 0.8, 0.9])
    coordinates = DummyCoordinates(
        a=np.array([1, 2, 3]),
        b=np.array([4, 5, 6]),
        c=np.array([7, 8, 9]),
        sigma_a=sigma_a,
        sigma_b=sigma_b,
        sigma_c=sigma_c,
        times=Time(np.array([59000.0, 59001.0, 59002.0]), format="mjd", scale="utc"),
    )

    with pytest.raises(TypeError):
        coordinates.to_df(sigmas="")

    with pytest.raises(TypeError):
        coordinates.to_df(covariances="")
