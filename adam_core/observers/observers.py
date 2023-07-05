import warnings

import pandas as pd
from astropy.time import Time
from mpc_obscodes import mpc_obscodes
from quivr import Float64Column, StringColumn, Table
from typing_extensions import Self

from ..coordinates.cartesian import CartesianCoordinates


class ObservatoryGeodetics(Table):
    code = StringColumn()
    longitude = Float64Column()
    cos_phi = Float64Column()
    sin_phi = Float64Column()
    name = StringColumn()


# Read MPC extended observatory codes file
# Ignore warning about pandas deprecation
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="The behavior of 'to_datetime' with 'unit' when parsing strings is deprecated",
    )
    OBSCODES = pd.read_json(
        mpc_obscodes,
        orient="index",
        dtype={"Longitude": float, "cos": float, "sin": float, "Name": str},
        encoding_errors="strict",
        precise_float=True,
    )
    OBSCODES.reset_index(inplace=True, names=["code"])

OBSERVATORY_GEODETICS = ObservatoryGeodetics.from_kwargs(
    code=OBSCODES["code"].values,
    longitude=OBSCODES["Longitude"].values,
    cos_phi=OBSCODES["cos"].values,
    sin_phi=OBSCODES["sin"].values,
    name=OBSCODES["Name"].values,
)

OBSERVATORY_CODES = {
    x for x in OBSERVATORY_GEODETICS.code.to_numpy(zero_copy_only=False)
}


class Observers(Table):
    code = StringColumn(nullable=False)
    coordinates = CartesianCoordinates.as_column()

    @classmethod
    def from_code(cls, code: str, times: Time) -> Self:
        """
        Instantiate an Observers table with a single code and multiple times.
        Times do not need to be unique. The observer state will be calculated
        for each time and correctly matched to the input times and replicated for
        duplicate times.

        To load multiple codes, use `from_code` and then concatenate the tables.

        Parameters
        ----------
        code : str
            MPC observatory code for which to find the states.
        times : `~astropy.time.core.Time` (N)
            Epochs for which to find the observatory locations.

        Returns
        -------
        observers : `~adam_core.observers.observers.Observers` (N)
            The observer and its state at each time.

        Examples
        --------
        >>> import numpy as np
        >>> from astropy.time import Time
        >>> from adam_core.observers import Observers
        >>> times = Time(np.arange(59000, 59000 + 100), scale="tdb", format="mjd")
        >>> observers = Observers.from_code("X05", times)
        """
        from .state import get_observer_state

        return cls.from_kwargs(
            code=[code] * len(times),
            coordinates=get_observer_state(code, times),
        )

    def iterate_codes(self):
        """
        Iterate over the codes in the Observers table.

        Yields
        ------
        code : str
            The code for observer.
        observers : `~adam_core.observers.observers.Observers`
            The Observers table for this observer.
        """
        for code in self.code.unique():
            yield code.as_py(), self.select("code", code)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the Observers table to a pandas DataFrame.

        Returns
        -------
        df : `~pandas.DataFrame`
            The Observers table as a DataFrame.
        """
        df = self.coordinates.to_dataframe(
            sigmas=False,
            covariances=False,
        )
        df.rename(
            columns={
                "jd1_tdb": "obs_jd1_tdb",
                "jd2_tdb": "obs_jd2_tdb",
                "x": "obs_x",
                "y": "obs_y",
                "z": "obs_z",
                "vx": "obs_vx",
                "vy": "obs_vy",
                "vz": "obs_vz",
                "origin.code": "obs_origin.code",
            },
            inplace=True,
        )
        df.insert(0, "obs_code", self.code)
        return df

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> Self:
        """
        Instantiate an Observers table from a pandas DataFrame.

        Parameters
        ----------
        df : `~pandas.DataFrame`
            The Observers table as a DataFrame.

        Returns
        -------
        observers : `~adam_core.observers.observers.Observers`
            The Observers table.
        """
        df_renamed = df.rename(
            columns={
                "obs_jd1_tdb": "jd1_tdb",
                "obs_jd2_tdb": "jd2_tdb",
                "obs_x": "x",
                "obs_y": "y",
                "obs_z": "z",
                "obs_vx": "vx",
                "obs_vy": "vy",
                "obs_vz": "vz",
                "obs_origin.code": "origin.code",
            }
        )
        coordinates = CartesianCoordinates.from_dataframe(
            df_renamed,
            frame="ecliptic",
        )
        return cls.from_kwargs(
            code=df["obs_code"].values,
            coordinates=coordinates,
        )
