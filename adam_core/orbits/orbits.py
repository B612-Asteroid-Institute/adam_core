import logging
from typing import Literal

import pandas as pd
from quivr import StringField, Table

from ..coordinates.cartesian import CartesianCoordinates

logger = logging.getLogger(__name__)


class Orbits(Table):

    orbit_ids = StringField(nullable=True)
    object_ids = StringField(nullable=True)
    coordinates = CartesianCoordinates.as_field(nullable=False)

    def to_dataframe(self, sigmas: bool = False, covariances: bool = True):
        """
        Represent the orbits as a pandas DataFrame.

        Parameters
        ----------
        sigmas : bool, optional
            If True, include 1-sigma uncertainties in the DataFrame.
        covariances : bool, optional
            If True, include covariance matrices in the DataFrame. Covariance matrices
            will be split into 21 columns, with the lower triangular elements stored.

        Returns
        -------
        df : `~pandas.Dataframe`
            DataFrame containing orbits and their Cartesian elements.
        """
        df = pd.DataFrame(
            {
                "orbit_ids": self.orbit_ids.to_pandas(),
                "object_ids": self.object_ids.to_pandas(),
            }
        )
        df = df.join(
            self.coordinates.to_dataframe(sigmas=sigmas, covariances=covariances)
        )
        return df

    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, frame: Literal["ecliptic", "equatorial"]
    ) -> "Orbits":
        """
        Create an Orbits object from a pandas DataFrame.

        Parameters
        ----------
        df : `~pandas.DataFrame`
            DataFrame containing orbits and their Cartesian elements.
        frame : {"ecliptic", "equatorial"}
            Frame in which coordinates are defined.

        Returns
        -------
        orbits : `~adam_core.orbits.orbits.Orbits`
            Orbits.
        """
        orbit_ids = df["orbit_ids"].values
        object_ids = df["object_ids"].values
        coordinates = CartesianCoordinates.from_dataframe(df, frame=frame)
        return cls.from_kwargs(
            orbit_ids=orbit_ids, object_ids=object_ids, coordinates=coordinates
        )
