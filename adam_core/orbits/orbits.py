import logging
import uuid
from typing import Iterable, Literal, Tuple

import pandas as pd
import pyarrow.compute as pc
import quivr as qv

from ..coordinates.cartesian import CartesianCoordinates

logger = logging.getLogger(__name__)


class Orbits(qv.Table):

    orbit_id = qv.StringColumn(default=lambda: uuid.uuid4().hex)
    object_id = qv.StringColumn(nullable=True)
    coordinates = CartesianCoordinates.as_column()

    def group_by_orbit_id(self) -> Iterable[Tuple[str, "Orbits"]]:
        """
        Group orbits by orbit ID and yield them.

        Yields
        ------
        orbit_id : str
            Orbit ID.
        orbits : `~adam_core.orbits.orbits.Orbits`
            Orbits belonging to this orbit ID.
        """
        unique_orbit_ids = self.orbit_id.unique()
        for orbit_id in unique_orbit_ids:
            mask = pc.equal(self.orbit_id, orbit_id)
            yield orbit_id, self.apply_mask(mask)

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
                "orbit_id": self.orbit_id.to_pandas(),
                "object_id": self.object_id.to_pandas(),
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
        orbit_id = df["orbit_id"].values
        object_id = df["object_id"].values
        coordinates = CartesianCoordinates.from_dataframe(df, frame=frame)
        return cls.from_kwargs(
            orbit_id=orbit_id, object_id=object_id, coordinates=coordinates
        )
