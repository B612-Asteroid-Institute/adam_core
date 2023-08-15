import pandas as pd
from quivr import StringColumn, Table
from typing_extensions import Self

from ..coordinates.spherical import SphericalCoordinates


class Ephemeris(Table):

    orbit_id = StringColumn(nullable=False)
    object_id = StringColumn()
    coordinates = SphericalCoordinates.as_column(nullable=False)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the Ephemeris table to a pandas DataFrame.

        Returns
        -------
        df : `~pandas.DataFrame`
            The Ephemeris table as a DataFrame.
        """
        assert self.coordinates.frame == "equatorial"
        df = pd.DataFrame()
        df["orbit_id"] = self.orbit_id
        df["object_id"] = self.object_id
        df_coordinates = self.coordinates.to_dataframe()

        df = pd.concat([df, df_coordinates], axis=1)
        df.rename(
            columns={
                "origin.code": "obs_code",
                "lon_eq": "ra",
                "lat_eq": "dec",
                "vlon_eq": "vra",
                "vlat_eq": "vdec",
                "rho_eq": "rho",
                "vrho_eq": "vrho",
            },
            inplace=True,
        )
        return df

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> Self:
        """
        Instantiate an Ephemeris table from a pandas DataFrame.

        Parameters
        ----------
        df : `~pandas.DataFrame`
            The Ephemeris table as a DataFrame.

        Returns
        -------
        ephemeris : `~adam_core.orbits.ephemeris.Ephemeris`
            The Ephemeris table.
        """
        coordinates = SphericalCoordinates.from_dataframe(
            df.rename(
                columns={
                    "ra": "lon_eq",
                    "dec": "lat_eq",
                    "vra": "vlon_eq",
                    "vdec": "vlat_eq",
                    "rho": "rho_eq",
                    "vrho": "vrho_eq",
                    "obs_code": "origin.code",
                },
            ),
        )
        return cls.from_kwargs(
            orbit_id=df["orbit_id"],
            object_id=df["object_id"],
            coordinates=coordinates,
        )
