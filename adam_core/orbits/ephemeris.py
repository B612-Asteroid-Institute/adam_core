import pandas as pd
from quivr import Float64Column, StringAttribute, StringColumn, Table
from typing_extensions import Self

from ..coordinates.spherical import SphericalCoordinates
from ..observers import Observers


class Ephemeris(Table):

    orbit_id = StringColumn(nullable=False)
    object_id = StringColumn()
    observer = Observers.as_column(nullable=False)
    rho = Float64Column()
    ra = Float64Column(nullable=False)
    dec = Float64Column(nullable=False)
    vrho = Float64Column()
    vra = Float64Column()
    vdec = Float64Column()
    frame = StringAttribute()

    def as_spherical_coordinates(self) -> SphericalCoordinates:
        return SphericalCoordinates.from_kwargs(
            rho=self.rho,
            lon=self.ra,
            lat=self.dec,
            vrho=self.vrho,
            vlon=self.vra,
            vlat=self.vdec,
            time=self.observer.coordinates.time,
            origin=self.observer.code,
            frame=self.frame,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the Ephemeris table to a pandas DataFrame.

        Returns
        -------
        df : `~pandas.DataFrame`
            The Ephemeris table as a DataFrame.
        """
        df = pd.DataFrame()
        df["orbit_id"] = self.orbit_id
        df["object_id"] = self.object_id
        df["ra"] = self.ra
        df["dec"] = self.dec
        df["vra"] = self.vra
        df["vdec"] = self.vdec
        df["rho"] = self.rho
        df["vrho"] = self.vrho

        df_obs = self.observer.to_dataframe()
        df = pd.concat([df, df_obs], axis=1)

        df = df[
            [
                "orbit_id",
                "object_id",
                "obs_code",
                "obs_jd1_tdb",
                "obs_jd2_tdb",
                "ra",
                "dec",
                "rho",
                "vra",
                "vdec",
                "vrho",
                "obs_x",
                "obs_y",
                "obs_z",
                "obs_vx",
                "obs_vy",
                "obs_vz",
                "obs_origin.code",
            ]
        ]
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
        observers = Observers.from_dataframe(df)

        return cls.from_kwargs(
            orbit_id=df["orbit_id"],
            object_id=df["object_id"],
            observer=observers,
            ra=df["ra"],
            dec=df["dec"],
            rho=df["rho"],
            vra=df["vra"],
            vdec=df["vdec"],
            vrho=df["vrho"],
            frame="equatorial",
        )
