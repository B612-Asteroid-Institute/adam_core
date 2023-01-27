from collections import OrderedDict
from typing import Optional, Union

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time

from .cartesian import CartesianCoordinates
from .coordinates import Coordinates
from .covariances import transform_covariances_jacobian

__all__ = [
    "SphericalCoordinates",
    "SPHERICAL_COLS",
    "SPHERICAL_UNITS",
]

SPHERICAL_COLS = OrderedDict()
SPHERICAL_UNITS = OrderedDict()
for i in ["rho", "lon", "lat", "vrho", "vlon", "vlat"]:
    SPHERICAL_COLS[i] = i
SPHERICAL_UNITS["rho"] = u.au
SPHERICAL_UNITS["lon"] = u.deg
SPHERICAL_UNITS["lat"] = u.deg
SPHERICAL_UNITS["vrho"] = u.au / u.d
SPHERICAL_UNITS["vlon"] = u.deg / u.d
SPHERICAL_UNITS["vlat"] = u.deg / u.d


class SphericalCoordinates(Coordinates):
    def __init__(
        self,
        rho: Optional[Union[int, float, np.ndarray]] = None,
        lon: Optional[Union[int, float, np.ndarray]] = None,
        lat: Optional[Union[int, float, np.ndarray]] = None,
        vrho: Optional[Union[int, float, np.ndarray]] = None,
        vlon: Optional[Union[int, float, np.ndarray]] = None,
        vlat: Optional[Union[int, float, np.ndarray]] = None,
        times: Optional[Time] = None,
        covariances: Optional[np.ndarray] = None,
        sigma_rho: Optional[np.ndarray] = None,
        sigma_lon: Optional[np.ndarray] = None,
        sigma_lat: Optional[np.ndarray] = None,
        sigma_vrho: Optional[np.ndarray] = None,
        sigma_vlon: Optional[np.ndarray] = None,
        sigma_vlat: Optional[np.ndarray] = None,
        origin: str = "heliocentric",
        frame: str = "ecliptic",
        names: OrderedDict = SPHERICAL_COLS,
        units: OrderedDict = SPHERICAL_UNITS,
    ):
        """

        Parameters
        ----------
        rho : `~numpy.ndarray` (N)
            Radial distance in units of au.
        lon : `~numpy.ndarray` (N)
            Longitudinal angle in units of degrees.
        lat : `~numpy.ndarray` (N)
            Latitudinal angle in units of degrees (geographic coordinate
            style with 0 degrees at the equator and ranging from -90 to 90).
        vrho : `~numpy.ndarray` (N)
            Radial velocity in units of au per day.
        vlon : `~numpy.ndarray` (N)
            Longitudinal velocity in units of degrees per day.
        vlat : `~numpy.ndarray` (N)
            Latitudinal velocity in units of degrees per day.
        """
        sigmas = (sigma_rho, sigma_lon, sigma_lat, sigma_vrho, sigma_vlon, sigma_vlat)
        Coordinates.__init__(
            self,
            rho=rho,
            lon=lon,
            lat=lat,
            vrho=vrho,
            vlon=vlon,
            vlat=vlat,
            covariances=covariances,
            sigmas=sigmas,
            times=times,
            origin=origin,
            frame=frame,
            names=names,
            units=units,
        )
        return

    @property
    def rho(self):
        return self._values[:, 0]

    @property
    def lon(self):
        return self._values[:, 1]

    @property
    def lat(self):
        return self._values[:, 2]

    @property
    def vrho(self):
        return self._values[:, 3]

    @property
    def vlon(self):
        return self._values[:, 4]

    @property
    def vlat(self):
        return self._values[:, 5]

    @property
    def sigma_rho(self):
        return self.sigmas[:, 0]

    @property
    def sigma_lon(self):
        return self.sigmas[:, 1]

    @property
    def sigma_lat(self):
        return self.sigmas[:, 2]

    @property
    def sigma_vrho(self):
        return self.sigmas[:, 3]

    @property
    def sigma_vlon(self):
        return self.sigmas[:, 4]

    @property
    def sigma_vlat(self):
        return self.sigmas[:, 5]

    def to_cartesian(self) -> CartesianCoordinates:
        from .transform import _spherical_to_cartesian, spherical_to_cartesian

        coords_cartesian = spherical_to_cartesian(self.values.filled())
        coords_cartesian = np.array(coords_cartesian)

        if self.covariances is not None:
            covariances_cartesian = transform_covariances_jacobian(
                self.values, self.covariances, _spherical_to_cartesian
            )
        else:
            covariances_cartesian = None

        coords = CartesianCoordinates(
            x=coords_cartesian[:, 0],
            y=coords_cartesian[:, 1],
            z=coords_cartesian[:, 2],
            vx=coords_cartesian[:, 3],
            vy=coords_cartesian[:, 4],
            vz=coords_cartesian[:, 5],
            times=self.times,
            covariances=covariances_cartesian,
            origin=self.origin,
            frame=self.frame,
        )
        return coords

    @classmethod
    def from_cartesian(cls, cartesian: CartesianCoordinates) -> "SphericalCoordinates":

        from .transform import _cartesian_to_spherical, cartesian_to_spherical

        coords_spherical = cartesian_to_spherical(cartesian.values.filled())
        coords_spherical = np.array(coords_spherical)

        if cartesian.covariances is not None and (~np.all(cartesian.covariances.mask)):
            covariances_spherical = transform_covariances_jacobian(
                cartesian.values, cartesian.covariances, _cartesian_to_spherical
            )
        else:
            covariances_spherical = None

        coords = cls(
            rho=coords_spherical[:, 0],
            lon=coords_spherical[:, 1],
            lat=coords_spherical[:, 2],
            vrho=coords_spherical[:, 3],
            vlon=coords_spherical[:, 4],
            vlat=coords_spherical[:, 5],
            times=cartesian.times,
            covariances=covariances_spherical,
            origin=cartesian.origin,
            frame=cartesian.frame,
        )

        return coords

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        coord_cols: OrderedDict = SPHERICAL_COLS,
        origin_col: str = "origin",
        frame_col: str = "frame",
    ) -> "SphericalCoordinates":
        """
        Create a SphericalCoordinates class from a dataframe.

        Parameters
        ----------
        df : `~pandas.DataFrame`
            Pandas DataFrame containing spherical coordinates and optionally their
            times and covariances.
        coord_cols : OrderedDict
            Ordered dictionary containing as keys the coordinate dimensions and their equivalent columns
            as values. For example,
                coord_cols = OrderedDict()
                coord_cols["rho"] = Column name of radial distance values
                coord_cols["lon"] = Column name of longitudinal values
                coord_cols["rho"] = Column name of latitudinal values
                coord_cols["vrho"] = Column name of the radial velocity values
                coord_cols["vlon"] = Column name of longitudinal velocity values
                coord_cols["vlat"] = Column name of latitudinal velocity values
        origin_col : str
            Name of the column containing the origin of each coordinate.
        frame_col : str
            Name of the column containing the coordinate frame.
        """
        data = Coordinates._dict_from_df(
            df, coord_cols=coord_cols, origin_col=origin_col, frame_col=frame_col
        )
        return cls(**data)
