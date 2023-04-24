from typing import Optional, Type, Union

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

SPHERICAL_COLS = {}
SPHERICAL_UNITS = {}
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
        origin: str = "heliocenter",
        frame: str = "ecliptic",
        names: dict = SPHERICAL_COLS,
        units: dict = SPHERICAL_UNITS,
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
        """
        Radial distance
        """
        return self._values[:, 0]

    @rho.setter
    def rho(self, value):
        self._values[:, 0] = value
        self._values[:, 0].mask = False

    @rho.deleter
    def rho(self):
        self._values[:, 0] = np.nan
        self._values[:, 0].mask = True

    @property
    def lon(self):
        """
        Longitude
        """
        return self._values[:, 1]

    @lon.setter
    def lon(self, value):
        self._values[:, 1] = value
        self._values[:, 1].mask = False

    @lon.deleter
    def lon(self):
        self._values[:, 1] = np.nan
        self._values[:, 1].mask = True

    @property
    def lat(self):
        """
        Latitude
        """
        return self._values[:, 2]

    @lat.setter
    def lat(self, value):
        self._values[:, 2] = value
        self._values[:, 2].mask = False

    @lat.deleter
    def lat(self):
        self._values[:, 2] = np.nan
        self._values[:, 2].mask = True

    @property
    def vrho(self):
        """
        Radial velocity
        """
        return self._values[:, 3]

    @vrho.setter
    def vrho(self, value):
        self._values[:, 3] = value
        self._values[:, 3].mask = False

    @vrho.deleter
    def vrho(self):
        self._values[:, 3] = np.nan
        self._values[:, 3].mask = True

    @property
    def vlon(self):
        """
        Longitudinal velocity
        """
        return self._values[:, 4]

    @vlon.setter
    def vlon(self, value):
        self._values[:, 4] = value
        self._values[:, 4].mask = False

    @vlon.deleter
    def vlon(self):
        self._values[:, 4] = np.nan
        self._values[:, 4].mask = True

    @property
    def vlat(self):
        """
        Latitudinal velocity
        """
        return self._values[:, 5]

    @vlat.setter
    def vlat(self, value):
        self._values[:, 5] = value
        self._values[:, 5].mask = False

    @vlat.deleter
    def vlat(self):
        self._values[:, 5] = np.nan
        self._values[:, 5].mask = True

    @property
    def sigma_rho(self):
        """
        1-sigma uncertainty in radial distance
        """
        return self.sigmas[:, 0]

    @sigma_rho.setter
    def sigma_rho(self, value):
        self._covariances[:, 0, 0] = value**2
        self._covariances[:, 0, 0].mask = False

    @sigma_rho.deleter
    def sigma_rho(self):
        self._covariances[:, 0, 0] = np.nan
        self._covariances[:, 0, 0].mask = True

    @property
    def sigma_lon(self):
        """
        1-sigma uncertainty in longitude
        """
        return self.sigmas[:, 1]

    @sigma_lon.setter
    def sigma_lon(self, value):
        self._covariances[:, 1, 1] = value**2
        self._covariances[:, 1, 1].mask = False

    @sigma_lon.deleter
    def sigma_lon(self):
        self._covariances[:, 1, 1] = np.nan
        self._covariances[:, 1, 1].mask = True

    @property
    def sigma_lat(self):
        """
        1-sigma uncertainty in latitude
        """
        return self.sigmas[:, 2]

    @sigma_lat.setter
    def sigma_lat(self, value):
        self._covariances[:, 2, 2] = value**2
        self._covariances[:, 2, 2].mask = False

    @sigma_lat.deleter
    def sigma_lat(self):
        self._covariances[:, 2, 2] = np.nan
        self._covariances[:, 2, 2].mask = True

    @property
    def sigma_vrho(self):
        """
        1-sigma uncertainty in radial velocity
        """
        return self.sigmas[:, 3]

    @sigma_vrho.setter
    def sigma_vrho(self, value):
        self._covariances[:, 3, 3] = value**2
        self._covariances[:, 3, 3].mask = False

    @sigma_vrho.deleter
    def sigma_vrho(self):
        self._covariances[:, 3, 3] = np.nan
        self._covariances[:, 3, 3].mask = True

    @property
    def sigma_vlon(self):
        """
        1-sigma uncertainty in longitudinal velocity
        """
        return self.sigmas[:, 4]

    @sigma_vlon.setter
    def sigma_vlon(self, value):
        self._covariances[:, 4, 4] = value**2
        self._covariances[:, 4, 4].mask = False

    @sigma_vlon.deleter
    def sigma_vlon(self):
        self._covariances[:, 4, 4] = np.nan
        self._covariances[:, 4, 4].mask = True

    @property
    def sigma_vlat(self):
        """
        1-sigma uncertainty in latitudinal velocity
        """
        return self.sigmas[:, 5]

    @sigma_vlat.setter
    def sigma_vlat(self, value):
        self._covariances[:, 5, 5] = value**2
        self._covariances[:, 5, 5].mask = False

    @sigma_vlat.deleter
    def sigma_vlat(self):
        self._covariances[:, 5, 5] = np.nan
        self._covariances[:, 5, 5].mask = True

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
        cls: Type["SphericalCoordinates"],
        df: pd.DataFrame,
        coord_cols: dict = SPHERICAL_COLS,
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
        coord_cols : dict
            Dictionary containing as keys the coordinate dimensions and their equivalent columns
            as values. For example,
                coord_cols = {}
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
