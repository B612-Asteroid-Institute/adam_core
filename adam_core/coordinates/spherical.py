from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import pandas as pd
from astropy import units as u
from quivr import Float64Column, StringAttribute, Table

from .cartesian import CartesianCoordinates
from .covariances import CoordinateCovariances, transform_covariances_jacobian
from .io import coords_from_dataframe, coords_to_dataframe
from .origin import Origin
from .times import Times

if TYPE_CHECKING:
    from .cometary import CometaryCoordinates
    from .keplerian import KeplerianCoordinates


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


class SphericalCoordinates(Table):

    rho = Float64Column(nullable=True)
    lon = Float64Column(nullable=True)
    lat = Float64Column(nullable=True)
    vrho = Float64Column(nullable=True)
    vlon = Float64Column(nullable=True)
    vlat = Float64Column(nullable=True)
    time = Times.as_column(nullable=True)
    covariance = CoordinateCovariances.as_column(nullable=True)
    origin = Origin.as_column()
    frame = StringAttribute()

    @property
    def values(self) -> np.ndarray:
        return np.array(
            self.table.select(["rho", "lon", "lat", "vrho", "vlon", "vlat"])
        )

    @property
    def sigma_rho(self):
        """
        1-sigma uncertainty in radial distance.
        """
        return self.covariance.sigmas[:, 0]

    @property
    def sigma_lon(self):
        """
        1-sigma uncertainty in longitude.
        """
        return self.covariance.sigmas[:, 1]

    @property
    def sigma_lat(self):
        """
        1-sigma uncertainty in latitude.
        """
        return self.covariance.sigmas[:, 2]

    @property
    def sigma_vrho(self):
        """
        1-sigma uncertainty in radial velocity.
        """
        return self.covariance.sigmas[:, 3]

    @property
    def sigma_vlon(self):
        """
        1-sigma uncertainty in longitudinal velocity.
        """
        return self.covariance.sigmas[:, 4]

    @property
    def sigma_vlat(self):
        """
        1-sigma uncertainty in latitudinal velocity.
        """
        return self.covariance.sigmas[:, 5]

    def to_unit_sphere(self, only_missing: bool = False) -> "SphericalCoordinates":
        """
        Convert to unit sphere. By default, all coordinates will have their rho values
        set to 1.0 and their vrho values set to 0.0. If only_missing is True, then only
        coordinates that have NaN values for rho will be set to 1.0 and coordinates that
        have NaN values for vrho will be set to 0.0.

        TODO: We could look at scaling the uncertainties as well, but this is not currently
        implemented nor probably necessary. This function will mostly be used to convert
        SphericalCoordinates that have missing radial distances to cartesian coordinates on a
        unit sphere.

        Parameters
        ----------
        only_missing : bool, optional
            If True, then only coordinates that have NaN values for rho will be set to 1.0 and
            coordinates that have NaN values for vrho will be set to 0.0. If False, then all
            coordinates will be set to 1.0 and 0.0, respectively. The default is False.

        Returns
        -------
        SphericalCoordinates
            Spherical coordinates on a unit sphere, with rho and vrho set to 1.0 and 0.0, respectively.
        """
        # Extract coordinate values
        coords = self.values

        # Set rho to 1.0 for all points that are NaN, or if force is True
        # then set rho to 1.0 for all points
        if not only_missing:
            mask = np.ones(len(coords), dtype=bool)
        else:
            mask = np.isnan(coords[:, 0])

        coords[mask, 0] = 1.0

        # Set vrho to 0.0 for all points that are NaN, or if force is True
        # then set vrho to 0.0 for all points
        if not only_missing:
            mask = np.ones(len(coords), dtype=bool)
        else:
            mask = np.isnan(coords[:, 3])

        coords[mask, 3] = 0.0

        # Convert back to spherical coordinates
        return SphericalCoordinates.from_kwargs(
            rho=coords[:, 0],
            lon=coords[:, 1],
            lat=coords[:, 2],
            vrho=coords[:, 3],
            vlon=coords[:, 4],
            vlat=coords[:, 5],
            time=self.time,
            covariance=self.covariance,
            origin=self.origin,
            frame=self.frame,
        )

    def to_cartesian(self) -> CartesianCoordinates:
        from .transform import _spherical_to_cartesian, spherical_to_cartesian

        coords_cartesian = spherical_to_cartesian(self.values)
        coords_cartesian = np.array(coords_cartesian)

        if not self.covariance.is_all_nan():
            covariances_spherical = self.covariance.to_matrix()
            covariances_cartesian = transform_covariances_jacobian(
                self.values, covariances_spherical, _spherical_to_cartesian
            )
        else:
            covariances_cartesian = np.empty(
                (len(coords_cartesian), 6, 6), dtype=np.float64
            )
            covariances_cartesian.fill(np.nan)

        covariances_cartesian = CoordinateCovariances.from_matrix(covariances_cartesian)
        coords = CartesianCoordinates.from_kwargs(
            x=coords_cartesian[:, 0],
            y=coords_cartesian[:, 1],
            z=coords_cartesian[:, 2],
            vx=coords_cartesian[:, 3],
            vy=coords_cartesian[:, 4],
            vz=coords_cartesian[:, 5],
            time=self.time,
            covariance=covariances_cartesian,
            origin=self.origin,
            frame=self.frame,
        )
        return coords

    @classmethod
    def from_cartesian(cls, cartesian: CartesianCoordinates) -> "SphericalCoordinates":
        from .transform import _cartesian_to_spherical, cartesian_to_spherical

        coords_spherical = cartesian_to_spherical(cartesian.values)
        coords_spherical = np.array(coords_spherical)

        if not cartesian.covariance.is_all_nan():
            cartesian_covariances = cartesian.covariance.to_matrix()
            covariances_spherical = transform_covariances_jacobian(
                cartesian.values, cartesian_covariances, _cartesian_to_spherical
            )
        else:
            covariances_spherical = np.empty(
                (len(coords_spherical), 6, 6), dtype=np.float64
            )
            covariances_spherical.fill(np.nan)

        covariances_spherical = CoordinateCovariances.from_matrix(covariances_spherical)
        coords = cls.from_kwargs(
            rho=coords_spherical[:, 0],
            lon=coords_spherical[:, 1],
            lat=coords_spherical[:, 2],
            vrho=coords_spherical[:, 3],
            vlon=coords_spherical[:, 4],
            vlat=coords_spherical[:, 5],
            time=cartesian.time,
            covariance=covariances_spherical,
            origin=cartesian.origin,
            frame=cartesian.frame,
        )

        return coords

    def to_cometary(self) -> "CometaryCoordinates":
        from .cometary import CometaryCoordinates

        return CometaryCoordinates.from_cartesian(self.to_cartesian())

    @classmethod
    def from_cometary(
        cls, cometary_coordinates: "CometaryCoordinates"
    ) -> "SphericalCoordinates":
        return cls.from_cartesian(cometary_coordinates.to_cartesian())

    def to_keplerian(self) -> "KeplerianCoordinates":
        from .keplerian import KeplerianCoordinates

        return KeplerianCoordinates.from_cartesian(self.to_cartesian())

    @classmethod
    def from_keplerian(
        cls, keplerian_coordinates: "KeplerianCoordinates"
    ) -> "SphericalCoordinates":
        return cls.from_cartesian(keplerian_coordinates.to_cartesian())

    @classmethod
    def from_spherical(
        cls, spherical_coordinates: "SphericalCoordinates"
    ) -> "SphericalCoordinates":
        return spherical_coordinates

    def to_dataframe(
        self, sigmas: Optional[bool] = None, covariances: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Convert coordinates to a pandas DataFrame.

        Parameters
        ----------
        sigmas : bool, optional
            If None, will only include sigmas if they are not null.
            If True, include 1-sigma uncertainties in the DataFrame. If False, do not include
            sigmas.
        covariances : bool, optional
            If None, will only include covariances if they are not null.
            If True, include covariance matrices in the DataFrame. Covariance matrices
            will be split into 21 columns, with the lower triangular elements stored. If False,
            do not include covariances.


        Returns
        -------
        df : `~pandas.Dataframe`
            DataFrame containing coordinates.
        """
        return coords_to_dataframe(
            self,
            ["rho", "lon", "lat", "vrho", "vlon", "vlat"],
            sigmas=sigmas,
            covariances=covariances,
        )

    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, frame: Literal["ecliptic", "equatorial"]
    ) -> "SphericalCoordinates":
        """
        Create coordinates from a pandas DataFrame.

        Parameters
        ----------
        df : `~pandas.Dataframe`
            DataFrame containing coordinates.
        frame : {"ecliptic", "equatorial"}
            Frame in which coordinates are defined.

        Returns
        -------
        coords : `~adam_core.coordinates.spherical.SphericalCoordinates`
            Spherical coordinates.
        """
        return coords_from_dataframe(
            cls,
            df,
            coord_names=["rho", "lon", "lat", "vrho", "vlon", "vlat"],
            frame=frame,
        )
