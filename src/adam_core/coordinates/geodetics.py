from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pyarrow.compute as pc
import quivr as qv

from ..constants import KM_P_AU
from ..time import Timestamp
from . import cartesian
from .covariances import CoordinateCovariances, transform_covariances_jacobian
from .origin import Origin

__all__ = [
    "GeodeticCoordinates",
]


@dataclass
class GeodeticConstants:
    """
    Geodetic constants for the Earth.

    Parameters
    ----------
    a : float
        Semi-major axis of the Earth in units of distance.
    b : float
        Semi-minor axis of the Earth in units of distance.
    f : float
        Flattening of the Earth.
    """

    a: float
    b: float
    f: float


WGS84 = GeodeticConstants(
    a=6378137.0 / KM_P_AU / 1000,
    b=6356752.31424518 / KM_P_AU / 1000,
    f=1.0 / 298.257223563,
)


class GeodeticCoordinates(qv.Table):

    alt = qv.Float64Column(nullable=True)
    lon = qv.Float64Column(nullable=True)
    lat = qv.Float64Column(nullable=True)
    vup = qv.Float64Column(nullable=True)
    veast = qv.Float64Column(nullable=True)
    vnorth = qv.Float64Column(nullable=True)
    time = Timestamp.as_column(nullable=True)
    covariance = CoordinateCovariances.as_column(nullable=True)
    origin = Origin.as_column()
    frame = qv.StringAttribute(default="unspecified")

    @property
    def values(self) -> np.ndarray:
        return np.array(
            self.table.select(["alt", "lon", "lat", "vup", "veast", "vnorth"])
        )

    @property
    def sigma_alt(self):
        """
        1-sigma uncertainty in altitude.
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
    def sigma_vup(self):
        """
        1-sigma uncertainty in up velocity.
        """
        return self.covariance.sigmas[:, 3]

    @property
    def sigma_veast(self):
        """
        1-sigma uncertainty in east velocity.
        """
        return self.covariance.sigmas[:, 4]

    @property
    def sigma_vnorth(self):
        """
        1-sigma uncertainty in north velocity.
        """
        return self.covariance.sigmas[:, 5]

    def google_maps_url(self, zoom: int = 18) -> List[str]:
        """
        Generate a Google Maps URL for the coordinates.
        """
        urls = []
        for lat, lon in zip(self.lat, self.lon):
            urls.append(
                f"https://www.google.com/maps/@{lat.as_py()},{lon.as_py() - 360},{zoom}z"
            )
        return urls

    @classmethod
    def from_cartesian(
        cls, cartesian: cartesian.CartesianCoordinates
    ) -> "GeodeticCoordinates":
        from .transform import _cartesian_to_geodetic, cartesian_to_geodetic

        assert (
            cartesian.frame == "itrf93"
        ), "Cartesian coordinates must be in ITRF93 frame"
        assert pc.all(
            pc.equal(cartesian.origin.code, "EARTH")
        ), "Cartesian coordinates must be in Earth-centered frame"

        coords_geodetic = cartesian_to_geodetic(cartesian.values, a=WGS84.a, f=WGS84.f)
        coords_geodetic = np.array(coords_geodetic)

        if not cartesian.covariance.is_all_nan():
            cartesian_covariances = cartesian.covariance.to_matrix()
            covariances_geodetic = transform_covariances_jacobian(
                cartesian.values, cartesian_covariances, _cartesian_to_geodetic
            )
        else:
            covariances_geodetic = np.empty(
                (len(coords_geodetic), 6, 6), dtype=np.float64
            )
            covariances_geodetic.fill(np.nan)

        covariances_geodetic = CoordinateCovariances.from_matrix(covariances_geodetic)
        coords = cls.from_kwargs(
            alt=coords_geodetic[:, 0],
            lon=coords_geodetic[:, 1],
            lat=coords_geodetic[:, 2],
            vup=coords_geodetic[:, 3],
            veast=coords_geodetic[:, 4],
            vnorth=coords_geodetic[:, 5],
            time=cartesian.time,
            covariance=covariances_geodetic,
            origin=cartesian.origin,
            frame=cartesian.frame,
        )

        return coords
