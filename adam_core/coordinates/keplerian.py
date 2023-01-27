from collections import OrderedDict
from typing import Optional, Union

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time
from jax import config

from ..constants import Constants as c
from .cartesian import CartesianCoordinates
from .coordinates import Coordinates
from .covariances import transform_covariances_jacobian

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

__all__ = [
    "KeplerianCoordinates",
    "KEPLERIAN_COLS",
    "KEPLERIAN_UNITS",
]

KEPLERIAN_COLS = OrderedDict()
KEPLERIAN_UNITS = OrderedDict()
for i in ["a", "e", "i", "raan", "ap", "M"]:
    KEPLERIAN_COLS[i] = i
KEPLERIAN_UNITS["a"] = u.au
KEPLERIAN_UNITS["e"] = u.dimensionless_unscaled
KEPLERIAN_UNITS["i"] = u.deg
KEPLERIAN_UNITS["raan"] = u.deg
KEPLERIAN_UNITS["ap"] = u.deg
KEPLERIAN_UNITS["M"] = u.deg

MU = c.MU


class KeplerianCoordinates(Coordinates):
    def __init__(
        self,
        a: Optional[Union[int, float, np.ndarray]] = None,
        e: Optional[Union[int, float, np.ndarray]] = None,
        i: Optional[Union[int, float, np.ndarray]] = None,
        raan: Optional[Union[int, float, np.ndarray]] = None,
        ap: Optional[Union[int, float, np.ndarray]] = None,
        M: Optional[Union[int, float, np.ndarray]] = None,
        times: Optional[Time] = None,
        covariances: Optional[np.ndarray] = None,
        sigma_a: Optional[np.ndarray] = None,
        sigma_e: Optional[np.ndarray] = None,
        sigma_i: Optional[np.ndarray] = None,
        sigma_raan: Optional[np.ndarray] = None,
        sigma_ap: Optional[np.ndarray] = None,
        sigma_M: Optional[np.ndarray] = None,
        origin: str = "heliocenter",
        frame: str = "ecliptic",
        names: OrderedDict = KEPLERIAN_COLS,
        units: OrderedDict = KEPLERIAN_UNITS,
        mu: float = MU,
    ):
        sigmas = (sigma_a, sigma_e, sigma_i, sigma_raan, sigma_ap, sigma_M)
        Coordinates.__init__(
            self,
            a=a,
            e=e,
            i=i,
            raan=raan,
            ap=ap,
            M=M,
            covariances=covariances,
            sigmas=sigmas,
            times=times,
            origin=origin,
            frame=frame,
            names=names,
            units=units,
        )
        self._mu = mu

        return

    def a(self):
        """
        Semi-major axis
        """
        return self._values[:, 0]

    def e(self):
        """
        Eccentricity
        """
        return self._values[:, 1]

    def i(self):
        """
        Inclination
        """
        return self._values[:, 2]

    def raan(self):
        """
        Right ascension of the ascending node
        """
        return self._values[:, 3]

    def ap(self):
        """
        Argument of periapsis
        """
        return self._values[:, 4]

    def M(self):
        """
        Mean anomaly
        """
        return self._values[:, 5]

    def sigma_a(self):
        """
        1-sigma uncertainty in semi-major axis
        """
        return self.sigmas[:, 0]

    def sigma_e(self):
        """
        1-sigma uncertainty in eccentricity
        """
        return self.sigmas[:, 1]

    def sigma_i(self):
        """
        1-sigma uncertainty in inclination
        """
        return self.sigmas[:, 2]

    def sigma_raan(self):
        """
        1-sigma uncertainty in right ascension of the ascending node
        """
        return self.sigmas[:, 3]

    def sigma_ap(self):
        """
        1-sigma uncertainty in argument of periapsis
        """
        return self.sigmas[:, 4]

    def sigma_M(self):
        """
        1-sigma uncertainty in mean anomaly
        """
        return self.sigmas[:, 5]

    def q(self):
        """
        Periapsis distance
        """
        return self.a * (1 - self.e)

    def Q(self):
        """
        Apoapsis distance
        """
        return self.a * (1 + self.e)

    def p(self):
        """
        Semi-latus rectum
        """
        return self.a / (1 - self.e**2)

    def P(self):
        """
        Period
        """
        return np.sqrt(4 * np.pi**2 * self.a**3 / self.mu)

    def mu(self):
        """
        Gravitational parameter
        """
        return self._mu

    def to_cartesian(self) -> CartesianCoordinates:

        from .transform import _keplerian_to_cartesian_a, keplerian_to_cartesian

        coords_cartesian = keplerian_to_cartesian(
            self.values.filled(),
            mu=MU,
            max_iter=1000,
            tol=1e-15,
        )
        coords_cartesian = np.array(coords_cartesian)

        if self.covariances is not None:
            covariances_cartesian = transform_covariances_jacobian(
                self.values, self.covariances, _keplerian_to_cartesian_a
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
    def from_cartesian(cls, cartesian: CartesianCoordinates, mu: float = MU):

        from .transform import _cartesian_to_keplerian6, cartesian_to_keplerian

        coords_keplerian = cartesian_to_keplerian(
            cartesian.values.filled(),
            cartesian.times.tdb.mjd,
            mu=mu,
        )
        coords_keplerian = np.array(coords_keplerian)

        if cartesian.covariances is not None and (~np.all(cartesian.covariances.mask)):
            covariances_keplerian = transform_covariances_jacobian(
                cartesian.values,
                cartesian.covariances,
                _cartesian_to_keplerian6,
                in_axes=(0, 0, None),
                out_axes=0,
                t0=cartesian.times.tdb.mjd,
                mu=mu,
            )
        else:
            covariances_keplerian = None

        coords = cls(
            a=coords_keplerian[:, 0],
            e=coords_keplerian[:, 4],
            i=coords_keplerian[:, 5],
            raan=coords_keplerian[:, 6],
            ap=coords_keplerian[:, 7],
            M=coords_keplerian[:, 8],
            times=cartesian.times,
            covariances=covariances_keplerian,
            origin=cartesian.origin,
            frame=cartesian.frame,
            mu=mu,
        )

        return coords

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        coord_cols: OrderedDict = KEPLERIAN_COLS,
        origin_col: str = "origin",
        frame_col: str = "frame",
    ) -> "KeplerianCoordinates":
        """
        Create a KeplerianCoordinates class from a dataframe.

        Parameters
        ----------
        df : `~pandas.DataFrame`
            Pandas DataFrame containing Keplerian coordinates and optionally their
            times and covariances.
        coord_cols : OrderedDict
            Ordered dictionary containing as keys the coordinate dimensions and their equivalent columns
            as values. For example,
                coord_cols = OrderedDict()
                coord_cols["a"] = Column name of semi-major axis values
                coord_cols["e"] = Column name of eccentricity values
                coord_cols["i"] = Column name of inclination values
                coord_cols["raan"] = Column name of longitude of ascending node values
                coord_cols["ap"] = Column name of argument of pericenter values
                coord_cols["M"] = Column name of mean anomaly values
        origin_col : str
            Name of the column containing the origin of each coordinate.
        frame_col : str
            Name of the column containing the coordinate frame.
        """
        data = Coordinates._dict_from_df(
            df, coord_cols=coord_cols, origin_col=origin_col, frame_col=frame_col
        )
        return cls(**data)
