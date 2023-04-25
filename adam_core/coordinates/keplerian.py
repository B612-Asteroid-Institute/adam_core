from typing import Optional, Type, Union

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

KEPLERIAN_COLS = {}
KEPLERIAN_UNITS = {}
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
        names: dict = KEPLERIAN_COLS,
        units: dict = KEPLERIAN_UNITS,
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

    @property
    def a(self):
        """
        Semi-major axis
        """
        return self._values[:, 0]

    @a.setter
    def a(self, value):
        self._values[:, 0] = value
        self._values[:, 0].mask = False

    @a.deleter
    def a(self):
        self._values[:, 0] = np.nan
        self._values[:, 0].mask = True

    @property
    def e(self):
        """
        Eccentricity
        """
        return self._values[:, 1]

    @e.setter
    def e(self, value):
        self._values[:, 1] = value
        self._values[:, 1].mask = False

    @e.deleter
    def e(self):
        self._values[:, 1] = np.nan
        self._values[:, 1].mask = True

    @property
    def i(self):
        """
        Inclination
        """
        return self._values[:, 2]

    @i.setter
    def i(self, value):
        self._values[:, 2] = value
        self._values[:, 2].mask = False

    @i.deleter
    def i(self):
        self._values[:, 2] = np.nan
        self._values[:, 2].mask = True

    @property
    def raan(self):
        """
        Right ascension of the ascending node
        """
        return self._values[:, 3]

    @raan.setter
    def raan(self, value):
        self._values[:, 3] = value
        self._values[:, 3].mask = False

    @raan.deleter
    def raan(self):
        self._values[:, 3] = np.nan
        self._values[:, 3].mask = True

    @property
    def ap(self):
        """
        Argument of periapsis
        """
        return self._values[:, 4]

    @ap.setter
    def ap(self, value):
        self._values[:, 4] = value
        self._values[:, 4].mask = False

    @ap.deleter
    def ap(self):
        self._values[:, 4] = np.nan
        self._values[:, 4].mask = True

    @property
    def M(self):
        """
        Mean anomaly
        """
        return self._values[:, 5]

    @M.setter
    def M(self, value):
        self._values[:, 5] = value
        self._values[:, 5].mask = False

    @M.deleter
    def M(self):
        self._values[:, 5] = np.nan
        self._values[:, 5].mask = True

    @property
    def sigma_a(self):
        """
        1-sigma uncertainty in semi-major axis
        """
        return self.sigmas[:, 0]

    @sigma_a.setter
    def sigma_a(self, value):
        self._covariances[:, 0, 0] = value**2
        self._covariances[:, 0, 0].mask = False

    @sigma_a.deleter
    def sigma_a(self):
        self._covariances[:, 0, 0] = np.nan
        self._covariances[:, 0, 0].mask = True

    @property
    def sigma_e(self):
        """
        1-sigma uncertainty in eccentricity
        """
        return self.sigmas[:, 1]

    @sigma_e.setter
    def sigma_e(self, value):
        self._covariances[:, 1, 1] = value**2
        self._covariances[:, 1, 1].mask = False

    @sigma_e.deleter
    def sigma_e(self):
        self._covariances[:, 1, 1] = np.nan
        self._covariances[:, 1, 1].mask = True

    @property
    def sigma_i(self):
        """
        1-sigma uncertainty in inclination
        """
        return self.sigmas[:, 2]

    @sigma_i.setter
    def sigma_i(self, value):
        self._covariances[:, 2, 2] = value**2
        self._covariances[:, 2, 2].mask = False

    @sigma_i.deleter
    def sigma_i(self):
        self._covariances[:, 2, 2] = np.nan
        self._covariances[:, 2, 2].mask = True

    @property
    def sigma_raan(self):
        """
        1-sigma uncertainty in right ascension of the ascending node
        """
        return self.sigmas[:, 3]

    @sigma_raan.setter
    def sigma_raan(self, value):
        self._covariances[:, 3, 3] = value**2
        self._covariances[:, 3, 3].mask = False

    @sigma_raan.deleter
    def sigma_raan(self):
        self._covariances[:, 3, 3] = np.nan
        self._covariances[:, 3, 3].mask = True

    @property
    def sigma_ap(self):
        """
        1-sigma uncertainty in argument of periapsis
        """
        return self.sigmas[:, 4]

    @sigma_ap.setter
    def sigma_ap(self, value):
        self._covariances[:, 4, 4] = value**2
        self._covariances[:, 4, 4].mask = False

    @sigma_ap.deleter
    def sigma_ap(self):
        self._covariances[:, 4, 4] = np.nan
        self._covariances[:, 4, 4].mask = True

    @property
    def sigma_M(self):
        """
        1-sigma uncertainty in mean anomaly
        """
        return self.sigmas[:, 5]

    @sigma_M.setter
    def sigma_M(self, value):
        self._covariances[:, 5, 5] = value**2
        self._covariances[:, 5, 5].mask = False

    @sigma_M.deleter
    def sigma_M(self):
        self._covariances[:, 5, 5] = np.nan
        self._covariances[:, 5, 5].mask = True

    @property
    def q(self):
        """
        Periapsis distance
        """
        return self.a * (1 - self.e)

    @q.setter
    def q(self, value):
        err = (
            "Cannot set periapsis distance (q) as it is"
            " derived from semi-major axis (a) and eccentricity (e)."
        )
        raise ValueError(err)

    @q.deleter
    def q(self):
        err = (
            "Cannot delete periapsis distance (q) as it is"
            " derived from semi-major axis (a) and eccentricity (e)."
        )
        raise ValueError(err)

    @property
    def Q(self):
        """
        Apoapsis distance
        """
        return self.a * (1 + self.e)

    @Q.setter
    def Q(self, value):
        err = (
            "Cannot set apoapsis distance (Q) as it is"
            " derived from semi-major axis (a) and eccentricity (e)."
        )
        raise ValueError(err)

    @Q.deleter
    def Q(self):
        err = (
            "Cannot delete apoapsis distance (Q) as it is"
            " derived from semi-major axis (a) and eccentricity (e)."
        )
        raise ValueError(err)

    @property
    def p(self):
        """
        Semi-latus rectum
        """
        return self.a / (1 - self.e**2)

    @p.setter
    def p(self, value):
        err = (
            "Cannot set semi-latus rectum (p) as it is"
            " derived from semi-major axis (a) and eccentricity (e)."
        )
        raise ValueError(err)

    @p.deleter
    def p(self):
        err = (
            "Cannot delete semi-latus rectum (p) as it is"
            " derived from semi-major axis (a) and eccentricity (e)."
        )
        raise ValueError(err)

    @property
    def P(self):
        """
        Period
        """
        return np.sqrt(4 * np.pi**2 * self.a**3 / self.mu)

    @P.setter
    def P(self, value):
        err = (
            "Cannot set period (P) as it is"
            " derived from semi-major axis (a) and gravitational parameter (mu)."
        )
        raise ValueError(err)

    @p.deleter
    def P(self):
        err = (
            "Cannot delete period (P) as it is"
            " derived from semi-major axis (a) and gravitational parameter (mu)."
        )
        raise ValueError(err)

    @property
    def mu(self):
        """
        Gravitational parameter
        """
        return self._mu

    @mu.setter
    def mu(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError(
                "Gravitational parameter (mu) should be an integer or a float."
            )
        self._mu = value

    @mu.deleter
    def mu(self):
        err = "Gravitational parameter (mu) cannot be deleted. Please set it to the desired value."
        raise ValueError(err)

    def to_cartesian(self) -> CartesianCoordinates:
        from .transform import _keplerian_to_cartesian_a, keplerian_to_cartesian

        coords_cartesian = keplerian_to_cartesian(
            self.values.filled(),
            mu=self.mu,
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
        cls: Type["KeplerianCoordinates"],
        df: pd.DataFrame,
        coord_cols: dict = KEPLERIAN_COLS,
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
        coord_cols : dict
            Dictionary containing as keys the coordinate dimensions and their equivalent columns
            as values. For example,
                coord_cols = {}
                coord_cols["a"] = Column name of semi-major axis values
                coord_cols["e"] = Column name of eccentricity values
                coord_cols["i"] = Column name of inclination values
                coord_cols["raan"] = Column name of longitude of ascending node values
                coord_cols["ap"] = Column name of argument of periapsis values
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
