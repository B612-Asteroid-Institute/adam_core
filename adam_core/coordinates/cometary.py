from typing import Optional, Union

import jax.numpy as jnp
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
    "CometaryCoordinates",
    "COMETARY_COLS",
    "COMETARY_UNITS",
]

COMETARY_COLS = {}
COMETARY_UNITS = {}
for i in ["q", "e", "i", "raan", "ap", "tp"]:
    COMETARY_COLS[i] = i
COMETARY_UNITS["q"] = u.au
COMETARY_UNITS["e"] = u.dimensionless_unscaled
COMETARY_UNITS["i"] = u.deg
COMETARY_UNITS["raan"] = u.deg
COMETARY_UNITS["ap"] = u.deg
COMETARY_UNITS["tp"] = u.d

MU = c.MU
Z_AXIS = jnp.array([0.0, 0.0, 1.0])


class CometaryCoordinates(Coordinates):
    def __init__(
        self,
        q: Optional[Union[int, float, np.ndarray]] = None,
        e: Optional[Union[int, float, np.ndarray]] = None,
        i: Optional[Union[int, float, np.ndarray]] = None,
        raan: Optional[Union[int, float, np.ndarray]] = None,
        ap: Optional[Union[int, float, np.ndarray]] = None,
        tp: Optional[Union[int, float, np.ndarray]] = None,
        times: Optional[Time] = None,
        covariances: Optional[np.ndarray] = None,
        sigma_q: Optional[np.ndarray] = None,
        sigma_e: Optional[np.ndarray] = None,
        sigma_i: Optional[np.ndarray] = None,
        sigma_raan: Optional[np.ndarray] = None,
        sigma_ap: Optional[np.ndarray] = None,
        sigma_tp: Optional[np.ndarray] = None,
        origin: str = "heliocenter",
        frame: str = "ecliptic",
        names: dict = COMETARY_COLS,
        units: dict = COMETARY_UNITS,
        mu: float = MU,
    ):
        sigmas = (sigma_q, sigma_e, sigma_i, sigma_raan, sigma_ap, sigma_tp)
        Coordinates.__init__(
            self,
            q=q,
            e=e,
            i=i,
            raan=raan,
            ap=ap,
            tp=tp,
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
    def q(self):
        """
        Periapsis distance
        """
        return self._values[:, 0]

    @q.setter
    def q(self, value):
        self._values[:, 0] = value
        self._values[:, 0].mask = False

    @q.deleter
    def q(self):
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
    def tp(self):
        """
        Time of periapse passage
        """
        return self._values[:, 5]

    @tp.setter
    def tp(self, value):
        self._values[:, 5] = value
        self._values[:, 5].mask = False

    @tp.deleter
    def tp(self):
        self._values[:, 5] = np.nan
        self._values[:, 5].mask = True

    @property
    def sigma_q(self):
        """
        1-sigma uncertainty in periapsis distance
        """
        return self.sigmas[:, 0]

    @sigma_q.setter
    def sigma_q(self, value):
        self._covariances[:, 0, 0] = value**2
        self._covariances[:, 0, 0].mask = False

    @sigma_q.deleter
    def sigma_q(self):
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
    def sigma_tp(self):
        """
        1-sigma uncertainty in time of periapse passage
        """
        return self.sigmas[:, 5]

    @sigma_tp.setter
    def sigma_tp(self, value):
        self._covariances[:, 5, 5] = value**2
        self._covariances[:, 5, 5].mask = False

    @sigma_tp.deleter
    def sigma_tp(self):
        self._covariances[:, 5, 5] = np.nan
        self._covariances[:, 5, 5].mask = True

    @property
    def a(self):
        """
        Semi-major axis
        """
        return self.q / (1 - self.e)

    @a.setter
    def a(self, value):
        err = (
            "Cannot set semi-major axis (a) as it is"
            " derived from the periapsis distance (q) and eccentricity (e)."
        )
        raise ValueError(err)

    @a.deleter
    def a(self):
        err = (
            "Cannot delete semi-major axis (a) as it is"
            " derived from the periapsis distance (q) and eccentricity (e)."
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
            " derived from the semi-major axis (a) and eccentricity (e)."
        )
        raise ValueError(err)

    @Q.deleter
    def Q(self):
        err = (
            "Cannot delete apoapsis distance (Q) as it is"
            " derived from the semi-major axis (a) and eccentricity (e)."
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
            " derived from the semi-major axis (a) and eccentricity (e)."
        )
        raise ValueError(err)

    @p.deleter
    def p(self):
        err = (
            "Cannot delete semi-latus rectum (p) as it is"
            " derived from the semi-major axis (a) and eccentricity (e)."
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
            " derived from the semi-major axis (a) and gravitational parameter (mu)."
        )
        raise ValueError(err)

    @p.deleter
    def P(self):
        err = (
            "Cannot delete period (P) as it is"
            " derived from the semi-major axis (a) and gravitational parameter (mu)."
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
        from .transform import _cometary_to_cartesian, cometary_to_cartesian

        if self.times is None:
            err = (
                "To convert Cometary coordinates to Cartesian coordinates, the times\n"
                "at which the Coordinates coordinates are defined is required to give\n"
                "the time of periapsis passage context."
            )
            raise ValueError(err)

        coords_cartesian = cometary_to_cartesian(
            self.values.filled(),
            t0=self.times.tdb.mjd,
            mu=self.mu,
            max_iter=100,
            tol=1e-15,
        )
        coords_cartesian = np.array(coords_cartesian)

        if self.covariances is not None:
            covariances_cartesian = transform_covariances_jacobian(
                self.values,
                self.covariances,
                _cometary_to_cartesian,
                in_axes=(0, 0, None, None, None),
                out_axes=0,
                t0=self.times.tdb.mjd,
                mu=self.mu,
                max_iter=100,
                tol=1e-15,
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
    def from_cartesian(
        cls, cartesian: CartesianCoordinates, mu: float = MU
    ) -> "CometaryCoordinates":
        from .transform import _cartesian_to_cometary, cartesian_to_cometary

        if cartesian.times is None:
            err = (
                "To convert Cometary coordinates to Cartesian coordinates, the times\n"
                "at which the Cartesian coordinates are defined is required to calculate\n"
                "the time of periapsis passage."
            )
            raise ValueError(err)

        coords_cometary = cartesian_to_cometary(
            cartesian.values.filled(),
            cartesian.times.tdb.mjd,
            mu=mu,
        )
        coords_cometary = np.array(coords_cometary)

        if cartesian.covariances is not None and (~np.all(cartesian.covariances.mask)):
            covariances_cometary = transform_covariances_jacobian(
                cartesian.values,
                cartesian.covariances,
                _cartesian_to_cometary,
                in_axes=(0, 0, None),
                out_axes=0,
                t0=cartesian.times.tdb.mjd,
                mu=mu,
            )
        else:
            covariances_cometary = None

        coords = cls(
            q=coords_cometary[:, 0],
            e=coords_cometary[:, 1],
            i=coords_cometary[:, 2],
            raan=coords_cometary[:, 3],
            ap=coords_cometary[:, 4],
            tp=coords_cometary[:, 5],
            times=cartesian.times,
            covariances=covariances_cometary,
            origin=cartesian.origin,
            frame=cartesian.frame,
            mu=mu,
        )

        return coords

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        coord_cols: dict = COMETARY_COLS,
        origin_col: str = "origin",
        frame_col: str = "frame",
    ) -> "CometaryCoordinates":
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
                coord_cols["q"] = Column name of periapsis distance values
                coord_cols["e"] = Column name of eccentricity values
                coord_cols["i"] = Column name of inclination values
                coord_cols["raan"] = Column name of longitude of ascending node values
                coord_cols["ap"] = Column name of argument of periapsis values
                coord_cols["tp"] = Column name of time of periapsis passage values.
        origin_col : str
            Name of the column containing the origin of each coordinate.
        frame_col : str
            Name of the column containing the coordinate frame.
        """
        data = Coordinates._dict_from_df(
            df, coord_cols=coord_cols, origin_col=origin_col, frame_col=frame_col
        )
        return cls(**data)
