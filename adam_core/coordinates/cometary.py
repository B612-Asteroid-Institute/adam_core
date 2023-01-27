from collections import OrderedDict
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

COMETARY_COLS = OrderedDict()
COMETARY_UNITS = OrderedDict()
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
        names: OrderedDict = COMETARY_COLS,
        units: OrderedDict = COMETARY_UNITS,
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
        return self._values[:, 0]

    @property
    def e(self):
        return self._values[:, 1]

    @property
    def i(self):
        return self._values[:, 2]

    @property
    def raan(self):
        return self._values[:, 3]

    @property
    def ap(self):
        return self._values[:, 4]

    @property
    def tp(self):
        return self._values[:, 5]

    @property
    def sigma_q(self):
        return self.sigmas[:, 0]

    @property
    def sigma_e(self):
        return self.sigmas[:, 1]

    @property
    def sigma_i(self):
        return self.sigmas[:, 2]

    @property
    def sigma_raan(self):
        return self.sigmas[:, 3]

    @property
    def sigma_ap(self):
        return self.sigmas[:, 4]

    @property
    def sigma_tp(self):
        return self.sigmas[:, 5]

    @property
    def a(self):
        # pericenter distance
        return self.q / (1 - self.e)

    @property
    def p(self):
        # apocenter distance
        return self.a * (1 + self.e)

    @property
    def mu(self):
        return self._mu

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
        coord_cols: OrderedDict = COMETARY_COLS,
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
        coord_cols : OrderedDict
            Ordered dictionary containing as keys the coordinate dimensions and their equivalent columns
            as values. For example,
                coord_cols = OrderedDict()
                coord_cols["q"] = Column name of pericenter distance values
                coord_cols["e"] = Column name of eccentricity values
                coord_cols["i"] = Column name of inclination values
                coord_cols["raan"] = Column name of longitude of ascending node values
                coord_cols["ap"] = Column name of argument of pericenter values
                coord_cols["tp"] = Column name of time of pericenter passage values.
        origin_col : str
            Name of the column containing the origin of each coordinate.
        frame_col : str
            Name of the column containing the coordinate frame.
        """
        data = Coordinates._dict_from_df(
            df, coord_cols=coord_cols, origin_col=origin_col, frame_col=frame_col
        )
        return cls(**data)
