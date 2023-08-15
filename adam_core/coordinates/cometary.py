from typing import TYPE_CHECKING, Optional

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
    from .keplerian import KeplerianCoordinates
    from .spherical import SphericalCoordinates


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


class CometaryCoordinates(Table):
    # TODO: Time of periapse passage could perhaps be represented
    # as a Times object. We could then modify self.values to only
    # grab the MJD column. That said, we would want to force it
    # the time scale to be in TDB..

    q = Float64Column()
    e = Float64Column()
    i = Float64Column()
    raan = Float64Column()
    ap = Float64Column()
    tp = Float64Column()
    time = Times.as_column(nullable=True)
    covariance = CoordinateCovariances.as_column(nullable=True)
    origin = Origin.as_column()
    frame = StringAttribute()

    @property
    def values(self) -> np.ndarray:
        return np.array(self.table.select(["q", "e", "i", "raan", "ap", "tp"])).T

    @property
    def sigma_q(self) -> np.ndarray:
        """
        1-sigma uncertainty in periapsis distance.
        """
        return self.covariance.sigmas[:, 0]

    @property
    def sigma_e(self) -> np.ndarray:
        """
        1-sigma uncertainty in eccentricity.
        """
        return self.covariance.sigmas[:, 1]

    @property
    def sigma_i(self) -> np.ndarray:
        """
        1-sigma uncertainty in inclination.
        """
        return self.covariance.sigmas[:, 2]

    @property
    def sigma_raan(self):
        """
        1-sigma uncertainty in right ascension of the ascending node.
        """
        return self.covariance.sigmas[:, 3]

    @property
    def sigma_ap(self) -> np.ndarray:
        """
        1-sigma uncertainty in argument of periapsis.
        """
        return self.covariance.sigmas[:, 4]

    @property
    def sigma_tp(self) -> np.ndarray:
        """
        1-sigma uncertainty in time of periapse passage.
        """
        return self.covariance.sigmas[:, 5]

    @property
    def a(self) -> np.ndarray:
        """
        Semi-major axis.
        """
        return self.q.to_numpy() / (1 - self.e.to_numpy())

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
    def Q(self) -> np.ndarray:
        """
        Apoapsis distance.
        """
        return self.a.to_numpy() * (1 + self.e.to_numpy())

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
    def p(self) -> np.ndarray:
        """
        Semi-latus rectum.
        """
        return self.a.to_numpy() / (1 - self.e.to_numpy() ** 2)

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
    def P(self) -> np.ndarray:
        """
        Period.
        """
        return np.sqrt(4 * np.pi**2 * self.a.to_numpy() ** 3 / self.origin.mu)

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

    def to_cartesian(self) -> CartesianCoordinates:
        from .transform import _cometary_to_cartesian, cometary_to_cartesian

        if self.time is None:
            err = (
                "To convert Cometary coordinates to Cartesian coordinates, the times\n"
                "at which the Coordinates coordinates are defined is required to give\n"
                "the time of periapsis passage context."
            )
            raise ValueError(err)

        # Extract gravitational parameter from origin
        mu = self.origin.mu

        coords_cartesian = cometary_to_cartesian(
            self.values,
            t0=self.time.to_astropy().tdb.mjd,
            mu=mu,
            max_iter=100,
            tol=1e-15,
        )
        coords_cartesian = np.array(coords_cartesian)

        if not self.covariance.is_all_nan():
            cometary_covariances = self.covariance.to_matrix()
            covariances_cartesian = transform_covariances_jacobian(
                self.values,
                cometary_covariances,
                _cometary_to_cartesian,
                in_axes=(0, 0, None, None, None),
                out_axes=0,
                t0=self.time.to_astropy().tdb.mjd,
                mu=mu,
                max_iter=100,
                tol=1e-15,
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
    def from_cartesian(cls, cartesian: CartesianCoordinates) -> "CometaryCoordinates":
        from .transform import _cartesian_to_cometary, cartesian_to_cometary

        if cartesian.time is None:
            err = (
                "To convert Cometary coordinates to Cartesian coordinates, the times\n"
                "at which the Cartesian coordinates are defined is required to calculate\n"
                "the time of periapsis passage."
            )
            raise ValueError(err)

        # Extract gravitational parameter from origin
        mu = cartesian.origin.mu

        coords_cometary = cartesian_to_cometary(
            cartesian.values,
            cartesian.time.to_astropy().tdb.mjd,
            mu=mu,
        )
        coords_cometary = np.array(coords_cometary)

        if not cartesian.covariance.is_all_nan():
            cartesian_covariances = cartesian.covariance.to_matrix()
            covariances_cometary = transform_covariances_jacobian(
                cartesian.values,
                cartesian_covariances,
                _cartesian_to_cometary,
                in_axes=(0, 0, None),
                out_axes=0,
                t0=cartesian.time.to_astropy().tdb.mjd,
                mu=mu,
            )
        else:
            covariances_cometary = np.empty(
                (len(coords_cometary), 6, 6), dtype=np.float64
            )
            covariances_cometary.fill(np.nan)

        covariances_cometary = CoordinateCovariances.from_matrix(covariances_cometary)
        coords = cls.from_kwargs(
            q=coords_cometary[:, 0],
            e=coords_cometary[:, 1],
            i=coords_cometary[:, 2],
            raan=coords_cometary[:, 3],
            ap=coords_cometary[:, 4],
            tp=coords_cometary[:, 5],
            time=cartesian.time,
            covariance=covariances_cometary,
            origin=cartesian.origin,
            frame=cartesian.frame,
        )

        return coords

    def to_keplerian(self) -> "KeplerianCoordinates":
        from .keplerian import KeplerianCoordinates

        return KeplerianCoordinates.from_cartesian(self.to_cartesian())

    @classmethod
    def from_keplerian(
        cls, keplerian_coordinates: "KeplerianCoordinates"
    ) -> "CometaryCoordinates":
        return cls.from_cartesian(keplerian_coordinates.to_cartesian())

    def to_spherical(self) -> "SphericalCoordinates":
        from .spherical import SphericalCoordinates

        return SphericalCoordinates.from_cartesian(self.to_cartesian())

    @classmethod
    def from_spherical(
        cls, spherical_coordinates: "SphericalCoordinates"
    ) -> "CometaryCoordinates":
        return cls.from_cartesian(spherical_coordinates.to_cartesian())

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
            ["q", "e", "i", "raan", "ap", "tp"],
            sigmas=sigmas,
            covariances=covariances,
        )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "CometaryCoordinates":
        """
        Create coordinates from a pandas DataFrame.

        Parameters
        ----------
        df : `~pandas.Dataframe`
            DataFrame containing coordinates.

        Returns
        -------
        coords : `~adam_core.coordinates.cometary.CometaryCoordinates`
            Cometary coordinates.
        """
        return coords_from_dataframe(
            cls, df, coord_names=["q", "e", "i", "raan", "ap", "tp"]
        )
