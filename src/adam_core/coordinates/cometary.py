from __future__ import annotations

import numpy as np
import quivr as qv

from ..time import Timestamp
from . import cartesian, keplerian, spherical
from .covariances import CoordinateCovariances, transform_covariances_jacobian
from .origin import Origin

__all__ = [
    "CometaryCoordinates",
]


class CometaryCoordinates(qv.Table):
    # TODO: Time of periapse passage could perhaps be represented
    # as a Times object. We could then modify self.values to only
    # grab the MJD column. That said, we would want to force it
    # the time scale to be in TDB..

    q = qv.Float64Column()
    e = qv.Float64Column()
    i = qv.Float64Column()
    raan = qv.Float64Column()
    ap = qv.Float64Column()
    tp = qv.Float64Column()
    time = Timestamp.as_column(nullable=True)
    covariance = CoordinateCovariances.as_column(nullable=True)
    origin = Origin.as_column()
    frame = qv.StringAttribute(default="unspecified")

    @property
    def values(self) -> np.ndarray:
        return np.array(self.table.select(["q", "e", "i", "raan", "ap", "tp"]))

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
        from ..dynamics.kepler import calc_semi_major_axis

        return np.array(calc_semi_major_axis(self.q.to_numpy(), self.e.to_numpy()))

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
        from ..dynamics.kepler import calc_apoapsis_distance

        return np.array(calc_apoapsis_distance(self.a, self.e.to_numpy()))

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
        from ..dynamics.kepler import calc_semi_latus_rectum

        return np.array(calc_semi_latus_rectum(self.a, self.e.to_numpy()))

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
        from ..dynamics.kepler import calc_period

        return np.array(calc_period(self.a, self.origin.mu()))

    @P.setter
    def P(self, value):
        err = (
            "Cannot set period (P) as it is"
            " derived from the semi-major axis (a) and gravitational parameter (mu)."
        )
        raise ValueError(err)

    @P.deleter
    def P(self):
        err = (
            "Cannot delete period (P) as it is"
            " derived from the semi-major axis (a) and gravitational parameter (mu)."
        )
        raise ValueError(err)

    @property
    def n(self):
        """
        Mean motion in degrees.
        """
        from ..dynamics.kepler import calc_mean_motion

        return np.degrees(np.array(calc_mean_motion(self.a, self.origin.mu())))

    @n.setter
    def n(self, value):
        err = (
            "Cannot set mean motion (n) as it is"
            " derived from semi-major axis (a) and gravitational parameter (mu)."
        )
        raise ValueError(err)

    @n.deleter
    def n(self):
        err = (
            "Cannot delete mean motion (n) as it is"
            " derived from semi-major axis (a) and gravitational parameter (mu)."
        )
        raise ValueError(err)

    def to_cartesian(self) -> cartesian.CartesianCoordinates:
        from .transform import _cometary_to_cartesian, cometary_to_cartesian

        if self.time is None:
            err = (
                "To convert Cometary coordinates to Cartesian coordinates, the times\n"
                "at which the Coordinates coordinates are defined is required to give\n"
                "the time of periapsis passage context."
            )
            raise ValueError(err)

        # Extract gravitational parameter from origin
        mu = self.origin.mu()

        coords_cartesian = cometary_to_cartesian(
            self.values,
            t0=self.time.to_numpy(),
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
                in_axes=(0, 0, 0, None, None),
                out_axes=0,
                t0=self.time.to_numpy(),
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
        coords = cartesian.CartesianCoordinates.from_kwargs(
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
    def from_cartesian(
        cls, cartesian: cartesian.CartesianCoordinates
    ) -> CometaryCoordinates:
        from .transform import _cartesian_to_cometary, cartesian_to_cometary

        if cartesian.time is None:
            err = (
                "To convert Cometary coordinates to Cartesian coordinates, the times\n"
                "at which the Cartesian coordinates are defined is required to calculate\n"
                "the time of periapsis passage."
            )
            raise ValueError(err)

        # Extract gravitational parameter from origin
        mu = cartesian.origin.mu()

        coords_cometary = cartesian_to_cometary(
            cartesian.values,
            cartesian.time.to_numpy(),
            mu=mu,
        )
        coords_cometary = np.array(coords_cometary)

        if not cartesian.covariance.is_all_nan():
            cartesian_covariances = cartesian.covariance.to_matrix()
            covariances_cometary = transform_covariances_jacobian(
                cartesian.values,
                cartesian_covariances,
                _cartesian_to_cometary,
                in_axes=(0, 0, 0),
                out_axes=0,
                t0=cartesian.time.to_numpy(),
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

    def to_keplerian(self) -> keplerian.KeplerianCoordinates:
        return keplerian.KeplerianCoordinates.from_cartesian(self.to_cartesian())

    @classmethod
    def from_keplerian(
        cls, keplerian_coordinates: keplerian.KeplerianCoordinates
    ) -> CometaryCoordinates:
        return cls.from_cartesian(keplerian_coordinates.to_cartesian())

    def to_spherical(self) -> spherical.SphericalCoordinates:
        return spherical.SphericalCoordinates.from_cartesian(self.to_cartesian())

    @classmethod
    def from_spherical(
        cls, spherical_coordinates: spherical.SphericalCoordinates
    ) -> CometaryCoordinates:
        return cls.from_cartesian(spherical_coordinates.to_cartesian())
