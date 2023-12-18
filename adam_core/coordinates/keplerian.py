from __future__ import annotations

import numpy as np
import quivr as qv

from ..time import Timestamp
from . import cartesian, cometary, spherical
from .covariances import CoordinateCovariances, transform_covariances_jacobian
from .origin import Origin

__all__ = [
    "KeplerianCoordinates",
]


class KeplerianCoordinates(qv.Table):

    a = qv.Float64Column()
    e = qv.Float64Column()
    i = qv.Float64Column()
    raan = qv.Float64Column()
    ap = qv.Float64Column()
    M = qv.Float64Column()
    time = Timestamp.as_column(nullable=True)
    covariance = CoordinateCovariances.as_column(nullable=True)
    origin = Origin.as_column()
    frame = qv.StringAttribute(default="unspecified")

    @property
    def values(self) -> np.ndarray:
        return np.array(self.table.select(["a", "e", "i", "raan", "ap", "M"]))

    @property
    def sigma_a(self) -> np.ndarray:
        """
        1-sigma uncertainty in semi-major axis.
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
    def sigma_raan(self) -> np.ndarray:
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
    def sigma_M(self) -> np.ndarray:
        """
        1-sigma uncertainty in mean anomaly.
        """
        return self.covariance.sigmas[:, 5]

    @property
    def q(self) -> np.ndarray:
        """
        Periapsis distance.
        """
        from ..dynamics.kepler import calc_periapsis_distance

        return np.array(calc_periapsis_distance(self.a.to_numpy(), self.e.to_numpy()))

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
    def Q(self) -> np.ndarray:
        """
        Apoapsis distance.
        """
        from ..dynamics.kepler import calc_apoapsis_distance

        return np.array(calc_apoapsis_distance(self.a.to_numpy(), self.e.to_numpy()))

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
    def p(self) -> np.ndarray:
        """
        Semi-latus rectum.
        """
        from ..dynamics.kepler import calc_semi_latus_rectum

        return np.array(calc_semi_latus_rectum(self.a.to_numpy(), self.e.to_numpy()))

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
    def P(self) -> np.ndarray:
        """
        Period.
        """
        from ..dynamics.kepler import calc_period

        return np.array(calc_period(self.a.to_numpy(), self.origin.mu()))

    @P.setter
    def P(self, value):
        err = (
            "Cannot set period (P) as it is"
            " derived from semi-major axis (a) and gravitational parameter (mu)."
        )
        raise ValueError(err)

    @P.deleter
    def P(self):
        err = (
            "Cannot delete period (P) as it is"
            " derived from semi-major axis (a) and gravitational parameter (mu)."
        )
        raise ValueError(err)

    @property
    def n(self):
        """
        Mean motion in degrees.
        """
        from ..dynamics.kepler import calc_mean_motion

        return np.degrees(
            np.array(calc_mean_motion(self.a.to_numpy(), self.origin.mu()))
        )

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
        from .transform import _keplerian_to_cartesian_a, keplerian_to_cartesian

        # Extract gravitational parameter from origin
        mu = self.origin.mu()

        coords_cartesian = keplerian_to_cartesian(
            self.values,
            mu=mu,
            max_iter=1000,
            tol=1e-15,
        )
        coords_cartesian = np.array(coords_cartesian)

        if not self.covariance.is_all_nan():
            covariances_keplerian = self.covariance.to_matrix()
            covariances_cartesian = transform_covariances_jacobian(
                self.values,
                covariances_keplerian,
                _keplerian_to_cartesian_a,
                in_axes=(0, 0, None, None),
                out_axes=0,
                mu=mu,
                max_iter=1000,
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
    def from_cartesian(cls, cartesian: cartesian.CartesianCoordinates):
        from .transform import _cartesian_to_keplerian6, cartesian_to_keplerian

        # Extract gravitational parameter from origin
        mu = cartesian.origin.mu()

        coords_keplerian = cartesian_to_keplerian(
            cartesian.values,
            cartesian.time.to_numpy(),
            mu=mu,
        )
        coords_keplerian = np.array(coords_keplerian)

        if not cartesian.covariance.is_all_nan():
            cartesian_covariances = cartesian.covariance.to_matrix()
            covariances_keplerian = transform_covariances_jacobian(
                cartesian.values,
                cartesian_covariances,
                _cartesian_to_keplerian6,
                in_axes=(0, 0, 0),
                out_axes=0,
                t0=cartesian.time.to_numpy(),
                mu=mu,
            )
        else:
            covariances_keplerian = np.empty(
                (len(coords_keplerian), 6, 6), dtype=np.float64
            )
            covariances_keplerian.fill(np.nan)

        covariances_keplerian = CoordinateCovariances.from_matrix(covariances_keplerian)
        coords = cls.from_kwargs(
            a=coords_keplerian[:, 0],
            e=coords_keplerian[:, 4],
            i=coords_keplerian[:, 5],
            raan=coords_keplerian[:, 6],
            ap=coords_keplerian[:, 7],
            M=coords_keplerian[:, 8],
            time=cartesian.time,
            covariance=covariances_keplerian,
            origin=cartesian.origin,
            frame=cartesian.frame,
        )
        return coords

    def to_cometary(self) -> cometary.CometaryCoordinates:
        return cometary.CometaryCoordinates.from_cartesian(self.to_cartesian())

    @classmethod
    def from_cometary(
        cls, cometary_coordinates: cometary.CometaryCoordinates
    ) -> KeplerianCoordinates:
        return cls.from_cartesian(cometary_coordinates.to_cartesian())

    def to_spherical(self) -> spherical.SphericalCoordinates:
        return spherical.SphericalCoordinates.from_cartesian(self.to_cartesian())

    @classmethod
    def from_spherical(
        cls, spherical_coordinates: spherical.SphericalCoordinates
    ) -> KeplerianCoordinates:
        return cls.from_cartesian(spherical_coordinates.to_cartesian())
