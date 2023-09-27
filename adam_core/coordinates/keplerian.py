from typing import TYPE_CHECKING

import numpy as np
import quivr as qv
from astropy import units as u

from .cartesian import CartesianCoordinates
from .covariances import CoordinateCovariances, transform_covariances_jacobian
from .origin import Origin
from ..time import Timestamp

if TYPE_CHECKING:
    from .cometary import CometaryCoordinates
    from .spherical import SphericalCoordinates


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
    frame = qv.StringAttribute()

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
        return self.a.to_numpy() * (1 - self.e.to_numpy())

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
        return self.a.to_numpy() * (1 + self.e.to_numpy())

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
        return self.a.to_numpy() / (1 - self.e.to_numpy() ** 2)

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
        return np.sqrt(4 * np.pi**2 * self.a.to_numpy() ** 3 / self.origin.mu)

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

    def to_cartesian(self) -> CartesianCoordinates:
        from .transform import _keplerian_to_cartesian_a, keplerian_to_cartesian

        # Extract gravitational parameter from origin
        mu = self.origin.mu

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
                in_axes=(0, None, None, None),
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
    def from_cartesian(cls, cartesian: CartesianCoordinates):
        from .transform import _cartesian_to_keplerian6, cartesian_to_keplerian

        # Extract gravitational parameter from origin
        mu = cartesian.origin.mu

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
                in_axes=(0, 0, None),
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

    def to_cometary(self) -> "CometaryCoordinates":
        from .cometary import CometaryCoordinates

        return CometaryCoordinates.from_cartesian(self.to_cartesian())

    @classmethod
    def from_cometary(
        cls, cometary_coordinates: "CometaryCoordinates"
    ) -> "KeplerianCoordinates":
        return cls.from_cartesian(cometary_coordinates.to_cartesian())

    def to_spherical(self) -> "SphericalCoordinates":
        from .spherical import SphericalCoordinates

        return SphericalCoordinates.from_cartesian(self.to_cartesian())

    @classmethod
    def from_spherical(
        cls, spherical_coordinates: "SphericalCoordinates"
    ) -> "KeplerianCoordinates":
        return cls.from_cartesian(spherical_coordinates.to_cartesian())
