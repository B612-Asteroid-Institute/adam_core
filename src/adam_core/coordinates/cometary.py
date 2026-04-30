from __future__ import annotations

import numpy as np
import quivr as qv

from ..time import Timestamp
from . import cartesian, keplerian, spherical
from .covariances import (
    CoordinateCovariances,
    rust_covariance_transform,
)
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
        # Pure-NumPy: a = q / (1 − e).
        q = self.q.to_numpy()
        e = self.e.to_numpy()
        return q / (1.0 - e)

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
        # Pure-NumPy: Q = a · (1 + e), or ∞ for e ≥ 1.
        a = self.a
        e = self.e.to_numpy()
        return np.where(e >= 1.0, np.inf, a * (1.0 + e))

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
        # Pure-NumPy: p = a · (1 − e²).
        a = self.a
        e = self.e.to_numpy()
        return a * (1.0 - e * e)

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
        # Pure-NumPy: P = 2π · sqrt(|a³|/μ), ∞ for a < 0 (hyperbolic).
        a = self.a
        mu = self.origin.mu()
        return np.where(a < 0.0, np.inf, 2.0 * np.pi * np.sqrt(np.abs(a**3) / mu))

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
        # Rust-backed NumPy kernel for concrete-array callers (1.6x faster
        # than JAX at N=50k per `migration/scripts/calc_mean_motion_bench.py`).
        from .._rust.api import calc_mean_motion_numpy as _rust_calc_mean_motion

        a = np.asarray(self.a, dtype=np.float64)
        mu = np.asarray(self.origin.mu(), dtype=np.float64)
        rust_out = _rust_calc_mean_motion(a, mu)
        return np.degrees(rust_out)

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
        from .transform import cometary_to_cartesian

        if self.time is None:
            err = (
                "To convert Cometary coordinates to Cartesian coordinates, the times\n"
                "at which the Coordinates coordinates are defined is required to give\n"
                "the time of periapsis passage context."
            )
            raise ValueError(err)

        # Extract gravitational parameter from origin
        mu = self.origin.mu()
        t0_np = self.time.to_numpy()

        if not self.covariance.is_all_nan():
            cometary_covariances = self.covariance.to_matrix()
            rust_result = rust_covariance_transform(
                self.values,
                cometary_covariances,
                "cometary",
                "cartesian",
                t0=np.ascontiguousarray(np.asarray(t0_np, dtype=np.float64)),
                mu=np.ascontiguousarray(np.asarray(mu, dtype=np.float64)),
                frame_in=self.frame,
                frame_out=self.frame,
            )
            coords_cartesian, covariances_cartesian = rust_result
        else:
            coords_cartesian = np.array(
                cometary_to_cartesian(
                    self.values, t0=t0_np, mu=mu, max_iter=100, tol=1e-15
                )
            )
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
        from .transform import cartesian_to_cometary

        if cartesian.time is None:
            err = (
                "To convert Cometary coordinates to Cartesian coordinates, the times\n"
                "at which the Cartesian coordinates are defined is required to calculate\n"
                "the time of periapsis passage."
            )
            raise ValueError(err)

        # Extract gravitational parameter from origin
        mu = cartesian.origin.mu()
        t0_np = cartesian.time.to_numpy()

        if not cartesian.covariance.is_all_nan():
            cartesian_covariances = cartesian.covariance.to_matrix()
            rust_result = rust_covariance_transform(
                cartesian.values,
                cartesian_covariances,
                "cartesian",
                "cometary",
                t0=np.ascontiguousarray(np.asarray(t0_np, dtype=np.float64)),
                mu=np.ascontiguousarray(np.asarray(mu, dtype=np.float64)),
                frame_in=cartesian.frame,
                frame_out=cartesian.frame,
            )
            coords_cometary, covariances_cometary = rust_result
        else:
            coords_cometary = np.array(
                cartesian_to_cometary(cartesian.values, t0_np, mu=mu)
            )
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
