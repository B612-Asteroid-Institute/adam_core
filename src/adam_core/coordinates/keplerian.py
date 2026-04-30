from __future__ import annotations

import numpy as np
import quivr as qv

from ..time import Timestamp
from . import cartesian, cometary, spherical
from .covariances import (
    CoordinateCovariances,
    rust_covariance_transform,
)
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
        # Pure-NumPy: q = a · (1 − e). No JAX dispatch overhead.
        a = self.a.to_numpy()
        e = self.e.to_numpy()
        return a * (1.0 - e)

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
        # Pure-NumPy: Q = a · (1 + e), or ∞ for e ≥ 1 (parabolic/hyperbolic).
        a = self.a.to_numpy()
        e = self.e.to_numpy()
        return np.where(e >= 1.0, np.inf, a * (1.0 + e))

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
        # Pure-NumPy: p = a · (1 − e²).
        a = self.a.to_numpy()
        e = self.e.to_numpy()
        return a * (1.0 - e * e)

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
        # Pure-NumPy: P = 2π · sqrt(a³/μ), or ∞ for a < 0 (hyperbolic).
        # `np.where` evaluates both branches, so clamp sqrt's input to
        # |a³| to avoid RuntimeWarning on the hyperbolic branch.
        a = self.a.to_numpy()
        mu = self.origin.mu()
        return np.where(a < 0.0, np.inf, 2.0 * np.pi * np.sqrt(np.abs(a**3) / mu))

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
        # Rust-backed NumPy kernel for concrete-array callers (1.6x faster
        # than JAX at N=50k per `migration/scripts/calc_mean_motion_bench.py`).
        from .._rust.api import calc_mean_motion_numpy as _rust_calc_mean_motion

        a = self.a.to_numpy()
        mu = self.origin.mu()
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
        from .transform import keplerian_to_cartesian

        # Extract gravitational parameter from origin
        mu = self.origin.mu()

        if not self.covariance.is_all_nan():
            covariances_keplerian = self.covariance.to_matrix()
            rust_result = rust_covariance_transform(
                self.values,
                covariances_keplerian,
                "keplerian",
                "cartesian",
                mu=np.ascontiguousarray(np.asarray(mu, dtype=np.float64)),
                frame_in=self.frame,
                frame_out=self.frame,
            )
            coords_cartesian, covariances_cartesian = rust_result
        else:
            coords_cartesian = np.array(
                keplerian_to_cartesian(self.values, mu=mu, max_iter=1000, tol=1e-15)
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
    def from_cartesian(cls, cartesian: cartesian.CartesianCoordinates):
        from .transform import cartesian_to_keplerian

        # Extract gravitational parameter from origin
        mu = cartesian.origin.mu()
        t0_np = cartesian.time.to_numpy()

        if not cartesian.covariance.is_all_nan():
            cartesian_covariances = cartesian.covariance.to_matrix()
            rust_result = rust_covariance_transform(
                cartesian.values,
                cartesian_covariances,
                "cartesian",
                "keplerian",
                t0=np.ascontiguousarray(np.asarray(t0_np, dtype=np.float64)),
                mu=np.ascontiguousarray(np.asarray(mu, dtype=np.float64)),
                frame_in=cartesian.frame,
                frame_out=cartesian.frame,
            )
            rust_coords, covariances_keplerian = rust_result
            # The 6-col generic returns (a, e, i, raan, ap, M).
            a_col, e_col, i_col, raan_col, ap_col, m_col = 0, 1, 2, 3, 4, 5
            coords_keplerian = rust_coords
        else:
            coords_keplerian = np.array(
                cartesian_to_keplerian(cartesian.values, t0_np, mu=mu)
            )
            a_col, e_col, i_col, raan_col, ap_col, m_col = 0, 4, 5, 6, 7, 8
            covariances_keplerian = np.empty(
                (len(coords_keplerian), 6, 6), dtype=np.float64
            )
            covariances_keplerian.fill(np.nan)

        covariances_keplerian = CoordinateCovariances.from_matrix(covariances_keplerian)
        coords = cls.from_kwargs(
            a=coords_keplerian[:, a_col],
            e=coords_keplerian[:, e_col],
            i=coords_keplerian[:, i_col],
            raan=coords_keplerian[:, raan_col],
            ap=coords_keplerian[:, ap_col],
            M=coords_keplerian[:, m_col],
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
