import jax.numpy as jnp
import numpy as np
from astropy import units as u
from quivr import Float64Field, Table

from ..constants import Constants as c
from .cartesian import CartesianCoordinates
from .covariances import CoordinateCovariances, transform_covariances_jacobian
from .frame import Frame
from .origin import Origin
from .times import Times

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


class CometaryCoordinates(Table):
    # TODO: Time of periapse passage could perhaps be represented
    # as a Times object. We could then modify self.values to only
    # grab the MJD column. That said, we would want to force it
    # the time scale to be in TDB..

    q = Float64Field(nullable=False)
    e = Float64Field(nullable=False)
    i = Float64Field(nullable=False)
    raan = Float64Field(nullable=False)
    ap = Float64Field(nullable=False)
    tp = Float64Field(nullable=False)
    times = Times.as_field(nullable=True)
    covariances = CoordinateCovariances.as_field(nullable=True)
    origin = Origin.as_field(nullable=False)
    frame = Frame.as_field(nullable=False)

    @property
    def values(self) -> np.ndarray:
        return self.table.to_pandas()[["q", "e", "i", "raan", "ap", "tp"]].values

    @property
    def sigma_q(self) -> np.ndarray:
        """
        1-sigma uncertainty in periapsis distance.
        """
        return self.covariances.sigmas[:, 0]

    @property
    def sigma_e(self) -> np.ndarray:
        """
        1-sigma uncertainty in eccentricity.
        """
        return self.covariances.sigmas[:, 1]

    @property
    def sigma_i(self) -> np.ndarray:
        """
        1-sigma uncertainty in inclination.
        """
        return self.covariances.sigmas[:, 2]

    @property
    def sigma_raan(self):
        """
        1-sigma uncertainty in right ascension of the ascending node.
        """
        return self.covariances.sigmas[:, 3]

    @property
    def sigma_ap(self) -> np.ndarray:
        """
        1-sigma uncertainty in argument of periapsis.
        """
        return self.covariances.sigmas[:, 4]

    @property
    def sigma_tp(self) -> np.ndarray:
        """
        1-sigma uncertainty in time of periapse passage.
        """
        return self.covariances.sigmas[:, 5]

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
        return np.sqrt(4 * np.pi**2 * self.a.to_numpy() ** 3 / self.mu)

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

    def to_cartesian(self, mu: float = MU) -> CartesianCoordinates:
        from .transform import _cometary_to_cartesian, cometary_to_cartesian

        if self.times is None:
            err = (
                "To convert Cometary coordinates to Cartesian coordinates, the times\n"
                "at which the Coordinates coordinates are defined is required to give\n"
                "the time of periapsis passage context."
            )
            raise ValueError(err)

        coords_cartesian = cometary_to_cartesian(
            self.values,
            t0=self.times.to_astropy().tdb.mjd,
            mu=mu,
            max_iter=100,
            tol=1e-15,
        )
        coords_cartesian = np.array(coords_cartesian)

        cometary_covariances = self.covariances.to_matrix()
        if not np.all(np.isnan(cometary_covariances)):
            covariances_cartesian = transform_covariances_jacobian(
                self.values,
                cometary_covariances,
                _cometary_to_cartesian,
                in_axes=(0, 0, None, None, None),
                out_axes=0,
                t0=self.times.to_astropy().tdb.mjd,
                mu=mu,
                max_iter=100,
                tol=1e-15,
            )
            covariances_cartesian = CoordinateCovariances.from_matrix(
                covariances_cartesian.filled()
            )
        else:
            covariances_cartesian = np.empty(
                (len(coords_cartesian), 6, 6), dtype=np.float64
            )
            covariances_cartesian.fill(np.nan)
            covariances_cartesian = CoordinateCovariances.from_matrix(
                covariances_cartesian
            )

        coords = CartesianCoordinates.from_kwargs(
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
            cartesian.values,
            cartesian.times.to_astropy().tdb.mjd,
            mu=mu,
        )
        coords_cometary = np.array(coords_cometary)

        cartesian_covariances = cartesian.covariances.to_matrix()
        if not np.all(np.isnan(cartesian_covariances)):
            covariances_cometary = transform_covariances_jacobian(
                cartesian.values,
                cartesian_covariances,
                _cartesian_to_cometary,
                in_axes=(0, 0, None),
                out_axes=0,
                t0=cartesian.times.to_astropy().tdb.mjd,
                mu=mu,
            )
        else:
            covariances_cometary = np.empty(
                (len(coords_cometary), 6, 6), dtype=np.float64
            )
            covariances_cometary.fill(np.nan)
            covariances_cometary = CoordinateCovariances.from_matrix(
                covariances_cometary
            )

        coords = cls.from_kwargs(
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
        )

        return coords
