import logging
from copy import deepcopy
from typing import Optional, Union

import jax.numpy as jnp
import numpy as np
from jax import config, jit, lax, vmap

from ..constants import Constants as c
from ..dynamics.barker import solve_barker
from ..dynamics.kepler import calc_mean_anomaly, solve_kepler
from .cartesian import CARTESIAN_UNITS, CartesianCoordinates
from .cometary import COMETARY_UNITS, CometaryCoordinates
from .conversions import convert_coordinates
from .keplerian import KEPLERIAN_UNITS, KeplerianCoordinates
from .spherical import SPHERICAL_UNITS, SphericalCoordinates

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

TRANSFORM_EQ2EC = np.zeros((6, 6))
TRANSFORM_EQ2EC[0:3, 0:3] = c.TRANSFORM_EQ2EC
TRANSFORM_EQ2EC[3:6, 3:6] = c.TRANSFORM_EQ2EC
TRANSFORM_EC2EQ = TRANSFORM_EQ2EC.T

logger = logging.getLogger(__name__)

__all__ = [
    "transform_coordinates",
    "_cartesian_to_keplerian",
    "_cartesian_to_keplerian6",
    "cartesian_to_keplerian",
    "_keplerian_to_cartesian_a",
    "_keplerian_to_cartesian_p",
    "_keplerian_to_cartesian_q",
    "_cartesian_to_cometary",
    "cartesian_to_cometary",
    "_cometary_to_cartesian",
    "cometary_to_cartesian",
]

CoordinateType = Union[
    CartesianCoordinates,
    KeplerianCoordinates,
    CometaryCoordinates,
    SphericalCoordinates,
]


MU = c.MU
Z_AXIS = jnp.array([0.0, 0.0, 1.0])
FLOAT_TOLERANCE = 1e-15


@jit
def _cartesian_to_spherical(
    coords_cartesian: Union[np.ndarray, jnp.ndarray]
) -> jnp.ndarray:
    """
    Convert a single Cartesian coordinate to a spherical coordinate.

    Parameters
    ----------
    coords_cartesian : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (6)
        3D Cartesian coordinate including time derivatives.
        x : x-position in units of distance.
        y : y-position in units of distance.
        z : z-position in units of distance.
        vx : x-velocity in the same units of x per arbitrary unit of time.
        vy : y-velocity in the same units of y per arbitrary unit of time.
        vz : z-velocity in the same units of z per arbitrary unit of time.

    Returns
    -------
    coords_spherical : `~jax.numpy.ndarray` (6)
        3D Spherical coordinate including time derivatives.
        rho : Radial distance in the same units of x, y, and z.
        lon : Longitude ranging from 0.0 to 360.0 degrees.
        lat : Latitude ranging from -90.0 to 90.0 degrees with 0 at the equator.
        vrho : Radial velocity in the same units as rho per arbitrary unit of time
            (same unit of time as the x, y, and z velocities).
        vlon : Longitudinal velocity in degrees per arbitrary unit of time
            (same unit of time as the x, y, and z velocities).
        vlat :Latitudinal velocity in degrees per arbitrary unit of time.
            (same unit of time as the x, y, and z velocities).
    """
    coords_spherical = jnp.zeros(6, dtype=jnp.float64)
    x = coords_cartesian[0]
    y = coords_cartesian[1]
    z = coords_cartesian[2]
    vx = coords_cartesian[3]
    vy = coords_cartesian[4]
    vz = coords_cartesian[5]

    rho = jnp.sqrt(x**2 + y**2 + z**2)
    lon = jnp.arctan2(y, x)
    lon = jnp.where(lon < 0.0, 2 * jnp.pi + lon, lon)
    lat = lax.cond(
        rho == 0.0,
        lambda _: 0.0,
        lambda _: jnp.arcsin(z / rho),
        None,
    )
    lat = jnp.where(
        (lat >= 3 * jnp.pi / 2) & (lat <= 2 * jnp.pi), lat - 2 * jnp.pi, lat
    )

    vrho = lax.cond(
        rho == 0.0,
        lambda _: 0.0,
        lambda _: (x * vx + y * vy + z * vz) / rho,
        None,
    )
    vlon = lax.cond(
        (x == 0.0) & (y == 0.0),
        lambda _: 0.0,
        lambda _: (vy * x - vx * y) / (x**2 + y**2),
        None,
    )
    vlat = lax.cond(
        ((x == 0.0) & (y == 0.0)) | (rho == 0.0),
        lambda _: 0.0,
        lambda _: (vz - vrho * z / rho) / jnp.sqrt(x**2 + y**2),
        None,
    )

    coords_spherical = coords_spherical.at[0].set(rho)
    coords_spherical = coords_spherical.at[1].set(jnp.degrees(lon))
    coords_spherical = coords_spherical.at[2].set(jnp.degrees(lat))
    coords_spherical = coords_spherical.at[3].set(vrho)
    coords_spherical = coords_spherical.at[4].set(jnp.degrees(vlon))
    coords_spherical = coords_spherical.at[5].set(jnp.degrees(vlat))

    return coords_spherical


# Vectorization Map: _cartesian_to_spherical
_cartesian_to_spherical_vmap = vmap(
    _cartesian_to_spherical,
    in_axes=(0,),
)


def cartesian_to_spherical(
    coords_cartesian: Union[np.ndarray, jnp.ndarray]
) -> jnp.ndarray:
    """
    Convert Cartesian coordinates to a spherical coordinates.

    Parameters
    ----------
    coords_cartesian : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (N, 6)
        3D Cartesian coordinates including time derivatives.
        x : x-position in units of distance.
        y : y-position in units of distance.
        z : z-position in units of distance.
        vx : x-velocity in the same units of x per arbitrary unit of time.
        vy : y-velocity in the same units of y per arbitrary unit of time.
        vz : z-velocity in the same units of z per arbitrary unit of time.

    Returns
    -------
    coords_spherical : ~jax.numpy.ndarray` (N, 6)
        3D Spherical coordinates including time derivatives.
        rho : Radial distance in the same units of x, y, and z.
        lon : Longitude ranging from 0.0 to 360.0 degrees.
        lat : Latitude ranging from -90.0 to 90.0 degrees with 0 at the equator.
        vrho : Radial velocity in the same units as rho per arbitrary unit of time
            (same unit of time as the x, y, and z velocities).
        vlon : Longitudinal velocity in degrees per arbitrary unit of time
            (same unit of time as the x, y, and z velocities).
        vlat : Latitudinal velocity in degrees per arbitrary unit of time.
            (same unit of time as the x, y, and z velocities).
    """
    coords_spherical = _cartesian_to_spherical_vmap(coords_cartesian)
    return coords_spherical


@jit
def _spherical_to_cartesian(
    coords_spherical: Union[np.ndarray, jnp.ndarray]
) -> jnp.ndarray:
    """
    Convert a single spherical coordinate to a Cartesian coordinate.

    Parameters
    ----------
    coords_spherical : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (6)
        3D Spherical coordinate including time derivatives.
        rho : Radial distance in the same units of x, y, and z.
        lon : Longitude ranging from 0.0 to 360.0 degrees.
        lat : Latitude ranging from -90.0 to 90.0 degrees with 0 at the equator.
        vrho : Radial velocity in the same units as rho per arbitrary unit of time
            (same unit of time as the x, y, and z velocities).
        vlon : Longitudinal velocity in degrees per arbitrary unit of time
            (same unit of time as the x, y, and z velocities).
        vlat : Latitudinal velocity in degrees per arbitrary unit of time.
            (same unit of time as the x, y, and z velocities).

    Returns
    -------
    coords_cartesian : `~jax.numpy.ndarray` (6)
        3D Cartesian coordinate including time derivatives.
        x : x-position in units of distance.
        y : y-position in units of distance.
        z : z-position in units of distance.
        vx : x-velocity in the same units of x per arbitrary unit of time.
        vy : y-velocity in the same units of y per arbitrary unit of time.
        vz : z-velocity in the same units of z per arbitrary unit of time.
    """
    coords_cartesian = jnp.zeros(6, dtype=jnp.float64)
    rho = coords_spherical[0]
    lon = jnp.radians(coords_spherical[1])
    lat = jnp.radians(coords_spherical[2])
    vrho = coords_spherical[3]
    vlon = jnp.radians(coords_spherical[4])
    vlat = jnp.radians(coords_spherical[5])

    cos_lat = jnp.cos(lat)
    sin_lat = jnp.sin(lat)
    cos_lon = jnp.cos(lon)
    sin_lon = jnp.sin(lon)

    x = rho * cos_lat * cos_lon
    y = rho * cos_lat * sin_lon
    z = rho * sin_lat

    vx = (
        cos_lat * cos_lon * vrho
        - rho * cos_lat * sin_lon * vlon
        - rho * sin_lat * cos_lon * vlat
    )
    vy = (
        cos_lat * sin_lon * vrho
        + rho * cos_lat * cos_lon * vlon
        - rho * sin_lat * sin_lon * vlat
    )
    vz = sin_lat * vrho + rho * cos_lat * vlat

    coords_cartesian = coords_cartesian.at[0].set(x)
    coords_cartesian = coords_cartesian.at[1].set(y)
    coords_cartesian = coords_cartesian.at[2].set(z)
    coords_cartesian = coords_cartesian.at[3].set(vx)
    coords_cartesian = coords_cartesian.at[4].set(vy)
    coords_cartesian = coords_cartesian.at[5].set(vz)

    return coords_cartesian


# Vectorization Map: _spherical_to_cartesian
_spherical_to_cartesian_vmap = vmap(
    _spherical_to_cartesian,
    in_axes=(0,),
)


def spherical_to_cartesian(
    coords_spherical: Union[np.ndarray, jnp.ndarray]
) -> jnp.ndarray:
    """
    Convert spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    coords_spherical : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (N, 6)
        3D Spherical coordinates including time derivatives.
        rho : Radial distance in the same units of x, y, and z.
        lon : Longitude ranging from 0.0 to 360.0 degrees.
        lat : Latitude ranging from -90.0 to 90.0 degrees with 0 at the equator.
        vrho : Radial velocity in the same units as rho per arbitrary unit of time
            (same unit of time as the x, y, and z velocities).
        vlon : Longitudinal velocity in degrees per arbitrary unit of time
            (same unit of time as the x, y, and z velocities).
        vlat :Latitudinal velocity in degrees per arbitrary unit of time.
            (same unit of time as the x, y, and z velocities).

    Returns
    -------
    coords_cartesian : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (N, 6)
        3D Cartesian coordinates including time derivatives.
        x : x-position in units of distance.
        y : y-position in units of distance.
        z : z-position in units of distance.
        vx : x-velocity in the same units of x per arbitrary unit of time.
        vy : y-velocity in the same units of y per arbitrary unit of time.
        vz : z-velocity in the same units of z per arbitrary unit of time.
    """
    coords_cartesian = _spherical_to_cartesian_vmap(coords_spherical)
    return coords_cartesian


@jit
def _cartesian_to_keplerian(
    coords_cartesian: Union[np.ndarray, jnp.ndarray],
    t0: float,
    mu: float = MU,
) -> jnp.ndarray:
    """
    Convert a single Cartesian coordinate to a Keplerian coordinate.

    If the orbit is found to be circular (e = 0 +- 1e-15) then
    the argument of periapsis is set to 0. The anomalies are then accordingly
    defined with this assumption.

    If the orbit's inclination is zero or 180 degrees (i = 0 +- 1e-15 or i = 180 +- 1e-15),
    then the longitude of the ascending node is set to 0 (located in the direction of
    the reference axis).

    Parameters
    ----------
    coords_cartesian : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (6)
        3D Cartesian coordinate including time derivatives.
        x : x-position in units of au.
        y : y-position in units of au.
        z : z-position in units of au.
        vx : x-velocity in units of au per day.
        vy : y-velocity in units of au per day.
        vz : z-velocity in units of au per day.
    t0 : float (1)
        Epoch at which cometary elements are defined in MJD TDB.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.

    Returns
    -------
    coords_keplerian : `~jax.numpy.ndarray` (13)
        13D Keplerian coordinate.
        a : semi-major axis in au.
        p : semi-latus rectum in au.
        q : periapsis distance in au.
        Q : apoapsis distance in au.
        e : eccentricity.
        i : inclination in degrees.
        raan : Right ascension (longitude) of the ascending node in degrees.
        ap : argument of periapsis in degrees.
        M : mean anomaly in degrees.
        nu : true anomaly in degrees.
        n : mean motion in degrees per day.
        P : period in days.
        tp : time of periapsis passage in days.

    References
    ----------
    [1] Bate, R. R; Mueller, D. D; White, J. E. (1971). Fundamentals of Astrodynamics. 1st ed.,
        Dover Publications, Inc. ISBN-13: 978-0486600611
    """
    coords_keplerian = jnp.zeros(13, dtype=jnp.float64)
    r = coords_cartesian[0:3]
    v = coords_cartesian[3:6]

    r_mag = jnp.linalg.norm(r)
    v_mag = jnp.linalg.norm(v)

    sme = v_mag**2 / 2 - mu / r_mag

    # Calculate the angular momentum vector
    # Equation 2.4-1 in Bate, Mueller, & White [1]
    h = jnp.cross(r, v)
    h_mag = jnp.linalg.norm(h)

    # Calculate the vector which is perpendicular to the
    # momentum vector and the Z-axis and points towards
    # the direction of the ascending node.
    # Equation 2.4-3 in Bate, Mueller, & White [1]
    n = jnp.cross(Z_AXIS, h)
    n_mag = jnp.linalg.norm(n)

    # Calculate the eccentricity vector which lies in the orbital plane
    # and points toward periapse.
    # Equation 2.4-5 in Bate, Mueller, & White [1]
    e_vec = ((v_mag**2 - mu / r_mag) * r - (jnp.dot(r, v)) * v) / mu
    e = jnp.linalg.norm(e_vec)

    # Calculate the semi-latus rectum
    p = h_mag**2 / mu

    # Calculate the inclination
    # Equation 2.4-7 in Bate, Mueller, & White [1]
    i = jnp.arccos(h[2] / h_mag)

    # Calculate the longitude of the ascending node
    # Equation 2.4-8 in Bate, Mueller, & White [1]
    raan = jnp.arccos(n[0] / n_mag)
    raan = jnp.where(n[1] < 0, 2 * jnp.pi - raan, raan)
    # In certain conventions when the orbit is zero inclined or 180 inclined
    # the ascending node is set to 0 as opposed to being undefined. This is what
    # SPICE does so we will do the same.
    raan = jnp.where(
        (i < FLOAT_TOLERANCE) | (jnp.abs(i - 2 * jnp.pi) < FLOAT_TOLERANCE), 0.0, raan
    )

    # Calculate the argument of periapsis
    # Equation 2.4-9 in Bate, Mueller, & White [1]
    ap = jnp.arccos(jnp.dot(n, e_vec) / (n_mag * e))
    # Adopt convention that if the orbit is circular the argument of
    # periapsis is set to 0
    ap = jnp.where(e_vec[2] < 0, 2 * jnp.pi - ap, ap)
    ap = jnp.where(jnp.abs(e) < FLOAT_TOLERANCE, 0.0, ap)

    # Calculate true anomaly (undefined for
    # circular orbits)
    # Equation 2.4-10 in Bate, Mueller, & White [1]
    nu = jnp.arccos(jnp.dot(e_vec, r) / (e * r_mag))
    nu = jnp.where(jnp.dot(r, v) < 0, 2 * jnp.pi - nu, nu)
    # nu = jnp.where(jnp.abs(e) < FLOAT_TOLERANCE, jnp.nan, nu)

    # Calculate the semi-major axis (undefined for parabolic
    # orbits)
    a = jnp.where(
        (e > (1.0 - FLOAT_TOLERANCE)) & (e < (1.0 + FLOAT_TOLERANCE)),
        jnp.nan,
        mu / (-2 * sme),
    )

    # Calculate the periapsis distance
    q = jnp.where(
        (e > (1.0 - FLOAT_TOLERANCE)) & (e < (1.0 + FLOAT_TOLERANCE)),
        p / 2,
        a * (1 - e),
    )

    # Calculate the apoapsis distance (infinite for
    # parabolic and hyperbolic orbits)
    Q = jnp.where(e < 1.0, a * (1 + e), jnp.inf)

    # Calculate the mean anomaly
    M = calc_mean_anomaly(nu, e)

    # Calculate the mean motion
    n = lax.cond(
        (e > (1.0 - FLOAT_TOLERANCE)) & (e < (1.0 + FLOAT_TOLERANCE)),
        lambda a, q: jnp.sqrt(mu / (2 * q**3)),
        lambda a, q: jnp.sqrt(mu / jnp.abs(a) ** 3),
        a,
        q,
    )

    # Calculate the orbital period which for parabolic and hyperbolic
    # orbits is infinite while for all closed orbits
    # is well defined.
    P = lax.cond(
        e < (1.0 - FLOAT_TOLERANCE), lambda n: 2 * jnp.pi / n, lambda n: jnp.inf, n
    )

    # In the case of closed orbits, if the mean anomaly is
    # greater than 180 degrees then the orbit is
    # approaching periapsis passage in which case
    # the periapsis will occur in the future
    # in less than half a period. If the mean anomaly is less
    # than 180 degrees, then the orbit is ascending from periapsis
    # passage and the most recent periapsis was in the past.
    dtp = M / n
    dtp = jnp.where((M > jnp.pi) & (e < (1.0 - FLOAT_TOLERANCE)), P - M / n, -M / n)
    tp = t0 + dtp

    coords_keplerian = coords_keplerian.at[0].set(a)
    coords_keplerian = coords_keplerian.at[1].set(p)
    coords_keplerian = coords_keplerian.at[2].set(q)
    coords_keplerian = coords_keplerian.at[3].set(Q)
    coords_keplerian = coords_keplerian.at[4].set(e)
    coords_keplerian = coords_keplerian.at[5].set(jnp.degrees(i))
    coords_keplerian = coords_keplerian.at[6].set(jnp.degrees(raan))
    coords_keplerian = coords_keplerian.at[7].set(jnp.degrees(ap))
    coords_keplerian = coords_keplerian.at[8].set(jnp.degrees(M))
    coords_keplerian = coords_keplerian.at[9].set(jnp.degrees(nu))
    coords_keplerian = coords_keplerian.at[10].set(jnp.degrees(n))
    coords_keplerian = coords_keplerian.at[11].set(P)
    coords_keplerian = coords_keplerian.at[12].set(tp)

    return coords_keplerian


# Vectorization Map: _cartesian_to_keplerian
_cartesian_to_keplerian_vmap = vmap(
    _cartesian_to_keplerian,
    in_axes=(0, 0, None),
)


@jit
def _cartesian_to_keplerian6(
    coords_cartesian: Union[np.ndarray, jnp.ndarray],
    t0: float,
    mu: float = MU,
) -> jnp.ndarray:
    """
    Limit conversion of Cartesian coordinates to Keplerian 6 fundamental coordinates.

    Parameters
    ----------
    coords_cartesian : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (6)
        3D Cartesian coordinate including time derivatives.
        x : x-position in units of au.
        y : y-position in units of au.
        z : z-position in units of au.
        vx : x-velocity in units of au per day.
        vy : y-velocity in units of au per day.
        vz : z-velocity in units of au per day.
    t0 : float (1)
        Epoch at which cometary elements are defined in MJD TDB.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.

    Returns
    -------
    coords_keplerian : `~jax.numpy.ndarray` (6)
        6D Keplerian coordinate.
        a : semi-major axis in au.
        e : eccentricity.
        i : inclination in degrees.
        raan : Right ascension (longitude) of the ascending node in degrees.
        ap : argument of periapsis in degrees.
        M : mean anomaly in degrees.
    """
    coords_keplerian = _cartesian_to_keplerian(coords_cartesian, t0, mu)
    return coords_keplerian[jnp.array([0, 4, 5, 6, 7, 8])]


def cartesian_to_keplerian(
    coords_cartesian: Union[np.ndarray, jnp.ndarray],
    t0: Union[np.ndarray, jnp.ndarray],
    mu: float = MU,
) -> jnp.ndarray:
    """
    Convert Cartesian coordinates to Keplerian coordinates.

    Parameters
    ----------
    coords_cartesian : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (N, 6)
        3D Cartesian coordinates including time derivatives.
        x : x-position in units of au.
        y : y-position in units of au.
        z : z-position in units of au.
        vx : x-velocity in units of au per day.
        vy : y-velocity in units of au per day.
        vz : z-velocity in units of au per day.
    t0 : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (N)
        Epoch at which cometary elements are defined in MJD TDB.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.

    Returns
    -------
    coords_keplerian : `~jax.numpy.ndarray` (N, 13)
        13D Keplerian coordinates.
        a : semi-major axis in au.
        p : semi-latus rectum in au.
        q : periapsis distance in au.
        Q : apoapsis distance in au.
        e : eccentricity.
        i : inclination in degrees.
        raan : Right ascension (longitude) of the ascending node in degrees.
        ap : argument of periapsis in degrees.
        M : mean anomaly in degrees.
        nu : true anomaly in degrees.
        n : mean motion in degrees per day.
        P : period in days.
        tp : time of periapsis passage in days.
    """
    coords_keplerian = _cartesian_to_keplerian_vmap(coords_cartesian, t0, mu)
    return coords_keplerian


@jit
def _keplerian_to_cartesian_p(
    coords_keplerian: Union[np.ndarray, jnp.ndarray],
    mu: float = MU,
    max_iter: int = 1000,
    tol: float = 1e-15,
) -> jnp.ndarray:
    """
    Convert a single Keplerian coordinate to a Cartesian coordinate.

    Parabolic orbits (e = 1.0 +- 1e-15) with elements (a, e, i, raan, ap, M) cannot be converted
    to Cartesian orbits since their semi-major axes are by definition undefined.
    Please consider representing the orbits with Cometary elements
    and using those to convert to Cartesian. See `~thor.coordinates.cometary._cometary_to_cartesian`.

    Parameters
    ----------
    coords_keplerian : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (6)
        6D Keplerian coordinate.
        p : semi-latus rectum in au.
        e : eccentricity.
        i : inclination in degrees.
        raan : Right ascension (longitude) of the ascending node in degrees.
        ap : argument of periapsis in degrees.
        M : mean anomaly in degrees.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.
    max_iter : int, optional
        Maximum number of iterations over which to converge. If number of iterations is
        exceeded, will use the value of the relevant anomaly at the last iteration.
    tol : float, optional
        Numerical tolerance to which to compute anomalies using the Newtown-Raphson
        method.

    Returns
    -------
    coords_cartesian : `~jax.numpy.ndarray` (6)
        3D Cartesian coordinate including time derivatives.
        x : x-position in units of au.
        y : y-position in units of au.
        z : z-position in units of au.
        vx : x-velocity in units of au per day.
        vy : y-velocity in units of au per day.
        vz : z-velocity in units of au per day.
    """
    coords_cartesian = jnp.zeros(6, dtype=jnp.float64)

    p = coords_keplerian[0]
    e = coords_keplerian[1]
    i = jnp.radians(coords_keplerian[2])
    raan = jnp.radians(coords_keplerian[3])
    ap = jnp.radians(coords_keplerian[4])
    M = jnp.radians(coords_keplerian[5])

    # Calculate the true anomaly
    nu = lax.cond(
        (e > (1.0 - FLOAT_TOLERANCE)) & (e < (1.0 + FLOAT_TOLERANCE)),
        lambda e_i, M_i: solve_barker(M_i),
        lambda e_i, M_i: solve_kepler(e_i, M_i, max_iter=max_iter, tol=tol),
        e,
        M,
    )

    # Calculate the perifocal rotation matrices
    r_PQW = jnp.array(
        [
            p * jnp.cos(nu) / (1 + e * jnp.cos(nu)),
            p * jnp.sin(nu) / (1 + e * jnp.cos(nu)),
            0,
        ]
    )
    v_PQW = jnp.array(
        [-jnp.sqrt(mu / p) * jnp.sin(nu), jnp.sqrt(mu / p) * (e + jnp.cos(nu)), 0]
    )

    cos_raan = jnp.cos(raan)
    sin_raan = jnp.sin(raan)
    cos_ap = jnp.cos(ap)
    sin_ap = jnp.sin(ap)
    cos_i = jnp.cos(i)
    sin_i = jnp.sin(i)

    P1 = jnp.array(
        [
            [cos_ap, -sin_ap, 0.0],
            [sin_ap, cos_ap, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.float64,
    )

    P2 = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_i, -sin_i],
            [0.0, sin_i, cos_i],
        ],
        dtype=jnp.float64,
    )

    P3 = jnp.array(
        [
            [cos_raan, -sin_raan, 0.0],
            [sin_raan, cos_raan, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.float64,
    )

    rotation_matrix = P3 @ P2 @ P1
    r = rotation_matrix @ r_PQW
    v = rotation_matrix @ v_PQW

    coords_cartesian = coords_cartesian.at[0].set(r[0])
    coords_cartesian = coords_cartesian.at[1].set(r[1])
    coords_cartesian = coords_cartesian.at[2].set(r[2])
    coords_cartesian = coords_cartesian.at[3].set(v[0])
    coords_cartesian = coords_cartesian.at[4].set(v[1])
    coords_cartesian = coords_cartesian.at[5].set(v[2])

    return coords_cartesian


# Vectorization Map: _keplerian_to_cartesian_p
_keplerian_to_cartesian_p_vmap = vmap(
    _keplerian_to_cartesian_p,
    in_axes=(0, None, None, None),
)


@jit
def _keplerian_to_cartesian_a(
    coords_keplerian: Union[np.ndarray, jnp.ndarray],
    mu: float = MU,
    max_iter: int = 1000,
    tol: float = 1e-15,
) -> jnp.ndarray:
    """
    Convert a single Keplerian coordinate to a Cartesian coordinate.

    Parabolic orbits (e = 1.0 +- 1e-15) with elements (a, e, i, raan, ap, M) cannot be converted
    to Cartesian orbits since their semi-major axes are by definition undefined.
    Please consider representing the orbits with Cometary elements
    and using those to convert to Cartesian. See `~thor.coordinates.cometary._cometary_to_cartesian`.

    Parameters
    ----------
    coords_keplerian : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (6)
        6D Keplerian coordinate.
        a : semi-major axis in au.
        e : eccentricity.
        i : inclination in degrees.
        raan : Right ascension (longitude) of the ascending node in degrees.
        ap : argument of periapsis in degrees.
        M : mean anomaly in degrees.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.
    max_iter : int, optional
        Maximum number of iterations over which to converge. If number of iterations is
        exceeded, will use the value of the relevant anomaly at the last iteration.
    tol : float, optional
        Numerical tolerance to which to compute anomalies using the Newtown-Raphson
        method.

    Returns
    -------
    coords_cartesian : `~jax.numpy.ndarray` (6)
        3D Cartesian coordinate including time derivatives.
        x : x-position in units of au.
        y : y-position in units of au.
        z : z-position in units of au.
        vx : x-velocity in units of au per day.
        vy : y-velocity in units of au per day.
        vz : z-velocity in units of au per day.
    """
    coords_keplerian_p = jnp.zeros(6, dtype=jnp.float64)
    coords_keplerian_p = coords_keplerian_p.at[1].set(coords_keplerian[1])
    coords_keplerian_p = coords_keplerian_p.at[2].set(coords_keplerian[2])
    coords_keplerian_p = coords_keplerian_p.at[3].set(coords_keplerian[3])
    coords_keplerian_p = coords_keplerian_p.at[4].set(coords_keplerian[4])
    coords_keplerian_p = coords_keplerian_p.at[5].set(coords_keplerian[5])

    # Calculate the semi-latus rectum
    a = coords_keplerian[0]
    e = coords_keplerian[1]
    p = lax.cond(
        (e > (1.0 - FLOAT_TOLERANCE)) & (e < (1.0 + FLOAT_TOLERANCE)),
        lambda a, e: jnp.nan,  # 2 * q (not enough information present)
        lambda a, e: a * (1 - e**2),
        a,
        e,
    )
    coords_keplerian_p = coords_keplerian_p.at[0].set(p)

    coords_cartesian = _keplerian_to_cartesian_p(
        coords_keplerian_p,
        mu=mu,
        max_iter=max_iter,
        tol=tol,
    )
    return coords_cartesian


# Vectorization Map: _keplerian_to_cartesian_a
_keplerian_to_cartesian_a_vmap = vmap(
    _keplerian_to_cartesian_a,
    in_axes=(0, None, None, None),
)


@jit
def _keplerian_to_cartesian_q(
    coords_keplerian: Union[np.ndarray, jnp.ndarray],
    mu: float = MU,
    max_iter: int = 1000,
    tol: float = 1e-15,
) -> jnp.ndarray:
    """
    Convert a single Keplerian coordinate to a Cartesian coordinate.

    Parabolic orbits (e = 1.0 +- 1e-15) with elements (a, e, i, raan, ap, M) cannot be converted
    to Cartesian orbits since their semi-major axes are by definition undefined.
    Please consider representing the orbits with Cometary elements
    and using those to convert to Cartesian. See `~thor.coordinates.cometary._cometary_to_cartesian`.

    Parameters
    ----------
    coords_keplerian : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (6)
        6D Keplerian coordinate.
        q : periapsis distance in au.
        e : eccentricity.
        i : inclination in degrees.
        raan : Right ascension (longitude) of the ascending node in degrees.
        ap : argument of periapsis in degrees.
        M : mean anomaly in degrees.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.
    max_iter : int, optional
        Maximum number of iterations over which to converge. If number of iterations is
        exceeded, will use the value of the relevant anomaly at the last iteration.
    tol : float, optional
        Numerical tolerance to which to compute anomalies using the Newtown-Raphson
        method.

    Returns
    -------
    coords_cartesian : `~jax.numpy.ndarray` (6)
        3D Cartesian coordinate including time derivatives.
        x : x-position in units of au.
        y : y-position in units of au.
        z : z-position in units of au.
        vx : x-velocity in units of au per day.
        vy : y-velocity in units of au per day.
        vz : z-velocity in units of au per day.
    """
    coords_keplerian_p = jnp.zeros(6, dtype=jnp.float64)
    coords_keplerian_p = coords_keplerian_p.at[1].set(coords_keplerian[1])
    coords_keplerian_p = coords_keplerian_p.at[2].set(coords_keplerian[2])
    coords_keplerian_p = coords_keplerian_p.at[3].set(coords_keplerian[3])
    coords_keplerian_p = coords_keplerian_p.at[4].set(coords_keplerian[4])
    coords_keplerian_p = coords_keplerian_p.at[5].set(coords_keplerian[5])

    # Calculate the semi-latus rectum
    q = coords_keplerian[0]
    e = coords_keplerian[1]
    p = lax.cond(
        (e > (1.0 - FLOAT_TOLERANCE)) & (e < (1.0 + FLOAT_TOLERANCE)),
        lambda e, q: 2 * q,
        lambda e, q: q / (1 - e) * (1 - e**2),  # a = q / (1 - e), p = a / (1 - e**2)
        e,
        q,
    )
    coords_keplerian_p = coords_keplerian_p.at[0].set(p)

    coords_cartesian = _keplerian_to_cartesian_p(
        coords_keplerian_p,
        mu=mu,
        max_iter=max_iter,
        tol=tol,
    )
    return coords_cartesian


# Vectorization Map: _keplerian_to_cartesian_q
_keplerian_to_cartesian_q_vmap = vmap(
    _keplerian_to_cartesian_q,
    in_axes=(0, None, None, None),
)


def keplerian_to_cartesian(
    coords_keplerian: Union[np.ndarray, jnp.ndarray],
    mu: float = MU,
    max_iter: int = 100,
    tol: float = 1e-15,
) -> jnp.ndarray:
    """
    Convert Keplerian coordinates to Cartesian coordinates.

    Parabolic orbits (e = 1.0 +- 1e-15) with elements (a, e, i, raan, ap, M) cannot be converted
    to Cartesian orbits since their semi-major axes are by definition undefined.
    Please consider representing these orbits with Cometary elements
    and using those to convert to Cartesian. See `~thor.coordinates.cometary.cometary_to_cartesian`.

    Parameters
    ----------
    coords_keplerian : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (N, 6)
        6D Keplerian coordinate.
        a : semi-major axis in au.
        e : eccentricity.
        i : inclination in degrees.
        raan : Right ascension (longitude) of the ascending node in degrees.
        ap : argument of periapsis in degrees.
        M : mean anomaly in degrees.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.
    max_iter : int, optional
        Maximum number of iterations over which to converge. If number of iterations is
        exceeded, will use the value of the relevant anomaly at the last iteration.
    tol : float, optional
        Numerical tolerance to which to compute anomalies using the Newtown-Raphson
        method.

    Returns
    -------
    coords_cartesian : `~jax.numpy.ndarray` (N, 6)
        3D Cartesian coordinates including time derivatives.
        x : x-position in units of au.
        y : y-position in units of au.
        z : z-position in units of au.
        vx : x-velocity in units of au per day.
        vy : y-velocity in units of au per day.
        vz : z-velocity in units of au per day.

    Raises
    ------
    ValueError: When semi-major axis is less than 0 for elliptical orbits or when
        semi-major axis is greater than 0 for hyperbolic orbits.
    """
    a = coords_keplerian[:, 0]
    e = coords_keplerian[:, 1]

    parabolic = np.where((e < (1.0 + FLOAT_TOLERANCE)) & (e > (1.0 - FLOAT_TOLERANCE)))[
        0
    ]
    if len(parabolic) > 0:
        msg = (
            "Parabolic orbits (e = 1.0 +- 1e-15) are best represented using Cometary coordinates.\n"
            "Conversion to Cartesian coordinates will not yield correct results as semi-major axis\n"
            "for parabolic orbits is undefined."
        )
        logger.critical(msg)

    hyperbolic_invalid = np.where((e > (1.0 + FLOAT_TOLERANCE)) & (a > 0))[0]
    if len(hyperbolic_invalid) > 0:
        err = (
            "Semi-major axis (a) for hyperbolic orbits (e > 1 + 1e-15) should be negative. "
            f"Instead found a = {a[hyperbolic_invalid][0]} with e = {e[hyperbolic_invalid][0]}."
        )
        raise ValueError(err)

    elliptical_invalid = np.where((e < (1.0 - FLOAT_TOLERANCE)) & (a < 0))[0]
    if len(elliptical_invalid) > 0:
        err = (
            "Semi-major axis (a) for elliptical orbits (e < 1 - 1e-15) should be positive. "
            f"Instead found a = {a[elliptical_invalid][0]} with e = {e[elliptical_invalid][0]}."
        )
        raise ValueError(err)

    coords_cartesian = _keplerian_to_cartesian_a_vmap(
        coords_keplerian, mu, max_iter, tol
    )
    return coords_cartesian


@jit
def _cartesian_to_cometary(
    coords_cartesian: Union[np.ndarray, jnp.ndarray],
    t0: float,
    mu: float = MU,
) -> jnp.ndarray:
    """
    Convert Cartesian coordinates to Cometary coordinates.

    Parameters
    ----------
    coords_cartesian : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (6)
        3D Cartesian coordinate including time derivatives.
        x : x-position in units of au.
        y : y-position in units of au.
        z : z-position in units of au.
        vx : x-velocity in units of au per day.
        vy : y-velocity in units of au per day.
        vz : z-velocity in units of au per day.
    t0 : float (1)
        Epoch at which cometary elements are defined in MJD TDB.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.

    Returns
    -------
    coords_cometary : `~jax.numpy.ndarray` (6)
        6D Cometary coordinate.
        q : periapsis distance in au.
        e : eccentricity.
        i : inclination in degrees.
        raan : Right ascension (longitude) of the ascending node in degrees.
        ap : argument of periapsis in degrees.
        tp : time of periapse passage in days.
    """
    coords_cometary = _cartesian_to_keplerian(coords_cartesian, t0, mu=mu)
    return coords_cometary[jnp.array([2, 4, 5, 6, 7, 12])]


# Vectorization Map: _cartesian_to_cometary
_cartesian_to_cometary_vmap = vmap(
    _cartesian_to_cometary,
    in_axes=(0, 0, None),
)


def cartesian_to_cometary(
    coords_cartesian: Union[np.ndarray, jnp.ndarray],
    t0: Union[np.ndarray, jnp.ndarray],
    mu: float = MU,
) -> jnp.ndarray:
    """
    Convert Cartesian coordinates to Keplerian coordinates.

    Parameters
    ----------
    coords_cartesian : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (N, 6)
        3D Cartesian coordinates including time derivatives.
        x : x-position in units of au.
        y : y-position in units of au.
        z : z-position in units of au.
        vx : x-velocity in units of au per day.
        vy : y-velocity in units of au per day.
        vz : z-velocity in units of au per day.
    t0 : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (N)
        Epoch at which cometary elements are defined in MJD TDB.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.

    Returns
    -------
    coords_cometary : `~jax.numpy.ndarray` (N, 6)
        6D Cometary coordinates.
        q : periapsis distance in au.
        e : eccentricity.
        i : inclination in degrees.
        raan : Right ascension (longitude) of the ascending node in degrees.
        ap : argument of periapsis in degrees.
        tp : time of periapse passage in days.
    """
    coords_cometary = _cartesian_to_cometary_vmap(coords_cartesian, t0, mu)
    return coords_cometary


@jit
def _cometary_to_cartesian(
    coords_cometary: Union[np.ndarray, jnp.ndarray],
    t0: float,
    mu: float = MU,
    max_iter: int = 100,
    tol: float = 1e-15,
) -> jnp.ndarray:
    """
    Convert a single Cometary coordinate to a Cartesian coordinate.

    Parameters
    ----------
    coords_cometary : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (6)
        6D Cometary coordinate.
        q : periapsis distance in au.
        e : eccentricity.
        i : inclination in degrees.
        raan : Right ascension (longitude) of the ascending node in degrees.
        ap : argument of periapsis in degrees.
        tp : time of periapse passage in days.
    t0 : float (1)
        Epoch at which cometary elements are defined in MJD TDB.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.
    max_iter : int, optional
        Maximum number of iterations over which to converge. If number of iterations is
        exceeded, will use the value of the relevant anomaly at the last iteration.
    tol : float, optional
        Numerical tolerance to which to compute anomalies using the Newtown-Raphson
        method.

    Returns
    -------
    coords_cartesian : `~jax.numpy.ndarray` (6)
        3D Cartesian coordinate including time derivatives.
        x : x-position in units of au.
        y : y-position in units of au.
        z : z-position in units of au.
        vx : x-velocity in units of au per day.
        vy : y-velocity in units of au per day.
        vz : z-velocity in units of au per day.
    """
    coords_keplerian = jnp.zeros(6, dtype=jnp.float64)

    q = coords_cometary[0]
    e = coords_cometary[1]
    i = coords_cometary[2]
    raan = coords_cometary[3]
    ap = coords_cometary[4]
    tp = coords_cometary[5]

    # Calculate the semi-major axis from the periapsis distance
    # The semi-major axis for parabolic orbits is undefined
    a = lax.cond(
        (e > (1.0 - FLOAT_TOLERANCE)) & (e < (1.0 + FLOAT_TOLERANCE)),
        lambda e, q: jnp.nan,
        lambda e, q: q / (1 - e),
        e,
        q,
    )

    # Calculate the mean motion
    n = lax.cond(
        (e > (1.0 - FLOAT_TOLERANCE)) & (e < (1.0 + FLOAT_TOLERANCE)),
        lambda a, q: jnp.sqrt(mu / (2 * q**3)),
        lambda a, q: jnp.sqrt(mu / jnp.abs(a) ** 3),
        a,
        q,
    )

    # Calculate the orbital period which for parabolic and hyperbolic
    # orbits is infinite while for all closed orbits
    # is well defined.
    # P = lax.cond(
    #    e < (1.0 - FLOAT_TOLERANCE), lambda n: 2 * jnp.pi / n, lambda n: jnp.inf, n
    # )

    # Calculate the mean anomaly
    dtp = tp - t0
    M = jnp.where(dtp > 0, 2 * jnp.pi - dtp * n, -dtp * n)

    coords_keplerian = coords_keplerian.at[0].set(q)
    coords_keplerian = coords_keplerian.at[1].set(e)
    coords_keplerian = coords_keplerian.at[2].set(i)
    coords_keplerian = coords_keplerian.at[3].set(raan)
    coords_keplerian = coords_keplerian.at[4].set(ap)
    coords_keplerian = coords_keplerian.at[5].set(jnp.degrees(M))

    coords_cartesian = _keplerian_to_cartesian_q(
        coords_keplerian, mu=mu, max_iter=max_iter, tol=tol
    )

    return coords_cartesian


# Vectorization Map: _cometary_to_cartesian
_cometary_to_cartesian_vmap = vmap(
    _cometary_to_cartesian,
    in_axes=(0, 0, None, None, None),
)


def cometary_to_cartesian(
    coords_cometary: Union[np.ndarray, jnp.ndarray],
    t0: Union[np.ndarray, jnp.ndarray],
    mu: float = MU,
    max_iter: int = 100,
    tol: float = 1e-15,
) -> jnp.ndarray:
    """
    Convert Cometary coordinates to Cartesian coordinates.

    Parameters
    ----------
    coords_cometary : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (N, 6)
        6D Cometary coordinate.
        q : periapsis distance in au.
        e : eccentricity.
        i : inclination in degrees.
        raan : Right ascension (longitude) of the ascending node in degrees.
        ap : argument of periapsis in degrees.
        tp : time of periapse passage in days.
    t0 : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (N)
        Epoch at which cometary elements are defined in MJD TDB.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.
    max_iter : int, optional
        Maximum number of iterations over which to converge. If number of iterations is
        exceeded, will use the value of the relevant anomaly at the last iteration.
    tol : float, optional
        Numerical tolerance to which to compute anomalies using the Newtown-Raphson
        method.

    Returns
    -------
    coords_cartesian : `~jax.numpy.ndarray` (N, 6)
        3D Cartesian coordinates including time derivatives.
        x : x-position in units of au.
        y : y-position in units of au.
        z : z-position in units of au.
        vx : x-velocity in units of au per day.
        vy : y-velocity in units of au per day.
        vz : z-velocity in units of au per day.
    """
    coords_cartesian = _cometary_to_cartesian_vmap(
        coords_cometary, t0, mu, max_iter, tol
    )
    return coords_cartesian


def cartesian_to_origin(
    coords: CartesianCoordinates, origin: str
) -> "CartesianCoordinates":
    """
    Translate coordinates to a different origin.

    Parameters
    ----------
    coords : `~adam_core.coordinates.cartesian.CartesianCoordinates`
        Cartesian coordinates and optionally their covariances.
    origin : {'heliocenter', 'barycenter'}
        Name of the desired origin.

    Returns
    -------
    CartesianCoordinates : `~thor.coordinates.cartesian.CartesianCoordinates`
        Translated Cartesian coordinates and their covariances.
    """
    unique_origins = np.unique(coords.origin)
    vectors = np.zeros(coords.values.shape, dtype=np.float64)

    for origin_in in unique_origins:
        raise NotImplementedError("get_perturber_state not implemented yet")
        # mask = np.where(coords.origin == origin_in)[0]
        # vectors[mask] = get_perturber_state(
        #    origin_in, coords.times[mask], frame=coords.frame, origin=origin
        # )

    return coords.translate(vectors, origin)


def cartesian_to_frame(
    coords: CartesianCoordinates, frame: str
) -> "CartesianCoordinates":
    """
    Rotate Cartesian coordinates and their covariances to the given frame.

    Parameters
    ----------
    coords : `~adam_core.coordinates.cartesian.CartesianCoordinates`
        Cartesian coordinates and optionally their covariances.
    frame : {'ecliptic', 'equatorial'}
        Desired reference frame of the output coordinates.

    Returns
    -------
    CartesianCoordinates : `~thor.coordinates.cartesian.CartesianCoordinates`
        Rotated Cartesian coordinates and their covariances.
    """
    if frame == "ecliptic" and coords.frame != "ecliptic":
        return coords.rotate(TRANSFORM_EC2EQ, "ecliptic")
    elif frame == "equatorial" and coords.frame != "equatorial":
        return coords.rotate(TRANSFORM_EQ2EC, "equatorial")
    elif frame == coords.frame:
        return coords
    else:
        err = "frame should be one of {'ecliptic', 'equatorial'}"
        raise ValueError(err)


def transform_coordinates(
    coords: CoordinateType,
    representation_out: str,
    frame_out: Optional[str] = None,
    origin_out: Optional[str] = None,
    unit_sphere: bool = True,
) -> CoordinateType:
    """
    Transform coordinates between frames ('ecliptic', 'equatorial')
    and/or representations ('cartesian', 'spherical', 'keplerian').

    Parameters
    ----------
    coords : `~thor.coordinates.Coordinates`
        Coordinates to transform between representations and frames.
    representation_out : {'cartesian', 'spherical', 'keplerian', 'cometary'}
        Desired coordinate type or representation of the output coordinates.
    frame_out : {'equatorial', 'ecliptic'}
        Desired reference frame of the output coordinates.
    origin_out : {'heliocenter', 'barycenter'}
        Desired origin of the output coordinates.
    unit_sphere : bool
        Assume the coordinates lie on a unit sphere. In many cases, spherical coordinates
        may not have a value for radial distance or radial velocity but transforms to other
        representations or frames are still meaningful. If this parameter is set to true,
        then if radial distance is not defined and/or radial velocity is not defined
        then they are assumed to be 1.0 au and 0.0 au/d, respectively.

    Returns
    -------
    coords_out : `~thor.coordinates.Coordinates`
        Coordinates in desired output representation and frame.

    Raises
    ------
    ValueError
        If frame_in, frame_out are not one of 'equatorial', 'ecliptic'.
        If representation_in, representation_out are not one of 'cartesian',
            'spherical', 'keplerian', 'cometary'.
    """
    # Check that coords is a thor.coordinates.Coordinates object
    if not isinstance(
        coords,
        (
            CartesianCoordinates,
            SphericalCoordinates,
            KeplerianCoordinates,
            CometaryCoordinates,
        ),
    ):
        err = (
            "Coords of type {} are not supported.\n"
            "Supported coordinates are:\n"
            "  CartesianCoordinates\n"
            "  SphericalCoordinates\n"
            "  KeplerianCoordinates\n"
            "  CometaryCoordinates\n"
        )
        raise TypeError(err)

    # Check that frame_in and frame_out are one of equatorial
    # or ecliptic, raise errors otherwise
    frame_err = ["{} should be one of:\n", "'equatorial' or 'ecliptic'"]
    if coords.frame != "equatorial" and coords.frame != "ecliptic":
        raise ValueError("".join(frame_err).format("frame_in"))

    if frame_out is not None:
        if frame_out != "equatorial" and frame_out != "ecliptic":
            raise ValueError("".join(frame_err).format("frame_out"))
    else:
        frame_out = coords.frame

    origin_err = ["{} should be one of:\n", "'heliocenter' or 'barycenter'"]
    if origin_out is not None:
        if origin_out != "barycenter" and origin_out != "heliocenter":
            raise ValueError("".join(origin_err).format("origin_out"))
    else:
        origin_out = coords.origin

    # Check that representation_in and representation_out are one of cartesian
    # or spherical, raise errors otherwise
    representation_err = [
        "{} should be one of:\n",
        "'cartesian', 'spherical', 'keplerian', 'cometary'",
    ]
    if representation_out not in (
        "cartesian",
        "spherical",
        "keplerian",
        "cometary",
    ):
        raise ValueError("".join(representation_err).format("representation_out"))

    # If coords are already in the desired frame, have the desired origin and representation
    # then return them unaltered
    if (
        coords.frame == frame_out
        and origin_out is None
        and np.all(coords.origin == origin_out)
    ):
        if (
            isinstance(coords, CartesianCoordinates)
            and representation_out == "cartesian"
        ):
            return coords
        elif (
            isinstance(coords, SphericalCoordinates)
            and representation_out == "spherical"
        ):
            return coords
        elif (
            isinstance(coords, KeplerianCoordinates)
            and representation_out == "keplerian"
        ):
            return coords
        elif (
            isinstance(coords, CometaryCoordinates) and representation_out == "cometary"
        ):
            return coords
        else:
            pass

    # At this point, some form of transformation is going to occur so make a
    # copy of the coords and then convert the coords to Cartesian if they aren't already and make sure
    # the units match the default units assumed for each class
    coords_ = deepcopy(coords)

    set_rho_nan = False
    set_vrho_nan = False
    if isinstance(coords_, CartesianCoordinates):
        if not coords_.has_units(CARTESIAN_UNITS):
            logger.info(
                "Cartesian coordinates do not have default units, converting units before transforming."
            )
            coords_ = convert_coordinates(coords_, CARTESIAN_UNITS)
        cartesian = coords_

    elif isinstance(coords_, SphericalCoordinates):
        if not coords_.has_units(SPHERICAL_UNITS):
            logger.info(
                "Spherical coordinates do not have default units, converting units before transforming."
            )
            coords_ = convert_coordinates(coords_, SPHERICAL_UNITS)

        if representation_out == "spherical" or representation_out == "cartesian":
            if unit_sphere:
                set_rho_nan = True
                if np.all(np.isnan(coords_.rho.filled())):
                    logger.debug(
                        "Spherical coordinates have no defined radial distance (rho),"
                        "assuming spherical coordinates lie on unit sphere."
                    )
                    coords_.values[:, 0] = 1.0
                    coords._values.mask[:, 0] = 0

                set_vrho_nan = True
                if np.all(np.isnan(coords_.vrho.filled())):
                    logger.debug(
                        "Spherical coordinates have no defined radial velocity (vrho),"
                        " assuming spherical coordinates lie on unit sphere with zero velocity."
                    )
                    coords_.covariances[:, 3, :] = 0.0
                    coords_.covariances[:, :, 3] = 0.0

        cartesian = coords_.to_cartesian()

    elif isinstance(coords_, KeplerianCoordinates):
        if not coords_.has_units(KEPLERIAN_UNITS):
            logger.info(
                "Keplerian coordinates do not have default units, converting units before transforming."
            )
            coords_ = convert_coordinates(coords_, KEPLERIAN_UNITS)

        cartesian = coords_.to_cartesian()

    elif isinstance(coords_, CometaryCoordinates):
        if not coords_.has_units(COMETARY_UNITS):
            logger.info(
                "Cometary coordinates do not have default units, converting units before transforming."
            )
            coords_ = convert_coordinates(coords_, COMETARY_UNITS)

        cartesian = coords_.to_cartesian()

    # Translate coordinates to new origin (if different from current)
    if origin_out is not None and np.all(cartesian.origin != origin_out):
        cartesian = cartesian_to_origin(cartesian, origin_out)

    # Rotate coordinates to new frame (if different from current)
    if cartesian.frame != frame_out:
        cartesian = cartesian_to_frame(cartesian, frame_out)

    if representation_out == "spherical":
        coords_out = SphericalCoordinates.from_cartesian(cartesian)

        # If we assumed the coordinates lie on a unit sphere and the
        # rho and vrho values were assumed then make sure the output coordinates
        # and covariances are set back to NaN values and masked
        if set_rho_nan:
            coords_out._values[:, 0] = np.NaN
            coords_out._values[:, 0].mask = 1

        if set_vrho_nan:
            coords_out._values[:, 3] = np.NaN
            coords_out._values[:, 3].mask = 1

    elif representation_out == "keplerian":
        coords_out = KeplerianCoordinates.from_cartesian(cartesian)
    elif representation_out == "cometary":
        coords_out = CometaryCoordinates.from_cartesian(cartesian)
    elif representation_out == "cartesian":
        coords_out = cartesian

    return coords_out
