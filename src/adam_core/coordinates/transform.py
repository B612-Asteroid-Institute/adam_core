from __future__ import annotations

import logging
from typing import Literal, Optional, Union

import jax.numpy as jnp
import numpy as np
import pyarrow.compute as pc
from jax import config, jit, lax, vmap

from ..constants import Constants as c
from ..utils.chunking import process_in_chunks
from . import types
from .cartesian import CartesianCoordinates
from .cometary import CometaryCoordinates
from .keplerian import KeplerianCoordinates
from .origin import OriginCodes
from .spherical import SphericalCoordinates

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

TRANSFORM_EQ2EC = np.zeros((6, 6))
TRANSFORM_EQ2EC[0:3, 0:3] = c.TRANSFORM_EQ2EC
TRANSFORM_EQ2EC[3:6, 3:6] = c.TRANSFORM_EQ2EC
TRANSFORM_EC2EQ = TRANSFORM_EQ2EC.T

logger = logging.getLogger(__name__)


CoordinatesClasses = (
    CartesianCoordinates,
    KeplerianCoordinates,
    CometaryCoordinates,
    SphericalCoordinates,
)


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
_cartesian_to_spherical_vmap = jit(
    vmap(
        _cartesian_to_spherical,
        in_axes=(0,),
    )
)


def cartesian_to_spherical(coords_cartesian: np.ndarray) -> np.ndarray:
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
    # Define chunk size
    chunk_size = 200

    # Process in chunk
    coords_spherical: np.ndarray = np.empty((0, 6))
    for cartesian_chunk in process_in_chunks(coords_cartesian, chunk_size):
        coords_spherical_chunk = _cartesian_to_spherical_vmap(cartesian_chunk)
        coords_spherical = np.concatenate(
            (coords_spherical, np.asarray(coords_spherical_chunk))
        )

    # Concatenate chunks and remove padding
    coords_spherical = coords_spherical[: len(coords_cartesian)]

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
_spherical_to_cartesian_vmap = jit(
    vmap(
        _spherical_to_cartesian,
        in_axes=(0,),
    )
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
    # Define chunk size
    chunk_size = 200

    # Process in chunks
    coords_cartesian: np.ndarray = np.empty((0, 6))
    for spherical_chunk in process_in_chunks(coords_spherical, chunk_size):
        coords_cartesian_chunk = _spherical_to_cartesian_vmap(spherical_chunk)
        coords_cartesian = np.concatenate(
            (coords_cartesian, np.asarray(coords_cartesian_chunk))
        )

    # Remove padding
    coords_cartesian = coords_cartesian[: len(coords_spherical)]

    return coords_cartesian


@jit
def _cartesian_to_keplerian(
    coords_cartesian: Union[np.ndarray, jnp.ndarray],
    t0: float,
    mu: float,
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
    mu : float (1)
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
    from ..dynamics.kepler import (
        calc_mean_anomaly,
        calc_mean_motion,
        calc_periapsis_distance,
    )

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
        calc_periapsis_distance(a, e),
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
        lambda a, q: calc_mean_motion(a, mu),
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
_cartesian_to_keplerian_vmap = jit(
    vmap(
        _cartesian_to_keplerian,
        in_axes=(0, 0, 0),
    )
)


@jit
def _cartesian_to_keplerian6(
    coords_cartesian: Union[np.ndarray, jnp.ndarray],
    t0: float,
    mu: float,
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
    mu : float (1)
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
    mu: Union[np.ndarray, jnp.ndarray],
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
    mu : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (N)
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
    # Define chunk size
    chunk_size = 200

    # Process in chunks
    coords_keplerian_chunks = []
    for cartesian_chunk, t0_chunk, mu_chunk in zip(
        process_in_chunks(coords_cartesian, chunk_size),
        process_in_chunks(t0, chunk_size),
        process_in_chunks(mu, chunk_size),
    ):
        coords_keplerian_chunk = _cartesian_to_keplerian_vmap(
            cartesian_chunk, t0_chunk, mu_chunk
        )
        coords_keplerian_chunks.append(coords_keplerian_chunk)

    # Concatenate chunks and remove padding
    coords_keplerian = jnp.concatenate(coords_keplerian_chunks, axis=0)
    coords_keplerian = coords_keplerian[: len(coords_cartesian)]

    return coords_keplerian


@jit
def _keplerian_to_cartesian_p(
    coords_keplerian: Union[np.ndarray, jnp.ndarray],
    mu: float,
    max_iter: int = 1000,
    tol: float = 1e-15,
) -> jnp.ndarray:
    """
    Convert a single Keplerian coordinate to a Cartesian coordinate.

    Parabolic orbits (e = 1.0 +- 1e-15) with elements (a, e, i, raan, ap, M) cannot be converted
    to Cartesian orbits since their semi-major axes are by definition undefined.
    Please consider representing the orbits with Cometary elements
    and using those to convert to Cartesian. See `~adam_core.coordinates.cometary._cometary_to_cartesian`.

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
    mu : float
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
    from ..dynamics.barker import solve_barker
    from ..dynamics.kepler import solve_kepler

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
_keplerian_to_cartesian_p_vmap = jit(
    vmap(
        _keplerian_to_cartesian_p,
        in_axes=(0, 0, None, None),
    )
)


@jit
def _keplerian_to_cartesian_a(
    coords_keplerian: Union[np.ndarray, jnp.ndarray],
    mu: float,
    max_iter: int = 1000,
    tol: float = 1e-15,
) -> jnp.ndarray:
    """
    Convert a single Keplerian coordinate to a Cartesian coordinate.

    Parabolic orbits (e = 1.0 +- 1e-15) with elements (a, e, i, raan, ap, M) cannot be converted
    to Cartesian orbits since their semi-major axes are by definition undefined.
    Please consider representing the orbits with Cometary elements
    and using those to convert to Cartesian. See `~adam_core.coordinates.cometary._cometary_to_cartesian`.

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
    mu : float (1)
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
_keplerian_to_cartesian_a_vmap = jit(
    vmap(
        _keplerian_to_cartesian_a,
        in_axes=(0, 0, None, None),
    )
)


@jit
def _keplerian_to_cartesian_q(
    coords_keplerian: Union[np.ndarray, jnp.ndarray],
    mu: float,
    max_iter: int = 1000,
    tol: float = 1e-15,
) -> jnp.ndarray:
    """
    Convert a single Keplerian coordinate to a Cartesian coordinate.

    Parabolic orbits (e = 1.0 +- 1e-15) with elements (a, e, i, raan, ap, M) cannot be converted
    to Cartesian orbits since their semi-major axes are by definition undefined.
    Please consider representing the orbits with Cometary elements
    and using those to convert to Cartesian. See `~adam_core.coordinates.cometary._cometary_to_cartesian`.

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
    mu : float (1)
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
_keplerian_to_cartesian_q_vmap = jit(
    vmap(
        _keplerian_to_cartesian_q,
        in_axes=(0, 0, None, None),
    )
)


def keplerian_to_cartesian(
    coords_keplerian: Union[np.ndarray, jnp.ndarray],
    mu: Union[np.ndarray, jnp.ndarray],
    max_iter: int = 100,
    tol: float = 1e-15,
) -> jnp.ndarray:
    """
    Convert Keplerian coordinates to Cartesian coordinates.

    Parabolic orbits (e = 1.0 +- 1e-15) with elements (a, e, i, raan, ap, M) cannot be converted
    to Cartesian orbits since their semi-major axes are by definition undefined.
    Please consider representing these orbits with Cometary elements
    and using those to convert to Cartesian. See `~adam_core.coordinates.cometary.cometary_to_cartesian`.

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
    mu : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (N)
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

    # Define chunk size
    chunk_size = 200

    # Process in chunks
    coords_cartesian_chunks = []
    for keplerian_chunk, mu_chunk in zip(
        process_in_chunks(coords_keplerian, chunk_size),
        process_in_chunks(mu, chunk_size),
    ):
        coords_cartesian_chunk = _keplerian_to_cartesian_a_vmap(
            keplerian_chunk, mu_chunk, max_iter, tol
        )
        coords_cartesian_chunks.append(coords_cartesian_chunk)

    # Concatenate chunks and remove padding
    coords_cartesian = jnp.concatenate(coords_cartesian_chunks, axis=0)
    coords_cartesian = coords_cartesian[: len(coords_keplerian)]

    return coords_cartesian


@jit
def _cartesian_to_cometary(
    coords_cartesian: Union[np.ndarray, jnp.ndarray],
    t0: float,
    mu: float,
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
    mu : float (1)
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
_cartesian_to_cometary_vmap = jit(
    vmap(
        _cartesian_to_cometary,
        in_axes=(0, 0, 0),
    )
)


def cartesian_to_cometary(
    coords_cartesian: Union[np.ndarray, jnp.ndarray],
    t0: Union[np.ndarray, jnp.ndarray],
    mu: Union[np.ndarray, jnp.ndarray],
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
    mu : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (N)
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
    mu: float,
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
    mu : float
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
_cometary_to_cartesian_vmap = jit(
    vmap(
        _cometary_to_cartesian,
        in_axes=(0, 0, 0, None, None),
    )
)


def cometary_to_cartesian(
    coords_cometary: Union[np.ndarray, jnp.ndarray],
    t0: Union[np.ndarray, jnp.ndarray],
    mu: Union[np.ndarray, jnp.ndarray],
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
    mu : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (N)
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
    # Define chunk size
    chunk_size = 200

    # Process in chunks
    coords_cartesian_chunks = []
    for cometary_chunk, t0_chunk, mu_chunk in zip(
        process_in_chunks(coords_cometary, chunk_size),
        process_in_chunks(t0, chunk_size),
        process_in_chunks(mu, chunk_size),
    ):
        coords_cartesian_chunk = _cometary_to_cartesian_vmap(
            cometary_chunk, t0_chunk, mu_chunk, max_iter, tol
        )
        coords_cartesian_chunks.append(coords_cartesian_chunk)

    # Concatenate chunks and remove padding
    coords_cartesian = jnp.concatenate(coords_cartesian_chunks, axis=0)
    coords_cartesian = coords_cartesian[: len(coords_cometary)]

    return coords_cartesian


def cartesian_to_origin(
    coords: CartesianCoordinates, origin: OriginCodes
) -> "CartesianCoordinates":
    """
    Translate coordinates to a different origin.

    Parameters
    ----------
    coords : `~adam_core.coordinates.cartesian.CartesianCoordinates`
        Cartesian coordinates and optionally their covariances.
    origin : `~adam_core.coordinates.origin.OriginCodes`
        Desired origin. Input origins may be either `~adam_core.coordinates.origin.OriginCodes`
        or a str of an observatory code, but the output origin (this kwarg) should always be an
        `~adam_core.coordinates.origin.OriginCodes`. If you are looking to generate ephemerides
        for an observatory, please use a `~adam_core.propagator.propagator.Propagator` instead.

    Returns
    -------
    CartesianCoordinates : `~adam_core.coordinates.cartesian.CartesianCoordinates`
        Translated Cartesian coordinates and their covariances.

    Raises
    ------
    ValueError
        If origin is not a `~adam_core.coordinates.origin.OriginCodes` object or
        a str of an observatory code.
    """
    from ..observers.state import OBSERVATORY_CODES, get_observer_state
    from ..utils.spice import get_perturber_state

    unique_origins = coords.origin.code.unique()
    vectors = np.empty(coords.values.shape, dtype=np.float64)
    times = coords.time

    for origin_in in unique_origins:

        mask = pc.equal(coords.origin.code, origin_in).to_numpy(zero_copy_only=False)

        origin_in_str = origin_in.as_py()
        # Could use try / except block here but this is more explicit
        if origin_in_str in OriginCodes.__members__:

            vectors[mask] = get_perturber_state(
                OriginCodes[origin_in_str],
                times.apply_mask(mask),
                frame=coords.frame,
                origin=origin,
            ).values

        elif origin_in_str in OBSERVATORY_CODES:

            vectors[mask] = get_observer_state(
                origin_in_str,
                times.apply_mask(mask),
                frame=coords.frame,
                origin=origin,
            ).values

        else:
            raise ValueError("Unsupported origin: {}".format(origin_in_str))

    return coords.translate(vectors, origin.name)


def cartesian_to_frame(
    coords: CartesianCoordinates, frame: Literal["ecliptic", "equatorial"]
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
    CartesianCoordinates : `~adam_core.coordinates.cartesian.CartesianCoordinates`
        Rotated Cartesian coordinates and their covariances.
    """
    if frame == "ecliptic" and coords.frame != "ecliptic":
        return coords.rotate(TRANSFORM_EQ2EC, "ecliptic")
    elif frame == "equatorial" and coords.frame != "equatorial":
        return coords.rotate(TRANSFORM_EC2EQ, "equatorial")
    elif frame == coords.frame:
        return coords
    else:
        err = "frame should be one of {'ecliptic', 'equatorial'}"
        raise ValueError(err)


def transform_coordinates(
    coords: types.CoordinateType,
    representation_out: Optional[type[types.CoordinateType]] = None,
    frame_out: Optional[Literal["ecliptic", "equatorial"]] = None,
    origin_out: Optional[OriginCodes] = None,
) -> types.CoordinateType:
    """
    Transform coordinates between frames ('ecliptic', 'equatorial'), origins,
    and/or representations ('cartesian', 'spherical', 'keplerian', 'cometary').

    Input coordinates may be defined from multiple origins but if origin_out is
    specified, all coordinates will be transformed to that origin.

    Parameters
    ----------
    coords:
        Coordinates to transform between representations and frames.
    representation_out:
        Desired coordinate type or representation of the output coordinates. If None,
        the output coordinates will be the same type as the input coordinates.
    frame_out : {'ecliptic', 'equatorial'}
        Desired reference frame of the output coordinates.
    origin_out : `~adam_core.coordinates.origin.OriginCodes`
        Desired origin. Input origins may be either `~adam_core.coordinates.origin.OriginCodes`
        or a str of an observatory code, but the output origin (this kwarg) should always be an
        `~adam_core.coordinates.origin.OriginCodes`. If you are looking to generate ephemerides
        for an observatory, please use a `~adam_core.propagator.propagator.Propagator` instead.

    Returns
    -------
    coords_out : `~adam_core.coordinates.Coordinates`
        Coordinates in desired output representation and frame.

    Raises
    ------
    ValueError
        If frame_in, frame_out are not one of 'equatorial', 'ecliptic'.
        If representation_in, representation_out are not one of 'cartesian',
            'spherical', 'keplerian', 'cometary'.
    TypeError
        If coords is not a `~adam_core.coordinates.Coordinates` object.
        If origin_out is not a `~adam_core.coordinates.OriginCodes` object.
    """
    # Check that coords is a thor.coordinates.Coordinates object
    if not isinstance(coords, CoordinatesClasses):
        raise TypeError("Unsupported coordinate type: {}".format(type(coords)))

    if frame_out not in {None, "equatorial", "ecliptic"}:
        raise ValueError("Unsupported frame_out: {}".format(frame_out))

    if origin_out is not None and not isinstance(origin_out, OriginCodes):
        raise TypeError("Unsupported origin_out type: {}".format(type(origin_out)))

    if representation_out is None:
        representation_out_ = coords.__class__
    else:
        representation_out_ = representation_out

    if representation_out_ not in CoordinatesClasses:
        raise ValueError(
            "Unsupported representation_out: {}".format(representation_out_)
        )

    coord_frame = coords.frame
    # Extract the origins from the input coordinates. These typically correspond
    # to the name of OriginCode enums but stored as an array of strings.
    coord_origin = coords.origin.code.to_numpy(zero_copy_only=False)

    if frame_out is None:
        frame_out = coord_frame

    # If origin out is not None, then origin_out will be an OriginCode
    # passed directly to this function. Otherwise, it will be an array of strings
    # extracted from the input coordinates.
    if origin_out is None:
        origin_out = coord_origin

    # `~adam_core.coordinates.origin.Origin` support equality checks with
    # `~adam_core.coordinates.origin.OriginCodes` so we can compare them directly.
    # If its not an OriginCodes enum then origin_out will be an array of strings which
    # also can be checked for equality.
    if type(coords) is representation_out_:
        if coord_frame == frame_out and np.all(coord_origin == origin_out):
            return coords

    if not isinstance(coords, CartesianCoordinates):
        cartesian = coords.to_cartesian()
    else:
        cartesian = coords

    # Translate coordinates to new origin (if any are different from current)
    if np.any(cartesian.origin != origin_out):
        cartesian = cartesian_to_origin(cartesian, origin_out)

    # Rotate coordinates to new frame (if different from current)
    if cartesian.frame != frame_out:
        cartesian = cartesian_to_frame(cartesian, frame_out)

    # You might think this should be 'isinstance', but no! We're
    # checking whether the input is a particular class variable, not
    # an instance of a class.
    if representation_out_ is CartesianCoordinates:
        return cartesian
    return representation_out_.from_cartesian(cartesian)
