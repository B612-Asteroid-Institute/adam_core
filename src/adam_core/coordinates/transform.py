from __future__ import annotations

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pyarrow.compute as pc

from .._rust import (
    cartesian_to_cometary_numpy,
    cartesian_to_geodetic_numpy,
    cartesian_to_keplerian_numpy,
    cartesian_to_spherical_numpy,
    cometary_to_cartesian_numpy,
    keplerian_to_cartesian_numpy,
    rotate_cartesian_time_varying_numpy,
    spherical_to_cartesian_numpy,
    transform_coordinates_numpy,
    transform_coordinates_with_covariance_numpy,
)
from ..constants import Constants as c
from ..utils.bounded_lru import bounded_lru_get, bounded_lru_put
from . import types
from .cartesian import COVARIANCE_ROTATION_TOLERANCE, CartesianCoordinates
from .cometary import CometaryCoordinates
from .covariances import CoordinateCovariances
from .geodetics import GeodeticCoordinates, WGS84
from .keplerian import KeplerianCoordinates
from .origin import Origin, OriginCodes
from .spherical import SphericalCoordinates


TRANSFORM_EQ2EC = np.zeros((6, 6))
TRANSFORM_EQ2EC[0:3, 0:3] = c.TRANSFORM_EQ2EC
TRANSFORM_EQ2EC[3:6, 3:6] = c.TRANSFORM_EQ2EC
TRANSFORM_EC2EQ = TRANSFORM_EQ2EC.T

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TranslationCacheKey:
    origin_in: str
    origin_out: str
    frame: str
    n: int
    first_key: int
    last_key: int
    time_digest: int


_TRANSLATION_CACHE_MAXSIZE = int(
    os.environ.get("ADAM_CORE_TRANSLATION_CACHE_MAXSIZE", "2048")
)
_TRANSLATION_CACHE: "OrderedDict[_TranslationCacheKey, np.ndarray]" = OrderedDict()
_TRANSLATION_CACHE_ENABLED = os.environ.get(
    "ADAM_CORE_TRANSLATION_CACHE", "1"
).lower() not in {
    "0",
    "false",
    "no",
}
_TRANSLATION_CACHE_ALLOWED = {
    ("SUN", "SOLAR_SYSTEM_BARYCENTER"),
    ("SOLAR_SYSTEM_BARYCENTER", "SUN"),
}


def _translation_cache_get(key: _TranslationCacheKey) -> np.ndarray | None:
    return bounded_lru_get(_TRANSLATION_CACHE, key, maxsize=_TRANSLATION_CACHE_MAXSIZE)


def _translation_cache_put(key: _TranslationCacheKey, vectors: np.ndarray) -> None:
    bounded_lru_put(
        _TRANSLATION_CACHE, key, vectors, maxsize=_TRANSLATION_CACHE_MAXSIZE
    )


def clear_translation_cache() -> None:
    """
    Clear the in-process translation cache used by `cartesian_to_origin`.

    Primarily intended for testing and benchmarking.
    """
    _TRANSLATION_CACHE.clear()


CoordinatesClasses = (
    CartesianCoordinates,
    KeplerianCoordinates,
    CometaryCoordinates,
    SphericalCoordinates,
    GeodeticCoordinates,
)


FLOAT_TOLERANCE = 1e-15

_RUST_TRANSFORM_REPRESENTATIONS = {
    CartesianCoordinates: "cartesian",
    SphericalCoordinates: "spherical",
    GeodeticCoordinates: "geodetic",
    KeplerianCoordinates: "keplerian",
    CometaryCoordinates: "cometary",
}


def _coordinates_from_rust_values(
    values: np.ndarray,
    coords_in: types.CoordinateType,
    representation_out: type[types.CoordinateType],
    frame_out: Literal["ecliptic", "equatorial", "itrf93"],
    covariance: CoordinateCovariances | None = None,
) -> types.CoordinateType:
    cov = coords_in.covariance if covariance is None else covariance
    if representation_out is CartesianCoordinates:
        return CartesianCoordinates.from_kwargs(
            x=values[:, 0],
            y=values[:, 1],
            z=values[:, 2],
            vx=values[:, 3],
            vy=values[:, 4],
            vz=values[:, 5],
            time=coords_in.time,
            covariance=cov,
            origin=coords_in.origin,
            frame=frame_out,
        )
    if representation_out is SphericalCoordinates:
        return SphericalCoordinates.from_kwargs(
            rho=values[:, 0],
            lon=values[:, 1],
            lat=values[:, 2],
            vrho=values[:, 3],
            vlon=values[:, 4],
            vlat=values[:, 5],
            time=coords_in.time,
            covariance=cov,
            origin=coords_in.origin,
            frame=frame_out,
        )
    if representation_out is GeodeticCoordinates:
        return GeodeticCoordinates.from_kwargs(
            alt=values[:, 0],
            lon=values[:, 1],
            lat=values[:, 2],
            vup=values[:, 3],
            veast=values[:, 4],
            vnorth=values[:, 5],
            time=coords_in.time,
            covariance=cov,
            origin=coords_in.origin,
            frame=frame_out,
        )
    if representation_out is KeplerianCoordinates:
        # The non-covariance path (cartesian_to_keplerian_flat6) returns 13 columns
        # (a, Q, q, e, i, raan, ap, M, nu, E, P, T, tp) for legacy compat; the
        # covariance path evaluates the 6-column generic kernel to keep the Jacobian
        # square. Dispatch on column count.
        if values.shape[1] == 13:
            kep_cols = (0, 4, 5, 6, 7, 8)
        else:
            kep_cols = (0, 1, 2, 3, 4, 5)
        a_col, e_col, i_col, raan_col, ap_col, m_col = kep_cols
        return KeplerianCoordinates.from_kwargs(
            a=values[:, a_col],
            e=values[:, e_col],
            i=values[:, i_col],
            raan=values[:, raan_col],
            ap=values[:, ap_col],
            M=values[:, m_col],
            time=coords_in.time,
            covariance=cov,
            origin=coords_in.origin,
            frame=frame_out,
        )
    if representation_out is CometaryCoordinates:
        return CometaryCoordinates.from_kwargs(
            q=values[:, 0],
            e=values[:, 1],
            i=values[:, 2],
            raan=values[:, 3],
            ap=values[:, 4],
            tp=values[:, 5],
            time=coords_in.time,
            covariance=cov,
            origin=coords_in.origin,
            frame=frame_out,
        )
    raise ValueError(f"Unsupported representation_out: {representation_out}")


_RUST_TRANSFORM_INPUT_TYPES = (
    CartesianCoordinates,
    SphericalCoordinates,
    KeplerianCoordinates,
    CometaryCoordinates,
)


def _rust_transform_supports(
    coords: types.CoordinateType,
    representation_out: type[types.CoordinateType],
    frame_out: Literal["ecliptic", "equatorial", "itrf93"],
    origin_out: OriginCodes | np.ndarray,
) -> bool:
    """
    Return True if the Rust single-crossing path covers this exact input combination.

    The supported set is explicit: input representation in
    (Cartesian, Spherical, Keplerian, Cometary), output representation in
    (Cartesian, Spherical, Keplerian, Cometary, Geodetic), same origin in/out
    (EARTH only for Geodetic output), covariance allowed (propagated via
    forward-mode autodiff in Rust; NaN covariance passes through), and frame
    transitions limited to identity or equatorial<->ecliptic (plus itrf93 for
    Geodetic output). Anything outside this set must fall back to legacy.
    """
    if type(coords) not in _RUST_TRANSFORM_INPUT_TYPES:
        return False
    if representation_out not in _RUST_TRANSFORM_REPRESENTATIONS:
        return False
    if frame_out != coords.frame:
        if {coords.frame, frame_out} == {"ecliptic", "equatorial"}:
            pass  # single-crossing Rust kernel rotates inline
        elif "itrf93" in (coords.frame, frame_out):
            # Time-varying ITRF93 rotation needs Cartesian states (no
            # physically meaningful ITRF93 Keplerian/Cometary/Spherical
            # representation to rotate from). Dispatcher routes through
            # rotate_cartesian_time_varying + identity-frame conversion.
            if type(coords) is not CartesianCoordinates:
                return False
        else:
            return False
    if representation_out is GeodeticCoordinates and frame_out != "itrf93":
        return False
    origin_differs = bool(np.any(coords.origin != origin_out))
    if origin_differs:
        # The dispatcher handles origin changes by routing through a
        # SPICE-backed Cartesian translation (covariance-invariant).
        # This needs a single target origin (`OriginCodes`); mixed-origin
        # arrays aren't a supported target.
        if not isinstance(origin_out, OriginCodes):
            return False
        # Geodetic output must be EARTH-centered ITRF93; if we're asking
        # for a different origin-out, we can't satisfy it.
        if representation_out is GeodeticCoordinates:
            if origin_out is not OriginCodes.EARTH:
                return False
    if representation_out is GeodeticCoordinates:
        origin_codes = coords.origin.code.to_numpy(zero_copy_only=False)
        if np.any(origin_codes != "EARTH") and not origin_differs:
            return False
    # Pure Cartesian frame-only change is faster on the legacy path (a single
    # 6x6 rotation) than through the Rust dispatcher's Python marshaling; keep
    # it on legacy so the promotion gate holds for all covered workloads.
    if (
        type(coords) is CartesianCoordinates
        and representation_out is CartesianCoordinates
    ):
        return False
    # Cartesian output with origin change: the dispatcher would just do
    # to_cartesian+translate+rotate, which is exactly what the legacy
    # fallthrough does (all three hops are already Rust after lever 1).
    # Going through the dispatcher adds Python marshaling overhead with no
    # fusion win, so keep this on legacy.
    if representation_out is CartesianCoordinates and origin_differs:
        return False
    return True


def _try_transform_coordinates_rust(
    coords: types.CoordinateType,
    representation_out: type[types.CoordinateType],
    frame_out: Literal["ecliptic", "equatorial", "itrf93"],
    origin_out: OriginCodes | np.ndarray,
) -> types.CoordinateType | None:
    """
    Attempt the Rust single-crossing transform path.

    Returns the transformed coordinates when the input combination is in the
    supported set (see `_rust_transform_supports`), otherwise returns None so
    the caller falls back to the legacy JAX path. Callers must treat a None
    return as an explicit "not supported by Rust yet" signal, not an error.
    """
    if not _rust_transform_supports(coords, representation_out, frame_out, origin_out):
        return None

    # Origin change: resolve the per-row SPICE translation vector in the
    # input frame, then pass it to the Rust dispatcher so the add fuses
    # with rep-in -> cart -> frame rotate -> rep-out -> covariance AD in a
    # single Python/Rust crossing. Translation is a constant offset
    # (identity Jacobian wrt state), so covariance propagation is
    # unaffected mathematically; the kernel just skips adding anything to
    # the Jacobian accumulation.
    origin_differs = bool(np.any(coords.origin != origin_out))
    translation_vectors: np.ndarray | None = None
    if origin_differs:
        assert isinstance(origin_out, OriginCodes)
        if type(coords) is CartesianCoordinates:
            translation_vectors = _resolve_origin_translation_vectors(coords, origin_out)
        else:
            # SPICE resolves translation vectors in Cartesian space, so we
            # need the rep-in-as-Cartesian (same frame, same origin) to
            # resolve them. That's a pure origin-query operation against
            # the *input* origin metadata — it doesn't depend on the
            # actual Cartesian values — so we build a synthetic Cartesian
            # view carrying just the time/origin/frame.
            synthetic = CartesianCoordinates.from_kwargs(
                x=np.zeros(len(coords)),
                y=np.zeros(len(coords)),
                z=np.zeros(len(coords)),
                vx=np.zeros(len(coords)),
                vy=np.zeros(len(coords)),
                vz=np.zeros(len(coords)),
                time=coords.time,
                origin=coords.origin,
                frame=coords.frame,
            )
            translation_vectors = _resolve_origin_translation_vectors(synthetic, origin_out)

    # ITRF93 frame changes: do the time-varying rotation in Rust first, then
    # re-dispatch as an identity-frame representation conversion so the
    # covariance transform runs through Rust forward-mode autodiff instead
    # of the JAX `from_cartesian` Jacobian path.
    if frame_out != coords.frame and "itrf93" in (coords.frame, frame_out):
        # ITRF93 path cannot combine with origin_differs (gated out in
        # _rust_transform_supports for non-EARTH targets, and ITRF93 with
        # EARTH target would still require the time-varying rotate path
        # first, then translate — handle via fallthrough here).
        if origin_differs:
            translated = cartesian_to_origin(coords, origin_out)
            rotated = apply_time_varying_rotation(translated, frame_out)
        else:
            rotated = apply_time_varying_rotation(coords, frame_out)
        if representation_out is CartesianCoordinates:
            return rotated
        return _try_transform_coordinates_rust(
            rotated, representation_out, frame_out, origin_out
        )

    representation_in_name = _RUST_TRANSFORM_REPRESENTATIONS[type(coords)]
    representation_out_name = _RUST_TRANSFORM_REPRESENTATIONS[representation_out]

    t0 = None
    mu = None
    a = None
    f = None
    if type(coords) is KeplerianCoordinates:
        mu = np.ascontiguousarray(coords.origin.mu(), dtype=np.float64)
    if type(coords) is CometaryCoordinates:
        t0 = np.ascontiguousarray(coords.time.to_numpy(), dtype=np.float64)
        mu = np.ascontiguousarray(coords.origin.mu(), dtype=np.float64)
    if representation_out is GeodeticCoordinates:
        a = float(WGS84.a)
        f = float(WGS84.f)
    if representation_out is KeplerianCoordinates:
        if t0 is None:
            t0 = np.ascontiguousarray(coords.time.to_numpy(), dtype=np.float64)
        if mu is None:
            mu = np.ascontiguousarray(coords.origin.mu(), dtype=np.float64)
    if representation_out is CometaryCoordinates:
        if t0 is None:
            t0 = np.ascontiguousarray(coords.time.to_numpy(), dtype=np.float64)
        if mu is None:
            mu = np.ascontiguousarray(coords.origin.mu(), dtype=np.float64)

    translation_kwarg = (
        np.ascontiguousarray(translation_vectors, dtype=np.float64)
        if translation_vectors is not None
        else None
    )

    if coords.covariance.is_all_nan():
        rust_coords = transform_coordinates_numpy(
            coords.values,
            representation_in_name,
            representation_out_name,
            t0=t0,
            mu=mu,
            a=a,
            f=f,
            max_iter=100,
            tol=1e-15,
            frame_in=coords.frame,
            frame_out=frame_out,
            translation_vectors=translation_kwarg,
        )
        if rust_coords is None:
            return None
        result = _coordinates_from_rust_values(
            np.asarray(rust_coords, dtype=np.float64),
            coords,
            representation_out,
            frame_out,
        )
        if origin_differs:
            result = result.set_column(
                "origin", Origin.from_kwargs(code=[origin_out.name] * len(result))
            )
        return result

    covariance_matrices = coords.covariance.to_matrix()
    n = covariance_matrices.shape[0]
    covariance_flat = np.ascontiguousarray(
        covariance_matrices.reshape(n, 36), dtype=np.float64
    )
    result = transform_coordinates_with_covariance_numpy(
        coords.values,
        covariance_flat,
        representation_in_name,
        representation_out_name,
        t0=t0,
        mu=mu,
        a=a,
        f=f,
        max_iter=100,
        tol=1e-15,
        frame_in=coords.frame,
        frame_out=frame_out,
        translation_vectors=translation_kwarg,
    )
    if result is None:
        return None
    rust_coords, rust_cov_flat = result
    cov_out = CoordinateCovariances.from_matrix(
        np.asarray(rust_cov_flat, dtype=np.float64).reshape(n, 6, 6)
    )
    out = _coordinates_from_rust_values(
        np.asarray(rust_coords, dtype=np.float64),
        coords,
        representation_out,
        frame_out,
        covariance=cov_out,
    )
    if origin_differs:
        out = out.set_column(
            "origin", Origin.from_kwargs(code=[origin_out.name] * len(out))
        )
    return out


def cartesian_to_geodetic(
    coords_cartesian: np.ndarray,
    a: float,
    f: float,
    max_iter: int = 100,
    tol: float = 1e-15,
) -> np.ndarray:
    """
    Convert Cartesian coordinates to a geodetic coordinate.

    Parameters
    ----------
    coords_cartesian : {`~numpy.ndarray`, `~jax.numpy.ndarray`} (N, 6)
        3D Cartesian coordinate including time derivatives.
        x : x-position in units of distance.
        y : y-position in units of distance.
        z : z-position in units of distance.
        vx : x-velocity in the same units of x per arbitrary unit of time.
        vy : y-velocity in the same units of y per arbitrary unit of time.
        vz : z-velocity in the same units of z per arbitrary unit of time.
    a : float (1)
        Semi-major axis of the Earth in units of distance.
    f : float (1)
        Flattening of the Earth.
    max_iter : int (1)
        Maximum number of iterations to perform.
    tol : float (1)
        Tolerance for the iteration.

    Returns
    -------
    coords_geodetic : `~jax.numpy.ndarray` (N, 6)
        3D geodetic coordinate including time derivatives.
        alt : Altitude in units of distance.
        lon : Longitude in degrees.
        lat : Latitude in degrees.
        vup : Up velocity in the same units of x per arbitrary unit of time.
        veast : East velocity in degrees per arbitrary unit of time.
        vnorth : North velocity in degrees per arbitrary unit of time.
    """
    coords_cartesian_np = np.ascontiguousarray(
        np.asarray(coords_cartesian, dtype=np.float64)
    )
    if coords_cartesian_np.ndim != 2 or coords_cartesian_np.shape[1] != 6:
        raise ValueError("coords_cartesian must have shape (N, 6)")

    rust_coords = cartesian_to_geodetic_numpy(coords_cartesian_np, a, f, max_iter, tol)
    assert rust_coords is not None
    return rust_coords


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
    coords_cartesian_np = np.ascontiguousarray(
        np.asarray(coords_cartesian, dtype=np.float64)
    )
    if coords_cartesian_np.ndim != 2 or coords_cartesian_np.shape[1] != 6:
        raise ValueError("coords_cartesian must have shape (N, 6)")

    rust_coords = cartesian_to_spherical_numpy(coords_cartesian_np)
    assert rust_coords is not None
    return rust_coords


def spherical_to_cartesian(
    coords_spherical: np.ndarray,
) -> np.ndarray:
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
    coords_spherical_np = np.ascontiguousarray(
        np.asarray(coords_spherical, dtype=np.float64)
    )
    if coords_spherical_np.ndim != 2 or coords_spherical_np.shape[1] != 6:
        raise ValueError("coords_spherical must have shape (N, 6)")

    rust_coords = spherical_to_cartesian_numpy(coords_spherical_np)
    assert rust_coords is not None
    return rust_coords


def cartesian_to_keplerian(
    coords_cartesian: np.ndarray,
    t0: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
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
    coords_cartesian_np = np.ascontiguousarray(
        np.asarray(coords_cartesian, dtype=np.float64)
    )
    t0_np = np.ascontiguousarray(np.asarray(t0, dtype=np.float64))
    mu_np = np.ascontiguousarray(np.asarray(mu, dtype=np.float64))
    if coords_cartesian_np.ndim != 2 or coords_cartesian_np.shape[1] != 6:
        raise ValueError("coords_cartesian must have shape (N, 6)")
    if t0_np.ndim != 1 or mu_np.ndim != 1:
        raise ValueError("t0 and mu must be one-dimensional")
    if (
        t0_np.shape[0] != coords_cartesian_np.shape[0]
        or mu_np.shape[0] != coords_cartesian_np.shape[0]
    ):
        raise ValueError(
            "t0 and mu must each have length N for coords_cartesian shape (N, 6)"
        )

    rust_coords = cartesian_to_keplerian_numpy(coords_cartesian_np, t0_np, mu_np)
    assert rust_coords is not None
    return rust_coords


def keplerian_to_cartesian(
    coords_keplerian: np.ndarray,
    mu: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-15,
) -> np.ndarray:
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
    coords_keplerian_np = np.ascontiguousarray(
        np.asarray(coords_keplerian, dtype=np.float64)
    )
    mu_np = np.ascontiguousarray(np.asarray(mu, dtype=np.float64))
    if coords_keplerian_np.ndim != 2 or coords_keplerian_np.shape[1] != 6:
        raise ValueError("coords_keplerian must have shape (N, 6)")
    if mu_np.ndim != 1:
        raise ValueError("mu must be one-dimensional")
    if mu_np.shape[0] != coords_keplerian_np.shape[0]:
        raise ValueError("mu must have length N for coords_keplerian shape (N, 6)")

    a = coords_keplerian_np[:, 0]
    e = coords_keplerian_np[:, 1]

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

    rust_coords = keplerian_to_cartesian_numpy(
        coords_keplerian_np, mu_np, max_iter=max_iter, tol=tol
    )
    assert rust_coords is not None
    return rust_coords


def cartesian_to_cometary(
    coords_cartesian: np.ndarray,
    t0: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
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
    coords_cometary : `~numpy.ndarray` (N, 6)
        6D Cometary coordinates.
        q : periapsis distance in au.
        e : eccentricity.
        i : inclination in degrees.
        raan : Right ascension (longitude) of the ascending node in degrees.
        ap : argument of periapsis in degrees.
        tp : time of periapse passage in days.
    """
    coords_cartesian_np = np.ascontiguousarray(
        np.asarray(coords_cartesian, dtype=np.float64)
    )
    t0_np = np.ascontiguousarray(np.asarray(t0, dtype=np.float64))
    mu_np = np.ascontiguousarray(np.asarray(mu, dtype=np.float64))
    rust_coords = cartesian_to_cometary_numpy(coords_cartesian_np, t0_np, mu_np)
    assert rust_coords is not None
    return rust_coords


def cometary_to_cartesian(
    coords_cometary: np.ndarray,
    t0: np.ndarray,
    mu: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-15,
) -> np.ndarray:
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
    coords_cartesian : `~numpy.ndarray` (N, 6)
        3D Cartesian coordinates including time derivatives.
    """
    coords_cometary_np = np.ascontiguousarray(
        np.asarray(coords_cometary, dtype=np.float64)
    )
    t0_np = np.ascontiguousarray(np.asarray(t0, dtype=np.float64))
    mu_np = np.ascontiguousarray(np.asarray(mu, dtype=np.float64))
    rust_coords = cometary_to_cartesian_numpy(
        coords_cometary_np, t0_np, mu_np, max_iter=max_iter, tol=tol
    )
    assert rust_coords is not None
    return rust_coords


def _resolve_origin_translation_vectors(
    coords: CartesianCoordinates, origin: OriginCodes
) -> np.ndarray:
    """
    Resolve per-row SPICE translation vectors taking `coords` from its current
    origin(s) to `origin`, expressed in `coords.frame`.

    Returns an `(N, 6)` `float64` ndarray `v` such that
    `coords.values + v == coords_with_new_origin.values`. Uses the translation
    cache when the (origin_in, origin_out) pair is eligible.

    Callers wanting the final translated `CartesianCoordinates` should use
    `cartesian_to_origin` instead. This helper is for callers that will pass
    the translation vectors into the Rust dispatcher so the add can fuse with
    the rest of the transform chain in a single Python/Rust crossing.
    """
    from ..observers.state import OBSERVATORY_CODES, get_observer_state
    from ..utils.spice import get_perturber_state

    unique_origins = coords.origin.code.unique()
    vectors = np.empty(coords.values.shape, dtype=np.float64)
    times = coords.time

    for origin_in in unique_origins:
        mask = pc.equal(coords.origin.code, origin_in).to_numpy(zero_copy_only=False)
        origin_in_str = origin_in.as_py()
        if origin_in_str in OriginCodes.__members__:
            times_masked = times.apply_mask(mask)
            origin_out_str = str(origin.name)
            use_cache = bool(_TRANSLATION_CACHE_ENABLED) and (
                (str(origin_in_str), origin_out_str) in _TRANSLATION_CACHE_ALLOWED
            )
            if use_cache:
                n, first, last, _ = times_masked.signature(scale="tdb")
                cache_key = _TranslationCacheKey(
                    origin_in=str(origin_in_str),
                    origin_out=origin_out_str,
                    frame=str(coords.frame),
                    n=int(n),
                    first_key=int(first),
                    last_key=int(last),
                    time_digest=int(times_masked.cache_digest(scale="tdb")),
                )
                cached = _translation_cache_get(cache_key)
                if cached is None:
                    v = get_perturber_state(
                        OriginCodes[origin_in_str],
                        times_masked,
                        frame=coords.frame,
                        origin=origin,
                    ).values
                    _translation_cache_put(cache_key, v)
                    vectors[mask] = v
                else:
                    vectors[mask] = cached
            else:
                vectors[mask] = get_perturber_state(
                    OriginCodes[origin_in_str],
                    times_masked,
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
    return vectors


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
    vectors = _resolve_origin_translation_vectors(coords, origin)
    return coords.translate(vectors, origin.name)


def apply_time_varying_rotation(
    coords: CartesianCoordinates, frame_out: Literal["ecliptic", "equatorial", "itrf93"]
) -> CartesianCoordinates:
    """
    Apply a time-varying rotation to the Cartesian coordinates. At the moment only
    rotations between itrf93 and ecliptic/equatorial are supported.

    Parameters
    ----------
    coords : CartesianCoordinates
        The coordinates to rotate and translate to the desired frame.
    frame_out : Literal["ecliptic", "equatorial", "itrf93"]
        The desired output frame.

    Returns
    -------
    CartesianCoordinates
        The rotated coordinates.
    """
    from ..utils.spice import setup_SPICE

    setup_SPICE()

    if coords.frame == "ecliptic" and frame_out == "equatorial":
        logger.warning(
            "Rotations between ecliptic and equatorial are not time-varying."
        )
    elif coords.frame == "equatorial" and frame_out == "ecliptic":
        logger.warning(
            "Rotations between equatorial and ecliptic are not time-varying."
        )

    assert len(pc.unique(coords.origin.code)) == 1

    # Transform to geocentric coordinates in the input frame
    if frame_out == "itrf93":
        frame_spice_out = "ITRF93"
    elif frame_out == "ecliptic":
        frame_spice_out = "ECLIPJ2000"
    elif frame_out == "equatorial":
        frame_spice_out = "J2000"
    else:
        raise ValueError("Unsupported frame: {}".format(frame_out))

    frame_in = coords.frame
    if frame_in == "itrf93":
        frame_spice_in = "ITRF93"
    elif frame_in == "ecliptic":
        frame_spice_in = "ECLIPJ2000"
    elif frame_in == "equatorial":
        frame_spice_in = "J2000"
    else:
        raise ValueError("Unsupported frame: {}".format(frame_in))

    from ..constants import KM_P_AU, S_P_DAY
    from ..utils.spice import _query_sxform_itrf93_batch

    # Dedup epochs then query sxforms in one batched call. Fewer unique
    # epochs than rows is the common case (e.g. many objects at a few
    # shared times), so paying O(U) sxform computations over an O(N)
    # per-row apply is the right trade.
    unique_times = coords.time.unique()
    unique_ets = (
        unique_times.et().to_numpy(zero_copy_only=False).astype(np.float64)
    )
    batched_kms = _query_sxform_itrf93_batch(frame_spice_in, frame_spice_out, unique_ets)

    # Unit conversion: sxform is km / km-s, our states are AU / AU-d.
    # Fold the scaling into the matrix table once so the Rust apply
    # kernel stays a pure (matrix @ state, matrix @ cov @ matrix^T).
    rotation_unit_conversion = np.zeros((6, 6))
    rotation_unit_conversion[:3, :3] = np.identity(3) * KM_P_AU
    rotation_unit_conversion[3:, 3:] = np.identity(3) * KM_P_AU / S_P_DAY
    inv_unit_conversion = np.linalg.inv(rotation_unit_conversion)
    matrices_aud = np.einsum(
        "ij,ujk,kl->uil", inv_unit_conversion, batched_kms, rotation_unit_conversion
    )

    # Build per-row time index into unique_times by matching on the
    # (days, nanos) pair packed into a single int64 key.
    def _time_keys(time_like) -> np.ndarray:
        days = time_like.days.to_numpy(zero_copy_only=False).astype(np.int64)
        nanos = time_like.nanos.to_numpy(zero_copy_only=False).astype(np.int64)
        return days * 86_400_000_000_000 + nanos

    keys_all = _time_keys(coords.time)
    keys_u = _time_keys(unique_times)
    order = np.argsort(keys_u)
    positions = np.searchsorted(keys_u[order], keys_all)
    time_index = order[positions]

    # Single Rust crossing does per-row M @ state and M @ Σ @ M^T
    # across all N rows in parallel. Covariance NaN handling matches
    # CartesianCoordinates.rotate exactly (NaN cells pass through).
    states = coords.values
    cov_matrix = coords.covariance.to_matrix()
    cov_flat = cov_matrix.reshape(cov_matrix.shape[0], 36)
    rotated_states, rotated_cov_flat = rotate_cartesian_time_varying_numpy(
        states,
        time_index,
        matrices_aud,
        cov_flat,
    )
    rotated_cov = rotated_cov_flat.reshape(cov_matrix.shape)

    # Preserve the existing near-zero cleanup so downstream covariance
    # consumers see the same small-element behaviour as the legacy path.
    near_zero_mask = np.abs(rotated_cov) < COVARIANCE_ROTATION_TOLERANCE
    if near_zero_mask.any():
        logger.debug(
            f"{int(near_zero_mask.sum())} covariance elements are within "
            f"{COVARIANCE_ROTATION_TOLERANCE:.0e} of zero after rotation, "
            "setting these elements to 0."
        )
        rotated_cov = np.where(near_zero_mask, 0.0, rotated_cov)

    return CartesianCoordinates.from_kwargs(
        x=rotated_states[:, 0],
        y=rotated_states[:, 1],
        z=rotated_states[:, 2],
        vx=rotated_states[:, 3],
        vy=rotated_states[:, 4],
        vz=rotated_states[:, 5],
        time=coords.time,
        covariance=CoordinateCovariances.from_matrix(rotated_cov),
        origin=coords.origin,
        frame=frame_out,
    )


def cartesian_to_frame(
    coords: CartesianCoordinates, frame: Literal["ecliptic", "equatorial", "itrf93"]
) -> "CartesianCoordinates":
    """
    Rotate Cartesian coordinates and their covariances to the given frame.

    Parameters
    ----------
    coords : `~adam_core.coordinates.cartesian.CartesianCoordinates`
        Cartesian coordinates and optionally their covariances.
    frame : {'ecliptic', 'equatorial', 'itrf93'}
        Desired reference frame of the output coordinates.

    Returns
    -------
    CartesianCoordinates : `~adam_core.coordinates.cartesian.CartesianCoordinates`
        Rotated Cartesian coordinates and their covariances.
    """
    if frame == "ecliptic" and coords.frame == "equatorial":
        return coords.rotate(TRANSFORM_EQ2EC, "ecliptic")
    elif frame == "equatorial" and coords.frame == "ecliptic":
        return coords.rotate(TRANSFORM_EC2EQ, "equatorial")
    elif frame == "itrf93" and coords.frame != "itrf93":
        return apply_time_varying_rotation(coords, frame)
    elif frame != "itrf93" and coords.frame == "itrf93":
        return apply_time_varying_rotation(coords, frame)
    elif frame == coords.frame:
        return coords
    else:
        err = "frame should be one of {'ecliptic', 'equatorial', 'itrf93'}"
        raise ValueError(err)


def transform_coordinates(
    coords: types.CoordinateType,
    representation_out: Optional[type[types.CoordinateType]] = None,
    frame_out: Optional[Literal["ecliptic", "equatorial", "itrf93"]] = None,
    origin_out: Optional[OriginCodes] = None,
) -> types.CoordinateType:
    """
    Transform coordinates between frames ('ecliptic', 'equatorial', 'itrf93'), origins,
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
            'spherical', 'keplerian', 'cometary', 'geodetic'.
    TypeError
        If coords is not a `~adam_core.coordinates.Coordinates` object.
        If origin_out is not a `~adam_core.coordinates.OriginCodes` object.
    """
    # Check that coords is a thor.coordinates.Coordinates object
    if not isinstance(coords, CoordinatesClasses):
        raise TypeError("Unsupported coordinate type: {}".format(type(coords)))

    if frame_out not in {None, "equatorial", "ecliptic", "itrf93"}:
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

    rust_coords = _try_transform_coordinates_rust(
        coords, representation_out_, frame_out, origin_out
    )
    if rust_coords is not None:
        return rust_coords

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
