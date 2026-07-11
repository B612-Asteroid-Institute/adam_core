from __future__ import annotations

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .._rust import (
    cartesian_to_cometary_numpy,
    cartesian_to_geodetic_numpy,
    cartesian_to_keplerian_numpy,
    cartesian_to_spherical_numpy,
    cometary_to_cartesian_numpy,
    keplerian_to_cartesian_numpy,
    spherical_to_cartesian_numpy,
)
from ..constants import Constants as c
from ..utils.bounded_lru import _bounded_lru_get, _bounded_lru_put
from .._rust.arrow import (
    ensure_spice_backend,
    stamp_adam_core_metadata,
    table_from_record_batch,
)
from . import types
from .cartesian import CartesianCoordinates
from .cometary import CometaryCoordinates
from .geodetics import WGS84, GeodeticCoordinates
from .keplerian import KeplerianCoordinates
from .origin import OriginCodes
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
    return _bounded_lru_get(_TRANSLATION_CACHE, key, maxsize=_TRANSLATION_CACHE_MAXSIZE)


def _translation_cache_put(key: _TranslationCacheKey, vectors: np.ndarray) -> None:
    _bounded_lru_put(
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


def _coordinate_record_batch(
    coords: types.CoordinateType, representation: str
) -> pa.RecordBatch:
    """Expose a coordinate table as one metadata-stamped RecordBatch."""
    table = stamp_adam_core_metadata(
        coords.table.combine_chunks(),
        representation=representation,
        frame=coords.frame,
        scale=coords.time.scale,
        schema_name="CoordinateBatch.cartesian.nested.quivr.v1",
    )
    batches = table.to_batches(max_chunksize=max(len(coords), 1))
    if batches:
        return batches[0]
    return pa.RecordBatch.from_arrays(
        [pa.array([], type=field.type) for field in table.schema],
        schema=table.schema,
    )


def _transform_coordinates_native(
    coords: types.CoordinateType,
    representation_out: type[types.CoordinateType],
    frame_out: Literal["ecliptic", "equatorial", "itrf93"],
    origin_out: OriginCodes | None,
) -> types.CoordinateType | None:
    """Transform one coordinate RecordBatch entirely in Rust.

    Python stamps schema metadata and directly wraps the returned RecordBatch;
    representation/frame/origin composition, SPICE calls, covariance AD, and
    Arrow table assembly remain behind this one crossing.
    """
    from .._rust import transform_coordinates_arrow

    representation_in_name = _RUST_TRANSFORM_REPRESENTATIONS.get(type(coords))
    representation_out_name = _RUST_TRANSFORM_REPRESENTATIONS.get(representation_out)
    if representation_in_name is None or representation_out_name is None:
        return None

    target_origin = origin_out.name if isinstance(origin_out, OriginCodes) else None
    itrf93_frame_change = frame_out != coords.frame and "itrf93" in (
        coords.frame,
        frame_out,
    )
    if target_origin is not None or itrf93_frame_change:
        ensure_spice_backend()

    result = transform_coordinates_arrow(
        _coordinate_record_batch(coords, representation_in_name),
        representation_out_name,
        frame_out,
        target_origin,
        a=float(WGS84.a) if representation_out is GeodeticCoordinates else 0.0,
        f=float(WGS84.f) if representation_out is GeodeticCoordinates else 0.0,
    )
    if result is None:
        return None
    return table_from_record_batch(representation_out, result)


def _try_transform_coordinates_rust(
    coords: types.CoordinateType,
    representation_out: type[types.CoordinateType],
    frame_out: Literal["ecliptic", "equatorial", "itrf93"],
    origin_out: OriginCodes | None,
) -> types.CoordinateType | None:
    """
    Run ``transform_coordinates`` on the fully-Rust single-crossing path.

    Returns the transformed coordinates when the native path covers the case,
    otherwise ``None`` so the caller uses the thin Python fallthrough
    composition (``to_cartesian`` -> ``cartesian_to_origin`` ->
    ``cartesian_to_frame`` -> ``from_cartesian``) for the residual cases the
    native path does not yet cover (non-Cartesian input into an ITRF93 frame
    change, geodetic input). A ``None`` return is an explicit "not covered
    natively" signal, not an error.
    """
    # The entire composition -- origin translation (perturber via spkez,
    # observatory via ground-observer), representation change, constant AND
    # time-varying ITRF93 frame rotation, and covariance forward-mode AD --
    # runs in Rust in a single Python->Rust crossing.
    return _transform_coordinates_native(
        coords, representation_out, frame_out, origin_out
    )


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
    # One Rust crossing owns origin resolution (perturber spkez / ground
    # observer states) and the translation for every backend-supported
    # origin. The legacy Python composition remains only as an explicit
    # fallback for origin codes the Rust backend cannot resolve (space/custom
    # observatories; see personal-cmy.37.2.16 for the dispatch boundary).
    try:
        native = _transform_coordinates_native(
            coords, CartesianCoordinates, coords.frame, origin
        )
    except Exception:
        native = None
    if native is not None:
        return native

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

    # Preserve the legacy frame validation errors before crossing.
    if frame_out not in ("itrf93", "ecliptic", "equatorial"):
        raise ValueError("Unsupported frame: {}".format(frame_out))
    if coords.frame not in ("itrf93", "ecliptic", "equatorial"):
        raise ValueError("Unsupported frame: {}".format(coords.frame))

    # One Rust crossing owns SPICE sxform lookup, epoch deduplication, unit
    # conversion, the per-row rotation, covariance transport with the NaN
    # policy, near-zero cleanup, and output table assembly.
    rotated = _transform_coordinates_native(
        coords, CartesianCoordinates, frame_out, None
    )
    if rotated is None:
        raise ValueError(
            "Unsupported time-varying rotation: " f"{coords.frame} -> {frame_out}"
        )
    return rotated


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
    if frame == coords.frame:
        return coords
    if frame not in ("ecliptic", "equatorial", "itrf93"):
        err = "frame should be one of {'ecliptic', 'equatorial', 'itrf93'}"
        raise ValueError(err)

    # Time-varying rotations preserve the legacy SPICE setup and warning
    # semantics; all rotations execute as one Rust crossing.
    if "itrf93" in (frame, coords.frame):
        return apply_time_varying_rotation(coords, frame)

    rotated = _transform_coordinates_native(coords, CartesianCoordinates, frame, None)
    if rotated is None:
        err = "frame should be one of {'ecliptic', 'equatorial', 'itrf93'}"
        raise ValueError(err)
    return rotated


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
    if frame_out is None:
        frame_out = coord_frame

    # A missing target means preserve the per-row origins already present in
    # the input RecordBatch. Avoid splitting the origin column just to express
    # that no origin translation is requested.
    origin_unchanged = origin_out is None or np.all(coords.origin == origin_out)
    if type(coords) is representation_out_:
        if coord_frame == frame_out and origin_unchanged:
            return coords

    rust_coords = _try_transform_coordinates_rust(
        coords, representation_out_, frame_out, origin_out
    )
    if rust_coords is not None:
        return rust_coords

    # Only the deliberately-uncovered legacy fallthrough needs a Python array
    # of per-row origins. The canonical Arrow-native path above never extracts
    # coordinate columns.
    fallback_origin_out: OriginCodes | np.ndarray
    if origin_out is None:
        fallback_origin_out = coords.origin.code.to_numpy(zero_copy_only=False)
    else:
        fallback_origin_out = origin_out

    if not isinstance(coords, CartesianCoordinates):
        cartesian = coords.to_cartesian()
    else:
        cartesian = coords

    # Translate coordinates to new origin (if any are different from current)
    if np.any(cartesian.origin != fallback_origin_out):
        cartesian = cartesian_to_origin(cartesian, fallback_origin_out)

    # Rotate coordinates to new frame (if different from current)
    if cartesian.frame != frame_out:
        cartesian = cartesian_to_frame(cartesian, frame_out)

    # You might think this should be 'isinstance', but no! We're
    # checking whether the input is a particular class variable, not
    # an instance of a class.
    if representation_out_ is CartesianCoordinates:
        return cartesian
    return representation_out_.from_cartesian(cartesian)
