from typing import Tuple

import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.covariances import (
    CoordinateCovariances,
    transform_covariances_jacobian,
)
from ..coordinates.origin import Origin, OriginCodes
from ..coordinates.spherical import SphericalCoordinates
from ..coordinates.transform import _cartesian_to_spherical, transform_coordinates
from ..observers.observers import Observers
from ..orbits.ephemeris import Ephemeris
from ..orbits.orbits import Orbits
from ..utils.chunking import process_in_chunks
from .aberrations import _add_light_time, add_stellar_aberration


@jit
def _generate_ephemeris_2body(
    propagated_orbit: np.ndarray,
    observation_time: float,
    observer_coordinates: jnp.ndarray,
    mu: float,
    lt_tol: float = 1e-10,
    max_iter: int = 100,
    tol: float = 1e-15,
    stellar_aberration: bool = False,
) -> Tuple[jnp.ndarray, jnp.float64]:
    """
    Given a propagated orbit, generate its on-sky ephemeris as viewed from the observer.
    This function calculates the light time delay between the propagated orbit and the observer,
    and then propagates the orbit backward by that amount to when the light from object was actually
    emitted towards the observer ("astrometric coordinates").

    The motion of the observer in an inertial frame will cause an object
    to appear in a different location than its true location, this is known as
    stellar aberration (often referred to in combination with other aberrations as "apparent
    coordinates"). Stellar aberration can optionally be applied after
    light time correction has been added but it should not be necessary when comparing to ephemerides
    of solar system small bodies extracted from astrometric catalogs. The stars to which the
    catalog is calibrated undergo the same aberration as the moving objects as seen from the observer.

    If stellar aberration is applied then the velocity of the input orbits are unmodified, only the position
    vector is modified with stellar aberration.

    For more details on aberrations see:
        https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/abcorr.html
        https://ssd.jpl.nasa.gov/horizons/manual.html#defs

    Parameters
    ----------
    propagated_orbit : `~jax.numpy.ndarray` (6)
        Barycentric Cartesian orbit propagated to the given time.
    observation_time : float
        Epoch at which orbit and observer coordinates are defined.
    observer_coordinates : `~jax.numpy.ndarray` (3)
        Barycentric Cartesian observer coordinates.
    mu : float (1)
        Gravitational parameter (GM) of the attracting body in units of
        AU**3 / d**2.
    lt_tol : float, optional
        Calculate aberration to within this value in time (units of days).
    max_iter : int, optional
        Maximum number of iterations over which to converge for propagation.
    tol : float, optional
        Numerical tolerance to which to compute universal anomaly during propagation using the Newtown-Raphson
        method.
    stellar_aberration : bool, optional
        Apply stellar aberration to the ephemerides.

    Returns
    -------
    ephemeris_spherical : `~jax.numpy.ndarray` (6)
        Topocentric Spherical ephemeris.
    lt : float
        Light time correction (t0 - corrected_t0).
    """
    # Add light time correction
    propagated_orbits_aberrated, light_time = _add_light_time(
        propagated_orbit,
        observation_time,
        observer_coordinates[0:3],
        lt_tol=lt_tol,
        mu=mu,
        max_iter=max_iter,
        tol=tol,
    )

    # Calculate topocentric coordinates
    topocentric_coordinates = propagated_orbits_aberrated - observer_coordinates

    # Apply stellar aberration to topocentric coordinates
    topocentric_coordinates = lax.cond(
        stellar_aberration,
        lambda topocentric_coords: topocentric_coords.at[0:3].set(
            add_stellar_aberration(
                propagated_orbits_aberrated.reshape(1, -1),
                observer_coordinates.reshape(1, -1),
            )[0],
        ),
        lambda topocentric_coords: topocentric_coords,
        topocentric_coordinates,
    )

    # Convert to spherical coordinates
    ephemeris_spherical = _cartesian_to_spherical(topocentric_coordinates)

    return ephemeris_spherical, light_time


# Vectorization Map: _generate_ephemeris_2body
_generate_ephemeris_2body_vmap = jit(
    vmap(
        _generate_ephemeris_2body,
        in_axes=(0, 0, 0, 0, None, None, None, None),
        out_axes=(0, 0),
    )
)


def generate_ephemeris_2body(
    propagated_orbits: Orbits,
    observers: Observers,
    lt_tol: float = 1e-10,
    max_iter: int = 1000,
    tol: float = 1e-15,
    stellar_aberration: bool = False,
) -> Ephemeris:
    """
    Generate on-sky ephemerides for each propagated orbit as viewed by the observers.
    This function calculates the light time delay between the propagated orbit and the observer,
    and then propagates the orbit backward by that amount to when the light from object was actually
    emitted towards the observer ("astrometric coordinates").

    The motion of the observer in an inertial frame will cause an object
    to appear in a different location than its true location, this is known as
    stellar aberration (often referred to in combination with other aberrations as "apparent
    coordinates"). Stellar aberration can optionally be applied after
    light time correction has been added but it should not be necessary when comparing to ephemerides
    of solar system small bodies extracted from astrometric catalogs. The stars to which the
    catalog is calibrated undergo the same aberration as the moving objects as seen from the observer.

    If stellar aberration is applied then the velocity of the input orbits are unmodified, only the position
    vector is modified with stellar aberration.

    For more details on aberrations see:
        https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/abcorr.html
        https://ssd.jpl.nasa.gov/horizons/manual.html#defs


    Parameters
    ----------
    propagated_orbits : `~adam_core.orbits.orbits.Orbits` (N)
        Propagated orbits.
    observers : `~adam_core.observers.observers.Observers` (N)
        Observers for which to generate ephemerides. Orbits should already have been
        propagated to the same times as the observers.
    lt_tol : float, optional
        Calculate aberration to within this value in time (units of days).
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        AU**3 / d**2.
    max_iter : int, optional
        Maximum number of iterations over which to converge for propagation.
    tol : float, optional
        Numerical tolerance to which to compute universal anomaly during propagation using the Newtown-Raphson
        method.
    stellar_aberration : bool, optional
        Apply stellar aberration to the ephemerides.

    Returns
    -------
    ephemeris : `~adam_core.orbits.ephemeris.Ephemeris` (N)
        Topocentric ephemerides for each propagated orbit as observed by the given observers.
    """
    num_entries = len(observers)
    assert (
        len(propagated_orbits) == num_entries
    ), "Orbits and observers must be paired and orbits must be propagated to observer times."

    # Transform both the orbits and observers to the barycenter if they are not already.
    propagated_orbits_barycentric = propagated_orbits.set_column(
        "coordinates",
        transform_coordinates(
            propagated_orbits.coordinates,
            CartesianCoordinates,
            frame_out="ecliptic",
            origin_out=OriginCodes.SOLAR_SYSTEM_BARYCENTER,
        ),
    )
    observers_barycentric = observers.set_column(
        "coordinates",
        transform_coordinates(
            observers.coordinates,
            CartesianCoordinates,
            frame_out="ecliptic",
            origin_out=OriginCodes.SOLAR_SYSTEM_BARYCENTER,
        ),
    )

    observer_coordinates = observers_barycentric.coordinates.values
    observer_codes = observers_barycentric.code.to_numpy(zero_copy_only=False)
    mu = observers_barycentric.coordinates.origin.mu()
    times = propagated_orbits.coordinates.time.mjd().to_numpy(zero_copy_only=False)

    # Define chunk size
    chunk_size = 200

    # Process in chunks
    ephemeris_spherical: np.ndarray = np.empty((0, 6))
    light_time: np.ndarray = np.empty((0,))

    for orbits_chunk, times_chunk, observer_coords_chunk, mu_chunk in zip(
        process_in_chunks(propagated_orbits_barycentric.coordinates.values, chunk_size),
        process_in_chunks(times, chunk_size),
        process_in_chunks(observer_coordinates, chunk_size),
        process_in_chunks(mu, chunk_size),
    ):
        ephemeris_chunk, light_time_chunk = _generate_ephemeris_2body_vmap(
            orbits_chunk,
            times_chunk,
            observer_coords_chunk,
            mu_chunk,
            lt_tol,
            max_iter,
            tol,
            stellar_aberration,
        )
        ephemeris_spherical = np.concatenate(
            (ephemeris_spherical, np.asarray(ephemeris_chunk))
        )
        light_time = np.concatenate((light_time, np.asarray(light_time_chunk)))

    # Concatenate chunks and remove padding
    ephemeris_spherical = np.array(ephemeris_spherical)[:num_entries]
    light_time = np.array(light_time)[:num_entries]

    if not propagated_orbits.coordinates.covariance.is_all_nan():

        cartesian_covariances = propagated_orbits.coordinates.covariance.to_matrix()
        covariances_spherical = transform_covariances_jacobian(
            propagated_orbits.coordinates.values,
            cartesian_covariances,
            _generate_ephemeris_2body,
            in_axes=(0, 0, 0, 0, None, None, None, None),
            out_axes=(0, 0),
            observation_times=times.utc.mjd,
            observer_coordinates=observer_coordinates,
            mu=mu,
            lt_tol=lt_tol,
            max_iter=max_iter,
            tol=tol,
            stellar_aberration=stellar_aberration,
        )
        covariances_spherical = CoordinateCovariances.from_matrix(
            np.array(covariances_spherical)
        )

    else:
        covariances_spherical = None

    spherical_coordinates = SphericalCoordinates.from_kwargs(
        time=propagated_orbits.coordinates.time,
        rho=ephemeris_spherical[:, 0],
        lon=ephemeris_spherical[:, 1],
        lat=ephemeris_spherical[:, 2],
        vrho=ephemeris_spherical[:, 3],
        vlon=ephemeris_spherical[:, 4],
        vlat=ephemeris_spherical[:, 5],
        covariance=covariances_spherical,
        origin=Origin.from_kwargs(code=observer_codes),
        frame="ecliptic",
    )

    # Rotate the spherical coordinates from the ecliptic frame
    # to the equatorial frame
    spherical_coordinates = transform_coordinates(
        spherical_coordinates, SphericalCoordinates, frame_out="equatorial"
    )

    return Ephemeris.from_kwargs(
        orbit_id=propagated_orbits_barycentric.orbit_id,
        object_id=propagated_orbits_barycentric.object_id,
        coordinates=spherical_coordinates,
        light_time=light_time,
    )
