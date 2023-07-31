from typing import Tuple

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

from ..constants import Constants as c
from ..coordinates.transform import _cartesian_to_spherical
from .aberrations import _add_light_time, add_stellar_aberration

MU = c.MU


@jit
def _generate_ephemeris_2body(
    propagated_orbit: np.ndarray,
    observation_time: float,
    observer_coordinates: jnp.ndarray,
    lt_tol: float = 1e-10,
    mu: float = MU,
    max_iter: int = 100,
    tol: float = 1e-15,
) -> Tuple[jnp.ndarray, jnp.float64]:
    """
    Given a propagated orbit, generate its on-sky ephemeris as viewed from the observer.
    This function calculates the light time delay between the propagated orbit and the observer,
    and then propagates the orbit backward by that amount to when the light from object was actually
    emitted towards the observer.

    The motion of the observer in an inertial frame will cause an object
    to appear in a different location than its true geometric location, this is known as
    stellar aberration. Stellar aberration is will also be applied after
    light time correction has been added.

    The velocity of the input orbits are unmodified only the position
    vector is modified with stellar aberration.

    Parameters
    ----------
    propagated_orbit : `~jax.numpy.ndarray` (6)
        Barycentric Cartesian orbit propagated to the given time.
    observation_time : float
        Epoch at which orbit and observer coordinates are defined.
    observer_coordinates : `~jax.numpy.ndarray` (3)
        Barycentric Cartesian observer coordinates.
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
    topocentric_coordinates = topocentric_coordinates.at[0:3].set(
        add_stellar_aberration(
            propagated_orbits_aberrated.reshape(1, -1),
            observer_coordinates.reshape(1, -1),
        )[0]
    )

    # Convert to spherical coordinates
    ephemeris_spherical = _cartesian_to_spherical(topocentric_coordinates)

    return ephemeris_spherical, light_time


# Vectorization Map: _generate_ephemeris_2body
_generate_ephemeris_2body_vmap = jit(
    vmap(
        _generate_ephemeris_2body,
        in_axes=(0, 0, 0, None, None, None, None),
        out_axes=(0, 0),
    )
)
