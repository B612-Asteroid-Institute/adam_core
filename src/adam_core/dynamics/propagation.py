import jax.numpy as jnp
import numpy as np
from jax import config, jit, vmap

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.covariances import (
    CoordinateCovariances,
    transform_covariances_jacobian,
)
from ..coordinates.origin import Origin
from ..orbits.orbits import Orbits
from ..time import Timestamp
from ..utils.chunking import process_in_chunks
from .lagrange import apply_lagrange_coefficients, calc_lagrange_coefficients

config.update("jax_enable_x64", True)


@jit
def _propagate_2body(
    orbit: jnp.ndarray,
    t0: float,
    t1: float,
    mu: float,
    max_iter: int = 1000,
    tol: float = 1e-14,
) -> jnp.ndarray:
    """
    Propagate an orbit from t0 to t1.

    Parameters
    ----------
    orbit : `~jax.numpy.ndarray` (6)
        Cartesian orbit with position in units of au and velocity in units of au per day.
    t0 : float (1)
        Epoch in MJD at which the orbit are defined.
    t1 : float (N)
        Epochs to which to propagate the given orbit.
    mu : float (1)
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.
    max_iter : int, optional
        Maximum number of iterations over which to converge. If number of iterations is
        exceeded, will return the value of the universal anomaly at the last iteration.
    tol : float, optional
        Numerical tolerance to which to compute universal anomaly using the Newtown-Raphson
        method.

    Returns
    -------
    orbits : `~jax.numpy.ndarray` (N, 6)
        Orbit propagated to each MJD with position in units of au and velocity in units
        of au per day.
    """
    r = orbit[0:3]
    v = orbit[3:6]
    dt = t1 - t0

    lagrange_coeffs, stumpff_coeffs, chi = calc_lagrange_coefficients(
        r, v, dt, mu=mu, max_iter=max_iter, tol=tol
    )
    r_new, v_new = apply_lagrange_coefficients(r, v, *lagrange_coeffs)

    return jnp.array([r_new[0], r_new[1], r_new[2], v_new[0], v_new[1], v_new[2]])


# Vectorization Map: _propagate_2body
_propagate_2body_vmap = jit(
    vmap(_propagate_2body, in_axes=(0, 0, 0, 0, None, None), out_axes=(0))
)


def propagate_2body(
    orbits: Orbits,
    times: Timestamp,
    max_iter: int = 1000,
    tol: float = 1e-14,
) -> Orbits:
    """
    Propagate orbits using the 2-body universal anomaly formalism.

    Parameters
    ----------
    orbits : `~adam_core.orbits.orbits.Orbits` (N)
        Cartesian orbits with position in units of au and velocity in units of au per day.
    times : Timestamp (M)
        Epochs to which to propagate each orbit. If a single epoch is given, all orbits are propagated to this
        epoch. If multiple epochs are given, then each orbit to will be propagated to each epoch.
    max_iter : int, optional
        Maximum number of iterations over which to converge. If number of iterations is
        exceeded, will return the value of the universal anomaly at the last iteration.
    tol : float, optional
        Numerical tolerance to which to compute universal anomaly using the Newtown-Raphson
        method.

    Returns
    -------
    orbits : `~adam_core.orbits.orbits.Orbits` (N*M)
        Orbits propagated to each MJD.
    """
    # Extract and prepare data
    cartesian_orbits = orbits.coordinates.values
    t0 = orbits.coordinates.time.rescale("tdb").mjd()
    t1 = times.rescale("tdb").mjd()
    mu = orbits.coordinates.origin.mu()
    orbit_ids = orbits.orbit_id.to_numpy(zero_copy_only=False)
    object_ids = orbits.object_id.to_numpy(zero_copy_only=False)

    # Define chunk size
    chunk_size = 200  # Changed from 1000

    # Prepare arrays for chunk processing
    # This creates a n x m matrix where n is the number of orbits and m is the number of times
    n_orbits = cartesian_orbits.shape[0]
    n_times = len(times)
    orbit_ids_ = np.repeat(orbit_ids, n_times)
    object_ids_ = np.repeat(object_ids, n_times)
    orbits_array_ = np.repeat(cartesian_orbits, n_times, axis=0)
    mu = np.repeat(mu, n_times)
    t0_ = np.repeat(t0, n_times)
    t1_ = np.tile(t1, n_orbits)

    # Process in chunks
    orbits_propagated: np.ndarray = np.empty((0, 6))
    for orbits_chunk, t0_chunk, t1_chunk, mu_chunk in zip(
        process_in_chunks(orbits_array_, chunk_size),
        process_in_chunks(t0_, chunk_size),
        process_in_chunks(t1_, chunk_size),
        process_in_chunks(mu, chunk_size),
    ):
        orbits_propagated_chunk = _propagate_2body_vmap(
            orbits_chunk, t0_chunk, t1_chunk, mu_chunk, max_iter, tol
        )
        orbits_propagated = np.concatenate(
            (orbits_propagated, np.asarray(orbits_propagated_chunk))
        )

    # Remove padding
    orbits_propagated = orbits_propagated[: n_orbits * n_times]

    if not orbits.coordinates.covariance.is_all_nan():
        cartesian_covariances = orbits.coordinates.covariance.to_matrix()
        covariances_array_ = np.repeat(cartesian_covariances, n_times, axis=0)

        cartesian_covariances = transform_covariances_jacobian(
            orbits_array_,
            covariances_array_,
            _propagate_2body,
            in_axes=(0, 0, 0, 0, None, None),
            out_axes=0,
            t0=t0_,
            t1=t1_,
            mu=mu,
            max_iter=max_iter,
            tol=tol,
        )
        cartesian_covariances = CoordinateCovariances.from_matrix(cartesian_covariances)

    else:
        cartesian_covariances = None

    origin_code = np.repeat(
        orbits.coordinates.origin.code.to_numpy(zero_copy_only=False), n_times
    )

    # Convert from the jax array to a numpy array
    orbits_propagated = np.asarray(orbits_propagated)

    orbits_propagated = Orbits.from_kwargs(
        orbit_id=orbit_ids_,
        object_id=object_ids_,
        coordinates=CartesianCoordinates.from_kwargs(
            x=orbits_propagated[:, 0],
            y=orbits_propagated[:, 1],
            z=orbits_propagated[:, 2],
            vx=orbits_propagated[:, 3],
            vy=orbits_propagated[:, 4],
            vz=orbits_propagated[:, 5],
            covariance=cartesian_covariances,
            time=Timestamp.from_mjd(t1_, scale="tdb"),
            origin=Origin.from_kwargs(code=origin_code),
            frame="ecliptic",
        ),
    )

    return orbits_propagated
