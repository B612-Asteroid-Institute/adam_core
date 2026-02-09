import multiprocessing as mp
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import quivr as qv
import ray
from jax import config, jit, vmap
from ray import ObjectRef

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.covariances import (
    CoordinateCovariances,
    transform_covariances_jacobian,
)
from ..coordinates.origin import Origin
from ..orbits.orbits import Orbits
from ..ray_cluster import initialize_use_ray
from ..time import Timestamp
from ..utils.chunking import process_in_chunks
from ..utils.iter import _iterate_chunks
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


def _propagate_2body_serial(
    orbits: Orbits,
    times: Timestamp,
    *,
    max_iter: int,
    tol: float,
) -> Orbits:
    """
    Serial (single-process) implementation of 2-body propagation.

    The Ray backend uses this function inside each worker.
    """
    # Extract and prepare data
    cartesian_orbits = orbits.coordinates.values
    t0 = orbits.coordinates.time.rescale("tdb").mjd()
    t1 = times.rescale("tdb").mjd()
    mu = orbits.coordinates.origin.mu()
    orbit_ids = orbits.orbit_id.to_numpy(zero_copy_only=False)
    object_ids = orbits.object_id.to_numpy(zero_copy_only=False)

    # Fixed chunk size to keep JAX shapes stable.
    chunk_size = 200

    n_orbits = cartesian_orbits.shape[0]
    n_times = len(times)
    orbit_ids_ = np.repeat(orbit_ids, n_times)
    object_ids_ = np.repeat(object_ids, n_times)
    orbits_array_ = np.repeat(cartesian_orbits, n_times, axis=0)
    mu_ = np.repeat(mu, n_times)
    t0_ = np.repeat(t0, n_times)
    t1_ = np.tile(t1, n_orbits)

    # Preserve physical parameters by repeating per-orbit rows across times.
    pp_idx = np.repeat(np.arange(n_orbits), n_times).tolist()
    physical_parameters_ = orbits.physical_parameters.take(pp_idx)

    num_entries = n_orbits * n_times
    orbits_propagated = np.empty((num_entries, 6), dtype=np.float64)
    start = 0
    for orbits_chunk, t0_chunk, t1_chunk, mu_chunk in zip(
        process_in_chunks(orbits_array_, chunk_size),
        process_in_chunks(t0_, chunk_size),
        process_in_chunks(t1_, chunk_size),
        process_in_chunks(mu_, chunk_size),
    ):
        valid = min(chunk_size, num_entries - start)
        orbits_propagated_chunk = _propagate_2body_vmap(
            orbits_chunk, t0_chunk, t1_chunk, mu_chunk, max_iter, tol
        )
        orbits_propagated[start : start + valid] = np.asarray(orbits_propagated_chunk)[
            :valid
        ]
        start += valid

    if start != num_entries:
        raise RuntimeError(
            f"Internal error: expected {num_entries} propagated rows, got {start}"
        )

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
            mu=mu_,
            max_iter=max_iter,
            tol=tol,
        )
        cartesian_covariances = CoordinateCovariances.from_matrix(cartesian_covariances)
    else:
        cartesian_covariances = None

    origin_code = np.repeat(
        orbits.coordinates.origin.code.to_numpy(zero_copy_only=False), n_times
    )

    return Orbits.from_kwargs(
        orbit_id=orbit_ids_,
        object_id=object_ids_,
        physical_parameters=physical_parameters_,
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


@ray.remote
def propagate_2body_worker_ray(
    start: int,
    idx_chunk: np.ndarray,
    orbits: Orbits,
    times: Timestamp,
    max_iter: int,
    tol: float,
) -> Tuple[int, Orbits]:
    orbits_chunk = orbits.take(idx_chunk)
    propagated = _propagate_2body_serial(
        orbits_chunk, times, max_iter=max_iter, tol=tol
    )
    return start, propagated


def propagate_2body(
    orbits: Orbits,
    times: Timestamp,
    max_iter: int = 1000,
    tol: float = 1e-14,
    *,
    max_processes: Optional[int] = 1,
    chunk_size: int = 100,
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
    if max_processes is None:
        max_processes = mp.cpu_count()

    if max_processes <= 1:
        return _propagate_2body_serial(orbits, times, max_iter=max_iter, tol=tol)

    initialize_use_ray(num_cpus=max_processes)

    # Put large inputs in object store once.
    orbits_ref = ray.put(orbits)  # type: ignore[name-defined]
    times_ref = ray.put(times)  # type: ignore[name-defined]

    idx = np.arange(0, len(orbits), dtype=np.int64)
    pending: List["ObjectRef"] = []  # type: ignore[name-defined]
    results: Dict[int, Orbits] = {}

    for idx_chunk in _iterate_chunks(idx, chunk_size):
        start = int(idx_chunk[0]) if len(idx_chunk) else 0
        pending.append(
            propagate_2body_worker_ray.remote(  # type: ignore[name-defined]
                start, idx_chunk, orbits_ref, times_ref, max_iter, tol
            )
        )

        if len(pending) >= max_processes * 1.5:
            finished, pending = ray.wait(pending, num_returns=1)  # type: ignore[name-defined]
            start_i, propagated_i = ray.get(finished[0])  # type: ignore[name-defined]
            results[int(start_i)] = propagated_i

    while pending:
        finished, pending = ray.wait(pending, num_returns=1)  # type: ignore[name-defined]
        start_i, propagated_i = ray.get(finished[0])  # type: ignore[name-defined]
        results[int(start_i)] = propagated_i

    chunks = [results[k] for k in sorted(results.keys())]
    return qv.concatenate(chunks) if chunks else Orbits.empty()
