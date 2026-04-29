"""
Implementation of SGD-MOID

References
----------
[1] Hedo, J. M. et al. (2019). Minimum orbital intersection distance: an asymptotic approach.
    Astronomy & Astrophysics, 633, A22. https://doi.org/10.1051/0004-6361/201936502

Algorithm 1:
From an arbitrary point P_0 on the primary ellipse,
project the point onto the plane of the other ellipse.

Then find the minimum distance from the projected point,
using a minimization function.

Algorithm 2:

1. Divide up the primary ellipse into a series of points
2. For each point, run algorithm 1
3. After finding the point with the lowest distance,
search for the overall minimum.
"""

from typing import List, Optional, Union

import numpy as np
import quivr as qv

from ..coordinates import Origin, OriginCodes
from ..orbits import Orbits
from ..time import Timestamp
from ..utils.spice import get_perturber_state


class PerturberMOIDs(qv.Table):
    orbit_id = qv.LargeStringColumn()
    perturber = Origin.as_column()
    moid = qv.Float64Column()
    time = Timestamp.as_column()


def calculate_moid(
    primary_ellipse: Orbits, secondary_ellipse: Orbits
) -> tuple[float, Timestamp]:
    """
    Calculate the Minimum Orbit Intersection Distance (MOID) between two orbits.

    Rust-backed fused kernel: nested Brent's-method bounded minimizations
    (outer over propagation dt, inner over ellipse angle), propagate +
    ellipse-distance all in Rust.
    """
    from .._rust.api import calculate_moid_numpy

    mu = float(primary_ellipse.coordinates.origin.mu()[0])
    primary_state = np.ascontiguousarray(
        primary_ellipse.coordinates.values[0], dtype=np.float64
    )
    secondary_state = np.ascontiguousarray(
        secondary_ellipse.coordinates.values[0], dtype=np.float64
    )

    result = calculate_moid_numpy(primary_state, secondary_state, mu)
    moid, dt_min = result
    moid_time = Timestamp.from_mjd(
        [primary_ellipse.coordinates.time.mjd()[0].as_py() + dt_min],
        scale=primary_ellipse.coordinates.time.scale,
    )
    return moid, moid_time


def calculate_perturber_moids(
    orbits: Orbits,
    perturber: Union[OriginCodes, List[OriginCodes]],
    chunk_size: int = 100,
    max_processes: Optional[int] = 1,
) -> PerturberMOIDs:
    """
    Calculate the minimum orbit intersection distance (MOID) for all orbits with respect
    to the perturbing body or bodies.

    Implementation: a single Rust call per perturber, parallelized internally
    via rayon. No Ray dispatch — `max_processes` and `chunk_size` are accepted
    for API compatibility but the Rust kernel auto-detects available cores.

    Parameters
    ----------
    orbits : Orbits
        The orbits to calculate the MOID for.
    perturber : OriginCodes or List[OriginCodes]
        The perturbing body or bodies to calculate the MOID for.
    chunk_size : int, optional
        Accepted for API compatibility; not used (rayon parallelizes the
        whole batch internally).
    max_processes : int, optional
        Accepted for API compatibility; not used.

    Returns
    -------
    PerturberMOIDs
        A table containing the MOID and time for each orbit with respect to
        the perturbing body or bodies.
    """
    from .._rust.api import calculate_moid_batch_numpy

    del chunk_size  # API compat
    del max_processes  # API compat

    assert len(orbits.orbit_id.unique()) == len(orbits)
    assert len(orbits.coordinates.origin.code.unique()) == 1

    if isinstance(perturber, OriginCodes):
        perturbers = [perturber]
    else:
        perturbers = perturber

    n = len(orbits)
    if n == 0:
        return PerturberMOIDs.empty()

    primary_states = np.ascontiguousarray(orbits.coordinates.values, dtype=np.float64)
    primary_mus = np.ascontiguousarray(
        np.asarray(orbits.coordinates.origin.mu(), dtype=np.float64)
    )
    primary_mjds = orbits.coordinates.time.mjd().to_numpy(zero_copy_only=False)
    time_scale = orbits.coordinates.time.scale
    orbit_ids_np = orbits.orbit_id.to_numpy(zero_copy_only=False)

    moids_list: List[PerturberMOIDs] = []
    for perturber_i in perturbers:
        states = get_perturber_state(
            perturber_i,
            orbits.coordinates.time,
            frame=orbits.coordinates.frame,
            origin=orbits.coordinates.origin[0].as_OriginCodes(),
        )
        secondary_states = np.ascontiguousarray(states.values, dtype=np.float64)
        result = calculate_moid_batch_numpy(
            primary_states, secondary_states, primary_mus
        )
        moids, dt_mins = result
        moid_mjds = primary_mjds + dt_mins
        moids_list.append(
            PerturberMOIDs.from_kwargs(
                orbit_id=orbit_ids_np,
                perturber=Origin.from_kwargs(
                    code=np.full(n, perturber_i.name, dtype=object)
                ),
                moid=moids,
                time=Timestamp.from_mjd(moid_mjds, scale=time_scale),
            )
        )

    return qv.concatenate(moids_list) if len(moids_list) > 1 else moids_list[0]
