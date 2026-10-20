"""
Deprecated legacy differential-correction entry points.

`od` and `differential_correction` keep their historical signatures but no
longer run their own finite-difference loop: each orbit is refined by
`refine` (``rejection="worst_residual"``), the native stage contract, on an
instance of the given propagator. The legacy loop's numerical knobs
(``delta``, ``max_iter``, ``method``) have no counterpart there and are
accepted and ignored. Both will be removed in a later release.
"""

import logging
import time
import warnings
from typing import Literal, Optional, Tuple, Type

import numpy as np
import numpy.typing as npt
import pyarrow.compute as pc
import quivr as qv

from ..orbit_determination import OrbitDeterminationObservations
from ..propagator import Propagator
from ..utils.iter import _iterate_chunks
from .fitted_orbits import FittedOrbitMembers, FittedOrbits
from .observation_uncertainty import ObservationUncertaintyModel

logger = logging.getLogger(__name__)

__all__ = ["differential_correction"]

_DEPRECATION_MESSAGE = (
    "adam_core.orbit_determination.od.{name} is deprecated and will be removed. "
    "It now delegates to refine(orbit, observations, propagator, "
    "rejection='worst_residual'); call refine (or run_od for the whole step) "
    "directly."
)


def od_worker(
    orbit_ids: npt.NDArray[np.str_],
    orbits: FittedOrbits,
    orbit_members: FittedOrbitMembers,
    observations: OrbitDeterminationObservations,
    propagator: Type[Propagator],
    rchi2_threshold: float = 100,
    min_obs: int = 5,
    min_arc_length: float = 1.0,
    contamination_percentage: float = 0.0,
    delta: float = 1e-6,
    max_iter: int = 20,
    method: Literal["central", "finite"] = "central",
    propagator_kwargs: dict = {},
) -> Tuple[FittedOrbits, FittedOrbitMembers]:

    # Pre-compute orbit_id -> row indices so each per-orbit lookup is O(1)
    # rather than O(N_orbits) / O(N_members) / O(N_observations) scans.
    orbits_id_arr = orbits.orbit_id.to_numpy(zero_copy_only=False)
    orbits_idx_by_id = {oid: i for i, oid in enumerate(orbits_id_arr)}

    om_orbit_ids = orbit_members.orbit_id.to_numpy(zero_copy_only=False)
    om_obs_ids = orbit_members.obs_id.to_numpy(zero_copy_only=False)
    om_rows_by_orbit: dict = {}
    for i, oid in enumerate(om_orbit_ids):
        om_rows_by_orbit.setdefault(oid, []).append(i)

    observations_id_arr = observations.id.to_numpy(zero_copy_only=False)
    obs_idx_by_id = {oid: i for i, oid in enumerate(observations_id_arr)}

    od_orbits_list: list[FittedOrbits] = []
    od_orbit_members_list: list[FittedOrbitMembers] = []
    for orbit_id in orbit_ids:
        time_start = time.time()
        logger.debug(f"Differentially correcting orbit {orbit_id}...")

        orbit_row = orbits_idx_by_id.get(orbit_id)
        if orbit_row is None:
            continue
        orbit = orbits.take([orbit_row])

        om_rows = om_rows_by_orbit.get(orbit_id, [])
        obs_row_idx = np.fromiter(
            (obs_idx_by_id[o] for o in om_obs_ids[om_rows] if o in obs_idx_by_id),
            dtype=np.int64,
        )
        orbit_observations = observations.take(obs_row_idx)

        # Sort observations by time
        orbit_observations = orbit_observations.sort_by(
            [
                "coordinates.time.days",
                "coordinates.time.nanos",
                "coordinates.origin.code",
            ]
        )

        od_orbit, od_orbit_orbit_members = _od(
            orbit,
            orbit_observations,
            rchi2_threshold=rchi2_threshold,
            min_obs=min_obs,
            min_arc_length=min_arc_length,
            contamination_percentage=contamination_percentage,
            delta=delta,
            max_iter=max_iter,
            method=method,
            propagator=propagator,
            propagator_kwargs=propagator_kwargs,
        )
        time_end = time.time()
        duration = time_end - time_start
        logger.debug(f"OD for orbit {orbit_id} completed in {duration:.3f}s.")
        od_orbits_list.append(od_orbit)
        od_orbit_members_list.append(od_orbit_orbit_members)

    od_orbits = (
        qv.concatenate(od_orbits_list) if od_orbits_list else FittedOrbits.empty()
    )
    od_orbit_members = (
        qv.concatenate(od_orbit_members_list)
        if od_orbit_members_list
        else FittedOrbitMembers.empty()
    )
    return od_orbits, od_orbit_members


def _od(
    orbit: FittedOrbits,
    observations: OrbitDeterminationObservations,
    propagator: Type[Propagator],
    rchi2_threshold: float = 100,
    min_obs: int = 5,
    min_arc_length: float = 1.0,
    contamination_percentage: float = 0.0,
    delta: float = 1e-6,
    max_iter: int = 20,
    method: Literal["central", "finite"] = "central",
    propagator_kwargs: dict = {},
    observatory_bias_model: Optional[ObservationUncertaintyModel] = None,
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    from .od_orchestration import refine

    if method not in ["central", "finite"]:
        err = "method should be one of 'central' or 'finite'."
        raise ValueError(err)
    if observatory_bias_model is not None:
        observations = observatory_bias_model.apply(observations)
    # As the legacy loop: an orbit with fewer than min_obs observations is
    # not fitted.
    if len(observations) < min_obs:
        logger.debug("This orbit has fewer than {} observations.".format(min_obs))
        return FittedOrbits.empty(), FittedOrbitMembers.empty()
    return refine(
        orbit,
        observations,
        propagator(**propagator_kwargs),
        rejection="worst_residual",
        rejection_settings={
            "rchi2_threshold": rchi2_threshold,
            "min_obs": min_obs,
            "min_arc_length": min_arc_length,
            "contamination_percentage": contamination_percentage,
        },
    )


def od(
    orbit: FittedOrbits,
    observations: OrbitDeterminationObservations,
    propagator: Type[Propagator],
    rchi2_threshold: float = 100,
    min_obs: int = 5,
    min_arc_length: float = 1.0,
    contamination_percentage: float = 0.0,
    delta: float = 1e-6,
    max_iter: int = 20,
    method: Literal["central", "finite"] = "central",
    propagator_kwargs: dict = {},
    observatory_bias_model: Optional[ObservationUncertaintyModel] = None,
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    """
    Deprecated: refine one orbit with `refine` (``rejection="worst_residual"``).

    ``propagator`` is instantiated with ``propagator_kwargs`` and must have a
    native backend (`refine` raises ``TypeError`` otherwise). ``delta``,
    ``max_iter`` and ``method`` belonged to the retired finite-difference
    loop and are ignored (``method`` is still validated).
    """
    warnings.warn(
        _DEPRECATION_MESSAGE.format(name="od"), DeprecationWarning, stacklevel=2
    )
    return _od(
        orbit,
        observations,
        propagator,
        rchi2_threshold=rchi2_threshold,
        min_obs=min_obs,
        min_arc_length=min_arc_length,
        contamination_percentage=contamination_percentage,
        delta=delta,
        max_iter=max_iter,
        method=method,
        propagator_kwargs=propagator_kwargs,
        observatory_bias_model=observatory_bias_model,
    )


def differential_correction(
    orbits: FittedOrbits,
    orbit_members: FittedOrbitMembers,
    observations: OrbitDeterminationObservations,
    propagator: Type[Propagator],
    min_obs: int = 5,
    min_arc_length: float = 1.0,
    contamination_percentage: float = 20,
    rchi2_threshold: float = 100,
    delta: float = 1e-8,
    max_iter: int = 20,
    method: Literal["central", "finite"] = "central",
    propagator_kwargs: dict = {},
    chunk_size: int = 10,
    max_processes: Optional[int] = 1,
    orbit_ids: Optional[npt.NDArray[np.str_]] = None,
    obs_ids: Optional[npt.NDArray[np.str_]] = None,
    observatory_bias_model: Optional[ObservationUncertaintyModel] = None,
) -> Tuple[FittedOrbits, FittedOrbitMembers]:
    """
    Deprecated: refine each orbit with `refine` (``rejection="worst_residual"``)
    on its member observations, serially. ``delta``, ``max_iter`` and
    ``method`` are ignored (see `od`).

    Parameters
    ----------
    chunk_size : int, optional
        Number of orbits to send to each job.
    num_jobs : int, optional
        Number of jobs to launch.
    parallel_backend : str, optional
        Which parallelization backend to use {'ray', 'mp', 'cf'}. Defaults to using Python's concurrent.futures
        module ('cf').
    observatory_bias_model : `~adam_core.orbit_determination.ObservationUncertaintyModel`, optional
        Observation uncertainty model applied to the observations once at this
        entry point before fitting. Default None leaves the observations unchanged.
    """
    warnings.warn(
        _DEPRECATION_MESSAGE.format(name="differential_correction"),
        DeprecationWarning,
        stacklevel=2,
    )
    time_start = time.perf_counter()
    logger.info("Running differential correction...")
    if observatory_bias_model is not None:
        # Applied once here, before any fitting, so every orbit sees the
        # transformed uncertainties.
        observations = observatory_bias_model.apply(observations)
    # Object-store inputs were part of the retired Ray execution contract.
    # Plain-table filtering remains available through the compatibility
    # selectors.
    if orbit_ids is not None:
        orbits = orbits.apply_mask(pc.is_in(orbits.orbit_id, orbit_ids))
        orbit_members = orbit_members.apply_mask(
            pc.is_in(orbit_members.orbit_id, orbit_ids)
        )
    if obs_ids is not None:
        orbit_members = orbit_members.apply_mask(
            pc.is_in(orbit_members.obs_id, obs_ids)
        )
        observations = observations.apply_mask(pc.is_in(observations.id, obs_ids))

    if len(orbits) == 0 or len(orbit_members) == 0:
        logger.info("Received no orbits or orbit members.")
        od_orbits = FittedOrbits.empty()
        od_orbit_members = FittedOrbitMembers.empty()
        time_end = time.perf_counter()
        logger.info(f"Differentially corrected {len(od_orbits)} orbits.")
        logger.info(
            f"Differential correction completed in {time_end - time_start:.3f} seconds."
        )
        return od_orbits, od_orbit_members

    orbit_ids = orbits.orbit_id.to_numpy(zero_copy_only=False)

    od_orbits_chunks: list[FittedOrbits] = []
    od_orbit_members_chunks: list[FittedOrbitMembers] = []

    # `max_processes` remains accepted for compatibility; each orbit is
    # refined natively, one after another (no Python-side parallelism).
    del max_processes
    for orbit_ids_chunk in _iterate_chunks(orbit_ids, chunk_size):
        od_orbits_chunk, od_orbit_members_chunk = od_worker(
            orbit_ids_chunk,
            orbits,
            orbit_members,
            observations,
            rchi2_threshold=rchi2_threshold,
            min_obs=min_obs,
            min_arc_length=min_arc_length,
            contamination_percentage=contamination_percentage,
            delta=delta,
            max_iter=max_iter,
            method=method,
            propagator=propagator,
            propagator_kwargs=propagator_kwargs,
        )
        od_orbits_chunks.append(od_orbits_chunk)
        od_orbit_members_chunks.append(od_orbit_members_chunk)

    od_orbits = (
        qv.concatenate(od_orbits_chunks) if od_orbits_chunks else FittedOrbits.empty()
    )
    od_orbit_members = (
        qv.concatenate(od_orbit_members_chunks)
        if od_orbit_members_chunks
        else FittedOrbitMembers.empty()
    )

    time_end = time.perf_counter()
    logger.info(f"Differentially corrected {len(od_orbits)} orbits.")
    logger.info(
        f"Differential correction completed in {time_end - time_start:.3f} seconds."
    )

    return od_orbits, od_orbit_members
