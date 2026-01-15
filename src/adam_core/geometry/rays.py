"""
Observation ray construction from point source detections.

This module provides functionality to convert point source detections with
RA/Dec coordinates into observation rays in the SSB ecliptic frame, including
observer positions and line-of-sight unit vectors.

ObservationRays are used to find line-of-sight intersections with orbit segments.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pyarrow.compute as pc
import quivr as qv
import ray

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.origin import Origin, OriginCodes
from ..coordinates.spherical import SphericalCoordinates
from ..coordinates.transform import transform_coordinates
from ..observations.detections import PointSourceDetections
from ..observations.exposures import Exposures
from ..observations.source_catalog import SourceCatalog
from ..observers.observers import Observers
from ..orbits.ephemeris import Ephemeris
from ..ray_cluster import initialize_use_ray
from ..utils.iter import _iterate_chunk_indices

__all__ = [
    "ObservationRays",
    "detections_to_rays",
    "ephemeris_to_rays",
    "source_catalog_to_rays",
]

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # Only for type checkers to avoid runtime import cycles
    from ..orbits.ephemeris import Ephemeris


class ObservationRays(qv.Table):
    """
    Observation rays representing line-of-sight vectors from observers to detections.

    Each row represents a ray from an observer position to a detected source,
    with both the observer position and line-of-sight direction in the SSB
    ecliptic frame for consistent geometric queries.
    """

    #: Unique identifier for the detection
    det_id = qv.LargeStringColumn()

    #: Optional originating orbit_id (when rays come from ephemerides)
    orbit_id = qv.LargeStringColumn(nullable=True)

    #: Observer and state in SSB ecliptic frame
    observer = Observers.as_column()

    #: Line-of-sight unit vector x-component in SSB ecliptic frame
    u_x = qv.Float64Column()

    #: Line-of-sight unit vector y-component in SSB ecliptic frame
    u_y = qv.Float64Column()

    #: Line-of-sight unit vector z-component in SSB ecliptic frame
    u_z = qv.Float64Column()


def detections_to_rays(
    detections: PointSourceDetections,
    exposures: Exposures,
) -> ObservationRays:
    """
    Convert point source detections to observation rays in SSB ecliptic frame.

    This function joins detections with exposures to obtain observatory codes,
    computes observer positions at detection times, and converts RA/Dec
    coordinates to line-of-sight unit vectors in the SSB ecliptic frame.

    Parameters
    ----------
    detections : PointSourceDetections
        Point source detections with RA/Dec coordinates
    exposures : Exposures
        Exposure information including observatory codes

    Returns
    -------
    rays : ObservationRays
        Observation rays with observer positions and line-of-sight vectors
        in SSB ecliptic frame
    """
    if len(detections) == 0:
        return ObservationRays.empty()

    # Pre-filter detections to only those with matching exposures (inner-join semantics)
    # This ensures tests expecting zero rays on no match will pass
    det_has_match_mask = pc.is_in(detections.exposure_id, exposures.id)
    detections = detections.apply_mask(det_has_match_mask)
    if len(detections) == 0:
        return ObservationRays.empty()

    # Vectorized linkage: map each detection's exposure_id to exposure row index
    idx_in_exposures = pc.index_in(detections.exposure_id, exposures.id)
    # Filter out any detections with missing exposure match (should be none after mask)
    has_match = pc.invert(pc.is_null(idx_in_exposures))
    detections = detections.apply_mask(has_match)
    idx_in_exposures = pc.fill_null(idx_in_exposures, -1)
    # Gather observatory codes aligned to detections length
    observatory_codes = pc.take(
        exposures.observatory_code, idx_in_exposures
    ).to_pylist()
    detection_times = detections.time

    # Compute observer states in SSB ecliptic frame
    observers = Observers.from_codes(observatory_codes, detection_times)

    # Ensure observer coordinates are in ecliptic frame with SUN origin
    observers = observers.set_column(
        "coordinates",
        transform_coordinates(
            observers.coordinates,
            CartesianCoordinates,
            frame_out="ecliptic",
            origin_out=OriginCodes.SUN,
        ),
    )

    # Convert RA/Dec to line-of-sight unit vectors
    ra_deg = detections.ra.to_numpy()
    dec_deg = detections.dec.to_numpy()

    # Create spherical coordinates on unit sphere (equatorial RA/Dec)
    spherical_coords = SphericalCoordinates.from_kwargs(
        rho=np.ones(len(detections)),
        lon=ra_deg,
        lat=dec_deg,
        vrho=np.zeros(len(detections)),
        vlon=np.zeros(len(detections)),
        vlat=np.zeros(len(detections)),
        time=detection_times,
        origin=Origin.from_kwargs(code=observers.code),
        frame="equatorial",
    )

    # Convert LOS to ecliptic/SUN to match BVH/index conventions
    cartesian_coords = transform_coordinates(
        spherical_coords,
        CartesianCoordinates,
        frame_out="ecliptic",
        origin_out=OriginCodes.SUN,
    )

    # Extract line-of-sight vectors relative to the observer (not SSB origin)
    los_points_ssb = (
        cartesian_coords.r
    )  # SSB coordinates of points at unit distance along LOS
    obs_ssb = np.column_stack(
        [
            observers.coordinates.x.to_numpy(),
            observers.coordinates.y.to_numpy(),
            observers.coordinates.z.to_numpy(),
        ]
    )
    los_vectors = los_points_ssb - obs_ssb
    los_magnitudes = np.linalg.norm(los_vectors, axis=1)
    # Normalize to ensure unit vectors (handle any numerical errors)
    u_vectors = los_vectors / los_magnitudes[:, np.newaxis]

    # Create observation rays table
    rays = ObservationRays.from_kwargs(
        det_id=detections.id.to_pylist(),
        orbit_id=[None] * len(detections),
        observer=observers,
        u_x=u_vectors[:, 0],
        u_y=u_vectors[:, 1],
        u_z=u_vectors[:, 2],
    )

    logger.info(
        f"Created {len(rays)} observation rays from {len(detections)} detections"
    )

    return rays


def ephemeris_to_rays_worker(
    ephemeris: Ephemeris,
    det_id: list[str] | None = None,
) -> ObservationRays:
    ephemeris = qv.defragment(ephemeris)
    times = ephemeris.coordinates.time
    station_codes = ephemeris.coordinates.origin.code
    observers = Observers.from_codes(times=times, codes=station_codes)

    # Ensure observer coordinates are in ecliptic frame with SUN origin
    observers = observers.set_column(
        "coordinates",
        transform_coordinates(
            observers.coordinates,
            CartesianCoordinates,
            frame_out="ecliptic",
            origin_out=OriginCodes.SUN,
        ),
    )

    spherical_coords = ephemeris.coordinates.set_column(
        "rho",
        np.ones(len(ephemeris)),
    )

    cartesian_coords = transform_coordinates(
        spherical_coords,
        CartesianCoordinates,
        frame_out="ecliptic",
        origin_out=OriginCodes.SUN,
    )

    los_points_ssb = cartesian_coords.r
    obs_ssb = np.column_stack(
        [
            observers.coordinates.x.to_numpy(),
            observers.coordinates.y.to_numpy(),
            observers.coordinates.z.to_numpy(),
        ]
    )
    los_vectors = los_points_ssb - obs_ssb
    magnitudes = np.linalg.norm(los_vectors, axis=1)
    u_vectors = los_vectors / magnitudes[:, np.newaxis]

    # Create or reuse detection IDs
    if det_id is None:
        det_id = [f"ephem_{i:06d}" for i in range(len(ephemeris))]

    rays = ObservationRays.from_kwargs(
        det_id=det_id,
        orbit_id=ephemeris.orbit_id,
        observer=observers,
        u_x=u_vectors[:, 0],
        u_y=u_vectors[:, 1],
        u_z=u_vectors[:, 2],
    )
    rays = qv.defragment(rays)
    return rays


@ray.remote
def ephemeris_to_rays_worker_remote(
    ephemeris: Ephemeris,
    det_id: list[str] | None = None,
) -> ObservationRays:
    return ephemeris_to_rays_worker(ephemeris, det_id)


def ephemeris_to_rays(
    ephemeris: Ephemeris,
    det_id: list[str] | None = None,
    max_processes: int | None = 1,
    chunk_size: int = 10_000,
) -> ObservationRays:
    """
    Convert ephemeris (topocentric RA/Dec) to observation rays in SSB ecliptic frame.

    Parameters
    ----------
    ephemeris : Ephemeris
        Ephemeris with spherical coordinates (RA/Dec) from observer perspective.
    det_id : list[str], optional
        Detection IDs to assign to rays; if omitted, synthetic IDs are generated.
    max_processes : int | None
        Number of processes to use. If None or <=1, runs single-threaded.
        If >1, uses Ray to parallelize processing across chunks.
    chunk_size : int
        Number of ephemeris rows to process per chunk when using Ray parallelization.
    Returns
    -------
    ObservationRays
        Rays with observer positions and LOS unit vectors in SSB ecliptic frame.
    """

    if len(ephemeris) == 0:
        return ObservationRays.empty()

    rays = ObservationRays.empty()
    if max_processes is None or max_processes <= 1:
        for start, end in _iterate_chunk_indices(ephemeris, chunk_size=chunk_size):
            ephemeris_chunk = ephemeris[start:end]
            det_id_chunk = det_id[start:end]
            rays = qv.concatenate(
                [rays, ephemeris_to_rays_worker(ephemeris_chunk, det_id_chunk)]
            )
        return rays

    initialize_use_ray(num_cpus=max_processes)
    futures = []
    for start, end in _iterate_chunk_indices(ephemeris, chunk_size=chunk_size):
        if len(futures) >= max_processes * 1.5:
            finished, futures = ray.wait(futures, num_returns=1)
            rays_chunk = ray.get(finished[0])
            rays = qv.concatenate([rays, rays_chunk])
        ephemeris_chunk = ephemeris[start:end]
        det_id_chunk = det_id[start:end]

        futures.append(
            ephemeris_to_rays_worker_remote.remote(ephemeris_chunk, det_id_chunk)
        )

    while futures:
        finished, futures = ray.wait(futures, num_returns=1)
        rays_chunk = ray.get(finished[0])
        rays = qv.concatenate([rays, rays_chunk])

    return rays


def source_catalog_to_rays_worker(
    source_catalog: SourceCatalog,
) -> ObservationRays:
    """
    Worker function to convert a chunk of SourceCatalog to ObservationRays.

    This is the core processing unit used by both local and Ray execution paths.
    If the SourceCatalog has an object_id column, it will be used to set the
    orbit_id on the rays for ground truth tracking.

    Parameters
    ----------
    source_catalog : SourceCatalog
        Chunk of denormalized source catalog.

    Returns
    -------
    ObservationRays
        Rays with observer positions and LOS unit vectors in SSB ecliptic frame.
        If object_id is present in the catalog, orbit_id will be set from it.
    """
    if len(source_catalog) == 0:
        return ObservationRays.empty()

    # Ensure defragmented for zero-copy performance
    source_catalog = qv.defragment(source_catalog)

    # Compute observer states at detection times
    observers = Observers.from_codes(
        source_catalog.observatory_code, source_catalog.time
    )

    # Ensure observer coordinates are in ecliptic frame with SUN origin
    observers = observers.set_column(
        "coordinates",
        transform_coordinates(
            observers.coordinates,
            CartesianCoordinates,
            frame_out="ecliptic",
            origin_out=OriginCodes.SUN,
        ),
    )

    # Convert RA/Dec to spherical coordinates (degrees)
    ra_deg = source_catalog.ra.to_numpy()
    dec_deg = source_catalog.dec.to_numpy()

    # Create spherical coordinates on unit sphere (equatorial RA/Dec)
    spherical_coords = SphericalCoordinates.from_kwargs(
        rho=np.ones(len(source_catalog)),
        lon=ra_deg,
        lat=dec_deg,
        vrho=np.zeros(len(source_catalog)),
        vlon=np.zeros(len(source_catalog)),
        vlat=np.zeros(len(source_catalog)),
        time=source_catalog.time,
        origin=Origin.from_kwargs(code=observers.code),
        frame="equatorial",
    )

    # Convert LOS to ecliptic/SUN to match BVH/index conventions
    cartesian_coords = transform_coordinates(
        spherical_coords,
        CartesianCoordinates,
        frame_out="ecliptic",
        origin_out=OriginCodes.SUN,
    )

    # Extract and normalize line-of-sight unit vectors relative to the observer
    los_points_ssb = cartesian_coords.r
    obs_ssb = np.column_stack(
        [
            observers.coordinates.x.to_numpy(),
            observers.coordinates.y.to_numpy(),
            observers.coordinates.z.to_numpy(),
        ]
    )
    los_vectors = los_points_ssb - obs_ssb
    los_magnitudes = np.linalg.norm(los_vectors, axis=1)
    u_vectors = los_vectors / los_magnitudes[:, np.newaxis]

    rays = ObservationRays.from_kwargs(
        det_id=source_catalog.id,
        orbit_id=source_catalog.object_id,
        observer=observers,
        u_x=u_vectors[:, 0],
        u_y=u_vectors[:, 1],
        u_z=u_vectors[:, 2],
    )
    rays = qv.defragment(rays)
    return rays


@ray.remote
def source_catalog_to_rays_worker_remote(
    source_catalog: SourceCatalog,
) -> ObservationRays:
    """Ray remote wrapper for source_catalog_to_rays_worker."""
    return source_catalog_to_rays_worker(source_catalog)


def source_catalog_to_rays(
    source_catalog: SourceCatalog,
    *,
    max_processes: int | None = 1,
    chunk_size: int = 100_000,
) -> ObservationRays:
    """
    Convert a denormalized SourceCatalog directly to ObservationRays.

    This function avoids intermediate PointSourceDetections/Exposures tables by
    using SourceCatalog's already joined fields (time, ra/dec, observatory_code).

    Parameters
    ----------
    source_catalog : SourceCatalog
        Denormalized source catalog with columns time, ra/dec and observatory_code.
    max_processes : int | None
        Number of processes to use. If None or <=1, runs single-threaded.
        If >1, uses Ray to parallelize processing across chunks.
    chunk_size : int
        Number of sources to process per chunk when using Ray parallelization.

    Returns
    -------
    ObservationRays
        Rays with observer positions and LOS unit vectors in SSB ecliptic frame.
    """
    if len(source_catalog) == 0:
        return ObservationRays.empty()

    # Single-threaded execution
    if max_processes is None or max_processes <= 1:
        return source_catalog_to_rays_worker(source_catalog)

    # Ray parallel execution
    initialize_use_ray(num_cpus=max_processes)

    futures: list[ray.ObjectRef] = []
    ray_chunks: list[ObservationRays] = []

    for start, end in _iterate_chunk_indices(source_catalog, chunk_size):
        print(f"  Queueing chunk {start} to {end}")
        chunk = source_catalog[start:end]
        future = source_catalog_to_rays_worker_remote.remote(chunk)
        futures.append(future)

        # Progressive collection to manage memory
        if len(futures) >= max_processes * 1.5:
            finished, futures = ray.wait(futures, num_returns=1)
            ray_chunks.append(ray.get(finished[0]))

    # Collect remaining futures
    while futures:
        finished, futures = ray.wait(futures, num_returns=1)
        ray_chunks.append(ray.get(finished[0]))

    # Concatenate all ray chunks
    print(f"  Concatenating {len(ray_chunks)} ray chunks")
    return qv.concatenate(ray_chunks, defrag=True)
