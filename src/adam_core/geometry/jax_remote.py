"""
JAX-based Ray remote functions for parallel geometric overlap detection.

This module provides Ray remote functions that use JAX kernels for efficient
parallel processing of geometric overlap queries. Uses Ray's object store
for shared BVH and segment data, following the pattern of @ray.remote functions
rather than actors.

This replaces the legacy actor-based parallel implementation with a more
efficient approach using pure functions and explicit object store management.
"""

import logging
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import ray
from quivr import concatenate

from ..observations.rays import ObservationRays
from ..orbits.polyline import OrbitPolylineSegments
from .adapters import (
    bvh_shard_to_arrays,
    hits_soa_to_overlap_hits,
    rays_to_arrays,
    segments_to_soa,
)
from .aggregator import aggregate_candidates
from .bvh import BVHShard
from .jax_kernels import compute_overlap_hits_jax
from .jax_types import BVHArrays, OrbitIdMapping, SegmentsSOA
from .overlap import OverlapHits

logger = logging.getLogger(__name__)

__all__ = [
    "process_ray_batch_remote",
    "query_bvh_parallel_jax",
]


@ray.remote
def process_ray_batch_remote(
    bvh_arrays_ref: ray.ObjectRef,
    segments_soa_ref: ray.ObjectRef,
    orbit_mapping_ref: ray.ObjectRef,
    ray_origins: np.ndarray,
    ray_directions: np.ndarray,
    observer_distances: np.ndarray,
    det_ids: List[str],
    guard_arcmin: float = 1.0,
    max_hits_per_ray: Optional[int] = None,
    max_candidates_per_ray: int = 64,
) -> Dict[str, np.ndarray]:
    """
    Ray remote function to process a batch of rays using JAX kernels.

    Parameters
    ----------
    bvh_arrays_ref : ray.ObjectRef
        Reference to BVHArrays in Ray's object store
    segments_soa_ref : ray.ObjectRef
        Reference to SegmentsSOA in Ray's object store
    orbit_mapping_ref : ray.ObjectRef
        Reference to OrbitIdMapping in Ray's object store
    ray_origins : np.ndarray
        Ray origin positions (N, 3)
    ray_directions : np.ndarray
        Ray direction vectors (N, 3)
    observer_distances : np.ndarray
        Observer distances (N,)
    det_ids : List[str]
        Detection IDs for this batch
    guard_arcmin : float, default=1.0
        Guard band tolerance in arcminutes
    max_hits_per_ray : int, optional
        Maximum number of hits to return per ray
    max_candidates_per_ray : int, default=64
        Maximum candidates per ray for aggregation

    Returns
    -------
    hits_dict : Dict[str, np.ndarray]
        Dictionary with hit arrays for reconstruction into OverlapHits
    """
    # Get shared data from Ray object store (Ray resolves refs when passed as args).
    # Be tolerant to receiving either resolved objects or ObjectRefs.
    bvh_arrays = (
        ray.get(bvh_arrays_ref)
        if isinstance(bvh_arrays_ref, ray.ObjectRef)
        else bvh_arrays_ref
    )
    segments_soa = (
        ray.get(segments_soa_ref)
        if isinstance(segments_soa_ref, ray.ObjectRef)
        else segments_soa_ref
    )
    orbit_mapping = (
        ray.get(orbit_mapping_ref)
        if isinstance(orbit_mapping_ref, ray.ObjectRef)
        else orbit_mapping_ref
    )

    if len(ray_origins) == 0 or bvh_arrays.num_primitives == 0:
        return {
            "det_indices": np.array([], dtype=np.int32),
            "orbit_indices": np.array([], dtype=np.int32),
            "seg_ids": np.array([], dtype=np.int32),
            "leaf_ids": np.array([], dtype=np.int32),
            "distances_au": np.array([], dtype=np.float64),
        }

    # Convert to JAX arrays
    ray_origins_jax = jnp.asarray(ray_origins)
    ray_directions_jax = jnp.asarray(ray_directions)
    observer_distances_jax = jnp.asarray(observer_distances)

    # Aggregate candidates using CPU-side BVH traversal
    candidates = aggregate_candidates(
        bvh_arrays,
        segments_soa,
        ray_origins_jax,
        ray_directions_jax,
        observer_distances_jax,
        max_candidates_per_ray=max_candidates_per_ray,
    )

    # Compute overlap hits using JAX kernel
    hits_soa = compute_overlap_hits_jax(
        candidates,
        guard_arcmin=guard_arcmin,
        max_hits_per_ray=max_hits_per_ray,
    )

    # Convert to numpy for serialization
    return {
        "det_indices": np.asarray(hits_soa.det_indices),
        "orbit_indices": np.asarray(hits_soa.orbit_indices),
        "seg_ids": np.asarray(hits_soa.seg_ids),
        "leaf_ids": np.asarray(hits_soa.leaf_ids),
        "distances_au": np.asarray(hits_soa.distances_au),
    }


def query_bvh_parallel_jax(
    bvh: BVHShard,
    segments: "OrbitPolylineSegments",
    rays: ObservationRays,
    guard_arcmin: float = 1.0,
    batch_size: int = 2000,
    max_candidates_per_ray: int = 64,
    max_hits_per_ray: Optional[int] = None,
) -> OverlapHits:
    """
    Query BVH in parallel using JAX kernels and Ray remote functions.

    Parameters
    ----------
    bvh : BVHShard
        Built BVH for the segments
    segments : OrbitPolylineSegments
        Orbit polyline segments with AABBs
    rays : ObservationRays
        Observation rays to query
    guard_arcmin : float, default=1.0
        Guard band tolerance in arcminutes
    batch_size : int, default=2000
        Number of rays per batch
    max_candidates_per_ray : int, default=64
        Maximum candidates per ray for aggregation
    max_hits_per_ray : int, optional
        Maximum number of hits to return per ray

    Returns
    -------
    hits : OverlapHits
        All geometric overlap hits
    """
    if len(rays) == 0 or bvh.num_primitives == 0:
        return OverlapHits.empty()

    logger.info(f"Processing {len(rays)} rays in batches of {batch_size}")

    # Convert to JAX-compatible types and put in object store
    orbit_mapping = OrbitIdMapping.from_orbit_ids(segments.orbit_id.to_pylist())
    bvh_arrays = bvh_shard_to_arrays(bvh, orbit_mapping)
    segments_soa = segments_to_soa(segments)

    bvh_arrays_ref = ray.put(bvh_arrays)
    segments_soa_ref = ray.put(segments_soa)
    orbit_mapping_ref = ray.put(orbit_mapping)

    # Convert rays to arrays
    ray_origins, ray_directions, observer_distances = rays_to_arrays(rays)

    # Precompute det_id list from rays
    det_ids = rays.det_id.to_pylist()

    # Launch batches
    futures = []
    batch_bounds = []
    for start_idx in range(0, len(rays), batch_size):
        end_idx = min(start_idx + batch_size, len(rays))

        batch_origins = ray_origins[start_idx:end_idx]
        batch_directions = ray_directions[start_idx:end_idx]
        batch_distances = observer_distances[start_idx:end_idx]
        batch_det_ids = det_ids[start_idx:end_idx]

        future = process_ray_batch_remote.remote(
            bvh_arrays_ref,
            segments_soa_ref,
            orbit_mapping_ref,
            batch_origins,
            batch_directions,
            batch_distances,
            batch_det_ids,
            guard_arcmin=guard_arcmin,
            max_hits_per_ray=max_hits_per_ray,
            max_candidates_per_ray=max_candidates_per_ray,
        )
        futures.append(future)
        batch_bounds.append((start_idx, end_idx))

    # Collect results
    batch_results = ray.get(futures)

    if not batch_results:
        return OverlapHits.empty()

    # Offset local det_indices to global indices
    for i, res in enumerate(batch_results):
        start_idx, _ = batch_bounds[i]
        if len(res["det_indices"]) > 0:
            res["det_indices"] = res["det_indices"] + start_idx

    # Combine results
    all_det_indices = []
    all_orbit_indices = []
    all_seg_ids = []
    all_leaf_ids = []
    all_distances = []

    # det_id index mapping is not required; we map indices back directly

    for batch_result in batch_results:
        if len(batch_result["det_indices"]) > 0:
            all_det_indices.append(batch_result["det_indices"])
            all_orbit_indices.append(batch_result["orbit_indices"])
            all_seg_ids.append(batch_result["seg_ids"])
            all_leaf_ids.append(batch_result["leaf_ids"])
            all_distances.append(batch_result["distances_au"])

    if not all_det_indices:
        return OverlapHits.empty()

    # Concatenate arrays
    combined_det_indices = np.concatenate(all_det_indices)
    combined_orbit_indices = np.concatenate(all_orbit_indices)
    combined_seg_ids = np.concatenate(all_seg_ids)
    combined_leaf_ids = np.concatenate(all_leaf_ids)
    combined_distances = np.concatenate(all_distances)

    # Convert back to OverlapHits
    det_ids = [rays.det_id.to_pylist()[i] for i in combined_det_indices]
    orbit_ids = [orbit_mapping.index_to_id[i] for i in combined_orbit_indices]

    hits = OverlapHits.from_kwargs(
        det_id=det_ids,
        orbit_id=orbit_ids,
        seg_id=combined_seg_ids.tolist(),
        leaf_id=combined_leaf_ids.tolist(),
        distance_au=combined_distances.tolist(),
    )

    logger.info(f"Found {len(hits)} overlap hits from {len(rays)} rays")

    return hits
