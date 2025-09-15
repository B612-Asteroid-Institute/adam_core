"""
JAX-accelerated kernels for geometric overlap detection.

This module provides JIT-compiled kernels for efficient distance computation
and guard band filtering, with support for both CPU and GPU execution.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import jax
import jax.numpy as jnp

from .aggregator import CandidateBatch
from .jax_types import HitsSOA

__all__ = [
    "ray_segment_distances_jax",
    "compute_overlap_hits_jax",
    "OverlapBackend",
]

logger = logging.getLogger(__name__)

OverlapBackend = Literal["jax"]


@jax.jit
def _ray_segment_distance_single(
    ray_origin: jax.Array,
    ray_direction: jax.Array,
    seg_start: jax.Array,
    seg_end: jax.Array,
) -> jax.Array:
    """
    Compute minimum distance between a ray and a line segment (single pair).

    JAX-compatible implementation with stable numerics.

    Parameters
    ----------
    ray_origin : jax.Array (3,)
        Ray origin point
    ray_direction : jax.Array (3,)
        Ray direction vector (should be normalized)
    seg_start : jax.Array (3,)
        Segment start point
    seg_end : jax.Array (3,)
        Segment end point

    Returns
    -------
    distance : jax.Array (scalar)
        Minimum distance between ray and segment
    """
    # Vector from ray origin to segment start
    w0 = ray_origin - seg_start

    # Segment direction vector
    seg_dir = seg_end - seg_start
    seg_length_sq = jnp.dot(seg_dir, seg_dir)

    # Handle degenerate segment (point)
    is_degenerate = seg_length_sq < 1e-15

    # For degenerate segments, compute distance from ray to point
    cross_prod = jnp.cross(ray_direction, w0)
    point_distance = jnp.linalg.norm(cross_prod)

    # For non-degenerate segments, compute closest approach parameters
    a = jnp.dot(ray_direction, ray_direction)  # Should be 1 if normalized
    b = jnp.dot(ray_direction, seg_dir)
    c = seg_length_sq
    d = jnp.dot(ray_direction, w0)
    e = jnp.dot(seg_dir, w0)

    denom = a * c - b * b

    # Handle parallel case
    is_parallel = jnp.abs(denom) < 1e-15
    parallel_distance = jnp.linalg.norm(cross_prod)

    # For non-parallel segments, compute parameters
    t_ray = jnp.where(is_parallel, 0.0, (b * e - c * d) / denom)
    t_seg = jnp.where(is_parallel, 0.0, (a * e - b * d) / denom)

    # Clamp ray parameter to non-negative (ray, not line)
    t_ray = jnp.maximum(0.0, t_ray)

    # Clamp segment parameter to [0, 1]
    t_seg = jnp.clip(t_seg, 0.0, 1.0)

    # For clamped segments, recompute t_ray
    is_clamped = (t_seg == 0.0) | (t_seg == 1.0)
    seg_point = seg_start + t_seg * seg_dir
    ray_to_seg = seg_point - ray_origin
    t_ray_recalc = jnp.maximum(0.0, jnp.dot(ray_to_seg, ray_direction))
    t_ray = jnp.where(is_clamped, t_ray_recalc, t_ray)

    # Compute closest points and distance
    ray_point = ray_origin + t_ray * ray_direction
    seg_point = seg_start + t_seg * seg_dir
    segment_distance = jnp.linalg.norm(ray_point - seg_point)

    # Return appropriate distance based on segment type
    return jnp.where(
        is_degenerate,
        point_distance,
        jnp.where(is_parallel, parallel_distance, segment_distance),
    )


@jax.jit
def ray_segment_distances_jax(
    ray_origins: jax.Array,
    ray_directions: jax.Array,
    seg_starts: jax.Array,
    seg_ends: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    """
    Vectorized ray-segment distance computation using JAX.

    Parameters
    ----------
    ray_origins : jax.Array (B, 3)
        Ray origin points for B rays
    ray_directions : jax.Array (B, 3)
        Ray direction vectors for B rays
    seg_starts : jax.Array (B, K, 3)
        Segment start points (K candidates per ray)
    seg_ends : jax.Array (B, K, 3)
        Segment end points (K candidates per ray)
    mask : jax.Array (B, K)
        Boolean mask indicating valid candidates

    Returns
    -------
    distances : jax.Array (B, K)
        Minimum distances between rays and segments
        Invalid entries (mask=False) have distance=inf
    """
    # Vectorize over both batch and candidate dimensions
    distance_fn = jax.vmap(
        jax.vmap(_ray_segment_distance_single, in_axes=(None, None, 0, 0)),
        in_axes=(0, 0, 0, 0),
    )

    distances = distance_fn(ray_origins, ray_directions, seg_starts, seg_ends)

    # Mask invalid candidates with infinity
    distances = jnp.where(mask, distances, jnp.inf)

    return distances


def compute_overlap_hits_jax(
    batch: CandidateBatch,
    guard_arcmin: float = 1.0,
    max_hits_per_ray: Optional[int] = None,
) -> HitsSOA:
    """
    Compute geometric overlap hits using JAX kernels.

    Note: This function is not JIT-compiled due to dynamic output shapes.
    The inner distance computation is JIT-compiled for performance.

    Parameters
    ----------
    batch : CandidateBatch
        Aggregated candidates from BVH traversal
    guard_arcmin : float, default=1.0
        Guard band tolerance in arcminutes
    max_hits_per_ray : int, optional
        Maximum number of hits to return per ray

    Returns
    -------
    HitsSOA
        Geometric overlap hits within guard band
    """
    # Convert guard band to radians
    theta_guard = guard_arcmin * jnp.pi / (180 * 60)

    # Compute distances for all candidates (JIT-compiled)
    distances = ray_segment_distances_jax(
        batch.ray_origins,
        batch.ray_directions,
        batch.seg_starts,
        batch.seg_ends,
        batch.mask,
    )

    # Compute dynamic guard band: θ * max(r_mid, d_obs)
    # Broadcast observer distances to match candidate shape
    d_obs_expanded = batch.observer_distances[:, None]  # (B, 1)
    max_distances = theta_guard * jnp.maximum(batch.r_mids, d_obs_expanded)

    # Apply guard band filter
    valid_hits = batch.mask & (distances <= max_distances)

    # Sort candidates by distance within each ray and optionally take top-K per ray
    B = batch.batch_size
    K = batch.max_candidates
    distances_valid = jnp.where(valid_hits, distances, jnp.inf)

    take_k = K if max_hits_per_ray is None else int(max_hits_per_ray)
    # Order candidate indices by ascending distance per ray
    order = jnp.argsort(distances_valid, axis=1)
    ordered_cands = order[:, :take_k]
    ordered_dists = jnp.take_along_axis(distances_valid, ordered_cands, axis=1)
    valid_top = jnp.isfinite(ordered_dists)

    # Flatten per-ray selections, keeping only finite distances
    rows = jnp.repeat(jnp.arange(B)[:, None], take_k, axis=1)
    flat_rows = rows.reshape(-1)
    flat_cands = ordered_cands.reshape(-1)
    flat_valid = valid_top.reshape(-1)

    keep_rows = flat_rows[flat_valid]
    keep_cands = flat_cands[flat_valid]

    if keep_rows.size == 0:
        return HitsSOA.empty()

    # Extract hit data in sorted order per ray
    hit_ray_indices = batch.ray_indices[keep_rows]
    hit_orbit_indices = batch.orbit_indices[keep_rows, keep_cands]
    hit_seg_ids = batch.seg_ids[keep_rows, keep_cands]
    hit_leaf_ids = batch.leaf_ids[keep_rows, keep_cands]
    hit_distances = distances[keep_rows, keep_cands]

    return HitsSOA(
        det_indices=hit_ray_indices,
        orbit_indices=hit_orbit_indices,
        seg_ids=hit_seg_ids,
        leaf_ids=hit_leaf_ids,
        distances_au=hit_distances,
    )


## Removed NumPy fallback implementation (compute_overlap_hits_numpy)
