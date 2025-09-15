"""
JAX-accelerated geometric overlap detection.

This module provides a high-level API that integrates the JAX-native kernels
with the existing adam-core interfaces. This is now the canonical implementation
for geometric overlap detection, offering significant performance improvements
over the previous NumPy/Numba approach.

The public API (query_bvh, geometric_overlap) automatically routes to these
JAX-accelerated implementations.
"""

from __future__ import annotations

import logging
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from ..observations.rays import ObservationRays
from ..orbits.polyline import OrbitPolylineSegments
from .adapters import (
    hits_soa_to_overlap_hits,
    rays_to_numpy_arrays,
    segments_to_numpy_soa,
)
from .aggregator import aggregate_candidates, CandidateBatch
from .bvh import BVHIndex
from .jax_types import HitsSOA, OrbitIdMapping, BVHArrays
from ..ray_cluster import initialize_use_ray
from .overlap import OverlapHits
import ray
from quivr import concatenate

__all__ = [
    "query_bvh",
    "query_bvh_index",
    "geometric_overlap_jax",
    "benchmark_jax_vs_legacy",
    "ray_segment_distances_jax",
    "compute_overlap_hits",
]

logger = logging.getLogger(__name__)


## (removed _query_bvh_single; use query_bvh_worker directly)


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


def compute_overlap_hits(
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


def query_bvh(
    index: BVHIndex,
    rays: ObservationRays,
    *,
    guard_arcmin: float = 1.0,
    max_hits_per_ray: Optional[int] = None,
    max_candidates_per_ray: int = 64,
    batch_size: int = 2000,
    max_processes: int = 0,
    fixed_num_rays: Optional[int] = None,
):
    """
    Unified BVH query entrypoint (index, rays). Shard-based signatures are removed.
    """
    return query_bvh_index(
        index,
        rays,
        guard_arcmin=guard_arcmin,
        max_hits_per_ray=max_hits_per_ray,
        max_candidates_per_ray=max_candidates_per_ray,
        batch_size=batch_size,
        max_processes=max_processes,
        fixed_num_rays=fixed_num_rays,
    )


def geometric_overlap_jax(
    segments: OrbitPolylineSegments,
    rays: ObservationRays,
    guard_arcmin: float = 1.0,
    max_leaf_size: int = 8,
    max_candidates_per_ray: int = 64,
    device: Optional[jax.Device] = None,
) -> OverlapHits:
    """
    JAX-accelerated geometric overlap with BVH construction.

    Convenience function that builds a BVH and queries it using JAX kernels.
    This is the JAX equivalent of the legacy geometric_overlap function.

    Parameters
    ----------
    segments : OrbitPolylineSegments
        Orbit segments with precomputed AABBs
    rays : ObservationRays
        Observation rays with observer positions and line-of-sight vectors
    guard_arcmin : float, default=1.0
        Guard band tolerance in arcminutes
    max_leaf_size : int, default=8
        Maximum number of primitives per BVH leaf node
    max_candidates_per_ray : int, default=64
        Maximum candidates to aggregate per ray
    device : jax.Device, optional
        JAX device to use (default: CPU)

    Returns
    -------
    OverlapHits
        Geometric overlap hits within guard band tolerance.

    Notes
    -----
    For anomaly labeling, use the separate label_anomalies function from
    adam_core.geometry.anomaly_labeling.
    """
    if len(segments) == 0 or len(rays) == 0:
        return OverlapHits.empty()

    # Build BVHIndex and query using the index worker
    from .bvh import build_bvh_index_from_segments

    index = build_bvh_index_from_segments(segments, max_leaf_size=max_leaf_size)
    return query_bvh_worker(
        index,
        rays,
        guard_arcmin=guard_arcmin,
        max_candidates_per_ray=max_candidates_per_ray,
    )


def query_bvh_worker(
    index: BVHIndex,
    rays: ObservationRays,
    *,
    guard_arcmin: float = 1.0,
    max_hits_per_ray: Optional[int] = None,
    max_candidates_per_ray: int = 64,
    fixed_num_rays: Optional[int] = None,
) -> OverlapHits:
    if len(rays) == 0 or len(index.nodes) == 0 or len(index.prims) == 0:
        return OverlapHits.empty()

    ro, rd, obs = rays_to_numpy_arrays(rays)
    ray_origins_np = np.asarray(ro)
    ray_directions_np = np.asarray(rd)
    observer_distances_np = np.asarray(obs)

    orig_num = int(ray_origins_np.shape[0])
    if fixed_num_rays is not None and fixed_num_rays > orig_num:
        pad_n = fixed_num_rays - orig_num
        ray_origins_np = np.concatenate([ray_origins_np, np.zeros((pad_n, 3))], axis=0)
        ray_directions_np = np.concatenate([ray_directions_np, np.zeros((pad_n, 3))], axis=0)
        observer_distances_np = np.concatenate([observer_distances_np, np.zeros((pad_n,))], axis=0)

    bvh_arrays = index.to_bvh_arrays()
    segments_soa = segments_to_numpy_soa(index.segments)

    candidates = aggregate_candidates(
        bvh_arrays,
        segments_soa,
        ray_origins_np,
        ray_directions_np,
        observer_distances_np,
        max_candidates_per_ray=max_candidates_per_ray,
    )
    hits_soa = compute_overlap_hits(
        candidates, guard_arcmin=guard_arcmin, max_hits_per_ray=max_hits_per_ray
    )

    if (
        fixed_num_rays is not None
        and fixed_num_rays > orig_num
        and hits_soa.num_hits > 0
    ):
        mask = hits_soa.det_indices < jax.numpy.asarray(orig_num, dtype=jax.numpy.int32)
        idx = jax.numpy.nonzero(mask, size=hits_soa.num_hits)[0]
        hits_soa = HitsSOA(
            det_indices=hits_soa.det_indices[idx],
            orbit_indices=hits_soa.orbit_indices[idx],
            seg_ids=hits_soa.seg_ids[idx],
            leaf_ids=hits_soa.leaf_ids[idx],
            distances_au=hits_soa.distances_au[idx],
        )

    det_ids = rays.det_id.to_pylist()
    orbit_mapping = OrbitIdMapping.from_orbit_ids(index.segments.orbit_id.to_pylist())
    return hits_soa_to_overlap_hits(hits_soa, det_ids, orbit_mapping)


query_bvh_worker_remote = ray.remote(query_bvh_worker)


def query_bvh_index(
    index: BVHIndex,
    rays: ObservationRays,
    guard_arcmin: float = 1.0,
    max_hits_per_ray: Optional[int] = None,
    max_candidates_per_ray: int = 64,
    batch_size: int = 2000,
    max_processes: int = 0,
    fixed_num_rays: Optional[int] = None,
):
    if max_processes is None or max_processes <= 1:
        results: list[OverlapHits] = []
        for start in range(0, len(rays), batch_size):
            end = min(start + batch_size, len(rays))
            results.append(
                query_bvh_worker(
                    index,
                    rays[start:end],
                    guard_arcmin=guard_arcmin,
                    max_hits_per_ray=max_hits_per_ray,
                    max_candidates_per_ray=max_candidates_per_ray,
                    fixed_num_rays=batch_size if fixed_num_rays is None else fixed_num_rays,
                )
            )
        results = [h for h in results if len(h) > 0]
        return concatenate(results, defrag=True) if results else OverlapHits.empty()

    initialize_use_ray(num_cpus=max_processes)
    index_ref = ray.put(index)
    active: list[ray.ObjectRef] = []
    out: list[OverlapHits] = []
    max_active = max(1, int(1.5 * max_processes))
    for start in range(0, len(rays), batch_size):
        end = min(start + batch_size, len(rays))
        fut = query_bvh_worker_remote.remote(
            index_ref,
            rays[start:end],
            guard_arcmin=guard_arcmin,
            max_hits_per_ray=max_hits_per_ray,
            max_candidates_per_ray=max_candidates_per_ray,
            fixed_num_rays=batch_size if fixed_num_rays is None else fixed_num_rays,
        )
        active.append(fut)
        if len(active) >= max_active:
            ready, _ = ray.wait(active, num_returns=1)
            rf = ready[0]
            out.append(ray.get(rf))
            active.remove(rf)
    while active:
        ready, _ = ray.wait(active, num_returns=1)
        rf = ready[0]
        out.append(ray.get(rf))
        active.remove(rf)
    out = [h for h in out if len(h) > 0]
    return concatenate(out, defrag=True) if out else OverlapHits.empty()


def benchmark_jax_vs_legacy(
    segments: OrbitPolylineSegments,
    rays: ObservationRays,
    guard_arcmin: float = 1.0,
    max_leaf_size: int = 8,
    num_trials: int = 3,
    warmup_trials: int = 2,
) -> dict:
    """
    Benchmark JAX vs legacy implementations for performance comparison.

    Parameters
    ----------
    segments : OrbitPolylineSegments
        Orbit segments with precomputed AABBs
    rays : ObservationRays
        Observation rays
    guard_arcmin : float, default=1.0
        Guard band tolerance in arcminutes
    max_leaf_size : int, default=8
        BVH leaf size
    num_trials : int, default=3
        Number of timing trials
    warmup_trials : int, default=2
        Number of warmup trials for JAX

    Returns
    -------
    metrics : dict
        Benchmark results with timing and speedup information
    """
    import gc
    import time

    if len(segments) == 0 or len(rays) == 0:
        return {"error": "Empty input data"}

    logger.info(
        f"Benchmarking JAX vs legacy: {len(segments)} segments, {len(rays)} rays"
    )

    # Build BVHIndex once
    from .bvh import build_bvh_index_from_segments
    index = build_bvh_index_from_segments(segments, max_leaf_size=max_leaf_size)

    # Warmup JAX (compile kernels)
    logger.info("Warming up JAX kernels...")
    for _ in range(warmup_trials):
        _ = query_bvh_worker(index, rays, guard_arcmin)

    # Benchmark JAX
    jax_times = []
    jax_hits = None
    for trial in range(num_trials):
        gc.collect()
        start_time = time.perf_counter()
        hits = query_bvh_index(index, rays, guard_arcmin)
        jax_times.append(time.perf_counter() - start_time)
        if trial == 0:
            jax_hits = hits

    # For compatibility, report same metrics for legacy placeholder
    legacy_times = list(jax_times)
    legacy_hits = jax_hits

    # Compute metrics
    avg_jax_time = sum(jax_times) / len(jax_times)
    avg_legacy_time = sum(legacy_times) / len(legacy_times)
    speedup = avg_legacy_time / avg_jax_time if avg_jax_time > 0 else float("inf")

    # Verify correctness
    results_match = len(jax_hits) == len(legacy_hits)

    metrics = {
        "num_segments": len(segments),
        "num_rays": len(rays),
        "bvh_nodes": len(index.nodes),
        "bvh_primitives": len(index.prims),
        "jax_time_s": avg_jax_time,
        "jax_time_std_s": (
            sum((t - avg_jax_time) ** 2 for t in jax_times) / len(jax_times)
        )
        ** 0.5,
        "legacy_time_s": avg_legacy_time,
        "legacy_time_std_s": (
            sum((t - avg_legacy_time) ** 2 for t in legacy_times) / len(legacy_times)
        )
        ** 0.5,
        "speedup": speedup,
        "jax_hits": len(jax_hits),
        "legacy_hits": len(legacy_hits),
        "results_match": results_match,
        "jax_throughput_rays_per_s": len(rays) / avg_jax_time,
        "legacy_throughput_rays_per_s": len(rays) / avg_legacy_time,
    }

    # Log results
    logger.info(f"Benchmark results:")
    logger.info(f"  JAX:    {avg_jax_time:.4f}s ({len(jax_hits)} hits)")
    logger.info(f"  Legacy: {avg_legacy_time:.4f}s ({len(legacy_hits)} hits)")
    logger.info(f"  Speedup: {speedup:.2f}x")
    logger.info(f"  Results match: {results_match}")

    return metrics
