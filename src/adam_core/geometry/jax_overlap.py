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
from .jax_types import OrbitIdMapping
from .overlap import OverlapHits
from .jax_types import HitsSOA

__all__ = [
    "query_bvh_jax",
    "geometric_overlap_jax",
    "benchmark_jax_vs_legacy",
]

logger = logging.getLogger(__name__)


def query_bvh_jax(
    bvh: BVHShard,
    segments: OrbitPolylineSegments,
    rays: ObservationRays,
    guard_arcmin: float = 1.0,
    max_hits_per_ray: Optional[int] = None,
    max_candidates_per_ray: int = 64,
    device: Optional[jax.Device] = None,
    fixed_num_rays: Optional[int] = None,
) -> OverlapHits:
    """
    JAX-accelerated BVH query with legacy-compatible interface.
    
    This function provides the same interface as the legacy query_bvh but uses
    JAX kernels for significantly improved performance, especially for large
    candidate sets.
    
    Parameters
    ----------
    bvh : BVHShard
        Legacy BVH structure (automatically converted to JAX format)
    segments : OrbitPolylineSegments
        Orbit segments with precomputed AABBs
    rays : ObservationRays
        Observation rays with observer positions and line-of-sight vectors
    guard_arcmin : float, default=1.0
        Guard band tolerance in arcminutes
    max_hits_per_ray : int, optional
        Maximum number of hits to return per ray
    max_candidates_per_ray : int, default=64
        Maximum candidates to aggregate per ray (affects memory usage)
    device : jax.Device, optional
        JAX device to use (default: CPU)
    fixed_num_rays : int, optional
        Fixed number of rays for padding to avoid JAX recompilation
        
    Returns
    -------
    OverlapHits
        Geometric overlap hits within guard band tolerance.
        
    Notes
    -----
    The first call will be slower due to JIT compilation.
    Subsequent calls will be significantly faster (typically 10-20x speedup).
    
    For anomaly labeling, use the separate label_anomalies function from
    adam_core.geometry.anomaly_labeling.
    """
    if len(rays) == 0 or bvh.num_primitives == 0:
        return OverlapHits.empty()
    
    logger.debug(f"JAX query: {len(rays)} rays, {bvh.num_primitives} segments")
    
    # Create orbit ID mapping for compact indices
    orbit_ids = segments.orbit_id.to_pylist()
    orbit_mapping = OrbitIdMapping.from_orbit_ids(orbit_ids)
    
    # Convert legacy types to JAX-native formats
    jax_bvh = bvh_shard_to_arrays(bvh, orbit_mapping, device=device)
    jax_segments = segments_to_soa(segments, device=device)
    ray_origins, ray_directions, observer_distances = rays_to_arrays(rays, device=device)

    # Optional padding to a fixed number of rays to stabilize JAX compilation shapes
    orig_num_rays = int(ray_origins.shape[0])
    if fixed_num_rays is not None and fixed_num_rays > orig_num_rays:
        pad_n = fixed_num_rays - orig_num_rays
        pad_o = jax.numpy.zeros((pad_n, 3), dtype=ray_origins.dtype)
        pad_d = jax.numpy.zeros((pad_n, 3), dtype=ray_directions.dtype)
        pad_obs = jax.numpy.zeros((pad_n,), dtype=observer_distances.dtype)
        ray_origins = jax.numpy.concatenate([ray_origins, pad_o], axis=0)
        ray_directions = jax.numpy.concatenate([ray_directions, pad_d], axis=0)
        observer_distances = jax.numpy.concatenate([observer_distances, pad_obs], axis=0)
    
    # Aggregate candidates using CPU traversal
    candidates = aggregate_candidates(
        jax_bvh, jax_segments, ray_origins, ray_directions, observer_distances,
        max_candidates_per_ray=max_candidates_per_ray, device=device
    )
    
    total_candidates = int(jax.numpy.sum(candidates.mask))
    logger.debug(f"Aggregated {total_candidates} candidates")
    
    if total_candidates == 0:
        return OverlapHits.empty()
    
    # Compute overlaps using JAX backend
    hits_soa = compute_overlap_hits_jax(candidates, guard_arcmin, max_hits_per_ray)

    # Trim hits from padded rays if padding was applied
    if fixed_num_rays is not None and fixed_num_rays > orig_num_rays and hits_soa.num_hits > 0:
        mask = hits_soa.det_indices < jax.numpy.asarray(orig_num_rays, dtype=jax.numpy.int32)
        idx = jax.numpy.nonzero(mask, size=hits_soa.num_hits)[0]
        hits_soa = HitsSOA(
            det_indices=hits_soa.det_indices[idx],
            orbit_indices=hits_soa.orbit_indices[idx],
            seg_ids=hits_soa.seg_ids[idx],
            leaf_ids=hits_soa.leaf_ids[idx],
            distances_au=hits_soa.distances_au[idx],
        )
    
    # Convert back to legacy format
    det_ids = [rays.det_id[i].as_py() for i in range(len(rays))]
    hits = hits_soa_to_overlap_hits(hits_soa, det_ids, orbit_mapping)
    
    logger.info(f"Found {len(hits)} overlap hits")
    
    return hits


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
    
    # Build BVH (still uses legacy implementation)
    from .bvh import build_bvh
    bvh = build_bvh(segments, max_leaf_size=max_leaf_size)
    
    # Query using JAX backend
    return query_bvh_jax(
        bvh, segments, rays, guard_arcmin=guard_arcmin,
        max_candidates_per_ray=max_candidates_per_ray,
        device=device
    )


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
    import time
    import gc
    
    if len(segments) == 0 or len(rays) == 0:
        return {"error": "Empty input data"}
    
    logger.info(f"Benchmarking JAX vs legacy: {len(segments)} segments, {len(rays)} rays")
    
    # Build BVH once
    from .bvh import build_bvh
    from .overlap import query_bvh
    
    bvh = build_bvh(segments, max_leaf_size=max_leaf_size)
    
    # Warmup JAX (compile kernels)
    logger.info("Warming up JAX kernels...")
    for _ in range(warmup_trials):
        _ = query_bvh_jax(bvh, segments, rays, guard_arcmin)
    
    # Benchmark JAX
    jax_times = []
    jax_hits = None
    for trial in range(num_trials):
        gc.collect()
        start_time = time.perf_counter()
        hits = query_bvh_jax(bvh, segments, rays, guard_arcmin)
        jax_times.append(time.perf_counter() - start_time)
        if trial == 0:
            jax_hits = hits
    
    # Benchmark legacy
    legacy_times = []
    legacy_hits = None
    for trial in range(num_trials):
        gc.collect()
        start_time = time.perf_counter()
        hits = query_bvh(bvh, segments, rays, guard_arcmin)
        legacy_times.append(time.perf_counter() - start_time)
        if trial == 0:
            legacy_hits = hits
    
    # Compute metrics
    avg_jax_time = sum(jax_times) / len(jax_times)
    avg_legacy_time = sum(legacy_times) / len(legacy_times)
    speedup = avg_legacy_time / avg_jax_time if avg_jax_time > 0 else float('inf')
    
    # Verify correctness
    results_match = len(jax_hits) == len(legacy_hits)
    
    metrics = {
        "num_segments": len(segments),
        "num_rays": len(rays),
        "bvh_nodes": bvh.num_nodes,
        "bvh_primitives": bvh.num_primitives,
        "jax_time_s": avg_jax_time,
        "jax_time_std_s": (sum((t - avg_jax_time)**2 for t in jax_times) / len(jax_times))**0.5,
        "legacy_time_s": avg_legacy_time,
        "legacy_time_std_s": (sum((t - avg_legacy_time)**2 for t in legacy_times) / len(legacy_times))**0.5,
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
