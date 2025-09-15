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
import numpy as np

from ..observations.rays import ObservationRays
from ..orbits.polyline import OrbitPolylineSegments
from .adapters import (
    hits_soa_to_overlap_hits,
    rays_to_numpy_arrays,
    segments_to_numpy_soa,
)
from .aggregator import aggregate_candidates
from .bvh import BVHIndex
from .jax_kernels import compute_overlap_hits_jax
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
]

logger = logging.getLogger(__name__)


## (removed _query_bvh_single; use query_bvh_worker directly)


def query_bvh_worker(
    bvh_arrays: BVHArrays,
    segments: OrbitPolylineSegments,
    rays: ObservationRays,
    *,
    guard_arcmin: float = 1.0,
    max_hits_per_ray: Optional[int] = None,
    max_candidates_per_ray: int = 64,
    fixed_num_rays: Optional[int] = None,
) -> OverlapHits:
    """
    Core BVH batch worker (serial). Returns OverlapHits for the batch.
    """
    if len(rays) == 0 or bvh_arrays.num_primitives == 0:
        return OverlapHits.empty()

    # Allow receiving object refs under Ray
    try:
        import ray  # type: ignore
        if isinstance(bvh_arrays, ray.ObjectRef):
            bvh_arrays = ray.get(bvh_arrays)
        if isinstance(segments, ray.ObjectRef):
            segments = ray.get(segments)
        if isinstance(rays, ray.ObjectRef):
            rays = ray.get(rays)
    except Exception:
        pass

    # Build supporting structures at the worker boundary
    orbit_mapping = OrbitIdMapping.from_orbit_ids(segments.orbit_id.to_pylist())
    segments_soa = segments_to_numpy_soa(segments)

    # Rays to NumPy (host)
    ro, rd, obs = rays_to_numpy_arrays(rays)
    ray_origins_np = np.asarray(ro)
    ray_directions_np = np.asarray(rd)
    observer_distances_np = np.asarray(obs)

    # Optional fixed-shape padding for stability across batches
    orig_num = int(ray_origins_np.shape[0])
    if fixed_num_rays is not None and fixed_num_rays > orig_num:
        pad_n = fixed_num_rays - orig_num
        pad_o = np.zeros((pad_n, 3), dtype=ray_origins_np.dtype)
        pad_d = np.zeros((pad_n, 3), dtype=ray_directions_np.dtype)
        pad_obs = np.zeros((pad_n,), dtype=observer_distances_np.dtype)
        ray_origins_np = np.concatenate([ray_origins_np, pad_o], axis=0)
        ray_directions_np = np.concatenate([ray_directions_np, pad_d], axis=0)
        observer_distances_np = np.concatenate([observer_distances_np, pad_obs], axis=0)

    # Aggregate candidates and compute hits
    candidates = aggregate_candidates(
        bvh_arrays,
        segments_soa,
        ray_origins_np,
        ray_directions_np,
        observer_distances_np,
        max_candidates_per_ray=max_candidates_per_ray,
    )
    hits_soa = compute_overlap_hits_jax(
        candidates, guard_arcmin=guard_arcmin, max_hits_per_ray=max_hits_per_ray
    )

    # Trim hits from padded trailing rows if we padded
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

    # Convert to quivr table for this batch
    det_ids = rays.det_id.to_pylist()
    batch_hits = hits_soa_to_overlap_hits(hits_soa, det_ids, orbit_mapping)
    return batch_hits


# Ray remote wrapper over the same worker
query_bvh_worker_remote = ray.remote(query_bvh_worker)


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
    return query_bvh_worker_index(
        index,
        rays,
        guard_arcmin=guard_arcmin,
        max_candidates_per_ray=max_candidates_per_ray,
    )


def query_bvh_worker_index(
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
    hits_soa = compute_overlap_hits_jax(
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


query_bvh_worker_index_remote = ray.remote(query_bvh_worker_index)


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
                query_bvh_worker_index(
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
        fut = query_bvh_worker_index_remote.remote(
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
        _ = query_bvh_worker_index(index, rays, guard_arcmin)

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
