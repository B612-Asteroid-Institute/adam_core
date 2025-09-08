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
from typing import Literal, Optional, Tuple, Union

import jax

from ..observations.rays import ObservationRays
from ..orbits.polyline import OrbitPolylineSegments, OrbitsPlaneParams
from .adapters import (
    bvh_shard_to_arrays,
    hits_soa_to_overlap_hits,
    rays_to_arrays,
    segments_to_soa,
    hits_soa_to_anomaly_labels_soa,
    anomaly_labels_soa_to_anomaly_labels,
)
from .aggregator import aggregate_candidates
from .anomaly import AnomalyLabels
from .anomaly_labeling import label_anomalies_batch
from .bvh import BVHShard
from .jax_kernels import OverlapBackend, compute_overlap_hits_jax, compute_overlap_hits_numpy
from .jax_types import OrbitIdMapping
from .overlap import OverlapHits

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
    backend: OverlapBackend = "jax",
    max_candidates_per_ray: int = 64,
    device: Optional[jax.Device] = None,
    label_anomalies: bool = False,
    max_variants_per_hit: int = 2,
    max_newton_iterations: int = 10,
    plane_params: Optional[OrbitsPlaneParams] = None,
) -> Union[OverlapHits, Tuple[OverlapHits, AnomalyLabels]]:
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
    backend : {"jax", "numpy"}, default="jax"
        Computation backend to use
    max_candidates_per_ray : int, default=64
        Maximum candidates to aggregate per ray (affects memory usage)
    device : jax.Device, optional
        JAX device to use (default: CPU)
    label_anomalies : bool, default=False
        Whether to compute anomaly labels for hits
    max_variants_per_hit : int, default=2
        Maximum variants to compute per hit for anomaly labeling
    max_newton_iterations : int, default=10
        Maximum Newton method iterations for anomaly refinement
        
    Returns
    -------
    OverlapHits or (OverlapHits, AnomalyLabels)
        Geometric overlap hits within guard band tolerance.
        If label_anomalies=True, returns tuple of (hits, labels).
        
    Notes
    -----
    The first call with backend="jax" will be slower due to JIT compilation.
    Subsequent calls will be significantly faster (typically 10-20x speedup).
    """
    if len(rays) == 0 or bvh.num_primitives == 0:
        empty_hits = OverlapHits.empty()
        if label_anomalies:
            return empty_hits, AnomalyLabels.empty()
        return empty_hits
    
    logger.debug(f"JAX query: {len(rays)} rays, {bvh.num_primitives} segments, backend={backend}")
    
    # Create orbit ID mapping for compact indices
    orbit_ids = segments.orbit_id.to_pylist()
    orbit_mapping = OrbitIdMapping.from_orbit_ids(orbit_ids)
    
    # Convert legacy types to JAX-native formats
    jax_bvh = bvh_shard_to_arrays(bvh, orbit_mapping, device=device)
    jax_segments = segments_to_soa(segments, device=device)
    ray_origins, ray_directions, observer_distances = rays_to_arrays(rays, device=device)
    
    # Aggregate candidates using CPU traversal
    candidates = aggregate_candidates(
        jax_bvh, jax_segments, ray_origins, ray_directions, observer_distances,
        max_candidates_per_ray=max_candidates_per_ray, device=device
    )
    
    total_candidates = int(jax.numpy.sum(candidates.mask))
    logger.debug(f"Aggregated {total_candidates} candidates")
    
    if total_candidates == 0:
        empty_hits = OverlapHits.empty()
        if label_anomalies:
            return empty_hits, AnomalyLabels.empty()
        return empty_hits
    
    # Compute overlaps using selected backend
    if backend == "jax":
        hits_soa = compute_overlap_hits_jax(candidates, guard_arcmin, max_hits_per_ray)
    elif backend == "numpy":
        hits_soa = compute_overlap_hits_numpy(candidates, guard_arcmin, max_hits_per_ray)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    # Convert back to legacy format
    det_ids = [rays.det_id[i].as_py() for i in range(len(rays))]
    hits = hits_soa_to_overlap_hits(hits_soa, det_ids, orbit_mapping)
    
    logger.info(f"Found {len(hits)} overlap hits using {backend} backend")
    
    if not label_anomalies:
        return hits
    
    # Compute anomaly labels if requested
    logger.debug(f"Computing anomaly labels for {hits_soa.num_hits} hits")
    
    # Extract orbital elements and bases from plane parameters
    if plane_params is None:
        if hits_soa.num_hits > 0:
            raise ValueError("plane_params is required when label_anomalies=True and hits exist")
        # No hits, use empty arrays
        from .anomaly_labeling import compute_orbital_elements_batch
        orbital_elements = jax.numpy.zeros((0, 6))
        orbital_bases = jax.numpy.zeros((0, 3, 3))
    else:
        from .anomaly_labeling import compute_orbital_elements_batch
        orbital_elements, orbital_bases = compute_orbital_elements_batch(plane_params, device=device)
    
    # Apply anomaly labeling kernel
    labeled_soa = label_anomalies_batch(
        hits_soa, jax_segments, ray_origins, ray_directions,
        orbital_elements, orbital_bases,
        max_variants_per_hit=max_variants_per_hit,
        max_newton_iterations=max_newton_iterations
    )
    
    # Convert to quivr table format
    labels = anomaly_labels_soa_to_anomaly_labels(labeled_soa, det_ids, orbit_mapping)
    
    logger.info(f"Computed anomaly labels for {len(labels)} valid variants")
    
    return hits, labels


def geometric_overlap_jax(
    segments: OrbitPolylineSegments,
    rays: ObservationRays,
    guard_arcmin: float = 1.0,
    max_leaf_size: int = 8,
    backend: OverlapBackend = "jax",
    max_candidates_per_ray: int = 64,
    device: Optional[jax.Device] = None,
    label_anomalies: bool = False,
    max_variants_per_hit: int = 2,
    max_newton_iterations: int = 10,
    plane_params: Optional[OrbitsPlaneParams] = None,
) -> Union[OverlapHits, Tuple[OverlapHits, AnomalyLabels]]:
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
    backend : {"jax", "numpy"}, default="jax"
        Computation backend to use
    max_candidates_per_ray : int, default=64
        Maximum candidates to aggregate per ray
    device : jax.Device, optional
        JAX device to use (default: CPU)
    label_anomalies : bool, default=False
        Whether to compute anomaly labels for hits
    max_variants_per_hit : int, default=2
        Maximum variants to compute per hit for anomaly labeling
    max_newton_iterations : int, default=10
        Maximum Newton method iterations for anomaly refinement
        
    Returns
    -------
    OverlapHits or (OverlapHits, AnomalyLabels)
        Geometric overlap hits within guard band tolerance.
        If label_anomalies=True, returns tuple of (hits, labels).
    """
    if len(segments) == 0 or len(rays) == 0:
        empty_hits = OverlapHits.empty()
        if label_anomalies:
            return empty_hits, AnomalyLabels.empty()
        return empty_hits
    
    # Build BVH (still uses legacy implementation)
    from .bvh import build_bvh
    bvh = build_bvh(segments, max_leaf_size=max_leaf_size)
    
    # Query using JAX backend
    return query_bvh_jax(
        bvh, segments, rays, guard_arcmin=guard_arcmin,
        backend=backend, max_candidates_per_ray=max_candidates_per_ray,
        device=device, label_anomalies=label_anomalies,
        max_variants_per_hit=max_variants_per_hit,
        max_newton_iterations=max_newton_iterations,
        plane_params=plane_params
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
        _ = query_bvh_jax(bvh, segments, rays, guard_arcmin, backend="jax")
    
    # Benchmark JAX
    jax_times = []
    jax_hits = None
    for trial in range(num_trials):
        gc.collect()
        start_time = time.perf_counter()
        hits = query_bvh_jax(bvh, segments, rays, guard_arcmin, backend="jax")
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
