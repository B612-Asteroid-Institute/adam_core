"""
Ray-parallel sharded BVH query implementation.

This module provides distributed querying of sharded BVH indices using
Ray remote functions for efficient parallel processing across multiple
shards and nodes.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow as pa
import ray

from ..observations.rays import ObservationRays
from .overlap import OverlapHits
from .sharded_query import _compute_effective_guard, _merge_overlap_hits
from .sharding_types import ShardManifest, ShardMeta

logger = logging.getLogger(__name__)

# Default batch size and concurrency limits
DEFAULT_RAY_BATCH_SIZE = 200_000
DEFAULT_MAX_CONCURRENCY = None  # No limit by default


@ray.remote
def _query_shard_remote(
    manifest_dir_str: str,
    meta_dict: dict[str, Any],
    rays_ref: ray.ObjectRef,
    ray_batch_indices: list[np.ndarray],
    guard_arcmin: float,
    alpha: float = 0.0,
) -> dict[str, Any]:
    """
    Ray remote function to query a single shard across multiple ray batches.

    Parameters
    ----------
    manifest_dir_str : str
        String path to manifest directory (Ray serializable).
    meta_dict : dict
        ShardMeta as dictionary (Ray serializable).
    rays_ref : ray.ObjectRef
        Reference to rays in object store.
    ray_batch_indices : list[np.ndarray]
        List of index arrays for ray batches.
    guard_arcmin : float
        Base guard radius in arcminutes.
    alpha : float
        Guard inflation factor.

    Returns
    -------
    dict
        OverlapHits table as dictionary of arrays.
    """
    from .jax_overlap import query_bvh_jax
    from .sharding import load_shard
    from .sharding_types import ShardMeta

    t0 = time.perf_counter()

    # Reconstruct types
    manifest_dir = Path(manifest_dir_str)
    meta = ShardMeta.from_dict(meta_dict)
    rays = ray.get(rays_ref)

    # Load shard with memory mapping (once per task)
    segments, bvh_shard = load_shard(manifest_dir, meta)
    t_load = time.perf_counter()

    # Compute effective guard
    guard_eff = _compute_effective_guard(guard_arcmin, meta.max_chord_arcmin, alpha)

    print(
        f"[ray-shard {meta.shard_id}] loaded in {t_load - t0:.3f}s, guard_eff={guard_eff:.3f}"
    )

    # Process all ray batches for this shard
    shard_hits = []

    # Choose a fixed batch size for padding (largest batch)
    fixed_size = max((len(b) for b in ray_batch_indices), default=0)

    batch_times = []
    for batch_indices in ray_batch_indices:
        if len(batch_indices) == 0:
            continue

        tb = time.perf_counter()
        # Extract ray batch
        ray_batch = rays.take(batch_indices)

        # Query BVH for this batch
        batch_hits = query_bvh_jax(
            rays=ray_batch,
            segments=segments,
            bvh=bvh_shard,
            guard_arcmin=guard_eff,
            fixed_num_rays=fixed_size,
        )
        q_time = time.perf_counter() - tb
        batch_times.append(q_time)

        print(
            f"[ray-shard {meta.shard_id}] batch size={len(batch_indices)} queried in {q_time:.3f}s, hits={len(batch_hits)}"
        )

        if len(batch_hits) > 0:
            shard_hits.append(batch_hits)

    # Merge all batches for this shard
    if shard_hits:
        merged_hits = _merge_overlap_hits(shard_hits)
        # Convert to dictionary for Ray serialization
        return {
            col_name: (
                getattr(merged_hits, col_name).to_numpy(zero_copy_only=False)
                if col_name not in {"det_id", "orbit_id"}
                else getattr(merged_hits, col_name).to_pylist()
            )
            for col_name in merged_hits.schema.names
        }
    else:
        # Return empty dictionary
        return {col_name: [] for col_name in OverlapHits.empty().schema.names}


def query_manifest_ray(
    manifest: ShardManifest,
    rays: ObservationRays,
    guard_arcmin: float,
    alpha: float = 0.0,
    ray_batch_size: int = DEFAULT_RAY_BATCH_SIZE,
    max_concurrency: Optional[int] = DEFAULT_MAX_CONCURRENCY,
    manifest_dir: Path | None = None,
) -> OverlapHits:
    """
    Query a sharded BVH manifest using Ray for parallel processing.

    This function distributes shard queries across Ray workers, with each
    worker loading a shard via memory mapping and processing all ray batches
    for that shard locally to amortize I/O costs.

    Parameters
    ----------
    manifest : ShardManifest
        Manifest describing the sharded index.
    rays : ObservationRays
        Observation rays to query.
    guard_arcmin : float
        Base guard radius in arcminutes.
    alpha : float, default 0.0
        Guard inflation factor (guard_eff = guard + alpha * chord).
    ray_batch_size : int, default 200_000
        Number of rays to process in each batch.
    max_concurrency : int, optional
        Maximum number of concurrent shard tasks. If None, no limit.
    manifest_dir : Path, optional
        Directory containing manifest and shard files.
        If None, inferred from manifest file location.

    Returns
    -------
    OverlapHits
        All hits from the sharded index, sorted deterministically.
    """
    if manifest_dir is None:
        raise ValueError("manifest_dir must be provided")

    if not ray.is_initialized():
        raise RuntimeError("Ray must be initialized before calling query_manifest_ray")

    logger.info(f"Ray query: {len(manifest.shards)} shards, {len(rays)} rays")
    logger.info(f"Guard: {guard_arcmin} arcmin, alpha: {alpha}")
    logger.info(f"Ray batch size: {ray_batch_size}, max concurrency: {max_concurrency}")

    start_time = time.time()

    # Put rays in object store once
    rays_ref = ray.put(rays)

    # Precompute ray batch indices (use one batch if small to avoid extra JAX compiles)
    num_rays = len(rays)
    if ray_batch_size >= num_rays:
        ray_batch_indices = [np.arange(0, num_rays, dtype=np.int64)]
    else:
        ray_batch_indices = []
        for batch_start in range(0, num_rays, ray_batch_size):
            batch_end = min(batch_start + ray_batch_size, num_rays)
            batch_indices = np.arange(batch_start, batch_end, dtype=np.int64)
            ray_batch_indices.append(batch_indices)

    print(f"[driver] created {len(ray_batch_indices)} ray batches")

    # Submit shard tasks with concurrency control
    shard_results: list[OverlapHits] = []
    active: list[tuple[int, str, ray.ObjectRef]] = []
    pending = list(enumerate(manifest.shards))

    def submit_one() -> None:
        shard_idx, meta = pending.pop(0)
        print(f"[driver] submitting shard {meta.shard_id}")
        fut = _query_shard_remote.remote(
            str(manifest_dir),
            meta.to_dict(),
            rays_ref,
            ray_batch_indices,
            guard_arcmin,
            alpha,
        )
        active.append((shard_idx, meta.shard_id, fut))

    # Prime submissions
    while pending and (max_concurrency is None or len(active) < max_concurrency):
        submit_one()

    # Process completions
    while active:
        futures_only = [f for (_, _, f) in active]
        ready, _ = ray.wait(futures_only, num_returns=1)
        # Find completed
        comp_idx = next(i for i, (_, _, f) in enumerate(active) if f in ready)
        shard_idx, shard_id, fut = active.pop(comp_idx)
        print(f"[driver] completed shard {shard_id}")
        try:
            result_dict = ray.get(fut)
            if any(len(v) > 0 for v in result_dict.values()):
                hits = OverlapHits.from_kwargs(**result_dict)
                shard_results.append(hits)
                print(f"[driver] shard {shard_id} hits={len(hits)}")
            else:
                print(f"[driver] shard {shard_id} hits=0")
        except Exception as e:
            logger.error(f"Failed to process shard {shard_id}: {e}")
        # Backfill submissions
        if pending and (max_concurrency is None or len(active) < max_concurrency):
            submit_one()

    # Merge all shard results
    final_hits = _merge_overlap_hits(shard_results)

    total_time = time.time() - start_time
    rays_per_sec = len(rays) / total_time if total_time > 0 else 0

    print(
        f"[driver] query complete: hits={len(final_hits)}, total={total_time:.2f}s, rays/s={rays_per_sec:.0f}"
    )

    # Log hit distribution
    if len(final_hits) > 0:
        distances = final_hits.distance_au.to_numpy(zero_copy_only=False)
        logger.info(
            f"Hit distances: "
            f"min={np.min(distances):.6f} AU, "
            f"max={np.max(distances):.6f} AU, "
            f"mean={np.mean(distances):.6f} AU"
        )

    return final_hits


def estimate_ray_query_resources(
    manifest: ShardManifest,
    ray_batch_size: int = DEFAULT_RAY_BATCH_SIZE,
    max_concurrency: Optional[int] = DEFAULT_MAX_CONCURRENCY,
) -> dict[str, Any]:
    """
    Estimate resource requirements for Ray-parallel sharded query.

    Parameters
    ----------
    manifest : ShardManifest
        Manifest to analyze.
    ray_batch_size : int
        Ray batch size for processing.
    max_concurrency : int, optional
        Maximum concurrent shard tasks.

    Returns
    -------
    dict
        Resource estimates with keys:
        - 'max_shard_bytes': Largest single shard
        - 'concurrent_shards': Number of concurrent shards
        - 'peak_memory_bytes': Peak memory across all workers
        - 'ray_batch_bytes': Memory for ray batch
    """
    # Find largest shard
    max_shard_bytes = max(meta.total_bytes for meta in manifest.shards)

    # Determine concurrency
    concurrent_shards = min(
        len(manifest.shards), max_concurrency or len(manifest.shards)
    )

    # Estimate ray batch memory
    ray_batch_bytes = ray_batch_size * 200  # ~200 bytes per ray

    # Peak memory: concurrent shards * (shard + ray batch + overhead)
    peak_memory_bytes = concurrent_shards * (
        max_shard_bytes + ray_batch_bytes + int(0.1 * max_shard_bytes)
    )

    return {
        "max_shard_bytes": max_shard_bytes,
        "concurrent_shards": concurrent_shards,
        "peak_memory_bytes": peak_memory_bytes,
        "ray_batch_bytes": ray_batch_bytes,
        "total_shards": len(manifest.shards),
    }
