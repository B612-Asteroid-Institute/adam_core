"""
Local sharded BVH query implementation.

This module provides functionality to query sharded BVH indices using
memory-mapped files for efficient access without loading entire indices
into memory.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv

from ..observations.rays import ObservationRays
from .overlap import OverlapHits
from .jax_overlap import query_bvh_jax as query_bvh
from .sharding_types import ShardManifest, ShardMeta

logger = logging.getLogger(__name__)

# Default batch size for ray processing
DEFAULT_RAY_BATCH_SIZE = 200_000


def _merge_overlap_hits(hits_list: list[OverlapHits]) -> OverlapHits:
    """
    Merge multiple OverlapHits tables into a single table.
    
    Results are sorted by (det_id, distance_au) for deterministic output.
    
    Parameters
    ----------
    hits_list : list[OverlapHits]
        List of hits to merge.
        
    Returns
    -------
    OverlapHits
        Merged and sorted hits.
    """
    if not hits_list:
        # Return a canonical empty table
        return OverlapHits.empty()
    
    if len(hits_list) == 1:
        return hits_list[0]

    # Concatenate via pyarrow, then construct quivr table idiomatically
    tables = [hits.table for hits in hits_list]
    combined_table = pa.concat_tables(tables)
    combined_hits = OverlapHits.from_pyarrow(combined_table)

    # Deterministic sort using quivr's sort_by
    sorted_hits = combined_hits.sort_by([("det_id", "ascending"), ("distance_au", "ascending")])
    return sorted_hits


def _compute_effective_guard(
    base_guard_arcmin: float,
    chord_arcmin: float,
    alpha: float = 0.0,
) -> float:
    """
    Compute effective guard radius with chord-based inflation.
    
    Parameters
    ----------
    base_guard_arcmin : float
        Base guard radius in arcminutes.
    chord_arcmin : float
        Chord length in arcminutes.
    alpha : float, default 0.0
        Inflation factor (guard_eff = base_guard + alpha * chord).
        
    Returns
    -------
    float
        Effective guard radius in arcminutes.
    """
    return base_guard_arcmin + alpha * chord_arcmin


def _query_single_shard(
    manifest_dir: Path,
    meta: ShardMeta,
    rays: ObservationRays,
    guard_arcmin: float,
    alpha: float = 0.0,
) -> OverlapHits:
    """
    Query a single shard locally with memory mapping.
    
    Parameters
    ----------
    manifest_dir : Path
        Directory containing manifest and shard files.
    meta : ShardMeta
        Metadata for the shard to query.
    rays : ObservationRays
        Observation rays to query.
    guard_arcmin : float
        Base guard radius in arcminutes.
    alpha : float, default 0.0
        Guard inflation factor.
        
    Returns
    -------
    OverlapHits
        Hits from this shard.
    """
    from .sharding import load_shard
    
    # Load shard with memory mapping
    segments, bvh_shard = load_shard(manifest_dir, meta)
    
    # Compute effective guard
    guard_eff = _compute_effective_guard(guard_arcmin, meta.max_chord_arcmin, alpha)
    
    # Query BVH
    hits = query_bvh(
        rays=rays,
        segments=segments,
        bvh=bvh_shard,
        guard_arcmin=guard_eff,
    )
    
    return hits


def query_manifest_local(
    manifest: ShardManifest,
    rays: ObservationRays,
    guard_arcmin: float,
    alpha: float = 0.0,
    ray_batch_size: int = DEFAULT_RAY_BATCH_SIZE,
    manifest_dir: Path | None = None,
) -> OverlapHits:
    """
    Query a sharded BVH manifest locally.
    
    This function loads shards on-demand using memory mapping and processes
    rays in batches to control memory usage.
    
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
    manifest_dir : Path, optional
        Directory containing manifest and shard files.
        If None, inferred from manifest file location.
        
    Returns
    -------
    OverlapHits
        All hits from the sharded index, sorted deterministically.
    """
    if manifest_dir is None:
        # Try to infer from manifest (this won't work if manifest was created in memory)
        raise ValueError("manifest_dir must be provided")
    
    logger.info(f"Querying {len(manifest.shards)} shards with {len(rays)} rays")
    logger.info(f"Guard: {guard_arcmin} arcmin, alpha: {alpha}")
    logger.info(f"Ray batch size: {ray_batch_size}")
    
    start_time = time.time()
    
    # Process rays in batches
    num_rays = len(rays)
    all_hits = []
    
    for batch_start in range(0, num_rays, ray_batch_size):
        batch_end = min(batch_start + ray_batch_size, num_rays)
        # quivr/pyarrow take requires an array-like of integers, not range
        batch_indices = np.arange(batch_start, batch_end, dtype=np.int64)
        ray_batch = rays.take(batch_indices)
        
        logger.info(f"Processing ray batch {batch_start}-{batch_end} "
                   f"({len(ray_batch)} rays)")
        
        # Query all shards for this ray batch
        batch_hits = []
        shard_times = []
        
        for shard_idx, meta in enumerate(manifest.shards):
            shard_start = time.time()
            
            # Query this shard
            shard_hits = _query_single_shard(
                manifest_dir, meta, ray_batch, guard_arcmin, alpha
            )
            
            shard_time = time.time() - shard_start
            shard_times.append(shard_time)
            
            if len(shard_hits) > 0:
                batch_hits.append(shard_hits)
                logger.debug(f"Shard {shard_idx}: {len(shard_hits)} hits "
                           f"in {shard_time:.3f}s")
        
        # Merge hits from all shards for this batch
        if batch_hits:
            merged_batch_hits = _merge_overlap_hits(batch_hits)
            all_hits.append(merged_batch_hits)
            
            logger.info(f"Batch {batch_start}-{batch_end}: "
                       f"{len(merged_batch_hits)} total hits")
        
        # Log shard timing stats
        if shard_times:
            logger.info(f"Shard query times: "
                       f"mean={np.mean(shard_times):.3f}s, "
                       f"max={np.max(shard_times):.3f}s")
    
    # Merge all batches
    final_hits = _merge_overlap_hits(all_hits)
    
    total_time = time.time() - start_time
    rays_per_sec = len(rays) / total_time if total_time > 0 else 0
    
    logger.info(f"Query complete: {len(final_hits)} total hits")
    logger.info(f"Performance: {rays_per_sec:.0f} rays/s, {total_time:.1f}s total")
    
    # Log hit distribution
    if len(final_hits) > 0:
        distances = final_hits.distance_au.to_numpy()
        logger.info(f"Hit distances: "
                   f"min={np.min(distances):.6f} AU, "
                   f"max={np.max(distances):.6f} AU, "
                   f"mean={np.mean(distances):.6f} AU")
    
    return final_hits


def estimate_query_memory(
    manifest: ShardManifest,
    ray_batch_size: int = DEFAULT_RAY_BATCH_SIZE,
) -> dict[str, int]:
    """
    Estimate memory usage for querying a sharded manifest.
    
    Parameters
    ----------
    manifest : ShardManifest
        Manifest to analyze.
    ray_batch_size : int
        Ray batch size for processing.
        
    Returns
    -------
    dict[str, int]
        Memory estimates in bytes with keys:
        - 'max_shard_bytes': Largest single shard
        - 'ray_batch_bytes': Memory for ray batch
        - 'peak_query_bytes': Peak memory during query
    """
    # Find largest shard
    max_shard_bytes = max(meta.estimated_bytes for meta in manifest.shards)
    
    # Estimate ray batch memory (rough estimate)
    # Each ray: ~200 bytes (coordinates, time, strings, etc.)
    ray_batch_bytes = ray_batch_size * 200
    
    # Peak memory: largest shard + ray batch + overhead
    peak_query_bytes = max_shard_bytes + ray_batch_bytes + int(0.1 * max_shard_bytes)
    
    return {
        'max_shard_bytes': max_shard_bytes,
        'ray_batch_bytes': ray_batch_bytes,
        'peak_query_bytes': peak_query_bytes,
    }
