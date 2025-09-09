"""
BVH sharding system for large-scale orbit indices.

This module provides functionality to build, save, and load sharded BVH
indices that can be memory-mapped for efficient access across multiple
processes and nodes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import quivr as qv

from ..orbits.orbits import Orbits
from ..orbits.polyline import OrbitPolylineSegments, compute_segment_aabbs, sample_ellipse_adaptive
from .bvh import build_bvh
from .jax_types import OrbitIdMapping
from .jax_types import save_bvh_arrays, save_segments_soa
from .sharding_types import ShardData, ShardManifest, ShardMeta

logger = logging.getLogger(__name__)

# Version for manifest compatibility
MANIFEST_VERSION = "1.1.0"

# Default target shard size (3 GB)
DEFAULT_TARGET_SHARD_BYTES = 3_000_000_000


def estimate_shard_bytes(
    num_orbits: int,
    seg_per_orbit: int,
    float_dtype: str = "float64",
) -> int:
    """
    Estimate memory usage for a shard with given parameters.
    
    Parameters
    ----------
    num_orbits : int
        Number of orbits in the shard.
    seg_per_orbit : int
        Average number of segments per orbit.
    float_dtype : str, default "float64"
        Floating point precision ("float32" or "float64").
        
    Returns
    -------
    int
        Estimated bytes for the shard.
    """
    num_segments = num_orbits * seg_per_orbit
    
    # Bytes per float
    float_bytes = 8 if float_dtype == "float64" else 4
    
    # Segment storage: 7 floats per segment (x0,y0,z0,x1,y1,z1,r_mid_au)
    segment_bytes = num_segments * 7 * float_bytes
    
    # BVH storage: roughly 2*N nodes for N segments, 6 floats per node (min/max xyz)
    bvh_bytes = num_segments * 2 * 6 * float_bytes
    
    # Orbit mapping and metadata overhead (~10% of data)
    overhead_bytes = int(0.1 * (segment_bytes + bvh_bytes))
    
    return segment_bytes + bvh_bytes + overhead_bytes


def _determine_shard_boundaries(
    orbit_ids: list[str],
    segments_per_orbit: list[int],
    target_shard_bytes: int,
    float_dtype: str = "float64",
) -> list[tuple[int, int]]:
    """
    Determine shard boundaries based on target byte size.
    
    Parameters
    ----------
    orbit_ids : list[str]
        List of orbit IDs in order.
    segments_per_orbit : list[int]
        Number of segments for each orbit.
    target_shard_bytes : int
        Target size per shard in bytes.
    float_dtype : str
        Floating point precision.
        
    Returns
    -------
    list[tuple[int, int]]
        List of (start_idx, end_idx) pairs for each shard.
        end_idx is exclusive (Python slice convention).
    """
    boundaries = []
    start_idx = 0
    current_bytes = 0
    
    for i, seg_count in enumerate(segments_per_orbit):
        orbit_bytes = estimate_shard_bytes(1, seg_count, float_dtype)
        
        # If adding this orbit would exceed target, finalize current shard
        if current_bytes > 0 and current_bytes + orbit_bytes > target_shard_bytes:
            boundaries.append((start_idx, i))
            start_idx = i
            current_bytes = orbit_bytes
        else:
            current_bytes += orbit_bytes
    
    # Add final shard
    if start_idx < len(orbit_ids):
        boundaries.append((start_idx, len(orbit_ids)))
    
    return boundaries


def build_bvh_shards(
    orbits: Orbits,
    max_chord_arcmin: float,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    float_dtype: str = "float64",
) -> list[ShardData]:
    """
    Build BVH shards from orbits.
    
    Parameters
    ----------
    orbits : Orbits
        Input orbits to shard.
    max_chord_arcmin : float
        Maximum chord length for polyline sampling in arcminutes.
    target_shard_bytes : int, default 3_000_000_000
        Target size per shard in bytes.
    float_dtype : str, default "float64"
        Floating point precision ("float32" or "float64").
        
    Returns
    -------
    list[ShardData]
        List of shard data ready for persistence.
    """
    logger.info(f"Building BVH shards for {len(orbits)} orbits")
    logger.info(f"Target shard size: {target_shard_bytes / 1e9:.1f} GB")
    logger.info(f"Max chord: {max_chord_arcmin} arcmin")
    
    # Sample all orbits to get segment counts
    logger.info("Sampling orbits to determine shard boundaries...")
    all_segments = []
    segments_per_orbit = []
    orbit_ids = orbits.orbit_id.to_pylist()
    
    for i, orbit_id in enumerate(orbit_ids):
        if i % 1000 == 0:
            logger.info(f"Sampled {i}/{len(orbit_ids)} orbits")
        
        # Sample single orbit
        single_orbit = orbits.take([i])
        plane_params, segments = sample_ellipse_adaptive(single_orbit, max_chord_arcmin)
        all_segments.append(segments)
        segments_per_orbit.append(len(segments))
    
    logger.info(f"Total segments: {sum(segments_per_orbit)}")
    logger.info(f"Avg segments/orbit: {np.mean(segments_per_orbit):.1f}")
    
    # Determine shard boundaries
    boundaries = _determine_shard_boundaries(
        orbit_ids, segments_per_orbit, target_shard_bytes, float_dtype
    )
    
    logger.info(f"Creating {len(boundaries)} shards")
    
    # Build shards
    shards = []
    for shard_idx, (start_idx, end_idx) in enumerate(boundaries):
        logger.info(f"Building shard {shard_idx + 1}/{len(boundaries)}")
        
        # Combine segments for this shard
        shard_segments_list = all_segments[start_idx:end_idx]
        shard_orbit_ids = orbit_ids[start_idx:end_idx]
        
        # Concatenate all segments
        if len(shard_segments_list) == 1:
            shard_segments = shard_segments_list[0]
        else:
            # Concatenate multiple OrbitPolylineSegments
            combined_data = {}
            for col_name in shard_segments_list[0].schema.names:
                combined_arrays = []
                for segments in shard_segments_list:
                    combined_arrays.append(getattr(segments, col_name).to_numpy(zero_copy_only=False))
                combined_data[col_name] = np.concatenate(combined_arrays)
            
            shard_segments = OrbitPolylineSegments.from_kwargs(**combined_data)
        
        # Compute AABBs
        aabbs = compute_segment_aabbs(shard_segments)
        
        # Build BVH
        bvh = build_bvh(aabbs)
        
        # Create orbit mapping
        orbit_mapping = OrbitIdMapping.from_orbit_ids(shard_orbit_ids)
        
        # Create shard data
        shard_id = f"shard_{shard_idx:04d}"
        shard_data = ShardData(
            segments=shard_segments,
            bvh=bvh,
            orbit_mapping=orbit_mapping,
            shard_id=shard_id,
            orbit_id_start=shard_orbit_ids[0],
            orbit_id_end=shard_orbit_ids[-1],
            max_chord_arcmin=max_chord_arcmin,
            float_dtype=float_dtype,
        )
        
        shards.append(shard_data)
        
        logger.info(f"Shard {shard_idx}: {len(shard_orbit_ids)} orbits, "
                   f"{len(shard_segments)} segments")
    
    return shards


def _compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def save_shard(out_dir: Path, shard: ShardData) -> ShardMeta:
    """
    Save a shard to disk and return its metadata.
    
    Parameters
    ----------
    out_dir : Path
        Output directory for shard files.
    shard : ShardData
        Shard data to save.
        
    Returns
    -------
    ShardMeta
        Metadata for the saved shard.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # File paths
    segments_npz = f"{shard.shard_id}_segments.npz"
    bvh_npz = f"{shard.shard_id}_bvh.npz"
    orbit_ids_json = f"{shard.shard_id}_orbit_ids.json"
    
    segments_path = out_dir / segments_npz
    bvh_path = out_dir / bvh_npz
    orbit_ids_path = out_dir / orbit_ids_json
    
    logger.info(f"Saving shard {shard.shard_id} to {out_dir}")
    
    # Save segments
    from .adapters import segments_to_soa
    segments_soa = segments_to_soa(shard.segments)
    save_segments_soa(segments_soa, segments_path)
    
    # Save BVH
    from .adapters import bvh_shard_to_arrays
    bvh_arrays = bvh_shard_to_arrays(shard.bvh, shard.orbit_mapping)
    save_bvh_arrays(bvh_arrays, bvh_path)
    
    # Save orbit IDs (v1.1.0)
    orbit_ids = list(shard.orbit_mapping.index_to_id)
    with open(orbit_ids_path, 'w') as f:
        json.dump(orbit_ids, f)
    
    # Compute file sizes and hashes
    segments_bytes = segments_path.stat().st_size
    bvh_bytes = bvh_path.stat().st_size
    orbit_ids_bytes = orbit_ids_path.stat().st_size
    total_bytes = segments_bytes + bvh_bytes + orbit_ids_bytes
    
    file_hashes = {
        segments_npz: _compute_file_hash(segments_path),
        bvh_npz: _compute_file_hash(bvh_path),
        orbit_ids_json: _compute_file_hash(orbit_ids_path),
    }
    
    # Legacy estimate for backward compatibility
    estimated_bytes = estimate_shard_bytes(
        len(shard.orbit_mapping.id_to_index),
        len(shard.segments),
        shard.float_dtype,
    )
    
    # Create metadata
    meta = ShardMeta(
        shard_id=shard.shard_id,
        orbit_id_start=shard.orbit_id_start,
        orbit_id_end=shard.orbit_id_end,
        num_orbits=len(shard.orbit_mapping.id_to_index),
        num_segments=len(shard.segments),
        num_bvh_nodes=len(shard.bvh.nodes_min),
        max_chord_arcmin=shard.max_chord_arcmin,
        float_dtype=shard.float_dtype,
        segments_npz=segments_npz,
        bvh_npz=bvh_npz,
        orbit_ids_json=orbit_ids_json,
        segments_bytes=segments_bytes,
        bvh_bytes=bvh_bytes,
        orbit_ids_bytes=orbit_ids_bytes,
        total_bytes=total_bytes,
        file_hashes=file_hashes,
        estimated_bytes=estimated_bytes,
    )
    
    logger.info(f"Saved shard {shard.shard_id}: {total_bytes / 1e6:.1f} MB")
    
    return meta


def save_manifest(
    out_dir: Path,
    shards: list[ShardData],
    max_chord_arcmin: float,
    float_dtype: str = "float64",
) -> ShardManifest:
    """
    Save all shards and create a manifest.
    
    Parameters
    ----------
    out_dir : Path
        Output directory.
    shards : list[ShardData]
        List of shards to save.
    max_chord_arcmin : float
        Maximum chord length used.
    float_dtype : str
        Floating point precision used.
        
    Returns
    -------
    ShardManifest
        The created manifest.
    """
    logger.info(f"Saving {len(shards)} shards to {out_dir}")
    
    # Save all shards
    shard_metas = []
    for shard in shards:
        meta = save_shard(out_dir, shard)
        shard_metas.append(meta)
    
    # Create manifest
    manifest = ShardManifest(
        version=MANIFEST_VERSION,
        build_time=datetime.now().isoformat(),
        max_chord_arcmin=max_chord_arcmin,
        float_dtype=float_dtype,
        shards=shard_metas,
        total_orbits=sum(meta.num_orbits for meta in shard_metas),
        total_segments=sum(meta.num_segments for meta in shard_metas),
        total_bvh_nodes=sum(meta.num_bvh_nodes for meta in shard_metas),
        total_estimated_bytes=sum(meta.estimated_bytes for meta in shard_metas),
    )
    
    # Save manifest
    manifest_path = out_dir / "manifest.json"
    manifest.save(manifest_path)
    
    logger.info(f"Saved manifest: {manifest.total_orbits} orbits, "
               f"{manifest.total_segments} segments, "
               f"{manifest.total_estimated_bytes / 1e9:.1f} GB")
    
    return manifest


def load_shard(
    manifest_dir: Path,
    meta: ShardMeta,
) -> tuple[Any, Any]:  # (OrbitPolylineSegments, BVHShard)
    """
    Load a shard from disk with memory mapping.
    
    Parameters
    ----------
    manifest_dir : Path
        Directory containing the manifest and shard files.
    meta : ShardMeta
        Metadata for the shard to load.
        
    Returns
    -------
    tuple[OrbitPolylineSegments, BVHShard]
        The loaded segments and BVH shard.
    """
    from .adapters import bvh_arrays_to_shard, segments_soa_to_segments
    from .jax_types import load_bvh_arrays, load_segments_soa
    
    # Load with memory mapping
    segments_path = manifest_dir / meta.segments_npz
    bvh_path = manifest_dir / meta.bvh_npz
    orbit_ids_path = manifest_dir / meta.orbit_ids_json
    
    segments_soa = load_segments_soa(segments_path, mmap_mode="r")
    bvh_arrays = load_bvh_arrays(bvh_path, mmap_mode="r")
    
    # Load orbit IDs (v1.1.0) with fallback for v1.0.0
    if orbit_ids_path.exists():
        with open(orbit_ids_path) as f:
            orbit_ids = json.load(f)
    else:
        orbit_ids = [f"orbit_{i}" for i in range(meta.num_orbits)]
        logger.warning(f"Missing orbit_ids.json for shard {meta.shard_id}, using dummy IDs")
    
    # Convert to high-level types using orbit_id_index
    segments = segments_soa_to_segments(segments_soa, orbit_ids)
    bvh_shard = bvh_arrays_to_shard(bvh_arrays, orbit_ids=orbit_ids)
    
    return segments, bvh_shard
