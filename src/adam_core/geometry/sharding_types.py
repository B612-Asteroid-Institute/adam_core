"""
Data types for BVH sharding system.

This module defines the core data structures used for sharding large BVH
indices across multiple files for efficient memory-mapped access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ShardMeta:
    """
    Metadata for a single BVH shard.

    A shard contains a contiguous range of orbit IDs and their associated
    polyline segments and BVH nodes, stored in memory-mappable .npz files.
    """

    # Shard identification
    shard_id: str
    orbit_id_start: str
    orbit_id_end: str

    # Content statistics
    num_orbits: int
    num_segments: int
    num_bvh_nodes: int

    # Build parameters
    max_chord_arcmin: float
    float_dtype: str

    # File paths (relative to manifest directory)
    segments_npz: str
    bvh_npz: str
    orbit_ids_json: str  # v1.1.0: JSON list of orbit IDs for this shard

    # File sizes and integrity (v1.1.0)
    segments_bytes: int
    bvh_bytes: int
    orbit_ids_bytes: int
    total_bytes: int
    file_hashes: dict[str, str]  # filename -> sha256 hex

    # Legacy field for backward compatibility
    estimated_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "shard_id": self.shard_id,
            "orbit_id_start": self.orbit_id_start,
            "orbit_id_end": self.orbit_id_end,
            "num_orbits": self.num_orbits,
            "num_segments": self.num_segments,
            "num_bvh_nodes": self.num_bvh_nodes,
            "max_chord_arcmin": self.max_chord_arcmin,
            "float_dtype": self.float_dtype,
            "segments_npz": self.segments_npz,
            "bvh_npz": self.bvh_npz,
            "orbit_ids_json": self.orbit_ids_json,
            "segments_bytes": self.segments_bytes,
            "bvh_bytes": self.bvh_bytes,
            "orbit_ids_bytes": self.orbit_ids_bytes,
            "total_bytes": self.total_bytes,
            "file_hashes": self.file_hashes,
            "estimated_bytes": self.estimated_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShardMeta:
        """Create from dictionary loaded from JSON with backward compatibility."""
        # v1.1.0 fields with defaults for v1.0.0 compatibility
        orbit_ids_json = data.get(
            "orbit_ids_json", f"{data['shard_id']}_orbit_ids.json"
        )
        segments_bytes = data.get("segments_bytes", 0)
        bvh_bytes = data.get("bvh_bytes", 0)
        orbit_ids_bytes = data.get("orbit_ids_bytes", 0)
        total_bytes = data.get("total_bytes", data["estimated_bytes"])
        file_hashes = data.get("file_hashes", {})

        return cls(
            shard_id=data["shard_id"],
            orbit_id_start=data["orbit_id_start"],
            orbit_id_end=data["orbit_id_end"],
            num_orbits=data["num_orbits"],
            num_segments=data["num_segments"],
            num_bvh_nodes=data["num_bvh_nodes"],
            max_chord_arcmin=data["max_chord_arcmin"],
            float_dtype=data["float_dtype"],
            segments_npz=data["segments_npz"],
            bvh_npz=data["bvh_npz"],
            orbit_ids_json=orbit_ids_json,
            segments_bytes=segments_bytes,
            bvh_bytes=bvh_bytes,
            orbit_ids_bytes=orbit_ids_bytes,
            total_bytes=total_bytes,
            file_hashes=file_hashes,
            estimated_bytes=data["estimated_bytes"],
        )


@dataclass(frozen=True)
class ShardManifest:
    """
    Manifest describing a complete sharded BVH index.

    Contains metadata for all shards and global statistics about the
    sharded index. Can be serialized to/from JSON.
    """

    # Build metadata
    version: str
    build_time: str  # ISO format
    max_chord_arcmin: float
    float_dtype: str

    # Shards
    shards: list[ShardMeta]

    # Global statistics
    total_orbits: int
    total_segments: int
    total_bvh_nodes: int
    total_estimated_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "build_time": self.build_time,
            "max_chord_arcmin": self.max_chord_arcmin,
            "float_dtype": self.float_dtype,
            "shards": [shard.to_dict() for shard in self.shards],
            "total_orbits": self.total_orbits,
            "total_segments": self.total_segments,
            "total_bvh_nodes": self.total_bvh_nodes,
            "total_estimated_bytes": self.total_estimated_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShardManifest:
        """Create from dictionary loaded from JSON."""
        return cls(
            version=data["version"],
            build_time=data["build_time"],
            max_chord_arcmin=data["max_chord_arcmin"],
            float_dtype=data["float_dtype"],
            shards=[ShardMeta.from_dict(shard_data) for shard_data in data["shards"]],
            total_orbits=data["total_orbits"],
            total_segments=data["total_segments"],
            total_bvh_nodes=data["total_bvh_nodes"],
            total_estimated_bytes=data["total_estimated_bytes"],
        )

    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> ShardManifest:
        """Load manifest from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class ShardData:
    """
    In-memory representation of shard data before persistence.

    Contains the actual orbit segments and BVH for a single shard,
    along with metadata needed for saving.
    """

    # Data
    segments: Any  # OrbitPolylineSegments
    bvh: Any  # BVHShard
    orbit_mapping: Any  # OrbitIdMapping

    # Metadata
    shard_id: str
    orbit_id_start: str
    orbit_id_end: str
    max_chord_arcmin: float
    float_dtype: str
