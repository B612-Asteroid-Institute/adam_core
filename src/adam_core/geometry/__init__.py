"""
Geometric algorithms and data structures for adam-core.

This package provides geometric primitives, spatial data structures,
and algorithms for efficient geometric queries.

The public API uses JAX-accelerated implementations by default for optimal performance.
"""

from ..observations.rays import ephemeris_to_rays
from .adapters import *
from .aggregator import *

# Import AnomalyLabels from anomaly module
from .anomaly import AnomalyLabels
from .anomaly_kernels import *
from .anomaly_labeling import *
from .bvh import *
from .clock_gating import *
from .jax_kernels import *

# Define canonical API that routes to JAX implementations
from .bvh_query import *
from .bvh_query import geometric_overlap_jax as geometric_overlap
from .bvh_query import query_bvh, query_bvh_index  # unified entrypoints
from .jax_types import *

# Import OverlapHits from overlap module
from .overlap import OverlapHits
from .projection import *
## Sharding is being removed; stop exporting sharded modules

__all__ = [
    # BVHIndex API (quivr/parquet)
    # Monolithic BVHIndex API (quivr/parquet)
    "BVHIndex",
    "BVHNodes",
    "BVHPrimitives",
    "build_bvh_index_from_segments",
    "query_bvh_index",
    # Public API (JAX-backed by default)
    "OverlapHits",
    "AnomalyLabels",
    "query_bvh",
    "geometric_overlap",
    # Clock gating
    "ClockGateConfig",
    "ClockGateResults",
    "apply_clock_gating",
    "compute_orbital_positions_at_times",
    # Anomaly labeling
    "label_anomalies",
    # Projection utilities
    "compute_orbital_plane_normal",
    "ray_to_plane_distance",
    "project_ray_to_orbital_plane",
    "ellipse_snap_distance",
    # JAX types
    "BVHArrays",
    "SegmentsSOA",
    "HitsSOA",
    "AnomalyLabelsSOA",
    "OrbitIdMapping",
    # JAX persistence
    "save_bvh_arrays",
    "load_bvh_arrays",
    "save_segments_soa",
    "load_segments_soa",
    # Adapters
    # shard adapters removed
    "segments_to_soa",
    "rays_to_arrays",
    # Conversions
    "ephemeris_to_rays",
    "hits_soa_to_overlap_hits",
    "overlap_hits_to_soa",
    "hits_soa_to_anomaly_labels_soa",
    "anomaly_labels_soa_to_anomaly_labels",
    # Aggregator
    "CandidateBatch",
    "aggregate_candidates",
    # JAX kernels
    "ray_segment_distances_jax",
    "compute_overlap_hits_jax",
    # JAX explicit API (kept for power users)
    "geometric_overlap_jax",
    "benchmark_jax_vs_legacy",
    # Monolithic BVHIndex API (quivr/parquet)
    "BVHIndex",
    "BVHNodes",
    "BVHPrimitives",
    "query_bvh_index",
    "build_bvh_index_from_segments",
    "build_bvh_index",
]
