"""
Geometric algorithms and data structures for adam-core.

This package provides geometric primitives, spatial data structures,
and algorithms for efficient geometric queries.

The public API uses JAX-accelerated implementations by default for optimal performance.
"""

from .bvh import *
from .jax_types import *
from .adapters import *
from .aggregator import *
from .jax_kernels import *
from .jax_overlap import *
from .jax_remote import *
from .anomaly_kernels import *
from .anomaly_labeling import *
from .anomaly_remote import *
from .clock_gating import *
from .sharding import *
from .sharding_types import *
from .sharded_query import *
from .sharded_query_ray import *
from ..observations.rays import ephemeris_to_rays

# Import OverlapHits from overlap module
from .overlap import OverlapHits

# Import AnomalyLabels from anomaly module  
from .anomaly import AnomalyLabels

# Define canonical API that routes to JAX implementations
from .jax_overlap import query_bvh_jax as query_bvh
from .jax_overlap import geometric_overlap_jax as geometric_overlap
from .jax_remote import query_bvh_parallel_jax as query_bvh_parallel

__all__ = [
    # BVH construction and persistence
    "BVHShard",
    "build_bvh", 
    "save_bvh",
    "load_bvh",
    # Public API (JAX-backed by default)
    "OverlapHits",
    "AnomalyLabels",
    "query_bvh",
    "query_bvh_parallel", 
    "geometric_overlap",
    # Clock gating
    "ClockGateConfig",
    "ClockGateResults",
    "apply_clock_gating",
    "compute_orbital_positions_at_times",
    # Anomaly labeling
    "label_anomalies",
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
    "bvh_shard_to_arrays",
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
    # JAX explicit API
    "query_bvh_jax",
    "geometric_overlap_jax", 
    "benchmark_jax_vs_legacy",
    # Ray remote functions
    "process_ray_batch_remote",
    "query_bvh_parallel_jax",
    # Ray anomaly labeling
    "process_anomaly_batch_remote",
    "label_anomalies_parallel",
    # Sharding
    "ShardMeta",
    "ShardManifest", 
    "ShardData",
    "build_bvh_shards",
    "save_shard",
    "save_manifest",
    "load_shard",
    "estimate_shard_bytes",
    # Sharded query
    "query_manifest_local",
    "query_manifest_ray",
    "estimate_query_memory",
    "estimate_ray_query_resources",
]
