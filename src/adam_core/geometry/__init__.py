"""
Geometric algorithms and data structures for adam-core.

This package provides geometric primitives, spatial data structures,
and algorithms for efficient geometric queries.

The public API uses JAX-accelerated implementations by default for optimal performance.
"""

from .anomaly import AnomalyLabels
from .anomaly_labeling import *
from .bvh import *
from .clock_gating import *
from .projection import *

__all__ = [
    "BVHIndex",
    "BVHNodes",
    "BVHPrimitives",
    "OverlapHits",
    "AnomalyLabels",
    "query_bvh",
    "label_anomalies",
    "compute_orbital_plane_normal",
    "ray_to_plane_distance",
    "project_ray_to_orbital_plane",
    "ellipse_snap_distance",
    "ClockGatingCandidates",
    "ClockGatedEdges",
    "KeplerChains",
    "KeplerChainMembers",
    "prepare_clock_gating_candidates",
    "build_clock_gated_edges",
    "extract_kepler_chains",
    "kepler_clock_gate",
    "ephemeris_to_rays",
    "find_bvh_matches",
    "build_bvh_index",
]
