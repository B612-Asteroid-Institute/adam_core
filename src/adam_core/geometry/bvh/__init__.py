"""
Bounding Volume Hierarchy (BVH) implementation for efficient geometric queries.

This module provides BVH data structures and algorithms for spatial indexing
and querying of orbit segments.

The main components are:
- BVHIndex: Container for BVH nodes, primitives, and segments
- build_bvh_index: High-level function to build BVH from orbits
- ObservationRays: Ray-based representation of observations
- query_bvh: Query BVH for ray-segment intersections

Orbits (as 2-body ellipses) are sampled into polyline segments, and then
a BVH is built over the segments. This provides a structure for efficiently querying
overlap with detections.

A benchmark grid was run to determine the best parameters for indexing and querying.
Using a synthetic orbit population across multiple orbital classes and a 180 day observations
window, the best parameters were found to be:

Index
max_chord_arcmin: 5.0
index_guard_arcmin: 0.65
max_segments_per_orbit: 512
epsilon_n_au: 1.0e-9
padding_method: baseline
max_leaf_size: 64

Query
query_guard_arcmin: 0.65
window_size: 32768
batch_size: 16384

These were the fastest parameters for both indexing and querying, while maintaining 100%
signal recall. A larger test with a neomod population (14k orbits, 90 days span) resulted
in 99.3% observation recall and 99.998 orbit recall. The BVH querying will fail to match
detections if significant n-body perturbations shift the orbit beyond the guard radius. Longer time deltas between the
orbit epoch definition and the observations will result in more missed detections as
n-body perturbations accumulate, but it is highly depending on the individual orbit.

Index guard should be >= to query guard in order to avoid missing intersections. Larger
guards will result in matching more noise and false positives, but can accomodate more
n-body divergence.

"""

# Utilities
# Main functions
# Core data structures
from .index import (
    BVHIndex,
    BVHNodes,
    BVHPrimitives,
    build_bvh_index,
    build_bvh_index_from_segments,
    get_leaf_primitives_numpy,
)
from .query import CandidatePairs, OverlapHits, find_bvh_matches, query_bvh

__all__ = [
    # Data structures
    "BVHIndex",
    "BVHNodes",
    "BVHPrimitives",
    "OverlapHits",
    "CandidatePairs",
    # Functions
    "build_bvh_index",
    "build_bvh_index_from_segments",
    "query_bvh",
    "find_bvh_matches",
    # Utilities
    "get_leaf_primitives_numpy",
]
