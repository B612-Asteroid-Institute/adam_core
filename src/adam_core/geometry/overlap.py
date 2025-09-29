"""
Geometric overlap detection data structures.

This module provides the OverlapHits table for representing geometric
overlap results. The actual overlap detection algorithms are implemented
in JAX-accelerated modules for optimal performance.
"""

from __future__ import annotations

import quivr as qv

__all__ = [
    "OverlapHits",
]


class OverlapHits(qv.Table):
    """
    Geometric overlap hits between observation rays and orbit segments.

    Each row represents a potential geometric overlap between an observation
    ray and an orbit segment, with distance and metadata for further processing.
    """

    #: Unique identifier for the detection
    det_id = qv.LargeStringColumn()

    #: Unique identifier for the orbit
    orbit_id = qv.LargeStringColumn()

    #: Segment identifier within the orbit
    seg_id = qv.Int32Column()

    #: BVH leaf node index containing this segment
    leaf_id = qv.Int32Column()

    #: Minimum distance between ray and segment in AU
    distance_au = qv.Float64Column()
