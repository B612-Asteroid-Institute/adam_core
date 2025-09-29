"""
Bounding Volume Hierarchy (BVH) implementation for efficient geometric queries.

This module provides a CPU-based BVH implementation using axis-aligned bounding
boxes (AABBs) with median split strategy for partitioning primitives.
"""

from __future__ import annotations

import logging
from typing import Optional
import quivr as qv
from quivr import concatenate
from ..utils.iter import _iterate_chunk_indices
from ..orbits.polyline import sample_ellipse_adaptive, compute_segment_aabbs
from ..ray_cluster import initialize_use_ray
import ray

import numpy as np
import numpy.typing as npt

from ..orbits.polyline import OrbitPolylineSegments
from ..orbits.orbits import Orbits
from .jax_types import BVHArrays

__all__ = [
    "BVHNodes",
    "BVHPrimitives",
    "BVHIndex",
    "build_bvh_index_from_segments",
    "build_bvh_index",
]

logger = logging.getLogger(__name__)


class BVHNodes(qv.Table):
    """
    Quivr table representing BVH nodes for a shard.
    """

    nodes_min_x = qv.Float64Column()
    nodes_min_y = qv.Float64Column()
    nodes_min_z = qv.Float64Column()
    nodes_max_x = qv.Float64Column()
    nodes_max_y = qv.Float64Column()
    nodes_max_z = qv.Float64Column()

    left_child = qv.Int32Column()
    right_child = qv.Int32Column()

    first_prim = qv.Int32Column()
    prim_count = qv.Int32Column()

    # Attributes
    shard_id = qv.StringAttribute(default="")
    version = qv.StringAttribute(default="1.0.0")
    float_dtype = qv.StringAttribute(default="float64")


class BVHPrimitives(qv.Table):
    """
    Quivr table representing BVH primitive arrays for a shard.
    """

    segment_row_index = qv.Int32Column()
    prim_seg_ids = qv.Int32Column()

    # Attributes
    shard_id = qv.StringAttribute(default="")
    version = qv.StringAttribute(default="1.0.0")


class BVHIndex:
    """
    Convenience wrapper bundling segments, nodes, and primitives for a BVH index.

    Provides simple parquet IO helpers consistent with other quivr table patterns.
    """

    def __init__(
        self,
        segments: "OrbitPolylineSegments",
        nodes: BVHNodes,
        prims: BVHPrimitives,
    ) -> None:
        self.segments = segments
        self.nodes = nodes
        self.prims = prims

    @classmethod
    def from_parquet(cls, directory: str) -> "BVHIndex":
        from ..orbits.polyline import OrbitPolylineSegments

        # Standard file names within the index directory
        seg_path = f"{directory.rstrip('/')}/segments.parquet"
        nodes_path = f"{directory.rstrip('/')}/bvh_nodes.parquet"
        prims_path = f"{directory.rstrip('/')}/bvh_prims.parquet"

        segments = OrbitPolylineSegments.from_parquet(seg_path)
        nodes = BVHNodes.from_parquet(nodes_path)
        prims = BVHPrimitives.from_parquet(prims_path)
        return cls(segments=segments, nodes=nodes, prims=prims)

    def to_parquet(self, directory: str) -> None:
        # Standard file names within the index directory
        seg_path = f"{directory.rstrip('/')}/segments.parquet"
        nodes_path = f"{directory.rstrip('/')}/bvh_nodes.parquet"
        prims_path = f"{directory.rstrip('/')}/bvh_prims.parquet"

        self.segments.to_parquet(seg_path)
        self.nodes.to_parquet(nodes_path)
        self.prims.to_parquet(prims_path)

    # ----- Convenience helpers -----
    def get_nodes_min_max_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return stacked node AABB arrays as NumPy with shape (num_nodes, 3).
        """
        nodes_min = np.column_stack(
            [
                np.asarray(self.nodes.nodes_min_x),
                np.asarray(self.nodes.nodes_min_y),
                np.asarray(self.nodes.nodes_min_z),
            ]
        )
        nodes_max = np.column_stack(
            [
                np.asarray(self.nodes.nodes_max_x),
                np.asarray(self.nodes.nodes_max_y),
                np.asarray(self.nodes.nodes_max_z),
            ]
        )
        return nodes_min, nodes_max

    def to_bvh_arrays(self) -> BVHArrays:
        """
        Materialize a BVHArrays view from Quivr tables.

        Computes is_leaf from child pointers to avoid redundant state.
        """
        nodes_min, nodes_max = self.get_nodes_min_max_numpy()

        left_child = np.asarray(self.nodes.left_child, dtype=np.int32)
        right_child = np.asarray(self.nodes.right_child, dtype=np.int32)
        is_leaf = (left_child == -1) if left_child.size else np.array([], dtype=bool)

        first_prim = np.asarray(self.nodes.first_prim, dtype=np.int32)
        prim_count = np.asarray(self.nodes.prim_count, dtype=np.int32)

        prim_row_index = np.asarray(self.prims.segment_row_index, dtype=np.int32)
        prim_seg_ids = np.asarray(self.prims.prim_seg_ids, dtype=np.int32)

        arrays = BVHArrays(
            nodes_min=nodes_min,
            nodes_max=nodes_max,
            left_child=left_child,
            right_child=right_child,
            is_leaf=is_leaf,
            first_prim=np.asarray(first_prim, dtype=np.int32),
            prim_count=np.asarray(prim_count, dtype=np.int32),
            prim_row_index=prim_row_index,
            prim_seg_ids=prim_seg_ids,
        )
        # Validate structure early to catch inconsistencies
        arrays.validate_structure()
        return arrays

    def validate(self) -> None:
        """
        Validate invariants across nodes and primitives.
        """
        n_nodes = len(self.nodes)
        n_prims = len(self.prims)

        # Basic length checks
        assert all(
            len(col) == n_nodes
            for col in [
                self.nodes.nodes_min_x,
                self.nodes.nodes_min_y,
                self.nodes.nodes_min_z,
                self.nodes.nodes_max_x,
                self.nodes.nodes_max_y,
                self.nodes.nodes_max_z,
                self.nodes.left_child,
                self.nodes.right_child,
                self.nodes.first_prim,
                self.nodes.prim_count,
            ]
        ), "Node columns must have consistent length"

        assert all(
            len(col) == n_prims
            for col in [
                self.prims.segment_row_index,
                self.prims.prim_seg_ids,
            ]
        ), "Primitive columns must have consistent length"

        # Leaf status derived from children
        left_child = np.asarray(self.nodes.left_child, dtype=np.int32)
        right_child = np.asarray(self.nodes.right_child, dtype=np.int32)
        is_leaf = left_child == -1
        # If a node is a leaf (left -1), right must be -1 too
        if np.any(is_leaf):
            assert np.all(right_child[is_leaf] == -1)

        # first_prim/prim_count ranges within primitive arrays
        first = np.asarray(self.nodes.first_prim, dtype=np.int32)
        count = np.asarray(self.nodes.prim_count, dtype=np.int32)
        total = len(self.prims.segment_row_index)

        # Internal nodes must have count == 0 and first == -1
        internal_mask = ~is_leaf
        if np.any(internal_mask):
            assert np.all(count[internal_mask] == 0)
            assert np.all(first[internal_mask] == -1)

        # Leaf nodes must have valid ranges when count > 0
        leaf_mask = is_leaf
        with_prims = leaf_mask & (count > 0)
        if np.any(with_prims):
            assert np.all(first[with_prims] >= 0)
            assert np.all(first[with_prims] + count[with_prims] <= total)


def orbits_to_segments_worker(
    orbits: Orbits,
    max_chord_arcmin: float = 2.0,
    guard_arcmin: float = 1.0,
) -> OrbitPolylineSegments:
    _, segs = sample_ellipse_adaptive(orbits, max_chord_arcmin=max_chord_arcmin)
    segs = compute_segment_aabbs(segs, guard_arcmin=guard_arcmin)
    return segs


@ray.remote
def orbits_segments_worker_remote(
    orbits: Orbits,
    max_chord_arcmin: float = 2.0,
    guard_arcmin: float = 1.0,
) -> OrbitPolylineSegments:
    return orbits_to_segments_worker(orbits, max_chord_arcmin=max_chord_arcmin, guard_arcmin=guard_arcmin)



def build_bvh_index(
    orbits: Orbits,
    *,
    max_chord_arcmin: float = 2.0,
    guard_arcmin: float = 1.0,
    max_leaf_size: int = 8,
    chunk_size_orbits: int = 1000,
    max_processes: int = 0,
) -> BVHIndex:
    """
    High-level builder with orbit chunking: Orbits → segments → AABBs → BVHIndex.

    Chunks the input orbits to bound peak memory during sampling/AABB computation.
    """

    num_orbits = len(orbits)
    if num_orbits == 0:
        return BVHIndex(
            segments=OrbitPolylineSegments.empty(),
            nodes=BVHNodes.empty(),
            prims=BVHPrimitives.empty(),
        )

    chunk_definitions: list[tuple[int, int]] = []
    for chunk in _iterate_chunk_indices(orbits, chunk_size_orbits):
        chunk_definitions.append(chunk)

    segments_chunks: list[OrbitPolylineSegments] = []

    if max_processes is None or max_processes <= 1:
        for start, end in chunk_definitions:
            segs = orbits_to_segments_worker(
                orbits[start:end],
                max_chord_arcmin=max_chord_arcmin,
                guard_arcmin=guard_arcmin,
            )
            segments_chunks.append(segs)
    else:
        initialize_use_ray(num_cpus=max_processes)
        futures: list[ray.ObjectRef] = []
        for start, end in chunk_definitions:
            future = orbits_segments_worker_remote.remote(
                orbits[start:end],
                max_chord_arcmin=max_chord_arcmin,
                guard_arcmin=guard_arcmin,
            )
            futures.append(future)
            if len(futures) >= max_processes * 1.5:
                finished, futures = ray.wait(futures, num_returns=1)
                segments_chunks.append(ray.get(finished[0]))
        while futures:
            finished, futures = ray.wait(futures, num_returns=1)
            segments_chunks.append(ray.get(finished[0]))


    # Concatenate all segments (defragment into a single contiguous table)
    segments_all = (
        concatenate(segments_chunks, defrag=True)
        if segments_chunks
        else segments_chunks[0]
    )

    # Build BVH over all segments
    return build_bvh_index_from_segments(segments_all, max_leaf_size=max_leaf_size)


def build_bvh_index_from_segments(
    segments: "OrbitPolylineSegments", max_leaf_size: int = 8
) -> BVHIndex:
    """
    Build a monolithic BVHIndex from polyline segments.
    """
    # Internal array-based builder
    (
        nodes_min_arr,
        nodes_max_arr,
        left_child_arr,
        right_child_arr,
        first_prim_arr,
        prim_count_arr,
        prim_orbit_ids,
        prim_seg_ids,
        prim_row_indices,
    ) = _build_bvh_arrays(segments, max_leaf_size=max_leaf_size)

    nodes = BVHNodes.from_kwargs(
        nodes_min_x=(
            nodes_min_arr[:, 0] if len(nodes_min_arr) else np.array([], dtype=float)
        ),
        nodes_min_y=(
            nodes_min_arr[:, 1] if len(nodes_min_arr) else np.array([], dtype=float)
        ),
        nodes_min_z=(
            nodes_min_arr[:, 2] if len(nodes_min_arr) else np.array([], dtype=float)
        ),
        nodes_max_x=(
            nodes_max_arr[:, 0] if len(nodes_max_arr) else np.array([], dtype=float)
        ),
        nodes_max_y=(
            nodes_max_arr[:, 1] if len(nodes_max_arr) else np.array([], dtype=float)
        ),
        nodes_max_z=(
            nodes_max_arr[:, 2] if len(nodes_max_arr) else np.array([], dtype=float)
        ),
        left_child=left_child_arr,
        right_child=right_child_arr,
        first_prim=first_prim_arr,
        prim_count=prim_count_arr,
        shard_id="index",
    )

    prims = BVHPrimitives.from_kwargs(
        segment_row_index=prim_row_indices,
        prim_seg_ids=prim_seg_ids,
        shard_id="index",
    )

    return BVHIndex(segments=segments, nodes=nodes, prims=prims)


def _compute_aabb_from_arrays(
    min_x: npt.NDArray[np.float64],
    min_y: npt.NDArray[np.float64],
    min_z: npt.NDArray[np.float64],
    max_x: npt.NDArray[np.float64],
    max_y: npt.NDArray[np.float64],
    max_z: npt.NDArray[np.float64],
    indices: Optional[npt.NDArray[np.int32]] = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute axis-aligned bounding box for a set of segments using pre-extracted arrays.

    Parameters
    ----------
    min_x, min_y, min_z, max_x, max_y, max_z : np.ndarray
        Arrays containing per-segment AABB bounds
    indices : np.ndarray, optional
        Indices of segments to include (if None, use all)

    Returns
    -------
    aabb_min : np.ndarray (3,)
        Minimum bounds
    aabb_max : np.ndarray (3,)
        Maximum bounds
    """
    if indices is None:
        if min_x.size == 0:
            return np.full(3, np.inf), np.full(3, -np.inf)
        sel_min_x = min_x
        sel_min_y = min_y
        sel_min_z = min_z
        sel_max_x = max_x
        sel_max_y = max_y
        sel_max_z = max_z
    else:
        if len(indices) == 0:
            return np.full(3, np.inf), np.full(3, -np.inf)
        sel_min_x = min_x[indices]
        sel_min_y = min_y[indices]
        sel_min_z = min_z[indices]
        sel_max_x = max_x[indices]
        sel_max_y = max_y[indices]
        sel_max_z = max_z[indices]

    return (
        np.array([sel_min_x.min(), sel_min_y.min(), sel_min_z.min()]),
        np.array([sel_max_x.max(), sel_max_y.max(), sel_max_z.max()]),
    )


def _compute_centroids(segments: OrbitPolylineSegments) -> npt.NDArray[np.float64]:
    """
    Compute centroids of segment AABBs.

    Parameters
    ----------
    segments : OrbitPolylineSegments
        Input segments

    Returns
    -------
    centroids : np.ndarray (N, 3)
        Centroid of each segment's AABB
    """
    min_x = segments.aabb_min_x.to_numpy()
    min_y = segments.aabb_min_y.to_numpy()
    min_z = segments.aabb_min_z.to_numpy()

    max_x = segments.aabb_max_x.to_numpy()
    max_y = segments.aabb_max_y.to_numpy()
    max_z = segments.aabb_max_z.to_numpy()

    centroids = np.column_stack(
        [
            (min_x + max_x) / 2,
            (min_y + max_y) / 2,
            (min_z + max_z) / 2,
        ]
    )

    return centroids


def _build_bvh_recursive(
    indices: npt.NDArray[np.int32],
    start: int,
    end: int,
    min_x: npt.NDArray[np.float64],
    min_y: npt.NDArray[np.float64],
    min_z: npt.NDArray[np.float64],
    max_x: npt.NDArray[np.float64],
    max_y: npt.NDArray[np.float64],
    max_z: npt.NDArray[np.float64],
    nodes_min: list[npt.NDArray[np.float64]],
    nodes_max: list[npt.NDArray[np.float64]],
    left_child: list[int],
    right_child: list[int],
    first_prim: list[int],
    prim_count: list[int],
    max_leaf_size: int = 8,
) -> int:
    """
    Recursively build BVH nodes.

    Parameters
    ----------
    segments : OrbitPolylineSegments
        All segments
    indices : np.ndarray
        Indices of segments in current node
    centroids : np.ndarray
        Precomputed centroids for all segments
    nodes_min, nodes_max : list
        Node bounds (modified in place)
    left_child, right_child : list
        Child indices (modified in place)
    first_prim, prim_count : list
        Leaf primitive info (modified in place)
    max_leaf_size : int
        Maximum primitives per leaf

    Returns
    -------
    node_idx : int
        Index of created node
    """
    node_idx = len(nodes_min)

    # Compute AABB for this node using pre-extracted arrays to avoid Arrow overhead
    aabb_min, aabb_max = _compute_aabb_from_arrays(
        min_x, min_y, min_z, max_x, max_y, max_z, indices[start:end]
    )
    nodes_min.append(aabb_min)
    nodes_max.append(aabb_max)

    # Check if we should create a leaf
    count_here = end - start
    if count_here <= max_leaf_size:
        # Create leaf node
        left_child.append(-1)
        right_child.append(-1)
        # Temporarily store range in the global indices array; will repoint after packing
        first_prim.append(start)
        prim_count.append(count_here)
        return node_idx

    # Find best split axis (largest extent)
    extent = aabb_max - aabb_min
    split_axis = np.argmax(extent)

    # Split indices by centroid along split axis using argpartition (O(n))
    if split_axis == 0:
        axis_centroids = (min_x + max_x) * 0.5
    elif split_axis == 1:
        axis_centroids = (min_y + max_y) * 0.5
    else:
        axis_centroids = (min_z + max_z) * 0.5

    # Work on the subrange [start:end]
    sub = indices[start:end]
    sub_centroids = axis_centroids[sub]
    mid = count_here // 2
    part_order = np.argpartition(sub_centroids, mid)
    # Reorder indices in place for this node's subrange
    indices[start:end] = sub[part_order]
    mid_idx = start + mid

    # Create internal node
    left_child.append(-1)  # Will be filled by recursive call
    right_child.append(-1)  # Will be filled by recursive call
    first_prim.append(-1)
    prim_count.append(0)
    # Placeholder alignment for internal node

    # Recursively build children
    left_idx = _build_bvh_recursive(
        indices,
        start,
        mid_idx,
        min_x,
        min_y,
        min_z,
        max_x,
        max_y,
        max_z,
        nodes_min,
        nodes_max,
        left_child,
        right_child,
        first_prim,
        prim_count,
        max_leaf_size,
    )
    right_idx = _build_bvh_recursive(
        indices,
        mid_idx,
        end,
        min_x,
        min_y,
        min_z,
        max_x,
        max_y,
        max_z,
        nodes_min,
        nodes_max,
        left_child,
        right_child,
        first_prim,
        prim_count,
        max_leaf_size,
    )

    # Update child pointers
    left_child[node_idx] = left_idx
    right_child[node_idx] = right_idx

    return node_idx


def _build_bvh_arrays(
    segments: OrbitPolylineSegments,
    max_leaf_size: int = 8,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    list[str],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """
    Build a BVH over orbit segments and return raw arrays (no object wrapper).
    """
    if len(segments) == 0:
        return (
            np.empty((0, 3)),
            np.empty((0, 3)),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            [],
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )

    # Validate that AABBs are computed (null-safe)
    import pyarrow.compute as pc
    if pc.any(pc.is_null(segments.aabb_min_x)).as_py():
        raise ValueError("Segments must have computed AABBs. Call compute_segment_aabbs() first.")

    # Pre-extract per-segment AABBs
    min_x = segments.aabb_min_x.to_numpy()
    min_y = segments.aabb_min_y.to_numpy()
    min_z = segments.aabb_min_z.to_numpy()
    max_x = segments.aabb_max_x.to_numpy()
    max_y = segments.aabb_max_y.to_numpy()
    max_z = segments.aabb_max_z.to_numpy()

    # Initialize node arrays
    nodes_min = []
    nodes_max = []
    left_child = []
    right_child = []
    first_prim = []
    prim_count = []

    # Build BVH recursively
    indices = np.arange(len(segments), dtype=np.int32)
    root_idx = _build_bvh_recursive(
        indices,
        0,
        int(len(indices)),
        min_x,
        min_y,
        min_z,
        max_x,
        max_y,
        max_z,
        nodes_min,
        nodes_max,
        left_child,
        right_child,
        first_prim,
        prim_count,
        max_leaf_size,
    )

    assert root_idx == 0, "Root should be at index 0"

    # Convert to numpy arrays
    nodes_min_arr = np.array(nodes_min)
    nodes_max_arr = np.array(nodes_max)
    left_child_arr = np.array(left_child, dtype=np.int32)
    right_child_arr = np.array(right_child, dtype=np.int32)
    first_prim_arr = np.array(first_prim, dtype=np.int32)
    prim_count_arr = np.array(prim_count, dtype=np.int32)

    # Pack leaf primitives contiguously without storing per-leaf lists
    packed_prim_indices: list[int] = []
    node_first_offsets = np.full(len(nodes_min_arr), -1, dtype=np.int32)
    node_counts = np.array(prim_count_arr, dtype=np.int32)

    offset = 0
    for node_idx in range(len(nodes_min_arr)):
        if left_child_arr[node_idx] == -1:  # leaf
            count = int(node_counts[node_idx])
            if count > 0:
                start = int(first_prim_arr[node_idx])
                node_first_offsets[node_idx] = offset
                slice_inds = indices[start : start + count]
                packed_prim_indices.extend(slice_inds.tolist())
                offset += count

    packed_prim_indices_arr = np.array(packed_prim_indices, dtype=np.int32)

    # Remap primitive arrays to packed order
    prim_orbit_ids_all = segments.orbit_id.to_pylist()
    prim_seg_ids_all = segments.seg_id.to_numpy()
    prim_orbit_ids = [prim_orbit_ids_all[i] for i in packed_prim_indices_arr]
    prim_seg_ids = prim_seg_ids_all[packed_prim_indices_arr]
    prim_row_indices = packed_prim_indices_arr.copy()  # Row indices for O(1) lookup

    # Update first_prim to use packed offsets
    first_prim_arr = node_first_offsets
    prim_count_arr = node_counts

    logger.info(
        f"Built BVH with {len(nodes_min_arr)} nodes over {len(prim_orbit_ids)} segments"
    )

    return (
        nodes_min_arr,
        nodes_max_arr,
        left_child_arr,
        right_child_arr,
        first_prim_arr,
        prim_count_arr,
        prim_orbit_ids,
        prim_seg_ids,
        prim_row_indices,
    )


## Removed legacy save/load stubs; BVHIndex parquet IO is canonical
