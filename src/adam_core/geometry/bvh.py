"""
Bounding Volume Hierarchy (BVH) implementation for efficient geometric queries.

This module provides a CPU-based BVH implementation using axis-aligned bounding
boxes (AABBs) with median split strategy for partitioning primitives.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import numpy.typing as npt

from ..orbits.polyline import OrbitPolylineSegments

__all__ = [
    "BVHShard",
    "build_bvh",
    "save_bvh",
    "load_bvh",
]

logger = logging.getLogger(__name__)


class BVHShard:
    """
    A bounding volume hierarchy for efficient ray-segment intersection queries.
    
    This class represents a binary tree of axis-aligned bounding boxes (AABBs)
    built over a collection of orbit segments. The tree enables logarithmic-time
    queries for finding segments that potentially intersect with a given ray.
    
    Attributes
    ----------
    nodes_min : np.ndarray (N, 3)
        Minimum bounds of each BVH node
    nodes_max : np.ndarray (N, 3)  
        Maximum bounds of each BVH node
    left_child : np.ndarray (N,)
        Index of left child node (-1 for leaves)
    right_child : np.ndarray (N,)
        Index of right child node (-1 for leaves)
    first_prim : np.ndarray (N,)
        Index of first primitive in leaf node (-1 for internal nodes)
    prim_count : np.ndarray (N,)
        Number of primitives in leaf node (0 for internal nodes)
    prim_orbit_ids : list[str]
        Orbit IDs for each primitive (parallel to flattened segments)
    prim_seg_ids : np.ndarray (M,)
        Segment IDs for each primitive
    prim_row_index : np.ndarray (M,)
        Row indices into original segments table for O(1) lookup
    """
    
    def __init__(
        self,
        nodes_min: npt.NDArray[np.float64],
        nodes_max: npt.NDArray[np.float64],
        left_child: npt.NDArray[np.int32],
        right_child: npt.NDArray[np.int32],
        first_prim: npt.NDArray[np.int32],
        prim_count: npt.NDArray[np.int32],
        prim_orbit_ids: list[str],
        prim_seg_ids: npt.NDArray[np.int32],
        prim_row_index: npt.NDArray[np.int32],
    ):
        self.nodes_min = nodes_min
        self.nodes_max = nodes_max
        self.left_child = left_child
        self.right_child = right_child
        self.first_prim = first_prim
        self.prim_count = prim_count
        self.prim_orbit_ids = prim_orbit_ids
        self.prim_seg_ids = prim_seg_ids
        self.prim_row_index = prim_row_index
        
        # Validate structure
        assert len(nodes_min) == len(nodes_max) == len(left_child) == len(right_child)
        assert len(nodes_min) == len(first_prim) == len(prim_count)
        assert len(prim_orbit_ids) == len(prim_seg_ids) == len(prim_row_index)
        
    @property
    def num_nodes(self) -> int:
        """Number of nodes in the BVH."""
        return len(self.nodes_min)
        
    @property
    def num_primitives(self) -> int:
        """Number of primitives (segments) in the BVH."""
        return len(self.prim_orbit_ids)
        
    def is_leaf(self, node_idx: int) -> bool:
        """Check if a node is a leaf."""
        return self.left_child[node_idx] == -1
        
    def get_leaf_primitives(self, node_idx: int) -> tuple[list[str], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """
        Get primitives contained in a leaf node.
        
        Parameters
        ----------
        node_idx : int
            Index of the leaf node
            
        Returns
        -------
        orbit_ids : list[str]
            Orbit IDs of primitives in this leaf
        seg_ids : np.ndarray
            Segment IDs of primitives in this leaf
        row_indices : np.ndarray
            Row indices into original segments table for O(1) lookup
        """
        if not self.is_leaf(node_idx):
            raise ValueError(f"Node {node_idx} is not a leaf")
            
        first = self.first_prim[node_idx]
        count = self.prim_count[node_idx]
        
        if first == -1 or count == 0:
            return [], np.array([], dtype=np.int32), np.array([], dtype=np.int32)
            
        orbit_ids = self.prim_orbit_ids[first:first + count]
        seg_ids = self.prim_seg_ids[first:first + count]
        row_indices = self.prim_row_index[first:first + count]
        
        return orbit_ids, seg_ids, row_indices


def _compute_aabb(
    segments: OrbitPolylineSegments,
    indices: Optional[npt.NDArray[np.int32]] = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute axis-aligned bounding box for a set of segments.
    
    Parameters
    ----------
    segments : OrbitPolylineSegments
        Segments to bound
    indices : np.ndarray, optional
        Indices of segments to include (if None, use all)
        
    Returns
    -------
    aabb_min : np.ndarray (3,)
        Minimum bounds
    aabb_max : np.ndarray (3,)
        Maximum bounds
    """
    if indices is not None:
        if len(indices) == 0:
            return np.full(3, np.inf), np.full(3, -np.inf)
        segments = segments.take(indices)
    
    if len(segments) == 0:
        return np.full(3, np.inf), np.full(3, -np.inf)
    
    # Use precomputed AABBs from segments
    min_x = np.min(segments.aabb_min_x.to_numpy())
    min_y = np.min(segments.aabb_min_y.to_numpy())
    min_z = np.min(segments.aabb_min_z.to_numpy())
    
    max_x = np.max(segments.aabb_max_x.to_numpy())
    max_y = np.max(segments.aabb_max_y.to_numpy())
    max_z = np.max(segments.aabb_max_z.to_numpy())
    
    return np.array([min_x, min_y, min_z]), np.array([max_x, max_y, max_z])


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
    
    centroids = np.column_stack([
        (min_x + max_x) / 2,
        (min_y + max_y) / 2,
        (min_z + max_z) / 2,
    ])
    
    return centroids


def _build_bvh_recursive(
    segments: OrbitPolylineSegments,
    indices: npt.NDArray[np.int32],
    centroids: npt.NDArray[np.float64],
    nodes_min: list[npt.NDArray[np.float64]],
    nodes_max: list[npt.NDArray[np.float64]],
    left_child: list[int],
    right_child: list[int],
    first_prim: list[int],
    prim_count: list[int],
    leaf_indices_by_node: list[list[int]],
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
    
    # Compute AABB for this node
    aabb_min, aabb_max = _compute_aabb(segments, indices)
    nodes_min.append(aabb_min)
    nodes_max.append(aabb_max)
    
    # Check if we should create a leaf
    if len(indices) <= max_leaf_size:
        # Create leaf node
        left_child.append(-1)
        right_child.append(-1)
        # Defer packing of primitives until after the full tree is built.
        first_prim.append(-1)
        prim_count.append(len(indices))
        # Record exact indices for this leaf
        leaf_indices_by_node.append(indices.tolist())
        return node_idx
    
    # Find best split axis (largest extent)
    extent = aabb_max - aabb_min
    split_axis = np.argmax(extent)
    
    # Sort indices by centroid along split axis
    axis_centroids = centroids[indices, split_axis]
    sorted_order = np.argsort(axis_centroids)
    sorted_indices = indices[sorted_order]
    
    # Split at median
    mid = len(sorted_indices) // 2
    left_indices = sorted_indices[:mid]
    right_indices = sorted_indices[mid:]
    
    # Create internal node
    left_child.append(-1)  # Will be filled by recursive call
    right_child.append(-1)  # Will be filled by recursive call
    first_prim.append(-1)
    prim_count.append(0)
    leaf_indices_by_node.append([])  # Placeholder for internal node
    
    # Recursively build children
    left_idx = _build_bvh_recursive(
        segments, left_indices, centroids,
        nodes_min, nodes_max, left_child, right_child, first_prim, prim_count,
        leaf_indices_by_node,
        max_leaf_size,
    )
    right_idx = _build_bvh_recursive(
        segments, right_indices, centroids,
        nodes_min, nodes_max, left_child, right_child, first_prim, prim_count,
        leaf_indices_by_node,
        max_leaf_size,
    )
    
    # Update child pointers
    left_child[node_idx] = left_idx
    right_child[node_idx] = right_idx
    
    return node_idx


def build_bvh(
    segments: OrbitPolylineSegments,
    max_leaf_size: int = 8,
) -> BVHShard:
    """
    Build a bounding volume hierarchy over orbit segments.
    
    This function constructs a binary tree of axis-aligned bounding boxes
    using a median split strategy. The resulting BVH enables efficient
    ray-segment intersection queries.
    
    Parameters
    ----------
    segments : OrbitPolylineSegments
        Input segments with precomputed AABBs
    max_leaf_size : int, default=8
        Maximum number of primitives per leaf node
        
    Returns
    -------
    bvh : BVHShard
        Constructed BVH ready for queries
    """
    if len(segments) == 0:
        return BVHShard(
            nodes_min=np.empty((0, 3)),
            nodes_max=np.empty((0, 3)),
            left_child=np.empty(0, dtype=np.int32),
            right_child=np.empty(0, dtype=np.int32),
            first_prim=np.empty(0, dtype=np.int32),
            prim_count=np.empty(0, dtype=np.int32),
            prim_orbit_ids=[],
            prim_seg_ids=np.empty(0, dtype=np.int32),
            prim_row_index=np.empty(0, dtype=np.int32),
        )
    
    # Validate that AABBs are computed
    if np.any(np.isnan(segments.aabb_min_x.to_numpy())):
        raise ValueError("Segments must have computed AABBs. Call compute_segment_aabbs() first.")
    
    # Compute centroids for splitting
    centroids = _compute_centroids(segments)
    
    # Initialize node arrays
    nodes_min = []
    nodes_max = []
    left_child = []
    right_child = []
    first_prim = []
    prim_count = []
    leaf_indices_by_node: list[list[int]] = []
    
    # Build BVH recursively
    indices = np.arange(len(segments), dtype=np.int32)
    root_idx = _build_bvh_recursive(
        segments, indices, centroids,
        nodes_min, nodes_max, left_child, right_child, first_prim, prim_count,
        leaf_indices_by_node,
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
    
    # Pack leaf primitives contiguously
    packed_prim_indices: list[int] = []
    node_first_offsets = np.full(len(nodes_min_arr), -1, dtype=np.int32)
    node_counts = np.array(prim_count_arr, dtype=np.int32)

    offset = 0
    for node_idx in range(len(nodes_min_arr)):
        if left_child_arr[node_idx] == -1:  # leaf
            leaf_inds = leaf_indices_by_node[node_idx]
            if leaf_inds:
                node_first_offsets[node_idx] = offset
                packed_prim_indices.extend(leaf_inds)
                offset += len(leaf_inds)

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
    
    bvh = BVHShard(
        nodes_min=nodes_min_arr,
        nodes_max=nodes_max_arr,
        left_child=left_child_arr,
        right_child=right_child_arr,
        first_prim=first_prim_arr,
        prim_count=prim_count_arr,
        prim_orbit_ids=prim_orbit_ids,
        prim_seg_ids=prim_seg_ids,
        prim_row_index=prim_row_indices,
    )
    
    logger.info(f"Built BVH with {bvh.num_nodes} nodes over {bvh.num_primitives} segments")
    
    return bvh


def save_bvh(bvh: BVHShard, filepath: str) -> None:
    """
    Save a BVH to disk for later loading with memory mapping.
    
    This function saves all BVH arrays to a compressed .npz file that can be
    efficiently loaded and memory-mapped for read-only access by multiple workers.
    
    Parameters
    ----------
    bvh : BVHShard
        BVH to save
    filepath : str
        Path to save the BVH file (should end with .npz)
        
    Examples
    --------
    >>> bvh = build_bvh(segments)
    >>> save_bvh(bvh, "orbit_bvh.npz")
    """
    import os
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save all BVH arrays to compressed npz file
    np.savez_compressed(
        filepath,
        nodes_min=bvh.nodes_min,
        nodes_max=bvh.nodes_max,
        left_child=bvh.left_child,
        right_child=bvh.right_child,
        first_prim=bvh.first_prim,
        prim_count=bvh.prim_count,
        prim_orbit_ids=np.array(bvh.prim_orbit_ids, dtype='U'),  # Unicode string array
        prim_seg_ids=bvh.prim_seg_ids,
        prim_row_index=bvh.prim_row_index,
    )
    
    logger.info(f"Saved BVH with {bvh.num_nodes} nodes to {filepath}")


def load_bvh(filepath: str, mmap_mode: Optional[str] = 'r') -> BVHShard:
    """
    Load a BVH from disk with optional memory mapping.
    
    This function loads a BVH from a .npz file created by save_bvh().
    Memory mapping allows multiple processes to share the same BVH data
    without duplicating it in memory.
    
    Parameters
    ----------
    filepath : str
        Path to the BVH file (.npz)
    mmap_mode : str, optional
        Memory mapping mode for numpy arrays:
        - 'r': read-only (default, recommended for workers)
        - 'r+': read-write
        - None: load into memory (no memory mapping)
        
    Returns
    -------
    bvh : BVHShard
        Loaded BVH ready for queries
        
    Examples
    --------
    >>> bvh = load_bvh("orbit_bvh.npz", mmap_mode='r')  # Memory-mapped read-only
    >>> bvh = load_bvh("orbit_bvh.npz", mmap_mode=None)  # Load into memory
    """
    if mmap_mode is not None:
        # Load with memory mapping
        data = np.load(filepath, mmap_mode=mmap_mode)
    else:
        # Load into memory
        data = np.load(filepath)
    
    # Reconstruct BVH from saved arrays
    bvh = BVHShard(
        nodes_min=data['nodes_min'],
        nodes_max=data['nodes_max'],
        left_child=data['left_child'],
        right_child=data['right_child'],
        first_prim=data['first_prim'],
        prim_count=data['prim_count'],
        prim_orbit_ids=data['prim_orbit_ids'].tolist(),  # Convert back to list
        prim_seg_ids=data['prim_seg_ids'],
        prim_row_index=data['prim_row_index'],
    )
    
    logger.info(f"Loaded BVH with {bvh.num_nodes} nodes from {filepath} (mmap_mode={mmap_mode})")
    
    return bvh
