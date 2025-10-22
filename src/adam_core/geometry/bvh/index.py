"""
Build BVH indices from orbits.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt
import pyarrow.compute as pc
import quivr as qv
import ray
from numba import njit  # type: ignore

from ...orbits.orbits import Orbits
from ...orbits.polyline import (
    OrbitPolylineSegments,
    compute_segment_aabbs,
    sample_ellipse_adaptive,
)
from ...ray_cluster import initialize_use_ray
from ...utils.iter import _iterate_chunk_indices

__all__ = [
    "BVHNodes",
    "BVHPrimitives",
    "BVHIndex",
    "build_bvh_index_from_segments",
    "build_bvh_index",
    "BVHLeafPrimitives",
    "get_leaf_primitives_numpy",
    "build_bvh_nodes_from_aabbs",
]

logger = logging.getLogger(__name__)


# =====================
# Common bit helpers
# =====================


def _part1by2_vec(v: np.ndarray) -> np.ndarray:
    """Vectorized interleave of 21-bit integers (uint64) to Morton bits."""
    v = v.astype(np.uint64, copy=False)
    v = (v | (v << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    v = (v | (v << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    v = (v | (v << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    v = (v | (v << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    v = (v | (v << np.uint64(2))) & np.uint64(0x1249249249249249)
    return v


class BVHNodes(qv.Table):
    """
    Quivr table representing BVH nodes for a shard.
    """

    nodes_min_x = qv.Float32Column()
    nodes_min_y = qv.Float32Column()
    nodes_min_z = qv.Float32Column()
    nodes_max_x = qv.Float32Column()
    nodes_max_y = qv.Float32Column()
    nodes_max_z = qv.Float32Column()

    left_child = qv.Int32Column()
    right_child = qv.Int32Column()

    first_prim = qv.Int32Column()
    prim_count = qv.Int32Column()

    # Attributes
    shard_id = qv.StringAttribute(default="")
    version = qv.StringAttribute(default="1.0.0")
    float_dtype = qv.StringAttribute(default="float32")
    # Provenance: build settings
    build_max_leaf_size = qv.IntAttribute(default=0)
    # Structural metrics
    bvh_max_depth = qv.IntAttribute(default=0)
    # AABB provenance (moved from segments)
    aabb_guard_arcmin = qv.FloatAttribute(default=0.0)
    aabb_epsilon_n_au = qv.FloatAttribute(default=0.0)
    aabb_padding_method = qv.StringAttribute(default="baseline")

    # Convenience accessors for NumPy kernels
    def min_max_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (nodes_min, nodes_max) stacked as (N,3) NumPy arrays."""
        nodes_min = np.column_stack(
            [
                np.asarray(self.nodes_min_x),
                np.asarray(self.nodes_min_y),
                np.asarray(self.nodes_min_z),
            ]
        )
        nodes_max = np.column_stack(
            [
                np.asarray(self.nodes_max_x),
                np.asarray(self.nodes_max_y),
                np.asarray(self.nodes_max_z),
            ]
        )
        return nodes_min, nodes_max

    def is_leaf_numpy(self) -> np.ndarray:
        """Return boolean array indicating leaf nodes (left_child == -1)."""
        left = np.asarray(self.left_child, dtype=np.int32)
        right = np.asarray(self.right_child, dtype=np.int32)
        is_leaf = left == -1
        if np.any(is_leaf):
            # Integrity check for well-formed BVH leaves
            assert np.all(right[is_leaf] == -1)
        return is_leaf


class BVHPrimitives(qv.Table):
    """
    Quivr table representing BVH primitive arrays for a shard.
    """

    segment_row_index = qv.Int32Column()
    prim_seg_ids = qv.Int32Column()

    # Attributes
    shard_id = qv.StringAttribute(default="")
    version = qv.StringAttribute(default="1.0.0")
    # Provenance: build settings (mirrored)
    build_max_leaf_size = qv.IntAttribute(default=0)


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
        from ...orbits.polyline import OrbitPolylineSegments

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
        os.makedirs(directory, exist_ok=True)
        seg_path = f"{directory.rstrip('/')}/segments.parquet"
        nodes_path = f"{directory.rstrip('/')}/bvh_nodes.parquet"
        prims_path = f"{directory.rstrip('/')}/bvh_prims.parquet"

        self.segments.to_parquet(seg_path)
        self.nodes.to_parquet(nodes_path)
        self.prims.to_parquet(prims_path)

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
    max_chord_arcmin: float,
    max_segments_per_orbit: int,
) -> OrbitPolylineSegments:
    # Use a conservative default for max segments per orbit to balance fidelity and cost
    orbits = qv.defragment(orbits)
    _, segs = sample_ellipse_adaptive(
        orbits,
        max_chord_arcmin=max_chord_arcmin,
        max_segments_per_orbit=max_segments_per_orbit,
    )
    # Defer AABB computation to BVH build (segments only store endpoints)
    return segs


@ray.remote
def orbits_segments_worker_remote(
    orbits: Orbits,
    max_chord_arcmin: float,
    max_segments_per_orbit: int,
) -> OrbitPolylineSegments:
    return orbits_to_segments_worker(orbits, max_chord_arcmin=max_chord_arcmin, max_segments_per_orbit=max_segments_per_orbit)


def build_bvh_index(
    orbits: Orbits,
    *,
    max_chord_arcmin: float = 5.0,
    guard_arcmin: float = 0.65,
    max_leaf_size: int = 64,
    chunk_size_orbits: int = 1000,
    max_processes: int = 1,
    max_segments_per_orbit: int = 512,
    epsilon_n_au: float = 1e-6,
    padding_method: str = "baseline",
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
                max_segments_per_orbit=max_segments_per_orbit,
            )
            segments_chunks.append(segs)
    else:
        initialize_use_ray(num_cpus=max_processes)
        futures: list[ray.ObjectRef] = []
        for start, end in chunk_definitions:
            future = orbits_segments_worker_remote.remote(
                orbits[start:end],
                max_chord_arcmin=max_chord_arcmin,
                max_segments_per_orbit=max_segments_per_orbit,
            )
            futures.append(future)
            if len(futures) >= max_processes * 1.5:
                finished, futures = ray.wait(futures, num_returns=1)
                segments_chunks.append(ray.get(finished[0]))
        while futures:
            finished, futures = ray.wait(futures, num_returns=1)
            segments_chunks.append(ray.get(finished[0]))

    # Concatenate all segments (defragment into a single contiguous table)
    segments_all = qv.concatenate(segments_chunks, defrag=True)

    # Build BVH over all segments
    index = build_bvh_index_from_segments(
        segments_all,
        max_leaf_size=max_leaf_size,
        guard_arcmin=guard_arcmin,
        epsilon_n_au=epsilon_n_au,
        padding_method=padding_method,
        max_processes=max_processes,
    )
    return index


def build_bvh_index_from_segments(
    segments: "OrbitPolylineSegments",
    max_leaf_size: int = 8,
    *,
    guard_arcmin: float = 1.0,
    epsilon_n_au: float = 1e-6,
    padding_method: str = "baseline",
    max_processes: int | None = 1,
) -> BVHIndex:
    """
    Build a monolithic BVHIndex from polyline segments.
    """
    # Internal array-based builder
    # Compute AABBs from segment endpoints
    min_x, min_y, min_z, max_x, max_y, max_z = compute_segment_aabbs(
        segments,
        guard_arcmin=guard_arcmin,
        epsilon_n_au=epsilon_n_au,
        padding_method=padding_method,
        max_processes=max_processes,
    )

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
    ) = _build_bvh_arrays_lbvh(
        segments,
        min_x.astype(np.float32, copy=False),
        min_y.astype(np.float32, copy=False),
        min_z.astype(np.float32, copy=False),
        max_x.astype(np.float32, copy=False),
        max_y.astype(np.float32, copy=False),
        max_z.astype(np.float32, copy=False),
        max_leaf_size=max_leaf_size,
    )

    # Compute max depth robustly (root discovery + cycle guard)
    if len(left_child_arr) == 0:
        max_depth_val = 0
    else:
        tot_nodes = int(len(left_child_arr))
        parent = np.full((tot_nodes,), -1, dtype=np.int32)
        for i in range(tot_nodes):
            lc = int(left_child_arr[i])
            rc = int(right_child_arr[i])
            if lc >= 0:
                parent[lc] = i
            if rc >= 0:
                parent[rc] = i
        roots = np.nonzero(parent == -1)[0]
        root_idx = int(roots[0]) if roots.size > 0 else 0
        max_depth_val = 1
        visited = np.zeros((tot_nodes,), dtype=np.bool_)
        stack: list[tuple[int, int]] = [(root_idx, 1)]
        while stack:
            node, depth = stack.pop()
            if visited[node]:
                # Cycle detected; bail out with conservative depth
                logger.warning(
                    "BVH max depth traversal detected a cycle; aborting depth scan"
                )
                break
            visited[node] = True
            lc = int(left_child_arr[node])
            rc = int(right_child_arr[node])
            if lc >= 0:
                stack.append((lc, depth + 1))
            if rc >= 0:
                stack.append((rc, depth + 1))
            if lc < 0 and rc < 0 and depth > max_depth_val:
                max_depth_val = depth

    nodes = BVHNodes.from_kwargs(
        nodes_min_x=(
            nodes_min_arr[:, 0]
            if len(nodes_min_arr)
            else np.array([], dtype=np.float32)
        ),
        nodes_min_y=(
            nodes_min_arr[:, 1]
            if len(nodes_min_arr)
            else np.array([], dtype=np.float32)
        ),
        nodes_min_z=(
            nodes_min_arr[:, 2]
            if len(nodes_min_arr)
            else np.array([], dtype=np.float32)
        ),
        nodes_max_x=(
            nodes_max_arr[:, 0]
            if len(nodes_max_arr)
            else np.array([], dtype=np.float32)
        ),
        nodes_max_y=(
            nodes_max_arr[:, 1]
            if len(nodes_max_arr)
            else np.array([], dtype=np.float32)
        ),
        nodes_max_z=(
            nodes_max_arr[:, 2]
            if len(nodes_max_arr)
            else np.array([], dtype=np.float32)
        ),
        left_child=left_child_arr,
        right_child=right_child_arr,
        first_prim=first_prim_arr,
        prim_count=prim_count_arr,
        shard_id="index",
        build_max_leaf_size=int(max_leaf_size),
        bvh_max_depth=int(max_depth_val),
        aabb_guard_arcmin=float(guard_arcmin),
        aabb_epsilon_n_au=float(epsilon_n_au),
        aabb_padding_method=str(padding_method),
    )

    prims = BVHPrimitives.from_kwargs(
        segment_row_index=prim_row_indices,
        prim_seg_ids=prim_seg_ids,
        shard_id="index",
        build_max_leaf_size=int(max_leaf_size),
    )

    return BVHIndex(segments=segments, nodes=nodes, prims=prims)


class BVHLeafPrimitives(qv.Table):
    """View of primitives referenced by one or more leaf nodes."""

    leaf_id = qv.Int32Column()
    segment_row_index = qv.Int32Column()
    prim_seg_ids = qv.Int32Column()


def get_leaf_primitives_numpy(
    nodes: BVHNodes,
    prims: BVHPrimitives,
    leaf_node_indices: int | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gather primitive row indices and seg IDs for one or many leaf nodes.

    Returns (row_indices, seg_ids, leaf_ids) as NumPy arrays.
    """
    first = np.asarray(nodes.first_prim, dtype=np.int32)
    count = np.asarray(nodes.prim_count, dtype=np.int32)
    seg_row = np.asarray(prims.segment_row_index, dtype=np.int32)
    seg_ids = np.asarray(prims.prim_seg_ids, dtype=np.int32)

    leaf_idx = (
        np.array([leaf_node_indices], dtype=np.int32)
        if np.isscalar(leaf_node_indices)
        else np.asarray(leaf_node_indices, dtype=np.int32)
    )
    # Only nodes with positive count produce primitives
    mask = count[leaf_idx] > 0
    if not np.any(mask):
        empty = np.array([], dtype=np.int32)
        return empty, empty, empty

    sel_nodes = leaf_idx[mask]
    starts = first[sel_nodes].astype(np.int64)
    counts = count[sel_nodes].astype(np.int64)
    total = int(counts.sum())

    out_rows = np.empty(total, dtype=np.int32)
    out_segs = np.empty(total, dtype=np.int32)
    out_leaf = np.empty(total, dtype=np.int32)

    pos = 0
    for n, s, c in zip(sel_nodes, starts, counts):
        rng = slice(s, s + c)
        span = c
        out_rows[pos : pos + span] = seg_row[rng]
        out_segs[pos : pos + span] = seg_ids[rng]
        out_leaf[pos : pos + span] = int(n)
        pos += span

    return out_rows, out_segs, out_leaf


@njit(cache=True, fastmath=False)  # type: ignore
def _fused_aabb_from_arrays_range(
    indices: npt.NDArray[np.int32],
    start: int,
    end: int,
    min_x: npt.NDArray[np.float32],
    min_y: npt.NDArray[np.float32],
    min_z: npt.NDArray[np.float32],
    max_x: npt.NDArray[np.float32],
    max_y: npt.NDArray[np.float32],
    max_z: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Compute axis-aligned bounding box for a set of segments using pre-extracted arrays.
    """
    mn0 = np.float32(np.inf)
    mn1 = np.float32(np.inf)
    mn2 = np.float32(np.inf)
    mx0 = np.float32(-np.inf)
    mx1 = np.float32(-np.inf)
    mx2 = np.float32(-np.inf)
    for i in range(start, end):
        idx = int(indices[i])
        v00 = min_x[idx]
        v01 = min_y[idx]
        v02 = min_z[idx]
        v10 = max_x[idx]
        v11 = max_y[idx]
        v12 = max_z[idx]
        if v00 < mn0:
            mn0 = v00
        if v01 < mn1:
            mn1 = v01
        if v02 < mn2:
            mn2 = v02
        if v10 > mx0:
            mx0 = v10
        if v11 > mx1:
            mx1 = v11
        if v12 > mx2:
            mx2 = v12
    return np.array([mn0, mn1, mn2], dtype=np.float32), np.array(
        [mx0, mx1, mx2], dtype=np.float32
    )


# =====================
# LBVH Numba helpers
# =====================


@njit(cache=True, fastmath=False)
def _clz64_numba(x: np.uint64) -> int:
    if x == 0:
        return 64
    n = 0
    m = x
    while (m & np.uint64(1 << 63)) == 0:
        n += 1
        m = m << np.uint64(1)
    return n


@njit(cache=True, fastmath=False)
def _mark_collapsible_subtrees(
    counts: npt.NDArray[np.int32],
    left: npt.NDArray[np.int32],
    right: npt.NDArray[np.int32],
    max_leaf_size: int,
    n_leaves: int,
) -> npt.NDArray[np.uint8]:
    """
    Mark roots of collapsible subtrees (count <= max_leaf_size).
    Returns mask over all nodes [0, 2n-2].
    """
    tot = 2 * n_leaves - 1
    n_internal = n_leaves - 1
    is_root_collapsible = np.zeros((tot,), dtype=np.uint8)

    # DFS from root (0); mark first collapsible encountered in each branch
    stack = [0]
    while len(stack) > 0:
        u = stack.pop()
        if counts[u] <= max_leaf_size:
            is_root_collapsible[u] = 1
            # Don't traverse below; descendants are not roots
        elif u < n_internal:
            # Traverse children
            if right[u] >= 0:
                stack.append(int(right[u]))
            if left[u] >= 0:
                stack.append(int(left[u]))

    return is_root_collapsible


@njit(cache=True, fastmath=False)
def _radix_sort_u64_stable(
    codes: npt.NDArray[np.uint64],
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int64]]:
    """
    Stable LSD radix sort of uint64 Morton codes.
    Returns (sorted_codes, order) where sorted_codes[i] = codes[order[i]].
    """
    n = codes.shape[0]
    if n == 0:
        return codes.copy(), np.empty((0,), dtype=np.int64)

    order = np.arange(n, dtype=np.int64)
    tmp_order = np.empty_like(order)
    codes_work = codes.copy()
    codes_tmp = np.empty_like(codes_work)

    # 8 passes of 8 bits each (little endian)
    for shift in range(0, 64, 8):
        # Count occurrences
        counts = np.zeros(256, dtype=np.int64)
        for i in range(n):
            bucket = int((codes_work[i] >> np.uint64(shift)) & np.uint64(0xFF))
            counts[bucket] += 1

        # Prefix sums to positions
        pos = np.empty(256, dtype=np.int64)
        running = np.int64(0)
        for b in range(256):
            pos[b] = running
            running += counts[b]

        # Stable scatter
        for i in range(n):
            bucket = int((codes_work[i] >> np.uint64(shift)) & np.uint64(0xFF))
            dst = pos[bucket]
            codes_tmp[dst] = codes_work[i]
            tmp_order[dst] = order[i]
            pos[bucket] = dst + 1

        # Swap buffers
        codes_work, codes_tmp = codes_tmp, codes_work
        order, tmp_order = tmp_order, order

    return codes_work, order


@njit(cache=True, fastmath=False)
def _lcp_numba(codes: np.ndarray, i: int, j: int) -> int:
    n = codes.shape[0]
    if j < 0 or j >= n:
        return -1
    ci = np.uint64(codes[i])
    cj = np.uint64(codes[j])
    if ci == cj:
        # Tie-break identical codes by index distance
        return 64 + _clz64_numba(np.uint64(i ^ j))
    x = np.uint64(ci ^ cj)
    return _clz64_numba(x)


@njit(cache=True, fastmath=False)
def _precompute_neighbor_lcps(
    codes: npt.NDArray[np.uint64],
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """
    Precompute LCP with immediate neighbors for each index i:
    lcp_prev[i] = lcp(i, i-1), lcp_next[i] = lcp(i, i+1).
    Out-of-bounds entries are set to -1.
    """
    n = codes.shape[0]
    lcp_prev = np.full((n,), -1, dtype=np.int32)
    lcp_next = np.full((n,), -1, dtype=np.int32)
    for i in range(n):
        # i-1
        if i - 1 >= 0:
            lcp_prev[i] = _lcp_numba(codes, i, i - 1)
        # i+1
        if i + 1 < n:
            lcp_next[i] = _lcp_numba(codes, i, i + 1)
    return lcp_prev, lcp_next


@njit(cache=True, fastmath=False)
def _lbvh_link_karras_with_neighbors(
    codes: npt.NDArray[np.uint64],
    lcp_prev: npt.NDArray[np.int32],
    lcp_next: npt.NDArray[np.int32],
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    n = codes.shape[0]
    left = np.empty((n - 1,), dtype=np.int32)
    right = np.empty((n - 1,), dtype=np.int32)

    def find_split(first: int, last: int) -> int:
        firstCode = np.uint64(codes[first])
        lastCode = np.uint64(codes[last])
        if firstCode == lastCode:
            return (first + last) // 2
        commonPrefix = _clz64_numba(np.uint64(firstCode ^ lastCode))
        split = first
        step = last - first
        while step > 1:
            step = (step + 1) // 2
            newSplit = split + step
            if newSplit < last:
                splitPrefix = _clz64_numba(
                    np.uint64(firstCode ^ np.uint64(codes[newSplit]))
                )
                if splitPrefix > commonPrefix:
                    split = newSplit
        return split

    for i in range(n - 1):
        lcp_l = lcp_prev[i]
        lcp_r = lcp_next[i]
        d = 1 if lcp_r >= lcp_l else -1
        delta_min = lcp_prev[i] if d == 1 else lcp_next[i]
        lmax = 2
        while True:
            j = i + lmax * d
            if j < 0 or j > n - 1:
                break
            if _lcp_numba(codes, i, j) > delta_min:
                lmax *= 2
            else:
                break
        l = 0
        step = lmax
        while step > 1:
            step //= 2
            j2 = i + (l + step) * d
            if not (0 <= j2 <= n - 1):
                continue
            if _lcp_numba(codes, i, j2) > delta_min:
                l += step
        j = i + l * d
        if j < 0:
            j = 0
        if j > n - 1:
            j = n - 1
        first = j if j < i else i
        last = i if j < i else j
        split = find_split(first, last)
        left[i] = (n - 1 + first) if split == first else split
        right[i] = (n - 1 + last) if (split + 1) == last else (split + 1)
    return left, right


@njit(cache=True, fastmath=False)
def _compute_bounds_lbvh_numba(
    left_child: np.ndarray,
    right_child: np.ndarray,
    min_x: np.ndarray,
    min_y: np.ndarray,
    min_z: np.ndarray,
    max_x: np.ndarray,
    max_y: np.ndarray,
    max_z: np.ndarray,
    packed_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute AABBs in postorder to ensure children are processed before parents.
    Leaves are at indices [n-1, 2n-2] in node arrays and map to packed_indices.
    """
    n = packed_indices.shape[0]
    tot = 2 * n - 1
    base = n - 1

    nodes_min = np.zeros((tot, 3), dtype=np.float32)
    nodes_max = np.zeros((tot, 3), dtype=np.float32)

    # Initialize leaves from packed segment indices
    for i in range(n):
        idx = int(packed_indices[i])
        dst = base + i
        nodes_min[dst, 0] = min_x[idx]
        nodes_min[dst, 1] = min_y[idx]
        nodes_min[dst, 2] = min_z[idx]
        nodes_max[dst, 0] = max_x[idx]
        nodes_max[dst, 1] = max_y[idx]
        nodes_max[dst, 2] = max_z[idx]

    # Build parent array and find root among internal nodes [0, n-2]
    parent = np.full((tot,), -1, dtype=np.int32)
    for i in range(n - 1):
        lc = int(left_child[i])
        rc = int(right_child[i])
        if lc >= 0:
            parent[lc] = i
        if rc >= 0:
            parent[rc] = i
    root = 0
    for i in range(n - 1):
        if parent[i] == -1:
            root = i
            break

    # Postorder traversal using explicit stack with state
    stack_nodes = np.empty((tot * 2,), dtype=np.int32)
    stack_state = np.empty((tot * 2,), dtype=np.int8)
    sp = 0
    stack_nodes[sp] = root
    stack_state[sp] = 0
    sp += 1

    while sp > 0:
        sp -= 1
        u = int(stack_nodes[sp])
        state = int(stack_state[sp])

        if state == 0 and u < (n - 1):
            # push self for processing after children
            stack_nodes[sp] = u
            stack_state[sp] = 1
            sp += 1
            # push children
            rc = int(right_child[u])
            lc = int(left_child[u])
            if rc >= 0:
                stack_nodes[sp] = rc
                stack_state[sp] = 0
                sp += 1
            if lc >= 0:
                stack_nodes[sp] = lc
                stack_state[sp] = 0
                sp += 1
            continue

        # Process node u (either leaf or internal after children)
        if u < (n - 1):
            lc = int(left_child[u])
            rc = int(right_child[u])
            # union of children
            a0 = nodes_min[lc, 0]
            a1 = nodes_min[lc, 1]
            a2 = nodes_min[lc, 2]
            b0 = nodes_min[rc, 0]
            b1 = nodes_min[rc, 1]
            b2 = nodes_min[rc, 2]
            c0 = nodes_max[lc, 0]
            c1 = nodes_max[lc, 1]
            c2 = nodes_max[lc, 2]
            d0 = nodes_max[rc, 0]
            d1 = nodes_max[rc, 1]
            d2 = nodes_max[rc, 2]
            nodes_min[u, 0] = a0 if a0 < b0 else b0
            nodes_min[u, 1] = a1 if a1 < b1 else b1
            nodes_min[u, 2] = a2 if a2 < b2 else b2
            nodes_max[u, 0] = c0 if c0 > d0 else d0
            nodes_max[u, 1] = c1 if c1 > d1 else d1
            nodes_max[u, 2] = c2 if c2 > d2 else d2

    return nodes_min, nodes_max


@njit(cache=True, fastmath=False)
def _compute_leaf_ranges_numba(
    left_child: npt.NDArray[np.int32],
    right_child: npt.NDArray[np.int32],
    n_leaves: int,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """
    Compute [first_leaf_idx, last_leaf_idx] for every node (internal+leaf)
    using a postorder traversal to honor arbitrary internal indexing.
    Leaf index is in [0, n_leaves-1].
    """
    tot = 2 * n_leaves - 1
    base = n_leaves - 1
    first_idx = np.full((tot,), -1, dtype=np.int32)
    last_idx = np.full((tot,), -1, dtype=np.int32)

    # Initialize leaf ranges
    for i in range(n_leaves):
        node = base + i
        first_idx[node] = i
        last_idx[node] = i

    # Build parent to find root
    parent = np.full((tot,), -1, dtype=np.int32)
    for u in range(n_leaves - 1):
        lc = int(left_child[u])
        rc = int(right_child[u])
        if lc >= 0:
            parent[lc] = u
        if rc >= 0:
            parent[rc] = u
    root = 0
    for u in range(n_leaves - 1):
        if parent[u] == -1:
            root = u
            break

    # Postorder traversal with explicit stack
    stack_nodes = np.empty((tot * 2,), dtype=np.int32)
    stack_state = np.empty((tot * 2,), dtype=np.int8)
    sp = 0
    stack_nodes[sp] = root
    stack_state[sp] = 0
    sp += 1

    while sp > 0:
        sp -= 1
        u = int(stack_nodes[sp])
        state = int(stack_state[sp])
        if state == 0 and u < (n_leaves - 1):
            # push self after children
            stack_nodes[sp] = u
            stack_state[sp] = 1
            sp += 1
            rc = int(right_child[u])
            lc = int(left_child[u])
            if rc >= 0:
                stack_nodes[sp] = rc
                stack_state[sp] = 0
                sp += 1
            if lc >= 0:
                stack_nodes[sp] = lc
                stack_state[sp] = 0
                sp += 1
            continue

        if u < (n_leaves - 1):
            lc = int(left_child[u])
            rc = int(right_child[u])
            f0 = first_idx[lc]
            f1 = first_idx[rc]
            l0 = last_idx[lc]
            l1 = last_idx[rc]
            # min/max over children (both must be set by now)
            first_idx[u] = f0 if (f0 <= f1) else f1
            last_idx[u] = l0 if (l0 >= l1) else l1

    return first_idx, last_idx


@njit(cache=True, fastmath=False)
def _bounds_init_group_leaves_numba(
    min_x: np.ndarray,
    min_y: np.ndarray,
    min_z: np.ndarray,
    max_x: np.ndarray,
    max_y: np.ndarray,
    max_z: np.ndarray,
    sorted_ids: np.ndarray,
    group_starts: np.ndarray,
    group_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize leaf nodes (grouped) AABBs for compacted LBVH.
    Returns (nodes_min, nodes_max) with only leaves written; internals left zeroed.
    """
    n = group_starts.shape[0]
    tot = 2 * n - 1
    nodes_min = np.zeros((tot, 3), dtype=np.float32)
    nodes_max = np.zeros((tot, 3), dtype=np.float32)
    base = n - 1
    for i in range(n):
        gstart = int(group_starts[i])
        gcount = int(group_counts[i])
        mn0 = np.float32(np.inf)
        mn1 = np.float32(np.inf)
        mn2 = np.float32(np.inf)
        mx0 = np.float32(-np.inf)
        mx1 = np.float32(-np.inf)
        mx2 = np.float32(-np.inf)
        for j in range(gcount):
            idx = int(sorted_ids[gstart + j])
            v00 = min_x[idx]
            v01 = min_y[idx]
            v02 = min_z[idx]
            v10 = max_x[idx]
            v11 = max_y[idx]
            v12 = max_z[idx]
            if v00 < mn0:
                mn0 = v00
            if v01 < mn1:
                mn1 = v01
            if v02 < mn2:
                mn2 = v02
            if v10 > mx0:
                mx0 = v10
            if v11 > mx1:
                mx1 = v11
            if v12 > mx2:
                mx2 = v12
        nodes_min[base + i, 0] = mn0
        nodes_min[base + i, 1] = mn1
        nodes_min[base + i, 2] = mn2
        nodes_max[base + i, 0] = mx0
        nodes_max[base + i, 1] = mx1
        nodes_max[base + i, 2] = mx2
    return nodes_min, nodes_max


@njit(cache=True, fastmath=False)
def _bounds_union_internals_postorder_numba(
    left_child: np.ndarray,
    right_child: np.ndarray,
    n_leaves: int,
    nodes_min: np.ndarray,
    nodes_max: np.ndarray,
) -> None:
    """Postorder union of internal nodes; updates nodes_min/max in place."""
    tot = 2 * n_leaves - 1
    parent = np.full((tot,), -1, dtype=np.int32)
    for i in range(n_leaves - 1):
        lc = int(left_child[i])
        rc = int(right_child[i])
        if lc >= 0:
            parent[lc] = i
        if rc >= 0:
            parent[rc] = i
    root = 0
    for i in range(n_leaves - 1):
        if parent[i] == -1:
            root = i
            break
    stack_nodes = np.empty((tot * 2,), dtype=np.int32)
    stack_state = np.empty((tot * 2,), dtype=np.int8)
    sp = 0
    stack_nodes[sp] = root
    stack_state[sp] = 0
    sp += 1
    while sp > 0:
        sp -= 1
        u = int(stack_nodes[sp])
        state = int(stack_state[sp])
        if state == 0 and u < (n_leaves - 1):
            stack_nodes[sp] = u
            stack_state[sp] = 1
            sp += 1
            rc = int(right_child[u])
            lc = int(left_child[u])
            if rc >= 0:
                stack_nodes[sp] = rc
                stack_state[sp] = 0
                sp += 1
            if lc >= 0:
                stack_nodes[sp] = lc
                stack_state[sp] = 0
                sp += 1
            continue
        if u < (n_leaves - 1):
            lc = int(left_child[u])
            rc = int(right_child[u])
            a0 = nodes_min[lc, 0]
            a1 = nodes_min[lc, 1]
            a2 = nodes_min[lc, 2]
            b0 = nodes_min[rc, 0]
            b1 = nodes_min[rc, 1]
            b2 = nodes_min[rc, 2]
            c0 = nodes_max[lc, 0]
            c1 = nodes_max[lc, 1]
            c2 = nodes_max[lc, 2]
            d0 = nodes_max[rc, 0]
            d1 = nodes_max[rc, 1]
            d2 = nodes_max[rc, 2]
            nodes_min[u, 0] = a0 if a0 < b0 else b0
            nodes_min[u, 1] = a1 if a1 < b1 else b1
            nodes_min[u, 2] = a2 if a2 < b2 else b2
            nodes_max[u, 0] = c0 if c0 > d0 else d0
            nodes_max[u, 1] = c1 if c1 > d1 else d1
            nodes_max[u, 2] = c2 if c2 > d2 else d2


def _compute_bounds_lbvh_groups(
    left_child: np.ndarray,
    right_child: np.ndarray,
    min_x: np.ndarray,
    min_y: np.ndarray,
    min_z: np.ndarray,
    max_x: np.ndarray,
    max_y: np.ndarray,
    max_z: np.ndarray,
    sorted_ids: np.ndarray,
    group_starts: np.ndarray,
    group_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    nodes_min, nodes_max = _bounds_init_group_leaves_numba(
        min_x, min_y, min_z, max_x, max_y, max_z, sorted_ids, group_starts, group_counts
    )
    _bounds_union_internals_postorder_numba(
        left_child, right_child, group_starts.shape[0], nodes_min, nodes_max
    )
    return nodes_min, nodes_max


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
    centroid_x: npt.NDArray[np.float64],
    centroid_y: npt.NDArray[np.float64],
    centroid_z: npt.NDArray[np.float64],
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
    centroid_x, centroid_y, centroid_z : np.ndarray
        Precomputed centroid components for all segments
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
    aabb_min, aabb_max = _fused_aabb_from_arrays_range(
        indices, start, end, min_x, min_y, min_z, max_x, max_y, max_z
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
        axis_centroids = centroid_x
    elif split_axis == 1:
        axis_centroids = centroid_y
    else:
        axis_centroids = centroid_z

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
        centroid_x,
        centroid_y,
        centroid_z,
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
        centroid_x,
        centroid_y,
        centroid_z,
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


def _build_bvh_arrays_lbvh(
    segments: OrbitPolylineSegments,
    min_x: npt.NDArray[np.float32],
    min_y: npt.NDArray[np.float32],
    min_z: npt.NDArray[np.float32],
    max_x: npt.NDArray[np.float32],
    max_y: npt.NDArray[np.float32],
    max_z: npt.NDArray[np.float32],
    *,
    max_leaf_size: int = 8,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    list[str],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """
    LBVH builder (Karras 2012): Morton codes + O(N) linking, per-primitive leaves.
    """
    N = int(len(segments))
    if N == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            [],
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    t0 = time.perf_counter()
    # 1) Centroids and normalization
    cx = (min_x + max_x) * np.float32(0.5)
    cy = (min_y + max_y) * np.float32(0.5)
    cz = (min_z + max_z) * np.float32(0.5)
    cmin0 = float(cx.min()) if cx.size else 0.0
    cmin1 = float(cy.min()) if cy.size else 0.0
    cmin2 = float(cz.min()) if cz.size else 0.0
    cmax0 = float(cx.max()) if cx.size else 1.0
    cmax1 = float(cy.max()) if cy.size else 1.0
    cmax2 = float(cz.max()) if cz.size else 1.0
    ext0 = cmax0 - cmin0
    ext1 = cmax1 - cmin1
    ext2 = cmax2 - cmin2
    if ext0 == 0.0:
        ext0 = 1.0
    if ext1 == 0.0:
        ext1 = 1.0
    if ext2 == 0.0:
        ext2 = 1.0
    nx = (cx - np.float32(cmin0)) / np.float32(ext0)
    ny = (cy - np.float32(cmin1)) / np.float32(ext1)
    nz = (cz - np.float32(cmin2)) / np.float32(ext2)
    nx = np.clip(nx, 0.0, 1.0).astype(np.float32, copy=False)
    ny = np.clip(ny, 0.0, 1.0).astype(np.float32, copy=False)
    nz = np.clip(nz, 0.0, 1.0).astype(np.float32, copy=False)

    # 2) Morton codes (21 bits per axis -> 63-bit code) - vectorized
    bits = 21
    scale = np.float32((1 << bits) - 1)
    ix = (nx * scale).astype(np.int64, copy=False)
    iy = (ny * scale).astype(np.int64, copy=False)
    iz = (nz * scale).astype(np.int64, copy=False)
    ix = np.clip(ix, 0, (1 << bits) - 1).astype(np.uint64, copy=False)
    iy = np.clip(iy, 0, (1 << bits) - 1).astype(np.uint64, copy=False)
    iz = np.clip(iz, 0, (1 << bits) - 1).astype(np.uint64, copy=False)

    xx = _part1by2_vec(ix)
    yy = _part1by2_vec(iy) << np.uint64(1)
    zz = _part1by2_vec(iz) << np.uint64(2)
    codes = xx | yy | zz

    # 3) Stable sort by code using radix sort
    sorted_codes, order = _radix_sort_u64_stable(codes)
    ids = np.arange(N, dtype=np.int32)
    sorted_ids = ids[order.astype(np.int32)]

    t_codesort = time.perf_counter()
    logger.info(f"LBVH: N={N} codes+sort {t_codesort - t0:.3f}s")
    # 4) Karras internal node linking for N leaves, N-1 internals (Numba)
    tot_nodes = 2 * N - 1

    # Precompute neighbor LCPs to reduce repeated work in linker
    lcp_prev, lcp_next = _precompute_neighbor_lcps(sorted_codes)
    left_internal, right_internal = _lbvh_link_karras_with_neighbors(
        sorted_codes, lcp_prev, lcp_next
    )
    t_link = time.perf_counter()
    logger.info(f"LBVH: link {t_link - t_codesort:.3f}s")

    # 5) Build initial per-primitive leaf tree arrays
    left_child = np.empty((tot_nodes,), dtype=np.int32)
    right_child = np.empty((tot_nodes,), dtype=np.int32)
    left_child[: N - 1] = left_internal
    right_child[: N - 1] = right_internal
    left_child[N - 1 :] = -1
    right_child[N - 1 :] = -1

    # Primitive packing in sorted order (global primitive storage)
    packed_indices = sorted_ids.copy()
    prim_row_indices = packed_indices.copy()
    prim_seg_ids_all = segments.seg_id.to_numpy()
    prim_seg_ids = prim_seg_ids_all[packed_indices]
    prim_orbit_ids_all = segments.orbit_id.to_pylist()
    prim_orbit_ids = [prim_orbit_ids_all[i] for i in packed_indices]

    # Leaf compaction: collapse subtrees with prim_count <= max_leaf_size
    if max_leaf_size > 1:
        # Compute leaf index ranges per node and derive subtree primitive counts from ranges
        first_idx, last_idx = _compute_leaf_ranges_numba(
            left_child[: N - 1], right_child[: N - 1], N
        )
        tot_old = 2 * N - 1
        base_old = N - 1
        counts = np.zeros((tot_old,), dtype=np.int32)
        for i in range(N):
            counts[base_old + i] = 1
        for u in range(N - 2, -1, -1):
            f = int(first_idx[u])
            l = int(last_idx[u])
            counts[u] = l - f + 1

        collapsible = _mark_collapsible_subtrees(
            counts, left_child[: N - 1], right_child[: N - 1], max_leaf_size, N
        )

        # Count new leaves via DFS using collapsible mask
        def count_new_leaves(u: int) -> int:
            if u >= base_old:
                return 1
            if collapsible[u] == 1:
                return 1
            return count_new_leaves(int(left_child[u])) + count_new_leaves(
                int(right_child[u])
            )

        L = count_new_leaves(0)
        tot_new = 2 * L - 1

        # Allocate new tree arrays
        left_new = np.full((tot_new,), -1, dtype=np.int32)
        right_new = np.full((tot_new,), -1, dtype=np.int32)
        first_prim = np.full((tot_new,), -1, dtype=np.int32)
        prim_count = np.zeros((tot_new,), dtype=np.int32)

        group_starts = np.empty((L,), dtype=np.int32)
        group_counts = np.empty((L,), dtype=np.int32)

        # Indices
        next_internal = 0
        next_leaf = 0
        base_new = L - 1

        def build_new(u: int) -> int:
            nonlocal next_internal, next_leaf
            if u >= base_old:
                # Old leaf -> new leaf with single primitive
                leaf_old = u - base_old
                idx = base_new + next_leaf
                group_starts[next_leaf] = leaf_old
                group_counts[next_leaf] = 1
                first_prim[idx] = int(leaf_old)
                prim_count[idx] = 1
                next_leaf += 1
                return idx
            if collapsible[u] == 1:
                # Collapsible subtree -> one new leaf with range
                f = int(first_idx[u])
                l = int(last_idx[u])
                idx = base_new + next_leaf
                group_starts[next_leaf] = f
                group_counts[next_leaf] = l - f + 1
                first_prim[idx] = f
                prim_count[idx] = l - f + 1
                next_leaf += 1
                return idx
            # Internal node
            idx = next_internal
            next_internal += 1
            lc_new = build_new(int(left_child[u]))
            rc_new = build_new(int(right_child[u]))
            left_new[idx] = lc_new
            right_new[idx] = rc_new
            return idx

        root_new = build_new(0)
        # Sanity
        # next_internal should be L-1 and next_leaf == L
        # Compute bounds using group-based method
        nodes_min, nodes_max = _compute_bounds_lbvh_groups(
            left_new,
            right_new,
            min_x,
            min_y,
            min_z,
            max_x,
            max_y,
            max_z,
            packed_indices,
            group_starts,
            group_counts,
        )
        # Replace children arrays with compacted ones
        left_child = left_new
        right_child = right_new
    else:
        # No compaction: per-primitive leaves
        first_prim = np.full((tot_nodes,), -1, dtype=np.int32)
        prim_count = np.zeros((tot_nodes,), dtype=np.int32)
        for i in range(N):
            leaf_idx = (N - 1) + i
            first_prim[leaf_idx] = i
            prim_count[leaf_idx] = 1
        nodes_min, nodes_max = _compute_bounds_lbvh_numba(
            left_child,
            right_child,
            min_x,
            min_y,
            min_z,
            max_x,
            max_y,
            max_z,
            packed_indices,
        )
        t_bounds = time.perf_counter()
        logger.info(f"LBVH: bounds {t_bounds - t_link:.3f}s total {t_bounds - t0:.3f}s")

    return (
        nodes_min,
        nodes_max,
        left_child,
        right_child,
        first_prim,
        prim_count,
        prim_orbit_ids,
        prim_seg_ids,
        prim_row_indices,
    )




def build_bvh_nodes_from_aabbs(
    min_x: npt.NDArray[np.float32],
    min_y: npt.NDArray[np.float32],
    min_z: npt.NDArray[np.float32],
    max_x: npt.NDArray[np.float32],
    max_y: npt.NDArray[np.float32],
    max_z: npt.NDArray[np.float32],
    *,
    max_leaf_size: int = 8,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],  # primitive order mapping into input AABBs
]:
    """
    Build a BVH over provided AABBs using a simple median-split strategy.

    Returns node arrays and a primitive order array indicating the order of
    primitives referenced by leaf ranges.
    """
    N = int(len(min_x))
    if N == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    mins = np.column_stack([min_x, min_y, min_z]).astype(np.float32, copy=False)
    maxs = np.column_stack([max_x, max_y, max_z]).astype(np.float32, copy=False)

    max_nodes = 2 * N - 1
    nodes_min = np.zeros((max_nodes, 3), dtype=np.float32)
    nodes_max = np.zeros((max_nodes, 3), dtype=np.float32)
    left_child = np.full((max_nodes,), -1, dtype=np.int32)
    right_child = np.full((max_nodes,), -1, dtype=np.int32)
    first_prim = np.full((max_nodes,), -1, dtype=np.int32)
    prim_count = np.zeros((max_nodes,), dtype=np.int32)
    prim_order: list[int] = []

    indices = np.arange(N, dtype=np.int32)

    def build_node(idx_list: npt.NDArray[np.int32]) -> int:
        node_index = build_node.next_index
        build_node.next_index += 1
        bb_min = mins[idx_list].min(axis=0)
        bb_max = maxs[idx_list].max(axis=0)
        nodes_min[node_index] = bb_min
        nodes_max[node_index] = bb_max

        if idx_list.size <= max_leaf_size:
            start = len(prim_order)
            prim_order.extend(idx_list.tolist())
            count = idx_list.size
            first_prim[node_index] = start
            prim_count[node_index] = int(count)
            return node_index

        cents = 0.5 * (mins[idx_list] + maxs[idx_list])
        extents = cents.max(axis=0) - cents.min(axis=0)
        axis = int(np.argmax(extents))
        order = np.argsort(cents[:, axis], kind="mergesort")
        sorted_idx = idx_list[order]
        mid = sorted_idx.size // 2
        lc = build_node(sorted_idx[:mid])
        rc = build_node(sorted_idx[mid:])
        left_child[node_index] = lc
        right_child[node_index] = rc
        return node_index

    build_node.next_index = 0  # type: ignore[attr-defined]
    root = build_node(indices)
    total_nodes = build_node.next_index  # type: ignore[attr-defined]

    nodes_min = nodes_min[:total_nodes]
    nodes_max = nodes_max[:total_nodes]
    left_child = left_child[:total_nodes]
    right_child = right_child[:total_nodes]
    first_prim = first_prim[:total_nodes]
    prim_count = prim_count[:total_nodes]

    return (
        nodes_min,
        nodes_max,
        left_child,
        right_child,
        first_prim,
        prim_count,
        np.asarray(prim_order, dtype=np.int32),
    )
