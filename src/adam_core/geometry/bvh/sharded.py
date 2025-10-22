"""
Sharded BVH index support: Morton-partitioned shards and TLAS (top-level BVH).

This module provides:
- Quivr tables for TLAS nodes/primitives, shard metadata, manifest, assignments
- Streaming sharded builder that writes per-shard segments and builds BLAS
- TLAS builder over shard root AABBs
- Resolver abstraction for loading per-shard BVH indices

Notes
- We use Morton partitioning on segment centroids to assign segments to shards.
- Build is streaming: orbits -> segments (in chunks) -> route to shard writers.
- After writing segments, we build a BVH per shard (BLAS) and a TLAS over shards.
"""

from __future__ import annotations

import os
import math
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import quivr as qv

from .index import (
    BVHIndex,
    BVHNodes,
    build_bvh_index_from_segments,
    orbits_to_segments_worker,
    build_bvh_nodes_from_aabbs,
)
from ...orbits.orbits import Orbits
from ...orbits.polyline import OrbitPolylineSegments, compute_segment_aabbs
from ...utils.iter import _iterate_chunk_indices


__all__ = [
    "TLASPrimitives",
    "ShardMetadata",
    "ShardedBVHManifest",
    "ShardAssignments",
    "ShardedBVH",
    "build_bvh_index_sharded",
    "FilesystemShardResolver",
]


logger = logging.getLogger(__name__)


# ============================
# Quivr table data structures
# ============================


## TLAS uses BVHNodes directly


class TLASPrimitives(qv.Table):
    """
    TLAS primitives, one per shard, mapping to shard_id and optional Morton range.
    """

    shard_id = qv.LargeStringColumn()
    morton_lo = qv.UInt64Column(nullable=True)
    morton_hi = qv.UInt64Column(nullable=True)

    # Attributes
    version = qv.StringAttribute(default="1.0.0")


class ShardMetadata(qv.Table):
    """
    Build-time metadata for shards. Useful for validation and inventory.
    """

    shard_id = qv.LargeStringColumn()
    min_x = qv.Float32Column()
    min_y = qv.Float32Column()
    min_z = qv.Float32Column()
    max_x = qv.Float32Column()
    max_y = qv.Float32Column()
    max_z = qv.Float32Column()

    num_segments = qv.Int64Column()
    num_nodes = qv.Int64Column()
    num_prims = qv.Int64Column()

    morton_lo = qv.UInt64Column()
    morton_hi = qv.UInt64Column()

    # Optional: local directory on builder node
    local_dir = qv.LargeStringColumn(nullable=True)

    # Attributes
    build_max_leaf_size = qv.IntAttribute(default=0)
    guard_arcmin = qv.FloatAttribute(default=0.0)
    epsilon_n_au = qv.FloatAttribute(default=0.0)
    padding_method = qv.StringAttribute(default="baseline")

    # Convenience metadata methods
    def num_shards(self) -> int:
        return int(len(self))

    def space_bounds(self) -> tuple[float, float, float, float, float, float]:
        if len(self) == 0:
            return (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        mnx = float(np.min(self.min_x.to_numpy()))
        mny = float(np.min(self.min_y.to_numpy()))
        mnz = float(np.min(self.min_z.to_numpy()))
        mxx = float(np.max(self.max_x.to_numpy()))
        mxy = float(np.max(self.max_y.to_numpy()))
        mxz = float(np.max(self.max_z.to_numpy()))
        return (mnx, mny, mnz, mxx, mxy, mxz)

    def strategy(self) -> str:
        return "spatial_morton"


class ShardAssignments(qv.Table):
    """
    Routing-only output: mapping detections to shard IDs.
    One row per (det_id, shard_id) pair.
    """

    det_id = qv.LargeStringColumn()
    shard_id = qv.LargeStringColumn()

    # Attributes
    version = qv.StringAttribute(default="1.0.0")
    max_shards_per_packet = qv.IntAttribute(default=0)
    tlas_depth = qv.IntAttribute(default=0)


# ============================
# Morton utilities
# ============================


def _normalize_points(
    pts: npt.NDArray[np.float64],
    space_min: npt.NDArray[np.float64],
    space_max: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Normalize points into [0, 1] within provided axis-aligned bounds.
    """
    denom = np.maximum(space_max - space_min, 1e-12)
    return np.clip((pts - space_min) / denom, 0.0, 1.0)


def _morton3d_encode_uint64(norm_pts: npt.NDArray[np.float64]) -> npt.NDArray[np.uint64]:
    """
    Encode normalized 3D points in [0,1] to 63-bit Morton codes (21 bits per axis).
    """
    # Quantize to 21 bits per axis
    scale = (1 << 21) - 1
    xyz = (norm_pts * scale + 0.5).astype(np.uint64)

    # Vectorized part1by2 (match implementation style in index.py)
    v = xyz
    x = v[:, 0]
    y = v[:, 1]
    z = v[:, 2]

    def part1by2_vec(a: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
        a = a.astype(np.uint64, copy=False)
        a = (a | (a << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
        a = (a | (a << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
        a = (a | (a << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
        a = (a | (a << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
        a = (a | (a << np.uint64(2))) & np.uint64(0x1249249249249249)
        return a

    xx = part1by2_vec(x)
    yy = part1by2_vec(y)
    zz = part1by2_vec(z)
    return (xx << np.uint64(2)) | (yy << np.uint64(1)) | zz


def _compute_cut_points(
    morton_codes: npt.NDArray[np.uint64], num_shards: int
) -> List[int]:
    """
    Compute (num_shards-1) cut points as uint64 values splitting codes by quantiles.
    """
    if num_shards <= 1:
        return []
    # Use equally spaced quantiles; guard against duplicates
    qs = np.linspace(0.0, 1.0, num=num_shards + 1, endpoint=True)[1:-1]
    cuts = np.quantile(morton_codes.astype(np.float64), qs, method="linear")
    cuts = np.asarray(cuts, dtype=np.uint64)
    # Ensure strictly increasing by forcing +1 when equal
    for i in range(1, len(cuts)):
        if cuts[i] <= cuts[i - 1]:
            cuts[i] = cuts[i - 1] + np.uint64(1)
    return [int(v) for v in cuts]


def _assign_shards(
    morton_codes: npt.NDArray[np.uint64], cut_points: List[int]
) -> npt.NDArray[np.int32]:
    """
    Assign each code to a shard index using cut_points.
    Shard ranges are: [0, cut0), [cut0, cut1), ..., [last, +inf)
    """
    cuts = np.asarray(cut_points, dtype=np.uint64)
    return np.searchsorted(cuts, morton_codes, side="right").astype(np.int32)


# ============================
# TLAS builder (median-split)
# ============================


def _build_tlas_from_shard_aabbs(
    mins: npt.NDArray[np.float32],
    maxs: npt.NDArray[np.float32],
    shard_ids: List[str],
    *,
    morton_lo: Optional[npt.NDArray[np.uint64]] = None,
    morton_hi: Optional[npt.NDArray[np.uint64]] = None,
) -> Tuple[BVHNodes, TLASPrimitives]:
    """
    Build TLAS using generic BVHNodes-from-AABBs helper; create a primitive row per shard.
    """
    n = len(shard_ids)
    if n == 0:
        return BVHNodes.empty(), TLASPrimitives.empty()

    nodes_min, nodes_max, left_child, right_child, first_prim, prim_count, prim_order = build_bvh_nodes_from_aabbs(
        mins[:, 0], mins[:, 1], mins[:, 2], maxs[:, 0], maxs[:, 1], maxs[:, 2], max_leaf_size=4
    )

    # Compute bvh_max_depth for TLAS nodes
    if len(left_child) == 0:
        max_depth_val = 0
    else:
        tot_nodes = int(len(left_child))
        parent = np.full((tot_nodes,), -1, dtype=np.int32)
        for i in range(tot_nodes):
            lc = int(left_child[i])
            rc = int(right_child[i])
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
                break
            visited[node] = True
            lc = int(left_child[node])
            rc = int(right_child[node])
            if lc >= 0:
                stack.append((lc, depth + 1))
            if rc >= 0:
                stack.append((rc, depth + 1))
            if lc < 0 and rc < 0 and depth > max_depth_val:
                max_depth_val = depth

    tlas_nodes = BVHNodes.from_kwargs(
        nodes_min_x=(nodes_min[:, 0] if len(nodes_min) else np.array([], dtype=np.float32)),
        nodes_min_y=(nodes_min[:, 1] if len(nodes_min) else np.array([], dtype=np.float32)),
        nodes_min_z=(nodes_min[:, 2] if len(nodes_min) else np.array([], dtype=np.float32)),
        nodes_max_x=(nodes_max[:, 0] if len(nodes_max) else np.array([], dtype=np.float32)),
        nodes_max_y=(nodes_max[:, 1] if len(nodes_max) else np.array([], dtype=np.float32)),
        nodes_max_z=(nodes_max[:, 2] if len(nodes_max) else np.array([], dtype=np.float32)),
        left_child=left_child,
        right_child=right_child,
        first_prim=first_prim,
        prim_count=prim_count,
        shard_id="index",
        build_max_leaf_size=4,
        bvh_max_depth=int(max_depth_val),
        aabb_guard_arcmin=0.0,
        aabb_epsilon_n_au=0.0,
        aabb_padding_method="baseline",
    )

    # Create primitives according to prim_order
    if prim_order.size == 0:
        return tlas_nodes, TLASPrimitives.empty()
    shard_ids_arr = [shard_ids[i] for i in prim_order]
    if morton_lo is None or morton_hi is None:
        morton_lo_arr = np.zeros((prim_order.size,), dtype=np.uint64)
        morton_hi_arr = np.zeros((prim_order.size,), dtype=np.uint64)
    else:
        morton_lo_arr = morton_lo[prim_order]
        morton_hi_arr = morton_hi[prim_order]
    tlas_prims = TLASPrimitives.from_kwargs(
        shard_id=shard_ids_arr,
        morton_lo=morton_lo_arr,
        morton_hi=morton_hi_arr,
    )

    return tlas_nodes, tlas_prims


# ============================
# Streaming sharded builder
# ============================


@dataclass
class _ShardWriter:
    writer: pq.ParquetWriter
    rows_written: int


class _LRUWriters:
    """
    LRU cache of ParquetWriter instances to limit open file descriptors.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self._cache: OrderedDict[str, _ShardWriter] = OrderedDict()

    def get(self, key: str) -> Optional[_ShardWriter]:
        w = self._cache.get(key)
        if w is not None:
            self._cache.move_to_end(key)
        return w

    def put(self, key: str, writer: pq.ParquetWriter) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return
        self._cache[key] = _ShardWriter(writer=writer, rows_written=0)
        if len(self._cache) > self.capacity:
            # Evict least-recently used
            old_key, old = self._cache.popitem(last=False)
            try:
                old.writer.close()
            except Exception:
                logger.exception("Failed to close ParquetWriter for shard %s", old_key)

    def close_all(self) -> None:
        for key, w in list(self._cache.items()):
            try:
                w.writer.close()
            except Exception:
                logger.exception("Failed to close ParquetWriter for shard %s", key)
        self._cache.clear()


def _iter_orbits_batches(
    orbits_source: Union[Orbits, str],
    *,
    chunk_size_orbits: int,
    orbits_parquet_batch_rows: int,
) -> Iterator[Orbits]:
    """
    Yield `Orbits` batches from either an in-memory table or a Parquet file path.
    """
    if isinstance(orbits_source, Orbits):
        for s, e in _iterate_chunk_indices(orbits_source, chunk_size_orbits):
            yield orbits_source[s:e]
        return

    # Parquet path
    pf = pq.ParquetFile(orbits_source)
    for rb in pf.iter_batches(batch_size=orbits_parquet_batch_rows):
        tbl = pa.Table.from_batches([rb])
        yield Orbits.from_pyarrow(tbl)


def _compute_centroids(segments: OrbitPolylineSegments) -> npt.NDArray[np.float64]:
    x0 = segments.x0.to_numpy()
    y0 = segments.y0.to_numpy()
    z0 = segments.z0.to_numpy()
    x1 = segments.x1.to_numpy()
    y1 = segments.y1.to_numpy()
    z1 = segments.z1.to_numpy()
    c = np.column_stack(
        [
            0.5 * (x0 + x1),
            0.5 * (y0 + y1),
            0.5 * (z0 + z1),
        ]
    ).astype(np.float64, copy=False)
    return c


def build_bvh_index_sharded(
    orbits_source: Union[Orbits, str],
    *,
    index_dir: str,
    num_shards: Optional[int] = None,
    target_shard_size_gb: Optional[float] = None,
    sample_fraction: float = 0.01,
    max_chord_arcmin: float = 5.0,
    guard_arcmin: float = 0.65,
    max_leaf_size: int = 64,
    max_segments_per_orbit: int = 512,
    epsilon_n_au: float = 1e-6,
    padding_method: str = "baseline",
    chunk_size_orbits: int = 10_000,
    max_processes: Optional[int] = 1,
    orbits_parquet_batch_rows: int = 1_000_000,
    max_open_writers: int = 256,
) -> "ShardedBVH":
    """
    Build a sharded BVH index on disk using Morton partitioning and a TLAS.

    Returns a `ShardedBVHManifest` quivr table (single row). Side effects: writes
    per-shard `segments.parquet`, then per-shard BVH nodes/prims, and TLAS files
    under `index_dir`.
    """
    os.makedirs(index_dir, exist_ok=True)
    manifest_dir = os.path.join(index_dir, "manifest")
    shards_root = os.path.join(index_dir, "shards")
    os.makedirs(manifest_dir, exist_ok=True)
    os.makedirs(shards_root, exist_ok=True)

    # Phase 0: sampling pass to estimate bounds and cut points
    logger.info("Sampling orbits for Morton partitioning")
    sample_centroids: List[npt.NDArray[np.float64]] = []
    total_sampled_segments = 0
    max_sample_segments = 500_000

    # For Parquet source, iterate only a few batches respecting sample_fraction
    for batch in _iter_orbits_batches(
        orbits_source,
        chunk_size_orbits=chunk_size_orbits,
        orbits_parquet_batch_rows=orbits_parquet_batch_rows,
    ):
        # Randomly subsample orbits in this batch
        if len(batch) == 0:
            continue
        take = max(1, int(len(batch) * float(sample_fraction)))
        if take < len(batch):
            # Quivr supports take via Arrow indices
            idx = np.random.choice(len(batch), size=take, replace=False)
            batch = batch.take(pa.array(idx, type=pa.int64()))

        segs = orbits_to_segments_worker(
            batch,
            max_chord_arcmin=max_chord_arcmin,
            max_segments_per_orbit=max_segments_per_orbit,
        )
        if len(segs) == 0:
            continue

        sample_centroids.append(_compute_centroids(segs))
        total_sampled_segments += int(len(segs))
        if total_sampled_segments >= max_sample_segments:
            break

    if not sample_centroids:
        # Empty input: write via ShardedBVH to centralize IO
        sharded = ShardedBVH(
            index_root=index_dir,
            tlas_nodes=BVHNodes.empty(),
            tlas_prims=TLASPrimitives.empty(),
            shards=ShardMetadata.empty(),
        )
        sharded.to_dir(index_dir)
        return sharded

    sample_pts = np.concatenate(sample_centroids, axis=0)
    # Compute space bounds with small padding
    mn = sample_pts.min(axis=0)
    mx = sample_pts.max(axis=0)
    pad = 1e-6 * np.maximum(1.0, np.abs(mx - mn))
    space_min = (mn - pad).astype(np.float64)
    space_max = (mx + pad).astype(np.float64)

    # Determine num_shards
    if num_shards is None:
        if target_shard_size_gb is None:
            num_shards = 512
        else:
            # Estimate bytes/segment from sample: segments.parquet table size / rows
            # Conservative: assume ~64 bytes per segment (positions + metadata)
            est_bytes_per_segment = 64.0
            est_total_segments = float(sample_pts.shape[0]) / float(sample_fraction)
            total_bytes = est_total_segments * est_bytes_per_segment
            num_shards = max(1, int(math.ceil(total_bytes / (target_shard_size_gb * (1024**3)))))
    num_shards = int(num_shards)

    # Morton codes and cut points
    sample_codes = _morton3d_encode_uint64(_normalize_points(sample_pts, space_min, space_max))
    cut_points = _compute_cut_points(sample_codes, num_shards)

    # Manifest removed; metadata is derived from ShardMetadata

    # Phase 1: streaming router and shard writers
    # Prepare writers and per-shard stats
    empty_schema = OrbitPolylineSegments.empty().table.schema
    writers = _LRUWriters(capacity=max_open_writers)

    shard_min = np.full((num_shards, 3), np.inf, dtype=np.float64)
    shard_max = np.full((num_shards, 3), -np.inf, dtype=np.float64)
    shard_counts = np.zeros((num_shards,), dtype=np.int64)

    for batch in _iter_orbits_batches(
        orbits_source,
        chunk_size_orbits=chunk_size_orbits,
        orbits_parquet_batch_rows=orbits_parquet_batch_rows,
    ):
        if len(batch) == 0:
            continue
        segs = orbits_to_segments_worker(
            batch,
            max_chord_arcmin=max_chord_arcmin,
            max_segments_per_orbit=max_segments_per_orbit,
        )
        if len(segs) == 0:
            continue

        cents = _compute_centroids(segs)
        codes = _morton3d_encode_uint64(
            _normalize_points(cents, space_min, space_max)
        )
        shard_idx = _assign_shards(codes, cut_points)

        # Compute per-segment AABBs to update shard root bounds
        min_x, min_y, min_z, max_x, max_y, max_z = compute_segment_aabbs(
            segs,
            guard_arcmin=guard_arcmin,
            epsilon_n_au=epsilon_n_au,
            padding_method=padding_method,  # type: ignore[arg-type]
            max_processes=max_processes,
        )
        seg_mins = np.column_stack([min_x, min_y, min_z])
        seg_maxs = np.column_stack([max_x, max_y, max_z])

        # Append rows to per-shard writers (by Arrow batch for efficiency)
        tbl = segs.table
        arr_shard = pa.array(shard_idx.astype(np.int32))
        tbl = tbl.append_column("_shard_tmp", arr_shard)
        for s in range(num_shards):
            mask = pa.compute.equal(tbl["_shard_tmp"], pa.scalar(int(s), type=pa.int32()))
            if not pa.compute.any(mask).as_py():
                continue
            sub = tbl.filter(mask)
            # Drop temp column
            sub = sub.drop_columns(["_shard_tmp"])
            if sub.num_rows == 0:
                continue
            shard_id = f"shard_{s:05d}"
            shard_dir = os.path.join(shards_root, shard_id)
            os.makedirs(shard_dir, exist_ok=True)
            seg_path = os.path.join(shard_dir, "segments.parquet")
            w = writers.get(shard_id)
            if w is None:
                writer = pq.ParquetWriter(seg_path, empty_schema)
                writers.put(shard_id, writer)
                w = writers.get(shard_id)
            assert w is not None
            w.writer.write_table(sub)
            w.rows_written += sub.num_rows

            # Update shard bounds and counts
            rows_bool = shard_idx == s
            if np.any(rows_bool):
                shard_mn = seg_mins[rows_bool]
                shard_mx = seg_maxs[rows_bool]
                shard_min[s] = np.minimum(shard_min[s], shard_mn.min(axis=0))
                shard_max[s] = np.maximum(shard_max[s], shard_mx.max(axis=0))
                shard_counts[s] += int(rows_bool.sum())

    writers.close_all()

    # Initialize shard metadata
    mortar_lo = np.zeros((num_shards,), dtype=np.uint64)
    mortar_hi = np.zeros((num_shards,), dtype=np.uint64)
    # Infer per-shard cut ranges from cut_points
    # Ranges: [0, cut0), [cut0, cut1), ... [last, +inf)
    last = np.uint64(0)
    for s in range(num_shards):
        lo = last
        hi = np.iinfo(np.uint64).max if s == num_shards - 1 else np.uint64(cut_points[s])
        mortar_lo[s] = lo
        mortar_hi[s] = hi
        last = hi

    shard_ids = [f"shard_{s:05d}" for s in range(num_shards)]
    shard_dirs = [os.path.join(shards_root, sid) for sid in shard_ids]

    # Prepare output arrays for metadata; finalize after BLAS build
    min_x_out = shard_min[:, 0].astype(np.float32, copy=False)
    min_y_out = shard_min[:, 1].astype(np.float32, copy=False)
    min_z_out = shard_min[:, 2].astype(np.float32, copy=False)
    max_x_out = shard_max[:, 0].astype(np.float32, copy=False)
    max_y_out = shard_max[:, 1].astype(np.float32, copy=False)
    max_z_out = shard_max[:, 2].astype(np.float32, copy=False)
    num_nodes_out = np.zeros_like(shard_counts)
    num_prims_out = np.zeros_like(shard_counts)

    # Phase 2: per-shard BLAS build
    for sid, sdir in zip(shard_ids, shard_dirs):
        seg_path = os.path.join(sdir, "segments.parquet")
        if not os.path.exists(seg_path):
            continue
        segs = OrbitPolylineSegments.from_parquet(seg_path)
        if len(segs) == 0:
            continue
        idx = build_bvh_index_from_segments(
            segs,
            max_leaf_size=max_leaf_size,
            guard_arcmin=guard_arcmin,
            epsilon_n_au=epsilon_n_au,
            padding_method=padding_method,
            max_processes=max_processes,
        )
        # Write full index using BVHIndex helper
        idx.to_parquet(sdir)

        # Update counts and validate min/max from nodes
        node_min = np.column_stack(
            [
                np.asarray(idx.nodes.nodes_min_x),
                np.asarray(idx.nodes.nodes_min_y),
                np.asarray(idx.nodes.nodes_min_z),
            ]
        )
        node_max = np.column_stack(
            [
                np.asarray(idx.nodes.nodes_max_x),
                np.asarray(idx.nodes.nodes_max_y),
                np.asarray(idx.nodes.nodes_max_z),
            ]
        )
        root_min = node_min.min(axis=0).astype(np.float32, copy=False)
        root_max = node_max.max(axis=0).astype(np.float32, copy=False)
        i = int(sid.split("_")[-1])
        # Update numpy arrays
        min_x_out[i] = float(root_min[0])
        min_y_out[i] = float(root_min[1])
        min_z_out[i] = float(root_min[2])
        max_x_out[i] = float(root_max[0])
        max_y_out[i] = float(root_max[1])
        max_z_out[i] = float(root_max[2])
        num_nodes_out[i] = int(len(idx.nodes))
        num_prims_out[i] = int(len(idx.prims))

    # Build ShardMetadata and persist
    meta = ShardMetadata.from_kwargs(
        shard_id=shard_ids,
        min_x=min_x_out,
        min_y=min_y_out,
        min_z=min_z_out,
        max_x=max_x_out,
        max_y=max_y_out,
        max_z=max_z_out,
        num_segments=shard_counts,
        num_nodes=num_nodes_out,
        num_prims=num_prims_out,
        morton_lo=mortar_lo,
        morton_hi=mortar_hi,
        local_dir=shard_dirs,
        build_max_leaf_size=int(max_leaf_size),
        guard_arcmin=float(guard_arcmin),
        epsilon_n_au=float(epsilon_n_au),
        padding_method=str(padding_method),
    )

    # Phase 3: TLAS build over shards
    mins = np.column_stack([min_x_out, min_y_out, min_z_out])
    maxs = np.column_stack([max_x_out, max_y_out, max_z_out])
    tlas_nodes, tlas_prims = _build_tlas_from_shard_aabbs(
        mins.astype(np.float32, copy=False),
        maxs.astype(np.float32, copy=False),
        shard_ids,
        morton_lo=mortar_lo,
        morton_hi=mortar_hi,
    )
    # Defer writing via ShardedBVH

    # Write assembled ShardedBVH
    sharded = ShardedBVH(
        index_root=index_dir,
        tlas_nodes=tlas_nodes,
        tlas_prims=tlas_prims,
        shards=meta,
    )
    sharded.to_dir(index_dir)
    return sharded


# ============================
# Resolver (filesystem default)
# ============================


class FilesystemShardResolver:
    """
    Resolve shard_id -> BVHIndex by reading from a base directory.

    Directory layout: <base_dir>/shards/<shard_id>/{segments,bvh_nodes,bvh_prims}.parquet
    """

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir

    def resolve(self, shard_id: str) -> BVHIndex:
        shard_dir = os.path.join(self.base_dir, "shards", shard_id)
        if not os.path.isdir(shard_dir):
            raise FileNotFoundError(f"Shard directory not found: {shard_dir}")
        return BVHIndex.from_parquet(shard_dir)


# ============================
# ShardedBVH wrapper
# ============================


class ShardedBVH:
    """
    Centralized wrapper for a sharded BVH index on disk.

    Provides single points of entry for reading/writing the manifest, TLAS, and
    shard metadata, and helpers to resolve and load shard BVH indices.
    """

    def __init__(
        self,
        *,
        index_root: str,
        tlas_nodes: BVHNodes,
        tlas_prims: TLASPrimitives,
        shards: ShardMetadata,
    ) -> None:
        self.index_root = index_root
        self.tlas_nodes = tlas_nodes
        self.tlas_prims = tlas_prims
        self.shards = shards

    @classmethod
    def from_dir(cls, index_root: str) -> "ShardedBVH":
        manifest_dir = os.path.join(index_root, "manifest")
        tlas_nodes = BVHNodes.from_parquet(
            os.path.join(manifest_dir, "tlas_nodes.parquet")
        )
        tlas_prims = TLASPrimitives.from_parquet(
            os.path.join(manifest_dir, "tlas_prims.parquet")
        )
        shards = ShardMetadata.from_parquet(
            os.path.join(manifest_dir, "shards.parquet")
        )
        return cls(
            index_root=index_root,
            tlas_nodes=tlas_nodes,
            tlas_prims=tlas_prims,
            shards=shards,
        )

    def to_dir(self, index_root: Optional[str] = None) -> None:
        base = index_root or self.index_root
        manifest_dir = os.path.join(base, "manifest")
        os.makedirs(manifest_dir, exist_ok=True)
        self.tlas_nodes.to_parquet(os.path.join(manifest_dir, "tlas_nodes.parquet"))
        self.tlas_prims.to_parquet(os.path.join(manifest_dir, "tlas_prims.parquet"))
        self.shards.to_parquet(os.path.join(manifest_dir, "shards.parquet"))

    def get_shard_metadata(self, shard_id: str) -> ShardMetadata:
        import pyarrow.compute as pc

        idx = pc.equal(self.shards.shard_id, shard_id)
        return self.shards.apply_mask(idx)

    def resolve_shard(self, shard_id: str, resolver: FilesystemShardResolver) -> BVHIndex:
        return resolver.resolve(shard_id)


