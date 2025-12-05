""" """

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import ray
from numba import njit

from ...ray_cluster import initialize_use_ray
import os
from ...utils.iter import _iterate_chunks
from ..rays import ObservationRays
from .index import BVHIndex, get_leaf_primitives_numpy
from .sharded import (
    ShardedBVH,
    ShardAssignments,
    FilesystemShardResolver,
)
from .index import BVHNodes

__all__ = [
    "OverlapHits",
    "query_bvh",
    "ray_segment_distance_window",
    "calc_ray_segment_distance_and_guard",
    "query_bvh_worker",
    "query_bvh_worker_remote",
    "QueryBVHTelemetry",
    "CandidatePairs",
    "find_bvh_matches",
    # Sharded TLAS routing
    "route_rays_to_shards",
    "write_routed_rays_by_shard",
    "query_bvh_sharded",
]

logger = logging.getLogger(__name__)


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

    # Query provenance attributes
    query_guard_arcmin = qv.FloatAttribute(default=0.0)
    query_max_hits_per_ray = qv.IntAttribute(default=0)
    query_max_candidates_per_ray = qv.IntAttribute(default=0)
    query_batch_size = qv.IntAttribute(default=0)
    query_max_processes = qv.IntAttribute(default=0)


class CandidatePairs(qv.Table):
    """
    Join table of ray–segment candidate pairs (Arrow-backed, size N).
    """

    det_index = qv.Int32Column()
    seg_row = qv.Int32Column()
    seg_id = qv.Int32Column()
    leaf_id = qv.Int32Column()


@dataclass
class QueryBVHTelemetry:
    truncation_occurred: bool
    max_leaf_visits_observed: int
    rays_with_zero_candidates: int
    packets_traversed: int
    pairs_total: int
    pairs_within_guard: int


@dataclass
class FindBVHTelemetry:
    pairs_total: int
    truncation_occurred: bool
    max_leaf_visits_observed: int
    rays_with_zero_candidates: int
    packets_traversed: int


# -----------------------------
# Coherence ordering utilities
# -----------------------------


def _direction_bins(u_x: np.ndarray, u_y: np.ndarray, u_z: np.ndarray) -> np.ndarray:
    """
    Cheap direction binning: major axis (x/y/z), sign bit, and 2-bit quantization
    of the two minor axes. Produces up to 192 bins.
    """
    vec = np.stack([u_x, u_y, u_z], axis=1)
    absvec = np.abs(vec)
    axis = np.argmax(absvec, axis=1).astype(np.int32)

    major_values = vec[np.arange(vec.shape[0]), axis]
    signbit = (major_values >= 0.0).astype(np.int32)

    # Prepare arrays for minor components
    minor0 = np.empty_like(u_x)
    minor1 = np.empty_like(u_x)

    mask0 = axis == 0
    if np.any(mask0):
        minor0[mask0] = u_y[mask0]
        minor1[mask0] = u_z[mask0]
    mask1 = axis == 1
    if np.any(mask1):
        minor0[mask1] = u_x[mask1]
        minor1[mask1] = u_z[mask1]
    mask2 = axis == 2
    if np.any(mask2):
        minor0[mask2] = u_x[mask2]
        minor1[mask2] = u_y[mask2]

    # 2-bit quantization for each minor in [-1,1]
    def quant2(v: np.ndarray) -> np.ndarray:
        q = ((v * 0.5 + 0.5) * 4.0).astype(np.int32)
        q = np.clip(q, 0, 3)
        return q

    q0 = quant2(minor0)
    q1 = quant2(minor1)

    # Combine: (axis*2 + sign) in [0..5], then 4*4 bins of minors
    head = (axis * 2 + signbit) * 16
    tail = q0 * 4 + q1
    return (head + tail).astype(np.int32)


# =========================
# Numba CPU traversal path
# =========================


@njit(cache=True, fastmath=False)
def _packet_intersect_aabb_numba(
    ro_pkt: np.ndarray,
    rd_pkt: np.ndarray,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    mask_in: np.ndarray,
) -> np.ndarray:
    P = ro_pkt.shape[0]
    out = np.zeros((P,), dtype=np.bool_)
    eps = np.float32(1e-15)
    for i in range(P):
        if not mask_in[i]:
            continue
        ox = ro_pkt[i, 0]
        oy = ro_pkt[i, 1]
        oz = ro_pkt[i, 2]
        dx = rd_pkt[i, 0]
        dy = rd_pkt[i, 1]
        dz = rd_pkt[i, 2]

        # X slab
        if np.abs(dx) > eps:
            tx1 = (aabb_min[0] - ox) / dx
            tx2 = (aabb_max[0] - ox) / dx
            tminx = tx1 if tx1 < tx2 else tx2
            tmaxx = tx2 if tx2 > tx1 else tx1
        else:
            if ox < aabb_min[0] or ox > aabb_max[0]:
                continue
            tminx = -np.inf
            tmaxx = np.inf

        # Y slab
        if np.abs(dy) > eps:
            ty1 = (aabb_min[1] - oy) / dy
            ty2 = (aabb_max[1] - oy) / dy
            tminy = ty1 if ty1 < ty2 else ty2
            tmaxy = ty2 if ty2 > ty1 else ty1
        else:
            if oy < aabb_min[1] or oy > aabb_max[1]:
                continue
            tminy = -np.inf
            tmaxy = np.inf

        # Z slab
        if np.abs(dz) > eps:
            tz1 = (aabb_min[2] - oz) / dz
            tz2 = (aabb_max[2] - oz) / dz
            tminz = tz1 if tz1 < tz2 else tz2
            tmaxz = tz2 if tz2 > tz1 else tz1
        else:
            if oz < aabb_min[2] or oz > aabb_max[2]:
                continue
            tminz = -np.inf
            tmaxz = np.inf

        tmin = tminx
        if tminy > tmin:
            tmin = tminy
        if tminz > tmin:
            tmin = tminz
        tmax = tmaxx
        if tmaxy < tmax:
            tmax = tmaxy
        if tmaxz < tmax:
            tmax = tmaxz

        if tmax >= (tmin if tmin > 0.0 else 0.0):
            out[i] = True
    return out


@njit(cache=True, fastmath=False)
def packet_traverse_bvh_numba(
    nodes_min: np.ndarray,
    nodes_max: np.ndarray,
    left_child: np.ndarray,
    right_child: np.ndarray,
    ro_pkt: np.ndarray,
    rd_pkt: np.ndarray,
    max_stack_depth: int,
    max_leaf_visits: int,
):
    P = ro_pkt.shape[0]
    stack_nodes = np.empty((max_stack_depth,), dtype=np.int32)
    stack_masks = np.zeros((max_stack_depth, P), dtype=np.bool_)
    top = 0

    visited_leaf_nodes = np.empty((max_leaf_visits,), dtype=np.int32)
    visited_masks = np.zeros((max_leaf_visits, P), dtype=np.bool_)
    num_vis = 0

    # Push root with all-active mask
    stack_nodes[top] = 0
    for i in range(P):
        stack_masks[top, i] = True
    top += 1

    while top > 0 and num_vis < max_leaf_visits:
        top -= 1
        node_idx = stack_nodes[top]
        mask = stack_masks[top]

        # Intersect packet with node AABB
        aabb_min = nodes_min[node_idx]
        aabb_max = nodes_max[node_idx]
        hits = _packet_intersect_aabb_numba(ro_pkt, rd_pkt, aabb_min, aabb_max, mask)
        any_hit = False
        for i in range(P):
            if hits[i]:
                any_hit = True
                break
        if not any_hit:
            continue

        if left_child[node_idx] == -1:
            visited_leaf_nodes[num_vis] = node_idx
            for i in range(P):
                visited_masks[num_vis, i] = hits[i]
            num_vis += 1
            continue

        # Push children with current hits mask
        lc = left_child[node_idx]
        rc = right_child[node_idx]
        stack_nodes[top] = lc
        for i in range(P):
            stack_masks[top, i] = hits[i]
        top += 1
        stack_nodes[top] = rc
        for i in range(P):
            stack_masks[top, i] = hits[i]
        top += 1

    return visited_leaf_nodes, visited_masks, num_vis


def find_bvh_matches(
    index: BVHIndex,
    rays: ObservationRays,
    *,
    packet_size: int = 64,
    max_leaf_visits: int = 262144,
    return_telemetry: bool = False,
) -> tuple[CandidatePairs, FindBVHTelemetry]:
    """
    Find candidates from BVH traversal across a batch of rays.

    This function performs CPU-side BVH traversal to collect all candidate
    segments, then packs them into a CandidateBatch.

    Parameters
    ----------
    index : BVHIndex
        BVH index (quivr tables for segments, nodes, prims)
    rays : ObservationRays
        Observation rays (quivr table)

    Returns
    -------
    results : tuple[CandidatePairs, FindBVHTelemetry]
        Candidate pairs and telemetry
    """
    num_rays = len(rays)

    # Materialize minimal arrays from quivr tables
    nodes = index.nodes
    nodes_min_np, nodes_max_np = nodes.min_max_numpy()
    left_child_np = np.asarray(nodes.left_child, dtype=np.int32)
    right_child_np = np.asarray(nodes.right_child, dtype=np.int32)
    # first_prim_np = np.asarray(nodes.first_prim, dtype=np.int32)
    prim_count_np = np.asarray(nodes.prim_count, dtype=np.int32)

    # Primitive arrays
    # Packed arrays available if needed for downstream processing
    # packed_row_np = np.asarray(index.prims.segment_row_index, dtype=np.int32)
    # packed_seg_np = np.asarray(index.prims.prim_seg_ids, dtype=np.int32)

    # Rays: observer positions and LOS directions (NumPy)
    ro_x = rays.observer.coordinates.x.to_numpy()
    ro_y = rays.observer.coordinates.y.to_numpy()
    ro_z = rays.observer.coordinates.z.to_numpy()
    ray_origins_cpu = np.column_stack([ro_x, ro_y, ro_z])

    rd_x = rays.u_x.to_numpy()
    rd_y = rays.u_y.to_numpy()
    rd_z = rays.u_z.to_numpy()
    ray_directions_cpu = np.column_stack([rd_x, rd_y, rd_z])

    # Ensure dtypes for Numba traversal
    nodes_min_np = nodes_min_np.astype(np.float32, copy=False)
    nodes_max_np = nodes_max_np.astype(np.float32, copy=False)

    # Use index's recorded max depth
    stack_capacity = index.nodes.bvh_max_depth

    det_idx_list: list[np.ndarray] = []
    seg_row_list: list[np.ndarray] = []
    seg_id_list: list[np.ndarray] = []
    leaf_id_list: list[np.ndarray] = []

    # Telemetry
    truncation_any = False
    max_visits_observed = 0
    packets_traversed = 0
    had_candidate = np.zeros((num_rays,), dtype=bool)

    # Pad rays once to multiple of packet_size and device-put once
    num_rays_padded = ((num_rays + packet_size - 1) // packet_size) * packet_size
    ray_origins_padded = np.zeros((num_rays_padded, 3), dtype=np.float32)
    ray_directions_padded = np.zeros_like(ray_origins_padded)
    ray_origins_padded[:num_rays] = ray_origins_cpu
    ray_directions_padded[:num_rays] = ray_directions_cpu

    for start in range(0, num_rays_padded, packet_size):
        end = start + packet_size
        P = min(packet_size, num_rays - start)
        ro_pkt = ray_origins_padded[start:end]
        rd_pkt = ray_directions_padded[start:end]

        visited_leaf_nodes, visited_masks, num_vis = packet_traverse_bvh_numba(
            nodes_min_np,
            nodes_max_np,
            left_child_np,
            right_child_np,
            ro_pkt,
            rd_pkt,
            stack_capacity,
            max_leaf_visits,
        )

        # Telemetry on num_visits
        num_vis = int(num_vis)
        if num_vis == 0:
            continue
        packets_traversed += 1
        if num_vis == int(max_leaf_visits):
            truncation_any = True
            logger.warning(
                "packet_traverse_bvh_numba reached max_leaf_visits=%d; consider increasing the cap or reducing packet_size",
                int(max_leaf_visits),
            )
        if num_vis > max_visits_observed:
            max_visits_observed = num_vis

        # Use traversal results directly (Numba outputs)
        leaf_ids_full = visited_leaf_nodes[:num_vis].astype(np.int32, copy=False)
        masks_full = visited_masks[:num_vis, :P]

        # Filter leaves with primitives
        counts_full = prim_count_np[leaf_ids_full]
        nz_mask = counts_full > 0
        if not np.any(nz_mask):
            continue
        leaf_ids = leaf_ids_full[nz_mask]
        masks = masks_full[nz_mask]
        counts = counts_full[nz_mask]

        # Gather primitives once for all visited leaves with primitives
        rows_all, segs_all, leafs_all = get_leaf_primitives_numpy(
            nodes, index.prims, leaf_ids
        )
        if rows_all.size == 0:
            continue

        # Offsets per leaf in concatenated arrays
        offsets = np.zeros_like(counts)
        if len(counts) > 0:
            offsets[1:] = np.cumsum(counts[:-1])

        for i in range(len(counts)):
            c_i = int(counts[i])
            if c_i <= 0:
                continue
            mask_i = masks[i]
            if not np.any(mask_i):
                continue
            leaf_i = int(leaf_ids[i])
            off = int(offsets[i])
            rows_i = rows_all[off : off + c_i]
            segs_i = segs_all[off : off + c_i]

            ray_local_idx = np.nonzero(mask_i)[0]
            if ray_local_idx.size == 0:
                continue
            ray_global_idx = ray_local_idx + start

            # Broadcast: Cartesian product of rays in mask with primitives in leaf
            det_idx_expanded = np.repeat(ray_global_idx.astype(np.int32), c_i)
            seg_rows_expanded = np.tile(rows_i, ray_local_idx.size)
            seg_ids_expanded = np.tile(segs_i, ray_local_idx.size)
            leaf_ids_expanded = np.full(det_idx_expanded.shape, leaf_i, dtype=np.int32)

            det_idx_list.append(det_idx_expanded)
            seg_row_list.append(seg_rows_expanded)
            seg_id_list.append(seg_ids_expanded)
            leaf_id_list.append(leaf_ids_expanded)

            # Mark rays that produced at least one candidate
            had_candidate[ray_global_idx] = True

    if not det_idx_list:
        pairs = CandidatePairs.from_kwargs(
            det_index=np.array([], dtype=np.int32),
            seg_row=np.array([], dtype=np.int32),
            seg_id=np.array([], dtype=np.int32),
            leaf_id=np.array([], dtype=np.int32),
        )
        rays_with_zero = int(num_rays)  # None produced any candidates
        telemetry = FindBVHTelemetry(
            pairs_total=0,
            truncation_occurred=truncation_any,
            max_leaf_visits_observed=int(max_visits_observed),
            rays_with_zero_candidates=rays_with_zero,
            packets_traversed=int(packets_traversed),
        )
        return (pairs, telemetry)

    det_indices_flat = np.concatenate(det_idx_list).astype(np.int32, copy=False)
    seg_rows_flat = np.concatenate(seg_row_list).astype(np.int32, copy=False)
    seg_ids_flat = np.concatenate(seg_id_list).astype(np.int32, copy=False)
    leaf_ids_flat = np.concatenate(leaf_id_list).astype(np.int32, copy=False)

    pairs = CandidatePairs.from_kwargs(
        det_index=det_indices_flat,
        seg_row=seg_rows_flat,
        seg_id=seg_ids_flat,
        leaf_id=leaf_ids_flat,
    )

    logger.debug(
        f"Aggregated {len(pairs)} candidates from {num_rays} rays (ragged, jax-packet traversal)"
    )
    rays_with_zero = int(np.count_nonzero(~had_candidate))
    telemetry = FindBVHTelemetry(
        pairs_total=int(len(pairs)),
        truncation_occurred=truncation_any,
        max_leaf_visits_observed=int(max_visits_observed),
        rays_with_zero_candidates=rays_with_zero,
        packets_traversed=int(packets_traversed),
    )
    return (pairs, telemetry)


@dataclass
class CandidateIndices:
    det_idx: np.ndarray
    seg_idx: np.ndarray
    seg_id: np.ndarray
    leaf_id: np.ndarray


@dataclass
class GeometryDevice:
    ro_d: jax.Array
    rd_d: jax.Array
    d_obs_d: jax.Array
    s0_d: jax.Array
    s1_d: jax.Array
    r_mid_d: jax.Array


def materialize_geometry_device(
    index: BVHIndex, rays: ObservationRays
) -> GeometryDevice:
    ro_d = jnp.asarray(
        np.column_stack(
            [
                rays.observer.coordinates.x.to_numpy(),
                rays.observer.coordinates.y.to_numpy(),
                rays.observer.coordinates.z.to_numpy(),
            ]
        )
    )
    rd_d = jnp.asarray(
        np.column_stack([rays.u_x.to_numpy(), rays.u_y.to_numpy(), rays.u_z.to_numpy()])
    )
    d_obs_d = jnp.asarray(rays.observer.coordinates.r_mag)
    s0_d = jnp.asarray(
        np.column_stack(
            [
                index.segments.x0.to_numpy(),
                index.segments.y0.to_numpy(),
                index.segments.z0.to_numpy(),
            ]
        )
    )
    s1_d = jnp.asarray(
        np.column_stack(
            [
                index.segments.x1.to_numpy(),
                index.segments.y1.to_numpy(),
                index.segments.z1.to_numpy(),
            ]
        )
    )
    r_mid_d = jnp.asarray(index.segments.r_mid_au.to_numpy())
    return GeometryDevice(
        ro_d=ro_d, rd_d=rd_d, d_obs_d=d_obs_d, s0_d=s0_d, s1_d=s1_d, r_mid_d=r_mid_d
    )


@partial(jax.jit, static_argnames=("chunk_size",))
def run_distances_jax_scan(
    ro_d: jax.Array,
    rd_d: jax.Array,
    s0_d: jax.Array,
    s1_d: jax.Array,
    r_mid_d: jax.Array,
    d_obs_d: jax.Array,
    det_chunks: jax.Array,  # (C, W)
    seg_chunks: jax.Array,  # (C, W)
    theta_guard: float,
    *,
    chunk_size: int,
) -> tuple[jax.Array, jax.Array]:
    C = det_chunks.shape[0]
    distances_out = jnp.zeros((C, chunk_size), dtype=ro_d.dtype)
    within_out = jnp.zeros((C, chunk_size), dtype=jnp.bool_)

    def body(carry, i):
        distances_out, within_out = carry
        det_pad = jax.lax.dynamic_index_in_dim(det_chunks, i, keepdims=False)
        seg_pad = jax.lax.dynamic_index_in_dim(seg_chunks, i, keepdims=False)

        ro = jnp.take(ro_d, det_pad, axis=0)
        rd = jnp.take(rd_d, det_pad, axis=0)
        s0 = jnp.take(s0_d, seg_pad, axis=0)
        s1 = jnp.take(s1_d, seg_pad, axis=0)
        r_mid = jnp.take(r_mid_d, seg_pad, axis=0)
        d_obs = jnp.take(d_obs_d, det_pad, axis=0)

        distances, within = calc_ray_segment_distance_and_guard(
            ro, rd, s0, s1, r_mid, d_obs, theta_guard
        )
        distances_out = distances_out.at[i].set(distances)
        within_out = within_out.at[i].set(within)
        return (distances_out, within_out), None

    (distances_out, within_out), _ = jax.lax.scan(
        body, (distances_out, within_out), jnp.arange(C)
    )
    return distances_out, within_out


# Fast distance calculations and filtering using jax
@jax.jit
def _ray_segment_distance_single(
    ray_origin: jax.Array,
    ray_direction: jax.Array,
    seg_start: jax.Array,
    seg_end: jax.Array,
) -> jax.Array:
    """
    Compute minimum distance between a ray and a line segment (single pair).

    JAX-compatible implementation with stable numerics.

    Parameters
    ----------
    ray_origin : jax.Array (3,)
        Ray origin point
    ray_direction : jax.Array (3,)
        Ray direction vector (should be normalized)
    seg_start : jax.Array (3,)
        Segment start point
    seg_end : jax.Array (3,)
        Segment end point

    Returns
    -------
    distance : jax.Array (scalar)
        Minimum distance between ray and segment
    """
    # Type constants to input dtype
    dtype = ray_origin.dtype
    eps = jnp.array(1e-15, dtype=dtype)
    zero = jnp.array(0.0, dtype=dtype)
    one = jnp.array(1.0, dtype=dtype)

    # Vector from ray origin to segment start
    w0 = ray_origin - seg_start

    # Segment direction vector
    seg_dir = seg_end - seg_start
    seg_length_sq = jnp.dot(seg_dir, seg_dir)

    # Handle degenerate segment (point)
    is_degenerate = seg_length_sq < eps

    # For degenerate segments, compute distance from ray to point
    cross_prod = jnp.cross(ray_direction, w0)
    point_distance = jnp.linalg.norm(cross_prod)

    # For non-degenerate segments, compute closest approach parameters
    a = jnp.dot(ray_direction, ray_direction)  # Should be 1 if normalized
    b = jnp.dot(ray_direction, seg_dir)
    c = seg_length_sq
    d = jnp.dot(ray_direction, w0)
    e = jnp.dot(seg_dir, w0)

    denom = a * c - b * b

    # Handle parallel case
    is_parallel = jnp.abs(denom) < eps
    parallel_distance = jnp.linalg.norm(cross_prod)

    # For non-parallel segments, compute parameters
    t_ray = jnp.where(is_parallel, zero, (b * e - c * d) / denom)
    t_seg = jnp.where(is_parallel, zero, (a * e - b * d) / denom)

    # Clamp ray parameter to non-negative (ray, not line)
    t_ray = jnp.maximum(zero, t_ray)

    # Clamp segment parameter to [0, 1]
    t_seg = jnp.clip(t_seg, zero, one)

    # For clamped segments, recompute t_ray
    is_clamped = (t_seg == zero) | (t_seg == one)
    seg_point = seg_start + t_seg * seg_dir
    ray_to_seg = seg_point - ray_origin
    t_ray_recalc = jnp.maximum(zero, jnp.dot(ray_to_seg, ray_direction))
    t_ray = jnp.where(is_clamped, t_ray_recalc, t_ray)

    # Compute closest points and distance
    ray_point = ray_origin + t_ray * ray_direction
    seg_point = seg_start + t_seg * seg_dir
    segment_distance = jnp.linalg.norm(ray_point - seg_point)

    # Return appropriate distance based on segment type
    return jnp.where(
        is_degenerate,
        point_distance,
        jnp.where(is_parallel, parallel_distance, segment_distance),
    )


@jax.jit
def ray_segment_distance_window(
    ray_origins: jax.Array,
    ray_directions: jax.Array,
    seg_starts: jax.Array,
    seg_ends: jax.Array,
) -> jax.Array:
    """
    Pairwise distances for flattened pairs of size W.
    """
    distance_fn = jax.vmap(_ray_segment_distance_single, in_axes=(0, 0, 0, 0))
    return distance_fn(ray_origins, ray_directions, seg_starts, seg_ends)


@jax.jit
def calc_ray_segment_distance_and_guard(
    ray_origins: jax.Array,
    ray_directions: jax.Array,
    seg_starts: jax.Array,
    seg_ends: jax.Array,
    r_mid_au: jax.Array,
    d_obs_au: jax.Array,
    theta_guard: float,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute ray-to-segment distances and apply a guard threshold per pair.

    Parameters
    ----------
    ray_origins : jax.Array, shape (W, 3), dtype float64
        Origins of W rays in SSB Cartesian coordinates (AU).
    ray_directions : jax.Array, shape (W, 3), dtype float64
        Unit direction vectors for W rays. Not strictly required to be normalized,
        but distances assume consistent scaling; inputs are typically normalized.
    seg_starts : jax.Array, shape (W, 3), dtype float64
        Segment start points for each of the W pairs (x0, y0, z0) in AU.
    seg_ends : jax.Array, shape (W, 3), dtype float64
        Segment end points for each of the W pairs (x1, y1, z1) in AU.
    r_mid_au : jax.Array, shape (W,), dtype float64
        Segment midpoint heliocentric distances (AU), used for angular guard scaling.
    d_obs_au : jax.Array, shape (W,), dtype float64
        Observer heliocentric distances (AU) for each ray, used for angular guard scaling.
    theta_guard : float
        Angular guard in radians. The threshold distance per pair is
        theta_guard * max(r_mid_au, d_obs_au).

    Returns
    -------
    distances : jax.Array, shape (W,), dtype float64
        Minimum distance (AU) between each ray and its corresponding segment.
    within_guard : jax.Array, shape (W,), dtype bool
        Boolean mask indicating whether each pair is within the guard threshold.
    """
    distances = ray_segment_distance_window(
        ray_origins, ray_directions, seg_starts, seg_ends
    )
    thresholds = theta_guard * jnp.maximum(r_mid_au, d_obs_au)
    return distances, distances <= thresholds


def query_bvh_worker(
    index: BVHIndex,
    rays: ObservationRays,
    *,
    guard_arcmin: float = 0.65,
    window_size: int = 32768,
) -> tuple[OverlapHits, QueryBVHTelemetry]:
    if len(rays) == 0:
        raise ValueError("query_bvh_worker: rays is empty")
    if len(index.nodes) == 0 or len(index.prims) == 0:
        raise ValueError("query_bvh_worker: index has no nodes/primitives")

    # Perform coherence ordering should reduce branch divergence in packet traversal
    # this is because if similarl direction rays are grouped together, they will traverse
    # the tree similarly
    obs_codes = rays.observer.code.to_numpy(zero_copy_only=False)
    # direction bin (cheap)
    u_x = rays.u_x.to_numpy()
    u_y = rays.u_y.to_numpy()
    u_z = rays.u_z.to_numpy()
    dbin = _direction_bins(u_x, u_y, u_z)
    # stable bucket key: (obs_code, tbin, dbin)
    # Convert codes to integers via factorization
    _, obs_idx = np.unique(obs_codes, return_inverse=True)
    key = obs_idx.astype(np.int64) * 10_000_000 + dbin.astype(np.int64)
    order = np.argsort(key, kind="stable")
    rays = rays.take(order)
    # Drop ordering scratch to free memory
    del obs_codes, u_x, u_y, u_z, dbin, obs_idx, key, order

    # Stage 1–2: candidate indices (Numba traversal + host assembly)
    candidates, bvh_telemetry = find_bvh_matches(index, rays)
    cand = CandidateIndices(
        det_idx=candidates.det_index.to_numpy(),
        seg_idx=candidates.seg_row.to_numpy(),
        seg_id=candidates.seg_id.to_numpy(),
        leaf_id=candidates.leaf_id.to_numpy(),
    )
    del candidates

    if cand.det_idx.size == 0:
        return (
            OverlapHits.from_kwargs(
                det_id=[],
                orbit_id=[],
                seg_id=[],
                leaf_id=[],
                distance_au=[],
                query_guard_arcmin=float(guard_arcmin),
                query_batch_size=0,
                query_max_processes=0,
            ),
            QueryBVHTelemetry(
                truncation_occurred=bvh_telemetry.truncation_occurred,
                max_leaf_visits_observed=bvh_telemetry.max_leaf_visits_observed,
                rays_with_zero_candidates=bvh_telemetry.rays_with_zero_candidates,
                packets_traversed=bvh_telemetry.packets_traversed,
                pairs_total=bvh_telemetry.pairs_total,
                pairs_within_guard=0,
            ),
        )
    det_idx = cand.det_idx
    seg_rows = cand.seg_idx
    seg_ids_arr = cand.seg_id
    leaf_ids_arr = cand.leaf_id
    # Release wrapper now that arrays are extracted
    del cand

    N = int(len(det_idx))

    theta_guard = guard_arcmin * np.pi / (180.0 * 60.0)

    pairs_within_guard = 0

    # Stage 3: device-put geometry and run single JAX scan over chunks
    geom = materialize_geometry_device(index, rays)
    # Chunk and pad indices for fixed-size scan
    M = int(N)
    C = (M + window_size - 1) // window_size
    pad = C * window_size - M
    if pad:
        det_chunks = np.pad(det_idx, (0, pad))
        seg_chunks = np.pad(seg_rows, (0, pad))
    else:
        det_chunks = det_idx
        seg_chunks = seg_rows
    det_chunks = det_chunks.reshape(C, window_size)
    seg_chunks = seg_chunks.reshape(C, window_size)

    distances_c, within_c = run_distances_jax_scan(
        geom.ro_d,
        geom.rd_d,
        geom.s0_d,
        geom.s1_d,
        geom.r_mid_d,
        geom.d_obs_d,
        jnp.asarray(det_chunks),
        jnp.asarray(seg_chunks),
        theta_guard,
        chunk_size=window_size,
    )
    distances_np = np.asarray(distances_c).reshape(C * window_size)[:M]
    valid = np.asarray(within_c).reshape(C * window_size)[:M]
    # Release device/host buffers and chunked indices
    del distances_c, within_c, det_chunks, seg_chunks, geom
    if not np.any(valid):
        return (
            OverlapHits.from_kwargs(
                det_id=[],
                orbit_id=[],
                seg_id=[],
                leaf_id=[],
                distance_au=[],
                query_guard_arcmin=float(guard_arcmin),
                query_batch_size=0,
                query_max_processes=0,
            ),
            QueryBVHTelemetry(
                truncation_occurred=bvh_telemetry.truncation_occurred,
                max_leaf_visits_observed=bvh_telemetry.max_leaf_visits_observed,
                rays_with_zero_candidates=bvh_telemetry.rays_with_zero_candidates,
                packets_traversed=bvh_telemetry.packets_traversed,
                pairs_total=bvh_telemetry.pairs_total,
                pairs_within_guard=0,
            ),
        )

    # Keep only valid pairs; avoid intermediate list concatenations
    keep = np.nonzero(valid)[0]
    pairs_within_guard = int(keep.size)

    det_kept = det_idx[keep].astype(np.int32, copy=False)
    seg_rows_kept = seg_rows[keep]
    seg_id_kept = seg_ids_arr[keep].astype(np.int32, copy=False)
    leaf_id_kept = leaf_ids_arr[keep].astype(np.int32, copy=False)
    dist_kept = distances_np[keep].astype(np.float64, copy=False)

    # Arrow-take only kept orbit_ids instead of materializing full column to NumPy
    seg_keep_idx = pa.array(seg_rows_kept, type=pa.int64())
    hit_orbit_ids_arrow = index.segments.orbit_id.take(seg_keep_idx)
    # Drop full arrays now that kept subsets are materialized
    del det_idx, seg_rows, seg_ids_arr, leaf_ids_arr, distances_np, valid

    # Map indices back to string IDs and sort by (det_id, distance)
    # Map det_id via Arrow take and sort using Arrow to avoid large Python string arrays
    det_keep_idx = pa.array(det_kept, type=pa.int64())
    hit_det_ids_arrow = rays.det_id.take(det_keep_idx)
    sort_tbl = pa.table(
        {
            "det_id": hit_det_ids_arrow,
            "distance": pa.array(dist_kept),
        }
    )
    order_arrow = pc.sort_indices(
        sort_tbl, sort_keys=[("det_id", "ascending"), ("distance", "ascending")]
    )
    del sort_tbl

    det_sorted = pc.take(hit_det_ids_arrow, order_arrow)
    orbit_sorted = pc.take(hit_orbit_ids_arrow, order_arrow)
    del hit_det_ids_arrow  # no longer needed

    # Reorder numeric arrays using Arrow indices to avoid NumPy conversions
    seg_id_sorted = pc.take(pa.array(seg_id_kept), order_arrow)
    leaf_id_sorted = pc.take(pa.array(leaf_id_kept), order_arrow)
    dist_sorted = pc.take(pa.array(dist_kept), order_arrow)

    hits = OverlapHits.from_kwargs(
        det_id=det_sorted,
        orbit_id=orbit_sorted,
        seg_id=seg_id_sorted,
        leaf_id=leaf_id_sorted,
        distance_au=dist_sorted,
        query_guard_arcmin=float(guard_arcmin),
        query_batch_size=0,
        query_max_processes=0,
    )
    # Proactively release large intermediates before returning
    del det_sorted, orbit_sorted, seg_id_sorted, leaf_id_sorted, dist_sorted
    del det_kept, seg_rows_kept, seg_id_kept, leaf_id_kept, dist_kept
    del order_arrow, hit_orbit_ids_arrow, seg_keep_idx
    del rays
    return (
        hits,
        QueryBVHTelemetry(
            truncation_occurred=bvh_telemetry.truncation_occurred,
            max_leaf_visits_observed=bvh_telemetry.max_leaf_visits_observed,
            rays_with_zero_candidates=bvh_telemetry.rays_with_zero_candidates,
            packets_traversed=bvh_telemetry.packets_traversed,
            pairs_total=int(N),
            pairs_within_guard=int(pairs_within_guard),
        ),
    )


# ======================================
# TLAS-only routing (no shard access)
# ======================================


@njit(cache=True, fastmath=False)
def _packet_intersect_aabb_numba_minmax(
    ro_pkt: np.ndarray,
    rd_pkt: np.ndarray,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    mask_in: np.ndarray,
) -> np.ndarray:
    return _packet_intersect_aabb_numba(ro_pkt, rd_pkt, aabb_min, aabb_max, mask_in)


def _tlas_minmax_arrays(nodes: BVHNodes) -> tuple[np.ndarray, np.ndarray]:
    nodes_min = np.column_stack(
        [
            np.asarray(nodes.nodes_min_x, dtype=np.float32),
            np.asarray(nodes.nodes_min_y, dtype=np.float32),
            np.asarray(nodes.nodes_min_z, dtype=np.float32),
        ]
    )
    nodes_max = np.column_stack(
        [
            np.asarray(nodes.nodes_max_x, dtype=np.float32),
            np.asarray(nodes.nodes_max_y, dtype=np.float32),
            np.asarray(nodes.nodes_max_z, dtype=np.float32),
        ]
    )
    return nodes_min, nodes_max


def route_rays_to_shards(
    sharded: ShardedBVH,
    rays: ObservationRays,
    *,
    batch_size: int = 65536,
    packet_size: int = 64,
    max_shards_per_packet: Optional[int] = None,
) -> tuple[ShardAssignments, QueryBVHTelemetry]:
    """
    Traverse TLAS to assign each detection to one or more shard_ids.

    Returns ShardAssignments (det_id, shard_id) and telemetry summarizing routing.
    """
    if len(rays) == 0:
        return ShardAssignments.empty(), QueryBVHTelemetry(
            pairs_total=0,
            pairs_within_guard=0,
            truncation_occurred=False,
            max_leaf_visits_observed=0,
            rays_with_zero_candidates=0,
            packets_traversed=0,
        )

    tlas_nodes = sharded.tlas_nodes
    tlas_prims = sharded.tlas_prims

    nodes_min_np, nodes_max_np = _tlas_minmax_arrays(tlas_nodes)
    left_child_np = np.asarray(tlas_nodes.left_child, dtype=np.int32)
    right_child_np = np.asarray(tlas_nodes.right_child, dtype=np.int32)
    first_prim_np = np.asarray(tlas_nodes.first_prim, dtype=np.int32)
    prim_count_np = np.asarray(tlas_nodes.prim_count, dtype=np.int32)

    num_rays = len(rays)
    ro_x = rays.observer.coordinates.x.to_numpy()
    ro_y = rays.observer.coordinates.y.to_numpy()
    ro_z = rays.observer.coordinates.z.to_numpy()
    rd_x = rays.u_x.to_numpy()
    rd_y = rays.u_y.to_numpy()
    rd_z = rays.u_z.to_numpy()
    ray_origins_cpu = np.column_stack([ro_x, ro_y, ro_z]).astype(np.float32, copy=False)
    ray_directions_cpu = np.column_stack([rd_x, rd_y, rd_z]).astype(np.float32, copy=False)

    # Packetization
    num_rays_padded = ((num_rays + packet_size - 1) // packet_size) * packet_size
    ray_origins_padded = np.zeros((num_rays_padded, 3), dtype=np.float32)
    ray_directions_padded = np.zeros_like(ray_origins_padded)
    ray_origins_padded[:num_rays] = ray_origins_cpu
    ray_directions_padded[:num_rays] = ray_directions_cpu

    det_ids: list[str] = rays.det_id.to_pylist()
    out_det: List[str] = []
    out_shard: List[str] = []

    packets_traversed = 0
    max_shards_observed = 0
    capped = 0

    # We'll visit TLAS using the same packet traversal used for BVH
    for start in range(0, num_rays_padded, packet_size):
        end = start + packet_size
        P = min(packet_size, num_rays - start)
        ro_pkt = ray_origins_padded[start:end]
        rd_pkt = ray_directions_padded[start:end]

        visited_tlas_nodes, visited_masks, num_vis = packet_traverse_bvh_numba(
            nodes_min_np,
            nodes_max_np,
            left_child_np,
            right_child_np,
            ro_pkt,
            rd_pkt,
            int(max(16, int(np.ceil(np.log2(len(left_child_np) + 1))))),
            1_000_000,
        )
        visited_tlas_nodes = visited_tlas_nodes[:num_vis]
        visited_masks = visited_masks[:num_vis]

        if int(num_vis) == 0:
            continue
        packets_traversed += 1

        # Expand shard_ids per visited TLAS leaf
        per_ray_shards: List[set[str]] = [set() for _ in range(P)]
        for i, node_idx in enumerate(visited_tlas_nodes):
            mask = visited_masks[i][:P]
            if not np.any(mask):
                continue
            first = int(first_prim_np[node_idx])
            count = int(prim_count_np[node_idx])
            if count <= 0 or first < 0:
                continue
            # For TLAS, each primitive maps to a shard_id
            shard_slice = tlas_prims.shard_id.to_pylist()[first : first + count]
            rays_idx = np.nonzero(mask)[0]
            for r in rays_idx:
                if max_shards_per_packet is not None and max_shards_per_packet > 0:
                    if len(per_ray_shards[r]) >= max_shards_per_packet:
                        continue
                per_ray_shards[r].update(shard_slice)

        # Materialize output rows
        for r in range(P):
            shards_r = list(per_ray_shards[r])
            if not shards_r:
                continue
            if max_shards_per_packet is not None and max_shards_per_packet > 0:
                if len(shards_r) > max_shards_per_packet:
                    shards_r = shards_r[:max_shards_per_packet]
                    capped += 1
            det = det_ids[start + r]
            for sid in shards_r:
                out_det.append(det)
                out_shard.append(sid)

        max_shards_observed = max(
            max_shards_observed, max((len(s) for s in per_ray_shards), default=0)
        )

    assignments = ShardAssignments.from_kwargs(det_id=out_det, shard_id=out_shard)
    telemetry = QueryBVHTelemetry(
        pairs_total=len(assignments),
        pairs_within_guard=0,
        truncation_occurred=False,
        max_leaf_visits_observed=max_shards_observed,
        rays_with_zero_candidates=max(0, len(rays) - len(set(assignments.det_id.to_pylist()))),
        packets_traversed=packets_traversed,
    )
    return assignments, telemetry


def _route_rays_to_shards_chunk(
    tlas_nodes: BVHNodes,
    tlas_prims,
    rays: ObservationRays,
    *,
    packet_size: int,
    max_shards_per_packet: Optional[int],
) -> tuple[ShardAssignments, QueryBVHTelemetry]:
    nodes_min_np, nodes_max_np = _tlas_minmax_arrays(tlas_nodes)
    left_child_np = np.asarray(tlas_nodes.left_child, dtype=np.int32)
    right_child_np = np.asarray(tlas_nodes.right_child, dtype=np.int32)
    first_prim_np = np.asarray(tlas_nodes.first_prim, dtype=np.int32)
    prim_count_np = np.asarray(tlas_nodes.prim_count, dtype=np.int32)

    num_rays = len(rays)
    ro_x = rays.observer.coordinates.x.to_numpy()
    ro_y = rays.observer.coordinates.y.to_numpy()
    ro_z = rays.observer.coordinates.z.to_numpy()
    rd_x = rays.u_x.to_numpy()
    rd_y = rays.u_y.to_numpy()
    rd_z = rays.u_z.to_numpy()
    ray_origins_cpu = np.column_stack([ro_x, ro_y, ro_z]).astype(np.float32, copy=False)
    ray_directions_cpu = np.column_stack([rd_x, rd_y, rd_z]).astype(np.float32, copy=False)

    num_rays_padded = ((num_rays + packet_size - 1) // packet_size) * packet_size
    ray_origins_padded = np.zeros((num_rays_padded, 3), dtype=np.float32)
    ray_directions_padded = np.zeros_like(ray_origins_padded)
    ray_origins_padded[:num_rays] = ray_origins_cpu
    ray_directions_padded[:num_rays] = ray_directions_cpu

    det_ids: list[str] = rays.det_id.to_pylist()
    out_det: List[str] = []
    out_shard: List[str] = []

    packets_traversed = 0
    max_shards_observed = 0
    capped = 0

    for start in range(0, num_rays_padded, packet_size):
        end = start + packet_size
        P = min(packet_size, num_rays - start)
        ro_pkt = ray_origins_padded[start:end]
        rd_pkt = ray_directions_padded[start:end]

        visited_tlas_nodes, visited_masks, num_vis = packet_traverse_bvh_numba(
            nodes_min_np,
            nodes_max_np,
            left_child_np,
            right_child_np,
            ro_pkt,
            rd_pkt,
            int(max(16, int(np.ceil(np.log2(len(left_child_np) + 1))))),
            1_000_000,
        )
        visited_tlas_nodes = visited_tlas_nodes[:num_vis]
        visited_masks = visited_masks[:num_vis]

        if int(num_vis) == 0:
            continue
        packets_traversed += 1

        per_ray_shards: List[set[str]] = [set() for _ in range(P)]
        for i, node_idx in enumerate(visited_tlas_nodes):
            mask = visited_masks[i][:P]
            if not np.any(mask):
                continue
            first = int(first_prim_np[node_idx])
            count = int(prim_count_np[node_idx])
            if count <= 0 or first < 0:
                continue
            shard_slice = tlas_prims.shard_id.to_pylist()[first : first + count]
            rays_idx = np.nonzero(mask)[0]
            for r in rays_idx:
                if max_shards_per_packet is not None and max_shards_per_packet > 0:
                    if len(per_ray_shards[r]) >= max_shards_per_packet:
                        continue
                per_ray_shards[r].update(shard_slice)

        for r in range(P):
            shards_r = list(per_ray_shards[r])
            if not shards_r:
                continue
            if max_shards_per_packet is not None and max_shards_per_packet > 0:
                if len(shards_r) > max_shards_per_packet:
                    shards_r = shards_r[:max_shards_per_packet]
                    capped += 1
            det = det_ids[start + r]
            for sid in shards_r:
                out_det.append(det)
                out_shard.append(sid)

        max_shards_observed = max(
            max_shards_observed, max((len(s) for s in per_ray_shards), default=0)
        )

    assignments = ShardAssignments.from_kwargs(det_id=out_det, shard_id=out_shard)
    telemetry = QueryBVHTelemetry(
        pairs_total=len(assignments),
        pairs_within_guard=0,
        truncation_occurred=False,
        max_leaf_visits_observed=max_shards_observed,
        rays_with_zero_candidates=max(0, len(rays) - len(set(assignments.det_id.to_pylist()))),
        packets_traversed=packets_traversed,
    )
    return assignments, telemetry


@ray.remote
def route_rays_to_shards_chunk_remote(
    tlas_nodes: BVHNodes,
    tlas_prims,
    rays: ObservationRays,
    *,
    packet_size: int,
    max_shards_per_packet: Optional[int],
) -> tuple[ShardAssignments, QueryBVHTelemetry]:
    return _route_rays_to_shards_chunk(
        tlas_nodes,
        tlas_prims,
        rays,
        packet_size=packet_size,
        max_shards_per_packet=max_shards_per_packet,
    )


def _route_rays_to_shards_parallel(
    sharded: ShardedBVH,
    rays: ObservationRays,
    *,
    batch_size: int,
    packet_size: int,
    max_shards_per_packet: Optional[int],
    max_processes: int,
) -> tuple[ShardAssignments, QueryBVHTelemetry]:
    initialize_use_ray(num_cpus=max_processes)
    tlas_nodes_ref = ray.put(sharded.tlas_nodes)
    tlas_prims_ref = ray.put(sharded.tlas_prims)

    futures: list[ray.ObjectRef] = []
    assignments_parts: list[ShardAssignments] = []
    telemetry_agg = QueryBVHTelemetry(
        pairs_total=0,
        pairs_within_guard=0,
        truncation_occurred=False,
        max_leaf_visits_observed=0,
        rays_with_zero_candidates=0,
        packets_traversed=0,
    )
    max_active = max(1, int(1.5 * max_processes))
    for chunk in _iterate_chunks(rays, batch_size):
        fut = route_rays_to_shards_chunk_remote.remote(
            tlas_nodes_ref,
            tlas_prims_ref,
            chunk,
            packet_size=packet_size,
            max_shards_per_packet=max_shards_per_packet,
        )
        futures.append(fut)
        if len(futures) >= max_active:
            finished, futures = ray.wait(futures, num_returns=1)
            part, tel = ray.get(finished[0])
            assignments_parts.append(part)
            telemetry_agg = QueryBVHTelemetry(
                pairs_total=telemetry_agg.pairs_total + tel.pairs_total,
                pairs_within_guard=0,
                truncation_occurred=telemetry_agg.truncation_occurred or tel.truncation_occurred,
                max_leaf_visits_observed=max(telemetry_agg.max_leaf_visits_observed, tel.max_leaf_visits_observed),
                rays_with_zero_candidates=telemetry_agg.rays_with_zero_candidates + tel.rays_with_zero_candidates,
                packets_traversed=telemetry_agg.packets_traversed + tel.packets_traversed,
            )
    while futures:
        finished, futures = ray.wait(futures, num_returns=1)
        part, tel = ray.get(finished[0])
        assignments_parts.append(part)
        telemetry_agg = QueryBVHTelemetry(
            pairs_total=telemetry_agg.pairs_total + tel.pairs_total,
            pairs_within_guard=0,
            truncation_occurred=telemetry_agg.truncation_occurred or tel.truncation_occurred,
            max_leaf_visits_observed=max(telemetry_agg.max_leaf_visits_observed, tel.max_leaf_visits_observed),
            rays_with_zero_candidates=telemetry_agg.rays_with_zero_candidates + tel.rays_with_zero_candidates,
            packets_traversed=telemetry_agg.packets_traversed + tel.packets_traversed,
        )

    if not assignments_parts:
        return ShardAssignments.empty(), telemetry_agg
    assignments = qv.concatenate(assignments_parts, defrag=True)
    return assignments, telemetry_agg


def write_routed_rays_by_shard(
    sharded: ShardedBVH,
    rays: ObservationRays,
    output_dir: str,
    *,
    batch_size: int = 65536,
    packet_size: int = 64,
    max_open_writers: int = 256,
    max_shards_per_packet: Optional[int] = None,
) -> tuple[int, QueryBVHTelemetry]:
    """
    Stream TLAS routing and append rays into per-shard Parquet files at output_dir.

    Returns (total_rows_written, telemetry).
    """
    os.makedirs(output_dir, exist_ok=True)

    if len(rays) == 0:
        return 0, QueryBVHTelemetry(
            pairs_total=0,
            pairs_within_guard=0,
            truncation_occurred=False,
            max_leaf_visits_observed=0,
            rays_with_zero_candidates=0,
            packets_traversed=0,
        )

    tlas_nodes = sharded.tlas_nodes
    tlas_prims = sharded.tlas_prims

    nodes_min_np, nodes_max_np = _tlas_minmax_arrays(tlas_nodes)
    left_child_np = np.asarray(tlas_nodes.left_child, dtype=np.int32)
    right_child_np = np.asarray(tlas_nodes.right_child, dtype=np.int32)
    first_prim_np = np.asarray(tlas_nodes.first_prim, dtype=np.int32)
    prim_count_np = np.asarray(tlas_nodes.prim_count, dtype=np.int32)

    # Prepare packetized ray arrays
    num_rays = len(rays)
    ro = np.column_stack([
        rays.observer.coordinates.x.to_numpy(),
        rays.observer.coordinates.y.to_numpy(),
        rays.observer.coordinates.z.to_numpy(),
    ]).astype(np.float32, copy=False)
    rd = np.column_stack([
        rays.u_x.to_numpy(),
        rays.u_y.to_numpy(),
        rays.u_z.to_numpy(),
    ]).astype(np.float32, copy=False)

    # Setup Parquet writers (LRU)
    from pyarrow import parquet as pq
    import pyarrow as pa

    schema = ObservationRays.empty().table.schema
    class _Writer:
        def __init__(self, path: str):
            self.path = path
            self.writer = pq.ParquetWriter(path, schema)
            self.rows = 0
        def write(self, tbl: pa.Table) -> None:
            self.writer.write_table(tbl)
            self.rows += tbl.num_rows
        def close(self) -> None:
            self.writer.close()

    from collections import OrderedDict
    class _LRUWriters:
        def __init__(self, cap: int):
            self.cap = cap
            self.od: OrderedDict[str, _Writer] = OrderedDict()
        def get(self, sid: str) -> Optional[_Writer]:
            w = self.od.get(sid)
            if w is not None:
                self.od.move_to_end(sid)
            return w
        def put(self, sid: str, writer: _Writer) -> _Writer:
            if sid in self.od:
                self.od.move_to_end(sid)
                return self.od[sid]
            self.od[sid] = writer
            if len(self.od) > self.cap:
                old_sid, old_w = self.od.popitem(last=False)
                try:
                    old_w.close()
                except Exception:
                    pass
            return writer
        def close_all(self) -> None:
            for _, w in list(self.od.items()):
                try:
                    w.close()
                except Exception:
                    pass
            self.od.clear()

    writers = _LRUWriters(max_open_writers)

    num_rays_padded = ((num_rays + packet_size - 1) // packet_size) * packet_size
    ro_pad = np.zeros((num_rays_padded, 3), dtype=np.float32)
    rd_pad = np.zeros_like(ro_pad)
    ro_pad[:num_rays] = ro
    rd_pad[:num_rays] = rd

    packets_traversed = 0
    max_shards_observed = 0
    rows_written = 0

    for start in range(0, num_rays_padded, packet_size):
        end = start + packet_size
        P = min(packet_size, num_rays - start)
        ro_pkt = ro_pad[start:end]
        rd_pkt = rd_pad[start:end]

        visited_tlas_nodes, visited_masks, num_vis = packet_traverse_bvh_numba(
            nodes_min_np,
            nodes_max_np,
            left_child_np,
            right_child_np,
            ro_pkt,
            rd_pkt,
            int(max(16, int(np.ceil(np.log2(len(left_child_np) + 1))))),
            1_000_000,
        )
        visited_tlas_nodes = visited_tlas_nodes[:num_vis]
        visited_masks = visited_masks[:num_vis]

        if int(num_vis) == 0:
            continue
        packets_traversed += 1

        # Collect per-ray shard ids
        per_ray_shards: List[List[str]] = [[] for _ in range(P)]
        for i, node_idx in enumerate(visited_tlas_nodes):
            mask = visited_masks[i][:P]
            if not np.any(mask):
                continue
            first = int(first_prim_np[node_idx])
            count = int(prim_count_np[node_idx])
            if count <= 0 or first < 0:
                continue
            shard_slice = tlas_prims.shard_id.to_pylist()[first : first + count]
            for r in np.nonzero(mask)[0]:
                lst = per_ray_shards[r]
                if max_shards_per_packet is not None and max_shards_per_packet > 0:
                    if len(lst) < max_shards_per_packet:
                        lst.extend(shard_slice)
                else:
                    lst.extend(shard_slice)

        # Unique per-shard and batch write by shard
        packet_tbl = rays[start : start + P].table
        # Build per-shard row indices
        shard_to_rows: dict[str, list[int]] = {}
        for r, sid_list in enumerate(per_ray_shards):
            if not sid_list:
                continue
            uniq = list(dict.fromkeys(sid_list))
            max_shards_observed = max(max_shards_observed, len(uniq))
            for sid in uniq:
                shard_to_rows.setdefault(sid, []).append(r)
        for sid, rows in shard_to_rows.items():
            shard_dir = os.path.join(output_dir, sid)
            os.makedirs(shard_dir, exist_ok=True)
            path = os.path.join(shard_dir, "rays.parquet")
            w = writers.get(sid)
            if w is None:
                w = writers.put(sid, _Writer(path))
            idx_arr = pa.array(rows, type=pa.int64())
            sub_tbl = packet_tbl.take(idx_arr)
            w.write(sub_tbl)
            rows_written += sub_tbl.num_rows

    writers.close_all()

    telemetry = QueryBVHTelemetry(
        pairs_total=rows_written,
        pairs_within_guard=0,
        truncation_occurred=False,
        max_leaf_visits_observed=max_shards_observed,
        rays_with_zero_candidates=0,
        packets_traversed=packets_traversed,
    )
    return rows_written, telemetry


def query_bvh_sharded(
    sharded: ShardedBVH,
    rays: ObservationRays,
    *,
    guard_arcmin: float = 0.65,
    resolver: FilesystemShardResolver,
    batch_size: int = 65536,
    window_size: int = 32768,
    packet_size: int = 64,
    max_shards_per_packet: Optional[int] = 8,
    max_processes: int = 1,
) -> tuple[OverlapHits, QueryBVHTelemetry]:
    """
    Single-node orchestration: TLAS-route rays, query only referenced shards, merge hits.
    """
    if max_processes is None or max_processes <= 1:
        assignments, _ = route_rays_to_shards(
            sharded,
            rays,
            batch_size=batch_size,
            packet_size=packet_size,
            max_shards_per_packet=max_shards_per_packet,
        )
    else:
        assignments, _ = _route_rays_to_shards_parallel(
            sharded,
            rays,
            batch_size=batch_size,
            packet_size=packet_size,
            max_shards_per_packet=max_shards_per_packet,
            max_processes=max_processes,
        )
    if len(assignments) == 0:
        return OverlapHits.empty(), QueryBVHTelemetry(
            pairs_total=0,
            pairs_within_guard=0,
            truncation_occurred=False,
            max_leaf_visits_observed=0,
            rays_with_zero_candidates=len(rays),
            packets_traversed=0,
        )

    # Group det_ids per shard
    import pyarrow as pa
    tbl = assignments.table
    # Get unique shard ids
    shard_ids = list(dict.fromkeys(assignments.shard_id.to_pylist()))
    hits_out: list[OverlapHits] = []
    telemetry_agg = QueryBVHTelemetry(
        pairs_total=0,
        pairs_within_guard=0,
        truncation_occurred=False,
        max_leaf_visits_observed=0,
        rays_with_zero_candidates=0,
        packets_traversed=0,
    )
    for sid in shard_ids:
        logger.info(f"Querying shard {sid}")
        mask = pa.compute.equal(tbl["shard_id"], pa.scalar(sid))
        det_ids_sid = pa.compute.take(tbl["det_id"], pa.compute.indices_nonzero(mask))
        # det_list = det_ids_sid.to_pylist()
        # Filter rays for this shard
        mask_det = pa.compute.is_in(rays.det_id, det_ids_sid)
        rays_sid = rays.apply_mask(mask_det)
        if len(rays_sid) == 0:
            continue
        index = resolver.resolve(sid)
        hits_sid, telemetry = query_bvh(
            index,
            rays_sid,
            guard_arcmin=0.65,
            batch_size=batch_size,
            window_size=window_size,
            max_processes=max_processes,
        )
        hits_out.append(hits_sid)
        telemetry_agg = QueryBVHTelemetry(
            pairs_total=telemetry_agg.pairs_total + telemetry.pairs_total,
            pairs_within_guard=telemetry_agg.pairs_within_guard + telemetry.pairs_within_guard,
            truncation_occurred=telemetry_agg.truncation_occurred or telemetry.truncation_occurred,
            max_leaf_visits_observed=max(telemetry_agg.max_leaf_visits_observed, telemetry.max_leaf_visits_observed),
            rays_with_zero_candidates=telemetry_agg.rays_with_zero_candidates + telemetry.rays_with_zero_candidates,
            packets_traversed=telemetry_agg.packets_traversed + telemetry.packets_traversed,
        )

    if not hits_out:
        return OverlapHits.empty(), telemetry_agg
    out = qv.concatenate(hits_out)
    return out, telemetry_agg


query_bvh_worker_remote = ray.remote(query_bvh_worker)


def query_bvh(
    index: BVHIndex,
    rays: ObservationRays,
    guard_arcmin: float = 0.65,
    batch_size: int = 65536,
    window_size: int = 32768,
    max_processes: int = 0,
) -> tuple[OverlapHits, QueryBVHTelemetry]:
    if len(rays) == 0:
        raise ValueError("query_bvh: rays is empty")
    if len(index.nodes) == 0 or len(index.prims) == 0:
        raise ValueError("query_bvh: index has no nodes/primitives")
    telemetry_agg = QueryBVHTelemetry(
        pairs_total=0,
        pairs_within_guard=0,
        truncation_occurred=False,
        max_leaf_visits_observed=0,
        rays_with_zero_candidates=0,
        packets_traversed=0,
    )

    # Stage 0: moved to worker; avoid full-table materialization here

    if max_processes is None or max_processes <= 1:
        results: list[OverlapHits] = []
        for ray_chunk in _iterate_chunks(rays, batch_size):
            hits, telemetry = query_bvh_worker(
                index,
                ray_chunk,
                guard_arcmin=guard_arcmin,
                window_size=window_size,
            )
            results.append(hits)
            telemetry_agg = QueryBVHTelemetry(
                pairs_total=telemetry_agg.pairs_total + telemetry.pairs_total,
                pairs_within_guard=telemetry_agg.pairs_within_guard
                + telemetry.pairs_within_guard,
                truncation_occurred=telemetry_agg.truncation_occurred
                or telemetry.truncation_occurred,
                max_leaf_visits_observed=max(
                    telemetry_agg.max_leaf_visits_observed,
                    telemetry.max_leaf_visits_observed,
                ),
                rays_with_zero_candidates=telemetry_agg.rays_with_zero_candidates
                + telemetry.rays_with_zero_candidates,
                packets_traversed=telemetry_agg.packets_traversed
                + telemetry.packets_traversed,
            )

        results = qv.concatenate(results)

        # Set the current query attributes
        out = OverlapHits.from_kwargs(
            det_id=results.det_id,
            orbit_id=results.orbit_id,
            seg_id=results.seg_id,
            leaf_id=results.leaf_id,
            distance_au=results.distance_au,
            query_guard_arcmin=float(guard_arcmin),
            query_batch_size=int(batch_size),
            query_max_processes=int(max_processes),
        )
        return out, telemetry_agg

    else:
        initialize_use_ray(num_cpus=max_processes)
        index_ref = ray.put(index)
        futures: list[ray.ObjectRef] = []
        out: list[OverlapHits] = []
        max_active = max(1, int(1.5 * max_processes))
        for ray_chunk in _iterate_chunks(rays, batch_size):
            fut = query_bvh_worker_remote.remote(
                index_ref,
                ray_chunk,
                guard_arcmin=guard_arcmin,
                window_size=window_size,
            )
            futures.append(fut)
            if len(futures) >= max_active:
                finished, futures = ray.wait(futures, num_returns=1)
                hits_part, telemetry = ray.get(finished[0])
                out.append(hits_part)
                telemetry_agg = QueryBVHTelemetry(
                    pairs_total=telemetry_agg.pairs_total + telemetry.pairs_total,
                    pairs_within_guard=telemetry_agg.pairs_within_guard
                    + telemetry.pairs_within_guard,
                    truncation_occurred=telemetry_agg.truncation_occurred
                    or telemetry.truncation_occurred,
                    max_leaf_visits_observed=max(
                        telemetry_agg.max_leaf_visits_observed,
                        telemetry.max_leaf_visits_observed,
                    ),
                    rays_with_zero_candidates=telemetry_agg.rays_with_zero_candidates
                    + telemetry.rays_with_zero_candidates,
                    packets_traversed=telemetry_agg.packets_traversed
                    + telemetry.packets_traversed,
                )
        while futures:
            finished, futures = ray.wait(futures, num_returns=1)
            hits_part, telemetry = ray.get(finished[0])
            out.append(hits_part)
            telemetry_agg = QueryBVHTelemetry(
                pairs_total=telemetry_agg.pairs_total + telemetry.pairs_total,
                pairs_within_guard=telemetry_agg.pairs_within_guard
                + telemetry.pairs_within_guard,
                truncation_occurred=telemetry_agg.truncation_occurred
                or telemetry.truncation_occurred,
                max_leaf_visits_observed=max(
                    telemetry_agg.max_leaf_visits_observed,
                    telemetry.max_leaf_visits_observed,
                ),
                rays_with_zero_candidates=telemetry_agg.rays_with_zero_candidates
                + telemetry.rays_with_zero_candidates,
                packets_traversed=telemetry_agg.packets_traversed
                + telemetry.packets_traversed,
            )

        out = qv.concatenate(out)
        # Set the current query attributes
        out = OverlapHits.from_kwargs(
            det_id=out.det_id,
            orbit_id=out.orbit_id,
            seg_id=out.seg_id,
            leaf_id=out.leaf_id,
            distance_au=out.distance_au,
            query_guard_arcmin=float(guard_arcmin),
            query_batch_size=int(batch_size),
            query_max_processes=int(max_processes),
        )
        return out, telemetry_agg
