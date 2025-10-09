"""
Utilities to filter an existing BVH index to a single orbit and prepare
lightweight arrays for a 3D viewer.

Inputs are Quivr-first (BVHIndex, ObservationRays), and outputs are NumPy
arrays suitable for direct upload to a WebGL viewer. No BVH rebuilding occurs
here; we only select and aggregate from the given index.
"""

from __future__ import annotations

from collections import deque
from typing import Optional, TypedDict

import numpy as np
import numpy.typing as npt
import pyarrow.compute as pc

from ...rays import ObservationRays
from .. import BVHIndex
from .. import OverlapHits



class OrbitNodesResult(TypedDict):
    """Typed result for orbit-filtered node data."""

    node_indices_i32: npt.NDArray[np.int32]
    node_depth_i32: npt.NDArray[np.int32]
    nodes_min_f32: npt.NDArray[np.float32]
    nodes_max_f32: npt.NDArray[np.float32]


class ViewData(TypedDict, total=False):
    """Typed payload for viewer consumption (single-orbit)."""

    segments_endpoints_f32: npt.NDArray[np.float32]  # (M, 6)
    nodes_min_f32: npt.NDArray[np.float32]  # (K, 3)
    nodes_max_f32: npt.NDArray[np.float32]  # (K, 3)
    node_depth_i32: npt.NDArray[np.int32]  # (K,)
    node_indices_i32: npt.NDArray[np.int32]  # (K,)
    rays_origins_f32: npt.NDArray[np.float32]  # (R, 3)
    rays_dirs_f32: npt.NDArray[np.float32]  # (R, 3)
    bounds_sphere_f32: npt.NDArray[np.float32]  # (4,) center_xyz, radius
    # Optional ray metadata
    rays_station_codes: list[str]
    rays_det_ids: list[str]
    rays_hit_mask: npt.NDArray[np.bool_]


def rays_to_numpy_arrays(
    rays: ObservationRays,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert ObservationRays to NumPy arrays using vectorized column access.
    """
    ro_x = rays.observer.coordinates.x.to_numpy(zero_copy_only=False)
    ro_y = rays.observer.coordinates.y.to_numpy(zero_copy_only=False)
    ro_z = rays.observer.coordinates.z.to_numpy(zero_copy_only=False)
    ray_origins = np.column_stack([ro_x, ro_y, ro_z]).astype(np.float64, copy=False)

    rd_x = rays.u_x.to_numpy(zero_copy_only=False)
    rd_y = rays.u_y.to_numpy(zero_copy_only=False)
    rd_z = rays.u_z.to_numpy(zero_copy_only=False)
    ray_directions = np.column_stack([rd_x, rd_y, rd_z]).astype(np.float64, copy=False)

    # Prefer coordinates' precomputed radial magnitude for distances
    # Note: r_mag is already a NumPy array in current schema
    observer_distances = rays.observer.coordinates.r_mag

    return ray_origins, ray_directions, observer_distances


def bvh_primitive_mask_for_orbit(
    index: BVHIndex, orbit_id: str
) -> npt.NDArray[np.bool_]:
    """
    Build a boolean mask over BVH primitives indicating membership in a selected orbit.

    Parameters
    ----------
    index : BVHIndex
        Existing BVH index built over many orbits.
    orbit_id : str
        The orbit identifier to select.

    Returns
    -------
    np.ndarray of bool, shape (num_primitives,)
        True for primitives (segments) belonging to the given orbit.
    """
    # Primitive array maps back to row indices in the segments table
    prim_row_index = np.asarray(index.prims.segment_row_index, dtype=np.int32)

    # Vectorized mask over segments using Arrow compute
    seg_mask_pa = pc.equal(index.segments.orbit_id, orbit_id)
    try:
        seg_mask_np = np.asarray(seg_mask_pa.to_numpy(zero_copy_only=False))
    except Exception:
        seg_mask_np = np.array(seg_mask_pa.to_pylist(), dtype=bool)

    # Map segment membership to primitive order
    prim_mask = seg_mask_np[prim_row_index]
    return prim_mask


def bvh_nodes_for_orbit(
    index: BVHIndex,
    prim_mask: npt.NDArray[np.bool_],
    *,
    tight_aabbs: bool = True,
) -> OrbitNodesResult:
    """
    Select BVH nodes that contain at least one primitive from the selected orbit.

    Optionally compute tight node AABBs for those nodes using only the selected
    primitives; otherwise reuse the original node bounds.

    Parameters
    ----------
    index : BVHIndex
        Existing BVH index built over many orbits.
    prim_mask : np.ndarray of bool, shape (num_primitives,)
        Mask over BVH primitives for the selected orbit.
    tight_aabbs : bool, default True
        When True, recompute node bounds using only selected primitives.

    Returns
    -------
    OrbitNodesResult
        node indices, depths, and min/max arrays (float32) for selected nodes.
    """
    # Extract arrays directly from BVHIndex tables (no intermediate wrapper)
    nodes = index.nodes
    prims = index.prims

    left_child = np.asarray(nodes.left_child, dtype=np.int32)
    right_child = np.asarray(nodes.right_child, dtype=np.int32)
    is_leaf = nodes.is_leaf_numpy().astype(bool, copy=False)
    first_prim = np.asarray(nodes.first_prim, dtype=np.int32)
    prim_count = np.asarray(nodes.prim_count, dtype=np.int32)
    nodes_min_orig, nodes_max_orig = nodes.min_max_numpy()
    prim_row_index = np.asarray(prims.segment_row_index, dtype=np.int32)

    num_nodes = int(left_child.shape[0])

    # Post-order traversal to compute selection bottom-up
    post_order: list[int] = []
    stack: list[tuple[int, bool]] = [(0, False)]  # (node_idx, visited)
    while stack:
        node, visited = stack.pop()
        if node < 0:
            continue
        if visited:
            post_order.append(node)
        else:
            stack.append((node, True))
            # Push children
            rc = int(right_child[node])
            lc = int(left_child[node])
            if rc >= 0:
                stack.append((rc, False))
            if lc >= 0:
                stack.append((lc, False))

    node_selected = np.zeros(num_nodes, dtype=bool)

    # Helper to test whether a leaf has any selected primitives
    def _leaf_has_selected(node_idx: int) -> bool:
        start = int(first_prim[node_idx])
        count = int(prim_count[node_idx])
        if start < 0 or count <= 0:
            return False
        sl = slice(start, start + count)
        return bool(np.any(prim_mask[sl]))

    for node in post_order:
        if is_leaf[node]:
            node_selected[node] = _leaf_has_selected(node)
        else:
            lc = int(left_child[node])
            rc = int(right_child[node])
            sel = False
            if lc >= 0:
                sel = sel or node_selected[lc]
            if rc >= 0:
                sel = sel or node_selected[rc]
            node_selected[node] = sel

    selected_indices = np.flatnonzero(node_selected).astype(np.int32)

    # Depth for all nodes via BFS (root depth = 0)
    node_depth = np.full(num_nodes, -1, dtype=np.int32)
    q: deque[int] = deque([0])
    node_depth[0] = 0
    while q:
        node = q.popleft()
        d = int(node_depth[node])
        lc = int(left_child[node])
        rc = int(right_child[node])
        if lc >= 0 and node_depth[lc] == -1:
            node_depth[lc] = d + 1
            q.append(lc)
        if rc >= 0 and node_depth[rc] == -1:
            node_depth[rc] = d + 1
            q.append(rc)

    if not tight_aabbs:
        nodes_min = nodes_min_orig[selected_indices].astype(np.float32, copy=False)
        nodes_max = nodes_max_orig[selected_indices].astype(np.float32, copy=False)
        return {
            "node_indices_i32": selected_indices,
            "node_depth_i32": node_depth[selected_indices],
            "nodes_min_f32": nodes_min,
            "nodes_max_f32": nodes_max,
        }

    # Tight AABBs: compute per-leaf bounds over selected primitives, then union upwards
    # Prepare segment AABB arrays from endpoints on demand
    from ....orbits.polyline import compute_segment_aabbs

    seg_min_x, seg_min_y, seg_min_z, seg_max_x, seg_max_y, seg_max_z = (
        compute_segment_aabbs(
            index.segments,
            guard_arcmin=float(index.nodes.aabb_guard_arcmin),
            epsilon_n_au=float(index.nodes.aabb_epsilon_n_au),
            padding_method=str(index.nodes.aabb_padding_method),
        )
    )

    # Storage for tight bounds per node (only filled for selected nodes)
    tight_min = np.zeros((num_nodes, 3), dtype=np.float64)
    tight_max = np.zeros((num_nodes, 3), dtype=np.float64)
    tight_valid = np.zeros(num_nodes, dtype=bool)

    # First pass: leaves
    for node in post_order:
        if not node_selected[node]:
            continue
        if is_leaf[node]:
            start = int(first_prim[node])
            count = int(prim_count[node])
            if start < 0 or count <= 0:
                continue
            sl = slice(start, start + count)
            mask_slice = prim_mask[sl]
            if not np.any(mask_slice):
                continue
            row_idx = prim_row_index[sl][mask_slice].astype(np.int64, copy=False)
            # Gather segment bounds and reduce
            mn = np.array(
                [
                    np.min(seg_min_x[row_idx]),
                    np.min(seg_min_y[row_idx]),
                    np.min(seg_min_z[row_idx]),
                ]
            )
            mx = np.array(
                [
                    np.max(seg_max_x[row_idx]),
                    np.max(seg_max_y[row_idx]),
                    np.max(seg_max_z[row_idx]),
                ]
            )
            tight_min[node] = mn
            tight_max[node] = mx
            tight_valid[node] = True

    # Second pass: internals (post-order ensures children are ready)
    for node in post_order:
        if not node_selected[node] or is_leaf[node]:
            continue
        lc = int(left_child[node])
        rc = int(right_child[node])
        have_l = lc >= 0 and tight_valid[lc]
        have_r = rc >= 0 and tight_valid[rc]

        if have_l and have_r:
            tight_min[node] = np.minimum(tight_min[lc], tight_min[rc])
            tight_max[node] = np.maximum(tight_max[lc], tight_max[rc])
            tight_valid[node] = True
        elif have_l:
            tight_min[node] = tight_min[lc]
            tight_max[node] = tight_max[lc]
            tight_valid[node] = True
        elif have_r:
            tight_min[node] = tight_min[rc]
            tight_max[node] = tight_max[rc]
            tight_valid[node] = True
        else:
            # Fallback: if neither child had selected prims (should not happen if selected),
            # use original bounds to avoid returning invalid values.
            tight_min[node] = nodes_min_orig[node]
            tight_max[node] = nodes_max_orig[node]
            tight_valid[node] = True

    nodes_min = tight_min[selected_indices].astype(np.float32, copy=False)
    nodes_max = tight_max[selected_indices].astype(np.float32, copy=False)

    return {
        "node_indices_i32": selected_indices,
        "node_depth_i32": node_depth[selected_indices],
        "nodes_min_f32": nodes_min,
        "nodes_max_f32": nodes_max,
    }


def orbit_segments_endpoints(index: BVHIndex, orbit_id: str) -> npt.NDArray[np.float32]:
    """
    Extract segment endpoints for a single orbit from the BVH index segments.

    Parameters
    ----------
    index : BVHIndex
        Existing BVH index.
    orbit_id : str
        Orbit identifier to filter.

    Returns
    -------
    np.ndarray, float32, shape (M, 6)
        [x0, y0, z0, x1, y1, z1] for each segment belonging to the orbit.
    """
    segs = index.segments
    # Arrow mask by orbit_id equality (avoids Python loops)
    seg_mask_pa = pc.equal(segs.orbit_id, orbit_id)
    segs_sel = segs.apply_mask(seg_mask_pa)
    if len(segs_sel) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    x0 = segs_sel.x0.to_numpy(zero_copy_only=False)
    y0 = segs_sel.y0.to_numpy(zero_copy_only=False)
    z0 = segs_sel.z0.to_numpy(zero_copy_only=False)
    x1 = segs_sel.x1.to_numpy(zero_copy_only=False)
    y1 = segs_sel.y1.to_numpy(zero_copy_only=False)
    z1 = segs_sel.z1.to_numpy(zero_copy_only=False)

    out = (
        np.column_stack([x0, y0, z0, x1, y1, z1]).astype(np.float32, copy=False)
    )
    return out


def select_rays(
    rays: ObservationRays, *, max_rays: Optional[int] = None
) -> ObservationRays:
    """
    Return the first N rays if a cap is provided; otherwise return as-is.
    """
    if max_rays is None or max_rays <= 0 or len(rays) <= max_rays:
        return rays
    return rays[: int(max_rays)]


def _compute_bounds_sphere(
    segments_endpoints_f32: npt.NDArray[np.float32],
    nodes_min_f32: npt.NDArray[np.float32],
    nodes_max_f32: npt.NDArray[np.float32],
    rays_origins_f32: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """
    Compute a simple bounding sphere [cx, cy, cz, r] from available geometry.

    Uses axis-aligned bounds across segment endpoints, node boxes, and ray origins.
    """
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []

    if segments_endpoints_f32.size:
        # Reshape to points (2 per segment)
        pts = segments_endpoints_f32.reshape(-1, 3)  # (2M, 3)
        mins.append(np.min(pts, axis=0))
        maxs.append(np.max(pts, axis=0))

    if nodes_min_f32.size:
        mins.append(np.min(nodes_min_f32, axis=0))
    if nodes_max_f32.size:
        maxs.append(np.max(nodes_max_f32, axis=0))

    if rays_origins_f32.size:
        mins.append(np.min(rays_origins_f32, axis=0))
        maxs.append(np.max(rays_origins_f32, axis=0))

    if not mins or not maxs:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    mn = np.min(np.stack(mins, axis=0), axis=0)
    mx = np.max(np.stack(maxs, axis=0), axis=0)
    center = (mn + mx) * 0.5
    radius = float(np.linalg.norm(mx - mn) * 0.5)
    return np.array([center[0], center[1], center[2], radius], dtype=np.float32)


def prepare_view_data_single_orbit(
    index: BVHIndex,
    orbit_id: str,
    rays: Optional[ObservationRays] = None,
    *,
    max_rays: Optional[int] = 1000,
    tight_aabbs: bool = True,
    hits: Optional[OverlapHits] = None,
) -> ViewData:
    """
    Prepare arrays for a viewer: single-orbit segments, correlated BVH nodes, and rays.

    No BVH construction is performed; data are filtered from the given index.
    Arrays are returned in float32/int32 for compact transport.
    """
    # Segments for selected orbit
    segments_endpoints_f32 = orbit_segments_endpoints(index, orbit_id)

    # Nodes for selected orbit
    prim_mask = bvh_primitive_mask_for_orbit(index, orbit_id)
    nodes = bvh_nodes_for_orbit(index, prim_mask, tight_aabbs=tight_aabbs)

    # Rays (optional)
    if rays is not None:
        rays_capped = select_rays(rays, max_rays=max_rays)
        ro, rd, _ = rays_to_numpy_arrays(rays_capped)
        rays_origins_f32 = np.asarray(ro, dtype=np.float32)
        rays_dirs_f32 = np.asarray(rd, dtype=np.float32)
        # Optional metadata for coloring and hit/miss filtering
        rays_station_codes = rays_capped.observer.code.to_pylist()
        rays_det_ids = rays_capped.det_id.to_pylist()
    else:
        rays_origins_f32 = np.zeros((0, 3), dtype=np.float32)
        rays_dirs_f32 = np.zeros((0, 3), dtype=np.float32)
        rays_station_codes = []
        rays_det_ids = []

    # Bounds sphere for camera fit
    bounds_sphere_f32 = _compute_bounds_sphere(
        segments_endpoints_f32,
        nodes["nodes_min_f32"],
        nodes["nodes_max_f32"],
        rays_origins_f32,
    )

    view: ViewData = {
        "segments_endpoints_f32": segments_endpoints_f32,
        "nodes_min_f32": nodes["nodes_min_f32"],
        "nodes_max_f32": nodes["nodes_max_f32"],
        "node_depth_i32": nodes["node_depth_i32"],
        "node_indices_i32": nodes["node_indices_i32"],
        "rays_origins_f32": rays_origins_f32,
        "rays_dirs_f32": rays_dirs_f32,
        "bounds_sphere_f32": bounds_sphere_f32,
    }

    # Attach optional ray metadata if available
    if rays_det_ids:
        view["rays_det_ids"] = rays_det_ids
        view["rays_station_codes"] = rays_station_codes

    # Optional: attach hit mask if hits provided
    if hits is not None and rays_det_ids:
        # Only consider hits against the selected orbit (Arrow filter then set membership)
        orbit_mask = pc.equal(hits.orbit_id, orbit_id)
        hits_sel = hits.apply_mask(orbit_mask)
        # Convert selected det_ids to a numpy hash set
        hit_det_ids = set(hits_sel.det_id.to_pylist())
        hit_mask = np.fromiter((det in hit_det_ids for det in rays_det_ids), dtype=bool)
        view["rays_hit_mask"] = hit_mask

    return view
