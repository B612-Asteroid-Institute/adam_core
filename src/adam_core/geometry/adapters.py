"""
Adapters between quivr/pyarrow types and JAX-native arrays.

This module provides zero-copy conversions where possible and handles
the boundary between the existing adam-core types and the new JAX-optimized
data structures.
"""

from __future__ import annotations

import logging
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import quivr as qv

from ..observations.rays import ObservationRays
from ..orbits.polyline import OrbitPolylineSegments
from .anomaly import AnomalyLabels
from .jax_types import AnomalyLabelsSOA, BVHArrays, HitsSOA, OrbitIdMapping, SegmentsSOA
from .overlap import OverlapHits

__all__ = [
    ## shard adapters removed; keep Quivr-first
    "segments_to_soa",
    "segments_to_numpy_soa",
    "segments_soa_to_segments",
    "rays_to_numpy_arrays",
    "hits_soa_to_overlap_hits",
    "overlap_hits_to_soa",
    "hits_soa_to_anomaly_labels_soa",
    "anomaly_labels_soa_to_anomaly_labels",
]

logger = logging.getLogger(__name__)


## (removed shard adapters; keep Quivr tables until kernel boundaries)


def segments_to_soa(
    segments: OrbitPolylineSegments,
    device: Optional[jax.Device] = None,
    include_normals: bool = False,
) -> SegmentsSOA:
    """
    Convert OrbitPolylineSegments to structure-of-arrays format.

    Parameters
    ----------
    segments : OrbitPolylineSegments
        Quivr table with segment data
    device : jax.Device, optional
        JAX device to place arrays on (default: CPU)
    include_normals : bool, default=False
        Whether to include orbital plane normals

    Returns
    -------
    SegmentsSOA
        JAX-native segments representation
    """
    # Compute stable per-segment orbit index using first-occurrence order
    orbit_ids_list = segments.orbit_id.to_pylist()
    unique_ids_in_order = list(dict.fromkeys(orbit_ids_list))
    id_to_idx = {oid: i for i, oid in enumerate(unique_ids_in_order)}
    orbit_id_index_np = np.array(
        [id_to_idx[oid] for oid in orbit_ids_list], dtype=np.int32
    )

    # Extract arrays with zero-copy where possible
    # PyArrow -> NumPy is zero-copy for compatible dtypes
    soa = SegmentsSOA(
        x0=np.asarray(segments.x0.to_numpy(zero_copy_only=False)),
        y0=np.asarray(segments.y0.to_numpy(zero_copy_only=False)),
        z0=np.asarray(segments.z0.to_numpy(zero_copy_only=False)),
        x1=np.asarray(segments.x1.to_numpy(zero_copy_only=False)),
        y1=np.asarray(segments.y1.to_numpy(zero_copy_only=False)),
        z1=np.asarray(segments.z1.to_numpy(zero_copy_only=False)),
        r_mid_au=np.asarray(segments.r_mid_au.to_numpy(zero_copy_only=False)),
        orbit_id_index=np.asarray(orbit_id_index_np),
    )

    # Include normals if requested and available
    if include_normals and hasattr(segments, "n_x"):
        soa.n_x = np.asarray(segments.n_x.to_numpy(zero_copy_only=False))
        soa.n_y = np.asarray(segments.n_y.to_numpy(zero_copy_only=False))
        soa.n_z = np.asarray(segments.n_z.to_numpy(zero_copy_only=False))

    soa.validate_structure()
    logger.debug(f"Converted {soa.num_segments} segments to SoA format")

    return soa


def segments_to_numpy_soa(
    segments: OrbitPolylineSegments,
    include_normals: bool = False,
) -> SegmentsSOA:
    """
    Convert OrbitPolylineSegments to SegmentsSOA using NumPy arrays.
    """
    orbit_ids_list = segments.orbit_id.to_pylist()
    unique_ids_in_order = list(dict.fromkeys(orbit_ids_list))
    id_to_idx = {oid: i for i, oid in enumerate(unique_ids_in_order)}
    orbit_id_index_np = np.array([id_to_idx[oid] for oid in orbit_ids_list], dtype=np.int32)

    soa = SegmentsSOA(
        x0=np.asarray(segments.x0.to_numpy(zero_copy_only=False)),
        y0=np.asarray(segments.y0.to_numpy(zero_copy_only=False)),
        z0=np.asarray(segments.z0.to_numpy(zero_copy_only=False)),
        x1=np.asarray(segments.x1.to_numpy(zero_copy_only=False)),
        y1=np.asarray(segments.y1.to_numpy(zero_copy_only=False)),
        z1=np.asarray(segments.z1.to_numpy(zero_copy_only=False)),
        r_mid_au=np.asarray(segments.r_mid_au.to_numpy(zero_copy_only=False)),
        orbit_id_index=np.asarray(orbit_id_index_np),
    )

    if include_normals and hasattr(segments, "n_x"):
        soa.n_x = np.asarray(segments.n_x.to_numpy(zero_copy_only=False))
        soa.n_y = np.asarray(segments.n_y.to_numpy(zero_copy_only=False))
        soa.n_z = np.asarray(segments.n_z.to_numpy(zero_copy_only=False))

    return soa


def segments_soa_to_segments(
    soa: SegmentsSOA, orbit_ids: list[str]
) -> OrbitPolylineSegments:
    """
    Convert JAX-native SegmentsSOA back to OrbitPolylineSegments.

    Parameters
    ----------
    soa : SegmentsSOA
        JAX segments structure

    Returns
    -------
    OrbitPolylineSegments
        Quivr table with segments
    """
    from ..orbits.polyline import OrbitPolylineSegments

    # Convert JAX arrays to numpy
    num_segments = soa.num_segments

    # Reconstruct orbit_id strings via orbit_id_index
    seg_ids = list(range(num_segments))
    seg_orbit_ids = [orbit_ids[int(idx)] for idx in np.array(soa.orbit_id_index)]

    return OrbitPolylineSegments.from_kwargs(
        orbit_id=seg_orbit_ids,
        seg_id=seg_ids,
        x0=np.array(soa.x0),
        y0=np.array(soa.y0),
        z0=np.array(soa.z0),
        x1=np.array(soa.x1),
        y1=np.array(soa.y1),
        z1=np.array(soa.z1),
        # Compute AABBs from endpoints
        aabb_min_x=np.minimum(np.array(soa.x0), np.array(soa.x1)),
        aabb_min_y=np.minimum(np.array(soa.y0), np.array(soa.y1)),
        aabb_min_z=np.minimum(np.array(soa.z0), np.array(soa.z1)),
        aabb_max_x=np.maximum(np.array(soa.x0), np.array(soa.x1)),
        aabb_max_y=np.maximum(np.array(soa.y0), np.array(soa.y1)),
        aabb_max_z=np.maximum(np.array(soa.z0), np.array(soa.z1)),
        r_mid_au=np.array(soa.r_mid_au),
        # Compute normals (dummy values for now)
        n_x=np.zeros(num_segments),
        n_y=np.zeros(num_segments),
        n_z=np.ones(num_segments),
    )

def rays_to_numpy_arrays(
    rays: ObservationRays,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert ObservationRays to NumPy arrays (host-side construction only).
    """
    num_rays = len(rays)
    ray_origins = np.zeros((num_rays, 3), dtype=np.float64)
    ray_directions = np.zeros((num_rays, 3), dtype=np.float64)
    observer_distances = np.zeros(num_rays, dtype=np.float64)

    for i in range(num_rays):
        observer_coords = rays.observer[i]
        ray_origins[i, 0] = observer_coords.x[0].as_py()
        ray_origins[i, 1] = observer_coords.y[0].as_py()
        ray_origins[i, 2] = observer_coords.z[0].as_py()
        ray_directions[i, 0] = rays.u_x[i].as_py()
        ray_directions[i, 1] = rays.u_y[i].as_py()
        ray_directions[i, 2] = rays.u_z[i].as_py()
        observer_distances[i] = np.linalg.norm(ray_origins[i])

    return ray_origins, ray_directions, observer_distances


def hits_soa_to_overlap_hits(
    hits_soa: HitsSOA, det_ids: list[str], orbit_mapping: OrbitIdMapping
) -> OverlapHits:
    """
    Convert JAX HitsSOA back to quivr OverlapHits table.

    Parameters
    ----------
    hits_soa : HitsSOA
        JAX-native hits structure
    det_ids : list[str]
        Detection ID strings (indexed by det_indices)
    orbit_mapping : OrbitIdMapping
        Mapping to convert orbit indices back to strings

    Returns
    -------
    OverlapHits
        Quivr table with hit data
    """
    if hits_soa.num_hits == 0:
        return OverlapHits.empty()

    # Convert arrays back to host
    det_indices = np.asarray(hits_soa.det_indices)
    orbit_indices = np.asarray(hits_soa.orbit_indices)
    seg_ids = np.asarray(hits_soa.seg_ids)
    leaf_ids = np.asarray(hits_soa.leaf_ids)
    distances = np.asarray(hits_soa.distances_au)

    # Map indices back to string IDs
    hit_det_ids = [det_ids[idx] for idx in det_indices]
    hit_orbit_ids = orbit_mapping.map_to_ids(orbit_indices)

    # Stable sort by (det_id, distance) to ensure deterministic order
    order = np.lexsort((distances, np.array(hit_det_ids, dtype=object)))

    return OverlapHits.from_kwargs(
        det_id=[hit_det_ids[i] for i in order],
        orbit_id=[hit_orbit_ids[i] for i in order],
        seg_id=seg_ids[order].tolist(),
        leaf_id=leaf_ids[order].tolist(),
        distance_au=distances[order].tolist(),
    )


def overlap_hits_to_soa(
    hits: OverlapHits, det_id_to_index: dict[str, int], orbit_mapping: OrbitIdMapping
) -> HitsSOA:
    """
    Convert quivr OverlapHits to JAX HitsSOA.

    Parameters
    ----------
    hits : OverlapHits
        Quivr table with hit data
    det_id_to_index : dict[str, int]
        Mapping from detection IDs to indices
    orbit_mapping : OrbitIdMapping
        Mapping to convert orbit IDs to indices

    Returns
    -------
    HitsSOA
        JAX-native hits structure
    """
    if len(hits) == 0:
        return HitsSOA.empty()

    # Convert string IDs to indices
    det_ids = hits.det_id.to_pylist()
    orbit_ids = hits.orbit_id.to_pylist()

    det_indices = np.array(
        [det_id_to_index[det_id] for det_id in det_ids], dtype=np.int32
    )
    orbit_indices = orbit_mapping.map_to_indices(orbit_ids)

    return HitsSOA(
        det_indices=jnp.asarray(det_indices),
        orbit_indices=jnp.asarray(orbit_indices),
        seg_ids=jnp.asarray(hits.seg_id.to_numpy()),
        leaf_ids=jnp.asarray(hits.leaf_id.to_numpy()),
        distances_au=jnp.asarray(hits.distance_au.to_numpy()),
    )


def hits_soa_to_anomaly_labels_soa(
    hits: HitsSOA, max_variants_per_hit: int = 3, device: Optional[jax.Device] = None
) -> AnomalyLabelsSOA:
    """
    Convert HitsSOA to AnomalyLabelsSOA structure for labeling.

    This creates a padded structure where each hit can have up to K variants.
    Initially all variants are masked as invalid and will be filled by the
    anomaly labeling kernel.

    Parameters
    ----------
    hits : HitsSOA
        Input hits structure
    max_variants_per_hit : int, default 3
        Maximum number of anomaly variants per hit
    device : jax.Device, optional
        Target JAX device

    Returns
    -------
    AnomalyLabelsSOA
        Padded structure ready for anomaly labeling
    """
    num_hits = hits.num_hits
    shape = (num_hits, max_variants_per_hit)

    # Replicate hit identification across variants
    det_indices = jnp.broadcast_to(hits.det_indices[:, None], shape)
    orbit_indices = jnp.broadcast_to(hits.orbit_indices[:, None], shape)
    seg_ids = jnp.broadcast_to(hits.seg_ids[:, None], shape)

    # Create variant IDs (0, 1, 2, ...)
    variant_ids = jnp.broadcast_to(
        jnp.arange(max_variants_per_hit, dtype=jnp.int32)[None, :], shape
    )

    # Initialize anomaly values to zero (will be overwritten)
    zeros_shape = jnp.zeros(shape, dtype=jnp.float64)

    # Initialize mask to False (no valid variants yet)
    mask = jnp.zeros(shape, dtype=jnp.bool_)

    labels_soa = AnomalyLabelsSOA(
        det_indices=det_indices,
        orbit_indices=orbit_indices,
        seg_ids=seg_ids,
        variant_ids=variant_ids,
        f_rad=zeros_shape,
        E_rad=zeros_shape,
        M_rad=zeros_shape,
        mean_motion_rad_day=zeros_shape,
        r_au=zeros_shape,
        snap_error=jnp.full(shape, jnp.inf, dtype=jnp.float64),
        plane_distance_au=zeros_shape,
        curvature_hint=zeros_shape,
        mask=mask,
    )

    if device is not None:
        labels_soa = jax.device_put(labels_soa, device)

    return labels_soa


def anomaly_labels_soa_to_anomaly_labels(
    labels_soa: AnomalyLabelsSOA,
    det_id_mapping: dict[int, str],
    orbit_mapping: OrbitIdMapping,
) -> AnomalyLabels:
    """
    Convert AnomalyLabelsSOA to quivr AnomalyLabels table.

    Only valid variants (mask=True) are included in the output.
    Results are sorted by (det_id, orbit_id, variant_id, snap_error).

    Parameters
    ----------
    labels_soa : AnomalyLabelsSOA
        Input anomaly labels structure
    det_id_mapping : dict[int, str]
        Mapping from detection indices to string IDs
    orbit_mapping : OrbitIdMapping
        Mapping from orbit indices to string IDs

    Returns
    -------
    AnomalyLabels
        Sorted quivr table with valid anomaly labels
    """
    # Extract only valid entries
    valid_mask = labels_soa.mask

    if not jnp.any(valid_mask):
        # No valid labels
        return AnomalyLabels.empty()

    # Flatten and filter by mask
    det_indices_flat = labels_soa.det_indices[valid_mask]
    orbit_indices_flat = labels_soa.orbit_indices[valid_mask]
    seg_ids_flat = labels_soa.seg_ids[valid_mask]
    variant_ids_flat = labels_soa.variant_ids[valid_mask]
    f_rad_flat = labels_soa.f_rad[valid_mask]
    E_rad_flat = labels_soa.E_rad[valid_mask]
    M_rad_flat = labels_soa.M_rad[valid_mask]
    n_rad_day_flat = labels_soa.mean_motion_rad_day[valid_mask]
    r_au_flat = labels_soa.r_au[valid_mask]
    snap_error_flat = labels_soa.snap_error[valid_mask]
    plane_distance_flat = labels_soa.plane_distance_au[valid_mask]

    # Convert to numpy for processing
    det_indices_np = np.asarray(det_indices_flat)
    orbit_indices_np = np.asarray(orbit_indices_flat)

    # Map indices back to string IDs
    det_ids = [det_id_mapping[idx] for idx in det_indices_np]
    orbit_ids = orbit_mapping.map_to_ids(orbit_indices_np)

    # Convert arrays to numpy
    seg_ids_np = np.asarray(seg_ids_flat)
    variant_ids_np = np.asarray(variant_ids_flat)
    f_rad_np = np.asarray(f_rad_flat)
    E_rad_np = np.asarray(E_rad_flat)
    M_rad_np = np.asarray(M_rad_flat)
    n_rad_day_np = np.asarray(n_rad_day_flat)
    r_au_np = np.asarray(r_au_flat)
    snap_error_np = np.asarray(snap_error_flat)
    plane_distance_np = np.asarray(plane_distance_flat)

    # Create table
    labels = AnomalyLabels.from_kwargs(
        det_id=det_ids,
        orbit_id=orbit_ids,
        seg_id=seg_ids_np,
        variant_id=variant_ids_np,
        f_rad=f_rad_np,
        E_rad=E_rad_np,
        M_rad=M_rad_np,
        n_rad_day=n_rad_day_np,
        r_au=r_au_np,
        snap_error=snap_error_np,
        plane_distance_au=plane_distance_np,
    )

    # Sort by (det_id, orbit_id, variant_id, snap_error) for deterministic output
    sort_keys = ["det_id", "orbit_id", "variant_id", "snap_error"]

    return labels.sort_by(sort_keys)
