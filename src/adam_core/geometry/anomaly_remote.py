"""
Ray remote functions for parallel anomaly labeling.

This module provides Ray remote functions for parallel computation of anomaly
labels from geometric overlap hits. Uses Ray's object store for shared orbital
parameters and follows the pattern of @ray.remote functions.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import ray

from ..orbits.orbits import Orbits
from ..observations.rays import ObservationRays
from .adapters import anomaly_labels_soa_to_anomaly_labels, hits_soa_to_anomaly_labels_soa
from .anomaly import AnomalyLabels
from .anomaly_labeling import label_anomalies
from .jax_types import AnomalyLabelsSOA, HitsSOA, OrbitIdMapping, SegmentsSOA
from .overlap import OverlapHits
from .adapters import overlap_hits_to_soa

logger = logging.getLogger(__name__)

__all__ = [
    "process_anomaly_batch_remote",
    "label_anomalies_parallel",
]


@ray.remote
def process_anomaly_batch_remote(
    hits_dict: Dict[str, np.ndarray],
    orbital_elements_ref: ray.ObjectRef,
    orbital_bases_ref: ray.ObjectRef,
    ray_origins,
    ray_directions,
    segments_soa_ref: ray.ObjectRef,
    max_variants_per_hit: int = 2,
    max_newton_iterations: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Ray remote function to process anomaly labeling for a batch of hits.
    
    Parameters
    ----------
    hits_dict : Dict[str, np.ndarray]
        Dictionary with hit arrays from overlap detection
    orbital_elements_ref : ray.ObjectRef
        Reference to orbital elements array in Ray's object store
    orbital_bases_ref : ray.ObjectRef
        Reference to orbital bases array in Ray's object store
    ray_origins : np.ndarray
        Ray origin positions (N, 3)
    ray_directions : np.ndarray
        Ray direction vectors (N, 3)
    segments_soa_ref : ray.ObjectRef
        Reference to segments SOA in Ray's object store
    max_variants_per_hit : int, default=2
        Maximum variants to compute per hit
    max_newton_iterations : int, default=10
        Maximum Newton iterations for anomaly refinement
        
    Returns
    -------
    labels_dict : Dict[str, np.ndarray]
        Dictionary with anomaly label arrays for reconstruction
    """
    # Get shared data from Ray object store
    orbital_elements = (
        ray.get(orbital_elements_ref) if isinstance(orbital_elements_ref, ray.ObjectRef) 
        else orbital_elements_ref
    )
    orbital_bases = (
        ray.get(orbital_bases_ref) if isinstance(orbital_bases_ref, ray.ObjectRef)
        else orbital_bases_ref
    )
    segments_soa_dict = (
        ray.get(segments_soa_ref) if isinstance(segments_soa_ref, ray.ObjectRef)
        else segments_soa_ref
    )
    # Ray arrays may be ObjectRefs as well
    ray_origins = ray.get(ray_origins) if isinstance(ray_origins, ray.ObjectRef) else ray_origins
    ray_directions = ray.get(ray_directions) if isinstance(ray_directions, ray.ObjectRef) else ray_directions
    
    # Reconstruct HitsSOA from dictionary
    if len(hits_dict["det_indices"]) == 0:
        # Return empty labels
        return {
            "det_indices": np.array([], dtype=np.int32),
            "orbit_indices": np.array([], dtype=np.int32),
            "seg_ids": np.array([], dtype=np.int32),
            "variant_ids": np.array([], dtype=np.int32),
            "f_rad": np.array([], dtype=np.float64),
            "E_rad": np.array([], dtype=np.float64),
            "M_rad": np.array([], dtype=np.float64),
            "mean_motion_rad_day": np.array([], dtype=np.float64),
            "r_au": np.array([], dtype=np.float64),
            "snap_error": np.array([], dtype=np.float64),
            "plane_distance_au": np.array([], dtype=np.float64),
            "curvature_hint": np.array([], dtype=np.float64),
            "mask": np.array([], dtype=np.bool_),
        }
    
    hits_soa = HitsSOA(
        det_indices=jnp.asarray(hits_dict["det_indices"], dtype=jnp.int32),
        orbit_indices=jnp.asarray(hits_dict["orbit_indices"], dtype=jnp.int32),
        seg_ids=jnp.asarray(hits_dict["seg_ids"], dtype=jnp.int32),
        leaf_ids=jnp.zeros_like(jnp.asarray(hits_dict["seg_ids"], dtype=jnp.int32)),
        distances_au=jnp.asarray(hits_dict["distances_au"], dtype=jnp.float64),
    )
    
    # Reconstruct SegmentsSOA from dictionary
    segments_soa = SegmentsSOA(
        x0=jnp.asarray(segments_soa_dict["x0"]),
        y0=jnp.asarray(segments_soa_dict["y0"]),
        z0=jnp.asarray(segments_soa_dict["z0"]),
        x1=jnp.asarray(segments_soa_dict["x1"]),
        y1=jnp.asarray(segments_soa_dict["y1"]),
        z1=jnp.asarray(segments_soa_dict["z1"]),
        r_mid_au=jnp.asarray(segments_soa_dict["r_mid_au"]),
    )
    
    # Convert to JAX arrays
    ray_origins_jax = jnp.asarray(ray_origins)
    ray_directions_jax = jnp.asarray(ray_directions)
    orbital_elements_jax = jnp.asarray(orbital_elements)
    orbital_bases_jax = jnp.asarray(orbital_bases)
    
    # Compute anomaly labels
    labels_soa = label_anomalies(
        hits_soa,
        segments_soa,  # Use real segments_soa for consistent seeding
        ray_origins_jax,
        ray_directions_jax,
        orbital_elements_jax,
        orbital_bases_jax,
        max_variants_per_hit=max_variants_per_hit,
        max_newton_iterations=max_newton_iterations,
    )
    
    # Convert to numpy for serialization
    return {
        "det_indices": np.asarray(labels_soa.det_indices),
        "orbit_indices": np.asarray(labels_soa.orbit_indices),
        "seg_ids": np.asarray(labels_soa.seg_ids),
        "variant_ids": np.asarray(labels_soa.variant_ids),
        "f_rad": np.asarray(labels_soa.f_rad),
        "E_rad": np.asarray(labels_soa.E_rad),
        "M_rad": np.asarray(labels_soa.M_rad),
        "mean_motion_rad_day": np.asarray(labels_soa.mean_motion_rad_day),
        "r_au": np.asarray(labels_soa.r_au),
        "snap_error": np.asarray(labels_soa.snap_error),
        "plane_distance_au": np.asarray(labels_soa.plane_distance_au),
        "curvature_hint": np.asarray(labels_soa.curvature_hint),
        "mask": np.asarray(labels_soa.mask),
    }


def label_anomalies_parallel(
    hits: OverlapHits,
    orbits: Orbits,
    rays: ObservationRays,
    segments_soa: SegmentsSOA | None = None,
    batch_size: int = 1000,
    max_k: int = 1,
    snap_error_max_au: float | None = None,
    dedupe_angle_tol: float = 1e-4,
    max_variants_per_hit: int = 2,  # Deprecated, use max_k
    max_newton_iterations: int = 10,  # Deprecated
    device: Optional[jax.Device] = None,
) -> AnomalyLabels:
    """
    Parallel labeling with multi-anomaly support.
    
    Currently delegates to the core label_anomalies function. For large workloads,
    this could be extended to use Ray batching with fixed-shape padding.
    
    Parameters
    ----------
    hits : OverlapHits
        Geometric overlaps between observations and test orbits.
    orbits : Orbits
        Original orbit data containing Keplerian elements.
    rays : ObservationRays
        Observation rays containing timing information.
    segments_soa : SegmentsSOA, optional
        Deprecated, not used in current implementation.
    batch_size : int, default=1000
        Batch size for potential Ray parallelization (not currently used).
    max_k : int, default=1
        Maximum number of anomaly variants per (det_id, orbit_id) pair.
    snap_error_max_au : float, optional
        Maximum allowed snap error (AU). Candidates above this are filtered out.
    dedupe_angle_tol : float, default=1e-4
        Angular tolerance (radians) for deduplicating near-identical solutions.
    max_variants_per_hit : int, default=2
        Deprecated parameter, use max_k instead.
    max_newton_iterations : int, default=10
        Deprecated parameter, Newton iterations are fixed in JAX kernels.
    device : jax.Device, optional
        Deprecated parameter, JAX device selection is automatic.
        
    Returns
    -------
    AnomalyLabels
        Table with anomaly labels, potentially with multiple variants per hit.
    """
    if len(hits) == 0:
        return AnomalyLabels.empty()
    
    # Use max_k if provided, otherwise fall back to max_variants_per_hit for compatibility
    effective_max_k = max_k if max_k > 1 else max_variants_per_hit
    
    return label_anomalies(
        hits, rays, orbits, 
        max_k=effective_max_k,
        snap_error_max_au=snap_error_max_au,
        dedupe_angle_tol=dedupe_angle_tol
    )
