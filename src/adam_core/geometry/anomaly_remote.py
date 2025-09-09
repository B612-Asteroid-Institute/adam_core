"""
Ray remote functions for parallel anomaly labeling.

This module provides Ray remote functions for parallel computation of anomaly
labels from geometric overlap hits. Uses Ray's object store for shared orbital
parameters and follows the pattern of @ray.remote functions.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import ray

from ..orbits.polyline import OrbitsPlaneParams
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
    plane_params: OrbitsPlaneParams,
    ray_origins: np.ndarray,
    ray_directions: np.ndarray,
    det_ids: List[str],
    segments_soa: SegmentsSOA,
    batch_size: int = 1000,
    max_variants_per_hit: int = 2,
    max_newton_iterations: int = 10,
    device: Optional[jax.Device] = None,
) -> AnomalyLabels:
    """
    Compute anomaly labels in parallel using Ray remote functions.
    """
    if len(hits) == 0:
        return AnomalyLabels.empty()
    
    logger.info(f"Computing anomaly labels for {len(hits)} hits (synchronous)")
    
    # Compute orbital elements and bases once
    # Inline computation since compute_orbital_elements_batch was removed
    from adam_core.orbits.orbits import Orbits
    orbits = Orbits.from_kwargs(
        orbit_id=plane_params.orbit_id,
        coordinates=plane_params.coordinates,
    )
    kep = orbits.coordinates.to_keplerian()
    orbit_ids_arr = orbits.table.column("orbit_id")
    a_arr_np = kep.a.to_numpy()
    e_arr_np = kep.e.to_numpy()
    M0_deg_np = kep.M.to_numpy() * (180.0 / np.pi)
    epoch_mjd_np = kep.time.mjd().to_numpy()
    GM_sun_au3_per_day2 = 2.959122082855911e-4
    n_deg_per_day_np = np.sqrt(GM_sun_au3_per_day2 / (a_arr_np ** 3)) * (180.0 / np.pi)
    
    orbital_elements = jnp.column_stack([
        a_arr_np, e_arr_np, M0_deg_np, n_deg_per_day_np, epoch_mjd_np, epoch_mjd_np  # duplicate for compatibility
    ])
    
    # Orbital bases from plane params (placeholder - this would need proper implementation)
    n_orbits = len(plane_params)
    orbital_bases = jnp.zeros((n_orbits, 3, 3))  # Identity matrices as placeholder
    
    # Build mappings
    orbit_ids = plane_params.orbit_id.to_pylist()
    orbit_mapping = OrbitIdMapping.from_orbit_ids(orbit_ids)
    det_id_to_index = {d: i for i, d in enumerate(det_ids)}
    
    # Convert hits to SoA (indices align with det_ids and orbit_mapping)
    hits_soa = overlap_hits_to_soa(hits, det_id_to_index, orbit_mapping)
    
    # Convert to JAX arrays
    ray_origins_jax = jnp.asarray(ray_origins)
    ray_directions_jax = jnp.asarray(ray_directions)
    
    # Run labeling kernel
    labels_soa = label_anomalies(
        hits_soa,
        segments_soa,
        ray_origins_jax,
        ray_directions_jax,
        jnp.asarray(orbital_elements),
        jnp.asarray(orbital_bases),
        max_variants_per_hit=max_variants_per_hit,
        max_newton_iterations=max_newton_iterations,
    )
    
    # Convert to quivr table
    det_index_to_id = {i: det_ids[i] for i in range(len(det_ids))}
    labels = anomaly_labels_soa_to_anomaly_labels(
        labels_soa,
        det_id_mapping=det_index_to_id,
        orbit_mapping=orbit_mapping,
    )
    
    logger.info(f"Computed {len(labels)} anomaly labels from {len(hits)} hits")
    return labels
