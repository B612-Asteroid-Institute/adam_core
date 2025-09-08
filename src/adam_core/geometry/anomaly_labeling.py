"""
High-level anomaly labeling functions.

This module provides the main API for converting geometric overlap hits
into orbital anomaly assignments using the JAX kernels.
"""

from __future__ import annotations

import functools
import logging
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .anomaly_kernels import (
    compute_anomaly_from_plane_point,
    compute_in_plane_coordinates,
    eccentric_and_mean_anomaly_from_true,
    ellipse_position_from_anomaly,
    project_ray_to_orbital_plane,
)
from .jax_types import AnomalyLabelsSOA, HitsSOA, SegmentsSOA

__all__ = [
    "label_anomalies_batch",
    "compute_orbital_elements_batch",
]

logger = logging.getLogger(__name__)


@functools.partial(jax.jit, static_argnames=['max_variants_per_hit', 'max_newton_iterations'])
def label_anomalies_batch(
    hits_soa: HitsSOA,
    segments_soa: SegmentsSOA,
    ray_origins: jax.Array,
    ray_directions: jax.Array,
    orbital_elements: jax.Array,
    orbital_bases: jax.Array,
    max_variants_per_hit: int = 3,
    max_newton_iterations: int = 5,
) -> AnomalyLabelsSOA:
    """
    Compute anomaly labels for a batch of hits.
    
    This is the main JAX-compiled kernel that processes all hits in parallel.
    
    Parameters
    ----------
    hits_soa : HitsSOA
        Input hits structure
    segments_soa : SegmentsSOA
        Segment data for looking up hit segments
    ray_origins : jax.Array (num_rays, 3)
        Ray origin points
    ray_directions : jax.Array (num_rays, 3)
        Ray direction vectors
    orbital_elements : jax.Array (num_orbits, 6)
        Orbital elements [a, e, i, Omega, omega, M0] for each orbit
    orbital_bases : jax.Array (num_orbits, 3, 3)
        Orbital plane basis vectors [p, q, n] for each orbit
    max_variants_per_hit : int, default 3
        Maximum anomaly variants per hit
    max_newton_iterations : int, default 5
        Maximum Newton iterations for anomaly refinement
        
    Returns
    -------
    AnomalyLabelsSOA
        Computed anomaly labels with validity mask
    """
    num_hits = hits_soa.num_hits
    shape = (num_hits, max_variants_per_hit)
    
    # Initialize output arrays
    det_indices = jnp.broadcast_to(hits_soa.det_indices[:, None], shape)
    orbit_indices = jnp.broadcast_to(hits_soa.orbit_indices[:, None], shape)
    seg_ids = jnp.broadcast_to(hits_soa.seg_ids[:, None], shape)
    variant_ids = jnp.broadcast_to(
        jnp.arange(max_variants_per_hit, dtype=jnp.int32)[None, :], 
        shape
    )
    
    # Initialize anomaly arrays
    f_rad = jnp.zeros(shape, dtype=jnp.float64)
    E_rad = jnp.zeros(shape, dtype=jnp.float64)
    M_rad = jnp.zeros(shape, dtype=jnp.float64)
    mean_motion_rad_day = jnp.zeros(shape, dtype=jnp.float64)
    r_au = jnp.zeros(shape, dtype=jnp.float64)
    snap_error = jnp.full(shape, jnp.inf, dtype=jnp.float64)
    plane_distance_au = jnp.zeros(shape, dtype=jnp.float64)
    curvature_hint = jnp.zeros(shape, dtype=jnp.float64)
    mask = jnp.zeros(shape, dtype=jnp.bool_)
    
    def process_hit(hit_idx):
        """Process a single hit to compute anomaly variants."""
        # Get hit data
        det_idx = hits_soa.det_indices[hit_idx]
        orbit_idx = hits_soa.orbit_indices[hit_idx]
        seg_id = hits_soa.seg_ids[hit_idx]
        
        # Get ray data
        ray_origin = ray_origins[det_idx]
        ray_direction = ray_directions[det_idx]
        
        # Get orbital elements and basis
        elements = orbital_elements[orbit_idx]  # [a, e, i, Omega, omega, M0]
        basis = orbital_bases[orbit_idx]        # [p, q, n] vectors
        
        a, e = elements[0], elements[1]
        basis_p, basis_q, basis_n = basis[0], basis[1], basis[2]
        
        # Project ray onto orbital plane
        plane_center = jnp.zeros(3)  # Assume heliocentric ellipse centered at origin
        intersection, plane_dist, _ = project_ray_to_orbital_plane(
            ray_origin, ray_direction, basis_n, plane_center
        )
        
        # Convert to in-plane coordinates
        x_target, y_target = compute_in_plane_coordinates(
            intersection, plane_center, basis_p, basis_q
        )
        
        # Initial anomaly seed from atan2
        f_seed = jnp.arctan2(y_target, x_target)
        
        # Compute curvature hint (inverse of distance from origin)
        target_distance = jnp.sqrt(x_target * x_target + y_target * y_target)
        curvature = jnp.where(target_distance > 1e-10, 1.0 / target_distance, 0.0)
        
        # Generate variants (for now, just use the main seed)
        # TODO: Add multi-variant logic for ambiguous cases
        variants_f = jnp.array([f_seed, f_seed, f_seed])[:max_variants_per_hit]
        variants_mask = jnp.array([True, False, False])[:max_variants_per_hit]
        
        # Refine each variant
        def refine_variant(variant_idx):
            f_initial = variants_f[variant_idx]
            is_valid = variants_mask[variant_idx]
            
            # Only refine if variant is valid
            # Compute refined values (always computed, but masked later)
            f_computed, residual_computed = compute_anomaly_from_plane_point(
                x_target, y_target, a, e, f_initial, max_newton_iterations
            )
            E_computed, M_computed = eccentric_and_mean_anomaly_from_true(f_computed, e)
            _, _, r_computed = ellipse_position_from_anomaly(f_computed, a, e)
            
            # Use computed values if valid, otherwise use defaults
            f_refined = jnp.where(is_valid, f_computed, f_initial)
            residual = jnp.where(is_valid, residual_computed, jnp.inf)
            E = jnp.where(is_valid, E_computed, 0.0)
            M = jnp.where(is_valid, M_computed, 0.0)
            r = jnp.where(is_valid, r_computed, 0.0)
            
            # Mean motion: n = k / a^{3/2} [rad/day], using Gaussian gravitational constant k
            k = jnp.float64(0.01720209895)
            n = jnp.where(is_valid, k / (a ** 1.5), 0.0)
            
            return f_refined, E, M, n, r, residual, plane_dist, curvature, is_valid
        
        # Process all variants for this hit
        variant_results = jax.vmap(refine_variant)(jnp.arange(max_variants_per_hit))
        
        return variant_results
    
    # Process all hits in parallel
    all_results = jax.vmap(process_hit)(jnp.arange(num_hits))
    
    # Unpack results
    (f_rad, E_rad, M_rad, mean_motion_rad_day, r_au, 
     snap_error, plane_distance_au, curvature_hint, mask) = all_results
    
    return AnomalyLabelsSOA(
        det_indices=det_indices,
        orbit_indices=orbit_indices,
        seg_ids=seg_ids,
        variant_ids=variant_ids,
        f_rad=f_rad,
        E_rad=E_rad,
        M_rad=M_rad,
        mean_motion_rad_day=mean_motion_rad_day,
        r_au=r_au,
        snap_error=snap_error,
        plane_distance_au=plane_distance_au,
        curvature_hint=curvature_hint,
        mask=mask,
    )


def compute_orbital_elements_batch(
    plane_params,  # OrbitsPlaneParams - import avoided to prevent circular dependency
    device: Optional[jax.Device] = None,
) -> Tuple[jax.Array, jax.Array]:
    """
    Extract orbital elements and basis vectors from OrbitsPlaneParams.
    
    Parameters
    ----------
    plane_params : OrbitsPlaneParams
        Orbital plane parameters with elements and basis vectors
    device : jax.Device, optional
        Target JAX device
        
    Returns
    -------
    orbital_elements : jax.Array (num_orbits, 6)
        Orbital elements [a, e, i, Omega, omega, M0] for each orbit
    orbital_bases : jax.Array (num_orbits, 3, 3)
        Orbital plane basis vectors [p, q, n] for each orbit
    """
    if len(plane_params) == 0:
        return (
            jnp.array([]).reshape(0, 6).astype(jnp.float64),
            jnp.array([]).reshape(0, 3, 3).astype(jnp.float64),
        )
    
    # Extract orbital elements - we have a and e, need to compute i, Omega, omega, M
    a = plane_params.a.to_numpy()
    e = plane_params.e.to_numpy()
    
    # Extract basis vectors
    p_vec = np.column_stack([
        plane_params.p_x.to_numpy(),
        plane_params.p_y.to_numpy(), 
        plane_params.p_z.to_numpy()
    ])
    q_vec = np.column_stack([
        plane_params.q_x.to_numpy(),
        plane_params.q_y.to_numpy(),
        plane_params.q_z.to_numpy()
    ])
    n_vec = np.column_stack([
        plane_params.n_x.to_numpy(),
        plane_params.n_y.to_numpy(),
        plane_params.n_z.to_numpy()
    ])
    
    # Reconstruct i, Omega, omega from basis vectors
    # n_hat = [sin(Omega)*sin(i), -cos(Omega)*sin(i), cos(i)]
    cos_i = n_vec[:, 2]
    sin_i = np.sqrt(1 - cos_i**2)
    i = np.arccos(cos_i)
    
    # Handle the case when sin_i ~ 0 (equatorial orbit)
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_Omega = np.where(sin_i > 1e-8, -n_vec[:, 1] / sin_i, 1.0)
        sin_Omega = np.where(sin_i > 1e-8, n_vec[:, 0] / sin_i, 0.0)
    
    Omega = np.arctan2(sin_Omega, cos_Omega)
    
    # Reconstruct omega from p_hat
    # p_hat = [cos(Omega)*cos(omega) - sin(Omega)*sin(omega)*cos(i), ...]
    cos_omega_cos_i = p_vec[:, 2] / np.where(sin_i > 1e-8, sin_i, 1.0)
    cos_omega = np.where(sin_i > 1e-8, cos_omega_cos_i, p_vec[:, 0])
    
    # From q_hat we can get sin(omega)
    sin_omega_cos_i = q_vec[:, 2] / np.where(sin_i > 1e-8, sin_i, 1.0) 
    sin_omega = np.where(sin_i > 1e-8, sin_omega_cos_i, p_vec[:, 1])
    
    omega = np.arctan2(sin_omega, cos_omega)
    
    # Set M0 = 0 for now (epoch mean anomaly)
    # This would ideally come from the orbit epoch and time
    M0 = np.zeros_like(a)
    
    # Stack into elements array [a, e, i, Omega, omega, M0]
    elements = np.column_stack([a, e, i, Omega, omega, M0])
    
    # Stack basis vectors [p, q, n] for each orbit
    bases = np.stack([p_vec, q_vec, n_vec], axis=1)  # (num_orbits, 3, 3)
    
    # Convert to JAX arrays and move to device
    elements_jax = jnp.asarray(elements, dtype=jnp.float64)
    bases_jax = jnp.asarray(bases, dtype=jnp.float64)
    
    if device is not None:
        elements_jax = jax.device_put(elements_jax, device)
        bases_jax = jax.device_put(bases_jax, device)
    
    return elements_jax, bases_jax
