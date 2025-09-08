"""
Candidate aggregator for efficient JAX kernel dispatch.

This module implements the CPU-side BVH traversal that collects all candidate
segments across a batch of rays, then packs them into padded arrays suitable
for vectorized JAX distance computation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .jax_types import BVHArrays, SegmentsSOA

__all__ = [
    "CandidateBatch",
    "aggregate_candidates",
    "_ray_aabb_intersect_jax",
]

logger = logging.getLogger(__name__)


@dataclass
class CandidateBatch:
    """
    Aggregated candidates from BVH traversal, ready for JAX kernel.
    
    All arrays are padded to the same size K for efficient vectorization.
    The mask indicates which entries are valid vs padding.
    """
    # Ray information (B rays)
    ray_indices: jax.Array      # int32[B] - original ray indices
    ray_origins: jax.Array      # float64[B, 3] - ray origin points
    ray_directions: jax.Array   # float64[B, 3] - ray direction vectors
    observer_distances: jax.Array  # float64[B] - observer distances
    
    # Candidate segments (B rays, up to K candidates each)
    seg_starts: jax.Array       # float64[B, K, 3] - segment start points
    seg_ends: jax.Array         # float64[B, K, 3] - segment end points
    r_mids: jax.Array          # float64[B, K] - segment midpoint distances
    orbit_indices: jax.Array    # int32[B, K] - compact orbit indices
    seg_ids: jax.Array         # int32[B, K] - segment IDs
    leaf_ids: jax.Array        # int32[B, K] - BVH leaf IDs
    
    # Validity mask
    mask: jax.Array            # bool[B, K] - True for valid candidates
    
    @property
    def batch_size(self) -> int:
        """Number of rays in batch."""
        return self.ray_indices.shape[0]
    
    @property
    def max_candidates(self) -> int:
        """Maximum candidates per ray (K)."""
        return self.seg_starts.shape[1]
    
    def validate_structure(self) -> None:
        """Validate batch structure and shapes."""
        B, K = self.batch_size, self.max_candidates
        
        # Ray arrays
        assert self.ray_origins.shape == (B, 3)
        assert self.ray_directions.shape == (B, 3)
        assert self.observer_distances.shape == (B,)
        
        # Candidate arrays
        assert self.seg_starts.shape == (B, K, 3)
        assert self.seg_ends.shape == (B, K, 3)
        assert self.r_mids.shape == (B, K)
        assert self.orbit_indices.shape == (B, K)
        assert self.seg_ids.shape == (B, K)
        assert self.leaf_ids.shape == (B, K)
        assert self.mask.shape == (B, K)
        
        # Dtypes
        assert self.ray_indices.dtype == jnp.int32
        assert self.ray_origins.dtype == jnp.float64
        assert self.ray_directions.dtype == jnp.float64
        assert self.observer_distances.dtype == jnp.float64
        assert self.seg_starts.dtype == jnp.float64
        assert self.seg_ends.dtype == jnp.float64
        assert self.r_mids.dtype == jnp.float64
        assert self.orbit_indices.dtype == jnp.int32
        assert self.seg_ids.dtype == jnp.int32
        assert self.leaf_ids.dtype == jnp.int32
        assert self.mask.dtype == jnp.bool_


def _ray_aabb_intersect_jax(
    ray_origin: jax.Array,
    ray_direction: jax.Array,
    aabb_min: jax.Array,
    aabb_max: jax.Array,
) -> bool:
    """
    JAX-compatible ray-AABB intersection test using slab method.
    
    Parameters
    ----------
    ray_origin : jax.Array (3,)
        Ray origin point
    ray_direction : jax.Array (3,)
        Ray direction vector
    aabb_min : jax.Array (3,)
        AABB minimum bounds
    aabb_max : jax.Array (3,)
        AABB maximum bounds
        
    Returns
    -------
    intersects : bool
        True if ray intersects AABB
    """
    # Handle near-zero direction components
    inv_dir = jnp.where(
        jnp.abs(ray_direction) < 1e-15,
        jnp.copysign(1e15, ray_direction),
        1.0 / ray_direction
    )
    
    # Compute slab intersection parameters
    t1 = (aabb_min - ray_origin) * inv_dir
    t2 = (aabb_max - ray_origin) * inv_dir
    
    # Ensure t1 <= t2 for each axis
    t_min = jnp.minimum(t1, t2)
    t_max = jnp.maximum(t1, t2)
    
    # Ray intersects AABB if overlap exists and tmax >= 0
    tmin = jnp.max(t_min)
    tmax = jnp.min(t_max)
    
    return (tmax >= 0.0) & (tmin <= tmax)


def aggregate_candidates(
    bvh: BVHArrays,
    segments: SegmentsSOA,
    ray_origins: jax.Array,
    ray_directions: jax.Array,
    observer_distances: jax.Array,
    max_candidates_per_ray: int = 64,
    device: Optional[jax.Device] = None
) -> CandidateBatch:
    """
    Aggregate candidates from BVH traversal across a batch of rays.
    
    This function performs CPU-side BVH traversal to collect all candidate
    segments, then packs them into padded arrays for efficient JAX processing.
    
    Parameters
    ----------
    bvh : BVHArrays
        JAX-native BVH structure
    segments : SegmentsSOA
        JAX-native segments structure
    ray_origins : jax.Array
        Ray origin points, shape (num_rays, 3)
    ray_directions : jax.Array
        Ray direction vectors, shape (num_rays, 3)
    observer_distances : jax.Array
        Observer distances, shape (num_rays,)
    max_candidates_per_ray : int, default=64
        Maximum candidates to collect per ray (K)
    device : jax.Device, optional
        Device to place result arrays on
        
    Returns
    -------
    CandidateBatch
        Aggregated candidates ready for JAX kernel
    """
    num_rays = ray_origins.shape[0]
    K = max_candidates_per_ray
    
    # Convert to numpy for CPU traversal
    bvh_cpu = jax.device_get(bvh)
    ray_origins_cpu = np.asarray(ray_origins)
    ray_directions_cpu = np.asarray(ray_directions)
    observer_distances_cpu = np.asarray(observer_distances)
    
    # Preallocate result arrays
    ray_indices = np.arange(num_rays, dtype=np.int32)
    
    seg_starts = np.zeros((num_rays, K, 3), dtype=np.float64)
    seg_ends = np.zeros((num_rays, K, 3), dtype=np.float64)
    r_mids = np.zeros((num_rays, K), dtype=np.float64)
    orbit_indices = np.zeros((num_rays, K), dtype=np.int32)
    seg_ids = np.zeros((num_rays, K), dtype=np.int32)
    leaf_ids = np.zeros((num_rays, K), dtype=np.int32)
    mask = np.zeros((num_rays, K), dtype=bool)
    
    # Process each ray
    for ray_idx in range(num_rays):
        ray_origin = ray_origins_cpu[ray_idx]
        ray_direction = ray_directions_cpu[ray_idx]
        
        candidates = []
        
        # BVH traversal using stack-based DFS
        node_stack = [0]  # Start with root node
        
        while node_stack and len(candidates) < K:
            node_idx = node_stack.pop()
            
            # Test ray against node AABB
            node_min = bvh_cpu.nodes_min[node_idx]
            node_max = bvh_cpu.nodes_max[node_idx]
            
            # Use numpy version for CPU traversal
            inv_dir = np.where(
                np.abs(ray_direction) < 1e-15,
                np.copysign(1e15, ray_direction),
                1.0 / ray_direction
            )
            
            t1 = (node_min - ray_origin) * inv_dir
            t2 = (node_max - ray_origin) * inv_dir
            t_min = np.minimum(t1, t2)
            t_max = np.maximum(t1, t2)
            tmin = np.max(t_min)
            tmax = np.min(t_max)
            
            if not (tmax >= 0.0 and tmin <= tmax):
                continue
            
            if bvh_cpu.is_leaf[node_idx]:
                # Process leaf node - collect all candidates
                row_indices, orbit_inds, seg_id_vals = bvh_cpu.get_leaf_primitives(node_idx)
                
                for i in range(len(row_indices)):
                    if len(candidates) >= K:
                        break
                    
                    row_idx = int(row_indices[i])
                    orbit_idx = int(orbit_inds[i])
                    seg_id = int(seg_id_vals[i])
                    
                    candidates.append((row_idx, orbit_idx, seg_id, node_idx))
            else:
                # Internal node - add children to stack
                left_child = int(bvh_cpu.left_child[node_idx])
                right_child = int(bvh_cpu.right_child[node_idx])
                
                if left_child >= 0:
                    node_stack.append(left_child)
                if right_child >= 0:
                    node_stack.append(right_child)
        
        # Pack candidates into arrays
        num_candidates = min(len(candidates), K)
        
        if num_candidates > 0:
            # Extract segment data using row indices
            segments_cpu = jax.device_get(segments)
            
            for i, (row_idx, orbit_idx, seg_id, leaf_id) in enumerate(candidates[:K]):
                seg_starts[ray_idx, i] = [
                    segments_cpu.x0[row_idx],
                    segments_cpu.y0[row_idx],
                    segments_cpu.z0[row_idx]
                ]
                seg_ends[ray_idx, i] = [
                    segments_cpu.x1[row_idx],
                    segments_cpu.y1[row_idx],
                    segments_cpu.z1[row_idx]
                ]
                r_mids[ray_idx, i] = segments_cpu.r_mid_au[row_idx]
                orbit_indices[ray_idx, i] = orbit_idx
                seg_ids[ray_idx, i] = seg_id
                leaf_ids[ray_idx, i] = leaf_id
                mask[ray_idx, i] = True
    
    # Convert to JAX arrays
    batch = CandidateBatch(
        ray_indices=jnp.asarray(ray_indices),
        ray_origins=jnp.asarray(ray_origins_cpu),
        ray_directions=jnp.asarray(ray_directions_cpu),
        observer_distances=jnp.asarray(observer_distances_cpu),
        seg_starts=jnp.asarray(seg_starts),
        seg_ends=jnp.asarray(seg_ends),
        r_mids=jnp.asarray(r_mids),
        orbit_indices=jnp.asarray(orbit_indices),
        seg_ids=jnp.asarray(seg_ids),
        leaf_ids=jnp.asarray(leaf_ids),
        mask=jnp.asarray(mask),
    )
    
    # Move to specified device if requested
    if device is not None:
        batch = jax.device_put(batch, device)
    
    batch.validate_structure()
    
    total_candidates = int(jnp.sum(batch.mask))
    logger.debug(f"Aggregated {total_candidates} candidates from {num_rays} rays (max {K} per ray)")
    
    return batch


# Register CandidateBatch as JAX PyTree
jax.tree_util.register_pytree_node(
    CandidateBatch,
    lambda batch: (
        (batch.ray_indices, batch.ray_origins, batch.ray_directions, 
         batch.observer_distances, batch.seg_starts, batch.seg_ends,
         batch.r_mids, batch.orbit_indices, batch.seg_ids, batch.leaf_ids, batch.mask),
        None
    ),
    lambda aux_data, children: CandidateBatch(*children)
)
