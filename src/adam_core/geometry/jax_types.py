"""
JAX-native data structures for geometric overlap detection.

This module provides PyTree dataclasses optimized for JAX kernels and Ray
distributed processing. All arrays use standard dtypes (float64, int32, bool)
and structure-of-arrays (SoA) layout for efficient vectorization.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

# Configure JAX for double precision by default (geometry accuracy)
jax.config.update("jax_enable_x64", True)

__all__ = [
    "BVHArrays",
    "SegmentsSOA",
    "HitsSOA",
    "AnomalyLabelsSOA",
    "OrbitIdMapping",
    "save_bvh_arrays",
    "load_bvh_arrays",
    "save_segments_soa",
    "load_segments_soa",
]


@dataclass
class OrbitIdMapping:
    """
    Mapping between string orbit IDs and compact integer indices.

    This enables efficient storage and computation while maintaining
    human-readable orbit identifiers at the API boundary.
    """

    id_to_index: dict[str, int]
    index_to_id: list[str]

    @classmethod
    def from_orbit_ids(cls, orbit_ids: list[str]) -> OrbitIdMapping:
        """Create mapping from a list of unique orbit IDs."""
        unique_ids = list(dict.fromkeys(orbit_ids))  # Preserve order, remove duplicates
        id_to_index = {oid: i for i, oid in enumerate(unique_ids)}
        return cls(id_to_index=id_to_index, index_to_id=unique_ids)

    def map_to_indices(self, orbit_ids: list[str]) -> npt.NDArray[np.int32]:
        """Convert orbit ID strings to integer indices."""
        return np.array([self.id_to_index[oid] for oid in orbit_ids], dtype=np.int32)

    def map_to_ids(self, indices: npt.NDArray[np.int32]) -> list[str]:
        """Convert integer indices back to orbit ID strings."""
        return [self.index_to_id[idx] for idx in indices]


@dataclass
class BVHArrays:
    """
    JAX-native BVH representation using structure-of-arrays layout.

    This replaces the object-oriented BVHShard with pure arrays that can be
    efficiently passed through JAX transformations and Ray workers.

    All arrays have the same leading dimension (num_nodes) and use standard
    dtypes for cross-platform compatibility.
    """

    # Node bounding boxes
    nodes_min: jax.Array  # float64[num_nodes, 3]
    nodes_max: jax.Array  # float64[num_nodes, 3]

    # Tree structure
    left_child: jax.Array  # int32[num_nodes] (-1 for leaves)
    right_child: jax.Array  # int32[num_nodes] (-1 for leaves)
    is_leaf: jax.Array  # bool[num_nodes]

    # Leaf primitive data
    first_prim: jax.Array  # int32[num_nodes] (-1 for internal nodes)
    prim_count: jax.Array  # int32[num_nodes] (0 for internal nodes)

    # Primitive arrays (parallel, length = total primitives)
    prim_row_index: jax.Array  # int32[num_primitives] - segment row lookup
    orbit_id_index: jax.Array  # int32[num_primitives] - compact orbit indices
    prim_seg_ids: jax.Array  # int32[num_primitives] - segment IDs

    @property
    def num_nodes(self) -> int:
        """Number of BVH nodes."""
        return self.nodes_min.shape[0]

    @property
    def num_primitives(self) -> int:
        """Number of primitives (segments)."""
        return self.prim_row_index.shape[0]

    def get_leaf_primitives(
        self, node_idx: int
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Get primitive indices for a leaf node.

        Returns
        -------
        row_indices : jax.Array
            Segment row indices for O(1) lookup
        orbit_indices : jax.Array
            Compact orbit indices
        seg_ids : jax.Array
            Segment IDs within orbits
        """
        if not self.is_leaf[node_idx]:
            raise ValueError(f"Node {node_idx} is not a leaf")

        first = self.first_prim[node_idx]
        count = self.prim_count[node_idx]

        if first == -1 or count == 0:
            return (
                jnp.array([], dtype=jnp.int32),
                jnp.array([], dtype=jnp.int32),
                jnp.array([], dtype=jnp.int32),
            )

        slice_end = first + count
        return (
            self.prim_row_index[first:slice_end],
            self.orbit_id_index[first:slice_end],
            self.prim_seg_ids[first:slice_end],
        )

    def validate_structure(self) -> None:
        """Validate BVH structure and array shapes."""
        # Check consistent shapes
        n_nodes = self.num_nodes
        assert self.nodes_max.shape == (n_nodes, 3)
        assert self.left_child.shape == (n_nodes,)
        assert self.right_child.shape == (n_nodes,)
        assert self.is_leaf.shape == (n_nodes,)
        assert self.first_prim.shape == (n_nodes,)
        assert self.prim_count.shape == (n_nodes,)

        n_prims = self.num_primitives
        assert self.orbit_id_index.shape == (n_prims,)
        assert self.prim_seg_ids.shape == (n_prims,)

        # Check dtypes
        assert self.nodes_min.dtype == jnp.float64
        assert self.nodes_max.dtype == jnp.float64
        assert self.left_child.dtype == jnp.int32
        assert self.right_child.dtype == jnp.int32
        assert self.is_leaf.dtype == jnp.bool_
        assert self.first_prim.dtype == jnp.int32
        assert self.prim_count.dtype == jnp.int32
        assert self.prim_row_index.dtype == jnp.int32
        assert self.orbit_id_index.dtype == jnp.int32
        assert self.prim_seg_ids.dtype == jnp.int32


@dataclass
class SegmentsSOA:
    """
    Structure-of-arrays representation for orbit segments.

    Optimized for vectorized distance computations and efficient
    memory access patterns in JAX kernels.
    """

    # Segment endpoints in SSB Cartesian (AU)
    x0: jax.Array  # float64[num_segments]
    y0: jax.Array  # float64[num_segments]
    z0: jax.Array  # float64[num_segments]
    x1: jax.Array  # float64[num_segments]
    y1: jax.Array  # float64[num_segments]
    z1: jax.Array  # float64[num_segments]

    # Segment midpoint distance for guard band scaling
    r_mid_au: jax.Array  # float64[num_segments]

    # Segment-to-orbit mapping (compact index into per-shard orbit_ids)
    orbit_id_index: jax.Array  # int32[num_segments]

    # Optional: orbital plane normal for advanced guard band computation
    n_x: Optional[jax.Array] = None  # float64[num_segments]
    n_y: Optional[jax.Array] = None  # float64[num_segments]
    n_z: Optional[jax.Array] = None  # float64[num_segments]

    @classmethod
    def empty(cls) -> "SegmentsSOA":
        zeros = jnp.array([], dtype=jnp.float64)
        return cls(
            x0=zeros,
            y0=zeros,
            z0=zeros,
            x1=zeros,
            y1=zeros,
            z1=zeros,
            r_mid_au=zeros,
            orbit_id_index=jnp.array([], dtype=jnp.int32),
        )

    @property
    def num_segments(self) -> int:
        """Number of segments."""
        return self.x0.shape[0]

    def get_starts(self, indices: jax.Array) -> jax.Array:
        """Get segment start points for given indices."""
        return jnp.column_stack([self.x0[indices], self.y0[indices], self.z0[indices]])

    def get_ends(self, indices: jax.Array) -> jax.Array:
        """Get segment end points for given indices."""
        return jnp.column_stack([self.x1[indices], self.y1[indices], self.z1[indices]])

    def validate_structure(self) -> None:
        """Validate segment array shapes and dtypes."""
        n_segs = self.num_segments

        # Check shapes
        for arr in [
            self.y0,
            self.z0,
            self.x1,
            self.y1,
            self.z1,
            self.r_mid_au,
            self.orbit_id_index,
        ]:
            assert arr.shape == (n_segs,)

        if self.n_x is not None:
            assert self.n_x.shape == (n_segs,)
            assert self.n_y.shape == (n_segs,)
            assert self.n_z.shape == (n_segs,)

        # Check dtypes
        for arr in [
            self.x0,
            self.y0,
            self.z0,
            self.x1,
            self.y1,
            self.z1,
            self.r_mid_au,
        ]:
            assert arr.dtype == jnp.float64
        assert self.orbit_id_index.dtype == jnp.int32


@dataclass
class HitsSOA:
    """
    Structure-of-arrays representation for geometric overlap hits.

    Uses compact integer indices internally for efficiency, with
    string IDs resolved at the API boundary.
    """

    det_indices: jax.Array  # int32[num_hits] - detection indices
    orbit_indices: jax.Array  # int32[num_hits] - compact orbit indices
    seg_ids: jax.Array  # int32[num_hits] - segment IDs
    leaf_ids: jax.Array  # int32[num_hits] - BVH leaf indices
    distances_au: jax.Array  # float64[num_hits] - ray-segment distances

    @property
    def num_hits(self) -> int:
        """Number of hits."""
        return self.det_indices.shape[0]

    def validate_structure(self) -> None:
        """Validate hit array shapes and dtypes."""
        n_hits = self.num_hits

        # Check shapes
        for arr in [self.orbit_indices, self.seg_ids, self.leaf_ids]:
            assert arr.shape == (n_hits,)
        assert self.distances_au.shape == (n_hits,)

        # Check dtypes
        for arr in [self.det_indices, self.orbit_indices, self.seg_ids, self.leaf_ids]:
            assert arr.dtype == jnp.int32
        assert self.distances_au.dtype == jnp.float64

    @classmethod
    def empty(cls) -> HitsSOA:
        """Create empty hits structure."""
        return cls(
            det_indices=jnp.array([], dtype=jnp.int32),
            orbit_indices=jnp.array([], dtype=jnp.int32),
            seg_ids=jnp.array([], dtype=jnp.int32),
            leaf_ids=jnp.array([], dtype=jnp.int32),
            distances_au=jnp.array([], dtype=jnp.float64),
        )


@dataclass
class AnomalyLabelsSOA:
    """
    Structure-of-arrays representation for anomaly labels.

    Each hit can have up to K anomaly variants (for ambiguous cases near nodes).
    Uses padded arrays with masking for efficient JAX vectorization.

    Shape: [num_hits, max_variants_per_hit]
    """

    # Hit identification (same across variants for a hit)
    det_indices: jax.Array  # int32[num_hits, K] - detection indices (repeated)
    orbit_indices: jax.Array  # int32[num_hits, K] - compact orbit indices (repeated)
    seg_ids: jax.Array  # int32[num_hits, K] - segment IDs (repeated)
    variant_ids: jax.Array  # int32[num_hits, K] - variant ID within hit (0, 1, 2, ...)

    # Anomaly values
    f_rad: jax.Array  # float64[num_hits, K] - true anomaly (radians)
    E_rad: jax.Array  # float64[num_hits, K] - eccentric anomaly (radians)
    M_rad: jax.Array  # float64[num_hits, K] - mean anomaly (radians)
    mean_motion_rad_day: jax.Array  # float64[num_hits, K] - mean motion (rad/day)
    r_au: jax.Array  # float64[num_hits, K] - heliocentric distance (AU)

    # Quality metrics
    snap_error: jax.Array  # float64[num_hits, K] - residual from ellipse fit
    plane_distance_au: jax.Array  # float64[num_hits, K] - distance from orbital plane
    curvature_hint: jax.Array  # float64[num_hits, K] - local curvature indicator

    # Validity mask
    mask: jax.Array  # bool[num_hits, K] - True for valid variants

    @property
    def shape(self) -> tuple[int, int]:
        """Shape (num_hits, max_variants_per_hit)."""
        return self.det_indices.shape

    @property
    def num_hits(self) -> int:
        """Number of hits."""
        return self.shape[0]

    @property
    def max_variants_per_hit(self) -> int:
        """Maximum variants per hit (K)."""
        return self.shape[1]

    def validate_structure(self) -> None:
        """Validate anomaly label array shapes and dtypes."""
        shape = self.shape

        # Check shapes - all arrays should have same shape
        arrays_to_check = [
            self.det_indices,
            self.orbit_indices,
            self.seg_ids,
            self.variant_ids,
            self.f_rad,
            self.E_rad,
            self.M_rad,
            self.mean_motion_rad_day,
            self.r_au,
            self.snap_error,
            self.plane_distance_au,
            self.curvature_hint,
            self.mask,
        ]

        for arr in arrays_to_check:
            assert arr.shape == shape, f"Array shape {arr.shape} != expected {shape}"

        # Check dtypes
        int_arrays = [
            self.det_indices,
            self.orbit_indices,
            self.seg_ids,
            self.variant_ids,
        ]
        for arr in int_arrays:
            assert arr.dtype == jnp.int32, f"Integer array has dtype {arr.dtype}"

        float_arrays = [
            self.f_rad,
            self.E_rad,
            self.M_rad,
            self.mean_motion_rad_day,
            self.r_au,
            self.snap_error,
            self.plane_distance_au,
            self.curvature_hint,
        ]
        for arr in float_arrays:
            assert arr.dtype == jnp.float64, f"Float array has dtype {arr.dtype}"

        assert self.mask.dtype == jnp.bool_, f"Mask has dtype {self.mask.dtype}"

    @classmethod
    def empty(cls, max_variants_per_hit: int = 3) -> AnomalyLabelsSOA:
        """Create empty anomaly labels structure."""
        shape = (0, max_variants_per_hit)

        return cls(
            det_indices=jnp.zeros(shape, dtype=jnp.int32),
            orbit_indices=jnp.zeros(shape, dtype=jnp.int32),
            seg_ids=jnp.zeros(shape, dtype=jnp.int32),
            variant_ids=jnp.zeros(shape, dtype=jnp.int32),
            f_rad=jnp.zeros(shape, dtype=jnp.float64),
            E_rad=jnp.zeros(shape, dtype=jnp.float64),
            M_rad=jnp.zeros(shape, dtype=jnp.float64),
            mean_motion_rad_day=jnp.zeros(shape, dtype=jnp.float64),
            r_au=jnp.zeros(shape, dtype=jnp.float64),
            snap_error=jnp.full(shape, jnp.inf, dtype=jnp.float64),
            plane_distance_au=jnp.zeros(shape, dtype=jnp.float64),
            curvature_hint=jnp.zeros(shape, dtype=jnp.float64),
            mask=jnp.zeros(shape, dtype=jnp.bool_),
        )


# Register as JAX PyTrees for transformations
jax.tree_util.register_pytree_node(
    BVHArrays,
    lambda bvh: (
        (
            bvh.nodes_min,
            bvh.nodes_max,
            bvh.left_child,
            bvh.right_child,
            bvh.is_leaf,
            bvh.first_prim,
            bvh.prim_count,
            bvh.prim_row_index,
            bvh.orbit_id_index,
            bvh.prim_seg_ids,
        ),
        None,
    ),
    lambda aux_data, children: BVHArrays(*children),
)

jax.tree_util.register_pytree_node(
    SegmentsSOA,
    lambda segs: (
        (
            segs.x0,
            segs.y0,
            segs.z0,
            segs.x1,
            segs.y1,
            segs.z1,
            segs.r_mid_au,
            segs.orbit_id_index,
            segs.n_x,
            segs.n_y,
            segs.n_z,
        ),
        None,
    ),
    lambda aux_data, children: SegmentsSOA(*children),
)

jax.tree_util.register_pytree_node(
    HitsSOA,
    lambda hits: (
        (
            hits.det_indices,
            hits.orbit_indices,
            hits.seg_ids,
            hits.leaf_ids,
            hits.distances_au,
        ),
        None,
    ),
    lambda aux_data, children: HitsSOA(*children),
)

jax.tree_util.register_pytree_node(
    AnomalyLabelsSOA,
    lambda labels: (
        (
            labels.det_indices,
            labels.orbit_indices,
            labels.seg_ids,
            labels.variant_ids,
            labels.f_rad,
            labels.E_rad,
            labels.M_rad,
            labels.mean_motion_rad_day,
            labels.r_au,
            labels.snap_error,
            labels.plane_distance_au,
            labels.curvature_hint,
            labels.mask,
        ),
        None,
    ),
    lambda aux_data, children: AnomalyLabelsSOA(*children),
)


# Persistence functions for JAX data structures


def save_bvh_arrays(bvh: BVHArrays, filepath: Union[str, Path]) -> None:
    """
    Save BVHArrays to disk with memory mapping support.

    Saves arrays to compressed .npz file and metadata to .json file.

    Parameters
    ----------
    bvh : BVHArrays
        BVH arrays to save
    filepath : str or Path
        Base path for saving (without extension)

    Examples
    --------
    >>> save_bvh_arrays(bvh_arrays, "data/orbit_bvh")
    # Creates: data/orbit_bvh.npz, data/orbit_bvh.json
    """
    filepath = Path(filepath)

    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert JAX arrays to numpy for saving
    arrays_np = jax.device_get(bvh)

    # Save arrays to compressed npz
    np.savez_compressed(
        filepath.with_suffix(".npz"),
        nodes_min=arrays_np.nodes_min,
        nodes_max=arrays_np.nodes_max,
        left_child=arrays_np.left_child,
        right_child=arrays_np.right_child,
        is_leaf=arrays_np.is_leaf,
        first_prim=arrays_np.first_prim,
        prim_count=arrays_np.prim_count,
        prim_row_index=arrays_np.prim_row_index,
        orbit_id_index=arrays_np.orbit_id_index,
        prim_seg_ids=arrays_np.prim_seg_ids,
    )

    # Save metadata to JSON
    metadata = {
        "type": "BVHArrays",
        "num_nodes": int(bvh.num_nodes),
        "num_primitives": int(bvh.num_primitives),
        "dtypes": {
            "nodes_min": str(arrays_np.nodes_min.dtype),
            "nodes_max": str(arrays_np.nodes_max.dtype),
            "left_child": str(arrays_np.left_child.dtype),
            "right_child": str(arrays_np.right_child.dtype),
            "is_leaf": str(arrays_np.is_leaf.dtype),
            "first_prim": str(arrays_np.first_prim.dtype),
            "prim_count": str(arrays_np.prim_count.dtype),
            "prim_row_index": str(arrays_np.prim_row_index.dtype),
            "orbit_id_index": str(arrays_np.orbit_id_index.dtype),
            "prim_seg_ids": str(arrays_np.prim_seg_ids.dtype),
        },
        "shapes": {
            "nodes_min": list(arrays_np.nodes_min.shape),
            "nodes_max": list(arrays_np.nodes_max.shape),
            "left_child": list(arrays_np.left_child.shape),
            "right_child": list(arrays_np.right_child.shape),
            "is_leaf": list(arrays_np.is_leaf.shape),
            "first_prim": list(arrays_np.first_prim.shape),
            "prim_count": list(arrays_np.prim_count.shape),
            "prim_row_index": list(arrays_np.prim_row_index.shape),
            "orbit_id_index": list(arrays_np.orbit_id_index.shape),
            "prim_seg_ids": list(arrays_np.prim_seg_ids.shape),
        },
    }

    with open(filepath.with_suffix(".json"), "w") as f:
        json.dump(metadata, f, indent=2)


def load_bvh_arrays(
    filepath: Union[str, Path],
    mmap_mode: Optional[str] = "r",
    device: Optional[jax.Device] = None,
) -> BVHArrays:
    """
    Load BVHArrays from disk with optional memory mapping.

    Parameters
    ----------
    filepath : str or Path
        Base path for loading (without extension)
    mmap_mode : str, optional
        Memory mapping mode for numpy arrays:
        - 'r': read-only (default, recommended for workers)
        - 'r+': read-write
        - None: load into memory (no memory mapping)
    device : jax.Device, optional
        JAX device to place arrays on (default: CPU)

    Returns
    -------
    BVHArrays
        Loaded BVH arrays

    Examples
    --------
    >>> bvh = load_bvh_arrays("data/orbit_bvh", mmap_mode='r')
    >>> bvh = load_bvh_arrays("data/orbit_bvh", mmap_mode=None)  # In memory
    """
    filepath = Path(filepath)

    # Load metadata
    with open(filepath.with_suffix(".json"), "r") as f:
        metadata = json.load(f)

    if metadata["type"] != "BVHArrays":
        raise ValueError(f"Expected BVHArrays, got {metadata['type']}")

    # Load arrays
    if mmap_mode is not None:
        data = np.load(filepath.with_suffix(".npz"), mmap_mode=mmap_mode)
    else:
        data = np.load(filepath.with_suffix(".npz"))

    # Reconstruct BVHArrays
    bvh = BVHArrays(
        nodes_min=jnp.asarray(data["nodes_min"]),
        nodes_max=jnp.asarray(data["nodes_max"]),
        left_child=jnp.asarray(data["left_child"]),
        right_child=jnp.asarray(data["right_child"]),
        is_leaf=jnp.asarray(data["is_leaf"]),
        first_prim=jnp.asarray(data["first_prim"]),
        prim_count=jnp.asarray(data["prim_count"]),
        prim_row_index=jnp.asarray(data["prim_row_index"]),
        orbit_id_index=jnp.asarray(data["orbit_id_index"]),
        prim_seg_ids=jnp.asarray(data["prim_seg_ids"]),
    )

    # Move to specified device if requested
    if device is not None:
        bvh = jax.device_put(bvh, device)

    # Validate structure
    bvh.validate_structure()

    return bvh


def save_segments_soa(segments: SegmentsSOA, filepath: Union[str, Path]) -> None:
    """
    Save SegmentsSOA to disk with memory mapping support.

    Parameters
    ----------
    segments : SegmentsSOA
        Segments to save
    filepath : str or Path
        Base path for saving (without extension)

    Examples
    --------
    >>> save_segments_soa(segments_soa, "data/orbit_segments")
    """
    filepath = Path(filepath)

    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert JAX arrays to numpy for saving
    segments_np = jax.device_get(segments)

    # Prepare arrays dict
    arrays_dict = {
        "x0": segments_np.x0,
        "y0": segments_np.y0,
        "z0": segments_np.z0,
        "x1": segments_np.x1,
        "y1": segments_np.y1,
        "z1": segments_np.z1,
        "r_mid_au": segments_np.r_mid_au,
        "orbit_id_index": segments_np.orbit_id_index,
    }

    # Add normals if present
    if segments_np.n_x is not None:
        arrays_dict.update(
            {
                "n_x": segments_np.n_x,
                "n_y": segments_np.n_y,
                "n_z": segments_np.n_z,
            }
        )

    # Save arrays to compressed npz
    np.savez_compressed(filepath.with_suffix(".npz"), **arrays_dict)

    # Save metadata to JSON
    metadata = {
        "type": "SegmentsSOA",
        "num_segments": int(segments.num_segments),
        "has_normals": segments_np.n_x is not None,
        "dtypes": {k: str(v.dtype) for k, v in arrays_dict.items()},
        "shapes": {k: list(v.shape) for k, v in arrays_dict.items()},
    }

    with open(filepath.with_suffix(".json"), "w") as f:
        json.dump(metadata, f, indent=2)


def load_segments_soa(
    filepath: Union[str, Path],
    mmap_mode: Optional[str] = "r",
    device: Optional[jax.Device] = None,
) -> SegmentsSOA:
    """
    Load SegmentsSOA from disk with optional memory mapping.

    Parameters
    ----------
    filepath : str or Path
        Base path for loading (without extension)
    mmap_mode : str, optional
        Memory mapping mode for numpy arrays
    device : jax.Device, optional
        JAX device to place arrays on

    Returns
    -------
    SegmentsSOA
        Loaded segments
    """
    filepath = Path(filepath)

    # Load metadata
    with open(filepath.with_suffix(".json"), "r") as f:
        metadata = json.load(f)

    if metadata["type"] != "SegmentsSOA":
        raise ValueError(f"Expected SegmentsSOA, got {metadata['type']}")

    # Load arrays
    if mmap_mode is not None:
        data = np.load(filepath.with_suffix(".npz"), mmap_mode=mmap_mode)
    else:
        data = np.load(filepath.with_suffix(".npz"))

    # Reconstruct SegmentsSOA
    segments = SegmentsSOA(
        x0=jnp.asarray(data["x0"]),
        y0=jnp.asarray(data["y0"]),
        z0=jnp.asarray(data["z0"]),
        x1=jnp.asarray(data["x1"]),
        y1=jnp.asarray(data["y1"]),
        z1=jnp.asarray(data["z1"]),
        r_mid_au=jnp.asarray(data["r_mid_au"]),
        orbit_id_index=jnp.asarray(data["orbit_id_index"]).astype(jnp.int32),
        n_x=jnp.asarray(data["n_x"]) if "n_x" in data else None,
        n_y=jnp.asarray(data["n_y"]) if "n_y" in data else None,
        n_z=jnp.asarray(data["n_z"]) if "n_z" in data else None,
    )

    # Move to specified device if requested
    if device is not None:
        segments = jax.device_put(segments, device)

    # Validate structure
    segments.validate_structure()

    return segments
