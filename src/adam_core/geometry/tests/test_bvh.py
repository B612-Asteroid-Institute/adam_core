"""
Tests for BVH (Bounding Volume Hierarchy) implementation.
"""

import numpy as np
import pytest

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.geometry.bvh import BVHShard, build_bvh
from adam_core.orbits.orbits import Orbits
from adam_core.orbits.polyline import compute_segment_aabbs, sample_ellipse_adaptive
from adam_core.time import Timestamp


def create_test_segments():
    """Create test segments for BVH testing."""
    # Create simple test orbits
    times = Timestamp.from_mjd([59000.0, 59000.0], scale="tdb")
    
    coords = CartesianCoordinates.from_kwargs(
        x=[1.0, 1.5],
        y=[0.0, 0.0],
        z=[0.0, 0.0],
        vx=[0.0, 0.0],
        vy=[0.017202, 0.014],
        vz=[0.0, 0.0],
        time=times,
        origin=Origin.from_kwargs(code=[OriginCodes.SUN.name] * 2),
        frame="ecliptic",
    )
    
    orbits = Orbits.from_kwargs(
        orbit_id=["orbit_1", "orbit_2"],
        coordinates=coords,
    )
    
    # Sample and compute AABBs
    _, segments = sample_ellipse_adaptive(orbits, max_chord_arcmin=1.0)
    segments_with_aabbs = compute_segment_aabbs(segments, guard_arcmin=1.0)
    
    return segments_with_aabbs


class TestBVHShard:
    """Test BVHShard class."""
    
    def test_empty_bvh(self):
        """Test creating empty BVH."""
        bvh = BVHShard(
            nodes_min=np.empty((0, 3)),
            nodes_max=np.empty((0, 3)),
            left_child=np.empty(0, dtype=np.int32),
            right_child=np.empty(0, dtype=np.int32),
            first_prim=np.empty(0, dtype=np.int32),
            prim_count=np.empty(0, dtype=np.int32),
            prim_orbit_ids=[],
            prim_seg_ids=np.empty(0, dtype=np.int32),
            prim_row_index=np.empty(0, dtype=np.int32),
        )
        
        assert bvh.num_nodes == 0
        assert bvh.num_primitives == 0
        
    def test_bvh_properties(self):
        """Test BVH properties and methods."""
        # Create a simple BVH with one internal node and two leaves
        nodes_min = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
        nodes_max = np.array([[2, 2, 2], [1, 1, 1], [2, 2, 2]], dtype=np.float64)
        left_child = np.array([1, -1, -1], dtype=np.int32)
        right_child = np.array([2, -1, -1], dtype=np.int32)
        first_prim = np.array([-1, 0, 1], dtype=np.int32)
        prim_count = np.array([0, 1, 1], dtype=np.int32)
        prim_orbit_ids = ["orbit_1", "orbit_2"]
        prim_seg_ids = np.array([0, 0], dtype=np.int32)
        
        bvh = BVHShard(
            nodes_min=nodes_min,
            nodes_max=nodes_max,
            left_child=left_child,
            right_child=right_child,
            first_prim=first_prim,
            prim_count=prim_count,
            prim_orbit_ids=prim_orbit_ids,
            prim_seg_ids=prim_seg_ids,
            prim_row_index=np.array([0, 1], dtype=np.int32),
        )
        
        assert bvh.num_nodes == 3
        assert bvh.num_primitives == 2
        
        # Test leaf detection
        assert not bvh.is_leaf(0)  # Root is internal
        assert bvh.is_leaf(1)      # Left child is leaf
        assert bvh.is_leaf(2)      # Right child is leaf
        
        # Test leaf primitive access
        orbit_ids, seg_ids, row_indices = bvh.get_leaf_primitives(1)
        assert orbit_ids == ["orbit_1"]
        assert seg_ids.tolist() == [0]
        assert row_indices.tolist() == [0]
        
        orbit_ids, seg_ids, row_indices = bvh.get_leaf_primitives(2)
        assert orbit_ids == ["orbit_2"]
        assert seg_ids.tolist() == [0]
        assert row_indices.tolist() == [1]
        
        # Test error on non-leaf
        with pytest.raises(ValueError):
            bvh.get_leaf_primitives(0)


class TestBuildBvh:
    """Test BVH construction."""
    
    def test_empty_segments(self):
        """Test building BVH with empty segments."""
        from adam_core.orbits.polyline import OrbitPolylineSegments
        
        empty_segments = OrbitPolylineSegments.empty()
        bvh = build_bvh(empty_segments)
        
        assert bvh.num_nodes == 0
        assert bvh.num_primitives == 0
        
    def test_single_segment(self):
        """Test building BVH with single segment."""
        from adam_core.orbits.polyline import OrbitPolylineSegments
        
        segments = OrbitPolylineSegments.from_kwargs(
            orbit_id=["test"],
            seg_id=[0],
            x0=[0.0], y0=[0.0], z0=[0.0],
            x1=[1.0], y1=[0.0], z1=[0.0],
            aabb_min_x=[-0.1], aabb_min_y=[-0.1], aabb_min_z=[-0.1],
            aabb_max_x=[1.1], aabb_max_y=[0.1], aabb_max_z=[0.1],
            r_mid_au=[0.5],
            n_x=[0.0], n_y=[0.0], n_z=[1.0],
        )
        
        bvh = build_bvh(segments)
        
        assert bvh.num_nodes == 1
        assert bvh.num_primitives == 1
        assert bvh.is_leaf(0)
        
        orbit_ids, seg_ids, row_indices = bvh.get_leaf_primitives(0)
        assert orbit_ids == ["test"]
        assert seg_ids.tolist() == [0]
        assert row_indices.tolist() == [0]
        
    def test_multiple_segments(self):
        """Test building BVH with multiple segments."""
        segments = create_test_segments()
        bvh = build_bvh(segments, max_leaf_size=4)
        
        assert bvh.num_nodes > 0
        assert bvh.num_primitives == len(segments)
        
        # Check that all primitives are accounted for
        total_prims_in_leaves = 0
        for i in range(bvh.num_nodes):
            if bvh.is_leaf(i):
                total_prims_in_leaves += bvh.prim_count[i]
                
        assert total_prims_in_leaves == len(segments)
        
    def test_bvh_structure_validity(self):
        """Test that BVH structure is valid."""
        segments = create_test_segments()
        bvh = build_bvh(segments, max_leaf_size=2)
        
        # Check that parent AABBs contain child AABBs
        for i in range(bvh.num_nodes):
            if not bvh.is_leaf(i):
                left_idx = bvh.left_child[i]
                right_idx = bvh.right_child[i]
                
                # Parent bounds should contain child bounds
                parent_min = bvh.nodes_min[i]
                parent_max = bvh.nodes_max[i]
                
                if left_idx >= 0:
                    left_min = bvh.nodes_min[left_idx]
                    left_max = bvh.nodes_max[left_idx]
                    
                    assert np.all(parent_min <= left_min + 1e-10)  # Small tolerance for floating point
                    assert np.all(parent_max >= left_max - 1e-10)
                    
                if right_idx >= 0:
                    right_min = bvh.nodes_min[right_idx]
                    right_max = bvh.nodes_max[right_idx]
                    
                    assert np.all(parent_min <= right_min + 1e-10)
                    assert np.all(parent_max >= right_max - 1e-10)
                    
    def test_leaf_size_constraint(self):
        """Test that leaf size constraint is respected."""
        segments = create_test_segments()
        max_leaf_size = 3
        bvh = build_bvh(segments, max_leaf_size=max_leaf_size)
        
        # Check that no leaf exceeds max size
        for i in range(bvh.num_nodes):
            if bvh.is_leaf(i):
                assert bvh.prim_count[i] <= max_leaf_size
                
    def test_missing_aabbs_error(self):
        """Test that error is raised for segments without AABBs."""
        from adam_core.orbits.polyline import OrbitPolylineSegments
        
        segments = OrbitPolylineSegments.from_kwargs(
            orbit_id=["test"],
            seg_id=[0],
            x0=[0.0], y0=[0.0], z0=[0.0],
            x1=[1.0], y1=[0.0], z1=[0.0],
            aabb_min_x=[np.nan], aabb_min_y=[np.nan], aabb_min_z=[np.nan],  # Missing AABBs
            aabb_max_x=[np.nan], aabb_max_y=[np.nan], aabb_max_z=[np.nan],
            r_mid_au=[0.5],
            n_x=[0.0], n_y=[0.0], n_z=[1.0],
        )
        
        with pytest.raises(ValueError, match="Segments must have computed AABBs"):
            build_bvh(segments)
            
    def test_bvh_primitive_mapping(self):
        """Test that BVH correctly maps primitives."""
        segments = create_test_segments()
        bvh = build_bvh(segments)
        
        # Collect all primitives from leaves
        all_orbit_ids = []
        all_seg_ids = []
        
        for i in range(bvh.num_nodes):
            if bvh.is_leaf(i):
                orbit_ids, seg_ids, row_indices = bvh.get_leaf_primitives(i)
                all_orbit_ids.extend(orbit_ids)
                all_seg_ids.extend(seg_ids.tolist())
                
        # Should match original segments
        original_orbit_ids = segments.orbit_id.to_pylist()
        original_seg_ids = segments.seg_id.to_pylist()
        
        assert len(all_orbit_ids) == len(original_orbit_ids)
        assert len(all_seg_ids) == len(original_seg_ids)
        
        # Order might be different due to BVH construction, so check sets
        assert set(all_orbit_ids) == set(original_orbit_ids)
        
    def test_large_bvh_performance(self):
        """Test BVH construction with larger dataset."""
        # Create more orbits for performance test
        n_orbits = 20
        times = Timestamp.from_mjd([59000.0] * n_orbits, scale="tdb")
        
        # Create varied orbits
        x_coords = np.linspace(0.5, 2.0, n_orbits)
        y_coords = np.zeros(n_orbits)
        z_coords = np.linspace(0.0, 0.5, n_orbits)
        vx_coords = np.zeros(n_orbits)
        vy_coords = np.linspace(0.01, 0.02, n_orbits)
        vz_coords = np.linspace(0.0, 0.01, n_orbits)
        
        coords = CartesianCoordinates.from_kwargs(
            x=x_coords, y=y_coords, z=z_coords,
            vx=vx_coords, vy=vy_coords, vz=vz_coords,
            time=times,
            origin=Origin.from_kwargs(code=[OriginCodes.SUN.name] * n_orbits),
            frame="ecliptic",
        )
        
        orbits = Orbits.from_kwargs(
            orbit_id=[f"orbit_{i}" for i in range(n_orbits)],
            coordinates=coords,
        )
        
        # Sample and build BVH
        _, segments = sample_ellipse_adaptive(orbits, max_chord_arcmin=0.5)
        segments_with_aabbs = compute_segment_aabbs(segments)
        
        bvh = build_bvh(segments_with_aabbs, max_leaf_size=8)
        
        # Should have reasonable structure
        assert bvh.num_nodes > 1  # Should not be just one leaf
        assert bvh.num_primitives == len(segments_with_aabbs)
        
        # Tree should be reasonably balanced (not too deep)
        # For n primitives, expect depth roughly log2(n/leaf_size)
        expected_max_depth = int(np.log2(len(segments_with_aabbs) / 8)) + 3  # Some tolerance
        
        # Check depth by traversing from root
        def compute_depth(node_idx, current_depth=0):
            if bvh.is_leaf(node_idx):
                return current_depth
            
            left_depth = 0
            right_depth = 0
            
            if bvh.left_child[node_idx] >= 0:
                left_depth = compute_depth(bvh.left_child[node_idx], current_depth + 1)
            if bvh.right_child[node_idx] >= 0:
                right_depth = compute_depth(bvh.right_child[node_idx], current_depth + 1)
                
            return max(left_depth, right_depth)
            
        actual_depth = compute_depth(0)
        assert actual_depth <= expected_max_depth


class TestIntegration:
    """Integration tests for BVH with polyline sampling."""
    
    def test_full_m1_pipeline(self):
        """Test complete M1 pipeline: sample -> AABB -> BVH."""
        # Create test orbits
        times = Timestamp.from_mjd([59000.0, 59000.0, 59000.0], scale="tdb")
        
        coords = CartesianCoordinates.from_kwargs(
            x=[1.0, 1.5, 0.8],
            y=[0.0, 0.0, 0.2],
            z=[0.0, 0.0, 0.1],
            vx=[0.0, 0.0, 0.0],
            vy=[0.017202, 0.014, 0.018],
            vz=[0.0, 0.0, 0.002],
            time=times,
            origin=Origin.from_kwargs(code=[OriginCodes.SUN.name] * 3),
            frame="ecliptic",
        )
        
        orbits = Orbits.from_kwargs(
            orbit_id=["orbit_a", "orbit_b", "orbit_c"],
            coordinates=coords,
        )
        
        # Run complete pipeline
        params, segments = sample_ellipse_adaptive(orbits, max_chord_arcmin=0.5)
        segments_with_aabbs = compute_segment_aabbs(segments, guard_arcmin=1.0)
        bvh = build_bvh(segments_with_aabbs, max_leaf_size=4)
        
        # Verify results
        assert len(params) == 3
        assert len(segments_with_aabbs) > 0
        assert bvh.num_primitives == len(segments_with_aabbs)
        assert bvh.num_nodes > 0
        
        # Verify all orbits are represented
        orbit_ids_in_bvh = set(bvh.prim_orbit_ids)
        orbit_ids_original = set(orbits.orbit_id.to_pylist())
        assert orbit_ids_in_bvh == orbit_ids_original
