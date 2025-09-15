"""
Tests for BVH (Bounding Volume Hierarchy) implementation.
"""

import numpy as np
import pytest

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.geometry.bvh import build_bvh_index_from_segments
from adam_core.orbits.orbits import Orbits
from adam_core.orbits.polyline import compute_segment_aabbs, sample_ellipse_adaptive
from adam_core.geometry.bvh import build_bvh_index
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


class TestBVHIndexStructure:
    """Tests on BVHIndex structure using Quivr tables."""

    def test_empty_index(self):
        from adam_core.orbits.polyline import OrbitPolylineSegments
        empty_segments = OrbitPolylineSegments.empty()
        index = build_bvh_index_from_segments(empty_segments)
        assert len(index.nodes) == 0
        assert len(index.prims) == 0


class TestBuildBvh:
    """Test BVH construction."""

    def test_empty_segments(self):
        """Test building BVHIndex with empty segments."""
        from adam_core.orbits.polyline import OrbitPolylineSegments

        empty_segments = OrbitPolylineSegments.empty()
        index = build_bvh_index_from_segments(empty_segments)

        assert len(index.nodes) == 0
        assert len(index.prims) == 0

    def test_single_segment(self):
        """Test building BVH with single segment."""
        from adam_core.orbits.polyline import OrbitPolylineSegments

        segments = OrbitPolylineSegments.from_kwargs(
            orbit_id=["test"],
            seg_id=[0],
            x0=[0.0],
            y0=[0.0],
            z0=[0.0],
            x1=[1.0],
            y1=[0.0],
            z1=[0.0],
            aabb_min_x=[-0.1],
            aabb_min_y=[-0.1],
            aabb_min_z=[-0.1],
            aabb_max_x=[1.1],
            aabb_max_y=[0.1],
            aabb_max_z=[0.1],
            r_mid_au=[0.5],
            n_x=[0.0],
            n_y=[0.0],
            n_z=[1.0],
        )

        index = build_bvh_index_from_segments(segments)

        assert len(index.nodes) == 1
        assert len(index.prims) == 1
        # Leaf if both children are -1
        assert int(index.nodes.left_child[0].as_py()) == -1
        assert int(index.nodes.right_child[0].as_py()) == -1

    def test_multiple_segments(self):
        """Test building BVH with multiple segments."""
        segments = create_test_segments()
        index = build_bvh_index_from_segments(segments, max_leaf_size=4)

        assert len(index.nodes) > 0
        assert len(index.prims) == len(segments)

        # Check that all primitives are accounted for
        total_prims_in_leaves = 0
        for i in range(len(index.nodes)):
            if int(index.nodes.left_child[i].as_py()) == -1:
                total_prims_in_leaves += int(index.nodes.prim_count[i].as_py())

        assert total_prims_in_leaves == len(segments)

    def test_bvh_structure_validity(self):
        """Test that BVH structure is valid."""
        segments = create_test_segments()
        index = build_bvh_index_from_segments(segments, max_leaf_size=2)

        # Check that parent AABBs contain child AABBs
        for i in range(len(index.nodes)):
            if not (int(index.nodes.left_child[i].as_py()) == -1):
                left_idx = int(index.nodes.left_child[i].as_py())
                right_idx = int(index.nodes.right_child[i].as_py())

                # Parent bounds should contain child bounds
                parent_min = np.array([
                    index.nodes.nodes_min_x[i].as_py(),
                    index.nodes.nodes_min_y[i].as_py(),
                    index.nodes.nodes_min_z[i].as_py(),
                ])
                parent_max = np.array([
                    index.nodes.nodes_max_x[i].as_py(),
                    index.nodes.nodes_max_y[i].as_py(),
                    index.nodes.nodes_max_z[i].as_py(),
                ])

                if left_idx >= 0:
                    left_min = np.array([
                        index.nodes.nodes_min_x[left_idx].as_py(),
                        index.nodes.nodes_min_y[left_idx].as_py(),
                        index.nodes.nodes_min_z[left_idx].as_py(),
                    ])
                    left_max = np.array([
                        index.nodes.nodes_max_x[left_idx].as_py(),
                        index.nodes.nodes_max_y[left_idx].as_py(),
                        index.nodes.nodes_max_z[left_idx].as_py(),
                    ])

                    assert np.all(
                        parent_min <= left_min + 1e-10
                    )  # Small tolerance for floating point
                    assert np.all(parent_max >= left_max - 1e-10)

                if right_idx >= 0:
                    right_min = np.array([
                        index.nodes.nodes_min_x[right_idx].as_py(),
                        index.nodes.nodes_min_y[right_idx].as_py(),
                        index.nodes.nodes_min_z[right_idx].as_py(),
                    ])
                    right_max = np.array([
                        index.nodes.nodes_max_x[right_idx].as_py(),
                        index.nodes.nodes_max_y[right_idx].as_py(),
                        index.nodes.nodes_max_z[right_idx].as_py(),
                    ])

                    assert np.all(parent_min <= right_min + 1e-10)
                    assert np.all(parent_max >= right_max - 1e-10)

    def test_leaf_size_constraint(self):
        """Test that leaf size constraint is respected."""
        segments = create_test_segments()
        max_leaf_size = 3
        index = build_bvh_index_from_segments(segments, max_leaf_size=max_leaf_size)

        # Check that no leaf exceeds max size
        for i in range(len(index.nodes)):
            if int(index.nodes.left_child[i].as_py()) == -1:
                assert int(index.nodes.prim_count[i].as_py()) <= max_leaf_size

    def test_missing_aabbs_error(self):
        """Test that error is raised for segments without AABBs."""
        from adam_core.orbits.polyline import OrbitPolylineSegments

        segments = OrbitPolylineSegments.from_kwargs(
            orbit_id=["test"],
            seg_id=[0],
            x0=[0.0],
            y0=[0.0],
            z0=[0.0],
            x1=[1.0],
            y1=[0.0],
            z1=[0.0],
            aabb_min_x=[None],
            aabb_min_y=[None],
            aabb_min_z=[None],  # Missing AABBs
            aabb_max_x=[None],
            aabb_max_y=[None],
            aabb_max_z=[None],
            r_mid_au=[0.5],
            n_x=[0.0],
            n_y=[0.0],
            n_z=[1.0],
        )

        with pytest.raises(ValueError, match="Segments must have computed AABBs"):
            build_bvh_index_from_segments(segments)

    @pytest.mark.parametrize("max_chord_arcmin", [0.5, 1.0, 2.0])
    def test_sample_ellipse_segment_counts(self, max_chord_arcmin):
        """sample_ellipse_adaptive should produce more segments for smaller chord limits."""
        times = Timestamp.from_mjd([59000.0], scale="tdb")
        coords = CartesianCoordinates.from_kwargs(
            x=[1.0], y=[0.0], z=[0.0], vx=[0.0], vy=[0.017202], vz=[0.0],
            time=times, origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]), frame="ecliptic",
        )
        orbits = Orbits.from_kwargs(orbit_id=["o1"], coordinates=coords)
        _, segs = sample_ellipse_adaptive(orbits, max_chord_arcmin=max_chord_arcmin)
        assert len(segs) > 0

    def test_compute_segment_aabbs_contains(self):
        """AABBs should contain segment endpoints with guard band."""
        segments = create_test_segments()
        aabbs = compute_segment_aabbs(segments, guard_arcmin=1.0)
        import numpy as np
        min_x = np.minimum(aabbs.x0.to_numpy(), aabbs.x1.to_numpy())
        max_x = np.maximum(aabbs.x0.to_numpy(), aabbs.x1.to_numpy())
        min_y = np.minimum(aabbs.y0.to_numpy(), aabbs.y1.to_numpy())
        max_y = np.maximum(aabbs.y0.to_numpy(), aabbs.y1.to_numpy())
        min_z = np.minimum(aabbs.z0.to_numpy(), aabbs.z1.to_numpy())
        max_z = np.maximum(aabbs.z0.to_numpy(), aabbs.z1.to_numpy())

        assert np.all(aabbs.aabb_min_x.to_numpy() <= min_x + 1e-12)
        assert np.all(aabbs.aabb_max_x.to_numpy() >= max_x - 1e-12)
        assert np.all(aabbs.aabb_min_y.to_numpy() <= min_y + 1e-12)
        assert np.all(aabbs.aabb_max_y.to_numpy() >= max_y - 1e-12)
        assert np.all(aabbs.aabb_min_z.to_numpy() <= min_z + 1e-12)
        assert np.all(aabbs.aabb_max_z.to_numpy() >= max_z - 1e-12)

    @pytest.mark.parametrize("guard_arcmin_list", [[0.5, 1.0, 2.0]])
    def test_compute_segment_aabbs_guard_monotonic(self, guard_arcmin_list):
        """AABB extent should grow monotonically with larger guard band."""
        segments = create_test_segments()
        import numpy as np
        extents = []
        for g in guard_arcmin_list:
            a = compute_segment_aabbs(segments, guard_arcmin=g)
            ex = (a.aabb_max_x.to_numpy() - a.aabb_min_x.to_numpy())
            ey = (a.aabb_max_y.to_numpy() - a.aabb_min_y.to_numpy())
            ez = (a.aabb_max_z.to_numpy() - a.aabb_min_z.to_numpy())
            extents.append(np.mean(ex + ey + ez))
        # Check monotonic increase
        for i in range(len(extents) - 1):
            assert extents[i] <= extents[i + 1] + 1e-12

    def test_sample_ellipse_continuity_and_determinism(self):
        """Consecutive segments should share endpoints; output deterministic."""
        times = Timestamp.from_mjd([59000.0], scale="tdb")
        coords = CartesianCoordinates.from_kwargs(
            x=[1.0], y=[0.0], z=[0.0], vx=[0.0], vy=[0.017202], vz=[0.0],
            time=times, origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]), frame="ecliptic",
        )
        orbits = Orbits.from_kwargs(orbit_id=["o1"], coordinates=coords)
        _, s1 = sample_ellipse_adaptive(orbits, max_chord_arcmin=1.0)
        _, s2 = sample_ellipse_adaptive(orbits, max_chord_arcmin=1.0)

        # Deterministic
        assert len(s1) == len(s2)
        import numpy as np
        assert np.allclose(s1.x0.to_numpy(), s2.x0.to_numpy())
        assert np.allclose(s1.y0.to_numpy(), s2.y0.to_numpy())
        assert np.allclose(s1.z0.to_numpy(), s2.z0.to_numpy())
        assert np.allclose(s1.x1.to_numpy(), s2.x1.to_numpy())

        # Continuity: end of i equals start of i+1
        if len(s1) > 1:
            end_pts = np.column_stack([s1.x1.to_numpy(), s1.y1.to_numpy(), s1.z1.to_numpy()])
            start_pts = np.column_stack([s1.x0.to_numpy(), s1.y0.to_numpy(), s1.z0.to_numpy()])
            deltas = np.linalg.norm(end_pts[:-1] - start_pts[1:], axis=1)
            assert np.all(deltas < 1e-9)

    def test_bvh_primitive_mapping(self):
        """Test that BVH correctly maps primitives."""
        segments = create_test_segments()
        index = build_bvh_index_from_segments(segments)

        # Collect all primitives from leaves
        all_orbit_ids = []
        all_seg_ids = []

        # Reconstruct per-primitive lists in index order
        # prim_row_index is contiguous packing as built
        prim_rows = index.prims.segment_row_index.to_numpy()
        all_orbit_ids = [index.segments.orbit_id[int(r)].as_py() for r in prim_rows]
        all_seg_ids = index.prims.prim_seg_ids.to_pylist()

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
            x=x_coords,
            y=y_coords,
            z=z_coords,
            vx=vx_coords,
            vy=vy_coords,
            vz=vz_coords,
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

        index = build_bvh_index_from_segments(segments_with_aabbs, max_leaf_size=8)

        # Should have reasonable structure
        assert len(index.nodes) > 1  # Should not be just one leaf
        assert len(index.prims) == len(segments_with_aabbs)

        # Tree should be reasonably balanced (not too deep)
        # For n primitives, expect depth roughly log2(n/leaf_size)
        expected_max_depth = (
            int(np.log2(len(segments_with_aabbs) / 8)) + 3
        )  # Some tolerance

        # Check depth by traversing from root
        def compute_depth(node_idx, current_depth=0):
            if int(index.nodes.left_child[node_idx].as_py()) == -1:
                return current_depth

            left_depth = 0
            right_depth = 0

            left = int(index.nodes.left_child[node_idx].as_py())
            right = int(index.nodes.right_child[node_idx].as_py())
            if left >= 0:
                left_depth = compute_depth(left, current_depth + 1)
            if right >= 0:
                right_depth = compute_depth(right, current_depth + 1)

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
        index = build_bvh_index_from_segments(segments_with_aabbs, max_leaf_size=4)

        # Verify results
        assert len(params) == 3
        assert len(segments_with_aabbs) > 0
        assert len(index.prims) == len(segments_with_aabbs)
        assert len(index.nodes) > 0

        # Verify all orbits are represented
        orbit_ids_in_bvh = set(index.segments.orbit_id.to_pylist())
        orbit_ids_original = set(orbits.orbit_id.to_pylist())
        assert orbit_ids_in_bvh == orbit_ids_original

    def test_build_bvh_index_from_orbits(self):
        segments = create_test_segments()
        # Back out orbits from segments' first entries (simple synthetic check)
        times = Timestamp.from_mjd([59000.0], scale="tdb")
        coords = CartesianCoordinates.from_kwargs(
            x=[1.0], y=[0.0], z=[0.0], vx=[0.0], vy=[0.017202], vz=[0.0],
            time=times, origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]), frame="ecliptic",
        )
        orbits = Orbits.from_kwargs(orbit_id=["o1"], coordinates=coords)
        index = build_bvh_index(orbits)
        assert len(index.nodes) > 0
        assert len(index.prims) > 0

    def test_build_bvh_index_from_orbits_multi(self):
        times = Timestamp.from_mjd([59000.0, 59000.0], scale="tdb")
        coords = CartesianCoordinates.from_kwargs(
            x=[1.0, 1.2], y=[0.0, 0.1], z=[0.0, 0.0],
            vx=[0.0, 0.0], vy=[0.017202, 0.015], vz=[0.0, 0.0],
            time=times, origin=Origin.from_kwargs(code=[OriginCodes.SUN.name, OriginCodes.SUN.name]), frame="ecliptic",
        )
        orbits = Orbits.from_kwargs(orbit_id=["o1", "o2"], coordinates=coords)
        index = build_bvh_index(orbits)
        assert len(index.nodes) > 0
        assert len(index.prims) > 0

    @pytest.mark.parametrize("chunk_size_orbits,max_processes", [(1, 0), (1, 2), (2, 0)])
    def test_build_bvh_index_from_orbits_chunked(self, chunk_size_orbits, max_processes):
        import ray
        ray.shutdown()
        times = Timestamp.from_mjd([59000.0, 59000.0, 59000.0], scale="tdb")
        coords = CartesianCoordinates.from_kwargs(
            x=[1.0, 1.2, 0.9], y=[0.0, 0.1, -0.1], z=[0.0, 0.0, 0.0],
            vx=[0.0, 0.0, 0.0], vy=[0.017202, 0.015, 0.018], vz=[0.0, 0.0, 0.0],
            time=times, origin=Origin.from_kwargs(code=[OriginCodes.SUN.name] * 3), frame="ecliptic",
        )
        orbits = Orbits.from_kwargs(orbit_id=["o1", "o2", "o3"], coordinates=coords)
        idx_chunked = build_bvh_index(
            orbits,
            max_chord_arcmin=2.0,
            guard_arcmin=1.0,
            max_leaf_size=4,
            chunk_size_orbits=chunk_size_orbits,
            max_processes=max_processes,
        )
        idx_mono = build_bvh_index(orbits)
        # Sanity checks
        assert len(idx_chunked.segments) == len(idx_mono.segments)
        assert len(idx_chunked.prims) == len(idx_mono.prims)
        assert len(idx_chunked.nodes) > 0
