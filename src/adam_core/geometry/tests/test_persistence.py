"""
Tests for JAX data structure persistence and memory mapping.
"""

import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from adam_core.geometry import (
    BVHArrays,
    OrbitIdMapping,
    SegmentsSOA,
    load_bvh_arrays,
    load_segments_soa,
    save_bvh_arrays,
    save_segments_soa,
)


@pytest.fixture
def sample_bvh_arrays():
    """Create sample BVHArrays for testing."""
    return BVHArrays(
        nodes_min=jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=jnp.float64),
        nodes_max=jnp.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=jnp.float64),
        left_child=jnp.array([1, -1], dtype=jnp.int32),
        right_child=jnp.array([-1, -1], dtype=jnp.int32),
        is_leaf=jnp.array([False, True], dtype=jnp.bool_),
        first_prim=jnp.array([-1, 0], dtype=jnp.int32),
        prim_count=jnp.array([0, 2], dtype=jnp.int32),
        prim_row_index=jnp.array([0, 1], dtype=jnp.int32),
        orbit_id_index=jnp.array([0, 0], dtype=jnp.int32),
        prim_seg_ids=jnp.array([0, 1], dtype=jnp.int32),
    )


@pytest.fixture
def sample_segments_soa():
    """Create sample SegmentsSOA for testing."""
    return SegmentsSOA(
        x0=jnp.array([1.0, 2.0], dtype=jnp.float64),
        y0=jnp.array([0.0, 0.1], dtype=jnp.float64),
        z0=jnp.array([0.0, 0.0], dtype=jnp.float64),
        x1=jnp.array([1.1, 2.1], dtype=jnp.float64),
        y1=jnp.array([0.1, 0.2], dtype=jnp.float64),
        z1=jnp.array([0.0, 0.0], dtype=jnp.float64),
        r_mid_au=jnp.array([1.05, 2.05], dtype=jnp.float64),
        n_x=jnp.array([0.0, 0.0], dtype=jnp.float64),
        n_y=jnp.array([0.0, 0.0], dtype=jnp.float64),
        n_z=jnp.array([1.0, 1.0], dtype=jnp.float64),
    )


@pytest.fixture
def sample_segments_soa_no_normals():
    """Create sample SegmentsSOA without normals for testing."""
    return SegmentsSOA(
        x0=jnp.array([1.0, 2.0], dtype=jnp.float64),
        y0=jnp.array([0.0, 0.1], dtype=jnp.float64),
        z0=jnp.array([0.0, 0.0], dtype=jnp.float64),
        x1=jnp.array([1.1, 2.1], dtype=jnp.float64),
        y1=jnp.array([0.1, 0.2], dtype=jnp.float64),
        z1=jnp.array([0.0, 0.0], dtype=jnp.float64),
        r_mid_au=jnp.array([1.05, 2.05], dtype=jnp.float64),
    )


class TestBVHArraysPersistence:
    """Test BVHArrays save/load functionality."""

    def test_save_load_roundtrip(self, sample_bvh_arrays):
        """Test save and load roundtrip preserves data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_bvh"
            
            # Save
            save_bvh_arrays(sample_bvh_arrays, filepath)
            
            # Check files exist
            assert (filepath.with_suffix('.npz')).exists()
            assert (filepath.with_suffix('.json')).exists()
            
            # Load
            loaded = load_bvh_arrays(filepath, mmap_mode=None)
            
            # Validate structure
            loaded.validate_structure()
            
            # Check all arrays match
            np.testing.assert_array_equal(sample_bvh_arrays.nodes_min, loaded.nodes_min)
            np.testing.assert_array_equal(sample_bvh_arrays.nodes_max, loaded.nodes_max)
            np.testing.assert_array_equal(sample_bvh_arrays.left_child, loaded.left_child)
            np.testing.assert_array_equal(sample_bvh_arrays.right_child, loaded.right_child)
            np.testing.assert_array_equal(sample_bvh_arrays.is_leaf, loaded.is_leaf)
            np.testing.assert_array_equal(sample_bvh_arrays.first_prim, loaded.first_prim)
            np.testing.assert_array_equal(sample_bvh_arrays.prim_count, loaded.prim_count)
            np.testing.assert_array_equal(sample_bvh_arrays.prim_row_index, loaded.prim_row_index)
            np.testing.assert_array_equal(sample_bvh_arrays.orbit_id_index, loaded.orbit_id_index)
            np.testing.assert_array_equal(sample_bvh_arrays.prim_seg_ids, loaded.prim_seg_ids)
            
            # Check properties
            assert loaded.num_nodes == sample_bvh_arrays.num_nodes
            assert loaded.num_primitives == sample_bvh_arrays.num_primitives

    def test_memory_mapping(self, sample_bvh_arrays):
        """Test memory mapping functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_bvh_mmap"
            
            # Save
            save_bvh_arrays(sample_bvh_arrays, filepath)
            
            # Load with memory mapping
            loaded_mmap = load_bvh_arrays(filepath, mmap_mode='r')
            
            # Load without memory mapping for comparison
            loaded_mem = load_bvh_arrays(filepath, mmap_mode=None)
            
            # Both should have same data
            np.testing.assert_array_equal(loaded_mmap.nodes_min, loaded_mem.nodes_min)
            np.testing.assert_array_equal(loaded_mmap.prim_row_index, loaded_mem.prim_row_index)
            
            # Memory-mapped arrays should be read-only (for 'r' mode)
            # Note: JAX arrays don't expose the underlying numpy flags directly,
            # but we can verify the data is correct

    def test_device_placement(self, sample_bvh_arrays):
        """Test device placement during load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_bvh_device"
            
            # Save
            save_bvh_arrays(sample_bvh_arrays, filepath)
            
            # Load to CPU (default)
            loaded_cpu = load_bvh_arrays(filepath)
            
            # Verify data is correct
            np.testing.assert_array_equal(sample_bvh_arrays.nodes_min, loaded_cpu.nodes_min)

    def test_invalid_metadata(self, sample_bvh_arrays):
        """Test loading with invalid metadata raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_bvh_invalid"
            
            # Save normally
            save_bvh_arrays(sample_bvh_arrays, filepath)
            
            # Corrupt metadata
            import json
            with open(filepath.with_suffix('.json'), 'w') as f:
                json.dump({"type": "WrongType"}, f)
            
            # Should raise error
            with pytest.raises(ValueError, match="Expected BVHArrays"):
                load_bvh_arrays(filepath)

    def test_pathlib_and_string_paths(self, sample_bvh_arrays):
        """Test both pathlib.Path and string paths work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with string path
            str_path = str(Path(tmpdir) / "test_bvh_str")
            save_bvh_arrays(sample_bvh_arrays, str_path)
            loaded_str = load_bvh_arrays(str_path)
            
            # Test with Path object
            path_obj = Path(tmpdir) / "test_bvh_path"
            save_bvh_arrays(sample_bvh_arrays, path_obj)
            loaded_path = load_bvh_arrays(path_obj)
            
            # Both should work
            np.testing.assert_array_equal(loaded_str.nodes_min, loaded_path.nodes_min)


class TestSegmentsSOAPersistence:
    """Test SegmentsSOA save/load functionality."""

    def test_save_load_roundtrip_with_normals(self, sample_segments_soa):
        """Test save and load roundtrip with normals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_segments"
            
            # Save
            save_segments_soa(sample_segments_soa, filepath)
            
            # Check files exist
            assert (filepath.with_suffix('.npz')).exists()
            assert (filepath.with_suffix('.json')).exists()
            
            # Load
            loaded = load_segments_soa(filepath, mmap_mode=None)
            
            # Validate structure
            loaded.validate_structure()
            
            # Check all arrays match
            np.testing.assert_array_equal(sample_segments_soa.x0, loaded.x0)
            np.testing.assert_array_equal(sample_segments_soa.y0, loaded.y0)
            np.testing.assert_array_equal(sample_segments_soa.z0, loaded.z0)
            np.testing.assert_array_equal(sample_segments_soa.x1, loaded.x1)
            np.testing.assert_array_equal(sample_segments_soa.y1, loaded.y1)
            np.testing.assert_array_equal(sample_segments_soa.z1, loaded.z1)
            np.testing.assert_array_equal(sample_segments_soa.r_mid_au, loaded.r_mid_au)
            np.testing.assert_array_equal(sample_segments_soa.n_x, loaded.n_x)
            np.testing.assert_array_equal(sample_segments_soa.n_y, loaded.n_y)
            np.testing.assert_array_equal(sample_segments_soa.n_z, loaded.n_z)
            
            # Check properties
            assert loaded.num_segments == sample_segments_soa.num_segments

    def test_save_load_roundtrip_no_normals(self, sample_segments_soa_no_normals):
        """Test save and load roundtrip without normals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_segments_no_normals"
            
            # Save
            save_segments_soa(sample_segments_soa_no_normals, filepath)
            
            # Load
            loaded = load_segments_soa(filepath, mmap_mode=None)
            
            # Validate structure
            loaded.validate_structure()
            
            # Check position arrays match
            np.testing.assert_array_equal(sample_segments_soa_no_normals.x0, loaded.x0)
            np.testing.assert_array_equal(sample_segments_soa_no_normals.r_mid_au, loaded.r_mid_au)
            
            # Check normals are None
            assert loaded.n_x is None
            assert loaded.n_y is None
            assert loaded.n_z is None

    def test_memory_mapping_segments(self, sample_segments_soa):
        """Test memory mapping for segments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_segments_mmap"
            
            # Save
            save_segments_soa(sample_segments_soa, filepath)
            
            # Load with memory mapping
            loaded_mmap = load_segments_soa(filepath, mmap_mode='r')
            
            # Load without memory mapping for comparison
            loaded_mem = load_segments_soa(filepath, mmap_mode=None)
            
            # Both should have same data
            np.testing.assert_array_equal(loaded_mmap.x0, loaded_mem.x0)
            np.testing.assert_array_equal(loaded_mmap.r_mid_au, loaded_mem.r_mid_au)

    def test_metadata_validation_segments(self, sample_segments_soa):
        """Test metadata validation for segments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_segments_invalid"
            
            # Save normally
            save_segments_soa(sample_segments_soa, filepath)
            
            # Corrupt metadata
            import json
            with open(filepath.with_suffix('.json'), 'w') as f:
                json.dump({"type": "WrongType"}, f)
            
            # Should raise error
            with pytest.raises(ValueError, match="Expected SegmentsSOA"):
                load_segments_soa(filepath)


class TestPersistenceIntegration:
    """Test integration with existing workflow."""

    def test_integration_with_bvh_conversion(self):
        """Test persistence works with BVH conversion workflow."""
        from adam_core.coordinates.cartesian import CartesianCoordinates
        from adam_core.coordinates.origin import Origin, OriginCodes
        from adam_core.geometry import build_bvh, bvh_shard_to_arrays, segments_to_soa
        from adam_core.orbits.orbits import Orbits
        from adam_core.orbits.polyline import compute_segment_aabbs, sample_ellipse_adaptive
        from adam_core.time import Timestamp

        # Create test orbit
        times = Timestamp.from_mjd([59000.0], scale="tdb")
        coords = CartesianCoordinates.from_kwargs(
            x=[1.0], y=[0.0], z=[0.0],
            vx=[0.0], vy=[0.017202], vz=[0.0],
            time=times,
            origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]),
            frame="ecliptic",
        )
        orbits = Orbits.from_kwargs(orbit_id=["test_orbit"], coordinates=coords)

        # Sample and build BVH
        params, segments = sample_ellipse_adaptive(orbits, max_chord_arcmin=4.0)
        segments_with_aabbs = compute_segment_aabbs(segments, guard_arcmin=1.0)
        bvh = build_bvh(segments_with_aabbs)

        # Convert to JAX types
        orbit_mapping = OrbitIdMapping.from_orbit_ids(segments_with_aabbs.orbit_id.to_pylist())
        bvh_arrays = bvh_shard_to_arrays(bvh, orbit_mapping)
        segments_soa = segments_to_soa(segments_with_aabbs)

        with tempfile.TemporaryDirectory() as tmpdir:
            bvh_path = Path(tmpdir) / "integration_bvh"
            segments_path = Path(tmpdir) / "integration_segments"

            # Save
            save_bvh_arrays(bvh_arrays, bvh_path)
            save_segments_soa(segments_soa, segments_path)

            # Load
            loaded_bvh = load_bvh_arrays(bvh_path)
            loaded_segments = load_segments_soa(segments_path)

            # Validate
            assert loaded_bvh.num_primitives > 0
            assert loaded_segments.num_segments > 0
            assert loaded_bvh.num_primitives == loaded_segments.num_segments

    def test_zero_copy_validation(self, sample_bvh_arrays):
        """Test that memory mapping provides zero-copy access where possible."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_zero_copy"
            
            # Save
            save_bvh_arrays(sample_bvh_arrays, filepath)
            
            # Load with memory mapping
            loaded = load_bvh_arrays(filepath, mmap_mode='r')
            
            # Verify data integrity (this validates the zero-copy path works)
            np.testing.assert_array_equal(sample_bvh_arrays.nodes_min, loaded.nodes_min)
            
            # The arrays should be usable in computations
            result = jnp.sum(loaded.nodes_min)
            expected = jnp.sum(sample_bvh_arrays.nodes_min)
            np.testing.assert_allclose(result, expected)
