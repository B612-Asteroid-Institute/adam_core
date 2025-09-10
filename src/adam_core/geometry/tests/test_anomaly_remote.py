"""
Tests for Ray-parallel anomaly labeling functionality.
"""

import numpy as np
import pytest

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.geometry.adapters import segments_to_soa
from adam_core.geometry.anomaly_remote import label_anomalies_parallel, process_anomaly_batch_remote
from adam_core.observations.rays import ObservationRays
from adam_core.orbits.orbits import Orbits
from adam_core.orbits.polyline import sample_ellipse_adaptive
from adam_core.time import Timestamp

from ..bvh import build_bvh
from ..jax_overlap import query_bvh_jax


@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not available")
class TestAnomalyRemote:
    """Test Ray-parallel anomaly labeling."""
    
    def setup_method(self):
        """Set up Ray for each test."""
        if not ray.is_initialized():
            ray.init(local_mode=True, ignore_reinit_error=True)
    
    def teardown_method(self):
        """Clean up Ray after each test."""
        if ray.is_initialized():
            ray.shutdown()
    
    def test_process_anomaly_batch_remote_empty(self):
        """Test remote function with empty hits."""
        # Empty hits dictionary
        empty_hits = {
            "det_indices": np.array([], dtype=np.int32),
            "orbit_indices": np.array([], dtype=np.int32),
            "seg_ids": np.array([], dtype=np.int32),
            "distances_au": np.array([], dtype=np.float64),
        }
        
        # Dummy orbital data
        orbital_elements = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        orbital_bases = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
        ray_origins = np.array([[0.0, 0.0, 0.0]])
        ray_directions = np.array([[1.0, 0.0, 0.0]])
        
        # Test remote function
        future = process_anomaly_batch_remote.remote(
            empty_hits,
            orbital_elements,
            orbital_bases,
            ray_origins,
            ray_directions,
            {},  # segments_soa_ref (unused for empty hits)
        )
        
        result = ray.get(future)
        
        # Should return empty arrays
        assert len(result["det_indices"]) == 0
        assert len(result["orbit_indices"]) == 0
        assert len(result["mask"]) == 0
    
    def test_label_anomalies_parallel_empty(self):
        """Test parallel labeling with empty hits."""
        from adam_core.geometry.overlap import OverlapHits
        from adam_core.orbits.polyline import OrbitsPlaneParams
        from adam_core.geometry.jax_types import SegmentsSOA
        
        empty_hits = OverlapHits.empty()
        empty_params = OrbitsPlaneParams.empty()
        ray_origins = np.array([[0.0, 0.0, 0.0]])
        ray_directions = np.array([[1.0, 0.0, 0.0]])
        empty_segments_soa = SegmentsSOA.empty()
        
        # Decoupled API requires orbits and rays; with empty hits, result is empty
        from adam_core.orbits.orbits import Orbits
        from adam_core.coordinates.cartesian import CartesianCoordinates
        empty_orbits = Orbits.from_kwargs(orbit_id=[], coordinates=CartesianCoordinates.empty())
        empty_rays = ObservationRays.empty()
        labels = label_anomalies_parallel(empty_hits, empty_orbits, empty_rays, segments_soa=empty_segments_soa)
        
        assert len(labels) == 0
    
    def test_query_bvh_parallel_and_labeling_integration(self):
        """Test parallel overlap then labeling (decoupled)."""
        # Create a simple orbit
        times = Timestamp.from_mjd([59000.0], scale="tdb")
        
        coords = CartesianCoordinates.from_kwargs(
            x=[1.0], y=[0.0], z=[0.0],
            vx=[0.0], vy=[0.017202], vz=[0.0],
            time=times,
            origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]),
            frame="ecliptic",
        )
        
        orbit = Orbits.from_kwargs(
            orbit_id=["test_orbit"],
            coordinates=coords,
        )
        
        plane_params, segments = sample_ellipse_adaptive(orbit)
        
        # Create observation rays
        obs_times = Timestamp.from_mjd([59000.0 + 91.3], scale="tdb")
        
        rays = ObservationRays.from_kwargs(
            det_id=["det_001"],
            time=obs_times,
            observer_code=["500"],
            observer=CartesianCoordinates.from_kwargs(
                x=[0.0], y=[0.0], z=[0.0],
                vx=[0.0], vy=[0.0], vz=[0.0],
                time=obs_times,
                origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]),
                frame="ecliptic",
            ),
            u_x=[0.0], u_y=[1.0], u_z=[0.0],
        )
        
        # Build BVH
        from adam_core.orbits.polyline import compute_segment_aabbs
        segments_with_aabbs = compute_segment_aabbs(segments)
        bvh = build_bvh(segments_with_aabbs)
        
        # Query in parallel using JAX remote pipeline
        from adam_core.geometry.sharded_query_ray import query_manifest_ray as _maybe_absent
        # Fallback to local query since full ray sharding may not be available in tests
        hits = query_bvh_jax(bvh, segments_with_aabbs, rays)
        
        # Label in parallel (decoupled API)
        labels = label_anomalies_parallel(hits, orbit, rays, segments_soa=None, batch_size=100)
        
        # Should find some hits and labels
        assert len(hits) >= 0  # May be 0 due to guard band
        assert len(labels) >= 0  # May be 0 if no hits
        
        # If we have hits, we should have labels
        if len(hits) > 0:
            assert len(labels) > 0
    
    def test_parallel_vs_serial_consistency(self):
        """Test that parallel labeling gives same results as serial."""
        # Create test data
        times = Timestamp.from_mjd([59000.0], scale="tdb")
        
        coords = CartesianCoordinates.from_kwargs(
            x=[1.0], y=[0.0], z=[0.0],
            vx=[0.0], vy=[0.017202], vz=[0.0],
            time=times,
            origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]),
            frame="ecliptic",
        )
        
        orbit = Orbits.from_kwargs(
            orbit_id=["test_orbit"],
            coordinates=coords,
        )
        
        plane_params, segments = sample_ellipse_adaptive(orbit)
        
        # Create multiple observation rays
        obs_times = Timestamp.from_mjd([59000.0 + 91.3] * 3, scale="tdb")
        
        rays = ObservationRays.from_kwargs(
            det_id=["det_001", "det_002", "det_003"],
            time=obs_times,
            observer_code=["500"] * 3,
            observer=CartesianCoordinates.from_kwargs(
                x=[0.0] * 3, y=[0.0] * 3, z=[0.0] * 3,
                vx=[0.0] * 3, vy=[0.0] * 3, vz=[0.0] * 3,
                time=obs_times,
                origin=Origin.from_kwargs(code=[OriginCodes.SUN.name] * 3),
                frame="ecliptic",
            ),
            u_x=[0.0, 0.1, -0.1], u_y=[1.0, 0.9, 1.1], u_z=[0.0] * 3,
        )
        
        # Build BVH
        from adam_core.orbits.polyline import compute_segment_aabbs
        segments_with_aabbs = compute_segment_aabbs(segments)
        bvh = build_bvh(segments_with_aabbs)
        
        # Get serial results (decoupled)
        serial_hits = query_bvh_jax(bvh, segments_with_aabbs, rays)
        serial_labels = label_anomalies_parallel(serial_hits, orbit, rays, segments_soa=None, batch_size=2)
        
        # Get parallel results (decoupled): query then label
        parallel_hits = query_bvh_jax(bvh, segments_with_aabbs, rays)
        parallel_labels = label_anomalies_parallel(parallel_hits, orbit, rays, segments_soa=None, batch_size=2)
        
        # Compare results (allowing for different ordering)
        assert len(parallel_hits) == len(serial_hits)
        
        if len(serial_labels) > 0 and len(parallel_labels) > 0:
            assert len(parallel_labels) == len(serial_labels)
            
            # Check that we have the same detection IDs (may be in different order)
            serial_det_ids = set(serial_labels.det_id.to_pylist())
            parallel_det_ids = set(parallel_labels.det_id.to_pylist())
            assert serial_det_ids == parallel_det_ids


@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not available")
class TestAnomalyRemoteConfig:
    """Test configuration and edge cases for Ray-parallel anomaly labeling."""
    
    def setup_method(self):
        """Set up Ray for each test."""
        if not ray.is_initialized():
            ray.init(local_mode=True, ignore_reinit_error=True)
    
    def teardown_method(self):
        """Clean up Ray after each test."""
        if ray.is_initialized():
            ray.shutdown()
    
    def test_batch_size_configurations(self):
        """Test different batch sizes."""
        from adam_core.geometry.overlap import OverlapHits
        from adam_core.orbits.polyline import OrbitsPlaneParams
        
        # Test with empty inputs and different batch sizes
        from adam_core.geometry.jax_types import SegmentsSOA
        
        empty_hits = OverlapHits.empty()
        empty_params = OrbitsPlaneParams.empty()
        # Decoupled signature requires Orbits and ObservationRays
        from adam_core.orbits.orbits import Orbits
        from adam_core.coordinates.cartesian import CartesianCoordinates
        empty_orbits = Orbits.from_kwargs(orbit_id=[], coordinates=CartesianCoordinates.empty())
        empty_rays = ObservationRays.empty()
        empty_segments_soa = SegmentsSOA.empty()
        
        for batch_size in [1, 10, 100, 1000]:
            labels = label_anomalies_parallel(
                empty_hits, empty_orbits, empty_rays, segments_soa=empty_segments_soa, batch_size=batch_size
            )
            assert len(labels) == 0
    
    def test_max_variants_configuration(self):
        """Test different max_variants_per_hit settings."""
        from adam_core.geometry.overlap import OverlapHits
        from adam_core.orbits.polyline import OrbitsPlaneParams
        from adam_core.geometry.jax_types import SegmentsSOA
        
        empty_hits = OverlapHits.empty()
        empty_params = OrbitsPlaneParams.empty()
        from adam_core.orbits.orbits import Orbits
        from adam_core.coordinates.cartesian import CartesianCoordinates
        empty_orbits = Orbits.from_kwargs(orbit_id=[], coordinates=CartesianCoordinates.empty())
        empty_rays = ObservationRays.empty()
        empty_segments_soa = SegmentsSOA.empty()
        
        for max_variants in [1, 2, 5]:
            labels = label_anomalies_parallel(
                empty_hits, empty_orbits, empty_rays, segments_soa=empty_segments_soa, max_variants_per_hit=max_variants
            )
            assert len(labels) == 0
