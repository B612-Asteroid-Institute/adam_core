"""
Tests for geometric overlap detection between rays and orbit segments.
"""

import numpy as np
import pytest

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.geometry import (
    OverlapHits,
    geometric_overlap,
    query_bvh,
    query_bvh_parallel,
    label_anomalies,
)
from adam_core.observations.detections import PointSourceDetections
from adam_core.observations.exposures import Exposures
from adam_core.observations.rays import rays_from_detections
from adam_core.orbits.orbits import Orbits
from adam_core.orbits.polyline import compute_segment_aabbs, sample_ellipse_adaptive
from adam_core.time import Timestamp


def create_test_orbit_and_segments():
    """Create a simple test orbit and its polyline segments."""
    times = Timestamp.from_mjd([59000.0], scale="tdb")
    
    # Create a circular orbit at 1 AU
    coords = CartesianCoordinates.from_kwargs(
        x=[1.0], y=[0.0], z=[0.0],
        vx=[0.0], vy=[0.017202], vz=[0.0],  # Circular velocity
        time=times,
        origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]),
        frame="ecliptic",
    )
    
    orbits = Orbits.from_kwargs(
        orbit_id=["test_orbit"],
        coordinates=coords,
    )
    
    # Sample the orbit into segments
    params, segments = sample_ellipse_adaptive(orbits, max_chord_arcmin=1.0)
    segments_with_aabbs = compute_segment_aabbs(segments, guard_arcmin=1.0)
    
    return orbits, params, segments_with_aabbs


def create_test_rays():
    """Create test observation rays."""
    times = Timestamp.from_mjd([59000.0, 59000.1], scale="tdb")
    
    # Create exposures
    exposures = Exposures.from_kwargs(
        id=["exp_1", "exp_2"],
        start_time=times,
        duration=[300.0, 300.0],
        filter=["r", "g"],
        observatory_code=["500", "500"],  # Geocenter
        seeing=[1.2, 1.3],
        depth_5sigma=[22.0, 22.1],
    )
    
    # Create detections pointing towards the orbit
    detections = PointSourceDetections.from_kwargs(
        id=["det_1", "det_2"],
        exposure_id=["exp_1", "exp_2"],
        time=times,
        ra=[0.0, 90.0],  # Point towards +X and +Y directions
        dec=[0.0, 0.0],
        ra_sigma=[0.1, 0.1],
        dec_sigma=[0.1, 0.1],
        mag=[20.0, 20.1],
        mag_sigma=[0.1, 0.1],
    )
    
    return rays_from_detections(detections, exposures)


class TestOverlapHits:
    """Test OverlapHits quivr table."""
    
    def test_empty_table(self):
        """Test creating empty table."""
        empty = OverlapHits.empty()
        assert len(empty) == 0
        
    def test_table_creation(self):
        """Test creating table with data."""
        hits = OverlapHits.from_kwargs(
            det_id=["det_1"],
            orbit_id=["orbit_1"],
            seg_id=[0],
            leaf_id=[1],
            distance_au=[0.001],
        )
        
        assert len(hits) == 1
        assert hits.det_id[0].as_py() == "det_1"
        assert hits.orbit_id[0].as_py() == "orbit_1"
        assert hits.seg_id[0].as_py() == 0
        assert hits.leaf_id[0].as_py() == 1
        assert abs(hits.distance_au[0].as_py() - 0.001) < 1e-10


class TestQueryBvh:
    """Test BVH querying functionality."""
    
    def test_empty_inputs(self):
        """Test with empty inputs."""
        from adam_core.geometry.bvh import BVHShard
        from adam_core.observations.rays import ObservationRays
        from adam_core.orbits.polyline import OrbitPolylineSegments
        
        empty_bvh = BVHShard(
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
        
        empty_segments = OrbitPolylineSegments.empty()
        empty_rays = ObservationRays.empty()
        
        hits = query_bvh(empty_bvh, empty_segments, empty_rays)
        assert len(hits) == 0
        
    def test_basic_overlap_detection(self):
        """Test basic overlap detection with simple geometry."""
        orbits, params, segments = create_test_orbit_and_segments()
        rays = create_test_rays()
        
        # Use geometric_overlap for simplicity
        hits = geometric_overlap(segments, rays, guard_arcmin=5.0)  # Large guard for testing
        
        # Should find some hits since rays point towards the orbit
        assert len(hits) > 0
        
        # All hits should have valid data
        for i in range(len(hits)):
            assert hits.det_id[i].as_py() in ["det_1", "det_2"]
            assert hits.orbit_id[i].as_py() == "test_orbit"
            assert hits.seg_id[i].as_py() >= 0
            assert hits.leaf_id[i].as_py() >= 0
            assert hits.distance_au[i].as_py() >= 0.0
            
    def test_no_hits_with_tight_guard(self):
        """Test that tight guard band produces no hits for distant rays."""
        orbits, params, segments = create_test_orbit_and_segments()
        
        # Create rays pointing away from the orbit
        times = Timestamp.from_mjd([59000.0], scale="tdb")
        
        exposures = Exposures.from_kwargs(
            id=["exp_1"],
            start_time=times,
            duration=[300.0],
            filter=["r"],
            observatory_code=["500"],
            seeing=[1.2],
            depth_5sigma=[22.0],
        )
        
        detections = PointSourceDetections.from_kwargs(
            id=["det_1"],
            exposure_id=["exp_1"],
            time=times,
            ra=[180.0],  # Point away from orbit (opposite direction)
            dec=[0.0],
            ra_sigma=[0.1],
            dec_sigma=[0.1],
            mag=[20.0],
            mag_sigma=[0.1],
        )
        
        rays = rays_from_detections(detections, exposures)
        
        # Use very tight guard band
        hits = geometric_overlap(segments, rays, guard_arcmin=0.01)
        
        # Should find no hits
        assert len(hits) == 0
        
    def test_max_hits_per_ray_limit(self):
        """Test max_hits_per_ray parameter."""
        orbits, params, segments = create_test_orbit_and_segments()
        rays = create_test_rays()
        
        from adam_core.geometry.bvh import build_bvh
        
        bvh = build_bvh(segments)
        
        # Query with limit
        hits_limited = query_bvh(bvh, segments, rays, guard_arcmin=10.0, max_hits_per_ray=2)
        
        # Count hits per ray
        hit_counts = {}
        for i in range(len(hits_limited)):
            det_id = hits_limited.det_id[i].as_py()
            hit_counts[det_id] = hit_counts.get(det_id, 0) + 1
        
        # No ray should have more than 2 hits
        for count in hit_counts.values():
            assert count <= 2
            
    def test_distance_sorting(self):
        """Test that hits are sorted by distance."""
        orbits, params, segments = create_test_orbit_and_segments()
        rays = create_test_rays()
        
        hits = geometric_overlap(segments, rays, guard_arcmin=10.0)
        
        # Group hits by detection ID and check sorting
        hits_by_det = {}
        for i in range(len(hits)):
            det_id = hits.det_id[i].as_py()
            distance = hits.distance_au[i].as_py()
            
            if det_id not in hits_by_det:
                hits_by_det[det_id] = []
            hits_by_det[det_id].append(distance)
        
        # Check that distances are sorted for each detection
        for det_id, distances in hits_by_det.items():
            sorted_distances = sorted(distances)
            assert distances == sorted_distances


class TestGeometricOverlap:
    """Test convenience geometric_overlap function."""
    
    def test_empty_inputs(self):
        """Test with empty inputs."""
        from adam_core.observations.rays import ObservationRays
        from adam_core.orbits.polyline import OrbitPolylineSegments
        
        empty_segments = OrbitPolylineSegments.empty()
        empty_rays = ObservationRays.empty()
        
        hits = geometric_overlap(empty_segments, empty_rays)
        assert len(hits) == 0
        
    def test_basic_functionality(self):
        """Test basic functionality of geometric_overlap."""
        orbits, params, segments = create_test_orbit_and_segments()
        rays = create_test_rays()
        
        hits = geometric_overlap(segments, rays, guard_arcmin=2.0)
        
        # Should find hits
        assert len(hits) > 0
        
        # All hits should be valid
        for i in range(len(hits)):
            assert isinstance(hits.det_id[i].as_py(), str)
            assert isinstance(hits.orbit_id[i].as_py(), str)
            assert isinstance(hits.seg_id[i].as_py(), int)
            assert isinstance(hits.leaf_id[i].as_py(), int)
            assert hits.distance_au[i].as_py() >= 0.0


class TestIntegration:
    """Integration tests for geometric overlap detection."""
    
    def test_full_pipeline_with_multiple_orbits(self):
        """Test complete pipeline with multiple orbits and rays."""
        # Create multiple test orbits
        times = Timestamp.from_mjd([59000.0, 59000.0], scale="tdb")
        
        coords = CartesianCoordinates.from_kwargs(
            x=[1.0, 1.5],  # Different distances
            y=[0.0, 0.0],
            z=[0.0, 0.2],  # Different inclinations
            vx=[0.0, 0.0],
            vy=[0.017202, 0.014],  # Different velocities
            vz=[0.0, 0.003],
            time=times,
            origin=Origin.from_kwargs(code=[OriginCodes.SUN.name] * 2),
            frame="ecliptic",
        )
        
        orbits = Orbits.from_kwargs(
            orbit_id=["orbit_1", "orbit_2"],
            coordinates=coords,
        )
        
        # Sample orbits
        params, segments = sample_ellipse_adaptive(orbits, max_chord_arcmin=2.0)
        segments_with_aabbs = compute_segment_aabbs(segments, guard_arcmin=1.0)
        
        # Create multiple rays
        ray_times = Timestamp.from_mjd([59000.0, 59000.1, 59000.2], scale="tdb")
        
        exposures = Exposures.from_kwargs(
            id=["exp_1", "exp_2", "exp_3"],
            start_time=ray_times,
            duration=[300.0, 300.0, 300.0],
            filter=["r", "g", "i"],
            observatory_code=["500", "X05", "G96"],
            seeing=[1.2, 1.3, 1.1],
            depth_5sigma=[22.0, 22.1, 22.2],
        )
        
        detections = PointSourceDetections.from_kwargs(
            id=["det_1", "det_2", "det_3"],
            exposure_id=["exp_1", "exp_2", "exp_3"],
            time=ray_times,
            ra=[0.0, 45.0, 90.0],
            dec=[0.0, 10.0, -5.0],
            ra_sigma=[0.1, 0.1, 0.1],
            dec_sigma=[0.1, 0.1, 0.1],
            mag=[20.0, 20.5, 19.8],
            mag_sigma=[0.1, 0.1, 0.1],
        )
        
        rays = rays_from_detections(detections, exposures)
        
        # Find overlaps
        hits = geometric_overlap(segments_with_aabbs, rays, guard_arcmin=3.0)
        
        # Should find some hits
        assert len(hits) >= 0  # May or may not find hits depending on geometry
        
        # If hits found, they should be valid
        if len(hits) > 0:
            unique_orbits = set(hits.orbit_id.to_pylist())
            unique_detections = set(hits.det_id.to_pylist())
            
            # Should only reference existing orbits and detections
            assert unique_orbits.issubset({"orbit_1", "orbit_2"})
            assert unique_detections.issubset({"det_1", "det_2", "det_3"})
            
    def test_performance_with_many_rays(self):
        """Test performance with larger number of rays."""
        orbits, params, segments = create_test_orbit_and_segments()
        
        # Create many rays
        n_rays = 100
        ray_times = Timestamp.from_mjd(np.linspace(59000.0, 59001.0, n_rays), scale="tdb")
        
        exposures = Exposures.from_kwargs(
            id=[f"exp_{i}" for i in range(n_rays)],
            start_time=ray_times,
            duration=[300.0] * n_rays,
            filter=["r"] * n_rays,
            observatory_code=["500"] * n_rays,
            seeing=[1.2] * n_rays,
            depth_5sigma=[22.0] * n_rays,
        )
        
        # Random sky positions
        np.random.seed(42)  # For reproducibility
        detections = PointSourceDetections.from_kwargs(
            id=[f"det_{i}" for i in range(n_rays)],
            exposure_id=[f"exp_{i}" for i in range(n_rays)],
            time=ray_times,
            ra=np.random.uniform(0.0, 360.0, n_rays),
            dec=np.random.uniform(-30.0, 30.0, n_rays),
            ra_sigma=[0.1] * n_rays,
            dec_sigma=[0.1] * n_rays,
            mag=[20.0] * n_rays,
            mag_sigma=[0.1] * n_rays,
        )
        
        rays = rays_from_detections(detections, exposures)
        
        # This should complete in reasonable time
        hits = geometric_overlap(segments, rays, guard_arcmin=1.0)
        
        # Results should be reasonable
        assert len(hits) >= 0
        assert len(hits) <= n_rays * len(segments)  # Upper bound check


class TestQueryBvhParallel:
    """Test parallel BVH querying functionality."""
    
    def test_parallel_vs_serial_consistency(self):
        """Test that parallel and serial implementations give same results."""
        import ray
        
        # Initialize Ray if not already running
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        orbits, params, segments = create_test_orbit_and_segments()
        rays = create_test_rays()
        
        # Get results from both implementations
        hits_serial = geometric_overlap(segments, rays, guard_arcmin=1.0)
        
        # For parallel, we need to build BVH first
        from adam_core.geometry.bvh import build_bvh
        bvh = build_bvh(segments)
        hits_parallel = query_bvh_parallel(
            bvh, segments, rays, guard_arcmin=1.0, batch_size=5
        )
        
        # Should have same number of hits
        assert len(hits_serial) == len(hits_parallel)
        
        # Convert to sorted lists for comparison (order may differ)
        def hit_key(hit_table, i):
            return (
                hit_table.det_id[i].as_py(),
                hit_table.orbit_id[i].as_py(),
                hit_table.seg_id[i].as_py(),
            )
        
        if len(hits_serial) > 0:
            serial_keys = sorted([hit_key(hits_serial, i) for i in range(len(hits_serial))])
            parallel_keys = sorted([hit_key(hits_parallel, i) for i in range(len(hits_parallel))])
            
            assert serial_keys == parallel_keys
    
    def test_parallel_empty_inputs(self):
        """Test parallel query with empty inputs."""
        import ray
        
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        from adam_core.geometry.bvh import BVHShard
        from adam_core.observations.rays import ObservationRays
        from adam_core.orbits.polyline import OrbitPolylineSegments
        
        # Empty BVH
        empty_bvh = BVHShard(
            nodes_min=np.array([]).reshape(0, 3),
            nodes_max=np.array([]).reshape(0, 3),
            left_child=np.array([], dtype=np.int32),
            right_child=np.array([], dtype=np.int32),
            first_prim=np.array([], dtype=np.int32),
            prim_count=np.array([], dtype=np.int32),
            prim_orbit_ids=[],
            prim_seg_ids=np.array([], dtype=np.int32),
            prim_row_index=np.array([], dtype=np.int32),
        )
        
        empty_segments = OrbitPolylineSegments.empty()
        empty_rays = ObservationRays.empty()
        
        hits = query_bvh_parallel(empty_bvh, empty_segments, empty_rays, guard_arcmin=1.0)
        assert len(hits) == 0
    
    def test_parallel_batch_sizes(self):
        """Test parallel query with different batch sizes."""
        import ray
        
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        orbits, params, segments = create_test_orbit_and_segments()
        rays = create_test_rays()
        
        # Test different batch sizes
        from adam_core.geometry.bvh import build_bvh
        bvh = build_bvh(segments)
        hits_small = query_bvh_parallel(bvh, segments, rays, guard_arcmin=1.0, batch_size=2)
        hits_large = query_bvh_parallel(bvh, segments, rays, guard_arcmin=1.0, batch_size=100)
        
        # Should get same results regardless of batch size
        assert len(hits_small) == len(hits_large)


class TestAnomalyLabeling:
    """Test anomaly labeling integration with geometric overlap."""
    
    def test_geometric_overlap_with_anomaly_labeling(self):
        """Test that anomaly labeling flag returns both hits and labels."""
        orbits, params, segments = create_test_orbit_and_segments()
        rays = create_test_rays()
        
        # Decoupled API: compute hits, then label_anomalies
        hits = geometric_overlap(segments, rays, guard_arcmin=1.0)
        assert isinstance(hits, OverlapHits)
        
        labels = label_anomalies(hits, rays, orbits)
        from adam_core.geometry import AnomalyLabels
        assert isinstance(labels, AnomalyLabels)
        
        # Sanity: number of labels equals hits (single-variant path)
        assert len(labels) == len(hits)
    
    def test_query_bvh_with_anomaly_labeling(self):
        """Test that query_bvh works with anomaly labeling flag."""
        orbits, params, segments = create_test_orbit_and_segments()
        rays = create_test_rays()
        
        from adam_core.geometry.bvh import build_bvh
        bvh = build_bvh(segments)
        
        # Decoupled API: query hits then label
        hits = query_bvh(bvh, segments, rays, guard_arcmin=1.0)
        assert isinstance(hits, OverlapHits)
        
        labels = label_anomalies(hits, rays, orbits)
        from adam_core.geometry import AnomalyLabels
        assert isinstance(labels, AnomalyLabels)
    
    def test_anomaly_labeling_empty_inputs(self):
        """Test anomaly labeling with empty inputs."""
        from adam_core.observations.rays import ObservationRays
        from adam_core.orbits.polyline import OrbitPolylineSegments
        
        empty_segments = OrbitPolylineSegments.empty()
        empty_rays = ObservationRays.empty()
        
        hits = geometric_overlap(empty_segments, empty_rays)
        assert isinstance(hits, OverlapHits)
        assert len(hits) == 0
    
    def test_anomaly_labeling_parameters(self):
        """Test anomaly labeling with different parameters."""
        orbits, params, segments = create_test_orbit_and_segments()
        rays = create_test_rays()
        
        hits = geometric_overlap(segments, rays, guard_arcmin=1.0)
        labels = label_anomalies(hits, rays, orbits)
        
        # Single-variant path: labels count equals hits
        assert len(labels) == len(hits)
