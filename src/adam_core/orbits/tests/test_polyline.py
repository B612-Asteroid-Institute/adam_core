"""
Tests for orbit polyline sampling and segment representation.
"""

import numpy as np
import pytest

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.orbits.orbits import Orbits
from adam_core.orbits.polyline import (
    OrbitsPlaneParams,
    OrbitPolylineSegments,
    compute_segment_aabbs,
    sample_ellipse_adaptive,
)
from adam_core.time import Timestamp


def create_test_orbits():
    """Create a small set of test orbits with different characteristics."""
    # Create test times
    times = Timestamp.from_mjd([59000.0, 59000.0, 59000.0], scale="tdb")
    
    # Create test Cartesian coordinates in SSB ecliptic frame
    # Orbit 1: Circular orbit at 1 AU
    coords1 = CartesianCoordinates.from_kwargs(
        x=[1.0], y=[0.0], z=[0.0],
        vx=[0.0], vy=[0.017202], vz=[0.0],  # ~1 AU/day for circular orbit
        time=times[0:1],
        origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]),
        frame="ecliptic",
    )
    
    # Orbit 2: Eccentric orbit (e=0.5)
    coords2 = CartesianCoordinates.from_kwargs(
        x=[1.5], y=[0.0], z=[0.0],
        vx=[0.0], vy=[0.014], vz=[0.0],
        time=times[1:2],
        origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]),
        frame="ecliptic",
    )
    
    # Orbit 3: Inclined orbit
    coords3 = CartesianCoordinates.from_kwargs(
        x=[1.0], y=[0.0], z=[0.5],
        vx=[0.0], vy=[0.015], vz=[0.008],
        time=times[2:3],
        origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]),
        frame="ecliptic",
    )
    
    # Combine coordinates
    all_coords = CartesianCoordinates.from_kwargs(
        x=[1.0, 1.5, 1.0],
        y=[0.0, 0.0, 0.0],
        z=[0.0, 0.0, 0.5],
        vx=[0.0, 0.0, 0.0],
        vy=[0.017202, 0.014, 0.015],
        vz=[0.0, 0.0, 0.008],
        time=times,
        origin=Origin.from_kwargs(code=[OriginCodes.SUN.name] * 3),
        frame="ecliptic",
    )
    
    orbits = Orbits.from_kwargs(
        orbit_id=["test_orbit_1", "test_orbit_2", "test_orbit_3"],
        coordinates=all_coords,
    )
    
    return orbits


class TestOrbitsPlaneParams:
    """Test OrbitsPlaneParams quivr table."""
    
    def test_empty_table(self):
        """Test creating empty table."""
        empty = OrbitsPlaneParams.empty()
        assert len(empty) == 0
        
    def test_table_creation(self):
        """Test creating table with data."""
        times = Timestamp.from_mjd([59000.0], scale="tdb")
        
        params = OrbitsPlaneParams.from_kwargs(
            orbit_id=["test"],
            t0=times,
            p_x=[1.0], p_y=[0.0], p_z=[0.0],
            q_x=[0.0], q_y=[1.0], q_z=[0.0],
            n_x=[0.0], n_y=[0.0], n_z=[1.0],
            r0_x=[0.0], r0_y=[0.0], r0_z=[0.0],
            a=[1.0], e=[0.0], M0=[0.0],
            frame="ecliptic",
            origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]),
        )
        
        assert len(params) == 1
        assert params.orbit_id[0].as_py() == "test"
        assert params.frame == "ecliptic"


class TestOrbitPolylineSegments:
    """Test OrbitPolylineSegments quivr table."""
    
    def test_empty_table(self):
        """Test creating empty table."""
        empty = OrbitPolylineSegments.empty()
        assert len(empty) == 0
        
    def test_table_creation(self):
        """Test creating table with data."""
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
        
        assert len(segments) == 1
        assert segments.orbit_id[0].as_py() == "test"
        assert segments.seg_id[0].as_py() == 0


class TestSampleEllipseAdaptive:
    """Test adaptive ellipse sampling."""
    
    def test_empty_orbits(self):
        """Test with empty orbit list."""
        empty_orbits = Orbits.empty()
        params, segments = sample_ellipse_adaptive(empty_orbits)
        
        assert len(params) == 0
        assert len(segments) == 0
        
    def test_basic_sampling(self):
        """Test basic sampling functionality."""
        orbits = create_test_orbits()
        params, segments = sample_ellipse_adaptive(orbits, max_chord_arcmin=1.0)
        
        # Should have parameters for each orbit
        assert len(params) == len(orbits)
        
        # Should have segments for each orbit
        assert len(segments) > 0
        
        # Check that all orbits are represented in segments
        orbit_ids_in_segments = set(segments.orbit_id.to_pylist())
        orbit_ids_in_params = set(params.orbit_id.to_pylist())
        assert orbit_ids_in_segments == orbit_ids_in_params
        
    def test_basis_orthonormality(self):
        """Test that orbital plane basis vectors are orthonormal."""
        orbits = create_test_orbits()
        params, _ = sample_ellipse_adaptive(orbits)
        
        tolerance = 1e-12
        
        for i in range(len(params)):
            # Extract basis vectors
            p = np.array([params.p_x[i].as_py(), params.p_y[i].as_py(), params.p_z[i].as_py()])
            q = np.array([params.q_x[i].as_py(), params.q_y[i].as_py(), params.q_z[i].as_py()])
            n = np.array([params.n_x[i].as_py(), params.n_y[i].as_py(), params.n_z[i].as_py()])
            
            # Check unit lengths
            assert abs(np.linalg.norm(p) - 1.0) < tolerance
            assert abs(np.linalg.norm(q) - 1.0) < tolerance
            assert abs(np.linalg.norm(n) - 1.0) < tolerance
            
            # Check orthogonality
            assert abs(np.dot(p, q)) < tolerance
            assert abs(np.dot(p, n)) < tolerance
            assert abs(np.dot(q, n)) < tolerance
            
    def test_chord_length_constraint(self):
        """Test that chord length constraint is respected."""
        orbits = create_test_orbits()
        max_chord_arcmin = 0.5
        params, segments = sample_ellipse_adaptive(orbits, max_chord_arcmin=max_chord_arcmin)
        
        # Convert to radians
        theta_max = max_chord_arcmin * np.pi / (180 * 60)
        
        # Check chord lengths for each segment
        for i in range(len(segments)):
            x0 = segments.x0[i].as_py()
            y0 = segments.y0[i].as_py()
            z0 = segments.z0[i].as_py()
            x1 = segments.x1[i].as_py()
            y1 = segments.y1[i].as_py()
            z1 = segments.z1[i].as_py()
            r_mid = segments.r_mid_au[i].as_py()
            
            # Compute chord length
            chord_length = np.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
            
            # Maximum allowed chord at this distance
            max_chord_au = theta_max * max(r_mid, 1.0)
            
            # Allow some tolerance for discretization
            assert chord_length <= max_chord_au * 1.1
            
    def test_segment_continuity(self):
        """Test that segments form continuous polylines."""
        orbits = create_test_orbits()
        params, segments = sample_ellipse_adaptive(orbits)
        
        # Group segments by orbit
        import pyarrow.compute as pc
        for orbit_id in params.orbit_id.to_pylist():
            mask = pc.equal(segments.orbit_id, orbit_id)
            orbit_segments = segments.apply_mask(mask).sort_by("seg_id")
            
            if len(orbit_segments) > 1:
                # Check that segments connect (allowing for wrap-around)
                for i in range(len(orbit_segments) - 1):
                    x0_next = orbit_segments.x0[i+1].as_py()
                    y0_next = orbit_segments.y0[i+1].as_py()
                    z0_next = orbit_segments.z0[i+1].as_py()
                    
                    x1_curr = orbit_segments.x1[i].as_py()
                    y1_curr = orbit_segments.y1[i].as_py()
                    z1_curr = orbit_segments.z1[i].as_py()
                    
                    # Should be the same point (within tolerance)
                    distance = np.sqrt(
                        (x0_next - x1_curr)**2 + 
                        (y0_next - y1_curr)**2 + 
                        (z0_next - z1_curr)**2
                    )
                    assert distance < 1e-10


class TestComputeSegmentAabbs:
    """Test AABB computation with guard band padding."""
    
    def test_empty_segments(self):
        """Test with empty segments."""
        empty_segments = OrbitPolylineSegments.empty()
        result = compute_segment_aabbs(empty_segments)
        assert len(result) == 0
        
    def test_aabb_computation(self):
        """Test basic AABB computation."""
        # Create test segments
        segments = OrbitPolylineSegments.from_kwargs(
            orbit_id=["test", "test"],
            seg_id=[0, 1],
            x0=[0.0, 1.0], y0=[0.0, 0.0], z0=[0.0, 0.0],
            x1=[1.0, 2.0], y1=[1.0, 1.0], z1=[0.0, 0.0],
            aabb_min_x=[np.nan, np.nan], aabb_min_y=[np.nan, np.nan], aabb_min_z=[np.nan, np.nan],
            aabb_max_x=[np.nan, np.nan], aabb_max_y=[np.nan, np.nan], aabb_max_z=[np.nan, np.nan],
            r_mid_au=[0.7, 1.5],  # Approximate midpoint distances
            n_x=[0.0, 0.0], n_y=[0.0, 0.0], n_z=[1.0, 1.0],
        )
        
        result = compute_segment_aabbs(segments, guard_arcmin=1.0)
        
        # Check that AABBs are no longer NaN
        assert not np.any(np.isnan(result.aabb_min_x.to_numpy()))
        assert not np.any(np.isnan(result.aabb_max_x.to_numpy()))
        
        # Check that AABBs contain segment endpoints
        for i in range(len(result)):
            x0, y0, z0 = result.x0[i].as_py(), result.y0[i].as_py(), result.z0[i].as_py()
            x1, y1, z1 = result.x1[i].as_py(), result.y1[i].as_py(), result.z1[i].as_py()
            
            min_x = result.aabb_min_x[i].as_py()
            max_x = result.aabb_max_x[i].as_py()
            min_y = result.aabb_min_y[i].as_py()
            max_y = result.aabb_max_y[i].as_py()
            min_z = result.aabb_min_z[i].as_py()
            max_z = result.aabb_max_z[i].as_py()
            
            # Both endpoints should be inside AABB
            assert min_x <= x0 <= max_x
            assert min_y <= y0 <= max_y
            assert min_z <= z0 <= max_z
            assert min_x <= x1 <= max_x
            assert min_y <= y1 <= max_y
            assert min_z <= z1 <= max_z
            
    def test_guard_band_padding(self):
        """Test that guard band padding is applied."""
        # Create a simple segment
        segments = OrbitPolylineSegments.from_kwargs(
            orbit_id=["test"],
            seg_id=[0],
            x0=[0.0], y0=[0.0], z0=[0.0],
            x1=[1.0], y1=[0.0], z1=[0.0],
            aabb_min_x=[np.nan], aabb_min_y=[np.nan], aabb_min_z=[np.nan],
            aabb_max_x=[np.nan], aabb_max_y=[np.nan], aabb_max_z=[np.nan],
            r_mid_au=[0.5],
            n_x=[0.0], n_y=[0.0], n_z=[1.0],
        )
        
        guard_arcmin = 2.0
        result = compute_segment_aabbs(segments, guard_arcmin=guard_arcmin)
        
        # Compute expected padding
        theta_guard = guard_arcmin * np.pi / (180 * 60)
        # Conservative padding uses max(r_mid, 1 AU)
        expected_pad = theta_guard * 1.0
        
        # Check that padding was applied
        min_x = result.aabb_min_x[0].as_py()
        max_x = result.aabb_max_x[0].as_py()
        
        # Unpadded AABB would be [0, 1] in x
        # With padding should be approximately [0-pad, 1+pad]
        assert min_x < 0.0
        assert max_x > 1.0
        assert abs(min_x - (-expected_pad)) < expected_pad * 0.1  # Allow some tolerance
        assert abs(max_x - (1.0 + expected_pad)) < expected_pad * 0.1


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_pipeline(self):
        """Test the complete M1 pipeline."""
        orbits = create_test_orbits()
        
        # Sample ellipses
        params, segments = sample_ellipse_adaptive(orbits, max_chord_arcmin=0.5)
        
        # Compute AABBs
        segments_with_aabbs = compute_segment_aabbs(segments, guard_arcmin=1.0)
        
        # Verify results
        assert len(params) == len(orbits)
        assert len(segments_with_aabbs) > 0
        assert not np.any(np.isnan(segments_with_aabbs.aabb_min_x.to_numpy()))
        
        # Check that each orbit has segments
        orbit_ids_params = set(params.orbit_id.to_pylist())
        orbit_ids_segments = set(segments_with_aabbs.orbit_id.to_pylist())
        assert orbit_ids_params == orbit_ids_segments
        
    def test_high_eccentricity_orbit(self):
        """Test with a high eccentricity orbit that should trigger refinement."""
        # Create a high-e orbit using Keplerian elements to ensure e ~ 0.9
        times = Timestamp.from_mjd([59000.0], scale="tdb")
        from adam_core.coordinates.keplerian import KeplerianCoordinates
        kep = KeplerianCoordinates.from_kwargs(
            a=[2.0],  # AU
            e=[0.9],
            i=[10.0],  # degrees
            raan=[30.0],
            ap=[45.0],
            M=[0.0],
            time=times,
            origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]),
            frame="ecliptic",
        )
        coords = kep.to_cartesian()
        orbits = Orbits.from_kwargs(orbit_id=["high_e_orbit"], coordinates=coords)
        
        # Sample with tight chord constraint
        params, segments = sample_ellipse_adaptive(orbits, max_chord_arcmin=0.1)
        
        # Should have many segments due to high curvature near perihelion
        assert len(segments) > 100  # Expect significant refinement
        
        # Verify parameters
        assert len(params) == 1
        assert params.e[0].as_py() > 0.8  # Should be high eccentricity
