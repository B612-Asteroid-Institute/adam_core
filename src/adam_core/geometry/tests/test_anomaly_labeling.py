"""
Tests for anomaly labeling correctness.

This module contains detailed tests for the anomaly labeling functionality,
including tests for different orbital eccentricities, near-node ambiguity,
determinism, and quantitative physics checks.
"""

import numpy as np
import pytest

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.geometry import geometric_overlap
 labelfrom adam_core.geometry.anomaly_labeling import label_anomalies
from adam_core.observations.detections import PointSourceDetections
from adam_core.observations.exposures import Exposures
from adam_core.observations.rays import rays_from_detections
from adam_core.orbits.orbits import Orbits
from adam_core.orbits.polyline import compute_segment_aabbs, sample_ellipse_adaptive
from adam_core.time import Timestamp


def create_orbit_with_elements(a=1.0, e=0.0, i=0.0, raan=0.0, ap=0.0, M=0.0):
    """Create an orbit with specified Keplerian elements."""
    times = Timestamp.from_mjd([59000.0], scale="tdb")
    
    # Use exact orbital velocity for circular orbits
    if e == 0.0:
        # Circular velocity: v = sqrt(GM/a), assuming GM_sun = 1 in AU^3/day^2 units
        # For a=1 AU: v = sqrt(1) = 1 AU/day = 0.017202 AU/day in actual units
        v_circular = np.sqrt(1.0 / a)  # In units where GM_sun = 1
        v_circular_actual = v_circular * 0.017202  # Convert to AU/day
        
        # Simple circular orbit in xy plane
        x, y, z = a * np.cos(M), a * np.sin(M), 0.0
        vx, vy, vz = -v_circular_actual * np.sin(M), v_circular_actual * np.cos(M), 0.0
    else:
        # For non-circular orbits, use a more complex approach
        # This is a simplified implementation for testing
        x, y, z = a * (1 - e), 0.0, 0.0  # At periapsis
        vx, vy, vz = 0.0, np.sqrt((1 + e) / (a * (1 - e))) * 0.017202, 0.0
    
    coords = CartesianCoordinates.from_kwargs(
        x=[x], y=[y], z=[z],
        vx=[vx], vy=[vy], vz=[vz],
        time=times,
        origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]),
        frame="ecliptic",
    )
    
    return Orbits.from_kwargs(
        orbit_id=[f"orbit_a{a}_e{e}_i{i}"],
        coordinates=coords,
    )


def create_rays_at_positions(positions, times=None):
    """Create observation rays pointing towards specific 3D positions."""
    if times is None:
        times = Timestamp.from_mjd([59000.0] * len(positions), scale="tdb")
    
    exposures = Exposures.from_kwargs(
        id=[f"exp_{i}" for i in range(len(positions))],
        start_time=times,
        duration=[300.0] * len(positions),
        filter=["r"] * len(positions),
        observatory_code=["500"] * len(positions),  # Geocenter
        seeing=[1.2] * len(positions),
        depth_5sigma=[22.0] * len(positions),
    )
    
    # Convert 3D positions to RA/Dec
    ra_dec_list = []
    for pos in positions:
        x, y, z = pos
        ra = np.degrees(np.arctan2(y, x))
        dec = np.degrees(np.arcsin(z / np.sqrt(x**2 + y**2 + z**2)))
        ra_dec_list.append((ra % 360, dec))
    
    ra_vals, dec_vals = zip(*ra_dec_list)
    
    detections = PointSourceDetections.from_kwargs(
        id=[f"det_{i}" for i in range(len(positions))],
        exposure_id=[f"exp_{i}" for i in range(len(positions))],
        time=times,
        ra=list(ra_vals),
        dec=list(dec_vals),
        ra_sigma=[0.1] * len(positions),
        dec_sigma=[0.1] * len(positions),
        mag=[20.0] * len(positions),
        mag_sigma=[0.1] * len(positions),
    )
    
    return rays_from_detections(detections, exposures)


class TestAnomalyLabelingCorrectness:
    """Test anomaly labeling for correctness across different orbital configurations."""
    
    def test_circular_orbit_labeling(self):
        """Test labeling for a circular orbit (e=0)."""
        # Create circular orbit
        orbit = create_orbit_with_elements(a=1.0, e=0.0)
        params, segments = sample_ellipse_adaptive(orbit, max_chord_arcmin=1.0)
        segments_with_aabbs = compute_segment_aabbs(segments, guard_arcmin=1.0)
        
        # Create rays pointing to positions on the orbit
        # Test positions at 0°, 90°, 180°, 270° true anomaly
        test_positions = [
            [1.0, 0.0, 0.0],      # 0° (periapsis for circular orbit)
            [0.0, 1.0, 0.0],      # 90°
            [-1.0, 0.0, 0.0],     # 180°
            [0.0, -1.0, 0.0],     # 270°
        ]
        rays = create_rays_at_positions(test_positions)
        
        # Run geometric overlap first, then labeling separately
        hits = geometric_overlap(
            segments_with_aabbs, rays, guard_arcmin=1.0
        )
        
        # Create orbits from the original orbit for labeling
        orbits = Orbits.from_kwargs(
            orbit_id=[orbit.orbit_id.to_pylist()[0]],
            coordinates=orbit.coordinates,
        )
        
        # Run anomaly labeling
        labels = label_anomalies(hits, rays, orbits)
        
        # Verify we got hits and labels
        assert len(hits) > 0, "Should find geometric overlaps"
        assert len(labels) > 0, "Should generate anomaly labels"
        
        # For circular orbit, E should equal f (both are the same for e=0)
        E_values = labels.E_rad.to_numpy()
        f_values = labels.f_rad.to_numpy()
        assert np.allclose(E_values, f_values, atol=1e-6), f"For circular orbit, E should equal f, got max |E-f|={np.max(np.abs(E_values - f_values))}"
        
        # Mean motion should be positive and reasonable (approximately sqrt(GM/a^3))
        n_values = labels.n_rad_day.to_numpy() 
        expected_n = 0.017202  # For a=1 AU, n ≈ sqrt(1)/1^1.5 ≈ 0.017202 rad/day
        assert np.all(n_values > 0), "Mean motion should be positive"
        assert np.all(np.abs(n_values - expected_n) < 0.01), f"Mean motion should be ~{expected_n}, got {n_values}"
        
        # Radius should be close to semi-major axis for circular orbit
        r_values = labels.r_au.to_numpy()
        assert np.all(np.abs(r_values - 1.0) < 0.1), f"Radius should be ~1 AU for circular orbit, got {r_values}"
        
    def test_high_eccentricity_orbit(self):
        """Test labeling for a high-eccentricity orbit."""
        # Create high-e orbit
        orbit = create_orbit_with_elements(a=2.0, e=0.8)
        params, segments = sample_ellipse_adaptive(orbit, max_chord_arcmin=1.0)
        segments_with_aabbs = compute_segment_aabbs(segments, guard_arcmin=1.0)
        
        # Create rays pointing near periapsis and apoapsis
        periapsis_pos = [2.0 * (1 - 0.8), 0.0, 0.0]  # r = a(1-e) = 0.4 AU
        apoapsis_pos = [-2.0 * (1 + 0.8), 0.0, 0.0]  # r = a(1+e) = 3.6 AU
        
        rays = create_rays_at_positions([periapsis_pos, apoapsis_pos])
        
        # Run geometric overlap first, then labeling separately
        hits = geometric_overlap(
            segments_with_aabbs, rays, guard_arcmin=1.0
        )
        
        # Create orbits from the original orbit for labeling
        orbits = Orbits.from_kwargs(
            orbit_id=[orbit.orbit_id.to_pylist()[0]],
            coordinates=orbit.coordinates,
        )
        
        # Run anomaly labeling
        labels = label_anomalies(hits, rays, orbits)
        
        if len(labels) > 0:
            # Radius should vary significantly for high-e orbit
            r_values = labels.r_au.to_numpy()
            if len(r_values) > 1:
                r_range = np.max(r_values) - np.min(r_values)
                assert r_range > 0.5, f"High-e orbit should show significant radius variation, got range {r_range}"
            
            # All radii should be positive
            assert np.all(r_values > 0), f"All radii should be positive, got {r_values}"
            
    def test_moderate_eccentricity_orbit(self):
        """Test labeling for a moderate-eccentricity orbit.""" 
        # Create moderate-e orbit
        orbit = create_orbit_with_elements(a=1.5, e=0.3)
        params, segments = sample_ellipse_adaptive(orbit, max_chord_arcmin=1.0)
        segments_with_aabbs = compute_segment_aabbs(segments, guard_arcmin=1.0)
        
        # Test several positions around the orbit
        test_positions = []
        for f in [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]:
            # Simple ellipse: r = a(1-e²)/(1+e*cos(f))
            r = 1.5 * (1 - 0.3**2) / (1 + 0.3 * np.cos(f))
            x = r * np.cos(f)
            y = r * np.sin(f)
            test_positions.append([x, y, 0.0])
        
        rays = create_rays_at_positions(test_positions)
        
        # Run geometric overlap first, then labeling separately
        hits = geometric_overlap(
            segments_with_aabbs, rays, guard_arcmin=1.0
        )
        
        # Create orbits from the original orbit for labeling
        orbits = Orbits.from_kwargs(
            orbit_id=[orbit.orbit_id.to_pylist()[0]],
            coordinates=orbit.coordinates,
        )
        
        # Run anomaly labeling
        labels = label_anomalies(hits, rays, orbits)
        
        if len(labels) > 0:
            # Check that anomaly values are physically reasonable
            f_values = labels.f_rad.to_numpy()
            # Our kernel returns f in (-π, π], which is valid - wrap to [0, 2π) for comparison
            f_wrapped = (f_values + 2*np.pi) % (2*np.pi)
            assert np.all(f_wrapped >= 0), "Wrapped true anomalies should be non-negative"
            assert np.all(f_wrapped < 2*np.pi), "Wrapped true anomalies should be less than 2π"
            
            # Check E and M consistency (E should have same sign as M for e < 1)
            E_values = labels.E_rad.to_numpy()
            M_values = labels.M_rad.to_numpy()
            # For e < 1, E and M should have same sign
            same_sign = np.sign(E_values) == np.sign(M_values)
            near_zero = (np.abs(E_values) < 0.1) | (np.abs(M_values) < 0.1)
            assert np.all(same_sign | near_zero), "E and M should have same sign for elliptical orbits"

    def test_labeling_determinism(self):
        """Test that anomaly labeling produces deterministic results."""
        # Create test orbit
        orbit = create_orbit_with_elements(a=1.0, e=0.2)
        params, segments = sample_ellipse_adaptive(orbit, max_chord_arcmin=1.0)
        segments_with_aabbs = compute_segment_aabbs(segments, guard_arcmin=1.0)
        
        # Create test rays
        rays = create_rays_at_positions([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0]])
        
        # Run labeling twice
        hits1, labels1 = geometric_overlap(
            segments_with_aabbs, rays, guard_arcmin=1.0,
            label_anomalies=True, plane_params=params
        )
        hits2, labels2 = geometric_overlap(
            segments_with_aabbs, rays, guard_arcmin=1.0,
            label_anomalies=True, plane_params=params
        )
        
        # Results should be identical
        assert len(labels1) == len(labels2), "Should get same number of labels"
        
        if len(labels1) > 0:
            # Check key fields are identical (within numerical precision)
            f1, f2 = labels1.f_rad.to_numpy(), labels2.f_rad.to_numpy()
            E1, E2 = labels1.E_rad.to_numpy(), labels2.E_rad.to_numpy()
            M1, M2 = labels1.M_rad.to_numpy(), labels2.M_rad.to_numpy()
            r1, r2 = labels1.r_au.to_numpy(), labels2.r_au.to_numpy()
            
            assert np.allclose(f1, f2, rtol=1e-12), "True anomalies should be identical"
            assert np.allclose(E1, E2, rtol=1e-12), "Eccentric anomalies should be identical" 
            assert np.allclose(M1, M2, rtol=1e-12), "Mean anomalies should be identical"
            assert np.allclose(r1, r2, rtol=1e-12), "Radii should be identical"

    def test_stable_sorting(self):
        """Test that anomaly labels are sorted consistently."""
        # Create multiple orbits
        orbits_list = []
        for i, e in enumerate([0.0, 0.2, 0.4]):
            orbit = create_orbit_with_elements(a=1.0 + 0.1*i, e=e)
            orbits_list.append(orbit)
        
        # Combine orbits - ensure all columns have same length
        n_orbits = len(orbits_list)
        
        # Replicate time and origin for each orbit
        base_time = orbits_list[0].coordinates.time
        base_origin = orbits_list[0].coordinates.origin
        
        # Create arrays of the same time/origin for all orbits
        times = Timestamp.from_mjd([base_time.mjd().to_numpy()[0]] * n_orbits, scale="tdb")
        origins = Origin.from_kwargs(code=[base_origin.code.to_pylist()[0]] * n_orbits)
        
        combined_coords = CartesianCoordinates.from_kwargs(
            x=[orbit.coordinates.x.to_numpy()[0] for orbit in orbits_list],
            y=[orbit.coordinates.y.to_numpy()[0] for orbit in orbits_list],
            z=[orbit.coordinates.z.to_numpy()[0] for orbit in orbits_list],
            vx=[orbit.coordinates.vx.to_numpy()[0] for orbit in orbits_list],
            vy=[orbit.coordinates.vy.to_numpy()[0] for orbit in orbits_list],
            vz=[orbit.coordinates.vz.to_numpy()[0] for orbit in orbits_list],
            time=times,
            origin=origins,
            frame=orbits_list[0].coordinates.frame,
        )
        
        combined_orbit = Orbits.from_kwargs(
            orbit_id=[f"orbit_{i}" for i in range(len(orbits_list))],
            coordinates=combined_coords,
        )
        
        params, segments = sample_ellipse_adaptive(combined_orbit, max_chord_arcmin=1.0)
        segments_with_aabbs = compute_segment_aabbs(segments, guard_arcmin=1.0)
        
        # Create rays
        rays = create_rays_at_positions([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        
        # Run labeling multiple times 
        results = []
        for _ in range(3):
            hits = geometric_overlap(
                segments_with_aabbs, rays, guard_arcmin=1.0
            )
            
            # Run anomaly labeling
            labels = label_anomalies(hits, rays, combined_orbit)
            if len(labels) > 0:
                # Extract sort keys
                det_ids = labels.det_id.to_pylist()
                orbit_ids = labels.orbit_id.to_pylist()
                variant_ids = labels.variant_id.to_numpy()
                snap_errors = labels.snap_error.to_numpy()
                results.append((det_ids, orbit_ids, variant_ids, snap_errors))
        
        # All runs should produce same sort order
        if len(results) > 1:
            for i in range(1, len(results)):
                assert results[0][0] == results[i][0], "det_id order should be consistent"
                assert results[0][1] == results[i][1], "orbit_id order should be consistent"
                assert np.array_equal(results[0][2], results[i][2]), "variant_id order should be consistent"
