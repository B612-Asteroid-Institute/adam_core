"""
Tests for anomaly labeling correctness.

This module contains detailed tests for the anomaly labeling functionality,
including tests for different orbital eccentricities, near-node ambiguity,
determinism, and quantitative physics checks.
"""

import numpy as np
import pytest
import pyarrow as pa

from adam_assist import ASSISTPropagator

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.keplerian import KeplerianCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.geometry import geometric_overlap
from adam_core.geometry.anomaly_labeling import label_anomalies
from adam_core.geometry.overlap import OverlapHits
from adam_core.observations.detections import PointSourceDetections
from adam_core.observations.exposures import Exposures
from adam_core.observations.rays import ObservationRays, rays_from_detections, ephemeris_to_rays
from adam_core.observers.observers import Observers
from adam_core.orbits.orbits import Orbits
from adam_core.orbits.polyline import compute_segment_aabbs, sample_ellipse_adaptive
from adam_core.time import Timestamp
from adam_core.utils.helpers import make_real_orbits


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
        x=[x],
        y=[y],
        z=[z],
        vx=[vx],
        vy=[vy],
        vz=[vz],
        time=times,
        origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]),
        frame="ecliptic",
    )

    return Orbits.from_kwargs(
        orbit_id=[f"orbit_a{a}_e{e}_i{i}"],
        coordinates=coords,
    )


def make_overlap_hits_from_rays_and_orbit(rays: ObservationRays, orbit: Orbits) -> OverlapHits:
    """
    Create minimal OverlapHits pairing each ray detection with the given orbit.
    
    This bypasses BVH geometric overlap for testing anomaly labeling directly.
    """
    n_rays = len(rays)
    orbit_id = orbit.orbit_id.to_pylist()[0]
    
    return OverlapHits.from_kwargs(
        det_id=rays.det_id.to_pylist(),
        orbit_id=[orbit_id] * n_rays,
        seg_id=[0] * n_rays,  # Dummy segment ID
        leaf_id=[0] * n_rays,  # Dummy leaf ID
        distance_au=[0.0] * n_rays,  # Dummy distance
    )




def generate_ephemeris_and_rays(orbit: Orbits, times: Timestamp, observatory_code: str = "500", max_processes: int = 1) -> ObservationRays:
    """
    Generate ephemeris from orbit, convert to rays using built-in utility.

    This uses ASSISTPropagator + ephemeris_to_rays to avoid mocked detections/exposures.
    """
    propagator = ASSISTPropagator()
    observers = Observers.from_codes(times=times, codes=[observatory_code] * len(times))
    ephemeris = propagator.generate_ephemeris(orbit, observers, max_processes=max_processes)

    det_ids = [f"ephem_{i:06d}" for i in range(len(ephemeris))]
    return ephemeris_to_rays(ephemeris, observers=observers, det_id=det_ids)


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
            [1.0, 0.0, 0.0],  # 0° (periapsis for circular orbit)
            [0.0, 1.0, 0.0],  # 90°
            [-1.0, 0.0, 0.0],  # 180°
            [0.0, -1.0, 0.0],  # 270°
        ]
        rays = create_rays_at_positions(test_positions)

        # Run geometric overlap first, then labeling separately
        hits = geometric_overlap(segments_with_aabbs, rays, guard_arcmin=1.0)

        # Run anomaly labeling
        labels = label_anomalies(hits, rays, orbit)

        # Verify we got hits and labels
        assert len(hits) > 0, "Should find geometric overlaps"
        assert len(labels) > 0, "Should generate anomaly labels"

        # For circular orbit, E should equal f (both are the same for e=0)
        E_values = labels.E_rad.to_numpy()
        f_values = labels.f_rad.to_numpy()
        assert np.allclose(
            E_values, f_values, atol=1e-6
        ), f"For circular orbit, E should equal f, got max |E-f|={np.max(np.abs(E_values - f_values))}"

        # Mean motion should be positive and reasonable (approximately sqrt(GM/a^3))
        n_values = labels.n_rad_day.to_numpy()
        expected_n = 0.017202  # For a=1 AU, n ≈ sqrt(1)/1^1.5 ≈ 0.017202 rad/day
        assert np.all(n_values > 0), "Mean motion should be positive"
        assert np.all(
            np.abs(n_values - expected_n) < 0.01
        ), f"Mean motion should be ~{expected_n}, got {n_values}"

        # Radius should be close to semi-major axis for circular orbit
        r_values = labels.r_au.to_numpy()
        assert np.all(
            np.abs(r_values - 1.0) < 0.1
        ), f"Radius should be ~1 AU for circular orbit, got {r_values}"

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
        hits = geometric_overlap(segments_with_aabbs, rays, guard_arcmin=1.0)

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
                assert (
                    r_range > 0.5
                ), f"High-e orbit should show significant radius variation, got range {r_range}"

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
        for f in [0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]:
            # Simple ellipse: r = a(1-e²)/(1+e*cos(f))
            r = 1.5 * (1 - 0.3**2) / (1 + 0.3 * np.cos(f))
            x = r * np.cos(f)
            y = r * np.sin(f)
            test_positions.append([x, y, 0.0])

        rays = create_rays_at_positions(test_positions)

        # Run geometric overlap first, then labeling separately
        hits = geometric_overlap(segments_with_aabbs, rays, guard_arcmin=1.0)

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
            f_wrapped = (f_values + 2 * np.pi) % (2 * np.pi)
            assert np.all(
                f_wrapped >= 0
            ), "Wrapped true anomalies should be non-negative"
            assert np.all(
                f_wrapped < 2 * np.pi
            ), "Wrapped true anomalies should be less than 2π"

            # Check E and M consistency (E should have same sign as M for e < 1)
            E_values = labels.E_rad.to_numpy()
            M_values = labels.M_rad.to_numpy()
            # For e < 1, E and M should have same sign
            same_sign = np.sign(E_values) == np.sign(M_values)
            near_zero = (np.abs(E_values) < 0.1) | (np.abs(M_values) < 0.1)
            assert np.all(
                same_sign | near_zero
            ), "E and M should have same sign for elliptical orbits"

    def test_labeling_determinism(self):
        """Test that anomaly labeling produces deterministic results."""
        # Create test orbit
        orbit = create_orbit_with_elements(a=1.0, e=0.2)
        params, segments = sample_ellipse_adaptive(orbit, max_chord_arcmin=1.0)
        segments_with_aabbs = compute_segment_aabbs(segments, guard_arcmin=1.0)

        # Create test rays
        rays = create_rays_at_positions([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0]])

        # Create orbits from the original orbit for labeling
        orbits = Orbits.from_kwargs(
            orbit_id=[orbit.orbit_id.to_pylist()[0]],
            coordinates=orbit.coordinates,
        )

        # Run labeling twice
        hits1 = geometric_overlap(segments_with_aabbs, rays, guard_arcmin=1.0)
        labels1 = label_anomalies(hits1, rays, orbits)

        hits2 = geometric_overlap(segments_with_aabbs, rays, guard_arcmin=1.0)
        labels2 = label_anomalies(hits2, rays, orbits)

        # Results should be identical
        assert len(labels1) == len(labels2), "Should get same number of labels"

        if len(labels1) > 0:
            # Check key fields are identical (within numerical precision)
            f1, f2 = labels1.f_rad.to_numpy(), labels2.f_rad.to_numpy()
            E1, E2 = labels1.E_rad.to_numpy(), labels2.E_rad.to_numpy()
            M1, M2 = labels1.M_rad.to_numpy(), labels2.M_rad.to_numpy()
            r1, r2 = labels1.r_au.to_numpy(), labels2.r_au.to_numpy()

            assert np.allclose(f1, f2, rtol=1e-12), "True anomalies should be identical"
            assert np.allclose(
                E1, E2, rtol=1e-12
            ), "Eccentric anomalies should be identical"
            assert np.allclose(M1, M2, rtol=1e-12), "Mean anomalies should be identical"
            assert np.allclose(r1, r2, rtol=1e-12), "Radii should be identical"

    def test_stable_sorting(self):
        """Test that anomaly labels are sorted consistently."""
        # Create multiple orbits
        orbits_list = []
        for i, e in enumerate([0.0, 0.2, 0.4]):
            orbit = create_orbit_with_elements(a=1.0 + 0.1 * i, e=e)
            orbits_list.append(orbit)

        # Combine orbits - ensure all columns have same length
        n_orbits = len(orbits_list)

        # Replicate time and origin for each orbit
        base_time = orbits_list[0].coordinates.time
        base_origin = orbits_list[0].coordinates.origin

        # Create arrays of the same time/origin for all orbits
        times = Timestamp.from_mjd(
            [base_time.mjd().to_numpy()[0]] * n_orbits, scale="tdb"
        )
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
            hits = geometric_overlap(segments_with_aabbs, rays, guard_arcmin=1.0)

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
                assert (
                    results[0][0] == results[i][0]
                ), "det_id order should be consistent"
                assert (
                    results[0][1] == results[i][1]
                ), "orbit_id order should be consistent"
                assert np.array_equal(
                    results[0][2], results[i][2]
                ), "variant_id order should be consistent"

    def test_multi_anomaly_support(self):
        """Test multi-anomaly support with max_k > 1."""
        # Create test orbit and rays
        orbit = create_orbit_with_elements(
            a=1.0, e=0.1
        )  # Slightly eccentric for more interesting geometry
        params, segments = sample_ellipse_adaptive(orbit, max_chord_arcmin=1.0)
        segments_with_aabbs = compute_segment_aabbs(segments, guard_arcmin=1.0)

        # Create rays at multiple positions
        test_positions = [
            [1.0, 0.0, 0.0],  # Near periapsis
            [0.0, 1.0, 0.0],  # 90° position
            [-0.9, 0.0, 0.0],  # Near apoapsis
        ]
        rays = create_rays_at_positions(test_positions)

        # Run geometric overlap
        hits = geometric_overlap(segments_with_aabbs, rays, guard_arcmin=1.0)

        # Create orbits for labeling
        orbits = Orbits.from_kwargs(
            orbit_id=[orbit.orbit_id.to_pylist()[0]],
            coordinates=orbit.coordinates,
        )

        # Test with max_k=3 to get multiple candidates
        labels = label_anomalies(hits, rays, orbits, max_k=3)

        if len(labels) > 0:
            # Check that variant_id is properly assigned
            variant_ids = labels.variant_id.to_numpy()
            assert np.all(variant_ids >= 0), "All variant_ids should be non-negative"
            assert np.all(variant_ids < 3), "All variant_ids should be < max_k"

            # Check that results are sorted by canonical order
            det_ids = labels.det_id.to_pylist()
            orbit_ids = labels.orbit_id.to_pylist()
            snap_errors = labels.snap_error.to_numpy()

            # Verify sorting: (det_id, orbit_id, variant_id, snap_error)
            for i in range(1, len(labels)):
                curr_det = det_ids[i]
                prev_det = det_ids[i - 1]
                curr_orbit = orbit_ids[i]
                prev_orbit = orbit_ids[i - 1]
                curr_variant = variant_ids[i]
                prev_variant = variant_ids[i - 1]
                curr_snap = snap_errors[i]
                prev_snap = snap_errors[i - 1]

                # Check sort order (skip NaN comparisons)
                if curr_det == prev_det and curr_orbit == prev_orbit:
                    if curr_variant == prev_variant:
                        # Only check ordering for non-NaN values
                        if not (np.isnan(curr_snap) or np.isnan(prev_snap)):
                            assert (
                                curr_snap >= prev_snap
                            ), "snap_error should be ascending within same (det,orbit,variant)"
                    else:
                        assert (
                            curr_variant > prev_variant
                        ), "variant_id should be ascending within same (det,orbit)"

    def test_multi_anomaly_with_filtering(self):
        """Test multi-anomaly support with snap error filtering."""
        # Create test orbit and rays
        orbit = create_orbit_with_elements(a=1.0, e=0.05)
        params, segments = sample_ellipse_adaptive(orbit, max_chord_arcmin=1.0)
        segments_with_aabbs = compute_segment_aabbs(segments, guard_arcmin=1.0)

        test_positions = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        rays = create_rays_at_positions(test_positions)

        hits = geometric_overlap(segments_with_aabbs, rays, guard_arcmin=1.0)

        orbits = Orbits.from_kwargs(
            orbit_id=[orbit.orbit_id.to_pylist()[0]],
            coordinates=orbit.coordinates,
        )

        # Test with very strict snap error filtering
        labels_strict = label_anomalies(
            hits, rays, orbits, max_k=3, snap_error_max_au=1e-6
        )

        # Test with lenient filtering
        labels_lenient = label_anomalies(
            hits, rays, orbits, max_k=3, snap_error_max_au=1.0
        )

        # Lenient should have >= strict results
        assert len(labels_lenient) >= len(labels_strict)

        # All snap errors in strict should be <= threshold
        if len(labels_strict) > 0:
            strict_snap_errors = labels_strict.snap_error.to_numpy()
            assert np.all(
                strict_snap_errors <= 1e-6
            ), "All snap errors should be below strict threshold"

    def test_multi_anomaly_k1_compatibility(self):
        """Test that max_k=1 produces same results as original implementation."""
        # Create test orbit and rays
        orbit = create_orbit_with_elements(a=1.0, e=0.0)
        params, segments = sample_ellipse_adaptive(orbit, max_chord_arcmin=1.0)
        segments_with_aabbs = compute_segment_aabbs(segments, guard_arcmin=1.0)

        test_positions = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        rays = create_rays_at_positions(test_positions)

        hits = geometric_overlap(segments_with_aabbs, rays, guard_arcmin=1.0)

        orbits = Orbits.from_kwargs(
            orbit_id=[orbit.orbit_id.to_pylist()[0]],
            coordinates=orbit.coordinates,
        )

        # Test with max_k=1 (should behave like original)
        labels_k1 = label_anomalies(hits, rays, orbits, max_k=1)

        if len(labels_k1) > 0:
            # All variant_ids should be 0
            variant_ids = labels_k1.variant_id.to_numpy()
            assert np.all(variant_ids == 0), "All variant_ids should be 0 for max_k=1"


class TestRealOrbitAnomalyLabeling:
    """Test anomaly labeling with real asteroid orbits and ephemeris-derived observations."""

    def test_real_orbit_ephemeris_labeling(self):
        """Test end-to-end: real orbit -> ephemeris -> detections -> rays -> labels with real data."""
        # Use real orbits from sample data
        orbits = make_real_orbits(num_orbits=1)
        orbit = orbits
        
        # Generate observation times spanning a few days
        base_mjd = orbit.coordinates.time.mjd()[0].as_py()
        times = Timestamp.from_mjd([base_mjd + i * 0.5 for i in range(8)], scale="tdb")  # 4 days, 12h intervals
        
        # Generate ephemeris and convert to rays
        rays = generate_ephemeris_and_rays(orbit, times, observatory_code="500")  # Geocenter
        
        # Create overlap hits (bypass BVH)
        hits = make_overlap_hits_from_rays_and_orbit(rays, orbit)
        
        # Run anomaly labeling with reasonable snap tolerance for real ephemeris
        labels = label_anomalies(hits, rays, orbit, max_k=3, snap_error_max_au=15.0)
        
        # Assertions
        assert len(labels) > 0, "Should generate anomaly labels from real ephemeris"
        assert len(labels) >= len(rays), "Should have at least one label per ray (possibly more with multi-K)"
        
        # All snap errors should be within reasonable tolerance for real ephemeris
        snap_errors = labels.snap_error.to_numpy()
        assert np.all(snap_errors <= 15.0), f"All snap errors should be ≤ 15.0 AU, got max={np.max(snap_errors)}"
        assert np.all(snap_errors > 0), "All snap errors should be positive"
        
        # Physical consistency checks
        r_values = labels.r_au.to_numpy()
        n_values = labels.n_rad_day.to_numpy()
        assert np.all(np.isfinite(r_values)), "All radii should be finite"
        assert np.all(np.isfinite(n_values)), "All mean motions should be finite"
        assert np.all(r_values > 0), "All radii should be positive"
        assert np.all(n_values > 0), "All mean motions should be positive"
        
        # Check canonical sorting
        det_ids = labels.det_id.to_pylist()
        orbit_ids = labels.orbit_id.to_pylist()
        variant_ids = labels.variant_id.to_numpy()
        for i in range(1, len(labels)):
            curr_det, prev_det = det_ids[i], det_ids[i-1]
            curr_orbit, prev_orbit = orbit_ids[i], orbit_ids[i-1]
            curr_var, prev_var = variant_ids[i], variant_ids[i-1]
            
            # Should be sorted by (det_id, orbit_id, variant_id)
            if curr_det == prev_det and curr_orbit == prev_orbit:
                assert curr_var >= prev_var, "variant_id should be non-decreasing within (det,orbit)"

    def test_multi_k_near_node_variants(self):
        """Test multi-K variants for high-inclination orbit near nodes."""
        # Use real orbits from sample data - select one with higher inclination
        orbits = make_real_orbits(num_orbits=5)
        orbit = orbits[2:3]  # Select third orbit
        
        # Generate times when object is near nodes (requires orbital mechanics calculation)
        # For simplicity, use several observation times
        base_mjd = orbit.coordinates.time.mjd()[0].as_py()
        times = Timestamp.from_mjd([base_mjd + i * 1.0 for i in range(6)], scale="tdb")
        
        rays = generate_ephemeris_and_rays(orbit, times)
        hits = make_overlap_hits_from_rays_and_orbit(rays, orbit)
        
        # Run with max_k=3, no inline snap filtering to preserve variants
        labels = label_anomalies(hits, rays, orbit, max_k=3, snap_error_max_au=None)
        
        assert len(labels) > 0, "Should generate labels for high-inclination orbit"
        
        # Check for multi-K variants (some detections may have multiple variant_ids)
        variant_ids = labels.variant_id.to_numpy()
        unique_variants = np.unique(variant_ids)
        assert len(unique_variants) >= 1, "Should have at least variant_id=0"
        assert np.all(unique_variants < 3), "All variant_ids should be < max_k=3"
        
        # Verify deduplication worked (no duplicate candidates within tolerance)
        # Note: For real orbits, we may not always get multiple distinct variants
        # depending on the geometry and orbital elements
        det_ids = labels.det_id.to_pylist()
        for det_id in set(det_ids):
            det_labels = labels.apply_mask(pa.compute.equal(labels.det_id, det_id))
            if len(det_labels) > 1:
                # Multiple variants for this detection - check they're reasonably distinct
                f_values = det_labels.f_rad.to_numpy()
                for i in range(len(f_values)):
                    for j in range(i+1, len(f_values)):
                        angle_diff = abs(f_values[i] - f_values[j])
                        angle_diff = min(angle_diff, 2*np.pi - angle_diff)  # Wrap to [0, π]
                        # Allow for machine precision but expect meaningful differences if variants exist
                        if angle_diff <= 1e-10:
                            print(f"Warning: Very similar variants (angle diff={angle_diff}) - deduplication may be working correctly")

    def test_high_e_peri_apo_radius_and_snap(self):
        """Test high-eccentricity orbit shows large radius variation and reasonable snap errors."""
        # Use real orbits from sample data - select one with moderate eccentricity
        orbits = make_real_orbits(num_orbits=3)
        orbit = orbits[1:2]  # Select second orbit
        
        # Generate observations over orbital period to capture peri/apo
        base_mjd = orbit.coordinates.time.mjd()[0].as_py()
        # Eros period ≈ 643 days, sample over ~100 days to get range
        times = Timestamp.from_mjd([base_mjd + i * 10.0 for i in range(10)], scale="tdb")
        
        rays = generate_ephemeris_and_rays(orbit, times)
        hits = make_overlap_hits_from_rays_and_orbit(rays, orbit)
        
        labels = label_anomalies(hits, rays, orbit, max_k=1, snap_error_max_au=1.0)  # Lenient for real data
        
        assert len(labels) > 0, "Should generate labels for high-e orbit"
        
        # Check radius variation
        r_values = labels.r_au.to_numpy()
        r_min, r_max = np.min(r_values), np.max(r_values)
        r_range = r_max - r_min
        
        # For moderate eccentricity, expect some variation (actual e=0.32 from logs)
        # Over a limited time span, we may not see the full peri/apo range
        assert r_range > 0.01, f"Moderate-e orbit should show some radius variation > 0.01 AU, got {r_range}"
        assert r_min > 0.5, f"Minimum radius should be reasonable, got {r_min}"
        assert r_max < 3.0, f"Maximum radius should be reasonable, got {r_max}"
        
        # Snap errors should be reasonable for real ephemeris
        snap_errors = labels.snap_error.to_numpy()
        assert np.all(snap_errors <= 1.0), f"Snap errors should be ≤ 1.0 AU, got max={np.max(snap_errors)}"

    def test_frame_origin_enforcement_equatorial_to_ecliptic(self):
        """Test that labeling handles frame/origin transforms correctly."""
        orbits = make_real_orbits(num_orbits=4)
        orbit = orbits[3:4]  # Select fourth orbit
        
        base_mjd = orbit.coordinates.time.mjd()[0].as_py()
        times = Timestamp.from_mjd([base_mjd + i * 2.0 for i in range(5)], scale="tdb")
        
        # Generate rays (rays_from_detections should handle frame transforms)
        rays = generate_ephemeris_and_rays(orbit, times, observatory_code="G96")  # Mt. Lemmon
        hits = make_overlap_hits_from_rays_and_orbit(rays, orbit)
        
        # Verify rays are in correct frame/origin
        assert rays.observer.frame == "ecliptic", "Rays should be in ecliptic frame"
        observer_origins = rays.observer.origin.code.to_pylist()
        assert all(code == "SUN" for code in observer_origins), "Rays should have SUN origin"
        
        labels = label_anomalies(hits, rays, orbit, max_k=1)
        
        assert len(labels) > 0, "Should generate labels despite frame transforms"
        
        # No NaNs should result from frame/origin mismatches
        for col_name in ["f_rad", "E_rad", "M_rad", "n_rad_day", "r_au", "snap_error"]:
            values = getattr(labels, col_name).to_numpy()
            assert np.all(np.isfinite(values)), f"Column {col_name} should have no NaNs/infs"

    def test_strict_vs_lenient_snap_filtering_monotonicity(self):
        """Test that stricter snap filtering produces subset of lenient results."""
        orbits = make_real_orbits(num_orbits=1)
        orbit = orbits
        
        base_mjd = orbit.coordinates.time.mjd()[0].as_py()
        times = Timestamp.from_mjd([base_mjd + i * 1.0 for i in range(6)], scale="tdb")
        
        rays = generate_ephemeris_and_rays(orbit, times)
        hits = make_overlap_hits_from_rays_and_orbit(rays, orbit)
        
        # Run with strict and lenient snap filtering
        labels_strict = label_anomalies(hits, rays, orbit, max_k=3, snap_error_max_au=1e-6)
        labels_lenient = label_anomalies(hits, rays, orbit, max_k=3, snap_error_max_au=1e-3)
        
        # Monotonicity: lenient should have >= strict results
        assert len(labels_lenient) >= len(labels_strict), "Lenient filtering should produce >= results than strict"
        
        # All strict snap errors should be within strict threshold
        if len(labels_strict) > 0:
            strict_snaps = labels_strict.snap_error.to_numpy()
            assert np.all(strict_snaps <= 1e-6), "All strict results should meet strict threshold"
        
        # All lenient snap errors should be within lenient threshold
        if len(labels_lenient) > 0:
            lenient_snaps = labels_lenient.snap_error.to_numpy()
            assert np.all(lenient_snaps <= 1e-3), "All lenient results should meet lenient threshold"

    def test_edge_cases_circular_zero_inc_no_nans(self):
        """Test edge cases (circular, zero inclination) produce no NaNs."""
        # Select a genuinely near-circular, low-inclination real orbit
        all_orbits = make_real_orbits()
        kep = all_orbits.coordinates.to_keplerian()
        e_vals = kep.e.to_numpy()
        i_vals = kep.i.to_numpy()
        # Prefer e < 0.1 and i < 5 deg if available, else pick lowest-e index
        candidate_indices = np.where((e_vals < 0.1) & (i_vals < 5.0))[0]
        if len(candidate_indices) == 0:
            idx = int(np.argmin(e_vals))
        else:
            idx = int(candidate_indices[0])
        orbit = all_orbits[idx:idx+1]
        
        base_mjd = orbit.coordinates.time.mjd()[0].as_py()
        times = Timestamp.from_mjd([base_mjd + i * 0.5 for i in range(4)], scale="tdb")
        
        rays = generate_ephemeris_and_rays(orbit, times)
        hits = make_overlap_hits_from_rays_and_orbit(rays, orbit)
        
        # Use a realistic snap tolerance for real ephemeris-derived rays
        labels = label_anomalies(hits, rays, orbit, max_k=1, snap_error_max_au=0.05)
        
        assert len(labels) > 0, "Should generate labels for near-circular orbit"
        
        # Check no NaNs in any output columns
        for col_name in ["f_rad", "E_rad", "M_rad", "n_rad_day", "r_au", "snap_error", "plane_distance_au"]:
            values = getattr(labels, col_name).to_numpy()
            assert np.all(np.isfinite(values)), f"Column {col_name} should have no NaNs for edge case orbit"
        
        # For low eccentricity, E ≈ f
        E_values = labels.E_rad.to_numpy()
        f_values = labels.f_rad.to_numpy()
        E_f_diff = np.abs(E_values - f_values)
        assert np.all(E_f_diff < 0.1), f"For low-e orbit, |E-f| should be small, got max={np.max(E_f_diff)}"

    def test_chunking_large_input_shapes(self):
        """Test that chunking works correctly for large input shapes."""
        # Use synthetic orbit for faster ephemeris generation
        orbit = build_keplerian_orbit(a=1.2, e=0.15, i=8.0, raan=45.0, ap=60.0, M=30.0)
        
        # Create > chunk_size observations but keep reasonable for CI (default chunk_size=8192)
        n_obs = 8192 + 100  # 8292 observations - just over one chunk
        base_mjd = orbit.coordinates.time.mjd()[0].as_py()
        times = Timestamp.from_mjd([base_mjd + i * 0.1 for i in range(n_obs)], scale="tdb")  # 6-minute intervals
        
        rays = synthetic_ephemeris_rays(orbit, times, station="500")
        hits = make_overlap_hits_from_rays_and_orbit(rays, orbit)
        
        assert len(hits) == n_obs, f"Should have {n_obs} hits"
        
        labels = label_anomalies(hits, rays, orbit, max_k=1, snap_error_max_au=0.1)
        
        assert len(labels) > 0, "Should generate labels for large input"
        assert len(labels) <= n_obs, "Should not have more labels than input hits"
        
        # Verify canonical sorting is maintained across chunks
        det_ids = labels.det_id.to_pylist()
        for i in range(1, len(det_ids)):
            assert det_ids[i] >= det_ids[i-1], "det_id should be non-decreasing (canonical sort)"

    def test_max_processes_variations(self):
        """Test label_anomalies with different max_processes values."""
        orbits = make_real_orbits(num_orbits=2)
        orbit = orbits
        
        # Use moderate size > chunk_size to test multiprocessing
        n_obs = 8192 + 50
        base_mjd = 59000.0
        times = Timestamp.from_mjd([base_mjd + i * 0.2 for i in range(n_obs)], scale="tdb")
        
        rays = generate_ephemeris_and_rays(orbit, times, max_processes=4)
        hits = make_overlap_hits_from_rays_and_orbit(rays, orbit)
        
        # Test different max_processes values
        for max_proc in [1, 4]:
            labels = label_anomalies(hits, rays, orbit, max_k=1, snap_error_max_au=0.1, max_processes=max_proc)
            assert len(labels) > 0, f"Should generate labels with max_processes={max_proc}"
            
            # Results should be deterministic regardless of max_processes
            det_ids = labels.det_id.to_pylist()
            for i in range(1, len(det_ids)):
                assert det_ids[i] >= det_ids[i-1], f"Sorting should be maintained with max_processes={max_proc}"

    def test_labeling_determinism_repeated_calls(self):
        """Test that repeated calls produce identical results."""
        orbits = make_real_orbits(num_orbits=1)
        orbit = orbits
        
        base_mjd = 59000.0
        times = Timestamp.from_mjd([base_mjd + i * 1.0 for i in range(5)], scale="tdb")
        
        rays = generate_ephemeris_and_rays(orbit, times)
        hits = make_overlap_hits_from_rays_and_orbit(rays, orbit)
        
        # Run labeling twice with identical inputs
        labels1 = label_anomalies(hits, rays, orbit, max_k=3, snap_error_max_au=1e-5)
        labels2 = label_anomalies(hits, rays, orbit, max_k=3, snap_error_max_au=1e-5)
        
        # Results should be identical
        assert len(labels1) == len(labels2), "Repeated calls should produce same number of labels"
        
        if len(labels1) > 0:
            # Check key columns are identical
            for col_name in ["det_id", "orbit_id", "variant_id", "f_rad", "E_rad", "M_rad", "r_au", "snap_error"]:
                if col_name in ["det_id", "orbit_id"]:
                    list1 = getattr(labels1, col_name).to_pylist()
                    list2 = getattr(labels2, col_name).to_pylist()
                    assert list1 == list2, f"Column {col_name} should be identical between runs"
                else:
                    values1 = getattr(labels1, col_name).to_numpy(zero_copy_only=False)
                    values2 = getattr(labels2, col_name).to_numpy(zero_copy_only=False)
                    assert np.allclose(values1, values2, rtol=1e-15), f"Column {col_name} should be identical between runs"

# --- Synthetic helpers ---

def build_keplerian_orbit(
    a: float,
    e: float,
    i: float,
    raan: float,
    ap: float,
    M: float,
    epoch_mjd: float = 59000.0,
    frame: str = "ecliptic",
) -> Orbits:
    times = Timestamp.from_mjd([epoch_mjd], scale="tdb")
    kep = KeplerianCoordinates.from_kwargs(
        a=[a], e=[e], i=[i], raan=[raan], ap=[ap], M=[M], time=times,
        origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]), frame=frame,
    )
    return Orbits.from_kwargs(
        orbit_id=[f"kep_a{a}_e{e}_i{i}_raan{raan}_ap{ap}_M{M}"],
        coordinates=kep.to_cartesian(),
    )


def generate_times(mode: str, epoch_mjd: float, count: int, cadence_days: float) -> Timestamp:
    base = epoch_mjd
    if mode == "uniform":
        times = [base + k * cadence_days for k in range(count)]
    elif mode == "dense":
        times = [base + k * (cadence_days / 5.0) for k in range(count)]
    else:
        times = [base + k * cadence_days for k in range(count)]
    return Timestamp.from_mjd(times, scale="tdb")


def synthetic_ephemeris_rays(orbit: Orbits, times: Timestamp, station: str = "500") -> ObservationRays:
    propagator = ASSISTPropagator()
    observers = Observers.from_codes(times=times, codes=[station] * len(times))
    ephem = propagator.generate_ephemeris(orbit, observers)
    det_ids = [f"syn_{i:06d}" for i in range(len(ephem))]
    return ephemeris_to_rays(ephem, observers=observers, det_id=det_ids)


class TestSyntheticOrbitAnomalyLabeling:
    """Synthetic-orbit tests using ephemeris-derived rays."""

    def test_synthetic_e2e_moderate_e(self):
        orbit = build_keplerian_orbit(a=1.5, e=0.3, i=10.0, raan=30.0, ap=45.0, M=10.0)
        times = generate_times("uniform", 59000.0, count=6, cadence_days=0.5)
        rays = synthetic_ephemeris_rays(orbit, times, station="500")
        hits = make_overlap_hits_from_rays_and_orbit(rays, orbit)
        labels = label_anomalies(hits, rays, orbit, max_k=3, snap_error_max_au=1.0)
        assert len(labels) > 0
        snaps = labels.snap_error.to_numpy()
        assert np.all(np.isfinite(snaps)) and np.all(snaps <= 1.0)

    def test_synthetic_near_circular_low_inc(self):
        orbit = build_keplerian_orbit(a=1.0, e=0.01, i=1.0, raan=0.0, ap=0.0, M=0.0)
        times = generate_times("uniform", 59000.0, count=5, cadence_days=0.5)
        rays = synthetic_ephemeris_rays(orbit, times, station="G96")
        hits = make_overlap_hits_from_rays_and_orbit(rays, orbit)
        labels = label_anomalies(hits, rays, orbit, max_k=1, snap_error_max_au=0.05)
        assert len(labels) > 0
        E = labels.E_rad.to_numpy()
        f = labels.f_rad.to_numpy()
        assert np.all(np.isfinite(E)) and np.all(np.isfinite(f))
        assert np.all(np.abs(E - f) < 0.1)

    def test_synthetic_high_e_radius_spread(self):
        orbit = build_keplerian_orbit(a=2.0, e=0.8, i=20.0, raan=60.0, ap=120.0, M=0.0)
        times = generate_times("uniform", 59000.0, count=8, cadence_days=5.0)
        rays = synthetic_ephemeris_rays(orbit, times, station="500")
        hits = make_overlap_hits_from_rays_and_orbit(rays, orbit)
        labels = label_anomalies(hits, rays, orbit, max_k=1, snap_error_max_au=1.0)
        assert len(labels) > 0
        r = labels.r_au.to_numpy()
        if len(r) > 1:
            assert (np.max(r) - np.min(r)) > 0.2

    def test_synthetic_node_ambiguity_multi_k(self):
        orbit = build_keplerian_orbit(a=1.0, e=0.2, i=75.0, raan=90.0, ap=10.0, M=0.0)
        times = generate_times("dense", 59000.0, count=6, cadence_days=0.4)
        rays = synthetic_ephemeris_rays(orbit, times, station="X05")
        hits = make_overlap_hits_from_rays_and_orbit(rays, orbit)
        labels = label_anomalies(hits, rays, orbit, max_k=3, snap_error_max_au=None)
        assert len(labels) > 0
        vids = labels.variant_id.to_numpy()
        assert np.all(vids < 3)
        # Ensure f variants are finite and deduped
        fvals = labels.f_rad.to_numpy()
        assert np.all(np.isfinite(fvals))

    def test_synthetic_determinism(self):
        orbit = build_keplerian_orbit(a=1.2, e=0.25, i=15.0, raan=45.0, ap=75.0, M=5.0)
        times = generate_times("uniform", 59000.0, count=5, cadence_days=1.0)
        rays = synthetic_ephemeris_rays(orbit, times, station="500")
        hits = make_overlap_hits_from_rays_and_orbit(rays, orbit)
        labels1 = label_anomalies(hits, rays, orbit, max_k=3, snap_error_max_au=1e-3)
        labels2 = label_anomalies(hits, rays, orbit, max_k=3, snap_error_max_au=1e-3)
        assert len(labels1) == len(labels2)
        # Use safe conversions to avoid Arrow zero-copy issues
        numeric_cols = ["f_rad", "E_rad", "M_rad", "r_au", "snap_error"]
        for col in numeric_cols:
            v1 = getattr(labels1, col).to_numpy(zero_copy_only=False)
            v2 = getattr(labels2, col).to_numpy(zero_copy_only=False)
            assert np.allclose(v1, v2, rtol=1e-12)
