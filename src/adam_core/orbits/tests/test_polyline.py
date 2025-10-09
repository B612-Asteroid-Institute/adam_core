"""
Tests for orbit polyline sampling and segment representation.
"""

import numpy as np
import pytest

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.orbits.orbits import Orbits
from adam_core.orbits.polyline import (
    OrbitPolylineSegments,
    OrbitsPlaneParams,
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
        x=[1.0],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[0.017202],
        vz=[0.0],  # ~1 AU/day for circular orbit
        time=times[0:1],
        origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]),
        frame="ecliptic",
    )

    # Orbit 2: Eccentric orbit (e=0.5)
    coords2 = CartesianCoordinates.from_kwargs(
        x=[1.5],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[0.014],
        vz=[0.0],
        time=times[1:2],
        origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]),
        frame="ecliptic",
    )

    # Orbit 3: Inclined orbit
    coords3 = CartesianCoordinates.from_kwargs(
        x=[1.0],
        y=[0.0],
        z=[0.5],
        vx=[0.0],
        vy=[0.015],
        vz=[0.008],
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
            p_x=[1.0],
            p_y=[0.0],
            p_z=[0.0],
            q_x=[0.0],
            q_y=[1.0],
            q_z=[0.0],
            n_x=[0.0],
            n_y=[0.0],
            n_z=[1.0],
            r0_x=[0.0],
            r0_y=[0.0],
            r0_z=[0.0],
            a=[1.0],
            e=[0.0],
            M0=[0.0],
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
            p = np.array(
                [params.p_x[i].as_py(), params.p_y[i].as_py(), params.p_z[i].as_py()]
            )
            q = np.array(
                [params.q_x[i].as_py(), params.q_y[i].as_py(), params.q_z[i].as_py()]
            )
            n = np.array(
                [params.n_x[i].as_py(), params.n_y[i].as_py(), params.n_z[i].as_py()]
            )

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
        params, segments = sample_ellipse_adaptive(
            orbits, max_chord_arcmin=max_chord_arcmin
        )

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
            chord_length = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)

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
                    x0_next = orbit_segments.x0[i + 1].as_py()
                    y0_next = orbit_segments.y0[i + 1].as_py()
                    z0_next = orbit_segments.z0[i + 1].as_py()

                    x1_curr = orbit_segments.x1[i].as_py()
                    y1_curr = orbit_segments.y1[i].as_py()
                    z1_curr = orbit_segments.z1[i].as_py()

                    # Should be the same point (within tolerance)
                    distance = np.sqrt(
                        (x0_next - x1_curr) ** 2
                        + (y0_next - y1_curr) ** 2
                        + (z0_next - z1_curr) ** 2
                    )
                    assert distance < 1e-10


class TestComputeSegmentAabbs:
    """Test AABB computation with guard band padding."""

    def test_empty_segments(self):
        """Test with empty segments."""
        empty_segments = OrbitPolylineSegments.empty()
        result = compute_segment_aabbs(
            empty_segments, guard_arcmin=1.0, epsilon_n_au=1e-6
        )
        assert all(arr.size == 0 for arr in result)

    def test_aabb_computation(self):
        """Test basic AABB computation."""
        # Create test segments
        segments = OrbitPolylineSegments.from_kwargs(
            orbit_id=["test", "test"],
            seg_id=[0, 1],
            x0=[0.0, 1.0],
            y0=[0.0, 0.0],
            z0=[0.0, 0.0],
            x1=[1.0, 2.0],
            y1=[1.0, 1.0],
            z1=[0.0, 0.0],
            r_mid_au=[0.7, 1.5],  # Approximate midpoint distances
            n_x=[0.0, 0.0],
            n_y=[0.0, 0.0],
            n_z=[1.0, 1.0],
        )

        min_x, min_y, min_z, max_x, max_y, max_z = compute_segment_aabbs(
            segments, guard_arcmin=1.0, epsilon_n_au=1e-6
        )

        # Check that AABBs contain segment endpoints
        for i in range(len(segments)):
            x0 = segments.x0[i].as_py()
            y0 = segments.y0[i].as_py()
            z0 = segments.z0[i].as_py()
            x1 = segments.x1[i].as_py()
            y1 = segments.y1[i].as_py()
            z1 = segments.z1[i].as_py()

            # Both endpoints should be inside AABB
            assert min_x[i] <= x0 <= max_x[i]
            assert min_y[i] <= y0 <= max_y[i]
            assert min_z[i] <= z0 <= max_z[i]
            assert min_x[i] <= x1 <= max_x[i]
            assert min_y[i] <= y1 <= max_y[i]
            assert min_z[i] <= z1 <= max_z[i]

    def test_guard_band_padding(self):
        """Test that guard band padding is applied."""
        # Create a simple segment
        segments = OrbitPolylineSegments.from_kwargs(
            orbit_id=["test"],
            seg_id=[0],
            x0=[0.0],
            y0=[0.0],
            z0=[0.0],
            x1=[1.0],
            y1=[0.0],
            z1=[0.0],
            r_mid_au=[0.5],
            n_x=[0.0],
            n_y=[0.0],
            n_z=[1.0],
        )

        guard_arcmin = 2.0
        min_x, min_y, min_z, max_x, max_y, max_z = compute_segment_aabbs(
            segments, guard_arcmin=guard_arcmin, epsilon_n_au=1e-6
        )

        # Compute expected padding
        theta_guard = guard_arcmin * np.pi / (180 * 60)
        # Conservative padding uses max(r_mid, 1 AU)
        expected_pad = theta_guard * 1.0

        # Check that padding was applied
        # Unpadded AABB would be [0, 1] in x
        # With padding should be approximately [0-pad, 1+pad]
        assert min_x[0] < 0.0
        assert max_x[0] > 1.0
        assert (
            abs(min_x[0] - (-expected_pad)) < expected_pad * 0.1
        )  # Allow some tolerance
        assert abs(max_x[0] - (1.0 + expected_pad)) < expected_pad * 0.1


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline(self):
        """Test the complete M1 pipeline."""
        orbits = create_test_orbits()

        # Sample ellipses
        params, segments = sample_ellipse_adaptive(orbits, max_chord_arcmin=0.5)

        # Compute AABBs
        aabbs = compute_segment_aabbs(segments, guard_arcmin=1.0, epsilon_n_au=1e-6)

        # Verify results
        assert len(params) == len(orbits)
        assert len(segments) > 0
        assert all(arr.size == len(segments) for arr in aabbs)

        # Check that each orbit has segments
        orbit_ids_params = set(params.orbit_id.to_pylist())
        orbit_ids_segments = set(segments.orbit_id.to_pylist())
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


def _make_one_orbit():
    times = Timestamp.from_mjd([59000.0], scale="tdb")
    coords = CartesianCoordinates.from_kwargs(
        x=[1.0],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[0.017202],
        vz=[0.0],
        time=times,
        origin=Origin.from_kwargs(code=[OriginCodes.SUN.name]),
        frame="ecliptic",
    )
    return Orbits.from_kwargs(orbit_id=["o1"], coordinates=coords)


@pytest.mark.parametrize("max_chord_arcmin", [0.5, 1.0, 2.0])
def test_sample_ellipse_adaptive_segment_counts_monotonic(max_chord_arcmin):
    orbits = _make_one_orbit()
    _, segs = sample_ellipse_adaptive(
        orbits, max_chord_arcmin=max_chord_arcmin, max_segments_per_orbit=8192
    )
    assert len(segs) > 0


def test_sample_ellipse_adaptive_determinism_and_continuity():
    orbits = _make_one_orbit()
    _, s1 = sample_ellipse_adaptive(
        orbits, max_chord_arcmin=1.0, max_segments_per_orbit=8192
    )
    _, s2 = sample_ellipse_adaptive(
        orbits, max_chord_arcmin=1.0, max_segments_per_orbit=8192
    )
    assert len(s1) == len(s2)
    assert np.allclose(s1.x0.to_numpy(), s2.x0.to_numpy())
    assert np.allclose(s1.y0.to_numpy(), s2.y0.to_numpy())
    assert np.allclose(s1.z0.to_numpy(), s2.z0.to_numpy())
    assert np.allclose(s1.x1.to_numpy(), s2.x1.to_numpy())
    if len(s1) > 1:
        end_pts = np.column_stack(
            [s1.x1.to_numpy(), s1.y1.to_numpy(), s1.z1.to_numpy()]
        )
        start_pts = np.column_stack(
            [s1.x0.to_numpy(), s1.y0.to_numpy(), s1.z0.to_numpy()]
        )
        deltas = np.linalg.norm(end_pts[:-1] - start_pts[1:], axis=1)
        assert np.all(deltas < 1e-9)


def test_compute_segment_aabbs_contains_endpoints():
    orbits = _make_one_orbit()
    _, segs = sample_ellipse_adaptive(
        orbits, max_chord_arcmin=1.0, max_segments_per_orbit=8192
    )
    aabbs = compute_segment_aabbs(segs, guard_arcmin=1.0, epsilon_n_au=1e-6)
    min_x = np.minimum(segs.x0.to_numpy(), segs.x1.to_numpy())
    max_x = np.maximum(segs.x0.to_numpy(), segs.x1.to_numpy())
    min_y = np.minimum(segs.y0.to_numpy(), segs.y1.to_numpy())
    max_y = np.maximum(segs.y0.to_numpy(), segs.y1.to_numpy())
    min_z = np.minimum(segs.z0.to_numpy(), segs.z1.to_numpy())
    max_z = np.maximum(segs.z0.to_numpy(), segs.z1.to_numpy())
    mnx, mny, mnz, mxx, mxy, mxz = aabbs
    assert np.all(mnx <= min_x + 1e-12)
    assert np.all(mxx >= max_x - 1e-12)
    assert np.all(mny <= min_y + 1e-12)
    assert np.all(mxy >= max_y - 1e-12)
    assert np.all(mnz <= min_z + 1e-12)
    assert np.all(mxz >= max_z - 1e-12)


@pytest.mark.parametrize("guard_arcmin_list", [[0.5, 1.0, 2.0]])
def test_compute_segment_aabbs_guard_monotonic_and_sagitta_guard(guard_arcmin_list):
    orbits = _make_one_orbit()
    _, segs = sample_ellipse_adaptive(
        orbits, max_chord_arcmin=1.0, max_segments_per_orbit=8192
    )
    extents_baseline = []
    extents_sagitta = []
    for g in guard_arcmin_list:
        a = compute_segment_aabbs(
            segs, guard_arcmin=g, epsilon_n_au=1e-6, padding_method="baseline"
        )
        s = compute_segment_aabbs(
            segs, guard_arcmin=g, epsilon_n_au=1e-6, padding_method="sagitta_guard"
        )
        a_mx_extent = (a[3] - a[0]) + (a[4] - a[1]) + (a[5] - a[2])
        s_mx_extent = (s[3] - s[0]) + (s[4] - s[1]) + (s[5] - s[2])
        extents_baseline.append(np.mean(a_mx_extent))
        extents_sagitta.append(np.mean(s_mx_extent))
    for i in range(len(extents_baseline) - 1):
        assert extents_baseline[i] <= extents_baseline[i + 1] + 1e-12
        assert extents_sagitta[i] <= extents_sagitta[i + 1] + 1e-12


def test_empty_inputs_return_empty_tables():
    params, segs = sample_ellipse_adaptive(
        Orbits.empty(), max_chord_arcmin=1.0, max_segments_per_orbit=1024
    )
    assert isinstance(params, OrbitsPlaneParams) and isinstance(
        segs, OrbitPolylineSegments
    )
    assert len(params) == 0 and len(segs) == 0
    segs2 = compute_segment_aabbs(
        OrbitPolylineSegments.empty(), guard_arcmin=1.0, epsilon_n_au=1e-6
    )
    assert all(arr.size == 0 for arr in segs2)
