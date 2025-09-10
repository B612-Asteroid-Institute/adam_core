"""
Tests for geometric projection utilities.

This module tests the projection functions used for computing
real plane_distance_au and snap_error metrics in anomaly labeling.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from adam_core.geometry.projection import (
    compute_orbital_plane_normal,
    ellipse_snap_distance,
    project_ray_to_orbital_plane,
    ray_to_plane_distance,
    transform_to_perifocal_2d,
)


class TestOrbitalPlaneNormal:
    """Test computation of orbital plane normal vectors."""

    def test_circular_orbit_xy_plane(self):
        """Test circular orbit in xy-plane (i=0)."""
        a = 1.0
        e = 0.0
        i = 0.0  # xy-plane
        raan = 0.0
        ap = 0.0

        normal = compute_orbital_plane_normal(a, e, i, raan, ap)

        # Should be z-axis for xy-plane orbit
        expected = jnp.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(normal, expected, atol=1e-10)

    def test_inclined_orbit(self):
        """Test orbit with 45° inclination."""
        a = 1.0
        e = 0.0
        i = np.pi / 4  # 45°
        raan = 0.123  # non-zero RAAN
        ap = 0.456  # non-zero argument of periapsis

        normal = compute_orbital_plane_normal(a, e, i, raan, ap)

        # Expected normal per formula: [sin i sin Ω, -sin i cos Ω, cos i]
        expected_x = np.sin(i) * np.sin(raan)
        expected_y = -np.sin(i) * np.cos(raan)
        expected_z = np.cos(i)
        expected = jnp.array([expected_x, expected_y, expected_z])

        np.testing.assert_allclose(normal, expected, atol=1e-10)

        # Should be unit vector
        assert abs(jnp.linalg.norm(normal) - 1.0) < 1e-10

    def test_polar_orbit(self):
        """Test polar orbit (i=90°)."""
        a = 1.0
        e = 0.0
        i = np.pi / 2  # 90°
        raan = 0.0
        ap = 0.0

        normal = compute_orbital_plane_normal(a, e, i, raan, ap)

        # For i=90°, raan=0°: h = [0, -1, 0]
        expected = jnp.array([0.0, -1.0, 0.0])
        np.testing.assert_allclose(normal, expected, atol=1e-10)


class TestRayToPlaneDistance:
    """Test ray-to-plane distance computation."""

    def test_ray_parallel_to_plane(self):
        """Test ray parallel to xy-plane."""
        ray_origin = jnp.array([0.0, 0.0, 1.0])  # 1 AU above xy-plane
        ray_direction = jnp.array([1.0, 0.0, 0.0])  # Along x-axis
        plane_normal = jnp.array([0.0, 0.0, 1.0])  # xy-plane normal

        distance = ray_to_plane_distance(ray_origin, ray_direction, plane_normal)

        # Should be 1.0 AU (height above plane)
        assert abs(distance - 1.0) < 1e-10

    def test_ray_in_plane(self):
        """Test ray that lies in the plane."""
        ray_origin = jnp.array([1.0, 1.0, 0.0])  # In xy-plane
        ray_direction = jnp.array([1.0, 0.0, 0.0])  # Along x-axis
        plane_normal = jnp.array([0.0, 0.0, 1.0])  # xy-plane normal

        distance = ray_to_plane_distance(ray_origin, ray_direction, plane_normal)

        # Should be 0.0 (ray is in plane)
        assert abs(distance) < 1e-10

    def test_ray_below_plane(self):
        """Test ray below the plane."""
        ray_origin = jnp.array([0.0, 0.0, -0.5])  # 0.5 AU below xy-plane
        ray_direction = jnp.array([1.0, 0.0, 0.0])  # Along x-axis
        plane_normal = jnp.array([0.0, 0.0, 1.0])  # xy-plane normal

        distance = ray_to_plane_distance(ray_origin, ray_direction, plane_normal)

        # Should be 0.5 AU (absolute distance)
        assert abs(distance - 0.5) < 1e-10


class TestProjectRayToPlane:
    """Test ray projection to orbital plane."""

    def test_perpendicular_intersection(self):
        """Test ray perpendicular to plane."""
        ray_origin = jnp.array([0.0, 0.0, 1.0])  # Above xy-plane
        ray_direction = jnp.array([0.0, 0.0, -1.0])  # Toward plane
        plane_normal = jnp.array([0.0, 0.0, 1.0])  # xy-plane normal

        intersection = project_ray_to_orbital_plane(
            ray_origin, ray_direction, plane_normal
        )

        # Should intersect at origin
        expected = jnp.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(intersection, expected, atol=1e-10)

    def test_oblique_intersection(self):
        """Test oblique ray intersection."""
        ray_origin = jnp.array([0.0, 0.0, 1.0])  # Above xy-plane
        ray_direction = jnp.array([1.0, 0.0, -1.0])  # 45° down and right
        ray_direction = ray_direction / jnp.linalg.norm(ray_direction)  # Normalize
        plane_normal = jnp.array([0.0, 0.0, 1.0])  # xy-plane normal

        intersection = project_ray_to_orbital_plane(
            ray_origin, ray_direction, plane_normal
        )

        # Should intersect at (1, 0, 0) since ray goes 1 unit right for 1 unit down
        expected = jnp.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(intersection, expected, atol=1e-10)


class TestEllipseSnapDistance:
    """Test ellipse snap distance computation."""

    def test_circular_orbit_on_ellipse(self):
        """Test point exactly on circular orbit."""
        a = 1.0
        e = 0.0  # Circular
        point_2d = jnp.array([1.0, 0.0])  # On circle at (a, 0)

        distance, E_closest = ellipse_snap_distance(point_2d, a, e)

        # Should be zero distance
        assert abs(distance) < 1e-6
        # E should be 0 (periapsis)
        assert abs(E_closest) < 1e-6

    def test_circular_orbit_inside(self):
        """Test point inside circular orbit."""
        a = 1.0
        e = 0.0  # Circular
        point_2d = jnp.array([0.5, 0.0])  # Inside circle

        distance, E_closest = ellipse_snap_distance(point_2d, a, e)

        # Should be 0.5 AU (distance from 0.5 to 1.0)
        assert abs(distance - 0.5) < 1e-6
        # E should be 0 (closest point is periapsis)
        assert abs(E_closest) < 1e-6

    def test_circular_orbit_outside(self):
        """Test point outside circular orbit."""
        a = 1.0
        e = 0.0  # Circular
        point_2d = jnp.array([2.0, 0.0])  # Outside circle

        distance, E_closest = ellipse_snap_distance(point_2d, a, e)

        # Should be 1.0 AU (distance from 2.0 to 1.0)
        assert abs(distance - 1.0) < 1e-6
        # E should be 0 (closest point is periapsis)
        assert abs(E_closest) < 1e-6

    def test_elliptical_orbit_periapsis(self):
        """Test point near periapsis of elliptical orbit."""
        a = 2.0
        e = 0.5
        # Periapsis is at x = a(1-e) = 1.0, y = 0
        point_2d = jnp.array([1.1, 0.0])  # Near periapsis

        distance, E_closest = ellipse_snap_distance(point_2d, a, e)

        # Should be small distance
        assert distance < 0.2  # Rough check
        # E should be near 0 (periapsis)
        assert abs(E_closest) < 0.5

    def test_elliptical_orbit_apoapsis(self):
        """Test point near apoapsis of elliptical orbit."""
        a = 2.0
        e = 0.5
        # Apoapsis is at x = -a(1+e) = -3.0, y = 0
        point_2d = jnp.array([-2.9, 0.0])  # Near apoapsis

        distance, E_closest = ellipse_snap_distance(point_2d, a, e)

        # Should be small distance
        assert distance < 0.2  # Rough check
        # E should be near π (apoapsis)
        assert abs(abs(E_closest) - np.pi) < 0.5


class TestTransformToPerifocal:
    """Test coordinate transformation to perifocal frame."""

    def test_identity_transformation(self):
        """Test transformation with no rotation (i=raan=ap=0)."""
        point_3d = jnp.array([1.0, 2.0, 3.0])
        i = 0.0
        raan = 0.0
        ap = 0.0

        point_2d = transform_to_perifocal_2d(point_3d, i, raan, ap)

        # Should just take x and y components
        expected = jnp.array([1.0, 2.0])
        np.testing.assert_allclose(point_2d, expected, atol=1e-10)

    def test_90_degree_rotation(self):
        """Test 90° rotation about z-axis."""
        point_3d = jnp.array([1.0, 0.0, 0.0])  # x-axis
        i = 0.0
        raan = np.pi / 2  # 90° rotation about z
        ap = 0.0

        point_2d = transform_to_perifocal_2d(point_3d, i, raan, ap)

        # With raan=90° rotation, x-axis becomes -y-axis in perifocal frame
        expected = jnp.array([0.0, -1.0])
        np.testing.assert_allclose(point_2d, expected, atol=1e-10)


class TestProjectionIntegration:
    """Integration tests combining multiple projection functions."""

    def test_circular_orbit_in_plane_ray(self):
        """Test complete projection pipeline for in-plane circular orbit."""
        # Circular orbit in xy-plane
        a = 1.0
        e = 0.0
        i = 0.0
        raan = 0.0
        ap = 0.0

        # Ray from observer at (0, 0, 0.001) pointing to (1, 0, 0) on orbit
        ray_origin = jnp.array([0.0, 0.0, 0.001])  # Slightly above plane
        target_point = jnp.array([1.0, 0.0, 0.0])  # On orbit
        ray_direction = target_point - ray_origin
        ray_direction = ray_direction / jnp.linalg.norm(ray_direction)

        # Compute plane normal
        normal = compute_orbital_plane_normal(a, e, i, raan, ap)

        # Compute plane distance
        plane_distance = ray_to_plane_distance(ray_origin, ray_direction, normal)

        # Should be very small (ray nearly in plane)
        assert plane_distance < 0.002

        # Project to plane
        projected = project_ray_to_orbital_plane(ray_origin, ray_direction, normal)

        # Transform to perifocal
        point_2d = transform_to_perifocal_2d(projected, i, raan, ap)

        # Compute snap distance
        snap_distance, _ = ellipse_snap_distance(point_2d, a, e)

        # Should be very small (projected point is near orbit)
        assert snap_distance < 0.01

    def test_high_eccentricity_orbit(self):
        """Test projection for high-eccentricity orbit."""
        # High-e orbit
        a = 2.0
        e = 0.8
        i = 0.0
        raan = 0.0
        ap = 0.0

        # Ray pointing to periapsis
        periapsis_x = a * (1 - e)  # 0.4 AU
        ray_origin = jnp.array([0.0, 0.0, 0.01])  # Slightly above plane
        target_point = jnp.array([periapsis_x, 0.0, 0.0])
        ray_direction = target_point - ray_origin
        ray_direction = ray_direction / jnp.linalg.norm(ray_direction)

        # Full projection pipeline
        normal = compute_orbital_plane_normal(a, e, i, raan, ap)
        plane_distance = ray_to_plane_distance(ray_origin, ray_direction, normal)
        projected = project_ray_to_orbital_plane(ray_origin, ray_direction, normal)
        point_2d = transform_to_perifocal_2d(projected, i, raan, ap)
        snap_distance, E_closest = ellipse_snap_distance(point_2d, a, e)

        # Plane distance should be small
        assert plane_distance < 0.02

        # Snap distance should be small (near periapsis)
        assert snap_distance < 0.1

        # E should be near 0 (periapsis)
        assert abs(E_closest) < 0.2
