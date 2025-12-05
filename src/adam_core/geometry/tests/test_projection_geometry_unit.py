from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from adam_core.geometry.projection import (
    ellipse_snap_distance,
    project_ray_to_orbital_plane,
    ray_to_plane_distance,
    transform_to_perifocal_2d,
)


def test_ray_to_plane_distance_parallel_and_nonparallel():
    n = jnp.array([0.0, 0.0, 1.0])
    # Non-parallel ray
    ro = jnp.array([1.0, 2.0, 3.0])
    rd = jnp.array([-1.0, -2.0, -3.0])
    d = float(ray_to_plane_distance(ro, rd, n))
    # Plane through origin z=0 → distance |z|
    npt.assert_allclose(d, 3.0, atol=1e-12)

    # Parallel ray (direction perpendicular to normal)
    rd_parallel = jnp.array([1.0, 0.0, 0.0])
    d2 = float(ray_to_plane_distance(ro, rd_parallel, n))
    npt.assert_allclose(d2, 3.0, atol=1e-12)


def test_project_ray_to_orbital_plane_intersection_on_plane():
    n = jnp.array([0.0, 0.0, 1.0])
    ro = jnp.array([1.0, 2.0, 3.0])
    rd = jnp.array([-1.0, -2.0, -1.0])
    p = project_ray_to_orbital_plane(ro, rd, n)
    # Resulting point lies on plane z=0
    npt.assert_allclose(jnp.dot(n, p), 0.0, atol=1e-10)


def test_transform_to_perifocal_2d_identity_for_zero_angles():
    # With i=raan=ap=0, perifocal x,y are inertial x,y
    i = jnp.array(0.0)
    raan = jnp.array(0.0)
    ap = jnp.array(0.0)
    pt = jnp.array([3.0, -4.0, 5.0])
    xy = transform_to_perifocal_2d(pt, i, raan, ap)
    npt.assert_allclose(np.asarray(xy), np.array([3.0, -4.0]), atol=1e-12)


def _wrap_angle(x: float) -> float:
    y = np.mod(x, 2.0 * np.pi)
    return y if y >= 0 else y + 2.0 * np.pi


def _angle_diff(a: float, b: float) -> float:
    da = _wrap_angle(a) - _wrap_angle(b)
    if da > np.pi:
        da -= 2.0 * np.pi
    if da < -np.pi:
        da += 2.0 * np.pi
    return da


def test_ellipse_snap_distance_exact_points_small_residual():
    a = jnp.array(2.0)
    e = jnp.array(0.3)
    E_true = 1.0  # radians
    b = float(a) * np.sqrt(1.0 - float(e) ** 2)
    x = float(a) * (np.cos(E_true) - float(e))
    y = b * np.sin(E_true)
    dist, E_est = ellipse_snap_distance(jnp.array([x, y]), a, e)
    npt.assert_allclose(float(dist), 0.0, atol=1e-8)
    npt.assert_allclose(_angle_diff(float(E_est), E_true), 0.0, atol=1e-6)
