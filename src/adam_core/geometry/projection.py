"""
Geometric projection utilities for anomaly labeling.

This module provides JAX-compatible functions for computing geometric
projections and distances used in anomaly labeling quality metrics.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

__all__ = [
    "compute_orbital_plane_normal",
    "ray_to_plane_distance",
    "project_ray_to_orbital_plane",
    "ellipse_snap_distance",
    "ellipse_snap_distance_multi_seed",
    "transform_to_perifocal_2d",
]


@jax.jit
def compute_orbital_plane_normal(
    a: jax.Array,
    e: jax.Array,
    i: jax.Array,
    raan: jax.Array,
    ap: jax.Array,
) -> jax.Array:
    """
    Compute the unit normal vector to the orbital plane.

    Parameters
    ----------
    a : jax.Array
        Semi-major axis (AU)
    e : jax.Array
        Eccentricity
    i : jax.Array
        Inclination (radians)
    raan : jax.Array
        Right ascension of ascending node (radians)
    ap : jax.Array
        Argument of periapsis (radians)

    Returns
    -------
    jax.Array
        Unit normal vector to orbital plane (3,)
    """
    # Standard orbital mechanics: normal vector is z-axis of perifocal frame
    # rotated to inertial frame
    # The orbital angular momentum vector is the normal to the orbital plane
    cos_raan = jnp.cos(raan)
    sin_raan = jnp.sin(raan)
    cos_i = jnp.cos(i)
    sin_i = jnp.sin(i)

    # Angular momentum vector (orbital plane normal) in inertial frame
    # h = sin(i) * sin(raan) * x_hat - sin(i) * cos(raan) * y_hat + cos(i) * z_hat
    nx = sin_i * sin_raan
    ny = -sin_i * cos_raan
    nz = cos_i

    # Normalize (should already be unit, but ensure numerical stability)
    norm = jnp.sqrt(nx * nx + ny * ny + nz * nz)
    return jnp.array([nx, ny, nz]) / norm


@jax.jit
def ray_to_plane_distance(
    ray_origin: jax.Array,
    ray_direction: jax.Array,
    plane_normal: jax.Array,
    plane_point: jax.Array = None,
) -> jax.Array:
    """
    Compute the minimum distance from a ray to a plane.

    Parameters
    ----------
    ray_origin : jax.Array
        Ray origin position (3,)
    ray_direction : jax.Array
        Ray direction vector (3,) - should be normalized
    plane_normal : jax.Array
        Plane unit normal vector (3,)
    plane_point : jax.Array, optional
        A point on the plane (3,). If None, assumes plane passes through origin.

    Returns
    -------
    jax.Array
        Minimum distance from ray to plane (scalar)
    """
    if plane_point is None:
        plane_point = jnp.zeros(3)

    # Vector from plane point to ray origin
    w = ray_origin - plane_point

    # Dot products
    n_dot_d = jnp.dot(plane_normal, ray_direction)
    n_dot_w = jnp.dot(plane_normal, w)

    # If ray is parallel to plane (n·d ≈ 0), distance is |n·w|
    # Otherwise, ray intersects plane and minimum distance is 0
    # But for small angles (grazing), we want the perpendicular distance

    # For astronomical applications, rays are nearly parallel to the plane
    # so we compute the perpendicular distance from ray origin to plane
    distance = jnp.abs(n_dot_w)

    return distance


@jax.jit
def project_ray_to_orbital_plane(
    ray_origin: jax.Array,
    ray_direction: jax.Array,
    plane_normal: jax.Array,
    plane_point: jax.Array = None,
) -> jax.Array:
    """
    Project a ray onto the orbital plane and return the intersection point.

    Parameters
    ----------
    ray_origin : jax.Array
        Ray origin position (3,)
    ray_direction : jax.Array
        Ray direction vector (3,) - should be normalized
    plane_normal : jax.Array
        Plane unit normal vector (3,)
    plane_point : jax.Array, optional
        A point on the plane (3,). If None, assumes plane passes through origin.

    Returns
    -------
    jax.Array
        Intersection point of ray with plane (3,)
    """
    if plane_point is None:
        plane_point = jnp.zeros(3)

    # Vector from plane point to ray origin
    w = ray_origin - plane_point

    # Dot products
    n_dot_d = jnp.dot(plane_normal, ray_direction)
    n_dot_w = jnp.dot(plane_normal, w)

    # Parameter t for ray intersection: ray_origin + t * ray_direction
    # Avoid division by zero for parallel rays
    t = jnp.where(
        jnp.abs(n_dot_d) > 1e-12, -n_dot_w / n_dot_d, 0.0  # If parallel, use ray origin
    )

    # Intersection point
    intersection = ray_origin + t * ray_direction

    return intersection


@jax.jit
def ellipse_snap_distance(
    point_2d: jax.Array,
    a: jax.Array,
    e: jax.Array,
    max_iterations: int = 10,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute the minimum distance from a 2D point to an ellipse.

    This uses Newton's method to find the eccentric anomaly E that minimizes
    the distance from the point to the ellipse.

    Parameters
    ----------
    point_2d : jax.Array
        2D point in the orbital plane (2,) - [x, y] in perifocal coordinates
    a : jax.Array
        Semi-major axis (AU)
    e : jax.Array
        Eccentricity
    max_iterations : int, default=10
        Maximum Newton iterations

    Returns
    -------
    distance : jax.Array
        Minimum distance from point to ellipse (AU)
    E_closest : jax.Array
        Eccentric anomaly of closest point on ellipse (radians)
    """
    x, y = point_2d[0], point_2d[1]
    b = a * jnp.sqrt(1 - e * e)  # Semi-minor axis

    # Initial guess for E: use angle from focus to point
    # Focus is at (-a*e, 0) in perifocal coordinates
    focus_x = -a * e
    dx = x - focus_x
    E_init = jnp.arctan2(y, dx)

    def newton_step(E):
        """Single Newton iteration to minimize distance squared."""
        cos_E = jnp.cos(E)
        sin_E = jnp.sin(E)

        # Point on ellipse at eccentric anomaly E
        ellipse_x = a * (cos_E - e)
        ellipse_y = b * sin_E

        # Distance vector from ellipse point to target point
        dx_dist = x - ellipse_x
        dy_dist = y - ellipse_y

        # Derivatives of ellipse point w.r.t. E
        dellipse_x_dE = -a * sin_E
        dellipse_y_dE = b * cos_E

        # First derivative of distance squared w.r.t. E
        f_prime = -2 * (dx_dist * dellipse_x_dE + dy_dist * dellipse_y_dE)

        # Second derivative of distance squared w.r.t. E
        f_double_prime = 2 * (
            dellipse_x_dE * dellipse_x_dE + dellipse_y_dE * dellipse_y_dE
        ) - 2 * (dx_dist * (-a * cos_E) + dy_dist * (-b * sin_E))

        # Newton update with safeguard against zero denominator
        dE = jnp.where(jnp.abs(f_double_prime) > 1e-12, -f_prime / f_double_prime, 0.0)

        # Clamp step size to avoid large jumps
        dE = jnp.clip(dE, -0.5, 0.5)

        return E + dE

    # Run Newton iterations using JAX lax.while_loop
    def cond_fun(state):
        E, E_prev, iteration = state
        converged = jnp.abs(E - E_prev) < 1e-10
        max_iter_reached = iteration >= max_iterations
        return jnp.logical_not(jnp.logical_or(converged, max_iter_reached))

    def body_fun(state):
        E, E_prev, iteration = state
        E_new = newton_step(E)
        return E_new, E, iteration + 1

    # Initial state: (current_E, previous_E, iteration)
    init_state = (E_init, E_init + 1.0, 0)  # Set prev != current to start loop
    final_E, _, _ = lax.while_loop(cond_fun, body_fun, init_state)

    E = final_E

    # Compute final distance
    cos_E = jnp.cos(E)
    sin_E = jnp.sin(E)
    ellipse_x = a * (cos_E - e)
    ellipse_y = b * sin_E

    distance = jnp.sqrt((x - ellipse_x) ** 2 + (y - ellipse_y) ** 2)

    return distance, E


@jax.jit
def transform_to_perifocal_2d(
    point_3d: jax.Array,
    i: jax.Array,
    raan: jax.Array,
    ap: jax.Array,
) -> jax.Array:
    """
    Transform a 3D inertial point to 2D perifocal coordinates.

    Parameters
    ----------
    point_3d : jax.Array
        3D point in inertial coordinates (3,)
    i : jax.Array
        Inclination (radians)
    raan : jax.Array
        Right ascension of ascending node (radians)
    ap : jax.Array
        Argument of periapsis (radians)

    Returns
    -------
    jax.Array
        2D point in perifocal coordinates (2,) - [x, y]
    """
    cos_raan = jnp.cos(raan)
    sin_raan = jnp.sin(raan)
    cos_i = jnp.cos(i)
    sin_i = jnp.sin(i)
    cos_ap = jnp.cos(ap)
    sin_ap = jnp.sin(ap)

    # Rotation matrix from inertial to perifocal
    # Standard transformation: R = R3(ap) * R1(i) * R3(raan)
    # where R3 is rotation about z-axis, R1 is rotation about x-axis
    R11 = cos_raan * cos_ap - sin_raan * sin_ap * cos_i
    R12 = sin_raan * cos_ap + cos_raan * sin_ap * cos_i
    R13 = sin_ap * sin_i

    R21 = -cos_raan * sin_ap - sin_raan * cos_ap * cos_i
    R22 = -sin_raan * sin_ap + cos_raan * cos_ap * cos_i
    R23 = cos_ap * sin_i

    # Transform point (only need x and y components in perifocal frame)
    x_peri = R11 * point_3d[0] + R12 * point_3d[1] + R13 * point_3d[2]
    y_peri = R21 * point_3d[0] + R22 * point_3d[1] + R23 * point_3d[2]

    return jnp.array([x_peri, y_peri])


@partial(jax.jit, static_argnums=(3, 4))
def ellipse_snap_distance_multi_seed(
    point_2d: jax.Array,
    a: jax.Array,
    e: jax.Array,
    max_k: int = 3,
    max_iterations: int = 10,
    dedupe_angle_tol: float = 1e-4,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Compute up to K candidate solutions for minimum distance from a 2D point to an ellipse.

    Single compiled function parameterized by static max_k/max_iterations; avoids per-K wrappers.
    """
    return _ellipse_snap_multi_kernel(point_2d, a, e, max_k, max_iterations, dedupe_angle_tol)


# Removed K-specific wrappers in favor of a single JIT with static max_k


def _ellipse_snap_multi_kernel(
    point_2d: jax.Array,
    a: jax.Array,
    e: jax.Array,
    max_k: int,
    max_iterations: int,
    dedupe_angle_tol: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Core kernel for multi-seed ellipse snapping."""
    x, y = point_2d[0], point_2d[1]

    # Clamp eccentricity to [0, 1) for numerical stability on elliptical case
    e_eff = jnp.clip(e, 0.0, 1.0 - 1e-9)

    # Generate seed angles: primary estimate and phase offsets
    focus_x = -a * e_eff
    dx = x - focus_x
    E_primary = jnp.arctan2(y, dx)

    # Create multiple seeds: primary, +π, -π, +π/2, -π/2 (fixed size for JAX)
    seed_offsets = jnp.array([0.0, jnp.pi, -jnp.pi, jnp.pi / 2, -jnp.pi / 2])
    # Use fixed-size array and mask instead of dynamic slicing
    seeds = E_primary + seed_offsets  # All 5 seeds
    seed_mask = jnp.arange(len(seed_offsets)) < max_k  # Which seeds to use

    # Solve for each seed using vmap
    def solve_single_seed(E_seed):
        return _ellipse_snap_newton_solve(point_2d, a, e_eff, E_seed, max_iterations)

    vmap_solve = jax.vmap(solve_single_seed)
    all_distances, all_E = vmap_solve(seeds)

    # Apply seed mask to filter out unused seeds
    # Mask non-finite results as invalid as well
    finite_mask = jnp.logical_and(jnp.isfinite(all_distances), jnp.isfinite(all_E))
    seed_valid_mask = jnp.logical_and(seed_mask, finite_mask)
    candidate_distances = jnp.where(seed_valid_mask, all_distances, jnp.inf)
    candidate_E = jnp.where(seed_valid_mask, all_E, jnp.nan)

    # Build output by taking first max_k valid candidates
    output_distances = jnp.full(max_k, jnp.inf)
    output_E = jnp.full(max_k, jnp.nan)
    output_valid = jnp.zeros(max_k, dtype=bool)

    # Sort by distance and take first max_k valid candidates
    sorted_indices = jnp.argsort(candidate_distances)

    def add_candidate(carry, i):
        out_dist, out_E, out_valid, count = carry
        idx = sorted_indices[i]

        # Check if this candidate should be added
        is_valid_seed = seed_valid_mask[idx]
        has_space = count < max_k
        should_add = jnp.logical_and(is_valid_seed, has_space)

        # Add to output arrays
        new_out_dist = out_dist.at[count].set(
            jnp.where(should_add, candidate_distances[idx], out_dist[count])
        )
        new_out_E = out_E.at[count].set(
            jnp.where(should_add, candidate_E[idx], out_E[count])
        )
        new_out_valid = out_valid.at[count].set(
            jnp.where(should_add, True, out_valid[count])
        )
        new_count = jnp.where(should_add, count + 1, count)

        return (new_out_dist, new_out_E, new_out_valid, new_count), None

    init_carry = (output_distances, output_E, output_valid, 0)
    (final_distances, final_E, final_valid, _), _ = lax.scan(
        add_candidate, init_carry, jnp.arange(len(seeds))
    )

    return final_distances, final_E, final_valid


@jax.jit
def _ellipse_snap_newton_solve(
    point_2d: jax.Array,
    a: jax.Array,
    e: jax.Array,
    E_init: jax.Array,
    max_iterations: int,
) -> tuple[jax.Array, jax.Array]:
    """
    Solve for ellipse snap distance using Newton's method from a given initial guess.

    This is a helper function for ellipse_snap_distance_multi_seed.
    """
    x, y = point_2d[0], point_2d[1]
    b = a * jnp.sqrt(1 - e * e)

    def newton_step(E):
        """Single Newton iteration to minimize distance squared."""
        cos_E = jnp.cos(E)
        sin_E = jnp.sin(E)

        # Point on ellipse at eccentric anomaly E
        ellipse_x = a * (cos_E - e)
        ellipse_y = b * sin_E

        # Distance vector from ellipse point to target point
        dx_dist = x - ellipse_x
        dy_dist = y - ellipse_y

        # Derivatives of ellipse point w.r.t. E
        dellipse_x_dE = -a * sin_E
        dellipse_y_dE = b * cos_E

        # First derivative of distance squared w.r.t. E
        f_prime = -2 * (dx_dist * dellipse_x_dE + dy_dist * dellipse_y_dE)

        # Second derivative of distance squared w.r.t. E
        f_double_prime = 2 * (
            dellipse_x_dE * dellipse_x_dE + dellipse_y_dE * dellipse_y_dE
        ) - 2 * (dx_dist * (-a * cos_E) + dy_dist * (-b * sin_E))

        # Newton update with safeguard against zero denominator
        dE = jnp.where(jnp.abs(f_double_prime) > 1e-12, -f_prime / f_double_prime, 0.0)

        # Clamp step size to avoid large jumps
        dE = jnp.clip(dE, -0.5, 0.5)

        return E + dE

    # Run Newton iterations using JAX lax.while_loop
    def cond_fun(state):
        E, E_prev, iteration = state
        converged = jnp.abs(E - E_prev) < 1e-10
        max_iter_reached = iteration >= max_iterations
        return jnp.logical_not(jnp.logical_or(converged, max_iter_reached))

    def body_fun(state):
        E, E_prev, iteration = state
        E_new = newton_step(E)
        return E_new, E, iteration + 1

    # Initial state: (current_E, previous_E, iteration)
    init_state = (E_init, E_init + 1.0, 0)  # Set prev != current to start loop
    final_E, _, _ = lax.while_loop(cond_fun, body_fun, init_state)

    E = final_E

    # Compute final distance
    cos_E = jnp.cos(E)
    sin_E = jnp.sin(E)
    ellipse_x = a * (cos_E - e)
    ellipse_y = b * sin_E

    distance = jnp.sqrt((x - ellipse_x) ** 2 + (y - ellipse_y) ** 2)

    return distance, E
