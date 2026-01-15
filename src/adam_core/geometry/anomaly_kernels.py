"""
JAX kernels for anomaly labeling.

This module provides JIT-compiled kernels for converting geometric overlap hits
into orbital anomaly assignments. The core algorithm projects detection rays
onto orbital planes and finds the best-fit anomaly on the ellipse.
"""

from __future__ import annotations

import logging
from typing import Tuple

import jax
import jax.numpy as jnp

__all__ = [
    "project_ray_to_orbital_plane",
    "compute_anomaly_from_plane_point",
    "ellipse_position_from_anomaly",
    "anomaly_derivatives",
]

logger = logging.getLogger(__name__)


@jax.jit
def project_ray_to_orbital_plane(
    ray_origin: jax.Array,
    ray_direction: jax.Array,
    plane_normal: jax.Array,
    plane_point: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Project a ray onto an orbital plane.

    Parameters
    ----------
    ray_origin : jax.Array (3,)
        Ray origin point in heliocentric coordinates
    ray_direction : jax.Array (3,)
        Ray direction vector (should be normalized)
    plane_normal : jax.Array (3,)
        Orbital plane normal vector (normalized)
    plane_point : jax.Array (3,)
        Reference point on the orbital plane (e.g., ellipse center)

    Returns
    -------
    intersection_point : jax.Array (3,)
        Ray-plane intersection point
    plane_distance : jax.Array (scalar)
        Distance from intersection to plane reference point
    ray_parameter : jax.Array (scalar)
        Parameter t where intersection = ray_origin + t * ray_direction
    """
    # Vector from plane point to ray origin
    w = ray_origin - plane_point

    # Ray-plane intersection parameter
    # t = -(w · n) / (d · n) where d is ray direction, n is plane normal
    denom = jnp.dot(ray_direction, plane_normal)

    # Handle parallel case (ray parallel to plane)
    is_parallel = jnp.abs(denom) < 1e-15
    t = jnp.where(
        is_parallel,
        0.0,  # Project to ray origin if parallel
        -jnp.dot(w, plane_normal) / denom,
    )

    # Clamp to non-negative (ray, not line)
    t = jnp.maximum(0.0, t)

    # Compute intersection point
    intersection = ray_origin + t * ray_direction

    # Distance from intersection to plane reference point
    plane_distance = jnp.linalg.norm(intersection - plane_point)

    return intersection, plane_distance, t


@jax.jit
def compute_in_plane_coordinates(
    point: jax.Array,
    plane_center: jax.Array,
    basis_p: jax.Array,
    basis_q: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """
    Convert 3D point to 2D coordinates in orbital plane basis.

    Parameters
    ----------
    point : jax.Array (3,)
        3D point to convert
    plane_center : jax.Array (3,)
        Center of the orbital plane coordinate system
    basis_p : jax.Array (3,)
        First basis vector (normalized)
    basis_q : jax.Array (3,)
        Second basis vector (normalized, orthogonal to basis_p)

    Returns
    -------
    x : jax.Array (scalar)
        Coordinate along basis_p
    y : jax.Array (scalar)
        Coordinate along basis_q
    """
    # Vector from center to point
    relative = point - plane_center

    # Project onto basis vectors
    x = jnp.dot(relative, basis_p)
    y = jnp.dot(relative, basis_q)

    return x, y


@jax.jit
def ellipse_position_from_anomaly(
    f: jax.Array,
    a: jax.Array,
    e: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Compute ellipse position from true anomaly.

    Parameters
    ----------
    f : jax.Array (scalar)
        True anomaly (radians)
    a : jax.Array (scalar)
        Semi-major axis
    e : jax.Array (scalar)
        Eccentricity

    Returns
    -------
    x : jax.Array (scalar)
        X coordinate in ellipse frame
    y : jax.Array (scalar)
        Y coordinate in ellipse frame
    r : jax.Array (scalar)
        Heliocentric distance
    """
    # Ellipse equation: r = a(1-e²)/(1+e*cos(f))
    cos_f = jnp.cos(f)
    sin_f = jnp.sin(f)

    r = a * (1 - e * e) / (1 + e * cos_f)

    x = r * cos_f
    y = r * sin_f

    return x, y, r


@jax.jit
def anomaly_derivatives(
    f: jax.Array,
    e: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """
    Compute derivatives for Newton's method anomaly refinement.

    Parameters
    ----------
    f : jax.Array (scalar)
        True anomaly (radians)
    e : jax.Array (scalar)
        Eccentricity

    Returns
    -------
    dr_df : jax.Array (scalar)
        Derivative of radius with respect to true anomaly
    d2r_df2 : jax.Array (scalar)
        Second derivative of radius with respect to true anomaly
    """
    cos_f = jnp.cos(f)
    sin_f = jnp.sin(f)

    # r = a(1-e²)/(1+e*cos(f))
    # Let u = 1 + e*cos(f), then r = a(1-e²)/u
    u = 1 + e * cos_f

    # dr/df = a(1-e²) * e*sin(f) / u²
    dr_df = e * sin_f / (u * u)

    # d²r/df² = a(1-e²) * [e*cos(f)/u² - 2*e²*sin²(f)/u³]
    d2r_df2 = (e * cos_f / (u * u)) - (2 * e * e * sin_f * sin_f / (u * u * u))

    return dr_df, d2r_df2


@jax.jit
def compute_anomaly_from_plane_point(
    x_target: jax.Array,
    y_target: jax.Array,
    a: jax.Array,
    e: jax.Array,
    f_seed: jax.Array,
    max_iterations: int = 5,
) -> Tuple[jax.Array, jax.Array]:
    """
    Refine true anomaly to minimize distance to target point.

    Uses Newton's method with fixed iterations for JAX compatibility.

    Parameters
    ----------
    x_target : jax.Array (scalar)
        Target X coordinate in ellipse frame
    y_target : jax.Array (scalar)
        Target Y coordinate in ellipse frame
    a : jax.Array (scalar)
        Semi-major axis
    e : jax.Array (scalar)
        Eccentricity
    f_seed : jax.Array (scalar)
        Initial guess for true anomaly
    max_iterations : int, default 5
        Maximum Newton iterations

    Returns
    -------
    f_refined : jax.Array (scalar)
        Refined true anomaly
    residual : jax.Array (scalar)
        Final distance residual
    """
    f = f_seed

    def newton_step(f_current):
        # Current ellipse position
        x_ellipse, y_ellipse, r = ellipse_position_from_anomaly(f_current, a, e)

        # Distance squared to target
        dx = x_ellipse - x_target
        dy = y_ellipse - y_target
        dist_sq = dx * dx + dy * dy

        # Derivatives for Newton's method
        cos_f = jnp.cos(f_current)
        sin_f = jnp.sin(f_current)

        # dr/df and d²r/df²
        dr_df, d2r_df2 = anomaly_derivatives(f_current, e)

        # dx/df = dr/df * cos(f) - r * sin(f)
        # dy/df = dr/df * sin(f) + r * cos(f)
        dx_df = dr_df * cos_f - r * sin_f
        dy_df = dr_df * sin_f + r * cos_f

        # Gradient of distance squared: d(dist²)/df = 2*(dx*dx/df + dy*dy/df)
        grad = 2 * (dx * dx_df + dy * dy_df)

        # Second derivative (simplified Hessian approximation)
        # d²x/df² = d²r/df² * cos(f) - 2*dr/df*sin(f) - r*cos(f)
        # d²y/df² = d²r/df² * sin(f) + 2*dr/df*cos(f) - r*sin(f)
        d2x_df2 = d2r_df2 * cos_f - 2 * dr_df * sin_f - r * cos_f
        d2y_df2 = d2r_df2 * sin_f + 2 * dr_df * cos_f - r * sin_f

        hess = 2 * (dx_df * dx_df + dx * d2x_df2 + dy_df * dy_df + dy * d2y_df2)

        # Newton step: f_new = f - grad/hess
        # Protect against zero or negative Hessian
        step = jnp.where(jnp.abs(hess) > 1e-15, grad / hess, 0.0)

        f_new = f_current - step

        return f_new, jnp.sqrt(dist_sq)

    # Fixed number of Newton iterations using JAX loop
    def body_fun(i, carry):
        f_current, _ = carry
        f_new, residual_new = newton_step(f_current)
        return f_new, residual_new

    f_final, residual_final = jax.lax.fori_loop(0, max_iterations, body_fun, (f, 0.0))

    return f_final, residual_final


@jax.jit
def eccentric_and_mean_anomaly_from_true(
    f: jax.Array,
    e: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """
    Convert true anomaly to eccentric and mean anomaly.

    Parameters
    ----------
    f : jax.Array (scalar)
        True anomaly (radians)
    e : jax.Array (scalar)
        Eccentricity

    Returns
    -------
    E : jax.Array (scalar)
        Eccentric anomaly (radians)
    M : jax.Array (scalar)
        Mean anomaly (radians)
    """
    # Eccentric anomaly from true anomaly
    # tan(E/2) = sqrt((1-e)/(1+e)) * tan(f/2)
    tan_half_f = jnp.tan(f / 2)
    tan_half_E = jnp.sqrt((1 - e) / (1 + e)) * tan_half_f
    E = 2 * jnp.arctan(tan_half_E)

    # Mean anomaly from eccentric anomaly
    # M = E - e*sin(E)
    M = E - e * jnp.sin(E)

    return E, M
