from __future__ import annotations

import numpy.typing as npt

from .._rust.api import (
    calc_apoapsis_distance_numpy as _rust_calc_apoapsis_distance_numpy,
    calc_mean_anomaly_numpy as _rust_calc_mean_anomaly_numpy,
    calc_mean_motion_numpy as _rust_calc_mean_motion_numpy,
    calc_periapsis_distance_numpy as _rust_calc_periapsis_distance_numpy,
    calc_period_numpy as _rust_calc_period_numpy,
    calc_semi_latus_rectum_numpy as _rust_calc_semi_latus_rectum_numpy,
    calc_semi_major_axis_numpy as _rust_calc_semi_major_axis_numpy,
    solve_kepler_numpy as _rust_solve_kepler_numpy,
)
from ._rust_compat import ScalarOrArray, broadcast_pair, require_rust, restore_shape


def calc_period(a: npt.ArrayLike, mu: npt.ArrayLike) -> ScalarOrArray:
    """Calculate orbital period from semi-major axis and gravitational parameter."""
    a_flat, mu_flat, shape = broadcast_pair(a, mu)
    out = require_rust(_rust_calc_period_numpy(a_flat, mu_flat), "dynamics.calc_period")
    return restore_shape(out, shape)


def calc_periapsis_distance(a: npt.ArrayLike, e: npt.ArrayLike) -> ScalarOrArray:
    """Calculate periapsis distance from semi-major axis and eccentricity."""
    a_flat, e_flat, shape = broadcast_pair(a, e)
    out = require_rust(
        _rust_calc_periapsis_distance_numpy(a_flat, e_flat),
        "dynamics.calc_periapsis_distance",
    )
    return restore_shape(out, shape)


def calc_apoapsis_distance(a: npt.ArrayLike, e: npt.ArrayLike) -> ScalarOrArray:
    """Calculate apoapsis distance from semi-major axis and eccentricity."""
    a_flat, e_flat, shape = broadcast_pair(a, e)
    out = require_rust(
        _rust_calc_apoapsis_distance_numpy(a_flat, e_flat),
        "dynamics.calc_apoapsis_distance",
    )
    return restore_shape(out, shape)


def calc_semi_major_axis(q: npt.ArrayLike, e: npt.ArrayLike) -> ScalarOrArray:
    """Calculate semi-major axis from periapsis distance and eccentricity."""
    q_flat, e_flat, shape = broadcast_pair(q, e)
    out = require_rust(
        _rust_calc_semi_major_axis_numpy(q_flat, e_flat),
        "dynamics.calc_semi_major_axis",
    )
    return restore_shape(out, shape)


def calc_semi_latus_rectum(a: npt.ArrayLike, e: npt.ArrayLike) -> ScalarOrArray:
    """Calculate semi-latus rectum from semi-major axis and eccentricity."""
    a_flat, e_flat, shape = broadcast_pair(a, e)
    out = require_rust(
        _rust_calc_semi_latus_rectum_numpy(a_flat, e_flat),
        "dynamics.calc_semi_latus_rectum",
    )
    return restore_shape(out, shape)


def calc_mean_motion(a: npt.ArrayLike, mu: npt.ArrayLike) -> ScalarOrArray:
    """Calculate mean motion from semi-major axis and gravitational parameter."""
    a_flat, mu_flat, shape = broadcast_pair(a, mu)
    out = require_rust(
        _rust_calc_mean_motion_numpy(a_flat, mu_flat),
        "dynamics.calc_mean_motion",
    )
    return restore_shape(out, shape)


def calc_mean_anomaly(nu: npt.ArrayLike, e: npt.ArrayLike) -> ScalarOrArray:
    """Calculate mean anomaly from true anomaly and eccentricity."""
    nu_flat, e_flat, shape = broadcast_pair(nu, e)
    out = require_rust(
        _rust_calc_mean_anomaly_numpy(nu_flat, e_flat),
        "dynamics.calc_mean_anomaly",
    )
    return restore_shape(out, shape)


def solve_kepler(
    e: npt.ArrayLike,
    M: npt.ArrayLike,
    max_iter: int = 100,
    tol: float = 1e-15,
) -> ScalarOrArray:
    """Solve Kepler's equation and return true anomaly in radians."""
    e_flat, m_flat, shape = broadcast_pair(e, M)
    out = require_rust(
        _rust_solve_kepler_numpy(e_flat, m_flat, max_iter=max_iter, tol=tol),
        "dynamics.solve_kepler",
    )
    return restore_shape(out, shape)
