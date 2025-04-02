"""
Implementation of Lambert's problem using Izzo's method.

This implementation follows the algorithm described in:
Izzo, D. (2015). Revisiting Lambert's problem. Celestial Mechanics and Dynamical Astronomy, 121(1), 1-15.

Credits: Based on poliastro implementation by Juan Luis Cano Rodríguez and lamberthub by Jorge Martinez
"""

from typing import Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax import config
from jax import debug as jax_debug
from jax import jit, lax, vmap

from ..constants import Constants as C

config.update("jax_enable_x64", True)

MU = C.MU


@jit
def _hyp2f1b(x):
    """Hypergeometric function 2F1(3, 1, 5/2, x), implemented with JAX."""

    # Using lax.while_loop for JAX compatibility
    def cond_fun(state):
        _, _, _, _, res, res_old = state
        return res != res_old

    def body_fun(state):
        x, ii, term, res, _, _ = state

        # Update term and result
        term = term * (3 + ii) * (1 + ii) / (5 / 2 + ii) * x / (ii + 1)
        res_old = res
        res += term

        return x, ii + 1, term, res, res, res_old

    # Initialize state
    init_state = (x, 0.0, 1.0, 1.0, 1.0, 0.0)

    # Run the loop
    x, _, _, res, _, _ = lax.while_loop(cond_fun, body_fun, init_state)

    # Return infinity for x >= 1
    return lax.cond(x >= 1.0, lambda _: jnp.inf, lambda _: res, None)


@jit
def _compute_y(x, ll):
    """Computes y."""
    return jnp.sqrt(1 - ll**2 * (1 - x**2))


@jit
def _compute_psi(x, y, ll):
    """Computes psi."""
    # Use lax.cond instead of if-else for JAX compatibility
    return lax.cond(
        x < 1.0,
        lambda _: jnp.arccos(x * y + ll * (1 - x**2)),  # Elliptic
        lambda _: lax.cond(
            x > 1.0,
            lambda _: jnp.arcsinh((y - x * ll) * jnp.sqrt(x**2 - 1)),  # Hyperbolic
            lambda _: 0.0,  # Parabolic (x == 1)
            None,
        ),
        None,
    )


@jit
def _tof_equation_y(x, y, T0, ll, M):
    """Time of flight equation with externally computed y."""

    # Special case for small number of revolutions and specific x range
    def small_M_case():
        eta = y - ll * x
        S_1 = (1 - ll - x * eta) * 0.5
        Q = 4 / 3 * _hyp2f1b(S_1)
        return (eta**3 * Q + 4 * ll * eta) * 0.5

    # General case
    def general_case():
        psi = _compute_psi(x, y, ll)
        return jnp.divide(
            jnp.divide(psi + M * jnp.pi, jnp.sqrt(jnp.abs(1 - x**2))) - x + ll * y,
            (1 - x**2),
        )

    # Use lax.cond for conditional execution
    T_ = lax.cond(
        (M == 0) & (jnp.sqrt(0.6) < x) & (x < jnp.sqrt(1.4)),
        lambda _: small_M_case(),
        lambda _: general_case(),
        None,
    )

    return T_ - T0


@jit
def _tof_equation(x, T0, ll, M):
    """Time of flight equation."""
    y = _compute_y(x, ll)
    return _tof_equation_y(x, y, T0, ll, M)


@jit
def _tof_equation_p(x, y, T, ll):
    """First derivative of the time of flight equation."""
    return (3 * T * x - 2 + 2 * ll**3 * x / y) / (1 - x**2)


@jit
def _tof_equation_p2(x, y, T, dT, ll):
    """Second derivative of the time of flight equation."""
    return (3 * T + 5 * x * dT + 2 * (1 - ll**2) * ll**3 / y**3) / (1 - x**2)


@jit
def _tof_equation_p3(x, y, _, dT, ddT, ll):
    """Third derivative of the time of flight equation."""
    return (7 * x * ddT + 8 * dT - 6 * (1 - ll**2) * ll**5 * x / y**5) / (1 - x**2)


@jit
def _initial_guess(T, ll, M, low_path):
    """Initial guess for the iterative algorithm."""

    # Single revolution case
    def single_rev():
        T_0 = jnp.arccos(ll) + ll * jnp.sqrt(1 - ll**2) + M * jnp.pi  # Equation 19
        T_1 = 2 * (1 - ll**3) / 3  # Equation 21

        # Determine initial guess based on T
        x_T0 = (T_0 / T) ** (2 / 3) - 1
        x_T1 = 5 / 2 * T_1 / T * (T_1 - T) / (1 - ll**5) + 1
        x_middle = jnp.exp(jnp.log(2) * jnp.log(T / T_0) / jnp.log(T_1 / T_0)) - 1

        x_T0_case = T >= T_0
        x_T1_case = T < T_1

        return lax.cond(
            x_T0_case,
            lambda _: x_T0,
            lambda _: lax.cond(x_T1_case, lambda _: x_T1, lambda _: x_middle, None),
            None,
        )

    # Multiple revolution case
    def multi_rev():
        x_0l = (((M * jnp.pi + jnp.pi) / (8 * T)) ** (2 / 3) - 1) / (
            ((M * jnp.pi + jnp.pi) / (8 * T)) ** (2 / 3) + 1
        )
        x_0r = (((8 * T) / (M * jnp.pi)) ** (2 / 3) - 1) / (
            ((8 * T) / (M * jnp.pi)) ** (2 / 3) + 1
        )

        # Choose high or low path
        return lax.cond(
            low_path,
            lambda _: jnp.maximum(x_0l, x_0r),
            lambda _: jnp.minimum(x_0l, x_0r),
            None,
        )

    # Branch based on the number of revolutions
    return lax.cond(M == 0, lambda _: single_rev(), lambda _: multi_rev(), None)


@jit
def _householder(p0, T0, ll, M, atol, rtol, maxiter):
    """Find a zero of time of flight equation using the Householder method."""

    def body_fun(state):
        p0, iter_count = state

        # Compute values needed for the Householder step
        y = _compute_y(p0, ll)
        fval = _tof_equation_y(p0, y, T0, ll, M)
        T = fval + T0
        fder = _tof_equation_p(p0, y, T, ll)
        fder2 = _tof_equation_p2(p0, y, T, fder, ll)
        fder3 = _tof_equation_p3(p0, y, T, fder, fder2, ll)

        # Householder step (quartic)
        numerator = fder**2 - fval * fder2 / 2
        denominator = fder * (fder**2 - fval * fder2) + fder3 * fval**2 / 6
        # Avoid division by zero
        safe_denominator = jnp.where(
            jnp.abs(denominator) < 1e-15, 1e-15 * jnp.sign(denominator), denominator
        )
        p = p0 - fval * (numerator / safe_denominator)

        # Debug output
        jax_debug.print(
            "Householder iteration {}: p0 = {}, p = {}, delta = {}",
            iter_count,
            p0,
            p,
            jnp.abs(p - p0),
        )

        return (p, iter_count + 1)

    def cond_fun(state):
        p0, iter_count = state
        y = _compute_y(p0, ll)
        fval = _tof_equation_y(p0, y, T0, ll, M)

        # Continue if not converged and not exceeded max iterations
        delta_p_small = jnp.abs(fval) < atol
        iter_remaining = iter_count < maxiter

        return (~delta_p_small) & iter_remaining

    # Run the Householder iterations
    (p, _) = lax.while_loop(cond_fun, body_fun, (p0, 0))

    return p


@jit
def izzo_lambert(
    r1: jnp.ndarray,
    r2: jnp.ndarray,
    tof: float,
    mu: float = MU,
    M: int = 0,
    prograde: bool = True,
    low_path: bool = True,
    maxiter: int = 35,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Solves Lambert problem using Izzo's devised algorithm.

    Parameters
    ----------
    r1: jnp.ndarray
        Initial position vector.
    r2: jnp.ndarray
        Final position vector.
    tof: float
        Time of flight.
    mu: float
        Gravitational parameter, equivalent to GM of attractor body.
    M: int
        Number of revolutions. Must be equal or greater than 0.
    prograde: bool
        If True, specifies prograde motion. Otherwise, retrograde motion is imposed.
    low_path: bool
        If two solutions are available, it selects between high or low path.
    maxiter: int
        Maximum number of iterations.
    atol: float
        Absolute tolerance.
    rtol: float
        Relative tolerance.

    Returns
    -------
    v1: jnp.ndarray
        Initial velocity vector.
    v2: jnp.ndarray
        Final velocity vector.
    """
    # Compute basic geometric quantities
    c = r2 - r1  # Chord
    c_norm = jnp.linalg.norm(c)
    r1_norm = jnp.linalg.norm(r1)
    r2_norm = jnp.linalg.norm(r2)

    # Semiperimeter
    s = (r1_norm + r2_norm + c_norm) * 0.5

    # Normalized vectors
    i_r1 = r1 / r1_norm
    i_r2 = r2 / r2_norm

    # Compute angular momentum unit vector
    i_h = jnp.cross(i_r1, i_r2)
    i_h_norm = jnp.linalg.norm(i_h)
    i_h = i_h / jnp.where(i_h_norm > 0, i_h_norm, 1.0)  # Avoid division by zero

    # Geometry of the problem: lambda parameter
    ll = jnp.sqrt(1 - jnp.minimum(1.0, c_norm / s))

    # Adjust lambda and compute transfer direction based on orbit inclination
    ll_sign = jnp.where(i_h[2] < 0, -1.0, 1.0)
    ll = ll * ll_sign

    # Compute tangential directions
    i_t1 = jnp.where(i_h[2] < 0, jnp.cross(i_r1, i_h), jnp.cross(i_h, i_r1))

    i_t2 = jnp.where(i_h[2] < 0, jnp.cross(i_r2, i_h), jnp.cross(i_h, i_r2))

    # Account for retrograde motion
    ll = jnp.where(~prograde, -ll, ll)
    i_t1 = jnp.where(~prograde, -i_t1, i_t1)
    i_t2 = jnp.where(~prograde, -i_t2, i_t2)

    # Non-dimensional time of flight
    T = jnp.sqrt(2 * mu / s**3) * tof

    # Find x using Householder iterations
    x_0 = _initial_guess(T, ll, M, low_path)
    jax_debug.print("Initial guess x_0 = {}", x_0)

    x = _householder(x_0, T, ll, M, atol, rtol, maxiter)
    jax_debug.print("Final x = {}", x)

    # Compute y from converged x
    y = _compute_y(x, ll)
    jax_debug.print("Final y = {}", y)

    # Reconstruct the solution
    gamma = jnp.sqrt(mu * s / 2)
    rho = (r1_norm - r2_norm) / c_norm
    sigma = jnp.sqrt(1 - rho**2)

    # Compute velocity components
    V_r1 = gamma * ((ll * y - x) - rho * (ll * y + x)) / r1_norm
    V_r2 = -gamma * ((ll * y - x) + rho * (ll * y + x)) / r2_norm
    V_t1 = gamma * sigma * (y + ll * x) / r1_norm
    V_t2 = gamma * sigma * (y + ll * x) / r2_norm

    # Construct velocity vectors
    v1 = V_r1 * i_r1 + V_t1 * i_t1
    v2 = V_r2 * i_r2 + V_t2 * i_t2

    return v1, v2


# Vectorize the Lambert solver
_izzo_lambert_vmap = jit(
    vmap(
        izzo_lambert,
        in_axes=(0, 0, 0, None, None, None, None, None, None, None),
        out_axes=(0, 0),
    )
)


def solve_lambert(
    r1: Union[np.ndarray, jnp.ndarray],
    r2: Union[np.ndarray, jnp.ndarray],
    tof: Union[np.ndarray, float],
    mu: float = MU,
    prograde: bool = True,
    max_iter: int = 35,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve Lambert's problem for multiple initial and final positions and times of flight.

    This implementation uses Izzo's method which is robust and handles all orbit types.

    Parameters
    ----------
    r1 : array_like (N, 3)
        Initial position vectors in au.
    r2 : array_like (N, 3)
        Final position vectors in au.
    tof : array_like (N) or float
        Times of flight in days.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of au³/day².
    prograde : bool, optional
        If True, assume prograde motion. If False, assume retrograde motion.
    max_iter : int, optional
        Maximum number of iterations for convergence.
    tol : float, optional
        Convergence tolerance.

    Returns
    -------
    v1 : ndarray (N, 3)
        Initial velocity vectors in au/day.
    v2 : ndarray (N, 3)
        Final velocity vectors in au/day.
    """
    # Convert inputs to jnp arrays
    r1 = jnp.asarray(r1)
    r2 = jnp.asarray(r2)

    # Handle scalar inputs
    if r1.ndim == 1:
        r1 = r1.reshape(1, -1)
    if r2.ndim == 1:
        r2 = r2.reshape(1, -1)

    # Convert tof to array
    if isinstance(tof, (int, float)):
        tof = jnp.full(r1.shape[0], tof)
    else:
        tof = jnp.asarray(tof)

    # Call vectorized solver (M=0 for single-revolution case)
    v1, v2 = _izzo_lambert_vmap(r1, r2, tof, mu, 0, prograde, True, max_iter, tol, tol)

    # Convert to numpy arrays
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)

    # Return scalar outputs if inputs were scalar
    if len(v1) == 1:
        v1 = v1[0]
        v2 = v2[0]

    return v1, v2
