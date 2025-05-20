"""
Implementation of Lambert's problem using Izzo's method.

This implementation follows the algorithm described in:
Izzo, D. (2015). Revisiting Lambert's problem. Celestial Mechanics and Dynamical Astronomy, 121(1), 1-15.

Credits: Based on poliastro implementation by Juan Luis Cano Rodríguez and lamberthub by Jorge Martinez
"""

from typing import Tuple, Union

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import config, jit, lax, vmap

from ..constants import Constants as C

config.update("jax_enable_x64", True)

MU = C.MU


@jit
def _hyp2f1b(x):
    """Hypergeometric function 2F1(3, 1, 5/2, x), implemented with JAX."""
    # Set a finite number of iterations instead of convergence check
    MAX_ITER = 100

    # Using lax.scan for JAX compatibility
    def body_fun(state, i):
        x, term, res = state

        # Update term and result
        term = term * (3 + i) * (1 + i) / (5 / 2 + i) * x / (i + 1)
        res += term

        return (x, term, res), None

    # Initialize state
    init_state = (
        x.astype(jnp.float64),
        jnp.ones_like(x, dtype=jnp.float64),
        jnp.ones_like(x, dtype=jnp.float64),
    )

    # Run the loop with fixed iterations
    (_, _, res), _ = lax.scan(
        body_fun, init_state, jnp.arange(MAX_ITER, dtype=jnp.float64)
    )

    # Set to infinity for x >= 1
    return jnp.where(x >= 1.0, jnp.inf, res)


@jit
def _compute_y(x, ll):
    """Computes y."""
    return jnp.sqrt(1 - ll**2 * (1 - x**2))


@jit
def _compute_psi(x, y, ll):
    """Computes psi."""
    # Compute the argument for arccos
    arccos_arg = x * y + ll * (1 - x**2)

    # Elliptic case (x < 1.0)
    elliptic = jnp.arccos(jnp.clip(arccos_arg, -1.0, 1.0))

    # Hyperbolic case (x > 1.0)
    hyperbolic = jnp.arcsinh((y - x * ll) * jnp.sqrt(x**2 - 1))

    # Parabolic case (x == 1.0)
    parabolic = jnp.zeros_like(x)

    # Use where statements instead of conditionals
    result = jnp.where(x < 1.0, elliptic, jnp.where(x > 1.0, hyperbolic, parabolic))

    return result


@jit
def _tof_equation_y(x, y, T0, ll, M):
    """Time of flight equation with externally computed y."""

    # Special case for small number of revolutions and specific x range
    # Calculate values for small M case
    eta = y - ll * x
    S_1 = (1 - ll - x * eta) * 0.5
    Q = 4 / 3 * _hyp2f1b(S_1)
    small_M_result = (eta**3 * Q + 4 * ll * eta) * 0.5

    # Calculate values for general case
    psi = _compute_psi(x, y, ll)
    sqrt_term = jnp.sqrt(jnp.abs(1 - x**2))
    numerator = jnp.divide(psi + M * jnp.pi, sqrt_term) - x + ll * y
    denominator = 1 - x**2
    general_result = jnp.divide(numerator, denominator)

    # Use where for conditional selection
    use_small_M = (M == 0) & (jnp.sqrt(0.6) < x) & (x < jnp.sqrt(1.4))
    T_ = jnp.where(use_small_M, small_M_result, general_result)

    result = T_ - T0
    return result


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

    # Single revolution case (M == 0)
    T_0 = jnp.arccos(ll) + ll * jnp.sqrt(1 - ll**2) + M * jnp.pi  # Equation 19
    T_1 = 2 * (1 - ll**3) / 3  # Equation 21

    # Determine initial guess based on T
    x_T0 = (T_0 / T) ** (2 / 3) - 1
    x_T1 = 5 / 2 * T_1 / T * (T_1 - T) / (1 - ll**5) + 1

    # For the middle case, use logarithmic interpolation
    x_middle = jnp.exp(jnp.log(2) * jnp.log(T / T_0) / jnp.log(T_1 / T_0)) - 1

    # Create masks for the cases
    case_T_ge_T0 = T >= T_0
    case_T_lt_T1 = T < T_1

    # Select appropriate single-revolution case
    x_single_rev = jnp.where(
        case_T_ge_T0, x_T0, jnp.where(case_T_lt_T1, x_T1, x_middle)
    )

    # Multiple revolution case
    x_0l = (((M * jnp.pi + jnp.pi) / (8 * T)) ** (2 / 3) - 1) / (
        ((M * jnp.pi + jnp.pi) / (8 * T)) ** (2 / 3) + 1
    )
    x_0r = (((8 * T) / (M * jnp.pi)) ** (2 / 3) - 1) / (
        ((8 * T) / (M * jnp.pi)) ** (2 / 3) + 1
    )

    # Choose high or low path for multi-rev
    x_multi_rev = jnp.where(low_path, jnp.maximum(x_0l, x_0r), jnp.minimum(x_0l, x_0r))

    # Final selection based on M
    result = jnp.where(M == 0, x_single_rev, x_multi_rev)

    return result


@jit
def _householder(p0, T0, ll, M, atol, rtol, maxiter):
    """Find a zero of time of flight equation using the Householder method."""

    def body_fun(state):
        p0, p, iter_count = state

        # Update p0 to be the current p
        p0 = p

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

        # Avoid division by zero with a safer approach
        safe_denominator = jnp.where(
            jnp.abs(denominator) < 1e-15, jnp.sign(denominator) * 1e-15, denominator
        )

        # Compute the new value with safeguards against large steps
        delta = fval * (numerator / safe_denominator)
        # Limit step size to prevent divergence
        max_step = jnp.maximum(0.1, jnp.abs(p0))
        delta = jnp.clip(delta, -max_step, max_step)
        p = p0 - delta

        return (p0, p, iter_count + 1)

    def cond_fun(state):
        p0, p, iter_count = state

        # Check both max iterations and convergence criteria
        delta_x = jnp.abs(p - p0)
        converged = delta_x < (rtol * jnp.abs(p0) + atol)
        iter_remaining = iter_count < maxiter

        return (~converged) & iter_remaining

    # Initialize state with different p value to ensure at least one iteration runs
    # Set p to be slightly different from p0 to trigger at least one iteration
    init_p = (
        p0 * 1.1 + 0.01
    )  # Ensure it's different enough to avoid immediate convergence
    init_state = (p0, init_p, 0)

    # Run the Householder iterations
    _, p, iteration_count = lax.while_loop(cond_fun, body_fun, init_state)

    return p


@jit
def _compute_T_min(ll, M, maxiter, atol, rtol):
    """Compute minimum T."""

    # Case 1: ll == 1
    x_T_min_case1 = 0.0
    T_min_case1 = _tof_equation(x_T_min_case1, 0.0, ll, M)

    # Case 2: ll != 1 and M == 0
    x_T_min_case2 = jnp.inf
    T_min_case2 = 0.0

    # Case 3: ll != 1 and M != 0
    # Set x_i > 0 to avoid problems at ll = -1
    x_i = 0.1
    T_i = _tof_equation(x_i, 0.0, ll, M)
    x_T_min_case3 = _halley(x_i, T_i, ll, atol, rtol, maxiter)
    T_min_case3 = _tof_equation(x_T_min_case3, 0.0, ll, M)

    # Select the appropriate case using masks
    is_ll_one = jnp.abs(ll - 1.0) < 1e-10
    is_m_zero = M == 0

    # First choose between case 2 and case 3 based on M
    x_T_min_case23 = jnp.where(is_m_zero, x_T_min_case2, x_T_min_case3)
    T_min_case23 = jnp.where(is_m_zero, T_min_case2, T_min_case3)

    # Then choose between case 1 and the result of case23 based on ll
    x_T_min = jnp.where(is_ll_one, x_T_min_case1, x_T_min_case23)
    T_min = jnp.where(is_ll_one, T_min_case1, T_min_case23)

    return x_T_min, T_min


@jit
def _halley(p0, T0, ll, atol, rtol, maxiter):
    """Find a minimum of time of flight equation using the Halley method."""

    def body_fun(state):
        p0, iter_count = state

        y = _compute_y(p0, ll)
        fder = _tof_equation_p(p0, y, T0, ll)
        fder2 = _tof_equation_p2(p0, y, T0, fder, ll)
        fder3 = _tof_equation_p3(p0, y, T0, fder, fder2, ll)

        # Halley step (cubic)
        p = p0 - 2 * fder * fder2 / (2 * fder2**2 - fder * fder3)

        return (p, iter_count + 1)

    def cond_fun(state):
        p0, iter_count = state
        p, _ = body_fun(state)
        return (jnp.abs(p - p0) >= rtol * jnp.abs(p0) + atol) & (iter_count < maxiter)

    # Run the Halley iterations
    (p, _) = lax.while_loop(cond_fun, body_fun, (p0, 0))
    return p


@jit
def _find_xy(ll, T, M, maxiter, atol, rtol, low_path):
    """Computes all x, y for given number of revolutions."""

    # Calculate M_max
    M_max = jnp.floor(T / jnp.pi)
    T_00 = jnp.arccos(jnp.abs(ll)) + jnp.abs(ll) * jnp.sqrt(1 - ll**2)

    # Possibly refine M_max using JAX-native operations
    need_refine = (T < T_00 + M_max * jnp.pi) & (M_max > 0)

    # Define the refine function
    def refine_fn(need_refine):
        _, T_min = _compute_T_min(ll, M_max, maxiter, atol, rtol)
        return jnp.where(T < T_min, M_max - 1, M_max)

    # Use lax.cond for this specific case as it's crucial
    M_max = lax.cond(
        need_refine, lambda _: refine_fn(need_refine), lambda _: M_max, None
    )

    # Check if solution exists - use explicit debug to verify condition
    valid_ll_value = jnp.abs(ll) < 1.0

    valid_M = M <= M_max

    # Get initial guess
    x_0 = _initial_guess(T, ll, M, low_path)

    # Run Householder iterations
    x = _householder(x_0, T, ll, M, atol, rtol, maxiter)
    y = _compute_y(x, ll)

    # Return NaN for invalid cases using jnp.where which is more reliable
    # Check validation condition explicitly again
    x_result = jnp.where(valid_ll_value & valid_M, x, jnp.nan)
    y_result = jnp.where(valid_ll_value & valid_M, y, jnp.nan)

    return x_result, y_result


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
    Solves Lambert's problem using Izzo's devised algorithm.

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

    # Find x, y using the new approach
    x, y = _find_xy(ll, T, M, maxiter, atol, rtol, low_path)
    # Perform explicit NaN check with JAX operations
    has_nan_x = jnp.isnan(x)
    has_nan_y = jnp.isnan(y)
    has_nans = has_nan_x | has_nan_y

    # Always compute all solution components
    # Reconstruct the solution - always compute these values
    gamma = jnp.sqrt(mu * s / 2)
    rho = (r1_norm - r2_norm) / c_norm
    sigma = jnp.sqrt(1 - rho**2)

    # Compute velocity components
    V_r1 = gamma * ((ll * y - x) - rho * (ll * y + x)) / r1_norm
    V_r2 = -gamma * ((ll * y - x) + rho * (ll * y + x)) / r2_norm
    V_t1 = gamma * sigma * (y + ll * x) / r1_norm
    V_t2 = gamma * sigma * (y + ll * x) / r2_norm

    # Construct velocity vectors
    v1_valid = V_r1 * i_r1 + V_t1 * i_t1
    v2_valid = V_r2 * i_r2 + V_t2 * i_t2

    # Create NaN vectors for invalid cases
    v1_nan = jnp.full_like(r1, jnp.nan)
    v2_nan = jnp.full_like(r2, jnp.nan)

    # Select the appropriate results using where
    v1 = jnp.where(has_nans, v1_nan, v1_valid)
    v2 = jnp.where(has_nans, v2_nan, v2_valid)

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
        Initial velocity vectors in au/day with origin at the attractor
    v2 : ndarray (N, 3)
        Final velocity vectors in au/day with origin at the attractor
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
    return v1, v2


@jit
def _calculate_c3(
    v1: Union[np.ndarray, jnp.ndarray], body_v: Union[np.ndarray, jnp.ndarray]
) -> jnp.ndarray:
    v_infinity = v1 - body_v

    # Use jnp.linalg.norm for JAX arrays to avoid TracerArrayConversionError
    c3 = jnp.linalg.norm(v_infinity, axis=1) ** 2
    return c3


def calculate_c3(
    v1: Union[np.ndarray, jnp.ndarray], body_v: Union[np.ndarray, jnp.ndarray]
) -> npt.NDArray[np.float64]:
    """
    Calculate the C3 of a spacecraft given its velocity relative to a body.

    Parameters
    ----------
    v1 : array_like (N, 3)
        Velocity of the spacecraft in au/d.
    body_v : array_like (N, 3)
        Velocity of the body in au/d.

    Returns
    -------
    c3 : array_like (N)
        C3 of the spacecraft in au^2/d^2.
    """

    c3 = _calculate_c3(v1, body_v)
    # Convert to numpy array before returning
    c3 = np.asarray(c3)
    return c3
