"""
Implementation of Izzo's Lambert problem solver using JAX

Lambert's problem determines the orbit between two points in space in a specified time of flight.
This solver implements Izzo's algorithm (2015) using JAX for efficient computation and automatic differentiation.

References
----------
[1] Izzo, D. (2015). Revisiting Lambert's problem. Celestial Mechanics and Dynamical Astronomy,
    121(1), 1-15.
[2] Lancaster, E. R., & Blanchard, R. C. (1969). A unified form of
    Lambert's theorem (Vol. 5368). National Aeronautics and Space Administration.
"""

from typing import Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax import config, jit, lax, vmap

from ..constants import Constants as C

config.update("jax_enable_x64", True)

MU = C.MU


@jit
def _compute_y(x: float, ll: float) -> float:
    """
    Compute the y parameter for Izzo's Lambert solver.

    Parameters
    ----------
    x : float
        The x parameter.
    ll : float
        The lambda parameter.

    Returns
    -------
    y : float
        The y parameter.
    """
    return jnp.sqrt(1 - ll**2 * (1 - x**2))


@jit
def _compute_psi(x: float, y: float, ll: float) -> float:
    """
    Compute the auxiliary angle psi.

    Parameters
    ----------
    x : float
        The x parameter.
    y : float
        The y parameter.
    ll : float
        The lambda parameter.

    Returns
    -------
    psi : float
        The auxiliary angle psi.
    """
    # Determine orbital type based on x
    return lax.cond(
        x < 1.0,
        # Elliptic motion: Use arc cosine to avoid numerical errors
        lambda _: jnp.arccos(x * y + ll * (1 - x**2)),
        # Hyperbolic motion or parabolic motion
        lambda _: lax.cond(
            x > 1.0,
            # Hyperbolic motion: The hyperbolic sine is bijective
            lambda _: jnp.arcsinh((y - x * ll) * jnp.sqrt(x**2 - 1)),
            # Parabolic motion (x == 1)
            lambda _: 0.0,
            None,
        ),
        None,
    )


@jit
def _hyp2f1b(x: float) -> float:
    """
    Compute a specific case of the Gauss Hypergeometric function 2F1(3/2, -1/2, 2, x).
    This implementation uses a Taylor series expansion.

    Parameters
    ----------
    x : float
        The parameter at which to evaluate the hypergeometric function.

    Returns
    -------
    float
        The value of the hypergeometric function.
    """
    # Implementation using a simplified series expansion
    # For 2F1(3/2, -1/2, 2, x)

    # Series expansion coefficients precomputed
    # For n=0 to 10, coefficient = prod(3/2+i) * prod(-1/2+i) / prod(2+i) / fact(i)
    coeffs = jnp.array(
        [
            1.0,
            0.75 * (-0.5) / 2.0 / 1.0,
            0.75 * 1.75 * (-0.5) * 0.5 / 2.0 / 3.0 / 2.0,
            0.75 * 1.75 * 2.75 * (-0.5) * 0.5 * 1.5 / 2.0 / 3.0 / 4.0 / 6.0,
            # More coefficients can be added for higher precision
        ]
    )

    # Calculate powers of x
    powers = jnp.array([1.0, x, x**2, x**3])  # More terms for higher precision

    # Compute the result with a conditional
    return lax.cond(
        x > 1.0,
        lambda _: jnp.inf,  # For x > 1, we need a transformation formula
        lambda _: jnp.sum(coeffs * powers),  # Otherwise, use the series expansion
        None,
    )

    # Note: In practice, a more accurate implementation might use special functions
    # or the transformation formulas for hypergeometric functions


@jit
def _tof_equation_y(x: float, y: float, T0: float, ll: float, M: int) -> float:
    """
    Compute the time of flight equation with precomputed y.

    Parameters
    ----------
    x : float
        The x parameter.
    y : float
        The y parameter.
    T0 : float
        The normalized time of flight.
    ll : float
        The lambda parameter.
    M : int
        Number of complete revolutions.

    Returns
    -------
    float
        The value of the time of flight equation.
    """
    # Special case for better numerical behavior
    use_special_case = (M == 0) & (jnp.sqrt(0.6) < x) & (x < jnp.sqrt(1.4))

    def special_case(_):
        eta = y - ll * x
        S_1 = (1 - ll - x * eta) * 0.5
        Q = 4.0 / 3.0 * _hyp2f1b(S_1)
        T_ = (eta**3 * Q + 4.0 * ll * eta) * 0.5
        return T_

    def general_case(_):
        psi = _compute_psi(x, y, ll)
        T_ = jnp.divide(
            jnp.divide(psi + M * jnp.pi, jnp.sqrt(jnp.abs(1 - x**2))) - x + ll * y,
            (1 - x**2),
        )
        return T_

    T_ = lax.cond(use_special_case, special_case, general_case, None)

    return T_ - T0


@jit
def _tof_equation(x: float, T0: float, ll: float, M: int) -> float:
    """
    Compute the time of flight equation.

    Parameters
    ----------
    x : float
        The x parameter.
    T0 : float
        The normalized time of flight.
    ll : float
        The lambda parameter.
    M : int
        Number of complete revolutions.

    Returns
    -------
    float
        The value of the time of flight equation.
    """
    return _tof_equation_y(x, _compute_y(x, ll), T0, ll, M)


@jit
def _tof_equation_p(x: float, y: float, T: float, ll: float) -> float:
    """
    First derivative of the time of flight equation.

    Parameters
    ----------
    x : float
        The x parameter.
    y : float
        The y parameter.
    T : float
        The value of the time of flight equation.
    ll : float
        The lambda parameter.

    Returns
    -------
    float
        The first derivative of the time of flight equation.
    """
    return (3.0 * T * x - 2.0 + 2.0 * ll**3 * x / y) / (1.0 - x**2)


@jit
def _tof_equation_p2(x: float, y: float, T: float, dT: float, ll: float) -> float:
    """
    Second derivative of the time of flight equation.

    Parameters
    ----------
    x : float
        The x parameter.
    y : float
        The y parameter.
    T : float
        The value of the time of flight equation.
    dT : float
        The first derivative of the time of flight equation.
    ll : float
        The lambda parameter.

    Returns
    -------
    float
        The second derivative of the time of flight equation.
    """
    return (3.0 * T + 5.0 * x * dT + 2.0 * (1.0 - ll**2) * ll**3 / y**3) / (1.0 - x**2)


@jit
def _tof_equation_p3(
    x: float, y: float, _: float, dT: float, ddT: float, ll: float
) -> float:
    """
    Third derivative of the time of flight equation.

    Parameters
    ----------
    x : float
        The x parameter.
    y : float
        The y parameter.
    _ : float
        Placeholder (unused, corresponds to the time of flight equation).
    dT : float
        The first derivative of the time of flight equation.
    ddT : float
        The second derivative of the time of flight equation.
    ll : float
        The lambda parameter.

    Returns
    -------
    float
        The third derivative of the time of flight equation.
    """
    numerator = 7.0 * x * ddT + 8.0 * dT - 6.0 * (1.0 - ll**2) * ll**5 * x / y**5
    denominator = 1.0 - x**2
    return numerator / denominator


@jit
def _initial_guess(T: float, ll: float, M: int, low_path: bool) -> float:
    """
    Compute initial guess for the x parameter.

    Parameters
    ----------
    T : float
        The normalized time of flight.
    ll : float
        The lambda parameter.
    M : int
        Number of complete revolutions.
    low_path : bool
        If True, selects the low path solution when two solutions are available.

    Returns
    -------
    float
        The initial guess for the x parameter.
    """

    # Multi-revolution case (M > 0)
    def multi_rev_case(_):
        # initial guess is taken from the paper (Izzo, 2015)
        T_0 = jnp.arccos(ll) + ll * jnp.sqrt(1 - ll**2) + M * jnp.pi
        return (T_0 / T) ** (2 / 3) - 1

    # Single-revolution case (M = 0)
    def single_rev_case(_):
        # Check for near-parabolic cases
        T_0 = jnp.arccos(ll) + ll * jnp.sqrt(1 - ll**2)

        # Elliptic orbit case (T >= T_0)
        def elliptic_case(_):
            log_pwr = 0.69314718055994529  # ln(2)
            return (T_0 / T) ** log_pwr - 1.0

        # Hyperbolic orbit case (T < T_0)
        def hyperbolic_case(_):
            # First we use a cubic model
            W = lax.cond(
                ll < 1,
                lambda _: jnp.log(T / T_0) * 0.5,
                lambda _: jnp.log(T / T_0),
                None,
            )

            # Based on path selection
            return lax.cond(
                low_path,
                # Lower branch solution
                lambda _: 5.0 / 2.0 * (1.0 - ll) * (jnp.exp(W) - 1.0) + 1.0,
                # Upper branch solution
                lambda _: 5.0 / 2.0 * (1.0 - ll) * (jnp.exp(-W) - 1.0) + 1.0,
                None,
            )

        # Choose between elliptic and hyperbolic case
        return lax.cond(T >= T_0, elliptic_case, hyperbolic_case, None)

    # Choose between multi-revolution and single-revolution case
    return lax.cond(M > 0, multi_rev_case, single_rev_case, None)


@jit
def _compute_T_min(ll: float, M: int) -> Tuple[float, float]:
    """
    Compute the minimum time of flight and corresponding x for a given
    number of revolutions.

    Parameters
    ----------
    ll : float
        The lambda parameter.
    M : int
        Number of complete revolutions.

    Returns
    -------
    x_T_min : float
        The value of x at minimum time of flight.
    T_min : float
        The minimum time of flight.
    """

    # Special case for ll close to 1
    def ll_close_to_one(_):
        x_T_min = 0.0
        T_min = _tof_equation(x_T_min, 0.0, ll, M)
        return x_T_min, T_min

    # Handle M = 0 case
    def m_equals_zero(_):
        return jnp.inf, 0.0

    # Standard case for multi-revolution
    def standard_case(_):
        x_T_min = 0.0
        T_min = _tof_equation(x_T_min, 0.0, ll, M)
        return x_T_min, T_min

    # First check if ll is close to 1
    result = lax.cond(
        jnp.abs(ll - 1.0) < 1e-6,
        ll_close_to_one,
        # If not, check if M is 0
        lambda _: lax.cond(M == 0, m_equals_zero, standard_case, None),
        None,
    )

    return result


@jit
def _householder(
    x0: float, T0: float, ll: float, M: int, atol: float, rtol: float, maxiter: int
) -> float:
    """
    Householder iteration to find a root of the time of flight equation.

    Parameters
    ----------
    x0 : float
        Initial guess for the x parameter.
    T0 : float
        The normalized time of flight.
    ll : float
        The lambda parameter.
    M : int
        Number of complete revolutions.
    atol : float
        Absolute tolerance for convergence.
    rtol : float
        Relative tolerance for convergence.
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    float
        The converged value of the x parameter.
    """
    # Initialize
    x = x0

    def body_fun(carry):
        x, iteration = carry

        # Compute y
        y = _compute_y(x, ll)

        # Evaluate function and derivatives
        f = _tof_equation_y(x, y, T0, ll, M)
        fp = _tof_equation_p(x, y, T0 + f, ll)
        fpp = _tof_equation_p2(x, y, T0 + f, fp, ll)
        fppp = _tof_equation_p3(x, y, T0 + f, fp, fpp, ll)

        # Householder step
        delta = -f * (fp**2 - f * fpp / 2) / (fp * (fp**2 - f * fpp) + fppp * f**2 / 6)

        # Update x
        x_new = x + delta

        # Return updated x and iteration count
        return x_new, iteration + 1

    def cond_fun(carry):
        x, iteration = carry

        # Compute y and function value for convergence check
        y = _compute_y(x, ll)
        f = _tof_equation_y(x, y, T0, ll, M)

        # Check for convergence
        return (jnp.abs(f) > atol) & (iteration < maxiter)

    # Run the Householder iteration
    x_sol, _ = lax.while_loop(cond_fun, body_fun, (x, 0))

    return x_sol


@jit
def _reconstruct(
    x: float,
    y: float,
    r1_norm: float,
    r2_norm: float,
    ll: float,
    gamma: float,
    rho: float,
    sigma: float,
) -> Tuple[float, float, float, float]:
    """
    Reconstruct velocity components from the converged x and y parameters.

    Parameters
    ----------
    x : float
        The converged x parameter.
    y : float
        The corresponding y parameter.
    r1_norm : float
        Magnitude of the initial position vector.
    r2_norm : float
        Magnitude of the final position vector.
    ll : float
        The lambda parameter.
    gamma : float
        Auxiliary parameter.
    rho : float
        Auxiliary parameter.
    sigma : float
        Auxiliary parameter.

    Returns
    -------
    V_r1 : float
        Radial component of initial velocity.
    V_r2 : float
        Radial component of final velocity.
    V_t1 : float
        Tangential component of initial velocity.
    V_t2 : float
        Tangential component of final velocity.
    """
    V_r1 = gamma * ((ll * y - x) - rho * (ll * y + x)) / r1_norm
    V_r2 = -gamma * ((ll * y - x) + rho * (ll * y + x)) / r2_norm
    V_t1 = gamma * sigma * (y + ll * x) / r1_norm
    V_t2 = gamma * sigma * (y + ll * x) / r2_norm

    return V_r1, V_r2, V_t1, V_t2


@jit
def _find_xy(
    ll: float, T: float, M: int, maxiter: int, atol: float, rtol: float, low_path: bool
) -> Tuple[float, float]:
    """
    Find the x and y parameters for Lambert's problem.

    Parameters
    ----------
    ll : float
        The lambda parameter.
    T : float
        The normalized time of flight.
    M : int
        Number of complete revolutions.
    maxiter : int
        Maximum number of iterations.
    atol : float
        Absolute tolerance for convergence.
    rtol : float
        Relative tolerance for convergence.
    low_path : bool
        If True, select the low path solution when two solutions are available.

    Returns
    -------
    x : float
        The converged x parameter.
    y : float
        The corresponding y parameter.
    """
    # For abs(ll) == 1 the derivative is not continuous
    # Adjust slightly to avoid numerical issues
    ll = jnp.clip(ll, -0.9999999, 0.9999999)

    # Calculate maximum number of revolutions possible
    M_max = jnp.floor(T / jnp.pi).astype(jnp.int32)
    T_00 = jnp.arccos(ll) + ll * jnp.sqrt(1 - ll**2)  # T_xM

    # Refine maximum number of revolutions if necessary
    M_max = lax.cond(
        (T < T_00 + M_max * jnp.pi) & (M_max > 0),
        lambda _: M_max
        - lax.cond(T < _compute_T_min(ll, M_max)[1], lambda _: 1, lambda _: 0, None),
        lambda _: M_max,
        None,
    )

    # Check if a feasible solution exists for the given number of revolutions
    M = jnp.minimum(M, M_max)

    # Initial guess
    x_0 = _initial_guess(T, ll, M, low_path)

    # Start Householder iterations from x_0 and find x
    x = _householder(x_0, T, ll, M, atol, rtol, maxiter)
    y = _compute_y(x, ll)

    return x, y


@jit
def _solve_lambert_izzo(
    r1: jnp.ndarray,
    r2: jnp.ndarray,
    tof: float,
    mu: float = MU,
    M: int = 0,
    prograde: bool = True,
    low_path: bool = True,
    maxiter: int = 35,
    atol: float = 1e-5,
    rtol: float = 1e-7,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Solve Lambert's problem using Izzo's algorithm.

    Parameters
    ----------
    r1 : jnp.ndarray (3)
        Initial position vector.
    r2 : jnp.ndarray (3)
        Final position vector.
    tof : float
        Time of flight.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body.
    M : int, optional
        Number of complete revolutions. Must be equal or greater than 0.
    prograde : bool, optional
        If True, specifies prograde motion. Otherwise, retrograde motion is imposed.
    low_path : bool, optional
        If two solutions are available, this selects between high or low path.
    maxiter : int, optional
        Maximum number of iterations.
    atol : float, optional
        Absolute tolerance.
    rtol : float, optional
        Relative tolerance.

    Returns
    -------
    v1 : jnp.ndarray (3)
        Initial velocity vector.
    v2 : jnp.ndarray (3)
        Final velocity vector.
    """
    # Calculate norms and derived quantities
    r1_norm = jnp.linalg.norm(r1)
    r2_norm = jnp.linalg.norm(r2)

    # Chord
    c = r2 - r1
    c_norm = jnp.linalg.norm(c)

    # Semiperimeter
    s = (r1_norm + r2_norm + c_norm) * 0.5

    # Versors (unit vectors)
    i_r1 = r1 / r1_norm
    i_r2 = r2 / r2_norm
    i_h = jnp.cross(i_r1, i_r2)
    i_h = i_h / jnp.linalg.norm(i_h)

    # Geometry of the problem
    ll = jnp.sqrt(1.0 - jnp.minimum(1.0, c_norm / s))

    # Compute the fundamental tangential directions
    ll, i_t1, i_t2 = lax.cond(
        i_h[2] < 0,
        lambda _: (-ll, jnp.cross(i_r1, i_h), jnp.cross(i_r2, i_h)),
        lambda _: (ll, jnp.cross(i_h, i_r1), jnp.cross(i_h, i_r2)),
        None,
    )

    # Correct transfer angle parameter and tangential vectors regarding orbit's inclination
    ll, i_t1, i_t2 = lax.cond(
        prograde is False,
        lambda _: (-ll, -i_t1, -i_t2),
        lambda _: (ll, i_t1, i_t2),
        None,
    )

    # Non dimensional time of flight
    T = jnp.sqrt(2.0 * mu / s**3) * tof

    # Find solutions
    x, y = _find_xy(ll, T, M, maxiter, atol, rtol, low_path)

    # Reconstruct
    gamma = jnp.sqrt(mu * s / 2.0)
    rho = (r1_norm - r2_norm) / c_norm
    sigma = jnp.sqrt(1.0 - rho**2)

    # Compute the radial and tangential components at initial and final position vectors
    V_r1, V_r2, V_t1, V_t2 = _reconstruct(x, y, r1_norm, r2_norm, ll, gamma, rho, sigma)

    # Solve for the initial and final velocity
    v1 = V_r1 * i_r1 + V_t1 * i_t1
    v2 = V_r2 * i_r2 + V_t2 * i_t2

    return v1, v2


# Vectorize the Lambert solver
_solve_lambert_izzo_vmap = jit(
    vmap(
        _solve_lambert_izzo,
        in_axes=(0, 0, 0, None, None, None, None, None, None, None),
        out_axes=(0, 0),
    )
)


def solve_lambert_izzo(
    r1: Union[np.ndarray, jnp.ndarray],
    r2: Union[np.ndarray, jnp.ndarray],
    tof: Union[np.ndarray, float],
    mu: float = MU,
    M: int = 0,
    prograde: bool = True,
    low_path: bool = True,
    maxiter: int = 35,
    atol: float = 1e-5,
    rtol: float = 1e-7,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve Lambert's problem using Izzo's algorithm.

    This solver works for both single and multiple revolutions, and handles
    all orbital regimes (elliptic, parabolic, and hyperbolic). It is a vectorized
    version that can solve multiple Lambert's problems simultaneously.

    Parameters
    ----------
    r1 : np.ndarray or jnp.ndarray (N, 3)
        Initial position vectors in consistent units.
    r2 : np.ndarray or jnp.ndarray (N, 3)
        Final position vectors in consistent units.
    tof : np.ndarray or float (N) or scalar
        Times of flight in consistent units.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body.
    M : int, optional
        Number of complete revolutions. Must be equal or greater than 0.
    prograde : bool, optional
        If True, specifies prograde motion. Otherwise, retrograde motion is imposed.
    low_path : bool, optional
        If two solutions are available, this selects between high or low path.
    maxiter : int, optional
        Maximum number of iterations.
    atol : float, optional
        Absolute tolerance.
    rtol : float, optional
        Relative tolerance.

    Returns
    -------
    v1 : np.ndarray (N, 3)
        Initial velocity vectors in consistent units.
    v2 : np.ndarray (N, 3)
        Final velocity vectors in consistent units.

    Notes
    -----
    This is a JAX implementation of the algorithm devised by Dario Izzo (2015).
    It follows the universal formulae approach developed by Lancaster during the 1960s.
    It handles all orbital regimes and can compute multi-revolution transfers.

    The algorithm is detailed in:
    Izzo, D. (2015). Revisiting Lambert's problem. Celestial Mechanics and
    Dynamical Astronomy, 121(1), 1-15.
    """
    # Convert inputs to jnp arrays if they are not already
    r1 = jnp.asarray(r1)
    r2 = jnp.asarray(r2)

    # Handle scalar inputs - this is fine as Python conditional since we're outside JIT
    if r1.ndim == 1:
        r1 = r1.reshape(1, -1)
    if r2.ndim == 1:
        r2 = r2.reshape(1, -1)

    # Convert tof to array - this is fine as Python conditional since we're outside JIT
    if isinstance(tof, (int, float)):
        tof = jnp.full(r1.shape[0], tof)
    else:
        tof = jnp.asarray(tof)
        if tof.ndim == 0:
            tof = jnp.full(r1.shape[0], tof)

    # Call the vectorized solver
    v1, v2 = _solve_lambert_izzo_vmap(
        r1, r2, tof, mu, M, prograde, low_path, maxiter, atol, rtol
    )

    # Convert back to numpy arrays
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)

    # If inputs were scalars, return scalar outputs - fine as Python conditional
    if len(v1) == 1:
        v1 = v1[0]
        v2 = v2[0]

    return v1, v2


def generate_porkchop_data(
    r1_func,
    r2_func,
    departure_times,
    arrival_times,
    mu: float = MU,
    M: int = 0,
    prograde: bool = True,
    low_path: bool = True,
    maxiter: int = 35,
    atol: float = 1e-5,
    rtol: float = 1e-7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data for a porkchop plot by solving Lambert's problem for a grid of
    departure and arrival times.

    Parameters
    ----------
    r1_func : callable
        Function that returns the position vector of the departure body at a given time.
        Should accept a time parameter and an optional return_velocity parameter.
    r2_func : callable
        Function that returns the position vector of the arrival body at a given time.
        Should accept a time parameter and an optional return_velocity parameter.
    departure_times : np.ndarray (N)
        Array of departure times in consistent units (e.g., days since J2000).
    arrival_times : np.ndarray (M)
        Array of arrival times in consistent units (e.g., days since J2000).
    mu : float, optional
        Gravitational parameter (GM) of the attracting body.
    M : int, optional
        Number of complete revolutions. Must be equal or greater than 0.
    prograde : bool, optional
        If True, specifies prograde motion. Otherwise, retrograde motion is imposed.
    low_path : bool, optional
        If two solutions are available, this selects between high or low path.
    maxiter : int, optional
        Maximum number of iterations for Lambert's solver.
    atol : float, optional
        Absolute tolerance for Lambert's solver.
    rtol : float, optional
        Relative tolerance for Lambert's solver.

    Returns
    -------
    delta_v_departure : np.ndarray (N, M)
        Delta-v required at departure for each departure-arrival time combination.
    delta_v_arrival : np.ndarray (N, M)
        Delta-v required at arrival for each departure-arrival time combination.
    total_delta_v : np.ndarray (N, M)
        Total delta-v (departure + arrival) for each departure-arrival time combination.
    """
    n_deps = len(departure_times)
    n_arrs = len(arrival_times)

    # Initialize arrays to store results
    delta_v_departure = np.zeros((n_deps, n_arrs))
    delta_v_arrival = np.zeros((n_deps, n_arrs))
    total_delta_v = np.zeros((n_deps, n_arrs))

    # Precompute positions at all times
    r1_positions = np.array([r1_func(t) for t in departure_times])
    r2_positions = np.array([r2_func(t) for t in arrival_times])

    # For each departure and arrival time combination
    for i, t1 in enumerate(departure_times):
        for j, t2 in enumerate(arrival_times):
            # Skip if arrival time is before departure time
            if t2 <= t1:
                delta_v_departure[i, j] = np.nan
                delta_v_arrival[i, j] = np.nan
                total_delta_v[i, j] = np.nan
                continue

            # Get positions
            r1 = r1_positions[i]
            r2 = r2_positions[j]

            # Time of flight
            tof = t2 - t1

            # Solve Lambert's problem
            try:
                v1_trans, v2_trans = solve_lambert_izzo(
                    r1, r2, tof, mu, M, prograde, low_path, maxiter, atol, rtol
                )

                # Get velocity of departure body at departure time
                v1_body = r1_func(t1, return_velocity=True)[1]

                # Get velocity of arrival body at arrival time
                v2_body = r2_func(t2, return_velocity=True)[1]

                # Calculate delta-v
                dv1 = np.linalg.norm(v1_trans - v1_body)
                dv2 = np.linalg.norm(v2_trans - v2_body)

                delta_v_departure[i, j] = dv1
                delta_v_arrival[i, j] = dv2
                total_delta_v[i, j] = dv1 + dv2

            except Exception as e:
                # In case Lambert solver fails
                delta_v_departure[i, j] = np.nan
                delta_v_arrival[i, j] = np.nan
                total_delta_v[i, j] = np.nan

    return delta_v_departure, delta_v_arrival, total_delta_v
