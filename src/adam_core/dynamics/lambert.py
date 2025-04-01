"""
Implementation of Lambert's problem

Lambert's problem determines the orbit between two points in space in a specified time of flight.
This solver uses the universal variables method to efficiently compute the solution.

References
----------
[1] Vallado, D. A. (2013). Fundamentals of Astrodynamics and Applications. 4th ed.,
    Microcosm Press. ISBN-13: 978-1881883180
[2] Izzo, D. (2015). Revisiting Lambert's problem. Celestial Mechanics and Dynamical Astronomy,
    121(1), 1-15.
"""
from typing import Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax import config
from jax import debug as jax_debug
from jax import jit, lax, vmap

from ..constants import Constants as C
from .stumpff import calc_stumpff  # Import from your existing module

config.update("jax_enable_x64", True)

MU = C.MU


@jit
def _lambert_iteration_step(
    x: float,
    ratio: float,
    iterations: int,
    a_min: float,
    beta: float,
    r1_mag: float,
    r2_mag: float,
    s: float,
    c: float,
    tof: float,
    mu: float,
    long_way: bool,
) -> Tuple[float, float, int]:
    """Single iteration step of Lambert solver."""
    # Compute the semi-major axis
    a = a_min / (1 - x**2 / 2)
    
    # Compute y based on x and a
    y = jnp.sqrt(r1_mag * r2_mag) * jnp.sin(beta / 2) / jnp.sqrt(a * (1 - jnp.cos(beta)))
    
    # Compute z = x^2/a
    z = x**2 / a
    
    # Calculate Stumpff functions using the module
    c0, c1, c2, c3, c4, c5 = calc_stumpff(z)
    
    # Compute time of flight for current x
    tof_x = jnp.sqrt(a**3 / mu) * (
        beta - x*y/jnp.sqrt(a) - 
        2*jnp.arcsin(jnp.sqrt(s/(2*a))) + 
        2*jnp.sqrt(a/s)*c2*z + 
        (2*(s - c)/jnp.sqrt(a))*c3*z
    )
    
    # If long-way transfer, adjust TOF calculation
    tof_x = jnp.where(long_way, -tof_x, tof_x)
    
    # Newton-Raphson update
    f = tof_x - tof
    df = jnp.sqrt(a**3 / mu) * (3*x*c3 - 2*x/a * c2 - y/jnp.sqrt(a))
    
    # Better approach with Halley's method 
    fpp = jnp.sqrt(a**3 / mu) * (3*c3 - 6*x*c4*z - 3*y/(2*a**(3/2)))
    denominator = 2*df**2 - f*fpp
    denominator = jnp.where(jnp.abs(denominator) < 1e-14, 1e-14 * jnp.sign(denominator), denominator)
    ratio = (2*f*df) / denominator
    x_new = x - ratio
    
    # Add damping to prevent oscillation
    x_new = jnp.where(
        jnp.abs(x_new - x) > 0.5,
        x + jnp.sign(x_new - x) * 0.5,
        x_new
    )
    
    # After computing the step
    # Add a backtracking line search
    alpha = 1.0
    for _ in range(5):  # Try a few backtracking steps
        x_temp = x - alpha * ratio
        # Evaluate f at x_temp
        # If |f(x_temp)| < |f(x)|, break
        alpha *= 0.5
    x_new = x - alpha * ratio
    
    jax_debug.print("x: {} -> {}, ratio: {}", x, x_new, ratio)
    
    return x_new, ratio, iterations + 1


@jit
def _lambert_iteration_condition(
    x: float,
    ratio: float,
    iterations: int,
    tol: float,
    max_iter: int,
    no_solution: bool,
) -> bool:
    """Condition for continuing Lambert iteration."""
    # Add check for NaN values and improve convergence criteria
    is_valid = ~(jnp.isnan(x) | jnp.isnan(ratio))
    converged = jnp.abs(ratio) < tol
    return ~converged & (iterations < max_iter) & (~no_solution) & is_valid


@jit
def _solve_lambert(
    r1: jnp.ndarray,
    r2: jnp.ndarray,
    tof: float,
    mu: float = MU,
    prograde: bool = True,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Solve Lambert's problem for two position vectors and time of flight.

    Parameters
    ----------
    r1 : `~jax.numpy.ndarray` (3)
        Initial position vector in au.
    r2 : `~jax.numpy.ndarray` (3)
        Final position vector in au.
    tof : float
        Time of flight in days.
    mu : float
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.
    prograde : bool, optional
        If True, assume prograde motion. If False, assume retrograde motion.
    max_iter : int, optional
        Maximum number of iterations over which to converge.
    tol : float, optional
        Numerical tolerance for convergence.

    Returns
    -------
    v1 : `~jax.numpy.ndarray` (3)
        Initial velocity vector in au/day.
    v2 : `~jax.numpy.ndarray` (3)
        Final velocity vector in au/day.
    """
    # Convert to non-dimensional form
    r1_mag = jnp.linalg.norm(r1)
    r2_mag = jnp.linalg.norm(r2)
    
    # Calculate the cross product of the position vectors to determine transfer plane
    h_cross = jnp.cross(r1, r2)
    h_cross_mag = jnp.linalg.norm(h_cross)

    
    # Check if the transfer is direct or retrograde
    # If h_cross_z is negative and prograde is True, or if h_cross_z is positive and prograde is False,
    # we need to flip the sign of the transfer angle
    sign_flip = jnp.where(
        (h_cross[2] < 0) == prograde,
        -1.0,
        1.0
    )
    
    # Compute the cosine and sine of the transfer angle
    cos_dnu = jnp.dot(r1, r2) / (r1_mag * r2_mag)
    cos_dnu = jnp.clip(cos_dnu, -1.0, 1.0)  # Ensure within valid range

   
        
    sin_dnu = sign_flip * h_cross_mag / (r1_mag * r2_mag)
    
    # Calculate the transfer angle (0 to 2π)
    dnu = jnp.arctan2(sin_dnu, cos_dnu)
    
    # Ensure positive transfer angle
    dnu = jnp.where(dnu < 0, dnu + 2 * jnp.pi, dnu)
    
    # Determine if the transfer is long-way (more than 180 degrees)
    long_way = dnu > jnp.pi
    
    # Adjust for long-way transfers by flipping sign for retrograde orbits
    dnu = jnp.where(
        ~prograde & long_way,
        2 * jnp.pi - dnu,
        dnu
    )
    long_way = jnp.where(~prograde, ~long_way, long_way)
    
    # Parameter of the problem
    c = jnp.sqrt(r1_mag**2 + r2_mag**2 - 2 * r1_mag * r2_mag * jnp.cos(dnu))  # chord
    s = (r1_mag + r2_mag + c) / 2  # semi-perimeter
    a_min = s / 2  # minimum energy semi-major axis
    beta = 2 * jnp.arcsin(jnp.sqrt((s - c) / s))  # transfer angle
    
    # Adjust beta for long-way transfers
    beta = jnp.where(long_way, 2 * jnp.pi - beta, beta)
    
    # Calculate the parabolic time of flight
    t_p = jnp.sqrt(2 / mu) * (s**(3/2) - jnp.sign(jnp.sin(dnu)) * (s - c)**(3/2)) / 3
    
    # Handle special cases
    # Parabolic orbit - TOF almost exactly matches parabolic TOF
    is_parabolic = jnp.abs(tof - t_p) < tol
    
    # No good solution exists if TOF < t_p
    no_solution = tof < t_p
    
    # Debug printing
    print("Debug values:")
    jax_debug.print("t_p: {t_p}", t_p=t_p)
    jax_debug.print("is_parabolic: {is_parabolic}", is_parabolic=is_parabolic)
    jax_debug.print("no_solution: {no_solution}", no_solution=no_solution)
    jax_debug.print("s: {s}", s=s)
    jax_debug.print("c: {c}", c=c)
    jax_debug.print("beta: {beta}", beta=beta)
    jax_debug.print("dnu: {dnu}", dnu=dnu)
    jax_debug.print("long_way: {long_way}", long_way=long_way)
    
    # Determine initial guess for x (the universal variable)
    # chi = jnp.sqrt(mu) * jnp.abs(alpha) * tof
    # psi = alpha * chi**2
    
    x = jnp.where(
        is_parabolic,
        0.0,  # For parabolic orbits, x = 0
        jnp.where(
            tof > t_p,  # Elliptical orbit
            0.99,  # Start with a small value for better convergence
            -0.99 # Start with a small value for hyperbolic orbits
        )
    )
    jax_debug.print("Initial x: {x}", x=x)
    
    # Initial values for iterative solution
    ratio = 1.0
    iterations = 0
    
    # Define carry structure for while loop
    def body_fun(carry):
        x, ratio, iterations = carry
        return _lambert_iteration_step(
            x, ratio, iterations,
            a_min, beta, r1_mag, r2_mag,
            s, c, tof, mu, long_way
        )
    
    def cond_fun(carry):
        x, ratio, iterations = carry
        return _lambert_iteration_condition(
            x, ratio, iterations,
            tol, max_iter, no_solution
        )
    
    # Solve iteratively
    x, ratio, iterations = lax.while_loop(
        cond_fun,
        body_fun,
        (x, ratio, iterations)
    )
    
    # Post-process results to compute velocity vectors
    # Compute semi-major axis
    jax_debug.print("x: {}, x**2: {}, x**2/2: {}", x, x**2, x**2/2)
    jax_debug.print("1 - x**2/2: {}", 1 - x**2/2)
    jax_debug.print("a_min: {}", a_min)
    a = a_min / (1 - x**2 / 2)
    jax_debug.print("Final a: {}", a)
    
    # Compute y based on x and a
    jax_debug.print("r1_mag: {}, r2_mag: {}", r1_mag, r2_mag)
    jax_debug.print("beta: {}", beta)
    jax_debug.print("a * (1 - cos(beta)): {}", a * (1 - jnp.cos(beta)))
    y = jnp.sqrt(r1_mag * r2_mag) * jnp.sin(beta / 2) / jnp.sqrt(a * (1 - jnp.cos(beta)))
    jax_debug.print("Final y: {}", y)
    
    # Compute z = x^2/a
    z = x**2 / a
    jax_debug.print("Final z: {}", z)
    
    # Get Stumpff functions
    c0, c1, c2, c3, c4, c5 = calc_stumpff(z)
    jax_debug.print("Final Stumpff c2, c3: {}, {}", c2, c3)
    
    # Compute parameters needed for Lagrange coefficients
    r12 = r1 - r2
    r12_mag = jnp.linalg.norm(r12)
    jax_debug.print("r12_mag: {}", r12_mag)
    
    # Compute the Lagrange coefficients f, g, fdot, gdot
    f = 1 - y**2 / r1_mag
    g = r1_mag * r2_mag * jnp.sin(beta) / jnp.sqrt(mu * a)
    g = jnp.where(long_way, -g, g)
    jax_debug.print("f: {}, g: {}", f, g)
    
    fdot = jnp.sqrt(mu) / (r1_mag * r2_mag) * y * (z * c3 - 1)
    gdot = 1 - y**2 / r2_mag
    jax_debug.print("fdot: {}, gdot: {}", fdot, gdot)
    
    # Compute velocities
    v1 = (r2 - f * r1) / g
    v2 = (gdot * r2 - r1) / g
    jax_debug.print("v1: {}, v2: {}", v1, v2)
    
    return v1, v2


# Vectorize the Lambert solver
_solve_lambert_vmap = jit(
    vmap(_solve_lambert, in_axes=(0, 0, 0, None, None, None, None), out_axes=(0, 0))
)


def solve_lambert(
    r1: Union[np.ndarray, jnp.ndarray],
    r2: Union[np.ndarray, jnp.ndarray],
    tof: Union[np.ndarray, float],
    mu: float = MU,
    prograde: bool = True,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve Lambert's problem for multiple initial and final positions and times of flight.

    Parameters
    ----------
    r1 : `~numpy.ndarray` or `~jax.numpy.ndarray` (N, 3)
        Initial position vectors in au.
    r2 : `~numpy.ndarray` or `~jax.numpy.ndarray` (N, 3)
        Final position vectors in au.
    tof : `~numpy.ndarray` or float (N) or scalar
        Times of flight in days.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.
    prograde : bool, optional
        If True, assume prograde motion. If False, assume retrograde motion.
    max_iter : int, optional
        Maximum number of iterations over which to converge.
    tol : float, optional
        Numerical tolerance for convergence.

    Returns
    -------
    v1 : `~numpy.ndarray` (N, 3)
        Initial velocity vectors in au/day.
    v2 : `~numpy.ndarray` (N, 3)
        Final velocity vectors in au/day.
    """
    # Convert inputs to jnp arrays if they are not already
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
    
    # Call the vectorized solver
    v1, v2 = _solve_lambert_vmap(r1, r2, tof, mu, prograde, max_iter, tol)
    
    # Convert back to numpy arrays
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    
    # If inputs were scalars, return scalar outputs
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
    prograde: bool = True,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data for a porkchop plot by solving Lambert's problem for a grid of
    departure and arrival times.

    Parameters
    ----------
    r1_func : callable
        Function that returns the position vector of the departure body at a given time.
    r2_func : callable
        Function that returns the position vector of the arrival body at a given time.
    departure_times : `~numpy.ndarray` (N)
        Array of departure times in days (e.g., MJD).
    arrival_times : `~numpy.ndarray` (M)
        Array of arrival times in days (e.g., MJD).
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        au**3 / d**2.
    prograde : bool, optional
        If True, assume prograde motion. If False, assume retrograde motion.
    max_iter : int, optional
        Maximum number of iterations for Lambert's solver.
    tol : float, optional
        Numerical tolerance for Lambert's solver.

    Returns
    -------
    delta_v_departure : `~numpy.ndarray` (N, M)
        Delta-v required at departure for each departure-arrival time combination.
    delta_v_arrival : `~numpy.ndarray` (N, M)
        Delta-v required at arrival for each departure-arrival time combination.
    total_delta_v : `~numpy.ndarray` (N, M)
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
                v1_trans, v2_trans = solve_lambert(r1, r2, tof, mu, prograde, max_iter, tol)
                
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
                
            except:
                # In case Lambert solver fails
                delta_v_departure[i, j] = np.nan
                delta_v_arrival[i, j] = np.nan
                total_delta_v[i, j] = np.nan
    
    return delta_v_departure, delta_v_arrival, total_delta_v
