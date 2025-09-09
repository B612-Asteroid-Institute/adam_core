"""
Kepler clock gating for fast detection-orbit filtering.

This module implements fast orbital mechanics-based filtering to eliminate
impossible detection-orbit pairings early in the THOR pipeline. Clock gating
uses Kepler's laws to predict where an orbit should be at observation times
and rejects pairings that are geometrically impossible within tolerances.
"""

from __future__ import annotations

import functools
import logging
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import quivr as qv

from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.origin import OriginCodes
from ..coordinates.spherical import SphericalCoordinates
from ..observations.rays import ObservationRays
from ..orbits.polyline import OrbitsPlaneParams
from ..time import Timestamp

__all__ = [
    "ClockGateConfig",
    "ClockGateResults", 
    "apply_clock_gating",
    "compute_orbital_positions_at_times",
]

logger = logging.getLogger(__name__)


class ClockGateConfig(qv.Table):
    """Configuration parameters for clock gating."""
    
    #: Maximum angular separation tolerance in arcseconds
    max_angular_sep_arcsec = qv.Float64Column()
    #: Maximum radial distance tolerance in AU  
    max_radial_sep_au = qv.Float64Column()
    #: Maximum time span for valid extrapolation in days
    max_extrapolation_days = qv.Float64Column()


class ClockGateResults(qv.Table):
    """Results of clock gating filter."""
    
    #: Detection identifier
    det_id = qv.LargeStringColumn()
    #: Orbit identifier
    orbit_id = qv.LargeStringColumn()
    #: Whether the pairing passed the filter
    passed = qv.BooleanColumn()
    #: Angular separation in arcseconds
    angular_sep_arcsec = qv.Float64Column()
    #: Radial separation in AU
    radial_sep_au = qv.Float64Column()
    #: Time extrapolation from orbit epoch in days
    extrapolation_days = qv.Float64Column()
    #: Predicted mean anomaly at observation time (radians)
    predicted_mean_anomaly = qv.Float64Column()
    #: Predicted true anomaly at observation time (radians)
    predicted_true_anomaly = qv.Float64Column()


@jax.jit
def _solve_kepler_equation_jax(
    M: jax.Array,
    e: jax.Array,
    max_iterations: int = 10,
    tolerance: float = 1e-12,
) -> jax.Array:
    """
    Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E.
    
    Uses Newton-Raphson iteration with JAX for performance.
    
    Parameters
    ----------
    M : jax.Array
        Mean anomaly (radians)
    e : jax.Array
        Eccentricity
    max_iterations : int
        Maximum Newton-Raphson iterations
    tolerance : float
        Convergence tolerance
        
    Returns
    -------
    E : jax.Array
        Eccentric anomaly (radians)
    """
    # Initial guess for eccentric anomaly
    E = M + e * jnp.sin(M)
    
    def body_fun(carry):
        E, _ = carry
        f = E - e * jnp.sin(E) - M
        df_dE = 1 - e * jnp.cos(E)
        E_new = E - f / df_dE
        return E_new, jnp.abs(E_new - E)
    
    def cond_fun(carry):
        _, residual = carry
        return residual > tolerance
    
    E_final, _ = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (E, jnp.array(1.0))
    )
    
    # Fallback to fori_loop for fixed iterations if while_loop doesn't converge
    def newton_step(i, E):
        f = E - e * jnp.sin(E) - M
        df_dE = 1 - e * jnp.cos(E)
        return E - f / df_dE
    
    E_final = jax.lax.fori_loop(0, max_iterations, newton_step, E_final)
    
    return E_final


@jax.jit
def _eccentric_to_true_anomaly_jax(
    E: jax.Array,
    e: jax.Array,
) -> jax.Array:
    """
    Convert eccentric anomaly to true anomaly.
    
    Parameters
    ----------
    E : jax.Array
        Eccentric anomaly (radians)
    e : jax.Array
        Eccentricity
        
    Returns
    -------
    f : jax.Array
        True anomaly (radians)
    """
    # tan(f/2) = sqrt((1+e)/(1-e)) * tan(E/2)
    tan_half_E = jnp.tan(E / 2)
    tan_half_f = jnp.sqrt((1 + e) / (1 - e)) * tan_half_E
    f = 2 * jnp.arctan(tan_half_f)
    
    return f


@jax.jit
def _orbital_position_from_anomaly_jax(
    f: jax.Array,
    a: jax.Array,
    e: jax.Array,
    p_hat: jax.Array,
    q_hat: jax.Array,
) -> jax.Array:
    """
    Compute 3D orbital position from true anomaly and orbital elements.
    
    Parameters
    ----------
    f : jax.Array
        True anomaly (radians)
    a : jax.Array
        Semi-major axis (AU)
    e : jax.Array
        Eccentricity
    p_hat : jax.Array (3,)
        Orbital basis vector towards perihelion
    q_hat : jax.Array (3,)
        Orbital basis vector perpendicular to p in orbital plane
        
    Returns
    -------
    position : jax.Array (3,)
        3D position in SSB ecliptic frame (AU)
    """
    # Compute radial distance
    cos_f = jnp.cos(f)
    sin_f = jnp.sin(f)
    
    r = a * (1 - e * e) / (1 + e * cos_f)
    
    # Position in orbital plane
    x_orb = r * cos_f
    y_orb = r * sin_f
    
    # Transform to 3D
    position = x_orb * p_hat + y_orb * q_hat
    
    return position


@jax.jit
def compute_orbital_positions_at_times(
    plane_params_elements: jax.Array,  # [a, e, M0, n]
    plane_params_bases: jax.Array,    # [p_hat, q_hat] (2, 3)
    epoch_mjd: jax.Array,
    obs_times_mjd: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Compute orbital positions at observation times using Kepler's laws.
    
    Parameters
    ----------
    plane_params_elements : jax.Array (4,)
        Orbital elements [a, e, M0, n] where n is mean motion (rad/day)
    plane_params_bases : jax.Array (2, 3)  
        Orbital plane basis vectors [p_hat, q_hat]
    epoch_mjd : jax.Array
        Epoch time (MJD)
    obs_times_mjd : jax.Array (N,)
        Observation times (MJD)
        
    Returns
    -------
    positions : jax.Array (N, 3)
        3D positions at observation times (AU)
    mean_anomalies : jax.Array (N,)
        Mean anomalies at observation times (radians)  
    true_anomalies : jax.Array (N,)
        True anomalies at observation times (radians)
    """
    a, e, M0, n = plane_params_elements
    p_hat, q_hat = plane_params_bases
    
    # Compute mean anomalies at observation times
    dt = obs_times_mjd - epoch_mjd
    mean_anomalies = M0 + n * dt
    
    # Wrap mean anomalies to [0, 2π)
    mean_anomalies = jnp.mod(mean_anomalies, 2 * jnp.pi)
    
    # Solve Kepler's equation for eccentric anomalies
    eccentric_anomalies = jax.vmap(_solve_kepler_equation_jax, in_axes=(0, None))(
        mean_anomalies, e
    )
    
    # Convert to true anomalies
    true_anomalies = jax.vmap(_eccentric_to_true_anomaly_jax, in_axes=(0, None))(
        eccentric_anomalies, e
    )
    
    # Compute 3D positions
    positions = jax.vmap(
        _orbital_position_from_anomaly_jax, 
        in_axes=(0, None, None, None, None)
    )(true_anomalies, a, e, p_hat, q_hat)
    
    return positions, mean_anomalies, true_anomalies


@functools.partial(jax.jit, static_argnames=['num_rays', 'num_orbits'])
def _apply_clock_gating_jax(
    obs_times_mjd: jax.Array,  # (num_rays,)
    obs_positions: jax.Array,  # (num_rays, 3)
    ray_directions: jax.Array,  # (num_rays, 3)
    orbit_elements: jax.Array,  # (num_orbits, 4) - [a, e, M0, n]
    orbit_bases: jax.Array,    # (num_orbits, 2, 3) - [p_hat, q_hat]
    epoch_mjds: jax.Array,     # (num_orbits,)
    max_angular_sep_rad: float,
    max_radial_sep_au: float,
    max_extrapolation_days: float,
    typical_distance_au: float = 2.5,
    num_rays: int = 0,
    num_orbits: int = 0,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    JAX-compiled clock gating computation.
    
    Returns
    -------
    passed_mask : jax.Array (num_orbits, num_rays)
        Boolean mask indicating which ray-orbit pairs passed gating
    angular_sep_arcsec : jax.Array (num_orbits, num_rays)
        Angular separations in arcseconds
    radial_sep_au : jax.Array (num_orbits, num_rays)
        Radial separations in AU
    extrapolation_days : jax.Array (num_orbits, num_rays)
        Time extrapolation in days
    predicted_M : jax.Array (num_orbits, num_rays)
        Predicted mean anomalies
    predicted_f : jax.Array (num_orbits, num_rays)
        Predicted true anomalies
    predicted_positions : jax.Array (num_orbits, num_rays, 3)
        Predicted 3D positions
    """
    # Approximate observed positions using typical distance
    obs_positions_approx = obs_positions + typical_distance_au * ray_directions
    obs_norms = jnp.linalg.norm(obs_positions_approx, axis=1)  # (num_rays,)
    
    def process_single_orbit(orbit_idx):
        """Process a single orbit against all rays."""
        # Extract orbital elements and basis
        a, e, M0, n = orbit_elements[orbit_idx]
        p_hat, q_hat = orbit_bases[orbit_idx]
        epoch_mjd = epoch_mjds[orbit_idx]
        
        # Compute predicted positions for all observation times
        elements_jax = jnp.array([a, e, M0, n])
        bases_jax = jnp.array([p_hat, q_hat])
        
        pred_positions, pred_M, pred_f = compute_orbital_positions_at_times(
            elements_jax, bases_jax, epoch_mjd, obs_times_mjd
        )
        
        # Time validity check
        dt = jnp.abs(obs_times_mjd - epoch_mjd)
        time_valid = dt <= max_extrapolation_days
        
        # Angular separation computation
        pred_norms = jnp.linalg.norm(pred_positions, axis=1)
        denom = jnp.maximum(obs_norms * pred_norms, 1e-15)
        cos_theta = jnp.sum(obs_positions_approx * pred_positions, axis=1) / denom
        cos_theta = jnp.clip(cos_theta, -1.0, 1.0)
        angular_separation = jnp.arccos(cos_theta)  # radians
        
        # Radial separation
        radial_separation = jnp.abs(pred_norms - obs_norms)
        
        # Gating decision (angle + time, ignoring radial for now as it's too restrictive)
        passed_mask = time_valid & (angular_separation <= max_angular_sep_rad)
        
        # Convert angular separation to arcseconds
        angular_sep_arcsec = angular_separation * 180.0 / jnp.pi * 3600.0
        
        return (
            passed_mask,
            angular_sep_arcsec,
            radial_separation,
            dt,
            pred_M,
            pred_f,
            pred_positions,
        )
    
    # Process all orbits in parallel
    results = jax.vmap(process_single_orbit)(jnp.arange(num_orbits))
    
    return results


def apply_clock_gating(
    observation_rays: ObservationRays,
    plane_params: OrbitsPlaneParams,
    config: ClockGateConfig,
    use_jax: bool = True,
) -> ClockGateResults:
    """
    Apply clock gating filter to observation rays and orbital plane parameters.
    
    This function computes predicted orbital positions at observation times
    and filters out detection-orbit pairings that exceed geometric tolerances.
    
    Parameters
    ----------
    observation_rays : ObservationRays
        Observation rays to filter
    plane_params : OrbitsPlaneParams
        Orbital plane parameters including elements and basis vectors
    config : ClockGateConfig
        Clock gating configuration parameters
    use_jax : bool, default=True
        Whether to use JAX-compiled computation for performance
        
    Returns
    -------
    results : ClockGateResults
        Clock gating filter results
    """
    if len(observation_rays) == 0 or len(plane_params) == 0:
        return ClockGateResults.empty()
    
    logger.info(f"Applying clock gating to {len(observation_rays)} observation rays and {len(plane_params)} orbits (JAX={use_jax})")
    
    # Extract configuration (as scalars)
    max_angular_sep_rad = float(config.max_angular_sep_arcsec.to_numpy()[0]) / 3600.0 * np.pi / 180.0
    max_radial_sep_au = float(config.max_radial_sep_au.to_numpy()[0])
    max_extrapolation_days = float(config.max_extrapolation_days.to_numpy()[0])
    
    # Extract ray information
    obs_times_mjd = observation_rays.time.mjd().to_numpy()
    observer_positions = np.column_stack([
        observation_rays.observer.x.to_numpy(),
        observation_rays.observer.y.to_numpy(),
        observation_rays.observer.z.to_numpy(),
    ])
    ray_directions = np.column_stack([
        observation_rays.u_x.to_numpy(),
        observation_rays.u_y.to_numpy(),
        observation_rays.u_z.to_numpy(),
    ])
    
    # Extract orbital parameters
    orbit_ids = plane_params.orbit_id.to_pylist()
    epoch_mjds = plane_params.t0.mjd().to_numpy()
    a_arr = plane_params.a.to_numpy()
    e_arr = plane_params.e.to_numpy()
    M0_arr = plane_params.M0.to_numpy()
    
    # Compute mean motions
    k = 0.01720209895  # rad/day * AU^(3/2)
    n_arr = k / (a_arr * np.sqrt(a_arr))
    
    # Basis vectors
    p_vecs = np.column_stack([
        plane_params.p_x.to_numpy(),
        plane_params.p_y.to_numpy(),
        plane_params.p_z.to_numpy(),
    ])
    q_vecs = np.column_stack([
        plane_params.q_x.to_numpy(),
        plane_params.q_y.to_numpy(),
        plane_params.q_z.to_numpy(),
    ])
    
    if use_jax:
        # Use JAX-compiled computation
        orbit_elements = jnp.column_stack([a_arr, e_arr, M0_arr, n_arr])  # (num_orbits, 4)
        orbit_bases = jnp.stack([p_vecs, q_vecs], axis=1)  # (num_orbits, 2, 3)
        
        # Call JAX kernel
        (passed_mask, angular_sep_arcsec, radial_sep_au, extrapolation_days,
         predicted_M, predicted_f, predicted_positions) = _apply_clock_gating_jax(
            jnp.asarray(obs_times_mjd),
            jnp.asarray(observer_positions),
            jnp.asarray(ray_directions),
            orbit_elements,
            orbit_bases,
            jnp.asarray(epoch_mjds),
            max_angular_sep_rad,
            max_radial_sep_au,
            max_extrapolation_days,
            num_rays=len(observation_rays),
            num_orbits=len(plane_params),
        )
        
        # Convert to numpy and flatten for table construction
        passed_mask_np = np.asarray(passed_mask).flatten()
        angular_sep_np = np.asarray(angular_sep_arcsec).flatten()
        radial_sep_np = np.asarray(radial_sep_au).flatten()
        extrapolation_np = np.asarray(extrapolation_days).flatten()
        predicted_M_np = np.asarray(predicted_M).flatten()
        predicted_f_np = np.asarray(predicted_f).flatten()
        
    else:
        # Use legacy NumPy computation (kept for validation)
        return _apply_clock_gating_legacy(
            observation_rays, plane_params, config,
            max_angular_sep_rad, max_radial_sep_au, max_extrapolation_days
        )
    
    # Build result arrays
    det_ids_list = observation_rays.det_id.to_pylist()
    num_rays = len(det_ids_list)
    num_orbits = len(orbit_ids)
    
    all_det_ids = []
    all_orbit_ids = []
    
    for orbit_id in orbit_ids:
        all_det_ids.extend(det_ids_list)
        all_orbit_ids.extend([orbit_id] * num_rays)
    
    # Create results table
    results = ClockGateResults.from_kwargs(
        det_id=all_det_ids,
        orbit_id=all_orbit_ids,
        passed=passed_mask_np.tolist(),
        angular_sep_arcsec=angular_sep_np.tolist(),
        radial_sep_au=radial_sep_np.tolist(),
        extrapolation_days=extrapolation_np.tolist(),
        predicted_mean_anomaly=predicted_M_np.tolist(),
        predicted_true_anomaly=predicted_f_np.tolist(),
    )
    
    n_passed = int(np.sum(passed_mask_np))
    n_total = len(passed_mask_np)
    logger.info(f"Clock gating: {n_passed}/{n_total} ({100*n_passed/n_total:.1f}%) pairings passed")
    
    return results


def _apply_clock_gating_legacy(
    observation_rays: ObservationRays,
    plane_params: OrbitsPlaneParams,
    config: ClockGateConfig,
    max_angular_sep_rad: float,
    max_radial_sep_au: float,
    max_extrapolation_days: float,
) -> ClockGateResults:
    """Legacy NumPy implementation for validation."""
    # Extract ray information
    obs_times_mjd = observation_rays.time.mjd().to_numpy()
    observer_positions = np.column_stack([
        observation_rays.observer.x.to_numpy(),
        observation_rays.observer.y.to_numpy(),
        observation_rays.observer.z.to_numpy(),
    ])
    ray_directions = np.column_stack([
        observation_rays.u_x.to_numpy(),
        observation_rays.u_y.to_numpy(),
        observation_rays.u_z.to_numpy(),
    ])
    
    # Approximate observed positions
    typical_distance = 2.5  # AU
    obs_positions = observer_positions + typical_distance * ray_directions
    obs_norms = np.linalg.norm(obs_positions, axis=1)
    
    # Results lists
    all_det_ids = []
    all_orbit_ids = []
    all_passed = []
    all_angular_sep = []
    all_radial_sep = []
    all_extrapolation = []
    all_predicted_M = []
    all_predicted_f = []

    # Vectorized plane params
    orbit_ids = plane_params.orbit_id.to_pylist()
    epoch_mjds = plane_params.t0.mjd().to_numpy()
    a_arr = plane_params.a.to_numpy()
    e_arr = plane_params.e.to_numpy()
    M0_arr = plane_params.M0.to_numpy()
    p_x_arr = plane_params.p_x.to_numpy()
    p_y_arr = plane_params.p_y.to_numpy()
    p_z_arr = plane_params.p_z.to_numpy()
    q_x_arr = plane_params.q_x.to_numpy()
    q_y_arr = plane_params.q_y.to_numpy()
    q_z_arr = plane_params.q_z.to_numpy()

    # Process each orbit
    for i in range(len(plane_params)):
        orbit_id = orbit_ids[i]
        epoch_mjd = float(epoch_mjds[i])
        
        # Extract orbital elements and basis
        a = float(a_arr[i])
        e = float(e_arr[i])
        M0 = float(M0_arr[i])
        
        # Compute mean motion (rad/day) - using Gaussian constant
        k = 0.01720209895  # rad/day * AU^(3/2)
        n = k / (a * np.sqrt(a))
        
        # Basis vectors
        p_hat = np.array([p_x_arr[i], p_y_arr[i], p_z_arr[i]], dtype=float)
        q_hat = np.array([q_x_arr[i], q_y_arr[i], q_z_arr[i]], dtype=float)
        
        # Convert to JAX arrays
        elements_jax = jnp.array([a, e, M0, n])
        bases_jax = jnp.array([p_hat, q_hat])
        epoch_jax = jnp.array(epoch_mjd)
        obs_times_jax = jnp.array(obs_times_mjd)
        
        # Compute predicted positions
        pred_positions, pred_M, pred_f = compute_orbital_positions_at_times(
            elements_jax, bases_jax, epoch_jax, obs_times_jax
        )
        
        pred_positions_np = np.array(pred_positions)  # (N,3)
        pred_M_np = np.array(pred_M)                 # (N,)
        pred_f_np = np.array(pred_f)                 # (N,)

        # Vectorized time validity
        dt = np.abs(obs_times_mjd - epoch_mjd)
        time_valid = dt <= max_extrapolation_days

        # Vectorized angular separation
        pred_norms = np.linalg.norm(pred_positions_np, axis=1)
        denom = np.maximum(obs_norms * pred_norms, 1e-15)
        cos_theta = np.einsum('ij,ij->i', obs_positions, pred_positions_np) / denom
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angular_separation = np.arccos(cos_theta)  # radians

        # Vectorized radial separation
        radial_separation = np.abs(pred_norms - obs_norms)

        # Gating decision (angle + time)
        passed_mask = time_valid & (angular_separation <= max_angular_sep_rad)

        # Extend results
        det_ids_list = observation_rays.det_id.to_pylist()
        all_det_ids.extend(det_ids_list)
        all_orbit_ids.extend([orbit_id] * len(det_ids_list))
        all_passed.extend(passed_mask.tolist())
        all_angular_sep.extend((angular_separation * 180.0 / np.pi * 3600.0).tolist())
        all_radial_sep.extend(radial_separation.tolist())
        all_extrapolation.extend(dt.tolist())
        all_predicted_M.extend(pred_M_np.tolist())
        all_predicted_f.extend(pred_f_np.tolist())
    
    # Create results table
    results = ClockGateResults.from_kwargs(
        det_id=all_det_ids,
        orbit_id=all_orbit_ids,
        passed=all_passed,
        angular_sep_arcsec=all_angular_sep,
        radial_sep_au=all_radial_sep,
        extrapolation_days=all_extrapolation,
        predicted_mean_anomaly=all_predicted_M,
        predicted_true_anomaly=all_predicted_f,
    )
    
    return results
