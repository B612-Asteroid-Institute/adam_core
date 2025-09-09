"""
Compute anomaly labels for observation-orbit overlaps.

This module provides functionality to compute orbital elements (M, n, r) at 
observation times for geometric overlaps, enabling downstream clock gating.
"""

from pathlib import Path
from typing import Dict, Any

import numpy as np
import quivr as qv
import pyarrow as pa
import pyarrow.compute as pc
import jax
import jax.numpy as jnp
from adam_core.geometry.overlap import OverlapHits
from adam_core.observations.rays import ObservationRays
from adam_core.orbits.orbits import Orbits
from adam_core.dynamics.kepler import solve_kepler as jax_solve_kepler
from .anomaly import AnomalyLabels as CanonicalAnomalyLabels


# Use the canonical AnomalyLabels from anomaly.py
AnomalyLabels = CanonicalAnomalyLabels


def label_anomalies(
    hits: OverlapHits,
    rays: ObservationRays,
    orbits: Orbits | None = None,
) -> CanonicalAnomalyLabels:
    """
    Compute anomaly labels for observation-orbit overlaps.
    
    This function takes geometric overlaps and computes the orbital elements
    (mean anomaly, mean motion, heliocentric distance) at each observation time,
    enabling downstream time-consistency checks.
    
    Parameters
    ----------
    hits : OverlapHits
        Geometric overlaps between observations and test orbits.
    rays : ObservationRays
        Observation rays containing timing information.
    orbits : Orbits, optional
        Original orbit data containing Keplerian elements. If provided,
        real orbital elements will be used instead of synthetic ones.
        
    Returns
    -------
    AnomalyLabels
        Table with anomaly labels for each overlap, sorted by (orbit_id, det_id).
        
    Notes
    -----
    This implementation computes labels using 2-body Keplerian elements at epoch.
    For now, it returns one label per hit; multi-anomaly support can be added later.
    """
    if len(hits) == 0:
        return CanonicalAnomalyLabels.empty()
    
    # Vectorized alignment of hits to ray times via index_in
    hits_table = hits.table
    hit_det = hits_table["det_id"]
    hit_orbit = hits_table["orbit_id"]
    hit_seg = hits_table["seg_id"]
    rays_det = rays.table["det_id"]
    rays_times = rays.time.mjd()

    ray_idx = pc.index_in(hit_det, rays_det)
    hit_times = pc.take(rays_times, ray_idx)
    valid_mask = pc.is_valid(hit_times)
    if pc.sum(pc.cast(valid_mask, pa.int64())).as_py() == 0:
        return CanonicalAnomalyLabels.empty()
    hit_det = pc.filter(hit_det, valid_mask)
    hit_orbit = pc.filter(hit_orbit, valid_mask)
    hit_seg = pc.filter(hit_seg, valid_mask)
    hit_times = pc.filter(hit_times, valid_mask)

    # Require real Orbits for non-empty hits
    if len(hit_det) > 0 and orbits is None:
        raise ValueError("label_anomalies requires 'orbits' when hits are non-empty")

    # Build orbit elements (Arrow) directly from Orbits.to_keplerian
    kep = orbits.coordinates.to_keplerian() if orbits is not None else None
    if kep is not None:
        orbit_ids_arr = orbits.table.column("orbit_id")
        a_arr_np = kep.a.to_numpy()
        e_arr_np = kep.e.to_numpy()
        M0_deg_np = kep.M.to_numpy() * (180.0 / np.pi)
        epoch_mjd_np = kep.time.mjd().to_numpy()
        GM_sun_au3_per_day2 = 2.959122082855911e-4
        n_deg_per_day_np = np.sqrt(GM_sun_au3_per_day2 / (a_arr_np ** 3)) * (180.0 / np.pi)

        elements_tbl = pa.table(
            {
                "orbit_id": orbit_ids_arr,
                "a": pa.array(a_arr_np, type=pa.float64()),
                "e": pa.array(e_arr_np, type=pa.float64()),
                "M0_deg": pa.array(M0_deg_np, type=pa.float64()),
                "n_deg_per_day": pa.array(n_deg_per_day_np, type=pa.float64()),
                "epoch_mjd": pa.array(epoch_mjd_np, type=pa.float64()),
            }
        )
    else:
        elements_tbl = pa.table(
            {
                "orbit_id": pa.array([], type=pa.large_string()),
                "a": pa.array([], type=pa.float64()),
                "e": pa.array([], type=pa.float64()),
                "M0_deg": pa.array([], type=pa.float64()),
                "n_deg_per_day": pa.array([], type=pa.float64()),
                "epoch_mjd": pa.array([], type=pa.float64()),
            }
        )

    elements_orbit = elements_tbl.column("orbit_id")
    orbit_idx = pc.index_in(hit_orbit, elements_orbit)
    a_hit = pc.take(elements_tbl.column("a"), orbit_idx).to_numpy(zero_copy_only=False)
    e_hit = pc.take(elements_tbl.column("e"), orbit_idx).to_numpy(zero_copy_only=False)
    M0_hit = pc.take(elements_tbl.column("M0_deg"), orbit_idx).to_numpy(zero_copy_only=False)
    n_hit = pc.take(elements_tbl.column("n_deg_per_day"), orbit_idx).to_numpy(zero_copy_only=False)
    epoch_hit = pc.take(elements_tbl.column("epoch_mjd"), orbit_idx).to_numpy(zero_copy_only=False)

    # Vectorized anomaly computation
    times_np = hit_times.to_numpy(zero_copy_only=False).astype(float)
    epoch_hit_clean = np.where(np.isnan(epoch_hit), times_np, epoch_hit)
    M0_hit_clean = np.where(np.isnan(M0_hit), 0.0, M0_hit)
    dt_days = times_np - epoch_hit_clean
    M_deg_arr = (M0_hit_clean + n_hit * dt_days) % 360.0
    M_rad_arr = np.radians(M_deg_arr)
    solve_v = jax.vmap(jax_solve_kepler, in_axes=(0, 0))
    nu_arr = np.array(solve_v(jnp.asarray(e_hit), jnp.asarray(M_rad_arr)))
    # Eccentric anomaly (elliptic)
    # For circular orbits (e=0), E = f
    # For elliptical orbits, use the standard conversion
    E_rad_arr = np.where(
        e_hit < 1e-4,  # Nearly circular (more generous threshold)
        nu_arr,  # E = f for circular orbits
        np.arctan2(
            np.sqrt(1 - e_hit**2) * np.sin(nu_arr) / (1 + e_hit * np.cos(nu_arr)),
            (e_hit + np.cos(nu_arr)) / (1 + e_hit * np.cos(nu_arr))
        )
    )
    r_arr = a_hit * (1 - e_hit * np.cos(E_rad_arr))
    zeros = np.zeros_like(r_arr)

    # Convert to canonical AnomalyLabels format
    M_rad_arr_final = np.radians(M_deg_arr)
    n_rad_day_arr = np.radians(n_hit)
    
    labels = CanonicalAnomalyLabels.from_kwargs(
        det_id=hit_det,
        orbit_id=hit_orbit,
        seg_id=hit_seg,
        variant_id=pa.array(zeros.astype(int), type=pa.int32()),  # Single variant per hit for now
        f_rad=pa.array(nu_arr, type=pa.float64()),
        E_rad=pa.array(E_rad_arr, type=pa.float64()),
        M_rad=pa.array(M_rad_arr_final, type=pa.float64()),
        n_rad_day=pa.array(n_rad_day_arr, type=pa.float64()),
        r_au=pa.array(r_arr, type=pa.float64()),
        snap_error=pa.array(zeros, type=pa.float64()),
        plane_distance_au=pa.array(zeros, type=pa.float64()),
    )
    
    # Sort deterministically
    if len(labels) > 0:
        labels = labels.sort_by([("orbit_id", "ascending"), ("det_id", "ascending")])
    
    return labels




def _compute_anomalies_at_time(elements: Dict[str, Any], time_mjd: float) -> tuple[float, float, float, float, float]:
    """
    Compute orbital anomalies at a given time using 2-body Keplerian motion.
    
    Parameters
    ----------
    elements : dict
        Orbital elements with keys: a, e, M0_deg, n_deg_per_day, epoch_mjd
    time_mjd : float
        Time at which to compute anomalies (MJD)
        
    Returns
    -------
    tuple[float, float, float, float, float]
        Mean anomaly (degrees), mean motion (deg/day), heliocentric distance (AU), E (rad), f (rad)
    """
    # Extract elements
    a = elements["a"]
    e = elements["e"]
    M0_deg = elements["M0_deg"]
    n_deg_per_day = elements["n_deg_per_day"]
    epoch_mjd = elements["epoch_mjd"]
    
    # Compute mean anomaly at observation time
    dt_days = time_mjd - epoch_mjd
    M_deg = (M0_deg + n_deg_per_day * dt_days) % 360.0
    M_rad = np.radians(M_deg)
    
    # Solve Kepler for true anomaly (JAX function returns nu)
    nu_rad = float(jax_solve_kepler(float(e), float(M_rad)))
    # Compute eccentric anomaly from true anomaly (elliptic case assumed here)
    if e < 1.0:
        # E from nu
        sinE = np.sqrt(1 - e**2) * np.sin(nu_rad) / (1 + e * np.cos(nu_rad))
        cosE = (e + np.cos(nu_rad)) / (1 + e * np.cos(nu_rad))
        E_rad = float(np.arctan2(sinE, cosE))
    else:
        # For non-elliptic, set E to NaN (not used) and keep r via conic formula
        E_rad = float('nan')
    
    # Radius from conic equation r = a(1 - e cos E) for elliptic; fallback to elliptical form for e<1
    r_au = a * (1 - e * np.cos(E_rad)) if e < 1.0 else a * (e * np.cosh(E_rad) - 1.0)
    
    return M_deg, n_deg_per_day, float(r_au), float(E_rad), float(nu_rad)


def _compute_plane_distance(elements: Dict[str, Any], det_id: str, rays: ObservationRays) -> float:
    """
    Compute the distance from the detection ray to the orbital plane.
    
    This is a placeholder implementation that returns a small random value.
    In a full implementation, this would:
    1. Extract the orbital plane normal from the elements
    2. Get the observer position and ray direction for this detection
    3. Compute the perpendicular distance from the ray to the plane
    
    Parameters
    ----------
    elements : dict
        Orbital elements for the orbit
    det_id : str
        Detection ID
    rays : ObservationRays
        Observation rays containing ray geometry
        
    Returns
    -------
    float
        Distance from ray to orbital plane in AU
    """
    # Placeholder: return a small value for testing
    # Real implementation would compute actual geometric distance
    return 0.001  # AU


def _compute_snap_error(elements: Dict[str, Any], det_id: str, rays: ObservationRays, seg_id: int) -> float:
    """
    Compute the projection error when snapping the detection to the ellipse.
    
    This is a placeholder implementation that returns a small random value.
    In a full implementation, this would:
    1. Project the detection ray onto the orbital plane
    2. Find the nearest point on the ellipse to the projected point
    3. Compute the distance between the projected point and nearest ellipse point
    
    Parameters
    ----------
    elements : dict
        Orbital elements for the orbit
    det_id : str
        Detection ID
    rays : ObservationRays
        Observation rays containing ray geometry
    seg_id : int
        Segment ID from the BVH hit
        
    Returns
    -------
    float
        Snap error in AU
    """
    # Placeholder: return a small value for testing
    # Real implementation would compute actual projection error
    return 0.0005  # AU
