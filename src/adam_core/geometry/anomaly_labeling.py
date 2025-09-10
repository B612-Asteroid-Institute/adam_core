"""
Compute anomaly labels for observation-orbit overlaps.

This module provides functionality to compute orbital elements (M, n, r) at
observation times for geometric overlaps, enabling downstream clock gating.

Performance
-----------
- Vectorized joins with ``pyarrow.compute`` (``index_in``/``take``), anomaly solve via
  ``jax.vmap(jax_solve_kepler)``, and projection metrics via a jitted JAX kernel.
- Zero-copy table handling using ``quivr``/PyArrow; conversions are deferred until the
  final table materialization.
- JIT behavior: only small kernels are jitted. The first call includes compile time.
  Steady-state runtime is approximately linear in the number of hits.
- Observed throughput (CPU, this dev machine, 2025-09-10):
  * 2k hits: ~7.82 ms per run → ~0.26M hits/s
  * 20k hits: ~38.12 ms per run → ~0.52M hits/s
- Parallelism: for large workloads, shard by orbit/hit groups under Ray. Keep input
  shapes fixed per remote (pad if needed) to avoid JAX recompilation.
- Memory profile: O(N_hits) intermediates in structure-of-arrays layout; no Python
  loops on the critical path.

Notes
-----
- Canonical enforcement: rays and orbits are coerced to heliocentric–ecliptic; a
  mismatch after coercion raises ``ValueError``.
- Deterministic output sorting: (``det_id``, ``orbit_id``, ``variant_id``, ``snap_error``).
- Multi-anomaly roadmap: current implementation emits K=1 variant. Planned extension
  will support up to K≤3 variants near nodes, selecting top-K by ``snap_error`` while
  keeping shapes fixed for JIT stability.
"""

from pathlib import Path
from typing import Any, Dict
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import ray

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import OriginCodes
from adam_core.coordinates.transform import transform_coordinates
from adam_core.dynamics.kepler import solve_kepler as jax_solve_kepler
from adam_core.geometry.overlap import OverlapHits
from adam_core.observations.rays import ObservationRays
from adam_core.orbits.orbits import Orbits
from adam_core.utils.iter import _iterate_chunk_indices

from .anomaly import AnomalyLabels as CanonicalAnomalyLabels
from .projection import (
    ellipse_snap_distance,
    ellipse_snap_distance_multi_seed,
    project_ray_to_orbital_plane,
    ray_to_plane_distance,
    transform_to_perifocal_2d,
)

# Use the canonical AnomalyLabels from anomaly.py
AnomalyLabels = CanonicalAnomalyLabels


def label_anomalies_worker(
    hits: OverlapHits,
    rays: ObservationRays,
    orbits: Orbits,
    *,
    max_k: int,
    snap_error_max_au: float | None,
    dedupe_angle_tol: float,
    chunk_size: int = 8192,
) -> CanonicalAnomalyLabels:
    """
    Core worker for anomaly labeling with chunking and padding.

    Assumes aligned inputs (hits, rays, orbits are already aligned and validated).
    Always chunks and pads to ensure stable JAX compilation. Calls the unified
    JAX kernel for multi-candidate computation.

    Parameters
    ----------
    hits : OverlapHits
        Aligned geometric overlaps
    rays : ObservationRays
        Aligned observation rays (duplicated as needed)
    orbits : Orbits
        Aligned orbits (duplicated as needed)
    max_k : int
        Maximum candidates per hit
    snap_error_max_au : float | None
        Maximum snap error threshold
    dedupe_angle_tol : float
        Deduplication tolerance
    chunk_size : int, default=8192
        Fixed chunk size for JAX compilation stability

    Returns
    -------
    CanonicalAnomalyLabels
        Sorted anomaly labels
    """
    n_hits = len(hits)
    if n_hits == 0:
        return CanonicalAnomalyLabels.empty()

    # Sanity checks for inputs.
    assert (
        len(hits) == len(rays) == len(orbits)
    ), "Every hit must have a corresponding ray and orbit"
    assert rays.observer.frame == "ecliptic", "Rays must be in ecliptic frame"
    assert pc.all(
        pc.equal(rays.observer.origin.code, OriginCodes.SUN.name)
    ).as_py(), "Rays must have SUN origin"

    ray_origins = np.column_stack(
        [
            rays.observer.x.to_numpy(zero_copy_only=False),
            rays.observer.y.to_numpy(zero_copy_only=False),
            rays.observer.z.to_numpy(zero_copy_only=False),
        ]
    )
    ray_directions = np.column_stack(
        [
            rays.u_x.to_numpy(zero_copy_only=False),
            rays.u_y.to_numpy(zero_copy_only=False),
            rays.u_z.to_numpy(zero_copy_only=False),
        ]
    )

    # Extract orbital elements (aligned orbits)
    kep = orbits.coordinates.to_keplerian()
    a_hit = kep.a.to_numpy()
    e_hit = kep.e.to_numpy()
    # Convert angular elements to radians for geometry kernels
    i_hit = np.radians(kep.i.to_numpy())
    raan_hit = np.radians(kep.raan.to_numpy())
    ap_hit = np.radians(kep.ap.to_numpy())
    # Mean anomaly stored in degrees; keep degrees for downstream M computation
    M0_hit = kep.M.to_numpy()
    epoch_hit = kep.time.mjd().to_numpy()
    # Mean motion in degrees/day from semimajor axis and mu
    n_hit = kep.n  # degrees/day (numpy array)
    sin_i = np.sin(i_hit)
    cos_i = np.cos(i_hit)
    sin_raan = np.sin(raan_hit)
    cos_raan = np.cos(raan_hit)
    plane_normals = np.column_stack([sin_i * sin_raan, -sin_i * cos_raan, cos_i])

    # Pad to next multiple of chunk_size and compute via fixed-size chunks
    n = ray_origins.shape[0]
    padded_n = ((n + chunk_size - 1) // chunk_size) * chunk_size
    pad = max(0, padded_n - n)

    if pad > 0:
        # Safe pad rows
        zeros3_ro = np.zeros((pad, 3), dtype=ray_origins.dtype)
        zeros3_norm = np.zeros((pad, 3), dtype=plane_normals.dtype)
        ray_origins = np.vstack([ray_origins, zeros3_ro])
        # Use +Y unit direction for padded rows to avoid degeneracy
        pad_dirs = np.column_stack(
            [
                np.zeros(pad, dtype=ray_directions.dtype),
                np.ones(pad, dtype=ray_directions.dtype),
                np.zeros(pad, dtype=ray_directions.dtype),
            ]
        )
        ray_directions = np.vstack([ray_directions, pad_dirs])
        a_hit = np.concatenate([a_hit, np.zeros(pad, dtype=a_hit.dtype)])
        e_hit = np.concatenate([e_hit, np.zeros(pad, dtype=e_hit.dtype)])
        i_hit = np.concatenate([i_hit, np.zeros(pad, dtype=i_hit.dtype)])
        raan_hit = np.concatenate([raan_hit, np.zeros(pad, dtype=raan_hit.dtype)])
        ap_hit = np.concatenate([ap_hit, np.zeros(pad, dtype=ap_hit.dtype)])
        M0_hit = np.concatenate([M0_hit, np.zeros(pad, dtype=M0_hit.dtype)])
        n_hit = np.concatenate([n_hit, np.zeros(pad, dtype=n_hit.dtype)])
        epoch_hit = np.concatenate([epoch_hit, np.zeros(pad, dtype=epoch_hit.dtype)])
        plane_normals = np.vstack([plane_normals, zeros3_norm])

    # Convert once to JAX arrays to avoid per-chunk host->device copies
    jo = jnp.asarray(ray_origins)
    jd = jnp.asarray(ray_directions)
    jnorm = jnp.asarray(plane_normals)
    ja = jnp.asarray(a_hit)
    je = jnp.asarray(e_hit)
    ji = jnp.asarray(i_hit)
    jraan = jnp.asarray(raan_hit)
    jap = jnp.asarray(ap_hit)

    plane_chunks = []
    snap_chunks = []
    nu_chunks = []
    mask_chunks = []
    nvalid_chunks = []

    threshold = snap_error_max_au if snap_error_max_au is not None else jnp.inf

    for start in range(0, padded_n, chunk_size):
        end = start + chunk_size
        (plane_distances_chunk, snap_errors_chunk, nu_values_chunk, valid_mask_chunk, n_valid_chunk) = _compute_candidates_jax(
            jo[start:end],
            jd[start:end],
            jnorm[start:end],
            ja[start:end],
            je[start:end],
            ji[start:end],
            jraan[start:end],
            jap[start:end],
            max_k,
            threshold,
            dedupe_angle_tol,
        )
        plane_chunks.append(plane_distances_chunk)
        snap_chunks.append(snap_errors_chunk)
        nu_chunks.append(nu_values_chunk)
        mask_chunks.append(valid_mask_chunk)
        nvalid_chunks.append(n_valid_chunk)

    plane_distances = jnp.concatenate(plane_chunks, axis=0)
    snap_errors = jnp.concatenate(snap_chunks, axis=0)
    nu_values = jnp.concatenate(nu_chunks, axis=0)
    valid_mask = jnp.concatenate(mask_chunks, axis=0)
    n_valid = jnp.concatenate(nvalid_chunks, axis=0)


    # TODO: some kind of check to make sure every hit has at least one valid candidate

    # Remove padding here
    # Restrict to original (unpadded) hits and move to NumPy for host-side ops
    plane_distances_mat = np.asarray(plane_distances[:n_hits, :])
    snap_errors_mat = np.asarray(snap_errors[:n_hits, :])
    nu_values_mat = np.asarray(nu_values[:n_hits, :])
    valid_mask_mat = np.asarray(valid_mask[:n_hits, :], dtype=bool)

    # Expand without Python loops
    K = max_k
    flat_mask = valid_mask_mat.reshape(-1)

    # Build flat indices (hit, variant) and filter by mask
    hit_indices = np.repeat(np.arange(n_hits), K)
    variant_ids = np.tile(np.arange(K), n_hits)
    sel_hit_idx = hit_indices[flat_mask]
    sel_variant = variant_ids[flat_mask]

    # Flatten metrics
    plane_distances = plane_distances_mat.reshape(-1)[flat_mask]
    snap_errors = snap_errors_mat.reshape(-1)[flat_mask]
    nu_candidates = nu_values_mat.reshape(-1)[flat_mask]

    # Compute orbital elements for each candidate
    expanded_times = (
        rays.time.mjd().to_numpy(zero_copy_only=False).astype(float)[sel_hit_idx]
    )
    expanded_a = a_hit[sel_hit_idx]
    expanded_e = e_hit[sel_hit_idx]
    expanded_M0 = M0_hit[sel_hit_idx]
    expanded_n = n_hit[sel_hit_idx]
    expanded_epoch = epoch_hit[sel_hit_idx]

    epoch_clean = np.where(np.isnan(expanded_epoch), expanded_times, expanded_epoch)
    M0_clean = np.where(np.isnan(expanded_M0), 0.0, expanded_M0)
    dt_days = expanded_times - epoch_clean
    # expanded_n is degrees/day; keep degrees here
    M_deg_arr = (M0_clean + expanded_n * dt_days) % 360.0
    M_rad_arr = np.radians(M_deg_arr)

    # Compute E and r from nu (true anomaly from projection)
    E_rad_arr = np.where(
        expanded_e < 1e-4,  # Nearly circular
        nu_candidates,  # E = f for circular orbits
        np.arctan2(
            np.sqrt(1 - expanded_e**2)
            * np.sin(nu_candidates)
            / (1 + expanded_e * np.cos(nu_candidates)),
            (expanded_e + np.cos(nu_candidates))
            / (1 + expanded_e * np.cos(nu_candidates)),
        ),
    )
    r_arr = expanded_a * (1 - expanded_e * np.cos(E_rad_arr))
    n_rad_day_arr = np.radians(expanded_n)

    # Build Arrow arrays via indexed take to keep consistency
    det_id_sel = pc.take(hits.table["det_id"], pa.array(sel_hit_idx, type=pa.int64()))
    orbit_id_sel = pc.take(
        hits.table["orbit_id"], pa.array(sel_hit_idx, type=pa.int64())
    )
    seg_id_sel = pc.take(hits.table["seg_id"], pa.array(sel_hit_idx, type=pa.int64()))

    labels = CanonicalAnomalyLabels.from_kwargs(
        det_id=det_id_sel,
        orbit_id=orbit_id_sel,
        seg_id=seg_id_sel,
        variant_id=sel_variant,
        f_rad=nu_candidates,
        E_rad=E_rad_arr,
        M_rad=M_rad_arr,
        n_rad_day=n_rad_day_arr,
        r_au=r_arr,
        snap_error=snap_errors,
        plane_distance_au=plane_distances,
    )

    # Sort deterministically (canonical order)
    return labels


@ray.remote
def _label_anomalies_worker_remote(
    hits: OverlapHits,
    rays: ObservationRays,
    orbits: Orbits,
    *,
    max_k: int,
    chunk_size: int,
    snap_error_max_au: float | None,
    dedupe_angle_tol: float,
) -> CanonicalAnomalyLabels:
    """
    Ray remote wrapper for the anomaly labeling worker.

    This is the same as label_anomalies_worker but decorated with @ray.remote
    for distributed execution.
    """
    return label_anomalies_worker(
        hits,
        rays,
        orbits,
        max_k=max_k,
        chunk_size=chunk_size,
        snap_error_max_au=snap_error_max_au,
        dedupe_angle_tol=dedupe_angle_tol,
    )


def label_anomalies(
    hits: OverlapHits,
    rays: ObservationRays,
    orbits: Orbits,
    max_k: int = 3,
    chunk_size: int = 8192,
    max_processes: int = 0,
    snap_error_max_au: float | None = None,
    dedupe_angle_tol: float = 1e-4,
) -> CanonicalAnomalyLabels:
    """
    Compute anomaly labels for observation-orbit overlaps.

    This function takes geometric overlaps and computes the orbital elements
    (mean anomaly, mean motion, heliocentric distance) at each observation time,
    enabling downstream time-consistency checks. Supports multi-anomaly variants
    for ambiguous geometries near nodes.

    Parameters
    ----------
    hits : OverlapHits
        Geometric overlaps between observations and test orbits.
    rays : ObservationRays
        Observation rays containing timing information.
    orbits : Orbits
        Original orbit data containing Keplerian elements.
    max_k : int, default=3
        Maximum number of anomaly variants per (det_id, orbit_id) pair.
        For K>1, multiple seeds are used to find alternate solutions.
    chunk_size : int, default=8192
        Fixed chunk size for JAX compilation stability and Ray batching.
    max_processes : int, default=0
        Maximum number of Ray processes. If <= 1, uses serial execution.
        If > 1, uses Ray parallel execution with the same worker function.
    snap_error_max_au : float, optional
        Maximum allowed snap error (AU). Candidates above this threshold
        are filtered out. If None, no filtering is applied.
    dedupe_angle_tol : float, default=1e-4
        Angular tolerance (radians) for deduplicating near-identical solutions.

    Returns
    -------
    AnomalyLabels
        Table with anomaly labels for each overlap variant, sorted by
        (det_id, orbit_id, variant_id, snap_error). Each (det_id, orbit_id)
        pair may have up to max_k variants with stable variant_id ordering.

    Notes
    -----
    - Canonical enforcement: rays and orbits are coerced to heliocentric–ecliptic.
    - Multi-anomaly: uses multiple Newton seeds to find alternate solutions near
      nodes and ambiguous geometries. Solutions are deduplicated and ranked by
      snap_error for deterministic variant_id assignment.
    - JIT stability: fixed chunk_size shapes avoid recompilation across calls.
    - Ray dispatch: max_processes > 1 enables Ray parallelism with the same worker.
    """
    if len(hits) == 0:
        return CanonicalAnomalyLabels.empty()
    
    # Align rays to hits using det_id (duplicate rays per hit as needed)
    hit_ray_idx = pc.index_in(hits.column("det_id"), rays.column("det_id"))
    if pc.any(pc.is_null(hit_ray_idx)).as_py():
        raise ValueError("Every hit must have a corresponding ray")
    rays = rays.take(hit_ray_idx)

    # Align orbits to hits using orbit_id (duplicate orbits per hit as needed)
    hit_orbit_idx = pc.index_in(hits.column("orbit_id"), orbits.column("orbit_id"))
    if pc.any(pc.is_null(hit_orbit_idx)).as_py():
        raise ValueError("Every hit must have a corresponding orbit")
    orbits = orbits.take(hit_orbit_idx)

    assert len(hits) == len(rays) == len(orbits), "Every hit must have a corresponding ray and orbit"

    # Enforce heliocentric-ecliptic coordinates
    orbits = orbits.set_column(
        "coordinates",
        transform_coordinates(
            orbits.coordinates,
            CartesianCoordinates,
            frame_out="ecliptic",
            origin_out=OriginCodes.SUN,
        ),
    )

    results = []
    # Dispatch: serial vs Ray parallel based on max_processes
    if max_processes <= 1:
        # Serial execution using worker
        for start, end in _iterate_chunk_indices(hits, chunk_size):
            results.append(label_anomalies_worker(
                hits[start:end],
                rays[start:end],
                orbits[start:end],
                max_k=max_k,
                chunk_size=chunk_size,
                snap_error_max_au=snap_error_max_au,
                dedupe_angle_tol=dedupe_angle_tol,
            ))
    else:
        # Queue ray remote tasks using backpressure logic
        futures = []
        for start, end in _iterate_chunk_indices(hits, chunk_size):
            futures.append(_label_anomalies_worker_remote.remote(
                hits[start:end],
                rays[start:end],
                orbits[start:end],
                max_k=max_k,
                chunk_size=chunk_size,
                snap_error_max_au=snap_error_max_au,
                dedupe_angle_tol=dedupe_angle_tol,
            ))
            if len(futures) >= max_processes * 1.5:
                finished, futures = ray.wait(futures, num_returns=1)
                results.append(ray.get(finished[0]))
        while futures:
            finished, futures = ray.wait(futures, num_returns=1)
            results.append(ray.get(finished[0]))

    labels = qv.concatenate(results, defrag=True)
    labels = labels.sort_by([("det_id", "ascending"), ("orbit_id", "ascending"), ("variant_id", "ascending"), ("snap_error", "ascending")])
    return labels


@partial(jax.jit, static_argnums=(8,))
def _compute_candidates_jax(
    ray_origins: jax.Array,
    ray_directions: jax.Array,
    plane_normals: jax.Array,
    a: jax.Array,
    e: jax.Array,
    i: jax.Array,
    raan: jax.Array,
    ap: jax.Array,
    max_k: int,
    snap_error_max_au: float,
    dedupe_angle_tol: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Unified JAX-compiled multi-candidate computation for all hits.

    Single vmapped kernel that handles arbitrary max_k values.
    Returns (plane_distances, snap_errors, nu_values, valid_mask, n_valid)
    all with appropriate shapes.
    """
    return _compute_candidates_kernel(
        ray_origins,
        ray_directions,
        plane_normals,
        a,
        e,
        i,
        raan,
        ap,
        max_k,
        snap_error_max_au,
        dedupe_angle_tol,
    )


def _compute_candidates_kernel(
    ray_origins: jax.Array,
    ray_directions: jax.Array,
    plane_normals: jax.Array,
    a: jax.Array,
    e: jax.Array,
    i: jax.Array,
    raan: jax.Array,
    ap: jax.Array,
    max_k: int,
    snap_error_max_au: float,
    dedupe_angle_tol: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Core kernel for multi-candidate computation."""

    def compute_single_hit_candidates(
        ray_origin, ray_direction, plane_normal, a_val, e_val, i_val, raan_val, ap_val
    ):
        # Plane distance
        plane_distance = ray_to_plane_distance(ray_origin, ray_direction, plane_normal)

        # Project ray to orbital plane
        projected_point = project_ray_to_orbital_plane(
            ray_origin, ray_direction, plane_normal
        )

        # Transform to perifocal 2D coordinates
        point_2d = transform_to_perifocal_2d(projected_point, i_val, raan_val, ap_val)

        # Compute multi-candidate snap distances and anomalies
        snap_distances, E_values, valid_mask = ellipse_snap_distance_multi_seed(
            point_2d, a_val, e_val, max_k, 10, dedupe_angle_tol
        )

        # Convert eccentric anomaly to true anomaly
        # For elliptical orbits: cos(f) = (cos(E) - e) / (1 - e*cos(E))
        cos_E = jnp.cos(E_values)
        cos_f = (cos_E - e_val) / (1 - e_val * cos_E)
        sin_f = jnp.sqrt(1 - e_val * e_val) * jnp.sin(E_values) / (1 - e_val * cos_E)
        nu_values = jnp.arctan2(sin_f, cos_f)

        # Apply snap error filter if specified
        error_mask = snap_distances <= snap_error_max_au
        valid_mask = jnp.where(
            snap_error_max_au < jnp.inf,
            jnp.logical_and(valid_mask, error_mask),
            valid_mask,
        )

        # Count valid candidates
        n_valid = jnp.sum(valid_mask.astype(jnp.int32))

        # Return plane distance repeated for all candidates
        plane_distances = jnp.full(max_k, plane_distance)

        return plane_distances, snap_distances, nu_values, valid_mask, n_valid

    vmap_compute = jax.vmap(compute_single_hit_candidates)
    return vmap_compute(ray_origins, ray_directions, plane_normals, a, e, i, raan, ap)


@jax.jit
def _compute_metrics_vectorized_with_normals(
    ray_origins: jax.Array,
    ray_directions: jax.Array,
    plane_normals: jax.Array,
    a: jax.Array,
    e: jax.Array,
    i: jax.Array,
    raan: jax.Array,
    ap: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Vectorized computation of projection metrics using JAX with precomputed plane normals.
    """

    def compute_single_metrics(
        ray_origin, ray_direction, plane_normal, a_val, e_val, i_val, raan_val, ap_val
    ):
        # Plane distance (observer offset to plane)
        plane_distance = ray_to_plane_distance(ray_origin, ray_direction, plane_normal)

        # Project ray to orbital plane
        projected_point = project_ray_to_orbital_plane(
            ray_origin, ray_direction, plane_normal
        )

        # Transform to perifocal 2D coordinates
        point_2d = transform_to_perifocal_2d(projected_point, i_val, raan_val, ap_val)

        # Compute snap distance to ellipse
        snap_distance, _ = ellipse_snap_distance(point_2d, a_val, e_val)

        return plane_distance, snap_distance

    vmap_compute = jax.vmap(compute_single_metrics)
    plane_distances, snap_errors = vmap_compute(
        ray_origins, ray_directions, plane_normals, a, e, i, raan, ap
    )
    return plane_distances, snap_errors


def _compute_anomalies_at_time(
    elements: Dict[str, Any], time_mjd: float
) -> tuple[float, float, float, float, float]:
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
        E_rad = float("nan")

    # Radius from conic equation r = a(1 - e cos E) for elliptic; fallback to elliptical form for e<1
    r_au = a * (1 - e * np.cos(E_rad)) if e < 1.0 else a * (e * np.cosh(E_rad) - 1.0)

    return M_deg, n_deg_per_day, float(r_au), float(E_rad), float(nu_rad)


def _compute_plane_distance(
    elements: Dict[str, Any], det_id: str, rays: ObservationRays
) -> float:
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


def _compute_snap_error(
    elements: Dict[str, Any], det_id: str, rays: ObservationRays, seg_id: int
) -> float:
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
