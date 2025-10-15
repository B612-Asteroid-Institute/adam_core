"""
Kepler clock gating: build time-consistent edges between anomaly-labeled detections.

Public API produces quivr tables and follows adam_core chunking/Ray patterns.
"""

from __future__ import annotations

from typing import Callable, Dict, List, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import ray
from numba import njit

from ..coordinates.keplerian import KeplerianCoordinates
from ..orbits.orbits import Orbits
from ..ray_cluster import initialize_use_ray
from .anomaly import AnomalyLabels
from .rays import ObservationRays
from ..utils.iter import _iterate_chunks

__all__ = [
    "ClockGatingCandidates",
    "ClockGatedEdges",
    "KeplerChains",
    "KeplerChainMembers",
    "prepare_clock_gating_candidates",
    "build_clock_gated_edges",
    "extract_kepler_chains",
    "kepler_clock_gate",
]

# Edge streaming: flush edges when accumulator grows beyond this many pairs
_EDGES_FLUSH_THRESHOLD = 500_000


class ClockGatingCandidates(qv.Table):
    """
    Per-orbit candidate vertices used by Kepler clock gating.

    Each row corresponds to an anomaly-labeled detection augmented with
    observation time and orbit eccentricity. Downstream, rows are grouped by
    `orbit_id` and only time-consistent pairs are connected.
    """

    det_id = qv.LargeStringColumn()
    orbit_id = qv.LargeStringColumn()
    seg_id = qv.Int32Column()
    variant_id = qv.Int32Column()

    time_tdb_mjd = qv.Float64Column()
    night_id = qv.Int64Column()
    observer_code = qv.LargeStringColumn()

    # Anomalies and dynamics (Float32 for kernel performance/memory)
    M_rad = qv.Float32Column()
    n_rad_day = qv.Float32Column()
    f_rad = qv.Float32Column()
    e = qv.Float32Column()
    sigma_M_rad = qv.Float32Column()

    # Methods below provide precomputed helpers when needed

    def M_wrapped_rad_f32(self) -> pa.Array:
        """Wrapped mean anomaly as Float32 Arrow array."""
        M = self.M_rad
        if M is None:
            return pa.nulls(0, type=pa.float32())
        M_np = M.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        wrapped = np.mod(M_np, np.float32(2.0 * np.pi))
        neg = wrapped < 0.0
        if np.any(neg):
            wrapped[neg] += np.float32(2.0 * np.pi)
        return pa.array(wrapped)

    def dM_df_f32(self) -> pa.Array:
        """Precomputed dM/df(e,f) as Float32 Arrow array."""
        e_np = self.e.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        f_np = self.f_rad.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        return pa.array(_dM_df(e_np, f_np).astype(np.float32, copy=False))

    # Optional guards / hints
    t_hat_plane_x = qv.Float32Column()
    t_hat_plane_y = qv.Float32Column()


class ClockGatedEdges(qv.Table):
    """
    Accepted time-consistent edges between candidates for a specific orbit.

    An edge (i -> j) is included if j occurs later in time than i and the
    measured change in mean anomaly is consistent with advancing at the
    orbit's mean motion (within the configured tolerance policy).
    """

    orbit_id = qv.LargeStringColumn()
    i_index = qv.Int32Column()
    j_index = qv.Int32Column()
    k_revs = qv.Int16Column()
    dt_days = qv.Float32Column()
    same_night = qv.BooleanColumn()

    # Parameters recorded for provenance
    tau_min_minutes = qv.FloatAttribute(default=0.0)
    alpha_min_per_day = qv.FloatAttribute(default=0.0)
    beta = qv.FloatAttribute(default=0.0)
    gamma = qv.FloatAttribute(default=0.0)
    time_bin_minutes = qv.IntAttribute(default=0)
    max_bins_ahead = qv.IntAttribute(default=0)
    horizon_days = qv.FloatAttribute(default=0.0)
    per_night_cap = qv.IntAttribute(default=0)
    max_k_span = qv.IntAttribute(default=0)
    batch_size = qv.IntAttribute(default=0)
    window_size = qv.IntAttribute(default=0)


class KeplerChainMembers(qv.Table):
    orbit_id = qv.LargeStringColumn()
    chain_id = qv.Int64Column()
    cand_index = qv.Int32Column()


class KeplerChains(qv.Table):
    orbit_id = qv.LargeStringColumn()
    chain_id = qv.Int64Column()
    size = qv.Int32Column()
    t_min_mjd = qv.Float64Column()
    t_max_mjd = qv.Float64Column()


def _factorize_strings(values: pa.Array) -> pa.Array:
    """
    Arrow-native factorization: map each string to a dense integer category.

    Parameters
    ----------
    values
        Input Arrow string array.

    Returns
    -------
    pa.Array (int32)
        Category index in [0, num_unique).
    """
    # Why: Create dense category ids to build compact integer keys instead of
    # string joins; improves performance and memory for grouping/joining.
    unique_string_values = pc.unique(values)
    return pc.index_in(values, unique_string_values)


def _floor_days(mjd_values: pa.Array) -> pa.Array:
    """Floor Modified Julian Date values to integer days (Arrow-native)."""
    # Why: Day-level bucketing is used to define per-night groupings and
    # coarser time windows for throttling candidate edges.
    return pc.cast(pc.floor(mjd_values), pa.int64())


def _compute_night_id(observer_codes: pa.Array, time_mjd: pa.Array) -> pa.Array:
    """
    Compute a stable per-night grouping key from (observer_code, floor(MJD)).

    The key is an Int64 composed as: (code_index << 32) | (day & 0xFFFFFFFF).
    """
    # Why: Build a stable, compact per-night key to cheaply group detections
    # by (observer, night) without expensive string operations downstream.
    observer_code_category_index = _factorize_strings(observer_codes)
    day_floor_mjd = _floor_days(time_mjd)
    observer_code_idx_i64 = pc.cast(observer_code_category_index, pa.int64())
    day_low32_masked = pc.bit_wise_and(
        day_floor_mjd, pa.scalar(0xFFFFFFFF, type=pa.int64())
    )
    return pc.bit_wise_or(pc.shift_left(observer_code_idx_i64, 32), day_low32_masked)


def prepare_clock_gating_candidates(
    labels: AnomalyLabels,
    rays: ObservationRays,
    orbits: Orbits,
) -> ClockGatingCandidates:
    """
    Join anomaly labels with observation timing and orbit eccentricity.

    Arrow-native joins are used to avoid materializing large Python objects.

    Parameters
    ----------
    labels
        Anomaly labels for (det_id, orbit_id[, variant]).
    rays
        Observation rays providing `det_id`, `observer.code`, and times.
    orbits
        Orbits table used to align orbit-level eccentricity.

    Returns
    -------
    ClockGatingCandidates
        Candidate vertices for clock gating, grouped by `orbit_id` downstream.
    """
    if len(labels) == 0:
        return ClockGatingCandidates.empty()

    # Why: Attach time and observer context to labels by aligning on det_id, using
    # Arrow-native index/lookup to avoid Python loops and minimize copies.
    label_to_ray_index = pc.index_in(labels.det_id, rays.det_id)
    if pc.any(pc.is_null(label_to_ray_index)).as_py():
        raise ValueError("prepare_clock_gating_candidates: every label must have a ray")
    rays_for_labels = rays.take(label_to_ray_index)

    # Why: Use TDB MJD to keep time units consistent with mean motion (rad/day)
    # and other orbital elements; convert to numpy for downstream kernels.
    observation_time_tdb_mjd = (
        rays_for_labels.observer.coordinates.time.rescale("tdb")
        .mjd()
        .to_numpy(zero_copy_only=False)
    )

    # Why: Observer code is needed to compute per-night grouping keys.
    observer_codes_for_labels = rays_for_labels.observer.code

    # Why: Compute (observer, night) keys to throttle same-night edges.
    night_id_per_label = _compute_night_id(
        observer_codes_for_labels, pa.array(observation_time_tdb_mjd)
    )

    # Why: Downstream adaptive tolerance depends on eccentricity via dM/df.
    # Join label orbits to orbits table once and extract aligned eccentricities.
    if len(orbits) == 0:
        raise ValueError("prepare_clock_gating_candidates: orbits is empty")
    orbits_keplerian: KeplerianCoordinates = orbits.coordinates.to_keplerian()
    label_to_orbit_index = pc.index_in(labels.orbit_id, orbits.orbit_id)
    if pc.any(pc.is_null(label_to_orbit_index)).as_py():
        raise ValueError(
            "prepare_clock_gating_candidates: label orbit_id missing in orbits"
        )
    label_eccentricity = pc.take(
        pa.array(orbits_keplerian.e), label_to_orbit_index
    ).to_numpy(zero_copy_only=False)

    # Why: Build a contiguous Arrow-backed table once to feed vectorized kernels
    # without per-column copies.
    # Cast numeric columns to Float32 where appropriate and precompute helpers
    M_rad_f32 = labels.M_rad.cast("float32")
    n_rad_day_f32 = labels.n_rad_day.cast("float32")
    f_rad_f32 = labels.f_rad.cast("float32")
    e_f32 = pa.array(label_eccentricity).cast(pa.float32())
    sigma_M_rad_f32 = labels.sigma_M_rad.cast("float32")

    candidates_table = ClockGatingCandidates.from_kwargs(
        det_id=labels.det_id,
        orbit_id=labels.orbit_id,
        seg_id=labels.seg_id,
        variant_id=labels.variant_id,
        time_tdb_mjd=observation_time_tdb_mjd,
        night_id=night_id_per_label,
        observer_code=rays_for_labels.observer.code,
        M_rad=M_rad_f32,
        n_rad_day=n_rad_day_f32,
        f_rad=f_rad_f32,
        e=e_f32,
        sigma_M_rad=sigma_M_rad_f32,
        t_hat_plane_x=labels.t_hat_plane_x,
        t_hat_plane_y=labels.t_hat_plane_y,
    )

    # Why: Defragment to ensure arrays are contiguous for fast downstream access.
    if candidates_table.fragmented():
        candidates_table = qv.defragment(candidates_table)
    return candidates_table


# ---- Edge building: declarations (implemented below) ----


def build_clock_gated_edges(
    candidates: ClockGatingCandidates,
    *,
    tau_min_minutes: float = 15.0,
    alpha_min_per_day: float = 0.02,
    beta: float = 1.0,
    gamma: float = 0.5,
    time_bin_minutes: int = 120,
    max_bins_ahead: int = 72,
    horizon_days: float = 90.0,
    per_night_cap: int = 0,
    max_k_span: int = 1,
    kernel_col_window_size: int = 2048,
    max_processes: int | None = 1,
    intra_bin_chunk_size: int = 100,
    inter_bin_chunk_size: int = 1000,
) -> ClockGatedEdges:
    """
    Build time-consistent edges between candidate detections for each orbit.

    The algorithm partitions candidates by time bins (e.g., 60-minute bins)
    and for each (bin_i, bin_j) pair within a lookahead window, evaluates the
    Kepler clock consistency criterion in a fully vectorized manner.

    Parameters
    ----------
    candidates
        Candidate vertices (single orbit).
    tau_min_minutes, alpha_min_per_day, beta, gamma
        Tolerance policy parameters (see _tau).
    time_bin_minutes
        Width of time bins used to limit pairwise comparisons.
    max_bins_ahead
        Number of subsequent bins to consider for each source bin (used with
        horizon_days; effective lookahead is limited by both).
    horizon_days
        Absolute lookahead horizon in days for inter-bin scanning.
    per_night_cap
        Maximum number of same-night outgoing edges per source detection.
    max_k_span
        Sweep size around the nearest integer revolutions k at bin center.
    kernel_col_window_size
        Number of destination columns (j) per kernel call; shapes are padded to
        this width to avoid JAX recompilation.
    max_processes
        If > 1, shard by orbit_id and run in parallel with Ray.
    """
    if len(candidates) == 0:
        return ClockGatedEdges.empty()

    assert len(candidates.orbit_id.unique()) == 1, "candidates must be a single orbit"

    # Build the bins and wrapped-M index directly from the candidates table
    bin_index = _build_bins_and_m_index(
        candidates,
        time_bin_minutes,
    )
    tolerance_scalars = _compute_per_detection_scalars(
        candidates,
        tau_min_minutes,
        alpha_min_per_day,
    )

    bin_pairs = calculate_bin_pairs(bin_index, max_bins_ahead, horizon_days)
    futures = []
    all_edges_list = []

    if max_processes >= 2:
        initialize_use_ray(num_cpus=max_processes)
        candidates_ref = ray.put(candidates)
        # bin_index_ref = ray.put(bin_index)

    for bin_id_chunk in _iterate_chunks(bin_index.unique_bins, chunk_size=intra_bin_chunk_size):
        if max_processes < 2:
            per_bin_intra_bin_edges = intra_bin_edges_worker(
                candidates,
                bin_index,
                tolerance_scalars,
                bin_id_chunk,
                beta=beta,
                gamma=gamma,
                horizon_days=horizon_days,
                per_night_cap=per_night_cap,
                kernel_col_window_size=kernel_col_window_size,
            )
            all_edges_list.append(per_bin_intra_bin_edges)
        else:
            futures.append(intra_bin_edges_worker_remote.remote(
                candidates_ref,
                bin_index,
                tolerance_scalars,
                bin_id_chunk,
                beta=beta,
                gamma=gamma,
                horizon_days=horizon_days,
                per_night_cap=per_night_cap,
                kernel_col_window_size=kernel_col_window_size,
            ))

            if len(futures) >= max_processes * 1.5:
                finished, futures = ray.wait(futures, num_returns=1)
                result = ray.get(finished[0])
                all_edges_list.append(result)

    for bin_pairs_chunk in _iterate_chunks(bin_pairs, chunk_size=inter_bin_chunk_size):
        if max_processes < 2:
            per_bin_inter_bin_edges = inter_bin_edges_worker(
                candidates,
                bin_index,
                tolerance_scalars,
                bin_pairs_chunk,
                beta=beta,
                gamma=gamma,
                horizon_days=horizon_days,
                per_night_cap=per_night_cap,
                kernel_col_window_size=kernel_col_window_size,
                max_k_span=max_k_span,
            )
            all_edges_list.append(per_bin_inter_bin_edges)
        else:
            futures.append(inter_bin_edges_worker_remote.remote(
                candidates_ref,
                bin_index,
                tolerance_scalars,
                bin_pairs_chunk,
                beta=beta,
                gamma=gamma,
                horizon_days=horizon_days,
                per_night_cap=per_night_cap,
                kernel_col_window_size=kernel_col_window_size,
                max_k_span=max_k_span,
            ))

            if len(futures) >= max_processes * 1.5:
                finished, futures = ray.wait(futures, num_returns=1)
                result = ray.get(finished[0])
                all_edges_list.append(result)

    while futures:
        finished, futures = ray.wait(futures, num_returns=1)
        result = ray.get(finished[0])
        all_edges_list.append(result)

    all_edges = qv.concatenate(all_edges_list)
    return all_edges

def kepler_clock_gate(
    candidates: ClockGatingCandidates,
    *,
    tau_min_minutes: float = 15.0,
    alpha_min_per_day: float = 0.02,
    beta: float = 1.0,
    gamma: float = 0.5,
    time_bin_minutes: int = 120,
    max_bins_ahead: int = 72,
    horizon_days: float = 90.0,
    per_night_cap: int = 0,
    max_k_span: int = 1,
    kernel_col_window_size: int = 2048,
    max_processes: int | None = 1,
    promote_min_size: int = 6,
    promote_min_span_days: float = 3.0,
    stream_to_disk: str | None = None,
) -> tuple[KeplerChains, KeplerChainMembers]:
    """
    Complete Kepler clock-gating stage: build edges and chains from candidates.

    Parameters
    ----------
    candidates
        Candidate vertices (one orbit at a time downstream).
    tau_min_minutes, alpha_min_per_day, beta, gamma
        Tolerance policy parameters (see _tau).
    time_bin_minutes
        Width of time bins used to limit pairwise comparisons.
    max_bins_ahead
        Number of subsequent bins to consider for each source bin (used with
        horizon_days; effective lookahead is limited by both).
    horizon_days
        Absolute lookahead horizon in days for inter-bin scanning.
    per_night_cap
        Maximum number of same-night outgoing edges per source detection.
    max_k_span
        Sweep size around the nearest integer revolutions k at bin center.
    kernel_col_window_size
        Number of destination columns (j) per kernel call; shapes are padded to
        this width to avoid JAX recompilation.
    max_processes
        If > 1, shard by orbit_id and run in parallel with Ray.
    promote_min_size
        Minimum size of a chain to promote.
    promote_min_span_days
        Minimum span of a chain to promote.
    stream_to_disk
        Optional directory path on disk to stream accumulated edges to avoid memory pressure.

    Returns
    -------
    chains
        Promoted chains.
    members
        Membership table for the chains.
    """

    # Since the edge finding and chain building
    # is specific to a single orbit, we will 
    # loop through the unique orbits and process them
    # independently.
    chains = KeplerChains.empty()
    members = KeplerChainMembers.empty()
    unique_orbits = candidates.orbit_id.unique()
    for orbit_id in unique_orbits:
        mask_for_orbit = pc.equal(candidates.orbit_id, orbit_id)
        candidates_for_orbit = candidates.apply_mask(mask_for_orbit)

        orbit_edges: ClockGatedEdges | str = build_clock_gated_edges(
            candidates_for_orbit,
            tau_min_minutes=tau_min_minutes,
            alpha_min_per_day=alpha_min_per_day,
            beta=beta,
            gamma=gamma,
            time_bin_minutes=time_bin_minutes,
            max_bins_ahead=max_bins_ahead,
            horizon_days=horizon_days,
            per_night_cap=per_night_cap,
            max_k_span=max_k_span,
            kernel_col_window_size=kernel_col_window_size,
            max_processes=max_processes,
        )
        orbit_chains, orbit_chain_members = extract_kepler_chains(
            candidates_for_orbit,
            orbit_edges,
            min_size=promote_min_size,
            min_span_days=promote_min_span_days,
        )
        chains = qv.concatenate([chains, orbit_chains])
        members = qv.concatenate([members, orbit_chain_members])
    return chains, members


# ------------------
# Internal utilities
# ------------------


def _wrap_2pi(angles_rad: np.ndarray) -> np.ndarray:
    """Wrap angles in radians to [0, 2π)."""
    # Why: Normalize periodic quantities so differences and comparisons are
    # consistent under 2π-periodicity.
    wrapped_angles = np.mod(angles_rad, 2.0 * np.pi)
    return np.where(wrapped_angles < 0.0, wrapped_angles + 2.0 * np.pi, wrapped_angles)


def _wrap_2pi_jax(angles_rad: jnp.ndarray) -> jnp.ndarray:
    # Why: JAX variant for use inside JIT-compiled kernels, mirroring numpy
    # behavior for 2π-periodic normalization.
    wrapped_angles = jnp.mod(angles_rad, 2.0 * jnp.pi)
    return jnp.where(
        wrapped_angles < 0.0, wrapped_angles + 2.0 * jnp.pi, wrapped_angles
    )


def _pair_kernel_impl(
    t_i: jnp.ndarray,
    M_i: jnp.ndarray,
    n_i: jnp.ndarray,
    sigmaMi_i: jnp.ndarray,
    dMdf_i: jnp.ndarray,
    t_j: jnp.ndarray,
    M_j: jnp.ndarray,
    tau_min_days: jnp.float32,
    alpha_days_per_day: jnp.float32,
    beta: jnp.float32,
    gamma: jnp.float32,
    horizon_days: jnp.float32,
    valid_i: jnp.ndarray,
    valid_j: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Shapes: t_i[B], t_j[W]; everything float32/int32
    # Why: Compute forward-in-time deltas within a finite horizon to limit
    # candidate pairs to physically plausible and computationally bounded windows.
    delta_time_days_matrix = t_j[None, :] - t_i[:, None]
    forward_time_mask = jnp.logical_and(
        delta_time_days_matrix > 0.0, delta_time_days_matrix <= horizon_days
    )

    # Why: Mean anomaly is 2π-periodic; wrap its difference before estimating
    # integer revolution counts to avoid branch cut artifacts.
    delta_wrapped_mean_anomaly_rad = _wrap_2pi_jax(M_j[None, :] - M_i[:, None])
    two_pi = jnp.float32(2.0 * jnp.pi)
    fractional_revolutions = (
        n_i[:, None] * delta_time_days_matrix - delta_wrapped_mean_anomaly_rad
    ) / two_pi
    rounded_revolutions = jnp.maximum(jnp.rint(fractional_revolutions), 0.0).astype(
        jnp.int32
    )

    # Why: Convert angular uncertainties to time by scaling with 1/n so τ has
    # units of days and is comparable to time residuals, especially for small n.
    tiny_epsilon = jnp.float32(1e-8)
    timing_residual_days = delta_time_days_matrix - (
        delta_wrapped_mean_anomaly_rad
        + two_pi * rounded_revolutions.astype(jnp.float32)
    ) / (n_i[:, None] + tiny_epsilon)
    inverse_mean_motion = 1.0 / (n_i[:, None] + tiny_epsilon)
    time_tolerance_days = (
        tau_min_days
        + alpha_days_per_day * delta_time_days_matrix
        + beta * (sigmaMi_i[:, None] * inverse_mean_motion)
        + gamma * (jnp.abs(dMdf_i[:, None]) * inverse_mean_motion)
    )

    # Why: Only accept pairs that move forward in time and whose clock residual
    # is consistent with the adaptive tolerance policy.
    keep_mask = jnp.logical_and(
        forward_time_mask, jnp.abs(timing_residual_days) <= time_tolerance_days
    )
    # Apply valid row/col masks (padding)
    keep_mask = jnp.logical_and(keep_mask, valid_i[:, None])
    keep_mask = jnp.logical_and(keep_mask, valid_j[None, :])
    return (
        keep_mask,
        rounded_revolutions,
        timing_residual_days,
        delta_time_days_matrix,
        delta_wrapped_mean_anomaly_rad,
    )


pair_kernel = jax.jit(_pair_kernel_impl, static_argnames=())


"""Removed unused helpers _choose_k and _clock_residual (now superseded by pair_kernel)."""


@njit(cache=True)
def _mband_union_mask_numba(
    M_sorted_j: np.ndarray,
    centers: np.ndarray,
    halfwidth: np.float32,
    two_pi: np.float32,
) -> np.ndarray:
    N = M_sorted_j.shape[0]
    out = np.zeros(N, dtype=np.bool_)
    for ci in range(centers.shape[0]):
        c = centers[ci]
        lo = c - halfwidth
        hi = c + halfwidth

        # Non-wrapping interval entirely inside [0, 2pi)
        if lo >= 0.0 and hi < two_pi:
            l = np.searchsorted(M_sorted_j, lo, side="left")
            r = np.searchsorted(M_sorted_j, hi, side="right")
            for idx in range(l, r):
                out[idx] = True
        else:
            # Left-wrap: lo < 0 => intervals [0, hi] and [lo+2pi, 2pi)
            if lo < 0.0:
                if hi > 0.0:
                    r1 = np.searchsorted(M_sorted_j, hi, side="right")
                    for idx in range(0, r1):
                        out[idx] = True
                lo2 = lo + two_pi
                l2 = np.searchsorted(M_sorted_j, lo2, side="left")
                for idx in range(l2, N):
                    out[idx] = True
            # Right-wrap: hi >= 2pi => intervals [0, hi-2pi] and [lo, 2pi)
            if hi >= two_pi:
                hi2 = hi - two_pi
                r1 = np.searchsorted(M_sorted_j, hi2, side="right")
                for idx in range(0, r1):
                    out[idx] = True
                if lo >= 0.0:
                    l2 = np.searchsorted(M_sorted_j, lo, side="left")
                else:
                    l2 = 0
                for idx in range(l2, N):
                    out[idx] = True
    return out


def _dM_df(eccentricity: np.ndarray, true_anomaly_rad: np.ndarray) -> np.ndarray:
    """Derivative dM/df for elliptical orbits (used for adaptive tolerance)."""
    # Why: Sensitivity of mean anomaly to true anomaly modulates the time
    # tolerance—higher sensitivity warrants larger tolerance.
    numerator = (1.0 - eccentricity * eccentricity) ** 1.5
    denominator = (1.0 + eccentricity * np.cos(true_anomaly_rad)) ** 2
    return numerator / (denominator + 1e-18)


def _tau(
    delta_time_days: np.ndarray,
    tau_min_days: float,
    alpha_days_per_day: float,
    beta_sigma_scale: float,
    sigma_mean_anomaly_rad: np.ndarray,
    gamma_dmdf_scale: float,
    dM_df_at_i: np.ndarray,
) -> np.ndarray:
    """
    Adaptive tolerance policy τ_ij in days.

    τ = τ_min + α Δt + β σ_M + γ |dM/df|.
    """
    # Why: Combine fixed, time-linear, and state-dependent terms to produce a
    # unit-correct time tolerance for the clock residual test.
    return (
        tau_min_days
        + alpha_days_per_day * delta_time_days
        + beta_sigma_scale * sigma_mean_anomaly_rad
        + gamma_dmdf_scale * np.abs(dM_df_at_i)
    )


# -------------------------------
# Refactor helpers (NamedTuples)
# -------------------------------


class KernelArrays(NamedTuple):
    # Deprecated: replaced by passing `ClockGatingCandidates` directly into workers.
    # Retained temporarily to ease transition; will be removed.
    time_tdb_mjd_days: np.ndarray
    mean_anomaly_rad: np.ndarray
    mean_motion_rad_per_day: np.ndarray
    true_anomaly_rad: np.ndarray
    eccentricity: np.ndarray
    sigma_mean_anomaly_rad: np.ndarray
    night_id: np.ndarray
    mean_anomaly_wrapped_rad: np.ndarray
    orbit_id_str: str


class BinIndex(NamedTuple):
    bin_width_days: float
    t0_min_days: float
    unique_bins: np.ndarray
    bin_to_unsorted_indices: Dict[int, np.ndarray]
    bin_to_time_sorted_indices: Dict[int, np.ndarray]
    bin_to_sorted_times: Dict[int, np.ndarray]
    bin_to_M_order_positions: Dict[int, np.ndarray]
    bin_to_sorted_M: Dict[int, np.ndarray]


class ToleranceScalars(NamedTuple):
    tau_min_days: float
    alpha_days_per_day: float
    dM_df_per_candidate: np.ndarray


class EdgeAccumulator(NamedTuple):
    source_candidate_indices: List[int]
    dest_candidate_indices: List[int]
    k_revolutions_rounded: List[int]
    delta_time_days: List[float]
    is_same_night: List[bool]
    outgoing_same_night_edge_counts: np.ndarray


def _extract_kernel_arrays(candidates_for_orbit: ClockGatingCandidates) -> KernelArrays:
    """Compatibility helper: build per-orbit numpy arrays for kernels from the table.

    Returns float32 arrays and per-orbit relative times for numerical stability.
    """
    compute_dtype = np.float32
    # Absolute time (float64 for precise min), then shift to local t0 and cast to f32
    absolute_time_mjd = candidates_for_orbit.time_tdb_mjd.to_numpy(
        zero_copy_only=False
    ).astype(np.float64)
    min_time_mjd = (
        float(np.min(absolute_time_mjd)) if absolute_time_mjd.size > 0 else 0.0
    )
    relative_time_days = (absolute_time_mjd - min_time_mjd).astype(compute_dtype)

    mean_anomaly_rad = candidates_for_orbit.M_rad.to_numpy(zero_copy_only=False).astype(
        compute_dtype
    )
    mean_motion_rad_per_day = candidates_for_orbit.n_rad_day.to_numpy(
        zero_copy_only=False
    ).astype(compute_dtype)
    true_anomaly_rad = candidates_for_orbit.f_rad.to_numpy(zero_copy_only=False).astype(
        compute_dtype
    )
    eccentricity = candidates_for_orbit.e.to_numpy(zero_copy_only=False).astype(
        compute_dtype
    )
    sigma_mean_anomaly_rad = candidates_for_orbit.sigma_M_rad.to_numpy(
        zero_copy_only=False
    ).astype(compute_dtype)
    night_id = candidates_for_orbit.night_id.to_numpy(zero_copy_only=False).astype(
        np.int64
    )
    orbit_id_value = (
        candidates_for_orbit.orbit_id[0].as_py()
        if len(candidates_for_orbit) > 0
        else ""
    )

    mean_anomaly_wrapped_rad = candidates_for_orbit.M_wrapped_rad_f32().to_numpy(
        zero_copy_only=False
    ).astype(np.float32, copy=False)

    return KernelArrays(
        time_tdb_mjd_days=relative_time_days,
        mean_anomaly_rad=mean_anomaly_rad,
        mean_motion_rad_per_day=mean_motion_rad_per_day,
        true_anomaly_rad=true_anomaly_rad,
        eccentricity=eccentricity,
        sigma_mean_anomaly_rad=sigma_mean_anomaly_rad,
        night_id=night_id,
        mean_anomaly_wrapped_rad=mean_anomaly_wrapped_rad,
        orbit_id_str=orbit_id_value,
    )


def _build_bins_and_m_index(
    candidates_for_orbit: ClockGatingCandidates,
    time_bin_minutes: int,
) -> BinIndex:
    """Build time bins and per-bin wrapped-M ordering/index for M-band queries."""
    # Why: Coarse time bins bound pairwise comparisons; per-bin M ordering
    # enables fast interval queries in wrapped mean anomaly space.
    # Build relative times on-the-fly for binning (Float32) and use precomputed wrapped M
    time_mjd_abs = candidates_for_orbit.time_tdb_mjd.to_numpy(zero_copy_only=False).astype(np.float64)
    bin_width_days = float(time_bin_minutes) / 1440.0
    t0_min_days = float(np.min(time_mjd_abs)) if time_mjd_abs.size > 0 else 0.0
    time_tdb_mjd_days = (time_mjd_abs - t0_min_days).astype(np.float32, copy=False)
    mean_anomaly_wrapped_rad = candidates_for_orbit.M_wrapped_rad_f32().to_numpy(zero_copy_only=False).astype(np.float32, copy=False)

    bin_index = np.floor((time_tdb_mjd_days - 0.0) / bin_width_days).astype(
        np.int64
    )
    indices_sorted_by_bin = np.argsort(bin_index, kind="stable")
    bin_indices_sorted = bin_index[indices_sorted_by_bin]
    unique_bins, first_index_per_bin, count_per_bin = np.unique(
        bin_indices_sorted, return_index=True, return_counts=True
    )

    bin_to_unsorted_indices: Dict[int, np.ndarray] = {}
    bin_to_time_sorted_indices: Dict[int, np.ndarray] = {}
    bin_to_sorted_times: Dict[int, np.ndarray] = {}
    bin_to_M_order_positions: Dict[int, np.ndarray] = {}
    bin_to_sorted_M: Dict[int, np.ndarray] = {}

    for position_in_unique, bin_id in enumerate(unique_bins):
        start = first_index_per_bin[position_in_unique]
        end = start + count_per_bin[position_in_unique]
        # Why: Recover original indices for this bin, then sort within-bin by time
        # for streaming, and by wrapped M for M-band queries.
        idx_unsorted = indices_sorted_by_bin[start:end]
        if idx_unsorted.size == 0:
            continue
        order_within_bin_by_time = np.argsort(
            time_tdb_mjd_days[idx_unsorted], kind="stable"
        )
        idx_time_sorted = idx_unsorted[order_within_bin_by_time]
        times_sorted_within_bin = time_tdb_mjd_days[idx_time_sorted]
        wrapped_M_unsorted_for_bin = mean_anomaly_wrapped_rad[idx_unsorted]
        order_positions_by_wrapped_M = np.argsort(
            wrapped_M_unsorted_for_bin, kind="stable"
        )
        wrapped_M_sorted = wrapped_M_unsorted_for_bin[order_positions_by_wrapped_M]

        b_int = int(bin_id)
        bin_to_unsorted_indices[b_int] = idx_unsorted
        bin_to_time_sorted_indices[b_int] = idx_time_sorted
        bin_to_sorted_times[b_int] = times_sorted_within_bin
        bin_to_M_order_positions[b_int] = order_positions_by_wrapped_M
        bin_to_sorted_M[b_int] = wrapped_M_sorted

    return BinIndex(
        bin_width_days=bin_width_days,
        t0_min_days=t0_min_days,
        unique_bins=unique_bins,
        bin_to_unsorted_indices=bin_to_unsorted_indices,
        bin_to_time_sorted_indices=bin_to_time_sorted_indices,
        bin_to_sorted_times=bin_to_sorted_times,
        bin_to_M_order_positions=bin_to_M_order_positions,
        bin_to_sorted_M=bin_to_sorted_M,
    )


def _compute_per_detection_scalars(
    candidates_for_orbit: ClockGatingCandidates,
    tau_min_minutes: float,
    alpha_min_per_day: float,
) -> ToleranceScalars:
    # Why: Convert policy parameters to days to match kernel units.
    tau_min_days = float(tau_min_minutes) / 1440.0
    alpha_days_per_day = float(alpha_min_per_day) / 1440.0
    # Use method-based precompute for dM/df from the candidates table (Float32)
    dM_df_per_detection = candidates_for_orbit.dM_df_f32().to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    return ToleranceScalars(
        tau_min_days=tau_min_days,
        alpha_days_per_day=alpha_days_per_day,
        dM_df_per_candidate=dM_df_per_detection,
    )


def _init_edge_accumulator(num_candidates: int) -> EdgeAccumulator:
    # Why: Centralized constructor to keep accumulator shape consistent and
    # initialize per-source same-night quotas.
    return EdgeAccumulator(
        source_candidate_indices=[],
        dest_candidate_indices=[],
        k_revolutions_rounded=[],
        delta_time_days=[],
        is_same_night=[],
        outgoing_same_night_edge_counts=np.zeros(num_candidates, dtype=np.int32),
    )


def intra_bin_edges_worker(
    candidates: ClockGatingCandidates,
    bin_index: BinIndex,
    tolerance_scalars: ToleranceScalars,
    bin_ids: List[int],
    *,
    beta: float,
    gamma: float,
    horizon_days: float,
    per_night_cap: int,
    kernel_col_window_size: int,
) -> ClockGatedEdges:
    """
    Worker function for building clock-gated edges within a single bin.

    Returns a `ClockGatedEdges` table built of intra-bin edges,
    following the adaptive τ policy and band-limited M queries.
    """
    # One-time numpy views for required columns
    time_rel_days = (
        candidates.time_tdb_mjd.to_numpy(zero_copy_only=False).astype(np.float64)
    )
    if time_rel_days.size > 0:
        t0 = float(np.min(time_rel_days))
        time_rel_days = (time_rel_days - t0).astype(np.float32, copy=False)
    else:
        time_rel_days = time_rel_days.astype(np.float32, copy=False)
    M_wrapped = candidates.M_wrapped_rad_f32().to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    n_rad_day = candidates.n_rad_day.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    sigma_M = candidates.sigma_M_rad.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    night_id = candidates.night_id.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)

    edge_accumulator = _init_edge_accumulator(len(time_rel_days))
    
    for bin_id in bin_ids:
        b_int = int(bin_id)
        indices_in_bin_sorted_by_time = bin_index.bin_to_time_sorted_indices.get(b_int)
        if indices_in_bin_sorted_by_time is None or indices_in_bin_sorted_by_time.size == 0:
            return ClockGatedEdges.empty()

        num_candidates_in_bin = int(indices_in_bin_sorted_by_time.size)
        # Why: Preallocate destination buffers once per bin and reuse across windows.
        dest_time_buffer = np.zeros((int(kernel_col_window_size),), dtype=np.float32)
        dest_wrapped_M_buffer = np.zeros((int(kernel_col_window_size),), dtype=np.float32)
        dest_valid_mask = np.zeros((int(kernel_col_window_size),), dtype=bool)

        for source_pos_in_bin in range(num_candidates_in_bin):
            source_index = int(indices_in_bin_sorted_by_time[source_pos_in_bin])
            source_time_array = time_rel_days[
                source_index : source_index + 1
            ]
            source_wrapped_M_array = M_wrapped[
                source_index : source_index + 1
            ]
            source_mean_motion_array = n_rad_day[
                source_index : source_index + 1
            ]
            source_sigma_M_array = sigma_M[
                source_index : source_index + 1
            ]
            source_dMdf_array = tolerance_scalars.dM_df_per_candidate[
                source_index : source_index + 1
            ]
            source_valid_mask = np.asarray([True], dtype=bool)

            for j_block_start in range(
                source_pos_in_bin + 1, num_candidates_in_bin, int(kernel_col_window_size)
            ):
                j_block_end = min(
                    num_candidates_in_bin, j_block_start + int(kernel_col_window_size)
                )
                dest_indices_window = indices_in_bin_sorted_by_time[
                    j_block_start:j_block_end
                ]
                num_dest_in_window = j_block_end - j_block_start
                if num_dest_in_window > 0:
                    dest_time_buffer[:num_dest_in_window] = time_rel_days[dest_indices_window]
                    dest_wrapped_M_buffer[:num_dest_in_window] = M_wrapped[dest_indices_window]
                if num_dest_in_window < int(kernel_col_window_size):
                    dest_time_buffer[num_dest_in_window:] = 0.0
                    dest_wrapped_M_buffer[num_dest_in_window:] = 0.0
                dest_valid_mask[:] = False
                dest_valid_mask[:num_dest_in_window] = True

                (
                    keep_mask,
                    rounded_revs_matrix,
                    _resid_unused,
                    delta_time_days_matrix,
                    _dM_unused,
                ) = pair_kernel(
                    source_time_array,
                    source_wrapped_M_array,
                    source_mean_motion_array,
                    source_sigma_M_array,
                    source_dMdf_array,
                    dest_time_buffer,
                    dest_wrapped_M_buffer,
                    tolerance_scalars.tau_min_days,
                    tolerance_scalars.alpha_days_per_day,
                    beta,
                    gamma,
                    horizon_days,
                    valid_i=source_valid_mask,
                    valid_j=dest_valid_mask,
                )
                # Convert kernel outputs once per window (drop resid/dM)
                keep_np = np.asarray(keep_mask)
                rounded_revs_np = np.asarray(rounded_revs_matrix)
                dt_np = np.asarray(delta_time_days_matrix)
                if not np.any(keep_np):
                    continue
                accepted_dest_cols = np.nonzero(keep_np[0])[0]
                if accepted_dest_cols.size == 0:
                    continue
                source_indices_global = np.full(
                    accepted_dest_cols.shape[0], source_index, dtype=np.int32
                )
                dest_indices_global = dest_indices_window[accepted_dest_cols]
                same_night_mask = (night_id[dest_indices_global] == night_id[source_index])
                k_revs_for_kept = rounded_revs_np[0, accepted_dest_cols]
                delta_time_days_kept = dt_np[0, accepted_dest_cols]

                keep_after_quota_mask = np.ones(source_indices_global.shape[0], dtype=bool)
                if per_night_cap > 0 and np.any(same_night_mask):
                    remaining_same_night_quota = (
                        per_night_cap
                        - edge_accumulator.outgoing_same_night_edge_counts[source_index]
                    )

                    if remaining_same_night_quota <= 0:
                        keep_after_quota_mask[same_night_mask] = False
                    else:
                        same_night_true_indices = np.nonzero(same_night_mask)[0]
                        if same_night_true_indices.size > remaining_same_night_quota:
                            keep_after_quota_mask[
                                same_night_true_indices[remaining_same_night_quota:]
                            ] = False
                        edge_accumulator.outgoing_same_night_edge_counts[
                            source_index
                        ] += min(remaining_same_night_quota, same_night_true_indices.size)

                # apply mask and append in bulk
                if np.any(keep_after_quota_mask):
                    edge_accumulator.source_candidate_indices.extend(
                        source_indices_global[keep_after_quota_mask].tolist()
                    )
                    edge_accumulator.dest_candidate_indices.extend(
                        dest_indices_global[keep_after_quota_mask].astype(np.int32).tolist()
                    )
                    edge_accumulator.k_revolutions_rounded.extend(
                        k_revs_for_kept[keep_after_quota_mask].astype(np.int16).tolist()
                    )
                    edge_accumulator.delta_time_days.extend(
                        delta_time_days_kept[keep_after_quota_mask]
                        .astype(np.float32)
                        .tolist()
                    )
                    edge_accumulator.is_same_night.extend(
                        same_night_mask[keep_after_quota_mask].tolist()
                    )

    orbit_id_str = candidates.orbit_id.unique()[0].as_py() if len(candidates) > 0 else ""
    clock_gated_edges = ClockGatedEdges.from_kwargs(
        orbit_id=[orbit_id_str] * len(edge_accumulator.source_candidate_indices),
        i_index=edge_accumulator.source_candidate_indices,
        j_index=edge_accumulator.dest_candidate_indices,
        k_revs=edge_accumulator.k_revolutions_rounded,
        dt_days=edge_accumulator.delta_time_days,
        same_night=edge_accumulator.is_same_night,
        tau_min_minutes=tolerance_scalars.tau_min_days * 60.0,
        alpha_min_per_day=tolerance_scalars.alpha_days_per_day * 60.0,
        beta=beta,
        gamma=gamma,
        window_size=kernel_col_window_size,
    )

    return clock_gated_edges


@ray.remote
def intra_bin_edges_worker_remote(
    candidates: ClockGatingCandidates,
    bin_index: BinIndex,
    tolerance_scalars: ToleranceScalars,
    bin_ids: List[int],
    *,
    beta: float,
    gamma: float,
    horizon_days: float,
    per_night_cap: int,
    kernel_col_window_size: int,
) -> ClockGatedEdges:
    return intra_bin_edges_worker(
        candidates,
        bin_index,
        tolerance_scalars,
        bin_ids,
        beta=beta,
        gamma=gamma,
        horizon_days=horizon_days,
        per_night_cap=per_night_cap,
        kernel_col_window_size=kernel_col_window_size,
    )

def inter_bin_edges_worker(
    candidates: ClockGatingCandidates,
    bin_index: BinIndex,
    tolerance_scalars: ToleranceScalars,
    bin_pairs: List[tuple[int, int]],
    *,
    beta: float,
    gamma: float,
    horizon_days: float,
    per_night_cap: int,
    kernel_col_window_size: int,
    max_k_span: int,
) -> ClockGatedEdges:
    """
    Worker function for building clock-gated edges between bins.

    Parameters
    ----------
    kernel_arrays
        Kernel arrays for the orbit.
    bin_index
        Bin index for the orbit.
    tolerance_scalars
        Tolerance scalars for the orbit.
    source_bin_id
        Source bin ID.
    dest_bin_id
        Dest bin ID.
    beta
        Beta parameter.
    gamma
        Gamma parameter.
    horizon_days
        Horizon days.
    per_night_cap
        Per night cap.
    kernel_col_window_size
        Kernel column window size.
    max_k_span
        Represents the number of revolutions around the orbit that are considered for the kernel.

    Returns
    -------
    ClockGatedEdges
        Clock-gated edges between bins.
    """
    # One-time numpy views
    time_rel_days = (
        candidates.time_tdb_mjd.to_numpy(zero_copy_only=False).astype(np.float64)
    )
    if time_rel_days.size > 0:
        t0 = float(np.min(time_rel_days))
        time_rel_days = (time_rel_days - t0).astype(np.float32, copy=False)
    else:
        time_rel_days = time_rel_days.astype(np.float32, copy=False)
    M_wrapped = candidates.M_wrapped_rad_f32().to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    n_rad_day = candidates.n_rad_day.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    sigma_M = candidates.sigma_M_rad.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    night_id = candidates.night_id.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)

    N = time_rel_days.shape[0]
    edge_accumulator = _init_edge_accumulator(N)

    for source_bin_id, dest_bin_id in bin_pairs:
        indices_time_sorted = bin_index.bin_to_time_sorted_indices.get(source_bin_id)
        if indices_time_sorted is None or indices_time_sorted.size == 0:
            return ClockGatedEdges.empty()

        idx_unsorted_j = bin_index.bin_to_unsorted_indices.get(dest_bin_id)
        if idx_unsorted_j is None or idx_unsorted_j.size == 0:
            return ClockGatedEdges.empty()

        # Check the horizon against the source bin
        # Work in the same (relative) time frame as time_rel_days
        t_center = (source_bin_id + 0.5) * bin_index.bin_width_days
        dt_center = float(t_center - time_rel_days[indices_time_sorted[0]])
        if dt_center <= 0.0 or dt_center > float(horizon_days):
            return ClockGatedEdges.empty()

        M_sorted_j = bin_index.bin_to_sorted_M[dest_bin_id]
        M_order_j = bin_index.bin_to_M_order_positions[dest_bin_id]

        two_pi = np.float32(2.0 * np.pi)
        max_k_span = np.int64(max_k_span)

        # Process one detection in source bin at a time
        for i_pos in range(indices_time_sorted.size):
            i_idx = int(indices_time_sorted[i_pos])
            time_i = float(time_rel_days[i_idx])
            mean_anomaly_i = float(M_wrapped[i_idx])
            mean_motion_i = float(n_rad_day[i_idx])
            sigma_i = float(sigma_M[i_idx])
            dMdf_i = float(tolerance_scalars.dM_df_per_candidate[i_idx])
            night_id_i = int(night_id[i_idx])

            # De-dup mask for this i
            seen_j = np.zeros(N, dtype=bool)

            # 1-element i buffers for kernel (views, no new allocations)
            times_i_arr = time_rel_days[i_idx : i_idx + 1]
            mean_anomaly_i_arr = M_wrapped[i_idx : i_idx + 1]
            mean_motion_i_arr = n_rad_day[i_idx : i_idx + 1]
            sigma_i_arr = sigma_M[i_idx : i_idx + 1]
            dMdf_i_arr = tolerance_scalars.dM_df_per_candidate[i_idx : i_idx + 1]
            i_valid = np.asarray([True], dtype=bool)


            # Build band positions for this single i
            t_center = (dest_bin_id + 0.5) * bin_index.bin_width_days
            dt_center = float(t_center - time_i)
            if dt_center <= 0.0 or dt_center > float(horizon_days):
                continue

            tau_center_days = float(
                tolerance_scalars.tau_min_days
                + tolerance_scalars.alpha_days_per_day * dt_center
                + beta * sigma_i
                + gamma * abs(dMdf_i)
            )
            mband_floor = 1e-5
            band_halfwidth_M_rad = float(
                max(
                    mband_floor,
                    mean_motion_i
                    * (tau_center_days + 2.0 * bin_index.bin_width_days),
                )
            )
            k_star_estimate = int(np.rint((mean_motion_i * dt_center) / two_pi))
            k_lo = max(0, int(k_star_estimate - max_k_span))
            k_hi = int(k_star_estimate + max_k_span)
            ks = np.arange(k_lo, k_hi + 1, dtype=np.int64)
            if ks.size == 0:
                continue

            centers = _wrap_2pi(
                mean_anomaly_i + mean_motion_i * dt_center - two_pi * ks
            )
            centers = np.atleast_1d(centers).astype(np.float32, copy=False)
            pos_mask_union = _mband_union_mask_numba(
                M_sorted_j.astype(np.float32, copy=False),
                centers,
                band_halfwidth_M_rad,
                two_pi,
            )

            band_positions = np.nonzero(pos_mask_union)[0]
            if band_positions.size == 0:
                continue
            candidate_j_indices = idx_unsorted_j[M_order_j[band_positions]]

            # Preallocate j buffers once per b2 and reuse
            times_j_block = np.zeros(
                (int(kernel_col_window_size),), dtype=np.float32
            )
            mean_anomaly_j_block = np.zeros(
                (int(kernel_col_window_size),), dtype=np.float32
            )
            j_valid_mask = np.zeros((int(kernel_col_window_size),), dtype=bool)

            for js in range(0, candidate_j_indices.size, kernel_col_window_size):
                j_indices_window = candidate_j_indices[js: js + kernel_col_window_size]
                if j_indices_window.size == 0:
                    continue
                count_j = j_indices_window.size
                if count_j > 0:
                    times_j_block[:count_j] = time_rel_days[j_indices_window]
                    mean_anomaly_j_block[:count_j] = M_wrapped[j_indices_window]
                if count_j < kernel_col_window_size:
                    times_j_block[count_j:] = 0.0
                    mean_anomaly_j_block[count_j:] = 0.0
                j_valid_mask[:] = False
                j_valid_mask[:count_j] = True

                keep, k_hat, _resid_unused, dt_mat, _dM_unused = pair_kernel(
                    times_i_arr,
                    mean_anomaly_i_arr,
                    mean_motion_i_arr,
                    sigma_i_arr,
                    dMdf_i_arr,
                    times_j_block,
                    mean_anomaly_j_block,
                    tolerance_scalars.tau_min_days,
                    tolerance_scalars.alpha_days_per_day,
                    beta,
                    gamma,
                    horizon_days,
                    valid_i=i_valid,
                    valid_j=j_valid_mask,
                )
                keep_np = np.asarray(keep)
                if not np.any(keep_np):
                    continue

                # Process single i row
                cols = np.nonzero(keep_np[0])[0]
                if cols.size == 0:
                    continue
                jg_all = j_indices_window[cols].astype(int)
                mask_seen = ~seen_j[jg_all]
                if not np.any(mask_seen):
                    continue
                jg = jg_all[mask_seen]
                cols2 = cols[mask_seen]
                dt_vals = np.asarray(dt_mat)[0, cols2]
                within_horizon = (dt_vals > 0.0) & (dt_vals <= float(horizon_days))
                if not np.any(within_horizon):
                    continue
                jg = jg[within_horizon]
                cols2 = cols2[within_horizon]
                dt_vals = dt_vals[within_horizon]
                same_vec = night_id[jg] == night_id_i

                if per_night_cap > 0 and np.any(same_vec):
                    quota = per_night_cap - edge_accumulator.outgoing_same_night_edge_counts[i_idx]
                    if quota <= 0:
                        same_vec[:] = False
                    else:
                        same_true_idx = np.nonzero(same_vec)[0]
                        if same_true_idx.size > quota:
                            drop_idx = same_true_idx[quota:]
                            same_vec[drop_idx] = False
                        edge_accumulator.outgoing_same_night_edge_counts[
                            i_idx
                        ] += min(quota, same_true_idx.size)

                k_vals = np.asarray(k_hat)[0, cols2]

                if jg.size > 0:
                    edge_accumulator.source_candidate_indices.extend(
                        [int(i_idx)] * int(jg.size)
                    )
                    edge_accumulator.dest_candidate_indices.extend(
                        jg.astype(np.int32).tolist()
                    )
                    edge_accumulator.k_revolutions_rounded.extend(
                        k_vals.astype(np.int16).tolist()
                    )
                    edge_accumulator.delta_time_days.extend(
                        dt_vals.astype(np.float32).tolist()
                    )
                    edge_accumulator.is_same_night.extend(
                        (night_id[jg] == night_id_i).tolist()
                    )
                    seen_j[jg] = True

    orbit_id_str = candidates.orbit_id.unique()[0].as_py() if len(candidates) > 0 else ""
    edges = ClockGatedEdges.from_kwargs(
        orbit_id=[orbit_id_str] * len(edge_accumulator.source_candidate_indices),
        i_index=edge_accumulator.source_candidate_indices,
        j_index=edge_accumulator.dest_candidate_indices,
        k_revs=edge_accumulator.k_revolutions_rounded,
        dt_days=edge_accumulator.delta_time_days,
        same_night=edge_accumulator.is_same_night,
        tau_min_minutes=tolerance_scalars.tau_min_days * 60.0,
        alpha_min_per_day=tolerance_scalars.alpha_days_per_day * 60.0,
        beta=beta,
        gamma=gamma,
        window_size=kernel_col_window_size,
    )
    return edges


@ray.remote
def inter_bin_edges_worker_remote(
    candidates: ClockGatingCandidates,
    bin_index: BinIndex,
    tolerance_scalars: ToleranceScalars,
    bin_pairs: List[tuple[int, int]],
    *,
    beta: float,
    gamma: float,
    horizon_days: float,
    per_night_cap: int,
    kernel_col_window_size: int,
    max_k_span: int,
) -> ClockGatedEdges:
    return inter_bin_edges_worker(
        candidates,
        bin_index,
        tolerance_scalars,
        bin_pairs,
        beta=beta,
        gamma=gamma,
        horizon_days=horizon_days,
        per_night_cap=per_night_cap,
        kernel_col_window_size=kernel_col_window_size,
        max_k_span=max_k_span,
    )


def calculate_bin_pairs(
    bin_index: BinIndex, max_bins_ahead: int, horizon_days: float
) -> list[tuple[int, int]]:
    """
    Determines pairs of bins to process for inter-bin edges based on time ordering and filters.
    """
    bin_pairs = []
    for source_bin_id in bin_index.unique_bins:
        for dest_bin_id in bin_index.unique_bins:
            if dest_bin_id > source_bin_id:
                bin_pairs.append((source_bin_id, dest_bin_id))
    return bin_pairs

def uf_find(uf_parent: np.ndarray, x: int) -> int:
    while uf_parent[x] != x:
        uf_parent[x] = uf_parent[uf_parent[x]]
        x = uf_parent[x]
    return x

def uf_union(uf_parent: np.ndarray, uf_rank: np.ndarray, a: int, b: int) -> None:
    ra, rb = uf_find(uf_parent, a), uf_find(uf_parent, b)
    if ra == rb:
        return
    if uf_rank[ra] < uf_rank[rb]:
        ra, rb = rb, ra
    uf_parent[rb] = ra
    if uf_rank[ra] == uf_rank[rb]:
        uf_rank[ra] += 1


@njit(cache=True)
def _uf_find_numba(uf_parent: np.ndarray, x: int) -> int:
    # Path compression
    while uf_parent[x] != x:
        uf_parent[x] = uf_parent[uf_parent[x]]
        x = uf_parent[x]
    return x


@njit(cache=True)
def _uf_union_numba(uf_parent: np.ndarray, uf_rank: np.ndarray, a: int, b: int) -> None:
    ra = _uf_find_numba(uf_parent, a)
    rb = _uf_find_numba(uf_parent, b)
    if ra == rb:
        return
    if uf_rank[ra] < uf_rank[rb]:
        ra, rb = rb, ra
    uf_parent[rb] = ra
    if uf_rank[ra] == uf_rank[rb]:
        uf_rank[ra] += 1


@njit(cache=True)
def _connected_components_labels_numba(
    num_nodes: int, src_indices: np.ndarray, dst_indices: np.ndarray
) -> np.ndarray:
    # Initialize union-find state
    uf_parent = np.arange(num_nodes, dtype=np.int32)
    uf_rank = np.zeros(num_nodes, dtype=np.int32)

    E = src_indices.shape[0]
    for e in range(E):
        _uf_union_numba(uf_parent, uf_rank, int(src_indices[e]), int(dst_indices[e]))

    # Compute root label for each node
    roots = np.empty(num_nodes, dtype=np.int32)
    for i in range(num_nodes):
        roots[i] = _uf_find_numba(uf_parent, i)
    return roots


def extract_kepler_chains(
    candidates: ClockGatingCandidates,
    edges: ClockGatedEdges,
    *,
    min_size: int = 6,
    min_span_days: float = 3.0,
) -> tuple[KeplerChains, KeplerChainMembers]:
    """
    Build connected components from edges and promote chains by size/span.
    """
    if len(candidates) == 0 or len(edges) == 0:
        return KeplerChains.empty(), KeplerChainMembers.empty()

    # We operate on a single orbit at a time
    assert len(candidates.orbit_id.unique()) == 1, "candidates must be a single orbit"

    orbit_id_str = candidates.orbit_id.unique()[0].as_py()

    num_candidates = len(candidates)
    # Build connected components using a Numba-accelerated union-find over edges
    src_indices = edges.i_index.to_numpy(zero_copy_only=False).astype(np.int32)
    dst_indices = edges.j_index.to_numpy(zero_copy_only=False).astype(np.int32)
    roots = _connected_components_labels_numba(num_candidates, src_indices, dst_indices)

    # Remap roots to compact labels [0..K-1]
    unique_roots, labels = np.unique(roots, return_inverse=True)
    labels = labels.astype(np.int64, copy=False)
    num_components = int(unique_roots.shape[0])

    # Vectorized per-component size/min/max
    time_mjd_local = candidates.time_tdb_mjd.to_numpy(zero_copy_only=False).astype(float)
    sizes = np.bincount(labels, minlength=num_components).astype(np.int32, copy=False)

    order = np.argsort(labels, kind="stable")
    labels_sorted = labels[order]
    times_sorted = time_mjd_local[order]
    # Start indices of each label group
    group_starts = np.concatenate((
        np.array([0], dtype=np.int64),
        np.nonzero(labels_sorted[1:] != labels_sorted[:-1])[0].astype(np.int64) + 1,
    ))
    t_min_by_label = np.minimum.reduceat(times_sorted, group_starts)
    t_max_by_label = np.maximum.reduceat(times_sorted, group_starts)

    # Filter by size and span
    keep_mask = (sizes >= int(min_size)) & ((t_max_by_label - t_min_by_label) >= float(min_span_days))
    if not np.any(keep_mask):
        return KeplerChains.empty(), KeplerChainMembers.empty()

    # Deterministic ordering: (-size, t_min, t_max)
    sizes_keep = sizes[keep_mask]
    tmin_keep = t_min_by_label[keep_mask]
    tmax_keep = t_max_by_label[keep_mask]
    ord_keep = np.lexsort((tmax_keep, tmin_keep, -sizes_keep))
    kept_labels = np.nonzero(keep_mask)[0][ord_keep]

    # Build chain table arrays
    num_kept = int(kept_labels.shape[0])
    chain_ids = np.arange(num_kept, dtype=np.int64)
    sizes_out = sizes[kept_labels].astype(np.int32, copy=False)
    tmins_out = t_min_by_label[kept_labels].astype(np.float64, copy=False)
    tmaxs_out = t_max_by_label[kept_labels].astype(np.float64, copy=False)

    # Map each label to a chain id, then gather member rows
    chain_id_for_label = np.full(num_components, -1, dtype=np.int64)
    chain_id_for_label[kept_labels] = chain_ids
    member_chain_ids_all = chain_id_for_label[labels]
    member_mask = member_chain_ids_all >= 0
    members_indices = np.nonzero(member_mask)[0].astype(np.int32)
    members_chain_ids = member_chain_ids_all[member_mask].astype(np.int64, copy=False)

    chains = KeplerChains.from_kwargs(
        orbit_id=[orbit_id_str] * num_kept,
        chain_id=chain_ids,
        size=sizes_out,
        t_min_mjd=tmins_out,
        t_max_mjd=tmaxs_out,
    )
    members = KeplerChainMembers.from_kwargs(
        orbit_id=[orbit_id_str] * len(members_chain_ids),
        chain_id=members_chain_ids,
        cand_index=members_indices,
    )

    return chains, members

