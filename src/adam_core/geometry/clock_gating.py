"""
Kepler clock gating: build time-consistent edges between anomaly-labeled detections.

Public API produces quivr tables and follows adam_core chunking/Ray patterns.
"""

from __future__ import annotations

from typing import Callable, Dict, List, NamedTuple
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import ray
from numba import njit
from functools import partial
import jax
import jax.numpy as jnp

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

# Default refinement tolerance in mean anomaly (radians): 0.036 arcseconds
_DEFAULT_REFINE_TOL_M_RAD = 1e-5


class ClockGatingCandidates(qv.Table):
    """
    Per-orbit candidate vertices used by Kepler clock gating.

    Each row corresponds to an anomaly-labeled detection augmented with
    observation time and orbit eccentricity. Downstream, rows are grouped by
    `orbit_id` and only time-consistent pairs are connected.
    """
    # Stable candidate identifier for downstream chain membership/output
    cand_id = qv.LargeStringColumn()
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

    time_bin_minutes = qv.IntAttribute(default=0)
    max_bins_ahead = qv.IntAttribute(default=0)
    horizon_days = qv.FloatAttribute(default=0.0)
    batch_size = qv.IntAttribute(default=0)
    mband_padding_bins = qv.FloatAttribute(default=1.0)
    refine_tol_M_rad = qv.FloatAttribute(default=0.0)


class KeplerChainMembers(qv.Table):
    orbit_id = qv.LargeStringColumn()
    chain_id = qv.Int64Column()
    cand_id = qv.LargeStringColumn()


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
    # Labels are Float32; only cast orbit eccentricity materialized from Keplerian
    e_f32 = pa.array(label_eccentricity).cast(pa.float32())

    # Assign a random UUID per candidate row for stable identity downstream
    cand_id_array = pa.array([str(uuid.uuid4()) for _ in range(len(labels))])

    candidates_table = ClockGatingCandidates.from_kwargs(
        cand_id=cand_id_array,
        det_id=labels.det_id,
        orbit_id=labels.orbit_id,
        seg_id=labels.seg_id,
        variant_id=labels.variant_id,
        time_tdb_mjd=observation_time_tdb_mjd,
        night_id=night_id_per_label,
        observer_code=rays_for_labels.observer.code,
        M_rad=labels.M_rad,
        n_rad_day=labels.n_rad_day,
        f_rad=labels.f_rad,
        e=e_f32,
        sigma_M_rad=labels.sigma_M_rad,
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
    time_bin_minutes: int = 120,
    max_bins_ahead: int | None = None,
    horizon_days: float = 90.0,
    max_processes: int | None = 1,
    inter_bin_chunk_size: int = 1000,
    mband_padding_bins: float = 0.5,
    mband_floor_per_day_rad: float | None = None,
    refine_tol_M_rad: float = _DEFAULT_REFINE_TOL_M_RAD,
    refine_window_size: int = 16384,
    diagnostics_sample_rate: float = 0.0,
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
    time_bin_minutes
        Width of time bins used to limit pairwise comparisons.
    max_bins_ahead
        Number of subsequent bins to consider for each source bin (used with
        horizon_days; effective lookahead is limited by both).
    horizon_days
        Absolute lookahead horizon in days for inter-bin scanning.
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

    bin_pairs = calculate_bin_pairs(bin_index, max_bins_ahead, horizon_days)
    futures = []
    all_edges_list = []

    if max_processes >= 2:
        initialize_use_ray(num_cpus=max_processes)
        candidates_ref = ray.put(candidates)
        # bin_index_ref = ray.put(bin_index)

    # Intra-bin edges are now handled by inter-bin worker by allowing src==dst bin pairs.

    for bin_pairs_chunk in _iterate_chunks(bin_pairs, chunk_size=inter_bin_chunk_size):
        if max_processes < 2:
            per_bin_inter_bin_edges = inter_bin_edges_worker(
                candidates,
                bin_index,
                bin_pairs_chunk,
                mband_padding_bins=mband_padding_bins,
                mband_floor_per_day_rad=mband_floor_per_day_rad,
                refine_tol_M_rad=refine_tol_M_rad,
                refine_window_size=refine_window_size,
                diagnostics_sample_rate=diagnostics_sample_rate,
            )
            all_edges_list.append(per_bin_inter_bin_edges)
        else:
            futures.append(inter_bin_edges_worker_remote.remote(
                candidates_ref,
                bin_index,
                bin_pairs_chunk,
                mband_padding_bins=float(mband_padding_bins),
                mband_floor_per_day_rad=mband_floor_per_day_rad,
                refine_tol_M_rad=float(refine_tol_M_rad),
                refine_window_size=int(refine_window_size),
                diagnostics_sample_rate=float(diagnostics_sample_rate),
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
    time_bin_minutes: int = 120,
    max_bins_ahead: int | None = None,
    horizon_days: float = 90.0,
    max_processes: int | None = 1,
    promote_min_size: int = 6,
    promote_min_span_days: float = 3.0,
    mband_padding_bins: float = 0.5,
    mband_floor_per_day_rad: float | None = None,
    refine_tol_M_rad: float = _DEFAULT_REFINE_TOL_M_RAD,
    refine_window_size: int = 16384,
    diagnostics_sample_rate: float = 0.0,
) -> tuple[KeplerChains, KeplerChainMembers]:
    """
    Complete Kepler clock-gating stage: build edges and chains from candidates.

    Parameters
    ----------
    candidates
        Candidate vertices (one orbit at a time downstream).
    time_bin_minutes
        Width of time bins used to limit pairwise comparisons.
    max_bins_ahead
        Number of subsequent bins to consider for each source bin (used with
        horizon_days; effective lookahead is limited by both).
    horizon_days
        Absolute lookahead horizon in days for inter-bin scanning.

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
            time_bin_minutes=time_bin_minutes,
            max_bins_ahead=max_bins_ahead,
            horizon_days=horizon_days,
            max_processes=max_processes,
            mband_padding_bins=mband_padding_bins,
            mband_floor_per_day_rad=mband_floor_per_day_rad,
            refine_tol_M_rad=refine_tol_M_rad,
            refine_window_size=refine_window_size,
            diagnostics_sample_rate=diagnostics_sample_rate,
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


"""Removed unused helpers _choose_k and _clock_residual (now superseded by pair_kernel)."""


@njit(cache=True)
def _mband_union_mask_numba(
    M_sorted_j: np.ndarray,
    centers: np.ndarray,
    halfwidth: np.float32,
    two_pi: np.float32,
) -> np.ndarray:
    """
    Compute the union of M-bands for a given set of centers and halfwidths.
    """
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


# ---------------------
# JAX refinement kernel
# ---------------------

@jax.jit
def _refine_edges_M_only_kernel(
    M_i: jax.Array,
    n_i: jax.Array,
    t_i: jax.Array,
    M_j: jax.Array,
    t_j: jax.Array,
    tol_M: jax.Array,
    valid_mask: jax.Array,
) -> jax.Array:
    """
    Fixed-shape kernel: mean-anomaly residual vs predicted M_i + n_i * dt.
    Accept if |wrap_shortest(M_j - M_pred)| <= tol_M for valid rows.

    Parameters
    ----------
    M_i
        Mean anomaly of the source candidate in radians.
    n_i
        Mean motion of the source candidate in radians per day.
    t_i
        Time of the source candidate in days.
    M_j
        Mean anomaly of the destination candidate in radians.
    t_j
        Time of the destination candidate in days.
    tol_M
        Static tolerance in mean anomaly (radians).
    valid_mask
        Mask of valid rows.

    Returns
    -------
    keep
        Mask of valid rows.
    """
    two_pi = jnp.float32(2.0 * jnp.pi)
    dt = t_j - t_i
    M_pred = jnp.mod(M_i + n_i * dt, two_pi)
    # Shortest signed difference in [-pi, pi]
    resid = ((M_j - M_pred + jnp.pi) % (2.0 * jnp.pi)) - jnp.pi
    keep = jnp.abs(resid) <= tol_M
    return jnp.logical_and(keep, valid_mask)

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
    bin_to_F_order_positions: Dict[int, np.ndarray]
    bin_to_sorted_F: Dict[int, np.ndarray]


class ToleranceScalars(NamedTuple):
    tau_min_days: float
    alpha_days_per_day: float
    dM_df_per_candidate: np.ndarray


class PairBuffer:
    """Growable SoA buffer for provisional or accepted pairs."""

    def __init__(self, initial_capacity: int = 16384) -> None:
        cap = 1024 if int(initial_capacity) <= 0 else int(initial_capacity)
        self._size = 0
        self._cap = cap
        self.src = np.empty(cap, dtype=np.int32)
        self.dst = np.empty(cap, dtype=np.int32)
        self.k = np.empty(cap, dtype=np.int16)
        self.dt = np.empty(cap, dtype=np.float32)
        self.same = np.empty(cap, dtype=np.bool_)

    @property
    def size(self) -> int:
        return self._size

    def clear(self) -> None:
        self._size = 0

    def _ensure_capacity(self, additional: int) -> None:
        required = self._size + int(additional)
        if required <= self._cap:
            return
        new_cap = self._cap * 2
        if new_cap < required:
            new_cap = required
        self.src = self._resize(self.src, new_cap)
        self.dst = self._resize(self.dst, new_cap)
        self.k = self._resize(self.k, new_cap)
        self.dt = self._resize(self.dt, new_cap)
        self.same = self._resize(self.same, new_cap)
        self._cap = new_cap

    def _resize(self, arr: np.ndarray, new_cap: int) -> np.ndarray:
        out = np.empty(new_cap, dtype=arr.dtype)
        s = self._size
        if s > 0:
            out[:s] = arr[:s]
        return out

    def append_many(
        self,
        src_indices: np.ndarray,
        dst_indices: np.ndarray,
        k_vals: np.ndarray,
        dt_vals: np.ndarray,
        same_flags: np.ndarray,
    ) -> None:
        n = int(dst_indices.shape[0])
        if n == 0:
            return
        self._ensure_capacity(n)
        s = self._size
        e = s + n
        self.src[s:e] = src_indices.astype(np.int32, copy=False)
        self.dst[s:e] = dst_indices.astype(np.int32, copy=False)
        self.k[s:e] = k_vals.astype(np.int16, copy=False)
        self.dt[s:e] = dt_vals.astype(np.float32, copy=False)
        self.same[s:e] = same_flags.astype(np.bool_, copy=False)
        self._size = e


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
    # Wrap true anomaly to [0, 2pi)
    f_vals = candidates_for_orbit.f_rad.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    true_anomaly_wrapped_rad = np.mod(f_vals, np.float32(2.0 * np.pi))
    mask_neg = true_anomaly_wrapped_rad < 0.0
    if np.any(mask_neg):
        true_anomaly_wrapped_rad[mask_neg] += np.float32(2.0 * np.pi)

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
    bin_to_F_order_positions: Dict[int, np.ndarray] = {}
    bin_to_sorted_F: Dict[int, np.ndarray] = {}

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

        wrapped_F_unsorted_for_bin = true_anomaly_wrapped_rad[idx_unsorted]
        order_positions_by_wrapped_F = np.argsort(
            wrapped_F_unsorted_for_bin, kind="stable"
        )
        wrapped_F_sorted = wrapped_F_unsorted_for_bin[order_positions_by_wrapped_F]

        b_int = int(bin_id)
        bin_to_unsorted_indices[b_int] = idx_unsorted
        bin_to_time_sorted_indices[b_int] = idx_time_sorted
        bin_to_sorted_times[b_int] = times_sorted_within_bin
        bin_to_M_order_positions[b_int] = order_positions_by_wrapped_M
        bin_to_sorted_M[b_int] = wrapped_M_sorted
        bin_to_F_order_positions[b_int] = order_positions_by_wrapped_F
        bin_to_sorted_F[b_int] = wrapped_F_sorted

    return BinIndex(
        bin_width_days=bin_width_days,
        t0_min_days=t0_min_days,
        unique_bins=unique_bins,
        bin_to_unsorted_indices=bin_to_unsorted_indices,
        bin_to_time_sorted_indices=bin_to_time_sorted_indices,
        bin_to_sorted_times=bin_to_sorted_times,
        bin_to_M_order_positions=bin_to_M_order_positions,
        bin_to_sorted_M=bin_to_sorted_M,
        bin_to_F_order_positions=bin_to_F_order_positions,
        bin_to_sorted_F=bin_to_sorted_F,
    )


def _init_edge_buffer(initial_capacity: int = 16384) -> PairBuffer:
    return PairBuffer(initial_capacity=initial_capacity)


# -----------------------------
# (removed duplicate PairBuffer definition)
# -----------------------------

# ---- Simple M-band inter-bin worker (diagnostic baseline) ----

def _floor_per_day_from_n(mean_motion_rad_per_day: float) -> float:
    """Return an n-based floor in rad/day (no eccentricity).

    Piecewise schedule chosen from sweep behavior:
      - n >= 0.05               -> 0.01
      - 0.02 <= n < 0.05        -> 0.03
      - 0.006 <= n < 0.02       -> 0.03
      - 0.001 <= n < 0.006      -> 0.05
      - n < 0.001               -> 0.07
    """
    n = float(mean_motion_rad_per_day)
    if n >= 0.05:
        return 0.01
    if n >= 0.02:
        return 0.03
    if n >= 0.006:
        return 0.03
    if n >= 0.001:
        return 0.05
    return 0.07


def inter_bin_edges_worker(
    candidates: ClockGatingCandidates,
    bin_index: BinIndex,
    bin_pairs: List[tuple[int, int]],
    *,
    mband_padding_bins: float = 0.5,
    mband_floor_per_day_rad: float | None = None,
    refine_tol_M_rad: float = _DEFAULT_REFINE_TOL_M_RAD,
    refine_window_size: int = 16384,
    diagnostics_sample_rate: float = 0.0,
) -> ClockGatedEdges:
    """
    Build inter-bin edges using a dial-free mean-anomaly bin-sweep rule.

    For each source detection i in a source bin and each destination bin [t0, t1],
    accept any j in that destination bin whose mean anomaly lies within the
    minimal wrapped arc swept by i from t0 to t1. Implemented via center/halfwidth:

        t_c = (t0 + t1) / 2
        M_center = wrap(M_i + n_i * (t_c - t_i))
        H_M = n_i * (bin_width_days / 2)

    Membership: |wrap_shortest(M_j - M_center)| <= H_M and dt = t_j - t_i >= 0.

    Optional bin-based padding: inflate halfwidth by `mband_padding_bins` bins
    (unitless): H_M = n_i * bin_width_days * (0.5 + mband_padding_bins).

    Intentionally ignores all higher-level dials (tau/alpha/beta/gamma, caps,
    dM/df scaling, padding, k-span, min-dt, per-night quotas). This is a
    diagnostic/simple baseline.
    """
    # One-time numpy views aligned with local time origin
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
    night_id = candidates.night_id.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)

    edge_buffer = _init_edge_buffer()
    tol_val = np.float32(refine_tol_M_rad)

    two_pi = np.float32(2.0 * np.pi)
    bin_width_days = float(bin_index.bin_width_days)
    bin_width_days_f32 = np.float32(bin_width_days)
    pad_bins_scalar = np.float32(mband_padding_bins) if float(mband_padding_bins) > 0.0 else np.float32(0.0)

    # Group destination bins by source bin to avoid recomputing per-dest invariants for each source i
    pairs_by_src: Dict[int, List[int]] = {}
    for src, dst in bin_pairs:
        lst = pairs_by_src.get(src)
        if lst is None:
            pairs_by_src[src] = [dst]
        else:
            lst.append(dst)

    for source_bin_id, dest_bin_list in pairs_by_src.items():
        indices_time_sorted = bin_index.bin_to_time_sorted_indices.get(source_bin_id)
        if indices_time_sorted is None or indices_time_sorted.size == 0:
            continue

        # Precompute per-destination bin data reused across all sources i in the same source bin
        dest_data_map: Dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.float32]] = {}
        for dest_bin_id in dest_bin_list:
            idx_unsorted_j = bin_index.bin_to_unsorted_indices.get(dest_bin_id)
            if idx_unsorted_j is None or idx_unsorted_j.size == 0:
                continue
            M_sorted_j = bin_index.bin_to_sorted_M[dest_bin_id]
            M_order_j = bin_index.bin_to_M_order_positions[dest_bin_id]
            M_sorted_j_f32 = M_sorted_j if M_sorted_j.dtype == np.float32 else M_sorted_j.astype(np.float32, copy=False)
            t_center_f32 = (np.float32(dest_bin_id) + np.float32(0.5)) * bin_width_days_f32
            dest_data_map[dest_bin_id] = (idx_unsorted_j, M_order_j, M_sorted_j_f32, t_center_f32)

        if not dest_data_map:
            continue

        centers_buffer = np.empty(1, dtype=np.float32)

        # Growable flattened staging for provisional pairs in this source bin
        stage = PairBuffer(initial_capacity=8192)

        # Process each source detection i once per source bin and reuse per-i constants
        for i_pos in range(indices_time_sorted.size):
            i_idx = int(indices_time_sorted[i_pos])
            time_i = time_rel_days[i_idx]
            mean_anomaly_i = M_wrapped[i_idx]
            mean_motion_i = n_rad_day[i_idx]
            night_id_i = int(night_id[i_idx])

            # Per-source half-width in M (constant across all destination bins)
            # H_base = n_i * bin_width_days * (0.5 + pad_bins)
            H_base = mean_motion_i * (bin_width_days_f32 * (np.float32(0.5) + pad_bins_scalar))
            if H_base < 0.0:
                H_base = np.float32(0.0)
            if mband_floor_per_day_rad is None:
                floor_per_day = np.float32(_floor_per_day_from_n(float(mean_motion_i)))
            else:
                floor_per_day = np.float32(mband_floor_per_day_rad)
            H_floor = (floor_per_day if floor_per_day > 0.0 else np.float32(0.0)) * bin_width_days_f32
            H_M = H_base if H_base >= H_floor else H_floor

            # Sweep destination bins for this source i
            for dest_bin_id, (idx_unsorted_j, M_order_j, M_sorted_j_f32, t_center_f32) in dest_data_map.items():
                dt_center = t_center_f32 - time_i
                # Center of swept M-band
                M_center = mean_anomaly_i + mean_motion_i * dt_center
                # Wrap to [0, 2pi)
                M_center = np.mod(M_center, two_pi)
                centers_buffer[0] = M_center

                pos_mask_union = _mband_union_mask_numba(
                    M_sorted_j_f32,
                    centers_buffer,
                    H_M,
                    two_pi,
                )

                band_positions = np.nonzero(pos_mask_union)[0]
                if band_positions.size == 0:
                    continue

                candidate_j_indices = idx_unsorted_j[M_order_j[band_positions]]
                if candidate_j_indices.size == 0:
                    continue

                # Forward-time filter (dt >= 0)
                dt_vals = time_rel_days[candidate_j_indices] - time_i
                keep_dt = dt_vals >= 0.0
                if not np.any(keep_dt):
                    continue
                candidate_j_indices = candidate_j_indices[keep_dt]
                dt_vals = dt_vals[keep_dt]

                k_vals = np.rint((mean_motion_i * dt_vals) / two_pi).astype(np.int16, copy=False)
                same_vec = (night_id[candidate_j_indices] == night_id_i)

                # Stage flattened rows for refinement (amortized growth)
                count = int(candidate_j_indices.size)
                if count == 0:
                    continue
                stage.append_many(
                    np.full(count, int(i_idx), dtype=np.int32),
                    candidate_j_indices.astype(np.int32, copy=False),
                    dt_vals.astype(np.float32, copy=False),
                    k_vals.astype(np.int16, copy=False),
                    same_vec.astype(np.bool_, copy=False),
                )

                if float(diagnostics_sample_rate) > 0.0:
                    if np.random.random() < float(diagnostics_sample_rate):
                        for idx_loc in range(min(3, candidate_j_indices.size)):
                            print(
                                f"[diag simple] dt={float(dt_vals[idx_loc]):.6f} k={int(k_vals[idx_loc])} same={bool(same_vec[idx_loc])} H_M={float(H_M):.6f}"
                            )

        # After finishing the i loop for this source bin, run refinement over staged rows
        if stage.size > 0:
            # Preallocate fixed-size buffers for JAX kernel
            win = int(refine_window_size) if int(refine_window_size) > 0 else 16384
            # Per-window numpy buffers
            Mi_buf = np.zeros(win, dtype=np.float32)
            Ni_buf = np.zeros(win, dtype=np.float32)
            Ti_buf = np.zeros(win, dtype=np.float32)
            Mj_buf = np.zeros(win, dtype=np.float32)
            Tj_buf = np.zeros(win, dtype=np.float32)
            valid_buf = np.zeros(win, dtype=np.bool_)

            tol_buf = jnp.full((win,), tol_val, dtype=jnp.float32)

            # Gather state vectors for staged pairs
            Mi_all = M_wrapped[stage.src[: stage.size]]
            Ni_all = n_rad_day[stage.src[: stage.size]]
            Ti_all = time_rel_days[stage.src[: stage.size]]
            Mj_all = M_wrapped[stage.dst[: stage.size]]
            Tj_all = time_rel_days[stage.dst[: stage.size]]

            L = int(stage.size)
            start = 0
            while start < L:
                end = min(start + win, L)
                cur = end - start
                # Fill buffers
                Mi_buf[:cur] = Mi_all[start:end]
                Ni_buf[:cur] = Ni_all[start:end]
                Ti_buf[:cur] = Ti_all[start:end]
                Mj_buf[:cur] = Mj_all[start:end]
                Tj_buf[:cur] = Tj_all[start:end]
                valid_buf[:cur] = True
                if cur < win:
                    Mi_buf[cur:] = 0.0
                    Ni_buf[cur:] = 0.0
                    Ti_buf[cur:] = 0.0
                    Mj_buf[cur:] = 0.0
                    Tj_buf[cur:] = 0.0
                    valid_buf[cur:] = False

                keep_mask = _refine_edges_M_only_kernel(
                    Mi_buf,
                    Ni_buf,
                    Ti_buf,
                    Mj_buf,
                    Tj_buf,
                    tol_buf,
                    valid_buf,
                )

                keep_np = np.asarray(keep_mask)
                if np.any(keep_np[:cur]):
                    sel = np.nonzero(keep_np[:cur])[0]
                    edge_buffer.append_many(
                        stage.src[start:end][sel],
                        stage.dst[start:end][sel],
                        stage.k[start:end][sel],
                        stage.dt[start:end][sel],
                        stage.same[start:end][sel],
                    )

                start = end

    orbit_id_str = candidates.orbit_id.unique()[0].as_py() if len(candidates) > 0 else ""
    edges = ClockGatedEdges.from_kwargs(
        orbit_id=[orbit_id_str] * int(edge_buffer.size),
        i_index=edge_buffer.src[: edge_buffer.size].astype(np.int32),
        j_index=edge_buffer.dst[: edge_buffer.size].astype(np.int32),
        k_revs=edge_buffer.k[: edge_buffer.size].astype(np.int16),
        dt_days=edge_buffer.dt[: edge_buffer.size].astype(np.float32),
        same_night=edge_buffer.same[: edge_buffer.size],
        mband_padding_bins=float(mband_padding_bins),
        refine_tol_M_rad=float(refine_tol_M_rad),
    )
    return edges


@ray.remote
def inter_bin_edges_worker_remote(
    candidates: ClockGatingCandidates,
    bin_index: BinIndex,
    bin_pairs: List[tuple[int, int]],
    *,
    mband_padding_bins: float,
    mband_floor_per_day_rad: float,
    refine_tol_M_rad: float,
    refine_window_size: int,
    diagnostics_sample_rate: float,
) -> ClockGatedEdges:
    return inter_bin_edges_worker(
        candidates,
        bin_index,
        bin_pairs,
        mband_padding_bins=mband_padding_bins,
        mband_floor_per_day_rad=mband_floor_per_day_rad,
        refine_tol_M_rad=refine_tol_M_rad,
        refine_window_size=int(refine_window_size),
        diagnostics_sample_rate=diagnostics_sample_rate,
    )


def calculate_bin_pairs(
    bin_index: BinIndex, max_bins_ahead: int | None, horizon_days: float
) -> list[tuple[int, int]]:
    """
    Determines pairs of bins to process for inter-bin edges based on time ordering and filters.
    """
    bin_pairs: list[tuple[int, int]] = []
    unique_bins = bin_index.unique_bins
    bin_width_days = float(bin_index.bin_width_days)
    if bin_width_days <= 0.0:
        return bin_pairs

    # Convert horizon in days to a maximum number of bins
    horizon_bins = int(np.floor(float(horizon_days) / bin_width_days)) if float(horizon_days) > 0.0 else 0

    # Convert unique bins to Python ints for faster membership tests
    unique_bin_ints = [int(b) for b in unique_bins.tolist()]
    unique_bin_set = set(unique_bin_ints)

    for src_bin_int in unique_bin_ints:
        # Compute farthest destination bin allowed by both constraints
        if max_bins_ahead is None:
            max_dst_by_ahead = src_bin_int + (horizon_bins if horizon_bins > 0 else 10**9)
        else:
            mba = int(max_bins_ahead)
            max_dst_by_ahead = src_bin_int + (mba if mba > 0 else (horizon_bins if horizon_bins > 0 else 10**9))
        max_dst_by_horizon = src_bin_int + horizon_bins if horizon_bins > 0 else max_dst_by_ahead
        max_dst = min(max_dst_by_ahead, max_dst_by_horizon)

        # Iterate contiguous destination indices and include only those that exist
        # Allow same-bin pairs (src==dst)
        for dst_bin_int in range(src_bin_int, max_dst + 1):
            if dst_bin_int not in unique_bin_set:
                continue
            if (float(dst_bin_int - src_bin_int) * bin_width_days) > float(horizon_days):
                continue
            bin_pairs.append((src_bin_int, dst_bin_int))

    return bin_pairs


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
    # Map member indices to stable candidate IDs for persistence
    members_cand_id = pc.take(candidates.cand_id, pa.array(members_indices, type=pa.int64()))
    members = KeplerChainMembers.from_kwargs(
        orbit_id=[orbit_id_str] * len(members_chain_ids),
        chain_id=members_chain_ids,
        cand_id=members_cand_id,
    )

    return chains, members

