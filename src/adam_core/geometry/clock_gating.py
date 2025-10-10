"""
Kepler clock gating: build time-consistent edges between anomaly-labeled detections.

Public API produces quivr tables and follows adam_core chunking/Ray patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import ray

from ..coordinates.keplerian import KeplerianCoordinates
from ..orbits.orbits import Orbits
from ..ray_cluster import initialize_use_ray
from ..time import Timestamp
from ..utils.iter import _iterate_chunk_indices
from .anomaly import AnomalyLabels
from .rays import ObservationRays

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


class ClockGatingCandidates(qv.Table):
    """
    Per-orbit candidate vertices for kepler clock gating.

    Rows correspond to anomaly-labeled detections augmented with timing and
    orbit eccentricity. These are grouped by `orbit_id` downstream.
    """

    det_id = qv.LargeStringColumn()
    orbit_id = qv.LargeStringColumn()
    seg_id = qv.Int32Column()
    variant_id = qv.Int32Column()

    time_tdb_mjd = qv.Float64Column()
    night_id = qv.Int64Column()
    observer_code = qv.LargeStringColumn()

    # Anomalies and dynamics
    M_rad = qv.Float64Column()
    n_rad_day = qv.Float64Column()
    f_rad = qv.Float64Column()
    e = qv.Float64Column()
    sigma_M_rad = qv.Float64Column()

    # Optional guards / hints
    t_hat_plane_x = qv.Float32Column()
    t_hat_plane_y = qv.Float32Column()
    sky_pa_deg = qv.Float32Column(nullable=True)


class ClockGatedEdges(qv.Table):
    """
    Accepted time-consistent edges between candidates for a specific orbit.
    """

    orbit_id = qv.LargeStringColumn()
    i_index = qv.Int32Column()
    j_index = qv.Int32Column()
    k_revs = qv.Int16Column()
    resid_days = qv.Float32Column()
    dt_days = qv.Float32Column()
    dM_wrapped_rad = qv.Float32Column()
    same_night = qv.BooleanColumn()

    # Parameters recorded for provenance
    tau_min_minutes = qv.FloatAttribute(default=0.0)
    alpha_min_per_day = qv.FloatAttribute(default=0.0)
    beta = qv.FloatAttribute(default=0.0)
    gamma = qv.FloatAttribute(default=0.0)
    time_bin_minutes = qv.IntAttribute(default=0)
    max_bins_ahead = qv.IntAttribute(default=0)
    heading_max_deg = qv.FloatAttribute(default=np.nan)


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


def _factorize_strings(arr: pa.Array) -> pa.Array:
    """
    Map each string to its position in the unique set [0..K-1] (Arrow-backed).
    """
    uniq = pc.unique(arr)
    return pc.index_in(arr, uniq)


def _floor_days(mjd: pa.Array) -> pa.Array:
    return pc.cast(pc.floor(mjd), pa.int64())


def _compute_night_id(codes: pa.Array, time_mjd: pa.Array) -> pa.Array:
    """
    Stable per-night grouping key from observer code and integer MJD.
    """
    code_idx = _factorize_strings(codes)
    day = _floor_days(time_mjd)
    # Combine into a single Int64 key: (code_idx << 32) | (day & 0xFFFFFFFF)
    code_i64 = pc.cast(code_idx, pa.int64())
    day_masked = pc.bit_wise_and(day, pa.scalar(0xFFFFFFFF, type=pa.int64()))
    return pc.bit_wise_or(pc.shift_left(code_i64, 32), day_masked)


def prepare_clock_gating_candidates(
    labels: AnomalyLabels,
    rays: ObservationRays,
    orbits: Orbits,
    *,
    include_sky_pa: bool = False,
) -> ClockGatingCandidates:
    """
    Join anomaly labels with observation times and orbit eccentricities.

    Parameters
    ----------
    labels : AnomalyLabels
        Anomaly labels for (det_id, orbit_id[, variant]).
    rays : ObservationRays
        Rays providing det_id, observer code, and times.
    orbits : Orbits
        Original orbits to derive eccentricity values.
    include_sky_pa : bool, default False
        If True and a sky position angle column is present upstream, include it.

    Returns
    -------
    ClockGatingCandidates
        Candidates table grouped later by orbit_id for edge building.
    """
    if len(labels) == 0:
        return ClockGatingCandidates.empty()

    # Align rays to labels via det_id
    idx_in_rays = pc.index_in(labels.det_id, rays.det_id)
    if pc.any(pc.is_null(idx_in_rays)).as_py():
        raise ValueError("prepare_clock_gating_candidates: every label must have a ray")
    rays_sel = rays.take(idx_in_rays)

    # Times as TDB MJD
    times_mjd = (
        rays_sel.observer.coordinates.time.rescale("tdb")
        .mjd()
        .to_numpy(zero_copy_only=False)
    )

    # Observer codes aligned to labels
    obs_codes = rays_sel.observer.code

    # Compute night_id (Int64)
    night_id = _compute_night_id(obs_codes, pa.array(times_mjd))

    # Orbit eccentricities by joining labels.orbit_id to orbits.orbit_id
    # Convert orbits to Keplerian once
    if len(orbits) == 0:
        raise ValueError("prepare_clock_gating_candidates: orbits is empty")
    kep: KeplerianCoordinates = orbits.coordinates.to_keplerian()
    # Indices of labels.orbit_id within orbits.orbit_id
    idx_in_orbits = pc.index_in(labels.orbit_id, orbits.orbit_id)
    if pc.any(pc.is_null(idx_in_orbits)).as_py():
        raise ValueError(
            "prepare_clock_gating_candidates: label orbit_id missing in orbits"
        )
    e_aligned = pc.take(pa.array(kep.e), idx_in_orbits).to_numpy(zero_copy_only=False)

    # Optional sky PA if present upstream; otherwise fill nulls
    if include_sky_pa and hasattr(rays_sel, "sky_pa_deg"):
        sky_pa = pc.take(
            rays_sel.sky_pa_deg, pc.indices_nonzero(pc.greater_equal(idx_in_rays, 0))
        )
    else:
        sky_pa = pa.nulls(len(labels), type=pa.float32())

    # Assemble candidates from Arrow-backed columns to minimize copies
    out = ClockGatingCandidates.from_kwargs(
        det_id=labels.det_id,
        orbit_id=labels.orbit_id,
        seg_id=labels.seg_id,
        variant_id=labels.variant_id,
        time_tdb_mjd=times_mjd,
        night_id=night_id,
        observer_code=obs_codes,
        M_rad=labels.M_rad,
        n_rad_day=labels.n_rad_day,
        f_rad=labels.f_rad,
        e=e_aligned,
        sigma_M_rad=labels.sigma_M_rad,
        t_hat_plane_x=labels.t_hat_plane_x,
        t_hat_plane_y=labels.t_hat_plane_y,
        sky_pa_deg=sky_pa,
    )

    if out.fragmented():
        out = qv.defragment(out)
    return out


# ---- Edge building: declarations (implemented below) ----


def build_clock_gated_edges(
    candidates: ClockGatingCandidates,
    *,
    tau_min_minutes: float = 15.0,
    alpha_min_per_day: float = 0.02,
    beta: float = 1.0,
    gamma: float = 0.5,
    time_bin_minutes: int = 60,
    max_bins_ahead: int = 72,
    heading_max_deg: float | None = None,
    chunk_size: int = 200_000,
    max_processes: int | None = 1,
) -> ClockGatedEdges:
    if len(candidates) == 0:
        return ClockGatedEdges.empty()

    # Serial path: iterate per orbit
    if max_processes is None or max_processes <= 1:
        all_edges: list[ClockGatedEdges] = []
        for orbit_id in candidates.orbit_id.unique():
            orbit_id_str = orbit_id.as_py()
            mask = pc.equal(candidates.orbit_id, orbit_id)
            cands_i = candidates.apply_mask(mask)
            edges_i = _edges_worker_per_orbit(
                cands_i,
                tau_min_minutes=tau_min_minutes,
                alpha_min_per_day=alpha_min_per_day,
                beta=beta,
                gamma=gamma,
                time_bin_minutes=time_bin_minutes,
                max_bins_ahead=max_bins_ahead,
                heading_max_deg=heading_max_deg,
            )
            if len(edges_i) > 0:
                all_edges.append(edges_i)
        if not all_edges:
            return ClockGatedEdges.empty()
        edges = qv.concatenate(all_edges, defrag=True)
        return edges

    # Ray parallel path: shard by orbit_id
    initialize_use_ray(num_cpus=max_processes)
    candidates_ref = ray.put(candidates)
    futures: list[ray.ObjectRef] = []
    out_parts: list[ClockGatedEdges] = []
    max_active = max(1, int(1.5 * max_processes))

    for orbit_id in candidates.orbit_id.unique():
        orbit_id_str = orbit_id.as_py()
        fut = _edges_worker_per_orbit_remote.remote(
            candidates_ref,
            orbit_id_str,
            tau_min_minutes=tau_min_minutes,
            alpha_min_per_day=alpha_min_per_day,
            beta=beta,
            gamma=gamma,
            time_bin_minutes=time_bin_minutes,
            max_bins_ahead=max_bins_ahead,
            heading_max_deg=heading_max_deg,
        )
        futures.append(fut)
        if len(futures) >= max_active:
            finished, futures = ray.wait(futures, num_returns=1)
            out_parts.append(ray.get(finished[0]))

    while futures:
        finished, futures = ray.wait(futures, num_returns=1)
        out_parts.append(ray.get(finished[0]))

    out_parts = [p for p in out_parts if len(p) > 0]
    if not out_parts:
        return ClockGatedEdges.empty()
    return qv.concatenate(out_parts, defrag=True)


def extract_kepler_chains(
    candidates: ClockGatingCandidates,
    edges: ClockGatedEdges,
    *,
    min_size: int = 6,
    min_span_days: float = 3.0,
) -> tuple[KeplerChains, KeplerChainMembers]: ...


def kepler_clock_gate(
    labels: AnomalyLabels,
    rays: ObservationRays,
    orbits: Orbits,
    *,
    tau_min_minutes: float = 15.0,
    alpha_min_per_day: float = 0.02,
    beta: float = 1.0,
    gamma: float = 0.5,
    time_bin_minutes: int = 60,
    max_bins_ahead: int = 72,
    per_night_cap: int = 2,
    heading_max_deg: float | None = None,
    chunk_size: int = 200_000,
    max_processes: int | None = 1,
    promote_min_size: int = 6,
    promote_min_span_days: float = 3.0,
) -> tuple[ClockGatingCandidates, ClockGatedEdges, KeplerChains, KeplerChainMembers]:
    cands = prepare_clock_gating_candidates(labels, rays, orbits)
    edges = build_clock_gated_edges(
        cands,
        tau_min_minutes=tau_min_minutes,
        alpha_min_per_day=alpha_min_per_day,
        beta=beta,
        gamma=gamma,
        time_bin_minutes=time_bin_minutes,
        max_bins_ahead=max_bins_ahead,
        per_night_cap=per_night_cap,
        heading_max_deg=heading_max_deg,
        chunk_size=chunk_size,
        max_processes=max_processes,
    )
    chains, members = extract_kepler_chains(
        cands,
        edges,
        min_size=promote_min_size,
        min_span_days=promote_min_span_days,
    )
    return cands, edges, chains, members


# ------------------
# Internal utilities
# ------------------


def _wrap_2pi(x: np.ndarray) -> np.ndarray:
    y = np.mod(x, 2.0 * np.pi)
    return np.where(y < 0.0, y + 2.0 * np.pi, y)


def _choose_k(n: np.ndarray, dt: np.ndarray, dM_wrapped: np.ndarray) -> np.ndarray:
    k = np.rint((n * dt - dM_wrapped) / (2.0 * np.pi)).astype(np.int64)
    return np.maximum(k, 0)


def _clock_residual(
    n: np.ndarray, dt: np.ndarray, dM_wrapped: np.ndarray, k: np.ndarray
) -> np.ndarray:
    return dt - (dM_wrapped + 2.0 * np.pi * k) / (n + 1e-18)


def _dM_df(e: np.ndarray, f: np.ndarray) -> np.ndarray:
    num = (1.0 - e * e) ** 1.5
    den = (1.0 + e * np.cos(f)) ** 2
    return num / (den + 1e-18)


def _tau(
    dt: np.ndarray,
    tau_min: float,
    alpha: float,
    beta: float,
    sigma_Mi: np.ndarray,
    gamma: float,
    dMdf_i: np.ndarray,
) -> np.ndarray:
    return tau_min + alpha * dt + beta * sigma_Mi + gamma * np.abs(dMdf_i)


def _edges_worker_per_orbit(
    cands: ClockGatingCandidates,
    *,
    tau_min_minutes: float,
    alpha_min_per_day: float,
    beta: float,
    gamma: float,
    time_bin_minutes: int,
    max_bins_ahead: int,
    heading_max_deg: float | None,
) -> ClockGatedEdges:
    N = len(cands)
    if N == 0:
        return ClockGatedEdges.empty()

    # Extract numpy arrays
    t = cands.time_tdb_mjd.to_numpy(zero_copy_only=False).astype(float)
    M = cands.M_rad.to_numpy(zero_copy_only=False).astype(float)
    n = cands.n_rad_day.to_numpy(zero_copy_only=False).astype(float)
    f = cands.f_rad.to_numpy(zero_copy_only=False).astype(float)
    e = cands.e.to_numpy(zero_copy_only=False).astype(float)
    sigma_M = cands.sigma_M_rad.to_numpy(zero_copy_only=False).astype(float)
    night_id = cands.night_id.to_numpy(zero_copy_only=False).astype(np.int64)
    orbit_id = cands.orbit_id.to_pylist()[0] if len(cands) > 0 else ""

    # Optional heading
    if heading_max_deg is not None and "sky_pa_deg" in cands.table.column_names:
        sky_pa = cands.sky_pa_deg.to_numpy(zero_copy_only=False).astype(float)
        has_pa = ~np.isnan(sky_pa)
    else:
        sky_pa = None
        has_pa = None

    # Convert n from rad/day already; ensure float64
    n = n.astype(float)

    # Time binning
    bin_w = float(time_bin_minutes) / 1440.0
    t0 = float(np.min(t))
    bin_idx = np.floor((t - t0) / bin_w).astype(np.int64)

    # Build bins dict {bin: np.ndarray of indices}
    bins: dict[int, np.ndarray] = {}
    for b in np.unique(bin_idx):
        bins[int(b)] = np.nonzero(bin_idx == b)[0]

    edges_i_list: list[int] = []
    edges_j_list: list[int] = []
    edges_k_list: list[int] = []
    edges_resid_list: list[float] = []
    edges_dt_list: list[float] = []
    edges_dM_list: list[float] = []
    edges_same_night_list: list[bool] = []

    tau_min = float(tau_min_minutes) / 1440.0
    alpha = float(alpha_min_per_day) / 1440.0

    # Helper heading check
    def heading_ok(i: int, j: int) -> bool:
        if sky_pa is None:
            return True
        if not (has_pa[i] and has_pa[j]):
            return True
        dpa = abs(sky_pa[j] - sky_pa[i])
        dpa = min(dpa, 360.0 - dpa)
        return dpa <= float(heading_max_deg)

    # Collect same-night edges (no capping now, but keep flagging for diagnostics)

    # Iterate bin pairs
    for b in sorted(bins.keys()):
        I = bins[b]
        if I.size == 0:
            continue
        for da in range(1, max_bins_ahead + 1):
            b2 = b + da
            if b2 not in bins:
                # Skip gaps but continue scanning further bins
                continue
            J = bins[b2]
            if J.size == 0:
                continue

            # Vectorized vs scalar path: J typically small; iterate I and vectorize over J
            for i in I:
                t_i = t[i]
                dt = t[J] - t_i
                if not np.any(dt > 0.0):
                    continue
                valid_dt = dt > 0.0
                Jv = J[valid_dt]
                dt_v = dt[valid_dt]

                # Optional heading guard
                if sky_pa is not None and heading_max_deg is not None:
                    headmask = np.array([heading_ok(i, j) for j in Jv], dtype=bool)
                    if not np.any(headmask):
                        continue
                    Jv = Jv[headmask]
                    dt_v = dt_v[headmask]
                    if Jv.size == 0:
                        continue

                dM = _wrap_2pi(M[Jv] - M[i])
                k = _choose_k(n[i], dt_v, dM)
                resid = _clock_residual(n[i], dt_v, dM, k)
                dMdf_i = _dM_df(e[i], f[i])
                tau = _tau(dt_v, tau_min, alpha, beta, sigma_M[i], gamma, dMdf_i)

                keep = np.abs(resid) <= tau
                if not np.any(keep):
                    continue
                Jk = Jv[keep]
                dt_k = dt_v[keep]
                dM_k = dM[keep]
                k_k = k[keep]
                resid_k = resid[keep]

                # Per-night grouping
                same_mask = night_id[Jk] == night_id[i]
                # Emit cross-night edges immediately (no cap)
                if np.any(~same_mask):
                    nn = np.nonzero(~same_mask)[0]
                    for idx in nn:
                        j = int(Jk[idx])
                        edges_i_list.append(int(i))
                        edges_j_list.append(j)
                        edges_k_list.append(int(k_k[idx]))
                        edges_resid_list.append(float(resid_k[idx]))
                        edges_dt_list.append(float(dt_k[idx]))
                        edges_dM_list.append(float(dM_k[idx]))
                        edges_same_night_list.append(False)

                # Collect same-night edges (no capping)
                if np.any(same_mask):
                    sn = np.nonzero(same_mask)[0]
                    for idx in sn:
                        j = int(Jk[idx])
                        edges_i_list.append(int(i))
                        edges_j_list.append(int(j))
                        edges_k_list.append(int(k_k[idx]))
                        edges_resid_list.append(float(resid_k[idx]))
                        edges_dt_list.append(float(dt_k[idx]))
                        edges_dM_list.append(float(dM_k[idx]))
                        edges_same_night_list.append(True)

    # No per-night capping: all edges already appended

    if not edges_i_list:
        return ClockGatedEdges.empty()

    # Build table
    edges_tbl = ClockGatedEdges.from_kwargs(
        orbit_id=[orbit_id] * len(edges_i_list),
        i_index=np.asarray(edges_i_list, dtype=np.int32),
        j_index=np.asarray(edges_j_list, dtype=np.int32),
        k_revs=np.asarray(edges_k_list, dtype=np.int16),
        resid_days=np.asarray(edges_resid_list, dtype=np.float32),
        dt_days=np.asarray(edges_dt_list, dtype=np.float32),
        dM_wrapped_rad=np.asarray(edges_dM_list, dtype=np.float32),
        same_night=np.asarray(edges_same_night_list, dtype=bool),
        tau_min_minutes=float(tau_min_minutes),
        alpha_min_per_day=float(alpha_min_per_day),
        beta=float(beta),
        gamma=float(gamma),
        time_bin_minutes=int(time_bin_minutes),
        max_bins_ahead=int(max_bins_ahead),
        heading_max_deg=float(heading_max_deg) if heading_max_deg is not None else -1.0,
    )
    if edges_tbl.fragmented():
        edges_tbl = qv.defragment(edges_tbl)
    return edges_tbl


@ray.remote
def _edges_worker_per_orbit_remote(
    candidates: ClockGatingCandidates,
    orbit_id: str,
    *,
    tau_min_minutes: float,
    alpha_min_per_day: float,
    beta: float,
    gamma: float,
    time_bin_minutes: int,
    max_bins_ahead: int,
    heading_max_deg: float | None,
) -> ClockGatedEdges:
    mask = pc.equal(candidates.orbit_id, orbit_id)
    cands_i = candidates.apply_mask(mask)
    return _edges_worker_per_orbit(
        cands_i,
        tau_min_minutes=tau_min_minutes,
        alpha_min_per_day=alpha_min_per_day,
        beta=beta,
        gamma=gamma,
        time_bin_minutes=time_bin_minutes,
        max_bins_ahead=max_bins_ahead,
        heading_max_deg=heading_max_deg,
    )


def extract_kepler_chains(
    candidates: ClockGatingCandidates,
    edges: ClockGatedEdges,
    *,
    min_size: int = 6,
    min_span_days: float = 3.0,
) -> tuple[KeplerChains, KeplerChainMembers]:
    if len(candidates) == 0 or len(edges) == 0:
        return KeplerChains.empty(), KeplerChainMembers.empty()

    chains_out: list[KeplerChains] = []
    members_out: list[KeplerChainMembers] = []

    # Group by orbit
    for orbit_id in candidates.orbit_id.unique():
        oid = orbit_id.as_py()
        cand_mask = pc.equal(candidates.orbit_id, orbit_id)
        edge_mask = pc.equal(edges.orbit_id, orbit_id)
        cands_i = candidates.apply_mask(cand_mask)
        edges_i = edges.apply_mask(edge_mask)
        if len(cands_i) == 0 or len(edges_i) == 0:
            continue

        N = len(cands_i)
        # Build union-find over [0..N-1]
        parent = np.arange(N, dtype=np.int32)
        rank = np.zeros(N, dtype=np.int32)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1

        ii = edges_i.i_index.to_numpy(zero_copy_only=False).astype(np.int32)
        jj = edges_i.j_index.to_numpy(zero_copy_only=False).astype(np.int32)
        for a, b in zip(ii, jj):
            union(int(a), int(b))

        # Components
        comp_members: dict[int, list[int]] = {}
        for idx in range(N):
            r = find(idx)
            comp_members.setdefault(r, []).append(idx)

        # Summarize and promote
        t = cands_i.time_tdb_mjd.to_numpy(zero_copy_only=False).astype(float)
        comps = []
        for root, members in comp_members.items():
            tm = t[members]
            comps.append((members, len(members), float(np.min(tm)), float(np.max(tm))))
        # Filter and order deterministically
        comps = [
            c
            for c in comps
            if c[1] >= int(min_size) and (c[3] - c[2]) >= float(min_span_days)
        ]
        if not comps:
            continue
        comps.sort(key=lambda x: (-x[1], x[2], x[3]))

        chain_ids = []
        sizes = []
        tmins = []
        tmaxs = []
        members_chain_ids = []
        members_indices = []
        for chain_id, (members, size, t_min, t_max) in enumerate(comps):
            chain_ids.append(chain_id)
            sizes.append(size)
            tmins.append(t_min)
            tmaxs.append(t_max)
            members_chain_ids.extend([chain_id] * len(members))
            members_indices.extend(members)

        chains_tbl = KeplerChains.from_kwargs(
            orbit_id=[oid] * len(chain_ids),
            chain_id=np.asarray(chain_ids, dtype=np.int64),
            size=np.asarray(sizes, dtype=np.int32),
            t_min_mjd=np.asarray(tmins, dtype=np.float64),
            t_max_mjd=np.asarray(tmaxs, dtype=np.float64),
        )
        members_tbl = KeplerChainMembers.from_kwargs(
            orbit_id=[oid] * len(members_chain_ids),
            chain_id=np.asarray(members_chain_ids, dtype=np.int64),
            cand_index=np.asarray(members_indices, dtype=np.int32),
        )
        chains_out.append(chains_tbl)
        members_out.append(members_tbl)

    if not chains_out:
        return KeplerChains.empty(), KeplerChainMembers.empty()
    return qv.concatenate(chains_out, defrag=True), qv.concatenate(
        members_out, defrag=True
    )
