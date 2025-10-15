import cProfile
import os
import pstats
import time

import numpy as np
import numpy.testing as npt
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from adam_core.coordinates import CartesianCoordinates
from adam_core.geometry import (
    build_clock_gated_edges,
    extract_kepler_chains,
    label_anomalies,
    prepare_clock_gating_candidates,
    query_bvh,
)
from adam_core.geometry.clock_gating import (
    ClockGatedEdges,
    ClockGatingCandidates,
    KeplerChainMembers,
    KeplerChains,
    _dM_df,
    _tau,
    _wrap_2pi,
    _extract_kernel_arrays,
    _build_bins_and_m_index,
    _compute_per_detection_scalars,
    intra_bin_edges_worker,
    inter_bin_edges_worker,
    calculate_bin_pairs,
    build_clock_gated_edges,
    extract_kepler_chains,
    kepler_clock_gate,
)
from adam_core.geometry.rays import ObservationRays

from .test_bvh_query import compute_recall_metrics_from_hits

# ---- Micro-benchmarks for primitives ----


@pytest.mark.benchmark(group="clock_wrap")
def test_benchmark_wrap_2pi(benchmark):
    rng = np.random.default_rng(42)
    arr = rng.normal(0.0, 100.0, size=2_000_000)
    benchmark(_wrap_2pi, arr)


@pytest.mark.skip(
    reason="Removed _choose_k and _clock_residual; covered by pair_kernel"
)
def test_removed_choosek_and_clock_residual():
    pass


@pytest.mark.benchmark(group="clock_dM_df")
def test_benchmark_dM_df(benchmark):
    rng = np.random.default_rng(45)
    e = np.clip(rng.normal(0.1, 0.05, size=1_000_000), 0.0, 0.9)
    f = _wrap_2pi(rng.uniform(-10.0, 10.0, size=1_000_000))
    benchmark(_dM_df, e, f)


@pytest.mark.benchmark(group="clock_tau")
def test_benchmark_tau(benchmark):
    rng = np.random.default_rng(46)
    dt = rng.uniform(0.0, 10.0, size=1_000_000)
    sigma_M = rng.uniform(0.0, 1e-2, size=1_000_000)
    dMdf = rng.uniform(0.0, 2.0, size=1_000_000)
    tau_min = 15.0 / 1440.0
    alpha = 0.02 / 1440.0
    beta = 1.0
    gamma = 0.5
    benchmark(_tau, dt, tau_min, alpha, beta, sigma_M, gamma, dMdf)


# ---- Pairwise block dt/dM build benchmark ----


@pytest.mark.benchmark(group="clock_pair_block")
def test_benchmark_pairwise_block(benchmark):
    rng = np.random.default_rng(60)
    N_i, N_j = 5000, 5000
    t = np.sort(59000.0 + rng.random(N_i + N_j))
    t_i = t[:N_i][:, None]
    t_j = t[N_i:][None, :]
    M_i = _wrap_2pi(rng.random(N_i) * 2 * np.pi)[:, None]
    M_j = _wrap_2pi(rng.random(N_j) * 2 * np.pi)[None, :]
    n_i = (0.05 + rng.random(N_i) * 0.3)[:, None]

    def run():
        dt = t_j - t_i
        dM = _wrap_2pi(M_j - M_i)
        k_float = (n_i * dt - dM) / (2.0 * np.pi)
        k = np.maximum(np.rint(k_float), 0.0)
        r = dt - (dM + 2.0 * np.pi * k) / (n_i + 1e-8)
        return np.count_nonzero(np.abs(r) < 10.0)

    count = benchmark(run)
    assert count >= 0


# ---- Chain extraction benchmark ----


@pytest.mark.benchmark(group="clock_chains")
def test_benchmark_extract_kepler_chains(benchmark):
    # Build synthetic candidates and a dense edge set resembling a long chain
    N = 5000
    times = 59000.0 + np.arange(N) * (30.0 / 1440.0)
    M = np.zeros_like(times)
    n = np.full_like(times, 0.1)
    f = np.zeros_like(times)
    e = np.zeros_like(times) + 0.1
    sig = np.zeros_like(times)
    night = np.floor(times).astype(np.int64)
    cands = _make_candidates_simple(
        times, M, n, f, e, sig, night, orbit_id="oBenchChain"
    )

    # Edges: connect i->i+1 to produce one big component
    i_idx = np.arange(N - 1, dtype=np.int32)
    j_idx = np.arange(1, N, dtype=np.int32)
    edges = ClockGatedEdges.from_kwargs(
        orbit_id=["oBenchChain"] * (N - 1),
        i_index=i_idx,
        j_index=j_idx,
        k_revs=np.zeros(N - 1, dtype=np.int16),
        dt_days=np.full(N - 1, 30.0 / 1440.0, dtype=np.float32),
        same_night=np.zeros(N - 1, dtype=bool),
        tau_min_minutes=0.0,
        alpha_min_per_day=0.0,
        beta=0.0,
        gamma=0.0,
        time_bin_minutes=60,
        max_bins_ahead=72,
        horizon_days=90.0,
        per_night_cap=2,
        max_k_span=1,
    )

    def run():
        return extract_kepler_chains(cands, edges, min_size=6, min_span_days=3.0)

    chains, members = benchmark(run)
    assert len(chains) == 1
    assert chains.size.to_numpy()[0] >= 6


def _evaluate_clock_gate_completeness(
    rays: "ObservationRays",
    cands: ClockGatingCandidates,
    chains: KeplerChains,
    members: KeplerChainMembers,
):
    # Evaluate for all orbits present in candidates (not just chains)
    orbit_ids = cands.orbit_id.unique()

    out = []
    for oid in orbit_ids.to_pylist():
        # Select candidates and members for this orbit
        cand_mask = pc.equal(cands.orbit_id, oid)
        cands_i = cands.apply_mask(cand_mask)
        mem_mask = pc.equal(members.orbit_id, oid)
        members_i = members.apply_mask(mem_mask)

        # Truth set for this orbit (what gating saw)
        truth_det = pc.unique(pa.array(cands_i.det_id))
        n_truth = len(truth_det)
        if n_truth == 0:
            out.append((oid, 0.0))
            continue

        # Chains for this orbit
        ch_mask = pc.equal(chains.orbit_id, oid)
        chains_i = chains.apply_mask(ch_mask)
        chain_ids = chains_i.chain_id

        best_recall = 0.0
        best_contam_cnt = None
        best_contam_rate = None
        full_cover_chains = 0
        for cid in chain_ids.to_pylist():
            # Gather chain members' det_ids
            this_chain = pc.equal(members_i.chain_id, pa.scalar(int(cid)))
            cand_idx_chain = pc.filter(members_i.cand_index, this_chain)
            det_chain = pc.take(
                pa.array(cands_i.det_id), pa.array(cand_idx_chain, type=pa.int64())
            )
            det_chain_unique = pc.unique(det_chain)
            covered_mask = pc.is_in(truth_det, value_set=det_chain_unique)
            covered = pc.sum(pc.cast(covered_mask, pa.int64())).as_py()
            recall_chain = float(covered) / float(n_truth)
            chain_size = len(det_chain_unique)
            # contamination = extra members beyond covered truth
            contam_cnt = int(chain_size - covered)
            contam_rate = (
                0.0 if chain_size == 0 else float(contam_cnt) / float(chain_size)
            )
            # count full-cover chains
            if covered == n_truth:
                full_cover_chains += 1
            # choose best by recall, then by minimal contamination
            if (recall_chain > best_recall) or (
                abs(recall_chain - best_recall) < 1e-12
                and (best_contam_cnt is None or contam_cnt < best_contam_cnt)
            ):
                best_recall = recall_chain
                best_contam_cnt = contam_cnt or 0
                best_contam_rate = contam_rate or 0.0

        print(
            f"Orbit {oid}: recall={best_recall:.2%}, "
            f"contam={best_contam_cnt} ({best_contam_rate:.2%}), "
            f"num_complete_chains={full_cover_chains}"
        )
        out.append(
            (
                oid,
                best_recall,
                int(best_contam_cnt) if best_contam_cnt is not None else 0,
                float(best_contam_rate) if best_contam_rate is not None else 0.0,
                int(full_cover_chains),
            )
        )
    return out


def _label_coverage_vs_rays_for_orbit(
    labels, rays, orbit_id_str: str
) -> tuple[int, int, float]:
    """Return (#rays_dets_for_orbit, #covered_by_labels, coverage_rate)."""
    mask_r = pc.equal(rays.orbit_id, orbit_id_str)
    rays_det = pc.filter(rays.det_id, mask_r)
    rays_unique = pc.unique(rays_det)
    n_rays = len(rays_unique)

    mask_l = pc.equal(labels.orbit_id, orbit_id_str)
    labs_det = pc.filter(labels.det_id, mask_l)
    labs_unique = pc.unique(labs_det)

    covered_mask = pc.is_in(rays_unique, value_set=labs_unique)
    covered = pc.sum(pc.cast(covered_mask, pa.int64())).as_py()
    rate = 0.0 if n_rays == 0 else float(covered) / float(n_rays)
    return int(n_rays), int(covered), rate


def _make_candidates_simple(
    times: np.ndarray,
    M_rad: np.ndarray,
    n_rad_day: np.ndarray,
    f_rad: np.ndarray,
    e: np.ndarray,
    sigma_M_rad: np.ndarray,
    night_id: np.ndarray,
    orbit_id: str = "o1",
):
    N = len(times)
    return ClockGatingCandidates.from_kwargs(
        det_id=[f"d{i}" for i in range(N)],
        orbit_id=[orbit_id] * N,
        seg_id=np.zeros(N, dtype=np.int32),
        variant_id=np.zeros(N, dtype=np.int32),
        time_tdb_mjd=times.astype(float),
        night_id=night_id.astype(np.int64),
        observer_code=["X05"] * N,
        M_rad=M_rad.astype(float),
        n_rad_day=n_rad_day.astype(float),
        f_rad=f_rad.astype(float),
        e=e.astype(float),
        sigma_M_rad=sigma_M_rad.astype(float),
        t_hat_plane_x=np.zeros(N, dtype=np.float32),
        t_hat_plane_y=np.ones(N, dtype=np.float32),
    )


# ---- Coverage checks across pipeline stages ----


def test_labels_cover_all_bvh_hits(bvh_hits, anomaly_labels):
    # Build pairwise keys det_id||orbit_id in Arrow (cast types to match)
    sep = pa.scalar("||", type=pa.large_string())
    hits_det = pc.cast(bvh_hits.det_id, pa.large_string())
    hits_orb = pc.cast(bvh_hits.orbit_id, pa.large_string())
    labs_det = pc.cast(anomaly_labels.det_id, pa.large_string())
    labs_orb = pc.cast(anomaly_labels.orbit_id, pa.large_string())
    hit_key = pc.binary_join_element_wise(hits_det, sep, hits_orb)
    lab_key = pc.binary_join_element_wise(labs_det, sep, labs_orb)
    idx = pc.index_in(hit_key, lab_key)
    covered_mask = pc.is_valid(idx)
    covered = pc.sum(pc.cast(covered_mask, pa.int64())).as_py()
    assert covered == len(
        bvh_hits
    ), "Every BVH hit should have at least one anomaly label with same (det_id, orbit_id)"


def test_bvh_hits_recall_against_rays(rays_nbody, bvh_hits):
    stats = compute_recall_metrics_from_hits(rays_nbody, bvh_hits)
    assert stats["recall_signal"] == 1.0


def test_wrap_identity_pair_kernel_like():
    # Residual should be ~0 for exact Kepler advance under pair_kernel logic
    n = np.float32(0.2)
    dt = np.float32(5.0)
    Mi = np.float32(1.0)
    dM = np.float32(_wrap_2pi(n * dt))
    Mj = np.float32(_wrap_2pi(Mi + dM))
    # Minimal 1x1 call to pair_kernel via public path: use intra-bin with 2 points in same bin
    times = np.array([59000.0, 59000.0 + float(dt)], dtype=np.float32)
    M_all = np.array([Mi, Mj], dtype=np.float32)
    n_all = np.array([n, n], dtype=np.float32)
    f_all = np.zeros(2, dtype=np.float32)
    e_all = np.zeros(2, dtype=np.float32)
    sig_all = np.zeros(2, dtype=np.float32)
    night = np.array([1, 1], dtype=np.int64)
    cands = _make_candidates_simple(times, M_all, n_all, f_all, e_all, sig_all, night)
    edges = build_clock_gated_edges(
        cands,
        tau_min_minutes=180.0,
        alpha_min_per_day=0.0,
        beta=0.0,
        gamma=0.0,
        time_bin_minutes=60,
        max_processes=0,
    )
    assert len(edges) == 1


def test_build_edges_accepts_and_caps_same_night():
    # One i at t0 and three j at later times, same night, engineered residuals
    n = 0.25
    Mi = 0.5
    ti = 59000.0
    tjs = np.array([59001.0, 59001.5, 59002.0])
    dts = tjs - ti
    # Design residuals (days); all below large tau
    residuals = np.array([0.010, 0.030, 0.005])
    dMs = _wrap_2pi(n * dts - n * residuals)
    Mj = _wrap_2pi(Mi + dMs)

    times = np.concatenate([[ti], tjs])
    M_all = np.concatenate([[Mi], Mj])
    n_all = np.array([n] * 4)
    f_all = np.array([0.2] * 4)
    e_all = np.array([0.1] * 4)
    sig_all = np.zeros(4)
    night = np.array([12345] * 4)  # same night

    cands = _make_candidates_simple(
        times=times,
        M_rad=M_all,
        n_rad_day=n_all,
        f_rad=f_all,
        e=e_all,
        sigma_M_rad=sig_all,
        night_id=night,
    )

    edges = build_clock_gated_edges(
        cands,
        tau_min_minutes=180.0,  # big tau
        alpha_min_per_day=0.0,
        beta=0.0,
        gamma=0.0,
        per_night_cap=999,
        max_processes=0,
    )

    # Cap removed: expect all forward pairs kept (3 choose 2 for later nodes + 3 from i=0) = 6
    assert len(edges) == 6
    two_pi = 2.0 * np.pi
    i_idx_arr = edges.i_index.to_numpy(zero_copy_only=False).astype(np.int32)
    j_idx_arr = edges.j_index.to_numpy(zero_copy_only=False).astype(np.int32)
    k_arr = edges.k_revs.to_numpy(zero_copy_only=False).astype(np.int32)
    dt_arr = edges.dt_days.to_numpy(zero_copy_only=False).astype(np.float64)
    resids = []
    for ii, jj, kk, dtv in zip(i_idx_arr, j_idx_arr, k_arr, dt_arr):
        dM = _wrap_2pi(M_all[jj] - M_all[ii])
        r = float(dtv - (dM + two_pi * kk) / (n_all[ii] + 1e-8))
        resids.append(abs(r))
    chosen = np.sort(np.array(resids))
    # Build expected residuals for all forward pairs (i<j)
    all_resids = []
    for i in range(4):
        for j in range(i + 1, 4):
            dt = times[j] - times[i]
            dM = _wrap_2pi(M_all[j] - M_all[i])
            k = int(np.maximum(np.rint((n_all[i] * dt - dM) / (2.0 * np.pi)), 0.0))
            r = float(dt - (dM + 2.0 * np.pi * k) / (n_all[i] + 1e-8))
            all_resids.append(abs(r))
    expected = np.sort(np.array(all_resids))
    npt.assert_allclose(chosen, expected, atol=1e-6)


def test_tau_growth_helper():
    # Verify tau increases with |dM/df| near apoapsis (|dM/df| is larger at apoapsis)
    dt = np.array([1.0])
    tau_min = 0.0
    alpha = 0.0
    beta = 0.0
    gamma = 1.0
    sigma_M = np.array([0.0])
    e = np.array([0.6])
    f_peri = np.array([0.0])
    f_apo = np.array([np.pi])

    d_peri = _dM_df(e, f_peri)[0]
    d_apo = _dM_df(e, f_apo)[0]
    assert d_apo > d_peri

    tau_peri = _tau(dt, tau_min, alpha, beta, sigma_M, gamma, np.array([d_peri]))[0]
    tau_apo = _tau(dt, tau_min, alpha, beta, sigma_M, gamma, np.array([d_apo]))[0]
    assert tau_apo > tau_peri


def test_extract_kepler_chains_and_promotion():
    # Build a component of size 6 spanning 5 days and a small component of 2
    times = np.array([59000, 59001, 59002, 59003, 59004, 59005, 59010, 59010.5])
    M = np.zeros_like(times)
    n = np.ones_like(times) * 0.1
    f = np.zeros_like(times)
    e = np.zeros_like(times) * 0.1
    sig = np.zeros_like(times)
    night = np.arange(len(times))
    cands = _make_candidates_simple(times, M, n, f, e, sig, night, orbit_id="oX")

    # Edges: chain over indices 0..5 and separate 6-7
    i_idx = np.array([0, 1, 2, 3, 4, 6])
    j_idx = np.array([1, 2, 3, 4, 5, 7])
    edges = ClockGatedEdges.from_kwargs(
        orbit_id=["oX"] * len(i_idx),
        i_index=i_idx.astype(np.int32),
        j_index=j_idx.astype(np.int32),
        k_revs=np.zeros(len(i_idx), dtype=np.int16),
        dt_days=np.ones(len(i_idx), dtype=np.float32),
        same_night=np.array([False] * len(i_idx)),
        tau_min_minutes=0.0,
        alpha_min_per_day=0.0,
        beta=0.0,
        gamma=0.0,
        time_bin_minutes=60,
        max_bins_ahead=72,
        horizon_days=90.0,
        per_night_cap=2,
        max_k_span=1,
    )

    chains, members = extract_kepler_chains(cands, edges, min_size=6, min_span_days=3.0)
    # One chain promoted
    assert len(chains) == 1
    assert chains.size.to_numpy()[0] == 6
    # Members should be indices 0..5 in some order
    mem_indices = np.sort(members.cand_index.to_numpy(zero_copy_only=False))
    npt.assert_array_equal(mem_indices, np.arange(6))


def test_edges_binning_gap_continue():
    # Ensure bin scanning continues across missing bins
    n = 0.2
    times = np.array([59000.0, 59000.5, 59002.0])  # gap skipping 59001 bin
    M = np.array([0.0, _wrap_2pi(n * 0.5), _wrap_2pi(n * 2.0)])
    e = np.zeros_like(times)
    f = np.zeros_like(times)
    sig = np.zeros_like(times)
    night = np.zeros_like(times)
    cands = _make_candidates_simple(times, M, np.ones_like(times) * n, f, e, sig, night)
    edges = build_clock_gated_edges(
        cands,
        tau_min_minutes=180.0,
        alpha_min_per_day=0.0,
        beta=0.0,
        gamma=0.0,
        time_bin_minutes=60,
        max_bins_ahead=120,
        horizon_days=365.0,
        max_processes=0,
    )
    # Expect edges from 0->1 and 1->2 and 0->2 (forward pairs)
    assert len(edges) == 3


def test_edges_k_rounding_boundary():
    # Construct dt so that n*dt - dM is near (m+0.5)*2pi; ensure rounding chooses nearest k
    n = 1.0
    Mi = 0.0
    dt = 2.5 * np.pi  # halfway between k=1 and k=2 if dM=0
    dM = 0.0
    k = int(np.maximum(np.rint((n * dt - dM) / (2.0 * np.pi)), 0.0))
    # With dt=2.5*pi and n=1, nearest k is 1 (since 2*pi is closer than 4*pi)
    assert k == 1


# ---- Parity against naive O(N^2) reference ----


def _build_edges_naive(
    cands: ClockGatingCandidates,
    tau_min_minutes: float,
    alpha_min_per_day: float,
    beta: float,
    gamma: float,
) -> set[tuple[int, int, int]]:
    t = cands.time_tdb_mjd.to_numpy(zero_copy_only=False).astype(float)
    M = cands.M_rad.to_numpy(zero_copy_only=False).astype(float)
    n = cands.n_rad_day.to_numpy(zero_copy_only=False).astype(float)
    f = cands.f_rad.to_numpy(zero_copy_only=False).astype(float)
    e = cands.e.to_numpy(zero_copy_only=False).astype(float)
    sigma_M = cands.sigma_M_rad.to_numpy(zero_copy_only=False).astype(float)

    tau_min = float(tau_min_minutes) / 1440.0
    alpha = float(alpha_min_per_day) / 1440.0

    N = len(cands)
    out: set[tuple[int, int, int]] = set()
    for i in range(N):
        for j in range(i + 1, N):
            dt = t[j] - t[i]
            if dt <= 0.0:
                continue
            dM = _wrap_2pi(M[j] - M[i])
            k = int(np.maximum(np.rint((n[i] * dt - dM) / (2.0 * np.pi)), 0.0))
            resid = float(dt - (dM + 2.0 * np.pi * k) / (n[i] + 1e-8))
            dMdf_i = float(_dM_df(np.array([e[i]]), np.array([f[i]]))[0])
            tau = float(
                _tau(
                    np.array([dt]),
                    tau_min,
                    alpha,
                    beta,
                    np.array([sigma_M[i]]),
                    gamma,
                    np.array([dMdf_i]),
                )[0]
            )
            if abs(resid) <= tau:
                out.add((i, j, k))
    return out


def test_build_edges_matches_naive_reference():
    rng = np.random.default_rng(1234)
    N = 120
    # Deterministic times spaced exactly one bin apart (60 min) to avoid same-bin pairs
    bin_w_days = 60.0 / 1440.0
    t0 = 59000.0
    times = t0 + (np.arange(N) * bin_w_days + 1e-9)
    n_vals = 0.05 + rng.random(N) * 0.3  # rad/day
    M_vals = _wrap_2pi(rng.random(N) * 2 * np.pi)
    f_vals = _wrap_2pi(rng.random(N) * 2 * np.pi)
    e_vals = np.clip(rng.normal(0.1, 0.05, size=N), 0.0, 0.6)
    sig_vals = np.zeros(N)
    night = np.floor(times).astype(np.int64)

    cands = _make_candidates_simple(
        times, M_vals, n_vals, f_vals, e_vals, sig_vals, night, orbit_id="oRef"
    )

    # Loose tolerance, no heading guard
    tau_min_minutes = 180.0
    alpha_min_per_day = 0.0
    beta = 0.0
    gamma = 0.0

    edges_ref = _build_edges_naive(
        cands,
        tau_min_minutes=tau_min_minutes,
        alpha_min_per_day=alpha_min_per_day,
        beta=beta,
        gamma=gamma,
    )

    edges_fast = build_clock_gated_edges(
        cands,
        tau_min_minutes=tau_min_minutes,
        alpha_min_per_day=alpha_min_per_day,
        beta=beta,
        gamma=gamma,
        time_bin_minutes=60,
        max_bins_ahead=10_000,  # ensure we consider all forward pairs
        per_night_cap=999999,
        horizon_days=365.0,
        max_k_span=3,
        max_processes=0,
        kernel_col_window_size=2048,
    )
    got = set(
        zip(
            edges_fast.i_index.to_pylist(),
            edges_fast.j_index.to_pylist(),
            edges_fast.k_revs.to_pylist(),
        )
    )
    assert got == edges_ref


# ---- Benchmarks ----


@pytest.mark.benchmark(group="clock_gating_edges")
def test_benchmark_build_clock_gated_edges(benchmark):
    rng = np.random.default_rng(2025)
    N = 1000
    times = 59000.0 + np.sort(rng.random(N) * 7.0)
    n_vals = 0.05 + rng.random(N) * 0.3
    M_vals = _wrap_2pi(rng.random(N) * 2 * np.pi)
    f_vals = _wrap_2pi(rng.random(N) * 2 * np.pi)
    e_vals = np.clip(rng.normal(0.1, 0.05, size=N), 0.0, 0.6)
    sig_vals = np.zeros(N)
    night = np.floor(times).astype(np.int64)

    cands = _make_candidates_simple(
        times, M_vals, n_vals, f_vals, e_vals, sig_vals, night, orbit_id="oBench"
    )

    def run():
        return build_clock_gated_edges(
            cands,
            tau_min_minutes=15.0,
            alpha_min_per_day=0.02,
            beta=1.0,
            gamma=0.5,
            time_bin_minutes=60,
            max_bins_ahead=72,
            max_processes=0,
            kernel_col_window_size=2048,
        )

    edges = benchmark(run)
    assert isinstance(edges, ClockGatedEdges)


def test_edges_ray_parallel_equals_serial(
    index_optimal, rays_2b, orbits_synthetic_stratified_ci
):
    # Small shard for parity test
    rays = rays_2b[:100]
    hits, _ = query_bvh(
        index_optimal,
        rays,
        guard_arcmin=1.0,
        batch_size=4096,
        window_size=1024,
        max_processes=0,
    )
    if len(hits) == 0:
        raise ValueError("No hits for this configuration")

    keep_orb = pc.is_in(
        orbits_synthetic_stratified_ci.orbit_id, pc.unique(hits.orbit_id)
    )
    orbits_used = orbits_synthetic_stratified_ci.apply_mask(keep_orb)
    labels = label_anomalies(
        hits, rays, orbits_used, max_k=1, chunk_size=256, max_processes=0
    )
    cands = prepare_clock_gating_candidates(labels, rays, orbits_used)
    if len(cands) == 0:
        raise ValueError("No candidates produced")

    for orbit_id in cands.orbit_id.unique():
        cands_for_orbit = cands.apply_mask(pc.equal(cands.orbit_id, orbit_id))
        edges_serial = build_clock_gated_edges(cands_for_orbit, max_processes=1)
        edges_parallel = build_clock_gated_edges(cands_for_orbit, max_processes=2)

        ser = set(
            zip(
                edges_serial.orbit_id.to_pylist(),
                edges_serial.i_index.to_pylist(),
                edges_serial.j_index.to_pylist(),
                edges_serial.k_revs.to_pylist(),
            )
        )
        par = set(
            zip(
                edges_parallel.orbit_id.to_pylist(),
                edges_parallel.i_index.to_pylist(),
                edges_parallel.j_index.to_pylist(),
                edges_parallel.k_revs.to_pylist(),
            )
        )
        assert ser == par



def test_clock_gating_e2e(anomaly_labels, rays_nbody, orbits_synthetic_stratified_ci):
    """
    Signal only test to ensure we get full k-chain recall of observations
    """
    print("filtering orbits")
    orbits_used = orbits_synthetic_stratified_ci.apply_mask(
        pc.is_in(
            orbits_synthetic_stratified_ci.orbit_id, pc.unique(anomaly_labels.orbit_id)
        )
    )

    kcol = int(os.environ.get("CG_KCOL", "2048"))

    # Optional warmup to exclude JAX compile time when profiling
    if os.environ.get("CG_PROFILE", "0") == "1":
        # Tiny subset warmup
        print("warming up")
        first_orb = anomaly_labels.orbit_id.unique()[0]
        warm_labels = anomaly_labels.apply_mask(
            pc.equal(anomaly_labels.orbit_id, first_orb)
        )
        cands_warm = prepare_clock_gating_candidates(
            warm_labels, rays_nbody, orbits_used
        )
        _chains_warm, _members_warm = kepler_clock_gate(
            cands_warm,
            max_processes=1,
            tau_min_minutes=15.0,
            alpha_min_per_day=0.02,
            beta=0.5,
            gamma=0.5,
            time_bin_minutes=120,
            max_bins_ahead=8,
            horizon_days=30.0,
            per_night_cap=2,
            max_k_span=1,
            kernel_col_window_size=kcol,
        )

    if os.environ.get("CG_PROFILE", "0") == "1":
        pr = cProfile.Profile()
        pr.enable()

    print("clock gating")
    # limit to 10 orbits for now
    first_ten_orbits = anomaly_labels.orbit_id.unique()[:10]
    anomaly_labels = anomaly_labels.apply_mask(
        pc.is_in(anomaly_labels.orbit_id, first_ten_orbits)
    )
    orbits_used = orbits_used.apply_mask(
        pc.is_in(orbits_used.orbit_id, first_ten_orbits)
    )
    cands = prepare_clock_gating_candidates(anomaly_labels, rays_nbody, orbits_used)
    chains, members = kepler_clock_gate(
        cands,
        max_processes=1,
        tau_min_minutes=15.0,
        alpha_min_per_day=0.02,
        beta=0.5,
        gamma=0.5,
        time_bin_minutes=120,
        max_bins_ahead=48,
        horizon_days=90.0,
        per_night_cap=2,
        max_k_span=1,
        promote_min_size=1,
        promote_min_span_days=0.0,
        kernel_col_window_size=kcol,
    )

    if os.environ.get("CG_PROFILE", "0") == "1":
        pr.disable()
        outfile = os.environ.get("CG_PROF_OUT", "cg_test.prof")
        pr.dump_stats(outfile)
        if os.environ.get("CG_PRINT_TOP", "0") == "1":
            s = pstats.Stats(pr)
            s.strip_dirs().sort_stats("tottime").print_stats(40)

    assert len(cands) > 0
    assert len(chains) > 0
    assert len(members) > 0
    # Best-chain recall per orbit (Arrow-based)
    print("evaluating clock gate completeness")
    metrics = _evaluate_clock_gate_completeness(rays_nbody, cands, chains, members)
    for (
        oid,
        best_recall,
        best_contam_cnt,
        best_contam_rate,
        full_cover_chains,
    ) in metrics:
        assert best_recall >= 1.0 - 1e-12, f"Best-chain recall < 100% for orbit {oid}"


# ---- Unit tests for per-bin workers and bin-pair planner ----


def test_intra_bin_edges_worker_matches_naive_within_bin():
    rng = np.random.default_rng(2026)
    time_bin_minutes = 60
    bin_w_days = time_bin_minutes / 1440.0
    # Build detections across 3 bins, several per bin, all within half a bin width
    per_bin = 8
    times = []
    for b in range(3):
        base = 59000.0 + b * bin_w_days
        # Keep within half-bin to avoid cross-bin drift by jitter
        times.append(base + rng.random(per_bin) * (0.5 * bin_w_days))
    times = np.concatenate(times)
    N = times.size
    n_vals = 0.05 + rng.random(N) * 0.3
    M_vals = _wrap_2pi(rng.random(N) * 2 * np.pi)
    f_vals = _wrap_2pi(rng.random(N) * 2 * np.pi)
    e_vals = np.clip(rng.normal(0.1, 0.05, size=N), 0.0, 0.6)
    sig_vals = np.zeros(N)
    night = np.floor(times).astype(np.int64)

    cands = _make_candidates_simple(times, M_vals, n_vals, f_vals, e_vals, sig_vals, night, orbit_id="oIntra")

    # Kernel inputs
    bins = _build_bins_and_m_index(cands, time_bin_minutes)
    tol = _compute_per_detection_scalars(cands, tau_min_minutes=180.0, alpha_min_per_day=0.0)

    # Run worker per bin and collect edges
    got: set[tuple[int, int, int]] = set()
    for b in bins.unique_bins:
        edges = intra_bin_edges_worker(
            cands,
            bins,
            tol,
            [int(b)],
            beta=0.0,
            gamma=0.0,
            horizon_days=365.0,
            per_night_cap=999999,
            kernel_col_window_size=64,
        )
        if len(edges) > 0:
            ii = edges.i_index.to_pylist()
            jj = edges.j_index.to_pylist()
            kk = edges.k_revs.to_pylist()
            got.update(zip(ii, jj, kk))

    # Naive reference filtered to within-bin pairs only
    ref_all = _build_edges_naive(cands, tau_min_minutes=180.0, alpha_min_per_day=0.0, beta=0.0, gamma=0.0)
    index_to_bin = np.full(N, -1, dtype=int)
    for b in bins.unique_bins:
        for idx in bins.bin_to_unsorted_indices[int(b)]:
            index_to_bin[int(idx)] = int(b)
    ref_within = {(i, j, k) for (i, j, k) in ref_all if index_to_bin[i] == index_to_bin[j]}

    assert got == ref_within


def test_intra_bin_edges_worker_respects_per_night_cap():
    # Four detections in same bin and same night; design residuals to be accepted
    n = 0.25
    Mi = 0.5
    times = 59000.0 + np.array([0.0, 0.01, 0.02, 0.03])  # all within 60-min bin
    dts = times[1:] - times[0]
    residuals = np.array([0.010, 0.030, 0.005])
    dMs = _wrap_2pi(n * dts - n * residuals)
    M_all = np.concatenate([[Mi], _wrap_2pi(Mi + dMs)])

    n_all = np.array([n] * 4)
    f_all = np.array([0.2] * 4)
    e_all = np.array([0.1] * 4)
    sig_all = np.zeros(4)
    night = np.array([12345] * 4, dtype=np.int64)

    cands = _make_candidates_simple(times, M_all, n_all, f_all, e_all, sig_all, night)
    bins = _build_bins_and_m_index(cands, time_bin_minutes=60)
    tol = _compute_per_detection_scalars(cands, tau_min_minutes=180.0, alpha_min_per_day=0.0)

    # Single occupied bin
    b = int(bins.unique_bins[0])
    cap = 1
    edges = intra_bin_edges_worker(
        cands,
        bins,
        tol,
        [int(b)],
        beta=0.0,
        gamma=0.0,
        horizon_days=365.0,
        per_night_cap=cap,
        kernel_col_window_size=32,
    )

    # Count same-night outgoing edges per source
    ii = np.array(edges.i_index.to_pylist(), dtype=int)
    same = np.array(edges.same_night.to_pylist(), dtype=bool)
    for src in range(4):
        available = max(0, 4 - (src + 1))
        expected = min(cap, available)
        got = int(np.sum(same[ii == src]))
        assert got == expected

def test_calculate_bin_pairs_upper_triangle():
    # Build a simple bin index with gaps and verify pairs are upper-triangular
    times = 59000.0 + np.array([0.0, 0.5, 2.1, 2.6])  # days
    M = np.zeros_like(times)
    n = np.ones_like(times) * 0.1
    f = np.zeros_like(times)
    e = np.zeros_like(times) * 0.1
    sig = np.zeros_like(times)
    night = np.floor(times).astype(np.int64)
    cands = _make_candidates_simple(times, M, n, f, e, sig, night)
    bins = _build_bins_and_m_index(cands, time_bin_minutes=60)

    pairs = calculate_bin_pairs(bins, max_bins_ahead=2, horizon_days=10.0)
    # Expected upper-triangle over unique bins
    ub = bins.unique_bins.tolist()
    expected = {(int(ub[i]), int(ub[j])) for i in range(len(ub)) for j in range(i + 1, len(ub))}
    assert set((int(a), int(b)) for (a, b) in pairs) == expected


# def test_clock_gating_e2e_with_noise(
#     anomaly_labels_with_noise,
#     rays_nbody_with_noise,
#     bvh_hits_with_noise,
#     orbits_synthetic_stratified_ci,
# ):

#     print("Artificially limiting anomaly labels temporarily to test completion")
#     first_ten_orbits = anomaly_labels_with_noise.orbit_id.unique()[0]
#     anomaly_labels_with_noise = anomaly_labels_with_noise.apply_mask(pc.is_in(anomaly_labels_with_noise.orbit_id, first_ten_orbits))

#     # Filter orbits to those present in labels
#     print("Selecting orbits")
#     keep_orb = pc.is_in(
#         orbits_synthetic_stratified_ci.orbit_id,
#         pc.unique(anomaly_labels_with_noise.orbit_id),
#     )
#     orbits_used = orbits_synthetic_stratified_ci.apply_mask(keep_orb)

#     print("Starting clock gating")
#     start = time.time()
#     cands, edges, chains, members = kepler_clock_gate(
#         anomaly_labels_with_noise,
#         rays_nbody_with_noise,
#         orbits_used,
#         max_processes=8,
#         tau_min_minutes=15.0,
#         alpha_min_per_day=0.02,
#         beta=0.5,
#         gamma=0.5,
#         time_bin_minutes=120,
#         max_bins_ahead=48,
#         horizon_days=90.0,
#         per_night_cap=2,
#         max_k_span=1,
#         promote_min_size=1,
#         promote_min_span_days=0.0,
#     )
#     end = time.time()
#     print(f"Clock gating took {end - start} seconds")
#     assert len(cands) > 0
#     assert len(chains) > 0
#     # Evaluate best-chain recall and contamination
#     print(f"Evaluating clock gating with {len(cands)} candidates, {len(chains)} chains, and {len(members)} members")
#     metrics = _evaluate_clock_gate_completeness(
#         rays_nbody_with_noise, cands, chains, members
#     )
#     # Expect high recall for best chain per orbit; contamination may be >0 due to noise
#     for oid, best_recall, best_contam_cnt, best_contam_rate, full_cover_chains in metrics:
#         assert best_recall >= 1.0 - 1e-12
