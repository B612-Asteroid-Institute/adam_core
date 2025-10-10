import numpy as np
import numpy.testing as npt
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
    _choose_k,
    _clock_residual,
    _dM_df,
    _tau,
    _wrap_2pi,
    build_clock_gated_edges,
    extract_kepler_chains,
    kepler_clock_gate,
)


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
        sky_pa_deg=np.array([np.nan] * N, dtype=np.float32),
    )


def test_wrap_choosek_residual_identity():
    # Single i->j pair with exact Kepler consistency: residual ~ 0
    n = 0.2  # rad/day
    dt = 5.0  # days
    Mi = 1.0
    dM = _wrap_2pi(n * dt)  # choose k=0, below 2pi
    Mj = _wrap_2pi(Mi + dM)
    k = _choose_k(np.array([n]), np.array([dt]), np.array([dM]))[0]
    r = _clock_residual(np.array([n]), np.array([dt]), np.array([dM]), np.array([k]))[0]
    assert k == 0
    npt.assert_allclose(r, 0.0, atol=1e-12)


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
        max_processes=0,
    )

    # Cap removed: expect all forward pairs kept (3 choose 2 for later nodes + 3 from i=0) = 6
    assert len(edges) == 6
    chosen = np.sort(np.abs(edges.resid_days.to_numpy(zero_copy_only=False)))
    # Build expected residuals for all forward pairs (i<j)
    all_resids = []
    for i in range(4):
        for j in range(i + 1, 4):
            dt = times[j] - times[i]
            dM = _wrap_2pi(M_all[j] - M_all[i])
            k = _choose_k(np.array([n_all[i]]), np.array([dt]), np.array([dM]))[0]
            r = _clock_residual(
                np.array([n_all[i]]), np.array([dt]), np.array([dM]), np.array([k])
            )[0]
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
        resid_days=np.zeros(len(i_idx), dtype=np.float32),
        dt_days=np.ones(len(i_idx), dtype=np.float32),
        dM_wrapped_rad=np.zeros(len(i_idx), dtype=np.float32),
        same_night=np.array([False] * len(i_idx)),
        tau_min_minutes=0.0,
        alpha_min_per_day=0.0,
        beta=0.0,
        gamma=0.0,
        time_bin_minutes=60,
        max_bins_ahead=72,
        heading_max_deg=np.nan,
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
    k = _choose_k(np.array([n]), np.array([dt]), np.array([dM]))[0]
    # With dt=2.5*pi and n=1, nearest k is 1 (since 2*pi is closer than 4*pi)
    assert k == 1


def test_edges_heading_guard_filters():
    # Provide sky_pa_deg to force reject if heading_max_deg very small
    times = np.array([59000.0, 59001.0])
    n = np.array([0.1, 0.1])
    M = np.array([0.0, _wrap_2pi(0.1)])
    f = np.zeros_like(times)
    e = np.zeros_like(times)
    sig = np.zeros_like(times)
    night = np.zeros_like(times)
    cands = _make_candidates_simple(times, M, n, f, e, sig, night)
    # Inject distinct sky PAs
    cands = cands.set_column("sky_pa_deg", np.array([0.0, 180.0], dtype=np.float32))
    edges = build_clock_gated_edges(
        cands,
        tau_min_minutes=180.0,
        alpha_min_per_day=0.0,
        beta=0.0,
        gamma=0.0,
        time_bin_minutes=60,
        max_bins_ahead=72,
        heading_max_deg=10.0,
        max_processes=0,
    )
    # Heading guard should reject the only forward pair
    assert len(edges) == 0


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

    edges_serial = build_clock_gated_edges(cands, max_processes=0)
    edges_parallel = build_clock_gated_edges(cands, max_processes=2)
    # Compare sets of tuples for equality
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

    orbits_used = orbits_synthetic_stratified_ci.apply_mask(pc.is_in(orbits_synthetic_stratified_ci.orbit_id, pc.unique(anomaly_labels.orbit_id)))

    cands, edges, chains, members = kepler_clock_gate(
        anomaly_labels,
        rays_nbody,
        orbits_used,
        max_processes=1,
        tau_min_minutes=15.0,
        alpha_min_per_day=0.02,
        beta=0.5,
        gamma=0.5,
        time_bin_minutes=120,
        max_bins_ahead=48,
        heading_max_deg=None,
    )
    assert len(cands) > 0
    assert len(edges) > 0
    assert len(chains) > 0
    assert len(members) > 0
