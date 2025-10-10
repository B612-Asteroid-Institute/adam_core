from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np
import pyarrow.compute as pc
import pytest
import quivr as qv

from adam_core.geometry.bvh.index import build_bvh_index_from_segments
from adam_core.geometry.bvh.query import (
    OverlapHits,
    calc_ray_segment_distance_and_guard,
    query_bvh,
    ray_segment_distance_window,
)
from adam_core.geometry.rays import ObservationRays

# Grid infrastructure removed - using fixed optimal parameters


def compute_recall_metrics_from_hits(
    rays: ObservationRays, hits_tbl: OverlapHits
) -> Dict[str, float]:
    det_id = rays.det_id
    signal_mask_pa = pc.invert(pc.starts_with(det_id, "noise_"))
    signal_det = pc.filter(det_id, signal_mask_pa)
    signal_det_unique = pc.unique(signal_det)
    total_signal = len(signal_det_unique)

    if total_signal == 0:
        return {
            "recall_signal": 0.0,
            "fp_rate_signal": 0.0,
            "dup_rate_signal": 0.0,
            "total_signal": 0.0,
        }

    parts = pc.split_pattern(det_id, ":")
    truth_pa = pc.list_element(parts, 0)
    hit_det = hits_tbl.det_id
    hit_orb = hits_tbl.orbit_id

    idx_in_rays = pc.index_in(hit_det, det_id)
    truth_at_hit = pc.take(truth_pa, idx_in_rays)
    signal_mask_at_hit = pc.take(signal_mask_pa, idx_in_rays)

    correct_hit_mask = pc.equal(hit_orb, truth_at_hit)
    correct_signal_mask = pc.and_(signal_mask_at_hit, correct_hit_mask)
    incorrect_signal_mask = pc.and_(signal_mask_at_hit, pc.invert(correct_hit_mask))

    hit_det_correct = pc.filter(hit_det, correct_signal_mask)
    correct_det_unique = pc.unique(hit_det_correct)
    num_signal_hits = len(correct_det_unique)
    recall = num_signal_hits / total_signal if total_signal > 0 else 0.0

    hit_det_incorrect = pc.filter(hit_det, incorrect_signal_mask)
    incorrect_det_unique = pc.unique(hit_det_incorrect)
    fp_rate = len(incorrect_det_unique) / total_signal if total_signal > 0 else 0.0

    idx_in_signal = pc.index_in(hit_det, signal_det_unique)
    idx_np = np.asarray(pc.fill_null(idx_in_signal, -1))
    idx_np = idx_np[idx_np >= 0]
    if len(idx_np) == 0:
        dup_rate = 0.0
    else:
        counts = np.bincount(idx_np, minlength=total_signal)
        dup_rate = float(np.sum(counts > 1)) / float(total_signal)

    # Noise counts and rate (available vs hits)
    noise_mask_pa = pc.starts_with(det_id, "noise_")
    noise_det_unique = pc.unique(pc.filter(det_id, noise_mask_pa))
    num_noise_available = len(noise_det_unique)
    noise_mask_at_hit = pc.take(noise_mask_pa, idx_in_rays)
    hit_det_noise = pc.filter(hit_det, noise_mask_at_hit)
    noise_det_hits_unique = pc.unique(hit_det_noise)
    num_noise_hits = len(noise_det_hits_unique)
    noise_hit_rate = (
        (num_noise_hits / num_noise_available) if num_noise_available > 0 else 0.0
    )

    return {
        "recall_signal": float(recall),
        "fp_rate_signal": float(fp_rate),
        "dup_rate_signal": float(dup_rate),
        "total_signal": float(total_signal),
        "num_signal_available": int(total_signal),
        "num_signal_hits": int(num_signal_hits),
        "num_noise_available": int(num_noise_available),
        "num_noise_hits": int(num_noise_hits),
        "noise_hit_rate": float(noise_hit_rate),
    }


# Grid infrastructure removed - using fixed optimal parameters


def test_query_bvh_basic_and_max_processes(segments_aabbs, rays_2b):
    # Build a fresh index to ensure valid depth/stack size
    index = build_bvh_index_from_segments(segments_aabbs[:256], max_leaf_size=32)
    hits0, _ = query_bvh(
        index, rays_2b, guard_arcmin=5.0, batch_size=64, max_processes=0
    )
    hits2, _ = query_bvh(
        index, rays_2b, guard_arcmin=5.0, batch_size=64, max_processes=2
    )
    assert isinstance(hits0, OverlapHits) and isinstance(hits2, OverlapHits)
    assert len(hits0) == len(hits2)


def test_query_bvh_raises_on_empty_inputs(segments_aabbs, rays_2b):
    index = build_bvh_index_from_segments(segments_aabbs[:64], max_leaf_size=16)
    from adam_core.geometry.rays import ObservationRays

    with pytest.raises(ValueError, match="rays is empty"):
        query_bvh(index, ObservationRays.empty())

    # Empty index
    from adam_core.geometry.bvh.index import BVHIndex, BVHNodes, BVHPrimitives

    empty_index = BVHIndex(
        segments=index.segments[:0], nodes=BVHNodes.empty(), prims=BVHPrimitives.empty()
    )
    with pytest.raises(ValueError, match="index has no nodes/primitives"):
        query_bvh(empty_index, rays_2b)


def test_distances_and_guard_pairs_jax_shapes():
    W = 64
    rng = np.random.default_rng(0)
    ro = rng.uniform(-5, 5, (W, 3))
    rd = rng.normal(0, 1, (W, 3))
    rd /= np.linalg.norm(rd, axis=1, keepdims=True)
    s0 = rng.uniform(-10, 10, (W, 3))
    s1 = rng.uniform(-10, 10, (W, 3))
    r_mid = rng.uniform(0.5, 3.0, (W,))
    d_obs = rng.uniform(0.9, 1.1, (W,))
    theta = 1.0 * np.pi / (180 * 60)

    distances, mask = calc_ray_segment_distance_and_guard(
        ro, rd, s0, s1, r_mid, d_obs, theta
    )
    assert distances.shape == (W,) and mask.shape == (W,)


def test_distances_and_guard_pairs_threshold_behavior():
    # Construct an easy case: identical segments and rays intersecting near origin
    ro = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    rd = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    s0 = np.array([[1.0, -0.01, 0.0], [0.0, 1.0, 0.0]])
    s1 = np.array([[2.0, 0.01, 0.0], [0.0, 2.0, 0.0]])
    r_mid = np.array([1.0, 1.0])
    d_obs = np.array([1.0, 1.0])
    # First pair is within small guard, second is exactly on-line so distance 0
    theta_small = 0.5 * np.pi / (180 * 60)
    theta_tiny = 1e-8
    d_small, m_small = calc_ray_segment_distance_and_guard(
        ro, rd, s0, s1, r_mid, d_obs, theta_small
    )
    d_tiny, m_tiny = calc_ray_segment_distance_and_guard(
        ro, rd, s0, s1, r_mid, d_obs, theta_tiny
    )
    assert m_small.shape == (2,) and m_tiny.shape == (2,)
    # With tiny guard, only perfectly colinear pair (index 1) should pass
    assert bool(np.asarray(m_tiny)[1]) is True
    # With larger guard, both should pass
    assert np.all(np.asarray(m_small))


def test_ray_segment_distances_pairs_jax_shapes():
    W = 32
    rng = np.random.default_rng(1)
    ro = rng.uniform(-5, 5, (W, 3))
    rd = rng.normal(0, 1, (W, 3))
    rd /= np.linalg.norm(rd, axis=1, keepdims=True)
    s0 = rng.uniform(-10, 10, (W, 3))
    s1 = rng.uniform(-10, 10, (W, 3))
    d = ray_segment_distance_window(ro, rd, s0, s1)
    assert d.shape == (W,)


def test_query_bvh_e2e(index_optimal, rays_nbody):
    hits, telemetry = query_bvh(
        index_optimal,
        rays_nbody,
        guard_arcmin=0.65,
        window_size=32768,
        batch_size=16384,
        max_processes=1,
    )
    print(telemetry)
    assert isinstance(hits, OverlapHits)
    stats = compute_recall_metrics_from_hits(rays_nbody, hits)
    assert stats["recall_signal"] == 1.0


@pytest.mark.benchmark
def test_query_bvh_benchmark(benchmark, index_optimal, rays_nbody):
    """Benchmark BVH querying with optimal parameters (cached fixtures, n-body rays)."""

    def _run() -> OverlapHits:
        return query_bvh(
            index_optimal,
            rays_nbody,
            guard_arcmin=0.65,
            batch_size=16384,
            max_processes=1,
            window_size=32768,
        )

    result = benchmark(_run)
    hits, telemetry = result
    assert isinstance(hits, OverlapHits)
    stats = compute_recall_metrics_from_hits(rays_nbody, hits)

    # Assert 100% recall
    assert stats["recall_signal"] == 1.0

    benchmark.extra_info.update(
        {
            "max_chord_arcmin": 5.0,
            "index_guard_arcmin": 0.65,
            "max_leaf_size": 64,
            "max_segments_per_orbit": 512,
            "epsilon_n_au": 1e-9,
            "padding_method": "baseline",
            "query_guard_arcmin": 0.65,
            "window_size": 32768,
            "batch_size": 16384,
            "noise_per_sqdeg": 1.0,
            # Recall metrics
            "num_signal_available": stats["num_signal_available"],
            "num_signal_hits": stats["num_signal_hits"],
            "recall_signal": stats["recall_signal"],
            "dup_rate_signal": stats.get("dup_rate_signal", 0.0),
            "fp_rate_signal": stats.get("fp_rate_signal", 0.0),
            # Telemetry
            "pairs_total": telemetry.pairs_total,
            "pairs_within_guard": telemetry.pairs_within_guard,
            "truncation_occurred": telemetry.truncation_occurred,
            "max_leaf_visits_observed": telemetry.max_leaf_visits_observed,
            "rays_with_zero_candidates": telemetry.rays_with_zero_candidates,
            "packets_traversed": telemetry.packets_traversed,
        }
    )
