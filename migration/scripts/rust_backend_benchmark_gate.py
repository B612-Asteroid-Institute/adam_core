"""Rust-only latency gate for adam-core.

After legacy JAX/Numba paths were removed from production (task #121,
2026-04-23), there is no legitimate "legacy" comparison to run on every
gate. This script measures Rust-only p50/p95 latency per rust-default API
and compares against a pinned baseline:

    migration/artifacts/rust_latency_baseline.json

Historical rust-vs-legacy numbers are preserved in
`migration/artifacts/history/rust_vs_legacy_final_snapshot_2026-04-23.json`
with annotations in the matching `README.md`.

Modes
-----
- Default: measure current Rust, compare to baseline, fail on regression.
- `--capture-baseline`: measure current Rust and write the baseline file.
  Commit the produced file for future gate runs.

Parity checks are intentionally NOT performed here. Use the dedicated
parity tests (`src/adam_core/*/tests/test_rust_*_parity.py`) for
correctness validation; this script is strictly a performance gate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from adam_core._rust import API_MIGRATIONS
from adam_core._rust import api as rust_api
from adam_core.constants import Constants as c
from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.coordinates.keplerian import KeplerianCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.coordinates.transform import transform_coordinates
from adam_core.time import Timestamp

RNG = np.random.default_rng(20260414)
MU = float(c.MU)
GEODETIC_A = 4.26352124542639e-05
GEODETIC_F = 0.0033528106647474805

BASELINE_PATH = Path("migration/artifacts/rust_latency_baseline.json")
CURRENT_PATH = Path("migration/artifacts/rust_latency_current.json")

# Regression tolerances. A latency ratio > tolerance is a gate failure.
# Empirically observed run-to-run variance on shared dev/CI hardware with
# microbenchmarks in the 0.1–10 ms range is p50 ±40% and p95 ±100% (even
# with 7 repeats per measurement). Defaults are set to absorb that noise
# while still catching multi-x regressions. On dedicated benchmark
# hardware, tighten via --p50-max-ratio / --p95-max-ratio flags
# (e.g., 1.15 / 1.30).
DEFAULT_P50_MAX_RATIO = 1.75
DEFAULT_P95_MAX_RATIO = 2.50


def _timed_runs(fn: Callable[[], Any], repeats: int) -> tuple[np.ndarray, Any]:
    if repeats < 2:
        raise ValueError("repeats must be >= 2 to estimate p50/p95 runtime.")
    out: Any = fn()  # prime
    times = np.empty(repeats, dtype=np.float64)
    for idx in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times[idx] = time.perf_counter() - t0
    return times, out


def _latency_summary(rust_times: np.ndarray) -> dict[str, Any]:
    return {
        "rust_seconds_p50": float(np.percentile(rust_times, 50)),
        "rust_seconds_p95": float(np.percentile(rust_times, 95)),
        "rust_samples_seconds": rust_times.tolist(),
    }


# ---------- input builders (unchanged from legacy gate) -------------------


def _build_cartesian(n: int) -> np.ndarray:
    pos = RNG.normal(0.0, 2.0, size=(n, 3))
    vel = RNG.normal(0.0, 0.02, size=(n, 3))
    return np.ascontiguousarray(np.hstack([pos, vel]), dtype=np.float64)


def _build_spherical(n: int) -> np.ndarray:
    rho = RNG.uniform(1e-6, 20.0, size=n)
    lon = RNG.uniform(0.0, 360.0, size=n)
    lat = RNG.uniform(-90.0, 90.0, size=n)
    vrho = RNG.normal(0.0, 0.05, size=n)
    vlon = RNG.normal(0.0, 0.1, size=n)
    vlat = RNG.normal(0.0, 0.1, size=n)
    return np.ascontiguousarray(
        np.column_stack([rho, lon, lat, vrho, vlon, vlat]), dtype=np.float64
    )


def _build_geodetic_cartesian(n: int, a: float, f: float) -> np.ndarray:
    b = a * (1.0 - f)
    e2 = (a * a - b * b) / (a * a)
    lon = RNG.uniform(0.0, 2.0 * np.pi, size=n)
    lat = RNG.uniform(np.radians(-80.0), np.radians(80.0), size=n)
    alt = RNG.uniform(-1e-8, 1e-8, size=n)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    n_curv = a / np.sqrt(1.0 - e2 * sin_lat**2)
    radius = n_curv + alt
    x = radius * cos_lat * cos_lon
    y = radius * cos_lat * sin_lon
    z = (n_curv * (1.0 - e2) + alt) * sin_lat
    v_east = RNG.normal(0.0, 1e-8, size=n)
    v_north = RNG.normal(0.0, 1e-8, size=n)
    v_up = RNG.normal(0.0, 1e-8, size=n)
    vx = -sin_lon * v_east - sin_lat * cos_lon * v_north + cos_lat * cos_lon * v_up
    vy = cos_lon * v_east - sin_lat * sin_lon * v_north + cos_lat * sin_lon * v_up
    vz = cos_lat * v_north + sin_lat * v_up
    return np.ascontiguousarray(
        np.column_stack([x, y, z, vx, vy, vz]), dtype=np.float64
    )


def _build_mean_motion_inputs(n: int) -> tuple[np.ndarray, np.ndarray]:
    a = np.ascontiguousarray(RNG.uniform(0.1, 50.0, size=n), dtype=np.float64)
    mu = np.ascontiguousarray(RNG.uniform(1e-6, 10.0, size=n), dtype=np.float64)
    return a, mu


def _build_cometary_inputs(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = RNG.uniform(0.3, 30.0, size=n)
    e_ell = RNG.uniform(0.0, 0.9, size=n // 2)
    e_hyp = RNG.uniform(1.1, 2.0, size=n - n // 2)
    e = np.concatenate([e_ell, e_hyp])
    i = RNG.uniform(0.0, 180.0, size=n)
    raan = RNG.uniform(0.0, 360.0, size=n)
    ap = RNG.uniform(0.0, 360.0, size=n)
    t0 = RNG.uniform(59000.0, 60000.0, size=n)
    tp = t0 + RNG.uniform(-1.0, 1.0, size=n)
    coords = np.ascontiguousarray(
        np.column_stack([q, e, i, raan, ap, tp]), dtype=np.float64
    )
    t0 = np.ascontiguousarray(t0, dtype=np.float64)
    mu = np.ascontiguousarray(RNG.uniform(1e-6, 1e-2, size=n), dtype=np.float64)
    return coords, t0, mu


def _build_keplerian_a_inputs(n: int) -> tuple[np.ndarray, np.ndarray]:
    n_e = n // 2
    n_h = n - n_e
    a_e = RNG.uniform(0.1, 50.0, size=n_e)
    e_e = RNG.uniform(0.0, 0.9, size=n_e)
    a_h = -RNG.uniform(0.1, 50.0, size=n_h)
    e_h = RNG.uniform(1.1, 2.0, size=n_h)
    a = np.concatenate([a_e, a_h])
    e = np.concatenate([e_e, e_h])
    i = RNG.uniform(0.0, 180.0, size=n)
    raan = RNG.uniform(0.0, 360.0, size=n)
    ap = RNG.uniform(0.0, 360.0, size=n)
    m = RNG.uniform(0.0, 360.0, size=n)
    coords = np.ascontiguousarray(
        np.column_stack([a, e, i, raan, ap, m]), dtype=np.float64
    )
    mu = np.ascontiguousarray(RNG.uniform(1e-6, 1e-2, size=n), dtype=np.float64)
    return coords, mu


def _build_gibbs_triplets(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r1 = np.ascontiguousarray(RNG.normal(0.0, 1.0, size=(n, 3)), dtype=np.float64)
    r2 = np.ascontiguousarray(RNG.normal(0.0, 1.0, size=(n, 3)), dtype=np.float64)
    r3 = np.ascontiguousarray(RNG.normal(0.0, 1.0, size=(n, 3)), dtype=np.float64)
    r1[np.linalg.norm(r1, axis=1) < 1e-6, 0] = 1.0
    r2[np.linalg.norm(r2, axis=1) < 1e-6, 1] = 1.0
    r3[np.linalg.norm(r3, axis=1) < 1e-6, 2] = 1.0
    return r1, r2, r3


def _build_herrick_times(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t1 = RNG.uniform(59000.0, 60000.0, size=n)
    t2 = t1 + RNG.uniform(1e-3, 5.0, size=n)
    t3 = t2 + RNG.uniform(1e-3, 5.0, size=n)
    return (
        np.ascontiguousarray(t1, dtype=np.float64),
        np.ascontiguousarray(t2, dtype=np.float64),
        np.ascontiguousarray(t3, dtype=np.float64),
    )


def _gibbs_rust_batch(r1: np.ndarray, r2: np.ndarray, r3: np.ndarray) -> np.ndarray:
    out = np.empty((r1.shape[0], 3), dtype=np.float64)
    for i in range(r1.shape[0]):
        v = rust_api.calc_gibbs_numpy(r1[i], r2[i], r3[i], MU)
        assert v is not None
        out[i] = np.asarray(v, dtype=np.float64)
    return out


def _herrick_rust_batch(
    r1: np.ndarray,
    r2: np.ndarray,
    r3: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    t3: np.ndarray,
) -> np.ndarray:
    out = np.empty((r1.shape[0], 3), dtype=np.float64)
    for i in range(r1.shape[0]):
        v = rust_api.calc_herrick_gibbs_numpy(
            r1[i], r2[i], r3[i], t1[i], t2[i], t3[i], MU
        )
        assert v is not None
        out[i] = np.asarray(v, dtype=np.float64)
    return out


def _gauss_rust_batch(
    r1: np.ndarray,
    r2: np.ndarray,
    r3: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    t3: np.ndarray,
) -> np.ndarray:
    out = np.empty((r1.shape[0], 3), dtype=np.float64)
    for i in range(r1.shape[0]):
        v = rust_api.calc_gauss_numpy(r1[i], r2[i], r3[i], t1[i], t2[i], t3[i], MU)
        assert v is not None
        out[i] = np.asarray(v, dtype=np.float64)
    return out


def _build_gaussiod_inputs(n: int):
    """Synthetic 3-observation triplets (RA/Dec deg, observer pos AU, mjd)."""
    coords_lon = RNG.uniform(0.0, 360.0, size=(n, 3))
    coords_lat = RNG.uniform(-30.0, 30.0, size=(n, 3))
    # 1-2 day spacing (avoid extreme close-spacing Gibbs ill-conditioning).
    t1 = RNG.uniform(59000.0, 60000.0, size=n)
    times = np.column_stack([t1, t1 + 1.0, t1 + 2.0]).astype(np.float64)
    coords_obs = np.tile(np.array([0.97, 0.18, -0.001], dtype=np.float64), (n, 3, 1))
    coords_obs[:, 1, 0] += 0.005
    coords_obs[:, 2, 0] += 0.010
    return coords_lon, coords_lat, times, np.ascontiguousarray(coords_obs)


def _gauss_iod_rust_batch(coords_lon, coords_lat, times, coords_obs):
    """Loop over rust_api.gauss_iod_fused_numpy for N triplets."""
    n = times.shape[0]
    for i in range(n):
        rust_api.gauss_iod_fused_numpy(
            coords_lon[i],
            coords_lat[i],
            times[i],
            coords_obs[i],
            "gibbs",
            True,
            MU,
            float(c.C),
        )


def _build_cartesian_bundle_with_covariance(n: int) -> CartesianCoordinates:
    pos = RNG.normal(0.0, 2.0, size=(n, 3))
    vel = RNG.normal(0.0, 0.02, size=(n, 3))
    cov = np.zeros((n, 6, 6), dtype=np.float64)
    for i in range(n):
        a = RNG.normal(size=(6, 6)) * 1e-6
        cov[i] = a @ a.T + np.eye(6) * 1e-9
    return CartesianCoordinates.from_kwargs(
        x=pos[:, 0],
        y=pos[:, 1],
        z=pos[:, 2],
        vx=vel[:, 0],
        vy=vel[:, 1],
        vz=vel[:, 2],
        time=Timestamp.from_mjd(np.full(n, 59000.0), scale="tdb"),
        covariance=CoordinateCovariances.from_matrix(cov),
        origin=Origin.from_kwargs(code=np.full(n, "SUN")),
        frame="ecliptic",
    )


def _build_propagate_inputs(n_orbits: int, n_times: int):
    orbits = np.zeros((n_orbits, 6), dtype=np.float64)
    orbits[:, 0] = 1.0 + 1.5 * RNG.random(n_orbits)
    orbits[:, 1] = 0.2 * RNG.standard_normal(n_orbits)
    orbits[:, 2] = 0.1 * RNG.standard_normal(n_orbits)
    orbits[:, 3] = -0.005 * RNG.standard_normal(n_orbits)
    orbits[:, 4] = 0.017 + 0.003 * RNG.standard_normal(n_orbits)
    orbits[:, 5] = 0.002 * RNG.standard_normal(n_orbits)
    dts = (500.0 + 3000.0 * RNG.random(n_orbits * n_times)).astype(np.float64)
    mus = np.full(n_orbits * n_times, MU, dtype=np.float64)
    orbits_flat = np.ascontiguousarray(np.repeat(orbits, n_times, axis=0))
    return orbits_flat, dts, mus


def _build_propagate_cov_inputs(n_orbits: int, n_times: int):
    orbits_flat, dts, mus = _build_propagate_inputs(n_orbits, n_times)
    n = orbits_flat.shape[0]
    cov = np.zeros((n, 6, 6), dtype=np.float64)
    for i in range(n):
        a = RNG.standard_normal((6, 6))
        cov[i] = 1e-10 * (a @ a.T) + 1e-8 * np.eye(6)
    return orbits_flat, cov.reshape(n, 36).copy(), dts, mus


def _build_photometry_inputs(n: int):
    theta_obj = 2.0 * np.pi * RNG.random(n)
    r_obj = 0.7 + 3.3 * RNG.random(n)
    object_pos = np.zeros((n, 3), dtype=np.float64)
    object_pos[:, 0] = r_obj * np.cos(theta_obj)
    object_pos[:, 1] = r_obj * np.sin(theta_obj)
    object_pos[:, 2] = 0.1 * RNG.standard_normal(n)
    theta_obs = 2.0 * np.pi * RNG.random(n)
    observer_pos = np.zeros((n, 3), dtype=np.float64)
    observer_pos[:, 0] = np.cos(theta_obs)
    observer_pos[:, 1] = np.sin(theta_obs)
    observer_pos[:, 2] = 1e-4 * RNG.standard_normal(n)
    h_v = 15.0 + 5.0 * RNG.random(n)
    g = np.full(n, 0.15, dtype=np.float64)
    return (
        np.ascontiguousarray(object_pos),
        np.ascontiguousarray(observer_pos),
        np.ascontiguousarray(h_v),
        np.ascontiguousarray(g),
    )


def _build_ephemeris_inputs(n_orbits: int, n_times: int):
    n = n_orbits * n_times
    orbits = np.zeros((n_orbits, 6), dtype=np.float64)
    orbits[:, 0] = 1.0 + 1.5 * RNG.random(n_orbits)
    orbits[:, 1] = 0.2 * RNG.standard_normal(n_orbits)
    orbits[:, 2] = 0.1 * RNG.standard_normal(n_orbits)
    orbits[:, 3] = -0.005 * RNG.standard_normal(n_orbits)
    orbits[:, 4] = 0.017 + 0.003 * RNG.standard_normal(n_orbits)
    orbits[:, 5] = 0.002 * RNG.standard_normal(n_orbits)
    orbits_flat = np.ascontiguousarray(np.repeat(orbits, n_times, axis=0))
    times = (59000.0 + 3650.0 * RNG.random(n)).astype(np.float64)
    theta = 2.0 * np.pi * RNG.random(n)
    observers = np.zeros((n, 6), dtype=np.float64)
    observers[:, 0] = np.cos(theta)
    observers[:, 1] = np.sin(theta)
    observers[:, 2] = 1e-4 * RNG.standard_normal(n)
    observers[:, 3] = -0.017 * np.sin(theta)
    observers[:, 4] = 0.017 * np.cos(theta)
    observers[:, 5] = 1e-6 * RNG.standard_normal(n)
    observers = np.ascontiguousarray(observers)
    mus = np.full(n, MU, dtype=np.float64)
    return orbits_flat, times, observers, mus


def _build_lambert_inputs(n: int):
    r1 = np.ascontiguousarray(RNG.normal(1.0, 0.3, size=(n, 3)), dtype=np.float64)
    r1 /= np.linalg.norm(r1, axis=1, keepdims=True)
    r1 *= RNG.uniform(0.7, 1.3, size=(n, 1))
    r2 = np.ascontiguousarray(RNG.normal(1.2, 0.3, size=(n, 3)), dtype=np.float64)
    r2 /= np.linalg.norm(r2, axis=1, keepdims=True)
    r2 *= RNG.uniform(1.0, 1.8, size=(n, 1))
    tof = np.ascontiguousarray(RNG.uniform(100.0, 500.0, size=n), dtype=np.float64)
    return r1, r2, tof


def _build_ephemeris_cov_inputs(n_orbits: int, n_times: int):
    orbits_flat, times, observers, mus = _build_ephemeris_inputs(n_orbits, n_times)
    n = orbits_flat.shape[0]
    cov = np.zeros((n, 6, 6), dtype=np.float64)
    for i in range(n):
        a = RNG.standard_normal((6, 6))
        cov[i] = 1e-10 * (a @ a.T) + 1e-8 * np.eye(6)
    return orbits_flat, cov.reshape(n, 36).copy(), times, observers, mus


# ---------- measurement orchestration ---------------------------------------


def _run_measurements(repeats: int) -> dict[str, dict[str, Any]]:
    coords = _build_cartesian(120_000)
    geodetic_coords = _build_geodetic_cartesian(120_000, GEODETIC_A, GEODETIC_F)
    keplerian_t0 = np.ascontiguousarray(
        RNG.uniform(59000.0, 60000.0, size=120_000), dtype=np.float64
    )
    keplerian_mu = np.ascontiguousarray(
        RNG.uniform(1e-6, 1e-2, size=120_000), dtype=np.float64
    )
    keplerian_a_coords, keplerian_a_mu = _build_keplerian_a_inputs(120_000)
    cometary_coords, cometary_t0, cometary_mu = _build_cometary_inputs(120_000)
    spherical_coords = _build_spherical(120_000)
    a, mu = _build_mean_motion_inputs(50_000)
    r1, r2, r3 = _build_gibbs_triplets(5_000)
    t1, t2, t3 = _build_herrick_times(5_000)
    transform_bundle = _build_cartesian_bundle_with_covariance(10_000)
    propagate_orbits, propagate_dts, propagate_mus = _build_propagate_inputs(1000, 20)
    cov_orbits, cov_cov_in, cov_dts, cov_mus = _build_propagate_cov_inputs(200, 10)
    eph_orbits, eph_times, eph_observers, eph_mus = _build_ephemeris_inputs(5000, 20)
    (
        eph_cov_orbits,
        eph_cov_cov_in,
        eph_cov_times,
        eph_cov_observers,
        eph_cov_mus,
    ) = _build_ephemeris_cov_inputs(2000, 20)
    lam_r1, lam_r2, lam_tof = _build_lambert_inputs(10_000)
    giod_lon, giod_lat, giod_times, giod_obs = _build_gaussiod_inputs(256)
    photo_obj, photo_obs, photo_h, photo_g = _build_photometry_inputs(100_000)
    # Bandpass predict_magnitudes: 5-entry delta table (V + 4 target filters).
    predict_delta_table = np.array([0.0, -0.2, 0.1, 0.3, -0.1], dtype=np.float64)
    predict_target_ids = RNG.integers(
        0, len(predict_delta_table), size=photo_h.shape[0]
    ).astype(np.int32)

    # Warm-up each Rust kernel on real-sized inputs to cover rayon pool spin-up.
    _ = rust_api.cartesian_to_spherical_numpy(coords[:1024])
    _ = rust_api.cartesian_to_geodetic_numpy(
        geodetic_coords[:1024], GEODETIC_A, GEODETIC_F, 100, 1e-15
    )
    _ = rust_api.cartesian_to_keplerian_numpy(
        coords[:1024], keplerian_t0[:1024], keplerian_mu[:1024]
    )
    _ = rust_api.keplerian_to_cartesian_numpy(
        keplerian_a_coords[:1024], keplerian_a_mu[:1024], 100, 1e-15
    )
    _ = rust_api.spherical_to_cartesian_numpy(spherical_coords[:1024])
    _ = rust_api.calc_mean_motion_numpy(a[:1024], mu[:1024])
    _ = rust_api.cartesian_to_cometary_numpy(
        coords[:1024], keplerian_t0[:1024], keplerian_mu[:1024]
    )
    _ = rust_api.cometary_to_cartesian_numpy(
        cometary_coords[:1024], cometary_t0[:1024], cometary_mu[:1024], 100, 1e-15
    )
    _ = _gibbs_rust_batch(r1[:128], r2[:128], r3[:128])
    _ = _herrick_rust_batch(r1[:128], r2[:128], r3[:128], t1[:128], t2[:128], t3[:128])
    _ = _gauss_rust_batch(r1[:128], r2[:128], r3[:128], t1[:128], t2[:128], t3[:128])
    _ = transform_coordinates(
        transform_bundle,
        representation_out=KeplerianCoordinates,
        origin_out=OriginCodes.SUN,
    )
    _ = rust_api.propagate_2body_numpy(
        propagate_orbits[:200], propagate_dts[:200], propagate_mus[:200], 1000, 1e-14
    )
    _ = rust_api.propagate_2body_with_covariance_numpy(
        cov_orbits[:50], cov_cov_in[:50], cov_dts[:50], cov_mus[:50], 1000, 1e-14
    )
    _ = rust_api.generate_ephemeris_2body_numpy(eph_orbits, eph_observers, eph_mus)
    _ = rust_api.generate_ephemeris_2body_with_covariance_numpy(
        eph_cov_orbits, eph_cov_cov_in, eph_cov_observers, eph_cov_mus
    )
    _ = rust_api.calculate_phase_angle_numpy(photo_obj, photo_obs)
    _ = rust_api.calculate_apparent_magnitude_v_numpy(
        photo_h, photo_obj, photo_obs, photo_g
    )
    _ = rust_api.calculate_apparent_magnitude_v_and_phase_angle_numpy(
        photo_h, photo_obj, photo_obs, photo_g
    )
    _ = rust_api.predict_magnitudes_bandpass_numpy(
        photo_h,
        photo_obj,
        photo_obs,
        photo_g,
        predict_target_ids,
        predict_delta_table,
    )
    _ = rust_api.izzo_lambert_numpy(lam_r1[:128], lam_r2[:128], lam_tof[:128], MU)
    _gauss_iod_rust_batch(giod_lon[:8], giod_lat[:8], giod_times[:8], giod_obs[:8])

    # Timed runs.
    rust_mm, _ = _timed_runs(lambda: rust_api.calc_mean_motion_numpy(a, mu), repeats)
    rust_cart, _ = _timed_runs(
        lambda: rust_api.cartesian_to_spherical_numpy(coords), repeats
    )
    rust_geodetic, _ = _timed_runs(
        lambda: rust_api.cartesian_to_geodetic_numpy(
            geodetic_coords, GEODETIC_A, GEODETIC_F, 100, 1e-15
        ),
        repeats,
    )
    rust_keplerian, _ = _timed_runs(
        lambda: rust_api.cartesian_to_keplerian_numpy(
            coords, keplerian_t0, keplerian_mu
        ),
        repeats,
    )
    rust_k2c, _ = _timed_runs(
        lambda: rust_api.keplerian_to_cartesian_numpy(
            keplerian_a_coords, keplerian_a_mu, 100, 1e-15
        ),
        repeats,
    )
    rust_sph2cart, _ = _timed_runs(
        lambda: rust_api.spherical_to_cartesian_numpy(spherical_coords), repeats
    )
    rust_gibbs, _ = _timed_runs(lambda: _gibbs_rust_batch(r1, r2, r3), repeats)
    rust_herrick, _ = _timed_runs(
        lambda: _herrick_rust_batch(r1, r2, r3, t1, t2, t3), repeats
    )
    rust_gauss, _ = _timed_runs(
        lambda: _gauss_rust_batch(r1, r2, r3, t1, t2, t3), repeats
    )
    rust_c2com, _ = _timed_runs(
        lambda: rust_api.cartesian_to_cometary_numpy(
            coords, keplerian_t0, keplerian_mu
        ),
        repeats,
    )
    rust_com2c, _ = _timed_runs(
        lambda: rust_api.cometary_to_cartesian_numpy(
            cometary_coords, cometary_t0, cometary_mu, 100, 1e-15
        ),
        repeats,
    )
    rust_transform, _ = _timed_runs(
        lambda: transform_coordinates(
            transform_bundle,
            representation_out=KeplerianCoordinates,
            origin_out=OriginCodes.SUN,
        ),
        repeats,
    )
    rust_prop, _ = _timed_runs(
        lambda: rust_api.propagate_2body_numpy(
            propagate_orbits, propagate_dts, propagate_mus, 1000, 1e-14
        ),
        repeats,
    )
    rust_prop_cov, _ = _timed_runs(
        lambda: rust_api.propagate_2body_with_covariance_numpy(
            cov_orbits, cov_cov_in, cov_dts, cov_mus, 1000, 1e-14
        ),
        repeats,
    )
    rust_eph, _ = _timed_runs(
        lambda: rust_api.generate_ephemeris_2body_numpy(
            eph_orbits, eph_observers, eph_mus
        ),
        repeats,
    )
    rust_eph_cov, _ = _timed_runs(
        lambda: rust_api.generate_ephemeris_2body_with_covariance_numpy(
            eph_cov_orbits, eph_cov_cov_in, eph_cov_observers, eph_cov_mus
        ),
        repeats,
    )
    rust_phase, _ = _timed_runs(
        lambda: rust_api.calculate_phase_angle_numpy(photo_obj, photo_obs), repeats
    )
    rust_mag, _ = _timed_runs(
        lambda: rust_api.calculate_apparent_magnitude_v_numpy(
            photo_h, photo_obj, photo_obs, photo_g
        ),
        repeats,
    )
    rust_mag_alpha, _ = _timed_runs(
        lambda: rust_api.calculate_apparent_magnitude_v_and_phase_angle_numpy(
            photo_h, photo_obj, photo_obs, photo_g
        ),
        repeats,
    )
    rust_lambert, _ = _timed_runs(
        lambda: rust_api.izzo_lambert_numpy(lam_r1, lam_r2, lam_tof, MU), repeats
    )
    rust_gauss_iod, _ = _timed_runs(
        lambda: _gauss_iod_rust_batch(giod_lon, giod_lat, giod_times, giod_obs),
        repeats,
    )
    rust_predict_mag, _ = _timed_runs(
        lambda: rust_api.predict_magnitudes_bandpass_numpy(
            photo_h,
            photo_obj,
            photo_obs,
            photo_g,
            predict_target_ids,
            predict_delta_table,
        ),
        repeats,
    )

    return {
        "cartesian_to_spherical": _latency_summary(rust_cart),
        "cartesian_to_geodetic": _latency_summary(rust_geodetic),
        "cartesian_to_keplerian": _latency_summary(rust_keplerian),
        "keplerian_to_cartesian": _latency_summary(rust_k2c),
        "spherical_to_cartesian": _latency_summary(rust_sph2cart),
        "cartesian_to_cometary": _latency_summary(rust_c2com),
        "cometary_to_cartesian": _latency_summary(rust_com2c),
        "calc_mean_motion": _latency_summary(rust_mm),
        "propagate_2body": _latency_summary(rust_prop),
        "propagate_2body_with_covariance": _latency_summary(rust_prop_cov),
        "generate_ephemeris_2body": _latency_summary(rust_eph),
        "generate_ephemeris_2body_with_covariance": _latency_summary(rust_eph_cov),
        "calculate_phase_angle": _latency_summary(rust_phase),
        "calculate_apparent_magnitude_v": _latency_summary(rust_mag),
        "calculate_apparent_magnitude_v_and_phase_angle": _latency_summary(
            rust_mag_alpha
        ),
        "calc_gibbs": _latency_summary(rust_gibbs),
        "calc_herrick_gibbs": _latency_summary(rust_herrick),
        "calc_gauss": _latency_summary(rust_gauss),
        "transform_coordinates": _latency_summary(rust_transform),
        "solve_lambert": _latency_summary(rust_lambert),
        "predict_magnitudes": _latency_summary(rust_predict_mag),
        "gauss_iod": _latency_summary(rust_gauss_iod),
    }


# ---------- gate ------------------------------------------------------------


BENCHMARK_TO_API_ID = {
    "cartesian_to_spherical": "coordinates.cartesian_to_spherical",
    "cartesian_to_geodetic": "coordinates.cartesian_to_geodetic",
    "cartesian_to_keplerian": "coordinates.cartesian_to_keplerian",
    "keplerian_to_cartesian": "coordinates.keplerian.to_cartesian",
    "cartesian_to_cometary": "coordinates.cartesian_to_cometary",
    "cometary_to_cartesian": "coordinates.cometary.to_cartesian",
    "spherical_to_cartesian": "coordinates.spherical.to_cartesian",
    "calc_mean_motion": "dynamics.calc_mean_motion",
    "propagate_2body": "dynamics.propagate_2body",
    "propagate_2body_with_covariance": "dynamics.propagate_2body_with_covariance",
    "generate_ephemeris_2body": "dynamics.generate_ephemeris_2body",
    "generate_ephemeris_2body_with_covariance": (
        "dynamics.generate_ephemeris_2body_with_covariance"
    ),
    "calculate_phase_angle": "photometry.calculate_phase_angle",
    "calculate_apparent_magnitude_v": "photometry.calculate_apparent_magnitude_v",
    "calculate_apparent_magnitude_v_and_phase_angle": (
        "photometry.calculate_apparent_magnitude_v_and_phase_angle"
    ),
    "calc_gibbs": "orbit_determination.calcGibbs",
    "calc_herrick_gibbs": "orbit_determination.calcHerrickGibbs",
    "calc_gauss": "orbit_determination.calcGauss",
    "transform_coordinates": "coordinates.transform_coordinates",
    "solve_lambert": "dynamics.solve_lambert",
    "predict_magnitudes": "photometry.predict_magnitudes",
    "gauss_iod": "orbit_determination.gaussIOD",
}

# APIs covered by other benchmark scripts.
EXTERNALLY_BENCHMARKED: set[str] = set()


def _check_coverage() -> None:
    default_rust_ids = {m.api_id for m in API_MIGRATIONS if m.default == "rust"}
    mapped = set(BENCHMARK_TO_API_ID.values())
    missing = sorted(default_rust_ids - mapped - EXTERNALLY_BENCHMARKED)
    if missing:
        raise SystemExit(
            "Missing benchmark coverage for rust-default APIs: " + ", ".join(missing)
        )


def _compare_to_baseline(
    current: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    p50_max_ratio: float,
    p95_max_ratio: float,
) -> list[str]:
    failures: list[str] = []
    for name in current:
        if name not in baseline:
            # New API since baseline was captured; not a failure, just skipped.
            print(f"  [skip] {name}: no baseline entry (new API)")
            continue
        cur = current[name]
        base = baseline[name]
        ratio_p50 = cur["rust_seconds_p50"] / base["rust_seconds_p50"]
        ratio_p95 = cur["rust_seconds_p95"] / base["rust_seconds_p95"]
        marker = (
            "  " if ratio_p50 <= p50_max_ratio and ratio_p95 <= p95_max_ratio else "!!"
        )
        print(
            f"{marker}{name:<48}"
            f"  p50 {cur['rust_seconds_p50']*1e3:8.3f} ms "
            f"(ratio {ratio_p50:5.2f})"
            f"  p95 {cur['rust_seconds_p95']*1e3:8.3f} ms "
            f"(ratio {ratio_p95:5.2f})"
        )
        if ratio_p50 > p50_max_ratio:
            failures.append(
                f"{name} p50 regression: {ratio_p50:.2f}x baseline (>{p50_max_ratio:.2f})"
            )
        if ratio_p95 > p95_max_ratio:
            failures.append(
                f"{name} p95 regression: {ratio_p95:.2f}x baseline (>{p95_max_ratio:.2f})"
            )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--capture-baseline",
        action="store_true",
        help="Write migration/artifacts/rust_latency_baseline.json with current "
        "Rust timings. Use when intentionally re-basing (e.g., after kernel "
        "port or on a new reference machine). Commit the produced file.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=BASELINE_PATH,
        help="Path to the pinned baseline JSON (defaults to the committed file).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CURRENT_PATH,
        help="Where to write this run's latency measurements.",
    )
    parser.add_argument("--p50-max-ratio", type=float, default=DEFAULT_P50_MAX_RATIO)
    parser.add_argument("--p95-max-ratio", type=float, default=DEFAULT_P95_MAX_RATIO)
    args = parser.parse_args()

    _check_coverage()

    print(f"Measuring Rust-only latency ({args.repeats} repeats)...")
    current = _run_measurements(args.repeats)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(current, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")

    if args.capture_baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(current, indent=2), encoding="utf-8")
        print(f"Wrote baseline to {args.baseline}")
        return

    if not args.baseline.exists():
        raise SystemExit(
            f"Baseline file {args.baseline} not found. Run with "
            f"--capture-baseline first to create it."
        )

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    print(
        f"\nComparing against baseline {args.baseline} "
        f"(p50 tol={args.p50_max_ratio:.2f}x, p95 tol={args.p95_max_ratio:.2f}x):"
    )
    failures = _compare_to_baseline(
        current, baseline, args.p50_max_ratio, args.p95_max_ratio
    )
    if failures:
        raise SystemExit("\nGate failures:\n  - " + "\n  - ".join(failures))
    print("\nAll APIs within regression tolerance.")


if __name__ == "__main__":
    main()
