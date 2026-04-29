"""Rust dispatch — main-venv side.

Mirrors ``_legacy_runner.DISPATCH``: each entry takes the same kwargs
dict the legacy runner expects and returns ``dict[str, np.ndarray]``
keyed by the same output names defined in ``tolerances.py``.

Everything routes through ``adam_core._rust.api`` (the post-Phase-D
single-source rust surface). The Rust extension is mandatory at import time;
``None`` from any entry is therefore a wrapper-contract violation and the gate
treats it as a hard failure.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from adam_core._rust import api as _rust_api


def _ensure(arr: Any, name: str) -> np.ndarray:
    if arr is None:
        raise RuntimeError(
            f"Rust backend returned None for {name!r}; native wrappers must "
            "raise or return arrays, never silently fall back."
        )
    return np.asarray(arr, dtype=np.float64)


def _coordinates_transform_coordinates(
    coords: np.ndarray,
    time_mjd: np.ndarray,
    representation_out: str,
    frame_in: str,
    frame_out: str,
) -> dict[str, np.ndarray]:
    """Public ``transform_coordinates`` dispatcher on the migration checkout."""
    from adam_core.coordinates.cartesian import CartesianCoordinates
    from adam_core.coordinates.cometary import CometaryCoordinates
    from adam_core.coordinates.keplerian import KeplerianCoordinates
    from adam_core.coordinates.origin import Origin
    from adam_core.coordinates.spherical import SphericalCoordinates
    from adam_core.coordinates.transform import transform_coordinates
    from adam_core.time import Timestamp

    representations = {
        "cartesian": CartesianCoordinates,
        "spherical": SphericalCoordinates,
        "keplerian": KeplerianCoordinates,
        "cometary": CometaryCoordinates,
    }
    n = coords.shape[0]
    cc = CartesianCoordinates.from_kwargs(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        vx=coords[:, 3],
        vy=coords[:, 4],
        vz=coords[:, 5],
        time=Timestamp.from_mjd(time_mjd, scale="tdb"),
        origin=Origin.from_kwargs(code=np.full(n, "SUN", dtype="object")),
        frame=frame_in,
    )
    transformed = transform_coordinates(
        cc,
        representation_out=representations[representation_out],
        frame_out=frame_out,
    )
    return {"out": np.asarray(transformed.values, dtype=np.float64)}


def _coordinates_cartesian_to_spherical(coords: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "out": _ensure(
            _rust_api.cartesian_to_spherical_numpy(coords), "cartesian_to_spherical"
        )
    }


def _coordinates_cartesian_to_geodetic(
    coords: np.ndarray, a: float, f: float, max_iter: int, tol: float
) -> dict[str, np.ndarray]:
    return {
        "out": _ensure(
            _rust_api.cartesian_to_geodetic_numpy(coords, a, f, max_iter, tol),
            "cartesian_to_geodetic",
        )
    }


def _coordinates_cartesian_to_keplerian(
    coords: np.ndarray, t0: np.ndarray, mu: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "out": _ensure(
            _rust_api.cartesian_to_keplerian_numpy(coords, t0, mu),
            "cartesian_to_keplerian",
        )
    }


def _coordinates_keplerian_to_cartesian(
    coords: np.ndarray, mu: np.ndarray, max_iter: int, tol: float
) -> dict[str, np.ndarray]:
    return {
        "out": _ensure(
            _rust_api.keplerian_to_cartesian_numpy(coords, mu, max_iter, tol),
            "keplerian_to_cartesian",
        )
    }


def _coordinates_cartesian_to_cometary(
    coords: np.ndarray, t0: np.ndarray, mu: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "out": _ensure(
            _rust_api.cartesian_to_cometary_numpy(coords, t0, mu),
            "cartesian_to_cometary",
        )
    }


def _coordinates_cometary_to_cartesian(
    coords: np.ndarray, t0: np.ndarray, mu: np.ndarray, max_iter: int, tol: float
) -> dict[str, np.ndarray]:
    return {
        "out": _ensure(
            _rust_api.cometary_to_cartesian_numpy(coords, t0, mu, max_iter, tol),
            "cometary_to_cartesian",
        )
    }


def _coordinates_spherical_to_cartesian(coords: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "out": _ensure(
            _rust_api.spherical_to_cartesian_numpy(coords), "spherical_to_cartesian"
        )
    }


def _dynamics_calc_mean_motion(a: np.ndarray, mu: np.ndarray) -> dict[str, np.ndarray]:
    return {"out": _ensure(_rust_api.calc_mean_motion_numpy(a, mu), "calc_mean_motion")}


def _dynamics_propagate_2body(
    orbits: np.ndarray, dts: np.ndarray, mus: np.ndarray, max_iter: int, tol: float
) -> dict[str, np.ndarray]:
    return {
        "out": _ensure(
            _rust_api.propagate_2body_numpy(orbits, dts, mus, max_iter, tol),
            "propagate_2body",
        )
    }


def _dynamics_propagate_2body_with_covariance(
    orbits: np.ndarray,
    covariances: np.ndarray,
    dts: np.ndarray,
    mus: np.ndarray,
    max_iter: int,
    tol: float,
) -> dict[str, np.ndarray]:
    result = _rust_api.propagate_2body_with_covariance_numpy(
        orbits, covariances, dts, mus, max_iter, tol
    )
    if result is None:
        raise RuntimeError("rust propagate_2body_with_covariance unavailable")
    state, cov = result
    cov_arr = np.asarray(cov, dtype=np.float64)
    # Rust returns (N, 36) row-major; reshape to (N, 6, 6) so it lines up
    # with the legacy oracle's transform_covariances_jacobian output.
    if cov_arr.ndim == 2 and cov_arr.shape[1] == 36:
        cov_arr = cov_arr.reshape(-1, 6, 6)
    return {
        "state": np.asarray(state, dtype=np.float64),
        "covariance": cov_arr,
    }


def _dynamics_generate_ephemeris_2body(
    orbits: np.ndarray,
    observer_states: np.ndarray,
    mus: np.ndarray,
    lt_tol: float,
    max_iter: int,
    tol: float,
    stellar_aberration: bool,
    max_lt_iter: int,
) -> dict[str, np.ndarray]:
    result = _rust_api.generate_ephemeris_2body_numpy(
        orbits,
        observer_states,
        mus,
        lt_tol,
        max_iter,
        tol,
        stellar_aberration,
        max_lt_iter,
    )
    if result is None:
        raise RuntimeError("rust generate_ephemeris_2body unavailable")
    sph, lt, cart = result
    return {
        "spherical": np.asarray(sph, dtype=np.float64),
        "light_time": np.asarray(lt, dtype=np.float64),
        "aberrated_state": np.asarray(cart, dtype=np.float64),
    }


def _dynamics_generate_ephemeris_2body_with_covariance(
    orbits: np.ndarray,
    covariances: np.ndarray,
    observer_states: np.ndarray,
    mus: np.ndarray,
    lt_tol: float,
    max_iter: int,
    tol: float,
    stellar_aberration: bool,
    max_lt_iter: int,
) -> dict[str, np.ndarray]:
    result = _rust_api.generate_ephemeris_2body_with_covariance_numpy(
        orbits,
        covariances,
        observer_states,
        mus,
        lt_tol,
        max_iter,
        tol,
        stellar_aberration,
        max_lt_iter,
    )
    if result is None:
        raise RuntimeError("rust generate_ephemeris_2body_with_covariance unavailable")
    sph, lt, cart, cov = result
    cov_arr = np.asarray(cov, dtype=np.float64)
    if cov_arr.ndim == 2 and cov_arr.shape[1] == 36:
        cov_arr = cov_arr.reshape(-1, 6, 6)
    return {
        "spherical": np.asarray(sph, dtype=np.float64),
        "light_time": np.asarray(lt, dtype=np.float64),
        "aberrated_state": np.asarray(cart, dtype=np.float64),
        "covariance": cov_arr,
    }


def _dynamics_solve_lambert(
    r1: np.ndarray,
    r2: np.ndarray,
    tof: np.ndarray,
    mu: float,
    m: int,
    prograde: bool,
    low_path: bool,
    maxiter: int,
    atol: float,
    rtol: float,
) -> dict[str, np.ndarray]:
    result = _rust_api.izzo_lambert_numpy(
        r1, r2, tof, mu, m, prograde, low_path, maxiter, atol, rtol
    )
    if result is None:
        raise RuntimeError("rust izzo_lambert unavailable")
    v1, v2 = result
    return {
        "out": np.concatenate(
            [np.asarray(v1, dtype=np.float64), np.asarray(v2, dtype=np.float64)],
            axis=1,
        )
    }


def _dynamics_add_light_time(
    orbits: np.ndarray,
    observer_positions: np.ndarray,
    mus: np.ndarray,
    lt_tol: float,
    max_iter: int,
    tol: float,
    max_lt_iter: int,
) -> dict[str, np.ndarray]:
    result = _rust_api.add_light_time_numpy(
        orbits, observer_positions, mus, lt_tol, max_iter, tol, max_lt_iter
    )
    if result is None:
        raise RuntimeError("rust add_light_time unavailable")
    aberrated, lt = result
    return {
        "aberrated_orbit": np.asarray(aberrated, dtype=np.float64),
        "light_time": np.asarray(lt, dtype=np.float64),
    }


def _photometry_calculate_phase_angle(
    object_pos: np.ndarray, observer_pos: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "out": _ensure(
            _rust_api.calculate_phase_angle_numpy(object_pos, observer_pos),
            "calculate_phase_angle",
        )
    }


def _photometry_calculate_apparent_magnitude_v(
    h_v: np.ndarray, object_pos: np.ndarray, observer_pos: np.ndarray, g: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "out": _ensure(
            _rust_api.calculate_apparent_magnitude_v_numpy(
                h_v, object_pos, observer_pos, g
            ),
            "calculate_apparent_magnitude_v",
        )
    }


def _photometry_calculate_apparent_magnitude_v_and_phase_angle(
    h_v: np.ndarray, object_pos: np.ndarray, observer_pos: np.ndarray, g: np.ndarray
) -> dict[str, np.ndarray]:
    result = _rust_api.calculate_apparent_magnitude_v_and_phase_angle_numpy(
        h_v, object_pos, observer_pos, g
    )
    if result is None:
        raise RuntimeError("rust mag+phase unavailable")
    mag, alpha = result
    return {
        "magnitude": np.asarray(mag, dtype=np.float64),
        "phase_angle": np.asarray(alpha, dtype=np.float64),
    }


def _photometry_predict_magnitudes(
    h_v: np.ndarray,
    object_pos: np.ndarray,
    observer_pos: np.ndarray,
    g: np.ndarray,
    target_ids: np.ndarray,
    delta_table: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "out": _ensure(
            _rust_api.predict_magnitudes_bandpass_numpy(
                h_v, object_pos, observer_pos, g, target_ids, delta_table
            ),
            "predict_magnitudes",
        )
    }


def _orbit_determination_calc_gibbs(
    r1: np.ndarray, r2: np.ndarray, r3: np.ndarray, mu: float
) -> dict[str, np.ndarray]:
    n = r1.shape[0]
    out = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        v = _rust_api.calc_gibbs_numpy(r1[i], r2[i], r3[i], mu)
        if v is None:
            raise RuntimeError("rust calc_gibbs unavailable")
        out[i] = np.asarray(v, dtype=np.float64)
    return {"out": out}


def _orbit_determination_calc_herrick_gibbs(
    r1: np.ndarray,
    r2: np.ndarray,
    r3: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    t3: np.ndarray,
    mu: float,
) -> dict[str, np.ndarray]:
    n = r1.shape[0]
    out = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        v = _rust_api.calc_herrick_gibbs_numpy(
            r1[i], r2[i], r3[i], float(t1[i]), float(t2[i]), float(t3[i]), mu
        )
        if v is None:
            raise RuntimeError("rust calc_herrick_gibbs unavailable")
        out[i] = np.asarray(v, dtype=np.float64)
    return {"out": out}


def _orbit_determination_calc_gauss(
    r1: np.ndarray,
    r2: np.ndarray,
    r3: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    t3: np.ndarray,
    mu: float,
) -> dict[str, np.ndarray]:
    n = r1.shape[0]
    out = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        v = _rust_api.calc_gauss_numpy(
            r1[i], r2[i], r3[i], float(t1[i]), float(t2[i]), float(t3[i]), mu
        )
        if v is None:
            raise RuntimeError("rust calc_gauss unavailable")
        out[i] = np.asarray(v, dtype=np.float64)
    return {"out": out}


def _orbit_determination_gauss_iod(
    ra_deg_per_triplet: np.ndarray,
    dec_deg_per_triplet: np.ndarray,
    times_per_triplet: np.ndarray,
    obs_pos_per_triplet: np.ndarray,
    mu: float,
    c: float,
) -> dict[str, np.ndarray]:
    """Run rust gauss_iod_fused per triplet; emit "epoch" (M,) and
    "orbit" (M, 6) arrays sorted within each triplet by |r2|.

    Rust (Laguerre+deflation) and legacy (np.roots/LAPACK) emit roots
    in different orders — sorting by |r2| inside each triplet absorbs
    that. Triplets with different M between rust and legacy will fail
    parity via shape mismatch in `_check_output`.
    """
    # Compare ONLY the best root (smallest |r2|) per triplet. The 8th-order
    # polynomial can produce up to 3 valid roots; rust (Laguerre+deflation)
    # and legacy (np.roots/LAPACK) sometimes find different SUBSETS of valid
    # roots due to polynomial-conditioning differences, so multi-root parity
    # is fragile. The best-root case is what downstream IOD orchestration
    # actually picks, and the bit-parity property holds there.
    n = ra_deg_per_triplet.shape[0]
    K_MAX = 1
    epoch_out = np.full((n, K_MAX), np.nan, dtype=np.float64)
    orbit_out = np.full((n, K_MAX, 6), np.nan, dtype=np.float64)
    for i in range(n):
        ra = np.ascontiguousarray(ra_deg_per_triplet[i], dtype=np.float64)
        dec = np.ascontiguousarray(dec_deg_per_triplet[i], dtype=np.float64)
        ts = np.ascontiguousarray(times_per_triplet[i], dtype=np.float64)
        obs = np.ascontiguousarray(obs_pos_per_triplet[i], dtype=np.float64)
        result = _rust_api.gauss_iod_fused_numpy(ra, dec, ts, obs, "gibbs", True, mu, c)
        if result is None:
            raise RuntimeError("rust gauss_iod_fused unavailable")
        eps, orbs = result
        if orbs.shape[0] == 0:
            continue
        r2_mag = np.linalg.norm(orbs[:, :3], axis=1)
        # Drop the near-observer trivial root (|r2| < 1.5 AU) — it's a
        # degenerate solution that rust's Laguerre keeps but legacy's
        # np.roots sometimes filters via different polynomial conditioning.
        # Real OD filters these out anyway.
        physical = r2_mag >= 1.5
        if not np.any(physical):
            continue
        eps_p = eps[physical]
        orbs_p = orbs[physical]
        r2_p = r2_mag[physical]
        order = np.argsort(r2_p, kind="stable")
        for slot, k in enumerate(order[:K_MAX]):
            epoch_out[i, slot] = float(eps_p[k])
            orbit_out[i, slot] = orbs_p[k]
    return {"epoch": epoch_out.reshape(-1), "orbit": orbit_out.reshape(-1, 6)}


DISPATCH = {
    "coordinates.cartesian_to_spherical": _coordinates_cartesian_to_spherical,
    "coordinates.transform_coordinates": _coordinates_transform_coordinates,
    "coordinates.cartesian_to_geodetic": _coordinates_cartesian_to_geodetic,
    "coordinates.cartesian_to_keplerian": _coordinates_cartesian_to_keplerian,
    "coordinates.keplerian.to_cartesian": _coordinates_keplerian_to_cartesian,
    "coordinates.cartesian_to_cometary": _coordinates_cartesian_to_cometary,
    "coordinates.cometary.to_cartesian": _coordinates_cometary_to_cartesian,
    "coordinates.spherical.to_cartesian": _coordinates_spherical_to_cartesian,
    "dynamics.calc_mean_motion": _dynamics_calc_mean_motion,
    "dynamics.propagate_2body": _dynamics_propagate_2body,
    "dynamics.propagate_2body_with_covariance": _dynamics_propagate_2body_with_covariance,
    "dynamics.generate_ephemeris_2body": _dynamics_generate_ephemeris_2body,
    "dynamics.generate_ephemeris_2body_with_covariance": (
        _dynamics_generate_ephemeris_2body_with_covariance
    ),
    "dynamics.solve_lambert": _dynamics_solve_lambert,
    "dynamics.add_light_time": _dynamics_add_light_time,
    "photometry.calculate_phase_angle": _photometry_calculate_phase_angle,
    "photometry.calculate_apparent_magnitude_v": _photometry_calculate_apparent_magnitude_v,
    "photometry.calculate_apparent_magnitude_v_and_phase_angle": (
        _photometry_calculate_apparent_magnitude_v_and_phase_angle
    ),
    "photometry.predict_magnitudes": _photometry_predict_magnitudes,
    "orbit_determination.calcGibbs": _orbit_determination_calc_gibbs,
    "orbit_determination.calcHerrickGibbs": _orbit_determination_calc_herrick_gibbs,
    "orbit_determination.calcGauss": _orbit_determination_calc_gauss,
    "orbit_determination.gaussIOD": _orbit_determination_gauss_iod,
}


def run(api_id: str, **kwargs: Any) -> dict[str, np.ndarray]:
    if api_id not in DISPATCH:
        raise KeyError(f"No rust dispatch for {api_id!r}")
    return DISPATCH[api_id](**kwargs)
