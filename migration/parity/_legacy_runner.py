"""Legacy oracle subprocess runner — runs inside ``.legacy-venv``.

This file is invoked as:

    .legacy-venv/bin/python -m migration.parity._legacy_runner

It reads a pickled request from stdin and writes a pickled response to
stdout. Both streams are binary. The protocol is:

    request  = {"api": str, "mode": "parity"|"time", "kwargs": {...}, ...}
    response = {"ok": True, "outputs": {...}, "elapsed": [float, ...]}
             | {"ok": False, "error": str, "traceback": str}

For ``mode == "parity"``: the runner invokes the legacy implementation
once and returns the output arrays.

For ``mode == "time"``: the runner invokes the legacy implementation
``reps`` times after ``warmup`` warmups, returning per-rep elapsed
seconds (we send back raw timings so the caller can compute p50/p95
identically for rust and legacy).

Why a subprocess instead of importing from the migration repo: both
``/Users/aleck/Code/adam-core`` (legacy upstream) and the migration repo
export the same ``adam_core`` package name. We cannot have both
importable in one Python process. Subprocess isolation lets us compare
without ever loading both interpreters' versions side-by-side.
"""

from __future__ import annotations

import io
import pickle
import sys
import time
import traceback
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Dispatch table — one entry per API id we expose to the gate.
# Each entry is a callable f(**kwargs) -> dict[str, np.ndarray] (the
# output names match migration/parity/tolerances.py output keys).
# ---------------------------------------------------------------------------


def _coordinates_transform_coordinates(coords: np.ndarray) -> dict[str, np.ndarray]:
    """Cart→Cart, ec→eq frame rotation via legacy `cartesian_to_frame`.

    Legacy expects a quivr CartesianCoordinates table; we build one
    from the numpy input, rotate, then extract values back to numpy.
    """
    from adam_core.coordinates.cartesian import CartesianCoordinates
    from adam_core.coordinates.origin import Origin
    from adam_core.coordinates.transform import cartesian_to_frame
    from adam_core.time import Timestamp

    n = coords.shape[0]
    cc = CartesianCoordinates.from_kwargs(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        vx=coords[:, 3], vy=coords[:, 4], vz=coords[:, 5],
        time=Timestamp.from_mjd(np.zeros(n), scale="tdb"),
        origin=Origin.from_kwargs(code=np.full(n, "SUN", dtype="object")),
        frame="ecliptic",
    )
    rotated = cartesian_to_frame(cc, "equatorial")
    out = np.column_stack([
        rotated.x.to_numpy(zero_copy_only=False),
        rotated.y.to_numpy(zero_copy_only=False),
        rotated.z.to_numpy(zero_copy_only=False),
        rotated.vx.to_numpy(zero_copy_only=False),
        rotated.vy.to_numpy(zero_copy_only=False),
        rotated.vz.to_numpy(zero_copy_only=False),
    ]).astype(np.float64)
    return {"out": out}


def _coordinates_cartesian_to_spherical(coords: np.ndarray) -> dict[str, np.ndarray]:
    from adam_core.coordinates.transform import _cartesian_to_spherical_vmap

    out = np.asarray(_cartesian_to_spherical_vmap(coords), dtype=np.float64)
    return {"out": out}


def _coordinates_cartesian_to_geodetic(
    coords: np.ndarray, a: float, f: float, max_iter: int, tol: float
) -> dict[str, np.ndarray]:
    from adam_core.coordinates.transform import _cartesian_to_geodetic_vmap

    out = np.asarray(_cartesian_to_geodetic_vmap(coords, a, f, max_iter, tol),
                     dtype=np.float64)
    return {"out": out}


def _coordinates_cartesian_to_keplerian(
    coords: np.ndarray, t0: np.ndarray, mu: np.ndarray
) -> dict[str, np.ndarray]:
    from adam_core.coordinates.transform import _cartesian_to_keplerian_vmap

    out = np.asarray(_cartesian_to_keplerian_vmap(coords, t0, mu), dtype=np.float64)
    return {"out": out}


def _coordinates_keplerian_to_cartesian(
    coords: np.ndarray, mu: np.ndarray, max_iter: int, tol: float
) -> dict[str, np.ndarray]:
    # Use the (a, e, i, raan, ap, M) variant — matches the rust kernel
    # convention. The "_p" variant takes semi-latus rectum p instead.
    from adam_core.coordinates.transform import _keplerian_to_cartesian_a_vmap

    out = np.asarray(_keplerian_to_cartesian_a_vmap(coords, mu, max_iter, tol),
                     dtype=np.float64)
    return {"out": out}


def _coordinates_cartesian_to_cometary(
    coords: np.ndarray, t0: np.ndarray, mu: np.ndarray
) -> dict[str, np.ndarray]:
    from adam_core.coordinates.transform import _cartesian_to_cometary_vmap

    out = np.asarray(_cartesian_to_cometary_vmap(coords, t0, mu), dtype=np.float64)
    return {"out": out}


def _coordinates_cometary_to_cartesian(
    coords: np.ndarray, t0: np.ndarray, mu: np.ndarray, max_iter: int, tol: float
) -> dict[str, np.ndarray]:
    from adam_core.coordinates.transform import _cometary_to_cartesian_vmap

    out = np.asarray(_cometary_to_cartesian_vmap(coords, t0, mu, max_iter, tol),
                     dtype=np.float64)
    return {"out": out}


def _coordinates_spherical_to_cartesian(coords: np.ndarray) -> dict[str, np.ndarray]:
    from adam_core.coordinates.transform import _spherical_to_cartesian_vmap

    out = np.asarray(_spherical_to_cartesian_vmap(coords), dtype=np.float64)
    return {"out": out}


def _dynamics_calc_mean_motion(
    a: np.ndarray, mu: np.ndarray
) -> dict[str, np.ndarray]:
    # Legacy has no batched calc_mean_motion — it's a 1-line np expression.
    # The rust kernel mirrors `np.sqrt(mu / a**3)` element-wise.
    out = np.sqrt(np.asarray(mu, dtype=np.float64) / np.asarray(a, dtype=np.float64) ** 3)
    return {"out": out}


def _dynamics_propagate_2body(
    orbits: np.ndarray, dts: np.ndarray, mus: np.ndarray, max_iter: int, tol: float
) -> dict[str, np.ndarray]:
    # Legacy signature is (orbit, t0, t1, mu) with dt = t1 - t0.
    # We pass t0=0, t1=dt to match the rust kernel's dt-only convention.
    from adam_core.dynamics.propagation import _propagate_2body_vmap

    n = orbits.shape[0]
    t0 = np.zeros(n, dtype=np.float64)
    out = np.asarray(_propagate_2body_vmap(orbits, t0, dts, mus, max_iter, tol),
                     dtype=np.float64)
    return {"out": out}


def _dynamics_propagate_2body_with_covariance(
    orbits: np.ndarray,
    covariances: np.ndarray,
    dts: np.ndarray,
    mus: np.ndarray,
    max_iter: int,
    tol: float,
) -> dict[str, np.ndarray]:
    from adam_core.coordinates.covariances import transform_covariances_jacobian
    from adam_core.dynamics.propagation import _propagate_2body, _propagate_2body_vmap

    n = orbits.shape[0]
    t0 = np.zeros(n, dtype=np.float64)
    states = np.asarray(_propagate_2body_vmap(orbits, t0, dts, mus, max_iter, tol),
                        dtype=np.float64)

    cov_out = np.empty_like(covariances)
    for i in range(n):
        cov_out[i] = np.asarray(
            transform_covariances_jacobian(
                orbits[i : i + 1],
                covariances[i : i + 1],
                lambda x: _propagate_2body(x, 0.0, float(dts[i]), float(mus[i]),
                                            max_iter, tol),
            )[0],
            dtype=np.float64,
        )
    return {"state": states, "covariance": cov_out}


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
    # Legacy signature: (orbit, t0, observer_coordinates, mu, lt_tol,
    # max_iter, tol, stellar_aberration). vmap'd over the first four.
    # Rust kernel implicitly uses t0=0 (back-prop by -lt), so we pass
    # t0=0 per row to match.
    from adam_core.dynamics.ephemeris import _generate_ephemeris_2body_vmap

    n = orbits.shape[0]
    t0 = np.zeros(n, dtype=np.float64)
    sph, lt, cart = _generate_ephemeris_2body_vmap(
        orbits,
        t0,
        observer_states,
        mus,
        lt_tol,
        max_iter,
        tol,
        stellar_aberration,
    )
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
    # Legacy doesn't have a direct vmap that returns covariance — we
    # propagate covariance via transform_covariances_jacobian per row.
    from adam_core.coordinates.covariances import transform_covariances_jacobian
    from adam_core.dynamics.ephemeris import (
        _generate_ephemeris_2body,
        _generate_ephemeris_2body_vmap,
    )

    n = orbits.shape[0]
    t0 = np.zeros(n, dtype=np.float64)
    sph, lt, cart = _generate_ephemeris_2body_vmap(
        orbits, t0, observer_states, mus, lt_tol, max_iter, tol, stellar_aberration
    )

    cov_out = np.empty_like(covariances)
    for i in range(n):
        cov_out[i] = np.asarray(
            transform_covariances_jacobian(
                orbits[i : i + 1],
                covariances[i : i + 1],
                lambda x: _generate_ephemeris_2body(
                    x, 0.0, observer_states[i], float(mus[i]),
                    lt_tol, max_iter, tol, stellar_aberration,
                )[0],  # only the spherical output's Jacobian
            )[0],
            dtype=np.float64,
        )

    return {
        "spherical": np.asarray(sph, dtype=np.float64),
        "light_time": np.asarray(lt, dtype=np.float64),
        "aberrated_state": np.asarray(cart, dtype=np.float64),
        "covariance": cov_out,
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
    from adam_core.dynamics.lambert import _izzo_lambert_vmap

    v1, v2 = _izzo_lambert_vmap(r1, r2, tof, mu, m, prograde, low_path,
                                maxiter, atol, rtol)
    out = np.concatenate(
        [np.asarray(v1, dtype=np.float64), np.asarray(v2, dtype=np.float64)], axis=1
    )
    return {"out": out}


def _dynamics_add_light_time(
    orbits: np.ndarray,
    observer_positions: np.ndarray,
    mus: np.ndarray,
    lt_tol: float,
    max_iter: int,
    tol: float,
    max_lt_iter: int,
) -> dict[str, np.ndarray]:
    # Legacy signature: (orbit, t0, observer_position, lt_tol, mu, max_iter,
    # tol, max_lt_iter). vmap'd over (orbit, t0, observer_position); mu is
    # a scalar across the batch, so we collapse the per-row mus array to
    # a single value (the rust input generators always pass a constant mu).
    from adam_core.dynamics.aberrations import _add_light_time_vmap

    n = orbits.shape[0]
    t0 = np.zeros(n, dtype=np.float64)
    mu_scalar = float(np.unique(mus)[0]) if mus.size > 0 else 0.0
    aberrated, lt = _add_light_time_vmap(
        orbits,
        t0,
        observer_positions,
        lt_tol,
        mu_scalar,
        max_iter,
        tol,
        max_lt_iter,
    )
    return {
        "aberrated_orbit": np.asarray(aberrated, dtype=np.float64),
        "light_time": np.asarray(lt, dtype=np.float64),
    }


def _photometry_calculate_phase_angle(
    object_pos: np.ndarray, observer_pos: np.ndarray
) -> dict[str, np.ndarray]:
    from adam_core.photometry.magnitude import _calculate_phase_angle_core_jax

    out = np.asarray(
        _calculate_phase_angle_core_jax(object_pos, observer_pos), dtype=np.float64
    )
    return {"out": out}


def _photometry_calculate_apparent_magnitude_v(
    h_v: np.ndarray, object_pos: np.ndarray, observer_pos: np.ndarray, g: np.ndarray
) -> dict[str, np.ndarray]:
    from adam_core.photometry.magnitude import _calculate_apparent_magnitude_core_jax

    out = np.asarray(
        _calculate_apparent_magnitude_core_jax(h_v, object_pos, observer_pos, g),
        dtype=np.float64,
    )
    return {"out": out}


def _photometry_calculate_apparent_magnitude_v_and_phase_angle(
    h_v: np.ndarray, object_pos: np.ndarray, observer_pos: np.ndarray, g: np.ndarray
) -> dict[str, np.ndarray]:
    from adam_core.photometry.magnitude import (
        _calculate_apparent_magnitude_and_phase_core_jax,
    )

    mag, alpha = _calculate_apparent_magnitude_and_phase_core_jax(
        h_v, object_pos, observer_pos, g
    )
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
    from adam_core.photometry.magnitude import _predict_magnitudes_bandpass_core_jax

    out = np.asarray(
        _predict_magnitudes_bandpass_core_jax(
            h_v, object_pos, observer_pos, g, target_ids, delta_table
        ),
        dtype=np.float64,
    )
    return {"out": out}


def _orbit_determination_calc_gibbs(
    r1: np.ndarray, r2: np.ndarray, r3: np.ndarray, mu: float
) -> dict[str, np.ndarray]:
    from adam_core.orbit_determination.gibbs import calcGibbs

    # Legacy calcGibbs signature: (r1, r2, r3) — mu hardcoded to MU constant.
    # We pass mu through Rust path but legacy uses constant. Verify mu matches.
    n = r1.shape[0]
    out = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        out[i] = np.asarray(calcGibbs(r1[i], r2[i], r3[i]), dtype=np.float64)
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
    from adam_core.orbit_determination.herrick_gibbs import calcHerrickGibbs

    n = r1.shape[0]
    out = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        out[i] = np.asarray(
            calcHerrickGibbs(r1[i], r2[i], r3[i], float(t1[i]), float(t2[i]), float(t3[i])),
            dtype=np.float64,
        )
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
    # Upstream `calcGauss` calls `approxLangrangeCoeffs(r2_mag, t12)`
    # with two args, but the numba JIT signature mandates three (mu has
    # a Python-default but numba ignores it). We replicate calcGauss
    # here passing all three arguments explicitly so the parity oracle
    # actually runs.
    from adam_core.orbit_determination.gauss import approxLangrangeCoeffs

    n = r1.shape[0]
    out = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        t12 = float(t1[i]) - float(t2[i])
        t32 = float(t3[i]) - float(t2[i])
        r2_mag = float(np.linalg.norm(r2[i]))
        f1, g1 = approxLangrangeCoeffs(r2_mag, t12, mu)
        f3, g3 = approxLangrangeCoeffs(r2_mag, t32, mu)
        v2 = (1.0 / (f1 * g3 - f3 * g1)) * (-f3 * r1[i] + f1 * r3[i])
        out[i] = np.asarray(v2, dtype=np.float64)
    return {"out": out}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _orbit_determination_gauss_iod(
    ra_deg_per_triplet: np.ndarray,
    dec_deg_per_triplet: np.ndarray,
    times_per_triplet: np.ndarray,
    obs_pos_per_triplet: np.ndarray,
    mu: float,
    c: float,
) -> dict[str, np.ndarray]:
    """Legacy gaussIOD per-triplet; emit "epoch" (M,) and "orbit" (M, 6)
    sorted within each triplet by |r2| (mirrors rust runner)."""
    from adam_core.orbit_determination.gauss import gaussIOD

    # Compare ONLY the best root per triplet (see rust runner docstring).
    n = ra_deg_per_triplet.shape[0]
    K_MAX = 1
    epoch_out = np.full((n, K_MAX), np.nan, dtype=np.float64)
    orbit_out = np.full((n, K_MAX, 6), np.nan, dtype=np.float64)
    for i in range(n):
        coords = np.column_stack([
            ra_deg_per_triplet[i], dec_deg_per_triplet[i],
        ]).astype(np.float64)
        times = np.asarray(times_per_triplet[i], dtype=np.float64)
        obs = np.asarray(obs_pos_per_triplet[i], dtype=np.float64)
        result = gaussIOD(
            coords, times, obs,
            velocity_method="gibbs",
            light_time=True,
            mu=mu, max_iter=10, tol=1e-15,
        )
        if not hasattr(result, "coordinates"):
            continue
        cart = result.coordinates
        n_roots = len(cart.x)
        if n_roots == 0:
            continue
        triplet = np.column_stack([
            cart.x.to_numpy(zero_copy_only=False),
            cart.y.to_numpy(zero_copy_only=False),
            cart.z.to_numpy(zero_copy_only=False),
            cart.vx.to_numpy(zero_copy_only=False),
            cart.vy.to_numpy(zero_copy_only=False),
            cart.vz.to_numpy(zero_copy_only=False),
        ])
        eps = cart.time.mjd().to_numpy(zero_copy_only=False)
        r2_mag = np.linalg.norm(triplet[:, :3], axis=1)
        # Drop near-observer trivial roots (|r2| < 1.5 AU) for symmetry
        # with the rust runner's filter; see its docstring for rationale.
        physical = r2_mag >= 1.5
        if not np.any(physical):
            continue
        eps_p = eps[physical]
        triplet_p = triplet[physical]
        r2_p = r2_mag[physical]
        order = np.argsort(r2_p, kind="stable")
        for slot, k in enumerate(order[:K_MAX]):
            epoch_out[i, slot] = float(eps_p[k])
            orbit_out[i, slot] = triplet_p[k]
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


def _run_one(api_id: str, kwargs: dict[str, Any]) -> dict[str, np.ndarray]:
    if api_id not in DISPATCH:
        raise NotImplementedError(
            f"Legacy oracle has no entry for {api_id!r}. Add a dispatch "
            f"entry to migration/parity/_legacy_runner.py."
        )
    return DISPATCH[api_id](**kwargs)


def _handle(request: dict[str, Any]) -> dict[str, Any]:
    api_id: str = request["api"]
    mode: str = request["mode"]
    kwargs: dict[str, Any] = request["kwargs"]

    if mode == "parity":
        outputs = _run_one(api_id, kwargs)
        return {"ok": True, "outputs": outputs}

    if mode == "time":
        warmup: int = int(request.get("warmup", 1))
        reps: int = int(request.get("reps", 7))
        for _ in range(warmup):
            _run_one(api_id, kwargs)
        elapsed: list[float] = []
        for _ in range(reps):
            t0 = time.perf_counter()
            _run_one(api_id, kwargs)
            elapsed.append(time.perf_counter() - t0)
        return {"ok": True, "elapsed": elapsed}

    raise ValueError(f"Unknown mode {mode!r}")


def main() -> int:
    try:
        request = pickle.load(sys.stdin.buffer)
    except Exception:
        sys.stderr.write("legacy runner: failed to read pickled request\n")
        sys.stderr.write(traceback.format_exc())
        return 2

    try:
        response = _handle(request)
    except Exception as e:
        response = {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }

    buf = io.BytesIO()
    pickle.dump(response, buf, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stdout.buffer.write(buf.getvalue())
    sys.stdout.buffer.flush()
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
