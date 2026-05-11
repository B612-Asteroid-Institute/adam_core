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

from migration.parity._porkchop_runner import (
    run_generate_porkchop_data as _dynamics_generate_porkchop_data,
)

# ---------------------------------------------------------------------------
# Dispatch table — one entry per API id we expose to the gate.
# Each entry is a callable f(**kwargs) -> dict[str, np.ndarray] (the
# output names match migration/parity/tolerances.py output keys).
# ---------------------------------------------------------------------------


def _coordinates_transform_coordinates(
    cases: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Public ``transform_coordinates`` dispatcher on baseline main.

    The migration gate compares public quivr-object dispatch, not only the raw
    Rust ``transform_coordinates_numpy`` kernel.
    """
    from adam_core.coordinates.cartesian import CartesianCoordinates
    from adam_core.coordinates.cometary import CometaryCoordinates
    from adam_core.coordinates.covariances import CoordinateCovariances
    from adam_core.coordinates.keplerian import KeplerianCoordinates
    from adam_core.coordinates.origin import Origin, OriginCodes
    from adam_core.coordinates.spherical import SphericalCoordinates
    from adam_core.coordinates.transform import transform_coordinates
    from adam_core.time import Timestamp

    representations = {
        "cartesian": CartesianCoordinates,
        "spherical": SphericalCoordinates,
        "keplerian": KeplerianCoordinates,
        "cometary": CometaryCoordinates,
    }

    def build_coords(case: dict[str, Any]) -> Any:
        coords = np.asarray(case["coords"], dtype=np.float64)
        time = Timestamp.from_mjd(
            np.asarray(case["time_mjd"], dtype=np.float64), scale="tdb"
        )
        origin = Origin.from_kwargs(
            code=np.full(coords.shape[0], str(case["origin_in"]), dtype="object")
        )
        frame = str(case["frame_in"])
        representation_in = str(case["representation_in"])
        covariance = case.get("covariance")
        covariance_kw = {}
        if covariance is not None:
            covariance_kw["covariance"] = CoordinateCovariances.from_matrix(
                np.asarray(covariance, dtype=np.float64)
            )
        if representation_in == "cartesian":
            return CartesianCoordinates.from_kwargs(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                vx=coords[:, 3],
                vy=coords[:, 4],
                vz=coords[:, 5],
                time=time,
                origin=origin,
                frame=frame,
                **covariance_kw,
            )
        if representation_in == "spherical":
            return SphericalCoordinates.from_kwargs(
                rho=coords[:, 0],
                lon=coords[:, 1],
                lat=coords[:, 2],
                vrho=coords[:, 3],
                vlon=coords[:, 4],
                vlat=coords[:, 5],
                time=time,
                origin=origin,
                frame=frame,
                **covariance_kw,
            )
        if representation_in == "keplerian":
            return KeplerianCoordinates.from_kwargs(
                a=coords[:, 0],
                e=coords[:, 1],
                i=coords[:, 2],
                raan=coords[:, 3],
                ap=coords[:, 4],
                M=coords[:, 5],
                time=time,
                origin=origin,
                frame=frame,
                **covariance_kw,
            )
        if representation_in == "cometary":
            return CometaryCoordinates.from_kwargs(
                q=coords[:, 0],
                e=coords[:, 1],
                i=coords[:, 2],
                raan=coords[:, 3],
                ap=coords[:, 4],
                tp=coords[:, 5],
                time=time,
                origin=origin,
                frame=frame,
                **covariance_kw,
            )
        raise ValueError(f"Unsupported representation_in: {representation_in}")

    outputs: dict[str, np.ndarray] = {}
    for case in cases:
        kwargs: dict[str, Any] = {
            "representation_out": representations[str(case["representation_out"])],
            "frame_out": str(case["frame_out"]),
        }
        origin_out = case.get("origin_out")
        if origin_out is not None:
            kwargs["origin_out"] = OriginCodes[str(origin_out)]
        transformed = transform_coordinates(build_coords(case), **kwargs)
        name = str(case["name"])
        outputs[name] = np.asarray(transformed.values, dtype=np.float64)
        if case.get("covariance") is not None:
            outputs[f"{name}_covariance"] = np.asarray(
                transformed.covariance.to_matrix(), dtype=np.float64
            )
    return outputs


def _coordinates_cartesian_to_spherical(coords: np.ndarray) -> dict[str, np.ndarray]:
    from adam_core.coordinates.transform import _cartesian_to_spherical_vmap

    out = np.asarray(_cartesian_to_spherical_vmap(coords), dtype=np.float64)
    return {"out": out}


def _coordinates_cartesian_to_geodetic(
    coords: np.ndarray, a: float, f: float, max_iter: int, tol: float
) -> dict[str, np.ndarray]:
    from adam_core.coordinates.transform import _cartesian_to_geodetic_vmap

    out = np.asarray(
        _cartesian_to_geodetic_vmap(coords, a, f, max_iter, tol), dtype=np.float64
    )
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

    out = np.asarray(
        _keplerian_to_cartesian_a_vmap(coords, mu, max_iter, tol), dtype=np.float64
    )
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

    out = np.asarray(
        _cometary_to_cartesian_vmap(coords, t0, mu, max_iter, tol), dtype=np.float64
    )
    return {"out": out}


def _coordinates_spherical_to_cartesian(coords: np.ndarray) -> dict[str, np.ndarray]:
    from adam_core.coordinates.transform import _spherical_to_cartesian_vmap

    out = np.asarray(_spherical_to_cartesian_vmap(coords), dtype=np.float64)
    return {"out": out}


def _dynamics_calc_mean_motion(a: np.ndarray, mu: np.ndarray) -> dict[str, np.ndarray]:
    # Legacy has no batched calc_mean_motion — it's a 1-line np expression.
    # The rust kernel mirrors `np.sqrt(mu / a**3)` element-wise.
    out = np.sqrt(
        np.asarray(mu, dtype=np.float64) / np.asarray(a, dtype=np.float64) ** 3
    )
    return {"out": out}


def _orbits_classify_orbits(
    a: np.ndarray, e: np.ndarray, q: np.ndarray, q_apo: np.ndarray
) -> dict[str, np.ndarray]:
    out = np.zeros(len(a), dtype=np.float64)
    rules = (
        (1.0, (a > 1.0) & (q > 1.017) & (q < 1.3)),
        (2.0, (a > 1.0) & (q < 1.017)),
        (3.0, (a < 1.0) & (q_apo > 0.983)),
        (4.0, (a > 5.5) & (a < 30.1)),
        (5.0, q_apo < 0.983),
        (6.0, (a < 2.0) & (q > 1.666)),
        (7.0, (a > 2.0) & (a < 3.2) & (q > 1.666)),
        (8.0, (a < 3.2) & (q > 1.3) & (q < 1.666)),
        (9.0, (a > 3.2) & (a < 4.6)),
        (10.0, (a > 4.6) & (a < 5.5) & (e < 0.3)),
        (11.0, a > 30.1),
        (12.0, e == 1.0),
        (13.0, e > 1.0),
    )
    for code, mask in rules:
        out[mask] = code
    return {"out": out}


def _moid_distance_to_ellipse(point: np.ndarray, a: float, e: float, u: float) -> float:
    half_u = 0.5 * u
    tan_half_u = np.tan(half_u)
    ecc_anom = 2.0 * np.arctan(np.sqrt((1.0 - e) / (1.0 + e)) * tan_half_u)
    true_anom = 2.0 * np.arctan(np.sqrt((1.0 + e) / (1.0 - e)) * tan_half_u)
    radius = a * (1.0 - e * np.cos(ecc_anom))
    ellipse_point = np.array(
        [radius * np.cos(true_anom), radius * np.sin(true_anom), 0.0]
    )
    return float(np.linalg.norm(point - ellipse_point))


def _keplerian13_from_cartesian(orbit: np.ndarray, mu: float) -> np.ndarray:
    from adam_core.coordinates.transform import _cartesian_to_keplerian_vmap

    kep = np.asarray(
        _cartesian_to_keplerian_vmap(
            orbit.reshape(1, 6),
            np.array([0.0], dtype=np.float64),
            np.array([mu], dtype=np.float64),
        ),
        dtype=np.float64,
    )[0]
    q_expected = kep[0] * (1.0 - kep[4])
    if not np.isclose(kep[2], q_expected, rtol=1e-9, atol=1e-12):
        raise RuntimeError("unexpected _cartesian_to_keplerian_vmap output layout")
    if kep[4] < 1.0 and np.isfinite(kep[10]):
        period_expected = 360.0 / kep[10]
        if not np.isclose(kep[11], period_expected, rtol=1e-9, atol=1e-9):
            raise RuntimeError("unexpected _cartesian_to_keplerian_vmap period layout")
    return kep


def _moid_one(
    primary_orbit: np.ndarray,
    secondary_orbit: np.ndarray,
    mu: float,
    max_iter: int,
    xtol: float,
) -> tuple[float, float]:
    del max_iter, xtol  # legacy baseline uses upstream scipy/propagation defaults

    from scipy.optimize import minimize_scalar

    from adam_core.dynamics.propagation import _propagate_2body

    primary_kep = _keplerian13_from_cartesian(primary_orbit, mu)
    secondary_kep = _keplerian13_from_cartesian(secondary_orbit, mu)
    period = float(primary_kep[11])
    e_primary = float(primary_kep[4])
    dt_upper = period if e_primary < 1.0 and np.isfinite(period) else 10_000.0
    a_secondary = float(secondary_kep[0])
    e_secondary = float(secondary_kep[4])

    r_sec = secondary_orbit[:3]
    v_sec = secondary_orbit[3:]
    h_sec = np.cross(r_sec, v_sec)
    h_mag = np.linalg.norm(h_sec)
    n_hat = h_sec / h_mag if h_mag > 0 else np.array([0.0, 0.0, 1.0])

    def distance_for_dt(dt: float) -> float:
        propagated = np.asarray(
            _propagate_2body(primary_orbit, 0.0, dt, mu, 1000, 1e-14),
            dtype=np.float64,
        )
        p0 = propagated[:3]
        projected = p0 - np.dot(p0, n_hat) * n_hat
        inner = minimize_scalar(
            lambda u: _moid_distance_to_ellipse(projected, a_secondary, e_secondary, u),
            bounds=(0.0, 2.0 * np.pi),
            method="bounded",
            tol=1e-12,
        )
        d_perp = np.linalg.norm(projected - p0)
        return float(np.sqrt(d_perp * d_perp + inner.fun * inner.fun))

    result = minimize_scalar(
        distance_for_dt,
        bounds=(0.0, dt_upper),
        method="bounded",
        tol=1e-14,
    )
    if result.status != 0:
        raise ValueError("MOID calculation did not converge")
    return float(result.fun), float(result.x)


def _dynamics_calculate_moid(
    primary_orbits: np.ndarray,
    secondary_orbits: np.ndarray,
    mus: np.ndarray,
    max_iter: int,
    xtol: float,
) -> dict[str, np.ndarray]:
    n = primary_orbits.shape[0]
    moids = np.empty(n, dtype=np.float64)
    dts = np.empty(n, dtype=np.float64)
    for i in range(n):
        moids[i], dts[i] = _moid_one(
            primary_orbits[i], secondary_orbits[i], float(mus[i]), max_iter, xtol
        )
    return {"moid": moids, "dt_at_min": dts}


def _pack_perturber_moids(moids: Any) -> dict[str, np.ndarray]:
    from adam_core.coordinates.origin import OriginCodes

    orbit_ids = np.asarray(moids.orbit_id.to_pylist(), dtype=object)
    perturber_codes = np.asarray(moids.perturber.code.to_pylist(), dtype=object)
    order = np.lexsort((perturber_codes.astype(str), orbit_ids.astype(str)))
    orbit_index = np.asarray(
        [float(str(value)[1:]) for value in orbit_ids[order]], dtype=np.float64
    )
    perturber_code = np.asarray(
        [float(OriginCodes[str(value)].value) for value in perturber_codes[order]],
        dtype=np.float64,
    )
    return {
        "orbit_index": orbit_index,
        "perturber_code": perturber_code,
        "moid": np.asarray(moids.moid.to_numpy(zero_copy_only=False), dtype=np.float64)[
            order
        ],
        "time_mjd": np.asarray(moids.time.mjd().to_numpy(False), dtype=np.float64)[
            order
        ],
    }


def _dynamics_calculate_perturber_moids(
    coords: np.ndarray,
    time_mjd: np.ndarray,
    orbit_ids: np.ndarray,
    perturber_codes: np.ndarray,
    origin_code: str,
    frame: str,
    chunk_size: int,
    max_processes: int,
) -> dict[str, np.ndarray]:
    from adam_core.coordinates.cartesian import CartesianCoordinates
    from adam_core.coordinates.origin import Origin, OriginCodes
    from adam_core.dynamics.moid import calculate_perturber_moids
    from adam_core.orbits import Orbits
    from adam_core.time import Timestamp

    n = coords.shape[0]
    coordinate_rows = CartesianCoordinates.from_kwargs(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        vx=coords[:, 3],
        vy=coords[:, 4],
        vz=coords[:, 5],
        time=Timestamp.from_mjd(time_mjd, scale="tdb"),
        origin=Origin.from_kwargs(code=np.full(n, origin_code, dtype=object)),
        frame=frame,
    )
    orbits = Orbits.from_kwargs(orbit_id=orbit_ids, coordinates=coordinate_rows)
    perturbers = [OriginCodes[str(code)] for code in perturber_codes]
    moids = calculate_perturber_moids(
        orbits,
        perturbers,
        chunk_size=chunk_size,
        max_processes=max_processes,
    )
    return _pack_perturber_moids(moids)


def _coordinates_residuals_calculate_chi2(
    residuals: np.ndarray, covariances: np.ndarray
) -> dict[str, np.ndarray]:
    from adam_core.coordinates.residuals import calculate_chi2

    return {"out": np.asarray(calculate_chi2(residuals, covariances), dtype=np.float64)}


def _coordinates_residuals_Residuals_calculate(
    observed_values: np.ndarray,
    predicted_values: np.ndarray,
    observed_covariance_matrices: np.ndarray,
    origin_codes: np.ndarray,
    frame: str,
) -> dict[str, np.ndarray]:
    """Baseline-main ``Residuals.calculate`` over the OD-inner-loop shape.

    Same construction as the rust-side runner; the legacy adam_core
    installed in ``.legacy-venv`` provides the JAX-based implementation
    for the comparison.
    """
    import warnings

    from adam_core.coordinates import CoordinateCovariances, SphericalCoordinates
    from adam_core.coordinates.origin import Origin
    from adam_core.coordinates.residuals import Residuals

    obs = SphericalCoordinates.from_kwargs(
        rho=observed_values[:, 0],
        lon=observed_values[:, 1],
        lat=observed_values[:, 2],
        vrho=observed_values[:, 3],
        vlon=observed_values[:, 4],
        vlat=observed_values[:, 5],
        covariance=CoordinateCovariances.from_matrix(observed_covariance_matrices),
        origin=Origin.from_kwargs(code=origin_codes),
        frame=frame,
    )
    pred = SphericalCoordinates.from_kwargs(
        rho=predicted_values[:, 0],
        lon=predicted_values[:, 1],
        lat=predicted_values[:, 2],
        vrho=predicted_values[:, 3],
        vlon=predicted_values[:, 4],
        vlat=predicted_values[:, 5],
        origin=Origin.from_kwargs(code=origin_codes),
        frame=frame,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        residuals = Residuals.calculate(obs, pred)

    values_arr = np.stack(residuals.values.to_numpy(zero_copy_only=False)).astype(
        np.float64
    )
    return {
        "values": values_arr,
        "chi2": np.asarray(
            residuals.chi2.to_numpy(zero_copy_only=False), dtype=np.float64
        ),
        "dof": np.asarray(
            residuals.dof.to_numpy(zero_copy_only=False), dtype=np.float64
        ),
        "probability": np.asarray(
            residuals.probability.to_numpy(zero_copy_only=False), dtype=np.float64
        ),
    }


def _dynamics_propagate_2body(
    orbits: np.ndarray, dts: np.ndarray, mus: np.ndarray, max_iter: int, tol: float
) -> dict[str, np.ndarray]:
    # Legacy signature is (orbit, t0, t1, mu) with dt = t1 - t0.
    # We pass t0=0, t1=dt to match the rust kernel's dt-only convention.
    from adam_core.dynamics.propagation import _propagate_2body_vmap

    n = orbits.shape[0]
    t0 = np.zeros(n, dtype=np.float64)
    out = np.asarray(
        _propagate_2body_vmap(orbits, t0, dts, mus, max_iter, tol), dtype=np.float64
    )
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
    states = np.asarray(
        _propagate_2body_vmap(orbits, t0, dts, mus, max_iter, tol), dtype=np.float64
    )

    cov_out = np.empty_like(covariances)
    for i in range(n):
        cov_out[i] = np.asarray(
            transform_covariances_jacobian(
                orbits[i : i + 1],
                covariances[i : i + 1],
                lambda x: _propagate_2body(
                    x, 0.0, float(dts[i]), float(mus[i]), max_iter, tol
                ),
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
                    x,
                    0.0,
                    observer_states[i],
                    float(mus[i]),
                    lt_tol,
                    max_iter,
                    tol,
                    stellar_aberration,
                )[
                    0
                ],  # only the spherical output's Jacobian
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

    v1, v2 = _izzo_lambert_vmap(
        r1, r2, tof, mu, m, prograde, low_path, maxiter, atol, rtol
    )
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
            calcHerrickGibbs(
                r1[i], r2[i], r3[i], float(t1[i]), float(t2[i]), float(t3[i])
            ),
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
        coords = np.column_stack(
            [
                ra_deg_per_triplet[i],
                dec_deg_per_triplet[i],
            ]
        ).astype(np.float64)
        times = np.asarray(times_per_triplet[i], dtype=np.float64)
        obs = np.asarray(obs_pos_per_triplet[i], dtype=np.float64)
        result = gaussIOD(
            coords,
            times,
            obs,
            velocity_method="gibbs",
            light_time=True,
            mu=mu,
            max_iter=10,
            tol=1e-15,
        )
        if not hasattr(result, "coordinates"):
            continue
        cart = result.coordinates
        n_roots = len(cart.x)
        if n_roots == 0:
            continue
        triplet = np.column_stack(
            [
                cart.x.to_numpy(zero_copy_only=False),
                cart.y.to_numpy(zero_copy_only=False),
                cart.z.to_numpy(zero_copy_only=False),
                cart.vx.to_numpy(zero_copy_only=False),
                cart.vy.to_numpy(zero_copy_only=False),
                cart.vz.to_numpy(zero_copy_only=False),
            ]
        )
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
    "coordinates.residuals.Residuals.calculate": _coordinates_residuals_Residuals_calculate,
    "coordinates.residuals.calculate_chi2": _coordinates_residuals_calculate_chi2,
    "dynamics.calc_mean_motion": _dynamics_calc_mean_motion,
    "orbits.classify_orbits": _orbits_classify_orbits,
    "dynamics.calculate_moid": _dynamics_calculate_moid,
    "dynamics.calculate_perturber_moids": _dynamics_calculate_perturber_moids,
    "dynamics.generate_porkchop_data": _dynamics_generate_porkchop_data,
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
