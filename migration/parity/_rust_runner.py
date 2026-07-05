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
from migration.parity._porkchop_runner import (
    run_generate_porkchop_data as _dynamics_generate_porkchop_data,
)


def _ensure(arr: Any, name: str) -> np.ndarray:
    if arr is None:
        raise RuntimeError(
            f"Rust backend returned None for {name!r}; native wrappers must "
            "raise or return arrays, never silently fall back."
        )
    return np.asarray(arr, dtype=np.float64)


def _coordinates_transform_coordinates(
    cases: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Public ``transform_coordinates`` dispatcher on the migration checkout."""
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


def _coordinates_transform_coordinates_with_covariance(
    cases: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    outputs: dict[str, np.ndarray] = {}
    for case in cases:
        name = str(case["name"])
        coords = np.ascontiguousarray(case["coords"], dtype=np.float64)
        n = coords.shape[0]
        covariances = np.ascontiguousarray(
            np.asarray(case["covariance"], dtype=np.float64).reshape(n, 36)
        )
        coords_out, covariances_out = (
            _rust_api.transform_coordinates_with_covariance_numpy(
                coords,
                covariances,
                str(case["representation_in"]),
                str(case["representation_out"]),
                t0=np.ascontiguousarray(case["time_mjd"], dtype=np.float64),
                mu=np.ascontiguousarray(case["mu"], dtype=np.float64),
                max_iter=1000,
                tol=1e-15,
                frame_in=str(case["frame_in"]),
                frame_out=str(case["frame_out"]),
            )
        )
        outputs[name] = _ensure(
            coords_out, f"transform_coordinates_with_covariance.{name}"
        )
        outputs[f"{name}_covariance"] = _ensure(
            covariances_out,
            f"transform_coordinates_with_covariance.{name}_covariance",
        ).reshape(n, 6, 6)
    return outputs


def _coordinates_rotate_cartesian_time_varying(
    coords: np.ndarray,
    time_index: np.ndarray,
    matrices: np.ndarray,
    covariances: np.ndarray,
) -> dict[str, np.ndarray]:
    coords_out, cov_out = _rust_api.rotate_cartesian_time_varying_numpy(
        coords, time_index, matrices, covariances
    )
    return {
        "coords": _ensure(coords_out, "rotate_cartesian_time_varying.coords"),
        "covariances": _ensure(cov_out, "rotate_cartesian_time_varying.covariances"),
    }


def _dynamics_calc_mean_motion(a: np.ndarray, mu: np.ndarray) -> dict[str, np.ndarray]:
    return {"out": _ensure(_rust_api.calc_mean_motion_numpy(a, mu), "calc_mean_motion")}


def _dynamics_tisserand_parameter(
    a: np.ndarray, e: np.ndarray, i: np.ndarray, third_body: str
) -> dict[str, np.ndarray]:
    from adam_core.dynamics.tisserand import calc_tisserand_parameter

    out = calc_tisserand_parameter(a, e, i, third_body=third_body)
    return {"out": _ensure(out, "tisserand_parameter")}


def _orbits_classify_orbits(
    a: np.ndarray, e: np.ndarray, q: np.ndarray, q_apo: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "out": _ensure(
            _rust_api.classify_orbits_numpy(a, e, q, q_apo), "classify_orbits"
        )
    }


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
        moid, dt = _rust_api.calculate_moid_numpy(
            primary_orbits[i], secondary_orbits[i], float(mus[i]), max_iter, xtol
        )
        moids[i] = moid
        dts[i] = dt
    return {"moid": moids, "dt_at_min": dts}


def _dynamics_calculate_moid_batch(
    primary_orbits: np.ndarray,
    secondary_orbits: np.ndarray,
    mus: np.ndarray,
    max_iter: int,
    xtol: float,
) -> dict[str, np.ndarray]:
    result = _rust_api.calculate_moid_batch_numpy(
        primary_orbits, secondary_orbits, mus, max_iter, xtol
    )
    if result is None:
        raise RuntimeError("rust calculate_moid_batch unavailable")
    moids, dts = result
    return {
        "moid": np.asarray(moids, dtype=np.float64),
        "dt_at_min": np.asarray(dts, dtype=np.float64),
    }


def _missions_porkchop_grid(
    dep_states: np.ndarray,
    dep_mjds: np.ndarray,
    arr_states: np.ndarray,
    arr_mjds: np.ndarray,
    mu: float,
    prograde: bool,
    maxiter: int,
    atol: float,
    rtol: float,
) -> dict[str, np.ndarray]:
    result = _rust_api.porkchop_grid_numpy(
        dep_states, dep_mjds, arr_states, arr_mjds, mu, prograde, maxiter, atol, rtol
    )
    if result is None:
        raise RuntimeError("rust porkchop_grid unavailable")
    dep_idx, arr_idx, v1, v2 = result
    return {
        "departure_index": np.asarray(dep_idx, dtype=np.float64),
        "arrival_index": np.asarray(arr_idx, dtype=np.float64),
        "solution_departure_velocity": np.asarray(v1, dtype=np.float64),
        "solution_arrival_velocity": np.asarray(v2, dtype=np.float64),
    }


def _observers_from_codes(codes: Any, mjd_utc: Any) -> dict[str, np.ndarray]:
    """Migration ``Observers.from_codes`` heliocentric observer states."""
    from adam_core.observers import Observers
    from adam_core.time import Timestamp

    times = Timestamp.from_mjd(np.asarray(mjd_utc, dtype=np.float64), scale="utc")
    observers = Observers.from_codes(list(codes), times)
    return {
        "coordinates": _ensure(
            np.asarray(observers.coordinates.values, dtype=np.float64),
            "observers.Observers.from_codes",
        ),
    }


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
    return {
        "out": _ensure(
            _rust_api.calculate_chi2_numpy(residuals, covariances), "calculate_chi2"
        )
    }


def _coordinates_residuals_bound_longitude_residuals(
    observed: np.ndarray, residuals: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "out": _ensure(
            _rust_api.bound_longitude_residuals_numpy(observed, residuals),
            "bound_longitude_residuals",
        )
    }


def _coordinates_residuals_apply_cosine_latitude_correction(
    lat: np.ndarray, residuals: np.ndarray, covariances: np.ndarray
) -> dict[str, np.ndarray]:
    result = _rust_api.apply_cosine_latitude_correction_numpy(
        lat, residuals, covariances
    )
    if result is None:
        raise RuntimeError("rust apply_cosine_latitude_correction unavailable")
    residuals_out, covariances_out = result
    return {
        "residuals": np.asarray(residuals_out, dtype=np.float64),
        "covariances": np.asarray(covariances_out, dtype=np.float64),
    }


def _statistics_weighted_mean(
    samples: np.ndarray, weights: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "out": _ensure(_rust_api.weighted_mean_numpy(samples, weights), "weighted_mean")
    }


def _statistics_weighted_covariance(
    mean: np.ndarray, samples: np.ndarray, weights: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "out": _ensure(
            _rust_api.weighted_covariance_numpy(mean, samples, weights),
            "weighted_covariance",
        )
    }


def _coordinates_residuals_Residuals_calculate(
    observed_values: np.ndarray,
    predicted_values: np.ndarray,
    observed_covariance_matrices: np.ndarray,
    origin_codes: np.ndarray,
    frame: str,
) -> dict[str, np.ndarray]:
    """End-to-end ``Residuals.calculate`` over the OD-inner-loop shape.

    Builds ``SphericalCoordinates`` from kwargs, calls ``Residuals.calculate``,
    extracts the four quivr columns as ndarrays. Suppresses the legacy
    off-diagonal NaN ``UserWarning`` so parity runs are quiet (the warning
    semantics are exercised by targeted tests).
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
    return {
        "out": _ensure(
            _rust_api.propagate_2body_numpy(orbits, dts, mus, max_iter, tol),
            "propagate_2body",
        )
    }


def _dynamics_propagate_2body_along_arc(
    orbit: np.ndarray, dts: np.ndarray, mu: float, max_iter: int, tol: float
) -> dict[str, np.ndarray]:
    return {
        "out": _ensure(
            _rust_api.propagate_2body_along_arc_numpy(orbit, dts, mu, max_iter, tol),
            "propagate_2body_along_arc",
        )
    }


def _dynamics_propagate_2body_arc_batch(
    orbits: np.ndarray, dts: np.ndarray, mus: np.ndarray, max_iter: int, tol: float
) -> dict[str, np.ndarray]:
    return {
        "out": _ensure(
            _rust_api.propagate_2body_arc_batch_numpy(orbits, dts, mus, max_iter, tol),
            "propagate_2body_arc_batch",
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


def _pack_absolute_magnitude_fit(result: tuple[Any, ...]) -> dict[str, np.ndarray]:
    h_hat, h_sigma, sigma_eff, chi2_red, n_used = result
    return {
        "h_hat": np.asarray(h_hat, dtype=np.float64).reshape(-1),
        "h_sigma": np.asarray(h_sigma, dtype=np.float64).reshape(-1),
        "sigma_eff": np.asarray(sigma_eff, dtype=np.float64).reshape(-1),
        "chi2_red": np.asarray(chi2_red, dtype=np.float64).reshape(-1),
        "n_used": np.asarray(n_used, dtype=np.float64).reshape(-1),
    }


def _photometry_fit_absolute_magnitude_rows(
    h_rows: np.ndarray, sigma_rows: np.ndarray
) -> dict[str, np.ndarray]:
    return _pack_absolute_magnitude_fit(
        _rust_api.fit_absolute_magnitude_rows_numpy(h_rows, sigma_rows)
    )


def _photometry_fit_absolute_magnitude_grouped(
    h_rows: np.ndarray, sigma_rows: np.ndarray, group_offsets: np.ndarray
) -> dict[str, np.ndarray]:
    return _pack_absolute_magnitude_fit(
        _rust_api.fit_absolute_magnitude_grouped_numpy(
            h_rows, sigma_rows, group_offsets
        )
    )


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
    # actually picks, and the shared-root tolerance is tight enough to catch
    # metre-scale state drift.
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


# ---------------------------------------------------------------------------
# Bridge signatures — W1 Arrow-bridge public surface (migration checkout side).
# ---------------------------------------------------------------------------


def _bridge_propagate_orbits_2body(
    coords: Any,
    epoch_mjd: Any,
    target_mjd: float,
    origin: str,
    frame: str,
) -> dict[str, np.ndarray]:
    """Bridge ``propagate_orbits_2body`` (Orbits -> Orbits) on the migration checkout."""
    from adam_core.coordinates.cartesian import CartesianCoordinates
    from adam_core.coordinates.origin import Origin
    from adam_core.orbits import Orbits
    from adam_core.orbits.arrow_bridge import _propagate_orbits_2body_ipc_candidate
    from adam_core.time import Timestamp

    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    cart = CartesianCoordinates.from_kwargs(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        vx=coords[:, 3],
        vy=coords[:, 4],
        vz=coords[:, 5],
        time=Timestamp.from_mjd(np.asarray(epoch_mjd, dtype=np.float64), scale="tdb"),
        origin=Origin.from_kwargs(code=np.full(n, str(origin), dtype="object")),
        frame=str(frame),
    )
    orbits = Orbits.from_kwargs(
        orbit_id=[str(i) for i in range(n)],
        coordinates=cart,
    )
    target = Timestamp.from_mjd(
        np.asarray([float(target_mjd)], dtype=np.float64), scale="tdb"
    )
    out = _propagate_orbits_2body_ipc_candidate(orbits, target)
    return {"out": _ensure(out.coordinates.values, "bridge.propagate_orbits_2body")}


def _bridge_rotate_orbits_frame(
    coords: Any,
    epoch_mjd: Any,
    covariance: Any,
    origin: str,
    frame_in: str,
    frame_out: str,
) -> dict[str, np.ndarray]:
    """Diagnostic Arrow-IPC frame-rotation candidate on the migration checkout."""
    from adam_core.coordinates.cartesian import CartesianCoordinates
    from adam_core.coordinates.covariances import CoordinateCovariances
    from adam_core.coordinates.origin import Origin
    from adam_core.orbits import Orbits
    from adam_core.orbits.arrow_bridge import _rotate_orbits_frame_ipc_candidate
    from adam_core.time import Timestamp

    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    cart = CartesianCoordinates.from_kwargs(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        vx=coords[:, 3],
        vy=coords[:, 4],
        vz=coords[:, 5],
        time=Timestamp.from_mjd(np.asarray(epoch_mjd, dtype=np.float64), scale="tdb"),
        origin=Origin.from_kwargs(code=np.full(n, str(origin), dtype="object")),
        frame=str(frame_in),
        covariance=CoordinateCovariances.from_matrix(
            np.asarray(covariance, dtype=np.float64)
        ),
    )
    orbits = Orbits.from_kwargs(
        orbit_id=[str(i) for i in range(n)],
        coordinates=cart,
    )
    out = _rotate_orbits_frame_ipc_candidate(orbits, str(frame_out))
    return {
        "values": _ensure(out.coordinates.values, "bridge.rotate_orbits_frame"),
        "covariance": _ensure(
            out.coordinates.covariance.to_matrix(), "bridge.rotate_orbits_frame.cov"
        ),
    }


def _bridge_sample_orbit_variants(
    coords: Any,
    epoch_mjd: Any,
    covariance: Any,
    origin: str,
    frame: str,
) -> dict[str, np.ndarray]:
    """Bridge candidate for ``VariantOrbits.create`` sigma-point sampling."""
    from adam_core.coordinates.cartesian import CartesianCoordinates
    from adam_core.coordinates.covariances import CoordinateCovariances
    from adam_core.coordinates.origin import Origin
    from adam_core.orbits import Orbits
    from adam_core.orbits.arrow_bridge import _sample_orbit_variants_ipc_candidate
    from adam_core.time import Timestamp

    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    cart = CartesianCoordinates.from_kwargs(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        vx=coords[:, 3],
        vy=coords[:, 4],
        vz=coords[:, 5],
        time=Timestamp.from_mjd(np.asarray(epoch_mjd, dtype=np.float64), scale="tdb"),
        origin=Origin.from_kwargs(code=np.full(n, str(origin), dtype="object")),
        frame=str(frame),
        covariance=CoordinateCovariances.from_matrix(
            np.asarray(covariance, dtype=np.float64)
        ),
    )
    orbits = Orbits.from_kwargs(
        orbit_id=[str(i) for i in range(n)],
        coordinates=cart,
    )
    out = _sample_orbit_variants_ipc_candidate(orbits, method="sigma-point")
    return {
        "coordinates": _ensure(out.coordinates.values, "bridge.sample_orbit_variants"),
        "weights": _ensure(
            out.weights.to_numpy(zero_copy_only=False),
            "bridge.sample_orbit_variants.weights",
        ),
        "weights_cov": _ensure(
            out.weights_cov.to_numpy(zero_copy_only=False),
            "bridge.sample_orbit_variants.weights_cov",
        ),
    }


def _bridge_evaluate_residuals_2body(
    orbit_coords: Any,
    observer_coords: Any,
    observed_sph: Any,
    observed_cov: Any,
    epoch_mjd: Any,
    origin: str,
    frame: str,
    observer_code: str,
    observed_origin: Any,
    observed_frame: str,
) -> dict[str, np.ndarray]:
    """Bridge ``evaluate_residuals_2body`` (OD inner loop) on the migration checkout."""
    from adam_core.coordinates.cartesian import CartesianCoordinates
    from adam_core.coordinates.covariances import CoordinateCovariances
    from adam_core.coordinates.origin import Origin
    from adam_core.coordinates.spherical import SphericalCoordinates
    from adam_core.observers import Observers
    from adam_core.orbits import Orbits
    from adam_core.orbits.arrow_bridge import (
        _evaluate_residuals_2body_ipc_candidate,
    )
    from adam_core.time import Timestamp

    orbit_coords = np.asarray(orbit_coords, dtype=np.float64)
    n = orbit_coords.shape[0]
    times = Timestamp.from_mjd(np.asarray(epoch_mjd, dtype=np.float64), scale="tdb")
    ssb = Origin.from_kwargs(code=np.full(n, str(origin), dtype="object"))
    orbits = Orbits.from_kwargs(
        orbit_id=[str(i) for i in range(n)],
        coordinates=CartesianCoordinates.from_kwargs(
            x=orbit_coords[:, 0],
            y=orbit_coords[:, 1],
            z=orbit_coords[:, 2],
            vx=orbit_coords[:, 3],
            vy=orbit_coords[:, 4],
            vz=orbit_coords[:, 5],
            time=times,
            origin=ssb,
            frame=str(frame),
        ),
    )
    observer_coords = np.asarray(observer_coords, dtype=np.float64)
    observers = Observers.from_kwargs(
        code=[str(observer_code)] * n,
        coordinates=CartesianCoordinates.from_kwargs(
            x=observer_coords[:, 0],
            y=observer_coords[:, 1],
            z=observer_coords[:, 2],
            vx=observer_coords[:, 3],
            vy=observer_coords[:, 4],
            vz=observer_coords[:, 5],
            time=times,
            origin=Origin.from_kwargs(code=np.full(n, str(origin), dtype="object")),
            frame=str(frame),
        ),
    )
    observed_sph = np.asarray(observed_sph, dtype=np.float64)
    observed = SphericalCoordinates.from_kwargs(
        rho=observed_sph[:, 0],
        lon=observed_sph[:, 1],
        lat=observed_sph[:, 2],
        vrho=observed_sph[:, 3],
        vlon=observed_sph[:, 4],
        vlat=observed_sph[:, 5],
        time=times,
        origin=Origin.from_kwargs(
            code=np.asarray(list(observed_origin), dtype="object")
        ),
        frame=str(observed_frame),
        covariance=CoordinateCovariances.from_matrix(
            np.asarray(observed_cov, dtype=np.float64)
        ),
    )
    chi2, _residuals = _evaluate_residuals_2body_ipc_candidate(
        orbits, observed, observers
    )
    return {"chi2": _ensure(chi2, "bridge.evaluate_residuals_2body")}


DISPATCH = {
    "coordinates.cartesian_to_spherical": _coordinates_cartesian_to_spherical,
    "coordinates.transform_coordinates": _coordinates_transform_coordinates,
    "coordinates.transform_coordinates_with_covariance": (
        _coordinates_transform_coordinates_with_covariance
    ),
    "coordinates.cartesian_to_geodetic": _coordinates_cartesian_to_geodetic,
    "coordinates.cartesian_to_keplerian": _coordinates_cartesian_to_keplerian,
    "coordinates.keplerian.to_cartesian": _coordinates_keplerian_to_cartesian,
    "coordinates.cartesian_to_cometary": _coordinates_cartesian_to_cometary,
    "coordinates.cometary.to_cartesian": _coordinates_cometary_to_cartesian,
    "coordinates.spherical.to_cartesian": _coordinates_spherical_to_cartesian,
    "coordinates.rotate_cartesian_time_varying": _coordinates_rotate_cartesian_time_varying,
    "coordinates.residuals.Residuals.calculate": _coordinates_residuals_Residuals_calculate,
    "coordinates.residuals.calculate_chi2": _coordinates_residuals_calculate_chi2,
    "coordinates.residuals.bound_longitude_residuals": (
        _coordinates_residuals_bound_longitude_residuals
    ),
    "coordinates.residuals.apply_cosine_latitude_correction": (
        _coordinates_residuals_apply_cosine_latitude_correction
    ),
    "statistics.weighted_mean": _statistics_weighted_mean,
    "statistics.weighted_covariance": _statistics_weighted_covariance,
    "dynamics.calc_mean_motion": _dynamics_calc_mean_motion,
    "dynamics.tisserand_parameter": _dynamics_tisserand_parameter,
    "orbits.classify_orbits": _orbits_classify_orbits,
    "dynamics.calculate_moid": _dynamics_calculate_moid,
    "dynamics.calculate_moid_batch": _dynamics_calculate_moid_batch,
    "dynamics.calculate_perturber_moids": _dynamics_calculate_perturber_moids,
    "observers.Observers.from_codes": _observers_from_codes,
    "missions.porkchop_grid": _missions_porkchop_grid,
    "dynamics.generate_porkchop_data": _dynamics_generate_porkchop_data,
    "dynamics.propagate_2body": _dynamics_propagate_2body,
    "dynamics.propagate_2body_along_arc": _dynamics_propagate_2body_along_arc,
    "dynamics.propagate_2body_arc_batch": _dynamics_propagate_2body_arc_batch,
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
    "photometry.fit_absolute_magnitude_rows": _photometry_fit_absolute_magnitude_rows,
    "photometry.fit_absolute_magnitude_grouped": (
        _photometry_fit_absolute_magnitude_grouped
    ),
    "orbit_determination.calcGibbs": _orbit_determination_calc_gibbs,
    "orbit_determination.calcHerrickGibbs": _orbit_determination_calc_herrick_gibbs,
    "orbit_determination.calcGauss": _orbit_determination_calc_gauss,
    "orbit_determination.gaussIOD": _orbit_determination_gauss_iod,
    "bridge.propagate_orbits_2body": _bridge_propagate_orbits_2body,
    "bridge.rotate_orbits_frame": _bridge_rotate_orbits_frame,
    "bridge.sample_orbit_variants": _bridge_sample_orbit_variants,
    "bridge.evaluate_residuals_2body": _bridge_evaluate_residuals_2body,
}


def run(api_id: str, **kwargs: Any) -> dict[str, np.ndarray]:
    if api_id not in DISPATCH:
        raise KeyError(f"No rust dispatch for {api_id!r}")
    return DISPATCH[api_id](**kwargs)
