"""Native-Rust timing adapters for the parity/performance report.

Unlike ``_rust_runner`` (which times the compatible Python entrypoint), each
adapter here invokes a Rust-owned timing loop once and receives per-repetition
samples measured by ``std::time::Instant`` around direct Rust function calls.
Python/PyO3 launches the benchmark and passes prepared inputs, but neither is
inside any recorded sample.

Surfaces without a Rust-internal timer return an explicit unavailable result;
the report renders a blank native-Rust column plus the owning TODO rather than
substituting a PyO3-crossing measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pyarrow as pa


@dataclass(frozen=True)
class NativeRustTiming:
    status: str
    sample_trials_s: list[list[float]] = field(default_factory=list)
    entrypoint: str = ""
    reason: str = ""
    todo: str = ""
    timing_boundary: str = ""


def _build_transform_coordinates_case(case: dict[str, Any]) -> Any:
    """Build one typed coordinate input outside every native Rust sample."""
    from adam_core.coordinates.cartesian import CartesianCoordinates
    from adam_core.coordinates.cometary import CometaryCoordinates
    from adam_core.coordinates.covariances import CoordinateCovariances
    from adam_core.coordinates.keplerian import KeplerianCoordinates
    from adam_core.coordinates.origin import Origin
    from adam_core.coordinates.spherical import SphericalCoordinates
    from adam_core.time import Timestamp

    values = np.asarray(case["coords"], dtype=np.float64)
    common: dict[str, Any] = {
        "time": Timestamp.from_mjd(
            np.asarray(case["time_mjd"], dtype=np.float64), scale="tdb"
        ),
        "origin": Origin.from_kwargs(
            code=np.full(values.shape[0], str(case["origin_in"]), dtype="object")
        ),
        "frame": str(case["frame_in"]),
    }
    if "covariance" in case:
        common["covariance"] = CoordinateCovariances.from_matrix(
            np.asarray(case["covariance"], dtype=np.float64)
        )
    representation = str(case["representation_in"])
    if representation == "cartesian":
        return CartesianCoordinates.from_kwargs(
            x=values[:, 0],
            y=values[:, 1],
            z=values[:, 2],
            vx=values[:, 3],
            vy=values[:, 4],
            vz=values[:, 5],
            **common,
        )
    if representation == "spherical":
        return SphericalCoordinates.from_kwargs(
            rho=values[:, 0],
            lon=values[:, 1],
            lat=values[:, 2],
            vrho=values[:, 3],
            vlon=values[:, 4],
            vlat=values[:, 5],
            **common,
        )
    if representation == "keplerian":
        return KeplerianCoordinates.from_kwargs(
            a=values[:, 0],
            e=values[:, 1],
            i=values[:, 2],
            raan=values[:, 3],
            ap=values[:, 4],
            M=values[:, 5],
            **common,
        )
    if representation == "cometary":
        return CometaryCoordinates.from_kwargs(
            q=values[:, 0],
            e=values[:, 1],
            i=values[:, 2],
            raan=values[:, 3],
            ap=values[:, 4],
            tp=values[:, 5],
            **common,
        )
    raise ValueError(f"unsupported native transform representation: {representation}")


def _transform_coordinates(
    *, cases: list[dict[str, Any]], reps: int, warmup: int, trials: int
) -> NativeRustTiming:
    from adam_core import _rust_native
    from adam_core._rust.arrow import ensure_spice_backend
    from adam_core.coordinates.geodetics import WGS84
    from adam_core.coordinates.transform import _coordinate_record_batch

    ensure_spice_backend()
    coordinates = [_build_transform_coordinates_case(case) for case in cases]
    batches = [
        _coordinate_record_batch(coords, str(case["representation_in"]))
        for coords, case in zip(coordinates, cases)
    ]
    representations_out = [str(case["representation_out"]) for case in cases]
    frames_out = [str(case["frame_out"]) for case in cases]
    target_origins = [case.get("origin_out") for case in cases]
    axes = [
        float(WGS84.a) if representation == "geodetic" else 0.0
        for representation in representations_out
    ]
    flattenings = [
        float(WGS84.f) if representation == "geodetic" else 0.0
        for representation in representations_out
    ]
    samples = _rust_native.benchmark_transform_coordinates_arrow(
        batches,
        representations_out,
        frames_out,
        target_origins,
        axes,
        flattenings,
        reps,
        trials,
        warmup,
    )
    return NativeRustTiming(
        status="measured",
        sample_trials_s=[[float(value) for value in trial] for trial in samples],
        entrypoint="adam_core_py::coordinates::transform_coordinates_record_batch",
        timing_boundary=(
            "Rust std::time::Instant around direct RecordBatch decode, composed "
            "coordinate/covariance transform, and RecordBatch assembly calls; outer "
            "Python/PyO3 launch and PyArrow conversion excluded"
        ),
    )


def _propagate_2body(
    *,
    reps: int,
    warmup: int,
    trials: int,
    max_iter: int,
    tol: float,
    **kwargs: Any,
) -> NativeRustTiming:
    from adam_core import _rust_native
    from adam_core.dynamics.propagation import _target_times_record_batch
    from adam_core.orbits.arrow_bridge import orbits_to_record_batch

    from ._public_facades import build_propagate_2body_inputs

    orbits, targets = build_propagate_2body_inputs(**kwargs)
    samples = _rust_native.benchmark_propagate_orbits_arrow(
        orbits_to_record_batch(orbits),
        _target_times_record_batch(targets),
        reps,
        trials,
        warmup,
        max_iter,
        tol,
        100,
        None,
    )
    return NativeRustTiming(
        status="measured",
        sample_trials_s=[[float(value) for value in trial] for trial in samples],
        entrypoint="adam_core_py::coordinates::propagate_orbits_record_batch",
        timing_boundary=(
            "Rust std::time::Instant around direct Orbits RecordBatch decode, "
            "typed 2-body cross-product propagation, covariance transport, and "
            "RecordBatch assembly calls; outer Python/PyO3 launch and PyArrow "
            "conversion excluded"
        ),
    )


def _generate_ephemeris_2body(
    *,
    reps: int,
    warmup: int,
    trials: int,
    lt_tol: float,
    max_iter: int,
    tol: float,
    stellar_aberration: bool,
    **kwargs: Any,
) -> NativeRustTiming:
    from adam_core import _rust_native
    from adam_core._rust.arrow import ensure_spice_backend
    from adam_core.orbits.arrow_bridge import (
        observers_to_record_batch,
        orbits_to_record_batch,
    )

    from ._public_facades import build_generate_ephemeris_inputs

    orbits, observers = build_generate_ephemeris_inputs(**kwargs)
    ensure_spice_backend()
    samples = _rust_native.benchmark_generate_ephemeris_arrow(
        orbits_to_record_batch(orbits),
        observers_to_record_batch(observers),
        reps,
        trials,
        warmup,
        lt_tol,
        max_iter,
        tol,
        stellar_aberration,
        False,
        False,
        100,
        None,
    )
    return NativeRustTiming(
        status="measured",
        sample_trials_s=[[float(value) for value in trial] for trial in samples],
        entrypoint="adam_core_py::coordinates::generate_ephemeris_record_batch",
        timing_boundary=(
            "Rust std::time::Instant around direct Orbits/Observers RecordBatch "
            "decode, frame/origin normalization, pairwise propagation, light-time, "
            "covariance/photometry, diagnostics, and Ephemeris RecordBatch assembly; "
            "outer Python/PyO3 launch and PyArrow conversion excluded"
        ),
    )


def _kernel_timing(
    function: Callable[..., Any],
    args: tuple[Any, ...],
    *,
    reps: int,
    warmup: int,
    trials: int,
    entrypoint: str,
    kernel: str,
) -> NativeRustTiming:
    """Launch one Rust-owned timer after all canonical input preparation."""
    samples = function(*args, reps, trials, warmup)
    return NativeRustTiming(
        status="measured",
        sample_trials_s=[[float(value) for value in trial] for trial in samples],
        entrypoint=entrypoint,
        timing_boundary=(
            f"Rust std::time::Instant around direct {kernel} calls; canonical "
            "NumPy input ownership, outer Python/PyO3 launch, and output conversion excluded"
        ),
    )


def _f64(value: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value, dtype=np.float64))


def _calc_mean_motion(
    *, a: Any, mu: Any, reps: int, warmup: int, trials: int, **_unused: Any
) -> NativeRustTiming:
    from adam_core import _rust_native

    return _kernel_timing(
        _rust_native.benchmark_calc_mean_motion_numpy,
        (_f64(a), _f64(mu)),
        reps=reps,
        warmup=warmup,
        trials=trials,
        entrypoint="adam_core_rs_coords::calc_mean_motion_batch",
        kernel="calc_mean_motion_batch",
    )


def _tisserand_parameter(
    *,
    a: Any,
    e: Any,
    i: Any,
    third_body: str,
    reps: int,
    warmup: int,
    trials: int,
    **_unused: Any,
) -> NativeRustTiming:
    from adam_core import _rust_native
    from adam_core.dynamics.tisserand import MAJOR_BODIES

    ap = float(MAJOR_BODIES[str(third_body)])
    return _kernel_timing(
        _rust_native.benchmark_tisserand_parameter_numpy,
        (_f64(a), _f64(e), _f64(i), ap),
        reps=reps,
        warmup=warmup,
        trials=trials,
        entrypoint="adam_core_rs_coords::tisserand_parameter_flat",
        kernel="tisserand_parameter_flat",
    )


def _calculate_moid(
    *,
    primary_orbits: Any,
    secondary_orbits: Any,
    mus: Any,
    max_iter: int,
    xtol: float,
    reps: int,
    warmup: int,
    trials: int,
    **_unused: Any,
) -> NativeRustTiming:
    from adam_core import _rust_native

    return _kernel_timing(
        _rust_native.benchmark_calculate_moid_numpy,
        (
            _f64(primary_orbits),
            _f64(secondary_orbits),
            _f64(mus),
            int(max_iter),
            float(xtol),
        ),
        reps=reps,
        warmup=warmup,
        trials=trials,
        entrypoint="adam_core_rs_coords::calculate_moid",
        kernel="canonical per-pair calculate_moid loop",
    )


def _calculate_moid_batch(
    *,
    primary_orbits: Any,
    secondary_orbits: Any,
    mus: Any,
    max_iter: int,
    xtol: float,
    reps: int,
    warmup: int,
    trials: int,
    **_unused: Any,
) -> NativeRustTiming:
    from adam_core import _rust_native

    return _kernel_timing(
        _rust_native.benchmark_calculate_moid_batch_numpy,
        (
            _f64(primary_orbits),
            _f64(secondary_orbits),
            _f64(mus),
            int(max_iter),
            float(xtol),
        ),
        reps=reps,
        warmup=warmup,
        trials=trials,
        entrypoint="adam_core_rs_coords::calculate_moid_batch",
        kernel="calculate_moid_batch",
    )


def _propagate_2body_along_arc(
    *, reps: int, warmup: int, trials: int, orbit: Any, dts: Any, mu: float,
    max_iter: int, tol: float, **_unused: Any
) -> NativeRustTiming:
    from adam_core import _rust_native

    return _kernel_timing(
        _rust_native.benchmark_propagate_2body_along_arc_numpy,
        (_f64(orbit), _f64(dts), float(mu), int(max_iter), float(tol)),
        reps=reps, warmup=warmup, trials=trials,
        entrypoint="adam_core_rs_coords::propagate_2body_along_arc",
        kernel="propagate_2body_along_arc",
    )


def _propagate_2body_arc_batch(
    *, reps: int, warmup: int, trials: int, orbits: Any, dts: Any, mus: Any,
    max_iter: int, tol: float, **_unused: Any
) -> NativeRustTiming:
    from adam_core import _rust_native

    return _kernel_timing(
        _rust_native.benchmark_propagate_2body_arc_batch_numpy,
        (_f64(orbits), _f64(dts), _f64(mus), int(max_iter), float(tol)),
        reps=reps, warmup=warmup, trials=trials,
        entrypoint="adam_core_rs_coords::propagate_2body_arc_batch_flat6",
        kernel="propagate_2body_arc_batch_flat6",
    )


def _propagate_2body_with_covariance(
    *, reps: int, warmup: int, trials: int, orbits: Any, covariances: Any,
    dts: Any, mus: Any, max_iter: int, tol: float, **_unused: Any
) -> NativeRustTiming:
    from adam_core import _rust_native

    return _kernel_timing(
        _rust_native.benchmark_propagate_2body_with_covariance_numpy,
        (
            _f64(orbits), _f64(covariances), _f64(dts), _f64(mus),
            int(max_iter), float(tol),
        ),
        reps=reps, warmup=warmup, trials=trials,
        entrypoint="adam_core_rs_coords::propagate_2body_with_covariance_flat6",
        kernel="state and covariance propagation",
    )


def _solve_lambert(
    *, reps: int, warmup: int, trials: int, r1: Any, r2: Any, tof: Any,
    mu: float, m: int, prograde: bool, low_path: bool, maxiter: int,
    atol: float, rtol: float, **_unused: Any
) -> NativeRustTiming:
    from adam_core import _rust_native

    return _kernel_timing(
        _rust_native.benchmark_izzo_lambert_numpy,
        (
            _f64(r1), _f64(r2), _f64(tof), float(mu), int(m), bool(prograde),
            bool(low_path), int(maxiter), float(atol), float(rtol),
        ),
        reps=reps, warmup=warmup, trials=trials,
        entrypoint="adam_core_rs_coords::izzo_lambert_batch_flat",
        kernel="izzo_lambert_batch_flat",
    )


def _add_light_time(
    *, reps: int, warmup: int, trials: int, orbits: Any,
    observer_positions: Any, mus: Any, lt_tol: float, max_iter: int,
    tol: float, max_lt_iter: int, **_unused: Any
) -> NativeRustTiming:
    from adam_core import _rust_native

    return _kernel_timing(
        _rust_native.benchmark_add_light_time_numpy,
        (
            _f64(orbits), _f64(observer_positions), _f64(mus), float(lt_tol),
            int(max_iter), float(tol), int(max_lt_iter),
        ),
        reps=reps, warmup=warmup, trials=trials,
        entrypoint="adam_core_rs_coords::add_light_time_batch_flat",
        kernel="add_light_time_batch_flat",
    )


def _porkchop_grid(
    *, reps: int, warmup: int, trials: int, dep_states: Any, dep_mjds: Any,
    arr_states: Any, arr_mjds: Any, mu: float, prograde: bool, maxiter: int,
    atol: float, rtol: float, **_unused: Any
) -> NativeRustTiming:
    from adam_core import _rust_native

    return _kernel_timing(
        _rust_native.benchmark_porkchop_grid_numpy,
        (
            _f64(dep_states), _f64(dep_mjds), _f64(arr_states), _f64(arr_mjds),
            float(mu), bool(prograde), int(maxiter), float(atol), float(rtol),
        ),
        reps=reps, warmup=warmup, trials=trials,
        entrypoint="adam_core_rs_coords::porkchop_grid_flat",
        kernel="porkchop_grid_flat",
    )


def _build_orbits_table(
    coords: Any,
    time_mjd: Any,
    orbit_ids: Any,
    origin_code: str,
    frame: str,
) -> Any:
    from adam_core.coordinates.cartesian import CartesianCoordinates
    from adam_core.coordinates.origin import Origin
    from adam_core.orbits import Orbits
    from adam_core.time import Timestamp

    values = np.asarray(coords, dtype=np.float64)
    rows = values.shape[0]
    return Orbits.from_kwargs(
        orbit_id=[str(orbit_id) for orbit_id in orbit_ids],
        coordinates=CartesianCoordinates.from_kwargs(
            x=values[:, 0],
            y=values[:, 1],
            z=values[:, 2],
            vx=values[:, 3],
            vy=values[:, 4],
            vz=values[:, 5],
            time=Timestamp.from_mjd(
                np.asarray(time_mjd, dtype=np.float64), scale="tdb"
            ),
            origin=Origin.from_kwargs(
                code=np.full(rows, str(origin_code), dtype="object")
            ),
            frame=str(frame),
        ),
    )


def _calculate_perturber_moids(
    *,
    reps: int,
    warmup: int,
    trials: int,
    coords: Any,
    time_mjd: Any,
    orbit_ids: Any,
    perturber_codes: Any,
    origin_code: str,
    frame: str,
    **_unused: Any,
) -> NativeRustTiming:
    from adam_core import _rust_native
    from adam_core._rust.arrow import ensure_spice_backend
    from adam_core.orbits.arrow_bridge import orbits_to_record_batch

    orbits = _build_orbits_table(coords, time_mjd, orbit_ids, origin_code, frame)
    ensure_spice_backend()
    samples = _rust_native.benchmark_calculate_perturber_moids_arrow(
        orbits_to_record_batch(orbits),
        [str(code) for code in perturber_codes],
        reps,
        trials,
        warmup,
        100,
        1e-10,
    )
    return NativeRustTiming(
        status="measured",
        sample_trials_s=[[float(value) for value in trial] for trial in samples],
        entrypoint="adam_core_py::coordinates::perturber_moids_record_batch",
        timing_boundary=(
            "Rust std::time::Instant around direct Orbits RecordBatch decode, "
            "SPICE perturber-state lookup, batched MOID kernel, and "
            "PerturberMOIDs RecordBatch assembly; outer Python/PyO3 launch and "
            "PyArrow conversion excluded"
        ),
    )


def _generate_porkchop_data(
    *,
    reps: int,
    warmup: int,
    trials: int,
    departure_coords: Any,
    arrival_coords: Any,
    departure_time_mjd: Any,
    arrival_time_mjd: Any,
    departure_orbit_ids: Any,
    arrival_orbit_ids: Any,
    propagation_origin: str,
    frame: str,
    prograde: bool,
    max_iter: int,
    tol: float,
    **_unused: Any,
) -> NativeRustTiming:
    from adam_core import _rust_native
    from adam_core.orbits.arrow_bridge import orbits_to_record_batch

    departure = _build_orbits_table(
        departure_coords,
        departure_time_mjd,
        departure_orbit_ids,
        propagation_origin,
        frame,
    )
    arrival = _build_orbits_table(
        arrival_coords,
        arrival_time_mjd,
        arrival_orbit_ids,
        propagation_origin,
        frame,
    )
    samples = _rust_native.benchmark_generate_porkchop_data_arrow(
        orbits_to_record_batch(departure),
        orbits_to_record_batch(arrival),
        str(propagation_origin),
        reps,
        trials,
        warmup,
        prograde,
        max_iter,
        tol,
    )
    return NativeRustTiming(
        status="measured",
        sample_trials_s=[[float(value) for value in trial] for trial in samples],
        entrypoint="adam_core_py::coordinates::porkchop_record_batch",
        timing_boundary=(
            "Rust std::time::Instant around direct departure/arrival Orbits "
            "RecordBatch decode, chronological sorting, meshgrid time filter, "
            "rayon-batched Lambert, and LambertSolutions RecordBatch assembly; "
            "outer Python/PyO3 launch and PyArrow conversion excluded"
        ),
    )


def _residuals_calculate(
    *,
    reps: int,
    warmup: int,
    trials: int,
    observed_values: Any,
    predicted_values: Any,
    observed_covariance_matrices: Any,
    origin_codes: Any,
    frame: str,
    **_unused: Any,
) -> NativeRustTiming:
    from adam_core import _rust_native
    from adam_core.coordinates import CoordinateCovariances, SphericalCoordinates
    from adam_core.coordinates.origin import Origin
    from adam_core.coordinates.transform import _coordinate_record_batch

    observed = SphericalCoordinates.from_kwargs(
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
    predicted = SphericalCoordinates.from_kwargs(
        rho=predicted_values[:, 0],
        lon=predicted_values[:, 1],
        lat=predicted_values[:, 2],
        vrho=predicted_values[:, 3],
        vlon=predicted_values[:, 4],
        vlat=predicted_values[:, 5],
        origin=Origin.from_kwargs(code=origin_codes),
        frame=frame,
    )
    samples = _rust_native.benchmark_residuals_calculate_arrow(
        _coordinate_record_batch(observed, "spherical"),
        _coordinate_record_batch(predicted, "spherical"),
        reps,
        trials,
        warmup,
        True,
    )
    return NativeRustTiming(
        status="measured",
        sample_trials_s=[[float(value) for value in trial] for trial in samples],
        entrypoint="adam_core_py::coordinates::residuals_calculate_record_batch",
        timing_boundary=(
            "Rust std::time::Instant around direct coordinate RecordBatch "
            "decode, fused residual/chi2 kernel, chi-squared survival "
            "probability, and Residuals RecordBatch assembly; outer "
            "Python/PyO3 launch and PyArrow conversion excluded"
        ),
    )


def _gauss_iod(
    *,
    reps: int,
    warmup: int,
    trials: int,
    ra_deg_per_triplet: Any,
    dec_deg_per_triplet: Any,
    times_per_triplet: Any,
    obs_pos_per_triplet: Any,
    mu: float,
    c: float,
    **_unused: Any,
) -> NativeRustTiming:
    from adam_core import _rust_native

    obs = np.ascontiguousarray(
        np.asarray(obs_pos_per_triplet, dtype=np.float64).reshape(-1, 3)
    )
    samples = _rust_native.benchmark_gauss_iod_orbits_arrow(
        np.ascontiguousarray(np.asarray(ra_deg_per_triplet, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(dec_deg_per_triplet, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(times_per_triplet, dtype=np.float64)),
        obs,
        reps,
        trials,
        warmup,
        "gibbs",
        True,
        float(mu),
        float(c),
    )
    return NativeRustTiming(
        status="measured",
        sample_trials_s=[[float(value) for value in trial] for trial in samples],
        entrypoint="adam_core_py::orbit_determination::gauss_iod_orbits_record_batch",
        timing_boundary=(
            "Rust std::time::Instant around the per-triplet fused Gauss-IOD "
            "kernel, non-finite filtering, orbit-id generation, and Orbits "
            "RecordBatch assembly; outer Python/PyO3 launch and PyArrow "
            "conversion excluded"
        ),
    )


def _variant_orbits_create(
    *,
    reps: int,
    warmup: int,
    trials: int,
    coords: Any,
    epoch_mjd: Any,
    covariance: Any,
    origin: str,
    frame: str,
    **_unused: Any,
) -> NativeRustTiming:
    from adam_core import _rust_native
    from adam_core.coordinates.cartesian import CartesianCoordinates
    from adam_core.coordinates.covariances import CoordinateCovariances
    from adam_core.coordinates.origin import Origin
    from adam_core.orbits import Orbits
    from adam_core.orbits.arrow_bridge import orbits_to_record_batch
    from adam_core.time import Timestamp

    values = np.asarray(coords, dtype=np.float64)
    rows = values.shape[0]
    orbits = Orbits.from_kwargs(
        orbit_id=[str(index) for index in range(rows)],
        coordinates=CartesianCoordinates.from_kwargs(
            x=values[:, 0],
            y=values[:, 1],
            z=values[:, 2],
            vx=values[:, 3],
            vy=values[:, 4],
            vz=values[:, 5],
            time=Timestamp.from_mjd(
                np.asarray(epoch_mjd, dtype=np.float64), scale="tdb"
            ),
            origin=Origin.from_kwargs(code=np.full(rows, str(origin), dtype="object")),
            frame=str(frame),
            covariance=CoordinateCovariances.from_matrix(
                np.asarray(covariance, dtype=np.float64)
            ),
        ),
    )
    samples = _rust_native.benchmark_sample_orbit_variants_arrow(
        orbits_to_record_batch(orbits),
        "sigma-point",
        reps,
        trials,
        warmup,
    )
    return NativeRustTiming(
        status="measured",
        sample_trials_s=[[float(value) for value in trial] for trial in samples],
        entrypoint="adam_core_py::coordinates::sample_orbit_variants_record_batch",
        timing_boundary=(
            "Rust std::time::Instant around direct Orbits RecordBatch decode, "
            "sigma-point unscented sampling, and VariantOrbits RecordBatch "
            "assembly; outer Python/PyO3 launch and PyArrow conversion excluded"
        ),
    )


def _photometry_timing(
    samples: Any, *, entrypoint: str, fitting: bool = False
) -> NativeRustTiming:
    semantic_scope = (
        "complete row/group fit setup, internal allocations, and direct fit call"
        if fitting
        else "semantic output allocation and direct photometry kernel call"
    )
    return NativeRustTiming(
        status="measured",
        sample_trials_s=[[float(value) for value in trial] for trial in samples],
        entrypoint=entrypoint,
        timing_boundary=(
            f"Rust std::time::Instant around {semantic_scope}; NumPy extraction, "
            "outer Python/PyO3 launch, and return conversion excluded"
        ),
    )


def _photometry_calculate_phase_angle(
    *, object_pos: Any, observer_pos: Any, reps: int, warmup: int, trials: int
) -> NativeRustTiming:
    from adam_core import _rust_native

    samples = _rust_native.benchmark_calculate_phase_angle_numpy(
        np.ascontiguousarray(object_pos, dtype=np.float64),
        np.ascontiguousarray(observer_pos, dtype=np.float64),
        reps,
        trials,
        warmup,
    )
    return _photometry_timing(
        samples,
        entrypoint="adam_core_rs_coords::calculate_phase_angle_into",
    )


def _photometry_calculate_apparent_magnitude_v(
    *, h_v: Any, object_pos: Any, observer_pos: Any, g: Any,
    reps: int, warmup: int, trials: int
) -> NativeRustTiming:
    from adam_core import _rust_native

    samples = _rust_native.benchmark_calculate_apparent_magnitude_v_numpy(
        np.ascontiguousarray(h_v, dtype=np.float64),
        np.ascontiguousarray(object_pos, dtype=np.float64),
        np.ascontiguousarray(observer_pos, dtype=np.float64),
        np.ascontiguousarray(g, dtype=np.float64),
        reps,
        trials,
        warmup,
    )
    return _photometry_timing(
        samples,
        entrypoint="adam_core_rs_coords::calculate_apparent_magnitude_v_into",
    )


def _photometry_calculate_apparent_magnitude_v_and_phase_angle(
    *, h_v: Any, object_pos: Any, observer_pos: Any, g: Any,
    reps: int, warmup: int, trials: int
) -> NativeRustTiming:
    from adam_core import _rust_native

    samples = (
        _rust_native.benchmark_calculate_apparent_magnitude_v_and_phase_angle_numpy(
            np.ascontiguousarray(h_v, dtype=np.float64),
            np.ascontiguousarray(object_pos, dtype=np.float64),
            np.ascontiguousarray(observer_pos, dtype=np.float64),
            np.ascontiguousarray(g, dtype=np.float64),
            reps,
            trials,
            warmup,
        )
    )
    return _photometry_timing(
        samples,
        entrypoint=(
            "adam_core_rs_coords::"
            "calculate_apparent_magnitude_v_and_phase_angle_into"
        ),
    )


def _photometry_predict_magnitudes(
    *, h_v: Any, object_pos: Any, observer_pos: Any, g: Any,
    target_ids: Any, delta_table: Any, reps: int, warmup: int, trials: int
) -> NativeRustTiming:
    from adam_core import _rust_native

    samples = _rust_native.benchmark_predict_magnitudes_bandpass_numpy(
        np.ascontiguousarray(h_v, dtype=np.float64),
        np.ascontiguousarray(object_pos, dtype=np.float64),
        np.ascontiguousarray(observer_pos, dtype=np.float64),
        np.ascontiguousarray(g, dtype=np.float64),
        np.ascontiguousarray(target_ids, dtype=np.int32),
        np.ascontiguousarray(delta_table, dtype=np.float64),
        reps,
        trials,
        warmup,
    )
    return _photometry_timing(
        samples,
        entrypoint="adam_core_rs_coords::predict_magnitudes_bandpass_into",
    )


def _photometry_fit_absolute_magnitude_rows(
    *, h_rows: Any, sigma_rows: Any, reps: int, warmup: int, trials: int
) -> NativeRustTiming:
    from adam_core import _rust_native

    samples = _rust_native.benchmark_fit_absolute_magnitude_rows_numpy(
        np.ascontiguousarray(h_rows, dtype=np.float64),
        np.ascontiguousarray(sigma_rows, dtype=np.float64),
        reps,
        trials,
        warmup,
    )
    return _photometry_timing(
        samples,
        entrypoint="adam_core_rs_coords::fit_absolute_magnitude_rows",
        fitting=True,
    )


def _photometry_fit_absolute_magnitude_grouped(
    *, h_rows: Any, sigma_rows: Any, group_offsets: Any,
    reps: int, warmup: int, trials: int
) -> NativeRustTiming:
    from adam_core import _rust_native

    samples = _rust_native.benchmark_fit_absolute_magnitude_grouped_numpy(
        np.ascontiguousarray(h_rows, dtype=np.float64),
        np.ascontiguousarray(sigma_rows, dtype=np.float64),
        np.ascontiguousarray(group_offsets, dtype=np.int64),
        reps,
        trials,
        warmup,
    )
    return _photometry_timing(
        samples,
        entrypoint="adam_core_rs_coords::fit_absolute_magnitude_grouped",
        fitting=True,
    )


def _observers_from_codes(
    *,
    codes: Any,
    mjd_utc: Any,
    reps: int,
    warmup: int,
    trials: int,
) -> NativeRustTiming:
    from adam_core._rust.arrow import ensure_spice_backend
    from adam_core.time import Timestamp

    times = Timestamp.from_mjd(np.asarray(mjd_utc, dtype=np.float64), scale="utc")
    code_column = pa.array([str(code) for code in codes], type=pa.large_string())
    time_table = times.table.combine_chunks()
    batch = pa.RecordBatch.from_arrays(
        [
            code_column,
            time_table.column("days").chunk(0),
            time_table.column("nanos").chunk(0),
        ],
        names=["code", "days", "nanos"],
    )
    backend = ensure_spice_backend()
    samples = backend.benchmark_observer_states_from_codes_arrow_rust(
        batch, times.scale, reps, warmup, trials
    )
    return NativeRustTiming(
        status="measured",
        sample_trials_s=[[float(value) for value in trial] for trial in samples],
        entrypoint="adam_core_py::spice::observer_states_from_codes_record_batch",
        timing_boundary=(
            "Rust std::time::Instant around direct RecordBatch->ObserverBatch->"
            "RecordBatch calls; outer Python/PyO3 launch and PyArrow conversion excluded"
        ),
    )


_ADAPTERS: dict[str, Callable[..., NativeRustTiming]] = {
    "coordinates.transform_coordinates": _transform_coordinates,
    "dynamics.calc_mean_motion": _calc_mean_motion,
    "dynamics.tisserand_parameter": _tisserand_parameter,
    "dynamics.calculate_moid": _calculate_moid,
    "dynamics.calculate_moid_batch": _calculate_moid_batch,
    "missions.porkchop_grid": _porkchop_grid,
    "dynamics.propagate_2body_along_arc": _propagate_2body_along_arc,
    "dynamics.propagate_2body_arc_batch": _propagate_2body_arc_batch,
    "dynamics.propagate_2body_with_covariance": _propagate_2body_with_covariance,
    "dynamics.solve_lambert": _solve_lambert,
    "dynamics.add_light_time": _add_light_time,
    "dynamics.propagate_2body": _propagate_2body,
    "dynamics.generate_ephemeris_2body": _generate_ephemeris_2body,
    "dynamics.generate_ephemeris_2body_with_covariance": _generate_ephemeris_2body,
    "dynamics.calculate_perturber_moids": _calculate_perturber_moids,
    "dynamics.generate_porkchop_data": _generate_porkchop_data,
    "orbit_determination.gaussIOD": _gauss_iod,
    "coordinates.residuals.Residuals.calculate": _residuals_calculate,
    "orbits.VariantOrbits.create": _variant_orbits_create,
    "photometry.calculate_phase_angle": _photometry_calculate_phase_angle,
    "photometry.calculate_apparent_magnitude_v": (
        _photometry_calculate_apparent_magnitude_v
    ),
    "photometry.calculate_apparent_magnitude_v_and_phase_angle": (
        _photometry_calculate_apparent_magnitude_v_and_phase_angle
    ),
    "photometry.predict_magnitudes": _photometry_predict_magnitudes,
    "photometry.fit_absolute_magnitude_rows": (
        _photometry_fit_absolute_magnitude_rows
    ),
    "photometry.fit_absolute_magnitude_grouped": (
        _photometry_fit_absolute_magnitude_grouped
    ),
    "observers.Observers.from_codes": _observers_from_codes,
}


def _todo_for(api_id: str) -> str:
    if api_id == "observers.Observers.from_codes":
        return "personal-3gg"
    if (
        api_id.startswith("coordinates.transform")
        or api_id.startswith("coordinates.cartesian")
        or api_id.startswith(
            ("coordinates.keplerian", "coordinates.cometary", "coordinates.spherical")
        )
        or api_id == "coordinates.rotate_cartesian_time_varying"
    ):
        return "personal-98v.1"
    if api_id.startswith("dynamics.propagate_2body"):
        return "personal-98v.1"
    if (
        api_id.startswith("dynamics.generate_ephemeris")
        or api_id == "dynamics.add_light_time"
    ):
        return "personal-cmy.36.5"
    if "moid" in api_id or "porkchop" in api_id or api_id == "dynamics.solve_lambert":
        # calculate_moid / calculate_moid_batch / porkchop_grid stay raw
        # numpy kernels by classification (bead personal-cmy.36.6); their
        # native columns route to the catch-all adapter bead.
        return "personal-98v.1"
    if api_id.startswith("orbit_determination"):
        # calcGibbs / calcHerrickGibbs / calcGauss stay scalar numpy vector
        # kernels by classification (bead personal-cmy.36.7); their native
        # columns route to the catch-all adapter bead.
        return "personal-98v.1"
    if api_id.startswith("photometry"):
        # All photometry surfaces are classified numpy-flat position/fitting
        # kernels (bead personal-cmy.36.8); native columns route to the
        # catch-all adapter bead.
        return "personal-98v.1.3"
    if (
        api_id.startswith(("coordinates.residuals", "statistics."))
        or api_id == "orbits.classify_orbits"
    ):
        # The remaining residuals/statistics helpers are classified numpy-flat
        # array kernels (bead personal-cmy.36.9); native columns route to the
        # catch-all adapter bead.
        return "personal-98v.1"
    return "personal-98v.1"


def measure(
    api_id: str,
    kwargs: dict[str, Any],
    *,
    reps: int,
    warmup: int,
    trials: int,
) -> NativeRustTiming:
    """Return Rust-internal samples or an explicit blank-column reason/TODO."""
    adapter = _ADAPTERS.get(api_id)
    if adapter is None:
        return NativeRustTiming(
            status="unavailable",
            reason=(
                "no Rust-internal Instant benchmark adapter for this surface; "
                "a Python->PyO3 call is not accepted as native-Rust timing"
            ),
            todo=_todo_for(api_id),
        )
    try:
        return adapter(reps=reps, warmup=warmup, trials=trials, **kwargs)
    except Exception as exc:
        return NativeRustTiming(
            status="unavailable",
            reason=f"native-Rust benchmark failed: {type(exc).__name__}: {exc}",
            todo=_todo_for(api_id),
        )
