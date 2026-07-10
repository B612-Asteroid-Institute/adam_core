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
    "dynamics.propagate_2body": _propagate_2body,
    "observers.Observers.from_codes": _observers_from_codes,
}


def _todo_for(api_id: str) -> str:
    if api_id == "observers.Observers.from_codes":
        return "personal-3gg"
    if api_id == "bridge.rotate_orbits_frame":
        return "personal-cmy.36.10"
    if (
        api_id.startswith("coordinates.transform")
        or api_id.startswith("coordinates.cartesian")
        or api_id.startswith(
            ("coordinates.keplerian", "coordinates.cometary", "coordinates.spherical")
        )
        or api_id == "coordinates.rotate_cartesian_time_varying"
    ):
        return "personal-98v.1"
    if api_id == "bridge.propagate_orbits_2body":
        return "personal-cmy.36.10"
    if api_id.startswith("dynamics.propagate_2body"):
        return "personal-98v.1"
    if (
        api_id.startswith("dynamics.generate_ephemeris")
        or api_id == "dynamics.add_light_time"
    ):
        return "personal-cmy.36.5"
    if "moid" in api_id or "porkchop" in api_id or api_id == "dynamics.solve_lambert":
        return "personal-cmy.36.6"
    if api_id.startswith("orbit_determination"):
        return "personal-cmy.36.7"
    if api_id.startswith("photometry"):
        return "personal-cmy.36.8"
    if (
        api_id.startswith(("coordinates.residuals", "statistics."))
        or api_id == "orbits.classify_orbits"
        or api_id == "bridge.evaluate_residuals_2body"
    ):
        return "personal-cmy.36.9"
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
