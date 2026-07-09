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
        or api_id == "bridge.rotate_orbits_frame"
    ):
        return "personal-cmy.36.3"
    if (
        api_id.startswith("dynamics.propagate_2body")
        or api_id == "bridge.propagate_orbits_2body"
    ):
        return "personal-cmy.36.4"
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
