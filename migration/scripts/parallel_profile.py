"""Confirm Ray adds no value on top of already-Rayon-parallel Rust kernels.

This is the dynamics-only slice of RM-WD3-001 step 2. Surfaces that touch
the n-body propagation line (anything backed by ASSIST: ``Propagator``
wrapper paths, ``dynamics.impacts``, ``orbit_determination.od``/``iod``)
are deliberately out of scope here — those decisions are deferred until
the n-body line is itself migrated to Rust traits backed by ``assist-rs``
(see RM-FUTURE-002).

The four surfaces measured all delegate (directly or via a tiny harness
propagator) to Rust kernels that are already Rayon-parallel. Adding Ray
on top means N Python processes each launching a Rust kernel that
itself wants every core — parallelism on parallelism. The numbers in
the artifact confirm what the structural argument predicts.

Usage:
    pdm run python migration/scripts/parallel_profile.py \
        [--surfaces propagate_2body generate_ephemeris_2body \
         propagator.propagate_orbits propagator.generate_ephemeris]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pyarrow as pa
import quivr as qv
import ray

from adam_core.dynamics.ephemeris import generate_ephemeris_2body
from adam_core.dynamics.propagation import propagate_2body
from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_core.propagator.propagator import EphemerisMixin, Propagator
from adam_core.time import Timestamp
from adam_core.utils.helpers.orbits import make_real_orbits

MAX_PROCESSES = (1, 4, 8)
DEFAULT_SURFACES = (
    "propagate_2body",
    "generate_ephemeris_2body",
    "propagator.propagate_orbits",
    "propagator.generate_ephemeris",
)


@dataclass(frozen=True)
class Run:
    surface: str
    max_processes: int
    cold_seconds: float
    warm_median_seconds: Optional[float]
    warm_min_seconds: Optional[float]
    warm_max_seconds: Optional[float]
    n_warm_reps: int
    workload: dict[str, int | str | float]


@dataclass
class Report:
    started_at: str
    cpu_count: int
    runs: List[Run] = field(default_factory=list)


class _HarnessPropagator(Propagator, EphemerisMixin):
    """Minimal Propagator subclass for profiling the wrapper surfaces.

    Delegates to ``propagate_2body`` (and inherits the EphemerisMixin
    default ``_generate_ephemeris``) with inner ``max_processes=1`` so
    the only Ray dispatch under measurement is at the wrapper level.
    Crucially, this delegates to a Rust kernel that does *not* touch the
    n-body propagation line.
    """

    def __getstate__(self) -> dict:
        return self.__dict__.copy()

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def _propagate_orbits(self, orbits: Orbits, times: Timestamp) -> Orbits:
        return propagate_2body(orbits, times, max_processes=1)


def _replicate_orbits(orbits: Orbits, copies: int) -> Orbits:
    if copies <= 1:
        return orbits
    parts: list[Orbits] = []
    base_ids = orbits.orbit_id.to_pylist()
    base_object_ids = orbits.object_id.to_pylist()
    for k in range(copies):
        new_ids = pa.array([f"{oid}_{k}" for oid in base_ids], type=pa.large_string())
        new_object_ids = pa.array(
            [f"{oid}_{k}" for oid in base_object_ids], type=pa.large_string()
        )
        parts.append(
            orbits.set_column("orbit_id", new_ids).set_column(
                "object_id", new_object_ids
            )
        )
    return qv.concatenate(parts)


def _build_propagation_workload() -> tuple[Orbits, Timestamp]:
    orbits = _replicate_orbits(make_real_orbits(27), copies=37)  # 999 orbits
    base_mjd = float(
        np.median(orbits.coordinates.time.mjd().to_numpy(zero_copy_only=False))
    )
    times = Timestamp.from_mjd(
        base_mjd + np.arange(0, 100, dtype=np.float64), scale="tdb"
    )
    return orbits, times


def _build_wrapper_ephemeris_workload() -> tuple[Orbits, Observers]:
    """Inputs for ``Propagator.generate_ephemeris`` (raw orbits + observers)."""
    orbits = _replicate_orbits(make_real_orbits(27), copies=8)  # 216 orbits
    base_mjd = float(
        np.median(orbits.coordinates.time.mjd().to_numpy(zero_copy_only=False))
    )
    n_obs = 60
    times = Timestamp.from_mjd(
        base_mjd + np.arange(0, n_obs, dtype=np.float64), scale="tdb"
    )
    codes = pa.array(["X05"] * n_obs)
    observers = Observers.from_codes(times=times, codes=codes)
    return orbits, observers


def _build_dynamics_ephemeris_workload() -> tuple[Orbits, Observers]:
    """Inputs for ``generate_ephemeris_2body`` (paired propagated rows)."""
    orbits, observers = _build_wrapper_ephemeris_workload()
    propagated = propagate_2body(orbits, observers.coordinates.time, max_processes=1)
    observers_tiled = qv.concatenate([observers] * len(orbits))
    assert len(propagated) == len(observers_tiled)
    return propagated, observers_tiled


def _ensure_ray(num_cpus: int) -> None:
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=num_cpus, include_dashboard=False, log_to_driver=False)


def _shutdown_ray() -> None:
    if ray.is_initialized():
        ray.shutdown()


def _time_call(fn: Callable[[], object]) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def _record_runs(
    surface: str,
    workload: dict[str, int | str | float],
    runner: Callable[[int], Callable[[], object]],
    *,
    n_warm_reps: int = 2,
) -> list[Run]:
    runs: list[Run] = []
    for mp_value in MAX_PROCESSES:
        if mp_value > 1:
            _ensure_ray(mp_value)
        else:
            _shutdown_ray()
        try:
            cold = _time_call(runner(mp_value))
            warm = [_time_call(runner(mp_value)) for _ in range(n_warm_reps)]
        finally:
            _shutdown_ray()

        runs.append(
            Run(
                surface=surface,
                max_processes=mp_value,
                cold_seconds=cold,
                warm_median_seconds=statistics.median(warm) if warm else None,
                warm_min_seconds=min(warm) if warm else None,
                warm_max_seconds=max(warm) if warm else None,
                n_warm_reps=len(warm),
                workload=workload,
            )
        )
        run = runs[-1]
        print(
            f"  mp={run.max_processes:>2}  cold={run.cold_seconds:7.3f}s"
            f"  warm_median={(run.warm_median_seconds or 0):7.3f}s"
            f"  warm_range=[{(run.warm_min_seconds or 0):.3f}"
            f", {(run.warm_max_seconds or 0):.3f}]s"
        )
    return runs


def _profile_propagate_2body() -> list[Run]:
    orbits, times = _build_propagation_workload()
    workload = {"n_orbits": len(orbits), "n_times": len(times)}
    print(f"== propagate_2body  workload={workload}")

    def runner(mp_value: int) -> Callable[[], object]:
        return lambda: propagate_2body(orbits, times, max_processes=mp_value)

    return _record_runs("propagate_2body", workload, runner, n_warm_reps=2)


def _profile_generate_ephemeris_2body() -> list[Run]:
    orbits, observers = _build_dynamics_ephemeris_workload()
    workload = {"n_paired_rows": len(orbits)}
    print(f"== generate_ephemeris_2body  workload={workload}")

    def runner(mp_value: int) -> Callable[[], object]:
        return lambda: generate_ephemeris_2body(
            orbits, observers, max_processes=mp_value
        )

    return _record_runs("generate_ephemeris_2body", workload, runner, n_warm_reps=2)


def _profile_propagator_propagate_orbits() -> list[Run]:
    orbits, times = _build_propagation_workload()
    workload = {
        "n_orbits": len(orbits),
        "n_times": len(times),
        "propagator": "_HarnessPropagator(2body)",
    }
    print(f"== propagator.propagate_orbits  workload={workload}")

    def runner(mp_value: int) -> Callable[[], object]:
        prop = _HarnessPropagator()
        return lambda: prop.propagate_orbits(orbits, times, max_processes=mp_value)

    return _record_runs("propagator.propagate_orbits", workload, runner, n_warm_reps=2)


def _profile_propagator_generate_ephemeris() -> list[Run]:
    orbits, observers = _build_wrapper_ephemeris_workload()
    workload = {
        "n_orbits": len(orbits),
        "n_observers": len(observers),
        "propagator": "_HarnessPropagator(2body)",
    }
    print(f"== propagator.generate_ephemeris  workload={workload}")

    def runner(mp_value: int) -> Callable[[], object]:
        prop = _HarnessPropagator()
        return lambda: prop.generate_ephemeris(
            orbits,
            observers,
            max_processes=mp_value,
            predict_magnitudes=False,
            predict_phase_angle=False,
        )

    return _record_runs(
        "propagator.generate_ephemeris", workload, runner, n_warm_reps=2
    )


SURFACE_RUNNERS: dict[str, Callable[[], list[Run]]] = {
    "propagate_2body": _profile_propagate_2body,
    "generate_ephemeris_2body": _profile_generate_ephemeris_2body,
    "propagator.propagate_orbits": _profile_propagator_propagate_orbits,
    "propagator.generate_ephemeris": _profile_propagator_generate_ephemeris,
}


def _format_seconds(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if value < 1e-3:
        return f"{value * 1e6:.0f}µs"
    if value < 1.0:
        return f"{value * 1e3:.1f}ms"
    return f"{value:.3f}s"


def _render_markdown(report: Report) -> str:
    lines: list[str] = [
        "# Parallel Backend Profile (RM-WD3-001 step 2 — dynamics-only)",
        "",
        f"- started_at: {report.started_at}",
        f"- cpu_count: {report.cpu_count}",
        "",
        "Surfaces that touch the n-body line (ASSIST-backed Propagator,"
        " impacts, OD, IOD) are deliberately out of scope here. Their Ray"
        " defaults are deferred until the n-body line itself is migrated"
        " to Rust (RM-FUTURE-002).",
        "",
    ]
    by_surface: dict[str, list[Run]] = {}
    for run in report.runs:
        by_surface.setdefault(run.surface, []).append(run)
    for surface, runs in by_surface.items():
        lines.append(f"## {surface}")
        if runs:
            workload = runs[0].workload
            lines.append(
                "- workload: " + ", ".join(f"{k}={v}" for k, v in workload.items())
            )
        lines.append("")
        lines.append(
            "| max_processes | cold | warm_median | warm_min | warm_max | n_warm |"
        )
        lines.append("|---|---|---|---|---|---|")
        baseline_warm = next(
            (r.warm_median_seconds for r in runs if r.max_processes == 1), None
        )
        for run in runs:
            ratio = ""
            if (
                baseline_warm is not None
                and run.warm_median_seconds is not None
                and run.warm_median_seconds > 0
            ):
                ratio = f"  (seq/this {baseline_warm / run.warm_median_seconds:.2f}x)"
            lines.append(
                f"| {run.max_processes} "
                f"| {_format_seconds(run.cold_seconds)} "
                f"| {_format_seconds(run.warm_median_seconds)}{ratio} "
                f"| {_format_seconds(run.warm_min_seconds)} "
                f"| {_format_seconds(run.warm_max_seconds)} "
                f"| {run.n_warm_reps} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    import multiprocessing

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--surfaces",
        nargs="*",
        choices=sorted(SURFACE_RUNNERS),
        default=list(DEFAULT_SURFACES),
        help="Subset of surfaces to profile.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("migration/artifacts/parallel_profile.json"),
        help="Path to write JSON results.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("migration/artifacts/parallel_profile.md"),
        help="Path to write a Markdown summary.",
    )
    args = parser.parse_args(argv)

    report = Report(
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        cpu_count=multiprocessing.cpu_count(),
    )

    for surface in args.surfaces:
        runner = SURFACE_RUNNERS[surface]
        try:
            report.runs.extend(runner())
        finally:
            _shutdown_ray()

    payload = {
        "started_at": report.started_at,
        "cpu_count": report.cpu_count,
        "runs": [asdict(run) for run in report.runs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.output}")

    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_render_markdown(report) + "\n")
    print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
