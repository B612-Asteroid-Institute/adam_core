"""Benchmark Python adam-assist against the Rust-backed ASSIST PyO3 shim.

This is the first RM-STANDALONE-007B apples-to-apples benchmark hook for
``adam_assist.ASSISTPropagator.propagate_orbits`` public semantics. It times
Python-callable public propagation for the Python package and the experimental
GPL ``adam_assist_rust`` package over identical quivr orbit/time workloads and
records timing plus residual metadata.
"""

from __future__ import annotations

import argparse
import gc
import json
import hashlib
import platform
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow as pa
from adam_assist import ASSISTPropagator as PythonASSISTPropagator
from adam_assist_rust import ASSISTPropagator as RustASSISTPropagator
from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.coordinates.transform import transform_coordinates
from adam_core.orbits.orbits import Orbits
from adam_core.orbits.variants import VariantOrbits
from adam_core.time import Timestamp
from jpl_small_bodies_de441_n16 import de441_n16
from naif_de440 import de440

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "migration"
    / "artifacts"
    / "assist_public_semantics_benchmark_2026-05-20.json"
)
NANOS_PER_DAY = 86_400_000_000_000
AU_METERS = 149_597_870_700.0
SECONDS_PER_DAY = 86_400.0
PACKAGE_NAMES = (
    "adam-assist",
    "adam-assist-rust",
    "assist",
    "rebound",
    "adam-core",
)
OrbitTable = Orbits | VariantOrbits


@dataclass(frozen=True)
class Workload:
    name: str
    description: str
    orbits: OrbitTable
    times: Timestamp
    chunk_size: int
    max_processes: int


def _package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _kernel_metadata(
    path_value: str, *, label: str, include_sha256: bool
) -> dict[str, Any]:
    path = Path(path_value)
    data: dict[str, Any] = {
        "label": label,
        "file_name": path.name,
        "size_bytes": path.stat().st_size,
    }
    if include_sha256:
        data["sha256"] = _sha256(path)
    return data


def _arrow_to_list(values: Any) -> list[Any]:
    if hasattr(values, "to_pylist"):
        return values.to_pylist()
    return values.to_numpy(zero_copy_only=False).tolist()


def _orbit_metadata(orbits: OrbitTable) -> dict[str, Any]:
    data: dict[str, Any] = {
        "table_type": type(orbits).__name__,
        "rows": len(orbits),
        "frame": orbits.coordinates.frame,
        "time_scale": orbits.coordinates.time.scale,
        "origin_codes": sorted(set(_arrow_to_list(orbits.coordinates.origin.code))),
        "unique_input_epochs": len(orbits.coordinates.time.unique()),
    }
    if isinstance(orbits, VariantOrbits):
        data["variant_rows"] = len(orbits.variant_id)
    return data


def _base_sun_ecliptic_orbits(rows: int, *, mixed_epochs: bool = False) -> Orbits:
    index = np.arange(rows, dtype=np.float64)
    epoch_offsets = (index % 4) * 0.125 if mixed_epochs else np.zeros(rows)
    return Orbits.from_kwargs(
        orbit_id=[f"bench-{i:04d}" for i in range(rows)],
        object_id=[f"bench-{i:04d}" for i in range(rows)],
        coordinates=CartesianCoordinates.from_kwargs(
            x=1.05 + 0.01 * index,
            y=0.02 - 0.003 * index,
            z=0.01 + 0.001 * (index % 3),
            vx=-0.0005 + 0.00002 * index,
            vy=0.0165 - 0.00004 * index,
            vz=0.0002 - 0.00001 * (index % 5),
            time=Timestamp.from_mjd(60000.0 + epoch_offsets, scale="tdb"),
            origin=Origin.from_kwargs(code=pa.repeat("SUN", rows)),
            frame="ecliptic",
        ),
    )


def _as_public_input(
    orbits: Orbits, *, origin_out: str, frame_out: str, time_scale: str
) -> Orbits:
    coordinates = transform_coordinates(
        orbits.coordinates,
        CartesianCoordinates,
        origin_out=origin_out,
        frame_out=frame_out,
    )
    coordinates = coordinates.set_column("time", coordinates.time.rescale(time_scale))
    return orbits.set_column("coordinates", coordinates)


def _variant_orbits(rows: int) -> VariantOrbits:
    base = _base_sun_ecliptic_orbits(rows, mixed_epochs=False)
    values = base.coordinates.values.copy()
    values[:, 0] += np.linspace(0.0, 0.001, rows)
    values[:, 4] -= np.linspace(0.0, 0.0001, rows)
    return VariantOrbits.from_kwargs(
        orbit_id=base.orbit_id,
        object_id=base.object_id,
        variant_id=[f"v{i:04d}" for i in range(rows)],
        weights=np.full(rows, 1.0 / rows),
        weights_cov=np.full(rows, 1.0 / rows),
        coordinates=CartesianCoordinates.from_kwargs(
            x=values[:, 0],
            y=values[:, 1],
            z=values[:, 2],
            vx=values[:, 3],
            vy=values[:, 4],
            vz=values[:, 5],
            time=base.coordinates.time,
            origin=base.coordinates.origin,
            frame=base.coordinates.frame,
        ),
    )


def _workloads() -> list[Workload]:
    return [
        Workload(
            name="sun_ecliptic_tdb_2x2_fixture_shape",
            description="Small public-semantics fixture-shaped SUN/ecliptic/TDB workload.",
            orbits=_base_sun_ecliptic_orbits(2, mixed_epochs=True),
            times=Timestamp.from_mjd([60002.0, 60001.0], scale="tdb"),
            chunk_size=1,
            max_processes=1,
        ),
        Workload(
            name="sun_ecliptic_tdb_8x8_same_epoch",
            description="Eight SUN/ecliptic/TDB orbits sharing one epoch propagated to eight epochs.",
            orbits=_base_sun_ecliptic_orbits(8, mixed_epochs=False),
            times=Timestamp.from_mjd(np.linspace(60000.25, 60004.0, 8), scale="tdb"),
            chunk_size=8,
            max_processes=1,
        ),
        Workload(
            name="ssb_equatorial_utc_8x8_same_epoch",
            description="Eight SSB/equatorial/UTC public-input rows exercising origin/frame/time restoration.",
            orbits=_as_public_input(
                _base_sun_ecliptic_orbits(8, mixed_epochs=False),
                origin_out=OriginCodes.SOLAR_SYSTEM_BARYCENTER,
                frame_out="equatorial",
                time_scale="utc",
            ),
            times=Timestamp.from_mjd(np.linspace(60000.25, 60004.0, 8), scale="utc"),
            chunk_size=8,
            max_processes=1,
        ),
        Workload(
            name="variant_sun_ecliptic_tdb_8x4",
            description="Eight VariantOrbits rows propagated to four TDB epochs.",
            orbits=_variant_orbits(8),
            times=Timestamp.from_mjd(np.linspace(60000.25, 60002.0, 4), scale="tdb"),
            chunk_size=8,
            max_processes=1,
        ),
    ]


def _timed_call(
    function: Callable[[], OrbitTable], *, repeats: int, warmups: int
) -> tuple[list[float], OrbitTable]:
    result = function()
    for _ in range(warmups):
        result = function()
    timings: list[float] = []
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter()
        result = function()
        timings.append(time.perf_counter() - started)
    return timings, result


def _p95(values: list[float]) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95))


def _state_residuals(actual: OrbitTable, expected: OrbitTable) -> dict[str, Any]:
    actual_values = actual.coordinates.values
    expected_values = expected.coordinates.values
    if actual_values.shape != expected_values.shape:
        raise AssertionError(
            f"state shape mismatch: {actual_values.shape} != {expected_values.shape}"
        )
    orbit_ids_actual = _arrow_to_list(actual.orbit_id)
    orbit_ids_expected = _arrow_to_list(expected.orbit_id)
    if orbit_ids_actual != orbit_ids_expected:
        raise AssertionError("orbit_id output order mismatch")
    if isinstance(actual, VariantOrbits) or isinstance(expected, VariantOrbits):
        if not isinstance(actual, VariantOrbits) or not isinstance(
            expected, VariantOrbits
        ):
            raise AssertionError("variant output type mismatch")
        if _arrow_to_list(actual.variant_id) != _arrow_to_list(expected.variant_id):
            raise AssertionError("variant_id output order mismatch")

    delta = np.abs(actual_values - expected_values)
    time_delta_ns = np.abs(
        (
            actual.coordinates.time.days.to_numpy(zero_copy_only=False).astype(np.int64)
            - expected.coordinates.time.days.to_numpy(zero_copy_only=False).astype(
                np.int64
            )
        )
        * NANOS_PER_DAY
        + (
            actual.coordinates.time.nanos.to_numpy(zero_copy_only=False).astype(
                np.int64
            )
            - expected.coordinates.time.nanos.to_numpy(zero_copy_only=False).astype(
                np.int64
            )
        )
    )
    max_position_au = float(delta[:, :3].max(initial=0.0))
    max_velocity_au_per_day = float(delta[:, 3:].max(initial=0.0))
    return {
        "rows": int(actual_values.shape[0]),
        "position_abs_au": max_position_au,
        "position_abs_m": max_position_au * AU_METERS,
        "velocity_abs_au_per_day": max_velocity_au_per_day,
        "velocity_abs_m_per_s": max_velocity_au_per_day * AU_METERS / SECONDS_PER_DAY,
        "time_abs_ns": int(time_delta_ns.max(initial=0)),
    }


def _benchmark_workload(
    workload: Workload,
    *,
    python_propagator: PythonASSISTPropagator,
    rust_propagator: RustASSISTPropagator,
    repeats: int,
    warmups: int,
) -> dict[str, Any]:
    def run_python() -> OrbitTable:
        return python_propagator.propagate_orbits(
            workload.orbits,
            workload.times,
            covariance=False,
            max_processes=workload.max_processes,
            chunk_size=workload.chunk_size,
        )

    def run_rust() -> OrbitTable:
        return rust_propagator.propagate_orbits(
            workload.orbits,
            workload.times,
            covariance=False,
            max_processes=workload.max_processes,
            chunk_size=workload.chunk_size,
        )

    python_timings, python_output = _timed_call(
        run_python, repeats=repeats, warmups=warmups
    )
    rust_timings, rust_output = _timed_call(run_rust, repeats=repeats, warmups=warmups)
    python_p50 = statistics.median(python_timings)
    python_p95 = _p95(python_timings)
    rust_p50 = statistics.median(rust_timings)
    rust_p95 = _p95(rust_timings)
    return {
        "name": workload.name,
        "description": workload.description,
        "input": _orbit_metadata(workload.orbits),
        "target_times": {
            "rows": len(workload.times),
            "scale": workload.times.scale,
            "mjd_min": float(workload.times.mjd().to_numpy(zero_copy_only=False).min()),
            "mjd_max": float(workload.times.mjd().to_numpy(zero_copy_only=False).max()),
        },
        "options": {
            "covariance": False,
            "chunk_size": workload.chunk_size,
            "max_processes": workload.max_processes,
        },
        "timing_seconds": {
            "python": {
                "values": python_timings,
                "p50": python_p50,
                "p95": python_p95,
            },
            "rust": {"values": rust_timings, "p50": rust_p50, "p95": rust_p95},
            "speedup": {
                "p50_python_over_rust": python_p50 / rust_p50,
                "p95_python_over_rust": python_p95 / rust_p95,
            },
        },
        "residuals": _state_residuals(rust_output, python_output),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--skip-kernel-sha256",
        action="store_true",
        help="Skip kernel SHA256 hashing for faster local diagnostics.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.repeats < 3:
        raise ValueError("--repeats must be at least 3 to report p50/p95")
    python_started = time.perf_counter()
    python_propagator = PythonASSISTPropagator()
    python_constructor_seconds = time.perf_counter() - python_started
    rust_started = time.perf_counter()
    rust_propagator = RustASSISTPropagator()
    rust_constructor_seconds = time.perf_counter() - rust_started

    results = [
        _benchmark_workload(
            workload,
            python_propagator=python_propagator,
            rust_propagator=rust_propagator,
            repeats=args.repeats,
            warmups=args.warmups,
        )
        for workload in _workloads()
    ]
    artifact = {
        "schema_version": 1,
        "benchmark_id": "assist_public_semantics_benchmark_2026-05-20",
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "packages": {name: _package_version(name) for name in PACKAGE_NAMES},
        "kernels": [
            _kernel_metadata(
                de440,
                label="naif_de440",
                include_sha256=not args.skip_kernel_sha256,
            ),
            _kernel_metadata(
                de441_n16,
                label="jpl_small_bodies_de441_n16",
                include_sha256=not args.skip_kernel_sha256,
            ),
        ],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "thread_mode": "single-process/single-thread public calls: max_processes=1; Rust thread_limit=1",
            "object_lifecycle": (
                "Propagator objects are constructed once before timed calls. Rust construction loads "
                "assist-rs kernels into AssistData; Python adam-assist loads assist.Ephem inside each propagation call."
            ),
            "constructor_seconds": {
                "python_adam_assist": python_constructor_seconds,
                "rust_adam_assist_rust": rust_constructor_seconds,
            },
        },
        "workloads": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(
        "\n| workload | rows | p50 speedup | p95 speedup | max pos (m) | max vel (m/s) |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    for row in results:
        timing = row["timing_seconds"]
        residuals = row["residuals"]
        print(
            f"| {row['name']} | {residuals['rows']} | "
            f"{timing['speedup']['p50_python_over_rust']:.3f} | "
            f"{timing['speedup']['p95_python_over_rust']:.3f} | "
            f"{residuals['position_abs_m']:.6e} | "
            f"{residuals['velocity_abs_m_per_s']:.6e} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
