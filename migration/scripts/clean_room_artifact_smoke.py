#!/usr/bin/env python3
"""Artifact-only public API smoke test for adam-core and adam-assist.

This script is copied outside both source trees and executed by
``run_clean_room_artifact_acceptance.py`` inside a fresh wheel-only virtual
environment.  It intentionally uses only public Python APIs.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import site
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow as pa
import quivr as qv

REPORT_SCHEMA_VERSION = 1
OBJECT_ID = "99942 Apophis"
TARGET_OFFSETS_DAYS = np.array([1.0, 7.0, 30.0], dtype=np.float64)


def _resolved_module_path(module: Any) -> Path:
    module_path = getattr(module, "__file__", None)
    if not module_path:
        raise AssertionError(f"module {module.__name__!r} has no __file__")
    return Path(module_path).resolve()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _finite(values: np.ndarray, label: str) -> None:
    if not np.isfinite(values).all():
        raise AssertionError(f"{label} contains non-finite values")


def _mjd_values(table: Any) -> np.ndarray:
    return np.asarray(
        table.coordinates.time.mjd().to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )


def _cache_files(cache: Path) -> list[str]:
    if not cache.exists():
        return []
    return sorted(
        str(path.relative_to(cache)) for path in cache.rglob("*") if path.is_file()
    )


def _distribution_version(name: str) -> str:
    return importlib.metadata.version(name)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _run_stage(
    report: dict[str, Any],
    report_path: Path,
    name: str,
    operation: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = operation()
    except Exception as error:
        report["status"] = "failed"
        report["failed_stage"] = name
        report["stages"][name] = {
            "status": "failed",
            "seconds": time.perf_counter() - started,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _write_report(report_path, report)
        raise
    report["stages"][name] = {
        "status": "passed",
        "seconds": time.perf_counter() - started,
        **result,
    }
    _write_report(report_path, report)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--forbid-root",
        action="append",
        type=Path,
        default=[],
        help="Source root that imported modules and kernel paths must not use.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report_path = args.report.resolve()
    forbidden_roots = [path.resolve() for path in args.forbid_root]
    runtime_prefix = Path(sys.prefix).resolve()
    kernel_cache_value = os.environ.get("ADAM_CORE_KERNEL_CACHE")
    if not kernel_cache_value:
        raise RuntimeError("ADAM_CORE_KERNEL_CACHE must identify the isolated cache")
    kernel_cache = Path(kernel_cache_value).resolve()

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "running",
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "prefix": str(runtime_prefix),
            "site_packages": site.getsitepackages(),
        },
        "environment": {
            "cwd": str(Path.cwd().resolve()),
            "path": os.environ.get("PATH", ""),
            "pythonpath": os.environ.get("PYTHONPATH"),
            "kernel_offline": os.environ.get("ADAM_CORE_KERNEL_OFFLINE"),
            "kernel_cache": str(kernel_cache),
            "assist_planets_override": os.environ.get(
                "ADAM_CORE_RS_ASSIST_PLANETS_PATH"
            ),
            "assist_asteroids_override": os.environ.get(
                "ADAM_CORE_RS_ASSIST_ASTEROIDS_PATH"
            ),
            "kernel_python_override": os.environ.get("ADAM_CORE_KERNEL_PYTHON"),
            "forbidden_roots": [str(path) for path in forbidden_roots],
        },
        "stages": {},
    }
    _write_report(report_path, report)

    state: dict[str, Any] = {}

    def imports_and_provenance() -> dict[str, Any]:
        import adam_assist
        import adam_assist._native
        import adam_core
        import adam_core._rust_native

        modules = {
            "adam_core": adam_core,
            "adam_core._rust_native": adam_core._rust_native,
            "adam_assist": adam_assist,
            "adam_assist._native": adam_assist._native,
        }
        module_paths = {
            name: _resolved_module_path(module) for name, module in modules.items()
        }
        for name, module_path in module_paths.items():
            if not _is_relative_to(module_path, runtime_prefix):
                raise AssertionError(
                    f"{name} imported outside runtime prefix: {module_path}"
                )
            for root in forbidden_roots:
                if _is_relative_to(module_path, root):
                    raise AssertionError(f"{name} leaked from source root {root}")
        for name in ("adam_core._rust_native", "adam_assist._native"):
            if module_paths[name].suffix not in {".so", ".pyd", ".dylib"}:
                raise AssertionError(f"{name} is not a native extension")
        if os.environ.get("PYTHONPATH"):
            raise AssertionError("PYTHONPATH must be unset in the clean runtime")
        if os.environ.get("ADAM_CORE_KERNEL_OFFLINE") != "1":
            raise AssertionError(
                "strict kernel lane requires ADAM_CORE_KERNEL_OFFLINE=1"
            )
        if os.environ.get("ADAM_CORE_KERNEL_PYTHON") is not None:
            raise AssertionError("kernel Python discovery must not be overridden")
        if os.environ.get("ADAM_CORE_RS_ASSIST_PLANETS_PATH") is not None:
            raise AssertionError("ASSIST planets path must not be overridden")
        if os.environ.get("ADAM_CORE_RS_ASSIST_ASTEROIDS_PATH") is not None:
            raise AssertionError("ASSIST asteroids path must not be overridden")
        if _cache_files(kernel_cache):
            raise AssertionError("isolated kernel cache was not empty at startup")
        return {
            "versions": {
                name: _distribution_version(name)
                for name in (
                    "adam-core",
                    "adam-assist",
                    "naif-de440",
                    "jpl-small-bodies-de441-n16",
                )
            },
            "module_paths": {name: str(path) for name, path in module_paths.items()},
        }

    _run_stage(
        report,
        report_path,
        "imports_and_provenance",
        imports_and_provenance,
    )

    def live_orbit_fetch() -> dict[str, Any]:
        from adam_core.orbits.query.sbdb import query_sbdb_new

        orbits = query_sbdb_new(
            [OBJECT_ID],
            max_concurrent_requests=1,
            timeout_s=60.0,
            max_attempts=5,
            orbit_id_from_input=True,
        )
        if len(orbits) != 1:
            raise AssertionError(f"expected one fetched orbit, received {len(orbits)}")
        if orbits.orbit_id.to_pylist() != [OBJECT_ID]:
            raise AssertionError("SBDB orbit ID did not preserve the requested ID")
        values = np.asarray(orbits.coordinates.values, dtype=np.float64)
        _finite(values, "fetched orbit")
        epoch = float(_mjd_values(orbits)[0])
        if orbits.coordinates.time.scale != "tdb":
            raise AssertionError("SBDB orbit epoch is not TDB")
        if orbits.coordinates.frame != "ecliptic":
            raise AssertionError("SBDB orbit frame is not ecliptic")
        if orbits.coordinates.origin.code.to_pylist() != ["SUN"]:
            raise AssertionError("SBDB orbit origin is not SUN")
        state["orbits"] = orbits
        state["epoch"] = epoch
        return {
            "object_id": orbits.object_id.to_pylist()[0],
            "orbit_id": orbits.orbit_id.to_pylist()[0],
            "epoch_mjd_tdb": epoch,
            "state": values[0].tolist(),
            "schema": str(orbits.table.schema),
        }

    _run_stage(report, report_path, "live_orbit_fetch", live_orbit_fetch)

    def public_propagation() -> dict[str, Any]:
        from adam_assist import ASSISTPropagator
        from adam_core.dynamics import propagate_2body
        from adam_core.time import Timestamp

        orbits = state["orbits"]
        epoch = state["epoch"]
        targets = Timestamp.from_mjd(epoch + TARGET_OFFSETS_DAYS, scale="tdb")
        two_body = propagate_2body(orbits, targets, max_processes=1, chunk_size=1)
        propagator = ASSISTPropagator()
        n_body = propagator.propagate_orbits(
            orbits, targets, covariance=False, max_processes=1, chunk_size=1
        )
        for label, result in (("two-body", two_body), ("n-body", n_body)):
            if len(result) != len(TARGET_OFFSETS_DAYS):
                raise AssertionError(f"{label} output length is incorrect")
            _finite(np.asarray(result.coordinates.values), f"{label} output")
            np.testing.assert_allclose(
                _mjd_values(result), epoch + TARGET_OFFSETS_DAYS, atol=1.0e-12
            )
            if result.orbit_id.to_pylist() != [OBJECT_ID] * len(TARGET_OFFSETS_DAYS):
                raise AssertionError(f"{label} orbit IDs were not preserved")

        position_delta = np.linalg.norm(
            two_body.coordinates.r - n_body.coordinates.r, axis=1
        )
        velocity_delta = np.linalg.norm(
            two_body.coordinates.v - n_body.coordinates.v, axis=1
        )
        if not (position_delta.max() < 1.0e-3 and position_delta.max() > 1.0e-10):
            raise AssertionError(
                "two-body/N-body position separation is not physically sensible"
            )
        if position_delta[-1] <= position_delta[0]:
            raise AssertionError("model separation did not grow across the toy arc")

        initial_values = np.asarray(orbits.coordinates.values, dtype=np.float64)
        return_time = Timestamp.from_mjd([epoch], scale="tdb")
        two_body_back = propagate_2body(two_body[1], return_time)
        n_body_back = propagator.propagate_orbits(
            n_body[1], return_time, max_processes=1, chunk_size=1
        )
        two_body_roundtrip = float(
            np.max(np.abs(two_body_back.coordinates.values - initial_values))
        )
        n_body_roundtrip = float(
            np.max(np.abs(n_body_back.coordinates.values - initial_values))
        )
        if two_body_roundtrip >= 1.0e-10 or n_body_roundtrip >= 1.0e-10:
            raise AssertionError("forward/backward round trip exceeded tolerance")

        planets_path = Path(propagator.planets_path).resolve()
        asteroids_path = Path(propagator.asteroids_path).resolve()
        for label, kernel_path in (
            ("planets", planets_path),
            ("asteroids", asteroids_path),
        ):
            if not kernel_path.is_file():
                raise AssertionError(f"{label} kernel does not exist: {kernel_path}")
            if not _is_relative_to(kernel_path, runtime_prefix):
                raise AssertionError(
                    f"{label} kernel was not resolved from installed packages"
                )
            for root in forbidden_roots:
                if _is_relative_to(kernel_path, root):
                    raise AssertionError(
                        f"{label} kernel leaked from source root {root}"
                    )

        state["propagator"] = propagator
        state["targets"] = targets
        return {
            "target_mjd_tdb": (epoch + TARGET_OFFSETS_DAYS).tolist(),
            "two_body_state": two_body.coordinates.values.tolist(),
            "n_body_state": n_body.coordinates.values.tolist(),
            "model_position_delta_au": position_delta.tolist(),
            "model_velocity_delta_au_per_day": velocity_delta.tolist(),
            "two_body_roundtrip_max_abs": two_body_roundtrip,
            "n_body_roundtrip_max_abs": n_body_roundtrip,
            "kernel_paths": {
                "planets": str(planets_path),
                "asteroids": str(asteroids_path),
            },
        }

    _run_stage(report, report_path, "public_propagation", public_propagation)

    def same_epoch_batch() -> dict[str, Any]:
        from adam_core.time import Timestamp

        orbits = state["orbits"]
        epoch = state["epoch"]
        propagator = state["propagator"]
        candidate_ids = ["candidate-00", "candidate-01", "candidate-02"]
        candidates = qv.concatenate([orbits, orbits, orbits]).set_column(
            "orbit_id", pa.array(candidate_ids, type=pa.large_string())
        )
        result = propagator.propagate_orbits(
            candidates,
            Timestamp.from_mjd([epoch + 7.0], scale="tdb"),
            max_processes=2,
            chunk_size=3,
        )
        if len(result) != len(candidate_ids):
            raise AssertionError("same-epoch batch output length is incorrect")
        if result.orbit_id.to_pylist() != candidate_ids:
            raise AssertionError("same-epoch candidate ordering changed")
        values = np.asarray(result.coordinates.values, dtype=np.float64)
        _finite(values, "same-epoch batch")
        max_span = float(np.max(np.ptp(values, axis=0)))
        if max_span >= 1.0e-13:
            raise AssertionError("identical candidates produced different states")
        return {
            "candidate_ids": candidate_ids,
            "output_rows": len(result),
            "max_identical_candidate_span": max_span,
        }

    _run_stage(report, report_path, "same_epoch_batch", same_epoch_batch)

    def observer_ephemeris() -> dict[str, Any]:
        from adam_core.observers import Observers
        from adam_core.time import Timestamp

        orbits = state["orbits"]
        epoch = state["epoch"]
        propagator = state["propagator"]
        observer_times = Timestamp.from_mjd(epoch + TARGET_OFFSETS_DAYS, scale="utc")
        observers = Observers.from_code("X05", observer_times)
        ephemeris = propagator.generate_ephemeris(
            orbits,
            observers,
            covariance=False,
            max_processes=1,
            chunk_size=1,
        )
        if len(ephemeris) != len(TARGET_OFFSETS_DAYS):
            raise AssertionError("ephemeris output length is incorrect")
        values = np.asarray(ephemeris.coordinates.values, dtype=np.float64)
        light_time = np.asarray(
            ephemeris.light_time.to_numpy(zero_copy_only=False), dtype=np.float64
        )
        _finite(values, "ephemeris coordinates")
        _finite(light_time, "ephemeris light time")
        if not np.all(values[:, 0] > 0.0):
            raise AssertionError("ephemeris range must be positive")
        if not np.all((values[:, 1] >= 0.0) & (values[:, 1] < 360.0)):
            raise AssertionError("ephemeris longitude is outside [0, 360)")
        if not np.all(np.abs(values[:, 2]) <= 90.0):
            raise AssertionError("ephemeris latitude is outside [-90, 90]")
        if not np.all(light_time > 0.0):
            raise AssertionError("ephemeris light time must be positive")
        if ephemeris.orbit_id.to_pylist() != [OBJECT_ID] * len(TARGET_OFFSETS_DAYS):
            raise AssertionError("ephemeris orbit IDs were not preserved")
        return {
            "observatory_code": "X05",
            "output_rows": len(ephemeris),
            "coordinates": values.tolist(),
            "light_time_days": light_time.tolist(),
            "output_time_scale": ephemeris.coordinates.time.scale,
        }

    _run_stage(report, report_path, "observer_ephemeris", observer_ephemeris)

    def kernel_cache_policy() -> dict[str, Any]:
        cache_files = _cache_files(kernel_cache)
        if cache_files:
            raise AssertionError(
                "Python-hosted execution duplicated installed kernels into cache: "
                + ", ".join(cache_files)
            )
        return {"cache_files": cache_files, "installed_package_paths_used": True}

    _run_stage(report, report_path, "kernel_cache_policy", kernel_cache_policy)
    report["status"] = "passed"
    report["failed_stage"] = None
    report["total_seconds"] = sum(
        float(stage["seconds"]) for stage in report["stages"].values()
    )
    _write_report(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"clean-room smoke failed: {type(error).__name__}: {error}", file=sys.stderr
        )
        raise
