#!/usr/bin/env python3
"""Validate one installed adam-core wheel and write its acceptance report."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import site
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-python", required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--wheel-filename", required=True)
    parser.add_argument("--wheel-sha256", required=True)
    return parser.parse_args(argv)


def _path_is_under(path: Path, roots: Sequence[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved.is_relative_to(root.resolve()) for root in roots)


def _validate_runtime_identity(
    *,
    expected_python: str,
    actual_python: str,
    expected_version: str,
    distribution_version: str,
    python_package_version: str,
    native_version: str,
    expected_commit: str,
    native_commit: str,
    package_path: Path,
    native_path: Path,
    site_packages: Sequence[Path],
) -> None:
    if actual_python != expected_python:
        raise AssertionError(
            f"expected CPython {expected_python}, got CPython {actual_python}"
        )
    versions = {
        expected_version,
        distribution_version,
        python_package_version,
        native_version,
    }
    if len(versions) != 1:
        raise AssertionError(
            "Core expected/distribution/Python/native versions differ: "
            f"{sorted(versions)!r}"
        )
    if native_commit != expected_commit:
        raise AssertionError(
            f"native commit mismatch: {native_commit!r} != {expected_commit!r}"
        )
    if not _path_is_under(package_path, site_packages):
        raise AssertionError(
            f"adam_core package is not installed under site-packages: {package_path}"
        )
    if not _path_is_under(native_path, site_packages):
        raise AssertionError(
            f"native module is not installed under site-packages: {native_path}"
        )
    if native_path.suffix not in {".so", ".dylib", ".pyd"}:
        raise AssertionError(f"not a native extension: {native_path}")


def _calculate_c3_smoke(calculate_c3: Callable[[Any, Any], Any]) -> list[float]:
    import numpy as np

    asteroid_velocity = np.array([[2.5, -1.0, 0.5]], dtype=np.float64)
    body_velocity = np.array([[1.5, -0.5, -1.0]], dtype=np.float64)
    result = np.asarray(calculate_c3(asteroid_velocity, body_velocity)).tolist()
    if result != [3.5]:
        raise AssertionError(f"calculate_c3 smoke mismatch: {result!r}")
    return result


def _acceptance_report(
    *,
    wheel_filename: str,
    wheel_sha256: str,
    python_runtime: str,
    platform_system: str,
    distribution_version: str,
    python_package_version: str,
    native_version: str,
    native_commit: str,
    package_path: Path,
    native_path: Path,
    site_packages: Sequence[Path],
    c3: list[float],
) -> dict[str, object]:
    return {
        "status": "passed",
        "python_runtime": python_runtime,
        "platform": platform_system,
        "wheel_filename": wheel_filename,
        "wheel_sha256": wheel_sha256,
        "distribution_version": distribution_version,
        "python_package_version": python_package_version,
        "native_version": native_version,
        "native_commit": native_commit,
        "package_path": str(package_path),
        "native_path": str(native_path),
        "site_packages": [str(path) for path in site_packages],
        "c3_au2_per_day2": c3,
    }


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    import adam_core
    from adam_core import _rust_native
    from adam_core.dynamics.lambert import calculate_c3

    distribution_version = importlib.metadata.version("adam-core")
    native_version, native_commit = _rust_native.build_identity()
    package_path = Path(adam_core.__file__).resolve()
    native_path = Path(_rust_native.__file__).resolve()
    site_packages = [Path(path).resolve() for path in site.getsitepackages()]
    actual_python = ".".join(platform.python_version_tuple()[:2])
    actual_platform = f"{platform.system()}-{platform.machine()}"

    _validate_runtime_identity(
        expected_python=args.expected_python,
        actual_python=actual_python,
        expected_version=args.expected_version,
        distribution_version=distribution_version,
        python_package_version=adam_core.__version__,
        native_version=native_version,
        expected_commit=args.expected_commit,
        native_commit=native_commit,
        package_path=package_path,
        native_path=native_path,
        site_packages=site_packages,
    )
    c3 = _calculate_c3_smoke(calculate_c3)
    report = _acceptance_report(
        wheel_filename=args.wheel_filename,
        wheel_sha256=args.wheel_sha256,
        python_runtime=platform.python_version(),
        platform_system=actual_platform,
        distribution_version=distribution_version,
        python_package_version=adam_core.__version__,
        native_version=native_version,
        native_commit=native_commit,
        package_path=package_path,
        native_path=native_path,
        site_packages=site_packages,
        c3=c3,
    )
    _write_report(args.report.resolve(), report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
