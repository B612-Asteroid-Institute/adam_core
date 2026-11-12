from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from migration.scripts import core_wheel_smoke, run_core_wheel_acceptance


def _runner_arguments(tmp_path: Path) -> list[str]:
    return [
        "--python",
        "/usr/bin/python3",
        "--wheelhouse",
        str(tmp_path / "wheels"),
        "--workspace",
        str(tmp_path / "acceptance"),
        "--report",
        str(tmp_path / "acceptance-report.json"),
        "--expected-python",
        "3.14",
        "--expected-version",
        "0.8.0",
        "--expected-commit",
        "a" * 40,
        "--expected-system",
        "Linux-aarch64",
        "--expected-wheel-tag",
        "manylinux*aarch64",
    ]


def test_runner_system_mismatch_fails_before_workspace_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_core_wheel_acceptance.platform, "system", lambda: "Linux")
    monkeypatch.setattr(run_core_wheel_acceptance.platform, "machine", lambda: "x86_64")

    with pytest.raises(AssertionError, match="runner architecture mismatch"):
        run_core_wheel_acceptance.main(_runner_arguments(tmp_path))

    assert not (tmp_path / "acceptance").exists()


def test_builder_python_mismatch_fails_before_workspace_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_core_wheel_acceptance.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        run_core_wheel_acceptance.platform, "machine", lambda: "aarch64"
    )
    monkeypatch.setattr(
        run_core_wheel_acceptance,
        "_builder_python_major_minor",
        lambda python: "3.13",
    )

    with pytest.raises(AssertionError, match="builder CPython mismatch"):
        run_core_wheel_acceptance.main(_runner_arguments(tmp_path))

    assert not (tmp_path / "acceptance").exists()


def test_selects_exactly_one_wheel_with_expected_platform_tag(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "adam_core-0.8.0-cp314-cp314-manylinux_2_17_x86_64.whl"
    wheel.touch()
    assert (
        run_core_wheel_acceptance._select_core_wheel(tmp_path, "manylinux*x86_64")
        == wheel
    )

    with pytest.raises(AssertionError, match="wheel tag mismatch"):
        run_core_wheel_acceptance._select_core_wheel(tmp_path, "macosx*arm64")

    (tmp_path / "adam_core-0.8.0-cp313-cp313-manylinux_2_17_x86_64.whl").touch()
    with pytest.raises(AssertionError, match="expected one adam_core wheel"):
        run_core_wheel_acceptance._select_core_wheel(tmp_path, "manylinux*x86_64")


def test_wheel_sha256_is_content_derived(tmp_path: Path) -> None:
    wheel = tmp_path / "adam_core-0.8.0-cp314-cp314-manylinux_x86_64.whl"
    wheel.write_bytes(b"wheel bytes")
    assert (
        run_core_wheel_acceptance._sha256(wheel)
        == hashlib.sha256(b"wheel bytes").hexdigest()
    )


def test_install_is_explicitly_binary_only_without_pip_upgrade() -> None:
    command = run_core_wheel_acceptance._install_command(
        Path("/venv/bin/python"), Path("/wheelhouse/adam_core.whl")
    )
    rendered = [str(part) for part in command]
    assert rendered == [
        "/venv/bin/python",
        "-m",
        "pip",
        "install",
        "--only-binary=:all:",
        "/wheelhouse/adam_core.whl",
    ]
    assert "upgrade" not in rendered


@pytest.mark.parametrize(
    "parser",
    [
        run_core_wheel_acceptance._parse_args,
        core_wheel_smoke._parse_args,
    ],
)
def test_all_cli_arguments_are_required(parser: object) -> None:
    with pytest.raises(SystemExit):
        parser([])  # type: ignore[operator]


def _valid_runtime_identity(site_packages: Path) -> dict[str, object]:
    return {
        "expected_python": "3.14",
        "actual_python": "3.14",
        "expected_version": "0.8.0",
        "distribution_version": "0.8.0",
        "python_package_version": "0.8.0",
        "native_version": "0.8.0",
        "expected_commit": "a" * 40,
        "native_commit": "a" * 40,
        "package_path": site_packages / "adam_core" / "__init__.py",
        "native_path": site_packages / "adam_core" / "_rust_native.so",
        "site_packages": [site_packages],
    }


def test_runtime_identity_requires_installed_paths_python_versions_and_commit(
    tmp_path: Path,
) -> None:
    values = _valid_runtime_identity(tmp_path / "site-packages")
    core_wheel_smoke._validate_runtime_identity(**values)  # type: ignore[arg-type]

    mutations = (
        ("actual_python", "3.13", "expected CPython"),
        ("expected_version", "0.8.1", "versions differ"),
        ("distribution_version", "0.8.1", "versions differ"),
        ("python_package_version", "0.8.1", "versions differ"),
        ("native_version", "0.8.1", "versions differ"),
        ("native_commit", "b" * 40, "native commit mismatch"),
        ("package_path", tmp_path / "source" / "__init__.py", "site-packages"),
        ("native_path", tmp_path / "source" / "_rust_native.so", "site-packages"),
        (
            "native_path",
            tmp_path / "site-packages" / "adam_core" / "_rust_native.py",
            "not a native extension",
        ),
    )
    for key, value, message in mutations:
        invalid = {**values, key: value}
        with pytest.raises(AssertionError, match=message):
            core_wheel_smoke._validate_runtime_identity(  # type: ignore[arg-type]
                **invalid
            )


def test_calculate_c3_smoke_requires_exact_native_result() -> None:
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def calculate_c3(v1: np.ndarray, body_v: np.ndarray) -> np.ndarray:
        calls.append((v1, body_v))
        return np.sum((v1 - body_v) ** 2, axis=1)

    assert core_wheel_smoke._calculate_c3_smoke(calculate_c3) == [3.5]
    assert len(calls) == 1

    with pytest.raises(AssertionError, match="calculate_c3 smoke mismatch"):
        core_wheel_smoke._calculate_c3_smoke(lambda v1, body_v: np.array([0.0]))


def test_smoke_creates_one_self_contained_report(tmp_path: Path) -> None:
    report_path = tmp_path / "acceptance-report.json"
    assert not report_path.exists()
    report = core_wheel_smoke._acceptance_report(
        wheel_filename="adam_core-0.8.0-cp314-cp314-manylinux_x86_64.whl",
        wheel_sha256="f" * 64,
        python_runtime="3.14.4",
        platform_system="Linux-x86_64",
        distribution_version="0.8.0",
        python_package_version="0.8.0",
        native_version="0.8.0",
        native_commit="a" * 40,
        package_path=Path("/venv/site-packages/adam_core/__init__.py"),
        native_path=Path("/venv/site-packages/adam_core/_rust_native.so"),
        site_packages=[Path("/venv/site-packages")],
        c3=[3.5],
    )
    core_wheel_smoke._write_report(report_path, report)

    written = json.loads(report_path.read_text())
    assert written == report
    assert written["status"] == "passed"
    assert written["wheel_sha256"] == "f" * 64
    assert written["platform"] == "Linux-x86_64"
    assert written["c3_au2_per_day2"] == [3.5]
