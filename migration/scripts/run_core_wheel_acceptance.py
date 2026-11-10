#!/usr/bin/env python3
"""Install and accept one prebuilt adam-core wheel in a fresh environment."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import platform
import shutil
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_SCRIPT = REPO_ROOT / "migration" / "scripts" / "core_wheel_smoke.py"


class CommandFailed(RuntimeError):
    def __init__(
        self,
        stage: str,
        command: Sequence[str],
        returncode: int,
        log_path: Path,
        tail: str,
    ) -> None:
        super().__init__(
            f"{stage} failed with exit code {returncode}: {' '.join(command)}\n{tail}"
        )
        self.stage = stage
        self.command = list(command)
        self.returncode = returncode
        self.log_path = log_path
        self.tail = tail


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-python", required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-system", required=True)
    parser.add_argument("--expected-wheel-tag", required=True)
    return parser.parse_args(argv)


def _validate_runner_system(expected: str, actual: str) -> None:
    if actual != expected:
        raise AssertionError(f"runner architecture mismatch: {actual} != {expected}")


def _validate_builder_python(expected: str, actual: str) -> None:
    if actual != expected:
        raise AssertionError(f"builder CPython mismatch: {actual} != {expected}")


def _builder_python_major_minor(python: Path) -> str:
    completed = subprocess.run(
        [
            str(python),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"cannot inspect builder Python {python}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _select_core_wheel(wheelhouse: Path, expected_tag: str) -> Path:
    wheels = sorted(wheelhouse.glob("adam_core-*.whl"))
    if len(wheels) != 1:
        raise AssertionError(
            f"expected one adam_core wheel in {wheelhouse}, found {wheels}"
        )
    wheel = wheels[0]
    pattern = f"*{expected_tag}.whl"
    if not fnmatch.fnmatchcase(wheel.name, pattern):
        raise AssertionError(
            f"wheel tag mismatch: {wheel.name!r} does not match {pattern!r}"
        )
    return wheel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _venv_python(root: Path) -> Path:
    if os.name == "nt":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def _tail(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def _run(
    stage: str,
    command: Sequence[str | Path],
    *,
    cwd: Path,
    log_dir: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    rendered = [str(item) for item in command]
    log_path = log_dir / f"{stage}.log"
    print(f"[{stage}] {' '.join(rendered)}", flush=True)
    started = time.perf_counter()
    completed = subprocess.run(
        rendered,
        cwd=cwd,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - started
    log_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
    print(f"[{stage}] exit={completed.returncode} seconds={elapsed:.2f}", flush=True)
    if completed.returncode:
        raise CommandFailed(
            stage,
            rendered,
            completed.returncode,
            log_path,
            _tail(log_path),
        )
    return completed


def _runtime_environment(runtime: Path) -> dict[str, str]:
    environment = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        environment.pop(name, None)
    environment.update(
        {
            "PATH": os.pathsep.join([str(_venv_python(runtime).parent), os.defpath]),
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_ONLY_BINARY": ":all:",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return environment


def _install_command(python: Path, wheel: Path) -> list[str | Path]:
    return [
        python,
        "-m",
        "pip",
        "install",
        "--only-binary=:all:",
        wheel,
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    actual_system = f"{platform.system()}-{platform.machine()}"
    _validate_runner_system(args.expected_system, actual_system)
    _validate_builder_python(
        args.expected_python, _builder_python_major_minor(args.python)
    )
    source_wheel = _select_core_wheel(
        args.wheelhouse.resolve(), args.expected_wheel_tag
    )
    wheel_sha256 = _sha256(source_wheel)

    workspace = args.workspace.resolve()
    if workspace.exists() and any(workspace.iterdir()):
        raise RuntimeError(f"workspace must be absent or empty: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)
    log_dir = workspace / "logs"
    log_dir.mkdir()
    wheelhouse = workspace / "wheelhouse"
    wheelhouse.mkdir()
    invocation = workspace / "invocation"
    invocation.mkdir()
    runtime = workspace / "runtime-venv"

    accepted_wheel = wheelhouse / source_wheel.name
    shutil.copy2(source_wheel, accepted_wheel)
    if _sha256(accepted_wheel) != wheel_sha256:
        raise AssertionError("accepted wheel copy does not match its source")
    copied_smoke = invocation / SMOKE_SCRIPT.name
    shutil.copy2(SMOKE_SCRIPT, copied_smoke)

    _run(
        "create_runtime_venv",
        [args.python, "-m", "venv", runtime],
        cwd=workspace,
        log_dir=log_dir,
    )
    runtime_python = _venv_python(runtime)
    environment = _runtime_environment(runtime)
    _run(
        "install_adam_core_wheel",
        _install_command(runtime_python, accepted_wheel),
        cwd=invocation,
        log_dir=log_dir,
        env=environment,
    )
    _run(
        "pip_check",
        [runtime_python, "-m", "pip", "check"],
        cwd=invocation,
        log_dir=log_dir,
        env=environment,
    )
    _run(
        "core_wheel_smoke",
        [
            runtime_python,
            copied_smoke,
            "--report",
            args.report.resolve(),
            "--expected-python",
            args.expected_python,
            "--expected-version",
            args.expected_version,
            "--expected-commit",
            args.expected_commit,
            "--wheel-filename",
            accepted_wheel.name,
            "--wheel-sha256",
            wheel_sha256,
        ],
        cwd=invocation,
        log_dir=log_dir,
        env=environment,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
