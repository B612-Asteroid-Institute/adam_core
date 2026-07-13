#!/usr/bin/env python3
"""Build and test adam-core/adam-assist wheels in an isolated runtime.

The driver clones exact Git revisions into a disposable workspace, builds PEP
517 release wheels, creates a runtime virtual environment containing only the
wheels and their published dependencies, and executes the public-API smoke
script from outside both source trees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import venv
from pathlib import Path
from typing import Any, Sequence
from zipfile import ZipFile

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_SCRIPT = REPO_ROOT / "migration" / "scripts" / "clean_room_artifact_smoke.py"
REPORT_SCHEMA_VERSION = 2


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adam-core-repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--adam-core-ref", default="HEAD")
    parser.add_argument("--adam-assist-repo", type=Path, required=True)
    parser.add_argument("--adam-assist-ref", default="HEAD")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--prebuilt-wheelhouse",
        type=Path,
        help=(
            "Use exact wheels already built by a platform-native release builder "
            "(for example a manylinux container) while retaining the same "
            "clone/provenance/runtime/smoke driver."
        ),
    )
    return parser.parse_args()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


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
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - started
    log_path.write_text(completed.stdout)
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


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _clone_exact(
    name: str,
    source: Path,
    ref: str,
    destination: Path,
    log_dir: Path,
) -> dict[str, str]:
    _run(
        f"clone_{name}",
        ["git", "clone", "--no-local", "--no-checkout", source, destination],
        cwd=destination.parent,
        log_dir=log_dir,
    )
    _run(
        f"checkout_{name}",
        ["git", "checkout", "--detach", ref],
        cwd=destination,
        log_dir=log_dir,
    )
    status = _git_output(destination, "status", "--porcelain")
    if status:
        raise AssertionError(f"{name} checkout was not clean:\n{status}")
    return {
        "requested_ref": ref,
        "commit": _git_output(destination, "rev-parse", "HEAD"),
        "tree": _git_output(destination, "rev-parse", "HEAD^{tree}"),
        "source": str(source.resolve()),
        "checkout": str(destination.resolve()),
    }


def _venv_python(root: Path) -> Path:
    if os.name == "nt":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wheel_info(path: Path, native_prefix: str) -> dict[str, Any]:
    with ZipFile(path) as archive:
        names = archive.namelist()
        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        native_names = [name for name in names if name.startswith(native_prefix)]
        if len(metadata_names) != 1:
            raise AssertionError(f"{path.name} has invalid METADATA entries")
        if not native_names:
            raise AssertionError(
                f"{path.name} does not contain {native_prefix} native module"
            )
        metadata_text = archive.read(metadata_names[0]).decode("utf-8")
        metadata: dict[str, str] = {}
        for line in metadata_text.splitlines():
            key, separator, value = line.partition(":")
            if separator and key in {"Name", "Version"}:
                metadata[key.lower()] = value.strip()
        return {
            "filename": path.name,
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "name": metadata.get("name"),
            "version": metadata.get("version"),
            "native_members": native_names,
        }


def _only_wheel(wheelhouse: Path, prefix: str) -> Path:
    matches = sorted(wheelhouse.glob(f"{prefix}-*.whl"))
    if len(matches) != 1:
        raise AssertionError(
            f"expected one {prefix} wheel in {wheelhouse}, found {matches}"
        )
    return matches[0]


def _runtime_environment(
    runtime: Path, home: Path, kernel_cache: Path
) -> dict[str, str]:
    environment = dict(os.environ)
    for name in list(environment):
        if name.startswith("ADAM_CORE_KERNEL_") or name.startswith(
            "ADAM_CORE_RS_ASSIST_"
        ):
            environment.pop(name)
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "ADAM_ASSIST_RUST_REPO",
    ):
        environment.pop(name, None)
    runtime_bin = _venv_python(runtime).parent
    environment.update(
        {
            "HOME": str(home),
            "XDG_CACHE_HOME": str(home / ".cache"),
            "ADAM_CORE_KERNEL_CACHE": str(kernel_cache),
            "ADAM_CORE_KERNEL_OFFLINE": "1",
            "PATH": os.pathsep.join([str(runtime_bin), "/usr/bin", "/bin"]),
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return environment


def main() -> int:
    args = _parse_args()
    workspace = args.workspace.resolve()
    report_path = args.report.resolve()
    if workspace.exists() and any(workspace.iterdir()):
        raise RuntimeError(f"workspace must be absent or empty: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)
    log_dir = workspace / "logs"
    log_dir.mkdir()
    sources = workspace / "sources"
    sources.mkdir()
    wheelhouse = workspace / "wheelhouse"
    wheelhouse.mkdir()
    runtime_home = workspace / "runtime-home"
    runtime_home.mkdir()
    kernel_cache = workspace / "kernel-cache"
    kernel_cache.mkdir()
    invocation_dir = workspace / "invocation"
    invocation_dir.mkdir()

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "running",
        "workspace": str(workspace),
        "builder_python": str(args.python.resolve()),
        "sources": {},
        "wheels": {},
        "runtime": {},
    }
    _write_report(report_path, report)

    try:
        core_source = sources / "adam-core"
        assist_source = sources / "adam-assist"
        report["sources"]["adam_core"] = _clone_exact(
            "adam_core",
            args.adam_core_repo.resolve(),
            args.adam_core_ref,
            core_source,
            log_dir,
        )
        report["sources"]["adam_assist"] = _clone_exact(
            "adam_assist",
            args.adam_assist_repo.resolve(),
            args.adam_assist_ref,
            assist_source,
            log_dir,
        )
        _write_report(report_path, report)

        if args.prebuilt_wheelhouse is not None:
            prebuilt_wheelhouse = args.prebuilt_wheelhouse.resolve()
            prebuilt_wheels = sorted(prebuilt_wheelhouse.glob("*.whl"))
            if not prebuilt_wheels:
                raise AssertionError(
                    f"prebuilt wheelhouse contains no wheels: {prebuilt_wheelhouse}"
                )
            for wheel in prebuilt_wheels:
                shutil.copy2(wheel, wheelhouse / wheel.name)
            report["build_mode"] = "prebuilt-platform-wheels"
            report["prebuilt_wheelhouse"] = str(prebuilt_wheelhouse)
        else:
            builder = workspace / "builder-venv"
            venv.EnvBuilder(with_pip=True, clear=True).create(builder)
            builder_python = _venv_python(builder)
            _run(
                "install_builder",
                [
                    builder_python,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "build",
                    "pip",
                ],
                cwd=workspace,
                log_dir=log_dir,
            )
            _run(
                "write_core_version",
                [builder_python, "migration/scripts/write_maturin_version.py"],
                cwd=core_source,
                log_dir=log_dir,
            )
            _run(
                "build_adam_core",
                [
                    builder_python,
                    "-m",
                    "build",
                    "--wheel",
                    "--outdir",
                    wheelhouse,
                    core_source,
                ],
                cwd=workspace,
                log_dir=log_dir,
            )
            _run(
                "build_adam_assist",
                [
                    builder_python,
                    "-m",
                    "build",
                    "--wheel",
                    "--outdir",
                    wheelhouse,
                    assist_source,
                ],
                cwd=workspace,
                log_dir=log_dir,
            )
            report["build_mode"] = "pep517-source-build"
        core_wheel = _only_wheel(wheelhouse, "adam_core")
        assist_wheel = _only_wheel(wheelhouse, "adam_assist")
        report["wheels"] = {
            "adam_core": _wheel_info(core_wheel, "adam_core/_rust_native"),
            "adam_assist": _wheel_info(assist_wheel, "adam_assist/_native"),
        }
        _write_report(report_path, report)

        runtime = workspace / "runtime-venv"
        venv.EnvBuilder(with_pip=True, clear=True).create(runtime)
        runtime_python = _venv_python(runtime)
        install_environment = dict(os.environ)
        install_environment.update(
            {
                "PIP_ONLY_BINARY": ":all:",
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PYTHONNOUSERSITE": "1",
            }
        )
        install_environment.pop("PYTHONPATH", None)
        _run(
            "upgrade_runtime_pip",
            [runtime_python, "-m", "pip", "install", "--upgrade", "pip"],
            cwd=workspace,
            log_dir=log_dir,
            env=install_environment,
        )
        _run(
            "install_adam_core_wheel",
            [runtime_python, "-m", "pip", "install", core_wheel],
            cwd=workspace,
            log_dir=log_dir,
            env=install_environment,
        )
        _run(
            "install_adam_assist_wheel",
            [runtime_python, "-m", "pip", "install", assist_wheel],
            cwd=workspace,
            log_dir=log_dir,
            env=install_environment,
        )
        pip_check = _run(
            "pip_check",
            [runtime_python, "-m", "pip", "check"],
            cwd=invocation_dir,
            log_dir=log_dir,
        )
        pip_freeze = _run(
            "pip_freeze",
            [runtime_python, "-m", "pip", "freeze", "--all"],
            cwd=invocation_dir,
            log_dir=log_dir,
        )

        copied_smoke = invocation_dir / SMOKE_SCRIPT.name
        shutil.copy2(SMOKE_SCRIPT, copied_smoke)
        smoke_report_path = invocation_dir / "smoke-report.json"
        runtime_environment = _runtime_environment(runtime, runtime_home, kernel_cache)
        smoke = _run(
            "artifact_smoke",
            [
                runtime_python,
                copied_smoke,
                "--report",
                smoke_report_path,
                "--forbid-root",
                core_source,
                "--forbid-root",
                assist_source,
                "--forbid-root",
                args.adam_core_repo.resolve(),
                "--forbid-root",
                args.adam_assist_repo.resolve(),
            ],
            cwd=invocation_dir,
            log_dir=log_dir,
            env=runtime_environment,
        )
        smoke_report = json.loads(smoke_report_path.read_text())
        if smoke_report.get("status") != "passed":
            raise AssertionError("artifact smoke report did not pass")
        report["runtime"] = {
            "python": str(runtime_python.resolve()),
            "pip_check": pip_check.stdout.strip(),
            "pip_freeze": pip_freeze.stdout.splitlines(),
            "smoke": smoke_report,
            "smoke_stdout_tail": "\n".join(smoke.stdout.splitlines()[-20:]),
            "compiler_hidden_path": runtime_environment["PATH"],
        }
        report["status"] = "passed"
        report["failed_stage"] = None
        _write_report(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        report["status"] = "failed"
        report["failed_stage"] = getattr(error, "stage", "driver")
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        if isinstance(error, CommandFailed):
            report["failed_command"] = error.command
            report["failed_log"] = str(error.log_path)
            report["failed_log_tail"] = error.tail
        _write_report(report_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
