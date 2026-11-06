"""Build and inspect Core in a fresh environment with declared build tools only."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
EXACT_REQUIREMENT = re.compile(r"^[A-Za-z0-9_.-]+==[^=<>!~]+$")


def declared_build_tools() -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    tools = data.get("dependency-groups", {}).get("build")
    if not isinstance(tools, list) or not tools:
        raise SystemExit(
            "pyproject.toml must declare a nonempty build dependency group"
        )
    if not all(
        isinstance(tool, str) and EXACT_REQUIREMENT.fullmatch(tool) for tool in tools
    ):
        raise SystemExit("every build tool must be a direct exact == pin")
    return tools


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, help="retain the inspected wheel in this directory"
    )
    parser.add_argument("maturin_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = list(args.maturin_args)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]

    tools = declared_build_tools()
    with tempfile.TemporaryDirectory(prefix="adam-core-release-build-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        wheelhouse = args.out or root / "wheelhouse"
        if not wheelhouse.is_absolute():
            wheelhouse = REPO_ROOT / wheelhouse
        if wheelhouse.exists() and any(wheelhouse.iterdir()):
            raise SystemExit(f"fresh build output is not empty: {wheelhouse}")
        subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
        python = environment / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--isolated",
                "--no-deps",
                "--only-binary=:all:",
                *tools,
            ],
            check=True,
        )
        build_environment = os.environ.copy()
        build_environment["PATH"] = (
            str(python.parent) + os.pathsep + build_environment.get("PATH", "")
        )
        subprocess.run(
            [
                str(python),
                str(REPO_ROOT / "migration/scripts/build_release_wheel.py"),
                "--out",
                str(wheelhouse),
                "--",
                *forwarded,
            ],
            cwd=REPO_ROOT,
            env=build_environment,
            check=True,
        )
        subprocess.run(
            [
                str(python),
                str(REPO_ROOT / "migration/scripts/check_wheel_artifacts.py"),
                str(wheelhouse),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        print(
            "Fresh release build environment passed with declared tools: "
            + ", ".join(tools)
        )


if __name__ == "__main__":
    main()
