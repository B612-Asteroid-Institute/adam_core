"""Audited, fail-closed adam-core release wheel builder."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from .check_wheel_artifacts import inspect_artifact_directory
    from .release_inventory import (
        REPO_ROOT,
        create_release_inventory,
        remove_packaged_inventory,
    )
    from .write_maturin_version import main as write_runtime_version
except ImportError:
    from check_wheel_artifacts import inspect_artifact_directory
    from release_inventory import (
        REPO_ROOT,
        create_release_inventory,
        remove_packaged_inventory,
    )
    from write_maturin_version import main as write_runtime_version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="dist", help="wheel output directory")
    parser.add_argument("maturin_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = list(args.maturin_args)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    output = Path(args.out)
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    write_runtime_version()
    inventory = create_release_inventory()
    env = {**os.environ, **inventory.environment()}
    command = [
        sys.executable,
        "-m",
        "maturin",
        "build",
        "--manifest-path",
        "rust/adam_core_py/Cargo.toml",
        "--release",
        "--out",
        str(output),
        *forwarded,
    ]
    try:
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
        inspect_artifact_directory(output)
    finally:
        remove_packaged_inventory()


if __name__ == "__main__":
    main()
