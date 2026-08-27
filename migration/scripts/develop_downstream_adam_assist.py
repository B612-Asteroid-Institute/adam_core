"""Install the downstream-owned Rust adam-assist package into this environment.

The GPL backend moved to the adam-assist migration branch (bead personal-yio).
Set ``ADAM_ASSIST_RUST_REPO`` to that checkout; otherwise a sibling
``../adam-assist`` checkout is used. adam-core retains no GPL crate sources.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO = PROJECT_ROOT.parent / "adam-assist"


def main() -> int:
    repo = Path(os.environ.get("ADAM_ASSIST_RUST_REPO", DEFAULT_REPO)).resolve()
    manifest = repo / "rust" / "adam_assist_rs" / "Cargo.toml"
    if not manifest.is_file():
        raise SystemExit(
            "downstream Rust adam-assist checkout not found; set "
            f"ADAM_ASSIST_RUST_REPO (looked for {manifest})"
        )
    maturin = shutil.which("maturin")
    if maturin is None:
        raise SystemExit("maturin is not installed in the active environment")
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if pip_check.returncode:
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
    env = dict(os.environ)
    env.setdefault("CARGO_NET_GIT_FETCH_WITH_CLI", "true")
    return subprocess.call(
        [
            maturin,
            "develop",
            "--release",
            "--features",
            "python,extension-module",
            "--locked",
        ],
        cwd=repo,
        env=env,
    )


if __name__ == "__main__":
    sys.exit(main())
