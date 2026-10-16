"""Install downstream adam-assist against the current adam-core checkout.

Set ``ADAM_ASSIST_RUST_REPO`` to an exact candidate checkout; otherwise a
sibling ``../adam-assist`` checkout is used. The build receives a temporary
Cargo patch for this checkout's public Core crates and refuses to replace an
existing downstream Cargo config. adam-core retains no GPL crate sources.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO = PROJECT_ROOT.parent / "adam-assist"
PUBLIC_CORE_CRATES = (
    "adam_core_rs_autodiff",
    "adam_core_rs_coords",
    "adam_core_rs_kernel_data",
    "adam_core_rs_orbit_determination",
    "adam_core_rs_spice",
)


def _write_core_patch(repo: Path) -> tuple[Path, bool]:
    cargo_dir = repo / ".cargo"
    config = cargo_dir / "config.toml"
    if config.exists():
        raise SystemExit(f"refusing to replace downstream Cargo config: {config}")

    paths = {
        crate: PROJECT_ROOT / "rust" / crate for crate in PUBLIC_CORE_CRATES
    }
    missing = [str(path) for path in paths.values() if not path.is_dir()]
    if missing:
        raise SystemExit(f"adam-core crate paths are missing: {missing}")

    created_dir = not cargo_dir.exists()
    cargo_dir.mkdir(exist_ok=True)
    lines = ["[patch.crates-io]"]
    lines.extend(
        f"{crate} = {{ path = {json.dumps(str(path))} }}"
        for crate, path in paths.items()
    )
    config.write_text("\n".join(lines) + "\n")
    return config, created_dir


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
    config, remove_cargo_dir = _write_core_patch(repo)
    try:
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
    finally:
        config.unlink(missing_ok=True)
        if remove_cargo_dir:
            config.parent.rmdir()


if __name__ == "__main__":
    sys.exit(main())
