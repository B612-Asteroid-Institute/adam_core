"""Install downstream adam-assist against the current adam-core checkout.

Set ``ADAM_ASSIST_RUST_REPO`` to an exact candidate checkout; otherwise a
sibling ``../adam-assist`` checkout is used. The build receives a temporary
Cargo patch for this checkout's public Core crates and refuses to replace an
existing downstream Cargo config. Compatible Core patch releases temporarily
refresh only those patched package entries in the dependent lock, then restore
the original lock byte-for-byte after a local develop build. adam-core retains
no GPL crate sources.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
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


@dataclass(frozen=True)
class PreparedDownstream:
    config: Path
    remove_cargo_dir: bool
    lock: Path
    original_lock: bytes


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


def _current_core_version() -> str:
    versions = {}
    for crate in PUBLIC_CORE_CRATES:
        manifest = PROJECT_ROOT / "rust" / crate / "Cargo.toml"
        with manifest.open("rb") as source:
            versions[crate] = tomllib.load(source)["package"]["version"]
    unique = set(versions.values())
    if len(unique) != 1:
        raise SystemExit(f"public Core crate versions are not aligned: {versions}")
    return unique.pop()


def _locked_versions(lock: Path) -> dict[str, set[str]]:
    with lock.open("rb") as source:
        packages = tomllib.load(source).get("package", [])
    result: dict[str, set[str]] = {}
    for package in packages:
        result.setdefault(package["name"], set()).add(package["version"])
    return result


def _refresh_core_lock(repo: Path, env: dict[str, str]) -> None:
    cargo = shutil.which("cargo")
    if cargo is None:
        raise SystemExit("cargo is not installed in the active environment")
    manifest = repo / "rust" / "adam_assist_rs" / "Cargo.toml"
    lock = manifest.with_name("Cargo.lock")
    if not manifest.is_file():
        raise SystemExit(f"downstream Rust adam-assist manifest is missing: {manifest}")
    if not lock.is_file():
        raise SystemExit(f"downstream Rust adam-assist lock is missing: {lock}")

    version = _current_core_version()
    before = _locked_versions(lock)
    for crate in PUBLIC_CORE_CRATES:
        subprocess.check_call(
            [
                cargo,
                "update",
                "--manifest-path",
                str(manifest),
                "--package",
                crate,
                "--precise",
                version,
            ],
            cwd=repo,
            env=env,
        )

    after = _locked_versions(lock)
    selected = {crate: after.get(crate, set()) for crate in PUBLIC_CORE_CRATES}
    expected = {version}
    invalid = {
        crate: values for crate, values in selected.items() if values != expected
    }
    if invalid:
        raise SystemExit(
            f"temporary downstream lock did not select Core {version}: {invalid}"
        )

    # Targeted `cargo update --package` calls may add/remove dependencies required
    # by a patched Core crate, but they must not change a package version that is
    # otherwise present on both sides of the dependent lock.
    core_names = set(PUBLIC_CORE_CRATES)
    unrelated_changes = {
        name: (before[name], after[name])
        for name in (before.keys() & after.keys()) - core_names
        if before[name] != after[name]
    }
    if unrelated_changes:
        raise SystemExit(
            "temporary Core lock refresh changed unrelated package versions: "
            f"{unrelated_changes}"
        )
    print(f"prepared downstream Cargo.lock for Core {version}")


def _prepare_downstream(repo: Path, env: dict[str, str]) -> PreparedDownstream:
    lock = repo / "rust" / "adam_assist_rs" / "Cargo.lock"
    if not lock.is_file():
        raise SystemExit(f"downstream Rust adam-assist lock is missing: {lock}")
    original_lock = lock.read_bytes()
    config, remove_cargo_dir = _write_core_patch(repo)
    prepared = PreparedDownstream(config, remove_cargo_dir, lock, original_lock)
    try:
        _refresh_core_lock(repo, env)
    except BaseException:
        _restore_downstream(prepared)
        raise
    return prepared


def _restore_downstream(prepared: PreparedDownstream) -> None:
    prepared.lock.write_bytes(prepared.original_lock)
    if prepared.lock.read_bytes() != prepared.original_lock:
        raise RuntimeError(f"failed to restore downstream Cargo lock: {prepared.lock}")
    prepared.config.unlink(missing_ok=True)
    if prepared.remove_cargo_dir and prepared.config.parent.exists():
        prepared.config.parent.rmdir()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="leave the temporary patch and refreshed lock for a later locked build",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo = Path(os.environ.get("ADAM_ASSIST_RUST_REPO", DEFAULT_REPO)).resolve()
    manifest = repo / "rust" / "adam_assist_rs" / "Cargo.toml"
    if not manifest.is_file():
        raise SystemExit(
            "downstream Rust adam-assist checkout not found; set "
            f"ADAM_ASSIST_RUST_REPO (looked for {manifest})"
        )
    env = dict(os.environ)
    env.setdefault("CARGO_NET_GIT_FETCH_WITH_CLI", "true")

    if args.prepare_only:
        _prepare_downstream(repo, env)
        return 0

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

    prepared = _prepare_downstream(repo, env)
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
        _restore_downstream(prepared)


if __name__ == "__main__":
    sys.exit(main())
