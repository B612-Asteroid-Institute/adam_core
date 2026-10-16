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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _downstream_paths(repo: Path) -> tuple[Path, Path]:
    manifest = repo / "rust" / "adam_assist_rs" / "Cargo.toml"
    lock = manifest.with_name("Cargo.lock")
    if not manifest.is_file():
        raise SystemExit(f"downstream Rust adam-assist manifest is missing: {manifest}")
    if not lock.is_file():
        raise SystemExit(f"downstream Rust adam-assist lock is missing: {lock}")
    return manifest, lock


def _write_core_patch(repo: Path) -> tuple[Path, bool]:
    cargo_dir = repo / ".cargo"
    config = cargo_dir / "config.toml"
    if config.exists():
        raise SystemExit(f"refusing to replace downstream Cargo config: {config}")

    paths = {crate: PROJECT_ROOT / "rust" / crate for crate in PUBLIC_CORE_CRATES}
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


def _locked_packages(lock: Path) -> list[dict[str, Any]]:
    with lock.open("rb") as source:
        packages = tomllib.load(source).get("package", [])
    if not isinstance(packages, list):
        raise SystemExit(f"downstream Cargo lock has invalid package data: {lock}")
    return packages


def _locked_versions(packages: list[dict[str, Any]]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for package in packages:
        result.setdefault(package["name"], set()).add(package["version"])
    return result


def _verify_patched_core_packages(packages: list[dict[str, Any]], version: str) -> None:
    invalid: dict[str, list[dict[str, Any]]] = {}
    for crate in PUBLIC_CORE_CRATES:
        selected = [package for package in packages if package.get("name") == crate]
        if (
            len(selected) != 1
            or selected[0].get("version") != version
            or "source" in selected[0]
        ):
            invalid[crate] = selected
    if invalid:
        raise SystemExit(
            "temporary downstream lock did not select exactly one local-path "
            f"Core {version} package per crate: {invalid}"
        )


def _reject_unrelated_version_additions(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> None:
    before_versions = _locked_versions(before)
    after_versions = _locked_versions(after)
    retained_non_core = (before_versions.keys() & after_versions.keys()) - set(
        PUBLIC_CORE_CRATES
    )
    additions = {
        name: sorted(after_versions[name] - before_versions[name])
        for name in retained_non_core
        if after_versions[name] - before_versions[name]
    }
    if additions:
        raise SystemExit(
            "temporary Core lock refresh added unrelated package versions: "
            f"{additions}"
        )


def _refresh_core_lock(
    repo: Path,
    env: dict[str, str],
    paths: tuple[Path, Path] | None = None,
) -> None:
    cargo = shutil.which("cargo")
    if cargo is None:
        raise SystemExit("cargo is not installed in the active environment")
    manifest, lock = paths or _downstream_paths(repo)

    version = _current_core_version()
    before = _locked_packages(lock)
    command = [cargo, "update", "--manifest-path", str(manifest)]
    for crate in PUBLIC_CORE_CRATES:
        command.extend(["--package", crate])
    subprocess.check_call(command, cwd=repo, env=env)

    after = _locked_packages(lock)
    _verify_patched_core_packages(after, version)

    # One Cargo transaction unlocks all five mutually exact-pinned Core crates.
    # It may remove obsolete dependency versions or introduce new dependency
    # names required by Core (for example rawzip), but it may not add a version
    # for any non-Core package name retained from the preliminary lock.
    _reject_unrelated_version_additions(before, after)
    print(f"prepared downstream Cargo.lock for Core {version}")


def _prepare_downstream(
    repo: Path,
    env: dict[str, str],
    paths: tuple[Path, Path] | None = None,
) -> PreparedDownstream:
    checked_paths = paths or _downstream_paths(repo)
    _, lock = checked_paths
    original_lock = lock.read_bytes()
    config, remove_cargo_dir = _write_core_patch(repo)
    prepared = PreparedDownstream(config, remove_cargo_dir, lock, original_lock)
    try:
        _refresh_core_lock(repo, env, checked_paths)
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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="leave the temporary patch and refreshed lock for a later locked build",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    repo = Path(os.environ.get("ADAM_ASSIST_RUST_REPO", DEFAULT_REPO)).resolve()
    paths = _downstream_paths(repo)
    env = dict(os.environ)
    env.setdefault("CARGO_NET_GIT_FETCH_WITH_CLI", "true")

    if args.prepare_only:
        _prepare_downstream(repo, env, paths)
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

    prepared = _prepare_downstream(repo, env, paths)
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
