"""Install downstream adam-assist against the current adam-core checkout.

Set ``ADAM_ASSIST_RUST_REPO`` to an exact candidate checkout; otherwise a
sibling ``../adam-assist`` checkout is used. The build receives a temporary
Cargo patch for this checkout's public Core crates and refuses to replace an
existing downstream Cargo config. Compatible Core source releases temporarily
refresh the patched package entries already selected downstream and admit only
dependency versions selected by Core's lock. When the reviewed downstream
source intentionally has no registry lock yet, a
patched temporary lock is generated instead. A local develop build restores
the original lock byte-for-byte, including restoring initial absence.
adam-core retains no GPL crate sources.
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
    "adam_core",
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
    original_lock: bytes | None


def _downstream_paths(repo: Path) -> tuple[Path, Path]:
    manifest = repo / "rust" / "adam_assist_rs" / "Cargo.toml"
    lock = manifest.with_name("Cargo.lock")
    if not manifest.is_file():
        raise SystemExit(f"downstream Rust adam-assist manifest is missing: {manifest}")
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


def _selected_core_crates(packages: list[dict[str, Any]]) -> tuple[str, ...]:
    package_names = {package.get("name") for package in packages}
    return tuple(crate for crate in PUBLIC_CORE_CRATES if crate in package_names)


def _verify_patched_core_packages(
    packages: list[dict[str, Any]],
    version: str,
    required_crates: Sequence[str] | None = None,
) -> None:
    selected_crates = _selected_core_crates(packages)
    required = (
        tuple(required_crates) if required_crates is not None else selected_crates
    )
    checked_crates = tuple(
        crate
        for crate in PUBLIC_CORE_CRATES
        if crate in required or crate in selected_crates
    )
    invalid: dict[str, list[dict[str, Any]]] = {}
    for crate in checked_crates:
        selected = [package for package in packages if package.get("name") == crate]
        if (
            len(selected) != 1
            or selected[0].get("version") != version
            or "source" in selected[0]
        ):
            invalid[crate] = selected
    if not required or invalid:
        raise SystemExit(
            "temporary downstream lock did not select exactly one local-path "
            f"Core {version} package for every downstream Core crate: {invalid}"
        )


def _reject_unrelated_version_additions(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> None:
    before_versions = _locked_versions(before)
    after_versions = _locked_versions(after)
    # A targeted Core update may replace a shared dependency generation (for
    # example Arrow/Parquet 54 with 58) or add a newly required package. Admit
    # only exact versions already selected by Core's reviewed lock. This keeps
    # the refresh future-proof without allowing an unrelated downstream update.
    core_versions = _locked_versions(_locked_packages(PROJECT_ROOT / "Cargo.lock"))
    additions = {}
    for name in after_versions.keys() - set(PUBLIC_CORE_CRATES):
        unexpected = (
            after_versions[name]
            - before_versions.get(name, set())
            - core_versions.get(name, set())
        )
        if unexpected:
            additions[name] = sorted(unexpected)
    if additions:
        raise SystemExit(
            "temporary Core lock refresh added package versions absent from "
            f"Core's lock: {additions}"
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
    before = _locked_packages(lock) if lock.is_file() else None
    selected_core_crates: tuple[str, ...] | None = None
    if before is None:
        command = [cargo, "generate-lockfile", "--manifest-path", str(manifest)]
    else:
        selected_core_crates = _selected_core_crates(before)
        if not selected_core_crates:
            raise SystemExit(
                f"downstream Cargo lock selects no public Core crates: {lock}"
            )
        command = [cargo, "update", "--manifest-path", str(manifest)]
        for crate in selected_core_crates:
            command.extend(["--package", crate])
    subprocess.check_call(command, cwd=repo, env=env)

    if not lock.is_file():
        raise SystemExit(f"Cargo did not create the temporary downstream lock: {lock}")
    after = _locked_packages(lock)
    _verify_patched_core_packages(after, version, selected_core_crates)

    if before is not None:
        # One Cargo transaction unlocks all Core crates selected by ASSIST. It
        # may remove obsolete versions, but every added non-Core version must be
        # an exact version already selected in Core's reviewed lock.
        _reject_unrelated_version_additions(before, after)
    print(f"prepared downstream Cargo.lock for Core {version}")


def _prepare_downstream(
    repo: Path,
    env: dict[str, str],
    paths: tuple[Path, Path] | None = None,
) -> PreparedDownstream:
    checked_paths = paths or _downstream_paths(repo)
    _, lock = checked_paths
    original_lock = lock.read_bytes() if lock.is_file() else None
    config, remove_cargo_dir = _write_core_patch(repo)
    prepared = PreparedDownstream(config, remove_cargo_dir, lock, original_lock)
    try:
        _refresh_core_lock(repo, env, checked_paths)
    except BaseException:
        _restore_downstream(prepared)
        raise
    return prepared


def _restore_downstream(prepared: PreparedDownstream) -> None:
    if prepared.original_lock is None:
        prepared.lock.unlink(missing_ok=True)
        if prepared.lock.exists():
            raise RuntimeError(
                f"failed to restore absent downstream Cargo lock: {prepared.lock}"
            )
    else:
        prepared.lock.write_bytes(prepared.original_lock)
        if prepared.lock.read_bytes() != prepared.original_lock:
            raise RuntimeError(
                f"failed to restore downstream Cargo lock: {prepared.lock}"
            )
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
