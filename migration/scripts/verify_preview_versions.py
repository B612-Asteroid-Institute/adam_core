#!/usr/bin/env python3
"""Verify adam-core Python/Rust prerelease versions before candidate builds."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any

from write_maturin_version import cargo_version_to_pep440

PUBLIC_CRATES = (
    "adam_core_rs_autodiff",
    "adam_core_rs_orbit_determination",
    "adam_core_rs_coords",
    "adam_core_rs_spice",
    "adam_core_rs_kernel_data",
    "adam_core",
)
PUBLIC_CRATE_SET = set(PUBLIC_CRATES)


def _manifest(repo: Path, crate: str) -> dict[str, Any]:
    with (repo / "rust" / crate / "Cargo.toml").open("rb") as source:
        return tomllib.load(source)


def verify(repo: Path, python_version: str, rust_version: str) -> None:
    if not any(marker in python_version for marker in ("a", "b", "rc")):
        raise ValueError(f"Python version is not a prerelease: {python_version}")
    if "-" not in rust_version:
        raise ValueError(f"Rust version is not a prerelease: {rust_version}")

    py_manifest = _manifest(repo, "adam_core_py")
    cargo_python_version = py_manifest["package"]["version"]
    normalized = cargo_version_to_pep440(cargo_python_version)
    if normalized != python_version:
        raise ValueError(
            f"adam_core_py {cargo_python_version} normalizes to {normalized}, "
            f"not {python_version}"
        )

    manifests = {crate: _manifest(repo, crate) for crate in PUBLIC_CRATES}
    mismatched = {
        crate: manifest["package"]["version"]
        for crate, manifest in manifests.items()
        if manifest["package"]["version"] != rust_version
    }
    if mismatched:
        raise ValueError(f"public Rust crate versions do not match: {mismatched}")

    expected_requirement = f"={rust_version}"
    dependency_manifests = {"adam_core_py": py_manifest, **manifests}
    invalid_dependencies: list[str] = []
    for crate, manifest in dependency_manifests.items():
        for name, specification in manifest.get("dependencies", {}).items():
            if name not in PUBLIC_CRATE_SET:
                continue
            if not isinstance(specification, dict):
                invalid_dependencies.append(f"{crate}->{name}: missing table")
                continue
            requirement = specification.get("version")
            if requirement != expected_requirement:
                invalid_dependencies.append(
                    f"{crate}->{name}: {requirement!r} != {expected_requirement!r}"
                )
    if invalid_dependencies:
        raise ValueError(
            "internal prerelease dependencies must be exact:\n  "
            + "\n  ".join(invalid_dependencies)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--rust-version", required=True)
    args = parser.parse_args()
    verify(args.repo.resolve(), args.python_version, args.rust_version)
    print(
        f"verified preview versions: adam-core=={args.python_version}; "
        f"six Rust crates =={args.rust_version} with exact internal pins"
    )


if __name__ == "__main__":
    main()
