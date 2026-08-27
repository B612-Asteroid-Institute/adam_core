#!/usr/bin/env python3
"""Upload exact prebuilt ``.crate`` archives without repackaging them.

This is intentionally guarded by ``--execute``. Release-candidate CI creates and
verifies the archives; the publication workflow downloads those exact bytes and
uses the Cargo registry publish protocol directly.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import tarfile
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PUBLICATION_ORDER = (
    "adam_core_rs_autodiff",
    "adam_core_rs_orbit_determination",
    "adam_core_rs_coords",
    "adam_core_rs_spice",
    "adam_core_rs_kernel_data",
    "adam_core",
)
PUBLIC_CRATES = set(PUBLICATION_ORDER)


def cargo_packages(repo: Path) -> dict[str, dict[str, Any]]:
    output = subprocess.check_output(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=repo,
        text=True,
    )
    metadata = json.loads(output)
    return {package["name"]: package for package in metadata["packages"]}


def publish_metadata(package: dict[str, Any]) -> dict[str, Any]:
    manifest_dir = Path(package["manifest_path"]).parent
    readme_path = package.get("readme")
    readme = None
    readme_file = None
    if readme_path:
        path = Path(readme_path)
        readme = path.read_text()
        readme_file = os.path.relpath(path, manifest_dir)

    dependencies = []
    for dependency in package["dependencies"]:
        dependencies.append(
            {
                "name": dependency["name"],
                "version_req": dependency["req"],
                "features": dependency["features"],
                "optional": dependency["optional"],
                "default_features": dependency["uses_default_features"],
                "target": dependency["target"],
                "kind": dependency["kind"] or "normal",
                "registry": dependency["registry"],
                "explicit_name_in_toml": dependency["rename"],
            }
        )
    dependencies.sort(
        key=lambda item: (
            item["kind"],
            item["target"] or "",
            item["explicit_name_in_toml"] or item["name"],
        )
    )

    return {
        "name": package["name"],
        "vers": package["version"],
        "deps": dependencies,
        "features": package["features"],
        "authors": package["authors"],
        "description": package["description"],
        "documentation": package["documentation"],
        "homepage": package["homepage"],
        "readme": readme,
        "readme_file": readme_file,
        "keywords": package["keywords"],
        "categories": package["categories"],
        "license": package["license"],
        "license_file": package["license_file"],
        "repository": package["repository"],
        "badges": {},
        "links": package["links"],
        "rust_version": package["rust_version"],
    }


def publish_body(metadata: dict[str, Any], archive: bytes) -> bytes:
    encoded = json.dumps(metadata, separators=(",", ":")).encode()
    return (
        struct.pack("<I", len(encoded))
        + encoded
        + struct.pack("<I", len(archive))
        + archive
    )


def archive_identity(path: Path) -> tuple[str, str]:
    """Read package identity from the exact normalized manifest in a .crate."""
    with tarfile.open(path, mode="r:gz") as archive:
        manifests = [
            member
            for member in archive.getmembers()
            if member.name.endswith("/Cargo.toml")
        ]
        if len(manifests) != 1:
            raise ValueError(f"{path.name} has {len(manifests)} root Cargo.toml files")
        source = archive.extractfile(manifests[0])
        if source is None:
            raise ValueError(f"could not read Cargo.toml from {path.name}")
        manifest = tomllib.loads(source.read().decode())
    package = manifest.get("package", {})
    name = package.get("name")
    version = package.get("version")
    if not isinstance(name, str) or not isinstance(version, str):
        raise ValueError(f"{path.name} archive manifest has no package identity")
    return name, version


def validate_preview_packages(
    packages: dict[str, dict[str, Any]], expected_version: str | None
) -> None:
    versions = {name: packages[name]["version"] for name in PUBLICATION_ORDER}
    if expected_version is not None:
        mismatched = {
            name: version
            for name, version in versions.items()
            if version != expected_version
        }
        if mismatched:
            raise ValueError(
                f"crate versions do not match {expected_version}: {mismatched}"
            )
    if any("-" not in version for version in versions.values()):
        raise ValueError(f"all publication versions must be prereleases: {versions}")

    for package_name in PUBLICATION_ORDER:
        package = packages[package_name]
        for dependency in package["dependencies"]:
            dependency_name = dependency["name"]
            if dependency_name not in PUBLIC_CRATES:
                continue
            dependency_version = versions[dependency_name]
            expected_requirement = f"={dependency_version}"
            if dependency["req"] != expected_requirement:
                raise ValueError(
                    f"{package_name}->{dependency_name} must use "
                    f"{expected_requirement}, got {dependency['req']}"
                )


def request_json(request: urllib.request.Request, timeout: float = 60.0) -> Any:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(
            f"registry request failed ({error.code}): {detail}"
        ) from error


def wait_for_version(api: str, name: str, version: str) -> None:
    url = f"{api.rstrip('/')}/api/v1/crates/{name}/{version}"
    for attempt in range(60):
        try:
            request_json(
                urllib.request.Request(url, headers={"Accept": "application/json"})
            )
            return
        except RuntimeError as error:
            if "(404)" not in str(error) or attempt == 59:
                raise
        time.sleep(2.0)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--registry-api", default="https://crates.io")
    parser.add_argument("--token-env", default="CARGO_REGISTRY_TOKEN")
    parser.add_argument("--expected-version")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    packages = cargo_packages(args.repo.resolve())
    validate_preview_packages(packages, args.expected_version)
    prepared = []
    for name in PUBLICATION_ORDER:
        package = packages[name]
        archive_path = args.archives / f"{name}-{package['version']}.crate"
        archive_name, archive_version = archive_identity(archive_path)
        if (archive_name, archive_version) != (name, package["version"]):
            raise ValueError(
                f"archive identity mismatch for {archive_path.name}: "
                f"{archive_name} {archive_version}"
            )
        archive = archive_path.read_bytes()
        metadata = publish_metadata(package)
        body = publish_body(metadata, archive)
        prepared.append((name, package["version"], archive_path, body))
        print(f"prepared {archive_path.name}: archive={len(archive)} body={len(body)}")

    if not args.execute:
        print("dry run only; pass --execute to publish")
        return
    token = os.environ.get(args.token_env)
    if not token:
        raise SystemExit(f"{args.token_env} is required with --execute")
    endpoint = f"{args.registry_api.rstrip('/')}/api/v1/crates/new"
    for name, version, archive_path, body in prepared:
        request = urllib.request.Request(
            endpoint,
            data=body,
            method="PUT",
            headers={
                "Accept": "application/json",
                "Authorization": token,
                "Content-Type": "application/octet-stream",
                "User-Agent": "adam-core-release-automation/1",
            },
        )
        request_json(request)
        print(f"published exact archive {archive_path.name}")
        wait_for_version(args.registry_api, name, version)


if __name__ == "__main__":
    main()
