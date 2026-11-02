"""Validate adam-core wheel artifacts before upload or clean-install smoke tests."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from email.parser import Parser
from email.utils import collapse_rfc2231_value
from pathlib import Path
from zipfile import BadZipFile, ZipFile

NATIVE_MODULE_PREFIX = "adam_core/_rust_native"
VERSION_MODULE = "adam_core/_version.py"
RELEASE_INVENTORY = "adam_core/_release_inventory.json"
RELEASE_INVENTORY_FORMAT = "adam-core-release-inventory-v2"
FORBIDDEN_RUNTIME_DEPENDENCIES = {"pdm"}

# The release wrapper produces CPython ABI-specific wheels without a build tag.
# Validate that exact PEP 427 subset using only the standard library so the
# fresh release matrix can import the builder before optional tooling exists.
_WHEEL_FILENAME = re.compile(
    r"^(?P<distribution>[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)-"
    r"(?P<version>[0-9]+(?:\.[0-9]+)+(?:a[0-9]+|b[0-9]+|rc[0-9]+)?)-"
    r"(?P<python>cp[0-9]+)-(?P<abi>cp[0-9]+)-"
    r"(?P<platform>[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*)\.whl$"
)


def _canonicalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _validated_wheel_filename(wheel: Path) -> tuple[str, str]:
    """Return canonical distribution/version from our PEP 427 wheel subset."""

    match = _WHEEL_FILENAME.fullmatch(wheel.name)
    if match is None or match["python"] != match["abi"]:
        raise SystemExit(
            f"{wheel} has an invalid wheel filename: expected "
            "<distribution>-<version>-cp<python>-cp<python>-<platform>.whl"
        )
    return _canonicalize_name(match["distribution"]), match["version"]


def _wheel_metadata(archive: ZipFile, wheel: Path) -> tuple[str, str, list[str]]:
    metadata_files = [
        name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
    ]
    if len(metadata_files) != 1:
        raise SystemExit(
            f"{wheel} should contain exactly one METADATA file, "
            f"found {len(metadata_files)}"
        )
    metadata = Parser().parsestr(
        archive.read(metadata_files[0]).decode("utf-8", errors="replace")
    )
    name = metadata.get("Name")
    version = metadata.get("Version")
    if name is None or version is None:
        raise SystemExit(f"{wheel} METADATA is missing Name or Version")
    requirements = metadata.get_all("Requires-Dist", [])
    return name, version, requirements


def _runtime_version(archive: ZipFile, wheel: Path) -> str:
    if VERSION_MODULE not in archive.namelist():
        raise SystemExit(f"{wheel} is missing {VERSION_MODULE}")
    namespace: dict[str, str] = {}
    exec(archive.read(VERSION_MODULE), namespace)
    version = namespace.get("__version__")
    if not isinstance(version, str):
        raise SystemExit(f"{wheel} {VERSION_MODULE} does not define __version__")
    return version


def _verify_release_inventory(archive: ZipFile, wheel: Path) -> None:
    if RELEASE_INVENTORY not in archive.namelist():
        raise SystemExit(f"{wheel} is missing {RELEASE_INVENTORY}")
    try:
        inventory = json.loads(archive.read(RELEASE_INVENTORY))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise SystemExit(
            f"{wheel} has an invalid release inventory: {error}"
        ) from error
    if inventory.get("format") != RELEASE_INVENTORY_FORMAT:
        raise SystemExit(f"{wheel} has an unsupported release inventory format")
    for field, size in (("commit", 40), ("digest", 64)):
        value = inventory.get(field)
        if (
            not isinstance(value, str)
            or len(value) != size
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise SystemExit(f"{wheel} release inventory has invalid {field}")
    expected = inventory.get("python_files")
    if not isinstance(expected, dict) or not all(
        isinstance(name, str) and isinstance(digest, str) and len(digest) == 64
        for name, digest in expected.items()
    ):
        raise SystemExit(f"{wheel} release inventory has invalid python_files")
    actual = {
        name
        for name in archive.namelist()
        if name.startswith("adam_core/")
        and name.endswith(".py")
        and "/tests/" not in name
        and not Path(name).name.startswith("test_")
    }
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        unexpected = sorted(actual - set(expected))
        raise SystemExit(
            f"{wheel} packaged Python file set does not match release inventory; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name, expected_digest in expected.items():
        actual_digest = hashlib.sha256(archive.read(name)).hexdigest()
        if actual_digest != expected_digest:
            raise SystemExit(
                f"{wheel} packaged Python file {name} does not match its attested digest"
            )


def _inspect_wheel(wheel: Path) -> str:
    filename_name, filename_version = _validated_wheel_filename(wheel)
    try:
        with ZipFile(wheel) as archive:
            name, metadata_version, requirements = _wheel_metadata(archive, wheel)
            if _canonicalize_name(name) != filename_name:
                raise SystemExit(
                    f"{wheel} filename distribution {filename_name!r} does not match "
                    f"METADATA name {name!r}"
                )
            if metadata_version != filename_version:
                raise SystemExit(
                    f"{wheel} filename version {filename_version!r} does not match "
                    f"METADATA version {metadata_version!r}"
                )
            runtime_version = _runtime_version(archive, wheel)
            _verify_release_inventory(archive, wheel)
            if name not in {"adam_core", "adam-core"}:
                raise SystemExit(f"{wheel} has unexpected package name {name!r}")
            if runtime_version != metadata_version:
                raise SystemExit(
                    f"{wheel} runtime version {runtime_version!r} does not match "
                    f"METADATA version {metadata_version!r}"
                )
            runtime_names = {
                requirement.split(";", 1)[0]
                .split("[", 1)[0]
                .split("=", 1)[0]
                .split("<", 1)[0]
                .split(">", 1)[0]
                .strip()
                .lower()
                for raw_requirement in requirements
                for requirement in [
                    (
                        collapse_rfc2231_value(raw_requirement)
                        if isinstance(raw_requirement, tuple)
                        else str(raw_requirement)
                    )
                ]
                if ";" not in requirement
            }
            forbidden = sorted(runtime_names & FORBIDDEN_RUNTIME_DEPENDENCIES)
            if forbidden:
                raise SystemExit(
                    f"{wheel} includes build-only runtime dependencies: "
                    + ", ".join(forbidden)
                )
            packaged_tests = [
                member
                for member in archive.namelist()
                if member.startswith("adam_core/")
                and ("/tests/" in member or Path(member).name.startswith("test_"))
            ]
            if packaged_tests:
                raise SystemExit(
                    f"{wheel} includes test modules, for example {packaged_tests[0]}"
                )
            has_native_extension = any(
                name.startswith(NATIVE_MODULE_PREFIX)
                and name.endswith((".so", ".pyd", ".dll", ".dylib"))
                for name in archive.namelist()
            )
            if not has_native_extension:
                raise SystemExit(f"{wheel} is missing adam_core._rust_native extension")
            return metadata_version
    except BadZipFile as exc:
        raise SystemExit(f"{wheel} is not a valid wheel: {exc}") from exc


def inspect_artifact_directory(artifact_dir: Path) -> None:
    if not artifact_dir.exists():
        raise SystemExit(f"{artifact_dir} does not exist")
    if not artifact_dir.is_dir():
        raise SystemExit(f"{artifact_dir} is not a directory")

    entries = sorted(artifact_dir.iterdir())
    wheels = [path for path in entries if path.suffix == ".whl"]
    unexpected = [path for path in entries if path.suffix != ".whl"]

    if unexpected:
        formatted = ", ".join(path.name for path in unexpected)
        raise SystemExit(
            f"{artifact_dir} contains non-wheel publish artifacts: {formatted}"
        )
    if not wheels:
        raise SystemExit(f"{artifact_dir} contains no wheel artifacts")

    inspected = [(wheel.name, _inspect_wheel(wheel)) for wheel in wheels]
    print(
        "Wheel artifact inspection passed: "
        + ", ".join(f"{name} ({version})" for name, version in inspected)
    )


def main() -> None:
    artifact_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dist")
    inspect_artifact_directory(artifact_dir)


if __name__ == "__main__":
    main()
