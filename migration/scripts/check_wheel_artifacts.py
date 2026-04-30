"""Validate adam-core wheel artifacts before upload or clean-install smoke tests."""

from __future__ import annotations

import sys
from email.parser import Parser
from pathlib import Path
from zipfile import BadZipFile, ZipFile

NATIVE_MODULE_PREFIX = "adam_core/_rust_native"
VERSION_MODULE = "adam_core/_version.py"


def _wheel_metadata(archive: ZipFile, wheel: Path) -> tuple[str, str]:
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
    return name, version


def _runtime_version(archive: ZipFile, wheel: Path) -> str:
    if VERSION_MODULE not in archive.namelist():
        raise SystemExit(f"{wheel} is missing {VERSION_MODULE}")
    namespace: dict[str, str] = {}
    exec(archive.read(VERSION_MODULE), namespace)
    version = namespace.get("__version__")
    if not isinstance(version, str):
        raise SystemExit(f"{wheel} {VERSION_MODULE} does not define __version__")
    return version


def _inspect_wheel(wheel: Path) -> str:
    try:
        with ZipFile(wheel) as archive:
            name, metadata_version = _wheel_metadata(archive, wheel)
            runtime_version = _runtime_version(archive, wheel)
            if name not in {"adam_core", "adam-core"}:
                raise SystemExit(f"{wheel} has unexpected package name {name!r}")
            if runtime_version != metadata_version:
                raise SystemExit(
                    f"{wheel} runtime version {runtime_version!r} does not match "
                    f"METADATA version {metadata_version!r}"
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


def main() -> None:
    artifact_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dist")
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


if __name__ == "__main__":
    main()
