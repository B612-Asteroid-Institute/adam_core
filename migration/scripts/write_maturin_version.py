"""Write ``adam_core._version`` from the Maturin Cargo package version.

Maturin uses ``rust/adam_core_py/Cargo.toml`` as the wheel version source.
Cargo requires SemVer (for example ``0.5.6-rc.1``), while Python wheel
metadata uses normalized PEP 440 (``0.5.6rc1``). This script performs that
small, explicit conversion and keeps an exact release tag honest.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CARGO_TOML = REPO_ROOT / "rust" / "adam_core_py" / "Cargo.toml"
VERSION_FILE = REPO_ROOT / "src" / "adam_core" / "_version.py"
PEP440_SAFE = re.compile(
    r"^[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+)?"
    r"(?:\.post[0-9]+)?(?:\.dev[0-9]+)?(?:\+[A-Za-z0-9.]+)?$"
)
CARGO_SEMVER = re.compile(
    r"^(?P<release>(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*))"
    r"(?:-(?P<phase>alpha|beta|rc)\.(?P<number>0|[1-9][0-9]*))?$"
)
VERSION_TAG = re.compile(r"^v(?P<version>.+)$")


def cargo_version_to_pep440(version: str) -> str:
    """Convert the supported Cargo SemVer release forms to canonical PEP 440."""
    match = CARGO_SEMVER.fullmatch(version)
    if match is None:
        raise ValueError(
            f"Cargo version {version!r} must be X.Y.Z or " "X.Y.Z-(alpha|beta|rc).N"
        )
    release = match.group("release")
    phase = match.group("phase")
    if phase is None:
        return release
    pep_phase = {"alpha": "a", "beta": "b", "rc": "rc"}[phase]
    return f"{release}{pep_phase}{match.group('number')}"


def _cargo_versions() -> tuple[str, str]:
    data = tomllib.loads(CARGO_TOML.read_text(encoding="utf-8"))
    version = data.get("package", {}).get("version")
    if not isinstance(version, str):
        raise SystemExit(f"{CARGO_TOML} is missing [package].version")
    try:
        python_version = cargo_version_to_pep440(version)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    return version, python_version


def _exact_version_tag() -> str | None:
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match", "--match", "v[0-9]*"],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        return None

    if result.returncode != 0:
        return None

    tag = result.stdout.strip()
    match = VERSION_TAG.fullmatch(tag)
    if match is None:
        raise SystemExit(f"Exact release tag {tag!r} is not a supported version tag")
    version = match.group("version")
    if PEP440_SAFE.fullmatch(version):
        return version
    try:
        return cargo_version_to_pep440(version)
    except ValueError as error:
        raise SystemExit(
            f"Exact release tag {tag!r} is not supported: {error}"
        ) from error


def _validate_exact_tag(python_version: str) -> None:
    tag_version = _exact_version_tag()
    if tag_version is None:
        return
    if tag_version != python_version:
        raise SystemExit(
            f"{CARGO_TOML.relative_to(REPO_ROOT)} Python version "
            f"{python_version!r} does not match exact release tag {tag_version!r}"
        )


def main() -> None:
    cargo_version, python_version = _cargo_versions()
    _validate_exact_tag(python_version)
    VERSION_FILE.write_text(f"__version__ = {python_version!r}\n", encoding="utf-8")
    print(
        f"Wrote {VERSION_FILE.relative_to(REPO_ROOT)} = {python_version} "
        f"from Cargo {cargo_version}"
    )


if __name__ == "__main__":
    main()
