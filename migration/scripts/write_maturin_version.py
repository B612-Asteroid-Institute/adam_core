"""Write adam_core._version from the maturin package version.

The adam-core wheel version source of truth is
`rust/adam_core_py/Cargo.toml` `[package].version`, because maturin uses that
value for wheel metadata when `project.version` is dynamic. This script mirrors
that version into the Python runtime module before build.
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
VERSION_TAG = re.compile(r"^v(?P<version>.+)$")


def _cargo_version() -> str:
    data = tomllib.loads(CARGO_TOML.read_text(encoding="utf-8"))
    version = data.get("package", {}).get("version")
    if not isinstance(version, str):
        raise SystemExit(f"{CARGO_TOML} is missing [package].version")
    if not PEP440_SAFE.fullmatch(version):
        raise SystemExit(
            f"{CARGO_TOML} version {version!r} is not safe to expose as PEP 440"
        )
    return version


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
    if not PEP440_SAFE.fullmatch(version):
        raise SystemExit(f"Exact release tag {tag!r} is not PEP 440 compatible")
    return version


def _validate_exact_tag(version: str) -> None:
    tag_version = _exact_version_tag()
    if tag_version is None:
        return
    if tag_version != version:
        raise SystemExit(
            f"{CARGO_TOML.relative_to(REPO_ROOT)} version {version!r} does not match "
            f"exact release tag {tag_version!r}"
        )


def main() -> None:
    version = _cargo_version()
    _validate_exact_tag(version)
    VERSION_FILE.write_text(f"__version__ = {version!r}\n", encoding="utf-8")
    print(f"Wrote {VERSION_FILE.relative_to(REPO_ROOT)} = {version}")


if __name__ == "__main__":
    main()
