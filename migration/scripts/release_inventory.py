"""Exact per-invocation source attestation for adam-core release wheels."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "src" / "adam_core"
PACKAGED_INVENTORY = PACKAGE_ROOT / "_release_inventory.json"
FORMAT = "adam-core-release-inventory-v2"


def _git(*arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "release inventory requires a readable Git checkout"
        ) from error


def _paths(raw: bytes) -> list[str]:
    try:
        return sorted(item.decode("utf-8") for item in raw.split(b"\0") if item)
    except UnicodeDecodeError as error:
        raise RuntimeError("release source paths must be valid UTF-8") from error


def _is_test_path(path: str) -> bool:
    parts = Path(path).parts
    return "tests" in parts or Path(path).name.startswith("test_")


def _allowed_generated_package_file(path: Path) -> bool:
    relative = path.relative_to(PACKAGE_ROOT).as_posix()
    return (
        relative in {"_release_inventory.json", "_version.py"}
        or "__pycache__" in path.parts
        or path.suffix in {".pyc", ".pyo"}
        or (
            path.name.startswith("_rust_native")
            and path.suffix in {".so", ".pyd", ".dll", ".dylib"}
        )
    )


def _source_affecting_rust_file(path: Path) -> bool:
    relative_path = path.relative_to(REPO_ROOT)
    if "target" in relative_path.parts:
        return False
    relative = relative_path.as_posix()
    return relative.endswith(("/Cargo.toml", "/build.rs")) or "/src/" in relative


def _inventory_digest(files: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(FORMAT.encode("ascii") + b"\0")
    for relative in files:
        payload = (REPO_ROOT / relative).read_bytes()
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _assert_clean_source(tracked: set[str]) -> None:
    dirty = _git(
        "status",
        "--porcelain",
        "-z",
        "--untracked-files=no",
        "--",
        ".",
        ":(exclude)adam-assist",
    )
    if dirty:
        raise RuntimeError(
            "release inventory requires a clean tracked adam-core checkout"
        )

    untracked_package: list[str] = []
    for path in PACKAGE_ROOT.rglob("*"):
        if not path.is_file() or _allowed_generated_package_file(path):
            continue
        relative = path.relative_to(REPO_ROOT).as_posix()
        if _is_test_path(relative):
            continue
        if relative not in tracked:
            untracked_package.append(relative)
    if untracked_package:
        raise RuntimeError(
            "release inventory refuses untracked package files: "
            + ", ".join(sorted(untracked_package))
        )

    untracked_rust: list[str] = []
    rust_root = REPO_ROOT / "rust"
    for path in rust_root.rglob("*"):
        if path.is_file() and _source_affecting_rust_file(path):
            relative = path.relative_to(REPO_ROOT).as_posix()
            if relative not in tracked:
                untracked_rust.append(relative)
    if untracked_rust:
        raise RuntimeError(
            "release inventory refuses untracked Rust source: "
            + ", ".join(sorted(untracked_rust))
        )


@dataclass(frozen=True)
class ReleaseInventory:
    commit: str
    nonce: str
    digest: str
    tracked_files: tuple[str, ...]
    python_files: dict[str, str]
    attestation_path: Path

    def environment(self) -> dict[str, str]:
        return {
            "ADAM_CORE_RELEASE_BUILD": "1",
            "ADAM_CORE_BUILD_COMMIT": self.commit,
            "ADAM_CORE_RELEASE_INVENTORY_DIGEST": self.digest,
            "ADAM_CORE_RELEASE_INVENTORY_NONCE": self.nonce,
        }


def create_release_inventory(*, nonce: str | None = None) -> ReleaseInventory:
    """Create and persist a fresh exact attestation immediately before build."""
    commit = _git("rev-parse", "HEAD").decode("ascii").strip()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise RuntimeError("release inventory requires an exact lowercase Git commit")
    tracked_files = [
        path
        for path in _paths(_git("ls-files", "-z"))
        if not path.startswith("adam-assist/")
    ]
    tracked = set(tracked_files)
    _assert_clean_source(tracked)
    generated_version = "src/adam_core/_version.py"
    if (REPO_ROOT / generated_version).is_file() and generated_version not in tracked:
        tracked_files.append(generated_version)
        tracked_files.sort()
    digest = _inventory_digest(tracked_files)
    nonce = nonce or secrets.token_hex(32)
    if len(nonce) != 64 or any(char not in "0123456789abcdef" for char in nonce):
        raise RuntimeError(
            "release inventory nonce must be 64 lowercase hexadecimal characters"
        )

    python_files = {
        path.removeprefix("src/"): hashlib.sha256(
            (REPO_ROOT / path).read_bytes()
        ).hexdigest()
        for path in tracked_files
        if path.startswith("src/adam_core/")
        and path.endswith(".py")
        and not _is_test_path(path)
    }
    raw_git_path = (
        _git("rev-parse", "--git-path", "adam-core-release-attestation")
        .decode()
        .strip()
    )
    attestation_path = Path(raw_git_path)
    if not attestation_path.is_absolute():
        attestation_path = REPO_ROOT / attestation_path
    lines = [
        FORMAT,
        f"commit={commit}",
        f"nonce={nonce}",
        f"digest={digest}",
        *(f"file={path}" for path in tracked_files),
    ]
    attestation_path.parent.mkdir(parents=True, exist_ok=True)
    attestation_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    PACKAGED_INVENTORY.write_text(
        json.dumps(
            {
                "format": FORMAT,
                "commit": commit,
                "digest": digest,
                "python_files": python_files,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return ReleaseInventory(
        commit,
        nonce,
        digest,
        tuple(tracked_files),
        python_files,
        attestation_path,
    )


def remove_packaged_inventory() -> None:
    try:
        PACKAGED_INVENTORY.unlink()
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    inventory = create_release_inventory(
        nonce=os.environ.get("ADAM_CORE_RELEASE_INVENTORY_NONCE")
    )
    print(
        json.dumps(
            {**inventory.environment(), "attestation": str(inventory.attestation_path)}
        )
    )
