"""Fail-closed release inventory and packaged-file-set contracts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from zipfile import ZipFile

import pytest

from migration.scripts import release_inventory
from migration.scripts.check_wheel_artifacts import _verify_release_inventory


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()


def _inventory_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    package = repo / "src/adam_core"
    (repo / "rust/example/src").mkdir(parents=True)
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n")
    (package / "_version.py").write_text("__version__ = '0'\n")
    (repo / "rust/example/Cargo.toml").write_text(
        "[package]\nname='x'\nversion='0.1.0'\n"
    )
    (repo / "rust/example/src/lib.rs").write_text("pub fn value() -> u8 { 1 }\n")
    (repo / ".gitignore").write_text(
        "lib/\n*_ignored.py\n_version.py\n" "/src/adam_core/_release_inventory.json\n"
    )
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "initial")
    monkeypatch.setattr(release_inventory, "REPO_ROOT", repo)
    monkeypatch.setattr(release_inventory, "PACKAGE_ROOT", package)
    monkeypatch.setattr(
        release_inventory, "PACKAGED_INVENTORY", package / "_release_inventory.json"
    )
    return repo


def test_release_inventory_refuses_visible_and_ignored_package_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _inventory_repo(tmp_path, monkeypatch)
    generated_target = repo / "rust/example/target/generated/src/lib.rs"
    generated_target.parent.mkdir(parents=True)
    generated_target.write_text("pub fn generated() {}\n")
    first = release_inventory.create_release_inventory(nonce="1" * 64)
    assert first.commit == _git(repo, "rev-parse", "HEAD")
    assert first.environment()["ADAM_CORE_RELEASE_INVENTORY_NONCE"] == "1" * 64
    assert "adam_core/__init__.py" in first.python_files
    assert "adam_core/_version.py" in first.python_files
    assert "src/adam_core/_version.py" in first.tracked_files

    visible = repo / "src/adam_core/new.py"
    visible.write_text("VALUE = 2\n")
    with pytest.raises(RuntimeError, match="untracked package"):
        release_inventory.create_release_inventory(nonce="2" * 64)
    visible.unlink()

    ignored = repo / "src/adam_core/lib/override.py"
    ignored.parent.mkdir()
    ignored.write_text("VALUE = 3\n")
    with pytest.raises(RuntimeError, match="lib/override.py"):
        release_inventory.create_release_inventory(nonce="3" * 64)
    ignored.unlink()
    ignored.parent.rmdir()

    ignored_name = repo / "src/adam_core/model_ignored.py"
    ignored_name.write_text("VALUE = 4\n")
    with pytest.raises(RuntimeError, match="model_ignored.py"):
        release_inventory.create_release_inventory(nonce="4" * 64)


def test_wheel_checker_rejects_python_set_or_hash_outside_attestation(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "probe.whl"
    inventory = {
        "format": release_inventory.FORMAT,
        "commit": "1" * 40,
        "digest": "2" * 64,
        "python_files": {
            "adam_core/__init__.py": __import__("hashlib")
            .sha256(b"VALUE = 1\n")
            .hexdigest()
        },
    }
    with ZipFile(wheel, "w") as archive:
        archive.writestr("adam_core/__init__.py", b"VALUE = 1\n")
        archive.writestr(
            "adam_core/_release_inventory.json",
            json.dumps(inventory, sort_keys=True),
        )
    with ZipFile(wheel) as archive:
        _verify_release_inventory(archive, wheel)

    with ZipFile(wheel, "a") as archive:
        archive.writestr("adam_core/untracked.py", b"VALUE = 2\n")
    with ZipFile(wheel) as archive:
        with pytest.raises(SystemExit, match="unexpected=.*untracked.py"):
            _verify_release_inventory(archive, wheel)


def test_wheel_checker_requires_inventory_and_exact_attested_contents(
    tmp_path: Path,
) -> None:
    missing_inventory = tmp_path / "missing-inventory.whl"
    with ZipFile(missing_inventory, "w") as archive:
        archive.writestr("adam_core/__init__.py", b"VALUE = 1\n")
    with ZipFile(missing_inventory) as archive:
        with pytest.raises(
            SystemExit, match="missing adam_core/_release_inventory.json"
        ):
            _verify_release_inventory(archive, missing_inventory)

    bad_hash = tmp_path / "bad-hash.whl"
    inventory = {
        "format": release_inventory.FORMAT,
        "commit": "1" * 40,
        "digest": "2" * 64,
        "python_files": {"adam_core/__init__.py": "0" * 64},
    }
    with ZipFile(bad_hash, "w") as archive:
        archive.writestr("adam_core/__init__.py", b"VALUE = 1\n")
        archive.writestr(
            "adam_core/_release_inventory.json", json.dumps(inventory, sort_keys=True)
        )
    with ZipFile(bad_hash) as archive:
        with pytest.raises(SystemExit, match="does not match its attested digest"):
            _verify_release_inventory(archive, bad_hash)

    missing_python = tmp_path / "missing-python.whl"
    python_files = inventory["python_files"]
    assert isinstance(python_files, dict)
    python_files["adam_core/missing.py"] = "3" * 64
    with ZipFile(missing_python, "w") as archive:
        archive.writestr("adam_core/__init__.py", b"VALUE = 1\n")
        archive.writestr(
            "adam_core/_release_inventory.json", json.dumps(inventory, sort_keys=True)
        )
    with ZipFile(missing_python) as archive:
        with pytest.raises(SystemExit, match="missing=.*missing.py"):
            _verify_release_inventory(archive, missing_python)
