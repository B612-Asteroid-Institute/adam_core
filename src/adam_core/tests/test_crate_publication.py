from __future__ import annotations

import pytest

from migration.scripts import publish_crate_archives


def _packages(version: str) -> dict[str, dict[str, object]]:
    packages = {
        name: {"version": version, "dependencies": []}
        for name in publish_crate_archives.PUBLICATION_ORDER
    }
    packages["adam_core"]["dependencies"] = [
        {"name": name, "req": f"={version}"}
        for name in publish_crate_archives.PUBLICATION_ORDER[:-1]
    ]
    return packages


def test_release_package_validation_distinguishes_preview_and_stable() -> None:
    preview = _packages("0.5.7-rc.1")
    publish_crate_archives.validate_release_packages(preview, "0.5.7-rc.1", "preview")
    with pytest.raises(ValueError, match="stable versions"):
        publish_crate_archives.validate_release_packages(
            preview, "0.5.7-rc.1", "stable"
        )

    stable = _packages("0.5.7")
    publish_crate_archives.validate_release_packages(stable, "0.5.7", "stable")
    with pytest.raises(ValueError, match="preview versions"):
        publish_crate_archives.validate_release_packages(stable, "0.5.7", "preview")


def test_crates_io_index_path() -> None:
    assert publish_crate_archives.crates_io_index_path("a") == "1/a"
    assert publish_crate_archives.crates_io_index_path("ab") == "2/ab"
    assert publish_crate_archives.crates_io_index_path("abc") == "3/a/abc"
    assert publish_crate_archives.crates_io_index_path("adam_core") == "ad/am/adam_core"


def test_existing_archive_must_be_exact_and_unyanked() -> None:
    entry = {"cksum": "abc", "yanked": False}
    publish_crate_archives.validate_existing_archive(entry, "adam_core", "0.5.7", "abc")

    with pytest.raises(ValueError, match="checksum"):
        publish_crate_archives.validate_existing_archive(
            entry, "adam_core", "0.5.7", "def"
        )
    with pytest.raises(ValueError, match="yanked"):
        publish_crate_archives.validate_existing_archive(
            {"cksum": "abc", "yanked": True},
            "adam_core",
            "0.5.7",
            "abc",
        )
