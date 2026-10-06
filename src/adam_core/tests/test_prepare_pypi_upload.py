from __future__ import annotations

from pathlib import Path

import pytest

from migration.scripts import prepare_pypi_upload


def _wheels(dist: Path) -> tuple[Path, Path]:
    first = dist / "package-1.0-cp311-cp311-manylinux.whl"
    second = dist / "package-1.0-cp312-cp312-manylinux.whl"
    first.write_bytes(b"first accepted wheel")
    second.write_bytes(b"second accepted wheel")
    return first, second


def test_missing_release_prepares_every_accepted_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    wheels = _wheels(dist)
    monkeypatch.setattr(
        prepare_pypi_upload, "fetch_release", lambda *args, **kwargs: None
    )

    state = prepare_pypi_upload.publication_state(
        dist, "package", "1.0", expected_count=2
    )
    upload = tmp_path / "upload"
    prepare_pypi_upload.prepare_upload_directory(dist, upload, state["missing"])

    assert state["existing"] == []
    assert state["missing"] == sorted(wheel.name for wheel in wheels)
    assert sorted(path.name for path in upload.iterdir()) == state["missing"]


def test_partial_release_skips_only_checksum_identical_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    first, second = _wheels(dist)
    release = {
        "urls": [
            {
                "filename": first.name,
                "digests": {"sha256": prepare_pypi_upload.sha256(first)},
                "yanked": False,
            }
        ]
    }
    monkeypatch.setattr(
        prepare_pypi_upload, "fetch_release", lambda *args, **kwargs: release
    )

    state = prepare_pypi_upload.publication_state(
        dist, "package", "1.0", expected_count=2
    )

    assert state["existing"] == [first.name]
    assert state["missing"] == [second.name]
    assert state["complete"] is False


@pytest.mark.parametrize(
    "public_file",
    [
        {
            "filename": "package-1.0-cp311-cp311-manylinux.whl",
            "digests": {"sha256": "0" * 64},
            "yanked": False,
        },
        {
            "filename": "unexpected.whl",
            "digests": {"sha256": "0" * 64},
            "yanked": False,
        },
        {
            "filename": "package-1.0-cp311-cp311-manylinux.whl",
            "digests": {"sha256": "0" * 64},
            "yanked": True,
        },
    ],
)
def test_publication_state_rejects_nonidentical_or_unexpected_public_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    public_file: dict[str, object],
) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    _wheels(dist)
    monkeypatch.setattr(
        prepare_pypi_upload,
        "fetch_release",
        lambda *args, **kwargs: {"urls": [public_file]},
    )

    with pytest.raises(ValueError, match="does not match"):
        prepare_pypi_upload.publication_state(dist, "package", "1.0", expected_count=2)
