from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from migration.scripts.clean_room_artifact_smoke import _validate_runtime_versions

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "migration" / "scripts" / "write_maturin_version.py"
SPEC = importlib.util.spec_from_file_location("write_maturin_version", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    ("cargo", "python"),
    [
        ("0.5.6", "0.5.6"),
        ("0.5.6-alpha.1", "0.5.6a1"),
        ("0.5.6-beta.2", "0.5.6b2"),
        ("0.5.6-rc.1", "0.5.6rc1"),
        ("0.5.6-rc.2", "0.5.6rc2"),
        ("0.5.6-rc.3", "0.5.6rc3"),
        ("0.5.6-rc.4", "0.5.6rc4"),
        ("0.5.6-rc.5", "0.5.6rc5"),
    ],
)
def test_cargo_version_to_pep440(cargo: str, python: str) -> None:
    assert MODULE.cargo_version_to_pep440(cargo) == python


@pytest.mark.parametrize(
    "version",
    ["0.5", "0.5.6-preview.1", "0.5.6-rc", "0.5.6-rc.01", "0.5.6+local"],
)
def test_cargo_version_to_pep440_rejects_unsupported_forms(version: str) -> None:
    with pytest.raises(ValueError):
        MODULE.cargo_version_to_pep440(version)


def test_runtime_versions_must_match_distribution_metadata() -> None:
    versions = {"adam-core": "0.5.6rc5", "adam-assist": "0.4.0rc6"}
    assert _validate_runtime_versions(versions, versions.copy()) == versions

    with pytest.raises(AssertionError, match="runtime package version mismatch"):
        _validate_runtime_versions(
            versions,
            {"adam-core": "0.0.0dev0", "adam-assist": "0.4.0rc6"},
        )


def test_release_matrix_generates_and_inspects_runtime_version() -> None:
    workflow = (
        ROOT / ".github/workflows/release-candidate-wheel-matrix.yml"
    ).read_text()
    writer = "python migration/scripts/write_maturin_version.py"
    builder = "uses: PyO3/maturin-action@v1"
    inspector = "python migration/scripts/check_wheel_artifacts.py"

    assert workflow.index(writer) < workflow.index(builder) < workflow.index(inspector)
    assert 'PYTHON_PREVIEW_VERSION: "0.5.6rc5"' in workflow
