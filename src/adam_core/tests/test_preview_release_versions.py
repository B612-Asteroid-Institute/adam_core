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


def test_rust_ci_is_reproducible_and_downstream_sources_are_exact() -> None:
    workflows = ROOT / ".github/workflows"
    normal_ci = (workflows / "pip-build-lint-test-coverage.yml").read_text()
    crate_ci = (workflows / "rust-crate-release-candidate.yml").read_text()
    tier1 = (workflows / "tier1-dependent-smoke.yml").read_text()
    assist_sha = "c7795c1e205f33ad3c4e2f81d572c8344fda7be0"

    assert "dtolnay/rust-toolchain@1.87.0" in normal_ci
    assert "components: rustfmt, clippy" in normal_ci
    assert "dtolnay/rust-toolchain@1.87.0" in crate_ci
    assert "components: rustfmt, clippy" in crate_ci
    assert "dtolnay/rust-toolchain@1.87.0" in tier1
    assert assist_sha in normal_ci
    assert assist_sha in tier1
    assert 'version("adam-core") == adam_core.__version__ == "0.5.6rc5"' in tier1
    assert 'version("adam-assist") == assist_version == "0.4.0rc6"' in tier1

    for workflow in workflows.glob("*.yml"):
        source = workflow.read_text()
        if "dtolnay/rust-toolchain@" not in source:
            continue
        assert "dtolnay/rust-toolchain@stable" not in source, workflow.name
        assert "dtolnay/rust-toolchain@1.87.0" in source, workflow.name
        assert "components: rustfmt, clippy" in source, workflow.name
