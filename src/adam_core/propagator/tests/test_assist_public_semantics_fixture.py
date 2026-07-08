from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

FIXTURE_PATH = (
    Path(__file__).resolve().parents[4]
    / "migration"
    / "artifacts"
    / "assist_public_semantics_fixture_2026-05-20.json"
)


def _without_kernel_hashes(fixture: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(fixture)
    for kernel in normalized["kernels"]:
        kernel.pop("sha256", None)
    return normalized


REPO_ROOT = Path(__file__).resolve().parents[4]
_LEGACY_ASSIST_PYTHON = REPO_ROOT / ".legacy-assist-venv" / "bin" / "python"


def _build_fixture_in_legacy_runtime() -> dict[str, Any]:
    """Regenerate the frozen public-semantics fixture in the isolated legacy
    adam_assist runtime (``.legacy-assist-venv``).

    The generator instantiates the downstream, composition-based
    ``adam_assist.ASSISTPropagator``, which is no longer instantiable in this
    (composition-deleted) runtime. Mirroring the adam_core parity oracle, the
    frozen reference is regenerated in the dedicated legacy runtime -- two
    runtimes, one serialized reference. Skips if the legacy runtime is absent.
    """
    if not _LEGACY_ASSIST_PYTHON.exists():
        pytest.skip(
            "legacy adam_assist runtime (.legacy-assist-venv) not built; "
            "see migration/parity/README"
        )
    code = (
        "import json, sys; "
        f"sys.path.insert(0, {str(REPO_ROOT)!r}); "
        "from migration.scripts.generate_assist_public_semantics_fixture "
        "import build_fixture; "
        "sys.stdout.write(json.dumps(build_fixture(include_kernel_sha256=False)))"
    )
    proc = subprocess.run(
        [str(_LEGACY_ASSIST_PYTHON), "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"legacy fixture regeneration failed:\n{proc.stderr}")
    return json.loads(proc.stdout)


def test_assist_public_semantics_fixture_is_current() -> None:
    expected = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    actual = _build_fixture_in_legacy_runtime()
    assert _without_kernel_hashes(expected) == actual


def test_assist_public_semantics_fixture_records_kernel_identity() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    kernels = fixture["kernels"]
    assert {kernel["label"] for kernel in kernels} == {
        "naif_de440",
        "jpl_small_bodies_de441_n16",
    }
    for kernel in kernels:
        assert kernel["size_bytes"] > 0
        assert len(kernel["sha256"]) == 64


def test_assist_public_semantics_fixture_covers_acceptance_surface() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    case_ids = {case["case_id"] for case in fixture["propagation_cases"]}
    assert case_ids == {
        "sun_ecliptic_tdb_input_tdb_targets",
        "ssb_equatorial_tdb_input_tdb_targets",
        "sun_ecliptic_utc_input_utc_targets",
        "variant_metadata_tdb_targets",
    }
    ephemeris_case_ids = {case["case_id"] for case in fixture["ephemeris_cases"]}
    assert ephemeris_case_ids == {"ephemeris_mixed_observers_utc_output"}
    assert (
        fixture["acceptance_target"] == "adam_assist.ASSISTPropagator public semantics"
    )
