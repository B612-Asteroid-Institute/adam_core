from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from migration.scripts.generate_assist_public_semantics_fixture import build_fixture

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


def test_assist_public_semantics_fixture_is_current() -> None:
    expected = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    actual = build_fixture(include_kernel_sha256=False)
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
