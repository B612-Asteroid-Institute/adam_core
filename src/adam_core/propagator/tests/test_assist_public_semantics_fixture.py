from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

FIXTURE_PATH = (
    Path(__file__).resolve().parents[4]
    / "migration"
    / "artifacts"
    / "assist_public_semantics_fixture_2026-05-20.json"
)
FIXTURE_SHA256 = "31de414652f86a4d23399610e32407bb11e37022e5c28af27e1fbf85dc1aa913"


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_assist_public_semantics_fixture_is_hash_pinned() -> None:
    assert hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest() == FIXTURE_SHA256
    assert _fixture()["packages"] == {
        "adam-assist": "0.3.11.dev12+gcb5bb14",
        "assist": "1.2.3",
        "rebound": "4.6.0",
        "adam-core": "0.5.7.dev39+g757c09fc",
    }


def test_assist_public_semantics_fixture_records_kernel_identity() -> None:
    kernels = _fixture()["kernels"]
    assert {kernel["label"] for kernel in kernels} == {
        "naif_de440",
        "jpl_small_bodies_de441_n16",
    }
    for kernel in kernels:
        assert kernel["size_bytes"] > 0
        assert len(kernel["sha256"]) == 64


def test_assist_public_semantics_fixture_covers_acceptance_surface() -> None:
    fixture = _fixture()
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
