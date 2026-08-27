"""Parity gate for the Rust MPC packed-designation helpers (W11) against the
frozen legacy fixture generated in the untouched legacy checkout via
``.legacy-venv/bin/python migration/scripts/generate_mpc_designation_fixture.py``.

Every public function must reproduce the legacy output or raise the same
exception type with the same message for every panel input."""

import json
from pathlib import Path

import pytest

from ..mpc import (
    pack_mpc_designation,
    pack_numbered_designation,
    pack_provisional_designation,
    pack_survey_designation,
    unpack_mpc_designation,
    unpack_numbered_designation,
    unpack_provisional_designation,
    unpack_survey_designation,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[4]
    / "migration"
    / "artifacts"
    / "mpc_designation_fixture_2026-07-06.json"
)

FUNCTIONS = {
    "pack_numbered_designation": pack_numbered_designation,
    "pack_provisional_designation": pack_provisional_designation,
    "pack_survey_designation": pack_survey_designation,
    "pack_mpc_designation": pack_mpc_designation,
    "unpack_numbered_designation": unpack_numbered_designation,
    "unpack_provisional_designation": unpack_provisional_designation,
    "unpack_survey_designation": unpack_survey_designation,
    "unpack_mpc_designation": unpack_mpc_designation,
}


@pytest.fixture(scope="module")
def fixture():
    assert FIXTURE_PATH.exists(), (
        "MPC designation fixture missing; generate it with the legacy "
        "interpreter: .legacy-venv/bin/python "
        "migration/scripts/generate_mpc_designation_fixture.py"
    )
    return json.loads(FIXTURE_PATH.read_text())


@pytest.mark.parametrize("function_name", sorted(FUNCTIONS))
def test_matches_legacy_fixture(fixture, function_name):
    function = FUNCTIONS[function_name]
    for value, expected in zip(fixture["panel"], fixture["cases"][function_name]):
        label = f"{function_name}({value!r})"
        if "output" in expected:
            assert function(value) == expected["output"], label
        else:
            try:
                result = function(value)
            except Exception as exc:  # noqa: BLE001 - comparing to legacy
                assert type(exc).__name__ == expected["error_type"], (
                    f"{label}: {type(exc).__name__} != {expected['error_type']} "
                    f"({exc})"
                )
                assert str(exc) == expected["error_message"], label
            else:
                raise AssertionError(
                    f"{label} returned {result!r}, expected "
                    f"{expected['error_type']}: {expected['error_message']}"
                )
