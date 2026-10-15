"""Compatibility gate for official forms in the frozen legacy MPC fixture.

Malformed-input exception quirks are intentionally no longer frozen: the
current codec rejects them uniformly while adding the official extended packed
provisional format. Valid historical forms must remain byte compatible.
"""

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


VALID_OFFICIAL_INPUTS = {
    "pack_numbered_designation": {
        "1",
        "3202",
        "50000",
        "99999",
        "100000",
        "100345",
        "203289",
        "360017",
        "619999",
        "620000",
        "620061",
        "3140113",
        "15396335",
    },
    "unpack_numbered_designation": {
        "00001",
        "03202",
        "50000",
        "99999",
        "A0000",
        "A0345",
        "K3289",
        "a0017",
        "z9999",
        "~0000",
        "~000z",
        "~AZaz",
        "~zzzz",
    },
    "pack_provisional_designation": {
        "1995 XA",
        "1995 XL1",
        "1995 FB13",
        "1998 SQ108",
        "1998 SV127",
        "1998 SS162",
        "2099 AZ193",
        "2008 AA360",
        "2007 TA418",
        "2008 AA619",
    },
    "unpack_provisional_designation": {
        "J95X00A",
        "J95X01L",
        "J95F13B",
        "J98SA8Q",
        "J98SC7V",
        "J98SG2S",
        "K99AJ3Z",
        "K08Aa0A",
        "K07Tf8A",
    },
    "pack_survey_designation": {"2040 P-L", "3138 T-1", "1010 T-2", "4101 T-3"},
    "unpack_survey_designation": {"PLS2040", "T1S3138", "T2S1010", "T3S4101"},
}
VALID_OFFICIAL_INPUTS["pack_mpc_designation"] = (
    VALID_OFFICIAL_INPUTS["pack_numbered_designation"]
    | VALID_OFFICIAL_INPUTS["pack_provisional_designation"]
    | VALID_OFFICIAL_INPUTS["pack_survey_designation"]
)
VALID_OFFICIAL_INPUTS["unpack_mpc_designation"] = (
    VALID_OFFICIAL_INPUTS["unpack_numbered_designation"]
    | VALID_OFFICIAL_INPUTS["unpack_provisional_designation"]
    | VALID_OFFICIAL_INPUTS["unpack_survey_designation"]
)


@pytest.mark.parametrize("function_name", sorted(FUNCTIONS))
def test_preserves_legacy_outputs_for_official_forms(fixture, function_name):
    function = FUNCTIONS[function_name]
    accepted = VALID_OFFICIAL_INPUTS[function_name]
    for value, expected in zip(fixture["panel"], fixture["cases"][function_name]):
        if value not in accepted:
            continue
        assert (
            "output" in expected
        ), f"missing legacy output for {function_name}({value!r})"
        assert function(value) == expected["output"]
