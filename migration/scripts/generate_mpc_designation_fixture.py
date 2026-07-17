"""Generate the MPC packed-designation parity fixture (W11 helper migration).

Run this with the LEGACY baseline interpreter (the untouched adam-core
checkout, no Rust work):

    .legacy-venv/bin/python migration/scripts/generate_mpc_designation_fixture.py

For a panel of designations covering every branch (documented MPC examples,
numbering boundaries, base-62 edge values, whitespace/sign quirks, and
adversarial inputs that exercise the legacy ValueError/KeyError/IndexError
behavior), the fixture freezes the output or the exact exception
(type + message) of all eight public pack/unpack functions. The migration
test ``src/adam_core/utils/tests/test_mpc_rust_parity.py`` asserts identical
behavior from the Rust-dispatched functions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adam_core.utils.mpc import (
    pack_mpc_designation,
    pack_numbered_designation,
    pack_provisional_designation,
    pack_survey_designation,
    unpack_mpc_designation,
    unpack_numbered_designation,
    unpack_provisional_designation,
    unpack_survey_designation,
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

PANEL = [
    # Documented MPC examples (both directions).
    "3202",
    "50000",
    "100345",
    "360017",
    "203289",
    "620000",
    "620061",
    "3140113",
    "15396335",
    "03202",
    "A0345",
    "a0017",
    "K3289",
    "~0000",
    "~000z",
    "~AZaz",
    "~zzzz",
    "1995 XA",
    "1995 XL1",
    "1995 FB13",
    "1998 SQ108",
    "1998 SV127",
    "1998 SS162",
    "2099 AZ193",
    "2008 AA360",
    "2007 TA418",
    "J95X00A",
    "J95X01L",
    "J95F13B",
    "J98SA8Q",
    "J98SC7V",
    "J98SG2S",
    "K99AJ3Z",
    "K08Aa0A",
    "K07Tf8A",
    "2040 P-L",
    "3138 T-1",
    "1010 T-2",
    "4101 T-3",
    "PLS2040",
    "T1S3138",
    "T2S1010",
    "T3S4101",
    # Numbering boundaries.
    "0",
    "1",
    "99999",
    "100000",
    "619999",
    "15396336",
    # Whitespace / sign quirks of Python int().
    "  5  ",
    "+7",
    "-5",
    # Provisional edge cases.
    "1995 IA",
    "1995 ZA",
    "1995 X-1",
    "1995 X",
    "2008 AA619",
    "2008 AA6200",
    "199 5XA",
    "1995_XA",
    # Survey edge cases (legacy does not validate the number part).
    "abcd P-L",
    "12 T-4",
    "PLS0500",
    "T4S1234",
    "PLSabcd",
    # Packed-form edge cases.
    "J95X001",
    "K08!a0A",
    "!0345",
    "~zzzzz",
    "0032",
    # Degenerate inputs.
    "",
    " ",
    "~",
]


def evaluate(function, value: str) -> dict:
    try:
        return {"output": function(value)}
    except Exception as exc:  # noqa: BLE001 - freezing legacy behavior
        return {"error_type": type(exc).__name__, "error_message": str(exc)}


def build_fixture() -> dict:
    return {
        "schema": "adam_core.mpc_designation_fixture",
        "version": 1,
        "generated_by": "migration/scripts/generate_mpc_designation_fixture.py",
        "source_contract": (
            "Legacy adam-core MPC packed-designation helpers, executed in the "
            "untouched legacy checkout."
        ),
        "panel": PANEL,
        "cases": {
            name: [evaluate(function, value) for value in PANEL]
            for name, function in FUNCTIONS.items()
        },
    }


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "migration"
        / "artifacts"
        / "mpc_designation_fixture_2026-07-06.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output_path())
    args = parser.parse_args()
    fixture = build_fixture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=1))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
