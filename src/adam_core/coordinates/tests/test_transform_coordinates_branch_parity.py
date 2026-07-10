"""Legacy-frozen branch parity for transform_coordinates.

Fixture generated from the untouched legacy checkout with:

    .legacy-venv/bin/python migration/scripts/generate_transform_coordinates_branch_fixture.py

This complements randomized parity fuzz by pinning every public dispatcher
branch, including branches that are too specific for random fuzz: identity
returns, validation errors, geodetic output, mixed origins, observatory origins,
ITRF93+covariance, and intentional fallbacks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from adam_core.coordinates.origin import OriginCodes
from adam_core.coordinates.transform import _transform_coordinates_native
from migration.scripts.generate_transform_coordinates_branch_fixture import (
    CASES,
    REPRESENTATIONS,
    build_coordinates,
    run_case,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[4]
    / "migration"
    / "artifacts"
    / "transform_coordinates_branch_fixture_2026-07-06.json"
)

EXPECTED_CASE_NAMES = {case["name"] for case in CASES}


@pytest.fixture(scope="module")
def fixture() -> dict[str, Any]:
    assert FIXTURE_PATH.exists(), (
        "transform_coordinates branch fixture missing; generate with the legacy "
        "interpreter: .legacy-venv/bin/python "
        "migration/scripts/generate_transform_coordinates_branch_fixture.py"
    )
    return json.loads(FIXTURE_PATH.read_text())


@pytest.fixture(scope="module")
def fixture_cases(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    return list(fixture["cases"])


def _case_id(case: dict[str, Any]) -> str:
    return str(case["name"])


def _assert_nan_aware_allclose(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    name: str,
    covariance: bool = False,
) -> None:
    assert actual.shape == expected.shape, name
    both_nan = np.isnan(actual) & np.isnan(expected)
    actual_clean = np.where(both_nan, 0.0, actual)
    expected_clean = np.where(both_nan, 0.0, expected)
    # ITRF93 public-dispatch rows compare CSPICE/spiceypy in the legacy fixture
    # to spicekit/Rust in the migration checkout. Keep the same scoped tolerance
    # as the canonical parity matrix; non-ITRF rows are much tighter.
    if "itrf93" in name:
        atol = 3e-8 if not covariance else 1e-16
        rtol = 1e-12 if not covariance else 1e-9
    elif "com_" in name or "_to_com_" in name:
        atol = 1e-9 if not covariance else 1e-18
        rtol = 1e-12 if not covariance else 1e-10
    else:
        atol = 1e-10 if not covariance else 1e-18
        rtol = 1e-12 if not covariance else 1e-10
    np.testing.assert_allclose(
        actual_clean,
        expected_clean,
        rtol=rtol,
        atol=atol,
        err_msg=name,
    )
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected), err_msg=name)


def _assert_outputs_match(
    actual: dict[str, Any], expected: dict[str, Any], name: str
) -> None:
    assert actual["ok"] == expected["ok"], name
    if not expected["ok"]:
        assert actual["error_type"] == expected["error_type"], name
        assert actual["error_message"] == expected["error_message"], name
        return

    actual_out = actual["output"]
    expected_out = expected["output"]
    for key in ("type", "frame", "origin", "days", "nanos", "scale"):
        assert actual_out[key] == expected_out[key], f"{name}: {key}"
    _assert_nan_aware_allclose(
        np.asarray(actual_out["values"], dtype=np.float64),
        np.asarray(expected_out["values"], dtype=np.float64),
        name=name,
    )
    _assert_nan_aware_allclose(
        np.asarray(actual_out["covariance"], dtype=np.float64),
        np.asarray(expected_out["covariance"], dtype=np.float64),
        name=f"{name}_covariance",
        covariance=True,
    )


def test_fixture_case_set_is_exhaustive(fixture_cases: list[dict[str, Any]]) -> None:
    assert {case["name"] for case in fixture_cases} == EXPECTED_CASE_NAMES
    assert len(fixture_cases) == 23


@pytest.mark.parametrize("case", CASES, ids=lambda case: str(case["name"]))
def test_transform_coordinates_branch_matches_legacy_fixture(
    fixture: dict[str, Any], case: dict[str, Any]
) -> None:
    fixture_by_name = {item["name"]: item for item in fixture["cases"]}
    fixture_case = fixture_by_name[case["name"]]
    # Guard against the test's checked-in branch matrix drifting without
    # regenerating the legacy fixture.
    assert fixture_case["input"] == case
    _assert_outputs_match(run_case(case), fixture_case["legacy"], case["name"])


# Value branches that the native single-crossing path does not cover and so
# fall through to the thin Python composition: non-Cartesian input into an
# ITRF93 frame change (there is no meaningful non-Cartesian ITRF93
# representation to rotate from). Everything else -- including plain Cartesian
# frame/origin changes, observatory-code origins, and time-varying ITRF93 --
# runs 100% in Rust in a single crossing.
_NATIVELY_UNCOVERED_VALUE_CASES = {"fallback_noncart_itrf93_input"}


@pytest.mark.parametrize(
    "case",
    [
        case
        for case in CASES
        if case.get("kind") == "value" and case.get("expect_rust_support") is not None
    ],
    ids=lambda case: str(case["name"]),
)
def test_transform_coordinates_value_cases_run_natively(
    case: dict[str, Any],
) -> None:
    """Each value branch runs entirely in Rust via ``_transform_coordinates_native``
    (a single Python->Rust crossing), except the deliberately-uncovered cases
    which return ``None`` so the public ``transform_coordinates`` uses the thin
    Python fallthrough composition."""
    coords = build_coordinates(case)
    representation_out = case.get("representation_out")
    representation_out_type = (
        REPRESENTATIONS[str(representation_out)]
        if representation_out is not None
        else coords.__class__
    )
    frame_out = case.get("frame_out") or coords.frame
    origin_out = case.get("origin_out")
    origin_out_value = OriginCodes[str(origin_out)] if origin_out is not None else None
    native = _transform_coordinates_native(
        coords,
        representation_out_type,
        frame_out,
        origin_out_value,
    )
    if case["name"] in _NATIVELY_UNCOVERED_VALUE_CASES:
        assert native is None
    else:
        assert native is not None
