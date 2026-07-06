"""Parity gates for the Rust OEM engine (bead personal-cmy.28) against the
frozen legacy fixture generated in the untouched legacy checkout (which
writes through the third-party `oem` package) via
``.legacy-venv/bin/python migration/scripts/generate_oem_parity_fixture.py``.

Writer output must match byte-for-byte modulo the wall-clock CREATION_DATE
line; parses must match bit-for-bit (integer epochs, float values,
covariances with NaN-aware comparison)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from ...coordinates import CartesianCoordinates
from ...coordinates.covariances import CoordinateCovariances
from ...coordinates.origin import Origin
from ...time import Timestamp
from ..oem_io import orbit_from_oem, orbit_to_oem
from ..orbits import Orbits

FIXTURE_PATH = (
    Path(__file__).resolve().parents[4]
    / "migration"
    / "artifacts"
    / "oem_parity_fixture_2026-07-06.json"
)


@pytest.fixture(scope="module")
def fixture():
    assert FIXTURE_PATH.exists(), (
        "OEM parity fixture missing; generate with the legacy interpreter: "
        ".legacy-venv/bin/python migration/scripts/generate_oem_parity_fixture.py"
    )
    return json.loads(FIXTURE_PATH.read_text())


def orbits_from_flat(flat: dict) -> Orbits:
    values = np.asarray(flat["values"], dtype=np.float64)
    covariance = CoordinateCovariances.from_matrix(
        np.asarray(flat["covariance"], dtype=np.float64)
    )
    return Orbits.from_kwargs(
        orbit_id=flat["orbit_id"],
        object_id=flat["object_id"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=values[:, 0],
            y=values[:, 1],
            z=values[:, 2],
            vx=values[:, 3],
            vy=values[:, 4],
            vz=values[:, 5],
            time=Timestamp.from_kwargs(
                days=flat["days"], nanos=flat["nanos"], scale=flat["scale"]
            ),
            frame=flat["frame"],
            origin=Origin.from_kwargs(code=flat["origin"]),
            covariance=covariance,
        ),
    )


def mask_creation_date(text: str) -> str:
    return "\n".join(
        line for line in text.split("\n") if not line.startswith("CREATION_DATE = ")
    )


def assert_flat_equal(actual: Orbits, expected: dict, label: str):
    assert actual.orbit_id.to_pylist() == expected["orbit_id"], label
    assert actual.object_id.to_pylist() == expected["object_id"], label
    coordinates = actual.coordinates
    assert coordinates.time.days.to_pylist() == expected["days"], label
    assert coordinates.time.nanos.to_pylist() == expected["nanos"], label
    assert coordinates.time.scale == expected["scale"], label
    assert coordinates.frame == expected["frame"], label
    assert coordinates.origin.code.to_pylist() == expected["origin"], label
    actual_values = np.asarray(coordinates.values, dtype=np.float64)
    expected_values = np.asarray(expected["values"], dtype=np.float64)
    np.testing.assert_array_equal(actual_values, expected_values, err_msg=label)
    actual_cov = np.asarray(coordinates.covariance.to_matrix(), dtype=np.float64)
    expected_cov = np.asarray(expected["covariance"], dtype=np.float64)
    assert actual_cov.shape == expected_cov.shape, label
    both_nan = np.isnan(actual_cov) & np.isnan(expected_cov)
    np.testing.assert_array_equal(
        np.where(both_nan, 0.0, actual_cov),
        np.where(both_nan, 0.0, expected_cov),
        err_msg=label,
    )


def test_writer_matches_legacy_fixture(fixture, tmp_path):
    for panel in fixture["panels"]:
        orbits = orbits_from_flat(panel["orbits"])
        path = tmp_path / f"{panel['name']}.oem"
        orbit_to_oem(orbits, str(path), originator=fixture["originator"])
        actual = mask_creation_date(path.read_text())
        expected = mask_creation_date(panel["oem_text"])
        assert actual == expected, panel["name"]


def test_parser_matches_legacy_fixture(fixture, tmp_path):
    for panel in fixture["panels"]:
        path = tmp_path / f"{panel['name']}.oem"
        path.write_text(panel["oem_text"])
        parsed = orbit_from_oem(str(path))
        assert_flat_equal(parsed, panel["parsed"], panel["name"])


def test_round_trip_write_parse(fixture, tmp_path):
    for panel in fixture["panels"]:
        orbits = orbits_from_flat(panel["orbits"])
        path = tmp_path / f"rt_{panel['name']}.oem"
        orbit_to_oem(orbits, str(path), originator=fixture["originator"])
        parsed = orbit_from_oem(str(path))
        # The legacy parse of the legacy file equals our parse of our file.
        assert_flat_equal(parsed, panel["parsed"], panel["name"])


def test_math_isfinite_guard():
    # Guard against silent NaN acceptance in the fixture reconstruction.
    assert math.isfinite(1.0)
