"""
This module tests using custom SPICE kernels using a small JWST kernel.
"""

from pathlib import Path

import numpy as np
import pytest

from ...coordinates.origin import OriginCodes
from ...time import Timestamp
from ...utils.spice import (
    get_spice_body_state,
    register_spice_kernel,
    unregister_spice_kernel,
)
from ...utils.spice_backend import get_backend
from ..observers import Observers
from ..state import get_observer_state

# JWST SPICE kernel paths
JWST_KERNEL_DIR = (
    Path(__file__).parent.parent.parent / "utils" / "tests" / "data" / "spice"
)
JWST_KERNEL_PATH = JWST_KERNEL_DIR / "jwst_horizons_20200101_20240101_v01.bsp"
JWST_FIXTURE_PATH = (
    Path(__file__).resolve().parents[4]
    / "migration"
    / "artifacts"
    / "jwst_observer_fixture_2026-07-06.json"
)


@pytest.fixture
def jwst_kernel():
    """Fixture to handle JWST kernel registration and cleanup."""
    if not JWST_KERNEL_PATH.exists():
        pytest.skip(
            "JWST SPICE kernel not found. Please download it to tests/data/spice/jwst_horizons_20200101_20240101_v01.bsp"
        )

    register_spice_kernel(str(JWST_KERNEL_PATH))
    yield
    unregister_spice_kernel(str(JWST_KERNEL_PATH))


def test_jwst_as_observer(jwst_kernel):
    """Test that we can use JWST as an observer using string identifier."""
    # Test with a date known to be in the JWST kernel range
    test_time = Timestamp.from_iso8601(["2022-01-01T00:00:00Z"])

    # Get JWST state vectors using the observer system with string identifier
    jwst_observer_state = get_observer_state(
        "JWST", test_time, frame="ecliptic", origin=OriginCodes.SUN
    )

    # Verify we got valid state vectors
    assert jwst_observer_state.values.shape == (1, 6)
    assert jwst_observer_state.frame == "ecliptic"
    assert jwst_observer_state.origin.code[0].as_py() == "SUN"

    # Get JWST state using the direct spice function
    jwst_id = get_backend().bodn2c("JWST")
    jwst_direct_state = get_spice_body_state(
        jwst_id, test_time, frame="ecliptic", origin=OriginCodes.SUN
    )

    # Both methods should return identical results
    np.testing.assert_allclose(
        jwst_observer_state.values, jwst_direct_state.values, rtol=1e-10
    )


def test_jwst_as_observer_from_code(jwst_kernel):
    """Test that we can use JWST as an observer using Observers.from_code."""
    # Test with a date known to be in the JWST kernel range
    test_time = Timestamp.from_iso8601(["2022-01-01T00:00:00Z"])

    observers = Observers.from_code("JWST", test_time)
    assert len(observers) == 1
    assert observers.code[0].as_py() == "JWST"
    assert observers.coordinates.values.shape == (1, 6)
    assert observers.coordinates.frame == "ecliptic"
    assert observers.coordinates.origin.code[0].as_py() == "SUN"

    observers = Observers.from_codes(["JWST"], test_time)
    assert len(observers) == 1
    assert observers.code[0].as_py() == "JWST"
    assert observers.coordinates.values.shape == (1, 6)
    assert observers.coordinates.frame == "ecliptic"
    assert observers.coordinates.origin.code[0].as_py() == "SUN"


def test_mixed_ground_and_custom_space_observers_share_one_batch(jwst_kernel):
    times = Timestamp.from_iso8601(["2022-01-01T00:00:00Z", "2022-01-02T00:00:00Z"])
    mixed = Observers.from_codes(["JWST", "500"], times)
    jwst = get_observer_state("JWST", times[:1])
    geocenter = get_observer_state("500", times[1:])
    np.testing.assert_allclose(
        mixed.coordinates.values[0], jwst.values[0], rtol=0, atol=0
    )
    np.testing.assert_allclose(
        mixed.coordinates.values[1], geocenter.values[0], rtol=0, atol=4e-18
    )
    assert mixed.code.to_pylist() == ["JWST", "500"]


def test_jwst_states_match_legacy_cspice_fixture(jwst_kernel):
    """Numeric parity for the custom-kernel space-observatory path
    (correction to the personal-cmy.27 audit): the spicekit-served JWST
    states must match the frozen legacy CSPICE fixture generated in the
    untouched legacy checkout on the same vendored kernel. Tolerance matches
    the spicekit-vs-CSPICE bound accepted for ground observers (cmy.6)."""
    import json

    import numpy as np

    assert JWST_FIXTURE_PATH.exists(), (
        "JWST observer fixture missing; generate with the legacy interpreter: "
        ".legacy-venv/bin/python migration/scripts/generate_jwst_observer_fixture.py"
    )
    fixture = json.loads(JWST_FIXTURE_PATH.read_text())

    times = Timestamp.from_mjd(
        np.asarray(fixture["epoch_mjds_tdb"], dtype=np.float64), scale="tdb"
    )
    for frame, expected in fixture["states"].items():
        coordinates = get_observer_state(
            "JWST", times, frame=frame, origin=OriginCodes.SUN
        )
        np.testing.assert_allclose(
            np.asarray(coordinates.values, dtype=np.float64),
            np.asarray(expected, dtype=np.float64),
            rtol=0.0,
            atol=1e-11,
            err_msg=f"JWST {frame} states diverge from legacy CSPICE",
        )
