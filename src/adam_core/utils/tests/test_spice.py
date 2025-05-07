from pathlib import Path

import numpy as np
import pytest
import spiceypy as sp
from naif_leapseconds import leapseconds

from ...time import Timestamp
from ..spice import (
    DEFAULT_KERNELS,
    _jd_tdb_to_et,
    get_spice_body_state,
    list_registered_kernels,
    register_spice_kernel,
    setup_SPICE,
    unregister_spice_kernel,
)

# JWST SPICE kernel paths
JWST_KERNEL_DIR = Path(__file__).parent / "data" / "spice"
JWST_KERNEL_PATH = JWST_KERNEL_DIR / "jwst_horizons_20200101_20240101_v01.bsp"

# JWST kernel time range (from filename)
JWST_START_TIME = Timestamp.from_mjd(np.array([58849.0]), scale="tdb")  # 2020-01-01
JWST_END_TIME = Timestamp.from_mjd(np.array([60314.0]), scale="tdb")  # 2024-01-01


def test_list_registered_kernels():
    """Test listing registered kernels."""
    # kernels could already be loaded, so let's list and unregister all of them first
    for kernel in list_registered_kernels():
        unregister_spice_kernel(kernel)

    assert len(list_registered_kernels()) == 0

    # register all the default kernels
    setup_SPICE(force=True)
    assert len(list_registered_kernels()) == len(DEFAULT_KERNELS)


@pytest.fixture
def jwst_kernel():
    """Fixture to handle JWST kernel registration and cleanup."""
    if not JWST_KERNEL_PATH.exists():
        raise Exception(
            "JWST SPICE kernel not found. Please download it to tests/data/spice/jwst_horizons_20200101_20240101_v01.bsp"
        )

    register_spice_kernel(str(JWST_KERNEL_PATH))
    yield
    unregister_spice_kernel(str(JWST_KERNEL_PATH))


def test_register_unregister_kernel():
    """Test registering and unregistering custom SPICE kernels."""
    # The kernel is already registered by the fixture
    assert str(JWST_KERNEL_PATH) not in list_registered_kernels()
    register_spice_kernel(str(JWST_KERNEL_PATH))
    assert str(JWST_KERNEL_PATH) in list_registered_kernels()
    # Unregister the kernel
    unregister_spice_kernel(str(JWST_KERNEL_PATH))
    assert str(JWST_KERNEL_PATH) not in list_registered_kernels()


def test_register_kernel_twice(jwst_kernel):
    """Test that registering the same kernel twice doesn't cause issues."""
    # Register again (kernel already registered by fixture)
    register_spice_kernel(str(JWST_KERNEL_PATH))

    # Should still only appear once in the list (since it's a set)
    assert str(JWST_KERNEL_PATH) in list_registered_kernels()


def test_unregister_nonexistent_kernel():
    """Test that unregistering a non-existent kernel doesn't cause issues."""
    previous_count = len(list_registered_kernels())
    unregister_spice_kernel("nonexistent.kernel")  # Should not raise an error
    assert len(list_registered_kernels()) == previous_count


def test_jwst_kernel_loading(jwst_kernel):
    """Test loading and using the JWST SPICE kernel."""
    # Verify JWST is available in SPICE
    jwst_id = sp.bodn2c("JWST")
    assert jwst_id == -170

    # Create test times within the kernel range
    times = Timestamp.from_iso8601(["2022-01-01T00:00:00Z"])

    # Get JWST state
    state = get_spice_body_state(jwst_id, times)

    # Verify we got valid state vectors
    assert state.values.shape == (1, 6)

    # Verify the state is in the correct frame
    assert state.frame == "ecliptic"

    # Verify the origin is correct
    assert state.origin.code[0].as_py() == "SUN"


def test_jwst_kernel_time_range(jwst_kernel):
    """Test JWST kernel behavior at different times."""
    jwst_id = sp.bodn2c("JWST")

    # Test times within range
    mid_time = Timestamp.from_iso8601("2022-01-01T00:00:00Z")
    state = get_spice_body_state(jwst_id, mid_time)
    assert state.values.shape == (1, 6)

    # Test times at range boundaries (allow some margin of error)
    # Use a date a bit after start date to avoid rounding issues
    start_time = Timestamp.from_iso8601("2020-01-02T00:00:00Z")
    end_time = Timestamp.from_iso8601("2023-12-31T00:00:00Z")

    start_state = get_spice_body_state(jwst_id, start_time)
    end_state = get_spice_body_state(jwst_id, end_time)

    assert start_state.values.shape == (1, 6)
    assert end_state.values.shape == (1, 6)

    # Test time before range
    before_time = Timestamp.from_iso8601(["2019-12-31T00:00:00Z"])
    with pytest.raises(ValueError):
        get_spice_body_state(jwst_id, before_time)

    # Test time after range (should raise ValueError)
    after_time = Timestamp.from_iso8601(["2025-01-01T00:00:00Z"])
    with pytest.raises(ValueError):
        get_spice_body_state(jwst_id, after_time)


def test_jwst_kernel_cleanup(jwst_kernel):
    """Test that JWST kernel is properly cleaned up."""
    # Verify JWST is available in SPICE
    jwst_id = sp.bodn2c("JWST")
    assert jwst_id == -170

    # Save a reference time
    test_time = Timestamp.from_iso8601("2022-01-01T00:00:00Z")

    # Verify we can get state before unregistering
    state_before = get_spice_body_state(jwst_id, test_time)
    assert state_before.values.shape == (1, 6)

    # Unregister the kernel
    unregister_spice_kernel(str(JWST_KERNEL_PATH))

    # Attempting to get JWST state now should fail because kernel is unloaded
    with pytest.raises(ValueError):
        get_spice_body_state(jwst_id, test_time)

    # Re-register for cleanup by fixture
    register_spice_kernel(str(JWST_KERNEL_PATH))


def test__jd_tdb_to_et():
    # Test that _jd_tdb_to_et returns the same values as SPICE's str2et
    sp.furnsh(leapseconds)

    times = Timestamp.from_mjd(np.arange(40000, 70000, 5), scale="tdb")
    jd_tdb = times.jd().to_numpy()

    et_actual = _jd_tdb_to_et(jd_tdb)
    et_expected = np.array([sp.str2et(f"JD {i:.16f} TDB") for i in jd_tdb])

    np.testing.assert_equal(et_actual, et_expected)
