from pathlib import Path

import numpy as np
import pytest

from ...coordinates.origin import OriginCodes
from ...time import Timestamp
from ..spice import (
    DEFAULT_KERNELS,
    _jd_tdb_to_et,
    clear_spkez_cache,
    get_perturber_state,
    get_spice_body_state,
    list_registered_kernels,
    register_spice_kernel,
    setup_SPICE,
    unregister_spice_kernel,
)
from ..spice_backend import get_backend

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
    # Verify JWST is available in the backend
    jwst_id = get_backend().bodn2c("JWST")
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
    jwst_id = get_backend().bodn2c("JWST")

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
    backend = get_backend()
    jwst_id = backend.bodn2c("JWST")
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
    # TDB → ET is pure arithmetic: `ET = (JD_TDB - 2451545.0) * 86400`.
    # This is the closed-form identity both CSpice and adam-core use; no
    # kernel required. Verifying the closed-form directly removes the
    # last spiceypy dependency in this file.
    times = Timestamp.from_mjd(np.arange(40000, 70000, 5), scale="tdb")
    jd_tdb = times.jd().to_numpy()

    et_actual = _jd_tdb_to_et(jd_tdb)
    et_expected = (jd_tdb - 2451545.0) * 86400.0

    np.testing.assert_equal(et_actual, et_expected)


def _force_legacy_perturber_path(monkeypatch):
    """Route get_perturber_state onto the retained legacy composition so the
    Python cache internals under test are actually exercised (the fused Rust
    crossing has its own Rust-side cache)."""
    from .. import spice as spice_mod

    def _raise(*args, **kwargs):
        raise RuntimeError("forced legacy path for cache test")

    monkeypatch.setattr(spice_mod, "_rust_backend_perturber_states", _raise)


def test_get_perturber_state_spkez_cache(monkeypatch):
    from .. import spice as spice_mod

    clear_spkez_cache()
    setup_SPICE(force=True)
    _force_legacy_perturber_path(monkeypatch)

    # Use a time grid with duplicates to ensure both in-call uniquing and cross-call caching.
    t = Timestamp.from_mjd(
        np.array([60000.0, 60000.0, 60000.5, 60001.0, 60001.0]), scale="tdb"
    )

    calls = {"n": 0}
    orig_query = spice_mod._query_states_km_kms_batch

    def _query_counted(*args, **kwargs):
        calls["n"] += 1
        return orig_query(*args, **kwargs)

    monkeypatch.setattr(spice_mod, "_query_states_km_kms_batch", _query_counted)

    _ = get_perturber_state(
        OriginCodes.SUN, t, frame="ecliptic", origin=OriginCodes.SOLAR_SYSTEM_BARYCENTER
    )
    n_first = int(calls["n"])
    assert n_first > 0

    # Second call should be entirely served from cache.
    _ = get_perturber_state(
        OriginCodes.SUN, t, frame="ecliptic", origin=OriginCodes.SOLAR_SYSTEM_BARYCENTER
    )
    assert int(calls["n"]) == n_first


def test_get_perturber_state_reverse_pair_cache(monkeypatch):
    from .. import spice as spice_mod

    clear_spkez_cache()
    setup_SPICE(force=True)
    _force_legacy_perturber_path(monkeypatch)

    t = Timestamp.from_mjd(np.array([60000.0, 60000.5, 60001.0]), scale="tdb")

    calls = {"n": 0}
    orig_query = spice_mod._query_states_km_kms_batch

    def _query_counted(*args, **kwargs):
        calls["n"] += 1
        return orig_query(*args, **kwargs)

    monkeypatch.setattr(spice_mod, "_query_states_km_kms_batch", _query_counted)

    sun_wrt_ssb = get_perturber_state(
        OriginCodes.SUN, t, frame="ecliptic", origin=OriginCodes.SOLAR_SYSTEM_BARYCENTER
    ).values
    n_first = int(calls["n"])
    assert n_first > 0

    # Reverse query should not trigger additional SPICE calls and should be exact negation.
    ssb_wrt_sun = get_perturber_state(
        OriginCodes.SOLAR_SYSTEM_BARYCENTER, t, frame="ecliptic", origin=OriginCodes.SUN
    ).values
    assert int(calls["n"]) == n_first
    np.testing.assert_allclose(ssb_wrt_sun, -sun_wrt_ssb, rtol=0.0, atol=0.0)


def test_get_perturber_state_fused_matches_legacy(monkeypatch):
    # The fused Rust crossing must reproduce the retained legacy composition
    # bit-for-bit: same TDB rescale, ET arithmetic, readers, and unit order.
    from .. import spice as spice_mod

    setup_SPICE(force=True)
    times_by_scale = [
        Timestamp.from_mjd(np.array([60000.0, 60000.0, 60000.5, 60001.0]), scale="tdb"),
        Timestamp.from_mjd(np.array([60000.0, 60001.25]), scale="utc"),
    ]
    cases = [
        (OriginCodes.SUN, OriginCodes.SOLAR_SYSTEM_BARYCENTER),
        (OriginCodes.EARTH, OriginCodes.SUN),
    ]
    for t in times_by_scale:
        for frame in ("ecliptic", "equatorial"):
            for target, origin in cases:
                clear_spkez_cache()
                fused = get_perturber_state(target, t, frame=frame, origin=origin)
                clear_spkez_cache()
                with monkeypatch.context() as ctx:

                    def _raise(*args, **kwargs):
                        raise RuntimeError("forced legacy")

                    ctx.setattr(spice_mod, "_rust_backend_perturber_states", _raise)
                    legacy = get_perturber_state(target, t, frame=frame, origin=origin)
                np.testing.assert_array_equal(fused.values, legacy.values)
                assert fused.frame == legacy.frame
                assert fused.origin.code.to_pylist() == legacy.origin.code.to_pylist()
                assert fused.time.days.to_pylist() == legacy.time.days.to_pylist()
                assert fused.time.nanos.to_pylist() == legacy.time.nanos.to_pylist()
                assert fused.time.scale == legacy.time.scale


def test_get_spice_body_state_fused_matches_legacy(monkeypatch):
    from .. import spice as spice_mod

    setup_SPICE(force=True)
    t = Timestamp.from_mjd(np.array([60000.0, 60000.5, 60001.0]), scale="tdb")
    for frame in ("ecliptic", "equatorial"):
        fused = get_spice_body_state(399, t, frame=frame, origin=OriginCodes.SUN)
        with monkeypatch.context() as ctx:

            def _raise(*args, **kwargs):
                raise RuntimeError("forced legacy")

            ctx.setattr(spice_mod, "_rust_backend_spice_body_states", _raise)
            legacy = get_spice_body_state(399, t, frame=frame, origin=OriginCodes.SUN)
        np.testing.assert_array_equal(fused.values, legacy.values)
        assert fused.frame == legacy.frame
        assert fused.origin.code.to_pylist() == legacy.origin.code.to_pylist()

    # Unknown body preserves the exact legacy wrapped ValueError.
    with pytest.raises(ValueError, match="Could not get state data for body ID"):
        get_spice_body_state(-987654, t)


def test_perturber_and_body_state_native_timing():
    import pyarrow as pa

    from ..spice import get_backend

    setup_SPICE(force=True)
    t = Timestamp.from_mjd(np.array([60000.0, 60000.5, 60001.0]), scale="tdb")
    time_table = t.table.combine_chunks()
    batch = pa.RecordBatch.from_arrays(
        [time_table.column("days").chunk(0), time_table.column("nanos").chunk(0)],
        names=["days", "nanos"],
    )
    backend = get_backend()
    perturber_samples = backend.benchmark_perturber_states_arrow_rust(
        batch,
        "tdb",
        int(OriginCodes.SUN.value),
        int(OriginCodes.SOLAR_SYSTEM_BARYCENTER.value),
        "SOLAR_SYSTEM_BARYCENTER",
        "ecliptic",
        200_000,
        2,
        2,
        1,
    )
    body_samples = backend.benchmark_spice_body_states_arrow_rust(
        batch, "tdb", 399, "SUN", "ecliptic", 2, 2, 1
    )
    assert all(s > 0.0 for trial in perturber_samples for s in trial)
    assert all(s > 0.0 for trial in body_samples for s in trial)
