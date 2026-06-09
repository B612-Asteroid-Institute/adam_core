"""Tests for the :class:`RustBackend` implementation and the module-level
backend accessor.

Parity against CSPICE is maintained upstream in the ``spicekit`` crate;
adam-core no longer links or imports CSPICE/spiceypy, so these tests
exercise only the Rust surface and the process-local accessor
semantics.
"""

from __future__ import annotations

import numpy as np
import pytest

from adam_core.utils.spice_backend import (
    NotCovered,
    RustBackend,
    get_backend,
)


@pytest.fixture(autouse=True)
def _isolate_backend():
    """Reset the module-level backend between tests so state doesn't
    leak and PID-change detection stays consistent."""
    from adam_core.utils import spice_backend as sb

    sb._BACKEND = None
    sb._BACKEND_PID = None
    try:
        yield
    finally:
        sb._BACKEND = None
        sb._BACKEND_PID = None


# ----------------------------------------------------------------------
# Text-kernel body-name bindings
# ----------------------------------------------------------------------


def test_rust_backend_picks_up_text_kernel_bindings(tmp_path):
    """Writing a minimal .tk with NAIF_BODY_NAME / NAIF_BODY_CODE
    should cause RustBackend.bodn2c to resolve the custom name after
    furnsh, and unresolve it after unload."""
    kernel_path = tmp_path / "custom_names.tk"
    kernel_path.write_text(
        "KPL/FK\n"
        "\\begindata\n"
        "  NAIF_BODY_NAME += ( 'ADAM_PROBE', 'APROBE' )\n"
        "  NAIF_BODY_CODE += ( -900001, -900001 )\n"
        "\\begintext\n"
    )
    rust = RustBackend()
    with pytest.raises(NotCovered):
        rust.bodn2c("ADAM_PROBE")
    rust.furnsh(str(kernel_path))
    assert rust.bodn2c("ADAM_PROBE") == -900001
    assert rust.bodn2c("APROBE") == -900001
    rust.unload(str(kernel_path))
    with pytest.raises(NotCovered):
        rust.bodn2c("ADAM_PROBE")


def test_text_kernel_last_loaded_wins(tmp_path):
    """When two text kernels declare the same name with different
    codes, the one loaded most recently wins (mirroring CSpice)."""
    k1 = tmp_path / "first.tk"
    k2 = tmp_path / "second.tk"
    k1.write_text(
        "\\begindata\n"
        "  NAIF_BODY_NAME += ( 'OVERRIDE_ME' )\n"
        "  NAIF_BODY_CODE += ( -1 )\n"
        "\\begintext\n"
    )
    k2.write_text(
        "\\begindata\n"
        "  NAIF_BODY_NAME += ( 'OVERRIDE_ME' )\n"
        "  NAIF_BODY_CODE += ( -2 )\n"
        "\\begintext\n"
    )
    rust = RustBackend()
    rust.furnsh(str(k1))
    assert rust.bodn2c("OVERRIDE_ME") == -1
    rust.furnsh(str(k2))
    assert rust.bodn2c("OVERRIDE_ME") == -2
    rust.unload(str(k2))
    assert rust.bodn2c("OVERRIDE_ME") == -1


def test_text_kernel_overrides_builtin(tmp_path):
    """A text kernel binding for a name that ALSO lives in the built-in
    table should win, matching CSpice's last-loaded-wins semantics."""
    k = tmp_path / "shadow.tk"
    k.write_text(
        "\\begindata\n"
        "  NAIF_BODY_NAME += ( 'EARTH' )\n"
        "  NAIF_BODY_CODE += ( -9999 )\n"
        "\\begintext\n"
    )
    rust = RustBackend()
    assert rust.bodn2c("EARTH") == 399
    rust.furnsh(str(k))
    assert rust.bodn2c("EARTH") == -9999
    rust.unload(str(k))
    assert rust.bodn2c("EARTH") == 399


# ----------------------------------------------------------------------
# NotCovered routing (no implicit fallback)
# ----------------------------------------------------------------------


def test_rust_backend_raises_notcovered_for_unknown_bodn2c():
    rust = RustBackend()
    with pytest.raises(NotCovered):
        rust.bodn2c("NOT-A-NAIF-NAME")


def test_rust_backend_raises_notcovered_for_itrf93_without_pck():
    rust = RustBackend()
    ets = np.array([0.0], dtype=np.float64)
    with pytest.raises(NotCovered):
        rust.pxform_batch("ITRF93", "J2000", ets)
    with pytest.raises(NotCovered):
        rust.sxform_batch("ITRF93", "J2000", ets)


# ----------------------------------------------------------------------
# Module-level accessor semantics
# ----------------------------------------------------------------------


def test_get_backend_returns_singleton_within_pid():
    b1 = get_backend()
    b2 = get_backend()
    assert b1 is b2


def test_get_backend_rebuilds_after_pid_change():
    from adam_core.utils import spice_backend as sb

    original = get_backend()
    sb._BACKEND_PID = -1
    rebuilt = get_backend()
    assert rebuilt is not original


def test_get_backend_returns_rust_backend():
    assert isinstance(get_backend(), RustBackend)
