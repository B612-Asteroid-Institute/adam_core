"""The external OrbitFitter science-plugin boundary is removed in Core 0.7."""

import importlib

import pytest


def test_orbit_fitter_module_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("adam_core.orbit_determination.orbit_fitter")
