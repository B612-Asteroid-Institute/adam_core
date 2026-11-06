"""NativeOrbitFitter settings dictionaries are replaced by typed config."""

import importlib

import pytest

from ..config import IodStageConfig, RefinementRunConfig


def test_native_orbit_fitter_module_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("adam_core.orbit_determination.native_orbit_fitter")


def test_typed_native_stage_declarations_replace_fitter_settings() -> None:
    assert IodStageConfig.core_v1()._native.validate() is None
    assert RefinementRunConfig.core_v1()._native.validate() is None
