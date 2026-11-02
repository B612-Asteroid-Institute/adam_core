"""Configuration reconstruction is native, typed, and non-executable."""

import pytest

from ..config import IodRunConfig, RefinementRunConfig
from ..profiles import OdProfile


def test_native_stage_round_trips_are_value_preserving() -> None:
    iod = IodRunConfig.core_v1()
    refinement = RefinementRunConfig.core_v1()
    assert IodRunConfig._from_native(iod._native) == iod
    assert RefinementRunConfig._from_native(refinement._native) == refinement


def test_profiles_are_nonconstructible_native_views() -> None:
    with pytest.raises(TypeError):
        OdProfile()  # type: ignore[call-arg]


def test_configuration_contains_no_callback_or_executable_model() -> None:
    encoded = IodRunConfig.core_v1()._native.canonical_json
    assert "callback" not in encoded
    assert "python" not in encoded.lower()
