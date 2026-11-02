"""Core 0.7 removes legacy OD aliases rather than translating them."""

import importlib

import pytest

import adam_core.orbit_determination as od
from adam_core import _rust_native


@pytest.mark.parametrize("name", ["run_iod", "run_od", "refine", "evaluate_orbits"])
def test_primary_batch_names_are_canonical_not_legacy_aliases(name: str) -> None:
    assert callable(getattr(od, name))
    assert getattr(od, name).__module__ == "adam_core.orbit_determination.workflows"


@pytest.mark.parametrize(
    "module",
    [
        "least_squares",
        "native_orbit_fitter",
        "orbit_fitter",
        "observation_uncertainty",
        "observation_models",
        "model_provenance",
        "provenance",
        "od_orchestration",
    ],
)
def test_removed_legacy_modules_have_no_compatibility_alias(module: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"adam_core.orbit_determination.{module}")


@pytest.mark.parametrize(
    "name",
    [
        "run_od_ipc",
        "run_od_2body_ipc",
        "full_od_ipc",
        "full_od_2body_ipc",
        "bias_table_model_apply_numpy",
    ],
)
def test_removed_native_model_routes_are_absent(name: str) -> None:
    assert not hasattr(_rust_native, name)
