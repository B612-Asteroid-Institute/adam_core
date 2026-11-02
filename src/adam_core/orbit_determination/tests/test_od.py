"""The legacy generic ``od`` facade is not part of the 0.7 foundation."""

import importlib

import pytest

import adam_core.orbit_determination as od_package


def test_generic_od_module_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("adam_core.orbit_determination.od")


def test_no_generic_od_alias_is_exported() -> None:
    assert not hasattr(od_package, "od")
