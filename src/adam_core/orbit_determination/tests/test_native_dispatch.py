"""Propagation-only native capsule dispatch contracts."""

import pytest

from ...orbits import Ephemeris, Orbits, VariantOrbits
from ...propagator import Propagator
from .._native_dispatch import native_propagator
from .test_differential_correction import NativeTwoBodyPropagator, TwoBodyPropagator


class RecordingNativePropagator(NativeTwoBodyPropagator):
    """Provider OD-shaped methods are never part of the capsule contract."""

    def run_od(self, *args: object, **kwargs: object) -> None:  # pragma: no cover
        raise AssertionError("provider-owned run_od was dispatched")

    def full_od(self, *args: object, **kwargs: object) -> None:  # pragma: no cover
        raise AssertionError("provider-owned full_od was dispatched")


class PythonOnly(Propagator):
    def propagate_orbits(
        self, *args: object, **kwargs: object
    ) -> Orbits | VariantOrbits:
        raise AssertionError("Python propagation must not run")

    def generate_ephemeris(self, *args: object, **kwargs: object) -> Ephemeris:
        raise AssertionError("Python ephemeris must not run")


def test_native_capsule_is_obtained_without_provider_od_dispatch() -> None:
    assert native_propagator(RecordingNativePropagator()) is not None


def test_python_only_provider_is_rejected_without_python_execution() -> None:
    with pytest.raises(TypeError, match="native|capsule"):
        native_propagator(PythonOnly())


def test_capsule_backed_test_propagator_is_accepted() -> None:
    assert native_propagator(TwoBodyPropagator()) is not None


def test_provider_boundary_has_no_science_configuration_methods() -> None:
    capsule_provider = RecordingNativePropagator()
    assert not hasattr(capsule_provider, "observation_pipeline")
    assert not hasattr(capsule_provider, "od_config")
