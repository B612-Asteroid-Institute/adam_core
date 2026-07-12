"""Contract tests for the abstract propagator framework.

adam_core ships no concrete propagator and no Python composition: only
Rust-backed propagators (e.g. ``adam_assist.ASSISTPropagator``) are
supported. These tests assert the abstract single-crossing contract
(``propagate_orbits`` / ``generate_ephemeris`` are abstract and enforced).

Behavioral coverage -- covariance sample/propagate/collapse, the light-time /
aberration / photometry ephemeris pipeline, variant handling, and parallelism
-- now lives inside the Rust backend and is validated by the backend suites in
the downstream adam-assist Rust/parity suites (propagate, covariance,
ephemeris, impacts) and by this repository's cross-package integration harness.
"""

import pytest

from ..propagator import EphemerisMixin, Propagator


def test_propagator_is_abstract() -> None:
    with pytest.raises(TypeError):
        Propagator()  # type: ignore[abstract]


def test_propagator_requires_propagate_orbits() -> None:
    class OnlyEphemeris(Propagator):
        def generate_ephemeris(self, orbits, observers, **kwargs):
            return None

    with pytest.raises(TypeError):
        OnlyEphemeris()  # type: ignore[abstract]


def test_propagator_requires_generate_ephemeris() -> None:
    class OnlyPropagate(Propagator):
        def propagate_orbits(self, orbits, times, **kwargs):
            return orbits

    with pytest.raises(TypeError):
        OnlyPropagate()  # type: ignore[abstract]


def test_minimal_concrete_propagator_instantiates() -> None:
    class Minimal(Propagator):
        def propagate_orbits(self, orbits, times, **kwargs):
            return orbits

        def generate_ephemeris(self, orbits, observers, **kwargs):
            return None

    prop = Minimal()
    assert isinstance(prop, Propagator)
    assert isinstance(prop, EphemerisMixin)


def test_ephemeris_mixin_generate_ephemeris_is_abstract() -> None:
    assert getattr(EphemerisMixin.generate_ephemeris, "__isabstractmethod__", False)


def test_rust_backend_satisfies_contract() -> None:
    """The in-repo Rust backend is a concrete Propagator implementing the
    single-crossing contract."""
    assist = pytest.importorskip("adam_assist")
    prop = assist.ASSISTPropagator()
    assert callable(prop.propagate_orbits)
    assert callable(prop.generate_ephemeris)
    assert callable(prop.detect_collisions)

    # The public Python methods are veneers over one compiled PyO3 owner, not
    # inherited composition from adam-core's abstract contracts.
    native = prop._native
    assert type(native).__name__ == "NativeAssistPropagator"
    assert type(native).__module__ == "builtins"
    assert callable(native.propagate_orbits)
    assert callable(native.generate_ephemeris)
    assert callable(native.detect_collisions)
