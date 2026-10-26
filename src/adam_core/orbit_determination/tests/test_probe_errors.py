"""
The covariance probe must not swallow backend errors (PR #217 review round
2, phase 1, item 2).

On the callback route a Python exception raised by the propagator while the
weak-direction probe evaluates the +/- 1-sigma displacement used to be
converted into a "covariance could not be validated" warning, after which
the next prediction cleared the stored exception. It now propagates
unchanged, like an exception raised during any other prediction.
"""

from __future__ import annotations

import warnings

import pytest

from ..differential_correction import _fit_least_squares as fit_least_squares
from .test_differential_correction import (
    TwoBodyPropagator,
    make_initial_guess,
    make_synthetic_observations,
)


class ProbeFailure(RuntimeError):
    """Marker exception type: must reach the caller as this type."""


class RaisesOnProbe(TwoBodyPropagator):
    """
    Raises only on the probe: with the analytic Jacobian every solver call
    predicts one candidate, the probe is the only two-candidate batch.
    """

    def __init__(self) -> None:
        self.calls: list[int] = []

    def generate_ephemeris(self, orbits, observers, **kwargs):
        self.calls.append(len(orbits))
        if len(orbits) == 2:
            raise ProbeFailure("probe displacement rejected by the backend")
        return super().generate_ephemeris(orbits, observers, **kwargs)


class RaisesOnSecondCall(TwoBodyPropagator):
    """Raises on its second prediction (a solver trial step)."""

    def __init__(self) -> None:
        self.calls = 0

    def generate_ephemeris(self, orbits, observers, **kwargs):
        self.calls += 1
        if self.calls == 2:
            raise ProbeFailure("second prediction rejected by the backend")
        return super().generate_ephemeris(orbits, observers, **kwargs)


def test_probe_exception_propagates_unchanged_without_warning() -> None:
    propagator = RaisesOnProbe()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ProbeFailure, match="probe displacement"):
            fit_least_squares(
                make_initial_guess(), make_synthetic_observations(), propagator
            )
    assert 2 in propagator.calls, "the probe batch was never requested"
    assert not [w for w in caught if "could not be validated" in str(w.message)]


def test_solver_step_exception_propagates_unchanged() -> None:
    propagator = RaisesOnSecondCall()
    with pytest.raises(ProbeFailure, match="second prediction"):
        fit_least_squares(
            make_initial_guess(), make_synthetic_observations(), propagator
        )
    assert propagator.calls == 2


def test_clean_propagator_still_validates_the_covariance() -> None:
    """Control: the probe runs and the fit succeeds when nothing raises."""
    propagator = RaisesOnProbe.__mro__[1]()  # plain TwoBodyPropagator
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fitted, _ = fit_least_squares(
            make_initial_guess(), make_synthetic_observations(), propagator
        )
    assert fitted.success[0].as_py()
