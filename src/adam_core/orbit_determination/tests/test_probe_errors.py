"""Python prediction overrides are never called from native OD."""

from __future__ import annotations

import warnings

import pytest

from ..algorithms.fitting import _fit_least_squares as fit_least_squares
from .test_differential_correction import (
    TwoBodyPropagator,
    make_initial_guess,
    make_synthetic_observations,
)


class PythonPredictionOverride(TwoBodyPropagator):
    called = False

    def generate_ephemeris(self, orbits, observers, **kwargs):
        type(self).called = True
        raise AssertionError("Python prediction callback ran")


def test_python_prediction_override_is_refused_before_it_runs() -> None:
    PythonPredictionOverride.called = False
    with pytest.raises(TypeError, match="not covered"):
        fit_least_squares(
            make_initial_guess(),
            make_synthetic_observations(),
            PythonPredictionOverride(),
        )
    assert not PythonPredictionOverride.called


def test_capsule_provider_still_validates_the_covariance() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fitted, _ = fit_least_squares(
            make_initial_guess(), make_synthetic_observations(), TwoBodyPropagator()
        )
    assert fitted.success[0].as_py()
