"""Jacobian declarations are explicit and capsule identity is authoritative."""

import warnings

import pytest

from adam_core.propagator.propagator import Propagator

from .._native_dispatch import _warn_if_analytic_jacobian_mismatches
from ..algorithms.fitting import fit_least_squares
from .test_differential_correction import (
    NativeTwoBodyPropagator,
    TwoBodyPropagator,
    make_initial_guess,
    make_synthetic_observations,
)

MATCH = "two-body dynamics"


class SpoofedDynamics(TwoBodyPropagator):
    """Its Python declaration disagrees with the retained native table."""

    two_body_dynamics = False


def mismatch_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [
        str(w.message)
        for w in caught
        if issubclass(w.category, UserWarning) and MATCH in str(w.message)
    ]


def test_dynamics_are_undeclared_unless_a_propagator_says_otherwise() -> None:
    assert Propagator.two_body_dynamics is False
    assert SpoofedDynamics.two_body_dynamics is False
    assert TwoBodyPropagator.two_body_dynamics is True
    assert NativeTwoBodyPropagator.two_body_dynamics is True


def test_helper_warns_only_for_analytic_on_undeclared_dynamics() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_analytic_jacobian_mismatches(TwoBodyPropagator(), "analytic")
        _warn_if_analytic_jacobian_mismatches(SpoofedDynamics(), "central")
        _warn_if_analytic_jacobian_mismatches(SpoofedDynamics(), "2-point")
    with pytest.warns(UserWarning, match=MATCH):
        _warn_if_analytic_jacobian_mismatches(SpoofedDynamics(), "analytic")


def test_capsule_identity_prevents_two_body_spoofing() -> None:
    with pytest.warns(UserWarning, match=MATCH):
        with pytest.raises(ValueError, match="identity mismatch"):
            fit_least_squares(
                make_initial_guess(),
                make_synthetic_observations(),
                SpoofedDynamics(),
                jacobian="analytic",
            )


def test_declared_two_body_capsule_runs_without_mismatch_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted, _ = fit_least_squares(
            make_initial_guess(),
            make_synthetic_observations(),
            NativeTwoBodyPropagator(),
        )
    assert len(fitted) == 1
    assert mismatch_warnings(caught) == []
