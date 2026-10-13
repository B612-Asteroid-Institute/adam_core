"""
The analytic two-body Jacobian is audible on undeclared dynamics (PR #217
review round 2): a propagator declares exact Kepler two-body dynamics with
``Propagator.two_body_dynamics = True``; any other propagator asked for
``jacobian="analytic"`` gets a `UserWarning` from every OD entry point,
on the fused and the callback route alike.
"""

import warnings

import pytest

from adam_core.propagator.propagator import Propagator

from .._native_dispatch import _warn_if_analytic_jacobian_mismatches
from ..differential_correction import fit_least_squares, iterative_fit
from ..native_orbit_fitter import NativeOrbitFitter
from ..od_orchestration import run_od
from ..rejection import cmc2003_fit_detailed
from .test_differential_correction import (
    NativeTwoBodyPropagator,
    TwoBodyPropagator,
    make_initial_guess,
    make_synthetic_observations,
)
from .test_run_od_null_covariance import two_body_arc  # noqa: F401

MATCH = "two-body dynamics"


class UndeclaredDynamics(TwoBodyPropagator):
    """Two-body in fact, but silent about it: a plugin propagator as seen
    from adam_core."""

    two_body_dynamics = False


class UndeclaredNativeDynamics(NativeTwoBodyPropagator):
    """The same, posing as a Rust-backed propagator (fused route)."""

    two_body_dynamics = False


def mismatch_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [
        str(w.message)
        for w in caught
        if issubclass(w.category, UserWarning) and MATCH in str(w.message)
    ]


def test_dynamics_are_undeclared_unless_a_propagator_says_otherwise() -> None:
    assert Propagator.two_body_dynamics is False
    assert UndeclaredDynamics.two_body_dynamics is False
    assert TwoBodyPropagator.two_body_dynamics is True
    assert NativeTwoBodyPropagator.two_body_dynamics is True


def test_helper_warns_only_for_the_analytic_jacobian_on_undeclared_dynamics() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_analytic_jacobian_mismatches(TwoBodyPropagator(), "analytic")
        _warn_if_analytic_jacobian_mismatches(UndeclaredDynamics(), "central")
        _warn_if_analytic_jacobian_mismatches(UndeclaredDynamics(), "2-point")
    with pytest.warns(UserWarning, match=MATCH):
        _warn_if_analytic_jacobian_mismatches(UndeclaredDynamics(), "analytic")


@pytest.mark.parametrize(
    "entry", ["fit_least_squares", "iterative_fit", "cmc2003_fit_detailed"]
)
def test_callback_route_entry_points_warn_once(entry: str) -> None:
    orbit = make_initial_guess()
    observations = make_synthetic_observations()
    run = {
        "fit_least_squares": lambda p, **kw: fit_least_squares(
            orbit, observations, p, **kw
        ),
        "iterative_fit": lambda p, **kw: iterative_fit(orbit, observations, p, **kw),
        "cmc2003_fit_detailed": lambda p, **kw: cmc2003_fit_detailed(
            orbit, observations, p, **kw
        ),
    }[entry]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run(UndeclaredDynamics())
    assert len(mismatch_warnings(caught)) == 1, mismatch_warnings(caught)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run(TwoBodyPropagator())
        run(UndeclaredDynamics(), jacobian="central")
    assert mismatch_warnings(caught) == []


def test_fused_route_warns_too(two_body_arc) -> None:  # noqa: F811
    orbit = make_initial_guess()
    observations = make_synthetic_observations()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_least_squares(orbit, observations, UndeclaredNativeDynamics())
    assert len(mismatch_warnings(caught)) == 1

    fitter = NativeOrbitFitter(
        propagator_class=UndeclaredNativeDynamics, iod_rchi2_threshold=1e4
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted, _ = run_od(
            two_body_arc, fitter, None, propagator=UndeclaredNativeDynamics()
        )
    assert len(fitted) == 1
    assert len(mismatch_warnings(caught)) == 1

    declared = NativeOrbitFitter(
        propagator_class=NativeTwoBodyPropagator, iod_rchi2_threshold=1e4
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_od(two_body_arc, declared, None, propagator=NativeTwoBodyPropagator())
    assert mismatch_warnings(caught) == []
