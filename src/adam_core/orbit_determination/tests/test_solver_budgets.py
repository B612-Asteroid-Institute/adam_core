"""
The Rust solver has independent iteration and residual-evaluation limits.

``max_iterations`` bounds optimizer iterations (one Jacobian each);
``max_nfev`` bounds every residual-vector evaluation: initial, numerical
Jacobian columns, and trials. ``iterations`` reports those evaluations while
``optimizer_iterations`` reports Jacobian/step iterations.
"""

from typing import Literal

import numpy.testing as npt
import pytest
from typing_extensions import TypedDict, Unpack

from adam_core.orbit_determination.algorithms.fitting import (
    fit_least_squares,
    iterative_fit,
)
from adam_core.orbit_determination.algorithms.rejection import cmc2003_fit_detailed
from adam_core.orbit_determination.evaluate import OrbitDeterminationObservations
from adam_core.orbit_determination.fitted_orbits import FittedOrbitMembers, FittedOrbits
from adam_core.orbit_determination.tests.test_differential_correction import (
    TwoBodyPropagator,
    make_initial_guess,
    make_synthetic_observations,
)
from adam_core.orbits import Orbits
from adam_core.propagator import Propagator


class _CmcOptions(TypedDict, total=False):
    max_iterations: int
    max_nfev: int
    validate_covariance: bool


def _cmc_tables(
    orbit: Orbits,
    observations: OrbitDeterminationObservations,
    propagator: Propagator,
    **kwargs: Unpack[_CmcOptions],
) -> tuple[FittedOrbits, FittedOrbitMembers]:
    result = cmc2003_fit_detailed(orbit, observations, propagator, **kwargs)
    return result.fitted_orbit, result.fitted_orbit_members


@pytest.fixture
def observations() -> OrbitDeterminationObservations:
    return make_synthetic_observations()


@pytest.mark.parametrize(
    "jacobian, jacobian_evals", [("analytic", 0), ("2-point", 6), ("central", 12)]
)
@pytest.mark.parametrize("budget", [1, 2, 7, 8, 13, 14])
def test_evaluation_budget_caps_all_solver_work(
    observations: OrbitDeterminationObservations,
    jacobian: Literal["analytic", "2-point", "central"],
    jacobian_evals: int,
    budget: int,
) -> None:
    propagator = TwoBodyPropagator()
    initial = make_initial_guess()
    fitted, _ = fit_least_squares(
        initial,
        observations,
        propagator,
        jacobian=jacobian,
        validate_covariance=False,
        max_nfev=budget,
        max_iterations=100,
        xtol=0.0,
        ftol=0.0,
        gtol=0.0,
    )
    nfev = fitted.iterations[0].as_py()
    optimizer_iterations = fitted.optimizer_iterations[0].as_py()
    assert fitted.success.to_pylist() == [False]
    assert fitted.status_code.to_pylist() == [0]
    assert nfev <= budget
    if budget == 1:
        assert nfev == 1
        assert optimizer_iterations == 0
        npt.assert_array_equal(fitted.coordinates.values, initial.coordinates.values)
    elif jacobian_evals and budget < 1 + jacobian_evals:
        # A numerical Jacobian is atomic and is not started unless every
        # finite-difference column fits in the remaining budget.
        assert nfev == 1
        assert optimizer_iterations == 0
    else:
        assert optimizer_iterations >= 1


@pytest.mark.parametrize(
    "jacobian, expected_nfev", [("analytic", 2), ("2-point", 8), ("central", 14)]
)
@pytest.mark.parametrize("max_nfev", [None, 1000])
def test_iteration_budget_is_separate(
    observations: OrbitDeterminationObservations,
    jacobian: str,
    expected_nfev: int,
    max_nfev: int,
) -> None:
    fitted, _ = fit_least_squares(
        make_initial_guess(),
        observations,
        TwoBodyPropagator(),
        jacobian=jacobian,
        validate_covariance=False,
        max_iterations=1,
        max_nfev=max_nfev,
        xtol=0.0,
        ftol=0.0,
        gtol=0.0,
    )
    assert fitted.iterations.to_pylist() == [expected_nfev]
    assert fitted.optimizer_iterations.to_pylist() == [1]
    assert fitted.status_code.to_pylist() == [0]
    assert fitted.success.to_pylist() == [False]


def test_propagator_jacobian_phases_share_tiny_budget(
    observations: OrbitDeterminationObservations,
) -> None:
    fitted, _ = fit_least_squares(
        make_initial_guess(),
        observations,
        TwoBodyPropagator(),
        jacobian="propagator",
        validate_covariance=False,
        max_iterations=1,
        max_nfev=1,
    )
    assert fitted.iterations.to_pylist() == [1]
    assert fitted.optimizer_iterations.to_pylist() == [0]
    assert fitted.status_code.to_pylist() == [0]
    assert fitted.success.to_pylist() == [False]


@pytest.mark.parametrize("limit", ["max_nfev", "max_iterations"])
def test_zero_solver_budget_is_rejected(
    observations: OrbitDeterminationObservations,
    limit: str,
) -> None:
    with pytest.raises(ValueError, match=limit):
        if limit == "max_nfev":
            fit_least_squares(
                make_initial_guess(), observations, TwoBodyPropagator(), max_nfev=0
            )
        else:
            fit_least_squares(
                make_initial_guess(),
                observations,
                TwoBodyPropagator(),
                max_iterations=0,
            )


@pytest.mark.parametrize("algorithm", ["worst_residual", "cmc2003"])
def test_rejection_loops_forward_the_evaluation_budget(
    observations: OrbitDeterminationObservations,
    algorithm: Literal["worst_residual", "cmc2003"],
) -> None:
    if algorithm == "worst_residual":
        fitted, _ = iterative_fit(
            make_initial_guess(),
            observations,
            TwoBodyPropagator(),
            contamination_percentage=0,
            max_iterations=3,
            max_nfev=1,
            validate_covariance=False,
        )
    else:
        fitted, _ = _cmc_tables(
            make_initial_guess(),
            observations,
            TwoBodyPropagator(),
            max_iterations=1,
            max_nfev=1,
            validate_covariance=False,
        )
    assert fitted.iterations.to_pylist() == [1]
    assert fitted.optimizer_iterations.to_pylist() == [0]
    assert fitted.success.to_pylist() == [False]


@pytest.mark.parametrize(
    "algorithm, kwargs, message",
    [
        ("worst_residual", {"min_obs": 2}, "min_obs"),
        (
            "worst_residual",
            {"contamination_percentage": float("nan")},
            "contamination_percentage",
        ),
        ("cmc2003", {"max_iterations": 0}, "cmc2003.max_iterations"),
        (
            "cmc2003",
            {"chi2_reject": 1.0, "chi2_recover": 2.0},
            "chi2_reject",
        ),
    ],
)
def test_refinement_settings_are_validated_by_rust(
    observations: OrbitDeterminationObservations,
    algorithm: Literal["worst_residual", "cmc2003"],
    kwargs: dict[str, int | float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        if algorithm == "worst_residual":
            if "min_obs" in kwargs:
                iterative_fit(
                    make_initial_guess(),
                    observations,
                    TwoBodyPropagator(),
                    min_obs=int(kwargs["min_obs"]),
                )
            else:
                iterative_fit(
                    make_initial_guess(),
                    observations,
                    TwoBodyPropagator(),
                    contamination_percentage=float(kwargs["contamination_percentage"]),
                )
        elif "max_iterations" in kwargs:
            _cmc_tables(
                make_initial_guess(),
                observations,
                TwoBodyPropagator(),
                max_iterations=int(kwargs["max_iterations"]),
            )
        else:
            cmc2003_fit_detailed(
                make_initial_guess(),
                observations,
                TwoBodyPropagator(),
                chi2_reject=float(kwargs["chi2_reject"]),
                chi2_recover=float(kwargs["chi2_recover"]),
            )
