"""
The Rust solver has independent iteration and residual-evaluation limits.

``max_iterations`` bounds optimizer iterations (one Jacobian each);
``max_nfev`` bounds every residual-vector evaluation: initial, numerical
Jacobian columns, and trials. ``iterations`` reports those evaluations while
``optimizer_iterations`` reports Jacobian/step iterations.
"""

import numpy.testing as npt
import pytest

from adam_core.orbit_determination.algorithms.fitting import (
    fit_least_squares,
    iterative_fit,
)
from adam_core.orbit_determination.algorithms.rejection import cmc2003_fit
from adam_core.orbit_determination.tests.test_differential_correction import (
    TwoBodyPropagator,
    make_initial_guess,
    make_synthetic_observations,
)


@pytest.fixture
def observations():
    return make_synthetic_observations()


@pytest.mark.parametrize(
    "jacobian, jacobian_evals", [("analytic", 0), ("2-point", 6), ("central", 12)]
)
@pytest.mark.parametrize("budget", [1, 2, 7, 8, 13, 14])
def test_evaluation_budget_caps_all_solver_work(
    observations, jacobian, jacobian_evals, budget
):
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
def test_iteration_budget_is_separate(observations, jacobian, expected_nfev, max_nfev):
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


def test_propagator_jacobian_phases_share_tiny_budget(observations):
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
def test_zero_solver_budget_is_rejected(observations, limit):
    with pytest.raises(ValueError, match=limit):
        fit_least_squares(
            make_initial_guess(), observations, TwoBodyPropagator(), **{limit: 0}
        )


@pytest.mark.parametrize("algorithm", ["worst_residual", "cmc2003"])
def test_rejection_loops_forward_the_evaluation_budget(observations, algorithm):
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
        fitted, _ = cmc2003_fit(
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
    observations, algorithm, kwargs, message
):
    function = iterative_fit if algorithm == "worst_residual" else cmc2003_fit
    with pytest.raises(ValueError, match=message):
        function(
            make_initial_guess(),
            observations,
            TwoBodyPropagator(),
            **kwargs,
        )
