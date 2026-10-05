"""The Rust solver has independent iteration and residual-evaluation limits."""

import numpy.testing as npt
import pytest

from adam_core.orbit_determination.differential_correction import (
    fit_least_squares,
    iterative_fit,
)
from adam_core.orbit_determination.native_orbit_fitter import NativeOrbitFitter
from adam_core.orbit_determination.rejection import cmc2003_fit
from adam_core.orbit_determination.tests.test_differential_correction import (
    NativeTwoBodyPropagator,
    TwoBodyPropagator,
    make_initial_guess,
    make_synthetic_observations,
)


class CountingPropagator(TwoBodyPropagator):
    def __init__(self):
        self.batches = []

    def generate_ephemeris(self, orbits, observers, **kwargs):
        self.batches.append(len(orbits))
        return super().generate_ephemeris(orbits, observers, **kwargs)


@pytest.fixture
def observations():
    return make_synthetic_observations()


@pytest.mark.parametrize("backend", [CountingPropagator, NativeTwoBodyPropagator])
@pytest.mark.parametrize(
    "jacobian, jacobian_evals", [("analytic", 0), ("2-point", 6), ("central", 12)]
)
@pytest.mark.parametrize("budget", [1, 2, 7, 8, 13, 14])
def test_evaluation_budget_caps_solver_work(
    observations, backend, jacobian, jacobian_evals, budget
):
    propagator = backend()
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
    assert 1 <= nfev <= budget
    if budget == 1:
        assert fitted.success.to_pylist() == [False]
        assert fitted.status_code.to_pylist() == [0]
        npt.assert_array_equal(fitted.coordinates.values, initial.coordinates.values)
    if isinstance(propagator, CountingPropagator):
        # Final covariance Jacobian and evaluation are separate diagnostics.
        assert sum(propagator.batches) == nfev + jacobian_evals + 1
        if budget <= jacobian_evals:
            assert nfev == 1  # Do not launch a partial numerical Jacobian.


@pytest.mark.parametrize("backend", [CountingPropagator, NativeTwoBodyPropagator])
@pytest.mark.parametrize(
    "jacobian, expected_nfev", [("analytic", 2), ("2-point", 8), ("central", 14)]
)
@pytest.mark.parametrize("max_nfev", [None, 1000])
def test_iteration_budget_is_separate(
    observations, backend, jacobian, expected_nfev, max_nfev
):
    fitted, _ = fit_least_squares(
        make_initial_guess(),
        observations,
        backend(),
        jacobian=jacobian,
        validate_covariance=False,
        max_iterations=1,
        max_nfev=max_nfev,
        xtol=0.0,
        ftol=0.0,
        gtol=0.0,
    )
    assert fitted.iterations.to_pylist() == [expected_nfev]
    assert fitted.status_code.to_pylist() == [0]
    assert fitted.success.to_pylist() == [False]


@pytest.mark.parametrize("backend", [CountingPropagator, NativeTwoBodyPropagator])
@pytest.mark.parametrize("limit", ["max_nfev", "max_iterations"])
def test_zero_solver_budget_is_rejected(observations, backend, limit):
    with pytest.raises(ValueError, match=limit):
        fit_least_squares(make_initial_guess(), observations, backend(), **{limit: 0})


@pytest.mark.parametrize("backend", [CountingPropagator, NativeTwoBodyPropagator])
@pytest.mark.parametrize("algorithm", ["worst_residual", "cmc2003"])
def test_rejection_loops_forward_the_evaluation_budget(
    observations, backend, algorithm
):
    if algorithm == "worst_residual":
        fitted, _ = iterative_fit(
            make_initial_guess(),
            observations,
            backend(),
            contamination_percentage=0,
            max_iterations=3,
            max_nfev=1,
            validate_covariance=False,
        )
    else:
        fitted, _ = cmc2003_fit(
            make_initial_guess(),
            observations,
            backend(),
            max_iterations=1,
            max_nfev=1,
            validate_covariance=False,
        )
    assert fitted.iterations.to_pylist() == [1]
    assert fitted.success.to_pylist() == [False]


def test_fitter_serializes_both_solver_limits():
    fitter = NativeOrbitFitter(
        propagator_class=TwoBodyPropagator,
        outlier_rejection="worst_residual",
        rejection_kwargs={"max_nfev": 11, "max_iterations": 3},
    )
    assert fitter.supports_native_full_od()
    _, _, settings = fitter.native_settings()
    assert settings["max_iterations"] == 3
    assert settings["max_nfev"] == 11
