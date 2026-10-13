"""
The native corrector's contract (bead od_experiments_setup-a6w.1): it is
scipy's trust-region-reflective solver without bounds, it never reports
convergence on a non-finite derivative, and its status codes mean one thing.
"""

import warnings

import numpy as np
import numpy.testing as npt
import pytest
from scipy.optimize import least_squares

from adam_core.coordinates import CartesianCoordinates
from adam_core.coordinates.origin import Origin
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

from ..differential_correction import (
    _analytic_jacobian,
    _analytic_jacobian_terms,
    fit_least_squares,
    residual_function,
)
from .test_differential_correction import (
    EPOCH_MJD_TDB,
    TRUTH_STATE,
    TwoBodyPropagator,
    make_synthetic_observations,
)


def orbit_from_state(state: np.ndarray) -> Orbits:
    return Orbits.from_kwargs(
        orbit_id=["seed"],
        object_id=["synthetic"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=state[0:1],
            y=state[1:2],
            z=state[2:3],
            vx=state[3:4],
            vy=state[4:5],
            vz=state[5:6],
            time=Timestamp.from_mjd([EPOCH_MJD_TDB], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )


def fitted_state(fitted) -> np.ndarray:
    return np.concatenate([fitted.coordinates.r[0], fitted.coordinates.v[0]])


class NanJacobianPropagator(TwoBodyPropagator):
    """Finite predictions for one orbit; NaN for any multi-orbit batch, which
    is what a finite-difference Jacobian asks for."""

    def generate_ephemeris(self, orbits, observers, **kwargs):
        ephemeris = super().generate_ephemeris(orbits, observers, **kwargs)
        if len(orbits) == 1:
            return ephemeris
        n = len(ephemeris)
        return ephemeris.set_column("coordinates.lon", np.full(n, np.nan)).set_column(
            "coordinates.lat", np.full(n, np.nan)
        )


@pytest.mark.parametrize("perturbation_scale", [1.0, 30.0])
def test_native_solver_reaches_scipy_trf_minimum(perturbation_scale: float) -> None:
    """From the same seed, the Rust solver and scipy's trf solver (the Python
    baseline's, x_scale='jac', analytic Jacobian) end at the same minimum."""
    observations = make_synthetic_observations()
    propagator = TwoBodyPropagator()
    seed = TRUTH_STATE + perturbation_scale * np.array(
        [1e-5, -8e-6, 5e-6, 1e-8, -1e-8, 5e-9]
    )
    terms = _analytic_jacobian_terms(observations)
    reference = least_squares(
        residual_function,
        seed,
        jac=lambda x, *_: _analytic_jacobian(x, EPOCH_MJD_TDB, terms),
        args=(EPOCH_MJD_TDB, observations, propagator),
        method="trf",
        x_scale="jac",
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    assert reference.success

    fitted, _ = fit_least_squares(
        orbit_from_state(seed), observations, propagator, validate_covariance=False
    )
    assert fitted.success[0].as_py(), fitted.status_code[0].as_py()
    assert fitted.status_code[0].as_py() in (1, 2, 3, 4)
    npt.assert_allclose(fitted.chi2[0].as_py(), 2.0 * reference.cost, rtol=1e-9)
    state = fitted_state(fitted)
    npt.assert_allclose(state[:3], reference.x[:3], atol=1e-10)
    npt.assert_allclose(state[3:], reference.x[3:], atol=1e-12)


def test_non_finite_jacobian_is_not_success() -> None:
    observations = make_synthetic_observations()
    seed = TRUTH_STATE + np.array([1e-5, -8e-6, 5e-6, 1e-8, -1e-8, 5e-9])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fitted, _ = fit_least_squares(
            orbit_from_state(seed),
            observations,
            NanJacobianPropagator(),
            jacobian="central",
            validate_covariance=False,
        )
    assert not fitted.success[0].as_py()
    assert fitted.status_code[0].as_py() == -2
    npt.assert_array_equal(fitted_state(fitted), seed)


def test_limits_report_status_zero_without_success() -> None:
    observations = make_synthetic_observations()
    seed = TRUTH_STATE + 30.0 * np.array([1e-5, -8e-6, 5e-6, 1e-8, -1e-8, 5e-9])
    propagator = TwoBodyPropagator()
    fitted, _ = fit_least_squares(
        orbit_from_state(seed),
        observations,
        propagator,
        validate_covariance=False,
        max_iterations=1,
        xtol=0.0,
        ftol=0.0,
        gtol=0.0,
    )
    assert not fitted.success[0].as_py()
    assert fitted.status_code[0].as_py() == 0
    assert fitted.iterations[0].as_py() == 2  # initial evaluation + one trial

    # max_nfev bounds trials, not finite-difference columns: one central
    # Jacobian (12 evaluations) plus the initial and one trial evaluation.
    fitted, _ = fit_least_squares(
        orbit_from_state(seed),
        observations,
        propagator,
        jacobian="central",
        validate_covariance=False,
        max_nfev=2,
        xtol=0.0,
        ftol=0.0,
        gtol=0.0,
    )
    assert not fitted.success[0].as_py()
    assert fitted.status_code[0].as_py() == 0
    assert fitted.iterations[0].as_py() == 14
