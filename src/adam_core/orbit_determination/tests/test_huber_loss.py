"""
Tests for the Huber robust loss in differential correction
(``fit_least_squares(loss="huber")``), one of the outlier-treatment
algorithms under evaluation alongside hard rejection.

Synthetic two-body observations of a known truth orbit exercise the complete
native solver with one injected gross outlier. Closed-form callback-oracle
coverage moved to the active Rust tests mapped in
``migration/od_native_oracle_coverage.md``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

from adam_core._rust.arrow import pa

from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import Origin
from ...orbits.orbits import Orbits
from ...time import Timestamp
from ..algorithms.fitting import (
    HUBER_F_SCALE_DEFAULT,
    LossType,
    _robust_cost,
    _robust_jacobian_scale,
    _robust_weights,
    _weak_direction_delta_chi2,
    fit_least_squares,
    iterative_fit,
    residual_function,
)
from ..fitted_orbits import FittedOrbitMembers, FittedOrbits
from .test_differential_correction import (
    EPOCH_MJD_TDB,
    TRUTH_STATE,
    TwoBodyPropagator,
    make_initial_guess,
    make_synthetic_observations,
)

K = HUBER_F_SCALE_DEFAULT


def state_of(fitted: FittedOrbits) -> np.ndarray:
    return np.concatenate([fitted.coordinates.r[0], fitted.coordinates.v[0]])


def orbit_at(state: np.ndarray) -> Orbits:
    return Orbits.from_kwargs(
        orbit_id=["start"],
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


def weak_direction_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    """Covariance-consistency warnings only (the synthetic covariances also
    emit unrelated NaN-off-diagonal notices)."""
    return [str(w.message) for w in caught if "weak-direction" in str(w.message)]


class TestRobustLossHelpers:
    def test_linear_cost_is_chi2_and_weights_are_one(self) -> None:
        r = np.array([0.5, -2.0, 3.0])
        assert _robust_cost(r, "linear", 1.0) == pytest.approx(float(r @ r))
        npt.assert_array_equal(_robust_weights(r, "linear", 1.0), 1.0)
        npt.assert_array_equal(_robust_jacobian_scale(r, "linear", 1.0), 1.0)

    def test_huber_cost_matches_closed_form(self) -> None:
        # Inside the core the Huber cost is chi2; in the tail it is
        # f_scale**2 * (2 |r| / f_scale - 1) per component.
        inside = np.array([0.3 * K, -0.9 * K])
        assert _robust_cost(inside, "huber", K) == pytest.approx(float(inside @ inside))
        tail = np.array([3.0 * K, -5.0 * K])
        expected = K**2 * ((2 * 3.0 - 1) + (2 * 5.0 - 1))
        assert _robust_cost(tail, "huber", K) == pytest.approx(expected)
        # The Huber cost is continuous at the transition
        assert _robust_cost(np.array([K]), "huber", K) == pytest.approx(K**2)

    def test_huber_cost_is_twice_scipy_cost(self) -> None:
        from ._scipy import scipy_least_squares

        r0 = np.array([0.2, -1.0, 2.5, -4.0])
        result = scipy_least_squares()(
            lambda x: r0 + x, np.zeros(4), loss="huber", f_scale=K, max_nfev=1
        )
        assert _robust_cost(np.asarray(result.fun), "huber", K) == pytest.approx(
            2.0 * result.cost
        )

    def test_huber_weights_and_jacobian_scale(self) -> None:
        r = np.array([0.5 * K, -K, 2.0 * K, -10.0 * K, 0.0])
        npt.assert_allclose(
            _robust_weights(r, "huber", K), [1.0, 1.0, 0.5, 0.1, 1.0], rtol=1e-12
        )
        scale = _robust_jacobian_scale(r, "huber", K)
        npt.assert_array_equal(scale[[0, 1, 4]], 1.0)
        assert np.all(scale[[2, 3]] < 1e-7)  # tail components carry no curvature


TWO_BODY_OUTLIER_OBS = 10
TWO_BODY_OUTLIER_SIGMAS = 10.0


@pytest.fixture(scope="module")
def fits() -> dict[str, Any]:
    """Linear and Huber fits of two-body observations with one 10-sigma outlier."""
    observations = make_synthetic_observations()
    lat = observations.coordinates.lat.to_numpy(zero_copy_only=False).copy()
    sigma = np.sqrt(
        observations.coordinates.covariance.to_matrix()[TWO_BODY_OUTLIER_OBS, 2, 2]
    )
    lat[TWO_BODY_OUTLIER_OBS] += TWO_BODY_OUTLIER_SIGMAS * sigma
    contaminated = observations.set_column("coordinates.lat", pa.array(lat))
    propagator = TwoBodyPropagator()
    out: dict[str, Any] = {"observations": contaminated, "propagator": propagator}
    losses: tuple[LossType, ...] = ("linear", "huber")
    for loss in losses:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted, members = fit_least_squares(
                make_initial_guess(), contaminated, propagator, loss=loss
            )
        out[loss] = (fitted, members, weak_direction_warnings(caught))
    return out


class TestTwoBodyWithOutlier:
    OUTLIER_OBS = TWO_BODY_OUTLIER_OBS

    def test_huber_recovers_truth_better_than_least_squares(
        self, fits: dict[str, Any]
    ) -> None:
        linear, _, _ = fits["linear"]
        huber, _, huber_warnings = fits["huber"]
        assert huber.success[0].as_py()
        assert huber_warnings == []
        error_linear = state_of(linear) - TRUTH_STATE
        error_huber = state_of(huber) - TRUTH_STATE
        # Position error shrinks by a factor of several (3x for this seed)
        assert np.linalg.norm(error_huber[:3]) < 0.5 * np.linalg.norm(error_linear[:3])
        covariance = huber.coordinates.covariance.to_matrix()[0]
        mahalanobis = np.sqrt(error_huber @ np.linalg.solve(covariance, error_huber))
        assert mahalanobis < 4.6

    def test_outlier_is_downweighted_not_rejected(self, fits: dict[str, Any]) -> None:
        huber, members, _ = fits["huber"]
        weights = members.weight.to_numpy(zero_copy_only=False)
        assert members.outlier.to_pylist() == [False] * len(members)
        assert huber.num_obs[0].as_py() == len(members)
        assert np.argmin(weights) == self.OUTLIER_OBS
        assert weights[self.OUTLIER_OBS] < 0.2
        residuals = residual_function(
            state_of(huber), EPOCH_MJD_TDB, fits["observations"], fits["propagator"]
        ).reshape(-1, 2)
        expected = np.minimum(1.0, K / np.abs(residuals)).min(axis=1)
        npt.assert_allclose(weights, expected, rtol=1e-9)
        # Most clean observations keep full weight
        assert np.sum(weights == 1.0) >= 0.7 * len(weights)

    def test_reported_chi2_stays_the_plain_chi2(self, fits: dict[str, Any]) -> None:
        huber, members, _ = fits["huber"]
        chi2 = np.stack(members.residuals.chi2.to_numpy(zero_copy_only=False))
        assert huber.chi2[0].as_py() == pytest.approx(float(np.sum(chi2)), rel=1e-9)

    def test_covariance_matches_robust_cost_surface(self, fits: dict[str, Any]) -> None:
        huber, _, _ = fits["huber"]
        state = state_of(huber)
        covariance = huber.coordinates.covariance.to_matrix()[0]
        residuals = residual_function(
            state, EPOCH_MJD_TDB, fits["observations"], fits["propagator"]
        )
        delta = _weak_direction_delta_chi2(
            state,
            EPOCH_MJD_TDB,
            fits["observations"],
            fits["propagator"],
            covariance,
            _robust_cost(residuals, "huber", K),
            loss="huber",
            f_scale=K,
        )
        assert delta == pytest.approx(1.0, rel=0.25)

    def test_central_jacobian_path_agrees(self, fits: dict[str, Any]) -> None:
        huber, _, _ = fits["huber"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            central, _ = fit_least_squares(
                make_initial_guess(),
                fits["observations"],
                fits["propagator"],
                loss="huber",
                jacobian="central",
            )
        sigma_analytic = np.sqrt(np.diag(huber.coordinates.covariance.to_matrix()[0]))
        sigma_central = np.sqrt(np.diag(central.coordinates.covariance.to_matrix()[0]))
        npt.assert_allclose(sigma_central, sigma_analytic, rtol=1e-3)


class TestWeightsBackCompat:
    def test_linear_fit_weights_are_one_and_zero_for_ignored(self) -> None:
        observations = make_synthetic_observations(n_obs=12)
        ignored = observations.id[2].as_py()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, members = fit_least_squares(
                make_initial_guess(),
                observations,
                TwoBodyPropagator(),
                ignore=[ignored],
            )
        weights = dict(zip(members.obs_id.to_pylist(), members.weight.to_pylist()))
        assert weights.pop(ignored) == 0.0
        assert set(weights.values()) == {1.0}
        assert members.table["weight"].null_count == 0

    def test_members_without_weights_are_null(self) -> None:
        members = FittedOrbitMembers.from_kwargs(orbit_id=["o"], obs_id=["a"])
        assert members.weight.to_pylist() == [None]


class TestIterativeFitAndFitter:
    def test_iterative_fit_huber_without_hard_rejection(self) -> None:
        observations = make_synthetic_observations(n_obs=12)
        lat = observations.coordinates.lat.to_numpy(zero_copy_only=False).copy()
        lat[4] += 10.0 * np.sqrt(
            observations.coordinates.covariance.to_matrix()[4, 2, 2]
        )
        observations = observations.set_column("coordinates.lat", pa.array(lat))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted, members = iterative_fit(
                make_initial_guess(),
                observations,
                TwoBodyPropagator(),
                contamination_percentage=0.0,
                loss="huber",
            )
        assert len(fitted) == 1
        assert members.outlier.to_pylist() == [False] * 12
        weights = members.weight.to_numpy(zero_copy_only=False)
        assert np.argmin(weights) == 4 and weights[4] < 0.5


class TestValidation:
    def test_unknown_loss_raises(self) -> None:
        with pytest.raises(ValueError, match="loss must be one of"):
            fit_least_squares(
                make_initial_guess(),
                make_synthetic_observations(n_obs=6),
                TwoBodyPropagator(),
                loss="cauchy",  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("f_scale", [0.0, -1.0, float("nan")])
    def test_bad_f_scale_raises(self, f_scale: float) -> None:
        with pytest.raises(ValueError, match="f_scale"):
            fit_least_squares(
                make_initial_guess(),
                make_synthetic_observations(n_obs=6),
                TwoBodyPropagator(),
                loss="huber",
                f_scale=f_scale,
            )

    def test_lm_method_incompatible_with_huber(self) -> None:
        with pytest.raises(ValueError, match="method='lm'"):
            fit_least_squares(
                make_initial_guess(),
                make_synthetic_observations(n_obs=6),
                TwoBodyPropagator(),
                loss="huber",
                method="lm",
            )
