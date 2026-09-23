"""
Tests for the Huber robust loss in differential correction
(``fit_least_squares(loss="huber")``), one of the outlier-treatment
algorithms under evaluation alongside hard rejection.

Synthetic inputs only: an exactly linear problem with a hand-computable
covariance (from ``test_differential_correction``) and two-body observations
of a known truth orbit, each with one injected gross outlier.
"""

from __future__ import annotations

import pickle
import warnings
from typing import Any, cast

import numpy as np
import numpy.testing as npt
import pyarrow as pa
import pytest

from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import Origin
from ...orbits.orbits import Orbits
from ...propagator.propagator import Propagator
from ...time import Timestamp
from .. import native_orbit_fitter as native_module
from ..differential_correction import (
    HUBER_F_SCALE_DEFAULT,
    LossType,
    _robust_cost,
    _robust_jacobian_scale,
    _robust_weights,
    _validated_covariance,
    _weak_direction_delta_chi2,
    fit_least_squares,
    iterative_fit,
    residual_function,
)
from ..fitted_orbits import FittedOrbitMembers, FittedOrbits
from ..native_orbit_fitter import NativeOrbitFitter
from .test_differential_correction import (
    EPOCH_MJD_TDB,
    TRUTH_STATE,
    TwoBodyPropagator,
    make_initial_guess,
    make_linear_problem,
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
        from scipy.optimize import least_squares

        r0 = np.array([0.2, -1.0, 2.5, -4.0])
        result = least_squares(
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


class TestLinearProblemWithOutlier:
    """Exactly linear residuals: the Huber M-estimate and its covariance are
    known in closed form."""

    OUTLIER_OBS = 3
    OUTLIER_SIGMAS = 20.0

    def make_problem(self) -> tuple[Any, Propagator, np.ndarray, np.ndarray, int]:
        observations, linear_propagator, x_star, expected = make_linear_problem()
        # Duck-typed fake propagator (generate_ephemeris only)
        propagator = cast(Propagator, linear_propagator)
        # Inject a gross latitude outlier; predictions stay anchored to the
        # clean observations so the injected offset is the only residual at x*.
        lat = observations.coordinates.lat.to_numpy(zero_copy_only=False).copy()
        sigma_lat = np.sqrt(observations.coordinates.covariance.to_matrix()[:, 2, 2])
        lat[self.OUTLIER_OBS] += self.OUTLIER_SIGMAS * sigma_lat[self.OUTLIER_OBS]
        contaminated = observations.set_column("coordinates.lat", pa.array(lat))
        component = 2 * self.OUTLIER_OBS + 1
        return contaminated, propagator, x_star, expected, component

    def fit(self, loss: LossType) -> tuple[FittedOrbits, FittedOrbitMembers, list[str]]:
        observations, propagator, x_star, _, _ = self.make_problem()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted, members = fit_least_squares(
                orbit_at(x_star),
                observations,
                propagator,
                jacobian="central",
                loss=loss,
                f_scale=K,
            )
        return fitted, members, weak_direction_warnings(caught)

    def test_huber_bounds_the_outlier_influence(self) -> None:
        _, _, x_star, expected, _ = self.make_problem()
        linear, _, _ = self.fit("linear")
        huber, _, _ = self.fit("huber")

        def shift(fitted: FittedOrbits) -> float:
            error = state_of(fitted) - x_star
            return float(np.sqrt(error @ np.linalg.solve(expected, error)))

        # Least squares is dragged by the 20-sigma outlier; the Huber estimate
        # feels it only as a bounded (f_scale-sized) pull.
        assert shift(linear) > 1.0
        assert shift(huber) < 0.3 * shift(linear)

    def test_huber_members_report_weights_not_rejections(self) -> None:
        observations, propagator, _, _, component = self.make_problem()
        fitted, members, _ = self.fit("huber")

        assert members.outlier.to_pylist() == [False] * len(members)
        assert members.solution.to_pylist() == [True] * len(members)
        weights = members.weight.to_numpy(zero_copy_only=False)
        residuals = residual_function(
            state_of(fitted), EPOCH_MJD_TDB, observations, propagator
        )
        expected_weight = K / abs(residuals[component])
        assert expected_weight < 0.2
        assert weights[self.OUTLIER_OBS] == pytest.approx(expected_weight, rel=1e-9)
        mask = np.ones(len(weights), dtype=bool)
        mask[self.OUTLIER_OBS] = False
        npt.assert_array_equal(weights[mask], 1.0)

    def test_huber_covariance_drops_the_outlier_information(self) -> None:
        # In the linear tail the Huber loss has no curvature, so the covariance
        # equals the weighted least-squares covariance without that component.
        observations, _, _, _, component = self.make_problem()
        _, prop_clean, _, _ = make_linear_problem()
        sigma_flat = np.empty(2 * len(observations))
        covariances = observations.coordinates.covariance.to_matrix()
        sigma_flat[0::2] = np.sqrt(covariances[:, 1, 1])
        sigma_flat[1::2] = np.sqrt(covariances[:, 2, 2])
        whitened_design = prop_clean.design_matrix / sigma_flat[:, None]
        keep = np.ones(len(sigma_flat), dtype=bool)
        keep[component] = False
        expected = np.linalg.inv(whitened_design[keep].T @ whitened_design[keep])

        fitted, _, probe_warnings = self.fit("huber")
        covariance = fitted.coordinates.covariance.to_matrix()[0]
        npt.assert_allclose(covariance, expected, rtol=1e-4)
        # ... and that covariance is consistent with the robust cost surface.
        assert probe_warnings == []

    def test_probe_uses_robust_cost(self) -> None:
        observations, propagator, _, _, _ = self.make_problem()
        fitted, _, _ = self.fit("huber")
        state = state_of(fitted)
        covariance = fitted.coordinates.covariance.to_matrix()[0]
        residuals = residual_function(state, EPOCH_MJD_TDB, observations, propagator)
        delta = _weak_direction_delta_chi2(
            state,
            EPOCH_MJD_TDB,
            observations,
            propagator,
            covariance,
            _robust_cost(residuals, "huber", K),
            loss="huber",
            f_scale=K,
        )
        assert delta == pytest.approx(1.0, rel=1e-3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _validated_covariance(
                covariance,
                "central",
                state,
                EPOCH_MJD_TDB,
                observations,
                propagator,
                _robust_cost(residuals, "huber", K),
                loss="huber",
                f_scale=K,
                residuals_solution=residuals,
            )
        npt.assert_array_equal(result, covariance)
        assert weak_direction_warnings(caught) == []


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

    def test_native_fitter_forwards_loss_and_pickles(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fitter = NativeOrbitFitter(
            propagator_class=TwoBodyPropagator, loss="huber", f_scale=2.0
        )
        restored = pickle.loads(pickle.dumps(fitter))
        assert restored.loss == "huber" and restored.f_scale == 2.0

        captured: dict[str, Any] = {}

        def fake_iterative_fit(
            orbit: Any, observations: Any, propagator: Any, **kwargs: Any
        ) -> Any:
            captured.update(kwargs)
            return FittedOrbits.empty(), FittedOrbitMembers.empty()

        monkeypatch.setattr(native_module, "iterative_fit", fake_iterative_fit)
        fitted = FittedOrbits.from_kwargs(
            orbit_id=["o"],
            coordinates=orbit_at(TRUTH_STATE).coordinates,
            arc_length=[1.0],
            num_obs=[6],
            chi2=[1.0],
            reduced_chi2=[0.5],
        )
        fitter.refine_fit(
            fitted, make_synthetic_observations(n_obs=6), TwoBodyPropagator()
        )
        assert captured["loss"] == "huber"
        assert captured["f_scale"] == 2.0

    def test_default_fitter_is_linear(self) -> None:
        fitter = NativeOrbitFitter(propagator_class=TwoBodyPropagator)
        assert fitter.loss == "linear"
        assert fitter.f_scale == HUBER_F_SCALE_DEFAULT


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
