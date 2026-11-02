"""
Tests for CMC2003 outlier rejection with re-inclusion (`cmc2003_fit`).

Decision rules are unit-tested on hand-built chi2 vectors; the end-to-end
behaviour is checked on synthetic two-body observations of a known truth
orbit with injected gross outliers (from ``test_differential_correction``).
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import numpy.testing as npt
import pytest
from typing_extensions import TypedDict, Unpack

from adam_core._rust.arrow import pa

from ...orbits.orbits import Orbits
from ...propagator.propagator import Propagator
from ..algorithms.fitting import fit_least_squares
from ..algorithms.rejection import (
    CMC2003_CHI2_FRAC,
    CMC2003_CHI2_RECOVER,
    CMC2003_CHI2_REJECT,
    CMC2003_MAX_REJECTED_FRACTION,
    CMC2003Fit,
    _apparitions,
    _cmc2003_select,
    _expected_residual_chi2,
    cmc2003_fit_detailed,
)
from ..evaluate import OrbitDeterminationObservations
from ..fitted_orbits import FittedOrbitMembers, FittedOrbits
from .test_differential_correction import (
    TRUTH_STATE,
    TwoBodyPropagator,
    make_initial_guess,
    make_synthetic_observations,
)


class _CmcOptions(TypedDict, total=False):
    loss: Literal["linear", "huber"]
    ignore: list[str]
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


def state_of(fitted: FittedOrbits) -> np.ndarray:
    return np.concatenate([fitted.coordinates.r[0], fitted.coordinates.v[0]])


def with_outliers(
    sigmas_by_index: dict[int, float],
) -> OrbitDeterminationObservations:
    """Synthetic observations with latitude offsets of k sigma injected."""
    observations = make_synthetic_observations()
    lat = observations.coordinates.lat.to_numpy(zero_copy_only=False).copy()
    covariances = observations.coordinates.covariance.to_matrix()
    for index, k in sigmas_by_index.items():
        lat[index] += k * np.sqrt(covariances[index, 2, 2])
    return observations.set_column("coordinates.lat", pa.array(lat))


def select(
    chi2: list[float],
    selected: list[bool],
    apparitions: list[int] | None = None,
    **overrides: Any,
) -> tuple[np.ndarray, int, int, set[str]]:
    n = len(chi2)
    options: dict[str, Any] = {
        "chi2_reject": CMC2003_CHI2_REJECT,
        "chi2_recover": CMC2003_CHI2_RECOVER,
        "chi2_frac": CMC2003_CHI2_FRAC,
        "max_rejected_fraction": CMC2003_MAX_REJECTED_FRACTION,
        "one_at_a_time": False,
    }
    options.update(overrides)
    return _cmc2003_select(
        np.asarray(chi2, dtype=np.float64),
        np.asarray(selected, dtype=bool),
        np.asarray(apparitions if apparitions is not None else [0] * n),
        **options,
    )


class TestSelectionRules:
    """Hysteresis, batch rule, guards. 30 observations: fudge is negligible."""

    def base(self, n: int = 30) -> tuple[list[float], list[bool]]:
        return [1.0] * n, [True] * n

    def test_nothing_happens_below_thresholds(self) -> None:
        chi2, selected = self.base()
        chi2[5] = 7.9  # below chi2_reject
        new, n_rej, n_rec, flags = select(chi2, selected)
        assert n_rej == 0 and n_rec == 0 and flags == set()
        assert new.all()

    def test_rejects_above_chi2_reject(self) -> None:
        chi2, selected = self.base()
        chi2[5] = 8.5
        new, n_rej, _, _ = select(chi2, selected)
        assert n_rej == 1 and not new[5] and new.sum() == 29

    def test_batch_rule_only_within_chi2_frac_of_worst(self) -> None:
        chi2, selected = self.base()
        chi2[5] = 100.0  # worst
        chi2[6] = 30.0  # >= 0.25 * 100 -> goes in the same pass
        chi2[7] = 20.0  # > chi2_reject but < 25 -> waits for the next pass
        new, n_rej, _, _ = select(chi2, selected)
        assert n_rej == 2
        assert not new[5] and not new[6] and new[7]

    def test_one_at_a_time_mode_rejects_only_the_worst(self) -> None:
        chi2, selected = self.base()
        chi2[5] = 100.0
        chi2[6] = 90.0
        new, n_rej, _, _ = select(chi2, selected, one_at_a_time=True)
        assert n_rej == 1 and not new[5] and new[6]

    def test_recovery_hysteresis(self) -> None:
        chi2, selected = self.base()
        selected[3] = False
        selected[4] = False
        chi2[3] = 6.9  # <= chi2_recover -> recovered
        chi2[4] = 7.5  # between recover and reject -> stays out
        new, n_rej, n_rec, _ = select(chi2, selected)
        assert n_rec == 1 and new[3] and not new[4] and n_rej == 0

    def test_small_sample_fudge_raises_thresholds(self) -> None:
        # 5 selected: fudge = 400 * 3**-5 = 1.646, so 9 < 8 + fudge stays
        chi2 = [1.0, 1.0, 1.0, 1.0, 9.0]
        _, n_rej, _, _ = select(chi2, [True] * 5, one_at_a_time=True)
        assert n_rej == 0
        chi2[4] = 10.0
        _, n_rej, _, _ = select(chi2, [True] * 5, one_at_a_time=True)
        assert n_rej == 1

    def test_max_rejected_fraction_cap(self) -> None:
        chi2, selected = self.base(n=10)
        for i in range(8):
            chi2[i] = 100.0 - i  # 8 offenders within chi2_frac of the worst
        new, n_rej, _, flags = select(chi2, selected)
        # never below ceil(10 * 0.5) = 5 selected
        assert new.sum() == 5 and n_rej == 5
        assert "max_rejected_fraction" in flags

    def test_last_observation_of_an_apparition_is_kept(self) -> None:
        chi2, selected = self.base(n=6)
        apparitions = [0, 0, 0, 0, 0, 1]
        chi2[5] = 50.0  # lone observation of apparition 1
        chi2[0] = 50.0
        new, n_rej, _, flags = select(chi2, selected, apparitions)
        assert new[5] and not new[0] and n_rej == 1
        assert "kept_last_in_apparition" in flags


class TestApparitions:
    def test_groups_split_at_gap(self) -> None:
        mjd = np.array([10.0, 12.0, 400.0, 11.0, 401.0])
        npt.assert_array_equal(_apparitions(mjd, 180.0), [0, 0, 1, 0, 1])

    def test_single_group_without_gaps(self) -> None:
        mjd = np.arange(20.0)
        assert set(_apparitions(mjd, 180.0).tolist()) == {0}


class TestExpectedResidualChi2:
    def test_reduces_to_plain_chi2_without_covariance(self) -> None:
        residuals = np.array([[1.0, 2.0], [0.5, 0.0]])
        chi2, flags = _expected_residual_chi2(
            residuals, np.zeros((4, 6)), None, np.array([True, False])
        )
        npt.assert_allclose(chi2, [5.0, 0.25])
        assert flags == {"no_fit_covariance"}

    def test_inside_fit_shrinks_expected_covariance(self) -> None:
        # P = 0.5 I for every observation: inside -> I - P = 0.5 I doubles the
        # chi2, outside -> I + P = 1.5 I shrinks it.
        residuals = np.array([[1.0, 0.0], [1.0, 0.0]])
        jacobian = np.zeros((4, 6))
        jacobian[0, 0] = jacobian[1, 1] = jacobian[2, 0] = jacobian[3, 1] = 1.0
        covariance = np.diag([0.5, 0.5, 0, 0, 0, 0]).astype(float)
        chi2, flags = _expected_residual_chi2(
            residuals, jacobian, covariance, np.array([True, False])
        )
        npt.assert_allclose(chi2, [2.0, 1.0 / 1.5])
        assert flags == set()

    def test_psd_floor_engages_when_prediction_dominates(self) -> None:
        residuals = np.array([[1.0, 0.0]])
        jacobian = np.zeros((2, 6))
        jacobian[0, 0] = jacobian[1, 1] = 1.0
        covariance = np.diag([0.99, 0.99, 0, 0, 0, 0]).astype(float)
        chi2, flags = _expected_residual_chi2(
            residuals, jacobian, covariance, np.array([True])
        )
        # I - P = 0.01 I, floored to 0.05 I -> chi2 = 1 / 0.05
        npt.assert_allclose(chi2, [20.0])
        assert "psd_floor" in flags


class TestTwoBodyIntegration:
    def test_clean_data_is_left_alone(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cmc2003_fit_detailed(
                make_initial_guess(), make_synthetic_observations(), TwoBodyPropagator()
            )
        assert isinstance(result, CMC2003Fit)
        assert result.n_iterations == 1
        assert result.n_rejected == 0 and result.n_recovered == 0
        assert result.flags == ()
        members = result.fitted_orbit_members
        assert members.outlier.to_pylist() == [False] * 25
        assert members.solution.to_pylist() == [True] * 25
        assert result.fitted_orbit.num_obs[0].as_py() == 25

    def test_gross_outliers_are_rejected_and_truth_recovered(self) -> None:
        observations = with_outliers({10: 10.0, 17: 15.0})
        propagator = TwoBodyPropagator()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = cmc2003_fit_detailed(
                make_initial_guess(), observations, propagator
            )
        assert not [w for w in caught if "weak-direction" in str(w.message)]

        members = result.fitted_orbit_members
        rejected = [i for i, flag in enumerate(members.outlier.to_pylist()) if flag]
        assert rejected == [10, 17]
        assert result.n_rejected == 2 and result.n_recovered == 0
        assert result.n_iterations <= 3
        assert result.flags == ()
        assert members.obs_id.to_pylist() == observations.id.to_pylist()
        assert members.table["residuals"].null_count == 0
        weights = members.weight.to_numpy(zero_copy_only=False)
        assert weights[10] == 0.0 and weights[17] == 0.0 and weights.sum() == 23.0
        assert result.fitted_orbit.num_obs[0].as_py() == 23

        error = state_of(result.fitted_orbit) - TRUTH_STATE
        covariance = result.fitted_orbit.coordinates.covariance.to_matrix()[0]
        mahalanobis = np.sqrt(error @ np.linalg.solve(covariance, error))
        assert mahalanobis < 4.6
        # ... whereas plain least squares is dragged off the truth
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plain, _ = fit_least_squares(make_initial_guess(), observations, propagator)
        error_plain = state_of(plain) - TRUTH_STATE
        assert np.linalg.norm(error[:3]) < 0.3 * np.linalg.norm(error_plain[:3])

    def test_several_outliers_over_multiple_passes(self) -> None:
        injected = {3: 8.0, 10: 10.0, 17: 15.0, 20: -12.0, 22: 9.0}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted, members = _cmc_tables(
                make_initial_guess(), with_outliers(injected), TwoBodyPropagator()
            )
        rejected = [i for i, flag in enumerate(members.outlier.to_pylist()) if flag]
        assert rejected == sorted(injected)
        assert fitted.num_obs[0].as_py() == 20

    def test_composes_with_huber_loss(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted, members = _cmc_tables(
                make_initial_guess(),
                with_outliers({10: 10.0}),
                TwoBodyPropagator(),
                loss="huber",
            )
        assert members.outlier.to_pylist()[10] is True
        assert fitted.success[0].as_py()

    def test_ignore_kwarg_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="ignore"):
            _cmc_tables(
                make_initial_guess(),
                make_synthetic_observations(n_obs=6),
                TwoBodyPropagator(),
                ignore=["obs-000"],
            )

    def test_empty_observations_raise(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            _cmc_tables(
                make_initial_guess(),
                make_synthetic_observations(n_obs=6)[:0],
                TwoBodyPropagator(),
            )


class TestCmc2003CapsuleDriver:
    """Behavioral checks of the single core-owned capsule CMC2003 driver."""

    @staticmethod
    def _fit(
        observations: OrbitDeterminationObservations,
        **kwargs: Unpack[_CmcOptions],
    ) -> CMC2003Fit:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return cmc2003_fit_detailed(
                make_initial_guess(), observations, TwoBodyPropagator(), **kwargs
            )

    def test_gross_outliers_are_rejected(self) -> None:
        result = self._fit(with_outliers({10: 10.0, 17: 15.0}))
        assert [
            i
            for i, flag in enumerate(result.fitted_orbit_members.outlier.to_pylist())
            if flag
        ] == [10, 17]
        assert result.n_rejected == 2
        assert result.flags == ()
        assert result.fitted_orbit.success[0].as_py()

    def test_clean_arc_finishes_without_rejection(self) -> None:
        result = self._fit(make_synthetic_observations())
        assert result.n_iterations == 1
        assert result.n_rejected == 0
        assert result.fitted_orbit_members.outlier.to_pylist() == [False] * 25

    def test_huber_reports_downweighting_without_fabricated_route_parity(self) -> None:
        result = self._fit(with_outliers({10: 10.0}), loss="huber")
        weights = result.fitted_orbit_members.weight.to_numpy(zero_copy_only=False)
        assert weights[10] < 1.0
        assert np.all(np.isfinite(state_of(result.fitted_orbit)))
