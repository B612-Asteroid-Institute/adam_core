"""
CMC2003 guards: a row whose rejection statistic cannot be established must
never look acceptable (PR #217 review round 2, phase 1, item 1).

The decision kernel is pinned on hand-built inputs (``+inf`` with the flag,
never ``0``); the end-to-end check runs `cmc2003_fit_detailed` on the
two-body backend with one crafted observation whose residual is NaN, which
must come out flagged and rejected while the fit recovers the truth from the
remaining observations.
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.testing as npt
import pyarrow as pa

from ..rejection import _expected_residual_chi2, cmc2003_fit_detailed
from .test_cmc2003_rejection import select, state_of
from .test_differential_correction import (
    TRUTH_STATE,
    TwoBodyPropagator,
    make_initial_guess,
    make_synthetic_observations,
)


def identity_jacobian(n: int) -> np.ndarray:
    jacobian = np.zeros((2 * n, 6))
    jacobian[0::2, 0] = 1.0
    jacobian[1::2, 1] = 1.0
    return jacobian


class TestStatisticGuards:
    def test_nan_residual_is_infinite_and_flagged(self) -> None:
        residuals = np.array([[1.0, 0.5], [np.nan, 0.0], [1.0, 0.0]])
        chi2, flags = _expected_residual_chi2(
            residuals, identity_jacobian(3), None, np.array([True, True, True])
        )
        assert np.isfinite(chi2[0]) and np.isfinite(chi2[2])
        assert chi2[1] == np.inf
        assert "non_finite_chi2" in flags

    def test_singular_covariance_is_infinite_unless_residual_is_zero(self) -> None:
        # P = I for a selected row with a zero eigenvalue floor: I - P = 0.
        covariance = np.zeros((6, 6))
        covariance[0, 0] = covariance[1, 1] = 1.0
        residuals = np.array([[0.3, 0.0], [0.0, 0.0]])
        chi2, flags = _expected_residual_chi2(
            residuals,
            identity_jacobian(2),
            covariance,
            np.array([True, True]),
            psd_floor_frac=0.0,
        )
        npt.assert_array_equal(chi2, [np.inf, 0.0])
        assert flags == {"singular_residual_covariance"}

    def test_infinite_rows_are_rejected_first_and_never_recovered(self) -> None:
        chi2 = [1.0] * 30
        chi2[5] = np.inf
        chi2[6] = 100.0
        selected, n_rejected, n_recovered, _ = select(chi2, [True] * 30)
        assert not selected[5] and selected[6]
        assert (n_rejected, n_recovered) == (1, 0)
        # chi2_frac = 0 must not turn the batch threshold into inf * 0.
        selected, n_rejected, _, _ = select(chi2, [True] * 30, chi2_frac=0.0)
        assert not selected[5] and selected[6] and n_rejected == 1
        excluded = [True] * 30
        excluded[5] = False
        selected, _, n_recovered, _ = select(chi2, excluded)
        assert not selected[5] and n_recovered == 0


def test_nan_residual_observation_is_flagged_and_rejected_end_to_end() -> None:
    """One observation with a NaN position: its prediction residual is NaN."""
    observations = make_synthetic_observations()
    bad = 12
    lon = observations.coordinates.lon.to_numpy(zero_copy_only=False).copy()
    lon[bad] = np.nan
    observations = observations.set_column("coordinates.lon", pa.array(lon))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = cmc2003_fit_detailed(
            make_initial_guess(), observations, TwoBodyPropagator()
        )

    assert "non_finite_chi2" in result.flags
    members = result.fitted_orbit_members
    outlier = members.outlier.to_pylist()
    assert outlier[bad] is True
    assert sum(outlier) == 1
    assert result.n_rejected == 1 and result.n_recovered == 0
    assert result.fitted_orbit.num_obs[0].as_py() == len(observations) - 1
    assert result.fitted_orbit.success[0].as_py()
    # The remaining observations recover the truth.
    error = state_of(result.fitted_orbit) - TRUTH_STATE
    covariance = result.fitted_orbit.coordinates.covariance.to_matrix()[0]
    assert np.all(np.isfinite(covariance))
    mahalanobis = np.sqrt(error @ np.linalg.solve(covariance, error))
    assert mahalanobis < 4.6
