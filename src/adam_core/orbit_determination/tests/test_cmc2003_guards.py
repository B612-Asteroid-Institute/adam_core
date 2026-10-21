"""
CMC2003 guards: a row whose rejection statistic cannot be established must
never look acceptable (PR #217 review round 2, phase 1, item 1).

The decision kernel is pinned on hand-built internal inputs (``+inf`` with
the flag, never ``0``). Public fitting rejects a non-finite observed position
eagerly instead of sending it through a provider or presenting a partial fit.
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.testing as npt
import pyarrow as pa
import pytest

from ..rejection import _expected_residual_chi2, cmc2003_fit_detailed
from .test_cmc2003_rejection import select
from .test_differential_correction import (
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


def test_nan_observation_is_rejected_eagerly_end_to_end() -> None:
    """A non-finite observed position is an invalid OD request, not an outlier."""
    observations = make_synthetic_observations()
    lon = observations.coordinates.lon.to_numpy(zero_copy_only=False).copy()
    lon[12] = np.nan
    observations = observations.set_column("coordinates.lon", pa.array(lon))

    with warnings.catch_warnings(), pytest.raises(ValueError, match="longitude"):
        warnings.simplefilter("ignore")
        cmc2003_fit_detailed(make_initial_guess(), observations, TwoBodyPropagator())
