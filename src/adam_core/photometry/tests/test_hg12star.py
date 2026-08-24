"""
Direct unit tests for the HG12* phase function against Penttila et al. (2016).

The basis functions Phi1, Phi2, Phi3 (Appendix A, Eq. A.1) are Hermite cubic
splines. The reference tables A.2/A.3 give, at each knot, the value of the basis
function and its derivative d(Phi)/d(alpha_rad). Evaluating a basis function at
a knot must return the tabulated value exactly, and a central difference at an
interior knot must reproduce the tabulated derivative, so the tables are used
as the ground truth here.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt
import pytest

from ..hg12star import (
    _XI1_D,
    _XI1_X,
    _XI1_Y,
    _XI2_D,
    _XI2_X,
    _XI2_Y,
    _XI3_X,
    _XI3_Y,
    _phi1,
    _phi2,
    _phi3,
    hg12star_correction,
)


def test_basis_functions_are_unity_at_opposition() -> None:
    """All three basis functions are normalized to 1 at zero phase angle."""
    assert _phi1(0.0) == pytest.approx(1.0)
    assert _phi2(0.0) == pytest.approx(1.0)
    assert _phi3(0.0) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "phi, xs, ys",
    [
        (_phi1, _XI1_X, _XI1_Y),
        (_phi2, _XI2_X, _XI2_Y),
        (_phi3, _XI3_X, _XI3_Y),
    ],
)
def test_basis_functions_interpolate_reference_knots(
    phi: Callable[[float], npt.NDArray[np.float64]],
    xs: np.ndarray,
    ys: np.ndarray,
) -> None:
    """Each basis function reproduces its tabulated knot values exactly."""
    got = np.array([phi(float(x)) for x in xs])
    np.testing.assert_allclose(got, ys, atol=1e-9)


@pytest.mark.parametrize(
    "phi, xs, ds",
    [
        (_phi1, _XI1_X, _XI1_D),
        (_phi2, _XI2_X, _XI2_D),
    ],
)
def test_basis_function_slopes_match_reference_table(
    phi: Callable[[float], npt.NDArray[np.float64]],
    xs: np.ndarray,
    ds: np.ndarray,
) -> None:
    """
    A central difference at each interior knot reproduces the tabulated
    derivative d(Phi)/d(alpha in radians), confirming the derivative tables feed
    the spline correctly.
    """
    h = 1e-5  # radians
    for k in range(1, len(xs) - 1):
        x_rad = np.deg2rad(xs[k])
        num = (phi(np.rad2deg(x_rad + h)) - phi(np.rad2deg(x_rad - h))) / (2 * h)
        assert num == pytest.approx(ds[k], rel=1e-3, abs=1e-3)


@pytest.mark.parametrize("alpha", [0.0, 1.5, 3.0, 7.5])
def test_phi1_phi2_follow_closed_form_below_7p5_deg(alpha: float) -> None:
    """Below 7.5 deg the first two basis functions are exactly linear in alpha."""
    assert _phi1(alpha) == pytest.approx(1.0 - (6.0 / np.pi) * np.deg2rad(alpha))
    assert _phi2(alpha) == pytest.approx(
        1.0 - (9.0 / (5.0 * np.pi)) * np.deg2rad(alpha)
    )


@pytest.mark.parametrize("alpha", [30.0, 45.0, 90.0, 150.0])
def test_phi3_is_zero_beyond_30_deg(alpha: float) -> None:
    """Phi3 is clamped to 0 for phase angles at or beyond 30 deg."""
    assert float(np.atleast_1d(_phi3(alpha))[0]) == 0.0


@pytest.mark.parametrize("g12star", [0.0, 0.2, 0.5, 0.8, 1.0])
def test_correction_is_zero_at_opposition(g12star: float) -> None:
    """At zero phase the combined phase function is 1, so the correction is 0."""
    value = float(np.atleast_1d(hg12star_correction(np.array([0.0]), g12star))[0])
    assert value == pytest.approx(0.0, abs=1e-9)


def test_correction_is_monotonic_in_phase() -> None:
    """
    For a physical G12* the magnitude correction grows monotonically with phase
    (the object dims as it moves away from opposition).
    """
    alpha = np.linspace(0.0, 60.0, 61)
    corr = hg12star_correction(alpha, 0.5)
    assert corr[0] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.diff(corr) >= -1e-9)
    assert corr[-1] > 0.0


def test_correction_scalar_and_vector_agree() -> None:
    """Elementwise scalar calls match a single vectorized call."""
    alpha = np.array([2.0, 10.0, 25.0, 55.0])
    vector = np.asarray(hg12star_correction(alpha, 0.4))
    elementwise = np.array(
        [float(np.atleast_1d(hg12star_correction(float(a), 0.4))[0]) for a in alpha]
    )
    np.testing.assert_allclose(vector, elementwise, atol=1e-12)
