"""
Synthetic, ground-truth unit tests for the per-band color fit.

The fixture tests exercise the full pipeline against real MPC data and paper
values, but cannot pin down exact behaviour. Here we build reduced magnitudes
directly from known per-band absolute magnitudes and a known phase function, then
check `_fit_per_band_h` recovers the injected colors, phase parameter, outlier
count, missing-band handling, and error scaling.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from ..color_determination import _fit_per_band_h
from ..hg12star import hg12star_correction
from ..magnitude_common import hg_phase_correction

PhiType = Literal["HG12star", "HG", "c1c2"]

# Injected truth: g-r = 0.6, g-i = 0.8, r-i = 0.2.
_H_TRUE = {"g": 18.0, "r": 17.4, "i": 17.2}


def _synthesize(
    phi_type: PhiType,
    phase_param: float | tuple[float, float],
    H_true: dict[str, float] = _H_TRUE,
    n_per_band: int = 60,
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (m_red, alpha_deg, channels, root_weights) for a known model.

    For "HG12star"/"HG", ``phase_param`` is the scalar slope (G12*/G); for "c1c2"
    it is a ``(c1, c2)`` pair with alpha in radians.
    """
    rng = np.random.default_rng(seed)
    bands = list(H_true)
    channels = np.array([b for b in bands for _ in range(n_per_band)], dtype=object)
    alpha = rng.uniform(1.0, 40.0, size=len(channels))
    base = np.array([H_true[c] for c in channels], dtype=np.float64)

    if phi_type == "c1c2":
        assert isinstance(phase_param, tuple)
        c1, c2 = phase_param
        alpha_rad = np.deg2rad(alpha)
        m_red = base + c1 * alpha_rad + c2 * alpha_rad**2
    else:
        assert not isinstance(phase_param, tuple)
        correction = (
            hg12star_correction(alpha, phase_param)
            if phi_type == "HG12star"
            else hg_phase_correction(alpha, phase_param)
        )
        m_red = base + np.asarray(correction)

    sigma = noise if noise > 0 else 1.0
    if noise > 0:
        m_red = m_red + rng.normal(0.0, noise, size=len(m_red))
    root_weights = np.full(len(m_red), 1.0 / sigma)
    return m_red, alpha, channels, root_weights


@pytest.mark.parametrize(
    "phi_type, phase_param",
    [("HG12star", 0.4), ("HG", 0.15)],
)
def test_fit_recovers_known_colors_and_phase(
    phi_type: PhiType, phase_param: float
) -> None:
    """With noiseless data the fit recovers the injected colors and slope exactly."""
    m_red, alpha, channels, rw = _synthesize(phi_type, phase_param)
    fit = _fit_per_band_h(m_red, alpha, channels, rw, phi_type)

    assert fit["H_g"] - fit["H_r"] == pytest.approx(0.6, abs=1e-4)
    assert fit["H_g"] - fit["H_i"] == pytest.approx(0.8, abs=1e-4)
    assert fit["H_r"] - fit["H_i"] == pytest.approx(0.2, abs=1e-4)
    assert fit["G"] == pytest.approx(phase_param, abs=1e-4)
    assert fit["converged"] is True
    assert fit["num_outliers"] == 0


def test_fit_recovers_known_colors_c1c2() -> None:
    """The linear c1c2 model recovers colors to machine precision and G is NaN."""
    m_red, alpha, channels, rw = _synthesize("c1c2", (0.03, -5e-4))
    fit = _fit_per_band_h(m_red, alpha, channels, rw, "c1c2")

    assert fit["H_g"] - fit["H_r"] == pytest.approx(0.6, abs=1e-6)
    assert fit["H_g"] - fit["H_i"] == pytest.approx(0.8, abs=1e-6)
    assert fit["H_r"] - fit["H_i"] == pytest.approx(0.2, abs=1e-6)
    assert np.isnan(fit["G"])


def test_fit_rejects_injected_outlier() -> None:
    """A single gross outlier is flagged and does not corrupt the recovered color."""
    m_red, alpha, channels, rw = _synthesize("HG12star", 0.4)
    m_red = m_red.copy()
    m_red[0] += 2.0  # 2-magnitude blunder on a g-band point

    fit = _fit_per_band_h(m_red, alpha, channels, rw, "HG12star")
    assert fit["num_outliers"] >= 1
    assert fit["H_g"] - fit["H_r"] == pytest.approx(0.6, abs=1e-3)


def test_fit_reports_nan_for_absent_band() -> None:
    """A band with no observations yields NaN magnitude and NaN uncertainty."""
    m_red, alpha, channels, rw = _synthesize(
        "HG12star", 0.4, H_true={"g": 18.0, "r": 17.4}
    )
    fit = _fit_per_band_h(m_red, alpha, channels, rw, "HG12star")

    assert np.isnan(fit["H_i"]) and np.isnan(fit["H_i_sigma"])
    assert np.isnan(fit["H_u"]) and np.isnan(fit["H_u_sigma"])
    assert np.isfinite(fit["H_g"]) and np.isfinite(fit["H_r"])
    assert fit["H_g"] - fit["H_r"] == pytest.approx(0.6, abs=1e-4)


def test_fit_uncertainties_scale_with_injected_noise() -> None:
    """
    When the weights match the true noise, the reduced chi-square is ~1 and the
    reported errors scale linearly with the noise level. Using the same seed makes
    the doubled-noise realization exactly twice the smaller one, so the reported
    color sigma must double.
    """

    def run(sigma: float) -> dict[str, float]:
        m_red, alpha, channels, rw = _synthesize(
            "HG12star", 0.4, n_per_band=300, noise=sigma, seed=7
        )
        return _fit_per_band_h(m_red, alpha, channels, rw, "HG12star")

    small = run(0.03)
    large = run(0.06)

    assert small["reduced_chi2"] == pytest.approx(1.0, abs=0.2)
    assert large["reduced_chi2"] == pytest.approx(1.0, abs=0.2)
    assert large["g_r_sigma"] == pytest.approx(2.0 * small["g_r_sigma"], rel=1e-6)
    # ~sigma / sqrt(N) per band scatter, loosely (covariance with G inflates it a bit).
    assert 0.0 < small["g_r_sigma"] < 0.03
