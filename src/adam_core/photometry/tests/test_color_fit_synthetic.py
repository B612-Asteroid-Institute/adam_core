"""
Synthetic, ground-truth unit tests for the per-band color fit.

The fixture tests exercise the full pipeline against real MPC data and paper
values, but cannot pin down exact behaviour. Here we build reduced magnitudes
directly from known per-band absolute magnitudes and a known phase function, then
check `_fit_per_band_h` recovers the injected colors, phase parameter, outlier
count, missing-band handling, error scaling, and, for fits that are sparse or
degenerate, the parameter count, rank, degrees of freedom and which quantities
the data can actually determine.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from .. import color_determination
from ..color_determination import _BANDS, _fit_per_band_h
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
    assert fit["num_clipped"] == 0


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
    assert fit["num_clipped"] >= 1
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


def _constant_phase(
    channels: list[str],
    alpha_deg: np.ndarray | float,
    H_true: dict[str, float] = _H_TRUE,
    g12star: float = 0.4,
    sigma: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a noiseless HG12* dataset at caller-chosen (possibly repeated) phase angles.

    Used for the degenerate cases, where the point is the geometry of the phase
    sampling rather than the noise: ``alpha_deg`` may be a scalar (one common
    angle for every row) or a per-row array.
    """
    ch = np.array(channels, dtype=object)
    alpha = np.broadcast_to(np.asarray(alpha_deg, dtype=np.float64), ch.shape).copy()
    m_red = np.array([H_true[c] for c in channels], dtype=np.float64) + np.asarray(
        hg12star_correction(alpha, g12star)
    )
    return m_red, alpha, ch, np.full(len(ch), 1.0 / sigma)


def _emptied_band_dataset(
    kept: str, emptied: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    30 clean ``kept``-band points plus two ``emptied``-band points, both blunders.

    The two ``emptied`` points sit at one phase angle and straddle the truth by
    +-50 sigma, so H lands between them and each is rejected in the same
    iteration, leaving the band with no surviving observation at all.
    """
    alpha = np.concatenate([np.linspace(2.0, 32.0, 30), [15.0, 15.0]])
    m_red, alpha, channels, rw = _constant_phase([kept] * 30 + [emptied] * 2, alpha)
    m_red = m_red.copy()
    m_red[30] += 50 * 0.02
    m_red[31] -= 50 * 0.02
    return m_red, alpha, channels, rw


@pytest.mark.parametrize("bands", [("g",), ("g", "r"), ("g", "r", "i")])
def test_dof_counts_only_observed_bands(bands: tuple[str, ...]) -> None:
    """
    Unobserved bands are not parameters, so rank and DOF follow the bands present.

    Four nominal H columns (three of them all-zero for a g-only fit) used to be
    charged to the parameter count, understating DOF by one per missing band and
    thereby inflating the reduced chi-square, the rescaled covariance and the
    clipping threshold.
    """
    H_true = {b: _H_TRUE[b] for b in bands}
    m_red, alpha, channels, rw = _synthesize(
        "HG12star", 0.4, H_true=H_true, n_per_band=60, noise=0.02, seed=11
    )
    fit = _fit_per_band_h(m_red, alpha, channels, rw, "HG12star")

    # G12* plus one absolute magnitude per observed band, all of them identified.
    expected = 1 + len(bands)
    assert fit["num_params"] == expected
    assert fit["rank"] == expected
    assert fit["dof"] == fit["num_used"] - expected
    assert fit["reduced_chi2"] == pytest.approx(1.0, abs=0.4)
    assert fit["chi2"] == pytest.approx(fit["reduced_chi2"] * fit["dof"])

    for band in bands:
        assert np.isfinite(fit[f"H_{band}"])
        assert fit[f"H_{band}_sigma"] > 0.0
    for band in set(_BANDS) - set(bands):
        assert np.isnan(fit[f"H_{band}"])
        assert np.isnan(fit[f"H_{band}_sigma"])


@pytest.mark.parametrize("bands", [("g",), ("g", "i"), ("g", "r", "i")])
def test_c1c2_dof_counts_only_observed_bands(bands: tuple[str, ...]) -> None:
    """Same accounting for the linear model, whose phase part costs two parameters."""
    H_true = {b: _H_TRUE[b] for b in bands}
    m_red, alpha, channels, rw = _synthesize(
        "c1c2", (0.03, -5e-4), H_true=H_true, n_per_band=60, noise=0.02, seed=13
    )
    fit = _fit_per_band_h(m_red, alpha, channels, rw, "c1c2")

    expected = 2 + len(bands)
    assert fit["num_params"] == expected
    assert fit["rank"] == expected
    assert fit["dof"] == fit["num_used"] - expected
    assert fit["reduced_chi2"] == pytest.approx(1.0, abs=0.4)


def test_sparse_single_band_fit_still_has_degrees_of_freedom() -> None:
    """
    Five g-only observations constrain two parameters, so DOF is 3, not 0.

    Charging all four nominal H columns gave dof = 5 - 5 = 0, which left the
    reduced chi-square and every uncertainty NaN and made the scatter estimate
    infinite, disabling outlier clipping entirely.
    """
    m_red, alpha, channels, rw = _constant_phase(
        ["g"] * 5, np.array([3.0, 9.0, 15.0, 22.0, 30.0])
    )
    noise = np.array([0.01, -0.02, 0.015, -0.005, 0.0])
    fit = _fit_per_band_h(m_red + noise, alpha, channels, rw, "HG12star")

    assert fit["num_params"] == 2
    assert fit["rank"] == 2
    assert fit["dof"] == 3
    assert np.isfinite(fit["reduced_chi2"])
    assert fit["H_g_sigma"] > 0.0
    assert fit["G_sigma"] > 0.0


def test_rank_based_scatter_keeps_outlier_clipping_alive() -> None:
    """
    A gross outlier in a 12-observation, single-band fit is clipped and G12* recovered.

    With the nominal five-parameter count the scatter was estimated over
    dof = 12 - 5 = 7 instead of 12 - 2 = 10, which raised the 3-sigma threshold
    enough that the blunder survived and dragged the slope off (G12* came out
    0.59 rather than the injected 0.40).
    """
    alpha = np.linspace(2.0, 32.0, 12)
    m_red, alpha, channels, rw = _constant_phase(["g"] * 12, alpha)
    m_red = m_red.copy()
    m_red[6] += 50 * 0.02  # 50-sigma blunder

    fit = _fit_per_band_h(m_red, alpha, channels, rw, "HG12star")
    assert fit["num_clipped"] == 1
    assert fit["dof"] == 11 - 2
    assert fit["G"] == pytest.approx(0.4, abs=1e-4)
    assert fit["H_g"] == pytest.approx(_H_TRUE["g"], abs=1e-4)


@pytest.mark.parametrize("kept, emptied", [("g", "r"), ("r", "g")])
def test_emptied_band_drops_out_of_the_parameter_set(kept: str, emptied: str) -> None:
    """
    Band presence follows the final included mask, not the pre-clipping one.

    Both of the emptied band's observations are rejected in one iteration; the
    active set is recomputed afterwards, so the fit is a two-parameter single-band
    one rather than one still charged for a magnitude it can no longer constrain.
    Judging presence before clipping instead returned an arbitrary finite H for
    that band (whatever the optimizer left its unconstrained parameter at), a
    zero uncertainty from the all-zero Jacobian column, and a finite color built
    on top of both.
    """
    fit = _fit_per_band_h(*_emptied_band_dataset(kept, emptied), "HG12star")
    assert fit["num_clipped"] == 2
    assert fit["num_params"] == 2  # G12* and the kept band's H only
    assert fit["rank"] == 2
    assert fit["dof"] == 30 - 2

    assert fit[f"num_used_{emptied}"] == 0
    assert fit[f"num_used_{kept}"] == 30
    assert np.isnan(fit[f"H_{emptied}"]) and np.isnan(fit[f"H_{emptied}_sigma"])
    # Every color involving the emptied band goes with it.
    for color in ("g_r", "g_i", "r_i"):
        if emptied in color.split("_"):
            assert np.isnan(fit[color]) and np.isnan(fit[f"{color}_sigma"])

    # The surviving band is unaffected and still recovers the injected truth.
    assert fit[f"H_{kept}"] == pytest.approx(_H_TRUE[kept], abs=1e-4)
    assert fit["G"] == pytest.approx(0.4, abs=1e-4)


def test_shared_single_phase_angle_keeps_color_but_not_magnitudes() -> None:
    """
    One common phase angle leaves the slope and the magnitudes non-identifiable.

    Every row then sees the same phase correction, so G12* trades off against all
    the absolute magnitudes by an equal amount: rank is one short of the parameter
    count, H_g/H_r/G12* are not individually fitable and are reported as NaN,
    but the trade-off cancels in g-r, which stays exact. The pseudoinverse alone
    would have handed back a finite number for every one of them.
    """
    m_red, alpha, channels, rw = _constant_phase(["g"] * 3 + ["r"] * 3, 10.0)
    fit = _fit_per_band_h(m_red, alpha, channels, rw, "HG12star")

    assert fit["num_params"] == 3
    assert fit["rank"] == 2
    assert fit["dof"] == 6 - 2
    assert np.isnan(fit["G"]) and np.isnan(fit["G_sigma"])
    assert np.isnan(fit["H_g"]) and np.isnan(fit["H_g_sigma"])
    assert np.isnan(fit["H_r"]) and np.isnan(fit["H_r_sigma"])
    assert fit["g_r"] == pytest.approx(0.6, abs=1e-6)
    # i was never observed, so that color is unavailable for a different reason.
    assert np.isnan(fit["g_i"]) and np.isnan(fit["r_i"])


def test_single_band_single_phase_angle_determines_nothing() -> None:
    """With one band at one angle there is no fitable parameter or color at all."""
    m_red, alpha, channels, rw = _constant_phase(["g"] * 5, 10.0)
    fit = _fit_per_band_h(m_red, alpha, channels, rw, "HG12star")

    assert fit["num_params"] == 2
    assert fit["rank"] == 1
    assert np.isnan(fit["G"]) and np.isnan(fit["H_g"])
    assert all(np.isnan(fit[c]) for c in ("g_r", "g_i", "r_i"))


def test_one_phase_angle_per_band_leaves_the_color_non_fitable() -> None:
    """
    Distinct single angles per band make even the color non-identifiable.

    The slope now trades off against H_g and H_r by *different* amounts, so the
    trade-off no longer cancels in g-r: unlike the shared-angle case, the color
    must be reported as NaN rather than retained.
    """
    m_red, alpha, channels, rw = _constant_phase(
        ["g"] * 3 + ["r"] * 3, np.array([10.0] * 3 + [25.0] * 3)
    )
    fit = _fit_per_band_h(m_red, alpha, channels, rw, "HG12star")

    assert fit["num_params"] == 3
    assert fit["rank"] == 2
    assert np.isnan(fit["g_r"]) and np.isnan(fit["g_r_sigma"])


def test_rank_deficiency_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """A rank-deficient fit names the parameters it had to NaN out."""
    m_red, alpha, channels, rw = _constant_phase(["g"] * 3 + ["r"] * 3, 10.0)
    with caplog.at_level("WARNING", logger="adam_core.photometry.color_determination"):
        _fit_per_band_h(m_red, alpha, channels, rw, "HG12star")

    assert "rank deficient (rank 2 of 3 parameters" in caplog.text
    assert "G12*, H_g, H_r" in caplog.text


@pytest.mark.parametrize("bands", [("g",), ("g", "r"), ("g", "r", "i")])
def test_uncertainties_shrink_as_the_root_of_the_sample_size(
    bands: tuple[str, ...],
) -> None:
    """
    Reported magnitude errors fall as 1/sqrt(N) once DOF is counted correctly.

    Quadrupling the observations per band must halve each H sigma, for any number
    of observed bands; a band-count-dependent DOF error would break that scaling.
    """
    H_true = {b: _H_TRUE[b] for b in bands}

    def run(n_per_band: int) -> dict[str, float]:
        m_red, alpha, channels, rw = _synthesize(
            "HG12star",
            0.4,
            H_true=H_true,
            n_per_band=n_per_band,
            noise=0.03,
            seed=23,
        )
        return _fit_per_band_h(m_red, alpha, channels, rw, "HG12star")

    few, many = run(50), run(200)
    for band in bands:
        assert many[f"H_{band}_sigma"] == pytest.approx(
            0.5 * few[f"H_{band}_sigma"], rel=0.25
        )


def _with_unsupported(
    dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Append ``count`` rows whose channel is ``None`` to an existing dataset.

    These stand in for valid photometry taken through a filter outside the
    g/i/r/u set (V, PS1_w, unfiltered reports): `_resolve_channels` hands them
    back as ``None`` and they can never enter the fit.
    """
    m_red, alpha, channels, rw = dataset
    return (
        np.concatenate([m_red, np.full(count, float(np.mean(m_red)))]),
        np.concatenate([alpha, np.full(count, 10.0)]),
        np.concatenate([channels, np.array([None] * count, dtype=object)]),
        np.concatenate([rw, np.full(count, float(rw[0]))]),
    )


def _clean_two_band_dataset(
    n_per_band: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A noiseless, well-sampled g/r dataset that must always fit."""
    alpha = np.tile(np.linspace(2.0, 32.0, n_per_band), 2)
    return _constant_phase(["g"] * n_per_band + ["r"] * n_per_band, alpha)


def test_unsupported_filters_do_not_block_a_clean_fit() -> None:
    """
    A clean g/r fit survives being accompanied by unsupported-filter rows.

    Those rows were previously seeded into the same mask as the statistical
    clip, so they counted as rejections on both sides of the retention
    threshold: 40 unfiltered reports alongside 20 perfectly good g/r
    observations pushed the retained fraction to 1/3 and the fit was refused
    outright, even though nothing about it was unreliable.
    """
    fit = _fit_per_band_h(*_with_unsupported(_clean_two_band_dataset(), 40), "HG12star")

    assert fit["num_unsupported_filter"] == 40
    assert fit["num_clipped"] == 0
    assert fit["num_used"] == 20
    assert fit["g_r"] == pytest.approx(0.6, abs=1e-6)
    assert fit["G"] == pytest.approx(0.4, abs=1e-6)


def test_disposal_counts_partition_the_input() -> None:
    """
    The reported counts account for every row exactly once, and per band.

    The dataset mixes all three buckets `_fit_per_band_h` can report: 30 clean
    g points, 2 r points that are both clipped, and 15 unsupported-filter rows.
    """
    dataset = _with_unsupported(_emptied_band_dataset("g", "r"), 15)
    fit = _fit_per_band_h(*dataset, "HG12star")

    assert fit["num_obs"] == 47
    assert fit["num_unsupported_filter"] == 15
    assert fit["num_clipped"] == 2
    assert fit["num_used"] == 30
    assert (
        fit["num_obs"]
        == fit["num_unsupported_filter"] + fit["num_clipped"] + fit["num_used"]
    )
    assert sum(fit[f"num_used_{b}"] for b in _BANDS) == fit["num_used"]
    assert fit["num_used_g"] == 30
    assert all(fit[f"num_used_{b}"] == 0 for b in ("i", "r", "u"))


def _over_clipped_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    A g-only dataset with a tight core and a geometrically escalating tail.

    Each clip pass shrinks the scatter estimate and exposes the next rung, so
    rejection cascades far enough to exercise the retention threshold.
    """
    n_core, n_tail = 10, 14
    alpha = np.linspace(2.0, 32.0, n_core + n_tail)
    m_red, alpha, channels, rw = _constant_phase(["g"] * (n_core + n_tail), alpha)
    m_red = m_red.copy()
    rungs = 0.1 * 2.0 ** np.arange(n_tail)
    m_red[n_core:] += rungs * np.where(np.arange(n_tail) % 2, 1.0, -1.0)
    return m_red, alpha, channels, rw


def test_retention_threshold_is_measured_among_eligible_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Unsupported-filter rows move neither the threshold nor what it reports.

    The threshold is raised here so the cascade in `_over_clipped_dataset`
    trips it; the point is that padding the same data with rows that could never
    have been fit leaves the verdict, the counts and the quoted denominator
    untouched. Before, each added row relaxed the threshold, the padding was
    charged to both the numerator and the denominator.
    """
    monkeypatch.setattr(color_determination, "_MIN_RETAINED_FRACTION", 0.9)
    dataset = _over_clipped_dataset()

    with pytest.raises(ValueError, match=r"removed 8 of the 24 observation\(s\)"):
        _fit_per_band_h(*dataset, "HG12star")

    # Same 24 eligible observations, now alongside 100 unsupported ones.
    with pytest.raises(ValueError, match=r"removed 8 of the 24 observation\(s\)"):
        _fit_per_band_h(*_with_unsupported(dataset, 100), "HG12star")


def test_retention_threshold_still_fires_on_genuine_over_clipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clipping most of the eligible data is still refused rather than reported."""
    monkeypatch.setattr(color_determination, "_MIN_RETAINED_FRACTION", 0.9)
    with pytest.raises(ValueError, match="fit is unreliable"):
        _fit_per_band_h(*_over_clipped_dataset(), "HG12star")


def test_all_unsupported_filters_is_reported_as_nothing_to_fit() -> None:
    """With no eligible observation at all there is no fit to attempt."""
    n = 12
    channels = np.array([None] * n, dtype=object)
    with pytest.raises(ValueError, match="resolve to a g/i/r/u color channel"):
        _fit_per_band_h(
            np.full(n, 18.0), np.full(n, 10.0), channels, np.full(n, 50.0), "HG12star"
        )
