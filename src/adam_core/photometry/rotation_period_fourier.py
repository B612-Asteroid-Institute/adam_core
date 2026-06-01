from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import cast

import numpy as np
import numpy.typing as npt

from .rotation_period_fourier_core import (
    FOURIER_PROFILES,
    _DesignInfo,
    _FitResult,
    _FitWithPeriod,
    _FourierProfile,
    _amplitude_from_fit,
    _apply_light_time_correction,
    _build_fixed_design,
    _fit_bic,
    _fit_frequency,
    _fit_frequency_unclipped,
    _fit_with_period,
    _harmonic_relative_mismatch,
    _ordered_unique,
    _phase_prior_rows,
    _select_order,
    _sigma_threshold_for_confidence,
    _summarize_sessions,
    _validate_inputs,
)
from .rotation_period_types import RotationPeriodObservations, RotationPeriodResult

_DEFAULT_HARMONIC_PERIOD_FACTORS = (
    0.5,
    2.0 / 3.0,
    0.75,
    1.0,
    4.0 / 3.0,
    1.5,
    2.0,
    3.0,
    4.0,
)
_LSM_MIN_PERIOD_DAYS = 0.00065
_LSM_MAX_PERIOD_DAYS = 3.0
_LSM_OVERSAMPLE_FACTOR = 100.0
_LSM_DEFAULT_MAX_SAMPLES = 20000
_LSM_DEFAULT_REFINE_SAMPLES = 2000
_LSM_DEFAULT_REFINE_ROUNDS = 2
_LSM_FAMILY_TOLERANCE = 0.05
_LSM_POWER_TIE_TOLERANCE = 2.0e-4
_SIMPLE_HARMONIC_FACTORS = (0.5, 1.0, 2.0)
_DEFAULT_JAX_FREQUENCY_BATCH_SIZE = 256
_DEFAULT_JAX_ROW_PAD_MULTIPLE = 64
_FOURIER_MAX_CLIP_ITERATIONS = 8

# --- D1 confidence-contract thresholds (bead rp-e4a.6) ---------------------
# Initial values per the D1 contract; the calibration study (bead rp-e4a.13)
# will tune these on a held-out split of the standard-candle set, so treat the
# values below as provisional defaults rather than validated cut points.
# False-alarm-probability ceiling for accepting an LSM/hybrid peak as signal.
FAP_SIGNIFICANT = 0.01
# Minimum fraction of rotational-phase bins that must be occupied to clear the
# signal gate (family-level) and to be eligible for ``single_period``.
PHASE_COVERAGE_MIN_FAMILY = 0.5
PHASE_COVERAGE_MIN_SINGLE = 0.7
# Minimum amplitude-to-noise ratio (amplitude_mag / residual_sigma_mag).
AMPLITUDE_SNR_MIN = 3.0
# Number of rotational-phase bins used to compute ``phase_coverage_fraction``.
_PHASE_COVERAGE_N_BINS = 20
# Longest period a ``single_period`` verdict may claim (bead rp-e4a.22 step 1).
# Recovered periods beyond this are treated as low-frequency drift fit as
# rotation rather than a real spin, so the confidence (not the value) is
# downgraded to ``period_family``. rp-e4a.13 may recalibrate; max real candle
# period across 327 LCDB+DAMIT reliable objects is 67.5h.
MAX_PLAUSIBLE_SINGLE_PERIOD_HOURS = 96.0

# --- Cheap pre-solve insufficiency thresholds (bead rp-e4a.19) -------------
# Period-INDEPENDENT screening run before the Fourier/LSM search so obviously
# under-determined / degenerate inputs can early-exit instead of building a
# frequency grid (or hanging on a single-night span).  Provisional values; the
# calibration study (bead rp-e4a.13) may retune them.
# Hard floor on the number of observations regardless of model size.
MIN_OBS = 8
# Extra observations required beyond the baseline free-parameter count
# (n_filters + 2*min_order + 2 phase terms) for the fit to be over-determined.
PRE_SOLVE_OBS_MARGIN = 2
# Robust-scatter-to-noise floor: fire ``amplitude_below_noise`` only when the
# distance-reduced magnitude scatter is clearly below the photometric noise.
AMP_SNR_FLOOR = 1.5
# Absolute mag scatter floor (mag) used when mag_sigma is entirely null, so the
# amplitude check stays conservative rather than firing on quiet-but-real data.
PRE_SOLVE_ABS_SCATTER_FLOOR = 0.02

_VERDICT_SINGLE = "single_period"
_VERDICT_FAMILY = "period_family"
_VERDICT_INSUFFICIENT = "insufficient_data"
_RELIABILITY_BY_VERDICT = {
    _VERDICT_SINGLE: "3",
    _VERDICT_FAMILY: "2",
    _VERDICT_INSUFFICIENT: "1",
}


@dataclass(slots=True)
class _FourierCluster:
    indices: npt.NDArray[np.int64]
    best: _FitWithPeriod
    period_lower_days: float
    period_upper_days: float
    sigma_best: float
    raw_weight: float


@dataclass(slots=True)
class _FourierSolution:
    chosen: _FitWithPeriod
    primary_cluster: _FourierCluster
    sigma_threshold: float
    clusters: list[_FourierCluster]
    period_lower_days: float
    period_upper_days: float
    relative_period_uncertainty: float
    alternate_period_days: list[float]
    is_valid: bool
    is_reliable: bool
    amplitude_mag: float
    used_session_offsets: bool
    fit_summary: _FitResult
    sigma_curve: npt.NDArray[np.float64]


@dataclass(slots=True)
class _LSMCandidate:
    period_days: float
    power: float
    coeffs: npt.NDArray[np.float64]
    frequency: float | None = None
    n_maxima: int | None = None
    n_minima: int | None = None
    amplitude_mag: float | None = None


@dataclass(slots=True)
class _LSMSolution:
    period_days: float | None
    power: float | None
    power_gap: float | None
    candidate_period_days: list[float]
    candidate_powers: list[float]
    is_reliable: bool
    amplitude_mag: float | None
    n_fit_observations: int
    n_clipped: int
    false_alarm_probability: float | None = None


@dataclass(slots=True)
class _LSMMethodResult:
    best_candidate: _LSMCandidate | None
    power_gap: float | None
    candidate_period_days: list[float]
    candidate_powers: list[float]
    is_reliable: bool
    amplitude_mag: float | None
    n_fit_observations: int | None = None
    n_clipped: int = 0
    false_alarm_probability: float | None = None


@dataclass(slots=True)
class _FourierMethodResult:
    chosen_fit: _FitResult
    best_period: _FitWithPeriod
    sigma_threshold: float
    period_lower_days: float
    period_upper_days: float
    relative_period_uncertainty: float
    alternate_period_days: list[float]
    is_valid: bool
    is_reliable: bool
    amplitude_mag: float


@dataclass(slots=True)
class _MethodFamily:
    representative_period_days: float
    fourier_cluster: _FourierCluster | None = None
    lsm_candidate: _LSMCandidate | None = None
    contains_fourier_primary: bool = False
    contains_lsm_primary: bool = False
    family_weight_fourier: float = 0.0
    family_weight_lsm: float = 0.0
    combined_weight: float = 0.0


def _resolve_search_fidelity(
    *,
    search_fidelity: str | None,
    search_strategy: str | None,
) -> str:
    if search_fidelity is not None:
        resolved = str(search_fidelity)
    elif search_strategy in {"surrogate_refine", "coarse_to_fine"}:
        resolved = "validated_staged"
    elif search_strategy == "grid":
        resolved = "exact_grid"
    else:
        resolved = "validated_staged"
    if resolved not in {"validated_staged", "exact_grid"}:
        raise ValueError("search_fidelity must be one of {'validated_staged', 'exact_grid'}")
    return resolved


def _resolve_paper_profile(paper_profile: str) -> _FourierProfile:
    if paper_profile not in FOURIER_PROFILES:
        raise ValueError("paper_profile must be one of {'greenstreet_2026', 'vavilov_2025'}")
    return FOURIER_PROFILES[paper_profile]


def _paper_profile(paper_profile: str) -> _FourierProfile:
    return _resolve_paper_profile(paper_profile)


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    normalized = sorted(
        [(min(int(start), int(end)), max(int(start), int(end))) for start, end in intervals],
        key=lambda item: item[0],
    )
    merged: list[tuple[int, int]] = [normalized[0]]
    for start, end in normalized[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _sample_indices_from_intervals(
    intervals: list[tuple[int, int]],
    *,
    stride: int,
    n_total: int,
) -> npt.NDArray[np.int64]:
    if n_total <= 0:
        return np.zeros(0, dtype=np.int64)
    if not intervals:
        intervals = [(0, n_total - 1)]
    idx_set: set[int] = set()
    for start, end in intervals:
        lo = max(0, min(int(start), n_total - 1))
        hi = max(0, min(int(end), n_total - 1))
        if hi < lo:
            lo, hi = hi, lo
        idx_set.add(lo)
        idx_set.add(hi)
        step = max(1, int(stride))
        first = lo + ((step - (lo % step)) % step)
        for idx in range(first, hi + 1, step):
            idx_set.add(idx)
    return np.asarray(sorted(idx_set), dtype=np.int64)


def _local_minima_positions(scores: npt.NDArray[np.float64]) -> list[int]:
    valid_positions = np.flatnonzero(np.isfinite(scores))
    if valid_positions.size == 0:
        return []
    minima: list[int] = []
    for pos in valid_positions.tolist():
        left = float(scores[pos - 1]) if pos > 0 and np.isfinite(scores[pos - 1]) else np.inf
        center = float(scores[pos])
        right = (
            float(scores[pos + 1])
            if pos + 1 < len(scores) and np.isfinite(scores[pos + 1])
            else np.inf
        )
        if center <= left and center <= right:
            minima.append(pos)
    return minima if minima else valid_positions.tolist()


def _candidate_intervals_from_scores(
    sample_indices: npt.NDArray[np.int64],
    scores: npt.NDArray[np.float64],
    *,
    n_total: int,
    radius: int,
    candidate_count: int,
) -> list[tuple[int, int]]:
    positions = _local_minima_positions(scores)
    positions = sorted(positions, key=lambda pos: float(scores[pos]))
    intervals: list[tuple[int, int]] = []
    chosen_centers: list[int] = []
    stride = 1 if sample_indices.size <= 1 else int(max(1, round(float(np.median(np.diff(sample_indices))))))
    for pos in positions:
        center = int(sample_indices[pos])
        if any(abs(center - existing) < stride for existing in chosen_centers):
            continue
        chosen_centers.append(center)
        intervals.append((max(0, center - radius), min(n_total - 1, center + radius)))
        if len(chosen_centers) >= int(candidate_count):
            break
    return _merge_intervals(intervals)


def _evaluate_frequency_indices_with_jax(
    *,
    t_rel: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    design_info: _DesignInfo,
    frequencies: npt.NDArray[np.float64],
    sample_indices: npt.NDArray[np.int64],
    order: int,
    clip_sigma: float,
    weights: npt.NDArray[np.float64] | None,
    jax_frequency_batch_size: int,
    jax_row_pad_multiple: int,
) -> tuple[npt.NDArray[np.float64], list[_FitResult | None], _FitResult | None]:
    try:
        from .rotation_period_jax import evaluate_frequency_indices_jax
    except Exception as exc:  # pragma: no cover - depends on optional local JAX install
        raise RuntimeError("exact_evaluation_backend='jax' requires JAX to be installed") from exc

    n_par = int(design_info.fixed.shape[1] + 2 * int(order))
    min_phase = float(np.min(design_info.fixed[:, design_info.phase_c1_idx]))
    prior_rows, prior_target, prior_weights = _phase_prior_rows(
        n_par,
        design_info,
        min_phase,
    )
    result = evaluate_frequency_indices_jax(
        time_rel=t_rel,
        y=y,
        fixed=design_info.fixed,
        weights=weights,
        prior_rows=prior_rows,
        prior_target=prior_target,
        prior_weights=prior_weights,
        frequencies=frequencies,
        sample_indices=sample_indices,
        fourier_order=int(order),
        clip_sigma=float(clip_sigma),
        jax_batch_size=int(jax_frequency_batch_size),
        row_pad_multiple=int(jax_row_pad_multiple),
        max_clip_iterations=_FOURIER_MAX_CLIP_ITERATIONS,
    )
    scores = np.asarray(result.scores, dtype=np.float64)
    fits: list[_FitResult | None] = [None] * int(sample_indices.size)
    finite_positions = np.flatnonzero(np.isfinite(scores))
    if not result.best_valid or finite_positions.size == 0:
        return scores, fits, None

    best_pos = int(finite_positions[int(np.argmin(scores[finite_positions]))])
    best_index = int(sample_indices[best_pos])
    best_mask = np.asarray(result.best_mask, dtype=bool)
    best_fit = _FitResult(
        frequency=float(frequencies[best_index]),
        fourier_order=int(order),
        coeffs=np.asarray(result.best_coeffs, dtype=np.float64),
        residual_sigma=float(result.best_sigma),
        rss=float(result.best_rss),
        df=int(result.best_df),
        n_par=int(n_par),
        mask=best_mask,
        n_fit=int(result.best_n_fit),
        n_clipped=int(result.best_n_clipped),
        n_filters=int(design_info.n_filters),
        phase_c1_idx=int(design_info.phase_c1_idx),
        phase_c2_idx=int(design_info.phase_c2_idx),
        sum_weights=None if weights is None else float(np.sum(np.asarray(weights, dtype=np.float64)[best_mask])),
    )
    fits[best_pos] = best_fit
    return scores, fits, best_fit


def _evaluate_frequency_indices(
    *,
    t_rel: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    design_info: _DesignInfo,
    frequencies: npt.NDArray[np.float64],
    sample_indices: npt.NDArray[np.int64],
    order: int,
    clip_sigma: float,
    weights: npt.NDArray[np.float64] | None,
    exact: bool,
    exact_evaluation_backend: str,
    jax_frequency_batch_size: int,
    jax_row_pad_multiple: int,
) -> tuple[npt.NDArray[np.float64], list[_FitResult | None], _FitResult | None]:
    if exact and exact_evaluation_backend == "jax":
        return _evaluate_frequency_indices_with_jax(
            t_rel=t_rel,
            y=y,
            design_info=design_info,
            frequencies=frequencies,
            sample_indices=sample_indices,
            order=order,
            clip_sigma=clip_sigma,
            weights=weights,
            jax_frequency_batch_size=jax_frequency_batch_size,
            jax_row_pad_multiple=jax_row_pad_multiple,
        )

    scores = np.full(sample_indices.shape, np.nan, dtype=np.float64)
    fits: list[_FitResult | None] = [None] * int(sample_indices.size)
    best_fit: _FitResult | None = None
    for pos, idx in enumerate(sample_indices.tolist()):
        frequency = float(frequencies[int(idx)])
        fit = (
            _fit_frequency(
                t_rel,
                y,
                design_info,
                frequency,
                int(order),
                clip_sigma=clip_sigma,
                weights=weights,
            )
            if exact
            else _fit_frequency_unclipped(
                t_rel,
                y,
                design_info,
                frequency,
                int(order),
                weights=weights,
            )
        )
        fits[pos] = fit
        if fit is None:
            continue
        scores[pos] = float(fit.residual_sigma)
        if best_fit is None or fit.residual_sigma < best_fit.residual_sigma:
            best_fit = fit
    return scores, fits, best_fit


def _search_best_fit(
    *,
    t_rel: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    design_info: _DesignInfo,
    frequencies: npt.NDArray[np.float64],
    order: int,
    clip_sigma: float,
    weights: npt.NDArray[np.float64] | None,
    search_fidelity: str,
    exact_evaluation_backend: str,
    jax_frequency_batch_size: int,
    jax_row_pad_multiple: int,
) -> _FitResult | None:
    n_total = int(frequencies.size)
    if n_total == 0:
        return None
    if search_fidelity == "exact_grid" or n_total <= 2048:
        sample_indices = np.arange(n_total, dtype=np.int64)
        _, _, best_fit = _evaluate_frequency_indices(
            t_rel=t_rel,
            y=y,
            design_info=design_info,
            frequencies=frequencies,
            sample_indices=sample_indices,
            order=order,
            clip_sigma=clip_sigma,
            weights=weights,
            exact=True,
            exact_evaluation_backend=exact_evaluation_backend,
            jax_frequency_batch_size=jax_frequency_batch_size,
            jax_row_pad_multiple=jax_row_pad_multiple,
        )
        return best_fit

    coarse_stride = max(1, int(ceil(n_total / 1024.0)))
    coarse_indices = np.arange(0, n_total, coarse_stride, dtype=np.int64)
    if coarse_indices[-1] != n_total - 1:
        coarse_indices = np.concatenate([coarse_indices, np.asarray([n_total - 1], dtype=np.int64)])
    coarse_scores, _, _ = _evaluate_frequency_indices(
        t_rel=t_rel,
        y=y,
        design_info=design_info,
        frequencies=frequencies,
        sample_indices=coarse_indices,
        order=order,
        clip_sigma=clip_sigma,
        weights=weights,
        exact=False,
        exact_evaluation_backend=exact_evaluation_backend,
        jax_frequency_batch_size=jax_frequency_batch_size,
        jax_row_pad_multiple=jax_row_pad_multiple,
    )
    intervals = _candidate_intervals_from_scores(
        coarse_indices,
        coarse_scores,
        n_total=n_total,
        radius=max(8, 4 * coarse_stride),
        candidate_count=12,
    )
    final_indices = _sample_indices_from_intervals(intervals, stride=1, n_total=n_total)
    _, _, best_fit = _evaluate_frequency_indices(
        t_rel=t_rel,
        y=y,
        design_info=design_info,
        frequencies=frequencies,
        sample_indices=final_indices,
        order=order,
        clip_sigma=clip_sigma,
        weights=weights,
        exact=True,
        exact_evaluation_backend=exact_evaluation_backend,
        jax_frequency_batch_size=jax_frequency_batch_size,
        jax_row_pad_multiple=jax_row_pad_multiple,
    )
    if best_fit is not None:
        return best_fit
    fallback_indices = np.arange(n_total, dtype=np.int64)
    _, _, best_fit = _evaluate_frequency_indices(
        t_rel=t_rel,
        y=y,
        design_info=design_info,
        frequencies=frequencies,
        sample_indices=fallback_indices,
        order=order,
        clip_sigma=clip_sigma,
        weights=weights,
        exact=True,
        exact_evaluation_backend=exact_evaluation_backend,
        jax_frequency_batch_size=jax_frequency_batch_size,
        jax_row_pad_multiple=jax_row_pad_multiple,
    )
    return best_fit


def _evaluate_full_order_curve(
    *,
    t_rel: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    design_info: _DesignInfo,
    frequencies: npt.NDArray[np.float64],
    order: int,
    clip_sigma: float,
    weights: npt.NDArray[np.float64] | None,
    exact_evaluation_backend: str,
    jax_frequency_batch_size: int,
    jax_row_pad_multiple: int,
) -> tuple[npt.NDArray[np.float64], list[_FitResult | None]]:
    sample_indices = np.arange(int(frequencies.size), dtype=np.int64)
    scores, fits, _ = _evaluate_frequency_indices(
        t_rel=t_rel,
        y=y,
        design_info=design_info,
        frequencies=frequencies,
        sample_indices=sample_indices,
        order=order,
        clip_sigma=clip_sigma,
        weights=weights,
        exact=True,
        exact_evaluation_backend=exact_evaluation_backend,
        jax_frequency_batch_size=jax_frequency_batch_size,
        jax_row_pad_multiple=jax_row_pad_multiple,
    )
    full_fits: list[_FitResult | None] = [None] * int(frequencies.size)
    for idx, fit in zip(sample_indices.tolist(), fits, strict=True):
        full_fits[int(idx)] = fit
    return scores, full_fits


# Hard cap on the Fourier frequency-grid size. n_freq grows as
# frequency_grid_scale * span_days * f_max, so a multi-apparition (multi-year) span
# would explode it into the millions and dominate runtime. The cap only bites on such
# pathological spans (single-apparition inputs stay far below it); when it bites the
# grid is merely coarser and the staged search still refines around peaks.
_FOURIER_MAX_FREQUENCY_SAMPLES = 200_000


def _build_frequency_grid(
    *,
    span_days: float,
    min_rotations_in_span: float,
    max_frequency_cycles_per_day: float,
    frequency_grid_scale: float,
    max_search_period_hours: float | None = None,
) -> npt.NDArray[np.float64]:
    f_min = float(min_rotations_in_span / span_days)
    f_max = float(max_frequency_cycles_per_day)
    if max_search_period_hours is not None:
        # f is cycles/day, so raising f_min caps the LONGEST searched period.
        f_min = max(f_min, 24.0 / float(max_search_period_hours))
    if not np.isfinite(f_min) or f_min <= 0.0:
        raise ValueError("derived minimum frequency is invalid")
    if f_max <= f_min:
        raise ValueError("max_frequency_cycles_per_day must exceed the minimum searchable frequency")
    n_freq = max(int(ceil(float(frequency_grid_scale) * span_days * (f_max - f_min)) + 1), 2)
    n_freq = min(n_freq, _FOURIER_MAX_FREQUENCY_SAMPLES)
    return np.linspace(f_min, f_max, n_freq, dtype=np.float64)


def _weights_from_sigma(mag_sigma: npt.NDArray[np.float64] | None) -> npt.NDArray[np.float64] | None:
    if mag_sigma is None:
        return None
    sigma = np.asarray(mag_sigma, dtype=np.float64)
    if sigma.ndim != 1 or sigma.size == 0:
        return None
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        return None
    return 1.0 / np.square(sigma)


def _cluster_fourier_solutions(
    *,
    fits: list[_FitResult | None],
    accepted_indices: npt.NDArray[np.int64],
    sigma_threshold: float,
    frequencies: npt.NDArray[np.float64] | None = None,
) -> list[_FourierCluster]:
    if accepted_indices.size == 0:
        return []
    best_sigma = float(
        min(
            fits[int(idx)].residual_sigma
            for idx in accepted_indices.tolist()
            if fits[int(idx)] is not None
        )
    )
    fit_items: list[tuple[int, _FitWithPeriod, float, float]] = []
    for idx in accepted_indices.tolist():
        fit = fits[int(idx)]
        if fit is None:
            continue
        fit_with_period = _fit_with_period(fit)
        center_frequency = float(fit.frequency)
        if frequencies is None:
            period_lower_days = float(fit_with_period.period_days)
            period_upper_days = float(fit_with_period.period_days)
        else:
            left_frequency = float(frequencies[int(idx - 1)]) if idx > 0 else center_frequency
            right_frequency = (
                float(frequencies[int(idx + 1)]) if int(idx) + 1 < len(frequencies) else center_frequency
            )
            frequency_half_step = 0.5 * max(
                abs(center_frequency - left_frequency),
                abs(right_frequency - center_frequency),
            )
            factor = 2.0 if fit_with_period.is_period_doubled else 1.0
            low_edge_frequency = max(center_frequency - frequency_half_step, np.finfo(np.float64).eps)
            high_edge_frequency = center_frequency + frequency_half_step
            edge_periods = np.asarray(
                [factor / high_edge_frequency, factor / low_edge_frequency],
                dtype=np.float64,
            )
            period_lower_days = float(np.min(edge_periods))
            period_upper_days = float(np.max(edge_periods))
        fit_items.append((int(idx), fit_with_period, period_lower_days, period_upper_days))
    if not fit_items:
        return []
    fit_items.sort(key=lambda item: (item[2], item[3], item[0]))
    clusters: list[_FourierCluster] = []
    denom = float(max(sigma_threshold - best_sigma, 0.0))
    merged_items: list[list[tuple[int, _FitWithPeriod, float, float]]] = [[fit_items[0]]]
    for item in fit_items[1:]:
        current_lower = float(item[2])
        previous_group = merged_items[-1]
        previous_upper = max(float(group_item[3]) for group_item in previous_group)
        if current_lower <= previous_upper + 1.0e-12:
            previous_group.append(item)
        else:
            merged_items.append([item])
    for group_items in merged_items:
        group_indices = np.asarray([item[0] for item in group_items], dtype=np.int64)
        group_fits = [item[1] for item in group_items]
        best = min(group_fits, key=lambda item: float(item.fit.residual_sigma))
        period_lower_days = float(min(item[2] for item in group_items))
        period_upper_days = float(max(item[3] for item in group_items))
        if denom > 0.0:
            raw_weight = max(0.0, 1.0 - (float(best.fit.residual_sigma) - best_sigma) / denom)
        else:
            raw_weight = 1.0 if abs(float(best.fit.residual_sigma) - best_sigma) <= 1.0e-12 else 0.0
        clusters.append(
            _FourierCluster(
                indices=group_indices,
                best=best,
                period_lower_days=period_lower_days,
                period_upper_days=period_upper_days,
                sigma_best=float(best.fit.residual_sigma),
                raw_weight=float(raw_weight),
            )
        )
    return sorted(clusters, key=lambda cluster: float(cluster.sigma_best))


def _relative_period_uncertainty(period_days: float, lower_days: float, upper_days: float) -> float:
    if not np.isfinite(period_days) or period_days <= 0.0:
        return float("inf")
    return float(max(abs(period_days - lower_days), abs(upper_days - period_days)) / period_days)


def _max_cluster_period_deviation(period_days: float, clusters: list[_FourierCluster]) -> float:
    if not np.isfinite(period_days) or period_days <= 0.0 or not clusters:
        return float("inf")
    return float(
        max(
            max(
                abs(float(cluster.period_lower_days) - period_days),
                abs(float(cluster.period_upper_days) - period_days),
            )
            for cluster in clusters
        )
    )


def _derive_fourier_solution(
    *,
    t_rel: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    design_info: _DesignInfo,
    frequencies: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64] | None,
    clip_sigma: float,
    profile: _FourierProfile,
    search_fidelity: str,
    exact_evaluation_backend: str,
    jax_frequency_batch_size: int,
    jax_row_pad_multiple: int,
    used_session_offsets: bool,
    fourier_orders: tuple[int, ...] | None,
) -> _FourierSolution:
    orders = tuple(sorted({int(order) for order in (fourier_orders or profile.orders)}))
    if not orders:
        raise ValueError("fourier order set must be non-empty")

    candidate_fits: dict[int, _FitResult] = {}
    for order in orders:
        fit = _search_best_fit(
            t_rel=t_rel,
            y=y,
            design_info=design_info,
            frequencies=frequencies,
            order=int(order),
            clip_sigma=clip_sigma,
            weights=weights,
            search_fidelity=search_fidelity,
            exact_evaluation_backend=exact_evaluation_backend,
            jax_frequency_batch_size=jax_frequency_batch_size,
            jax_row_pad_multiple=jax_row_pad_multiple,
        )
        if fit is not None:
            candidate_fits[int(order)] = fit
    if not candidate_fits:
        raise ValueError("no valid rotation-period fit could be found")

    chosen_order_fit = _select_order(candidate_fits, profile.order_selection_confidence)
    sigma_curve, fits = _evaluate_full_order_curve(
        t_rel=t_rel,
        y=y,
        design_info=design_info,
        frequencies=frequencies,
        order=int(chosen_order_fit.fourier_order),
        clip_sigma=clip_sigma,
        weights=weights,
        exact_evaluation_backend=exact_evaluation_backend,
        jax_frequency_batch_size=jax_frequency_batch_size,
        jax_row_pad_multiple=jax_row_pad_multiple,
    )
    if not np.any(np.isfinite(sigma_curve)):
        raise ValueError("failed to evaluate Fourier sigma curve")
    best_idx = int(np.nanargmin(sigma_curve))
    best_fit = fits[best_idx]
    if best_fit is None:
        raise ValueError("best Fourier fit is unavailable")
    best_with_period = _fit_with_period(best_fit)
    sigma_threshold = float(_sigma_threshold_for_confidence(best_fit, profile.sigma_threshold_confidence))
    accepted_indices = np.flatnonzero(np.isfinite(sigma_curve) & (sigma_curve <= sigma_threshold))
    if accepted_indices.size == 0:
        accepted_indices = np.asarray([best_idx], dtype=np.int64)
    for idx in accepted_indices.tolist():
        if fits[int(idx)] is not None:
            continue
        fits[int(idx)] = _fit_frequency(
            t_rel,
            y,
            design_info,
            float(frequencies[int(idx)]),
            int(chosen_order_fit.fourier_order),
            clip_sigma=clip_sigma,
            weights=weights,
        )
    clusters = _cluster_fourier_solutions(
        fits=fits,
        accepted_indices=accepted_indices,
        sigma_threshold=sigma_threshold,
        frequencies=frequencies,
    )
    if not clusters:
        clusters = [
            _FourierCluster(
                indices=np.asarray([best_idx], dtype=np.int64),
                best=best_with_period,
                period_lower_days=float(best_with_period.period_days),
                period_upper_days=float(best_with_period.period_days),
                sigma_best=float(best_fit.residual_sigma),
                raw_weight=1.0,
            )
        ]
    primary_cluster = next(
        (cluster for cluster in clusters if best_idx in set(cluster.indices.tolist())),
        clusters[0],
    )
    period_lower_days = float(primary_cluster.period_lower_days)
    period_upper_days = float(primary_cluster.period_upper_days)
    relative_uncertainty = _relative_period_uncertainty(
        best_with_period.period_days,
        period_lower_days,
        period_upper_days,
    )
    alternate_period_days = [
        float(cluster.best.period_days)
        for cluster in clusters
        if cluster is not primary_cluster
    ]
    amplitude_mag = float(_amplitude_from_fit(best_fit))

    is_valid = True
    if profile.valid_relative_uncertainty_max is not None:
        is_valid = bool(relative_uncertainty <= float(profile.valid_relative_uncertainty_max))

    is_reliable = is_valid
    if profile.reliable_relative_multiplier is not None and profile.reliable_absolute_hours is not None:
        uncertainty_days = _max_cluster_period_deviation(best_with_period.period_days, clusters)
        is_reliable = bool(
            uncertainty_days <= max(
                float(profile.reliable_relative_multiplier) * best_with_period.period_days,
                float(profile.reliable_absolute_hours) / 24.0,
            )
        )

    return _FourierSolution(
        chosen=best_with_period,
        primary_cluster=primary_cluster,
        sigma_threshold=sigma_threshold,
        clusters=clusters,
        period_lower_days=period_lower_days,
        period_upper_days=period_upper_days,
        relative_period_uncertainty=relative_uncertainty,
        alternate_period_days=alternate_period_days,
        is_valid=bool(is_valid),
        is_reliable=bool(is_reliable),
        amplitude_mag=amplitude_mag,
        used_session_offsets=bool(used_session_offsets),
        fit_summary=best_fit,
        sigma_curve=sigma_curve,
    )


def _band_intercept_design(filter_labels: npt.NDArray[np.object_]) -> tuple[npt.NDArray[np.float64], list[str]]:
    unique_filters = _ordered_unique(filter_labels)
    n = len(filter_labels)
    cols: list[npt.NDArray[np.float64]] = [np.ones(n, dtype=np.float64)]
    for label in unique_filters[1:]:
        cols.append((filter_labels == label).astype(np.float64))
    return np.column_stack(cols).astype(np.float64, copy=False), unique_filters


def _weighted_fit_with_clipping(
    *,
    design: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64] | None,
    clip_sigma: float,
    max_iterations: int = 6,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    mask = np.ones(target.shape[0], dtype=bool)
    for _ in range(max(1, int(max_iterations))):
        idx = np.flatnonzero(mask)
        if idx.size <= design.shape[1]:
            break
        design_use = design[idx]
        target_use = target[idx]
        if weights is None:
            coeffs, *_ = np.linalg.lstsq(design_use, target_use, rcond=None)
            residuals = target_use - design_use @ coeffs
        else:
            w_use = np.asarray(weights[idx], dtype=np.float64)
            sqrt_w = np.sqrt(w_use)
            coeffs, *_ = np.linalg.lstsq(design_use * sqrt_w[:, None], target_use * sqrt_w, rcond=None)
            residuals = target_use - design_use @ coeffs
        sigma = float(np.std(residuals, ddof=max(1, design_use.shape[1])))
        if not np.isfinite(sigma) or sigma <= 0.0:
            break
        keep = np.abs(residuals) <= float(clip_sigma) * sigma
        if np.all(keep):
            return np.asarray(coeffs, dtype=np.float64), mask
        new_mask = mask.copy()
        new_mask[idx[~keep]] = False
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    idx = np.flatnonzero(mask)
    design_use = design[idx]
    target_use = target[idx]
    if weights is None:
        coeffs, *_ = np.linalg.lstsq(design_use, target_use, rcond=None)
    else:
        w_use = np.asarray(weights[idx], dtype=np.float64)
        sqrt_w = np.sqrt(w_use)
        coeffs, *_ = np.linalg.lstsq(design_use * sqrt_w[:, None], target_use * sqrt_w, rcond=None)
    return np.asarray(coeffs, dtype=np.float64), mask


def _prepare_lsm_inputs(
    *,
    mag: npt.NDArray[np.float64],
    reduced_mag: npt.NDArray[np.float64],
    predicted_mag_v: npt.NDArray[np.float64] | None,
    phase_angle: npt.NDArray[np.float64],
    filter_labels: npt.NDArray[np.object_],
    weights: npt.NDArray[np.float64] | None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.object_],
    npt.NDArray[np.float64] | None,
    int,
    npt.NDArray[np.bool_],
]:
    n_obs = int(mag.shape[0])
    active_mask = np.ones(n_obs, dtype=bool)
    if predicted_mag_v is not None and np.all(np.isfinite(predicted_mag_v)):
        corrected = np.asarray(mag, dtype=np.float64) - np.asarray(predicted_mag_v, dtype=np.float64)
    else:
        fixed_design, _ = _band_intercept_design(filter_labels)
        phase_cols = np.column_stack(
            [fixed_design, np.asarray(phase_angle, dtype=np.float64), np.square(phase_angle)]
        )
        coeffs, mask = _weighted_fit_with_clipping(
            design=phase_cols,
            target=np.asarray(reduced_mag, dtype=np.float64),
            weights=weights,
            clip_sigma=3.0,
        )
        trend = phase_cols @ coeffs
        corrected = np.asarray(reduced_mag, dtype=np.float64) - trend
        if not np.all(mask):
            active_mask &= np.asarray(mask, dtype=bool)
            corrected = corrected[mask]
            filter_labels = filter_labels[mask]
            weights = None if weights is None else np.asarray(weights[mask], dtype=np.float64)

    corrected = np.asarray(corrected, dtype=np.float64) - float(np.median(corrected))
    mean = float(np.mean(corrected))
    sigma = float(np.std(corrected))
    if not np.isfinite(sigma) or sigma <= 0.0:
        keep = np.ones(corrected.shape[0], dtype=bool)
    else:
        keep = np.abs(corrected - mean) <= 3.0 * sigma
    corrected = corrected[keep]
    filter_labels = filter_labels[keep]
    clipped = int(np.count_nonzero(~keep))
    if weights is not None:
        weights = np.asarray(weights[keep], dtype=np.float64)
    if np.any(~active_mask):
        surviving = np.flatnonzero(active_mask)
        final_mask = np.zeros(n_obs, dtype=bool)
        final_mask[surviving[keep]] = True
    else:
        final_mask = np.asarray(keep, dtype=bool)
    return corrected, filter_labels, weights, clipped, final_mask


def _lsm_frequency_grid(
    *,
    span_days: float,
    max_samples: int,
) -> npt.NDArray[np.float64]:
    f_min = 1.0 / _LSM_MAX_PERIOD_DAYS
    f_max = 1.0 / _LSM_MIN_PERIOD_DAYS
    ideal_count = int(ceil(_LSM_OVERSAMPLE_FACTOR * span_days * (f_max - f_min)) + 1)
    n_samples = max(256, min(int(max_samples), ideal_count))
    return np.linspace(f_min, f_max, n_samples, dtype=np.float64)


def _lsm_design(
    *,
    t_rel: npt.NDArray[np.float64],
    filter_labels: npt.NDArray[np.object_],
    frequency: float,
    order: int = 2,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    intercepts, _ = _band_intercept_design(filter_labels)
    omega = 2.0 * np.pi * float(frequency) * np.asarray(t_rel, dtype=np.float64)
    periodic_cols: list[npt.NDArray[np.float64]] = []
    for harmonic in range(1, int(order) + 1):
        angle = harmonic * omega
        periodic_cols.append(np.cos(angle))
        periodic_cols.append(np.sin(angle))
    periodic = np.column_stack(periodic_cols).astype(np.float64, copy=False)
    return intercepts, np.concatenate([intercepts, periodic], axis=1)


def _weighted_chi2(
    residuals: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64] | None,
) -> float:
    if weights is None:
        return float(np.sum(np.square(residuals)))
    w = np.asarray(weights, dtype=np.float64)
    return float(np.sum(w * np.square(residuals)))


def _fit_lsm_frequency(
    *,
    t_rel: npt.NDArray[np.float64],
    corrected: npt.NDArray[np.float64],
    filter_labels: npt.NDArray[np.object_],
    weights: npt.NDArray[np.float64] | None,
    frequency: float,
) -> tuple[float, npt.NDArray[np.float64]]:
    flat_design, full_design = _lsm_design(
        t_rel=t_rel,
        filter_labels=filter_labels,
        frequency=frequency,
        order=2,
    )
    if weights is None:
        flat_coeffs, *_ = np.linalg.lstsq(flat_design, corrected, rcond=None)
        flat_resid = corrected - flat_design @ flat_coeffs
        coeffs, *_ = np.linalg.lstsq(full_design, corrected, rcond=None)
        resid = corrected - full_design @ coeffs
    else:
        w = np.asarray(weights, dtype=np.float64)
        sqrt_w = np.sqrt(w)
        flat_coeffs, *_ = np.linalg.lstsq(flat_design * sqrt_w[:, None], corrected * sqrt_w, rcond=None)
        flat_resid = corrected - flat_design @ flat_coeffs
        coeffs, *_ = np.linalg.lstsq(full_design * sqrt_w[:, None], corrected * sqrt_w, rcond=None)
        resid = corrected - full_design @ coeffs
    chi2_0 = _weighted_chi2(flat_resid, weights)
    chi2 = _weighted_chi2(resid, weights)
    if chi2_0 <= 0.0:
        power = 0.0
    else:
        power = max(0.0, 1.0 - chi2 / chi2_0)
    return float(power), np.asarray(coeffs, dtype=np.float64)


def _periodic_extrema_counts_from_coeffs(coeffs: npt.NDArray[np.float64], order: int = 2) -> tuple[int, int]:
    phase = np.linspace(0.0, 1.0, 2048, endpoint=False)
    periodic = np.zeros_like(phase)
    start = coeffs.size - 2 * int(order)
    for harmonic in range(1, int(order) + 1):
        idx = start + 2 * (harmonic - 1)
        periodic += coeffs[idx] * np.cos(2.0 * np.pi * harmonic * phase)
        periodic += coeffs[idx + 1] * np.sin(2.0 * np.pi * harmonic * phase)
    prev_vals = np.roll(periodic, 1)
    next_vals = np.roll(periodic, -1)
    maxima = (periodic > prev_vals) & (periodic >= next_vals)
    minima = (periodic < prev_vals) & (periodic <= next_vals)
    return int(np.count_nonzero(maxima)), int(np.count_nonzero(minima))


def _amplitude_from_lsm_coeffs(coeffs: npt.NDArray[np.float64], order: int = 2) -> float:
    phase = np.linspace(0.0, 1.0, 4096, endpoint=False)
    periodic = np.zeros_like(phase)
    start = coeffs.size - 2 * int(order)
    for harmonic in range(1, int(order) + 1):
        idx = start + 2 * (harmonic - 1)
        periodic += coeffs[idx] * np.cos(2.0 * np.pi * harmonic * phase)
        periodic += coeffs[idx + 1] * np.sin(2.0 * np.pi * harmonic * phase)
    return float(np.max(periodic) - np.min(periodic))


def _local_maxima_indices(values: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
    if values.size == 0:
        return np.zeros(0, dtype=np.int64)
    maxima: list[int] = []
    for idx in range(values.size):
        left = float(values[idx - 1]) if idx > 0 else -np.inf
        center = float(values[idx])
        right = float(values[idx + 1]) if idx + 1 < values.size else -np.inf
        if center >= left and center > right:
            maxima.append(idx)
    return np.asarray(maxima, dtype=np.int64)


def _lsm_false_alarm_probability(
    *,
    time_lsm: npt.NDArray[np.float64],
    corrected: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64] | None,
    best_frequency: float,
) -> float | None:
    """Astropy Baluev false-alarm probability for the LSM peak frequency.

    Computes the Lomb-Scargle periodogram on the trend-corrected magnitudes and
    returns the analytic Baluev FAP of the peak power. Returns ``None`` when the
    FAP cannot be evaluated (too few points, degenerate inputs, or an Astropy
    failure) so the caller can treat it as "no significant peak".
    """
    if not np.isfinite(best_frequency) or best_frequency <= 0.0:
        return None
    times = np.asarray(time_lsm, dtype=np.float64)
    values = np.asarray(corrected, dtype=np.float64)
    if times.size < 4 or values.size != times.size:
        return None
    # Convert fit weights (1/sigma^2) back to per-point uncertainties when present.
    dy: npt.NDArray[np.float64] | None = None
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape == values.shape and np.all(np.isfinite(w)) and np.all(w > 0.0):
            dy = 1.0 / np.sqrt(w)
    try:
        from astropy.timeseries import LombScargle  # type: ignore[import-untyped]

        # Astropy only implements an analytic FAP for the single-term (nterms=1)
        # periodogram, so the signal gate asks the standard question "is the
        # strongest periodicity in this data significant?". A doubly-peaked
        # rotation puts most single-term power at 2x the LSM frequency, so take
        # the more significant of (LSM peak, global periodogram peak).
        model = LombScargle(times, values, dy=dy) if dy is not None else LombScargle(times, values)
        power_at_best = float(model.power(float(best_frequency)))
        frequency_grid, power_grid = model.autopower(
            minimum_frequency=float(best_frequency) / 4.0,
            maximum_frequency=float(best_frequency) * 4.0,
            samples_per_peak=5,
        )
        power_peak = max(power_at_best, float(np.max(power_grid)))
        fap = float(model.false_alarm_probability(power_peak, method="baluev"))
    except Exception:
        return None
    if not np.isfinite(fap):
        return None
    return float(min(max(fap, 0.0), 1.0))


def _estimate_lsm_solution(
    *,
    t_rel: npt.NDArray[np.float64],
    mag: npt.NDArray[np.float64],
    reduced_mag: npt.NDArray[np.float64],
    predicted_mag_v: npt.NDArray[np.float64] | None,
    phase_angle: npt.NDArray[np.float64],
    filter_labels: npt.NDArray[np.object_],
    weights: npt.NDArray[np.float64] | None,
    lsm_max_frequency_samples: int,
    lsm_refine_samples: int,
    lsm_refine_rounds: int,
) -> _LSMSolution:
    corrected, filter_lsm, weights_lsm, clipped, keep_mask = _prepare_lsm_inputs(
        mag=mag,
        reduced_mag=reduced_mag,
        predicted_mag_v=predicted_mag_v,
        phase_angle=phase_angle,
        filter_labels=np.asarray(filter_labels, dtype=object),
        weights=weights,
    )
    if corrected.size < 8:
        return _LSMSolution(
            period_days=None,
            power=None,
            power_gap=None,
            candidate_period_days=[],
            candidate_powers=[],
            is_reliable=False,
            amplitude_mag=None,
            n_fit_observations=int(corrected.size),
            n_clipped=int(clipped),
        )

    time_lsm = np.asarray(t_rel[keep_mask], dtype=np.float64)
    if time_lsm.size != corrected.size:
        raise ValueError("internal error: LSM masking produced misaligned time and magnitude arrays")
    span_days = float(np.max(time_lsm) - np.min(time_lsm))
    span_days = max(span_days, 1.0e-6)
    frequencies = _lsm_frequency_grid(span_days=span_days, max_samples=lsm_max_frequency_samples)

    powers = np.full(frequencies.shape, np.nan, dtype=np.float64)
    coeffs_by_index: list[npt.NDArray[np.float64] | None] = [None] * int(frequencies.size)
    for idx, frequency in enumerate(frequencies.tolist()):
        power, coeffs = _fit_lsm_frequency(
            t_rel=time_lsm,
            corrected=corrected,
            filter_labels=filter_lsm,
            weights=weights_lsm,
            frequency=float(frequency),
        )
        powers[idx] = float(power)
        coeffs_by_index[idx] = coeffs

    local_maxima = _local_maxima_indices(powers)
    survivors: list[_LSMCandidate] = []
    for idx in local_maxima.tolist():
        coeffs = coeffs_by_index[int(idx)]
        if coeffs is None:
            continue
        n_maxima, n_minima = _periodic_extrema_counts_from_coeffs(coeffs, order=2)
        if n_maxima != 2 or n_minima != 2:
            continue
        survivors.append(
            _LSMCandidate(
                period_days=float(1.0 / frequencies[int(idx)]),
                power=float(powers[int(idx)]),
                coeffs=np.asarray(coeffs, dtype=np.float64),
            )
        )

    survivors.sort(key=lambda candidate: float(candidate.power), reverse=True)
    if not survivors:
        return _LSMSolution(
            period_days=None,
            power=None,
            power_gap=None,
            candidate_period_days=[],
            candidate_powers=[],
            is_reliable=False,
            amplitude_mag=None,
            n_fit_observations=int(corrected.size),
            n_clipped=int(clipped),
        )

    best_power = float(survivors[0].power)
    tied_survivors = [
        candidate
        for candidate in survivors
        if best_power - float(candidate.power) <= _LSM_POWER_TIE_TOLERANCE
    ]
    best = max(tied_survivors, key=lambda candidate: float(candidate.period_days))
    ordered_survivors = [best] + [candidate for candidate in survivors if candidate is not best]
    second_power = float(ordered_survivors[1].power) if len(ordered_survivors) >= 2 else 0.0
    power_gap = float(best.power - second_power) if len(ordered_survivors) >= 2 else float(best.power)
    amplitude_mag = float(_amplitude_from_lsm_coeffs(best.coeffs, order=2))
    best_frequency = 1.0 / float(best.period_days) if best.period_days > 0.0 else float("nan")
    false_alarm_probability = _lsm_false_alarm_probability(
        time_lsm=time_lsm,
        corrected=corrected,
        weights=weights_lsm,
        best_frequency=best_frequency,
    )
    # Keep power/power_gap as diagnostics; the false-alarm probability is the
    # validated significance gate (D1). ``None`` FAP fails the gate.
    is_reliable = bool(
        best.power >= 0.1
        and power_gap >= 0.02
        and false_alarm_probability is not None
        and false_alarm_probability <= FAP_SIGNIFICANT
    )
    return _LSMSolution(
        period_days=float(best.period_days),
        power=float(best.power),
        power_gap=float(power_gap),
        candidate_period_days=[float(candidate.period_days) for candidate in ordered_survivors[:12]],
        candidate_powers=[float(candidate.power) for candidate in ordered_survivors[:12]],
        is_reliable=is_reliable,
        amplitude_mag=amplitude_mag,
        n_fit_observations=int(corrected.size),
        n_clipped=int(clipped),
        false_alarm_probability=false_alarm_probability,
    )


def _run_lsm(
    *,
    t_rel: npt.NDArray[np.float64],
    mag: npt.NDArray[np.float64],
    reduced_mag: npt.NDArray[np.float64],
    predicted_mag_v: npt.NDArray[np.float64] | None,
    phase_angle: npt.NDArray[np.float64],
    filter_labels: npt.NDArray[np.object_],
    weights: npt.NDArray[np.float64] | None,
    lsm_max_frequency_samples: int,
    lsm_refine_samples: int,
    lsm_refine_rounds: int,
) -> _LSMMethodResult:
    solution = _estimate_lsm_solution(
        t_rel=t_rel,
        mag=mag,
        reduced_mag=reduced_mag,
        predicted_mag_v=predicted_mag_v,
        phase_angle=phase_angle,
        filter_labels=filter_labels,
        weights=weights,
        lsm_max_frequency_samples=lsm_max_frequency_samples,
        lsm_refine_samples=lsm_refine_samples,
        lsm_refine_rounds=lsm_refine_rounds,
    )
    best_candidate: _LSMCandidate | None = None
    if solution.period_days is not None and solution.power is not None:
        frequency = 1.0 / float(solution.period_days)
        amplitude_mag = solution.amplitude_mag
        best_candidate = _LSMCandidate(
            frequency=float(frequency),
            period_days=float(solution.period_days),
            power=float(solution.power),
            coeffs=np.zeros(4, dtype=np.float64),
            n_maxima=2 if solution.is_reliable else None,
            n_minima=2 if solution.is_reliable else None,
            amplitude_mag=None if amplitude_mag is None else float(amplitude_mag),
        )
    return _LSMMethodResult(
        best_candidate=best_candidate,
        power_gap=None if solution.power_gap is None else float(solution.power_gap),
        candidate_period_days=list(solution.candidate_period_days),
        candidate_powers=list(solution.candidate_powers),
        is_reliable=bool(solution.is_reliable),
        amplitude_mag=None if solution.amplitude_mag is None else float(solution.amplitude_mag),
        n_fit_observations=int(solution.n_fit_observations),
        n_clipped=int(solution.n_clipped),
        false_alarm_probability=(
            None
            if solution.false_alarm_probability is None
            else float(solution.false_alarm_probability)
        ),
    )


def _build_fourier_result(
    *,
    chosen_fit: _FitResult,
    order_grid_results: dict[int, tuple[npt.NDArray[np.float64], dict[int, _FitResult]]],
    frequencies: npt.NDArray[np.float64],
    profile: _FourierProfile,
) -> _FourierMethodResult:
    scores, fits_by_index = order_grid_results[int(chosen_fit.fourier_order)]
    sigma_threshold = float(_sigma_threshold_for_confidence(chosen_fit, profile.sigma_threshold_confidence))
    accepted_indices = np.flatnonzero(np.isfinite(scores) & (scores <= sigma_threshold))
    if accepted_indices.size == 0:
        accepted_indices = np.asarray(
            [int(np.nanargmin(np.asarray(scores, dtype=np.float64)))],
            dtype=np.int64,
        )
    fits: list[_FitResult | None] = [fits_by_index.get(int(idx)) for idx in range(len(frequencies))]
    clusters = _cluster_fourier_solutions(
        fits=fits,
        accepted_indices=accepted_indices,
        sigma_threshold=sigma_threshold,
        frequencies=frequencies,
    )
    best_period = _fit_with_period(chosen_fit)
    primary = next(
        (
            cluster
            for cluster in clusters
            if any(
                fits_by_index.get(int(idx)) is chosen_fit
                or (
                    fits_by_index.get(int(idx)) is not None
                    and abs(float(fits_by_index[int(idx)].frequency) - float(chosen_fit.frequency)) <= 1.0e-12
                )
                for idx in cluster.indices.tolist()
            )
        ),
        clusters[0],
    )
    relative_uncertainty = _relative_period_uncertainty(
        best_period.period_days,
        float(primary.period_lower_days),
        float(primary.period_upper_days),
    )
    alternate_period_days = [
        float(cluster.best.period_days)
        for cluster in clusters
        if cluster is not primary
    ]
    is_valid = True
    if profile.valid_relative_uncertainty_max is not None:
        is_valid = bool(relative_uncertainty <= float(profile.valid_relative_uncertainty_max))
    is_reliable = is_valid
    if profile.reliable_relative_multiplier is not None and profile.reliable_absolute_hours is not None:
        uncertainty_days = _max_cluster_period_deviation(best_period.period_days, clusters)
        is_reliable = bool(
            uncertainty_days <= max(
                float(profile.reliable_relative_multiplier) * best_period.period_days,
                float(profile.reliable_absolute_hours) / 24.0,
            )
        )
    return _FourierMethodResult(
        chosen_fit=chosen_fit,
        best_period=best_period,
        sigma_threshold=sigma_threshold,
        period_lower_days=float(primary.period_lower_days),
        period_upper_days=float(primary.period_upper_days),
        relative_period_uncertainty=float(relative_uncertainty),
        alternate_period_days=alternate_period_days,
        is_valid=bool(is_valid),
        is_reliable=bool(is_reliable),
        amplitude_mag=float(_amplitude_from_fit(chosen_fit)),
    )


def _normalize_method_weights(raw_weights: list[float]) -> list[float]:
    total = float(sum(max(0.0, weight) for weight in raw_weights))
    if total <= 0.0:
        return [0.0 for _ in raw_weights]
    return [float(max(0.0, weight) / total) for weight in raw_weights]


def _best_harmonic_factor(
    period_a: float,
    period_b: float,
    harmonic_period_factors: tuple[float, ...],
) -> tuple[float, float]:
    if period_a <= 0.0 or period_b <= 0.0 or not np.isfinite(period_a) or not np.isfinite(period_b):
        return 1.0, float("inf")
    best_factor = 1.0
    best_mismatch = float("inf")
    for factor in harmonic_period_factors:
        mismatch = float(abs(period_a * factor - period_b) / max(abs(period_b), np.finfo(np.float64).eps))
        if mismatch < best_mismatch:
            best_mismatch = mismatch
            best_factor = float(factor)
    return float(best_factor), float(best_mismatch)


def _is_simple_harmonic_factor(factor: float) -> bool:
    return any(abs(float(factor) - simple) <= 1.0e-12 for simple in _SIMPLE_HARMONIC_FACTORS)


def _family_match(period_a: float, period_b: float, harmonic_period_factors: tuple[float, ...]) -> bool:
    return _harmonic_relative_mismatch(
        period_a,
        period_b,
        harmonic_period_factors=harmonic_period_factors,
    ) <= _LSM_FAMILY_TOLERANCE


def _build_hybrid_families(
    *,
    fourier_solution: _FourierSolution,
    lsm_solution: _LSMSolution,
    harmonic_period_factors: tuple[float, ...],
) -> list[_MethodFamily]:
    families: list[_MethodFamily] = []
    fourier_primary_period = float(fourier_solution.chosen.period_days)
    lsm_primary_period = None if lsm_solution.period_days is None else float(lsm_solution.period_days)

    for cluster in fourier_solution.clusters:
        period = float(cluster.best.period_days)
        match = next(
            (
                family
                for family in families
                if _family_match(period, family.representative_period_days, harmonic_period_factors)
            ),
            None,
        )
        if match is None:
            match = _MethodFamily(representative_period_days=period)
            families.append(match)
        if match.fourier_cluster is None or cluster.raw_weight > match.fourier_cluster.raw_weight:
            match.fourier_cluster = cluster
            match.representative_period_days = period
        if _family_match(period, fourier_primary_period, harmonic_period_factors):
            match.contains_fourier_primary = True

    lsm_candidates = [
        _LSMCandidate(period_days=float(period), power=float(power), coeffs=np.zeros(4, dtype=np.float64))
        for period, power in zip(lsm_solution.candidate_period_days, lsm_solution.candidate_powers, strict=True)
    ]
    for candidate in lsm_candidates:
        period = float(candidate.period_days)
        match = next(
            (
                family
                for family in families
                if _family_match(period, family.representative_period_days, harmonic_period_factors)
            ),
            None,
        )
        if match is None:
            match = _MethodFamily(representative_period_days=period)
            families.append(match)
        if match.lsm_candidate is None or candidate.power > match.lsm_candidate.power:
            match.lsm_candidate = candidate
        if lsm_primary_period is not None and _family_match(period, lsm_primary_period, harmonic_period_factors):
            match.contains_lsm_primary = True

    fourier_raw = [0.0 if family.fourier_cluster is None else float(family.fourier_cluster.raw_weight) for family in families]
    lsm_best_power = max((float(candidate.power) for candidate in lsm_candidates), default=0.0)
    lsm_raw = [
        0.0
        if family.lsm_candidate is None or lsm_best_power <= 0.0
        else float(family.lsm_candidate.power) / lsm_best_power
        for family in families
    ]
    fourier_norm = _normalize_method_weights(fourier_raw)
    lsm_norm = _normalize_method_weights(lsm_raw)
    for family, wf, wl in zip(families, fourier_norm, lsm_norm, strict=True):
        family.family_weight_fourier = float(wf)
        family.family_weight_lsm = float(wl)
        family.combined_weight = float(0.5 * wf + 0.5 * wl)
    families.sort(key=lambda family: float(family.combined_weight), reverse=True)
    return families


def _observation_count_sufficient(filter_labels: npt.NDArray[np.object_]) -> bool:
    """Replicate the legacy observation-count guard (top filter[s] >= 30 obs)."""
    unique_filters = _ordered_unique(filter_labels)
    counts = sorted(
        [int(np.count_nonzero(filter_labels == label)) for label in unique_filters],
        reverse=True,
    )
    if len(counts) >= 2:
        return bool(counts[0] >= 30 and counts[1] >= 30)
    return bool(counts and counts[0] >= 30)


def _phase_coverage_fraction(
    *,
    times: npt.NDArray[np.float64],
    period_days: float,
    n_bins: int = _PHASE_COVERAGE_N_BINS,
) -> float | None:
    """Fraction of rotational-phase bins occupied at ``period_days``.

    Folds the observation times to the chosen period, bins the phase onto
    ``[0, 1)`` and returns the fraction of bins that contain >= 1 observation.
    Returns ``None`` when the period or times are degenerate.
    """
    if not np.isfinite(period_days) or period_days <= 0.0:
        return None
    t = np.asarray(times, dtype=np.float64)
    if t.size == 0 or n_bins <= 0:
        return None
    phase = np.mod(t / float(period_days), 1.0)
    bin_index = np.floor(phase * int(n_bins)).astype(np.int64)
    bin_index = np.clip(bin_index, 0, int(n_bins) - 1)
    occupied = int(np.unique(bin_index).size)
    return float(occupied) / float(n_bins)


def _amplitude_snr(amplitude_mag: float | None, residual_sigma_mag: float | None) -> float | None:
    """``amplitude_mag / residual_sigma_mag`` with division/None guards."""
    if amplitude_mag is None or residual_sigma_mag is None:
        return None
    if not np.isfinite(amplitude_mag) or not np.isfinite(residual_sigma_mag):
        return None
    if residual_sigma_mag <= 0.0:
        return None
    return float(amplitude_mag) / float(residual_sigma_mag)


def _classify_confidence(
    *,
    primary_method: str,
    amplitude_snr: float | None,
    phase_coverage_fraction: float | None,
    n_rotations_spanned: float | None,
    min_rotations_in_span: float,
    lsm_false_alarm_probability: float | None,
    n_significant_aliases: int | None,
    is_period_doubled: bool,
    observation_count_sufficient: bool,
    is_reliable: bool,
    is_valid: bool,
    cross_method_agree: bool | None,
) -> tuple[str, str, list[str], list[str]]:
    """Single deterministic verdict (D1 §"decision tree").

    Returns ``(period_verdict, reliability_code, confidence_flags,
    insufficiency_reasons)``.  ``cross_method_agree`` is ``None`` for
    single-method modes; ``True``/``False`` for hybrid.
    """
    flags: list[str] = []
    reasons: list[str] = []

    uses_fap = primary_method in {"lsm", "hybrid"}

    # ---- 1. Signal gate ----------------------------------------------------
    if amplitude_snr is None or not np.isfinite(amplitude_snr) or amplitude_snr < AMPLITUDE_SNR_MIN:
        reasons.append("amplitude_below_noise")
    if (
        phase_coverage_fraction is None
        or not np.isfinite(phase_coverage_fraction)
        or phase_coverage_fraction < PHASE_COVERAGE_MIN_FAMILY
    ):
        reasons.append("phase_coverage_low")
    if (
        n_rotations_spanned is None
        or not np.isfinite(n_rotations_spanned)
        or n_rotations_spanned < float(min_rotations_in_span)
    ):
        reasons.append("spans_too_few_rotations")
    if not observation_count_sufficient:
        reasons.append("too_few_observations")
    if uses_fap and (
        lsm_false_alarm_probability is None
        or not np.isfinite(lsm_false_alarm_probability)
        or lsm_false_alarm_probability > FAP_SIGNIFICANT
    ):
        reasons.append("no_significant_peak")

    if reasons:
        return _VERDICT_INSUFFICIENT, _RELIABILITY_BY_VERDICT[_VERDICT_INSUFFICIENT], flags, reasons

    # Signal present -> record positive flags.
    if uses_fap and lsm_false_alarm_probability is not None and lsm_false_alarm_probability <= FAP_SIGNIFICANT:
        flags.append("fap_significant")
    if phase_coverage_fraction is not None and phase_coverage_fraction >= PHASE_COVERAGE_MIN_SINGLE:
        flags.append("good_phase_coverage")
    if n_rotations_spanned is not None and n_rotations_spanned >= 2.0 * float(min_rotations_in_span):
        flags.append("multi_night")
    if not is_period_doubled:
        flags.append("two_max_two_min")

    # ---- 2. Alias gate -----------------------------------------------------
    n_aliases = 0 if n_significant_aliases is None else int(n_significant_aliases)
    alias_ambiguous = n_aliases >= 2
    single_max_ambiguous = bool(is_period_doubled)
    good_coverage_for_single = bool(
        phase_coverage_fraction is not None and phase_coverage_fraction >= PHASE_COVERAGE_MIN_SINGLE
    )

    eligible_single = (not alias_ambiguous) and (not single_max_ambiguous) and good_coverage_for_single
    if alias_ambiguous:
        reasons.append("conflicting_aliases")
    if single_max_ambiguous:
        reasons.append("single_max_alias")
    if not good_coverage_for_single and not alias_ambiguous and not single_max_ambiguous:
        reasons.append("phase_coverage_low")

    # ---- 3. Precision gate -------------------------------------------------
    # Greenstreet reliable (uncertainty <= max(2P, 7h)) -> code "3" eligible.
    # Vavilov valid-but-not-reliable -> at most period_family / "2".
    if eligible_single and is_reliable:
        verdict = _VERDICT_SINGLE
    elif is_valid:
        verdict = _VERDICT_FAMILY
    else:
        # Signal exists but neither precision criterion met: believe the family.
        verdict = _VERDICT_FAMILY
        if "no_precision" not in reasons:
            reasons.append("no_precision")

    # ---- 4. Cross-method (hybrid only) ------------------------------------
    if cross_method_agree is True:
        flags.append("dual_method_agree")
        if is_reliable and eligible_single:
            verdict = _VERDICT_SINGLE
    elif cross_method_agree is False:
        # Disagreement caps at period_family.
        if verdict == _VERDICT_SINGLE:
            verdict = _VERDICT_FAMILY
            if "conflicting_aliases" not in reasons:
                reasons.append("conflicting_aliases")

    # LSM-primary results have no validated uncertainty interval (D2): cap at family.
    if primary_method == "lsm" and verdict == _VERDICT_SINGLE:
        verdict = _VERDICT_FAMILY
        if "no_precision" not in reasons:
            reasons.append("no_precision")

    return verdict, _RELIABILITY_BY_VERDICT[verdict], flags, reasons


def _periods_close(period_a: float | None, period_b: float | None) -> bool:
    if period_a is None or period_b is None:
        return False
    if not np.isfinite(period_a) or not np.isfinite(period_b):
        return False
    return bool(abs(float(period_a) - float(period_b)) <= 1.0e-10 * max(1.0, abs(period_a), abs(period_b)))


def _verdict_diagnostics(
    *,
    period_days: float,
    amplitude_mag: float | None,
    residual_sigma_mag: float | None,
    n_significant_aliases: int | None,
    t_rel: npt.NDArray[np.float64],
    span_days: float,
) -> dict[str, object]:
    """Compute the D1 numeric verdict diagnostics for the chosen period."""
    if np.isfinite(period_days) and period_days > 0.0:
        n_rotations_spanned: float | None = float(span_days) / float(period_days)
        phase_coverage_fraction = _phase_coverage_fraction(times=t_rel, period_days=float(period_days))
    else:
        n_rotations_spanned = None
        phase_coverage_fraction = None
    return {
        "amplitude_snr": _amplitude_snr(amplitude_mag, residual_sigma_mag),
        "phase_coverage_fraction": phase_coverage_fraction,
        "n_rotations_spanned": n_rotations_spanned,
        "n_significant_aliases": (
            None if n_significant_aliases is None else int(n_significant_aliases)
        ),
    }


def _primary_from_method(
    *,
    method_mode: str,
    fourier_solution: _FourierSolution,
    lsm_solution: _LSMSolution,
    families: list[_MethodFamily],
    harmonic_period_factors: tuple[float, ...],
    filter_labels: npt.NDArray[np.object_],
    t_rel: npt.NDArray[np.float64],
    span_days: float,
    min_rotations_in_span: float,
    residual_sigma_mag: float | None,
) -> dict[str, object]:
    observation_count_sufficient = _observation_count_sufficient(filter_labels)
    n_fourier_aliases = int(len(fourier_solution.clusters))

    if method_mode == "fourier":
        amplitude_mag = float(fourier_solution.amplitude_mag)
        diagnostics = _verdict_diagnostics(
            period_days=float(fourier_solution.chosen.period_days),
            amplitude_mag=amplitude_mag,
            residual_sigma_mag=residual_sigma_mag,
            n_significant_aliases=n_fourier_aliases,
            t_rel=t_rel,
            span_days=span_days,
        )
        period_verdict, reliability_code, confidence_flags, insufficiency_reasons = _classify_confidence(
            primary_method="fourier",
            amplitude_snr=diagnostics["amplitude_snr"],
            phase_coverage_fraction=diagnostics["phase_coverage_fraction"],
            n_rotations_spanned=diagnostics["n_rotations_spanned"],
            min_rotations_in_span=min_rotations_in_span,
            lsm_false_alarm_probability=None,
            n_significant_aliases=n_fourier_aliases,
            is_period_doubled=bool(fourier_solution.chosen.is_period_doubled),
            observation_count_sufficient=observation_count_sufficient,
            is_reliable=bool(fourier_solution.is_reliable),
            is_valid=bool(fourier_solution.is_valid),
            cross_method_agree=None,
        )
        return {
            "primary_method": "fourier",
            "period_days": float(fourier_solution.chosen.period_days),
            "period_lower_days": float(fourier_solution.period_lower_days),
            "period_upper_days": float(fourier_solution.period_upper_days),
            "relative_period_uncertainty": float(fourier_solution.relative_period_uncertainty),
            "alternate_period_days": list(fourier_solution.alternate_period_days),
            "period_verdict": period_verdict,
            "reliability_code": reliability_code,
            "confidence_flags": confidence_flags,
            "insufficiency_reasons": insufficiency_reasons,
            "is_valid": bool(period_verdict in {_VERDICT_SINGLE, _VERDICT_FAMILY}),
            "is_reliable": bool(period_verdict == _VERDICT_SINGLE),
            "method_agreement_class": None,
            "decision_margin": None,
            "decision_support_count": 1,
            "winner_contains_fourier_primary": True,
            "winner_contains_lsm_primary": False,
            "family_weight_fourier": 1.0,
            "family_weight_lsm": 0.0,
            "selected_method_amplitude_mag": amplitude_mag,
            **diagnostics,
        }

    if method_mode == "lsm":
        period_days = lsm_solution.period_days
        if period_days is None:
            return {
                "primary_method": "lsm",
                "period_days": float("nan"),
                "period_lower_days": None,
                "period_upper_days": None,
                "relative_period_uncertainty": None,
                "alternate_period_days": [],
                "period_verdict": _VERDICT_INSUFFICIENT,
                "reliability_code": _RELIABILITY_BY_VERDICT[_VERDICT_INSUFFICIENT],
                "confidence_flags": [],
                "insufficiency_reasons": ["no_significant_peak"],
                "is_valid": False,
                "is_reliable": False,
                "method_agreement_class": None,
                "decision_margin": None,
                "decision_support_count": 0,
                "winner_contains_fourier_primary": False,
                "winner_contains_lsm_primary": False,
                "family_weight_fourier": 0.0,
                "family_weight_lsm": 1.0,
                "selected_method_amplitude_mag": None,
                "amplitude_snr": None,
                "phase_coverage_fraction": None,
                "n_rotations_spanned": None,
                "n_significant_aliases": 0,
            }
        amplitude_mag = lsm_solution.amplitude_mag
        n_lsm_aliases = int(len(lsm_solution.candidate_period_days))
        diagnostics = _verdict_diagnostics(
            period_days=float(period_days),
            amplitude_mag=amplitude_mag,
            residual_sigma_mag=residual_sigma_mag,
            n_significant_aliases=n_lsm_aliases,
            t_rel=t_rel,
            span_days=span_days,
        )
        period_verdict, reliability_code, confidence_flags, insufficiency_reasons = _classify_confidence(
            primary_method="lsm",
            amplitude_snr=diagnostics["amplitude_snr"],
            phase_coverage_fraction=diagnostics["phase_coverage_fraction"],
            n_rotations_spanned=diagnostics["n_rotations_spanned"],
            min_rotations_in_span=min_rotations_in_span,
            lsm_false_alarm_probability=lsm_solution.false_alarm_probability,
            n_significant_aliases=n_lsm_aliases,
            is_period_doubled=False,
            observation_count_sufficient=observation_count_sufficient,
            is_reliable=bool(lsm_solution.is_reliable),
            is_valid=True,
            cross_method_agree=None,
        )
        alternate_periods = list(lsm_solution.candidate_period_days[1:])
        return {
            "primary_method": "lsm",
            "period_days": float(period_days),
            "period_lower_days": None,
            "period_upper_days": None,
            "relative_period_uncertainty": None,
            "alternate_period_days": alternate_periods,
            "period_verdict": period_verdict,
            "reliability_code": reliability_code,
            "confidence_flags": confidence_flags,
            "insufficiency_reasons": insufficiency_reasons,
            "is_valid": bool(period_verdict in {_VERDICT_SINGLE, _VERDICT_FAMILY}),
            "is_reliable": bool(period_verdict == _VERDICT_SINGLE),
            "method_agreement_class": None,
            "decision_margin": None,
            "decision_support_count": 1,
            "winner_contains_fourier_primary": False,
            "winner_contains_lsm_primary": True,
            "family_weight_fourier": 0.0,
            "family_weight_lsm": 1.0,
            "selected_method_amplitude_mag": amplitude_mag,
            **diagnostics,
        }

    if not families:
        raise ValueError("hybrid mode requires at least one method family")
    ordered_candidates = sorted(
        families,
        key=lambda family: float(family.combined_weight),
        reverse=True,
    )
    winner = ordered_candidates[0]
    runner_up_weight = max(
        (
            float(candidate.combined_weight)
            for candidate in ordered_candidates
            if candidate is not winner
        ),
        default=0.0,
    )
    winner_weight = float(winner.combined_weight)
    support_count = int(winner.fourier_cluster is not None) + int(winner.lsm_candidate is not None)
    fourier_weight = float(winner.family_weight_fourier)
    lsm_weight = float(winner.family_weight_lsm)

    selected_cluster: _FourierCluster | None = None
    if winner.fourier_cluster is not None and (
        winner.lsm_candidate is None or fourier_weight >= lsm_weight
    ):
        selected_method = "fourier"
        selected_cluster = winner.fourier_cluster
        period_days = float(selected_cluster.best.period_days)
        period_lower_days = float(selected_cluster.period_lower_days)
        period_upper_days = float(selected_cluster.period_upper_days)
        relative_uncertainty = _relative_period_uncertainty(
            period_days,
            period_lower_days,
            period_upper_days,
        )
        alternate_period_days = [
            float(cluster.best.period_days)
            for cluster in fourier_solution.clusters
            if cluster is not selected_cluster
        ]
        selected_method_reliable = bool(fourier_solution.is_reliable)
        selected_method_valid = bool(fourier_solution.is_valid)
        selected_amplitude = float(fourier_solution.amplitude_mag)
    elif winner.lsm_candidate is not None:
        selected_method = "lsm"
        selected_candidate = winner.lsm_candidate
        period_days = float(selected_candidate.period_days)
        period_lower_days = None
        period_upper_days = None
        relative_uncertainty = None
        alternate_period_days = [
            float(candidate_period)
            for candidate_period in lsm_solution.candidate_period_days
            if not _periods_close(float(candidate_period), period_days)
        ]
        selected_method_reliable = bool(lsm_solution.is_reliable)
        selected_method_valid = bool(np.isfinite(period_days) and period_days > 0.0)
        selected_amplitude = None if lsm_solution.amplitude_mag is None else float(lsm_solution.amplitude_mag)
    else:
        raise ValueError("winning family has no method support")

    gap = float(max(0.0, winner_weight - runner_up_weight))
    if support_count == 2:
        agreement_class = "consensus" if gap >= 0.25 else "weak_consensus"
    elif selected_method_reliable:
        agreement_class = "method_dominant"
    else:
        agreement_class = "conflict"

    # The selected family carries both methods (support_count == 2) -> Fourier and
    # LSM agree on the rotational family; that drives the cross-method upgrade/cap.
    cross_method_agree = bool(support_count == 2)
    # is_period_doubled flag belongs to the Fourier solution; LSM survivors are
    # required to be two-max/two-min so doubling only applies to a Fourier winner.
    selected_is_period_doubled = bool(
        selected_method == "fourier"
        and selected_cluster is not None
        and selected_cluster.best.is_period_doubled
    )
    n_hybrid_aliases = int(len(families))

    diagnostics = _verdict_diagnostics(
        period_days=float(period_days),
        amplitude_mag=selected_amplitude,
        residual_sigma_mag=residual_sigma_mag,
        n_significant_aliases=n_hybrid_aliases,
        t_rel=t_rel,
        span_days=span_days,
    )
    period_verdict, reliability_code, confidence_flags, insufficiency_reasons = _classify_confidence(
        primary_method="hybrid" if selected_method == "fourier" else "lsm",
        amplitude_snr=diagnostics["amplitude_snr"],
        phase_coverage_fraction=diagnostics["phase_coverage_fraction"],
        n_rotations_spanned=diagnostics["n_rotations_spanned"],
        min_rotations_in_span=min_rotations_in_span,
        lsm_false_alarm_probability=lsm_solution.false_alarm_probability,
        n_significant_aliases=n_hybrid_aliases,
        is_period_doubled=selected_is_period_doubled,
        observation_count_sufficient=observation_count_sufficient,
        is_reliable=bool(selected_method_reliable),
        is_valid=bool(selected_method_valid),
        cross_method_agree=cross_method_agree,
    )

    return {
        "primary_method": selected_method,
        "period_days": float(period_days),
        "period_lower_days": None if period_lower_days is None else float(period_lower_days),
        "period_upper_days": None if period_upper_days is None else float(period_upper_days),
        "relative_period_uncertainty": (
            None if relative_uncertainty is None else float(relative_uncertainty)
        ),
        "alternate_period_days": alternate_period_days,
        "period_verdict": period_verdict,
        "reliability_code": reliability_code,
        "confidence_flags": confidence_flags,
        "insufficiency_reasons": insufficiency_reasons,
        "is_valid": bool(period_verdict in {_VERDICT_SINGLE, _VERDICT_FAMILY}),
        "is_reliable": bool(period_verdict == _VERDICT_SINGLE),
        "method_agreement_class": agreement_class,
        "decision_margin": float(gap),
        "decision_support_count": int(support_count),
        "winner_contains_fourier_primary": bool(winner.contains_fourier_primary),
        "winner_contains_lsm_primary": bool(winner.contains_lsm_primary),
        "family_weight_fourier": float(fourier_weight),
        "family_weight_lsm": float(lsm_weight),
        "selected_method_amplitude_mag": selected_amplitude,
        **diagnostics,
    }


def _pre_solve_insufficiency(
    *,
    n_obs: int,
    filter_labels: npt.NDArray[np.object_],
    span_days: float,
    reduced_mag: npt.NDArray[np.float64],
    mag_sigma: npt.NDArray[np.float64] | None,
    min_order: int,
    min_rotations_in_span: float,
    max_frequency_cycles_per_day: float,
) -> list[str]:
    """Cheap, period-INDEPENDENT insufficiency screen (bead rp-e4a.19).

    Returns the subset of D1 ``insufficiency_reasons`` that are detectable
    without running the search; an empty list means the input looks solvable.
    Reason strings match the full-path ``_classify_confidence`` so the early
    and full verdicts stay consistent.
    """
    reasons: list[str] = []

    # --- too_few_observations: below model size + margin, or below hard floor.
    n_filters = int(len(_ordered_unique(filter_labels)))
    baseline_free_params = n_filters + 2 * int(min_order) + 2
    if n_obs < MIN_OBS or n_obs < baseline_free_params + PRE_SOLVE_OBS_MARGIN:
        reasons.append("too_few_observations")

    # --- insufficient_time_span: zero span, or a degenerate frequency grid
    # (f_min >= f_max) that would otherwise make _build_frequency_grid raise.
    f_min = (
        float("inf")
        if not np.isfinite(span_days) or span_days <= 0.0
        else float(min_rotations_in_span) / float(span_days)
    )
    if not np.isfinite(span_days) or span_days <= 0.0 or f_min >= float(max_frequency_cycles_per_day):
        reasons.append("insufficient_time_span")

    # --- amplitude_below_noise: robust scatter (1.4826*MAD) of the reduced
    # magnitude below ~AMP_SNR_FLOOR x the noise level.  Conservative: only fire
    # when there is clearly no signal.
    y = np.asarray(reduced_mag, dtype=np.float64)
    if y.size > 0 and np.all(np.isfinite(y)):
        robust_scatter = 1.4826 * float(np.median(np.abs(y - np.median(y))))
        if mag_sigma is not None:
            finite_sigma = np.asarray(mag_sigma, dtype=np.float64)
            finite_sigma = finite_sigma[np.isfinite(finite_sigma)]
            noise_level = (
                float(np.median(finite_sigma)) if finite_sigma.size > 0 else PRE_SOLVE_ABS_SCATTER_FLOOR
            )
        else:
            noise_level = PRE_SOLVE_ABS_SCATTER_FLOOR
        if robust_scatter < AMP_SNR_FLOOR * noise_level:
            reasons.append("amplitude_below_noise")

    return reasons


def _insufficient_result(
    *,
    reasons: list[str],
    method_mode: str,
    profile: _FourierProfile,
    observations: RotationPeriodObservations,
    filter_labels: npt.NDArray[np.object_],
    session_labels: npt.NDArray[np.object_] | None,
    time_lt: npt.NDArray[np.float64],
) -> RotationPeriodResult:
    """Build the one-row ``insufficient_data`` result for the early-exit path.

    Mirrors the LSM-no-period branch of ``_primary_from_method``: NaN period /
    frequency, ``None`` for every nullable diagnostic, and the same D1 verdict
    fields.  All non-nullable columns are populated so the row is valid.
    """
    session_summary = _summarize_sessions(time_lt, filter_labels, session_labels)
    primary_method = method_mode if method_mode in {"fourier", "lsm", "hybrid"} else "none"
    return RotationPeriodResult.from_kwargs(
        period_days=[float("nan")],
        period_hours=[float("nan")],
        frequency_cycles_per_day=[float("nan")],
        primary_method=[primary_method],
        paper_profile=[profile.name],
        period_verdict=[_VERDICT_INSUFFICIENT],
        reliability_code=[_RELIABILITY_BY_VERDICT[_VERDICT_INSUFFICIENT]],
        confidence_flags=[[]],
        insufficiency_reasons=[list(reasons)],
        is_valid=[False],
        is_reliable=[False],
        period_lower_days=[None],
        period_upper_days=[None],
        relative_period_uncertainty=[None],
        alternate_period_days=[[]],
        fourier_period_days=[None],
        fourier_order=[None],
        fourier_sigma_threshold=[None],
        fourier_phase_c1=[None],
        fourier_phase_c2=[None],
        residual_sigma_mag=[None],
        fourier_is_valid=[None],
        fourier_is_reliable=[None],
        fourier_alternate_period_days=[[]],
        lsm_period_days=[None],
        lsm_power=[None],
        lsm_power_gap=[None],
        lsm_candidate_period_days=[[]],
        lsm_candidate_powers=[[]],
        lsm_is_reliable=[None],
        lsm_false_alarm_probability=[None],
        method_agreement_class=[None],
        decision_margin=[None],
        decision_support_count=[None],
        winner_contains_fourier_primary=[None],
        winner_contains_lsm_primary=[None],
        family_weight_fourier=[None],
        family_weight_lsm=[None],
        phase_coverage_fraction=[None],
        n_rotations_spanned=[None],
        amplitude_snr=[None],
        n_significant_aliases=[None],
        n_observations=[int(len(observations))],
        n_fit_observations=[0],
        n_clipped=[0],
        n_filters=[int(len(_ordered_unique(filter_labels)))],
        n_sessions=[int(session_summary.n_sessions)],
        used_session_offsets=[False],
        is_period_doubled=[False],
    )


def estimate_rotation_period(
    observations: RotationPeriodObservations,
    *,
    method_mode: str = "hybrid",
    paper_profile: str = "greenstreet_2026",
    search_fidelity: str | None = None,
    search_strategy: str | None = None,
    fourier_orders: tuple[int, ...] | None = None,
    clip_sigma: float = 3.0,
    min_rotations_in_span: float = 2.0,
    max_frequency_cycles_per_day: float = 1000.0,
    frequency_grid_scale: float = 30.0,
    max_search_period_hours: float | None = None,
    early_exit_on_insufficient: bool = False,
    exact_evaluation_backend: str = "numpy",
    jax_frequency_batch_size: int = _DEFAULT_JAX_FREQUENCY_BATCH_SIZE,
    jax_row_pad_multiple: int = _DEFAULT_JAX_ROW_PAD_MULTIPLE,
    session_mode: str = "auto",
    auto_session_min_observations_per_group: int = 6,
    auto_session_max_period_to_session_span_ratio: float = 1.0,
    auto_session_bic_improvement: float = 10.0,
    harmonic_period_factors: tuple[float, ...] = _DEFAULT_HARMONIC_PERIOD_FACTORS,
    lsm_max_frequency_samples: int = _LSM_DEFAULT_MAX_SAMPLES,
    lsm_refine_samples: int = _LSM_DEFAULT_REFINE_SAMPLES,
    lsm_refine_rounds: int = _LSM_DEFAULT_REFINE_ROUNDS,
) -> RotationPeriodResult:
    if method_mode not in {"fourier", "lsm", "hybrid"}:
        raise ValueError("method_mode must be one of {'fourier', 'lsm', 'hybrid'}")
    if clip_sigma <= 0.0:
        raise ValueError("clip_sigma must be positive")
    if min_rotations_in_span <= 0.0:
        raise ValueError("min_rotations_in_span must be positive")
    if max_frequency_cycles_per_day <= 0.0:
        raise ValueError("max_frequency_cycles_per_day must be positive")
    if frequency_grid_scale <= 0.0:
        raise ValueError("frequency_grid_scale must be positive")
    if max_search_period_hours is not None and max_search_period_hours <= 0.0:
        raise ValueError("max_search_period_hours must be positive when set")
    if session_mode not in {"ignore", "use", "auto"}:
        raise ValueError("session_mode must be one of {'ignore', 'use', 'auto'}")
    if auto_session_min_observations_per_group <= 0:
        raise ValueError("auto_session_min_observations_per_group must be positive")
    if auto_session_max_period_to_session_span_ratio <= 0.0:
        raise ValueError("auto_session_max_period_to_session_span_ratio must be positive")
    if auto_session_bic_improvement < 0.0:
        raise ValueError("auto_session_bic_improvement must be non-negative")
    if exact_evaluation_backend not in {"numpy", "jax"}:
        raise ValueError("exact_evaluation_backend must be one of {'numpy', 'jax'}")
    if jax_frequency_batch_size <= 0:
        raise ValueError("jax_frequency_batch_size must be positive")
    if jax_row_pad_multiple <= 0:
        raise ValueError("jax_row_pad_multiple must be positive")
    resolved_fidelity = _resolve_search_fidelity(
        search_fidelity=search_fidelity,
        search_strategy=search_strategy,
    )
    profile = _resolve_paper_profile(paper_profile)

    (
        time,
        mag,
        r_au,
        delta_au,
        phase_angle,
        filter_labels,
        session_labels,
        mag_sigma,
        predicted_mag_v,
    ) = _validate_inputs(observations)

    time_lt = _apply_light_time_correction(time, delta_au)
    t_rel = np.asarray(time_lt - float(np.min(time_lt)), dtype=np.float64)
    span_days = float(np.max(t_rel) - np.min(t_rel))
    reduced_mag = np.asarray(mag - 5.0 * np.log10(r_au * delta_au), dtype=np.float64)

    # Cheap, period-independent insufficiency screen (bead rp-e4a.19).  When the
    # caller opts in, an under-determined / degenerate input early-exits with the
    # standard ``insufficient_data`` verdict instead of building the grid (which
    # can hang or raise on single-night spans).
    if early_exit_on_insufficient:
        pre_solve_reasons = _pre_solve_insufficiency(
            n_obs=int(len(observations)),
            filter_labels=filter_labels,
            span_days=span_days,
            reduced_mag=reduced_mag,
            mag_sigma=mag_sigma,
            min_order=int(min(profile.orders)),
            min_rotations_in_span=min_rotations_in_span,
            max_frequency_cycles_per_day=max_frequency_cycles_per_day,
        )
        if pre_solve_reasons:
            return _insufficient_result(
                reasons=pre_solve_reasons,
                method_mode=method_mode,
                profile=profile,
                observations=observations,
                filter_labels=filter_labels,
                session_labels=session_labels,
                time_lt=time_lt,
            )

    if not np.isfinite(span_days) or span_days <= 0.0:
        raise ValueError("observation time span must be positive")
    weights = _weights_from_sigma(mag_sigma)
    session_summary = _summarize_sessions(time_lt, filter_labels, session_labels)
    frequencies = _build_frequency_grid(
        span_days=span_days,
        min_rotations_in_span=min_rotations_in_span,
        max_frequency_cycles_per_day=max_frequency_cycles_per_day,
        frequency_grid_scale=frequency_grid_scale,
        max_search_period_hours=max_search_period_hours,
    )

    base_design = _build_fixed_design(filter_labels, None, phase_angle)
    baseline_fourier = _derive_fourier_solution(
        t_rel=t_rel,
        y=reduced_mag,
        design_info=base_design,
        frequencies=frequencies,
        weights=weights,
        clip_sigma=clip_sigma,
        profile=profile,
        search_fidelity=resolved_fidelity,
        exact_evaluation_backend=exact_evaluation_backend,
        jax_frequency_batch_size=int(jax_frequency_batch_size),
        jax_row_pad_multiple=int(jax_row_pad_multiple),
        used_session_offsets=False,
        fourier_orders=fourier_orders,
    )
    fourier_solution = baseline_fourier
    used_session_offsets = False

    if session_labels is not None and session_mode in {"use", "auto"}:
        session_design = _build_fixed_design(filter_labels, session_labels, phase_angle)
        session_fourier = _derive_fourier_solution(
            t_rel=t_rel,
            y=reduced_mag,
            design_info=session_design,
            frequencies=frequencies,
            weights=weights,
            clip_sigma=clip_sigma,
            profile=profile,
            search_fidelity=resolved_fidelity,
            exact_evaluation_backend=exact_evaluation_backend,
            jax_frequency_batch_size=int(jax_frequency_batch_size),
            jax_row_pad_multiple=int(jax_row_pad_multiple),
            used_session_offsets=True,
            fourier_orders=fourier_orders,
        )
        if session_mode == "use":
            fourier_solution = session_fourier
            used_session_offsets = True
        else:
            session_eligible = bool(
                session_summary.n_sessions >= 2
                and session_summary.min_group_count >= auto_session_min_observations_per_group
                and session_summary.median_session_span_days > 0.0
                and session_fourier.chosen.period_days
                <= auto_session_max_period_to_session_span_ratio * session_summary.median_session_span_days
            )
            if session_eligible and (
                _fit_bic(session_fourier.fit_summary) + auto_session_bic_improvement
                < _fit_bic(baseline_fourier.fit_summary)
            ):
                fourier_solution = session_fourier
                used_session_offsets = True

    lsm_solution = _LSMSolution(
        period_days=None,
        power=None,
        power_gap=None,
        candidate_period_days=[],
        candidate_powers=[],
        is_reliable=False,
        amplitude_mag=None,
        n_fit_observations=0,
        n_clipped=0,
    )
    if method_mode in {"lsm", "hybrid"}:
        lsm_result = _run_lsm(
            t_rel=t_rel,
            mag=mag,
            reduced_mag=reduced_mag,
            predicted_mag_v=predicted_mag_v,
            phase_angle=phase_angle,
            filter_labels=filter_labels,
            weights=weights,
            lsm_max_frequency_samples=lsm_max_frequency_samples,
            lsm_refine_samples=lsm_refine_samples,
            lsm_refine_rounds=lsm_refine_rounds,
        )
        lsm_solution = _LSMSolution(
            period_days=None if lsm_result.best_candidate is None else float(lsm_result.best_candidate.period_days),
            power=None if lsm_result.best_candidate is None else float(lsm_result.best_candidate.power),
            power_gap=None if lsm_result.power_gap is None else float(lsm_result.power_gap),
            candidate_period_days=list(lsm_result.candidate_period_days),
            candidate_powers=list(lsm_result.candidate_powers),
            is_reliable=bool(lsm_result.is_reliable),
            amplitude_mag=None if lsm_result.amplitude_mag is None else float(lsm_result.amplitude_mag),
            n_fit_observations=int(len(observations))
            if lsm_result.n_fit_observations is None
            else int(lsm_result.n_fit_observations),
            n_clipped=int(lsm_result.n_clipped),
            false_alarm_probability=(
                None
                if lsm_result.false_alarm_probability is None
                else float(lsm_result.false_alarm_probability)
            ),
        )

    families = (
        _build_hybrid_families(
            fourier_solution=fourier_solution,
            lsm_solution=lsm_solution,
            harmonic_period_factors=harmonic_period_factors,
        )
        if method_mode == "hybrid"
        else []
    )

    primary = _primary_from_method(
        method_mode=method_mode,
        fourier_solution=fourier_solution,
        lsm_solution=lsm_solution,
        families=families,
        harmonic_period_factors=harmonic_period_factors,
        filter_labels=filter_labels,
        t_rel=t_rel,
        span_days=span_days,
        min_rotations_in_span=min_rotations_in_span,
        residual_sigma_mag=float(fourier_solution.fit_summary.residual_sigma),
    )
    primary_method = str(primary["primary_method"])
    period_days = float(primary["period_days"])
    period_hours = float(period_days * 24.0)

    # Long-period confidence guardrail (bead rp-e4a.22 step 1).  A
    # ``single_period`` claim at an implausibly long period is almost always
    # low-frequency drift fit as rotation; keep the reported value but downgrade
    # confidence to ``period_family``.
    if (
        str(primary["period_verdict"]) == _VERDICT_SINGLE
        and np.isfinite(period_hours)
        and period_hours > MAX_PLAUSIBLE_SINGLE_PERIOD_HOURS
    ):
        primary["period_verdict"] = _VERDICT_FAMILY
        primary["reliability_code"] = _RELIABILITY_BY_VERDICT[_VERDICT_FAMILY]
        downgrade_reasons = list(cast("list[str]", primary["insufficiency_reasons"]))
        if "period_implausibly_long" not in downgrade_reasons:
            downgrade_reasons.append("period_implausibly_long")
        primary["insufficiency_reasons"] = downgrade_reasons
        primary["is_valid"] = True
        primary["is_reliable"] = False

    selected_period_lower_days = (
        None if primary["period_lower_days"] is None else float(primary["period_lower_days"])
    )
    selected_period_upper_days = (
        None if primary["period_upper_days"] is None else float(primary["period_upper_days"])
    )
    selected_relative_uncertainty = (
        None
        if primary["relative_period_uncertainty"] is None
        else float(primary["relative_period_uncertainty"])
    )
    alternate_period_days = [float(value) for value in primary["alternate_period_days"]]

    if primary_method == "fourier":
        n_fit_observations = int(fourier_solution.fit_summary.n_fit)
        n_clipped = int(fourier_solution.fit_summary.n_clipped)
        is_period_doubled = bool(fourier_solution.chosen.is_period_doubled)
    else:
        n_fit_observations = int(lsm_solution.n_fit_observations)
        n_clipped = int(lsm_solution.n_clipped)
        is_period_doubled = False

    frequency_cycles_per_day = (
        float("nan") if not np.isfinite(period_days) or period_days <= 0.0 else float(1.0 / period_days)
    )

    return RotationPeriodResult.from_kwargs(
        period_days=[period_days],
        period_hours=[period_hours],
        frequency_cycles_per_day=[frequency_cycles_per_day],
        primary_method=[primary_method],
        paper_profile=[profile.name],
        period_verdict=[str(primary["period_verdict"])],
        reliability_code=[str(primary["reliability_code"])],
        confidence_flags=[list(primary["confidence_flags"])],
        insufficiency_reasons=[list(primary["insufficiency_reasons"])],
        is_valid=[bool(primary["is_valid"])],
        is_reliable=[bool(primary["is_reliable"])],
        period_lower_days=[selected_period_lower_days],
        period_upper_days=[selected_period_upper_days],
        relative_period_uncertainty=[selected_relative_uncertainty],
        alternate_period_days=[alternate_period_days],
        fourier_period_days=[float(fourier_solution.chosen.period_days)],
        fourier_order=[int(fourier_solution.fit_summary.fourier_order)],
        fourier_sigma_threshold=[float(fourier_solution.sigma_threshold)],
        fourier_phase_c1=[float(fourier_solution.fit_summary.coeffs[fourier_solution.fit_summary.phase_c1_idx])],
        fourier_phase_c2=[float(fourier_solution.fit_summary.coeffs[fourier_solution.fit_summary.phase_c2_idx])],
        residual_sigma_mag=[float(fourier_solution.fit_summary.residual_sigma)],
        fourier_is_valid=[bool(fourier_solution.is_valid)],
        fourier_is_reliable=[bool(fourier_solution.is_reliable)],
        fourier_alternate_period_days=[list(fourier_solution.alternate_period_days)],
        lsm_period_days=[None if lsm_solution.period_days is None else float(lsm_solution.period_days)],
        lsm_power=[None if lsm_solution.power is None else float(lsm_solution.power)],
        lsm_power_gap=[None if lsm_solution.power_gap is None else float(lsm_solution.power_gap)],
        lsm_candidate_period_days=[list(lsm_solution.candidate_period_days)],
        lsm_candidate_powers=[list(lsm_solution.candidate_powers)],
        lsm_is_reliable=[bool(lsm_solution.is_reliable)],
        lsm_false_alarm_probability=[
            None
            if lsm_solution.false_alarm_probability is None
            else float(lsm_solution.false_alarm_probability)
        ],
        method_agreement_class=[primary["method_agreement_class"]],
        decision_margin=[None if primary["decision_margin"] is None else float(primary["decision_margin"])],
        decision_support_count=[
            None if primary["decision_support_count"] is None else int(primary["decision_support_count"])
        ],
        winner_contains_fourier_primary=[
            None
            if primary["winner_contains_fourier_primary"] is None
            else bool(primary["winner_contains_fourier_primary"])
        ],
        winner_contains_lsm_primary=[
            None
            if primary["winner_contains_lsm_primary"] is None
            else bool(primary["winner_contains_lsm_primary"])
        ],
        family_weight_fourier=[
            None if primary["family_weight_fourier"] is None else float(primary["family_weight_fourier"])
        ],
        family_weight_lsm=[
            None if primary["family_weight_lsm"] is None else float(primary["family_weight_lsm"])
        ],
        phase_coverage_fraction=[
            None
            if primary["phase_coverage_fraction"] is None
            else float(primary["phase_coverage_fraction"])
        ],
        n_rotations_spanned=[
            None if primary["n_rotations_spanned"] is None else float(primary["n_rotations_spanned"])
        ],
        amplitude_snr=[None if primary["amplitude_snr"] is None else float(primary["amplitude_snr"])],
        n_significant_aliases=[
            None if primary["n_significant_aliases"] is None else int(primary["n_significant_aliases"])
        ],
        n_observations=[int(len(observations))],
        n_fit_observations=[n_fit_observations],
        n_clipped=[n_clipped],
        n_filters=[int(len(_ordered_unique(filter_labels)))],
        n_sessions=[int(session_summary.n_sessions)],
        used_session_offsets=[bool(used_session_offsets)],
        is_period_doubled=[bool(is_period_doubled)],
    )
