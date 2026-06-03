from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from scipy.stats import f as f_dist

from .rotation_period_types import RotationPeriodObservations

LIGHT_SPEED_AU_PER_DAY = 173.14463267424034
_HG_G_MIN = -0.25
_HG_G_MAX = 0.95
_HG_PHASE_MIN_DEG = 1.0
_HG_PHASE_MAX_DEG = 120.0
_HG_ARC_HALF_WIDTH_DEG = 0.5
_PRIOR_SIGMA_FLOOR = 1.0e-6


@dataclass(slots=True)
class _FitResult:
    frequency: float
    fourier_order: int
    coeffs: npt.NDArray[np.float64]
    residual_sigma: float
    rss: float
    df: int
    n_par: int
    mask: npt.NDArray[np.bool_]
    n_fit: int
    n_clipped: int
    n_filters: int
    phase_c1_idx: int
    phase_c2_idx: int
    sum_weights: float | None = None
    n_local_maxima: int | None = None


@dataclass(slots=True)
class _DesignInfo:
    fixed: npt.NDArray[np.float64]
    unique_filters: list[str]
    n_filters: int
    phase_c1_idx: int
    phase_c2_idx: int


@dataclass(slots=True)
class _SessionSummary:
    n_sessions: int
    min_group_count: int
    median_group_count: float
    median_session_span_days: float


@dataclass(slots=True)
class _FitWithPeriod:
    fit: _FitResult
    period_days: float
    period_hours: float
    is_period_doubled: bool


@dataclass(slots=True)
class _HarmonicAdjudication:
    selected: _FitWithPeriod
    near_tie_candidates: int
    had_near_tie: bool


@dataclass(frozen=True, slots=True)
class _FourierProfile:
    name: str
    orders: tuple[int, ...]
    order_selection_confidence: float
    sigma_threshold_confidence: float
    valid_relative_uncertainty_max: float | None
    reliable_relative_multiplier: float | None
    reliable_absolute_hours: float | None


# Single MVP solver profile, built off the Greenstreet (2026) rotation-period method:
# search Fourier orders 2-6, pick the order by F-test at 90% confidence, cluster aliases
# at the 95% sigma threshold, and call a period "reliable" when its uncertainty is within
# max(2P, 7h). (An experimental second profile was dropped for the MVP — one profile only.)
FOURIER_PROFILES: dict[str, _FourierProfile] = {
    "greenstreet_2026": _FourierProfile(
        name="greenstreet_2026",
        orders=(2, 3, 4, 5, 6),
        order_selection_confidence=0.90,
        sigma_threshold_confidence=0.95,
        valid_relative_uncertainty_max=None,
        reliable_relative_multiplier=2.0,
        reliable_absolute_hours=7.0,
    ),
}


def _to_numpy_1d(values: pa.Array | pa.ChunkedArray | npt.ArrayLike) -> npt.NDArray:
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        return np.asarray(values.to_numpy(zero_copy_only=False))
    return np.asarray(values)


def _ordered_unique(values: npt.NDArray[np.object_]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values.tolist():
        label = "__missing__" if value is None else str(value)
        if label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered


def _normalized_string_labels(
    values: npt.NDArray[np.object_],
    *,
    missing_label: str,
) -> npt.NDArray[np.object_]:
    return np.asarray(
        [missing_label if value is None else str(value) for value in values.tolist()],
        dtype=object,
    )


def _validate_inputs(
    observations: RotationPeriodObservations,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.object_],
    npt.NDArray[np.object_] | None,
    npt.NDArray[np.float64] | None,
    npt.NDArray[np.float64] | None,
]:
    n = len(observations)
    if n == 0:
        raise ValueError("observations must be non-empty")

    time = np.asarray(observations.time.rescale("tdb").mjd().to_numpy(False), dtype=np.float64)
    mag = np.asarray(observations.mag.to_numpy(zero_copy_only=False), dtype=np.float64)
    r_au = np.asarray(observations.r_au.to_numpy(zero_copy_only=False), dtype=np.float64)
    delta_au = np.asarray(observations.delta_au.to_numpy(zero_copy_only=False), dtype=np.float64)
    phase_angle = np.asarray(
        observations.phase_angle_deg.to_numpy(zero_copy_only=False), dtype=np.float64
    )

    if any(arr.shape != (n,) for arr in (time, mag, r_au, delta_au, phase_angle)):
        raise ValueError("all observation columns must be 1D and aligned")
    if not (
        np.all(np.isfinite(time))
        and np.all(np.isfinite(mag))
        and np.all(np.isfinite(r_au))
        and np.all(np.isfinite(delta_au))
        and np.all(np.isfinite(phase_angle))
    ):
        raise ValueError(
            "observations must contain finite time, mag, r_au, delta_au, and phase_angle_deg"
        )
    if np.any(r_au <= 0.0) or np.any(delta_au <= 0.0):
        raise ValueError("r_au and delta_au must be positive")

    filter_values = _to_numpy_1d(observations.filter)
    if filter_values.shape != (n,):
        raise ValueError("filter column must be 1D and aligned with observations")
    filter_labels = _normalized_string_labels(
        np.asarray(filter_values, dtype=object),
        missing_label="__missing_filter__",
    )

    session_labels = None
    session_values = _to_numpy_1d(observations.session_id)
    if session_values.shape != (n,):
        raise ValueError("session_id column must be 1D and aligned with observations")
    session_labels_raw = np.asarray(session_values, dtype=object)
    if np.any(np.asarray([value is not None for value in session_labels_raw.tolist()], dtype=bool)):
        session_labels = np.asarray(
            [None if value is None else str(value) for value in session_labels_raw.tolist()],
            dtype=object,
        )

    mag_sigma = None
    raw_sigma = np.asarray(observations.mag_sigma.to_numpy(zero_copy_only=False), dtype=np.float64)
    if raw_sigma.shape != (n,):
        raise ValueError("mag_sigma column must be 1D and aligned with observations")
    if np.any(np.isfinite(raw_sigma)):
        mag_sigma = raw_sigma

    predicted_mag_v = None
    raw_predicted = np.asarray(
        observations.predicted_mag_v.to_numpy(zero_copy_only=False), dtype=np.float64
    )
    if raw_predicted.shape != (n,):
        raise ValueError("predicted_mag_v column must be 1D and aligned with observations")
    if np.any(np.isfinite(raw_predicted)):
        predicted_mag_v = raw_predicted

    return (
        time,
        mag,
        r_au,
        delta_au,
        phase_angle,
        filter_labels,
        session_labels,
        mag_sigma,
        predicted_mag_v,
    )


def _apply_light_time_correction(
    time_mjd_tdb: npt.NDArray[np.float64],
    delta_au: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return np.asarray(time_mjd_tdb, dtype=np.float64) - np.asarray(delta_au, dtype=np.float64) / LIGHT_SPEED_AU_PER_DAY


def _build_fixed_design(
    filter_labels: npt.NDArray[np.object_],
    session_labels: npt.NDArray[np.object_] | None,
    phase_angle: npt.NDArray[np.float64],
) -> _DesignInfo:
    unique_filters = _ordered_unique(filter_labels)
    n = len(filter_labels)
    cols: list[npt.NDArray[np.float64]] = [np.ones(n, dtype=np.float64)]
    for label in unique_filters[1:]:
        cols.append((filter_labels == label).astype(np.float64))

    if session_labels is not None:
        normalized_sessions = _normalized_string_labels(
            np.asarray(session_labels, dtype=object),
            missing_label="__missing_session__",
        )
        for filter_label in unique_filters:
            filter_mask = filter_labels == filter_label
            ordered_sessions = _ordered_unique(normalized_sessions[filter_mask])
            for session_label in ordered_sessions[1:]:
                cols.append(
                    np.logical_and(filter_mask, normalized_sessions == session_label).astype(
                        np.float64
                    )
                )

    phase_angle = np.asarray(phase_angle, dtype=np.float64)
    cols.append(phase_angle)
    cols.append(np.square(phase_angle))
    fixed = np.column_stack(cols).astype(np.float64, copy=False)
    return _DesignInfo(
        fixed=fixed,
        unique_filters=unique_filters,
        n_filters=len(unique_filters),
        phase_c1_idx=fixed.shape[1] - 2,
        phase_c2_idx=fixed.shape[1] - 1,
    )


def _summarize_sessions(
    time: npt.NDArray[np.float64],
    filter_labels: npt.NDArray[np.object_],
    session_labels: npt.NDArray[np.object_] | None,
) -> _SessionSummary:
    if session_labels is None:
        return _SessionSummary(
            n_sessions=0,
            min_group_count=0,
            median_group_count=0.0,
            median_session_span_days=0.0,
        )

    session_labels_norm = _normalized_string_labels(
        np.asarray(session_labels, dtype=object),
        missing_label="__missing_session__",
    )
    unique_sessions = _ordered_unique(session_labels_norm)
    unique_filters = _ordered_unique(filter_labels)
    group_counts: list[int] = []
    spans: list[float] = []
    for session_label in unique_sessions:
        session_mask = session_labels_norm == session_label
        session_times = np.asarray(time[session_mask], dtype=np.float64)
        if session_times.size > 0:
            spans.append(float(np.max(session_times) - np.min(session_times)))
        for filter_label in unique_filters:
            count = int(np.count_nonzero(np.logical_and(session_mask, filter_labels == filter_label)))
            if count > 0:
                group_counts.append(count)

    return _SessionSummary(
        n_sessions=len(unique_sessions),
        min_group_count=min(group_counts) if group_counts else 0,
        median_group_count=float(np.median(np.asarray(group_counts, dtype=np.float64)))
        if group_counts
        else 0.0,
        median_session_span_days=float(np.median(np.asarray(spans, dtype=np.float64)))
        if spans
        else 0.0,
    )


def _build_fourier_columns(
    t_rel: npt.NDArray[np.float64],
    frequency: float,
    fourier_order: int,
) -> npt.NDArray[np.float64]:
    omega_t = 2.0 * np.pi * float(frequency) * np.asarray(t_rel, dtype=np.float64)
    cols: list[npt.NDArray[np.float64]] = []
    for harmonic in range(1, int(fourier_order) + 1):
        angle = harmonic * omega_t
        cols.append(np.cos(angle))
        cols.append(np.sin(angle))
    return np.column_stack(cols).astype(np.float64, copy=False)


def _hg_phase_reduced(alpha_deg: npt.NDArray[np.float64], g_value: float) -> npt.NDArray[np.float64]:
    alpha_rad = np.radians(np.asarray(alpha_deg, dtype=np.float64))
    tan_half = np.tan(0.5 * alpha_rad)
    phi1 = np.exp(-3.33 * np.power(tan_half, 0.63))
    phi2 = np.exp(-1.87 * np.power(tan_half, 1.22))
    phase = (1.0 - g_value) * phi1 + g_value * phi2
    phase = np.clip(phase, 1.0e-12, None)
    return -2.5 * np.log10(phase)


def _phase_prior_bounds(min_alpha_deg: float) -> tuple[tuple[float, float], tuple[float, float]]:
    alpha_center = float(np.clip(min_alpha_deg, _HG_PHASE_MIN_DEG, _HG_PHASE_MAX_DEG))
    alpha_arc = np.linspace(
        max(_HG_PHASE_MIN_DEG, alpha_center - _HG_ARC_HALF_WIDTH_DEG),
        min(_HG_PHASE_MAX_DEG, alpha_center + _HG_ARC_HALF_WIDTH_DEG),
        25,
        dtype=np.float64,
    )
    coeffs_min = np.polyfit(alpha_arc, _hg_phase_reduced(alpha_arc, _HG_G_MIN), deg=2)
    coeffs_max = np.polyfit(alpha_arc, _hg_phase_reduced(alpha_arc, _HG_G_MAX), deg=2)
    c1_min, c1_max = sorted((float(coeffs_min[1]), float(coeffs_max[1])))
    c2_min, c2_max = sorted((float(coeffs_min[0]), float(coeffs_max[0])))
    return (c1_min, c1_max), (c2_min, c2_max)


def _phase_prior_rows(
    n_par: int,
    design_info: _DesignInfo,
    min_phase_angle_deg: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    (c1_min, c1_max), (c2_min, c2_max) = _phase_prior_bounds(min_phase_angle_deg)
    rows = np.zeros((2, n_par), dtype=np.float64)
    rows[0, design_info.phase_c1_idx] = 1.0
    rows[1, design_info.phase_c2_idx] = 1.0
    target = np.asarray(
        [0.5 * (c1_min + c1_max), 0.5 * (c2_min + c2_max)],
        dtype=np.float64,
    )
    sigma = np.asarray(
        [
            max(0.5 * abs(c1_max - c1_min), _PRIOR_SIGMA_FLOOR),
            max(0.5 * abs(c2_max - c2_min), _PRIOR_SIGMA_FLOOR),
        ],
        dtype=np.float64,
    )
    weights = 1.0 / np.square(sigma)
    return rows, target, weights


def _weighted_lstsq(
    design: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64] | None,
) -> tuple[npt.NDArray[np.float64], float, int]:
    if weights is None:
        coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
        resid = target - design @ coeffs
        rss = float(np.sum(np.square(resid)))
    else:
        w = np.asarray(weights, dtype=np.float64)
        sqrt_w = np.sqrt(w)
        coeffs, *_ = np.linalg.lstsq(design * sqrt_w[:, None], target * sqrt_w, rcond=None)
        resid = target - design @ coeffs
        rss = float(np.sum(w * np.square(resid)))
    return np.asarray(coeffs, dtype=np.float64), rss, int(target.size - design.shape[1])


def _paper_sigma(
    residuals: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64] | None,
    *,
    n_obs: int,
    n_fit: int,
    n_par: int,
) -> tuple[float, float, int]:
    df = int(n_fit - n_par)
    if df <= 0:
        return float("inf"), float("inf"), df
    if weights is None:
        rss = float(np.sum(np.square(residuals)))
        sigma2 = rss / float(df)
        return float(np.sqrt(max(sigma2, 0.0))), rss, df

    w = np.asarray(weights, dtype=np.float64)
    rss = float(np.sum(w * np.square(residuals)))
    weight_sum = float(np.sum(w))
    if weight_sum <= 0.0:
        return float("inf"), float("inf"), df
    sigma2 = (float(n_obs) / weight_sum) * rss / float(df)
    return float(np.sqrt(max(sigma2, 0.0))), rss, df


def _fit_frequency(
    t_rel: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    design_info: _DesignInfo,
    frequency: float,
    fourier_order: int,
    *,
    clip_sigma: float,
    weights: npt.NDArray[np.float64] | None,
    max_clip_iterations: int = 8,
) -> _FitResult | None:
    n_obs = len(y)
    fixed = np.asarray(design_info.fixed, dtype=np.float64)
    n_par = int(fixed.shape[1] + 2 * int(fourier_order))
    if n_obs <= n_par:
        return None

    if weights is None:
        weights_real = None
    else:
        weights_real = np.asarray(weights, dtype=np.float64)

    min_phase = float(np.min(fixed[:, design_info.phase_c1_idx]))
    prior_rows, prior_target, prior_weights = _phase_prior_rows(n_par, design_info, min_phase)
    mask = np.ones(n_obs, dtype=bool)

    for _ in range(max(1, int(max_clip_iterations))):
        idx = np.flatnonzero(mask)
        if idx.size <= n_par:
            return None
        fourier = _build_fourier_columns(t_rel[idx], frequency, fourier_order)
        design_real = np.concatenate([fixed[idx], fourier], axis=1)
        target_real = np.asarray(y[idx], dtype=np.float64)
        design = np.vstack([design_real, prior_rows])
        target = np.concatenate([target_real, prior_target])
        if weights_real is None:
            weights_aug = np.concatenate(
                [np.ones(target_real.size, dtype=np.float64), prior_weights]
            )
            real_weights = None
        else:
            weights_aug = np.concatenate([weights_real[idx], prior_weights])
            real_weights = weights_real[idx]
        coeffs, _, _ = _weighted_lstsq(design, target, weights_aug)
        residuals = target_real - design_real @ coeffs
        sigma, rss, df = _paper_sigma(
            residuals,
            real_weights,
            n_obs=n_obs,
            n_fit=idx.size,
            n_par=n_par,
        )
        if not np.isfinite(sigma):
            return None
        clip_mask = np.abs(residuals) <= float(clip_sigma) * sigma
        if np.all(clip_mask):
            return _FitResult(
                frequency=float(frequency),
                fourier_order=int(fourier_order),
                coeffs=np.asarray(coeffs, dtype=np.float64),
                residual_sigma=float(sigma),
                rss=float(rss),
                df=int(df),
                n_par=int(n_par),
                mask=mask.copy(),
                n_fit=int(idx.size),
                n_clipped=int(n_obs - idx.size),
                n_filters=int(design_info.n_filters),
                phase_c1_idx=int(design_info.phase_c1_idx),
                phase_c2_idx=int(design_info.phase_c2_idx),
                sum_weights=None if real_weights is None else float(np.sum(real_weights)),
            )
        new_mask = mask.copy()
        new_mask[idx[~clip_mask]] = False
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    idx = np.flatnonzero(mask)
    if idx.size <= n_par:
        return None
    fourier = _build_fourier_columns(t_rel[idx], frequency, fourier_order)
    design_real = np.concatenate([fixed[idx], fourier], axis=1)
    target_real = np.asarray(y[idx], dtype=np.float64)
    design = np.vstack([design_real, prior_rows])
    target = np.concatenate([target_real, prior_target])
    if weights_real is None:
        weights_aug = np.concatenate([np.ones(target_real.size, dtype=np.float64), prior_weights])
        real_weights = None
    else:
        weights_aug = np.concatenate([weights_real[idx], prior_weights])
        real_weights = weights_real[idx]
    coeffs, _, _ = _weighted_lstsq(design, target, weights_aug)
    residuals = target_real - design_real @ coeffs
    sigma, rss, df = _paper_sigma(
        residuals,
        real_weights,
        n_obs=n_obs,
        n_fit=idx.size,
        n_par=n_par,
    )
    if not np.isfinite(sigma):
        return None
    return _FitResult(
        frequency=float(frequency),
        fourier_order=int(fourier_order),
        coeffs=np.asarray(coeffs, dtype=np.float64),
        residual_sigma=float(sigma),
        rss=float(rss),
        df=int(df),
        n_par=int(n_par),
        mask=mask.copy(),
        n_fit=int(idx.size),
        n_clipped=int(n_obs - idx.size),
        n_filters=int(design_info.n_filters),
        phase_c1_idx=int(design_info.phase_c1_idx),
        phase_c2_idx=int(design_info.phase_c2_idx),
        sum_weights=None if real_weights is None else float(np.sum(real_weights)),
    )


def _fit_frequency_unclipped(
    t_rel: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    design_info: _DesignInfo,
    frequency: float,
    fourier_order: int,
    *,
    weights: npt.NDArray[np.float64] | None,
) -> _FitResult | None:
    n_obs = len(y)
    fixed = np.asarray(design_info.fixed, dtype=np.float64)
    n_par = int(fixed.shape[1] + 2 * int(fourier_order))
    if n_obs <= n_par:
        return None

    fourier = _build_fourier_columns(t_rel, frequency, fourier_order)
    design_real = np.concatenate([fixed, fourier], axis=1)
    target_real = np.asarray(y, dtype=np.float64)
    min_phase = float(np.min(fixed[:, design_info.phase_c1_idx]))
    prior_rows, prior_target, prior_weights = _phase_prior_rows(n_par, design_info, min_phase)
    design = np.vstack([design_real, prior_rows])
    target = np.concatenate([target_real, prior_target])
    if weights is None:
        weights_aug = np.concatenate([np.ones(n_obs, dtype=np.float64), prior_weights])
        real_weights = None
    else:
        real_weights = np.asarray(weights, dtype=np.float64)
        weights_aug = np.concatenate([real_weights, prior_weights])
    coeffs, _, _ = _weighted_lstsq(design, target, weights_aug)
    residuals = target_real - design_real @ coeffs
    sigma, rss, df = _paper_sigma(
        residuals,
        real_weights,
        n_obs=n_obs,
        n_fit=n_obs,
        n_par=n_par,
    )
    if not np.isfinite(sigma):
        return None
    return _FitResult(
        frequency=float(frequency),
        fourier_order=int(fourier_order),
        coeffs=np.asarray(coeffs, dtype=np.float64),
        residual_sigma=float(sigma),
        rss=float(rss),
        df=int(df),
        n_par=int(n_par),
        mask=np.ones(n_obs, dtype=bool),
        n_fit=int(n_obs),
        n_clipped=0,
        n_filters=int(design_info.n_filters),
        phase_c1_idx=int(design_info.phase_c1_idx),
        phase_c2_idx=int(design_info.phase_c2_idx),
        sum_weights=None if real_weights is None else float(np.sum(real_weights)),
    )


def _f_test_confidence(small: _FitResult, large: _FitResult) -> float:
    if large.n_par <= small.n_par:
        return 0.0
    if not np.isfinite(small.residual_sigma) or not np.isfinite(large.residual_sigma):
        return 0.0
    df_small = int(small.df)
    df_large = int(large.df)
    if df_small <= 0 or df_large <= 0:
        return 0.0
    variance_small = float(small.residual_sigma) ** 2
    variance_large = float(large.residual_sigma) ** 2
    if variance_small <= variance_large or variance_large <= 0.0:
        return 0.0
    f_value = variance_small / variance_large
    if not np.isfinite(f_value) or f_value <= 1.0:
        return 0.0
    # The papers describe F-tests in terms of the fitted sigma values and
    # the number of fitted observations, so use a directional variance-ratio
    # confidence rather than a nested-model extra-sum-of-squares test.
    return float(f_dist.cdf(f_value, df_small, df_large))


def _directional_f_test_confidence(small: _FitResult, large: _FitResult) -> float:
    return _f_test_confidence(small, large)


def _fit_bic(fit: _FitResult) -> float:
    n = int(fit.n_fit)
    if n <= fit.n_par or fit.rss <= 0.0:
        return float("inf")
    return float(n * np.log(fit.rss / float(n)) + fit.n_par * np.log(float(n)))


def _select_order(candidate_fits: dict[int, _FitResult], required_confidence: float) -> _FitResult:
    valid_orders = sorted(candidate_fits)
    for i, order in enumerate(valid_orders):
        candidate = candidate_fits[order]
        significantly_worse = False
        for higher_order in valid_orders[i + 1 :]:
            higher = candidate_fits[higher_order]
            if _f_test_confidence(candidate, higher) > float(required_confidence):
                significantly_worse = True
                break
        if not significantly_worse:
            return candidate
    return candidate_fits[valid_orders[-1]]


def _sigma_threshold_for_confidence(best_fit: _FitResult, confidence: float) -> float:
    if best_fit.df <= 0:
        return float("inf")
    f_critical = float(f_dist.ppf(float(confidence), best_fit.df, best_fit.df))
    if not np.isfinite(f_critical) or f_critical <= 0.0:
        return float("inf")
    return float(best_fit.residual_sigma * np.sqrt(f_critical))


def _sigma_threshold_from_confidence(
    best_sigma: float,
    df: int,
    confidence: float,
) -> float:
    if df <= 0:
        return float("inf")
    f_critical = float(f_dist.ppf(float(confidence), int(df), int(df)))
    if not np.isfinite(f_critical) or f_critical <= 0.0:
        return float("inf")
    return float(float(best_sigma) * np.sqrt(f_critical))


def _count_local_extrema(
    coeffs: npt.NDArray[np.float64],
    fourier_order: int,
    dense_points: int = 2048,
) -> tuple[int, int]:
    phase = np.linspace(0.0, 1.0, dense_points, endpoint=False)
    periodic = np.zeros_like(phase)
    start = coeffs.size - 2 * int(fourier_order)
    for harmonic in range(1, int(fourier_order) + 1):
        idx = start + 2 * (harmonic - 1)
        periodic += coeffs[idx] * np.cos(2.0 * np.pi * harmonic * phase)
        periodic += coeffs[idx + 1] * np.sin(2.0 * np.pi * harmonic * phase)
    prev_vals = np.roll(periodic, 1)
    next_vals = np.roll(periodic, -1)
    maxima = (periodic > prev_vals) & (periodic >= next_vals)
    minima = (periodic < prev_vals) & (periodic <= next_vals)
    return int(np.count_nonzero(maxima)), int(np.count_nonzero(minima))


def _amplitude_from_fit(
    fit: _FitResult,
    dense_points: int = 4096,
) -> float:
    phase = np.linspace(0.0, 1.0, dense_points, endpoint=False)
    periodic = np.zeros_like(phase)
    start = fit.coeffs.size - 2 * int(fit.fourier_order)
    for harmonic in range(1, int(fit.fourier_order) + 1):
        idx = start + 2 * (harmonic - 1)
        periodic += fit.coeffs[idx] * np.cos(2.0 * np.pi * harmonic * phase)
        periodic += fit.coeffs[idx + 1] * np.sin(2.0 * np.pi * harmonic * phase)
    return float(np.max(periodic) - np.min(periodic))


def _fit_with_period(fit: _FitResult) -> _FitWithPeriod:
    n_maxima = fit.n_local_maxima
    if n_maxima is None:
        n_maxima, _ = _count_local_extrema(fit.coeffs, fit.fourier_order)
    is_period_doubled = n_maxima == 1
    period_days = (2.0 if is_period_doubled else 1.0) / float(fit.frequency)
    return _FitWithPeriod(
        fit=fit,
        period_days=float(period_days),
        period_hours=float(period_days * 24.0),
        is_period_doubled=bool(is_period_doubled),
    )
