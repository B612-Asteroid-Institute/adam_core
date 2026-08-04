from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import quivr as qv
from scipy.stats import f as f_dist  # type: ignore[import-untyped]

from ...constants import Constants
from ...time import Timestamp
from ..magnitude_common import hg_phase_correction

if TYPE_CHECKING:
    from ...coordinates.cartesian import CartesianCoordinates
    from ...observations.detections import PointSourceDetections
    from ...observations.exposures import Exposures

# Speed of light in au/day. Use the canonical adam_core constant rather than a local
# literal to avoid drift.
LIGHT_SPEED_AU_PER_DAY = Constants.C
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
    n_fit: int
    n_clipped: int
    phase_c1_idx: int
    phase_c2_idx: int


@dataclass(slots=True)
class _DesignInfo:
    fixed: npt.NDArray[np.float64]
    n_filters: int
    phase_c1_idx: int
    phase_c2_idx: int


@dataclass(slots=True)
class _SessionSummary:
    n_sessions: int
    min_group_count: int
    median_session_span_days: float


@dataclass(slots=True)
class _FitWithPeriod:
    fit: _FitResult
    period_days: float
    period_hours: float
    is_period_doubled: bool


@dataclass(frozen=True, slots=True)
class _FourierProfile:
    name: str
    orders: tuple[int, ...]
    order_selection_confidence: float
    sigma_threshold_confidence: float
    valid_relative_uncertainty_max: float | None
    reliable_relative_multiplier: float | None
    reliable_absolute_hours: float | None


# Solver profile, built off the Greenstreet (2026) rotation-period method:
# search Fourier orders 2-6, pick the order by F-test at 90% confidence, cluster aliases
# at the 95% sigma threshold, and call a period "reliable" when its uncertainty is within
# max(2P, 7h).
PROFILES: dict[str, _FourierProfile] = {
    "default": _FourierProfile(
        name="default",
        orders=(2, 3, 4, 5, 6),
        order_selection_confidence=0.90,
        sigma_threshold_confidence=0.95,
        valid_relative_uncertainty_max=None,
        reliable_relative_multiplier=2.0,
        reliable_absolute_hours=7.0,
    ),
}


def _to_numpy_1d(
    values: pa.Array | pa.ChunkedArray | npt.ArrayLike,
) -> npt.NDArray[Any]:
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
]:
    n = len(observations)
    if n == 0:
        raise ValueError("observations must be non-empty")

    time = np.asarray(
        observations.time.rescale("tdb").mjd().to_numpy(False), dtype=np.float64
    )
    mag = np.asarray(observations.mag.to_numpy(zero_copy_only=False), dtype=np.float64)
    r_au = np.asarray(
        observations.r_au.to_numpy(zero_copy_only=False), dtype=np.float64
    )
    delta_au = np.asarray(
        observations.delta_au.to_numpy(zero_copy_only=False), dtype=np.float64
    )
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
    if np.any(
        np.asarray(
            [value is not None for value in session_labels_raw.tolist()], dtype=bool
        )
    ):
        session_labels = np.asarray(
            [
                None if value is None else str(value)
                for value in session_labels_raw.tolist()
            ],
            dtype=object,
        )

    mag_sigma = None
    raw_sigma = np.asarray(
        observations.mag_sigma.to_numpy(zero_copy_only=False), dtype=np.float64
    )
    if raw_sigma.shape != (n,):
        raise ValueError("mag_sigma column must be 1D and aligned with observations")
    if np.any(np.isfinite(raw_sigma)):
        mag_sigma = raw_sigma

    return (
        time,
        mag,
        r_au,
        delta_au,
        phase_angle,
        filter_labels,
        session_labels,
        mag_sigma,
    )


def _apply_light_time_correction(
    time_mjd_tdb: npt.NDArray[np.float64],
    delta_au: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    corrected = np.asarray(time_mjd_tdb, dtype=np.float64) - (
        np.asarray(delta_au, dtype=np.float64) / LIGHT_SPEED_AU_PER_DAY
    )
    return np.asarray(corrected, dtype=np.float64)


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
                    np.logical_and(
                        filter_mask, normalized_sessions == session_label
                    ).astype(np.float64)
                )

    phase_angle = np.asarray(phase_angle, dtype=np.float64)
    cols.append(phase_angle)
    cols.append(np.square(phase_angle))
    fixed = np.column_stack(cols).astype(np.float64, copy=False)
    return _DesignInfo(
        fixed=fixed,
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
            count = int(
                np.count_nonzero(
                    np.logical_and(session_mask, filter_labels == filter_label)
                )
            )
            if count > 0:
                group_counts.append(count)

    return _SessionSummary(
        n_sessions=len(unique_sessions),
        min_group_count=min(group_counts) if group_counts else 0,
        median_session_span_days=(
            float(np.median(np.asarray(spans, dtype=np.float64))) if spans else 0.0
        ),
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


def _phase_prior_bounds(
    min_alpha_deg: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    alpha_center = float(np.clip(min_alpha_deg, _HG_PHASE_MIN_DEG, _HG_PHASE_MAX_DEG))
    alpha_arc = np.linspace(
        max(_HG_PHASE_MIN_DEG, alpha_center - _HG_ARC_HALF_WIDTH_DEG),
        min(_HG_PHASE_MAX_DEG, alpha_center + _HG_ARC_HALF_WIDTH_DEG),
        25,
        dtype=np.float64,
    )
    coeffs_min = np.polyfit(alpha_arc, hg_phase_correction(alpha_arc, _HG_G_MIN), deg=2)
    coeffs_max = np.polyfit(alpha_arc, hg_phase_correction(alpha_arc, _HG_G_MAX), deg=2)
    c1_min, c1_max = sorted((float(coeffs_min[1]), float(coeffs_max[1])))
    c2_min, c2_max = sorted((float(coeffs_min[0]), float(coeffs_max[0])))
    return (c1_min, c1_max), (c2_min, c2_max)


def _phase_prior_rows(
    n_par: int,
    design_info: _DesignInfo,
    min_phase_angle_deg: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Two weighted pseudo-observations constraining the phase-angle (H-G) coefficients.

    NOTE: these rows make every Fourier fit a RIDGE-REGULARIZED
    least-squares solve, not ordinary least squares. The reported RSS / residual_sigma
    / df / BIC / F-test are computed on the REAL observations only and exclude this
    prior penalty, so those statistics (and the order selection, sigma thresholds, and
    uncertainty intervals derived from them) are regularized/approximate, not exact OLS
    quantities. Treat them as such when interpreting the confidence surface.
    """
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
) -> npt.NDArray[np.float64]:
    if weights is None:
        coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    else:
        sqrt_w = np.sqrt(np.asarray(weights, dtype=np.float64))
        coeffs, *_ = np.linalg.lstsq(
            design * sqrt_w[:, None], target * sqrt_w, rcond=None
        )
    return np.asarray(coeffs, dtype=np.float64)


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
    prior_rows, prior_target, prior_weights = _phase_prior_rows(
        n_par, design_info, min_phase
    )
    mask = np.ones(n_obs, dtype=bool)

    def _solve(
        idx: npt.NDArray[np.int64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, float, int]:
        """Weighted, prior-augmented Fourier fit at the unmasked rows ``idx``."""
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
        coeffs = _weighted_lstsq(design, target, weights_aug)
        residuals = target_real - design_real @ coeffs
        sigma, rss, df = _paper_sigma(
            residuals, real_weights, n_obs=n_obs, n_fit=idx.size, n_par=n_par
        )
        return coeffs, residuals, sigma, rss, df

    def _result(
        idx: npt.NDArray[np.int64],
        coeffs: npt.NDArray[np.float64],
        sigma: float,
        rss: float,
        df: int,
    ) -> _FitResult:
        return _FitResult(
            frequency=float(frequency),
            fourier_order=int(fourier_order),
            coeffs=np.asarray(coeffs, dtype=np.float64),
            residual_sigma=float(sigma),
            rss=float(rss),
            df=int(df),
            n_par=int(n_par),
            n_fit=int(idx.size),
            n_clipped=int(n_obs - idx.size),
            phase_c1_idx=int(design_info.phase_c1_idx),
            phase_c2_idx=int(design_info.phase_c2_idx),
        )

    for _ in range(max(1, int(max_clip_iterations))):
        idx = np.flatnonzero(mask)
        if idx.size <= n_par:
            return None
        coeffs, residuals, sigma, rss, df = _solve(idx)
        if not np.isfinite(sigma):
            return None
        clip_mask = np.abs(residuals) <= float(clip_sigma) * sigma
        if np.all(clip_mask):
            return _result(idx, coeffs, sigma, rss, df)
        new_mask = mask.copy()
        new_mask[idx[~clip_mask]] = False
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    idx = np.flatnonzero(mask)
    if idx.size <= n_par:
        return None
    coeffs, residuals, sigma, rss, df = _solve(idx)
    if not np.isfinite(sigma):
        return None
    return _result(idx, coeffs, sigma, rss, df)


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
    prior_rows, prior_target, prior_weights = _phase_prior_rows(
        n_par, design_info, min_phase
    )
    design = np.vstack([design_real, prior_rows])
    target = np.concatenate([target_real, prior_target])
    if weights is None:
        weights_aug = np.concatenate([np.ones(n_obs, dtype=np.float64), prior_weights])
        real_weights = None
    else:
        real_weights = np.asarray(weights, dtype=np.float64)
        weights_aug = np.concatenate([real_weights, prior_weights])
    coeffs = _weighted_lstsq(design, target, weights_aug)
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
        n_fit=int(n_obs),
        n_clipped=0,
        phase_c1_idx=int(design_info.phase_c1_idx),
        phase_c2_idx=int(design_info.phase_c2_idx),
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


def _fit_bic(fit: _FitResult) -> float:
    n = int(fit.n_fit)
    if n <= fit.n_par or fit.rss <= 0.0:
        return float("inf")
    return float(n * np.log(fit.rss / float(n)) + fit.n_par * np.log(float(n)))


def _select_order(
    candidate_fits: dict[int, _FitResult], required_confidence: float
) -> _FitResult:
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


def _periodic_amplitude_from_coeffs(
    coeffs: npt.NDArray[np.float64],
    fourier_order: int,
    dense_points: int = 4096,
) -> float:
    """Peak-to-peak amplitude of the periodic (Fourier) part of a coefficient vector.

    Canonical amplitude helper for the Fourier path (start = len - 2*order
    convention), shared by ``_amplitude_from_fit``.
    """
    phase = np.linspace(0.0, 1.0, dense_points, endpoint=False)
    periodic = np.zeros_like(phase)
    start = coeffs.size - 2 * int(fourier_order)
    for harmonic in range(1, int(fourier_order) + 1):
        idx = start + 2 * (harmonic - 1)
        periodic += coeffs[idx] * np.cos(2.0 * np.pi * harmonic * phase)
        periodic += coeffs[idx + 1] * np.sin(2.0 * np.pi * harmonic * phase)
    return float(np.max(periodic) - np.min(periodic))


def _amplitude_from_fit(
    fit: _FitResult,
    dense_points: int = 4096,
) -> float:
    return _periodic_amplitude_from_coeffs(fit.coeffs, fit.fourier_order, dense_points)


def _fit_with_period(fit: _FitResult) -> _FitWithPeriod:
    n_maxima, _ = _count_local_extrema(fit.coeffs, fit.fourier_order)
    is_period_doubled = n_maxima == 1
    period_days = (2.0 if is_period_doubled else 1.0) / float(fit.frequency)
    return _FitWithPeriod(
        fit=fit,
        period_days=float(period_days),
        period_hours=float(period_days * 24.0),
        is_period_doubled=bool(is_period_doubled),
    )


# ---------------------------------------------------------------------------
# Public data model: rotation-period observations and results.
# ---------------------------------------------------------------------------


class RotationPeriodObservations(qv.Table):
    time = Timestamp.as_column()
    mag = qv.Float64Column()
    mag_sigma = qv.Float64Column(nullable=True)
    filter = qv.LargeStringColumn(nullable=True)
    session_id = qv.LargeStringColumn(nullable=True)
    r_au = qv.Float64Column()
    delta_au = qv.Float64Column()
    phase_angle_deg = qv.Float64Column()

    @classmethod
    def from_point_source_observations(
        cls,
        detections: PointSourceDetections,
        exposures: Exposures,
        object_coords: CartesianCoordinates,
    ) -> RotationPeriodObservations:
        """Build observations from adam_core point-source detections + exposures.

        Links this table to the core adam_core observation primitives: one row per
        ``PointSourceDetections`` entry, with ``filter`` and the per-exposure
        observing geometry (heliocentric distance ``r_au``, observer distance
        ``delta_au``, and solar ``phase_angle_deg``) derived from the aligned
        ``Exposures`` and the object's heliocentric ``CartesianCoordinates``.

        ``object_coords`` must be heliocentric (origin=SUN) and the same length and
        order as ``detections``; ``detections.exposure_id`` is used to align each
        detection to its exposure. ``mag`` / ``r_au`` / ``delta_au`` /
        ``phase_angle_deg`` must be finite (and the distances positive) or a
        ``ValueError`` is raised.
        """
        # Lazy import avoids a module-load cycle (the wrappers module imports this
        # one); the geometry pipeline lives in that adapter module.
        from .wrappers import (
            build_rotation_period_observations_from_detections,
        )

        return build_rotation_period_observations_from_detections(
            detections, exposures, object_coords
        )


class RotationPeriodResult(qv.Table):
    period_days = qv.Float64Column()
    period_hours = qv.Float64Column()
    frequency_cycles_per_day = qv.Float64Column()
    profile = qv.LargeStringColumn()
    period_verdict = qv.LargeStringColumn()
    # LCDB-U-style reliability code as a STRING ("3"/"2"/"1", highest first). Kept a
    # string (not int) to mirror LCDB U codes and stay forward-compatible with
    # qualified codes (e.g. "1+"); do NOT sort or compare it numerically.
    reliability_code = qv.LargeStringColumn()
    confidence_flags = qv.LargeListColumn(pa.large_string(), nullable=True)
    insufficiency_reasons = qv.LargeListColumn(pa.large_string(), nullable=True)
    is_valid = qv.BooleanColumn()
    is_reliable = qv.BooleanColumn()
    period_lower_days = qv.Float64Column(nullable=True)
    period_upper_days = qv.Float64Column(nullable=True)
    relative_period_uncertainty = qv.Float64Column(nullable=True)
    alternate_period_days = qv.LargeListColumn(pa.float64(), nullable=True)
    fourier_period_days = qv.Float64Column(nullable=True)
    fourier_order = qv.Int64Column(nullable=True)
    fourier_sigma_threshold = qv.Float64Column(nullable=True)
    fourier_phase_c1 = qv.Float64Column(nullable=True)
    fourier_phase_c2 = qv.Float64Column(nullable=True)
    residual_sigma_mag = qv.Float64Column(nullable=True)
    fourier_is_valid = qv.BooleanColumn(nullable=True)
    fourier_is_reliable = qv.BooleanColumn(nullable=True)
    fourier_alternate_period_days = qv.LargeListColumn(pa.float64(), nullable=True)
    phase_coverage_fraction = qv.Float64Column(nullable=True)
    n_rotations_spanned = qv.Float64Column(nullable=True)
    amplitude_snr = qv.Float64Column(nullable=True)
    n_significant_aliases = qv.Int64Column(nullable=True)
    n_observations = qv.Int64Column()
    n_fit_observations = qv.Int64Column()
    n_clipped = qv.Int64Column()
    n_filters = qv.Int64Column()
    n_sessions = qv.Int64Column()
    used_session_offsets = qv.BooleanColumn()
    is_period_doubled = qv.BooleanColumn()

    @classmethod
    def single_insufficient(
        cls,
        *,
        reasons: list[str],
        confidence_flags: list[str] | None = None,
        n_observations: int = 0,
        n_filters: int = 0,
        n_sessions: int = 0,
        profile: str = "default",
    ) -> "RotationPeriodResult":
        """One-row ``insufficient_data`` result: NaN period, every nullable diagnostic
        ``None``.

        The canonical builder for the insufficient verdict. Used both by the solver's
        early-exit path and by the detection wrappers when an object cannot be solved,
        so a grouped solve returns one row per object id rather than silently dropping
        failures. ``period_verdict``/``reliability_code`` are the contract constants
        (``"insufficient_data"`` / ``"1"``).
        """
        return cls.from_kwargs(
            period_days=[float("nan")],
            period_hours=[float("nan")],
            frequency_cycles_per_day=[float("nan")],
            profile=[profile],
            period_verdict=["insufficient_data"],
            reliability_code=["1"],
            confidence_flags=[list(confidence_flags or [])],
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
            phase_coverage_fraction=[None],
            n_rotations_spanned=[None],
            amplitude_snr=[None],
            n_significant_aliases=[None],
            n_observations=[int(n_observations)],
            n_fit_observations=[0],
            n_clipped=[0],
            n_filters=[int(n_filters)],
            n_sessions=[int(n_sessions)],
            used_session_offsets=[False],
            is_period_doubled=[False],
        )


class GroupedRotationPeriodResults(qv.Table):
    object_id = qv.LargeStringColumn()
    result = RotationPeriodResult.as_column()


# ---------------------------------------------------------------------------
# Period-recovery scoring metrics (standard-candle validation). All periods in
# hours; recovered and LCDB truth are both synodic (no sidereal correction).
# ---------------------------------------------------------------------------

HARMONIC_FACTORS: list[float] = [
    0.25,
    1.0 / 3.0,
    0.5,
    2.0 / 3.0,
    0.75,
    1.0,
    4.0 / 3.0,
    1.5,
    2.0,
    3.0,
    4.0,
]
"""Harmonic factors applied to the recovered period when diagnosing aliases."""


def relative_error_pct(p_rec: float, p_true: float) -> float:
    """Strict relative error of a recovered period, in percent.

    Parameters
    ----------
    p_rec : float
        Recovered period (hours).
    p_true : float
        True (LCDB) period (hours); must be positive.

    Returns
    -------
    float
        ``100 * |p_rec - p_true| / p_true``.
    """
    return float(100.0 * abs(float(p_rec) - float(p_true)) / float(p_true))


def harmonic_adjusted_error_pct(p_rec: float, p_true: float) -> tuple[float, float]:
    """Harmonic-tolerant relative error and the best-fitting harmonic factor.

    Minimises ``|p_rec * f - p_true| / p_true`` over :data:`HARMONIC_FACTORS`.

    Parameters
    ----------
    p_rec : float
        Recovered period (hours).
    p_true : float
        True (LCDB) period (hours); must be positive.

    Returns
    -------
    tuple of (float, float)
        The minimum harmonic-adjusted error in percent, and the factor ``f``
        that achieves it.
    """
    factors = np.asarray(HARMONIC_FACTORS, dtype=np.float64)
    errors = np.abs(float(p_rec) * factors - float(p_true)) / float(p_true)
    best = int(np.argmin(errors))
    return float(100.0 * errors[best]), float(factors[best])


def alias_bucket(best_factor: float) -> str:
    """Label the harmonic-alias bucket for the best-fitting factor.

    Parameters
    ----------
    best_factor : float
        The harmonic factor returned by :func:`harmonic_adjusted_error_pct`.

    Returns
    -------
    str
        One of ``"1x"``, ``"1/4x"``, ``"1/3x"``, ``"1/2x"``, ``"2/3x"``,
        ``"3/4x"``, ``"4/3x"``, ``"3/2x"``, ``"2x"``, ``"3x"``, ``"4x"``, or
        ``"other"`` if no factor is within 5% of ``best_factor``.
    """
    labels: list[tuple[float, str]] = [
        (1.0, "1x"),
        (0.25, "1/4x"),
        (1.0 / 3.0, "1/3x"),
        (0.5, "1/2x"),
        (2.0 / 3.0, "2/3x"),
        (0.75, "3/4x"),
        (4.0 / 3.0, "4/3x"),
        (1.5, "3/2x"),
        (2.0, "2x"),
        (3.0, "3x"),
        (4.0, "4x"),
    ]
    factor = float(best_factor)
    for value, label in labels:
        if abs(factor - value) <= 0.05 * value:
            return label
    return "other"


def within_tolerance(p_rec: float, p_true: float, tolerance_fraction: float) -> bool:
    """Whether the raw relative error is within the fixture's tolerance.

    Uses the strict :func:`relative_error_pct` (no harmonic adjustment) compared
    against the per-fixture ``tolerance_fraction`` from the npz.

    Parameters
    ----------
    p_rec : float
        Recovered period (hours).
    p_true : float
        True (LCDB) period (hours); must be positive.
    tolerance_fraction : float
        Allowed fractional error (e.g. ``0.01`` for a tight U=3 fixture).

    Returns
    -------
    bool
        ``True`` if ``relative_error_pct(p_rec, p_true) <= tolerance_fraction * 100``.
    """
    return bool(relative_error_pct(p_rec, p_true) <= float(tolerance_fraction) * 100.0)


def near_day_alias(
    p_rec_hours: float,
    p_true_hours: float,
    tolerance_fraction: float = 0.02,
) -> bool:
    """Whether the recovered frequency is a diurnal-cadence alias of the truth.

    Flags lock-on to a day-aliased frequency (cycles per day): the recovered
    frequency sits within ``tolerance_fraction`` of ``f_true +/- n`` for ``n`` in
    ``{1, 2}`` cycles/day, i.e. ``|f_rec - (f_true +/- n)| / f_true <= tol``.

    Parameters
    ----------
    p_rec_hours : float
        Recovered period (hours); must be positive.
    p_true_hours : float
        True (LCDB) period (hours); must be positive.
    tolerance_fraction : float, optional
        Fractional tolerance on the day-aliased frequency match (default 0.02).

    Returns
    -------
    bool
        ``True`` if a day-alias relationship holds.
    """
    p_rec = float(p_rec_hours)
    p_true = float(p_true_hours)
    if p_rec <= 0.0 or p_true <= 0.0:
        return False
    f_rec = 24.0 / p_rec
    f_true = 24.0 / p_true
    tol = float(tolerance_fraction)
    for n in (1, 2):
        for aliased in (f_true + n, f_true - n):
            if aliased <= 0.0:
                continue
            if abs(f_rec - aliased) / f_true <= tol:
                return True
    return False
