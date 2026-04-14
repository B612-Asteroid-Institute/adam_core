from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from scipy.stats import f as f_dist

from .rotation_period_types import RotationPeriodObservations


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
        raise ValueError("observations must contain finite time, mag, r_au, delta_au, and phase_angle_deg")
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
    if hasattr(observations, "session_id"):
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
    if hasattr(observations, "mag_sigma"):
        raw_sigma = np.asarray(
            observations.mag_sigma.to_numpy(zero_copy_only=False), dtype=np.float64
        )
        if raw_sigma.shape != (n,):
            raise ValueError("mag_sigma column must be 1D and aligned with observations")
        if np.any(np.isfinite(raw_sigma)):
            mag_sigma = raw_sigma

    return time, mag, r_au, delta_au, phase_angle, filter_labels, session_labels, mag_sigma


def _build_fixed_design(
    filter_labels: npt.NDArray[np.object_],
    session_labels: npt.NDArray[np.object_] | None,
    phase_angle: npt.NDArray[np.float64],
) -> _DesignInfo:
    unique_filters = _ordered_unique(filter_labels)
    n = len(filter_labels)
    n_filters = len(unique_filters)

    cols: list[npt.NDArray[np.float64]] = [np.ones(n, dtype=np.float64)]
    for label in unique_filters[1:]:
        cols.append((filter_labels == label).astype(np.float64))

    if session_labels is not None:
        normalized_sessions = _normalized_string_labels(
            session_labels,
            missing_label="__missing_session__",
        )
        for filter_label in unique_filters:
            filter_mask = filter_labels == filter_label
            filter_sessions = normalized_sessions[filter_mask]
            ordered_sessions = _ordered_unique(filter_sessions)
            for session_label in ordered_sessions[1:]:
                cols.append(
                    np.logical_and(filter_mask, normalized_sessions == session_label).astype(
                        np.float64
                    )
                )

    cols.append(np.asarray(phase_angle, dtype=np.float64))
    cols.append(np.square(phase_angle))
    fixed = np.column_stack(cols).astype(np.float64, copy=False)
    return _DesignInfo(
        fixed=fixed,
        unique_filters=unique_filters,
        n_filters=n_filters,
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
    omega_t = 2.0 * np.pi * frequency * t_rel
    cols = []
    for harmonic in range(1, fourier_order + 1):
        angle = harmonic * omega_t
        cols.append(np.cos(angle))
        cols.append(np.sin(angle))
    return np.column_stack(cols).astype(np.float64, copy=False)


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
        sqrt_w = np.sqrt(weights)
        coeffs, *_ = np.linalg.lstsq(design * sqrt_w[:, None], target * sqrt_w, rcond=None)
        resid = target - design @ coeffs
        rss = float(np.sum(weights * np.square(resid)))
    return np.asarray(coeffs, dtype=np.float64), rss, int(target.size - design.shape[1])


def _fit_frequency(
    t_rel: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    design_info: _DesignInfo,
    frequency: float,
    fourier_order: int,
    *,
    clip_sigma: float,
    weights: npt.NDArray[np.float64] | None,
) -> _FitResult | None:
    n = len(y)
    fixed = design_info.fixed
    n_par = fixed.shape[1] + 2 * fourier_order
    if n <= n_par:
        return None

    mask = np.ones(n, dtype=bool)

    for _ in range(3):
        idx = np.flatnonzero(mask)
        if idx.size <= n_par:
            return None
        fourier = _build_fourier_columns(t_rel[idx], frequency, fourier_order)
        design = np.concatenate([fixed[idx], fourier], axis=1)
        target = y[idx]
        coeffs, rss, df = _weighted_lstsq(
            design,
            target,
            None if weights is None else weights[idx],
        )
        if df <= 0:
            return None
        sigma = float(np.sqrt(rss / df))
        if not np.isfinite(sigma):
            return None
        resid = target - design @ coeffs
        clip_mask = np.abs(resid) <= clip_sigma * sigma
        if np.all(clip_mask):
            break
        new_mask = mask.copy()
        new_mask[idx[~clip_mask]] = False
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    idx = np.flatnonzero(mask)
    if idx.size <= n_par:
        return None

    fourier = _build_fourier_columns(t_rel[idx], frequency, fourier_order)
    design = np.concatenate([fixed[idx], fourier], axis=1)
    target = y[idx]
    coeffs, rss, df = _weighted_lstsq(
        design,
        target,
        None if weights is None else weights[idx],
    )
    if df <= 0:
        return None
    sigma = float(np.sqrt(rss / df))
    if not np.isfinite(sigma):
        return None

    return _FitResult(
        frequency=float(frequency),
        fourier_order=int(fourier_order),
        coeffs=np.asarray(coeffs, dtype=np.float64),
        residual_sigma=sigma,
        rss=float(rss),
        df=int(df),
        n_par=int(n_par),
        mask=mask,
        n_fit=int(idx.size),
        n_clipped=int(n - idx.size),
        n_filters=int(design_info.n_filters),
        phase_c1_idx=int(design_info.phase_c1_idx),
        phase_c2_idx=int(design_info.phase_c2_idx),
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
    n = len(y)
    fixed = design_info.fixed
    n_par = fixed.shape[1] + 2 * fourier_order
    if n <= n_par:
        return None

    fourier = _build_fourier_columns(t_rel, frequency, fourier_order)
    design = np.concatenate([fixed, fourier], axis=1)
    coeffs, rss, df = _weighted_lstsq(design, y, weights)
    if df <= 0:
        return None
    sigma = float(np.sqrt(rss / df))
    if not np.isfinite(sigma):
        return None

    return _FitResult(
        frequency=float(frequency),
        fourier_order=int(fourier_order),
        coeffs=np.asarray(coeffs, dtype=np.float64),
        residual_sigma=sigma,
        rss=float(rss),
        df=int(df),
        n_par=int(n_par),
        mask=np.ones(n, dtype=bool),
        n_fit=int(n),
        n_clipped=0,
        n_filters=int(design_info.n_filters),
        phase_c1_idx=int(design_info.phase_c1_idx),
        phase_c2_idx=int(design_info.phase_c2_idx),
    )


def _f_test_p_value(small: _FitResult, large: _FitResult) -> float:
    if large.df <= 0 or small.df <= large.df:
        return 1.0
    if small.rss <= large.rss:
        return 1.0

    df_num = small.df - large.df
    if df_num <= 0:
        return 1.0
    numerator = (small.rss - large.rss) / float(df_num)
    denominator = large.rss / float(large.df)
    if denominator <= 0.0 or numerator <= 0.0:
        return 0.0
    f_value = numerator / denominator
    return float(f_dist.sf(f_value, df_num, large.df))


def _fit_bic(fit: _FitResult) -> float:
    n = int(fit.n_fit)
    if n <= fit.n_par or fit.rss <= 0.0:
        return float("inf")
    return float(n * np.log(fit.rss / float(n)) + fit.n_par * np.log(float(n)))


def _select_order(candidate_fits: dict[int, _FitResult], order_selection_p_value: float) -> _FitResult:
    valid_orders = sorted(candidate_fits)
    for i, order in enumerate(valid_orders):
        small = candidate_fits[order]
        if small is None:
            continue
        significant_better = False
        for higher_order in valid_orders[i + 1 :]:
            large = candidate_fits[higher_order]
            if large is None:
                continue
            if _f_test_p_value(small, large) < order_selection_p_value:
                significant_better = True
                break
        if not significant_better:
            return small

    for order in reversed(valid_orders):
        fit = candidate_fits[order]
        if fit is not None:
            return fit
    raise ValueError("no valid Fourier order fits were found")


def _count_local_maxima(
    coeffs: npt.NDArray[np.float64],
    fourier_order: int,
    dense_points: int = 2048,
) -> int:
    phase = np.linspace(0.0, 1.0, dense_points, endpoint=False)
    periodic = np.zeros_like(phase)
    start = coeffs.size - 2 * fourier_order
    for harmonic in range(1, fourier_order + 1):
        idx = start + 2 * (harmonic - 1)
        periodic += coeffs[idx] * np.cos(2.0 * np.pi * harmonic * phase)
        periodic += coeffs[idx + 1] * np.sin(2.0 * np.pi * harmonic * phase)

    prev_vals = np.roll(periodic, 1)
    next_vals = np.roll(periodic, -1)
    maxima = (periodic > prev_vals) & (periodic >= next_vals)
    return int(np.count_nonzero(maxima))


def _fit_with_period(fit: _FitResult) -> _FitWithPeriod:
    is_period_doubled = _count_local_maxima(fit.coeffs, fit.fourier_order) == 1
    period_days = (2.0 if is_period_doubled else 1.0) / fit.frequency
    return _FitWithPeriod(
        fit=fit,
        period_days=float(period_days),
        period_hours=float(period_days * 24.0),
        is_period_doubled=bool(is_period_doubled),
    )


def _harmonic_relative_mismatch(
    period_days_a: float,
    period_days_b: float,
    *,
    harmonic_period_factors: tuple[float, ...],
) -> float:
    if not np.isfinite(period_days_a) or period_days_a <= 0.0:
        return float("inf")
    if not np.isfinite(period_days_b) or period_days_b <= 0.0:
        return float("inf")
    factors = np.asarray(sorted({float(factor) for factor in harmonic_period_factors}), dtype=np.float64)
    if factors.size == 0:
        return float(abs(period_days_a - period_days_b) / period_days_b)
    return float(np.min(np.abs(period_days_a * factors - period_days_b) / period_days_b))
