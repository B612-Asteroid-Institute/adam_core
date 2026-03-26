from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from scipy.stats import f as f_dist

from .rotation_period_types import RotationPeriodObservations, RotationPeriodResult


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


def estimate_rotation_period(
    observations: RotationPeriodObservations,
    *,
    fourier_orders: tuple[int, ...] = (2, 3, 4, 5, 6),
    clip_sigma: float = 3.0,
    order_selection_p_value: float = 0.05,
    min_rotations_in_span: float = 2.0,
    max_frequency_cycles_per_day: float = 1000.0,
    frequency_grid_scale: float = 30.0,
) -> RotationPeriodResult:
    """
    Estimate a rotation period from reduced photometric observations using a
    high-order Fourier search.
    """
    if clip_sigma <= 0.0:
        raise ValueError("clip_sigma must be positive")
    if order_selection_p_value <= 0.0 or order_selection_p_value >= 1.0:
        raise ValueError("order_selection_p_value must be in (0, 1)")
    if min_rotations_in_span <= 0.0:
        raise ValueError("min_rotations_in_span must be positive")
    if max_frequency_cycles_per_day <= 0.0:
        raise ValueError("max_frequency_cycles_per_day must be positive")
    if frequency_grid_scale <= 0.0:
        raise ValueError("frequency_grid_scale must be positive")

    orders = sorted({int(order) for order in fourier_orders})
    if not orders:
        raise ValueError("fourier_orders must be non-empty")
    if any(order < 2 for order in orders):
        raise ValueError("fourier_orders must be >= 2")

    (
        time,
        mag,
        r_au,
        delta_au,
        phase_angle,
        filter_labels,
        session_labels,
        mag_sigma,
    ) = _validate_inputs(observations)

    y = mag - 5.0 * np.log10(r_au * delta_au)
    time_rel = time - float(np.min(time))
    span = float(np.max(time_rel) - np.min(time_rel))
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError("observation time span must be positive")

    f_min = float(min_rotations_in_span / span)
    f_max = float(max_frequency_cycles_per_day)
    if not np.isfinite(f_min) or f_min <= 0.0:
        raise ValueError("derived minimum frequency is invalid")
    if f_max <= f_min:
        raise ValueError("max_frequency_cycles_per_day must exceed the minimum searchable frequency")

    n_freq = int(ceil(frequency_grid_scale * span * (f_max - f_min)) + 1)
    n_freq = max(n_freq, 2)
    frequencies = np.linspace(f_min, f_max, n_freq, dtype=np.float64)

    valid_sigma = mag_sigma is not None and np.all(np.isfinite(mag_sigma)) and np.all(mag_sigma > 0.0)
    weights = None if not valid_sigma else 1.0 / np.square(np.asarray(mag_sigma, dtype=np.float64))

    design_info = _build_fixed_design(filter_labels, session_labels, phase_angle)
    candidate_fits: dict[int, _FitResult] = {}
    for order in orders:
        best_fit: _FitResult | None = None
        for frequency in frequencies:
            fit = _fit_frequency(
                time_rel,
                y,
                design_info,
                float(frequency),
                int(order),
                clip_sigma=clip_sigma,
                weights=weights,
            )
            if fit is None:
                continue
            if best_fit is None or fit.residual_sigma < best_fit.residual_sigma:
                best_fit = fit
        if best_fit is not None:
            candidate_fits[int(order)] = best_fit

    if not candidate_fits:
        raise ValueError("no valid rotation-period fit could be found")

    chosen = _select_order(candidate_fits, order_selection_p_value)
    local_maxima = _count_local_maxima(chosen.coeffs, chosen.fourier_order)
    is_period_doubled = local_maxima == 1

    period_days = (2.0 if is_period_doubled else 1.0) / chosen.frequency
    period_hours = period_days * 24.0

    n_obs = int(len(observations))
    n_fit = int(chosen.n_fit)
    n_clipped = int(chosen.n_clipped)
    n_filters = int(chosen.n_filters)

    return RotationPeriodResult.from_kwargs(
        period_days=[period_days],
        period_hours=[period_hours],
        frequency_cycles_per_day=[chosen.frequency],
        fourier_order=[chosen.fourier_order],
        phase_c1=[float(chosen.coeffs[chosen.phase_c1_idx])],
        phase_c2=[float(chosen.coeffs[chosen.phase_c2_idx])],
        residual_sigma_mag=[float(chosen.residual_sigma)],
        n_observations=[n_obs],
        n_fit_observations=[n_fit],
        n_clipped=[n_clipped],
        n_filters=[n_filters],
        is_period_doubled=[bool(is_period_doubled)],
    )
