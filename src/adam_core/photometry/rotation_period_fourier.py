from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import ray
from scipy.stats import f as f_dist

from ..ray_cluster import initialize_use_ray
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


@dataclass(slots=True)
class _SessionSummary:
    n_sessions: int
    min_group_count: int
    median_group_count: float
    median_session_span_days: float


@dataclass(slots=True)
class _RaySearchState:
    time_rel_ref: object
    y_ref: object
    design_info_ref: object
    frequencies_ref: object
    weights_ref: object | None
    max_processes: int


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


def _sample_indices_from_intervals(
    intervals: list[tuple[int, int]],
    stride: int,
    n_total: int,
) -> npt.NDArray[np.int64]:
    if n_total <= 0:
        return np.zeros(0, dtype=np.int64)
    if stride <= 1 and len(intervals) == 1 and intervals[0] == (0, n_total - 1):
        return np.arange(n_total, dtype=np.int64)

    idx_set: set[int] = set()
    for start, end in intervals:
        lo = max(0, min(int(start), n_total - 1))
        hi = max(0, min(int(end), n_total - 1))
        if hi < lo:
            lo, hi = hi, lo
        idx_set.add(lo)
        idx_set.add(hi)
        first = lo + ((stride - (lo % stride)) % stride)
        for idx in range(first, hi + 1, max(1, stride)):
            idx_set.add(idx)
    return np.asarray(sorted(idx_set), dtype=np.int64)


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


def _select_seed_indices(
    sample_indices: npt.NDArray[np.int64],
    scores: npt.NDArray[np.float64],
    *,
    candidate_count: int,
) -> list[int]:
    valid_positions = np.flatnonzero(np.isfinite(scores))
    if valid_positions.size == 0:
        return []

    local_minima: list[int] = []
    for pos in valid_positions.tolist():
        center = float(scores[pos])
        left = float(scores[pos - 1]) if pos > 0 and np.isfinite(scores[pos - 1]) else np.inf
        right = (
            float(scores[pos + 1])
            if pos + 1 < len(scores) and np.isfinite(scores[pos + 1])
            else np.inf
        )
        if center <= left and center <= right:
            local_minima.append(pos)

    positions = local_minima if local_minima else valid_positions.tolist()
    positions = sorted(positions, key=lambda pos: float(scores[pos]))
    sample_step = 1
    if sample_indices.size > 1:
        sample_step = int(
            max(
                1,
                round(float(np.median(np.diff(np.asarray(sample_indices, dtype=np.int64))))),
            )
        )

    chosen: list[int] = []
    for pos in positions:
        idx = int(sample_indices[pos])
        if any(abs(idx - existing) < sample_step for existing in chosen):
            continue
        chosen.append(idx)
        if len(chosen) >= candidate_count:
            break

    if not chosen:
        chosen.append(int(sample_indices[int(valid_positions[0])]))
    return chosen


def _candidate_intervals_from_scores(
    scores_by_order: dict[int, tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]],
    *,
    candidate_count: int,
    radius: int,
    n_total: int,
    include_harmonic_partners: bool = False,
) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for sample_indices, scores in scores_by_order.values():
        for center in _select_seed_indices(
            sample_indices,
            scores,
            candidate_count=candidate_count,
        ):
            centers = [center]
            if include_harmonic_partners:
                half_center = int(round(center / 2.0))
                double_center = int(round(center * 2.0))
                if 0 <= half_center < n_total:
                    centers.append(half_center)
                if 0 <= double_center < n_total:
                    centers.append(double_center)
            for active_center in centers:
                intervals.append(
                    (
                        max(0, active_center - radius),
                        min(n_total - 1, active_center + radius),
                    )
                )
    merged = _merge_intervals(intervals)
    if not merged:
        return [(0, n_total - 1)]
    return merged


def _build_ray_search_state(
    *,
    time_rel: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    design_info: _DesignInfo,
    frequencies: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64] | None,
    max_processes: int | None,
) -> _RaySearchState | None:
    if max_processes is None or max_processes <= 1:
        return None
    use_ray = initialize_use_ray(num_cpus=max_processes)
    if not use_ray:
        return None
    return _RaySearchState(
        time_rel_ref=ray.put(np.asarray(time_rel, dtype=np.float64)),
        y_ref=ray.put(np.asarray(y, dtype=np.float64)),
        design_info_ref=ray.put(design_info),
        frequencies_ref=ray.put(np.asarray(frequencies, dtype=np.float64)),
        weights_ref=None if weights is None else ray.put(np.asarray(weights, dtype=np.float64)),
        max_processes=int(max_processes),
    )


def _resolve_parallel_chunk_size(
    sample_size: int,
    *,
    max_processes: int,
    parallel_chunk_size: int | None,
) -> int:
    if parallel_chunk_size is not None:
        return max(1, min(int(parallel_chunk_size), sample_size))
    target_chunks = max(1, max_processes * 4)
    return max(64, int(ceil(sample_size / target_chunks)))


@ray.remote
def _evaluate_frequency_chunk_ray(
    time_rel: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    design_info: _DesignInfo,
    frequencies: npt.NDArray[np.float64],
    sample_indices: npt.NDArray[np.int64],
    order: int,
    clip_sigma: float,
    weights: npt.NDArray[np.float64] | None,
    exact_evaluation_backend: str = "numpy",
    jax_frequency_batch_size: int = 256,
    jax_row_pad_multiple: int = 64,
    jax_max_clip_iterations: int = 3,
) -> tuple[npt.NDArray[np.float64], _FitResult | None]:
    if exact_evaluation_backend == "jax":
        from .rotation_period_jax import evaluate_frequency_indices_jax

        jax_result = evaluate_frequency_indices_jax(
            time_rel=np.asarray(time_rel, dtype=np.float64),
            y=np.asarray(y, dtype=np.float64),
            fixed=np.asarray(design_info.fixed, dtype=np.float64),
            weights=None if weights is None else np.asarray(weights, dtype=np.float64),
            frequencies=np.asarray(frequencies, dtype=np.float64),
            sample_indices=np.asarray(sample_indices, dtype=np.int64),
            fourier_order=int(order),
            clip_sigma=float(clip_sigma),
            jax_batch_size=int(jax_frequency_batch_size),
            row_pad_multiple=int(jax_row_pad_multiple),
            max_clip_iterations=int(jax_max_clip_iterations),
        )
        if not jax_result.best_valid:
            return jax_result.scores, None
        best_pos = int(np.nanargmin(jax_result.scores))
        best_frequency = float(frequencies[int(sample_indices[best_pos])])
        return jax_result.scores, _FitResult(
            frequency=best_frequency,
            fourier_order=int(order),
            coeffs=np.asarray(jax_result.best_coeffs, dtype=np.float64),
            residual_sigma=float(jax_result.best_sigma),
            rss=float(jax_result.best_rss),
            df=int(jax_result.best_df),
            n_par=int(design_info.fixed.shape[1] + 2 * order),
            mask=np.asarray(jax_result.best_mask, dtype=bool),
            n_fit=int(jax_result.best_n_fit),
            n_clipped=int(jax_result.best_n_clipped),
            n_filters=int(design_info.n_filters),
            phase_c1_idx=int(design_info.phase_c1_idx),
            phase_c2_idx=int(design_info.phase_c2_idx),
        )

    scores = np.full(sample_indices.shape, np.nan, dtype=np.float64)
    best_fit: _FitResult | None = None
    for pos, idx in enumerate(sample_indices.tolist()):
        fit = _fit_frequency(
            time_rel,
            y,
            design_info,
            float(frequencies[int(idx)]),
            int(order),
            clip_sigma=clip_sigma,
            weights=weights,
        )
        if fit is None:
            continue
        scores[pos] = float(fit.residual_sigma)
        if best_fit is None or fit.residual_sigma < best_fit.residual_sigma:
            best_fit = fit
    return scores, best_fit


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
    ray_state: _RaySearchState | None = None,
    parallel_chunk_size: int | None = None,
    exact_evaluation_backend: str = "numpy",
    jax_frequency_batch_size: int = 256,
    jax_row_pad_multiple: int = 64,
    jax_max_clip_iterations: int = 3,
) -> tuple[npt.NDArray[np.float64], _FitResult | None]:
    use_ray_exact = (
        exact
        and ray_state is not None
        and sample_indices.size >= max(128, ray_state.max_processes * 32)
    )

    if exact and exact_evaluation_backend == "jax" and not use_ray_exact:
        from .rotation_period_jax import evaluate_frequency_indices_jax

        jax_result = evaluate_frequency_indices_jax(
            time_rel=np.asarray(t_rel, dtype=np.float64),
            y=np.asarray(y, dtype=np.float64),
            fixed=np.asarray(design_info.fixed, dtype=np.float64),
            weights=None if weights is None else np.asarray(weights, dtype=np.float64),
            frequencies=np.asarray(frequencies, dtype=np.float64),
            sample_indices=np.asarray(sample_indices, dtype=np.int64),
            fourier_order=int(order),
            clip_sigma=float(clip_sigma),
            jax_batch_size=int(jax_frequency_batch_size),
            row_pad_multiple=int(jax_row_pad_multiple),
            max_clip_iterations=int(jax_max_clip_iterations),
        )
        if not jax_result.best_valid:
            return jax_result.scores, None
        best_pos = int(np.nanargmin(jax_result.scores))
        best_frequency = float(frequencies[int(sample_indices[best_pos])])
        return jax_result.scores, _FitResult(
            frequency=best_frequency,
            fourier_order=int(order),
            coeffs=np.asarray(jax_result.best_coeffs, dtype=np.float64),
            residual_sigma=float(jax_result.best_sigma),
            rss=float(jax_result.best_rss),
            df=int(jax_result.best_df),
            n_par=int(design_info.fixed.shape[1] + 2 * order),
            mask=np.asarray(jax_result.best_mask, dtype=bool),
            n_fit=int(jax_result.best_n_fit),
            n_clipped=int(jax_result.best_n_clipped),
            n_filters=int(design_info.n_filters),
            phase_c1_idx=int(design_info.phase_c1_idx),
            phase_c2_idx=int(design_info.phase_c2_idx),
        )

    if use_ray_exact:
        chunk_size = _resolve_parallel_chunk_size(
            int(sample_indices.size),
            max_processes=ray_state.max_processes,
            parallel_chunk_size=parallel_chunk_size,
        )
        chunks = [
            np.asarray(sample_indices[start : start + chunk_size], dtype=np.int64)
            for start in range(0, int(sample_indices.size), chunk_size)
        ]
        futures = [
            _evaluate_frequency_chunk_ray.remote(
                ray_state.time_rel_ref,
                ray_state.y_ref,
                ray_state.design_info_ref,
                ray_state.frequencies_ref,
                chunk,
                int(order),
                float(clip_sigma),
                ray_state.weights_ref,
                exact_evaluation_backend,
                int(jax_frequency_batch_size),
                int(jax_row_pad_multiple),
                int(jax_max_clip_iterations),
            )
            for chunk in chunks
        ]
        score_parts: list[npt.NDArray[np.float64]] = []
        best_fit: _FitResult | None = None
        for chunk_scores, chunk_best in ray.get(futures):
            score_parts.append(np.asarray(chunk_scores, dtype=np.float64))
            if chunk_best is not None and (
                best_fit is None or chunk_best.residual_sigma < best_fit.residual_sigma
            ):
                best_fit = chunk_best
        if score_parts:
            return np.concatenate(score_parts), best_fit
        return np.full(sample_indices.shape, np.nan, dtype=np.float64), None

    scores = np.full(sample_indices.shape, np.nan, dtype=np.float64)
    best_fit: _FitResult | None = None
    for pos, idx in enumerate(sample_indices.tolist()):
        frequency = float(frequencies[int(idx)])
        if exact:
            fit = _fit_frequency(
                t_rel,
                y,
                design_info,
                frequency,
                int(order),
                clip_sigma=clip_sigma,
                weights=weights,
            )
        else:
            fit = _fit_frequency_unclipped(
                t_rel,
                y,
                design_info,
                frequency,
                int(order),
                weights=weights,
            )
        if fit is not None:
            scores[pos] = float(fit.residual_sigma)
            if best_fit is None or fit.residual_sigma < best_fit.residual_sigma:
                best_fit = fit
    return scores, best_fit


def _run_period_search_grid(
    *,
    time_rel: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    phase_angle: npt.NDArray[np.float64],
    filter_labels: npt.NDArray[np.object_],
    session_labels: npt.NDArray[np.object_] | None,
    weights: npt.NDArray[np.float64] | None,
    orders: list[int],
    frequencies: npt.NDArray[np.float64],
    clip_sigma: float,
    order_selection_p_value: float,
    max_processes: int | None,
    parallel_chunk_size: int | None,
    exact_evaluation_backend: str,
    jax_frequency_batch_size: int,
    jax_row_pad_multiple: int,
    jax_max_clip_iterations: int,
) -> _FitResult:
    design_info = _build_fixed_design(filter_labels, session_labels, phase_angle)
    ray_state = _build_ray_search_state(
        time_rel=time_rel,
        y=y,
        design_info=design_info,
        frequencies=frequencies,
        weights=weights,
        max_processes=max_processes,
    )
    candidate_fits: dict[int, _FitResult] = {}
    sample_indices = np.arange(frequencies.size, dtype=np.int64)
    for order in orders:
        scores, best_fit = _evaluate_frequency_indices(
            t_rel=time_rel,
            y=y,
            design_info=design_info,
            frequencies=frequencies,
            sample_indices=sample_indices,
            order=int(order),
            clip_sigma=clip_sigma,
            weights=weights,
            exact=True,
            ray_state=ray_state,
            parallel_chunk_size=parallel_chunk_size,
            exact_evaluation_backend=exact_evaluation_backend,
            jax_frequency_batch_size=jax_frequency_batch_size,
            jax_row_pad_multiple=jax_row_pad_multiple,
            jax_max_clip_iterations=jax_max_clip_iterations,
        )
        if best_fit is not None:
            candidate_fits[int(order)] = best_fit

    if not candidate_fits:
        raise ValueError("no valid rotation-period fit could be found")

    return _select_order(candidate_fits, order_selection_p_value)


def _run_period_search_surrogate_refine(
    *,
    time_rel: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    phase_angle: npt.NDArray[np.float64],
    filter_labels: npt.NDArray[np.object_],
    session_labels: npt.NDArray[np.object_] | None,
    weights: npt.NDArray[np.float64] | None,
    orders: list[int],
    frequencies: npt.NDArray[np.float64],
    clip_sigma: float,
    order_selection_p_value: float,
    max_processes: int | None,
    parallel_chunk_size: int | None,
    exact_evaluation_backend: str,
    jax_frequency_batch_size: int,
    jax_row_pad_multiple: int,
    jax_max_clip_iterations: int,
) -> _FitResult:
    design_info = _build_fixed_design(filter_labels, session_labels, phase_angle)
    ray_state = _build_ray_search_state(
        time_rel=time_rel,
        y=y,
        design_info=design_info,
        frequencies=frequencies,
        weights=weights,
        max_processes=max_processes,
    )
    n_total = int(frequencies.size)
    coarse_stride = max(1, n_total // 256)
    if coarse_stride <= 1:
        return _run_period_search_grid(
            time_rel=time_rel,
            y=y,
            phase_angle=phase_angle,
            filter_labels=filter_labels,
            session_labels=session_labels,
            weights=weights,
            orders=orders,
            frequencies=frequencies,
            clip_sigma=clip_sigma,
            order_selection_p_value=order_selection_p_value,
            max_processes=max_processes,
            parallel_chunk_size=parallel_chunk_size,
            exact_evaluation_backend=exact_evaluation_backend,
            jax_frequency_batch_size=jax_frequency_batch_size,
            jax_row_pad_multiple=jax_row_pad_multiple,
            jax_max_clip_iterations=jax_max_clip_iterations,
        )

    coarse_indices = _sample_indices_from_intervals(
        [(0, n_total - 1)],
        coarse_stride,
        n_total,
    )
    surrogate_scores: dict[int, tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]] = {}
    for order in orders:
        scores, _ = _evaluate_frequency_indices(
            t_rel=time_rel,
            y=y,
            design_info=design_info,
            frequencies=frequencies,
            sample_indices=coarse_indices,
            order=int(order),
            clip_sigma=clip_sigma,
            weights=weights,
            exact=False,
            ray_state=ray_state,
            parallel_chunk_size=parallel_chunk_size,
            exact_evaluation_backend=exact_evaluation_backend,
            jax_frequency_batch_size=jax_frequency_batch_size,
            jax_row_pad_multiple=jax_row_pad_multiple,
            jax_max_clip_iterations=jax_max_clip_iterations,
        )
        surrogate_scores[int(order)] = (coarse_indices, scores)

    intervals = _candidate_intervals_from_scores(
        surrogate_scores,
        candidate_count=5,
        radius=max(1, 4 * coarse_stride),
        n_total=n_total,
    )
    medium_stride = max(1, coarse_stride // 4)
    if medium_stride < coarse_stride:
        medium_indices = _sample_indices_from_intervals(intervals, medium_stride, n_total)
        medium_scores: dict[int, tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]] = {}
        for order in orders:
            scores, _ = _evaluate_frequency_indices(
                t_rel=time_rel,
                y=y,
                design_info=design_info,
                frequencies=frequencies,
                sample_indices=medium_indices,
                order=int(order),
                clip_sigma=clip_sigma,
                weights=weights,
                exact=True,
                ray_state=ray_state,
                parallel_chunk_size=parallel_chunk_size,
                exact_evaluation_backend=exact_evaluation_backend,
                jax_frequency_batch_size=jax_frequency_batch_size,
                jax_row_pad_multiple=jax_row_pad_multiple,
                jax_max_clip_iterations=jax_max_clip_iterations,
            )
            medium_scores[int(order)] = (medium_indices, scores)
        intervals = _candidate_intervals_from_scores(
            medium_scores,
            candidate_count=4,
            radius=max(1, 3 * medium_stride),
            n_total=n_total,
        )

    final_indices = _sample_indices_from_intervals(intervals, 1, n_total)
    candidate_fits: dict[int, _FitResult] = {}
    for order in orders:
        scores, best_fit = _evaluate_frequency_indices(
            t_rel=time_rel,
            y=y,
            design_info=design_info,
            frequencies=frequencies,
            sample_indices=final_indices,
            order=int(order),
            clip_sigma=clip_sigma,
            weights=weights,
            exact=True,
            ray_state=ray_state,
            parallel_chunk_size=parallel_chunk_size,
            exact_evaluation_backend=exact_evaluation_backend,
            jax_frequency_batch_size=jax_frequency_batch_size,
            jax_row_pad_multiple=jax_row_pad_multiple,
            jax_max_clip_iterations=jax_max_clip_iterations,
        )
        if best_fit is not None:
            candidate_fits[int(order)] = best_fit

    if not candidate_fits:
        return _run_period_search_grid(
            time_rel=time_rel,
            y=y,
            phase_angle=phase_angle,
            filter_labels=filter_labels,
            session_labels=session_labels,
            weights=weights,
            orders=orders,
            frequencies=frequencies,
            clip_sigma=clip_sigma,
            order_selection_p_value=order_selection_p_value,
            max_processes=max_processes,
            parallel_chunk_size=parallel_chunk_size,
            exact_evaluation_backend=exact_evaluation_backend,
            jax_frequency_batch_size=jax_frequency_batch_size,
            jax_row_pad_multiple=jax_row_pad_multiple,
            jax_max_clip_iterations=jax_max_clip_iterations,
        )

    return _select_order(candidate_fits, order_selection_p_value)


def _run_period_search_coarse_to_fine(
    *,
    time_rel: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    phase_angle: npt.NDArray[np.float64],
    filter_labels: npt.NDArray[np.object_],
    session_labels: npt.NDArray[np.object_] | None,
    weights: npt.NDArray[np.float64] | None,
    orders: list[int],
    frequencies: npt.NDArray[np.float64],
    clip_sigma: float,
    order_selection_p_value: float,
    max_processes: int | None,
    parallel_chunk_size: int | None,
    exact_evaluation_backend: str,
    jax_frequency_batch_size: int,
    jax_row_pad_multiple: int,
    jax_max_clip_iterations: int,
) -> _FitResult:
    design_info = _build_fixed_design(filter_labels, session_labels, phase_angle)
    ray_state = _build_ray_search_state(
        time_rel=time_rel,
        y=y,
        design_info=design_info,
        frequencies=frequencies,
        weights=weights,
        max_processes=max_processes,
    )
    n_total = int(frequencies.size)
    stride = max(1, n_total // 256)
    if stride <= 1:
        return _run_period_search_grid(
            time_rel=time_rel,
            y=y,
            phase_angle=phase_angle,
            filter_labels=filter_labels,
            session_labels=session_labels,
            weights=weights,
            orders=orders,
            frequencies=frequencies,
            clip_sigma=clip_sigma,
            order_selection_p_value=order_selection_p_value,
            max_processes=max_processes,
            parallel_chunk_size=parallel_chunk_size,
            exact_evaluation_backend=exact_evaluation_backend,
            jax_frequency_batch_size=jax_frequency_batch_size,
            jax_row_pad_multiple=jax_row_pad_multiple,
            jax_max_clip_iterations=jax_max_clip_iterations,
        )

    intervals: list[tuple[int, int]] = [(0, n_total - 1)]
    while True:
        sample_indices = _sample_indices_from_intervals(intervals, stride, n_total)
        stage_scores: dict[int, tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]] = {}
        stage_candidate_fits: dict[int, _FitResult] = {}
        for order in orders:
            scores, best_fit = _evaluate_frequency_indices(
                t_rel=time_rel,
                y=y,
                design_info=design_info,
                frequencies=frequencies,
                sample_indices=sample_indices,
                order=int(order),
                clip_sigma=clip_sigma,
                weights=weights,
                exact=True,
                ray_state=ray_state,
                parallel_chunk_size=parallel_chunk_size,
                exact_evaluation_backend=exact_evaluation_backend,
                jax_frequency_batch_size=jax_frequency_batch_size,
                jax_row_pad_multiple=jax_row_pad_multiple,
                jax_max_clip_iterations=jax_max_clip_iterations,
            )
            stage_scores[int(order)] = (sample_indices, scores)
            if best_fit is not None:
                if np.any(np.isfinite(scores)):
                    best_fit.frequency = float(frequencies[int(sample_indices[int(np.nanargmin(scores))])])
                stage_candidate_fits[int(order)] = best_fit

        if stride == 1:
            if not stage_candidate_fits:
                return _run_period_search_grid(
                    time_rel=time_rel,
                    y=y,
                    phase_angle=phase_angle,
                    filter_labels=filter_labels,
                    session_labels=session_labels,
                    weights=weights,
                    orders=orders,
                    frequencies=frequencies,
                    clip_sigma=clip_sigma,
                    order_selection_p_value=order_selection_p_value,
                    max_processes=max_processes,
                    parallel_chunk_size=parallel_chunk_size,
                    exact_evaluation_backend=exact_evaluation_backend,
                    jax_frequency_batch_size=jax_frequency_batch_size,
                    jax_row_pad_multiple=jax_row_pad_multiple,
                    jax_max_clip_iterations=jax_max_clip_iterations,
                )
            return _select_order(stage_candidate_fits, order_selection_p_value)

        intervals = _candidate_intervals_from_scores(
            stage_scores,
            candidate_count=8,
            radius=max(1, 6 * stride),
            n_total=n_total,
            include_harmonic_partners=True,
        )
        next_stride = max(1, stride // 4)
        stride = 1 if next_stride == stride else next_stride


def _run_period_search(
    *,
    time_rel: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    phase_angle: npt.NDArray[np.float64],
    filter_labels: npt.NDArray[np.object_],
    session_labels: npt.NDArray[np.object_] | None,
    weights: npt.NDArray[np.float64] | None,
    orders: list[int],
    frequencies: npt.NDArray[np.float64],
    clip_sigma: float,
    order_selection_p_value: float,
    search_strategy: str,
    max_processes: int | None,
    parallel_chunk_size: int | None,
    exact_evaluation_backend: str,
    jax_frequency_batch_size: int,
    jax_row_pad_multiple: int,
    jax_max_clip_iterations: int,
) -> _FitResult:
    if search_strategy == "grid":
        return _run_period_search_grid(
            time_rel=time_rel,
            y=y,
            phase_angle=phase_angle,
            filter_labels=filter_labels,
            session_labels=session_labels,
            weights=weights,
            orders=orders,
            frequencies=frequencies,
            clip_sigma=clip_sigma,
            order_selection_p_value=order_selection_p_value,
            max_processes=max_processes,
            parallel_chunk_size=parallel_chunk_size,
            exact_evaluation_backend=exact_evaluation_backend,
            jax_frequency_batch_size=jax_frequency_batch_size,
            jax_row_pad_multiple=jax_row_pad_multiple,
            jax_max_clip_iterations=jax_max_clip_iterations,
        )
    if search_strategy == "surrogate_refine":
        return _run_period_search_surrogate_refine(
            time_rel=time_rel,
            y=y,
            phase_angle=phase_angle,
            filter_labels=filter_labels,
            session_labels=session_labels,
            weights=weights,
            orders=orders,
            frequencies=frequencies,
            clip_sigma=clip_sigma,
            order_selection_p_value=order_selection_p_value,
            max_processes=max_processes,
            parallel_chunk_size=parallel_chunk_size,
            exact_evaluation_backend=exact_evaluation_backend,
            jax_frequency_batch_size=jax_frequency_batch_size,
            jax_row_pad_multiple=jax_row_pad_multiple,
            jax_max_clip_iterations=jax_max_clip_iterations,
        )
    if search_strategy == "coarse_to_fine":
        return _run_period_search_coarse_to_fine(
            time_rel=time_rel,
            y=y,
            phase_angle=phase_angle,
            filter_labels=filter_labels,
            session_labels=session_labels,
            weights=weights,
            orders=orders,
            frequencies=frequencies,
            clip_sigma=clip_sigma,
            order_selection_p_value=order_selection_p_value,
            max_processes=max_processes,
            parallel_chunk_size=parallel_chunk_size,
            exact_evaluation_backend=exact_evaluation_backend,
            jax_frequency_batch_size=jax_frequency_batch_size,
            jax_row_pad_multiple=jax_row_pad_multiple,
            jax_max_clip_iterations=jax_max_clip_iterations,
        )
    raise ValueError("unknown search_strategy")


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


def _fit_with_period_key(fit_with_period: _FitWithPeriod) -> tuple[int, int, int]:
    return (
        int(fit_with_period.fit.fourier_order),
        int(round(float(fit_with_period.fit.frequency) * 1.0e12)),
        int(fit_with_period.is_period_doubled),
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


def _adjudicate_harmonic_aliases(
    *,
    chosen_fit: _FitResult,
    time_rel: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    design_info: _DesignInfo,
    weights: npt.NDArray[np.float64] | None,
    orders: list[int],
    frequencies: npt.NDArray[np.float64],
    clip_sigma: float,
    order_selection_p_value: float,
    harmonic_period_factors: tuple[float, ...],
    harmonic_sigma_tolerance_mag: float,
    harmonic_identity_tolerance: float,
    harmonic_refinement_window: int,
    max_processes: int | None,
    parallel_chunk_size: int | None,
    exact_evaluation_backend: str,
    jax_frequency_batch_size: int,
    jax_row_pad_multiple: int,
    jax_max_clip_iterations: int,
) -> _HarmonicAdjudication:
    if frequencies.size == 0:
        selected = _fit_with_period(chosen_fit)
        return _HarmonicAdjudication(
            selected=selected,
            near_tie_candidates=0,
            had_near_tie=False,
        )

    selected_seed = _fit_with_period(chosen_fit)
    candidates: list[_FitWithPeriod] = [selected_seed]
    period_factors = sorted({1.0, *[float(factor) for factor in harmonic_period_factors if factor > 0.0]})
    frequencies_min = float(frequencies[0])
    frequencies_max = float(frequencies[-1])
    ray_state = _build_ray_search_state(
        time_rel=time_rel,
        y=y,
        design_info=design_info,
        frequencies=frequencies,
        weights=weights,
        max_processes=max_processes,
    )

    for factor in period_factors:
        target_period_days = selected_seed.period_days * float(factor)
        if not np.isfinite(target_period_days) or target_period_days <= 0.0:
            continue
        target_frequency = 1.0 / target_period_days
        if target_frequency < frequencies_min or target_frequency > frequencies_max:
            continue

        idx_center = int(np.searchsorted(frequencies, target_frequency))
        idx_center = min(max(idx_center, 0), int(frequencies.size - 1))
        lo = max(0, idx_center - harmonic_refinement_window)
        hi = min(int(frequencies.size - 1), idx_center + harmonic_refinement_window)
        if hi < lo:
            continue

        sample_indices = np.arange(lo, hi + 1, dtype=np.int64)
        order_fits: dict[int, _FitResult] = {}
        for order in orders:
            _, best_fit = _evaluate_frequency_indices(
                t_rel=time_rel,
                y=y,
                design_info=design_info,
                frequencies=frequencies,
                sample_indices=sample_indices,
                order=int(order),
                clip_sigma=clip_sigma,
                weights=weights,
                exact=True,
                ray_state=ray_state,
                parallel_chunk_size=parallel_chunk_size,
                exact_evaluation_backend=exact_evaluation_backend,
                jax_frequency_batch_size=jax_frequency_batch_size,
                jax_row_pad_multiple=jax_row_pad_multiple,
                jax_max_clip_iterations=jax_max_clip_iterations,
            )
            if best_fit is not None:
                order_fits[int(order)] = best_fit

        if not order_fits:
            continue

        factor_fit = _select_order(order_fits, order_selection_p_value)
        candidates.append(_fit_with_period(factor_fit))

    unique: dict[tuple[int, int, int], _FitWithPeriod] = {}
    for candidate in candidates:
        key = _fit_with_period_key(candidate)
        current = unique.get(key)
        if current is None or candidate.fit.residual_sigma < current.fit.residual_sigma:
            unique[key] = candidate
    candidates = list(unique.values())

    sigma_best = float(min(candidate.fit.residual_sigma for candidate in candidates))
    tolerance = float(max(0.0, harmonic_sigma_tolerance_mag))
    within_tol = [
        candidate
        for candidate in candidates
        if candidate.fit.residual_sigma <= sigma_best + tolerance + 1.0e-12
    ]
    selected = min(
        within_tol,
        key=lambda candidate: (
            int(candidate.fit.fourier_order),
            float(candidate.fit.residual_sigma),
            abs(candidate.period_days - selected_seed.period_days),
        ),
    )
    near_tie_candidates = 0
    for candidate in candidates:
        if (
            int(candidate.fit.fourier_order) == int(selected.fit.fourier_order)
            and abs(float(candidate.fit.frequency) - float(selected.fit.frequency)) <= 1.0e-12
            and bool(candidate.is_period_doubled) == bool(selected.is_period_doubled)
        ):
            continue

        sigma_delta = float(candidate.fit.residual_sigma - selected.fit.residual_sigma)
        if sigma_delta > tolerance + 1.0e-12:
            continue

        mismatch = abs(candidate.period_days - selected.period_days) / selected.period_days
        if mismatch <= harmonic_identity_tolerance:
            continue

        near_tie_candidates += 1

    return _HarmonicAdjudication(
        selected=selected,
        near_tie_candidates=int(near_tie_candidates),
        had_near_tie=bool(near_tie_candidates > 0),
    )


def _count_nonharmonic_near_ties(
    *,
    selected: _FitWithPeriod,
    time_rel: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    design_info: _DesignInfo,
    weights: npt.NDArray[np.float64] | None,
    orders: list[int],
    frequencies: npt.NDArray[np.float64],
    clip_sigma: float,
    harmonic_period_factors: tuple[float, ...],
    harmonic_sigma_tolerance_mag: float,
    harmonic_identity_tolerance: float,
    max_processes: int | None,
    parallel_chunk_size: int | None,
    exact_evaluation_backend: str,
    jax_frequency_batch_size: int,
    jax_row_pad_multiple: int,
    jax_max_clip_iterations: int,
    global_near_tie_max_samples: int,
    global_near_tie_candidate_count: int,
    global_near_tie_refinement_window: int,
) -> int:
    n_total = int(frequencies.size)
    if n_total == 0:
        return 0

    candidate_count = max(1, int(global_near_tie_candidate_count))
    max_samples = max(64, int(global_near_tie_max_samples))
    stride = max(1, int(ceil(n_total / max_samples)))
    sample_indices = np.arange(0, n_total, stride, dtype=np.int64)
    if sample_indices[-1] != n_total - 1:
        sample_indices = np.concatenate(
            (sample_indices, np.asarray([n_total - 1], dtype=np.int64))
        )

    ray_state = _build_ray_search_state(
        time_rel=time_rel,
        y=y,
        design_info=design_info,
        frequencies=frequencies,
        weights=weights,
        max_processes=max_processes,
    )
    scores_by_order: dict[int, tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]] = {}
    for order in orders:
        scores, _ = _evaluate_frequency_indices(
            t_rel=time_rel,
            y=y,
            design_info=design_info,
            frequencies=frequencies,
            sample_indices=sample_indices,
            order=int(order),
            clip_sigma=clip_sigma,
            weights=weights,
            exact=False,
            ray_state=ray_state,
            parallel_chunk_size=parallel_chunk_size,
            exact_evaluation_backend=exact_evaluation_backend,
            jax_frequency_batch_size=jax_frequency_batch_size,
            jax_row_pad_multiple=jax_row_pad_multiple,
            jax_max_clip_iterations=jax_max_clip_iterations,
        )
        scores_by_order[int(order)] = (sample_indices, scores)

    if not scores_by_order:
        return 0

    intervals = _candidate_intervals_from_scores(
        scores_by_order,
        candidate_count=max(candidate_count, 4),
        radius=max(1, int(global_near_tie_refinement_window)),
        n_total=n_total,
        include_harmonic_partners=True,
    )
    exact_indices = _sample_indices_from_intervals(intervals, stride=1, n_total=n_total)
    if exact_indices.size == 0:
        return 0

    sigma_tolerance = float(max(0.0, harmonic_sigma_tolerance_mag))
    sigma_threshold = float(selected.fit.residual_sigma + sigma_tolerance + 1.0e-12)
    selected_key = _fit_with_period_key(selected)
    tie_keys: set[tuple[int, int, int]] = set()
    max_candidates_per_order = max(candidate_count * 4, 12)

    for order in orders:
        scores, _ = _evaluate_frequency_indices(
            t_rel=time_rel,
            y=y,
            design_info=design_info,
            frequencies=frequencies,
            sample_indices=exact_indices,
            order=int(order),
            clip_sigma=clip_sigma,
            weights=weights,
            exact=True,
            ray_state=ray_state,
            parallel_chunk_size=parallel_chunk_size,
            exact_evaluation_backend=exact_evaluation_backend,
            jax_frequency_batch_size=jax_frequency_batch_size,
            jax_row_pad_multiple=jax_row_pad_multiple,
            jax_max_clip_iterations=jax_max_clip_iterations,
        )
        valid_positions = np.flatnonzero(np.isfinite(scores) & (scores <= sigma_threshold))
        if valid_positions.size == 0:
            continue
        sorted_positions = valid_positions[np.argsort(scores[valid_positions])]
        for pos in sorted_positions[:max_candidates_per_order].tolist():
            idx = int(exact_indices[int(pos)])
            frequency = float(frequencies[idx])
            fit = _fit_frequency(
                time_rel,
                y,
                design_info,
                frequency,
                int(order),
                clip_sigma=clip_sigma,
                weights=weights,
            )
            if fit is None:
                continue
            candidate = _fit_with_period(fit)
            key = _fit_with_period_key(candidate)
            if key == selected_key or key in tie_keys:
                continue
            sigma_delta = float(candidate.fit.residual_sigma - selected.fit.residual_sigma)
            if sigma_delta > sigma_tolerance + 1.0e-12:
                continue
            relative_mismatch = abs(candidate.period_days - selected.period_days) / selected.period_days
            if relative_mismatch <= harmonic_identity_tolerance:
                continue
            tie_keys.add(key)
            if len(tie_keys) >= candidate_count:
                return int(len(tie_keys))

    return int(len(tie_keys))


def _count_window_alias_near_ties(
    *,
    selected: _FitWithPeriod,
    time_rel: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    phase_angle: npt.NDArray[np.float64],
    filter_labels: npt.NDArray[np.object_],
    session_labels: npt.NDArray[np.object_] | None,
    weights: npt.NDArray[np.float64] | None,
    orders: list[int],
    frequencies: npt.NDArray[np.float64],
    clip_sigma: float,
    order_selection_p_value: float,
    harmonic_period_factors: tuple[float, ...],
    harmonic_sigma_tolerance_mag: float,
    harmonic_identity_tolerance: float,
    max_processes: int | None,
    parallel_chunk_size: int | None,
    exact_evaluation_backend: str,
    jax_frequency_batch_size: int,
    jax_row_pad_multiple: int,
    jax_max_clip_iterations: int,
    window_alias_frequency_offsets: tuple[float, ...],
    window_alias_refinement_window: int,
) -> int:
    if frequencies.size == 0:
        return 0

    offsets = sorted(
        {
            float(abs(offset))
            for offset in window_alias_frequency_offsets
            if np.isfinite(offset) and float(offset) > 0.0
        }
    )
    if not offsets:
        return 0

    f_min = float(frequencies[0])
    f_max = float(frequencies[-1])
    selected_frequency = float(selected.fit.frequency)
    target_periods: list[float] = []
    for offset in offsets:
        for sign in (-1.0, 1.0):
            alias_frequency = abs(selected_frequency + sign * offset)
            if alias_frequency <= 0.0 or alias_frequency < f_min or alias_frequency > f_max:
                continue
            target_periods.append(float(1.0 / alias_frequency))
    if not target_periods:
        return 0

    sigma_tolerance = float(max(0.0, harmonic_sigma_tolerance_mag))
    selected_key = _fit_with_period_key(selected)
    seen_keys: set[tuple[int, int, int]] = set()
    near_ties = 0
    for target_period in target_periods:
        alias_fit = _run_period_search_targeted_grid(
            time_rel=time_rel,
            y=y,
            phase_angle=phase_angle,
            filter_labels=filter_labels,
            session_labels=session_labels,
            weights=weights,
            orders=orders,
            frequencies=frequencies,
            target_period_days=[float(target_period)],
            target_window_indices=max(1, int(window_alias_refinement_window)),
            clip_sigma=clip_sigma,
            order_selection_p_value=order_selection_p_value,
            max_processes=max_processes,
            parallel_chunk_size=parallel_chunk_size,
            exact_evaluation_backend=exact_evaluation_backend,
            jax_frequency_batch_size=jax_frequency_batch_size,
            jax_row_pad_multiple=jax_row_pad_multiple,
            jax_max_clip_iterations=jax_max_clip_iterations,
        )
        if alias_fit is None:
            continue
        alias_candidate = _fit_with_period(alias_fit)
        alias_key = _fit_with_period_key(alias_candidate)
        if alias_key == selected_key or alias_key in seen_keys:
            continue
        seen_keys.add(alias_key)
        sigma_delta = float(alias_candidate.fit.residual_sigma - selected.fit.residual_sigma)
        if sigma_delta > sigma_tolerance + 1.0e-12:
            continue
        relative_mismatch = abs(alias_candidate.period_days - selected.period_days) / selected.period_days
        if relative_mismatch <= harmonic_identity_tolerance:
            continue
        near_ties += 1
    return int(near_ties)


def _estimate_lsm_frequency(
    *,
    time_rel: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    filter_labels: npt.NDArray[np.object_],
    weights: npt.NDArray[np.float64] | None,
    f_min: float,
    f_max: float,
    lsm_max_frequency_samples: int,
    lsm_refine_samples: int,
    lsm_refine_rounds: int,
) -> float | None:
    if time_rel.size < 5:
        return None
    if lsm_max_frequency_samples < 8 or lsm_refine_samples < 8:
        return None
    if not np.isfinite(f_min) or not np.isfinite(f_max) or f_min <= 0.0 or f_max <= f_min:
        return None

    try:
        from astropy.timeseries import LombScargle, LombScargleMultiband
    except Exception:
        return None

    dy = None
    if weights is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            dy_candidate = 1.0 / np.sqrt(np.asarray(weights, dtype=np.float64))
        if np.all(np.isfinite(dy_candidate)) and np.all(dy_candidate > 0.0):
            dy = dy_candidate

    unique_filters = _ordered_unique(np.asarray(filter_labels, dtype=object))
    try:
        if len(unique_filters) > 1:
            bands = np.asarray(
                [str(value) for value in np.asarray(filter_labels, dtype=object).tolist()],
                dtype=object,
            )
            model = LombScargleMultiband(
                np.asarray(time_rel, dtype=np.float64),
                np.asarray(y, dtype=np.float64),
                bands,
                dy=dy,
            )
        else:
            model = LombScargle(
                np.asarray(time_rel, dtype=np.float64),
                np.asarray(y, dtype=np.float64),
                dy=dy,
            )
    except Exception:
        return None

    coarse_n = int(max(256, lsm_max_frequency_samples))
    coarse_frequencies = np.linspace(f_min, f_max, coarse_n, dtype=np.float64)
    try:
        power = np.asarray(model.power(coarse_frequencies), dtype=np.float64)
    except Exception:
        return None
    finite = np.isfinite(power)
    if not np.any(finite):
        return None
    best_idx = int(np.argmax(np.where(finite, power, -np.inf)))
    best_frequency = float(coarse_frequencies[best_idx])
    coarse_step = float((f_max - f_min) / max(coarse_n - 1, 1))

    current_half_width = max(coarse_step * 4.0, coarse_step)
    for _ in range(int(max(0, lsm_refine_rounds))):
        lo = max(f_min, best_frequency - current_half_width)
        hi = min(f_max, best_frequency + current_half_width)
        if hi <= lo:
            break
        refine_frequencies = np.linspace(lo, hi, int(max(16, lsm_refine_samples)), dtype=np.float64)
        try:
            refine_power = np.asarray(model.power(refine_frequencies), dtype=np.float64)
        except Exception:
            break
        finite_refine = np.isfinite(refine_power)
        if not np.any(finite_refine):
            break
        refine_idx = int(np.argmax(np.where(finite_refine, refine_power, -np.inf)))
        best_frequency = float(refine_frequencies[refine_idx])
        current_half_width = max((hi - lo) / 8.0, coarse_step / 4.0)

    return float(best_frequency)


def _run_period_search_targeted_grid(
    *,
    time_rel: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    phase_angle: npt.NDArray[np.float64],
    filter_labels: npt.NDArray[np.object_],
    session_labels: npt.NDArray[np.object_] | None,
    weights: npt.NDArray[np.float64] | None,
    orders: list[int],
    frequencies: npt.NDArray[np.float64],
    target_period_days: list[float],
    target_window_indices: int,
    clip_sigma: float,
    order_selection_p_value: float,
    max_processes: int | None,
    parallel_chunk_size: int | None,
    exact_evaluation_backend: str,
    jax_frequency_batch_size: int,
    jax_row_pad_multiple: int,
    jax_max_clip_iterations: int,
) -> _FitResult | None:
    if frequencies.size == 0 or target_window_indices <= 0:
        return None

    design_info = _build_fixed_design(filter_labels, session_labels, phase_angle)
    ray_state = _build_ray_search_state(
        time_rel=time_rel,
        y=y,
        design_info=design_info,
        frequencies=frequencies,
        weights=weights,
        max_processes=max_processes,
    )

    n_total = int(frequencies.size)
    f_min = float(frequencies[0])
    f_max = float(frequencies[-1])
    index_set: set[int] = set()
    for period_days in target_period_days:
        if not np.isfinite(period_days) or period_days <= 0.0:
            continue
        frequency = 1.0 / float(period_days)
        if frequency < f_min or frequency > f_max:
            continue
        idx_center = int(np.searchsorted(frequencies, frequency))
        idx_center = min(max(idx_center, 0), n_total - 1)
        lo = max(0, idx_center - target_window_indices)
        hi = min(n_total - 1, idx_center + target_window_indices)
        for idx in range(lo, hi + 1):
            index_set.add(idx)

    if not index_set:
        return None

    sample_indices = np.asarray(sorted(index_set), dtype=np.int64)
    candidate_fits: dict[int, _FitResult] = {}
    for order in orders:
        _, best_fit = _evaluate_frequency_indices(
            t_rel=time_rel,
            y=y,
            design_info=design_info,
            frequencies=frequencies,
            sample_indices=sample_indices,
            order=int(order),
            clip_sigma=clip_sigma,
            weights=weights,
            exact=True,
            ray_state=ray_state,
            parallel_chunk_size=parallel_chunk_size,
            exact_evaluation_backend=exact_evaluation_backend,
            jax_frequency_batch_size=jax_frequency_batch_size,
            jax_row_pad_multiple=jax_row_pad_multiple,
            jax_max_clip_iterations=jax_max_clip_iterations,
        )
        if best_fit is not None:
            candidate_fits[int(order)] = best_fit

    if not candidate_fits:
        return None
    return _select_order(candidate_fits, order_selection_p_value)


def estimate_rotation_period(
    observations: RotationPeriodObservations,
    *,
    fourier_orders: tuple[int, ...] = (2, 3, 4),
    clip_sigma: float = 3.0,
    order_selection_p_value: float = 0.05,
    min_rotations_in_span: float = 2.0,
    max_frequency_cycles_per_day: float = 1000.0,
    frequency_grid_scale: float = 30.0,
    search_strategy: str = "surrogate_refine",
    exact_evaluation_backend: str = "numpy",
    max_processes: int | None = None,
    parallel_chunk_size: int | None = None,
    jax_frequency_batch_size: int = 256,
    jax_row_pad_multiple: int = 64,
    jax_max_clip_iterations: int = 3,
    session_mode: str = "auto",
    auto_session_min_observations_per_group: int = 6,
    auto_session_max_period_to_session_span_ratio: float = 1.0,
    auto_session_bic_improvement: float = 10.0,
    enable_harmonic_adjudication: bool = True,
    harmonic_period_factors: tuple[float, ...] = (
        0.5,
        2.0 / 3.0,
        0.75,
        1.0,
        4.0 / 3.0,
        1.5,
        2.0,
        3.0,
        4.0,
    ),
    harmonic_sigma_tolerance_mag: float = 0.05,
    harmonic_identity_tolerance: float = 0.02,
    harmonic_refinement_window: int = 8,
    harmonic_grid_fallback_on_near_tie: bool = True,
    grid_fallback_refinement_window: int = 64,
    enable_global_near_tie_check: bool = True,
    global_near_tie_max_samples: int = 4096,
    global_near_tie_candidate_count: int = 8,
    global_near_tie_refinement_window: int = 8,
    enable_window_alias_check: bool = True,
    window_alias_frequency_offsets: tuple[float, ...] = (1.0, 2.0),
    window_alias_refinement_window: int = 16,
    enable_lsm_crosscheck: bool = True,
    lsm_harmonic_tolerance: float = 0.05,
    lsm_max_frequency_samples: int = 4096,
    lsm_refine_samples: int = 512,
    lsm_refine_rounds: int = 2,
) -> RotationPeriodResult:
    """
    Estimate a rotation period from reduced photometric observations using a
    high-order Fourier search.

    Default compute path rationale:
    - `search_strategy="surrogate_refine"` is the expected efficient default for
      wide searches because it keeps paper-like coverage while avoiding exhaustive
      evaluation of every sampled frequency.
    - `fourier_orders=(2,3,4)` is the expected robust default for this pipeline:
      it reduces high-order overfitting and alias preference while preserving
      recovery accuracy on synthetic and real fixtures.
    - `exact_evaluation_backend="numpy"` is the expected efficient default for
      one-off/library calls because JAX has nontrivial first-call compile overhead.
      Use `exact_evaluation_backend="jax"` in warm, repeated workloads.
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
    if search_strategy not in {"grid", "surrogate_refine", "coarse_to_fine"}:
        raise ValueError(
            "search_strategy must be one of {'grid', 'surrogate_refine', 'coarse_to_fine'}"
        )
    if exact_evaluation_backend not in {"numpy", "jax"}:
        raise ValueError("exact_evaluation_backend must be one of {'numpy', 'jax'}")
    if max_processes is not None and max_processes <= 0:
        raise ValueError("max_processes must be positive when provided")
    if parallel_chunk_size is not None and parallel_chunk_size <= 0:
        raise ValueError("parallel_chunk_size must be positive when provided")
    if jax_frequency_batch_size <= 0:
        raise ValueError("jax_frequency_batch_size must be positive")
    if jax_row_pad_multiple <= 0:
        raise ValueError("jax_row_pad_multiple must be positive")
    if jax_max_clip_iterations <= 0:
        raise ValueError("jax_max_clip_iterations must be positive")
    if session_mode not in {"ignore", "use", "auto"}:
        raise ValueError("session_mode must be one of {'ignore', 'use', 'auto'}")
    if auto_session_min_observations_per_group <= 0:
        raise ValueError("auto_session_min_observations_per_group must be positive")
    if auto_session_max_period_to_session_span_ratio <= 0.0:
        raise ValueError("auto_session_max_period_to_session_span_ratio must be positive")
    if auto_session_bic_improvement < 0.0:
        raise ValueError("auto_session_bic_improvement must be non-negative")
    if harmonic_sigma_tolerance_mag < 0.0:
        raise ValueError("harmonic_sigma_tolerance_mag must be non-negative")
    if harmonic_identity_tolerance < 0.0:
        raise ValueError("harmonic_identity_tolerance must be non-negative")
    if harmonic_refinement_window < 0:
        raise ValueError("harmonic_refinement_window must be non-negative")
    if grid_fallback_refinement_window <= 0:
        raise ValueError("grid_fallback_refinement_window must be positive")
    if global_near_tie_max_samples <= 0:
        raise ValueError("global_near_tie_max_samples must be positive")
    if global_near_tie_candidate_count <= 0:
        raise ValueError("global_near_tie_candidate_count must be positive")
    if global_near_tie_refinement_window < 0:
        raise ValueError("global_near_tie_refinement_window must be non-negative")
    if window_alias_refinement_window <= 0:
        raise ValueError("window_alias_refinement_window must be positive")
    if lsm_harmonic_tolerance < 0.0:
        raise ValueError("lsm_harmonic_tolerance must be non-negative")
    if lsm_max_frequency_samples <= 0:
        raise ValueError("lsm_max_frequency_samples must be positive")
    if lsm_refine_samples <= 0:
        raise ValueError("lsm_refine_samples must be positive")
    if lsm_refine_rounds < 0:
        raise ValueError("lsm_refine_rounds must be non-negative")

    orders = sorted({int(order) for order in fourier_orders})
    if not orders:
        raise ValueError("fourier_orders must be non-empty")
    if any(order < 2 for order in orders):
        raise ValueError("fourier_orders must be >= 2")
    if len(harmonic_period_factors) == 0:
        raise ValueError("harmonic_period_factors must be non-empty")
    if any(float(factor) <= 0.0 for factor in harmonic_period_factors):
        raise ValueError("harmonic_period_factors must all be positive")
    if len(window_alias_frequency_offsets) == 0:
        raise ValueError("window_alias_frequency_offsets must be non-empty")
    if any(float(offset) <= 0.0 for offset in window_alias_frequency_offsets):
        raise ValueError("window_alias_frequency_offsets must all be positive")

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
    session_summary = _summarize_sessions(time, filter_labels, session_labels)

    baseline = _run_period_search(
        time_rel=time_rel,
        y=y,
        phase_angle=phase_angle,
        filter_labels=filter_labels,
        session_labels=None,
        weights=weights,
        orders=orders,
        frequencies=frequencies,
        clip_sigma=clip_sigma,
        order_selection_p_value=order_selection_p_value,
        search_strategy=search_strategy,
        max_processes=max_processes,
        parallel_chunk_size=parallel_chunk_size,
        exact_evaluation_backend=exact_evaluation_backend,
        jax_frequency_batch_size=jax_frequency_batch_size,
        jax_row_pad_multiple=jax_row_pad_multiple,
        jax_max_clip_iterations=jax_max_clip_iterations,
    )
    chosen = baseline
    used_session_offsets = False

    if session_mode == "use":
        if session_labels is not None:
            chosen = _run_period_search(
                time_rel=time_rel,
                y=y,
                phase_angle=phase_angle,
                filter_labels=filter_labels,
                session_labels=session_labels,
                weights=weights,
                orders=orders,
                frequencies=frequencies,
                clip_sigma=clip_sigma,
                order_selection_p_value=order_selection_p_value,
                search_strategy=search_strategy,
                max_processes=max_processes,
                parallel_chunk_size=parallel_chunk_size,
                exact_evaluation_backend=exact_evaluation_backend,
                jax_frequency_batch_size=jax_frequency_batch_size,
                jax_row_pad_multiple=jax_row_pad_multiple,
                jax_max_clip_iterations=jax_max_clip_iterations,
            )
            used_session_offsets = True
    elif session_mode == "auto" and session_labels is not None and session_summary.n_sessions >= 2:
        baseline_period_days = 1.0 / baseline.frequency
        session_eligible = (
            session_summary.min_group_count >= auto_session_min_observations_per_group
            and session_summary.median_session_span_days > 0.0
            and baseline_period_days
            <= (
                auto_session_max_period_to_session_span_ratio
                * session_summary.median_session_span_days
            )
        )
        if session_eligible:
            session_fit = _run_period_search(
                time_rel=time_rel,
                y=y,
                phase_angle=phase_angle,
                filter_labels=filter_labels,
                session_labels=session_labels,
                weights=weights,
                orders=orders,
                frequencies=frequencies,
                clip_sigma=clip_sigma,
                order_selection_p_value=order_selection_p_value,
                search_strategy=search_strategy,
                max_processes=max_processes,
                parallel_chunk_size=parallel_chunk_size,
                exact_evaluation_backend=exact_evaluation_backend,
                jax_frequency_batch_size=jax_frequency_batch_size,
                jax_row_pad_multiple=jax_row_pad_multiple,
                jax_max_clip_iterations=jax_max_clip_iterations,
            )
            if _fit_bic(session_fit) + auto_session_bic_improvement < _fit_bic(baseline):
                chosen = session_fit
                used_session_offsets = True

    active_session_labels = session_labels if used_session_offsets else None
    design_info_active = _build_fixed_design(filter_labels, active_session_labels, phase_angle)
    adjudication = _HarmonicAdjudication(
        selected=_fit_with_period(chosen),
        near_tie_candidates=0,
        had_near_tie=False,
    )
    if enable_harmonic_adjudication:
        adjudication = _adjudicate_harmonic_aliases(
            chosen_fit=chosen,
            time_rel=time_rel,
            y=y,
            design_info=design_info_active,
            weights=weights,
            orders=orders,
            frequencies=frequencies,
            clip_sigma=clip_sigma,
            order_selection_p_value=order_selection_p_value,
            harmonic_period_factors=harmonic_period_factors,
            harmonic_sigma_tolerance_mag=harmonic_sigma_tolerance_mag,
            harmonic_identity_tolerance=harmonic_identity_tolerance,
            harmonic_refinement_window=harmonic_refinement_window,
            max_processes=max_processes,
            parallel_chunk_size=parallel_chunk_size,
            exact_evaluation_backend=exact_evaluation_backend,
            jax_frequency_batch_size=jax_frequency_batch_size,
            jax_row_pad_multiple=jax_row_pad_multiple,
            jax_max_clip_iterations=jax_max_clip_iterations,
        )

    lsm_frequency: float | None = None
    if enable_lsm_crosscheck:
        lsm_frequency = _estimate_lsm_frequency(
            time_rel=time_rel,
            y=y,
            filter_labels=filter_labels,
            weights=weights,
            f_min=f_min,
            f_max=f_max,
            lsm_max_frequency_samples=lsm_max_frequency_samples,
            lsm_refine_samples=lsm_refine_samples,
            lsm_refine_rounds=lsm_refine_rounds,
        )

    def _compute_lsm_summary(
        current_period_days: float,
    ) -> tuple[float | None, float | None, float | None, bool | None]:
        if lsm_frequency is None or not np.isfinite(lsm_frequency) or lsm_frequency <= 0.0:
            return None, None, None, None
        lsm_period_days_local = float(1.0 / lsm_frequency)
        lsm_period_hours_local = float(lsm_period_days_local * 24.0)
        lsm_mismatch = _harmonic_relative_mismatch(
            current_period_days,
            lsm_period_days_local,
            harmonic_period_factors=harmonic_period_factors,
        )
        lsm_agreement_local = bool(lsm_mismatch <= lsm_harmonic_tolerance)
        return (
            lsm_period_days_local,
            lsm_period_hours_local,
            float(lsm_frequency),
            lsm_agreement_local,
        )

    (
        lsm_period_days,
        lsm_period_hours,
        lsm_frequency_cycles_per_day,
        lsm_harmonic_agreement,
    ) = _compute_lsm_summary(adjudication.selected.period_days)

    used_grid_fallback = False
    if (
        harmonic_grid_fallback_on_near_tie
        and search_strategy != "grid"
        and (
            adjudication.had_near_tie
            or (enable_lsm_crosscheck and lsm_harmonic_agreement is False)
        )
    ):
        target_periods = [
            float(adjudication.selected.period_days * factor)
            for factor in sorted({1.0, *[float(value) for value in harmonic_period_factors]})
            if float(adjudication.selected.period_days * factor) > 0.0
        ]
        grid_fit = _run_period_search_targeted_grid(
            time_rel=time_rel,
            y=y,
            phase_angle=phase_angle,
            filter_labels=filter_labels,
            session_labels=active_session_labels,
            weights=weights,
            orders=orders,
            frequencies=frequencies,
            target_period_days=target_periods,
            target_window_indices=grid_fallback_refinement_window,
            clip_sigma=clip_sigma,
            order_selection_p_value=order_selection_p_value,
            max_processes=max_processes,
            parallel_chunk_size=parallel_chunk_size,
            exact_evaluation_backend=exact_evaluation_backend,
            jax_frequency_batch_size=jax_frequency_batch_size,
            jax_row_pad_multiple=jax_row_pad_multiple,
            jax_max_clip_iterations=jax_max_clip_iterations,
        )
        if grid_fit is None:
            grid_fit = _run_period_search_grid(
                time_rel=time_rel,
                y=y,
                phase_angle=phase_angle,
                filter_labels=filter_labels,
                session_labels=active_session_labels,
                weights=weights,
                orders=orders,
                frequencies=frequencies,
                clip_sigma=clip_sigma,
                order_selection_p_value=order_selection_p_value,
                max_processes=max_processes,
                parallel_chunk_size=parallel_chunk_size,
                exact_evaluation_backend=exact_evaluation_backend,
                jax_frequency_batch_size=jax_frequency_batch_size,
                jax_row_pad_multiple=jax_row_pad_multiple,
                jax_max_clip_iterations=jax_max_clip_iterations,
            )
        used_grid_fallback = True
        if enable_harmonic_adjudication:
            adjudication = _adjudicate_harmonic_aliases(
                chosen_fit=grid_fit,
                time_rel=time_rel,
                y=y,
                design_info=design_info_active,
                weights=weights,
                orders=orders,
                frequencies=frequencies,
                clip_sigma=clip_sigma,
                order_selection_p_value=order_selection_p_value,
                harmonic_period_factors=harmonic_period_factors,
                harmonic_sigma_tolerance_mag=harmonic_sigma_tolerance_mag,
                harmonic_identity_tolerance=harmonic_identity_tolerance,
                harmonic_refinement_window=harmonic_refinement_window,
                max_processes=max_processes,
                parallel_chunk_size=parallel_chunk_size,
                exact_evaluation_backend=exact_evaluation_backend,
                jax_frequency_batch_size=jax_frequency_batch_size,
                jax_row_pad_multiple=jax_row_pad_multiple,
                jax_max_clip_iterations=jax_max_clip_iterations,
            )
        else:
            adjudication = _HarmonicAdjudication(
                selected=_fit_with_period(grid_fit),
                near_tie_candidates=0,
                had_near_tie=False,
            )
        (
            lsm_period_days,
            lsm_period_hours,
            lsm_frequency_cycles_per_day,
            lsm_harmonic_agreement,
        ) = _compute_lsm_summary(adjudication.selected.period_days)

    chosen = adjudication.selected.fit
    is_period_doubled = bool(adjudication.selected.is_period_doubled)
    period_days = float(adjudication.selected.period_days)
    period_hours = float(adjudication.selected.period_hours)

    nonharmonic_near_ties = 0
    window_alias_near_ties = 0
    prelim_ambiguous = adjudication.had_near_tie or (
        enable_lsm_crosscheck and lsm_harmonic_agreement is False
    )
    if not prelim_ambiguous:
        if enable_global_near_tie_check:
            nonharmonic_near_ties = _count_nonharmonic_near_ties(
                selected=adjudication.selected,
                time_rel=time_rel,
                y=y,
                design_info=design_info_active,
                weights=weights,
                orders=orders,
                frequencies=frequencies,
                clip_sigma=clip_sigma,
                harmonic_period_factors=harmonic_period_factors,
                harmonic_sigma_tolerance_mag=harmonic_sigma_tolerance_mag,
                harmonic_identity_tolerance=harmonic_identity_tolerance,
                max_processes=max_processes,
                parallel_chunk_size=parallel_chunk_size,
                exact_evaluation_backend=exact_evaluation_backend,
                jax_frequency_batch_size=jax_frequency_batch_size,
                jax_row_pad_multiple=jax_row_pad_multiple,
                jax_max_clip_iterations=jax_max_clip_iterations,
                global_near_tie_max_samples=global_near_tie_max_samples,
                global_near_tie_candidate_count=global_near_tie_candidate_count,
                global_near_tie_refinement_window=global_near_tie_refinement_window,
            )
        if enable_window_alias_check:
            window_alias_near_ties = _count_window_alias_near_ties(
                selected=adjudication.selected,
                time_rel=time_rel,
                y=y,
                phase_angle=phase_angle,
                filter_labels=filter_labels,
                session_labels=active_session_labels,
                weights=weights,
                orders=orders,
                frequencies=frequencies,
                clip_sigma=clip_sigma,
                order_selection_p_value=order_selection_p_value,
                harmonic_period_factors=harmonic_period_factors,
                harmonic_sigma_tolerance_mag=harmonic_sigma_tolerance_mag,
                harmonic_identity_tolerance=harmonic_identity_tolerance,
                max_processes=max_processes,
                parallel_chunk_size=parallel_chunk_size,
                exact_evaluation_backend=exact_evaluation_backend,
                jax_frequency_batch_size=jax_frequency_batch_size,
                jax_row_pad_multiple=jax_row_pad_multiple,
                jax_max_clip_iterations=jax_max_clip_iterations,
                window_alias_frequency_offsets=window_alias_frequency_offsets,
                window_alias_refinement_window=window_alias_refinement_window,
            )

    ambiguity_reasons: list[str] = []
    if adjudication.had_near_tie:
        ambiguity_reasons.append("harmonic_near_tie")
    if enable_lsm_crosscheck and lsm_harmonic_agreement is False:
        ambiguity_reasons.append("lsm_disagreement")
    if nonharmonic_near_ties > 0:
        ambiguity_reasons.append("global_near_tie")
    if window_alias_near_ties > 0:
        ambiguity_reasons.append("window_alias_near_tie")
    is_ambiguous = len(ambiguity_reasons) > 0
    if not is_ambiguous:
        confidence_label = "high"
    elif adjudication.had_near_tie:
        # Distinguish harmonic alias uncertainty from other ambiguity sources.
        confidence_label = "harmonic_ambiguous"
    else:
        confidence_label = "ambiguous"
    ambiguity_reason = ",".join(ambiguity_reasons) if ambiguity_reasons else None

    n_obs = int(len(observations))
    n_fit = int(chosen.n_fit)
    n_clipped = int(chosen.n_clipped)
    n_filters = int(chosen.n_filters)
    n_sessions = int(session_summary.n_sessions)

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
        n_sessions=[n_sessions],
        used_session_offsets=[bool(used_session_offsets)],
        is_period_doubled=[bool(is_period_doubled)],
        is_ambiguous=[bool(is_ambiguous)],
        confidence_label=[confidence_label],
        ambiguity_reason=[ambiguity_reason],
        n_harmonic_near_ties=[int(adjudication.near_tie_candidates)],
        used_grid_fallback=[bool(used_grid_fallback)],
        harmonic_sigma_tolerance_mag=[float(harmonic_sigma_tolerance_mag)],
        lsm_period_days=[lsm_period_days],
        lsm_period_hours=[lsm_period_hours],
        lsm_frequency_cycles_per_day=[lsm_frequency_cycles_per_day],
        lsm_harmonic_agreement=[lsm_harmonic_agreement],
    )
