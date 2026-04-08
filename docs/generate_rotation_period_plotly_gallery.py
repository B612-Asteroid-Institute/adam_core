from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pyarrow as pa
from plotly.subplots import make_subplots

from adam_core.observers.utils import calculate_observing_night
from adam_core.photometry.rotation_period_fourier import (
    _build_fixed_design,
    _build_fourier_columns,
    _fit_frequency,
    _validate_inputs,
    estimate_rotation_period,
)
from adam_core.photometry.rotation_period_types import RotationPeriodObservations
from adam_core.time import Timestamp


FIXTURE_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "adam_core"
    / "photometry"
    / "tests"
    / "data"
)
OUTPUT_PATH = Path(__file__).resolve().parent / "rotation_period_plotly_gallery.html"
SEARCH_ORDERS = (2, 3, 4, 5, 6)
SEARCH_SAMPLE_LIMIT = 180
PHASE_MODEL_POINTS = 600
GALLERY_OBJECT_NUMBERS = {289, 511, 702, 1323}
SESSION_COLORS = [
    "#ef476f",
    "#118ab2",
    "#ffd166",
    "#06d6a0",
    "#8d99ae",
    "#f78c6b",
    "#7b2cbf",
    "#ff9f1c",
    "#5e60ce",
    "#2a9d8f",
]


def _fixture_to_observations(fx: np.lib.npyio.NpzFile) -> RotationPeriodObservations:
    mag_sigma = np.asarray(fx["mag_sigma"], dtype=np.float64)
    return RotationPeriodObservations.from_kwargs(
        time=Timestamp.from_iso8601(fx["time_iso"].astype(object).tolist(), scale="utc"),
        mag=np.asarray(fx["mag_obs"], dtype=np.float64),
        mag_sigma=pa.array(
            mag_sigma,
            mask=~np.isfinite(mag_sigma),
            type=pa.float64(),
        ),
        filter=fx["filter"].astype(object).tolist(),
        session_id=fx["session_id"].astype(object).tolist(),
        r_au=np.asarray(fx["r_au"], dtype=np.float64),
        delta_au=np.asarray(fx["delta_au"], dtype=np.float64),
        phase_angle_deg=np.asarray(fx["phase_angle_deg"], dtype=np.float64),
    )


def _periodic_component_for_phase(
    fourier_coeffs: np.ndarray,
    fourier_order: int,
    phase_fit: np.ndarray,
) -> np.ndarray:
    periodic = np.zeros_like(phase_fit, dtype=np.float64)
    for harmonic in range(1, fourier_order + 1):
        idx = 2 * (harmonic - 1)
        periodic += fourier_coeffs[idx] * np.cos(2.0 * np.pi * harmonic * phase_fit)
        periodic += fourier_coeffs[idx + 1] * np.sin(
            2.0 * np.pi * harmonic * phase_fit
        )
    return periodic


def _ordered_unique(values: np.ndarray | list[object]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in np.asarray(values, dtype=object).tolist():
        label = str(value)
        if label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered


def _order_session_ids(session_labels: np.ndarray | None) -> list[str]:
    if session_labels is None:
        return ["all"]
    ordered: list[str] = []
    seen: set[str] = set()
    for value in session_labels.tolist():
        label = "all" if value is None else str(value)
        if label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered


def _session_color_map(session_ids: list[str]) -> dict[str, str]:
    return {
        session_id: SESSION_COLORS[idx % len(SESSION_COLORS)]
        for idx, session_id in enumerate(session_ids)
    }


def build_observatory_night_session_ids(
    observatory_codes: pa.Array | pa.ChunkedArray | list[str],
    times: Timestamp,
) -> list[str]:
    codes = (
        observatory_codes
        if isinstance(observatory_codes, (pa.Array, pa.ChunkedArray))
        else pa.array(observatory_codes, type=pa.large_string())
    )
    nights = calculate_observing_night(codes, times.rescale("utc"))
    code_values = np.asarray(codes.to_numpy(zero_copy_only=False), dtype=object)
    night_values = np.asarray(nights.to_numpy(zero_copy_only=False), dtype=np.int64)
    return [
        f"{str(code)}:{int(night)}"
        for code, night in zip(code_values.tolist(), night_values.tolist())
    ]


def _normalize_point_sessions(
    session_labels: np.ndarray | None,
    n_points: int,
) -> tuple[list[str], np.ndarray]:
    if session_labels is None:
        return ["all"], np.asarray(["all"] * n_points, dtype=object)
    ordered_sessions = _order_session_ids(session_labels)
    point_sessions = np.asarray(
        ["all" if value is None else str(value) for value in session_labels.tolist()],
        dtype=object,
    )
    return ordered_sessions, point_sessions


def _build_default_session_meta(
    point_sessions: np.ndarray,
    filter_labels: np.ndarray,
    ordered_sessions: list[str],
) -> dict[str, dict[str, object]]:
    session_meta: dict[str, dict[str, object]] = {}
    for session_id in ordered_sessions:
        session_mask = point_sessions == session_id
        session_filters = np.asarray(filter_labels[session_mask], dtype=object)
        filter_text = ",".join(_ordered_unique(session_filters)) if session_filters.size else "ALL"
        session_meta[str(session_id)] = {
            "filter": filter_text,
            "count": int(np.count_nonzero(session_mask)),
        }
    return session_meta


def _merge_session_meta(
    base_meta: dict[str, dict[str, object]],
    override_meta: dict[str, dict[str, object]] | None,
) -> dict[str, dict[str, object]]:
    if override_meta is None:
        return base_meta
    merged = {key: dict(value) for key, value in base_meta.items()}
    for session_id, session_info in override_meta.items():
        merged[str(session_id)] = {
            **merged.get(str(session_id), {}),
            **dict(session_info),
        }
    return merged


def _extract_local_minima(
    periods: np.ndarray,
    sigma: np.ndarray,
    n: int,
) -> list[dict[str, float]]:
    valid = np.isfinite(sigma)
    if np.count_nonzero(valid) < 3:
        return []
    p = periods[valid]
    s = sigma[valid]
    minima = []
    for idx in range(1, len(s) - 1):
        if s[idx] <= s[idx - 1] and s[idx] <= s[idx + 1]:
            minima.append((float(p[idx]), float(s[idx])))
    minima.sort(key=lambda item: item[1])
    return [
        {"period_hours": period, "sigma": sigma_value}
        for period, sigma_value in minima[:n]
    ]


def _reference_period_hours(diag: dict[str, object]) -> float | None:
    if not bool(diag.get("has_reference_period", True)):
        return None
    reference = diag.get("reference_period_hours", diag.get("expected_period_hours"))
    if reference is None:
        return None
    return float(reference)


def _figure_title(diag: dict[str, object], key: str, fallback: str) -> str:
    value = diag.get(key)
    return fallback if value in (None, "") else str(value)


def build_rotation_period_diagnostics(
    observations: RotationPeriodObservations,
    *,
    search_kwargs: dict[str, object],
    name: str,
    object_number: int | str,
    key: str,
    tier: str,
    expected_period_hours: float | None = None,
    lcdb_u: int = -1,
    dataset_label: str | None = None,
    session_meta: dict[str, dict[str, object]] | None = None,
    top_alias_count: int = 0,
    dashboard_plot_title: str | None = None,
    session_audit_plot_title: str | None = None,
    search_surface_plot_title: str | None = None,
) -> dict[str, object]:
    result = estimate_rotation_period(observations, **search_kwargs)

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
    valid_sigma = (
        mag_sigma is not None
        and np.all(np.isfinite(mag_sigma))
        and np.all(mag_sigma > 0.0)
    )
    weights = None if not valid_sigma else 1.0 / np.square(np.asarray(mag_sigma))

    used_session_offsets = bool(result.used_session_offsets[0].as_py())
    model_session_labels = session_labels if used_session_offsets else None
    design_info = _build_fixed_design(filter_labels, model_session_labels, phase_angle)

    chosen_frequency = float(result.frequency_cycles_per_day[0].as_py())
    chosen_order = int(result.fourier_order[0].as_py())
    fit = _fit_frequency(
        time_rel,
        y,
        design_info,
        chosen_frequency,
        chosen_order,
        clip_sigma=3.0,
        weights=weights,
    )
    if fit is None:
        raise RuntimeError("Could not reconstruct the chosen fit for visualization.")

    fixed_cols = design_info.fixed.shape[1]
    fixed_component = design_info.fixed @ fit.coeffs[:fixed_cols]
    periodic_obs = _build_fourier_columns(time_rel, fit.frequency, fit.fourier_order) @ fit.coeffs[
        fixed_cols:
    ]
    fitted_full = fixed_component + periodic_obs
    normalized_mag = y - fixed_component
    residuals = normalized_mag - periodic_obs

    reported_period_days = float(result.period_days[0].as_py())
    reported_period_hours = float(result.period_hours[0].as_py())
    reported_phase = np.mod(time_rel / reported_period_days, 1.0)
    phase_twice = np.concatenate([reported_phase, reported_phase + 1.0])
    normalized_twice = np.concatenate([normalized_mag, normalized_mag])
    residuals_twice = np.concatenate([residuals, residuals])

    ordered_sessions, point_sessions = _normalize_point_sessions(session_labels, len(time_rel))
    merged_session_meta = _merge_session_meta(
        _build_default_session_meta(point_sessions, filter_labels, ordered_sessions),
        session_meta,
    )
    session_colors = _session_color_map(ordered_sessions)
    rel_days = np.asarray(time_rel, dtype=np.float64)
    order_idx = np.argsort(rel_days, kind="mergesort")

    phase_grid = np.linspace(0.0, 2.0, PHASE_MODEL_POINTS, dtype=np.float64)
    if bool(result.is_period_doubled[0].as_py()):
        fit_phase_grid = np.mod(2.0 * phase_grid, 1.0)
    else:
        fit_phase_grid = np.mod(phase_grid, 1.0)
    model_phase = _periodic_component_for_phase(
        fit.coeffs[fixed_cols:],
        fit.fourier_order,
        fit_phase_grid,
    )

    f_min = float(search_kwargs["min_rotations_in_span"] / span)
    f_max = float(search_kwargs["max_frequency_cycles_per_day"])
    n_freq = max(
        int(math.ceil(search_kwargs["frequency_grid_scale"] * span * (f_max - f_min)) + 1),
        2,
    )
    frequencies_full = np.linspace(f_min, f_max, n_freq, dtype=np.float64)
    if frequencies_full.size > SEARCH_SAMPLE_LIMIT:
        sample_idx = np.linspace(
            0,
            frequencies_full.size - 1,
            SEARCH_SAMPLE_LIMIT,
            dtype=np.int64,
        )
        sample_idx = np.unique(sample_idx)
        frequencies = np.asarray(frequencies_full[sample_idx], dtype=np.float64)
    else:
        frequencies = frequencies_full
    sigma_by_order: dict[int, np.ndarray] = {}
    for order in SEARCH_ORDERS:
        sigma = np.full(frequencies.shape, np.nan, dtype=np.float64)
        for idx, frequency in enumerate(frequencies):
            candidate = _fit_frequency(
                time_rel,
                y,
                design_info,
                float(frequency),
                int(order),
                clip_sigma=3.0,
                weights=weights,
            )
            if candidate is not None:
                sigma[idx] = float(candidate.residual_sigma)
        sigma_by_order[int(order)] = sigma

    reference_period = None if expected_period_hours is None else float(expected_period_hours)
    top_aliases = []
    if top_alias_count > 0:
        periods_hours = 24.0 / frequencies
        top_aliases = _extract_local_minima(
            periods_hours,
            np.asarray(sigma_by_order[chosen_order], dtype=np.float64),
            top_alias_count,
        )

    return {
        "name": str(name),
        "object_number": object_number,
        "key": str(key),
        "tier": str(tier),
        "dataset_label": str(dataset_label) if dataset_label is not None else str(name),
        "lcdb_u": int(lcdb_u),
        "has_reference_period": reference_period is not None,
        "reference_period_hours": reference_period,
        "expected_period_hours": (
            float(reference_period) if reference_period is not None else reported_period_hours
        ),
        "fitted_period_hours": reported_period_hours,
        "relative_error_pct": (
            abs(reported_period_hours - reference_period) / reference_period * 100.0
            if reference_period is not None
            else 0.0
        ),
        "fourier_order": chosen_order,
        "frequency_cycles_per_day": chosen_frequency,
        "residual_sigma_mag": float(result.residual_sigma_mag[0].as_py()),
        "is_period_doubled": bool(result.is_period_doubled[0].as_py()),
        "n_observations": int(result.n_observations[0].as_py()),
        "n_fit_observations": int(result.n_fit_observations[0].as_py()),
        "n_clipped": int(result.n_clipped[0].as_py()),
        "n_filters": int(result.n_filters[0].as_py()),
        "n_sessions": int(result.n_sessions[0].as_py()),
        "used_session_offsets": used_session_offsets,
        "session_mode_requested": str(search_kwargs.get("session_mode", "auto")),
        "session_ids": ordered_sessions,
        "session_colors": session_colors,
        "session_meta": merged_session_meta,
        "rel_days": rel_days,
        "raw_reduced_mag": y,
        "fitted_full": fitted_full,
        "normalized_mag": normalized_mag,
        "periodic_obs": periodic_obs,
        "residuals": residuals,
        "phase_twice": phase_twice,
        "normalized_twice": normalized_twice,
        "residuals_twice": residuals_twice,
        "point_sessions": point_sessions,
        "fit_mask": np.asarray(fit.mask, dtype=bool),
        "model_phase_grid": phase_grid,
        "model_phase_values": model_phase,
        "time_order": order_idx,
        "search_frequencies": frequencies,
        "search_sigma_by_order": sigma_by_order,
        "top_aliases": top_aliases,
        "dashboard_plot_title": dashboard_plot_title,
        "session_audit_plot_title": session_audit_plot_title,
        "search_surface_plot_title": search_surface_plot_title,
    }


def _build_diagnostics(
    fx: np.lib.npyio.NpzFile,
) -> dict[str, object]:
    observations = _fixture_to_observations(fx)
    search_kwargs = dict(
        frequency_grid_scale=float(fx["frequency_grid_scale"][0]),
        max_frequency_cycles_per_day=float(fx["max_frequency_cycles_per_day"][0]),
        min_rotations_in_span=float(fx["min_rotations_in_span"][0]),
        session_mode="use",
    )
    session_meta = {
        str(session_id): {"filter": str(filter_value) if filter_value is not None else "UNKNOWN"}
        for session_id, filter_value in zip(
            fx["session_mdid"].astype(object).tolist(),
            fx["session_filter"].astype(object).tolist(),
        )
    }
    return build_rotation_period_diagnostics(
        observations,
        search_kwargs=search_kwargs,
        name=str(fx["object_name"][0]),
        object_number=int(fx["object_number"][0]),
        key=f"{int(fx['object_number'][0])} {str(fx['object_name'][0])}",
        tier=str(fx["tier"][0]),
        expected_period_hours=float(fx["expected_period_hours"][0]),
        lcdb_u=int(fx["lcdb_u"][0]),
        session_meta=session_meta,
    )


def _apply_common_layout(fig: go.Figure, title: str) -> None:
    fig.update_layout(
        title=dict(
            text=title,
            x=0.01,
            xanchor="left",
            y=0.985,
            yanchor="top",
            pad=dict(t=6, b=12),
        ),
        paper_bgcolor="#0b1220",
        plot_bgcolor="#111827",
        font=dict(color="#dbe4ff", family="Avenir Next, Segoe UI, Helvetica Neue, sans-serif"),
        margin=dict(l=70, r=36, t=120, b=72),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.10,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(15, 23, 42, 0.65)",
        ),
        hovermode="closest",
    )
    fig.update_xaxes(automargin=True, title_standoff=18)
    fig.update_yaxes(automargin=True, title_standoff=14)


def _add_session_points(
    fig: go.Figure,
    *,
    x: np.ndarray,
    y: np.ndarray,
    point_sessions: np.ndarray,
    ordered_sessions: list[str],
    session_colors: dict[str, str],
    row: int,
    col: int,
    showlegend: bool,
    name_suffix: str = "",
    clipped_mask: np.ndarray | None = None,
) -> None:
    clipped_mask = (
        np.zeros(x.shape, dtype=bool) if clipped_mask is None else np.asarray(clipped_mask)
    )
    retained_mask = ~clipped_mask

    for session_id in ordered_sessions:
        session_mask = point_sessions == session_id
        if np.any(session_mask & retained_mask):
            fig.add_trace(
                go.Scattergl(
                    x=x[session_mask & retained_mask],
                    y=y[session_mask & retained_mask],
                    mode="markers",
                    marker=dict(
                        color=session_colors[session_id],
                        size=8,
                        line=dict(width=0),
                        opacity=0.82,
                    ),
                    name=f"{session_id}{name_suffix}",
                    legendgroup=session_id,
                    showlegend=showlegend,
                    hovertemplate=(
                        "session=%{fullData.legendgroup}<br>"
                        "x=%{x:.3f}<br>y=%{y:.4f}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
        if np.any(session_mask & clipped_mask):
            fig.add_trace(
                go.Scattergl(
                    x=x[session_mask & clipped_mask],
                    y=y[session_mask & clipped_mask],
                    mode="markers",
                    marker=dict(
                        color=session_colors[session_id],
                        size=9,
                        symbol="x",
                        line=dict(width=1.2),
                    ),
                    name=f"{session_id} clipped{name_suffix}",
                    legendgroup=session_id,
                    showlegend=False,
                    hovertemplate=(
                        "session=%{fullData.legendgroup}<br>"
                        "clipped=yes<br>x=%{x:.3f}<br>y=%{y:.4f}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )


def _make_dashboard_figure(diag: dict[str, object]) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Distance-Corrected Lightcurve",
            "Phase-Folded Signal",
            "Frequency Search",
            "Residuals After Model",
        ),
        vertical_spacing=0.20,
        horizontal_spacing=0.10,
    )

    clipped_mask = ~np.asarray(diag["fit_mask"], dtype=bool)

    _add_session_points(
        fig,
        x=np.asarray(diag["rel_days"], dtype=np.float64),
        y=np.asarray(diag["raw_reduced_mag"], dtype=np.float64),
        point_sessions=np.asarray(diag["point_sessions"], dtype=object),
        ordered_sessions=list(diag["session_ids"]),
        session_colors=dict(diag["session_colors"]),
        row=1,
        col=1,
        showlegend=True,
        clipped_mask=clipped_mask,
    )
    order_idx = np.asarray(diag["time_order"], dtype=np.int64)
    fig.add_trace(
        go.Scatter(
            x=np.asarray(diag["rel_days"], dtype=np.float64)[order_idx],
            y=np.asarray(diag["fitted_full"], dtype=np.float64)[order_idx],
            mode="lines",
            line=dict(color="#f8fafc", width=2.4),
            name="full fit",
            hovertemplate="days=%{x:.3f}<br>model=%{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    _add_session_points(
        fig,
        x=np.asarray(diag["phase_twice"], dtype=np.float64),
        y=np.asarray(diag["normalized_twice"], dtype=np.float64),
        point_sessions=np.concatenate(
            [
                np.asarray(diag["point_sessions"], dtype=object),
                np.asarray(diag["point_sessions"], dtype=object),
            ]
        ),
        ordered_sessions=list(diag["session_ids"]),
        session_colors=dict(diag["session_colors"]),
        row=1,
        col=2,
        showlegend=False,
        clipped_mask=np.concatenate([clipped_mask, clipped_mask]),
    )
    fig.add_trace(
        go.Scatter(
            x=np.asarray(diag["model_phase_grid"], dtype=np.float64),
            y=np.asarray(diag["model_phase_values"], dtype=np.float64),
            mode="lines",
            line=dict(color="#f8fafc", width=3),
            name="periodic model",
            hovertemplate="phase=%{x:.3f}<br>model=%{y:.4f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    periods_hours = 24.0 / np.asarray(diag["search_frequencies"], dtype=np.float64)
    for order, sigma in dict(diag["search_sigma_by_order"]).items():
        fig.add_trace(
            go.Scatter(
                x=periods_hours,
                y=np.asarray(sigma, dtype=np.float64),
                mode="lines",
                name=f"order {order}",
                line=dict(width=2),
                hovertemplate=(
                    f"order={order}<br>period=%{{x:.3f}} h<br>"
                    "sigma=%{y:.4f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    _add_session_points(
        fig,
        x=np.asarray(diag["phase_twice"], dtype=np.float64),
        y=np.asarray(diag["residuals_twice"], dtype=np.float64),
        point_sessions=np.concatenate(
            [
                np.asarray(diag["point_sessions"], dtype=object),
                np.asarray(diag["point_sessions"], dtype=object),
            ]
        ),
        ordered_sessions=list(diag["session_ids"]),
        session_colors=dict(diag["session_colors"]),
        row=2,
        col=2,
        showlegend=False,
        clipped_mask=np.concatenate([clipped_mask, clipped_mask]),
    )
    fig.add_hline(y=0.0, line=dict(color="#94a3b8", dash="dot"), row=2, col=2)

    reference_period = _reference_period_hours(diag)
    fitted_period = float(diag["fitted_period_hours"])
    if reference_period is not None:
        fig.add_vline(
            x=reference_period,
            line=dict(color="#ffd166", dash="dash"),
            row=2,
            col=1,
        )
    fig.add_vline(
        x=fitted_period,
        line=dict(color="#ef476f", dash="dot"),
        row=2,
        col=1,
    )

    fig.update_xaxes(title_text="days since first observation", row=1, col=1)
    fig.update_yaxes(title_text="reduced magnitude", autorange="reversed", row=1, col=1)
    fig.update_xaxes(title_text="phase (two cycles)", range=[0.0, 2.0], row=1, col=2)
    fig.update_yaxes(
        title_text="normalized magnitude",
        autorange="reversed",
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="candidate period (hours)", type="log", row=2, col=1)
    fig.update_yaxes(title_text="fit sigma (mag)", row=2, col=1)
    fig.update_xaxes(title_text="phase (two cycles)", range=[0.0, 2.0], row=2, col=2)
    fig.update_yaxes(title_text="residual (mag)", row=2, col=2)

    _apply_common_layout(
        fig,
        _figure_title(
            diag,
            "dashboard_plot_title",
            f"Variant 1: Analyst Dashboard for {diag['object_number']} {diag['name']}",
        ),
    )
    return fig


def _make_session_audit_figure(diag: dict[str, object]) -> go.Figure:
    ordered_sessions = list(diag["session_ids"])
    n_sessions = len(ordered_sessions)
    cols = 2 if n_sessions > 1 else 1
    rows = int(math.ceil(n_sessions / cols))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[
            (
                f"session {session_id} "
                f"({diag['session_meta'][session_id]['filter']}, "
                f"n={diag['session_meta'][session_id]['count']})"
            )
            for session_id in ordered_sessions
        ],
        vertical_spacing=0.16,
        horizontal_spacing=0.08,
    )

    point_sessions = np.asarray(diag["point_sessions"], dtype=object)
    phase_twice = np.asarray(diag["phase_twice"], dtype=np.float64)
    normalized_twice = np.asarray(diag["normalized_twice"], dtype=np.float64)
    repeated_sessions = np.concatenate([point_sessions, point_sessions])

    for panel_index, session_id in enumerate(ordered_sessions):
        row = panel_index // cols + 1
        col = panel_index % cols + 1
        session_mask = repeated_sessions == session_id
        fig.add_trace(
            go.Scattergl(
                x=phase_twice[session_mask],
                y=normalized_twice[session_mask],
                mode="markers",
                marker=dict(
                    color=dict(diag["session_colors"])[session_id],
                    size=8,
                    opacity=0.82,
                ),
                name=session_id,
                legendgroup=session_id,
                showlegend=False,
                hovertemplate=(
                    f"session={session_id}<br>phase=%{{x:.3f}}<br>"
                    "normalized=%{y:.4f}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=np.asarray(diag["model_phase_grid"], dtype=np.float64),
                y=np.asarray(diag["model_phase_values"], dtype=np.float64),
                mode="lines",
                line=dict(color="#f8fafc", width=2.4),
                name="periodic model",
                showlegend=False,
                hovertemplate="phase=%{x:.3f}<br>model=%{y:.4f}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="phase (two cycles)", range=[0.0, 2.0], row=row, col=col)
        fig.update_yaxes(
            title_text="normalized magnitude" if col == 1 else None,
            autorange="reversed",
            row=row,
            col=col,
        )

    _apply_common_layout(
        fig,
        _figure_title(
            diag,
            "session_audit_plot_title",
            f"Variant 2: Session Audit for {diag['object_number']} {diag['name']}",
        ),
    )
    fig.update_layout(height=max(520, 320 * rows))
    return fig


def _make_search_surface_figure(diag: dict[str, object]) -> go.Figure:
    periods_hours = 24.0 / np.asarray(diag["search_frequencies"], dtype=np.float64)
    sort_idx = np.argsort(periods_hours)
    period_axis = periods_hours[sort_idx]
    orders = list(dict(diag["search_sigma_by_order"]).keys())
    heatmap_z = np.vstack(
        [np.asarray(dict(diag["search_sigma_by_order"])[order], dtype=np.float64)[sort_idx] for order in orders]
    )
    best_sigma = np.asarray(
        [
            np.nanmin(np.asarray(dict(diag["search_sigma_by_order"])[order], dtype=np.float64))
            for order in orders
        ],
        dtype=np.float64,
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.22,
        subplot_titles=("Search Surface by Order", "Best Sigma by Fourier Order"),
    )

    fig.add_trace(
        go.Heatmap(
            x=period_axis,
            y=orders,
            z=heatmap_z,
            colorscale="Turbo",
            colorbar=dict(title="fit sigma"),
            hovertemplate=(
                "order=%{y}<br>period=%{x:.3f} h<br>"
                "sigma=%{z:.4f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    reference_period = _reference_period_hours(diag)
    if reference_period is not None:
        fig.add_vline(
            x=reference_period,
            line=dict(color="#ffd166", dash="dash"),
            row=1,
            col=1,
        )
    fig.add_vline(
        x=float(diag["fitted_period_hours"]),
        line=dict(color="#ef476f", dash="dot"),
        row=1,
        col=1,
    )

    bar_colors = ["#64748b"] * len(orders)
    chosen_order = int(diag["fourier_order"])
    if chosen_order in orders:
        bar_colors[orders.index(chosen_order)] = "#ef476f"
    fig.add_trace(
        go.Bar(
            x=orders,
            y=best_sigma,
            marker=dict(color=bar_colors),
            hovertemplate="order=%{x}<br>best sigma=%{y:.4f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(title_text="candidate period (hours)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Fourier order", row=1, col=1)
    fig.update_xaxes(title_text="Fourier order", row=2, col=1)
    fig.update_yaxes(title_text="best sigma (mag)", row=2, col=1)

    _apply_common_layout(
        fig,
        _figure_title(
            diag,
            "search_surface_plot_title",
            f"Variant 3: Search Surface for {diag['object_number']} {diag['name']}",
        ),
    )
    return fig


def _build_gallery_payload() -> dict[str, object]:
    payload: dict[str, object] = {}
    for fixture_path in sorted(FIXTURE_DIR.glob("rotation_period_pds_fixture_*.npz")):
        with np.load(fixture_path, allow_pickle=True) as fx:
            if int(fx["object_number"][0]) not in GALLERY_OBJECT_NUMBERS:
                continue
            diag = _build_diagnostics(fx)
        payload[str(diag["key"])] = {
            "meta": {
                "object_number": diag["object_number"],
                "name": diag["name"],
                "tier": str(diag["tier"]).upper(),
                "expected_period_hours": round(float(diag["expected_period_hours"]), 6),
                "fitted_period_hours": round(float(diag["fitted_period_hours"]), 6),
                "relative_error_pct": round(float(diag["relative_error_pct"]), 4),
                "fourier_order": int(diag["fourier_order"]),
                "frequency_cycles_per_day": round(
                    float(diag["frequency_cycles_per_day"]), 6
                ),
                "lcdb_u": int(diag["lcdb_u"]),
                "n_observations": int(diag["n_observations"]),
                "n_fit_observations": int(diag["n_fit_observations"]),
                "n_clipped": int(diag["n_clipped"]),
                "n_sessions": int(diag["n_sessions"]),
                "n_filters": int(diag["n_filters"]),
                "session_mode_requested": str(diag["session_mode_requested"]),
                "used_session_offsets": bool(diag["used_session_offsets"]),
                "is_period_doubled": bool(diag["is_period_doubled"]),
            },
            "variants": {
                "dashboard": json.loads(_make_dashboard_figure(diag).to_json()),
                "session_audit": json.loads(_make_session_audit_figure(diag).to_json()),
                "search_surface": json.loads(_make_search_surface_figure(diag).to_json()),
            },
        }
        print(f"Prepared {diag['key']}", flush=True)
    return payload


def _build_html(payload: dict[str, object]) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Rotation Period Plotly Gallery</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
      :root {{
        --bg: #08101d;
        --panel: rgba(15, 23, 42, 0.82);
        --panel-strong: rgba(17, 24, 39, 0.96);
        --ink: #e8eefc;
        --muted: #9fb2d8;
        --accent: #ef476f;
        --accent-2: #06d6a0;
        --accent-3: #ffd166;
        --border: rgba(148, 163, 184, 0.18);
        --shadow: 0 20px 60px rgba(2, 6, 23, 0.42);
      }}

      * {{
        box-sizing: border-box;
      }}

      body {{
        margin: 0;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(17, 138, 178, 0.18), transparent 28%),
          radial-gradient(circle at top right, rgba(239, 71, 111, 0.18), transparent 30%),
          linear-gradient(180deg, #08101d 0%, #0d1728 100%);
        font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
      }}

      .page {{
        max-width: 1540px;
        margin: 0 auto;
        padding: 40px 28px 72px;
      }}

      .hero {{
        display: grid;
        grid-template-columns: minmax(260px, 1.15fr) minmax(280px, 0.85fr);
        gap: 22px;
        align-items: start;
        margin-bottom: 24px;
      }}

      .hero-copy, .hero-controls, .variant-card {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 24px;
        box-shadow: var(--shadow);
        backdrop-filter: blur(14px);
      }}

      .hero-copy {{
        padding: 28px;
      }}

      .hero-copy h1 {{
        margin: 0 0 10px;
        font-size: 2.1rem;
        letter-spacing: -0.03em;
      }}

      .hero-copy p {{
        margin: 0 0 12px;
        color: var(--muted);
        line-height: 1.55;
      }}

      .hero-copy .recommendations {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        margin-top: 18px;
      }}

      .rec {{
        padding: 14px 16px;
        border-radius: 18px;
        background: rgba(8, 16, 29, 0.55);
        border: 1px solid var(--border);
      }}

      .rec strong {{
        display: block;
        margin-bottom: 6px;
        font-size: 0.92rem;
        color: var(--ink);
      }}

      .rec span {{
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.45;
      }}

      .hero-controls {{
        padding: 24px;
      }}

      .controls-label {{
        display: block;
        margin-bottom: 10px;
        color: var(--muted);
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }}

      select {{
        width: 100%;
        padding: 15px 18px;
        border-radius: 14px;
        border: 1px solid rgba(159, 178, 216, 0.28);
        background: var(--panel-strong);
        color: var(--ink);
        font-size: 1rem;
      }}

      .summary-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin-top: 20px;
      }}

      .summary-card {{
        padding: 14px 16px;
        border-radius: 18px;
        border: 1px solid var(--border);
        background: rgba(8, 16, 29, 0.55);
      }}

      .summary-card .label {{
        color: var(--muted);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
      }}

      .summary-card .value {{
        font-size: 1.08rem;
        font-weight: 600;
      }}

      .variant-stack {{
        display: grid;
        gap: 18px;
      }}

      .variant-card {{
        padding: 22px 22px 12px;
      }}

      .variant-header {{
        display: flex;
        justify-content: space-between;
        gap: 20px;
        align-items: baseline;
        margin-bottom: 8px;
      }}

      .variant-header h2 {{
        margin: 0;
        font-size: 1.18rem;
        letter-spacing: -0.02em;
      }}

      .variant-header span {{
        color: var(--accent-3);
        font-size: 0.9rem;
      }}

      .variant-card p {{
        margin: 0 0 14px;
        color: var(--muted);
        line-height: 1.5;
      }}

      .plot {{
        min-height: 520px;
      }}

      .plot.tall {{
        min-height: 760px;
      }}

      @media (max-width: 1100px) {{
        .hero {{
          grid-template-columns: 1fr;
        }}

        .hero-copy .recommendations {{
          grid-template-columns: 1fr;
        }}

        .summary-grid {{
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }}
      }}

      @media (max-width: 720px) {{
        .page {{
          padding: 20px 14px 44px;
        }}

        .summary-grid {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
  </head>
  <body>
    <div class="page">
      <section class="hero">
        <div class="hero-copy">
          <h1>Rotation Period Visualization Gallery</h1>
          <p>
            This page renders the frozen PDS validation set through three Plotly
            variants. Each one answers a different question: what the solver fit,
            whether session offsets are behaving, and where the alias structure lives.
          </p>
          <p>
            Recommended short list for the eventual module model:
            start with the analyst dashboard, keep the session audit for diagnosis,
            and use the search surface when period ambiguity matters.
          </p>
          <div class="recommendations">
            <div class="rec">
              <strong>Variant 1: Analyst Dashboard</strong>
              <span>Best default view. It combines time-domain behavior, the folded signal, the search curve, and residuals.</span>
            </div>
            <div class="rec">
              <strong>Variant 2: Session Audit</strong>
              <span>Best debugging view. It shows whether the session-aware nuisance terms actually align separate campaigns.</span>
            </div>
            <div class="rec">
              <strong>Variant 3: Search Surface</strong>
              <span>Best expert view. It exposes alias families and whether the chosen Fourier order is actually justified.</span>
            </div>
          </div>
        </div>
        <div class="hero-controls">
          <label class="controls-label" for="object-select">Fixture Object</label>
          <select id="object-select"></select>
          <div class="summary-grid" id="summary-grid"></div>
        </div>
      </section>

      <section class="variant-stack">
        <article class="variant-card">
          <div class="variant-header">
            <h2>Variant 1: Analyst Dashboard</h2>
            <span>Default choice for general review</span>
          </div>
          <p>
            Use this when you want the single most useful page per object. It is the best
            candidate for a first-class module visualization because it exposes both the fit
            and the failure modes without forcing the user into expert-only diagnostics.
          </p>
          <div class="plot" id="dashboard-plot"></div>
        </article>

        <article class="variant-card">
          <div class="variant-header">
            <h2>Variant 2: Session Audit</h2>
            <span>Best for debugging mixed campaigns</span>
          </div>
          <p>
            Use this when the question is whether session offsets, passband mixing, or
            curation choices are driving the answer. This is the view I would reach for first
            when a real-data recovery looks suspicious.
          </p>
          <div class="plot tall" id="session-audit-plot"></div>
        </article>

        <article class="variant-card">
          <div class="variant-header">
            <h2>Variant 3: Search Surface</h2>
            <span>Best for alias and order sensitivity</span>
          </div>
          <p>
            Use this when you need to explain why the solver chose a specific period and order.
            It is the most technical view here, but it is the cleanest way to surface ambiguity.
          </p>
          <div class="plot" id="search-surface-plot"></div>
        </article>
      </section>
    </div>

    <script>
      const payload = {payload_json};
      const objectKeys = Object.keys(payload).sort((a, b) => {{
        const aNum = parseInt(a.split(" ")[0], 10);
        const bNum = parseInt(b.split(" ")[0], 10);
        return aNum - bNum;
      }});

      const select = document.getElementById("object-select");
      const summaryGrid = document.getElementById("summary-grid");
      const plots = {{
        dashboard: document.getElementById("dashboard-plot"),
        session_audit: document.getElementById("session-audit-plot"),
        search_surface: document.getElementById("search-surface-plot"),
      }};

      const summaryFields = [
        ["Tier", meta => meta.tier],
        ["Period", meta => `${{meta.fitted_period_hours.toFixed(6)}} h`],
        ["LCDB", meta => `${{meta.expected_period_hours.toFixed(6)}} h (U=${{meta.lcdb_u}})`],
        ["Error", meta => `${{meta.relative_error_pct.toFixed(4)}}%`],
        ["Order", meta => `k=${{meta.fourier_order}}`],
        ["Frequency", meta => `${{meta.frequency_cycles_per_day.toFixed(6)}} c/d`],
        ["Observations", meta => `${{meta.n_fit_observations}} / ${{meta.n_observations}} fit, ${{meta.n_filters}} filter(s)`],
        ["Sessions", meta => `${{meta.n_sessions}} raw session(s), ${{meta.n_clipped}} clipped`],
        ["Session Model", meta => `${{meta.session_mode_requested}} / ${{meta.used_session_offsets ? "used" : "ignored"}}`],
      ];

      function renderSummary(meta) {{
        summaryGrid.innerHTML = summaryFields.map(([label, formatter]) => `
          <div class="summary-card">
            <div class="label">${{label}}</div>
            <div class="value">${{formatter(meta)}}</div>
          </div>
        `).join("");
      }}

      function renderFigures(objectKey) {{
        const item = payload[objectKey];
        renderSummary(item.meta);

        Plotly.react(
          plots.dashboard,
          item.variants.dashboard.data,
          item.variants.dashboard.layout,
          {{ responsive: true, displaylogo: false }}
        );
        Plotly.react(
          plots.session_audit,
          item.variants.session_audit.data,
          item.variants.session_audit.layout,
          {{ responsive: true, displaylogo: false }}
        );
        Plotly.react(
          plots.search_surface,
          item.variants.search_surface.data,
          item.variants.search_surface.layout,
          {{ responsive: true, displaylogo: false }}
        );
      }}

      objectKeys.forEach(key => {{
        const option = document.createElement("option");
        option.value = key;
        option.textContent = key;
        select.appendChild(option);
      }});

      select.addEventListener("change", event => renderFigures(event.target.value));
      renderFigures(objectKeys[0]);
    </script>
  </body>
</html>
"""


def main() -> None:
    payload = _build_gallery_payload()
    OUTPUT_PATH.write_text(_build_html(payload), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
