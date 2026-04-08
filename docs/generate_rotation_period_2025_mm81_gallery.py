from __future__ import annotations

import astropy.time
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
from google.cloud import bigquery
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adam_core.coordinates.origin import OriginCodes
from adam_core.observations.exposures import Exposures
from adam_core.orbits.query.horizons import query_horizons
from adam_core.photometry.rotation_period_fourier import (
    _build_fixed_design,
    _build_fourier_columns,
    _fit_frequency,
    _validate_inputs,
    estimate_rotation_period,
)
from adam_core.photometry.rotation_period_types import RotationPeriodObservations
from adam_core.time import Timestamp

DOCS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DOCS_DIR))
import generate_rotation_period_plotly_gallery as base  # noqa: E402


OUTPUT_PATH = DOCS_DIR / "rotation_period_2025_mm81_gallery.html"
DATASET_ID = "moeyens-thor-dev.mpc_sbn_aurora"
PROVID = "2025 MM81"
STATION = "X05"
SEARCH_ORDERS = (2, 3, 4, 5, 6)
TOP_ALIAS_COUNT = 3


@dataclass(frozen=True)
class DatasetVariant:
    key: str
    label: str
    description: str
    bands: tuple[str, ...]
    frequency_grid_scale: float = 40.0
    max_frequency_cycles_per_day: float = 24.0
    min_rotations_in_span: float = 2.0


VARIANTS: tuple[DatasetVariant, ...] = (
    DatasetVariant(
        key="all",
        label="All Bands",
        description="Combined Rubin `g/r/i` fit. This is the closest match to the real production use case for a new object.",
        bands=("g", "r", "i"),
    ),
    DatasetVariant(
        key="r",
        label="r Only",
        description="Densest single-band subset. Useful for separating real signal from cross-band coupling.",
        bands=("r",),
    ),
    DatasetVariant(
        key="g",
        label="g Only",
        description="Lower-count cross-check on the same object. Useful for testing whether the favored period repeats independently.",
        bands=("g",),
    ),
    DatasetVariant(
        key="i",
        label="i Only",
        description="Sparsest band in this object. Useful as a stress test for stability rather than a primary solution.",
        bands=("i",),
    ),
)


def _timestamp_from_bq_obstime(obstime: pa.Array | pa.ChunkedArray) -> Timestamp:
    if isinstance(obstime, pa.ChunkedArray):
        obstime = obstime.combine_chunks()
    dt64 = obstime.to_numpy(zero_copy_only=False)
    t = astropy.time.Time(dt64, format="datetime64", scale="utc")
    return Timestamp.from_astropy(t)


def _query_rows(client: bigquery.Client, variant: DatasetVariant) -> pa.Table:
    bands_sql = ", ".join(f"'{band}'" for band in variant.bands)
    query = f"""
    SELECT
      obsid,
      obstime,
      mag,
      rmsmag,
      TRIM(band) AS band,
      TRIM(stn) AS stn
    FROM `{DATASET_ID}.public_obs_sbn`
    WHERE provid = @provid
      AND TRIM(stn) = @stn
      AND TRIM(band) IN ({bands_sql})
      AND obstime IS NOT NULL
      AND mag IS NOT NULL
    ORDER BY obstime ASC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("provid", "STRING", PROVID),
            bigquery.ScalarQueryParameter("stn", "STRING", STATION),
        ]
    )
    return client.query(query, job_config=job_config).result().to_arrow(
        create_bqstorage_client=True
    )


def _build_geometry(
    *,
    times_utc: Timestamp,
    filters: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(times_utc)
    exposures = Exposures.from_kwargs(
        id=[f"{PROVID}_{i}" for i in range(n)],
        start_time=times_utc,
        duration=np.zeros(n, dtype=np.float64),
        filter=filters,
        observatory_code=[STATION] * n,
    )
    observers = exposures.observers(frame="ecliptic", origin=OriginCodes.SUN)
    coords = query_horizons(
        object_ids=[PROVID],
        times=times_utc,
        coordinate_type="cartesian",
        location="@sun",
        id_type="smallbody",
    ).coordinates

    object_r = np.asarray(coords.r, dtype=np.float64)
    observer_r = np.asarray(observers.coordinates.r, dtype=np.float64)
    r_au = np.linalg.norm(object_r, axis=1)
    delta_vec = object_r - observer_r
    delta_au = np.linalg.norm(delta_vec, axis=1)
    observer_sun_au = np.linalg.norm(observer_r, axis=1)
    cos_alpha = np.clip(
        (r_au * r_au + delta_au * delta_au - observer_sun_au * observer_sun_au)
        / (2.0 * r_au * delta_au),
        -1.0,
        1.0,
    )
    phase_angle_deg = np.degrees(np.arccos(cos_alpha))
    return r_au, delta_au, phase_angle_deg


def _periodic_component_for_phase(
    fourier_coeffs: np.ndarray,
    fourier_order: int,
    phase_fit: np.ndarray,
) -> np.ndarray:
    periodic = np.zeros_like(phase_fit, dtype=np.float64)
    for harmonic in range(1, fourier_order + 1):
        idx = 2 * (harmonic - 1)
        periodic += fourier_coeffs[idx] * np.cos(2.0 * np.pi * harmonic * phase_fit)
        periodic += fourier_coeffs[idx + 1] * np.sin(2.0 * np.pi * harmonic * phase_fit)
    return periodic


def _extract_local_minima(periods: np.ndarray, sigma: np.ndarray, n: int) -> list[dict[str, float]]:
    valid = np.isfinite(sigma)
    if np.count_nonzero(valid) < 3:
        return []
    p = periods[valid]
    s = sigma[valid]
    minima = []
    for idx in range(1, len(s) - 1):
        if s[idx] <= s[idx - 1] and s[idx] <= s[idx + 1]:
            minima.append((float(p[idx]), float(s[idx])))
    if not minima:
        return []
    minima.sort(key=lambda item: item[1])
    top = minima[:n]
    return [{"period_hours": period, "sigma": sigma_value} for period, sigma_value in top]


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
    base._add_session_points(
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

    base._add_session_points(
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
    fig.add_vline(
        x=float(diag["fitted_period_hours"]),
        line=dict(color="#ef476f", dash="dot"),
        row=2,
        col=1,
    )

    base._add_session_points(
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

    fig.update_xaxes(title_text="days since first observation", row=1, col=1)
    fig.update_yaxes(title_text="reduced magnitude", autorange="reversed", row=1, col=1)
    fig.update_xaxes(title_text="phase (two cycles)", range=[0.0, 2.0], row=1, col=2)
    fig.update_yaxes(title_text="normalized magnitude", autorange="reversed", row=1, col=2)
    fig.update_xaxes(title_text="candidate period (hours)", type="log", row=2, col=1)
    fig.update_yaxes(title_text="fit sigma (mag)", row=2, col=1)
    fig.update_xaxes(title_text="phase (two cycles)", range=[0.0, 2.0], row=2, col=2)
    fig.update_yaxes(title_text="residual (mag)", row=2, col=2)

    base._apply_common_layout(fig, f"2025 MM81 - {diag['dataset_label']} Dashboard")
    return fig


def _make_session_audit_figure(diag: dict[str, object]) -> go.Figure:
    return base._make_session_audit_figure(diag)


def _make_search_surface_figure(diag: dict[str, object]) -> go.Figure:
    periods_hours = 24.0 / np.asarray(diag["search_frequencies"], dtype=np.float64)
    sort_idx = np.argsort(periods_hours)
    period_axis = periods_hours[sort_idx]
    orders = list(dict(diag["search_sigma_by_order"]).keys())
    heatmap_z = np.vstack(
        [
            np.asarray(dict(diag["search_sigma_by_order"])[order], dtype=np.float64)[sort_idx]
            for order in orders
        ]
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
            hovertemplate="order=%{y}<br>period=%{x:.3f} h<br>sigma=%{z:.4f}<extra></extra>",
        ),
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
    base._apply_common_layout(fig, f"2025 MM81 - {diag['dataset_label']} Search Surface")
    return fig


def _build_variant_payload(client: bigquery.Client, variant: DatasetVariant) -> tuple[str, dict[str, object]]:
    rows = _query_rows(client, variant)
    if rows.num_rows == 0:
        raise RuntimeError(f"No observations found for variant {variant.key}")

    times_utc = _timestamp_from_bq_obstime(rows.column("obstime"))
    bands = np.asarray(rows.column("band").to_numpy(zero_copy_only=False), dtype=object)
    r_au, delta_au, phase_angle_deg = _build_geometry(
        times_utc=times_utc,
        filters=bands.tolist(),
    )

    observations = RotationPeriodObservations.from_kwargs(
        time=times_utc,
        mag=np.asarray(rows.column("mag").to_numpy(zero_copy_only=False), dtype=np.float64),
        mag_sigma=pa.array(
            np.asarray(rows.column("rmsmag").to_numpy(zero_copy_only=False), dtype=np.float64),
            mask=~np.isfinite(np.asarray(rows.column("rmsmag").to_numpy(zero_copy_only=False), dtype=np.float64)),
            type=pa.float64(),
        ),
        filter=bands.tolist(),
        session_id=base.build_observatory_night_session_ids(
            [STATION] * rows.num_rows,
            times_utc,
        ),
        r_au=r_au,
        delta_au=delta_au,
        phase_angle_deg=phase_angle_deg,
    )

    diag = base.build_rotation_period_diagnostics(
        observations,
        search_kwargs={
            "frequency_grid_scale": variant.frequency_grid_scale,
            "max_frequency_cycles_per_day": variant.max_frequency_cycles_per_day,
            "min_rotations_in_span": variant.min_rotations_in_span,
            "session_mode": "auto",
        },
        name=PROVID,
        object_number=PROVID,
        key=variant.key,
        tier="uncataloged",
        dataset_label=variant.label,
        top_alias_count=TOP_ALIAS_COUNT,
        dashboard_plot_title=f"2025 MM81 - {variant.label} Dashboard",
        session_audit_plot_title=f"2025 MM81 - {variant.label} Session Audit",
        search_surface_plot_title=f"2025 MM81 - {variant.label} Search Surface",
    )

    diag = {
        **diag,
        "dataset_label": variant.label,
    }

    payload = {
        "meta": {
            "object_id": PROVID,
            "dataset_label": variant.label,
            "description": variant.description,
            "station": STATION,
            "bands": list(variant.bands),
            "estimated_period_hours": round(float(diag["fitted_period_hours"]), 6),
            "frequency_cycles_per_day": round(float(diag["frequency_cycles_per_day"]), 6),
            "fourier_order": int(diag["fourier_order"]),
            "n_observations": int(diag["n_observations"]),
            "n_fit_observations": int(diag["n_fit_observations"]),
            "n_clipped": int(diag["n_clipped"]),
            "n_sessions": int(diag["n_sessions"]),
            "n_filters": int(diag["n_filters"]),
            "is_period_doubled": bool(diag["is_period_doubled"]),
            "residual_sigma_mag": round(float(diag["residual_sigma_mag"]), 6),
            "session_mode_requested": str(diag["session_mode_requested"]),
            "used_session_offsets": bool(diag["used_session_offsets"]),
            "top_aliases": list(diag["top_aliases"]),
        },
        "variants": {
            "dashboard": json.loads(base._make_dashboard_figure(diag).to_json()),
            "session_audit": json.loads(base._make_session_audit_figure(diag).to_json()),
            "search_surface": json.loads(base._make_search_surface_figure(diag).to_json()),
        },
    }
    return variant.key, payload


def _build_html(payload: dict[str, object]) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>2025 MM81 Rotation-Period Gallery</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
      :root {{
        --bg: #08101d;
        --panel: rgba(15, 23, 42, 0.84);
        --panel-strong: rgba(17, 24, 39, 0.96);
        --ink: #edf2ff;
        --muted: #9fb2d8;
        --border: rgba(148, 163, 184, 0.18);
        --shadow: 0 20px 60px rgba(2, 6, 23, 0.42);
      }}
      * {{ box-sizing: border-box; }}
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
        grid-template-columns: minmax(280px, 1.15fr) minmax(300px, 0.85fr);
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
      .hero-copy {{ padding: 28px; }}
      .hero-copy h1 {{ margin: 0 0 10px; font-size: 2.05rem; letter-spacing: -0.03em; }}
      .hero-copy p {{ margin: 0 0 12px; color: var(--muted); line-height: 1.55; }}
      .hero-copy .callouts {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        margin-top: 18px;
      }}
      .callout {{
        padding: 14px 16px;
        border-radius: 18px;
        background: rgba(8, 16, 29, 0.55);
        border: 1px solid var(--border);
      }}
      .callout strong {{ display: block; margin-bottom: 6px; font-size: 0.92rem; }}
      .callout span {{ color: var(--muted); font-size: 0.88rem; line-height: 1.45; }}
      .hero-controls {{ padding: 24px; }}
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
      .summary-card .value {{ font-size: 1.02rem; font-weight: 600; }}
      .alias-box {{
        margin-top: 16px;
        padding: 16px 18px;
        border-radius: 18px;
        border: 1px solid var(--border);
        background: rgba(8, 16, 29, 0.55);
      }}
      .alias-box h3 {{
        margin: 0 0 10px;
        font-size: 0.9rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }}
      .alias-list {{
        display: grid;
        gap: 8px;
      }}
      .alias-item {{
        display: flex;
        justify-content: space-between;
        gap: 10px;
        font-size: 0.93rem;
      }}
      .variant-stack {{ display: grid; gap: 18px; }}
      .variant-card {{ padding: 22px 22px 12px; }}
      .variant-header {{
        display: flex;
        justify-content: space-between;
        gap: 20px;
        align-items: baseline;
        margin-bottom: 8px;
      }}
      .variant-header h2 {{ margin: 0; font-size: 1.18rem; letter-spacing: -0.02em; }}
      .variant-header span {{ color: #ffd166; font-size: 0.9rem; }}
      .variant-card p {{ margin: 0 0 14px; color: var(--muted); line-height: 1.5; }}
      .plot {{ min-height: 520px; }}
      .plot.tall {{ min-height: 760px; }}
      @media (max-width: 1100px) {{
        .hero {{ grid-template-columns: 1fr; }}
        .hero-copy .callouts {{ grid-template-columns: 1fr; }}
        .summary-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      }}
      @media (max-width: 720px) {{
        .page {{ padding: 20px 14px 44px; }}
        .summary-grid {{ grid-template-columns: 1fr; }}
      }}
    </style>
  </head>
  <body>
    <div class="page">
      <section class="hero">
        <div class="hero-copy">
          <h1>2025 MM81 Rotation-Period Gallery</h1>
          <p>
            This page uses live MPC mirror photometry for <code>2025 MM81</code>.
            The object is currently represented here by Rubin Observatory station
            <code>X05</code>, with <code>390</code> photometric points across
            <code>7</code> nights and three filters.
          </p>
          <p>
            There is no LCDB reference period for this object in this workflow, so the
            figures are diagnostic rather than comparative. The main questions are:
            what period the solver favors, whether different nights and bands support it,
            and what alias competitors are nearby.
          </p>
          <div class="callouts">
            <div class="callout">
              <strong>All Bands</strong>
              <span>Closest to the real new-object workflow. Best first look.</span>
            </div>
            <div class="callout">
              <strong>Single-Band Checks</strong>
              <span>Use these to see whether the favored period survives without cross-band coupling.</span>
            </div>
            <div class="callout">
              <strong>Search Surface</strong>
              <span>Most important when there is no catalog truth. This is where alias risk is visible.</span>
            </div>
          </div>
        </div>
        <div class="hero-controls">
          <label class="controls-label" for="dataset-select">Dataset Variant</label>
          <select id="dataset-select"></select>
          <div class="summary-grid" id="summary-grid"></div>
          <div class="alias-box">
            <h3>Top Candidate Minima</h3>
            <div class="alias-list" id="alias-list"></div>
          </div>
        </div>
      </section>

      <section class="variant-stack">
        <article class="variant-card">
          <div class="variant-header">
            <h2>Variant 1: Analyst Dashboard</h2>
            <span>Best default view for a new object</span>
          </div>
          <p id="dataset-description-1"></p>
          <div class="plot" id="dashboard-plot"></div>
        </article>

        <article class="variant-card">
          <div class="variant-header">
            <h2>Variant 2: Session Audit</h2>
            <span>Best for night-level consistency</span>
          </div>
          <p>Shows whether separate Rubin nights phase up in a coherent way or whether one night is dominating the fit.</p>
          <div class="plot tall" id="session-audit-plot"></div>
        </article>

        <article class="variant-card">
          <div class="variant-header">
            <h2>Variant 3: Search Surface</h2>
            <span>Best for alias diagnosis</span>
          </div>
          <p>Shows the candidate-period landscape by Fourier order. This is the key plot when there is no external reference period.</p>
          <div class="plot" id="search-surface-plot"></div>
        </article>
      </section>
    </div>

    <script>
      const payload = {payload_json};
      const keys = Object.keys(payload);
      const select = document.getElementById("dataset-select");
      const summaryGrid = document.getElementById("summary-grid");
      const aliasList = document.getElementById("alias-list");
      const description1 = document.getElementById("dataset-description-1");
      const plots = {{
        dashboard: document.getElementById("dashboard-plot"),
        session_audit: document.getElementById("session-audit-plot"),
        search_surface: document.getElementById("search-surface-plot"),
      }};

      const summaryFields = [
        ["Station / Bands", meta => `${{meta.station}} / ${{meta.bands.join(', ')}}`],
        ["Estimated Period", meta => `${{meta.estimated_period_hours.toFixed(6)}} h`],
        ["Frequency", meta => `${{meta.frequency_cycles_per_day.toFixed(6)}} c/d`],
        ["Order", meta => `k=${{meta.fourier_order}}`],
        ["Observations", meta => `${{meta.n_fit_observations}} / ${{meta.n_observations}} fit`],
        ["Sessions", meta => `${{meta.n_sessions}} raw session(s), ${{meta.n_clipped}} clipped`],
        ["Filters", meta => `${{meta.n_filters}} modeled filter(s)`],
        ["Residual Sigma", meta => `${{meta.residual_sigma_mag.toFixed(6)}} mag`],
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

      function renderAliases(meta) {{
        if (!meta.top_aliases || meta.top_aliases.length === 0) {{
          aliasList.innerHTML = `<div class="alias-item"><span>No sampled local minima found</span><span></span></div>`;
          return;
        }}
        aliasList.innerHTML = meta.top_aliases.map((item, idx) => `
          <div class="alias-item">
            <span>#${{idx + 1}}: ${{item.period_hours.toFixed(6)}} h</span>
            <span>sigma=${{item.sigma.toFixed(4)}}</span>
          </div>
        `).join("");
      }}

      function renderVariant(key) {{
        const item = payload[key];
        const meta = item.meta;
        renderSummary(meta);
        renderAliases(meta);
        description1.textContent = meta.description;
        Plotly.react(plots.dashboard, item.variants.dashboard.data, item.variants.dashboard.layout, {{ responsive: true, displaylogo: false }});
        Plotly.react(plots.session_audit, item.variants.session_audit.data, item.variants.session_audit.layout, {{ responsive: true, displaylogo: false }});
        Plotly.react(plots.search_surface, item.variants.search_surface.data, item.variants.search_surface.layout, {{ responsive: true, displaylogo: false }});
      }}

      for (const key of keys) {{
        const option = document.createElement("option");
        option.value = key;
        option.textContent = payload[key].meta.dataset_label;
        select.appendChild(option);
      }}

      select.addEventListener("change", event => renderVariant(event.target.value));
      renderVariant(keys[0]);
    </script>
  </body>
</html>
"""


def main() -> None:
    client = bigquery.Client(project="moeyens-thor-dev", location="us")
    payload: dict[str, object] = {}
    for variant in VARIANTS:
        key, item = _build_variant_payload(client, variant)
        payload[key] = item
        print(f"Prepared {key}", flush=True)

    OUTPUT_PATH.write_text(_build_html(payload), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
