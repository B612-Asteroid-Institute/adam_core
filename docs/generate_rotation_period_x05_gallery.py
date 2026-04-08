from __future__ import annotations

import astropy.time
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
from google.cloud import bigquery

from adam_core.coordinates.origin import OriginCodes
from adam_core.observations.exposures import Exposures
from adam_core.orbits.query.horizons import query_horizons
from adam_core.photometry.rotation_period_types import RotationPeriodObservations
from adam_core.time import Timestamp

DOCS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DOCS_DIR))
import generate_rotation_period_plotly_gallery as base  # noqa: E402


OUTPUT_PATH = DOCS_DIR / "rotation_period_x05_gallery.html"
DATASET_ID = "moeyens-thor-dev.mpc_sbn_aurora"
SEARCH_ORDERS = (2, 3, 4, 5, 6)


@dataclass(frozen=True)
class X05Case:
    permid: str
    name: str
    band: str
    expected_period_hours: float
    quality_u: int
    label: str
    note: str
    frequency_grid_scale: float = 40.0
    max_frequency_cycles_per_day: float = 24.0
    min_rotations_in_span: float = 2.0


CASES: tuple[X05Case, ...] = (
    X05Case(
        permid="89063",
        name="2001 TV146",
        band="Lr",
        expected_period_hours=4.362,
        quality_u=2,
        label="auto-regime miss",
        note="Under the current auto session policy, the solver prefers a long-period alias here. Useful as a live failure example.",
    ),
    X05Case(
        permid="61461",
        name="2000 QA31",
        band="Lr",
        expected_period_hours=12.2,
        quality_u=2,
        label="long-period challenge",
        note="Longer period over the same six Rubin nights. Recovery is plausible but less tight.",
    ),
    X05Case(
        permid="120968",
        name="1998 VH32",
        band="Lr",
        expected_period_hours=3.12,
        quality_u=2,
        label="clean recovery",
        note="With the current auto session policy, this is the cleanest short-period recovery in the X05 set.",
    ),
)


def _timestamp_from_bq_obstime(obstime: pa.Array | pa.ChunkedArray) -> Timestamp:
    if isinstance(obstime, pa.ChunkedArray):
        obstime = obstime.combine_chunks()
    dt64 = obstime.to_numpy(zero_copy_only=False)
    t = astropy.time.Time(dt64, format="datetime64", scale="utc")
    return Timestamp.from_astropy(t)


def _build_geometry(
    *,
    object_id: str,
    station_code: str,
    band: str,
    times_utc: Timestamp,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(times_utc)
    exposures = Exposures.from_kwargs(
        id=[f"{object_id}_{i}" for i in range(n)],
        start_time=times_utc,
        duration=np.zeros(n, dtype=np.float64),
        filter=[band] * n,
        observatory_code=[station_code] * n,
    )
    observers = exposures.observers(frame="ecliptic", origin=OriginCodes.SUN)
    coords = query_horizons(
        object_ids=[object_id],
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


def _query_case_rows(client: bigquery.Client, case: X05Case) -> pa.Table:
    query = f"""
    SELECT
      obsid,
      obstime,
      mag,
      rmsmag
    FROM `{DATASET_ID}.public_obs_sbn`
    WHERE TRIM(stn) = 'X05'
      AND permid = @permid
      AND TRIM(band) = @band
      AND obstime IS NOT NULL
      AND mag IS NOT NULL
    ORDER BY obstime ASC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("permid", "STRING", case.permid),
            bigquery.ScalarQueryParameter("band", "STRING", case.band),
        ]
    )
    return client.query(query, job_config=job_config).result().to_arrow(
        create_bqstorage_client=True
    )
def _build_case_payload(client: bigquery.Client, case: X05Case) -> tuple[str, dict[str, object]]:
    rows = _query_case_rows(client, case)
    if rows.num_rows == 0:
        raise RuntimeError(f"No X05 rows found for {case.permid} {case.name} ({case.band})")

    times_utc = _timestamp_from_bq_obstime(rows.column("obstime"))
    r_au, delta_au, phase_angle_deg = _build_geometry(
        object_id=case.permid,
        station_code="X05",
        band=case.band,
        times_utc=times_utc,
    )
    time_iso = times_utc.to_iso8601().to_pylist()
    mag = np.asarray(rows.column("mag").to_numpy(zero_copy_only=False), dtype=np.float64)
    mag_sigma = np.asarray(
        rows.column("rmsmag").to_numpy(zero_copy_only=False), dtype=np.float64
    )

    observations = RotationPeriodObservations.from_kwargs(
        time=Timestamp.from_iso8601(time_iso, scale="utc"),
        mag=mag,
        mag_sigma=pa.array(
            mag_sigma,
            mask=~np.isfinite(mag_sigma),
            type=pa.float64(),
        ),
        filter=[case.band] * len(mag),
        session_id=base.build_observatory_night_session_ids(["X05"] * len(mag), times_utc),
        r_au=r_au,
        delta_au=delta_au,
        phase_angle_deg=phase_angle_deg,
    )

    diag = base.build_rotation_period_diagnostics(
        observations,
        search_kwargs={
            "frequency_grid_scale": case.frequency_grid_scale,
            "max_frequency_cycles_per_day": case.max_frequency_cycles_per_day,
            "min_rotations_in_span": case.min_rotations_in_span,
            "session_mode": "auto",
        },
        name=case.name,
        object_number=int(case.permid),
        key=f"{case.permid} {case.name}",
        tier=case.label,
        expected_period_hours=float(case.expected_period_hours),
        lcdb_u=case.quality_u,
        dataset_label=case.label,
        dashboard_plot_title=f"{case.permid} {case.name} Dashboard",
        session_audit_plot_title=f"{case.permid} {case.name} Session Audit",
        search_surface_plot_title=f"{case.permid} {case.name} Search Surface",
    )

    payload = {
        "meta": {
            "object_number": int(case.permid),
            "name": case.name,
            "label": case.label,
            "band": case.band,
            "station": "X05",
            "note": case.note,
            "expected_period_hours": round(float(case.expected_period_hours), 6),
            "fitted_period_hours": round(float(diag["fitted_period_hours"]), 6),
            "relative_error_pct": round(float(diag["relative_error_pct"]), 4),
            "fourier_order": int(diag["fourier_order"]),
            "frequency_cycles_per_day": round(float(diag["frequency_cycles_per_day"]), 6),
            "lcdb_u": int(case.quality_u),
            "n_observations": int(diag["n_observations"]),
            "n_fit_observations": int(diag["n_fit_observations"]),
            "n_clipped": int(diag["n_clipped"]),
            "n_filters": int(diag["n_filters"]),
            "n_sessions": int(diag["n_sessions"]),
            "session_mode_requested": str(diag["session_mode_requested"]),
            "used_session_offsets": bool(diag["used_session_offsets"]),
            "is_period_doubled": bool(diag["is_period_doubled"]),
        },
        "variants": {
            "dashboard": json.loads(base._make_dashboard_figure(diag).to_json()),
            "session_audit": json.loads(base._make_session_audit_figure(diag).to_json()),
            "search_surface": json.loads(base._make_search_surface_figure(diag).to_json()),
        },
    }
    return str(diag["key"]), payload


def _build_html(payload: dict[str, object]) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Rubin X05 Rotation Period Gallery</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
      :root {{
        --bg: #09111f;
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
          radial-gradient(circle at top right, rgba(255, 209, 102, 0.18), transparent 28%),
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
      .hero-copy .recs {{
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
      .rec strong {{ display: block; margin-bottom: 6px; font-size: 0.92rem; }}
      .rec span {{ color: var(--muted); font-size: 0.88rem; line-height: 1.45; }}
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
      .summary-card .value {{ font-size: 1.05rem; font-weight: 600; }}
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
        .hero-copy .recs {{ grid-template-columns: 1fr; }}
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
          <h1>Rubin/LSST `X05` Rotation-Period Gallery</h1>
          <p>
            These are live examples pulled from the MPC mirror for Rubin Observatory station
            code <code>X05</code>. The point here is not just a pretty plot; it is to show
            how the same visualization patterns behave on denser, survey-style data rather
            than the curated PDS fixtures.
          </p>
          <p>
            This page includes one clean recovery, one moderate long-period case, and one
            miss under the current auto-session regime. That is more useful than a hand-picked
            success-only page because it shows what these figures look like when the solver is
            right, borderline, and wrong on live Rubin-reported MPC photometry.
          </p>
          <div class="recs">
            <div class="rec">
              <strong>Dashboard</strong>
              <span>Best overall choice. It shows the time-domain data, folded fit, search behavior, and residuals together.</span>
            </div>
            <div class="rec">
              <strong>Session Audit</strong>
              <span>Best diagnostic view for Rubin night-to-night consistency and whether the phase-folded solution is coherent.</span>
            </div>
            <div class="rec">
              <strong>Search Surface</strong>
              <span>Best expert view for alias families, especially when the solver locks onto the wrong period family.</span>
            </div>
          </div>
        </div>
        <div class="hero-controls">
          <label class="controls-label" for="object-select">Rubin X05 Object</label>
          <select id="object-select"></select>
          <div class="summary-grid" id="summary-grid"></div>
        </div>
      </section>

      <section class="variant-stack">
        <article class="variant-card">
          <div class="variant-header">
            <h2>Variant 1: Analyst Dashboard</h2>
            <span>Best default view</span>
          </div>
          <p>The main review surface. This is the one I would productize first.</p>
          <div class="plot" id="dashboard-plot"></div>
        </article>

        <article class="variant-card">
          <div class="variant-header">
            <h2>Variant 2: Session Audit</h2>
            <span>Best for night-level inspection</span>
          </div>
          <p>This makes it obvious whether separate Rubin nights reinforce the same phased signal or fight each other.</p>
          <div class="plot tall" id="session-audit-plot"></div>
        </article>

        <article class="variant-card">
          <div class="variant-header">
            <h2>Variant 3: Search Surface</h2>
            <span>Best for aliases</span>
          </div>
          <p>This is the clearest view when you want to understand why the chosen period won or why the solver failed.</p>
          <div class="plot" id="search-surface-plot"></div>
        </article>
      </section>
    </div>

    <script>
      const payload = {payload_json};
      const objectKeys = Object.keys(payload).sort((a, b) => parseInt(a, 10) - parseInt(b, 10));
      const select = document.getElementById("object-select");
      const summaryGrid = document.getElementById("summary-grid");
      const plots = {{
        dashboard: document.getElementById("dashboard-plot"),
        session_audit: document.getElementById("session-audit-plot"),
        search_surface: document.getElementById("search-surface-plot"),
      }};

      const summaryFields = [
        ["Status", meta => meta.label],
        ["Station / Band", meta => `${{meta.station}} / ${{meta.band}}`],
        ["LCDB", meta => `${{meta.expected_period_hours.toFixed(6)}} h (U=${{meta.lcdb_u}})`],
        ["Fitted", meta => `${{meta.fitted_period_hours.toFixed(6)}} h`],
        ["Error", meta => `${{meta.relative_error_pct.toFixed(4)}}%`],
        ["Order", meta => `k=${{meta.fourier_order}}`],
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
        Plotly.react(plots.dashboard, item.variants.dashboard.data, item.variants.dashboard.layout, {{ responsive: true, displaylogo: false }});
        Plotly.react(plots.session_audit, item.variants.session_audit.data, item.variants.session_audit.layout, {{ responsive: true, displaylogo: false }});
        Plotly.react(plots.search_surface, item.variants.search_surface.data, item.variants.search_surface.layout, {{ responsive: true, displaylogo: false }});
      }}

      objectKeys.forEach(key => {{
        const option = document.createElement("option");
        option.value = key;
        option.textContent = `${{key}} - ${{payload[key].meta.name}}`;
        select.appendChild(option);
      }});

      select.addEventListener("change", event => renderFigures(event.target.value));
      renderFigures(objectKeys[0]);
    </script>
  </body>
</html>
"""


def main() -> None:
    client = bigquery.Client(project="moeyens-thor-dev", location="us")
    payload: dict[str, object] = {}
    for case in CASES:
        key, item = _build_case_payload(client, case)
        payload[key] = item
        print(f"Prepared {key} {case.name}", flush=True)

    OUTPUT_PATH.write_text(_build_html(payload), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
