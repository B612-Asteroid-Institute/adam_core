"""
Photometry regression tests using observations and orbits from the MPC and JPL Horizons databases.

Overview
--------
These tests load frozen "fixtures" (.npz files) containing observed magnitudes, observatory
metadata, and precise heliocentric geometry for specific objects. For each fixture, they:

1. Predict magnitudes using the `magnitude` model.
2. Calculate residuals (predicted - observed) using absolute magnitude (H) values from:
   - MPC (stored in `H_v_mpc`)
   - JPL SBDB (stored in `H_v_jpl`)
3. Validate that current residuals do not exceed the baseline residual ceilings (median absolute
   and 95th percentile absolute) stored within the fixture.

Data Sources
------------
- Orbits: JPL Horizons
- Observations: MPC database (pulled 01/13/2026)

Benchmark Results
-----------------
A larger benchmark of 1000 observations spanning at least 10 objects (where possible) gave the
following average residuals for each filter:

filter_id |    n | med_abs_mpc | p95_abs_mpc | mean_mpc | med_abs_jpl | p95_abs_jpl | mean_jpl
----------+------+-------------+-------------+----------+-------------+-------------+---------
ATLAS_c   | 1000 |      0.1494 |      0.5156 |   0.0155 |      0.1469 |      0.5193 |   0.0162
ATLAS_o   | 2000 |      0.2687 |      0.5828 |   0.2408 |      0.2704 |      0.5822 |   0.2440
DECam_VR  |  490 |      0.2016 |     80.3238 |   8.0554 |      0.1761 |      0.5677 |   0.0107
DECam_Y   |   24 |      0.4604 |      0.7152 |   0.3850 |      0.4614 |      0.7172 |   0.3866
DECam_g   | 1000 |      0.2571 |      0.6291 |  -0.2106 |      0.2562 |      0.6282 |  -0.2108
DECam_i   | 1000 |      0.2706 |      0.6542 |   0.2299 |      0.2659 |      0.6472 |   0.2261
DECam_r   | 1000 |      0.1782 |      0.7725 |   0.1441 |      0.1835 |      0.7730 |   0.1464
DECam_u   |   13 |      1.3321 |      1.4790 |  -1.3035 |      1.3411 |      1.4880 |  -1.3125
DECam_z   | 1000 |      0.3547 |      0.7997 |   0.2579 |      0.3507 |      0.7998 |   0.2519
ZTF_g     | 1000 |      0.2275 |      0.5885 |  -0.2308 |      0.2294 |      0.5905 |  -0.2318
ZTF_i     | 1000 |      0.2520 |      0.5271 |   0.2529 |      0.2487 |      0.5267 |   0.2510
ZTF_r     | 1000 |      0.1928 |      0.5008 |   0.1622 |      0.1896 |      0.5028 |   0.1634

Most residuals are on the order of 0.2-0.3 mag, with the exception of DECam_u and DECam_Y for which
there are very few observations.

Note that some objects in the MPC database have their H values incorrectly set to 99.99, causing unusually
high residuals for those objects.

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin
from adam_core.observations.exposures import Exposures
from adam_core.photometry.bandpasses import resolve_filter_ids
from adam_core.photometry.magnitude import predict_magnitudes
from adam_core.photometry.tests.data.fixture_generation import (
    observers_from_heliocentric_positions,
)
from adam_core.time import Timestamp

DATA_DIR = Path(__file__).parent / "data"

# Filled in once fixtures are generated (see `generate_mpc_magnitude_fixtures.py`).
# For multi-station benchmarking, auto-discover fixtures on disk.
FIXTURES: list[str] = sorted(
    p.name for p in DATA_DIR.glob("mpc_magnitude_fixture_*.npz")
)
if not FIXTURES:
    FIXTURES = ["__NO_FIXTURES__"]


def _format_residual_table(rows: list[dict[str, object]], cols: list[str]) -> str:
    # Compute widths
    widths: dict[str, int] = {c: len(c) for c in cols}
    for r in rows:
        for c in cols:
            widths[c] = max(widths[c], len(str(r.get(c, ""))))

    def fmt_row(r: dict[str, object]) -> str:
        parts = [str(r.get(c, "")).rjust(widths[c]) for c in cols]
        return "  ".join(parts)

    header = "  ".join([c.rjust(widths[c]) for c in cols])
    sep = "  ".join(["-" * widths[c] for c in cols])
    body = "\n".join(fmt_row(r) for r in rows)
    return "\n".join([header, sep, body])


def _phase_angle_deg(object_pos: np.ndarray, observer_pos: np.ndarray) -> np.ndarray:
    """
    Phase angle (Sun-object-observer) in degrees, matching the H-G geometry used by the model.

    Inputs are heliocentric vectors in AU with shape (N, 3).
    """
    obj = np.asarray(object_pos, dtype=np.float64)
    obs = np.asarray(observer_pos, dtype=np.float64)
    if obj.ndim != 2 or obs.ndim != 2 or obj.shape != obs.shape or obj.shape[1] != 3:
        raise ValueError("object_pos and observer_pos must both have shape (N, 3)")

    r = np.sqrt(np.sum(obj * obj, axis=1))
    delta_vec = obj - obs
    delta = np.sqrt(np.sum(delta_vec * delta_vec, axis=1))
    obs_sun = np.sqrt(np.sum(obs * obs, axis=1))
    numer = r**2 + delta**2 - obs_sun**2
    denom = 2.0 * r * delta
    cos_phase = np.clip(numer / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cos_phase))


def _slope_abs_resid_vs_phase(resid: np.ndarray, phase_deg: np.ndarray) -> float:
    x = np.asarray(phase_deg, dtype=np.float64)
    y = np.asarray(np.abs(resid), dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(m)) < 2:
        return float("nan")
    x = x[m]
    y = y[m]
    x0 = x - float(np.mean(x))
    denom = float(np.sum(x0 * x0))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(x0 * (y - float(np.mean(y)))) / denom)


def _slope_resid_vs_phase(resid: np.ndarray, phase_deg: np.ndarray) -> float:
    x = np.asarray(phase_deg, dtype=np.float64)
    y = np.asarray(resid, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(m)) < 3:
        return float("nan")
    x = x[m]
    y = y[m]
    if float(np.ptp(x)) < 1e-6:
        return float("nan")
    x0 = x - float(np.mean(x))
    denom = float(np.sum(x0 * x0))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(x0 * (y - float(np.mean(y)))) / denom)


def _pearson_r(resid: np.ndarray, phase_deg: np.ndarray) -> float:
    x = np.asarray(phase_deg, dtype=np.float64)
    y = np.asarray(resid, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(m)) < 3:
        return float("nan")
    x = x[m]
    y = y[m]
    if float(np.ptp(x)) < 1e-6:
        return float("nan")
    x0 = x - float(np.mean(x))
    y0 = y - float(np.mean(y))
    sx = float(np.sqrt(np.sum(x0 * x0)))
    sy = float(np.sqrt(np.sum(y0 * y0)))
    if sx <= 0.0 or sy <= 0.0:
        return float("nan")
    return float(np.sum(x0 * y0) / (sx * sy))


@pytest.mark.parametrize("fixture_name", FIXTURES)
def test_mpc_magnitude_regression_fixture(
    monkeypatch, fixture_name: str, pytestconfig
) -> None:
    if fixture_name == "__NO_FIXTURES__":
        pytest.skip("No MPC photometry fixtures found on disk.")

    fixture_path = DATA_DIR / fixture_name
    if not fixture_path.exists():
        pytest.skip(
            f"Missing fixture {fixture_name}. Generate it with "
            f"`python -m adam_core.photometry.tests.data.generate_mpc_magnitude_fixtures --overwrite`"
        )

    fx = np.load(fixture_path, allow_pickle=True)
    if "H_v_mpc" not in fx.files or "H_v_jpl" not in fx.files:
        pytest.skip(f"Legacy fixture missing dual-H fields: {fixture_name}")
    h_source = (
        str(pytestconfig.getoption("--photometry-fixtures-h-source")).strip().lower()
    )
    if h_source not in {"both", "mpc"}:
        raise ValueError(f"Invalid --photometry-fixtures-h-source: {h_source}")

    station = str(fx["station"][0])
    H_v_mpc = float(fx["H_v_mpc"][0])
    G_mpc = float(fx["G_mpc"][0])
    H_v_jpl = float(fx["H_v_jpl"][0])
    G_jpl = float(fx["G_jpl"][0])
    if not (np.isfinite(H_v_mpc) and np.isfinite(H_v_jpl)):
        pytest.skip(
            f"Missing H in MPC or JPL for fixture object {fx['object_id'][0]}; skipping."
        )
    time_iso = fx["time_iso"].astype(object).tolist()
    bands = fx["band"].astype(object).tolist()
    mag_obs = np.asarray(fx["mag_obs"], dtype=np.float64)
    obj_pos = np.asarray(fx["object_pos"], dtype=np.float64)
    obs_pos = np.asarray(fx["observer_pos"], dtype=np.float64)

    baseline_keys = fx["baseline_keys"].astype(object).tolist()
    baseline_mpc_median_abs = np.asarray(
        fx["baseline_mpc_median_abs"], dtype=np.float64
    )
    baseline_mpc_p95_abs = np.asarray(fx["baseline_mpc_p95_abs"], dtype=np.float64)
    baseline_jpl_median_abs = np.asarray(
        fx["baseline_jpl_median_abs"], dtype=np.float64
    )
    baseline_jpl_p95_abs = np.asarray(fx["baseline_jpl_p95_abs"], dtype=np.float64)

    n = len(mag_obs)
    assert obj_pos.shape == (n, 3)
    assert obs_pos.shape == (n, 3)
    assert len(bands) == n

    times = Timestamp.from_iso8601(time_iso, scale="utc")

    exposures = Exposures.from_kwargs(
        id=[f"e{i}" for i in range(n)],
        start_time=times,
        duration=np.zeros(n, dtype=np.float64),
        filter=bands,
        observatory_code=[station] * n,
    )

    canonical_filter_ids = resolve_filter_ids(
        exposures.observatory_code, exposures.filter
    )
    exposures = exposures.set_column(
        "filter", pa.array(canonical_filter_ids.tolist(), type=pa.large_string())
    )

    observers = observers_from_heliocentric_positions(
        station_code=station,
        times_utc=times,
        heliocentric_pos_au=obs_pos,
    )
    monkeypatch.setattr(exposures, "observers", lambda *args, **kwargs: observers)

    object_coords = CartesianCoordinates.from_kwargs(
        x=obj_pos[:, 0],
        y=obj_pos[:, 1],
        z=obj_pos[:, 2],
        vx=np.zeros(n),
        vy=np.zeros(n),
        vz=np.zeros(n),
        time=times,
        frame="ecliptic",
        origin=Origin.from_kwargs(code=["SUN"] * n),
    )

    out_mpc = predict_magnitudes(
        H=H_v_mpc,
        object_coords=object_coords,
        exposures=exposures,
        G=G_mpc,
        reference_filter="V",
        composition="NEO",
    )

    resid_mpc = np.asarray(out_mpc, dtype=np.float64) - mag_obs
    resid_jpl: np.ndarray | None
    if h_source == "both":
        out_jpl = predict_magnitudes(
            H=H_v_jpl,
            object_coords=object_coords,
            exposures=exposures,
            G=G_jpl,
            reference_filter="V",
            composition="NEO",
        )
        resid_jpl = np.asarray(out_jpl, dtype=np.float64) - mag_obs
    else:
        resid_jpl = None
    canon = np.asarray(canonical_filter_ids, dtype=object)
    phase_deg = _phase_angle_deg(obj_pos, obs_pos)

    rows: list[dict[str, object]] = []
    for i, k in enumerate(baseline_keys):
        m0_mpc = float(baseline_mpc_median_abs[i])
        p0_mpc = float(baseline_mpc_p95_abs[i])
        m0_jpl = float(baseline_jpl_median_abs[i])
        p0_jpl = float(baseline_jpl_p95_abs[i])
        xs_mpc = resid_mpc[canon == str(k)]
        xs_jpl = (
            resid_jpl[canon == str(k)]
            if resid_jpl is not None
            else np.asarray([], dtype=np.float64)
        )
        ph = phase_deg[canon == str(k)]
        if xs_mpc.size == 0 and (resid_jpl is None or xs_jpl.size == 0):
            continue

        xs_mpc_f = np.asarray(xs_mpc, dtype=np.float64)
        xs_jpl_f = np.asarray(xs_jpl, dtype=np.float64)
        ph_f = np.asarray(ph, dtype=np.float64)

        med_abs_mpc = (
            float(np.abs(np.median(xs_mpc_f))) if xs_mpc_f.size else float("nan")
        )
        p95_abs_mpc = (
            float(np.quantile(np.abs(xs_mpc_f), 0.95))
            if xs_mpc_f.size
            else float("nan")
        )
        mean_mpc = float(np.mean(xs_mpc_f)) if xs_mpc_f.size else float("nan")

        med_abs_jpl = (
            float(np.abs(np.median(xs_jpl_f))) if xs_jpl_f.size else float("nan")
        )
        p95_abs_jpl = (
            float(np.quantile(np.abs(xs_jpl_f), 0.95))
            if xs_jpl_f.size
            else float("nan")
        )
        mean_jpl = float(np.mean(xs_jpl_f)) if xs_jpl_f.size else float("nan")

        d_med = med_abs_jpl - med_abs_mpc
        d_p95 = p95_abs_jpl - p95_abs_mpc

        row: dict[str, object] = {
            "filter_id": str(k),
            "n": int(max(xs_mpc_f.size, xs_jpl_f.size)),
            "phase_deg_span": f"{float(np.ptp(ph_f)):.2f}" if ph_f.size else "nan",
            "slope_resid_mpc(mag/deg)": (
                f"{_slope_resid_vs_phase(xs_mpc_f, ph_f):+.4f}"
                if xs_mpc_f.size
                else "nan"
            ),
            "pearson_r_mpc(resid,phase)": (
                f"{_pearson_r(xs_mpc_f, ph_f):+.3f}" if xs_mpc_f.size else "nan"
            ),
            "median_abs_mpc": f"{med_abs_mpc:.6f} (base {m0_mpc:.6f})",
            "p95_abs_mpc": f"{p95_abs_mpc:.6f} (base {p0_mpc:.6f})",
            "mean_mpc": f"{mean_mpc:.6f}",
        }
        if resid_jpl is not None:
            row.update(
                {
                    "slope_resid_jpl(mag/deg)": (
                        f"{_slope_resid_vs_phase(xs_jpl_f, ph_f):+.4f}"
                        if xs_jpl_f.size
                        else "nan"
                    ),
                    "pearson_r_jpl(resid,phase)": (
                        f"{_pearson_r(xs_jpl_f, ph_f):+.3f}" if xs_jpl_f.size else "nan"
                    ),
                    "median_abs_jpl": f"{med_abs_jpl:.6f} (base {m0_jpl:.6f})",
                    "p95_abs_jpl": f"{p95_abs_jpl:.6f} (base {p0_jpl:.6f})",
                    "mean_jpl": f"{mean_jpl:.6f}",
                    "delta_median_abs(jpl-mpc)": f"{d_med:+.6f}",
                    "delta_p95_abs(jpl-mpc)": f"{d_p95:+.6f}",
                }
            )
        rows.append(row)

    rows.sort(key=lambda r: str(r["filter_id"]))
    if resid_jpl is None:
        cols = [
            "filter_id",
            "n",
            "phase_deg_span",
            "slope_resid_mpc(mag/deg)",
            "pearson_r_mpc(resid,phase)",
            "median_abs_mpc",
            "p95_abs_mpc",
            "mean_mpc",
        ]
    else:
        cols = [
            "filter_id",
            "n",
            "phase_deg_span",
            "slope_resid_mpc(mag/deg)",
            "pearson_r_mpc(resid,phase)",
            "median_abs_mpc",
            "p95_abs_mpc",
            "mean_mpc",
            "slope_resid_jpl(mag/deg)",
            "pearson_r_jpl(resid,phase)",
            "median_abs_jpl",
            "p95_abs_jpl",
            "mean_jpl",
            "delta_median_abs(jpl-mpc)",
            "delta_p95_abs(jpl-mpc)",
        ]
    table = _format_residual_table(rows, cols)

    # Always print the table so runs can be diffed against previous baselines without
    # failing the test due to residual changes.
    if resid_jpl is None:
        print(
            f"\nResidual summary for {fixture_name} (station={station}, object={fx['object_id'][0]}): "
            f"H_mpc={H_v_mpc:.3f}\n{table}\n"
        )
    else:
        print(
            f"\nResidual summary for {fixture_name} (station={station}, object={fx['object_id'][0]}): "
            f"H_mpc={H_v_mpc:.3f}, H_jpl={H_v_jpl:.3f}\n{table}\n"
        )
