from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin
from adam_core.observations.exposures import Exposures
from adam_core.observers.observers import Observers
from adam_core.photometry.bandpasses import resolve_filter_ids as suggested_filters
from adam_core.photometry.magnitude import predict_magnitudes
from adam_core.time import Timestamp

DATA_DIR = Path(__file__).parent / "data"

# MPC/ADES photometric band encodings for LSST commonly appear as:
# - canonical IDs: 'LSST_i', 'LSST_r', ...
# - naked bands: 'i', 'r', ...
# - MPC/ADES-style prefixed bands: 'Li', 'Lr', ...
_LSST_MPC_BAND_TO_FILTER_ID: dict[str, str] = {
    "Lu": "LSST_u",
    "Lg": "LSST_g",
    "Lr": "LSST_r",
    "Li": "LSST_i",
    "Lz": "LSST_z",
    "Ly": "LSST_y",
    "LY": "LSST_y",  # occasional uppercase variant
}


def _x05_canonical_filter_id(raw_band: object) -> str:
    """Map fixture band/filter values to bandpass canonical filter_id (e.g. Li -> LSST_i)."""
    s = str(raw_band).strip()
    if s in _LSST_MPC_BAND_TO_FILTER_ID:
        return _LSST_MPC_BAND_TO_FILTER_ID[s]
    if s.startswith("LSST_"):
        return s
    if len(s) == 1:
        return f"LSST_{s}"
    return s


def _x05_reported_band(raw_band: object) -> str:
    """Map fixture band/filter values to X05 reported band (e.g. Li -> i, LSST_i -> i)."""
    fid = _x05_canonical_filter_id(raw_band)
    if fid.startswith("LSST_"):
        return fid.split("_", 1)[1]
    return str(raw_band).strip()


def _residual_summary(residuals: np.ndarray) -> dict[str, float]:
    res = np.asarray(residuals, dtype=np.float64)
    return {
        "n": float(res.size),
        "mean": float(np.mean(res)),
        "median": float(np.median(res)),
        "std": float(np.std(res)),
        "rms": float(np.sqrt(np.mean(res**2))),
        "mean_abs": float(np.mean(np.abs(res))),
        "median_abs": float(np.median(np.abs(res))),
        "min": float(np.min(res)),
        "max": float(np.max(res)),
    }


def _wrms(residuals: np.ndarray, sigma: np.ndarray) -> float | None:
    res = np.asarray(residuals, dtype=np.float64)
    sig = np.asarray(sigma, dtype=np.float64)
    mask = np.isfinite(sig) & (sig > 0) & np.isfinite(res)
    if not np.any(mask):
        return None
    w = 1.0 / sig[mask] ** 2
    return float(np.sqrt(np.sum(w * res[mask] ** 2) / np.sum(w)))


def _format_residual_table(rows: list[dict[str, object]]) -> str:
    cols = [
        "filter_id",
        "n",
        "phase_deg_span",
        "slope_resid(mag/deg)",
        "pearson_r(resid,phase)",
        "median_abs",
        "p95_abs",
        "mean",
        "std",
        "min",
        "max",
        "baseline_median_abs",
        "baseline_p95_abs",
    ]
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


@pytest.mark.parametrize(
    "fixture_name",
    [
        # Generated via `generate_lsst_magnitude_fixture.py`
        "lsst_magnitude_fixture_2014_QH289.npz",
    ],
)
def test_lsst_magnitude_regression_fixture(
    monkeypatch, fixture_name: str, pytestconfig
) -> None:
    fixture_path = DATA_DIR / fixture_name
    if not fixture_path.exists():
        pytest.skip(
            f"Missing fixture {fixture_name}. Generate it with "
            f"`python -m adam_core.photometry.tests.data.generate_lsst_magnitude_fixture ...`"
        )

    fx = np.load(fixture_path, allow_pickle=True)
    H_v = float(fx["H_v"][0])
    G = float(fx["G"][0])
    time_iso = fx["time_iso"].astype(object).tolist()
    filters = fx["filters"].astype(object).tolist()
    filters = [_x05_canonical_filter_id(f) for f in filters]
    obj_pos = np.asarray(fx["object_pos"], dtype=np.float64)
    obs_pos = np.asarray(fx["observer_pos"], dtype=np.float64)
    mag_obs = np.asarray(fx["mag_obs"], dtype=np.float64)
    baseline_keys = fx["baseline_keys"].astype(object).tolist()
    baseline_median_abs = np.asarray(fx["baseline_conv_median_abs"], dtype=np.float64)
    baseline_p95_abs = np.asarray(fx["baseline_conv_p95_abs"], dtype=np.float64)

    n = len(mag_obs)
    assert obj_pos.shape == (n, 3)
    assert obs_pos.shape == (n, 3)
    assert len(filters) == n

    # Keep UTC here: the magnitude calculation uses only the position vectors and does not depend
    # on the time scale, but using UTC avoids accidental time-scale mismatches between stored
    # `observer_pos` (computed from UTC midpoints) and fixture metadata.
    times = Timestamp.from_iso8601(time_iso, scale="utc")

    exposures = Exposures.from_kwargs(
        id=[f"e{i}" for i in range(n)],
        start_time=times,
        duration=np.zeros(n, dtype=np.float64),
        filter=filters,
        observatory_code=["X05"] * n,
    )

    observers = Observers.from_kwargs(
        code=["X05"] * n,
        coordinates=CartesianCoordinates.from_kwargs(
            x=obs_pos[:, 0],
            y=obs_pos[:, 1],
            z=obs_pos[:, 2],
            vx=np.zeros(n),
            vy=np.zeros(n),
            vz=np.zeros(n),
            time=times,
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"] * n),
        ),
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

    out = predict_magnitudes(
        H=H_v,
        object_coords=object_coords,
        exposures=exposures,
        G=G,
        reference_filter="V",
        composition="NEO",
    )

    assert out.shape == mag_obs.shape
    assert np.all(np.isfinite(out))

    resid = np.asarray(out, dtype=np.float64) - mag_obs
    phase_deg = _phase_angle_deg(obj_pos, obs_pos)
    # Pass if residuals are <= the baseline ceilings stored in the fixture.
    # (Improvements pass; regressions fail.)
    eps = 1e-9
    # This codebase's predictor is bandpass-based; use the bandpass ceilings.
    baseline_median_abs = np.asarray(
        fx["baseline_bandpass_median_abs"], dtype=np.float64
    )
    baseline_p95_abs = np.asarray(fx["baseline_bandpass_p95_abs"], dtype=np.float64)

    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for k, med0, p950 in zip(baseline_keys, baseline_median_abs, baseline_p95_abs):
        xs = resid[np.asarray(filters, dtype=object) == str(k)]
        ph = phase_deg[np.asarray(filters, dtype=object) == str(k)]
        if xs.size == 0:
            continue
        xs_f = np.asarray(xs, dtype=np.float64)
        ph_f = np.asarray(ph, dtype=np.float64)
        med_abs = float(np.abs(np.median(xs_f)))
        p95_abs = float(np.quantile(np.abs(xs_f), 0.95))
        rows.append(
            {
                "filter_id": str(k),
                "n": int(xs_f.size),
                "phase_deg_span": f"{float(np.ptp(ph_f)):.2f}" if ph_f.size else "nan",
                "slope_resid(mag/deg)": f"{_slope_resid_vs_phase(xs_f, ph_f):+.4f}",
                "pearson_r(resid,phase)": f"{_pearson_r(xs_f, ph_f):+.3f}",
                "median_abs": f"{med_abs:.6f}",
                "p95_abs": f"{p95_abs:.6f}",
                "mean": f"{float(np.mean(xs_f)):.6f}",
                "std": f"{float(np.std(xs_f)):.6f}",
                "min": f"{float(np.min(xs_f)):.6f}",
                "max": f"{float(np.max(xs_f)):.6f}",
                "baseline_median_abs": f"{float(med0):.6f}",
                "baseline_p95_abs": f"{float(p950):.6f}",
            }
        )
        if med_abs > float(med0) + eps:
            failures.append(
                f"{k}: median_abs {med_abs:.6f} > baseline {float(med0):.6f}"
            )
        if p95_abs > float(p950) + eps:
            failures.append(f"{k}: p95_abs {p95_abs:.6f} > baseline {float(p950):.6f}")

    rows.sort(key=lambda r: str(r["filter_id"]))
    table = _format_residual_table(rows)
    verbose = bool(pytestconfig.getoption("--photometry-fixtures-verbose")) or bool(
        os.environ.get("ADAM_CORE_PHOTOMETRY_FIXTURE_VERBOSE")
    )
    if verbose:
        print(f"\nResidual summary for {fixture_name} (station=X05):\n{table}\n")

    if failures:
        msg = "\n".join(
            [
                f"Photometry regression fixture residuals exceeded baseline: {fixture_name}",
                *failures,
                "",
                table,
            ]
        )
        raise AssertionError(msg)


@pytest.mark.parametrize(
    "fixture_name",
    [
        "lsst_magnitude_fixture_2014_QH289.npz",
    ],
)
def test_lsst_magnitude_regression_fixture_bandpass(
    monkeypatch, fixture_name: str, pytestconfig
) -> None:
    """
    Same regression fixture, but evaluated through the bandpass-based predictor.

    This test exists to enable direct, apples-to-apples comparison on identical geometry
    and exposure metadata.
    """
    fixture_path = DATA_DIR / fixture_name
    if not fixture_path.exists():
        pytest.skip(
            f"Missing fixture {fixture_name}. Generate it with "
            f"`python -m adam_core.photometry.tests.data.generate_lsst_magnitude_fixture ...`"
        )

    fx = np.load(fixture_path, allow_pickle=True)
    H_v = float(fx["H_v"][0])
    G = float(fx["G"][0])
    time_iso = fx["time_iso"].astype(object).tolist()
    filters = fx["filters"].astype(object).tolist()
    filters = [_x05_canonical_filter_id(f) for f in filters]
    obj_pos = np.asarray(fx["object_pos"], dtype=np.float64)
    obs_pos = np.asarray(fx["observer_pos"], dtype=np.float64)
    mag_obs = np.asarray(fx["mag_obs"], dtype=np.float64)
    baseline_keys = fx["baseline_keys"].astype(object).tolist()
    baseline_median_abs = np.asarray(
        fx["baseline_bandpass_median_abs"], dtype=np.float64
    )
    baseline_p95_abs = np.asarray(fx["baseline_bandpass_p95_abs"], dtype=np.float64)

    n = len(mag_obs)
    assert obj_pos.shape == (n, 3)
    assert obs_pos.shape == (n, 3)
    assert len(filters) == n

    times = Timestamp.from_iso8601(time_iso, scale="utc")

    # Fixture commonly uses reported bands ('i', 'r', ...). Convert those to canonical
    # bandpass `filter_id` values (e.g. 'LSST_i') using the repo's resolver.
    reported_bands = [_x05_reported_band(f) for f in filters]
    exposures = Exposures.from_kwargs(
        id=[f"e{i}" for i in range(n)],
        start_time=times,
        duration=np.zeros(n, dtype=np.float64),
        filter=reported_bands,
        observatory_code=["X05"] * n,
    )
    canonical_filter_ids = suggested_filters(
        exposures.observatory_code, exposures.filter
    )
    exposures = exposures.set_column(
        "filter", pa.array(canonical_filter_ids, type=pa.large_string())
    )

    observers = Observers.from_kwargs(
        code=["X05"] * n,
        coordinates=CartesianCoordinates.from_kwargs(
            x=obs_pos[:, 0],
            y=obs_pos[:, 1],
            z=obs_pos[:, 2],
            vx=np.zeros(n),
            vy=np.zeros(n),
            vz=np.zeros(n),
            time=times,
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"] * n),
        ),
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

    # Explicit composition required by API. This fixture is for Rubin/LSST,
    # so use a realistic "NEO" mixture by default.
    out = predict_magnitudes(
        H=H_v,
        object_coords=object_coords,
        exposures=exposures,
        G=G,
        reference_filter="V",
        composition="NEO",
    )

    assert out.shape == mag_obs.shape
    assert np.all(np.isfinite(out))

    resid = np.asarray(out, dtype=np.float64) - mag_obs
    phase_deg = _phase_angle_deg(obj_pos, obs_pos)
    eps = 1e-9

    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for k, med0, p950 in zip(baseline_keys, baseline_median_abs, baseline_p95_abs):
        xs = resid[np.asarray(filters, dtype=object) == str(k)]
        ph = phase_deg[np.asarray(filters, dtype=object) == str(k)]
        if xs.size == 0:
            continue
        xs_f = np.asarray(xs, dtype=np.float64)
        ph_f = np.asarray(ph, dtype=np.float64)
        med_abs = float(np.abs(np.median(xs_f)))
        p95_abs = float(np.quantile(np.abs(xs_f), 0.95))
        rows.append(
            {
                "filter_id": str(k),
                "n": int(xs_f.size),
                "phase_deg_span": f"{float(np.ptp(ph_f)):.2f}" if ph_f.size else "nan",
                "slope_resid(mag/deg)": f"{_slope_resid_vs_phase(xs_f, ph_f):+.4f}",
                "pearson_r(resid,phase)": f"{_pearson_r(xs_f, ph_f):+.3f}",
                "median_abs": f"{med_abs:.6f}",
                "p95_abs": f"{p95_abs:.6f}",
                "mean": f"{float(np.mean(xs_f)):.6f}",
                "std": f"{float(np.std(xs_f)):.6f}",
                "min": f"{float(np.min(xs_f)):.6f}",
                "max": f"{float(np.max(xs_f)):.6f}",
                "baseline_median_abs": f"{float(med0):.6f}",
                "baseline_p95_abs": f"{float(p950):.6f}",
            }
        )
        if med_abs > float(med0) + eps:
            failures.append(
                f"{k}: median_abs {med_abs:.6f} > baseline {float(med0):.6f}"
            )
        if p95_abs > float(p950) + eps:
            failures.append(f"{k}: p95_abs {p95_abs:.6f} > baseline {float(p950):.6f}")

    rows.sort(key=lambda r: str(r["filter_id"]))
    table = _format_residual_table(rows)
    verbose = bool(pytestconfig.getoption("--photometry-fixtures-verbose")) or bool(
        os.environ.get("ADAM_CORE_PHOTOMETRY_FIXTURE_VERBOSE")
    )
    if verbose:
        print(f"\nResidual summary for {fixture_name} (station=X05):\n{table}\n")

    if failures:
        msg = "\n".join(
            [
                f"Photometry regression fixture residuals exceeded baseline: {fixture_name}",
                *failures,
                "",
                table,
            ]
        )
        raise AssertionError(msg)
