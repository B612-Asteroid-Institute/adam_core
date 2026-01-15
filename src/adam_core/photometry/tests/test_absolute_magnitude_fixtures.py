from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin
from adam_core.observations.detections import PointSourceDetections
from adam_core.observations.exposures import Exposures
from adam_core.observers.observers import Observers
from adam_core.photometry.absolute_magnitude import (
    estimate_absolute_magnitude_v_from_detections,
)
from adam_core.photometry.bandpasses.api import map_to_canonical_filter_bands
from adam_core.photometry.magnitude import predict_magnitudes
from adam_core.time import Timestamp

DATA_DIR = Path(__file__).parent / "data"

MPC_FIXTURES: list[str] = sorted(
    p.name for p in DATA_DIR.glob("mpc_magnitude_fixture_*.npz")
)
if not MPC_FIXTURES:
    MPC_FIXTURES = ["__NO_FIXTURES__"]


def _observers_from_heliocentric_positions(
    station_code: str, times: Timestamp, heliocentric_pos_au: np.ndarray
) -> Observers:
    pos = np.asarray(heliocentric_pos_au, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("heliocentric_pos_au must have shape (N, 3)")
    n = int(pos.shape[0])
    return Observers.from_kwargs(
        code=[station_code] * n,
        coordinates=CartesianCoordinates.from_kwargs(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            vx=np.zeros(n),
            vy=np.zeros(n),
            vz=np.zeros(n),
            time=times,
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"] * n),
        ),
    )


def _canonicalize_exposure_filters(
    exposures: Exposures, *, strict: bool = False
) -> Exposures:
    canonical = map_to_canonical_filter_bands(
        exposures.observatory_code,
        exposures.filter,
        allow_fallback_filters=not strict,
    )
    return exposures.set_column(
        "filter", pa.array(canonical.tolist(), type=pa.large_string())
    )


def _sse(resid: np.ndarray) -> float:
    x = np.asarray(resid, dtype=np.float64)
    return float(np.sum(x * x))


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


def _format_residual_table(rows: list[dict[str, object]], cols: list[str]) -> str:
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


@pytest.mark.parametrize("fixture_name", MPC_FIXTURES)
def test_mpc_absolute_magnitude_from_fixture(
    monkeypatch, fixture_name: str, pytestconfig, capsys: pytest.CaptureFixture[str]
) -> None:
    if fixture_name == "__NO_FIXTURES__":
        pytest.skip("No photometry fixtures found on disk.")

    fixture_path = DATA_DIR / fixture_name
    if not fixture_path.exists():
        pytest.skip(f"Missing fixture {fixture_name}")

    fx = np.load(fixture_path, allow_pickle=True)
    station = str(fx["station"][0])
    H_v_mpc = float(fx["H_v_mpc"][0])
    G_mpc = float(fx["G_mpc"][0])
    time_iso = fx["time_iso"].astype(object).tolist()
    bands = fx["band"].astype(object).tolist()
    mag_obs = np.asarray(fx["mag_obs"], dtype=np.float64)
    obj_pos = np.asarray(fx["object_pos"], dtype=np.float64)
    obs_pos = np.asarray(fx["observer_pos"], dtype=np.float64)

    n = len(mag_obs)
    assert obj_pos.shape == (n, 3)
    assert obs_pos.shape == (n, 3)
    assert len(bands) == n

    times = Timestamp.from_iso8601(time_iso, scale="utc")

    # Build a SMALL exposures table (unique by band) so the estimator must join via exposure_id
    # and expand back to per-detection rows internally.
    uniq_bands = sorted({str(b).strip() for b in bands})
    exposure_ids = [f"e_{b}" for b in uniq_bands]
    exposures = Exposures.from_kwargs(
        id=exposure_ids,
        start_time=Timestamp.from_iso8601(
            [time_iso[0]] * len(exposure_ids), scale="utc"
        ),
        duration=np.zeros(len(exposure_ids), dtype=np.float64),
        filter=uniq_bands,
        observatory_code=[station] * len(exposure_ids),
    )

    # Critical sanity: patch the *class* method so it applies to Exposures objects created
    # inside the estimator via .take()/.set_column(). Use fixture-provided observer_pos.
    def fake_observers(self, *args, **kwargs):  # noqa: ARG001
        if len(self) != n:
            raise AssertionError(f"Expected exposures length {n}, got {len(self)}")
        return _observers_from_heliocentric_positions(station, times, obs_pos)

    monkeypatch.setattr(Exposures, "observers", fake_observers)

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

    detections = PointSourceDetections.from_kwargs(
        id=[f"d{i}" for i in range(n)],
        exposure_id=[f"e_{str(b).strip()}" for b in bands],
        time=times,
        ra=np.zeros(n),
        dec=np.zeros(n),
        mag=mag_obs,
        mag_sigma=None,
    )

    pp_hat = estimate_absolute_magnitude_v_from_detections(
        detections,
        exposures,
        object_coords,
        composition="NEO",
        G=G_mpc,
        strict_band_mapping=False,
        reference_filter="V",
    )
    H_hat = float(pp_hat.H_v[0].as_py())
    assert np.isfinite(H_hat)

    # For evaluation, align + canonicalize exposures to get a valid forward-model run.
    idx = pc.fill_null(pc.index_in(detections.exposure_id, value_set=exposures.id), -1)
    idx_np = np.asarray(idx.to_numpy(zero_copy_only=False), dtype=np.int32)
    if np.any(idx_np < 0):
        raise AssertionError("fixture test exposure_id mapping failed unexpectedly")
    exposures_aligned = exposures.take(pa.array(idx_np, type=pa.int32()))
    exposures_canon = _canonicalize_exposure_filters(exposures_aligned, strict=False)

    # Compare fit quality using a single forward-model call at H=0.
    m0 = predict_magnitudes(
        H=0.0,
        object_coords=object_coords,
        exposures=exposures_canon,
        G=G_mpc,
        reference_filter="V",
        composition="NEO",
    )
    resid_hat = (np.asarray(m0, dtype=np.float64) + H_hat) - mag_obs
    resid_mpc = (np.asarray(m0, dtype=np.float64) + H_v_mpc) - mag_obs

    # The estimator is least-squares optimal in H for fixed geometry, G, and composition.
    assert _sse(resid_hat) <= _sse(resid_mpc) + 1e-10

    # Strong sanity check: with missing per-point sigma, the optimal H is mean(mag - m0).
    H_opt = float(np.mean(mag_obs - np.asarray(m0, dtype=np.float64)))
    assert H_hat == pytest.approx(H_opt, abs=1e-12)

    verbose = bool(pytestconfig.getoption("--photometry-fixtures-verbose")) or bool(
        os.environ.get("ADAM_CORE_PHOTOMETRY_FIXTURE_VERBOSE")
    )
    if verbose:
        canon = np.asarray(
            exposures_canon.filter.to_numpy(zero_copy_only=False), dtype=object
        ).astype(str)
        phase_deg = _phase_angle_deg(obj_pos, obs_pos)

        cols = [
            "H_mpc",
            "H_hat",
            "dH",
            "filter_id",
            "n",
            "phase_deg_span",
            "abs_resid_median_hat",
            "abs_resid_p95_hat",
            "resid_mean_hat",
            "abs_resid_median_mpc",
            "abs_resid_p95_mpc",
            "resid_mean_mpc",
        ]
        rows: list[dict[str, object]] = []
        for k in sorted(set(canon.tolist())):
            m = canon == str(k)
            xs_hat = np.asarray(resid_hat[m], dtype=np.float64)
            xs_mpc = np.asarray(resid_mpc[m], dtype=np.float64)
            ph = np.asarray(phase_deg[m], dtype=np.float64)
            rows.append(
                {
                    "H_mpc": f"{H_v_mpc:.4f}",
                    "H_hat": f"{H_hat:.4f}",
                    "dH": f"{(H_hat - H_v_mpc):+.4f}",
                    "filter_id": str(k),
                    "n": int(xs_hat.size),
                    "phase_deg_span": f"{float(np.ptp(ph)):.2f}" if ph.size else "nan",
                    "abs_resid_median_hat": f"{float(np.median(np.abs(xs_hat))):.6f}",
                    "abs_resid_p95_hat": f"{float(np.quantile(np.abs(xs_hat), 0.95)):.6f}",
                    "resid_mean_hat": f"{float(np.mean(xs_hat)):.6f}",
                    "abs_resid_median_mpc": f"{float(np.median(np.abs(xs_mpc))):.6f}",
                    "abs_resid_p95_mpc": f"{float(np.quantile(np.abs(xs_mpc), 0.95)):.6f}",
                    "resid_mean_mpc": f"{float(np.mean(xs_mpc)):.6f}",
                }
            )
        rows.sort(key=lambda r: str(r["filter_id"]))
        table = _format_residual_table(rows, cols)
        with capsys.disabled():
            print(
                "\n".join(
                    [
                        f"\nAbs-mag fixture summary for {fixture_name} (station={station}):",
                        f"H_mpc={H_v_mpc:.4f}, G_mpc={G_mpc:.4f}, H_hat={H_hat:.4f}, dH={H_hat - H_v_mpc:+.4f}",
                        table,
                        "",
                    ]
                )
            )


"""
NOTE: X05/LSST is covered by the unified MPC-format fixtures (station='X05')
generated via `generate_mpc_magnitude_fixtures.py`; we do not maintain a separate
LSST-only fixture schema/test here.
"""
