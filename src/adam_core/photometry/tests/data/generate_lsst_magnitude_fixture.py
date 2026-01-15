"""
Generate a small, offline photometry regression fixture from real MPCQ BigQuery + Horizons.

This script is intended to be run manually by a developer (it requires network access and ADC).
It writes a `.npz` fixture used by `test_lsst_magnitude_fixture.py`.

Example:
  python -m adam_core.photometry.tests.data.generate_lsst_magnitude_fixture \
    --dataset-id moeyens-thor-dev.mpc_sbn_aurora \
    --views-dataset-id moeyens-thor-dev.mpc_sbn_aurora_views \
    --object-id "2014 QH289" \
    --station X05 \
    --limit 30 \
    --out adam_core/src/adam_core/photometry/tests/data/lsst_magnitude_fixture_2014_QH289.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from mpcq.client import BigQueryMPCClient

from adam_core.observations.exposures import Exposures
from adam_core.orbits.query.horizons import query_horizons
from adam_core.photometry.bandpasses import resolve_filter_ids
from adam_core.photometry.magnitude import predict_magnitudes
from adam_core.photometry.tests.data.fixture_generation import query_jpl_hg


def _normalize_x05_reported_band(band: str) -> str | None:
    """
    Normalize MPC-reported bands for LSST/X05 into the bandpass lookup's reported-band values.

    We accept both:
    - 'g','r','i','u','z' (common)
    - 'Lg','Lr','Li','Lu','Lz' (seen in some LSST submissions)
    """
    b = band.strip()
    if not b:
        return None
    if len(b) == 2 and b[0] == "L":
        b = b[1:]
    # The vendored band map includes both X05|y and X05|Y as pragmatic aliases.
    if b in {"u", "g", "r", "i", "z", "y", "Y"}:
        return b
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--views-dataset-id", required=True)
    ap.add_argument(
        "--object-id", required=True, help="MPC designation (e.g. '2014 QH289')"
    )
    ap.add_argument("--station", default="X05")
    ap.add_argument(
        "--limit", type=int, default=30, help="Max observations to include in fixture"
    )
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    client = BigQueryMPCClient(
        dataset_id=args.dataset_id, views_dataset_id=args.views_dataset_id
    )

    obs = client.query_observations([args.object_id])
    # Filter station and valid fields
    obs = obs.apply_mask(pc.equal(pc.utf8_trim_whitespace(obs.stn), args.station))
    obs = obs.apply_mask(
        pc.and_(
            pc.is_valid(obs.obstime.days),
            pc.and_(pc.is_valid(obs.obstime.nanos), pc.is_valid(obs.band)),
        )
    )
    if len(obs) == 0:
        raise RuntimeError(
            f"No observations found for {args.object_id} at station {args.station}"
        )

    # Map bands -> LSST filters, drop unrecognized
    bands = [str(x).strip() for x in pc.utf8_trim_whitespace(obs.band).to_pylist()]
    normalized_bands: list[str] = []
    keep: list[bool] = []
    for b in bands:
        nb = _normalize_x05_reported_band(b)
        if nb is None:
            keep.append(False)
        else:
            keep.append(True)
            normalized_bands.append(nb)

    obs = obs.apply_mask(pa.array(keep, type=pa.bool_()))
    if len(obs) == 0:
        raise RuntimeError(f"No observations with LSST-like bands for {args.object_id}")

    # Limit deterministically by time
    obs = obs.sort_by(["obstime.days", "obstime.nanos"])
    if len(obs) > args.limit:
        obs = obs.take(pa.array(np.arange(args.limit), type=pa.int64()))
        bands = [str(x).strip() for x in pc.utf8_trim_whitespace(obs.band).to_pylist()]
        normalized_bands = [
            nb for b in bands if (nb := _normalize_x05_reported_band(b)) is not None
        ]

    # Pull H/G from JPL (SBDB), not MPC.
    H_v, G = query_jpl_hg(args.object_id)

    exposures = Exposures.from_kwargs(
        id=obs.obsid,
        start_time=obs.obstime,
        duration=np.zeros(len(obs), dtype=np.float64),
        # Store reported bands in the fixture; convert to canonical filter_id only for prediction.
        filter=pa.array(normalized_bands, type=pa.large_string()),
        observatory_code=pa.array([args.station] * len(obs), type=pa.large_string()),
    )
    times_utc = exposures.midpoint()

    # Geometry from Horizons (heliocentric)
    orbits_at_times = query_horizons(
        object_ids=[args.object_id],
        times=times_utc,
        coordinate_type="cartesian",
        location="@sun",
        id_type="smallbody",
    )
    object_coords = orbits_at_times.coordinates
    observers = exposures.observers()  # heliocentric ecliptic observer positions

    # ---------------------------------------------------------------------
    # Freeze observed mags + compute baseline residual ceilings (but do NOT
    # store per-observation predicted magnitudes).
    # ---------------------------------------------------------------------
    mag_obs = np.asarray(obs.mag.to_numpy(zero_copy_only=False), dtype=np.float64)
    canonical = resolve_filter_ids([args.station] * len(exposures), normalized_bands)
    canonical_np = np.asarray(canonical, dtype=object)
    exposures_canon = exposures.set_column(
        "filter", pa.array(canonical.tolist(), type=pa.large_string())
    )

    pred_conv = predict_magnitudes(
        H=H_v,
        object_coords=object_coords,
        exposures=exposures_canon,
        G=G,
        reference_filter="V",
        composition="NEO",
    )
    resid_conv = np.asarray(pred_conv, dtype=np.float64) - mag_obs

    # Second baseline: mimic the test path that starts from reported bands and resolves.
    resolved = resolve_filter_ids([args.station] * len(exposures), normalized_bands)
    exposures_resolved = exposures.set_column(
        "filter", pa.array(resolved.tolist(), type=pa.large_string())
    )
    pred_bp = predict_magnitudes(
        H=H_v,
        object_coords=object_coords,
        exposures=exposures_resolved,
        G=G,
        reference_filter="V",
        composition="NEO",
    )
    resid_bp = np.asarray(pred_bp, dtype=np.float64) - mag_obs

    def _stats_by_filter(resid: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
        keys = sorted({str(f) for f in canonical_np.tolist()})
        med = np.full(len(keys), np.nan, dtype=np.float64)
        p95 = np.full(len(keys), np.nan, dtype=np.float64)
        for i, k in enumerate(keys):
            xs = resid[canonical_np == k]
            if xs.size == 0:
                continue
            med[i] = float(np.abs(np.median(xs)))
            p95[i] = float(np.quantile(np.abs(xs), 0.95))
        return keys, med, p95

    keys, conv_med_abs, conv_p95_abs = _stats_by_filter(resid_conv)
    _, bp_med_abs, bp_p95_abs = _stats_by_filter(resid_bp)

    # Serialize minimal arrays
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        object_id=np.array([args.object_id], dtype=object),
        station=np.array([args.station], dtype=object),
        H_v=np.array([H_v], dtype=np.float64),
        G=np.array([G], dtype=np.float64),
        time_iso=np.array(times_utc.to_iso8601().to_pylist(), dtype=object),
        obsid=np.array(obs.obsid.to_pylist(), dtype=object),
        band=np.array(
            [str(x).strip() for x in pc.utf8_trim_whitespace(obs.band).to_pylist()],
            dtype=object,
        ),
        # Historical field name: `filters`. We store reported bands (normalized) here.
        filters=np.array(normalized_bands, dtype=object),
        mag_obs=np.asarray(obs.mag.to_numpy(zero_copy_only=False), dtype=np.float64),
        rmsmag=np.asarray(obs.rmsmag.to_numpy(zero_copy_only=False), dtype=np.float64),
        ra=np.asarray(obs.ra.to_numpy(zero_copy_only=False), dtype=np.float64),
        dec=np.asarray(obs.dec.to_numpy(zero_copy_only=False), dtype=np.float64),
        object_pos=np.asarray(object_coords.r, dtype=np.float64),
        observer_pos=np.asarray(observers.coordinates.r, dtype=np.float64),
        # Baseline residual ceilings (absolute) for the *current* implementation.
        # Tests enforce that future changes do not make residuals larger than these.
        baseline_keys=np.array(keys, dtype=object),
        baseline_conv_median_abs=conv_med_abs,
        baseline_conv_p95_abs=conv_p95_abs,
        baseline_bandpass_median_abs=bp_med_abs,
        baseline_bandpass_p95_abs=bp_p95_abs,
    )


if __name__ == "__main__":
    main()
