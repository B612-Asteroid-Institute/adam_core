"""
Generate offline rotation-period regression fixtures from MPCQ BigQuery + Horizons.

This script is intended to be run manually by a developer with network access.
The resulting `.npz` files are used by an optional offline pytest suite.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
from google.cloud import bigquery

from adam_core.observations.exposures import Exposures
from adam_core.orbits.query.horizons import query_horizons
from adam_core.photometry.tests.data.fixture_generation import timestamp_from_bq_obstime
from adam_core.time import Timestamp
from adam_core.coordinates.origin import OriginCodes


@dataclass(frozen=True)
class RotationFixtureCase:
    object_id: str
    station_code: str
    band: str
    expected_period_hours: float
    tolerance_fraction: float
    source_title: str
    source_url: str
    frequency_grid_scale: float = 40.0
    max_frequency_cycles_per_day: float = 24.0
    min_rotations_in_span: float = 2.0


CASES: tuple[RotationFixtureCase, ...] = (
    RotationFixtureCase(
        object_id="289",
        station_code="071",
        band="R",
        expected_period_hours=6.902,
        tolerance_fraction=0.01,
        source_title="Lucas et al. 2011 / MPB 38-4, 289 Nenetta",
        source_url="https://mpbulletin.org/issues/MPB_38-4.pdf",
    ),
    RotationFixtureCase(
        object_id="702",
        station_code="071",
        band="R",
        expected_period_hours=16.7072,
        tolerance_fraction=0.20,
        source_title="Colazo et al. 2022 / MPB 49-3, 702 Alauda",
        source_url="https://mpbulletin.org/issues/MPB_49-3.pdf",
    ),
    RotationFixtureCase(
        object_id="1011",
        station_code="071",
        band="R",
        expected_period_hours=5.171,
        tolerance_fraction=0.30,
        source_title="Lupishko et al. 2024 / MPB 51-3, 1011 Laodamia",
        source_url="https://mpbulletin.org/issues/MPB_51-3.pdf",
    ),
)


def _query_rows(
    client: bigquery.Client,
    *,
    dataset_id: str,
    case: RotationFixtureCase,
) -> pa.Table:
    query = f"""
    SELECT
      obsid,
      obstime,
      mag,
      rmsmag
    FROM `{dataset_id}.public_obs_sbn`
    WHERE permid = @permid
      AND TRIM(stn) = @stn
      AND TRIM(band) = @band
      AND obstime IS NOT NULL
      AND mag IS NOT NULL
    ORDER BY obstime ASC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("permid", "STRING", case.object_id),
            bigquery.ScalarQueryParameter("stn", "STRING", case.station_code),
            bigquery.ScalarQueryParameter("band", "STRING", case.band),
        ]
    )
    return client.query(query, job_config=job_config).result().to_arrow(
        create_bqstorage_client=True
    )


def _build_geometry(
    *,
    case: RotationFixtureCase,
    times_utc: Timestamp,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(times_utc)
    exposures = Exposures.from_kwargs(
        id=[f"{case.object_id}_{i}" for i in range(n)],
        start_time=times_utc,
        duration=np.zeros(n, dtype=np.float64),
        filter=[case.band] * n,
        observatory_code=[case.station_code] * n,
    )
    observers = exposures.observers(frame="ecliptic", origin=OriginCodes.SUN)

    last_err: Exception | None = None
    for attempt in range(5):
        try:
            coords = query_horizons(
                object_ids=[case.object_id],
                times=times_utc,
                coordinate_type="cartesian",
                location="@sun",
                id_type="smallbody",
            ).coordinates
            last_err = None
            break
        except Exception as err:  # pragma: no cover - network retries
            last_err = err
            time.sleep(min(8.0, 0.5 * (2**attempt)))
    if last_err is not None:
        raise RuntimeError(
            f"Horizons query failed for object={case.object_id}: {last_err}"
        )

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


def _slug(case: RotationFixtureCase) -> str:
    return (
        f"rotation_period_fixture_{case.object_id}_{case.station_code}_{case.band}"
    )


def _build_fixture(
    client: bigquery.Client,
    *,
    dataset_id: str,
    case: RotationFixtureCase,
    out_dir: Path,
    overwrite: bool,
) -> Path:
    out_path = out_dir / f"{_slug(case)}.npz"
    if out_path.exists() and not overwrite:
        raise RuntimeError(f"Fixture already exists: {out_path}")

    rows = _query_rows(client, dataset_id=dataset_id, case=case)
    if rows.num_rows == 0:
        raise RuntimeError(
            f"No rows found for object={case.object_id} station={case.station_code} band={case.band}"
        )

    times_utc = timestamp_from_bq_obstime(rows.column("obstime"))
    r_au, delta_au, phase_angle_deg = _build_geometry(case=case, times_utc=times_utc)
    mag = np.asarray(rows.column("mag").to_numpy(zero_copy_only=False), dtype=np.float64)
    rmsmag = np.asarray(
        rows.column("rmsmag").to_numpy(zero_copy_only=False), dtype=np.float64
    )

    np.savez_compressed(
        out_path,
        object_id=np.array([case.object_id], dtype=object),
        station=np.array([case.station_code], dtype=object),
        band=np.array([case.band], dtype=object),
        source_title=np.array([case.source_title], dtype=object),
        source_url=np.array([case.source_url], dtype=object),
        expected_period_hours=np.array([case.expected_period_hours], dtype=np.float64),
        tolerance_fraction=np.array([case.tolerance_fraction], dtype=np.float64),
        frequency_grid_scale=np.array([case.frequency_grid_scale], dtype=np.float64),
        max_frequency_cycles_per_day=np.array(
            [case.max_frequency_cycles_per_day], dtype=np.float64
        ),
        min_rotations_in_span=np.array([case.min_rotations_in_span], dtype=np.float64),
        time_iso=np.array(times_utc.to_iso8601().to_pylist(), dtype=object),
        mag_obs=mag,
        rmsmag=rmsmag,
        filter=np.array([case.band] * rows.num_rows, dtype=object),
        r_au=r_au,
        delta_au=delta_au,
        phase_angle_deg=phase_angle_deg,
    )
    return out_path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-id",
        default=os.environ.get("MPCQ_DATASET_ID", "moeyens-thor-dev.mpc_sbn_aurora"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory to write fixtures into (default: this folder).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing fixture files if present.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    client = bigquery.Client()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for case in CASES:
        out_path = _build_fixture(
            client,
            dataset_id=str(args.dataset_id),
            case=case,
            out_dir=out_dir,
            overwrite=bool(args.overwrite),
        )
        print(f"Wrote {out_path.name}")


if __name__ == "__main__":
    main()
