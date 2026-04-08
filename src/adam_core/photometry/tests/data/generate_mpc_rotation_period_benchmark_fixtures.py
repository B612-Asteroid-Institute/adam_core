"""
Generate frozen real-data benchmark fixtures for rotation-period timing.

This is intended for manual developer use with network access. The generated
`.npz` files are consumed by the benchmark suite so we can compare search
strategies on fixed real-world photometry without paying live query costs.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import astropy.time
import numpy as np
import pyarrow as pa
from google.cloud import bigquery

from adam_core.coordinates.origin import OriginCodes
from adam_core.observations.exposures import Exposures
from adam_core.observers.utils import calculate_observing_night
from adam_core.orbits.query.horizons import query_horizons
from adam_core.photometry.rotation_period_fourier import estimate_rotation_period
from adam_core.photometry.rotation_period_types import RotationPeriodObservations
from adam_core.time import Timestamp


@dataclass(frozen=True)
class BenchmarkFixtureCase:
    object_id: str
    station_code: str
    bands: tuple[str, ...]
    frequency_grid_scale: float
    max_frequency_cycles_per_day: float
    min_rotations_in_span: float
    session_mode: str

    @property
    def slug(self) -> str:
        band_slug = "-".join(self.bands)
        return (
            f"rotation_period_benchmark_fixture_"
            f"{self.object_id.replace(' ', '_')}_{self.station_code}_{band_slug}"
        )


CASES: tuple[BenchmarkFixtureCase, ...] = (
    BenchmarkFixtureCase(
        object_id="2025 MM81",
        station_code="X05",
        bands=("g", "r", "i"),
        frequency_grid_scale=40.0,
        max_frequency_cycles_per_day=24.0,
        min_rotations_in_span=2.0,
        session_mode="auto",
    ),
)


def _timestamp_from_bq_obstime(obstime: pa.Array | pa.ChunkedArray) -> Timestamp:
    if isinstance(obstime, pa.ChunkedArray):
        obstime = obstime.combine_chunks()
    dt64 = obstime.to_numpy(zero_copy_only=False)
    t = astropy.time.Time(dt64, format="datetime64", scale="utc")
    return Timestamp.from_astropy(t)


def _query_rows(
    client: bigquery.Client,
    *,
    dataset_id: str,
    case: BenchmarkFixtureCase,
) -> pa.Table:
    bands_sql = ", ".join(f"'{band}'" for band in case.bands)
    query = f"""
    SELECT
      obsid,
      obstime,
      mag,
      rmsmag,
      TRIM(band) AS band,
      TRIM(stn) AS stn
    FROM `{dataset_id}.public_obs_sbn`
    WHERE provid = @provid
      AND TRIM(stn) = @stn
      AND TRIM(band) IN ({bands_sql})
      AND obstime IS NOT NULL
      AND mag IS NOT NULL
    ORDER BY obstime ASC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("provid", "STRING", case.object_id),
            bigquery.ScalarQueryParameter("stn", "STRING", case.station_code),
        ]
    )
    return client.query(query, job_config=job_config).result().to_arrow(
        create_bqstorage_client=True
    )


def _build_geometry(
    *,
    case: BenchmarkFixtureCase,
    times_utc: Timestamp,
    filters: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(times_utc)
    exposures = Exposures.from_kwargs(
        id=[f"{case.object_id}_{i}" for i in range(n)],
        start_time=times_utc,
        duration=np.zeros(n, dtype=np.float64),
        filter=filters,
        observatory_code=[case.station_code] * n,
    )
    observers = exposures.observers(frame="ecliptic", origin=OriginCodes.SUN)
    coords = query_horizons(
        object_ids=[case.object_id],
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


def _session_ids(codes: list[str], times_utc: Timestamp) -> list[str]:
    nights = calculate_observing_night(pa.array(codes, type=pa.string()), times_utc)
    night_values = np.asarray(nights.to_numpy(zero_copy_only=False), dtype=np.int64)
    return [f"{code}:{int(night)}" for code, night in zip(codes, night_values, strict=True)]


def _build_observations(
    rows: pa.Table,
    case: BenchmarkFixtureCase,
) -> RotationPeriodObservations:
    times_utc = _timestamp_from_bq_obstime(rows.column("obstime"))
    bands = np.asarray(rows.column("band").to_numpy(zero_copy_only=False), dtype=object)
    mag_sigma = np.asarray(
        rows.column("rmsmag").to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    r_au, delta_au, phase_angle_deg = _build_geometry(
        case=case,
        times_utc=times_utc,
        filters=bands.tolist(),
    )
    codes = [case.station_code] * rows.num_rows
    return RotationPeriodObservations.from_kwargs(
        time=times_utc,
        mag=np.asarray(rows.column("mag").to_numpy(zero_copy_only=False), dtype=np.float64),
        mag_sigma=pa.array(
            mag_sigma,
            mask=~np.isfinite(mag_sigma),
            type=pa.float64(),
        ),
        filter=bands.tolist(),
        session_id=_session_ids(codes, times_utc),
        r_au=r_au,
        delta_au=delta_au,
        phase_angle_deg=phase_angle_deg,
    )


def _write_fixture(
    client: bigquery.Client,
    *,
    dataset_id: str,
    case: BenchmarkFixtureCase,
    out_dir: Path,
    overwrite: bool,
) -> Path:
    out_path = out_dir / f"{case.slug}.npz"
    if out_path.exists() and not overwrite:
        raise RuntimeError(f"Fixture already exists: {out_path}")

    rows = _query_rows(client, dataset_id=dataset_id, case=case)
    if rows.num_rows == 0:
        raise RuntimeError(f"No rows found for object={case.object_id}")

    observations = _build_observations(rows, case)
    baseline = estimate_rotation_period(
        observations,
        frequency_grid_scale=case.frequency_grid_scale,
        max_frequency_cycles_per_day=case.max_frequency_cycles_per_day,
        min_rotations_in_span=case.min_rotations_in_span,
        session_mode=case.session_mode,
        search_strategy="grid",
    )

    np.savez_compressed(
        out_path,
        object_id=np.array([case.object_id], dtype=object),
        station=np.array([case.station_code], dtype=object),
        bands=np.array(case.bands, dtype=object),
        frequency_grid_scale=np.array([case.frequency_grid_scale], dtype=np.float64),
        max_frequency_cycles_per_day=np.array(
            [case.max_frequency_cycles_per_day], dtype=np.float64
        ),
        min_rotations_in_span=np.array([case.min_rotations_in_span], dtype=np.float64),
        session_mode=np.array([case.session_mode], dtype=object),
        expected_period_hours=np.array(
            [float(baseline.period_hours[0].as_py())], dtype=np.float64
        ),
        expected_fourier_order=np.array(
            [int(baseline.fourier_order[0].as_py())], dtype=np.int64
        ),
        expected_used_session_offsets=np.array(
            [bool(baseline.used_session_offsets[0].as_py())],
            dtype=bool,
        ),
        time_iso=np.array(observations.time.to_iso8601().to_pylist(), dtype=object),
        mag_obs=np.asarray(observations.mag, dtype=np.float64),
        mag_sigma=np.asarray(observations.mag_sigma.to_numpy(False), dtype=np.float64),
        filter=np.asarray(observations.filter.to_pylist(), dtype=object),
        session_id=np.asarray(observations.session_id.to_pylist(), dtype=object),
        r_au=np.asarray(observations.r_au, dtype=np.float64),
        delta_au=np.asarray(observations.delta_au, dtype=np.float64),
        phase_angle_deg=np.asarray(observations.phase_angle_deg, dtype=np.float64),
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
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    client = bigquery.Client()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        out_path = _write_fixture(
            client,
            dataset_id=str(args.dataset_id),
            case=case,
            out_dir=out_dir,
            overwrite=bool(args.overwrite),
        )
        print(f"Wrote {out_path.name}")


if __name__ == "__main__":
    main()
