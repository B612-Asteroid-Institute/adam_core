"""
Generate offline rotation-period regression fixtures from the PDS lightcurve archive.

This script is intentionally separate from the older MPC-based fixture generator.
It uses:
- LCDB summary rows as the published-period source
- ALCDEF metadata + lcdata tables as the raw lightcurve source

The script saves compact `.npz` fixtures that are consumed by an optional offline
pytest suite. The fixture payload is solver-ready: observation times, magnitudes,
errors, filters, session ids, geometry arrays, expected period, tolerance, and
source metadata.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import astropy.coordinates
import astropy.time
import astropy.units as u
import numpy as np
import pyarrow as pa
import spiceypy as sp

from adam_core.coordinates.origin import OriginCodes
from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.orbits.query.horizons import query_horizons
from adam_core.observers.state import get_observer_state
from adam_core.time import Timestamp


LCDB_SUMMARY_URL = (
    "https://sbnarchive.psi.edu/pds4/non_mission/"
    "ast-lightcurve-database_V4_0/data/lc_summary.csv"
)


@dataclass(frozen=True)
class PDSRotationFixtureCase:
    object_number: int
    object_name: str
    lcdb_name: str
    metadata_url: str
    lcdata_url: str
    tolerance_fraction: float
    source_title: str
    source_url: str
    tier: str = "gold"
    included_mdids: tuple[int, ...] | None = None
    frequency_grid_scale: float = 30.0
    max_frequency_cycles_per_day: float = 24.0
    min_rotations_in_span: float = 2.0


CASES: tuple[PDSRotationFixtureCase, ...] = (
    PDSRotationFixtureCase(
        object_number=289,
        object_name="Nenetta",
        lcdb_name="Nenetta",
        metadata_url=(
            "https://sbnarchive.psi.edu/pds4/non_mission/"
            "gbo.ast.alcdef-database_V1_0/data/200/alcdef_metadata-200-299.csv"
        ),
        lcdata_url=(
            "https://sbnarchive.psi.edu/pds4/non_mission/"
            "gbo.ast.alcdef-database_V1_0/data/200/alcdef_lcdata-200-299.csv"
        ),
        tolerance_fraction=0.01,
        source_title="LCDB v4.0 / 289 Nenetta",
        source_url=LCDB_SUMMARY_URL,
        tier="gold",
        included_mdids=(2808, 2809, 2810, 2811, 2812, 2813),
    ),
    PDSRotationFixtureCase(
        object_number=1011,
        object_name="Laodamia",
        lcdb_name="Laodamia",
        metadata_url=(
            "https://sbnarchive.psi.edu/pds4/non_mission/"
            "gbo.ast.alcdef-database_V1_0/data/1000/alcdef_metadata-1000-1999.csv"
        ),
        lcdata_url=(
            "https://sbnarchive.psi.edu/pds4/non_mission/"
            "gbo.ast.alcdef-database_V1_0/data/1000/alcdef_lcdata-1000-1999.csv"
        ),
        tolerance_fraction=0.01,
        source_title="LCDB v4.0 / 1011 Laodamia",
        source_url=LCDB_SUMMARY_URL,
        tier="gold",
        included_mdids=(9027, 9028, 9029),
    ),
    PDSRotationFixtureCase(
        object_number=886,
        object_name="Washingtonia",
        lcdb_name="Washingtonia",
        metadata_url=(
            "https://sbnarchive.psi.edu/pds4/non_mission/"
            "gbo.ast.alcdef-database_V1_0/data/800/alcdef_metadata-800-899.csv"
        ),
        lcdata_url=(
            "https://sbnarchive.psi.edu/pds4/non_mission/"
            "gbo.ast.alcdef-database_V1_0/data/800/alcdef_lcdata-800-899.csv"
        ),
        tolerance_fraction=0.01,
        source_title="LCDB v4.0 / 886 Washingtonia",
        source_url=LCDB_SUMMARY_URL,
        tier="gold",
        included_mdids=(8177, 8178, 8179),
    ),
    PDSRotationFixtureCase(
        object_number=1282,
        object_name="Utopia",
        lcdb_name="Utopia",
        metadata_url=(
            "https://sbnarchive.psi.edu/pds4/non_mission/"
            "gbo.ast.alcdef-database_V1_0/data/1000/alcdef_metadata-1000-1999.csv"
        ),
        lcdata_url=(
            "https://sbnarchive.psi.edu/pds4/non_mission/"
            "gbo.ast.alcdef-database_V1_0/data/1000/alcdef_lcdata-1000-1999.csv"
        ),
        tolerance_fraction=0.01,
        source_title="LCDB v4.0 / 1282 Utopia",
        source_url=LCDB_SUMMARY_URL,
        tier="gold",
        included_mdids=(11414, 11415, 11416, 11417),
    ),
    PDSRotationFixtureCase(
        object_number=1323,
        object_name="Tugela",
        lcdb_name="Tugela",
        metadata_url=(
            "https://sbnarchive.psi.edu/pds4/non_mission/"
            "gbo.ast.alcdef-database_V1_0/data/1000/alcdef_metadata-1000-1999.csv"
        ),
        lcdata_url=(
            "https://sbnarchive.psi.edu/pds4/non_mission/"
            "gbo.ast.alcdef-database_V1_0/data/1000/alcdef_lcdata-1000-1999.csv"
        ),
        tolerance_fraction=0.02,
        source_title="LCDB v4.0 / 1323 Tugela",
        source_url=LCDB_SUMMARY_URL,
        tier="gold",
        included_mdids=(11675, 11676, 11677, 11678),
    ),
    PDSRotationFixtureCase(
        object_number=1627,
        object_name="Ivar",
        lcdb_name="Ivar",
        metadata_url=(
            "https://sbnarchive.psi.edu/pds4/non_mission/"
            "gbo.ast.alcdef-database_V1_0/data/1000/alcdef_metadata-1000-1999.csv"
        ),
        lcdata_url=(
            "https://sbnarchive.psi.edu/pds4/non_mission/"
            "gbo.ast.alcdef-database_V1_0/data/1000/alcdef_lcdata-1000-1999.csv"
        ),
        tolerance_fraction=0.01,
        source_title="LCDB v4.0 / 1627 Ivar",
        source_url=LCDB_SUMMARY_URL,
        tier="gold",
        included_mdids=(13838, 13839, 13840, 13841, 13842, 13843),
    ),
    PDSRotationFixtureCase(
        object_number=511,
        object_name="Davida",
        lcdb_name="Davida",
        metadata_url=(
            "https://sbnarchive.psi.edu/pds4/non_mission/"
            "gbo.ast.alcdef-database_V1_0/data/500/alcdef_metadata-500-599.csv"
        ),
        lcdata_url=(
            "https://sbnarchive.psi.edu/pds4/non_mission/"
            "gbo.ast.alcdef-database_V1_0/data/500/alcdef_lcdata-500-599.csv"
        ),
        tolerance_fraction=0.05,
        source_title="LCDB v4.0 / 511 Davida",
        source_url=LCDB_SUMMARY_URL,
        tier="challenge",
        included_mdids=(5065, 5066, 5067),
    ),
    PDSRotationFixtureCase(
        object_number=702,
        object_name="Alauda",
        lcdb_name="Alauda",
        metadata_url=(
            "https://sbnarchive.psi.edu/pds4/non_mission/"
            "gbo.ast.alcdef-database_V1_0/data/700/alcdef_metadata-700-799.csv"
        ),
        lcdata_url=(
            "https://sbnarchive.psi.edu/pds4/non_mission/"
            "gbo.ast.alcdef-database_V1_0/data/700/alcdef_lcdata-700-799.csv"
        ),
        tolerance_fraction=0.1,
        source_title="LCDB v4.0 / 702 Alauda",
        source_url=LCDB_SUMMARY_URL,
        tier="challenge",
        included_mdids=(6666, 6667, 6668, 6669, 6670, 6671, 6672, 6673),
    ),
)


def _fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode("utf-8")


def _csv_records(url: str, header_prefix: str) -> list[dict[str, str]]:
    text = _fetch_text(url)
    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.startswith(header_prefix):
            start = idx
            break
    if start is None:
        raise RuntimeError(f"Header '{header_prefix}' not found in {url}")

    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    return [dict(row) for row in reader]


def _parse_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text or text == "-99.9" or text == "-99":
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"-", "-99", "-9"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _normalize_str(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "-":
        return None
    return text


def _load_lcdb_row(object_number: int) -> dict[str, str]:
    for row in _csv_records(LCDB_SUMMARY_URL, "Number,"):
        if _parse_int(row.get("Number")) == object_number:
            return row
    raise RuntimeError(f"LCDB summary row not found for object {object_number}")


def _object_number_from_lcdata_row(row: dict[str, str]) -> int | None:
    number = row.get("ObjectNumber")
    if number is None:
        return None
    text = str(number).strip().strip('"')
    try:
        return int(text)
    except ValueError:
        return None


def _build_object_times(jd: np.ndarray) -> Timestamp:
    return Timestamp.from_astropy(astropy.time.Time(jd, format="jd", scale="utc"))


def _observer_state_from_geodetic(
    *,
    times: Timestamp,
    obs_long_deg: float,
    obs_lat_deg: float,
) -> CartesianCoordinates:
    t_astropy = times.to_astropy()
    location = astropy.coordinates.EarthLocation.from_geodetic(
        lon=obs_long_deg * u.deg,
        lat=obs_lat_deg * u.deg,
        height=0.0 * u.m,
    )
    topo_itrs = location.get_itrs(obstime=t_astropy).cartesian.xyz.to_value(u.au)
    topo_itrs = np.asarray(topo_itrs, dtype=np.float64).T

    # Start from geocenter state and add the topocentric offset in the
    # Earth-fixed frame rotated into the ecliptic frame for each epoch.
    geocenter = get_observer_state("500", times, frame="ecliptic", origin=OriginCodes.SUN)
    r_obs = np.asarray(geocenter.r, dtype=np.float64).copy()
    v_obs = np.asarray(geocenter.v, dtype=np.float64).copy()

    et = np.asarray(times.et(), dtype=np.float64)
    for epoch in np.unique(et):
        mask = et == epoch
        rot = sp.pxform("ITRF93", "ECLIPJ2000", float(epoch))
        r_obs[mask] += (rot @ topo_itrs[mask].T).T

    return CartesianCoordinates.from_kwargs(
        time=times,
        x=r_obs[:, 0],
        y=r_obs[:, 1],
        z=r_obs[:, 2],
        vx=v_obs[:, 0],
        vy=v_obs[:, 1],
        vz=v_obs[:, 2],
        frame="ecliptic",
        origin=geocenter.origin,
    )


def _observer_state_for_session(
    *,
    times: Timestamp,
    observatory_code: str | None,
    obs_long_deg: float,
    obs_lat_deg: float,
) -> CartesianCoordinates:
    if observatory_code is not None:
        return get_observer_state(observatory_code, times, frame="ecliptic", origin=OriginCodes.SUN)
    return _observer_state_from_geodetic(
        times=times,
        obs_long_deg=obs_long_deg,
        obs_lat_deg=obs_lat_deg,
    )


def _query_object_coords(
    *,
    object_number: int,
    times: Timestamp,
) -> CartesianCoordinates:
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            result = query_horizons(
                object_ids=[str(object_number)],
                times=times,
                coordinate_type="cartesian",
                location="@sun",
                id_type="smallbody",
            )
            return result.coordinates
        except Exception as err:  # pragma: no cover - network retry path
            last_err = err
            time.sleep(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"Horizons query failed for {object_number}: {last_err}")


def _save_fixture(
    *,
    out_path: Path,
    case: PDSRotationFixtureCase,
    lcdb_row: dict[str, str],
    metadata_rows: list[dict[str, str]],
    lcdata_rows: list[dict[str, str]],
) -> None:
    session_rows: dict[int, dict[str, str]] = {
        sid: row
        for sid, row in (
            (_parse_int(row.get("ID")), row) for row in metadata_rows if _parse_int(row.get("ID")) is not None
        )
    }
    for row in metadata_rows:
        if _parse_int(row.get("ID")) is None:
            continue
        if _parse_int(row.get("ObjNumber")) != case.object_number:
            raise RuntimeError(
                f"metadata row {row.get('ID')} does not match object {case.object_number}"
            )

    rows_by_session: dict[int, list[dict[str, str]]] = {}
    for row in lcdata_rows:
        if _parse_int(row.get("ObjectNumber")) != case.object_number:
            continue
        mdid = _parse_int(row.get("MDID"))
        if mdid is None:
            continue
        rows_by_session.setdefault(mdid, []).append(row)

    session_ids = sorted(rows_by_session)
    if not session_ids:
        raise RuntimeError(f"No lcdata rows found for object {case.object_number}")

    obs_time_chunks: list[Timestamp] = []
    mag_chunks: list[np.ndarray] = []
    mag_sigma_chunks: list[np.ndarray] = []
    filter_chunks: list[list[str]] = []
    session_id_chunks: list[list[str]] = []
    r_chunks: list[np.ndarray] = []
    delta_chunks: list[np.ndarray] = []
    phase_chunks: list[np.ndarray] = []

    session_meta: dict[str, list[object]] = {
        "session_id": [],
        "submitter": [],
        "session_datetime": [],
        "observatory_code": [],
        "obs_long_deg": [],
        "obs_lat_deg": [],
        "filter": [],
        "mag_band": [],
        "diff_mags": [],
        "diff_zero_mag": [],
        "reduced_mags": [],
        "unity_cor": [],
        "ltc_applied": [],
        "ltc_type": [],
        "ltc_days": [],
        "publication": [],
        "bibcode": [],
        "comments": [],
    }

    for mdid in session_ids:
        meta = session_rows.get(mdid)
        if meta is None:
            raise RuntimeError(f"Missing metadata row for MDID {mdid}")

        session_rows_sorted = sorted(
            rows_by_session[mdid],
            key=lambda row: _parse_float(row.get("JD")),
        )
        jd = np.asarray([_parse_float(row.get("JD")) for row in session_rows_sorted], dtype=np.float64)
        if np.any(~np.isfinite(jd)):
            raise RuntimeError(f"Non-finite JD encountered for MDID {mdid}")

        times = _build_object_times(jd)
        observatory_code = _normalize_str(meta.get("MPCCode"))
        obs_long = _parse_float(meta.get("ObsLong"))
        obs_lat = _parse_float(meta.get("ObsLat"))
        if observatory_code is not None and observatory_code == "-":
            observatory_code = None
        if observatory_code is None and (not np.isfinite(obs_long) or not np.isfinite(obs_lat)):
            raise RuntimeError(
                f"Cannot determine observer geometry for MDID {mdid}: missing MPCCode and geodetic coordinates"
            )

        observer_state = _observer_state_for_session(
            times=times,
            observatory_code=observatory_code,
            obs_long_deg=obs_long,
            obs_lat_deg=obs_lat,
        )
        object_coords = _query_object_coords(object_number=case.object_number, times=times)

        object_r = np.asarray(object_coords.r, dtype=np.float64)
        observer_r = np.asarray(observer_state.r, dtype=np.float64)
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

        mags = np.asarray([_parse_float(row.get("Mag")) for row in session_rows_sorted], dtype=np.float64)
        mag_errs = np.asarray(
            [_parse_float(row.get("MagErr")) for row in session_rows_sorted],
            dtype=np.float64,
        )
        if np.any(~np.isfinite(mags)):
            raise RuntimeError(f"Non-finite magnitude encountered for MDID {mdid}")

        filter_label = _normalize_str(meta.get("Filter")) or _normalize_str(meta.get("MagBand"))
        if filter_label is None:
            filter_label = "UNKNOWN"

        obs_time_chunks.append(times)
        mag_chunks.append(mags)
        mag_sigma_chunks.append(mag_errs)
        filter_chunks.append([filter_label] * len(session_rows_sorted))
        session_id_chunks.append([str(mdid)] * len(session_rows_sorted))
        r_chunks.append(r_au)
        delta_chunks.append(delta_au)
        phase_chunks.append(phase_angle_deg)

        session_meta["session_id"].append(str(mdid))
        session_meta["submitter"].append(_normalize_str(meta.get("Submitter")))
        session_meta["session_datetime"].append(_normalize_str(meta.get("SessionDateTime")))
        session_meta["observatory_code"].append(observatory_code)
        session_meta["obs_long_deg"].append(obs_long)
        session_meta["obs_lat_deg"].append(obs_lat)
        session_meta["filter"].append(_normalize_str(meta.get("Filter")))
        session_meta["mag_band"].append(_normalize_str(meta.get("MagBand")))
        session_meta["diff_mags"].append(_normalize_str(meta.get("DiffMags")))
        session_meta["diff_zero_mag"].append(_normalize_str(meta.get("DiffZeroMag")))
        session_meta["reduced_mags"].append(_normalize_str(meta.get("ReducedMags")))
        session_meta["unity_cor"].append(_normalize_str(meta.get("UnityCor")))
        session_meta["ltc_applied"].append(_normalize_str(meta.get("LTCApplied")))
        session_meta["ltc_type"].append(_normalize_str(meta.get("LTCType")))
        session_meta["ltc_days"].append(_parse_float(meta.get("LTCDays")))
        session_meta["publication"].append(_normalize_str(meta.get("Publication")))
        session_meta["bibcode"].append(_normalize_str(meta.get("BibCode")))
        session_meta["comments"].append(_normalize_str(meta.get("Comments")))

    time = Timestamp.from_iso8601(
        np.concatenate([chunk.to_iso8601().to_pylist() for chunk in obs_time_chunks]),
        scale="utc",
    )
    mags = np.concatenate(mag_chunks)
    mag_sigma = np.concatenate(mag_sigma_chunks)
    filters = np.concatenate([np.asarray(chunk, dtype=object) for chunk in filter_chunks])
    session_id = np.concatenate([np.asarray(chunk, dtype=object) for chunk in session_id_chunks])
    r_au = np.concatenate(r_chunks)
    delta_au = np.concatenate(delta_chunks)
    phase_angle_deg = np.concatenate(phase_chunks)

    order = np.argsort(np.asarray(time.mjd(), dtype=np.float64), kind="mergesort")
    time = time.take(pa.array(order, type=pa.int32()))
    mags = mags[order]
    mag_sigma = mag_sigma[order]
    filters = filters[order]
    session_id = session_id[order]
    r_au = r_au[order]
    delta_au = delta_au[order]
    phase_angle_deg = phase_angle_deg[order]

    lcdb_period_hours = _parse_float(lcdb_row.get("Period"))
    lcdb_u = _parse_int(lcdb_row.get("U"))
    if not np.isfinite(lcdb_period_hours):
        raise RuntimeError(f"LCDB period missing for object {case.object_number}")
    if lcdb_u is None:
        raise RuntimeError(f"LCDB U value missing for object {case.object_number}")

    np.savez_compressed(
        out_path,
        object_number=np.array([case.object_number], dtype=np.int64),
        object_name=np.array([case.object_name], dtype=object),
        lcdb_name=np.array([case.lcdb_name], dtype=object),
        tier=np.array([case.tier], dtype=object),
        lcdb_period_hours=np.array([lcdb_period_hours], dtype=np.float64),
        lcdb_u=np.array([lcdb_u], dtype=np.int64),
        expected_period_hours=np.array([lcdb_period_hours], dtype=np.float64),
        tolerance_fraction=np.array([case.tolerance_fraction], dtype=np.float64),
        source_title=np.array([case.source_title], dtype=object),
        source_url=np.array([case.source_url], dtype=object),
        frequency_grid_scale=np.array([case.frequency_grid_scale], dtype=np.float64),
        max_frequency_cycles_per_day=np.array([case.max_frequency_cycles_per_day], dtype=np.float64),
        min_rotations_in_span=np.array([case.min_rotations_in_span], dtype=np.float64),
        time_iso=np.array(time.to_iso8601().to_pylist(), dtype=object),
        mag_obs=mags,
        mag_sigma=mag_sigma,
        filter=filters,
        session_id=session_id,
        r_au=r_au,
        delta_au=delta_au,
        phase_angle_deg=phase_angle_deg,
        session_mdid=np.array(session_meta["session_id"], dtype=object),
        session_submitter=np.array(session_meta["submitter"], dtype=object),
        session_datetime=np.array(session_meta["session_datetime"], dtype=object),
        session_observatory_code=np.array(session_meta["observatory_code"], dtype=object),
        session_obs_long_deg=np.array(session_meta["obs_long_deg"], dtype=np.float64),
        session_obs_lat_deg=np.array(session_meta["obs_lat_deg"], dtype=np.float64),
        session_filter=np.array(session_meta["filter"], dtype=object),
        session_mag_band=np.array(session_meta["mag_band"], dtype=object),
        session_diff_mags=np.array(session_meta["diff_mags"], dtype=object),
        session_diff_zero_mag=np.array(session_meta["diff_zero_mag"], dtype=object),
        session_reduced_mags=np.array(session_meta["reduced_mags"], dtype=object),
        session_unity_cor=np.array(session_meta["unity_cor"], dtype=object),
        session_ltc_applied=np.array(session_meta["ltc_applied"], dtype=object),
        session_ltc_type=np.array(session_meta["ltc_type"], dtype=object),
        session_ltc_days=np.array(session_meta["ltc_days"], dtype=np.float64),
        session_publication=np.array(session_meta["publication"], dtype=object),
        session_bibcode=np.array(session_meta["bibcode"], dtype=object),
        session_comments=np.array(session_meta["comments"], dtype=object),
    )


def _build_fixture(
    case: PDSRotationFixtureCase,
    *,
    out_dir: Path,
    overwrite: bool,
) -> Path:
    lcdb_row = _load_lcdb_row(case.object_number)
    metadata_rows = [
        row for row in _csv_records(case.metadata_url, "ID,") if _parse_int(row.get("ObjNumber")) == case.object_number
    ]
    if not metadata_rows:
        raise RuntimeError(f"No ALCDEF metadata rows found for object {case.object_number}")
    if case.included_mdids is not None:
        included_mdids = set(case.included_mdids)
        metadata_rows = [
            row for row in metadata_rows if _parse_int(row.get("ID")) in included_mdids
        ]
        missing_mdids = sorted(included_mdids - {_parse_int(row.get("ID")) for row in metadata_rows})
        if missing_mdids:
            raise RuntimeError(
                f"Metadata rows missing for object {case.object_number}: {missing_mdids}"
            )

    lcdata_rows = [
        row for row in _csv_records(case.lcdata_url, "ObjectNumber,") if _parse_int(row.get("ObjectNumber")) == case.object_number
    ]
    if not lcdata_rows:
        raise RuntimeError(f"No ALCDEF lcdata rows found for object {case.object_number}")
    if case.included_mdids is not None:
        included_mdids = set(case.included_mdids)
        lcdata_rows = [
            row for row in lcdata_rows if _parse_int(row.get("MDID")) in included_mdids
        ]
        found_mdids = {_parse_int(row.get("MDID")) for row in lcdata_rows}
        missing_mdids = sorted(included_mdids - found_mdids)
        if missing_mdids:
            raise RuntimeError(
                f"Lcdata rows missing for object {case.object_number}: {missing_mdids}"
            )

    out_path = out_dir / f"rotation_period_pds_fixture_{case.object_number}_{case.object_name}.npz"
    if out_path.exists() and not overwrite:
        raise RuntimeError(f"Fixture already exists: {out_path}")

    _save_fixture(
        out_path=out_path,
        case=case,
        lcdb_row=lcdb_row,
        metadata_rows=metadata_rows,
        lcdata_rows=lcdata_rows,
    )
    return out_path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for case in CASES:
        out_path = _build_fixture(case, out_dir=out_dir, overwrite=bool(args.overwrite))
        print(f"Wrote {out_path.name}")


if __name__ == "__main__":
    main()
