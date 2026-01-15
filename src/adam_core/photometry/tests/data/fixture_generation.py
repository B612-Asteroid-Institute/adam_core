from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import astropy.time
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import requests
from google.cloud import bigquery
from mpcq.client import BigQueryMPCClient

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin
from adam_core.observations.exposures import Exposures
from adam_core.observers.observers import Observers
from adam_core.orbits.query.horizons import query_horizons
from adam_core.photometry.bandpasses import (
    map_to_canonical_filter_bands,
    load_observatory_band_map,
)
from adam_core.photometry.magnitude import predict_magnitudes
from adam_core.time import Timestamp


def _extract_float(value) -> float | None:
    """
    SBDB often returns nested dicts like {'value': 18.2, 'sigma': ...}.
    Accept both bare numerics and those dict shapes.
    """
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    if isinstance(value, dict):
        v = value.get("value")
        if v is None:
            return None
        return float(v)
    return None


def _sbdb_query_json(
    object_id: str,
    *,
    timeout_s: int = 60,
    max_attempts: int = 5,
) -> dict:
    """
    Query JPL SBDB via the public JSON API (ssd-api.jpl.nasa.gov) with retries.

    We use direct requests here (instead of astroquery) so we can control timeout/retry
    behavior and keep fixture generation resilient to transient network slowness.
    """
    obj = str(object_id).strip()
    if not obj:
        raise ValueError("object_id must be non-empty")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be > 0")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be > 0")

    url = "https://ssd-api.jpl.nasa.gov/sbdb.api"
    params = {
        "sstr": obj,
        # boolean parameters in this API are 'true'/'false' strings
        "phys-par": "true",
        "full-prec": "true",
    }

    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            resp = requests.get(url, params=params, timeout=timeout_s)
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as err:
            last_err = err
            # Exponential backoff with a small cap.
            sleep_s = min(8.0, 0.5 * (2**attempt))
            time.sleep(sleep_s)
        except Exception:
            # Non-retryable (HTTP 4xx, JSON decode, etc.)
            raise

    raise RuntimeError(f"SBDB query failed after {max_attempts} attempts: {last_err}")


def query_jpl_hg(object_id: str) -> tuple[float, float]:
    """
    Fetch H and G from JPL SBDB for a small body.

    Notes:
    - We intentionally do NOT use MPC orbits for H/G in test fixtures.
    - SBDB's H is V-band absolute magnitude.
    - G may be missing; we default to 0.15.
    """
    result = _sbdb_query_json(object_id, timeout_s=90, max_attempts=5)
    phys_list = result.get("phys_par") or []
    phys_by_name = {
        str(d.get("name")): d
        for d in phys_list
        if isinstance(d, dict) and d.get("name") is not None
    }

    H = _extract_float(phys_by_name.get("H")) or _extract_float(phys_by_name.get("H_mag"))
    if H is None:
        raise RuntimeError(f"JPL SBDB did not provide H for {object_id}")

    G = _extract_float(phys_by_name.get("G"))
    if G is None:
        G = 0.15
    return float(H), float(G)


def query_jpl_hg_and_kind(
    object_id: str,
) -> tuple[float | None, float | None, str | None]:
    """
    Query SBDB for (H, G, kind). Returns (None, None, kind) if H is missing.

    `kind` is SBDB's object type code (e.g. asteroids tend to be 'an'/'au';
    comets tend to start with 'c'). If missing, returns None.
    """
    result = _sbdb_query_json(object_id, timeout_s=90, max_attempts=5)
    obj = result.get("object") or {}
    kind = obj.get("kind")

    phys_list = result.get("phys_par") or []
    phys_by_name = {
        str(d.get("name")): d
        for d in phys_list
        if isinstance(d, dict) and d.get("name") is not None
    }
    H = _extract_float(phys_by_name.get("H")) or _extract_float(phys_by_name.get("H_mag"))
    G = _extract_float(phys_by_name.get("G"))
    return H, G, kind


def _is_comet_designation(object_id: str) -> bool:
    s = str(object_id).strip()
    # Common MPC comet prefixes: P/, C/, D/, X/, A/, I/
    if "/" in s:
        return True
    return s.startswith(("P", "C", "D", "X", "A", "I"))


def is_sbdb_asteroid(kind: str | None, object_id: str) -> bool:
    """
    Decide asteroid-vs-comet. We restrict to asteroids.
    - If SBDB kind is present and starts with 'c' -> comet.
    - Otherwise fall back to designation heuristics.
    """
    if kind is not None and str(kind):
        k = str(kind).strip().lower()
        if k.startswith("c"):
            return False
        # Most asteroid kinds are 'an'/'au' etc.
        return True
    # Fallback: crude but effective for MPC-style comet designations.
    return not _is_comet_designation(object_id)


def query_mpc_hg(
    client: BigQueryMPCClient, object_id: str
) -> tuple[float | None, float | None]:
    """
    Query MPCQ orbits table for H/G. Returns (None, None) if H is missing.
    """
    obj = str(object_id).strip()
    orbits = client.query_orbits([obj])
    h_mask = pc.is_valid(orbits.h)
    orbits_h = orbits.apply_mask(h_mask)
    if len(orbits_h) == 0:
        return None, None

    H_v = float(orbits_h.h[-1].as_py())
    if pc.is_valid(orbits_h.g[-1]).as_py():
        G = float(orbits_h.g[-1].as_py())
    else:
        G = 0.15
    return H_v, G


@dataclass(frozen=True)
class CandidateObject:
    object_id: str
    arc_days: int
    # canonical filter_id -> count of observations contributing to that filter_id
    obs_counts_by_filter_id: dict[str, int]


@dataclass(frozen=True)
class FixtureSelectionConfig:
    min_arc_days: int = 7
    min_obs_per_filter: int = 3
    max_obs_per_band_in_fixture: int = 10
    candidate_limit: int = 5000
    # Cost control: if set, restrict candidate queries to observations on/after
    # CURRENT_TIMESTAMP() - since_days. (This can dramatically reduce bytes scanned.)
    since_days: int | None = None


def timestamp_from_bq_obstime(obstime: pa.Array | pa.ChunkedArray) -> Timestamp:
    """
    Convert a BigQuery TIMESTAMP column (Arrow timestamp array) into `adam_core.time.Timestamp`.

    We intentionally avoid string formatting here because Arrow's strftime support can be
    backend/version-dependent (e.g. `%f` handling).
    """
    if isinstance(obstime, pa.ChunkedArray):
        obstime = obstime.combine_chunks()
    dt64 = obstime.to_numpy(zero_copy_only=False)
    t = astropy.time.Time(dt64, format="datetime64", scale="utc")
    return Timestamp.from_astropy(t)


def _slugify(value: str) -> str:
    s = str(value).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_") or "object"


def slugify_object_id(object_id: str) -> str:
    return _slugify(object_id)


def _bq_string_literal(value: str) -> str:
    # BigQuery Standard SQL: escape single quotes by doubling them.
    v = str(value)
    return "'" + v.replace("'", "''") + "'"


def normalize_reported_band_for_station(station_code: str, band: str) -> str:
    """
    Normalize station-specific reported band values into the canonical "reported band"
    strings expected by `ObservatoryBandMap` / bandpass filter resolution.

    For LSST (X05), MPC/ADES band encodings can include: 'g', 'Lg', 'LSST_g', etc.
    We normalize to 'u','g','r','i','z','y' (and accept 'Y' as 'y').
    """
    stn = str(station_code).strip()
    b = str(band).strip()
    if not b:
        return b

    if stn == "X05":
        if len(b) == 2 and b[0] == "L":
            b = b[1:]
        if b.startswith("LSST_"):
            b = b.split("_", 1)[1]
        if b == "Y":
            b = "y"
        return b

    return b


def band_variants_for_station(station_code: str, reported_band: str) -> set[str]:
    """
    Expand a (station, band) into a set of variants that may appear in MPCQ observations.
    This is used to widen BigQuery predicates without changing the downstream mapping.
    """
    stn = str(station_code).strip()
    b = str(reported_band).strip()
    if not stn or not b:
        return set()

    out = {b}
    if stn == "X05" and b in {"u", "g", "r", "i", "z", "y"}:
        out |= {f"L{b}", f"LSST_{b}"}
        if b == "y":
            out.add("LY")  # occasional uppercase variant
    return out


def target_filter_map_by_code(
    codes: Iterable[str],
) -> dict[str, dict[str, set[str]]]:
    """
    Return mapping:
        code -> (canonical_filter_id -> set(reported_band_variants))

    This treats aliases (e.g. W84|VR vs W84|vr) as the same canonical filter_id.
    """
    wanted = {str(c).strip() for c in codes if str(c).strip()}
    if not wanted:
        raise ValueError("No observatory codes provided")

    mapping = load_observatory_band_map()
    code_arr = pc.utf8_trim_whitespace(mapping.observatory_code)
    mask = pc.is_in(
        code_arr, value_set=pa.array(sorted(wanted), type=pa.large_string())
    )
    subset = mapping.apply_mask(mask)
    if len(subset) == 0:
        raise ValueError(f"No band map entries found for codes: {sorted(wanted)}")

    out: dict[str, dict[str, set[str]]] = {}
    for code, band, fid in zip(
        pc.utf8_trim_whitespace(subset.observatory_code).to_pylist(),
        pc.utf8_trim_whitespace(subset.reported_band).to_pylist(),
        pc.utf8_trim_whitespace(subset.filter_id).to_pylist(),
    ):
        c = str(code).strip()
        b = str(band).strip()
        f = str(fid).strip()
        if not c or not b or not f:
            continue
        variants = band_variants_for_station(c, b)
        if not variants:
            continue
        out.setdefault(c, {}).setdefault(f, set()).update(variants)

    missing = sorted(wanted - set(out.keys()))
    if missing:
        raise ValueError(f"No usable band map entries for codes: {missing}")
    return out


def query_distinct_bands(client: BigQueryMPCClient, code: str) -> set[str]:
    """
    Query BigQuery for distinct (trimmed) `band` values present for a station code.
    """
    c = str(code).strip()
    if not c:
        raise ValueError("code must be non-empty")

    query = f"""
    SELECT DISTINCT
      TRIM(band) AS band
    FROM `{client.dataset_id}.public_obs_sbn`
    WHERE TRIM(stn) = @stn
      AND band IS NOT NULL
      AND TRIM(band) != ''
    ORDER BY band
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("stn", "STRING", c)]
    )
    results = client.client.query(query, job_config=job_config).result()
    table = results.to_arrow(create_bqstorage_client=True)
    bands = pc.utf8_trim_whitespace(table.column("band"))
    return {
        str(x).strip() for x in bands.to_pylist() if x is not None and str(x).strip()
    }


def query_candidate_objects(
    client: BigQueryMPCClient,
    *,
    code: str,
    filter_id_to_bands: dict[str, set[str]],
    min_arc_days: int,
    min_obs_per_filter: int,
    limit: int,
    since_days: int | None = None,
) -> list[CandidateObject]:
    """
    Query candidate objects for a given station code, returning per-filter observation counts.

    The coverage units are canonical `filter_id` values; for each such filter_id we count
    observations whose reported `band` is any of the mapped band variants.
    """
    stn = str(code).strip()
    if not stn:
        raise ValueError("code must be non-empty")
    if min_arc_days < 0:
        raise ValueError("min_arc_days must be >= 0")
    if min_obs_per_filter <= 0:
        raise ValueError("min_obs_per_filter must be > 0")
    if limit <= 0:
        raise ValueError("limit must be > 0")
    if not filter_id_to_bands:
        raise ValueError("filter_id_to_bands must be non-empty")

    # Build COUNTIF columns per canonical filter_id.
    col_specs: list[tuple[str, str]] = []
    relevant_bands: set[str] = set()
    for filter_id, bands in sorted(filter_id_to_bands.items()):
        band_list = sorted({str(b).strip() for b in bands if str(b).strip()})
        if not band_list:
            continue
        relevant_bands |= set(band_list)
        col_name = "n_" + re.sub(r"[^A-Za-z0-9_]+", "_", str(filter_id)).strip("_")
        lits = ", ".join(_bq_string_literal(b) for b in band_list)
        expr = f"COUNTIF(band IN ({lits})) AS {col_name}"
        col_specs.append((filter_id, expr))

    if not col_specs:
        raise ValueError("No usable filter_id->bands mapping to query")
    if not relevant_bands:
        raise ValueError("No usable relevant bands derived from filter_id_to_bands")

    count_exprs_sql = ",\n        ".join(expr for _, expr in col_specs)
    count_col_names = [expr.split(" AS ", 1)[1].strip() for _, expr in col_specs]
    count_cols_sql = ", ".join([f"c.{name}" for name in count_col_names])
    greatest_sql = (
        "GREATEST(" + ", ".join([f"c.{name}" for name in count_col_names]) + ")"
    )

    time_clause = ""
    query_parameters: list[bigquery.QueryParameter] = [
        bigquery.ScalarQueryParameter("stn", "STRING", stn),
        bigquery.ScalarQueryParameter("min_arc_days", "INT64", int(min_arc_days)),
        bigquery.ScalarQueryParameter(
            "min_obs_per_filter", "INT64", int(min_obs_per_filter)
        ),
        bigquery.ArrayQueryParameter(
            "relevant_bands", "STRING", sorted(relevant_bands)
        ),
    ]
    if since_days is not None and int(since_days) > 0:
        time_clause = "AND obstime >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @since_days DAY)"
        query_parameters.append(
            bigquery.ScalarQueryParameter("since_days", "INT64", int(since_days))
        )

    query = f"""
    WITH station_obs AS (
      SELECT
        provid,
        obstime,
        TRIM(band) AS band
      FROM `{client.dataset_id}.public_obs_sbn`
      WHERE TRIM(stn) = @stn
        AND provid IS NOT NULL
        AND obstime IS NOT NULL
        AND band IS NOT NULL
        AND TRIM(band) != ''
        AND TRIM(band) IN UNNEST(@relevant_bands)
        {time_clause}
    ),
    arc AS (
      SELECT
        provid AS object_id,
        DATETIME_DIFF(MAX(obstime), MIN(obstime), DAY) AS arc_days,
        COUNT(*) AS n_total
      FROM station_obs
      GROUP BY object_id
      HAVING arc_days >= @min_arc_days
    ),
    counts AS (
      SELECT
        provid AS object_id,
        {count_exprs_sql}
      FROM station_obs
      GROUP BY object_id
    )
    SELECT
      a.object_id,
      a.arc_days,
      a.n_total,
      {count_cols_sql}
    FROM arc a
    JOIN counts c
      USING (object_id)
    WHERE {greatest_sql} >= @min_obs_per_filter
    ORDER BY a.arc_days DESC, a.n_total DESC, a.object_id ASC
    LIMIT {int(limit)}
    """
    job_config = bigquery.QueryJobConfig(query_parameters=query_parameters)
    results = client.client.query(query, job_config=job_config).result()
    table = results.to_arrow(create_bqstorage_client=True)
    if table.num_rows == 0:
        return []

    object_ids = table.column("object_id").to_pylist()
    arc_days = table.column("arc_days").to_pylist()

    out: list[CandidateObject] = []
    for i in range(int(table.num_rows)):
        oid = object_ids[i]
        if oid is None:
            continue
        a = arc_days[i]
        if a is None:
            continue
        counts: dict[str, int] = {}
        for filter_id, expr in col_specs:
            col_name = expr.split(" AS ", 1)[1].strip()
            v = table.column(col_name)[i].as_py()
            counts[str(filter_id)] = int(v or 0)
        # Optional early filtering: skip candidates that cover nothing.
        if not any(v >= int(min_obs_per_filter) for v in counts.values()):
            continue
        out.append(
            CandidateObject(
                object_id=str(oid), arc_days=int(a), obs_counts_by_filter_id=counts
            )
        )

    return out


def query_obs_sbn_rows_for_object_station(
    client: BigQueryMPCClient,
    *,
    object_id: str,
    station_code: str,
    allowed_bands: set[str],
    max_rows_per_band: int,
) -> pa.Table:
    """
    Efficiently fetch only the observation rows needed for a fixture.

    This avoids `client.query_observations(...)`, which can pull a large cross-designation
    history and touches additional tables. Here we query `public_obs_sbn` directly with
    strong predicates: provid+stn+band, and cap rows per band in SQL.
    """
    obj = str(object_id).strip()
    stn = str(station_code).strip()
    if not obj or not stn:
        raise ValueError("object_id and station_code must be non-empty")
    if max_rows_per_band <= 0:
        raise ValueError("max_rows_per_band must be > 0")

    bands = sorted({str(b).strip() for b in allowed_bands if str(b).strip()})
    if not bands:
        raise ValueError("allowed_bands must be non-empty")

    query = f"""
    WITH filtered AS (
      SELECT
        obsid,
        obstime,
        ra,
        dec,
        mag,
        rmsmag,
        TRIM(band) AS band
      FROM `{client.dataset_id}.public_obs_sbn`
      WHERE TRIM(stn) = @stn
        AND provid = @provid
        AND obstime IS NOT NULL
        AND mag IS NOT NULL
        AND band IS NOT NULL
        AND TRIM(band) IN UNNEST(@bands)
    ),
    ranked AS (
      SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY band ORDER BY obstime ASC) AS rn
      FROM filtered
    )
    SELECT
      obsid,
      obstime,
      ra,
      dec,
      mag,
      rmsmag,
      band
    FROM ranked
    WHERE rn <= @max_rows_per_band
    ORDER BY obstime ASC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("stn", "STRING", stn),
            bigquery.ScalarQueryParameter("provid", "STRING", obj),
            bigquery.ArrayQueryParameter("bands", "STRING", bands),
            bigquery.ScalarQueryParameter(
                "max_rows_per_band", "INT64", int(max_rows_per_band)
            ),
        ]
    )
    results = client.client.query(query, job_config=job_config).result()
    return results.to_arrow(create_bqstorage_client=True)


def select_objects_greedy(
    candidates: Sequence[CandidateObject],
    *,
    required_filter_ids: set[str],
    min_obs_per_filter: int,
) -> list[str]:
    """
    Greedy set cover:
    - Each candidate covers filter_ids where count >= min_obs_per_filter.
    - Choose candidates until all required_filter_ids are covered.
    """
    uncovered = {str(x) for x in required_filter_ids if str(x)}
    if not uncovered:
        return []

    remaining = list(candidates)
    chosen: list[str] = []

    while uncovered:
        best_idx: int | None = None
        best_gain = -1
        best_arc = -1
        best_id = ""

        for idx, cand in enumerate(remaining):
            covered = {
                fid
                for fid, n in cand.obs_counts_by_filter_id.items()
                if n >= int(min_obs_per_filter)
            }
            gain = len(covered & uncovered)
            if gain <= 0:
                continue
            # tie-break: larger gain, then longer arc, then lexicographically
            if gain > best_gain:
                best_idx, best_gain, best_arc, best_id = (
                    idx,
                    gain,
                    cand.arc_days,
                    cand.object_id,
                )
                continue
            if gain == best_gain and cand.arc_days > best_arc:
                best_idx, best_arc, best_id = idx, cand.arc_days, cand.object_id
                continue
            if (
                gain == best_gain
                and cand.arc_days == best_arc
                and cand.object_id < best_id
            ):
                best_idx, best_id = idx, cand.object_id

        if best_idx is None:
            break

        picked = remaining.pop(best_idx)
        chosen.append(picked.object_id)
        newly = {
            fid
            for fid, n in picked.obs_counts_by_filter_id.items()
            if n >= int(min_obs_per_filter)
        }
        uncovered -= newly

    if uncovered:
        raise RuntimeError(
            "Unable to cover all required filter_ids. Missing: "
            + ", ".join(sorted(uncovered))
        )

    return chosen


def build_fixture_for_object(
    client: BigQueryMPCClient,
    *,
    object_id: str,
    station_code: str,
    required_filter_ids: set[str],
    filter_id_to_bands: dict[str, set[str]],
    min_obs_per_filter: int,
    max_obs_per_filter: int,
    out_path: Path,
) -> Path:
    """
    Build a compact offline regression fixture for a single (object, station).

    The fixture stores:
    - observed mags (mag_obs) and geometry (object_pos, observer_pos)
    - canonical filter_ids (via find_suggested_filter_bands)
    - per-filter residual ceilings for the current implementation (median abs + p95 abs)
    """
    obj = str(object_id).strip()
    stn = str(station_code).strip()
    if not obj or not stn:
        raise ValueError("object_id and station_code must be non-empty")
    if min_obs_per_filter <= 0:
        raise ValueError("min_obs_per_filter must be > 0")
    if max_obs_per_filter <= 0:
        raise ValueError("max_obs_per_filter must be > 0")
    if max_obs_per_filter < min_obs_per_filter:
        raise ValueError("max_obs_per_filter must be >= min_obs_per_filter")

    required = {str(x).strip() for x in required_filter_ids if str(x).strip()}
    if not required:
        raise ValueError("required_filter_ids must be non-empty")
    if not filter_id_to_bands:
        raise ValueError("filter_id_to_bands must be non-empty")

    relevant_bands: set[str] = set()
    for fid in required:
        bs = filter_id_to_bands.get(fid)
        if not bs:
            raise ValueError(f"Missing band variants for required filter_id: {fid}")
        relevant_bands |= {str(b).strip() for b in bs if str(b).strip()}
    if not relevant_bands:
        raise ValueError("No relevant bands derived from required_filter_ids")

    # Cost control: fetch only the (object, station, band) rows we might use, directly from
    # `public_obs_sbn` with strong predicates and a per-band cap in SQL.
    raw = query_obs_sbn_rows_for_object_station(
        client,
        object_id=obj,
        station_code=stn,
        allowed_bands=relevant_bands,
        max_rows_per_band=int(max_obs_per_filter),
    )
    if raw.num_rows == 0:
        raise RuntimeError(
            f"No observations found for {obj} at station {stn} in bands={sorted(relevant_bands)}"
        )

    bands_all_raw = [
        str(x).strip() for x in pc.utf8_trim_whitespace(raw.column("band")).to_pylist()
    ]
    bands_all = [normalize_reported_band_for_station(stn, b) for b in bands_all_raw]
    canonical_all = map_to_canonical_filter_bands(
        [stn] * len(bands_all),
        bands_all,
        allow_fallback_filters=False,
    )
    canonical_all = np.asarray(canonical_all, dtype=object)

    # Validate minimum per required canonical filter_id (within fetched rows).
    for fid in sorted(required):
        n_f = int(np.sum(canonical_all == fid))
        if n_f < int(min_obs_per_filter):
            raise RuntimeError(
                f"Insufficient observations for {obj} {stn} filter_id '{fid}': "
                f"{n_f} < {min_obs_per_filter}"
            )

    # Deterministic cap per required canonical filter_id.
    keep_idx: list[int] = []
    counts: dict[str, int] = {fid: 0 for fid in required}
    for i, fid in enumerate(canonical_all.tolist()):
        if fid not in counts:
            continue
        if counts[fid] >= int(max_obs_per_filter):
            continue
        counts[fid] += 1
        keep_idx.append(i)

    raw = raw.take(pa.array(keep_idx, type=pa.int64()))
    bands_raw = [
        str(x).strip() for x in pc.utf8_trim_whitespace(raw.column("band")).to_pylist()
    ]
    bands = [normalize_reported_band_for_station(stn, b) for b in bands_raw]
    canonical = map_to_canonical_filter_bands(
        [stn] * len(bands),
        bands,
        allow_fallback_filters=False,
    )

    # Require asteroid + H in both MPC and JPL (for A/B benchmarking).
    H_jpl, G_jpl, kind = query_jpl_hg_and_kind(obj)
    if not is_sbdb_asteroid(kind, obj):
        raise RuntimeError(f"Object is not an asteroid (SBDB kind={kind}): {obj}")
    if H_jpl is None:
        raise RuntimeError(f"Missing JPL SBDB H for {obj}")
    if G_jpl is None:
        G_jpl = 0.15

    H_mpc, G_mpc = query_mpc_hg(client, obj)
    if H_mpc is None:
        raise RuntimeError(f"Missing MPC H for {obj}")
    if G_mpc is None:
        G_mpc = 0.15

    times = timestamp_from_bq_obstime(raw.column("obstime"))
    n = int(raw.num_rows)

    exposures = Exposures.from_kwargs(
        id=pa.array(raw.column("obsid").to_pylist(), type=pa.large_string()),
        start_time=times,
        duration=np.zeros(n, dtype=np.float64),
        filter=pa.array(bands, type=pa.large_string()),
        observatory_code=pa.array([stn] * n, type=pa.large_string()),
    )
    times_utc = exposures.midpoint()

    # Heliocentric geometry (network call): retry a few times for transient slowness.
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            orbits_at_times = query_horizons(
                object_ids=[obj],
                times=times_utc,
                coordinate_type="cartesian",
                location="@sun",
                id_type="smallbody",
            )
            last_err = None
            break
        except Exception as e:
            last_err = e
            time.sleep(min(8.0, 0.5 * (2**attempt)))
    if last_err is not None:
        raise RuntimeError(f"Horizons query failed after retries for {obj}: {last_err}")
    object_coords = orbits_at_times.coordinates
    observers = exposures.observers()

    # Predict magnitudes in canonical filters.
    exposures = exposures.set_column(
        "filter", pa.array(canonical.tolist(), type=pa.large_string())
    )

    mag_obs = np.asarray(
        raw.column("mag").to_numpy(zero_copy_only=False), dtype=np.float64
    )
    canonical_np = np.asarray(canonical, dtype=object)
    keys = sorted({str(x) for x in canonical_np.tolist()})

    def _stats_for(H_v: float, G: float) -> tuple[np.ndarray, np.ndarray]:
        pred = predict_magnitudes(
            H=H_v,
            object_coords=object_coords,
            exposures=exposures,
            G=G,
            reference_filter="V",
            composition="NEO",
        )
        resid = np.asarray(pred, dtype=np.float64) - mag_obs
        med_abs = np.full(len(keys), np.nan, dtype=np.float64)
        p95_abs = np.full(len(keys), np.nan, dtype=np.float64)
        for i, k in enumerate(keys):
            xs = resid[canonical_np == k]
            if xs.size == 0:
                continue
            med_abs[i] = float(np.abs(np.median(xs)))
            p95_abs[i] = float(np.quantile(np.abs(xs), 0.95))
        return med_abs, p95_abs

    baseline_mpc_median_abs, baseline_mpc_p95_abs = _stats_for(H_mpc, G_mpc)
    baseline_jpl_median_abs, baseline_jpl_p95_abs = _stats_for(
        float(H_jpl), float(G_jpl)
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        object_id=np.array([obj], dtype=object),
        station=np.array([stn], dtype=object),
        # Store both sources so benchmark tests can compare.
        H_v_mpc=np.array([H_mpc], dtype=np.float64),
        G_mpc=np.array([G_mpc], dtype=np.float64),
        H_v_jpl=np.array([float(H_jpl)], dtype=np.float64),
        G_jpl=np.array([float(G_jpl)], dtype=np.float64),
        sbdb_kind=np.array([kind if kind is not None else ""], dtype=object),
        time_iso=np.array(times_utc.to_iso8601().to_pylist(), dtype=object),
        obsid=np.array(raw.column("obsid").to_pylist(), dtype=object),
        band=np.array(bands, dtype=object),
        filter_id=np.array(canonical.tolist(), dtype=object),
        mag_obs=mag_obs,
        rmsmag=np.asarray(
            raw.column("rmsmag").to_numpy(zero_copy_only=False), dtype=np.float64
        ),
        ra=np.asarray(
            raw.column("ra").to_numpy(zero_copy_only=False), dtype=np.float64
        ),
        dec=np.asarray(
            raw.column("dec").to_numpy(zero_copy_only=False), dtype=np.float64
        ),
        object_pos=np.asarray(object_coords.r, dtype=np.float64),
        observer_pos=np.asarray(observers.coordinates.r, dtype=np.float64),
        baseline_keys=np.array(keys, dtype=object),
        baseline_mpc_median_abs=baseline_mpc_median_abs,
        baseline_mpc_p95_abs=baseline_mpc_p95_abs,
        baseline_jpl_median_abs=baseline_jpl_median_abs,
        baseline_jpl_p95_abs=baseline_jpl_p95_abs,
    )

    return out_path


def observers_from_heliocentric_positions(
    *,
    station_code: str,
    times_utc: Timestamp,
    heliocentric_pos_au: np.ndarray,
) -> Observers:
    stn = str(station_code).strip()
    if not stn:
        raise ValueError("station_code must be non-empty")
    pos = np.asarray(heliocentric_pos_au, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("heliocentric_pos_au must have shape (N, 3)")
    if len(times_utc) != pos.shape[0]:
        raise ValueError("times_utc length must match heliocentric_pos_au rows")

    return Observers.from_kwargs(
        code=[stn] * pos.shape[0],
        coordinates=CartesianCoordinates.from_kwargs(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            vx=np.zeros(pos.shape[0]),
            vy=np.zeros(pos.shape[0]),
            vz=np.zeros(pos.shape[0]),
            time=times_utc,
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"] * pos.shape[0]),
        ),
    )
