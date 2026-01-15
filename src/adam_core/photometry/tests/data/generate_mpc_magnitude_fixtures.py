"""
Generate offline photometry regression fixtures from MPCQ BigQuery + Horizons.

This is intended to be run manually by a developer (requires network access + ADC).
The generated `.npz` files are used by offline pytest regression tests.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from mpcq.client import BigQueryMPCClient

from adam_core.photometry.tests.data.fixture_generation import (
    FixtureSelectionConfig,
    build_fixture_for_object,
    query_candidate_objects,
    query_distinct_bands,
    slugify_object_id,
    target_filter_map_by_code,
)

DEFAULT_CODES: tuple[str, ...] = (
    "I41",
    "Q55",
    "X05",
    "T05",
    "T08",
    "M22",
    "W68",
    "W84",
    "695",
    "V00",
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-id",
        default=os.environ.get("MPCQ_DATASET_ID", "moeyens-thor-dev.mpc_sbn_aurora"),
    )
    ap.add_argument(
        "--views-dataset-id",
        default=os.environ.get(
            "MPCQ_VIEWS_DATASET_ID", "moeyens-thor-dev.mpc_sbn_aurora_views"
        ),
    )
    ap.add_argument("--codes", nargs="+", default=list(DEFAULT_CODES))
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory to write fixtures into (default: this folder).",
    )
    ap.add_argument("--min-arc-days", type=int, default=7)
    ap.add_argument("--min-obs-per-filter", type=int, default=5)
    ap.add_argument("--max-obs-per-filter", type=int, default=10)
    ap.add_argument("--candidate-limit", type=int, default=5000)
    ap.add_argument("--objects-per-filter", type=int, default=3)
    ap.add_argument(
        "--include-object",
        action="append",
        default=[],
        help=(
            "Attempt to include this object (MPC designation/provid) for every (station, filter_id) "
            "if it has enough observations AND has H in both MPC and JPL and is an asteroid. "
            "May be repeated. Example for Bennu: --include-object 101955 --include-object '1999 RQ36'"
        ),
    )
    ap.add_argument(
        "--since-days",
        type=int,
        default=None,
        help="If set, restrict candidate selection to observations from the last N days.",
    )
    ap.add_argument(
        "--check-distinct-bands",
        action="store_true",
        help="If set, run an extra DISTINCT band query per station (extra BigQuery cost).",
    )
    ap.add_argument(
        "--allow-missing-filters",
        action="store_true",
        help="If set, do not fail when a mapped filter_id has no data in BigQuery for that station.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing fixture files if present.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = FixtureSelectionConfig(
        min_arc_days=int(args.min_arc_days),
        min_obs_per_filter=int(args.min_obs_per_filter),
        max_obs_per_band_in_fixture=int(args.max_obs_per_filter),
        candidate_limit=int(args.candidate_limit),
        since_days=(int(args.since_days) if args.since_days is not None else None),
    )

    client = BigQueryMPCClient(
        dataset_id=str(args.dataset_id), views_dataset_id=str(args.views_dataset_id)
    )

    codes = [str(c).strip() for c in args.codes if str(c).strip()]
    filter_map = target_filter_map_by_code(codes)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for code in codes:
        print("=" * 88)
        print(f"Station {code}")

        filter_id_to_bands = filter_map[code]
        required_filter_ids = set(filter_id_to_bands.keys())
        if args.check_distinct_bands:
            distinct_bands = query_distinct_bands(client, code)
            print(f"Distinct reported bands in BigQuery: {sorted(distinct_bands)}")

        candidates = query_candidate_objects(
            client,
            code=code,
            filter_id_to_bands=filter_id_to_bands,
            min_arc_days=cfg.min_arc_days,
            # Keep candidates broad: we want "top 3 even if <min" fallback.
            min_obs_per_filter=1,
            limit=cfg.candidate_limit,
            since_days=cfg.since_days,
        )
        print(f"Candidate objects returned: {len(candidates)}")
        if not candidates:
            msg = f"No candidates found for {code}"
            if args.allow_missing_filters:
                print("WARNING:", msg)
                continue
            raise RuntimeError(msg)

        # For each (code, filter_id), choose up to N objects, preferring those with >= min obs
        # in that filter. If fewer than N satisfy the min, fall back to top-N by count.
        n_pick = int(args.objects_per_filter)
        if n_pick <= 0:
            raise ValueError("--objects-per-filter must be > 0")

        for filter_id in sorted(required_filter_ids):
            # ------------------------------------------------------------------
            # Optional: attempt to force-include specific objects (e.g., Bennu)
            # without reducing the normal per-filter picks.
            # ------------------------------------------------------------------
            include_objects = [
                str(x).strip() for x in (args.include_object or []) if str(x).strip()
            ]
            for obj0 in include_objects:
                out_path = (
                    out_dir
                    / f"mpc_magnitude_fixture_{code}_{filter_id}_{slugify_object_id(obj0)}.npz"
                )
                if out_path.exists() and not args.overwrite:
                    # If we're not overwriting, don't fail the whole run; just skip.
                    continue
                try:
                    build_fixture_for_object(
                        client,
                        object_id=obj0,
                        station_code=code,
                        required_filter_ids={filter_id},
                        filter_id_to_bands=filter_id_to_bands,
                        min_obs_per_filter=cfg.min_obs_per_filter,
                        max_obs_per_filter=cfg.max_obs_per_band_in_fixture,
                        out_path=out_path,
                    )
                    print(
                        f"Wrote forced-included fixture: {out_path.name} "
                        f"(station={code}, filter_id={filter_id}, object_id={obj0})"
                    )
                except Exception as e:
                    # Only warn; this is best-effort.
                    print(
                        f"WARNING: could not include {obj0} for {code} {filter_id}: {e}"
                    )

            scored = []
            for c in candidates:
                n = int(c.obs_counts_by_filter_id.get(filter_id, 0))
                if n <= 0:
                    continue
                scored.append((n, c.arc_days, c.object_id))

            if not scored:
                msg = f"No candidates with any observations for {code} filter_id={filter_id}"
                if args.allow_missing_filters:
                    print("WARNING:", msg)
                    continue
                raise RuntimeError(msg)

            scored.sort(key=lambda t: (-t[0], -t[1], t[2]))

            strong = [t for t in scored if t[0] >= cfg.min_obs_per_filter]
            weak = [t for t in scored if t[0] < cfg.min_obs_per_filter]
            ranked = strong + weak

            built = 0
            for n_obs, arc_days, object_id in ranked:
                if built >= n_pick:
                    break
                out_path = (
                    out_dir
                    / f"mpc_magnitude_fixture_{code}_{filter_id}_{slugify_object_id(object_id)}.npz"
                )
                if out_path.exists() and not args.overwrite:
                    raise RuntimeError(
                        f"Fixture already exists: {out_path}. Re-run with --overwrite."
                    )

                try:
                    build_fixture_for_object(
                        client,
                        object_id=object_id,
                        station_code=code,
                        required_filter_ids={filter_id},
                        filter_id_to_bands=filter_id_to_bands,
                        min_obs_per_filter=cfg.min_obs_per_filter,
                        max_obs_per_filter=cfg.max_obs_per_band_in_fixture,
                        out_path=out_path,
                    )
                except Exception as e:
                    # Requirement: asteroid-only, and must have H in BOTH MPC and JPL.
                    # If this candidate fails those constraints, skip and continue.
                    s = str(e)
                    if (
                        "Missing JPL SBDB H" in s
                        or "Missing MPC H" in s
                        or "not an asteroid" in s
                        or "SBDB did not provide H" in s
                    ):
                        print(
                            f"WARNING: skipping {code} {filter_id} {object_id} "
                            f"(must have H in both MPC+JPL and be asteroid): {e}"
                        )
                        continue
                    print(
                        f"WARNING: strict fixture build failed for {code} {filter_id} {object_id} "
                        f"(n_obs_in_candidate={n_obs}, arc_days={arc_days}): {e}"
                    )
                    try:
                        build_fixture_for_object(
                            client,
                            object_id=object_id,
                            station_code=code,
                            required_filter_ids={filter_id},
                            filter_id_to_bands=filter_id_to_bands,
                            min_obs_per_filter=1,
                            max_obs_per_filter=cfg.max_obs_per_band_in_fixture,
                            out_path=out_path,
                        )
                    except Exception as e2:
                        s2 = str(e2)
                        if (
                            "Missing JPL SBDB H" in s2
                            or "Missing MPC H" in s2
                            or "not an asteroid" in s2
                            or "SBDB did not provide H" in s2
                        ):
                            print(
                                f"WARNING: skipping {code} {filter_id} {object_id} "
                                f"(must have H in both MPC+JPL and be asteroid): {e2}"
                            )
                            continue
                        raise

                print(
                    f"Wrote fixture: {out_path.name} "
                    f"(filter_id={filter_id}, candidate_n={n_obs}, candidate_arc_days={arc_days})"
                )
                built += 1

            if built < n_pick:
                msg = (
                    f"Only built {built}/{n_pick} fixtures for station {code} filter_id={filter_id}. "
                    f"(Common cause: candidates missing MPC+JPL H or are comets.)"
                )
                if args.allow_missing_filters:
                    print("WARNING:", msg)
                else:
                    raise RuntimeError(msg)

        print(f"Completed fixtures for {code}: {sorted(required_filter_ids)}")


if __name__ == "__main__":
    main()
