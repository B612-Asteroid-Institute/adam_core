"""Capture parity/performance evidence for upstream 9b756803 additions.

The historical 44-API benchmark oracle remains pinned at 936cc636. This
separate conversion fixture uses the exact upstream source tree at 9b756803 for
Obs80, Scout snapshots/orbits, and Trajectory, which did not exist at the older
oracle commit. Candidate deterministic Scout payloads are injected without
network access; the migrated side exercises the Rust recorded-response path.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
ORACLE_COMMIT = "9b756803ab3afbe11e33df9e57d30a28e7976b92"
DEFAULT_ORACLE = Path("/tmp/adam-core-latest-main-oracle")
DEFAULT_OUTPUT = ROOT / "migration" / "artifacts" / "latest_main_additions_parity.json"

OBS80_LINES = [
    "     A11EpSe*0C2026 07 08.17725719 41 24.185-30 19 19.42         19.35oVNEOCPW68",
    "     A11EpSe KC2026 07 14.53636 19 37 22.30 -29 16 44.5          19.0 GVNEOCPE23",
]


def scout_observation_payload(repeat: int = 1) -> dict[str, Any]:
    lines = OBS80_LINES * repeat
    return {
        "objectName": "A11EpSe",
        "nObs": len(lines) - 1,
        "lastRun": "2026-07-14 13:33",
        "signature": {"version": "1.3", "source": "NASA/JPL Scout API"},
        "fileMPC": "\n".join(lines) + "\n",
    }


def scout_orbit_payload() -> dict[str, Any]:
    return {
        "signature": {"version": "1.3", "source": "NASA/JPL Scout API"},
        "orbits": {
            "fields": [
                "idx",
                "epoch",
                "ec",
                "qr",
                "tp",
                "om",
                "w",
                "inc",
                "H",
                "dca",
                "tca",
                "moid",
                "vinf",
                "geoEcc",
                "impFlag",
            ],
            "data": [
                [
                    0,
                    "2461234.5",
                    "0.2",
                    "1.0",
                    "2461200.0",
                    "10.0",
                    "20.0",
                    "5.0",
                    "22.0",
                    None,
                    None,
                    "0.01",
                    None,
                    "0.2",
                    0,
                ]
            ],
        },
    }


def median_seconds(run: Callable[[], Any], *, reps: int = 20, warmup: int = 3) -> float:
    for _ in range(warmup):
        run()
    samples = []
    for _ in range(reps):
        started = time.perf_counter()
        run()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def serialize_obs80(table: Any) -> dict[str, Any]:
    return {
        "raw_line": table.raw_line.to_pylist(),
        "designation": table.designation.to_pylist(),
        "discovery": table.discovery.to_pylist(),
        "note1": table.note1.to_pylist(),
        "note2": table.note2.to_pylist(),
        "observatory_code": table.observatory_code.to_pylist(),
        "time_scale": table.time.scale,
        "time_days": table.time.days.to_pylist(),
        "time_nanos": table.time.nanos.to_pylist(),
        "ra_deg": table.ra_deg.to_pylist(),
        "dec_deg": table.dec_deg.to_pylist(),
        "mag": table.mag.to_pylist(),
        "band": table.band.to_pylist(),
        "astrometric_catalog": table.astrometric_catalog.to_pylist(),
        "reference": table.reference.to_pylist(),
    }


def serialize_scout_observations(table: Any) -> dict[str, Any]:
    return {
        "object_id": table.object_id.to_pylist(),
        "solution_date_utc": table.solution_date_utc.to_pylist(),
        "declared_n_obs": table.declared_n_obs.to_pylist(),
        "snapshot_sha256": table.snapshot_sha256.to_pylist(),
        "snapshot_observation_count": table.snapshot_observation_count.to_pylist(),
        "signature_version": table.signature_version.to_pylist(),
        "signature_source": table.signature_source.to_pylist(),
        "observation_index": table.observation_index.to_pylist(),
        "observation": serialize_obs80(table.observation),
    }


def serialize_variants(table: Any) -> dict[str, Any]:
    return {
        "orbit_id": table.orbit_id.to_pylist(),
        "object_id": table.object_id.to_pylist(),
        "variant_id": table.variant_id.to_pylist(),
        "values": table.coordinates.values.tolist(),
        "time_days": table.coordinates.time.days.to_pylist(),
        "time_nanos": table.coordinates.time.nanos.to_pylist(),
        "time_scale": table.coordinates.time.scale,
        "origin": table.coordinates.origin.code.to_pylist(),
        "frame": table.coordinates.frame,
    }


def build_trajectory(rows: int = 256) -> Any:
    from adam_core.coordinates import CartesianCoordinates
    from adam_core.coordinates.origin import Origin
    from adam_core.orbits.orbits import Orbits
    from adam_core.orbits.trajectory import Trajectory
    from adam_core.time import Timestamp

    starts = [59000.0 + 2.0 * row for row in range(rows)]
    ends = [value + 2.0 for value in starts]
    epochs = [value + 1.0 for value in starts]
    orbits = Orbits.from_kwargs(
        orbit_id=[f"A-{row}" for row in range(rows)],
        object_id=["A"] * rows,
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0] * rows,
            y=[0.1] * rows,
            z=[0.0] * rows,
            vx=[0.0] * rows,
            vy=[0.017] * rows,
            vz=[0.0] * rows,
            time=Timestamp.from_mjd(epochs, scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN"] * rows),
            frame="ecliptic",
        ),
    )
    return Trajectory.from_kwargs(
        object_id=["A"] * rows,
        segment_id=[f"A.seg{row}" for row in range(rows)],
        coverage_start=Timestamp.from_mjd(starts, scale="tdb"),
        coverage_end=Timestamp.from_mjd(ends, scale="tdb"),
        orbit=orbits,
        source=["fixture"] * rows,
        source_version=["9b756803"] * rows,
        max_propagation_days=[2.0] * rows,
        is_maneuver_boundary=[False] * rows,
    )


def error_message(run: Callable[[], Any]) -> str | None:
    try:
        run()
    except Exception as error:  # fixture intentionally records public errors
        return f"{type(error).__name__}: {error}"
    return None


def worker(mode: str) -> dict[str, Any]:
    import numpy as np

    from adam_core.observations.obs80 import (
        ScoutObservations,
        parse_optical_obs80,
        parse_optical_obs80_file,
    )
    from adam_core.orbits.query.scout import query_scout, query_scout_observations

    raw = "\n".join(OBS80_LINES) + "\n"
    obs80 = parse_optical_obs80_file(raw)
    malformed = OBS80_LINES[0][:15] + "not a valid date!" + OBS80_LINES[0][32:]

    trajectory = build_trajectory()
    trajectory.validate_coverage()
    trajectory_result = {
        "coverage_start_mjd": trajectory.coverage_start_mjd().tolist(),
        "coverage_end_mjd": trajectory.coverage_end_mjd().tolist(),
        "epoch_mjd": trajectory.epoch_mjd().tolist(),
        "object_ids": trajectory.object_ids(),
        "segment_at_start": trajectory.segment_for(59000.0).segment_id.to_pylist(),
        "segment_at_boundary": trajectory.segment_for(59002.0).segment_id.to_pylist(),
        "gap_before": trajectory.segment_for(58999.0) is None,
        "last_end_excluded": trajectory.segment_for(59512.0) is None,
    }

    invalid = build_trajectory(1)
    invalid = type(invalid).from_kwargs(
        object_id=invalid.object_id,
        segment_id=invalid.segment_id,
        coverage_start=invalid.coverage_start,
        coverage_end=invalid.coverage_start,
        orbit=invalid.orbit,
        source=invalid.source,
        source_version=invalid.source_version,
        max_propagation_days=invalid.max_propagation_days,
        is_maneuver_boundary=invalid.is_maneuver_boundary,
    )
    trajectory_result["invalid_window_error"] = error_message(invalid.validate_coverage)
    empty = type(trajectory).empty()
    trajectory_result["empty"] = {
        "coverage_start_mjd": empty.coverage_start_mjd().tolist(),
        "coverage_end_mjd": empty.coverage_end_mjd().tolist(),
        "epoch_mjd": empty.epoch_mjd().tolist(),
        "object_ids": empty.object_ids(),
        "validate_returns_self": empty.validate_coverage() is empty,
        "segment_is_none": empty.segment_for(59000.0) is None,
    }

    observation_payload = scout_observation_payload()
    orbit_payload = scout_orbit_payload()

    class Response:
        def __init__(self, payload: dict[str, Any]):
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self.payload

    scout_observations_fallback = query_scout_observations(
        ["A11EpSe"], http_get=lambda *args, **kwargs: Response(observation_payload)
    )
    scout_orbits_fallback = query_scout(
        ["A11EpSe"], http_get=lambda *args, **kwargs: Response(orbit_payload)
    )

    def scout_observations_timing_run() -> Any:
        return query_scout_observations(
            ["A11EpSe"],
            http_get=lambda *args, **kwargs: Response(observation_payload),
        )

    def scout_orbits_timing_run() -> Any:
        return query_scout(
            ["A11EpSe"], http_get=lambda *args, **kwargs: Response(orbit_payload)
        )

    native: dict[str, Any] | None = None
    native_timings: dict[str, Any] | None = None
    if mode == "current":
        from adam_core import _rust_native
        from adam_core._rust.arrow import table_from_record_batch
        from adam_core.orbits.variants import VariantOrbits

        native_observations = table_from_record_batch(
            ScoutObservations,
            _rust_native.query_scout_observations_arrow(
                ["A11EpSe"], [json.dumps(observation_payload)]
            ),
        )
        native_orbits = table_from_record_batch(
            VariantOrbits,
            _rust_native.query_scout_arrow(["A11EpSe"], [json.dumps(orbit_payload)]),
        )
        native = {
            "scout_observations": serialize_scout_observations(native_observations),
            "scout_orbits": serialize_variants(native_orbits),
        }

        def native_scout_observations() -> Any:
            return table_from_record_batch(
                ScoutObservations,
                _rust_native.query_scout_observations_arrow(
                    ["A11EpSe"], [json.dumps(observation_payload)]
                ),
            )

        def native_scout_orbits() -> Any:
            return table_from_record_batch(
                VariantOrbits,
                _rust_native.query_scout_arrow(
                    ["A11EpSe"], [json.dumps(orbit_payload)]
                ),
            )

        scout_observations_timing_run = native_scout_observations
        scout_orbits_timing_run = native_scout_orbits
        batch = trajectory._native_batch()
        native_timings = {
            "obs80_seconds": _rust_native.benchmark_parse_optical_obs80(
                raw * 100, 5, 3, 1, True, True
            ),
            "scout_observations_seconds": _rust_native.benchmark_query_client_processing(
                "scout-observations", [json.dumps(observation_payload)], 5, 3, 1
            ),
            "scout_orbits_seconds": _rust_native.benchmark_query_client_processing(
                "scout", [json.dumps(orbit_payload)], 5, 3, 1
            ),
            "trajectory_validate_seconds": _rust_native.benchmark_trajectory_arrow(
                batch, "validate_coverage", 5, 3, 1
            ),
            "trajectory_segment_seconds": _rust_native.benchmark_trajectory_arrow(
                batch, "segment_for", 5, 3, 1, 59001.0, "A"
            ),
        }

    return {
        "mode": mode,
        "obs80": serialize_obs80(obs80),
        "obs80_errors": {
            "short": error_message(lambda: parse_optical_obs80("too short")),
            "date": error_message(lambda: parse_optical_obs80(malformed)),
        },
        "trajectory": trajectory_result,
        "scout_fallback": {
            "observations": serialize_scout_observations(scout_observations_fallback),
            "orbits": serialize_variants(scout_orbits_fallback),
        },
        "native": native,
        "native_timings": native_timings,
        "python_timings": {
            "obs80_file_seconds": median_seconds(
                lambda: parse_optical_obs80_file(raw * 100), reps=15
            ),
            "trajectory_validate_seconds": median_seconds(
                trajectory.validate_coverage, reps=15
            ),
            "trajectory_segment_seconds": median_seconds(
                lambda: trajectory.segment_for(59200.0, object_id="A"), reps=15
            ),
            "scout_observations_processing_seconds": median_seconds(
                scout_observations_timing_run, reps=15
            ),
            "scout_orbits_processing_seconds": median_seconds(
                scout_orbits_timing_run, reps=15
            ),
        },
        "numpy_version": np.__version__,
    }


def run_worker(mode: str, oracle: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    if mode == "oracle":
        environment["PYTHONPATH"] = str(oracle / "src")
    else:
        environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", mode],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def max_float_difference(left: Any, right: Any) -> float:
    import numpy as np

    return float(
        np.max(
            np.abs(
                np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
            )
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["oracle", "current"])
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker), allow_nan=False))
        return 0
    if not (args.oracle / "src" / "adam_core").exists():
        parser.error(
            f"oracle source is missing at {args.oracle}; extract git archive {ORACLE_COMMIT}"
        )

    oracle = run_worker("oracle", args.oracle)
    current = run_worker("current", args.oracle)
    obs80_exact = current["obs80"] == oracle["obs80"]
    obs80_errors_exact = current["obs80_errors"] == oracle["obs80_errors"]
    trajectory_exact = current["trajectory"] == oracle["trajectory"]
    scout_observations_exact = (
        current["native"]["scout_observations"]
        == oracle["scout_fallback"]["observations"]
    )
    scout_orbits_oracle = oracle["scout_fallback"]["orbits"]
    scout_orbits_current = current["native"]["scout_orbits"]
    scout_orbit_metadata_exact = {
        key: scout_orbits_current[key] == scout_orbits_oracle[key]
        for key in scout_orbits_oracle
        if key != "values"
    }
    scout_orbit_max_abs = max_float_difference(
        scout_orbits_current["values"], scout_orbits_oracle["values"]
    )
    speedups = {
        key: oracle["python_timings"][key] / current["python_timings"][key]
        for key in oracle["python_timings"]
    }
    parity = {
        "obs80_exact": obs80_exact,
        "obs80_errors_exact": obs80_errors_exact,
        "trajectory_exact": trajectory_exact,
        "scout_observations_exact": scout_observations_exact,
        "scout_orbit_metadata_exact": scout_orbit_metadata_exact,
        "scout_orbit_max_abs_difference": scout_orbit_max_abs,
    }
    passed = (
        obs80_exact
        and obs80_errors_exact
        and trajectory_exact
        and scout_observations_exact
        and all(scout_orbit_metadata_exact.values())
        and scout_orbit_max_abs <= 1e-14
    )
    output = {
        "schema_version": 1,
        "oracle_commit": ORACLE_COMMIT,
        "current_git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "surface": [
            "parse_optical_obs80",
            "parse_optical_obs80_file",
            "query_scout",
            "query_scout_observations",
            "Trajectory.coverage_start_mjd",
            "Trajectory.coverage_end_mjd",
            "Trajectory.epoch_mjd",
            "Trajectory.object_ids",
            "Trajectory.validate_coverage",
            "Trajectory.segment_for",
        ],
        "parity": parity,
        "python_control_timings": {
            "oracle_seconds": oracle["python_timings"],
            "current_seconds": current["python_timings"],
            "speedup": speedups,
        },
        "native_rust_instant_timings": current["native_timings"],
        "status": "passed" if passed else "failed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {args.output}: {output['status']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
