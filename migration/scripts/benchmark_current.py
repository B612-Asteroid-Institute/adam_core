"""Benchmark the current adam-core implementation without a legacy runtime.

The suite reuses the canonical ``API_MIGRATIONS`` registry, parity workload
generators, tiny/small/large lane shapes, current public runner, and genuine
Rust-owned ``std::time::Instant`` adapters. It never imports or invokes the
frozen Python oracle and does not read a legacy timing cache.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from adam_core._rust.status import API_MIGRATIONS, API_MIGRATIONS_BY_ID
from migration.parity import _inputs, _native_rust_runner, _threading
from migration.parity.parity_speed import (
    CANONICAL_SPEED_TRIALS,
    SPEED_TIMING_AGGREGATION,
    SpeedLane,
    _shape_json,
    _shape_label,
    _shape_rows,
    _time_rust_trials,
    _timing_summary,
    build_speed_lanes,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "migration" / "artifacts" / "benchmark_current_core.json"
DEFAULT_MARKDOWN = ROOT / "migration" / "artifacts" / "benchmark_current_core.md"
LANE_ALIASES = {
    "tiny": "tiny-n",
    "tiny-n": "tiny-n",
    "small": "small-n",
    "small-n": "small-n",
    "large": "large-n",
    "large-n": "large-n",
}


def _package_version() -> str:
    try:
        return metadata.version("adam-core")
    except metadata.PackageNotFoundError:
        return "editable"


def _duration(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 1e-3:
        return f"{value * 1e6:.1f} µs"
    if value < 1.0:
        return f"{value * 1e3:.2f} ms"
    return f"{value:.3f} s"


def _selected_api_ids(
    requested: list[str] | None, domains: list[str] | None
) -> list[str]:
    all_ids = [migration.api_id for migration in API_MIGRATIONS]
    if requested:
        unknown = sorted(set(requested) - set(all_ids))
        if unknown:
            raise ValueError(f"unknown API ids: {unknown}")
        all_ids = [api_id for api_id in all_ids if api_id in requested]
    if domains:
        prefixes = tuple(f"{domain}." for domain in domains)
        all_ids = [api_id for api_id in all_ids if api_id.startswith(prefixes)]
    if not all_ids:
        raise ValueError("no APIs selected")
    return all_ids


def _lanes(args: argparse.Namespace) -> list[SpeedLane]:
    if args.quick:
        tiny_reps = small_reps = large_reps = 3
        tiny_warmup = small_warmup = large_warmup = 1
    else:
        tiny_reps = args.tiny_reps
        small_reps = args.small_reps
        large_reps = args.large_reps
        tiny_warmup = args.tiny_warmup
        small_warmup = args.small_warmup
        large_warmup = args.large_warmup
    lanes = build_speed_lanes(
        n=args.small_n,
        reps=small_reps,
        warmup=small_warmup,
        measure_cold=False,
        include_tiny=True,
        tiny_reps=tiny_reps,
        tiny_warmup=tiny_warmup,
        include_large=True,
        large_reps=large_reps,
        large_warmup=large_warmup,
        large_enforced=False,
    )
    requested = {LANE_ALIASES[lane] for lane in args.lanes}
    return [lane for lane in lanes if lane.name in requested]


def _measure_row(
    api_id: str,
    lane: SpeedLane,
    *,
    seed: int,
    trials: int,
) -> dict[str, Any]:
    workload = lane.workload_for(api_id)
    shape = _shape_json(workload)
    migration = API_MIGRATIONS_BY_ID[api_id]
    rng = np.random.default_rng(seed)
    try:
        sample = _inputs.make(api_id, rng, workload)
        current_trials = _time_rust_trials(
            api_id,
            sample.rust_kwargs,
            reps=lane.reps,
            warmup=lane.warmup,
            trials=trials,
        )
        current = _timing_summary(current_trials)
        native_timing = _native_rust_runner.measure(
            api_id,
            sample.rust_kwargs,
            reps=lane.reps,
            warmup=lane.warmup,
            trials=trials,
        )
        native = (
            _timing_summary(native_timing.sample_trials_s)
            if native_timing.sample_trials_s
            else None
        )
        current_p50 = float(current["p50_s"])
        current_p95 = float(current["p95_s"])
        native_p50 = float(native["p50_s"]) if native else None
        native_p95 = float(native["p95_s"]) if native else None
        return {
            "api_id": api_id,
            "domain": api_id.split(".", 1)[0],
            "lane": lane.name,
            "lane_description": lane.description,
            "workload_shape": shape,
            "workload_label": _shape_label(workload),
            "rows": _shape_rows(workload),
            "reps": lane.reps,
            "warmup": lane.warmup,
            "timing_trials": trials,
            "timing_aggregation": SPEED_TIMING_AGGREGATION,
            "current_boundary": migration.boundary,
            "current_status": migration.status,
            "current_entrypoint": migration.rust_module,
            "current_python": current,
            "native_rust": {
                "status": native_timing.status,
                "entrypoint": native_timing.entrypoint,
                "timing_boundary": native_timing.timing_boundary,
                "reason": native_timing.reason,
                "todo": native_timing.todo,
                **(
                    {
                        "p50_s": native_p50,
                        "p95_s": native_p95,
                        "p50_trials_s": native["p50_trials_s"],
                        "p95_trials_s": native["p95_trials_s"],
                        "sample_trials_s": native["sample_trials_s"],
                    }
                    if native
                    else {}
                ),
            },
            "public_over_native": {
                "p50": (
                    current_p50 / native_p50
                    if native_p50 is not None and native_p50 > 0.0
                    else None
                ),
                "p95": (
                    current_p95 / native_p95
                    if native_p95 is not None and native_p95 > 0.0
                    else None
                ),
            },
            "error": None,
        }
    except Exception as exc:
        return {
            "api_id": api_id,
            "domain": api_id.split(".", 1)[0],
            "lane": lane.name,
            "lane_description": lane.description,
            "workload_shape": shape,
            "workload_label": _shape_label(workload),
            "rows": _shape_rows(workload),
            "reps": lane.reps,
            "warmup": lane.warmup,
            "timing_trials": trials,
            "current_boundary": migration.boundary,
            "current_status": migration.status,
            "current_entrypoint": migration.rust_module,
            "current_python": None,
            "native_rust": {"status": "unavailable"},
            "public_over_native": {"p50": None, "p95": None},
            "error": f"{type(exc).__name__}: {exc}",
        }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Current adam-core benchmark suite",
        "",
        (
            f"{payload['api_count']} registered APIs × {payload['lane_count']} lanes = "
            f"{len(payload['rows'])} workload rows. No frozen Python runtime or "
            "legacy timing cache is used."
        ),
        "",
        "| API | Lane and workload | Current public p50/p95 | Native Rust p50/p95 | Public/native p50/p95 | Native evidence |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        current = row.get("current_python") or {}
        native = row.get("native_rust") or {}
        overhead = row.get("public_over_native") or {}
        evidence = (
            f"measured (`{native.get('entrypoint', '')}`)"
            if native.get("status") == "measured"
            else f"unavailable ({native.get('todo') or native.get('reason') or 'no adapter'})"
        )
        error = f"; ERROR {row['error']}" if row.get("error") else ""
        p50_overhead = overhead.get("p50")
        p95_overhead = overhead.get("p95")
        lines.append(
            f"| `{row['api_id']}` | `{row['lane']}`: {row['workload_label']} "
            f"| {_duration(current.get('p50_s'))} / {_duration(current.get('p95_s'))} "
            f"| {_duration(native.get('p50_s'))} / {_duration(native.get('p95_s'))} "
            f"| {f'{p50_overhead:.2f}×' if p50_overhead is not None else '—'} / "
            f"{f'{p95_overhead:.2f}×' if p95_overhead is not None else '—'} "
            f"| {evidence}{error} |"
        )
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--apis", nargs="*")
    parser.add_argument("--domains", nargs="*")
    parser.add_argument(
        "--lanes",
        nargs="+",
        choices=tuple(LANE_ALIASES),
        default=["tiny", "small", "large"],
    )
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument(
        "--threads", choices=("single", "multi-thread"), default="multi-thread"
    )
    parser.add_argument("--trials", type=int, default=CANONICAL_SPEED_TRIALS)
    parser.add_argument("--small-n", type=int, default=2000)
    parser.add_argument("--tiny-reps", type=int, default=101)
    parser.add_argument("--tiny-warmup", type=int, default=3)
    parser.add_argument("--small-reps", type=int, default=21)
    parser.add_argument("--small-warmup", type=int, default=3)
    parser.add_argument("--large-reps", type=int, default=7)
    parser.add_argument("--large-warmup", type=int, default=1)
    parser.add_argument(
        "--require-native",
        action="store_true",
        help="Fail if any selected row lacks genuine Rust-owned timing.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use three repetitions and one warmup per lane for local smoke runs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.trials < 1:
        raise ValueError("--trials must be at least 1")
    _threading.apply_thread_mode(args.threads)
    api_ids = _selected_api_ids(args.apis, args.domains)
    lanes = _lanes(args)
    rows = [
        _measure_row(api_id, lane, seed=args.seed, trials=args.trials)
        for lane in lanes
        for api_id in api_ids
    ]
    payload = {
        "schema_version": 1,
        "benchmark_id": "adam-core-current-only",
        "comparison_mode": "current_public_and_native_rust_no_legacy",
        "legacy_timing_included": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "package_version": _package_version(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "thread_mode": args.threads,
            "thread_env": _threading.snapshot_thread_env(),
        },
        "api_count": len(api_ids),
        "lane_count": len(lanes),
        "selected_apis": api_ids,
        "selected_lanes": [lane.name for lane in lanes],
        "error_count": sum(row["error"] is not None for row in rows),
        "native_measured_count": sum(
            row["native_rust"].get("status") == "measured" for row in rows
        ),
        "native_unavailable_count": sum(
            row["native_rust"].get("status") != "measured" for row in rows
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(payload))
    print(args.output)
    print(args.markdown)
    failed = payload["error_count"] > 0 or (
        args.require_native and payload["native_unavailable_count"] > 0
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
