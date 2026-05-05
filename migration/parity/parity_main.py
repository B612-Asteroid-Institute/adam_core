"""Baseline-main parity gate orchestrator.

Runs both ``parity_fuzz`` and ``parity_speed`` over the full API set and
writes a single JSON artifact. Exit code is 0 only if BOTH gates pass
across all APIs.

Usage::

    .venv/bin/python -m migration.parity.parity_main \
        --output migration/artifacts/parity_gate.json

    # Or scope to a single API while iterating on a port:
    .venv/bin/python -m migration.parity.parity_main \
        --apis dynamics.propagate_2body
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from . import _threading


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Baseline-main parity + speedup gate for wired Rust APIs."
    )
    p.add_argument(
        "--apis", nargs="*", default=None, help="Specific API ids (default: ALL)."
    )
    p.add_argument(
        "--fuzz-seeds",
        type=int,
        default=8,
        help="Random seeds per API for parity fuzz (default: 8).",
    )
    p.add_argument(
        "--fuzz-n",
        type=int,
        default=128,
        help="Workload size per parity-fuzz seed (default: 128).",
    )
    p.add_argument(
        "--speed-n",
        type=int,
        default=2000,
        help="Workload size for speed gate (default: 2000).",
    )
    p.add_argument(
        "--speed-reps", type=int, default=7, help="Speed gate reps (default: 7)."
    )
    p.add_argument(
        "--speed-warmup",
        type=int,
        default=1,
        help="Speed gate warmup reps (default: 1).",
    )
    p.add_argument(
        "--speed-tiny",
        action="store_true",
        help="Also measure the enforced tiny-N one-off speed lane.",
    )
    p.add_argument(
        "--speed-tiny-reps",
        type=int,
        default=None,
        help="Timing reps for the tiny speed lane (default: --speed-reps).",
    )
    p.add_argument(
        "--speed-tiny-warmup",
        type=int,
        default=None,
        help="Warmup reps for the tiny speed lane (default: --speed-warmup).",
    )
    p.add_argument(
        "--speed-large",
        action="store_true",
        help="Also measure the enforced API-shaped large-N speed lane.",
    )
    p.add_argument(
        "--speed-large-n",
        type=int,
        default=None,
        help="Override the large speed lane to use this n for every API.",
    )
    p.add_argument(
        "--speed-large-reps",
        type=int,
        default=None,
        help="Timing reps for the large speed lane (default: --speed-reps).",
    )
    p.add_argument(
        "--speed-large-warmup",
        type=int,
        default=None,
        help="Warmup reps for the large speed lane (default: --speed-warmup).",
    )
    p.add_argument(
        "--speed-large-cold",
        action="store_true",
        help="Also collect cold-call timings for the large speed lane.",
    )
    p.add_argument(
        "--speed-large-enforced",
        action="store_true",
        help="Deprecated compatibility flag; large-N is enforced by default.",
    )
    p.add_argument(
        "--speed-large-diagnostic",
        action="store_true",
        help="Ad-hoc escape hatch: measure large-N but exclude it from pass/fail.",
    )
    p.add_argument(
        "--speed-legacy-cache",
        type=Path,
        default=None,
        help=(
            "Optional JSON cache for baseline-main legacy speed timings. Missing "
            "or stale entries fail unless --speed-refresh-legacy-cache is set."
        ),
    )
    p.add_argument(
        "--speed-refresh-legacy-cache",
        action="store_true",
        help=(
            "Recapture missing/requested --speed-legacy-cache entries and merge "
            "them into the existing cache. Existing entries for other lanes/APIs "
            "are preserved."
        ),
    )
    p.add_argument(
        "--speed-replace-legacy-cache",
        action="store_true",
        help=(
            "With --speed-refresh-legacy-cache, discard the existing speed cache "
            "before capturing requested lanes/APIs."
        ),
    )
    p.add_argument(
        "--base-seed",
        type=int,
        default=20260425,
        help="RNG base seed (default: today's date).",
    )
    p.add_argument(
        "--threads",
        choices=("single", "multi-thread", "native"),
        default="single",
        help=(
            "Thread policy for warm speed gate and parity subprocesses "
            "(default: single). Use 'multi-thread' only for separate scaling "
            "artifacts (allows both Rust Rayon and the legacy NumPy/JAX/BLAS "
            "pools to scale across available cores). 'native' is accepted as "
            "a deprecated alias for 'multi-thread'."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("migration/artifacts/parity_gate.json"),
        help="JSON artifact path.",
    )
    p.add_argument(
        "--skip-fuzz",
        action="store_true",
        help="Skip the parity-fuzz gate (speed only).",
    )
    p.add_argument(
        "--skip-speed",
        action="store_true",
        help="Skip the speedup gate (parity-fuzz only).",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    _threading.apply_thread_mode(args.threads)

    from . import _inputs, parity_fuzz, parity_speed

    if args.speed_refresh_legacy_cache and args.speed_legacy_cache is None:
        parser.error("--speed-refresh-legacy-cache requires --speed-legacy-cache")
    if args.speed_replace_legacy_cache and not args.speed_refresh_legacy_cache:
        parser.error(
            "--speed-replace-legacy-cache requires --speed-refresh-legacy-cache"
        )

    api_ids = args.apis or list(_inputs.all_api_ids())

    artifact: dict = {"api_ids": api_ids, "thread_mode": args.threads}
    fuzz_pass = True
    speed_pass = True

    if not args.skip_fuzz:
        print("=" * 72)
        print("PARITY FUZZ GATE")
        print("=" * 72)
        fuzz_results = parity_fuzz.fuzz_all(
            api_ids,
            seeds=args.fuzz_seeds,
            n=args.fuzz_n,
            base_seed=args.base_seed,
        )
        print(parity_fuzz.format_summary(fuzz_results))
        artifact["parity_fuzz"] = parity_fuzz.to_json(fuzz_results)
        fuzz_pass = artifact["parity_fuzz"]["all_passed"]

    if not args.skip_speed:
        print()
        print("=" * 72)
        print("SPEEDUP GATE (lane-specific Rust vs baseline-main thresholds)")
        print("=" * 72)
        legacy_cache = parity_speed.prepare_legacy_timing_cache(
            args.speed_legacy_cache,
            refresh=args.speed_refresh_legacy_cache,
            replace=args.speed_replace_legacy_cache,
        )
        speed_lanes = parity_speed.build_speed_lanes(
            n=args.speed_n,
            reps=args.speed_reps,
            warmup=args.speed_warmup,
            measure_cold=False,
            include_tiny=args.speed_tiny,
            tiny_reps=args.speed_tiny_reps,
            tiny_warmup=args.speed_tiny_warmup,
            include_large=args.speed_large,
            large_n=args.speed_large_n,
            large_reps=args.speed_large_reps,
            large_warmup=args.speed_large_warmup,
            large_cold=args.speed_large_cold,
            large_enforced=args.speed_large_enforced or not args.speed_large_diagnostic,
        )
        speed_results = parity_speed.measure_lanes(
            api_ids,
            speed_lanes,
            seed=args.base_seed,
            thread_mode=args.threads,
            legacy_cache=legacy_cache,
        )
        parity_speed.write_legacy_timing_cache(legacy_cache)
        print(parity_speed.format_summary(speed_results))
        artifact["parity_speed"] = parity_speed.to_json(
            speed_results,
            legacy_cache=legacy_cache,
        )
        speed_pass = artifact["parity_speed"]["all_passed"]

    artifact["all_passed"] = fuzz_pass and speed_pass

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2))
    print(f"\nwrote {args.output}", file=sys.stderr)

    if not artifact["all_passed"]:
        print(
            f"\nGATE FAILED: parity_fuzz={fuzz_pass} parity_speed={speed_pass}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
