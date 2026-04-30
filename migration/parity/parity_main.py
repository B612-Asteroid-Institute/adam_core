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

from . import _inputs, parity_fuzz, parity_speed


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
        "--base-seed",
        type=int,
        default=20260425,
        help="RNG base seed (default: today's date).",
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

    api_ids = args.apis or list(_inputs.all_api_ids())

    artifact: dict = {"api_ids": api_ids}
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
        print("SPEEDUP GATE (>=1.2x Rust vs baseline main)")
        print("=" * 72)
        speed_results = parity_speed.measure_all(
            api_ids,
            n=args.speed_n,
            reps=args.speed_reps,
            warmup=args.speed_warmup,
            seed=args.base_seed,
        )
        print(parity_speed.format_summary(speed_results))
        artifact["parity_speed"] = parity_speed.to_json(speed_results)
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
