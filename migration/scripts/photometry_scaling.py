"""Stepped batch-size sweep for the photometry kernels (and reference APIs).

Runs each API at n ∈ {10, 100, 1k, 10k, 100k} via the parity_speed
machinery so the rust-vs-legacy curve is visible across orders of
magnitude. Writes a single JSON to migration/artifacts/.

Defaults are conservative (5 reps, 1 warmup). Cold-call latency is
NOT measured here — that's a fixed ~70 ms (rust) vs ~2 s (legacy)
process+import floor across all APIs and was already characterized
in parity_speed_cold_warm.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from migration.parity import parity_speed


DEFAULT_APIS = (
    # Element-wise, transcendental-heavy (the visible problem):
    "photometry.calculate_phase_angle",
    "photometry.calculate_apparent_magnitude_v",
    "photometry.calculate_apparent_magnitude_v_and_phase_angle",
    "photometry.predict_magnitudes",
    # Reference: another simple element-wise (sqrt + atan2 + asin):
    "coordinates.cartesian_to_spherical",
    # Reference: control-flow heavy (chi solver, ~120 iters):
    "dynamics.propagate_2body",
)

DEFAULT_NS = (10, 100, 1_000, 10_000, 100_000)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Photometry batch-size scaling sweep.")
    p.add_argument("--apis", nargs="*", default=list(DEFAULT_APIS),
                   help="API ids to sweep (default: photometry + refs).")
    p.add_argument("--ns", nargs="*", type=int, default=list(DEFAULT_NS),
                   help="Batch sizes (default: 10 100 1000 10000 100000).")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260425)
    p.add_argument("--output", type=Path,
                   default=Path("migration/artifacts/photometry_scaling.json"))
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    artifact: dict = {"apis": {}}

    for api in args.apis:
        rows = []
        for n in args.ns:
            r = parity_speed.measure(
                api, n=n, reps=args.reps, warmup=args.warmup, seed=args.seed
            )
            rows.append({
                "n": r.n,
                "rust_p50_s": r.rust_p50,
                "legacy_p50_s": r.legacy_p50,
                "speedup_p50": r.speedup_p50,
                "rust_p95_s": r.rust_p95,
                "legacy_p95_s": r.legacy_p95,
                "speedup_p95": r.speedup_p95,
                "passed": r.passed,
                "error": r.error,
            })
            print(
                f"{api:50s}  n={n:>7}  "
                f"rust={r.rust_p50*1e6:>10.1f}μs  "
                f"leg={r.legacy_p50*1e6:>10.1f}μs  "
                f"×{r.speedup_p50:>6.2f}",
                flush=True,
            )
        artifact["apis"][api] = rows

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2))
    print(f"\nwrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
