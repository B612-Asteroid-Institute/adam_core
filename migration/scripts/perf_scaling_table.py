"""Scaling sweep — every wired rust-default API at multiple batch sizes.

For each API in `migration.parity._inputs.GENERATORS`, run the
`parity_speed.measure` machinery at n ∈ {10, 100, 1k, 10k, 100k}
(configurable). Emit a markdown table:

    | API | n=10 | n=100 | n=1k | n=10k | n=100k |
    | --- | --- | --- | --- | --- | --- |
    | api  | rust/legacy/×  | … |

`rust/legacy/×` shows rust p50, legacy p50, and the speedup ratio.

This is the apples-to-apples table the user asked for: same workload
through both paths, real wall-clock at p50.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from migration.parity import _inputs, parity_speed


DEFAULT_NS = (10, 100, 1_000, 10_000, 100_000)

# Per-API N caps when legacy timing becomes prohibitive.
# `propagate_2body_with_covariance` legacy uses JAX
# `transform_covariances_jacobian` per row → tens of seconds at N=10k,
# minutes at N=100k. Capping keeps the wall-clock reasonable while still
# showing the full scaling curve up to where the comparison stays
# meaningful.
PER_API_MAX_N = {
    "dynamics.propagate_2body_with_covariance": 10_000,
    "dynamics.generate_ephemeris_2body_with_covariance": 10_000,
}


def _fmt_time(t: float) -> str:
    if t >= 1.0:
        return f"{t:.2f}s"
    if t >= 1e-3:
        return f"{t * 1e3:.2f}ms"
    return f"{t * 1e6:.1f}μs"


def _fmt_speedup(s: float) -> str:
    if s >= 100:
        return f"{s:.0f}×"
    if s >= 10:
        return f"{s:.1f}×"
    return f"{s:.2f}×"


def _format_markdown(
    api_ids: list[str],
    ns: list[int],
    measurements: dict[tuple[str, int], parity_speed.SpeedResult],
) -> str:
    lines = []
    header = ["| API |"] + [f" n={n} |" for n in ns]
    lines.append("".join(header))
    lines.append("| --- |" + " --- |" * len(ns))
    for api in api_ids:
        row = [f"| `{api}` |"]
        for n in ns:
            r = measurements.get((api, n))
            if r is None or r.error:
                row.append(" — |")
                continue
            cell = (
                f" {_fmt_time(r.rust_p50)} / "
                f"{_fmt_time(r.legacy_p50)} / "
                f"{_fmt_speedup(r.speedup_p50)} |"
            )
            row.append(cell)
        lines.append("".join(row))
    return "\n".join(lines)


def _format_speedup_only_md(
    api_ids: list[str],
    ns: list[int],
    measurements: dict[tuple[str, int], parity_speed.SpeedResult],
) -> str:
    """Compact speedup-only table — easier to read at a glance."""
    lines = []
    header = ["| API |"] + [f" n={n} |" for n in ns]
    lines.append("".join(header))
    lines.append("| --- |" + " ---: |" * len(ns))
    for api in api_ids:
        row = [f"| `{api}` |"]
        for n in ns:
            r = measurements.get((api, n))
            if r is None or r.error:
                row.append(" err |")
                continue
            mark = "" if r.passed else "⚠"
            row.append(f" {_fmt_speedup(r.speedup_p50)}{mark} |")
        lines.append("".join(row))
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Per-API speedup scaling table.")
    p.add_argument("--apis", nargs="*", default=None)
    p.add_argument("--ns", nargs="*", type=int, default=list(DEFAULT_NS))
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260426)
    p.add_argument(
        "--json-output", type=Path,
        default=Path("migration/artifacts/perf_scaling_table.json"),
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    api_ids = args.apis or list(_inputs.all_api_ids())

    measurements: dict[tuple[str, int], parity_speed.SpeedResult] = {}
    for api in api_ids:
        max_n = PER_API_MAX_N.get(api)
        for n in args.ns:
            if max_n is not None and n > max_n:
                continue  # skip pathologically slow legacy cells
            r = parity_speed.measure(
                api, n=n, reps=args.reps, warmup=args.warmup, seed=args.seed
            )
            measurements[(api, n)] = r
            print(
                f"{api:55s} n={n:>7}  "
                f"rust={_fmt_time(r.rust_p50):>10}  "
                f"leg={_fmt_time(r.legacy_p50):>10}  "
                f"×{r.speedup_p50:>5.2f}",
                file=sys.stderr,
                flush=True,
            )

    print("# Apples-to-apples performance scaling — rust vs legacy p50\n")
    print(f"Workload: {args.reps} reps after {args.warmup} warmup, "
          f"seed={args.seed}, dim shown across n.\n")
    print("Each cell: `rust_p50 / legacy_p50 / speedup`. ⚠ flag in compact "
          "view = speedup < 1.2× (gate failure).\n")
    print("## Compact (speedup only)\n")
    print(_format_speedup_only_md(api_ids, args.ns, measurements))
    print("\n## Full (rust / legacy / speedup)\n")
    print(_format_markdown(api_ids, args.ns, measurements))

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps({
        "ns": args.ns,
        "measurements": {
            f"{api}|{n}": parity_speed.to_json([m])["apis"][0]
            for (api, n), m in measurements.items()
        },
    }, indent=2))
    print(f"\nwrote {args.json_output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
