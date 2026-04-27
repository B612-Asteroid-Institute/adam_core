"""Generate the parity-tolerance table for every wired rust-default API.

Runs the random-fuzz parity gate once, then joins each result with the
configured per-API tolerance from `migration/parity/tolerances.py`. Emits
a markdown table showing:

    | API | output | atol | rtol | worst_abs | worst_rel | margin | rationale |

`margin` = atol / worst_abs (how much headroom the tolerance leaves).
`rationale` is the short text from `ToleranceSpec.rationale`, truncated.

Also lists `tolerances.TOLERANCES` entries that are NOT wired in the
random-fuzz `GENERATORS`, so the table doubles as a coverage gap report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from migration.parity import _inputs, parity_fuzz, tolerances


def _truncate(text: str, n: int) -> str:
    text = " ".join(text.split())
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


def _build_rows(
    fuzz_results: list[parity_fuzz.ApiResult],
) -> list[dict]:
    rows = []
    by_api = {r.api_id: r for r in fuzz_results}

    measured_ids = set(by_api.keys())
    wired_ids = set(_inputs.all_api_ids())
    declared_ids = set(tolerances.all_api_ids())

    # Measured entries (one row per output)
    for api_id in sorted(measured_ids):
        spec = tolerances.get(api_id)
        api_result = by_api[api_id]
        for out_name, tol in spec.outputs.items():
            worst_abs = 0.0
            worst_rel = 0.0
            for s in api_result.seeds:
                for o in s.outputs:
                    if o.name == out_name:
                        worst_abs = max(worst_abs, o.max_abs)
                        worst_rel = max(worst_rel, o.max_rel)
            margin = (tol.atol / worst_abs) if worst_abs > 0 else float("inf")
            rows.append({
                "api_id": api_id,
                "output": out_name,
                "atol": tol.atol,
                "rtol": tol.rtol,
                "worst_abs": worst_abs,
                "worst_rel": worst_rel,
                "margin": margin,
                "passed": api_result.passed,
                "investigate": spec.investigate,
                "investigate_task": spec.investigate_task,
                "rationale": spec.rationale,
                "dominant_column": spec.dominant_column,
                "physical_magnitude": spec.physical_magnitude,
                "root_cause": spec.root_cause,
                "verdict": spec.verdict,
                "state": "measured",
            })

    # Orchestration APIs whose parity is structurally implied by their
    # underlying kernel entries — recognized by a "ORCHESTRATION" prefix
    # in the rationale string. We surface them as their own row state.
    orchestration_markers = ("ORCHESTRATION",)

    for api_id in sorted(declared_ids - measured_ids):
        spec = tolerances.get(api_id)
        is_orch = any(m in spec.rationale for m in orchestration_markers)
        if is_orch:
            state = "orchestration-implied"
        elif api_id in wired_ids:
            state = "wired-not-measured"
        else:
            state = "unwired"
        for out_name, tol in spec.outputs.items():
            rows.append({
                "api_id": api_id,
                "output": out_name,
                "atol": tol.atol,
                "rtol": tol.rtol,
                "worst_abs": None,
                "worst_rel": None,
                "margin": None,
                "passed": None,
                "investigate": spec.investigate,
                "investigate_task": spec.investigate_task,
                "rationale": spec.rationale,
                "dominant_column": spec.dominant_column,
                "physical_magnitude": spec.physical_magnitude,
                "root_cause": spec.root_cause,
                "verdict": spec.verdict,
                "state": state,
            })
    return rows


def _format_markdown(rows: list[dict]) -> str:
    lines = []
    lines.append(
        "| API | output | atol | rtol | worst_abs | result | dominant col | physical | root cause | verdict |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---|---|---|---|")
    for r in rows:
        if r["state"] != "measured":
            wa = "—"
            result = "—"
        else:
            wa = f"{r['worst_abs']:.2e}" if r["worst_abs"] > 0 else "0"
            # The margin column now shows pass/fail derived from the actual
            # parity check (worst_abs ≤ atol + rtol·|val|). For atol-only
            # outputs (rtol=0) we also show the bare margin atol/worst_abs.
            if r.get("passed") is True:
                if r["rtol"] == 0:
                    if r["worst_abs"] == 0:
                        result = "PASS (∞×)"
                    else:
                        margin_val = r["atol"] / r["worst_abs"]
                        result = f"PASS ({margin_val:.1f}×)"
                else:
                    # rtol-bound — atol/worst_abs alone is misleading.
                    result = "PASS (rtol)"
            elif r.get("passed") is False:
                result = "FAIL"
            else:
                result = "—"
        flag = ""
        if r["state"] == "unwired":
            flag = " (UNWIRED)"
        elif r["state"] == "wired-not-measured":
            flag = " (skipped)"
        elif r["state"] == "orchestration-implied":
            flag = " (orchestration)"
        elif r["investigate"]:
            flag = f" ⚠ {r['investigate_task'] or 'investigate'}"
        rtol_s = f"{r['rtol']:.0e}" if r["rtol"] > 0 else "0"
        dom = r.get("dominant_column") or "—"
        phys = _truncate(r.get("physical_magnitude") or "—", 60)
        rc = _truncate(r.get("root_cause") or _truncate(r["rationale"], 90), 110)
        verdict = r.get("verdict") or "—"
        # Verdict cell: highlight non-bit-parity verdicts
        if verdict and verdict != "—" and not verdict.startswith("bit-parity"):
            verdict = _truncate(verdict, 100)
        lines.append(
            f"| `{r['api_id']}`{flag} "
            f"| {r['output']} "
            f"| {r['atol']:.0e} | {rtol_s} "
            f"| {wa} | {result} "
            f"| {dom} | {phys} | {rc} | {verdict} |"
        )
    return "\n".join(lines)


def _coverage_summary(rows: list[dict]) -> str:
    declared_apis = sorted({r["api_id"] for r in rows})
    measured_apis = sorted({r["api_id"] for r in rows if r["state"] == "measured"})
    wired_not_measured = sorted({
        r["api_id"] for r in rows if r["state"] == "wired-not-measured"
    })
    orchestration = sorted({
        r["api_id"] for r in rows if r["state"] == "orchestration-implied"
    })
    unwired = sorted({r["api_id"] for r in rows if r["state"] == "unwired"})
    flagged = sorted({
        r["api_id"] for r in rows if r["state"] == "measured" and r["investigate"]
    })
    direct = len(measured_apis) + len(wired_not_measured)
    indirect = len(orchestration)
    lines = []
    lines.append(
        f"**Coverage**: {direct} of {len(declared_apis)} declared APIs "
        f"wired directly in random-fuzz GENERATORS; "
        f"{indirect} additional orchestration APIs covered indirectly via "
        f"underlying kernel parity. "
        f"{len(measured_apis)} measured this run."
    )
    if wired_not_measured:
        lines.append("")
        lines.append("**Wired but excluded from this run by `--apis` filter**:")
        for a in wired_not_measured:
            lines.append(f"- `{a}`")
    if orchestration:
        lines.append("")
        lines.append("**Orchestration (covered indirectly via underlying kernel parity)**:")
        for a in orchestration:
            lines.append(f"- `{a}`")
    if unwired:
        lines.append("")
        lines.append("**Declared but UNWIRED in random fuzz** (no parity, action needed):")
        for a in unwired:
            lines.append(f"- `{a}`")
    if flagged:
        lines.append("")
        lines.append("**Flagged (`investigate=True`)**:")
        for a in flagged:
            lines.append(f"- `{a}`")
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parity tolerance + result table.")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--base-seed", type=int, default=20260426)
    p.add_argument("--apis", nargs="*", default=None)
    p.add_argument(
        "--json-output", type=Path, default=None,
        help="Write full row data to this JSON path (in addition to stdout markdown).",
    )
    p.add_argument(
        "--use-cache", type=Path, default=None,
        help="Read fuzz results from this JSON instead of running parity_fuzz.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    api_ids = args.apis or list(_inputs.all_api_ids())

    if args.use_cache and args.use_cache.exists():
        cached = json.loads(args.use_cache.read_text())
        # Reconstruct ApiResult-ish objects for the joiner
        fuzz_results = []
        for entry in cached.get("apis", []):
            seeds = []
            for s in entry["seeds"]:
                outs = [
                    parity_fuzz.OutputResult(
                        name=o["name"], max_abs=o["max_abs"], max_rel=o["max_rel"],
                        atol=o["atol"], rtol=o["rtol"], passed=o["passed"],
                        nan_disagreement=o.get("nan_disagreement", 0),
                    )
                    for o in s["outputs"]
                ]
                seeds.append(parity_fuzz.SeedResult(
                    seed=s["seed"], n=s["n"], outputs=outs, error=s["error"],
                ))
            fuzz_results.append(parity_fuzz.ApiResult(
                api_id=entry["api_id"], seeds=seeds,
                investigate=entry.get("investigate", False),
                investigate_task=entry.get("investigate_task", ""),
            ))
    else:
        fuzz_results = parity_fuzz.fuzz_all(
            api_ids, seeds=args.seeds, n=args.n, base_seed=args.base_seed,
        )

    rows = _build_rows(fuzz_results)

    print("# Parity tolerance + observed-difference table\n")
    print(_coverage_summary(rows))
    print()
    print(_format_markdown(rows))

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json_output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
