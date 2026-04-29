"""Pretty-print parity tolerance/RCA and performance tables.

Reads the current parity artifacts (or runs parity_fuzz when requested),
then joins each result with the configured per-API tolerance from
`migration/parity/tolerances.py`. Emits markdown tables suitable for
handoffs and reviews.

The parity table includes the tolerance/RCA fields:

    | API | output | atol | rtol | worst_abs | worst_rel | result |
    | rationale | physical magnitude | root cause | verdict |

The performance table includes warm/cold speedup and waiver state when a
speed miss is explicitly waived.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from migration.parity import _inputs, parity_fuzz, tolerances

DEFAULT_PARITY_ARTIFACT = Path("migration/artifacts/parity_gate.json")
DEFAULT_SPEED_ARTIFACT = Path("migration/artifacts/parity_speed_cold_warm.json")


def _truncate(text: str, n: int | None) -> str:
    text = " ".join(text.split())
    if n is None or n <= 0:
        return text
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
            rows.append(
                {
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
                }
            )

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
            rows.append(
                {
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
                }
            )
    return rows


def _format_parity_markdown(rows: list[dict], *, max_text: int | None) -> str:
    lines = []
    lines.append(
        "| API | output | atol | rtol | worst_abs | worst_rel | result | rationale | physical | root cause | verdict |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---|---|---|---|---|")
    for r in rows:
        if r["state"] != "measured":
            wa = "—"
            wr = "—"
            result = "—"
        else:
            wa = f"{r['worst_abs']:.2e}" if r["worst_abs"] > 0 else "0"
            wr = f"{r['worst_rel']:.2e}" if r["worst_rel"] > 0 else "0"
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
        rationale = _truncate(r.get("rationale") or "—", max_text)
        phys = _truncate(r.get("physical_magnitude") or "—", max_text)
        rc = _truncate(r.get("root_cause") or "—", max_text)
        verdict = _truncate(r.get("verdict") or "—", max_text)
        lines.append(
            f"| `{r['api_id']}`{flag} "
            f"| {r['output']} "
            f"| {r['atol']:.0e} | {rtol_s} "
            f"| {wa} | {wr} | {result} "
            f"| {rationale} | {phys} | {rc} | {verdict} |"
        )
    return "\n".join(lines)


def _format_speed(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}x"


def _load_speed_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    data = json.loads(path.read_text())
    if "parity_speed" in data:
        return list(data["parity_speed"].get("apis", []))
    return list(data.get("apis", []))


def _format_speed_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| API | Warm ×p50 | Warm ×p95 | Cold × | Gate | Waiver |",
        "|---|---:|---:|---:|---|---|",
    ]
    for r in rows:
        raw_passed = bool(r.get("raw_passed", r.get("passed", False)))
        waived = bool(r.get("waived", False))
        if waived:
            gate = "WAIVED"
        elif raw_passed:
            gate = "PASS"
        elif r.get("passed"):
            gate = "PASS"
        else:
            gate = "FAIL"
        waiver = r.get("waiver") or "—"
        lines.append(
            f"| `{r['api_id']}` "
            f"| {_format_speed(r.get('speedup_p50'))} "
            f"| {_format_speed(r.get('speedup_p95'))} "
            f"| {_format_speed(r.get('speedup_cold'))} "
            f"| {gate} | {waiver} |"
        )
    return "\n".join(lines)


def _coverage_summary(rows: list[dict]) -> str:
    declared_apis = sorted({r["api_id"] for r in rows})
    measured_apis = sorted({r["api_id"] for r in rows if r["state"] == "measured"})
    wired_not_measured = sorted(
        {r["api_id"] for r in rows if r["state"] == "wired-not-measured"}
    )
    orchestration = sorted(
        {r["api_id"] for r in rows if r["state"] == "orchestration-implied"}
    )
    unwired = sorted({r["api_id"] for r in rows if r["state"] == "unwired"})
    flagged = sorted(
        {r["api_id"] for r in rows if r["state"] == "measured" and r["investigate"]}
    )
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
        lines.append(
            "**Orchestration (covered indirectly via underlying kernel parity)**:"
        )
        for a in orchestration:
            lines.append(f"- `{a}`")
    if unwired:
        lines.append("")
        lines.append(
            "**Declared but UNWIRED in random fuzz** (no parity, action needed):"
        )
        for a in unwired:
            lines.append(f"- `{a}`")
    if flagged:
        lines.append("")
        lines.append("**Flagged (`investigate=True`)**:")
        for a in flagged:
            lines.append(f"- `{a}`")
    return "\n".join(lines)


def _api_result_from_json(entry: dict[str, Any]) -> parity_fuzz.ApiResult:
    seeds = []
    for s in entry["seeds"]:
        outs = [
            parity_fuzz.OutputResult(
                name=o["name"],
                max_abs=o["max_abs"],
                max_rel=o["max_rel"],
                atol=o["atol"],
                rtol=o["rtol"],
                passed=o["passed"],
                nan_disagreement=o.get("nan_disagreement", 0),
            )
            for o in s["outputs"]
        ]
        seeds.append(
            parity_fuzz.SeedResult(
                seed=s["seed"],
                n=s["n"],
                outputs=outs,
                error=s["error"],
            )
        )
    return parity_fuzz.ApiResult(
        api_id=entry["api_id"],
        seeds=seeds,
        investigate=entry.get("investigate", False),
        investigate_task=entry.get("investigate_task", ""),
    )


def _load_fuzz_results(path: Path) -> list[parity_fuzz.ApiResult]:
    cached = json.loads(path.read_text())
    if "parity_fuzz" in cached:
        cached = cached["parity_fuzz"]
    return [_api_result_from_json(entry) for entry in cached.get("apis", [])]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parity tolerance + result table.")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--base-seed", type=int, default=20260426)
    p.add_argument("--apis", nargs="*", default=None)
    p.add_argument(
        "--parity-artifact",
        type=Path,
        default=DEFAULT_PARITY_ARTIFACT,
        help=(
            "Read fuzz results from this artifact. Accepts either "
            "parity_gate.json or parity_fuzz.json. Defaults to current "
            "migration/artifacts/parity_gate.json."
        ),
    )
    p.add_argument(
        "--speed-artifact",
        type=Path,
        default=DEFAULT_SPEED_ARTIFACT,
        help=(
            "Read speed results from this artifact. Accepts either "
            "parity_speed_cold_warm.json or parity_gate.json. Defaults "
            "to current migration/artifacts/parity_speed_cold_warm.json."
        ),
    )
    p.add_argument(
        "--no-speed",
        action="store_true",
        help="Only print parity tolerance/RCA table.",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Run parity_fuzz instead of reading --parity-artifact.",
    )
    p.add_argument(
        "--max-text",
        type=int,
        default=0,
        help=(
            "Maximum characters per rationale/RCA cell. Defaults to 0, "
            "which prints full text. Use a positive value for compact output."
        ),
    )
    p.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Write full row data to this JSON path (in addition to stdout markdown).",
    )
    p.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="Write the markdown report to this path as well as stdout.",
    )
    p.add_argument(
        "--use-cache",
        type=Path,
        default=None,
        help="Deprecated alias for --parity-artifact.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    api_ids = args.apis or list(_inputs.all_api_ids())
    parity_artifact = args.use_cache or args.parity_artifact

    if not args.refresh and parity_artifact and parity_artifact.exists():
        fuzz_results = _load_fuzz_results(parity_artifact)
    else:
        fuzz_results = parity_fuzz.fuzz_all(
            api_ids,
            seeds=args.seeds,
            n=args.n,
            base_seed=args.base_seed,
        )

    rows = _build_rows(fuzz_results)
    speed_rows = [] if args.no_speed else _load_speed_rows(args.speed_artifact)

    max_text = None if args.max_text == 0 else args.max_text
    sections = [
        "# Rust Migration Parity And Performance Tables",
        "",
        "## Parity Tolerance + Observed Difference",
        "",
        _coverage_summary(rows),
        "",
        _format_parity_markdown(rows, max_text=max_text),
    ]
    if speed_rows:
        sections.extend(
            [
                "",
                "## Performance",
                "",
                _format_speed_markdown(speed_rows),
            ]
        )
    report = "\n".join(sections)

    print(report)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json_output}", file=sys.stderr)

    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(report)
        print(f"wrote {args.markdown_output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
