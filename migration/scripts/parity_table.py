"""Pretty-print parity tolerance/RCA and performance tables.

Reads the current parity artifacts (or runs parity_fuzz/fixed fixtures when
requested), then joins each result with the configured per-API tolerance from
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

from adam_core._rust.status import API_MIGRATIONS_BY_ID, validate_api_migrations

from migration.parity import _inputs, parity_fixed, parity_fuzz, tolerances

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
    fixed_results: list[parity_fixed.ApiResult] | None = None,
) -> list[dict]:
    validate_api_migrations()
    rows = []
    by_api = {r.api_id: r for r in fuzz_results}
    fixed_by_api = {r.api_id: r for r in fixed_results or []}

    measured_ids = set(by_api.keys())
    fixed_ids = set(fixed_by_api.keys())
    wired_ids = set(_inputs.all_api_ids())
    declared_ids = set(tolerances.all_api_ids())
    missing_registry = sorted(declared_ids - set(API_MIGRATIONS_BY_ID))
    if missing_registry:
        raise RuntimeError(
            "Parity tolerance entries missing from API_MIGRATIONS registry: "
            + ", ".join(missing_registry)
        )

    # Measured entries (one row per output)
    for api_id in sorted(measured_ids):
        spec = tolerances.get(api_id)
        api_result = by_api[api_id]
        migration = API_MIGRATIONS_BY_ID[api_id]
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
                    "registry_status": migration.status,
                    "parity_coverage": migration.parity_coverage,
                    "coverage_note": migration.coverage_note,
                    "covered_subcases": migration.covered_subcases,
                    "excluded_subcases": migration.excluded_subcases,
                }
            )

    for api_id in sorted(fixed_ids):
        spec = tolerances.get(api_id)
        api_result = fixed_by_api[api_id]
        migration = API_MIGRATIONS_BY_ID[api_id]
        for out_name, tol in spec.outputs.items():
            worst_abs = 0.0
            worst_rel = 0.0
            for fixture in api_result.fixtures:
                for output in fixture.outputs:
                    if output.name == out_name:
                        worst_abs = max(worst_abs, output.max_abs)
                        worst_rel = max(worst_rel, output.max_rel)
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
                    "state": (
                        "fixed-fixture-supplemental"
                        if api_id in measured_ids
                        else "fixed-fixture"
                    ),
                    "registry_status": migration.status,
                    "parity_coverage": migration.parity_coverage,
                    "coverage_note": migration.coverage_note,
                    "covered_subcases": migration.covered_subcases,
                    "excluded_subcases": migration.excluded_subcases,
                    "fixture_names": tuple(
                        fixture.name for fixture in api_result.fixtures
                    ),
                }
            )

    for api_id in sorted(declared_ids - measured_ids - fixed_ids):
        spec = tolerances.get(api_id)
        migration = API_MIGRATIONS_BY_ID[api_id]
        if migration.parity_coverage == "orchestration-implied":
            state = "orchestration-implied"
        elif migration.parity_coverage == "fixed-fixture":
            state = "fixed-fixture-missing"
        elif migration.parity_coverage == "random-fuzz-excluded":
            state = "random-fuzz-excluded"
        elif migration.parity_coverage == "targeted-tests":
            state = "targeted-tests"
        elif migration.parity_coverage == "manual-only":
            state = "manual-only"
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
                    "registry_status": migration.status,
                    "parity_coverage": migration.parity_coverage,
                    "coverage_note": migration.coverage_note,
                    "covered_subcases": migration.covered_subcases,
                    "excluded_subcases": migration.excluded_subcases,
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
        observed = r["state"] in {
            "measured",
            "fixed-fixture",
            "fixed-fixture-supplemental",
        }
        if not observed:
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
        elif r["state"] == "fixed-fixture":
            flag = " (fixed fixture)"
        elif r["state"] == "fixed-fixture-supplemental":
            flag = " (supplemental fixed fixture)"
        elif r["state"] == "fixed-fixture-missing":
            flag = " (fixed fixture missing)"
        elif r["state"] == "random-fuzz-excluded":
            flag = " (random-fuzz excluded)"
        elif r["state"] == "targeted-tests":
            flag = " (targeted tests)"
        elif r["state"] == "manual-only":
            flag = " (manual only)"
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


def _load_speed_artifact(
    path: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path is None or not path.exists():
        return [], {}
    data = json.loads(path.read_text())
    if "parity_speed" in data:
        speed = data["parity_speed"]
        return list(speed.get("apis", [])), dict(speed)
    return list(data.get("apis", [])), data


def _format_speed_metadata(metadata: dict[str, Any]) -> str:
    if not metadata:
        return ""
    warm = metadata.get("thread_mode", "unknown")
    cold = metadata.get("cold_thread_mode", "unknown")
    thread_policy = metadata.get("thread_policy")
    lane_policy = metadata.get("lane_policy")
    lines = [
        f"**Thread mode**: warm p50/p95 `{warm}`; cold `{cold}`.",
        "Single-thread mode caps thread pools only; SIMD/ILP remain enabled within each CPU core.",
    ]
    if thread_policy:
        lines.append(f"Thread policy: {thread_policy}")
    if lane_policy:
        lines.append(f"Lane policy: {lane_policy}")
    legacy_cache = metadata.get("legacy_timing_cache")
    if isinstance(legacy_cache, dict):
        hits = legacy_cache.get("hits") or {}
        writes = legacy_cache.get("writes") or {}
        lines.append(
            "Legacy timing cache: "
            f"`{legacy_cache.get('path', 'unknown')}`; "
            f"refresh={legacy_cache.get('refresh', False)}; "
            f"hits={hits}; writes={writes}."
        )
        freshness = legacy_cache.get("entry_freshness")
        if isinstance(freshness, dict):
            parts = []
            for section in ("warm", "cold"):
                summary = freshness.get(section)
                if not isinstance(summary, dict):
                    continue
                parts.append(
                    f"{section}: {summary.get('entries', 0)} entries, "
                    f"captured {summary.get('captured_at_min') or '—'} to "
                    f"{summary.get('captured_at_max') or '—'}, "
                    f"identities={summary.get('distinct_legacy_identities', 0)}"
                )
            if parts:
                lines.append("Legacy cache freshness: " + "; ".join(parts) + ".")

    lanes = metadata.get("lanes") or []
    if lanes:
        lines.append("")
        lines.append("**Speed lanes**:")
        for lane in lanes:
            name = lane.get("name", "unknown")
            enforced = "enforced" if lane.get("enforced") else "diagnostic"
            n_values = lane.get("n_values") or []
            if n_values:
                size = ", ".join(str(n) for n in n_values)
            else:
                size = str(lane.get("n", "unknown"))
            lines.append(
                f"- `{name}` ({enforced}): n={size}; {lane.get('description', '')}"
            )
    return "\n".join(lines)


def _speed_gate_label(row: dict[str, Any]) -> str:
    raw_passed = bool(row.get("raw_passed", row.get("passed", False)))
    waived = bool(row.get("waived", False))
    if waived:
        gate = "WAIVED"
    elif raw_passed:
        gate = "PASS"
    elif row.get("passed"):
        gate = "PASS"
    else:
        gate = "FAIL"
    if not bool(row.get("lane_enforced", True)):
        return f"DIAG {gate}"
    return gate


def _workload_label(row: dict[str, Any]) -> str:
    label = row.get("workload_label")
    if label:
        return str(label)
    shape = row.get("workload_shape")
    if isinstance(shape, dict):
        label = shape.get("label")
        if label:
            return str(label)
        rows = shape.get("rows", row.get("n", "—"))
        return f"rows={rows}"
    if shape:
        return str(shape)
    return f"rows={row.get('n', '—')}"


def _format_lane_cell(row: dict[str, Any] | None) -> str:
    if row is None:
        return "—"
    waiver = row.get("waiver") or "—"
    return (
        f"{_workload_label(row)}<br>"
        f"p50 {_format_speed(row.get('speedup_p50'))}; "
        f"p95 {_format_speed(row.get('speedup_p95'))}; "
        f"cold {_format_speed(row.get('speedup_cold'))}<br>"
        f"{_speed_gate_label(row)}; waiver {waiver}"
    )


def _format_speed_long_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Lane | API | Size/shape | Warm ×p50 | Warm ×p95 | Cold × | Gate | Waiver |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for r in rows:
        gate = _speed_gate_label(r)
        waiver = r.get("waiver") or "—"
        lane = r.get("lane", "small-n")
        lines.append(
            f"| `{lane}` "
            f"| `{r['api_id']}` "
            f"| {_workload_label(r)} "
            f"| {_format_speed(r.get('speedup_p50'))} "
            f"| {_format_speed(r.get('speedup_p95'))} "
            f"| {_format_speed(r.get('speedup_cold'))} "
            f"| {gate} | {waiver} |"
        )
    return "\n".join(lines)


def _format_speed_markdown(rows: list[dict[str, Any]], *, long: bool = False) -> str:
    if long:
        return _format_speed_long_markdown(rows)

    lane_order = list(dict.fromkeys(row.get("lane", "small-n") for row in rows))
    by_api: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_api.setdefault(row["api_id"], {})[row.get("lane", "small-n")] = row

    lines = [
        "| API | " + " | ".join(f"{lane} speed" for lane in lane_order) + " |",
        "|---" + "|---" * len(lane_order) + "|",
    ]
    for api_id in sorted(by_api):
        cells = [_format_lane_cell(by_api[api_id].get(lane)) for lane in lane_order]
        lines.append(f"| `{api_id}` | " + " | ".join(cells) + " |")
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
    fixed_only = sorted({r["api_id"] for r in rows if r["state"] == "fixed-fixture"})
    fixed_supplemental = sorted(
        {r["api_id"] for r in rows if r["state"] == "fixed-fixture-supplemental"}
    )
    fixed_missing = sorted(
        {r["api_id"] for r in rows if r["state"] == "fixed-fixture-missing"}
    )
    random_excluded = sorted(
        {r["api_id"] for r in rows if r["state"] == "random-fuzz-excluded"}
    )
    targeted = sorted({r["api_id"] for r in rows if r["state"] == "targeted-tests"})
    manual = sorted({r["api_id"] for r in rows if r["state"] == "manual-only"})
    unwired = sorted({r["api_id"] for r in rows if r["state"] == "unwired"})
    flagged = sorted(
        {
            r["api_id"]
            for r in rows
            if r["state"] in {"measured", "fixed-fixture", "fixed-fixture-supplemental"}
            and r["investigate"]
        }
    )
    direct = len(measured_apis) + len(wired_not_measured)
    indirect = len(orchestration)
    fixed_only_word = "API" if len(fixed_only) == 1 else "APIs"
    fixed_supplemental_word = "API" if len(fixed_supplemental) == 1 else "APIs"
    random_excluded_word = "API" if len(random_excluded) == 1 else "APIs"
    measured_word = "API" if len(measured_apis) == 1 else "APIs"
    lines = []
    lines.append(
        f"**Coverage**: {direct} of {len(declared_apis)} declared APIs "
        f"wired directly in random-fuzz GENERATORS; "
        f"{indirect} additional orchestration APIs covered indirectly via "
        f"underlying kernel parity. "
        f"{len(fixed_only)} fixed-fixture {fixed_only_word} governed outside "
        f"randomized fuzz. "
        f"{len(fixed_supplemental)} supplemental fixed-fixture "
        f"{fixed_supplemental_word} also covered by randomized fuzz. "
        f"{len(random_excluded)} {random_excluded_word} intentionally excluded "
        f"from randomized fuzz with no fixed fixture in this artifact. "
        f"{len(measured_apis)} random-fuzz {measured_word} measured this run."
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
    if fixed_only:
        lines.append("")
        lines.append("**Fixed-fixture parity (not randomized fuzz)**:")
        for a in fixed_only:
            note = API_MIGRATIONS_BY_ID[a].coverage_note
            suffix = f" — {note}" if note else ""
            lines.append(f"- `{a}`{suffix}")
    if fixed_supplemental:
        lines.append("")
        lines.append("**Supplemental fixed-fixture parity (also randomized fuzz)**:")
        for a in fixed_supplemental:
            note = API_MIGRATIONS_BY_ID[a].coverage_note
            suffix = f" — {note}" if note else ""
            lines.append(f"- `{a}`{suffix}")
    if fixed_missing:
        lines.append("")
        lines.append("**Fixed-fixture parity missing from this artifact**:")
        for a in fixed_missing:
            note = API_MIGRATIONS_BY_ID[a].coverage_note
            suffix = f" — {note}" if note else ""
            lines.append(f"- `{a}`{suffix}")
    if random_excluded:
        lines.append("")
        lines.append("**Declared but intentionally excluded from randomized fuzz**:")
        for a in random_excluded:
            note = API_MIGRATIONS_BY_ID[a].coverage_note
            suffix = f" — {note}" if note else ""
            lines.append(f"- `{a}`{suffix}")
    if targeted:
        lines.append("")
        lines.append("**Covered by targeted tests, not baseline-main random fuzz**:")
        for a in targeted:
            lines.append(f"- `{a}`")
    if manual:
        lines.append("")
        lines.append("**Manual-only parity coverage**:")
        for a in manual:
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
    entries = cached.get("apis", [])
    if entries and "seeds" not in entries[0]:
        return []
    return [_api_result_from_json(entry) for entry in entries]


def _fixed_result_from_json(entry: dict[str, Any]) -> parity_fixed.ApiResult:
    fixtures = []
    for fixture in entry["fixtures"]:
        outputs = [
            parity_fuzz.OutputResult(
                name=output["name"],
                max_abs=output["max_abs"],
                max_rel=output["max_rel"],
                atol=output["atol"],
                rtol=output["rtol"],
                passed=output["passed"],
                nan_disagreement=output.get("nan_disagreement", 0),
            )
            for output in fixture["outputs"]
        ]
        fixtures.append(
            parity_fixed.FixtureResult(
                name=fixture["name"],
                description=fixture.get("description", ""),
                n=fixture["n"],
                outputs=outputs,
                error=fixture["error"],
            )
        )
    return parity_fixed.ApiResult(
        api_id=entry["api_id"],
        fixtures=fixtures,
        investigate=entry.get("investigate", False),
        investigate_task=entry.get("investigate_task", ""),
    )


def _load_fixed_results(path: Path) -> list[parity_fixed.ApiResult]:
    cached = json.loads(path.read_text())
    if "fixed_fixtures" in cached:
        cached = cached["fixed_fixtures"]
    entries = cached.get("apis", [])
    if entries and "fixtures" not in entries[0]:
        return []
    return [_fixed_result_from_json(entry) for entry in entries]


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
        "--speed-long",
        action="store_true",
        help="Render speed rows in long lane-per-row form instead of pivoting by API.",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Run parity_fuzz and fixed fixtures instead of reading --parity-artifact.",
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
    random_api_ids = list(_inputs.all_api_ids())
    fixed_api_ids = list(parity_fixed.all_api_ids())
    if args.apis is None:
        api_ids = random_api_ids
        requested_fixed_api_ids = fixed_api_ids
    else:
        supported = set(random_api_ids) | set(fixed_api_ids)
        unknown = sorted(set(args.apis) - supported)
        if unknown:
            raise SystemExit(
                "No parity fixture or generator for: " + ", ".join(unknown)
            )
        api_ids = [api_id for api_id in args.apis if api_id in random_api_ids]
        requested_fixed_api_ids = [
            api_id for api_id in args.apis if api_id in fixed_api_ids
        ]
    parity_artifact = args.use_cache or args.parity_artifact

    if not args.refresh and parity_artifact and parity_artifact.exists():
        fuzz_results = _load_fuzz_results(parity_artifact)
        fixed_results = _load_fixed_results(parity_artifact)
    else:
        fuzz_results = parity_fuzz.fuzz_all(
            api_ids,
            seeds=args.seeds,
            n=args.n,
            base_seed=args.base_seed,
        )
        fixed_results = parity_fixed.fixed_all(requested_fixed_api_ids)

    rows = _build_rows(fuzz_results, fixed_results)
    speed_rows, speed_metadata = (
        ([], {}) if args.no_speed else _load_speed_artifact(args.speed_artifact)
    )

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
        speed_metadata_md = _format_speed_metadata(speed_metadata)
        sections.extend(["", "## Performance", ""])
        if speed_metadata_md:
            sections.extend([speed_metadata_md, ""])
        sections.append(_format_speed_markdown(speed_rows, long=args.speed_long))
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
