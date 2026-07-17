"""Pretty-print parity tolerance/RCA and performance tables.

Reads the current parity artifacts (or runs parity_fuzz/fixed fixtures when
requested), then joins each result with the configured per-API tolerance from
`migration/parity/tolerances.py`. Emits markdown tables suitable for
handoffs and reviews.

The parity table includes the tolerance/RCA fields:

    | API | output | atol | rtol | worst_abs | worst_rel | nan_mismatch | result |
    | rationale | physical magnitude | root cause | verdict |

The performance table includes warm/cold speedup and waiver state when a
speed miss is explicitly waived.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adam_core._rust.status import API_MIGRATIONS_BY_ID, validate_api_migrations

from migration.parity import (
    _inputs,
    backend_candidates,
    comparison_metadata,
    parity_fixed,
    parity_fuzz,
    tolerances,
)

DEFAULT_PARITY_ARTIFACT = Path("migration/artifacts/parity_gate.json")
DEFAULT_SPEED_ARTIFACT = Path("migration/artifacts/parity_speed_cold_warm.json")
DEFAULT_LATEST_MAIN_ARTIFACT = Path(
    "migration/artifacts/latest_main_additions_parity.json"
)

# GPL ASSIST lane artifacts. These compare the Rust GPL backend
# (downstream adam-assist over assist-rs + adam-core contracts) against the current
# Python adam_assist.ASSISTPropagator public semantics. The legacy checkout has
# no ASSIST surface, so this lane is current-Python vs current-Rust rather than
# legacy-vs-current. Both sides load the same DE440/SB441 kernels from the
# PyPI data packages installed with the package dependencies (naif-de440,
# jpl-small-bodies-de441-n16); the Rust side receives the resolved
# site-packages paths via ADAM_CORE_RS_ASSIST_{PLANETS,ASTEROIDS}_PATH. The
# lane stays artifact-driven for the same reason as the frozen legacy speed
# baselines: the benchmarks are multi-minute Ray/Rayon suites behind the GPL
# crate build, refreshed intentionally rather than on every report render.
DEFAULT_ASSIST_RESIDUALS_ARTIFACT = Path(
    "migration/artifacts/assist_public_semantics_residuals_2026-05-20.json"
)
DEFAULT_ASSIST_PROPAGATION_BENCHMARK = Path(
    "migration/artifacts/assist_public_semantics_benchmark_2026-05-26.json"
)
DEFAULT_ASSIST_COVARIANCE_BENCHMARK = Path(
    "migration/artifacts/assist_public_semantics_covariance_benchmark_2026-06-20.json"
)
DEFAULT_ASSIST_IMPACTS_BENCHMARK = Path(
    "migration/artifacts/assist_impacts_benchmark_2026-07-03.json"
)
ASSIST_COMPARISON_MODE = "gpl_rust_assist_backend_vs_current_python_adam_assist"
_DASH = "\u2014"


def _assist_comparison_mode(*sources: dict[str, Any] | None) -> str:
    """Comparison mode recorded by regenerated assist artifacts.

    Historical tracked artifacts predate the two-runtime legacy-adam_assist
    benchmark harness and do not carry this field, so they retain the previous
    current-python comparison label until intentionally refreshed.
    """
    for source in sources:
        if not source:
            continue
        mode = source.get("comparison_mode")
        if mode:
            return str(mode)
    return ASSIST_COMPARISON_MODE


def _truncate(text: str, n: int | None) -> str:
    text = " ".join(text.split())
    if n is None or n <= 0:
        return text
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


def _headroom_from_ratio(max_tolerance_ratio: float) -> float:
    if max_tolerance_ratio == 0.0:
        return float("inf")
    if math.isinf(max_tolerance_ratio):
        return 0.0
    return 1.0 / max_tolerance_ratio


def _governance_fields(api_id: str) -> dict[str, Any]:
    candidate = backend_candidates.get(api_id)
    if candidate is not None:
        return {
            "backend_candidate": True,
            "backend_candidate_id": candidate.candidate_id,
            "canonical_api_id": candidate.canonical_api_id,
            "canonical_name": candidate.canonical_name,
            "implementation_label": candidate.implementation_label,
            "boundary": candidate.boundary,
            "rust_module": candidate.rust_module,
            "legacy_comparator": candidate.legacy_comparator,
            "candidate_note": candidate.note,
            **comparison_metadata.for_api(api_id),
            "registry_status": "backend-candidate",
            "parity_coverage": "backend-candidate",
            "coverage_note": candidate.note,
            "covered_subcases": (
                f"{candidate.implementation_label} compared against {candidate.legacy_comparator}",
            ),
            "excluded_subcases": (),
        }
    migration = API_MIGRATIONS_BY_ID[api_id]
    return {
        "backend_candidate": False,
        **comparison_metadata.for_api(api_id),
        "registry_status": migration.status,
        "parity_coverage": migration.parity_coverage,
        "coverage_note": migration.coverage_note,
        "covered_subcases": migration.covered_subcases,
        "excluded_subcases": migration.excluded_subcases,
    }


def _api_markdown_label(api_id: str, row: dict[str, Any] | None = None) -> str:
    candidate = None
    if row and row.get("backend_candidate"):
        metadata = row.get("backend_candidate")
        if isinstance(metadata, dict):
            candidate = {
                "canonical_name": metadata.get("canonical_name"),
                "implementation_label": metadata.get("implementation_label"),
            }
        else:
            candidate = {
                "canonical_name": row.get("canonical_name"),
                "implementation_label": row.get("implementation_label"),
            }
    else:
        registry_candidate = backend_candidates.get(api_id)
        if registry_candidate is not None:
            candidate = {
                "canonical_name": registry_candidate.canonical_name,
                "implementation_label": registry_candidate.implementation_label,
            }
    if candidate is not None:
        canonical = candidate.get("canonical_name") or "canonical API TBD"
        label = candidate.get("implementation_label") or "implementation candidate"
        return f"`{canonical}`<br>" f"<sub>impl candidate: `{api_id}` ({label})</sub>"
    return f"`{api_id}`"


def _is_backend_candidate_row(row: dict[str, Any]) -> bool:
    return bool(row.get("backend_candidate")) or str(row.get("state", "")).startswith(
        "backend-candidate"
    )


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
    candidate_ids = set(backend_candidates.BACKEND_CANDIDATES_BY_ID)
    missing_registry = sorted(declared_ids - set(API_MIGRATIONS_BY_ID) - candidate_ids)
    if missing_registry:
        raise RuntimeError(
            "Parity tolerance entries missing from API_MIGRATIONS registry: "
            + ", ".join(missing_registry)
        )

    # Measured entries (one row per output)
    for api_id in sorted(measured_ids):
        spec = tolerances.get(api_id)
        api_result = by_api[api_id]
        governance = _governance_fields(api_id)
        state = (
            "backend-candidate-measured"
            if governance["backend_candidate"]
            else "measured"
        )
        for out_name, tol in spec.outputs.items():
            worst_abs = 0.0
            worst_rel = 0.0
            worst_rel_above_floor = 0.0
            max_tolerance_ratio = 0.0
            nan_disagreement = 0
            for s in api_result.seeds:
                for o in s.outputs:
                    if o.name == out_name:
                        worst_abs = max(worst_abs, o.max_abs)
                        worst_rel = max(worst_rel, o.max_rel)
                        worst_rel_above_floor = max(
                            worst_rel_above_floor, o.max_rel_above_atol_floor
                        )
                        max_tolerance_ratio = max(
                            max_tolerance_ratio, o.max_tolerance_ratio
                        )
                        nan_disagreement += o.nan_disagreement
            margin = _headroom_from_ratio(max_tolerance_ratio)
            rows.append(
                {
                    "api_id": api_id,
                    "output": out_name,
                    "atol": tol.atol,
                    "rtol": tol.rtol,
                    "worst_abs": worst_abs,
                    "worst_rel": worst_rel,
                    "worst_rel_above_atol_floor": worst_rel_above_floor,
                    "max_tolerance_ratio": max_tolerance_ratio,
                    "nan_disagreement": nan_disagreement,
                    "margin": margin,
                    "passed": api_result.passed,
                    "investigate": spec.investigate,
                    "investigate_task": spec.investigate_task,
                    "rationale": spec.rationale,
                    "dominant_column": spec.dominant_column,
                    "physical_magnitude": spec.physical_magnitude,
                    "root_cause": spec.root_cause,
                    "verdict": spec.verdict,
                    "state": state,
                    **governance,
                }
            )

    for api_id in sorted(fixed_ids):
        spec = tolerances.get(api_id)
        api_result = fixed_by_api[api_id]
        governance = _governance_fields(api_id)
        output_names = sorted(
            {
                output.name
                for fixture in api_result.fixtures
                for output in fixture.outputs
            }
        )
        for out_name in output_names:
            output_results = [
                output
                for fixture in api_result.fixtures
                for output in fixture.outputs
                if output.name == out_name
            ]
            worst_abs = max((output.max_abs for output in output_results), default=0.0)
            worst_rel = max((output.max_rel for output in output_results), default=0.0)
            worst_rel_above_floor = max(
                (output.max_rel_above_atol_floor for output in output_results),
                default=0.0,
            )
            max_tolerance_ratio = max(
                (output.max_tolerance_ratio for output in output_results),
                default=0.0,
            )
            nan_disagreement = sum(output.nan_disagreement for output in output_results)
            atol = (
                output_results[0].atol
                if output_results
                else spec.outputs[out_name].atol
            )
            rtol = (
                output_results[0].rtol
                if output_results
                else spec.outputs[out_name].rtol
            )
            margin = _headroom_from_ratio(max_tolerance_ratio)
            rows.append(
                {
                    "api_id": api_id,
                    "output": out_name,
                    "atol": atol,
                    "rtol": rtol,
                    "worst_abs": worst_abs,
                    "worst_rel": worst_rel,
                    "worst_rel_above_atol_floor": worst_rel_above_floor,
                    "max_tolerance_ratio": max_tolerance_ratio,
                    "nan_disagreement": nan_disagreement,
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
                    **governance,
                    "fixture_names": tuple(
                        fixture.name for fixture in api_result.fixtures
                    ),
                }
            )

    for api_id in sorted(declared_ids - measured_ids - fixed_ids):
        spec = tolerances.get(api_id)
        governance = _governance_fields(api_id)
        if governance["backend_candidate"]:
            state = (
                "backend-candidate-skipped"
                if api_id in wired_ids
                else "backend-candidate-unwired"
            )
        else:
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
                    "worst_rel_above_atol_floor": None,
                    "max_tolerance_ratio": None,
                    "nan_disagreement": None,
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
                    **governance,
                }
            )
    return rows


def _format_parity_markdown(rows: list[dict], *, max_text: int | None) -> str:
    lines = []
    lines.append(
        "| API | mode | output | atol | rtol | worst_abs | worst_rel | rel_above_floor | nan_mismatch | result | rationale | physical | root cause | verdict |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|")
    for r in rows:
        observed = r["state"] in {
            "measured",
            "fixed-fixture",
            "fixed-fixture-supplemental",
            "backend-candidate-measured",
        }
        if not observed:
            wa = "—"
            wr = "—"
            wr_floor = "—"
            nan_mismatch = "—"
            result = "—"
        else:
            wa = f"{r['worst_abs']:.2e}" if r["worst_abs"] > 0 else "0"
            wr = f"{r['worst_rel']:.2e}" if r["worst_rel"] > 0 else "0"
            wr_floor_value = r.get("worst_rel_above_atol_floor")
            wr_floor = (
                f"{wr_floor_value:.2e}"
                if wr_floor_value and wr_floor_value > 0
                else "0"
            )
            nan_mismatch = str(r.get("nan_disagreement") or 0)
            # The result column reports pass/fail from the actual allclose budget
            # and, when passing, the minimum budget headroom over all finite cells.
            if r.get("passed") is True:
                ratio = r.get("max_tolerance_ratio")
                if ratio == 0:
                    result = "PASS (∞×)"
                elif ratio is not None and ratio > 0:
                    result = f"PASS ({1.0 / ratio:.1f}×)"
                else:
                    result = "PASS"
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
        elif r["state"] == "backend-candidate-measured":
            flag = " (impl candidate)"
        elif r["state"] == "backend-candidate-skipped":
            flag = " (impl candidate skipped)"
        elif r["state"] == "backend-candidate-unwired":
            flag = " (impl candidate UNWIRED)"
        elif r["investigate"]:
            flag = f" ⚠ {r['investigate_task'] or 'investigate'}"
        rtol_s = f"{r['rtol']:.0e}" if r["rtol"] > 0 else "0"
        rationale = _truncate(r.get("rationale") or "—", max_text)
        phys = _truncate(r.get("physical_magnitude") or "—", max_text)
        rc = _truncate(r.get("root_cause") or "—", max_text)
        verdict = _truncate(r.get("verdict") or "—", max_text)
        api_label = _api_markdown_label(r["api_id"], r)
        lines.append(
            f"| {api_label}{flag} "
            f"| {_comparison_mode_label(r)} "
            f"| {r['output']} "
            f"| {r['atol']:.0e} | {rtol_s} "
            f"| {wa} | {wr} | {wr_floor} | {nan_mismatch} | {result} "
            f"| {rationale} | {phys} | {rc} | {verdict} |"
        )
    return "\n".join(lines)


def _format_speed(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}x"


def _legacy_current_speedup(speedup: dict[str, Any], percentile: str) -> float | None:
    """Prefer explicit three-column vocabulary, retaining old-artifact fallback."""
    return speedup.get(
        f"{percentile}_legacy_over_current_python",
        speedup.get(f"{percentile}_python_over_rust"),
    )


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 1.0e-3:
        return f"{value * 1.0e6:.1f}µs"
    if value < 1.0:
        return f"{value * 1.0e3:.2f}ms"
    return f"{value:.3f}s"


def _flatten_timing_samples(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        return [sample for item in value for sample in _flatten_timing_samples(item)]
    return []


def _format_latest_main_additions(path: Path) -> str:
    if not path.exists():
        return ""
    artifact = json.loads(path.read_text())
    parity = artifact.get("parity", {})
    metadata_exact = parity.get("scout_orbit_metadata_exact", {})
    rows = [
        (
            "Obs80 parser/file",
            bool(parity.get("obs80_exact")) and bool(parity.get("obs80_errors_exact")),
            "exact tables and error contracts",
        ),
        (
            "Trajectory six-method surface",
            bool(parity.get("trajectory_exact")),
            "exact values, half-open selection, gaps, and validation errors",
        ),
        (
            "Scout file=mpc observations",
            bool(parity.get("scout_observations_exact")),
            "exact nested table, ordering, signature, hash, and metadata",
        ),
        (
            "Scout sampled orbits",
            bool(metadata_exact) and all(metadata_exact.values()),
            "metadata exact; max state abs difference "
            f"{float(parity.get('scout_orbit_max_abs_difference', math.inf)):.3g}",
        ),
    ]
    lines = [
        "## Upstream 9b756803 Additions",
        "",
        f"Oracle: `{artifact.get('oracle_commit', 'unknown')}`. Overall status: "
        f"**{str(artifact.get('status', 'unknown')).upper()}**. This supplemental "
        "conversion fixture covers public APIs added after the historical 44-API "
        "oracle pin; it does not rotate the older benchmark identity.",
        "",
        "| Surface | Result | Evidence |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {surface} | {'PASS' if passed else 'FAIL'} | {evidence} |"
        for surface, passed, evidence in rows
    )

    oracle = artifact.get("python_control_timings", {}).get("oracle_seconds", {})
    current = artifact.get("python_control_timings", {}).get("current_seconds", {})
    speedups = artifact.get("python_control_timings", {}).get("speedup", {})
    native = artifact.get("native_rust_instant_timings", {})
    native_keys = {
        "obs80_file_seconds": "obs80_seconds",
        "trajectory_validate_seconds": "trajectory_validate_seconds",
        "trajectory_segment_seconds": "trajectory_segment_seconds",
        "scout_observations_processing_seconds": "scout_observations_seconds",
        "scout_orbits_processing_seconds": "scout_orbits_seconds",
    }
    labels = {
        "obs80_file_seconds": "Obs80 file parse (200 rows)",
        "trajectory_validate_seconds": "Trajectory validate (256 segments)",
        "trajectory_segment_seconds": "Trajectory segment lookup (256 segments)",
        "scout_observations_processing_seconds": "Scout observation response processing",
        "scout_orbits_processing_seconds": "Scout orbit response processing",
    }
    lines.extend(
        [
            "",
            "| Operation | upstream Python p50 | current facade p50 | speedup | native Rust-Instant p50 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key, label in labels.items():
        samples = _flatten_timing_samples(native.get(native_keys[key], []))
        native_p50 = statistics.median(samples) if samples else None
        lines.append(
            f"| {label} | {_format_seconds(oracle.get(key))} | "
            f"{_format_seconds(current.get(key))} | "
            f"{float(speedups.get(key, math.nan)):.2f}x | "
            f"{_format_seconds(native_p50)} |"
        )
    return "\n".join(lines)


def _format_timing_pair(data: dict[str, Any]) -> str:
    p50 = data.get("p50")
    if p50 is not None:
        p95 = data.get("p95")
        return (
            f"{_format_seconds(p50)} / {_format_seconds(p95)}"
            if p95 is not None
            else _format_seconds(p50)
        )
    todo = data.get("todo")
    return f"— ({todo})" if todo else "—"


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
    timing_policy = metadata.get("timing_policy")
    lane_policy = metadata.get("lane_policy")
    trials = metadata.get("canonical_speed_trials")
    aggregation = metadata.get("timing_aggregation")
    trial_text = (
        f" Built-in speed trials: `{trials}` using `{aggregation}`."
        if trials and aggregation
        else ""
    )
    lines = [
        f"**Thread mode**: warm p50/p95 `{warm}`; cold `{cold}`.{trial_text}",
        "Single-thread mode caps thread pools only; SIMD/ILP remain enabled within each CPU core.",
    ]
    if timing_policy:
        lines.append(f"Timing policy: {timing_policy}")
    columns = metadata.get("performance_columns")
    if isinstance(columns, dict):
        lines.append(
            "Performance columns: "
            f"legacy adam_core = {columns.get('legacy_adam_core', 'unknown')}; "
            f"current through Python = {columns.get('current_python', 'unknown')}; "
            f"native Rust = {columns.get('native_rust', 'unknown')}; "
            f"gate = {columns.get('gate', 'unknown')}."
        )
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
            enforcement = lane.get("enforcement")
            if enforcement == "mixed":
                enforced = (
                    f"mixed: {lane.get('enforced_api_count', '—')} enforced, "
                    f"{lane.get('diagnostic_api_count', '—')} diagnostic"
                )
            elif enforcement:
                enforced = str(enforcement)
            else:
                enforced = "enforced" if lane.get("enforced") else "diagnostic"
            n_values = lane.get("n_values") or []
            if n_values:
                size = ", ".join(str(n) for n in n_values)
            else:
                size = str(lane.get("n", "unknown"))
            trials = lane.get("timing_trials") or "—"
            aggregation = lane.get("timing_aggregation") or "—"
            lines.append(
                f"- `{name}` ({enforced}): n={size}; trials={trials}; "
                f"aggregation={aggregation}; {lane.get('description', '')}"
            )
    return "\n".join(lines)


def _comparison_mode_label(row: dict[str, Any]) -> str:
    label = row.get("comparison_mode_short")
    if label:
        return str(label)
    api_id = row.get("api_id")
    if api_id:
        meta = comparison_metadata.for_api(str(api_id))
        if meta.get("comparison_mode") != comparison_metadata.UNKNOWN:
            return str(meta.get("comparison_mode_short", "unknown"))
    label = row.get("comparison_mode_label") or row.get("comparison_mode")
    return str(label) if label else "unknown"


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
    legacy_p50 = row.get("legacy_p50_s")
    legacy_p95 = row.get("legacy_p95_s")
    current_p50 = row.get("current_python_p50_s", row.get("rust_p50_s"))
    current_p95 = row.get("current_python_p95_s", row.get("rust_p95_s"))
    native_p50 = row.get("native_rust_p50_s")
    native_p95 = row.get("native_rust_p95_s")
    if native_p50 is None:
        todo = row.get("native_rust_todo")
        native = f"native Rust —{f' ({todo})' if todo else ''}"
    else:
        native = (
            f"native Rust {_format_seconds(native_p50)}/{_format_seconds(native_p95)}; "
            f"Python/native {_format_speed(row.get('current_python_over_native_rust_p50'))}/"
            f"{_format_speed(row.get('current_python_over_native_rust_p95'))}"
        )
    return (
        f"{_workload_label(row)}<br>"
        f"legacy adam_core {_format_seconds(legacy_p50)}/{_format_seconds(legacy_p95)}; "
        f"current Python {_format_seconds(current_p50)}/{_format_seconds(current_p95)}; "
        f"{native}<br>"
        f"legacy/current p50 {_format_speed(row.get('speedup_p50'))}; "
        f"p95 {_format_speed(row.get('speedup_p95'))}; "
        f"cold {_format_speed(row.get('speedup_cold'))}<br>"
        f"{_speed_gate_label(row)}; waiver {waiver}"
    )


def _format_speed_long_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Lane | API / implementation | Mode | Size/shape | Legacy adam_core p50/p95 | Current Python p50/p95 | Native Rust p50/p95 | Python/native p50/p95 | Legacy/current p50/p95 | Cold × | Gate | Waiver |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        gate = _speed_gate_label(r)
        waiver = r.get("waiver") or "—"
        lane = r.get("lane", "small-n")
        legacy_p50 = r.get("legacy_p50_s")
        legacy_p95 = r.get("legacy_p95_s")
        current_p50 = r.get("current_python_p50_s", r.get("rust_p50_s"))
        current_p95 = r.get("current_python_p95_s", r.get("rust_p95_s"))
        native_p50 = r.get("native_rust_p50_s")
        native_p95 = r.get("native_rust_p95_s")
        native_todo = r.get("native_rust_todo")
        native_cell = (
            f"{_format_seconds(native_p50)} / {_format_seconds(native_p95)}"
            if native_p50 is not None
            else f"—{f' ({native_todo})' if native_todo else ''}"
        )
        lines.append(
            f"| `{lane}` "
            f"| {_api_markdown_label(r['api_id'], r)} "
            f"| {_comparison_mode_label(r)} "
            f"| {_workload_label(r)} "
            f"| {_format_seconds(legacy_p50)} / {_format_seconds(legacy_p95)} "
            f"| {_format_seconds(current_p50)} / {_format_seconds(current_p95)} "
            f"| {native_cell} "
            f"| {_format_speed(r.get('current_python_over_native_rust_p50'))} / {_format_speed(r.get('current_python_over_native_rust_p95'))} "
            f"| {_format_speed(r.get('speedup_p50'))} / {_format_speed(r.get('speedup_p95'))} "
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
        "| API / implementation | Mode | "
        + " | ".join(f"{lane} speed" for lane in lane_order)
        + " |",
        "|---|---" + "|---" * len(lane_order) + "|",
    ]
    for api_id in sorted(by_api):
        cells = [_format_lane_cell(by_api[api_id].get(lane)) for lane in lane_order]
        first_row = next(iter(by_api[api_id].values()))
        lines.append(
            f"| {_api_markdown_label(api_id, first_row)} | {_comparison_mode_label(first_row)} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines)


def _simple_timing_pair(p50: float | None, p95: float | None) -> str:
    """Format a compact timing pair, leaving unavailable measurements blank."""
    if p50 is None:
        return ""
    if p95 is None:
        return _format_seconds(p50)
    return f"{_format_seconds(p50)} / {_format_seconds(p95)}"


def _simple_display_name(row: dict[str, Any]) -> str:
    """Use the canonical public surface before any temporary candidate label."""
    api_id = str(row["api_id"])
    candidate = backend_candidates.get(api_id)
    if candidate is None:
        return api_id
    return f"{candidate.canonical_name} — {candidate.implementation_label}"


def _format_simple_timing_table(
    title: str,
    rows: list[tuple[str, str, str, str]],
) -> str:
    lines = [
        f"## {title}",
        "",
        "| Name | Legacy p50 / p95 | Current Python p50 / p95 | Native Rust p50 / p95 |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| `{name}` | {legacy} | {current} | {native} |"
        for name, legacy, current, native in rows
    )
    return "\n".join(lines)


def _format_simple_speed_timing_tables(rows: list[dict[str, Any]]) -> str:
    sections: list[str] = []
    lane_order = list(dict.fromkeys(str(row.get("lane", "small-n")) for row in rows))
    for lane in lane_order:
        timing_rows: list[tuple[str, str, str, str]] = []
        for row in sorted(
            (row for row in rows if row.get("lane", "small-n") == lane),
            key=_simple_display_name,
        ):
            timing_rows.append(
                (
                    _simple_display_name(row),
                    _simple_timing_pair(
                        row.get("legacy_p50_s"), row.get("legacy_p95_s")
                    ),
                    _simple_timing_pair(
                        row.get("current_python_p50_s", row.get("rust_p50_s")),
                        row.get("current_python_p95_s", row.get("rust_p95_s")),
                    ),
                    _simple_timing_pair(
                        row.get("native_rust_p50_s"),
                        row.get("native_rust_p95_s"),
                    ),
                )
            )
        sections.append(_format_simple_timing_table(f"adam_core — {lane}", timing_rows))
    return "\n\n".join(sections)


def _coverage_summary(rows: list[dict]) -> str:
    public_rows = [r for r in rows if not _is_backend_candidate_row(r)]
    candidate_rows = [r for r in rows if _is_backend_candidate_row(r)]
    declared_apis = sorted({r["api_id"] for r in public_rows})
    measured_apis = sorted(
        {r["api_id"] for r in public_rows if r["state"] == "measured"}
    )
    wired_not_measured = sorted(
        {r["api_id"] for r in public_rows if r["state"] == "wired-not-measured"}
    )
    orchestration = sorted(
        {r["api_id"] for r in public_rows if r["state"] == "orchestration-implied"}
    )
    fixed_only = sorted(
        {r["api_id"] for r in public_rows if r["state"] == "fixed-fixture"}
    )
    fixed_supplemental = sorted(
        {r["api_id"] for r in public_rows if r["state"] == "fixed-fixture-supplemental"}
    )
    fixed_missing = sorted(
        {r["api_id"] for r in public_rows if r["state"] == "fixed-fixture-missing"}
    )
    random_excluded = sorted(
        {r["api_id"] for r in public_rows if r["state"] == "random-fuzz-excluded"}
    )
    targeted = sorted(
        {r["api_id"] for r in public_rows if r["state"] == "targeted-tests"}
    )
    manual = sorted({r["api_id"] for r in public_rows if r["state"] == "manual-only"})
    unwired = sorted({r["api_id"] for r in public_rows if r["state"] == "unwired"})
    backend_candidate_ids = sorted({r["api_id"] for r in candidate_rows})
    backend_candidate_measured = sorted(
        {
            r["api_id"]
            for r in candidate_rows
            if r["state"] == "backend-candidate-measured"
        }
    )
    flagged = sorted(
        {
            r["api_id"]
            for r in public_rows
            if r["state"] in {"measured", "fixed-fixture", "fixed-fixture-supplemental"}
            and r["investigate"]
        }
    )
    mode_counts: dict[str, int] = {}
    rust_native_apis: list[str] = []
    for api_id in declared_apis:
        meta = comparison_metadata.for_api(api_id)
        short = str(meta.get("comparison_mode_short", "unknown"))
        mode_counts[short] = mode_counts.get(short, 0) + 1
        if meta.get("rust_native_top_level"):
            rust_native_apis.append(api_id)
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
    mode_parts = [f"{count} {mode}" for mode, count in sorted(mode_counts.items())]
    lines.append("")
    lines.append(
        "**Comparison modes** (what each row actually measures): "
        + "; ".join(mode_parts)
        + f"; plus {len(backend_candidate_ids)} impl candidates (diagnostic). "
        "`public facade` and `thin wrapper` rows enter through the canonical "
        "Python API on both sides, so measured speedups include Python/PyO3 "
        "marshalling. `raw kernel` rows are diagnostic PyO3 bindings compared "
        "against a legacy Python oracle."
    )
    lines.append("")
    lines.append(
        f"**Direct pure-Rust benchmark entrypoints**: {len(rust_native_apis)} of "
        f"{len(declared_apis)} registry rows bypass Python entirely "
        "(`rust_native_top_level=true`). This field describes the benchmark "
        "entrypoint, not migration completeness: `public facade` and `thin "
        "wrapper` rows satisfy the migration contract when one crossing owns "
        "the adam-core computation. Complete-surface disposition is governed "
        "by `migration/public_surface/manifest.json` and its domain audits."
    )
    if backend_candidate_ids:
        lines.append("")
        lines.append(
            "**Backend/transport implementation candidates** "
            "(diagnostic; not public API identities):"
        )
        for candidate_id in backend_candidate_ids:
            row = next(r for r in candidate_rows if r["api_id"] == candidate_id)
            measured = (
                "measured"
                if candidate_id in backend_candidate_measured
                else "not measured"
            )
            lines.append(f"- {_api_markdown_label(candidate_id, row)} — {measured}")

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


def _load_json_artifact(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def _format_assist_residuals(data: dict[str, Any]) -> list[str]:
    thresholds = data.get("thresholds", {})
    pos_thr = thresholds.get("position_abs_m")
    vel_thr = thresholds.get("velocity_abs_m_per_s")
    lines = [
        "### Public-semantics parity (live Rust vs frozen Python fixture)",
        "",
        f"Fixture `{data.get('fixture_id', 'unknown')}`; "
        f"thresholds: |\u0394pos| \u2264 {pos_thr:.3e} m, |\u0394vel| \u2264 {vel_thr:.3e} m/s.",
        "",
        "| Case | rows | max \\|\u0394pos\\| (m) | max \\|\u0394vel\\| (m/s) | max \\|\u0394t\\| (ns) | result |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for case in data.get("cases", []):
        pos = case.get("max_position_abs_m", float("inf"))
        vel = case.get("max_velocity_abs_m_per_s", float("inf"))
        t_ns = case.get("max_time_abs_ns", 0)
        passed = pos <= pos_thr and vel <= vel_thr and t_ns == 0
        lines.append(
            f"| `{case.get('case_id', 'unknown')}` "
            f"| {case.get('rows', _DASH)} "
            f"| {pos:.3e} | {vel:.3e} | {t_ns} "
            f"| {'PASS' if passed else 'FAIL'} |"
        )
    return lines


def _format_assist_propagation(data: dict[str, Any]) -> list[str]:
    env = data.get("environment", {})
    lines = [
        "### N-body propagation performance (Python adam-assist vs Rust backend)",
        "",
        f"Benchmark `{data.get('benchmark_id', 'unknown')}`; "
        f"thread mode: {env.get('thread_mode', 'unknown')}.",
        "",
        "| Lane | Workload | Shape (orbits\u00d7epochs\u2192rows) | Legacy p50/p95 | Current Python p50/p95 | Native Rust p50/p95 | "
        "Legacy/current p50/p95 | max \\|\u0394pos\\| (m) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for workload in data.get("workloads", []):
        shape = workload.get("workload_shape", {})
        timing = workload.get("timing_seconds", {})
        legacy = timing.get("legacy_adam_core", timing.get("python", {}))
        current = timing.get("current_python", timing.get("rust", {}))
        native = timing.get("native_rust", {})
        speedup = timing.get("speedup", {})
        residuals = workload.get("residuals", {})
        shape_label = (
            f"{shape.get('n_orbits', _DASH)}\u00d7{shape.get('n_target_times', _DASH)}"
            f"\u2192{shape.get('output_rows', _DASH)}"
        )
        lines.append(
            f"| `{workload.get('lane', _DASH)}` "
            f"| `{workload.get('name', 'unknown')}` "
            f"| {shape_label} "
            f"| {_format_timing_pair(legacy)} "
            f"| {_format_timing_pair(current)} "
            f"| {_format_timing_pair(native)} "
            f"| {_format_speed(_legacy_current_speedup(speedup, 'p50'))} / {_format_speed(_legacy_current_speedup(speedup, 'p95'))} "
            f"| {residuals.get('position_abs_m', float('nan')):.3e} |"
        )
    return lines


def _format_assist_covariance(data: dict[str, Any]) -> list[str]:
    lines = [
        "### Covariance propagation performance (sampled covariance, both sides)",
        "",
        f"Benchmark `{data.get('benchmark_id', 'unknown')}`. "
        "sigma-point/auto are deterministic (element-wise parity expected); "
        "monte-carlo uses different RNGs and is compared statistically.",
        "",
        "| Lane | Workload | Method | Parity expected | Legacy p50/p95 | Current Python p50/p95 | Native Rust p50/p95 | Legacy/current p50/p95 "
        "| max σ rel | max \\|Δpos\\| (m) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for workload in data.get("workloads", []):
        covariance = workload.get("covariance", {})
        timing = workload.get("timing_seconds", {})
        speedup = timing.get("speedup", {})
        legacy = timing.get("legacy_adam_core", timing.get("python", {}))
        current = timing.get("current_python", timing.get("rust", {}))
        native = timing.get("native_rust", {})
        cov_res = workload.get("covariance_residuals", {})
        state_res = workload.get("state_residuals", {})
        sigma_rel = cov_res.get("max_sigma_rel")
        sigma_cell = f"{sigma_rel:.3e}" if sigma_rel is not None else "\u2014"
        lines.append(
            f"| `{workload.get('lane', _DASH)}` "
            f"| `{workload.get('name', 'unknown')}` "
            f"| {covariance.get('method', _DASH)} "
            f"| {'yes' if covariance.get('parity_expected') else 'statistical'} "
            f"| {_format_timing_pair(legacy)} "
            f"| {_format_timing_pair(current)} "
            f"| {_format_timing_pair(native)} "
            f"| {_format_speed(_legacy_current_speedup(speedup, 'p50'))} / {_format_speed(_legacy_current_speedup(speedup, 'p95'))} "
            f"| {sigma_cell} "
            f"| {state_res.get('position_abs_m', float('nan')):.3e} |"
        )
    return lines


def _format_assist_impacts(data: dict[str, Any]) -> list[str]:
    lines = [
        "### Impact detection performance (`detect_collisions`)",
        "",
        f"{data.get('description', '')}",
        "",
        "| Orbits | Days | Impacts | Legacy p50/p95 | Current Python p50/p95 | Native Rust p50/p95 | Legacy/current p50/p95 "
        "| max impact-time \u0394 (days) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lane in data.get("lanes", []):
        lines.append(
            f"| {lane.get('n_orbits', _DASH)} "
            f"| {lane.get('num_days', _DASH)} "
            f"| {lane.get('n_impacts', _DASH)} "
            f"| {_format_timing_pair({'p50': lane.get('legacy_adam_core_p50_s', lane.get('python_p50_s')), 'p95': lane.get('legacy_adam_core_p95_s', lane.get('python_p95_s'))})} "
            f"| {_format_timing_pair({'p50': lane.get('current_python_p50_s', lane.get('rust_p50_s')), 'p95': lane.get('current_python_p95_s', lane.get('rust_p95_s'))})} "
            f"| {_format_timing_pair({'p50': lane.get('native_rust_p50_s'), 'p95': lane.get('native_rust_p95_s'), 'todo': lane.get('native_rust_todo')})} "
            f"| {_format_speed(lane.get('speedup_p50'))} / {_format_speed(lane.get('speedup_p95'))} "
            f"| {lane.get('max_impact_time_diff_days', float('nan')):.3e} |"
        )
    return lines


def _simple_assist_payload_row(
    name: str, timing: dict[str, Any]
) -> tuple[str, str, str, str]:
    legacy = timing.get("legacy_adam_core", timing.get("python", {}))
    current = timing.get("current_python", timing.get("rust", {}))
    native = timing.get("native_rust", {})
    return (
        name,
        _simple_timing_pair(legacy.get("p50"), legacy.get("p95")),
        _simple_timing_pair(current.get("p50"), current.get("p95")),
        _simple_timing_pair(native.get("p50"), native.get("p95")),
    )


def _format_simple_assist_timing_tables(
    propagation_path: Path | None = DEFAULT_ASSIST_PROPAGATION_BENCHMARK,
    covariance_path: Path | None = DEFAULT_ASSIST_COVARIANCE_BENCHMARK,
    impacts_path: Path | None = DEFAULT_ASSIST_IMPACTS_BENCHMARK,
) -> str:
    sections: list[str] = []
    propagation = _load_json_artifact(propagation_path)
    if propagation:
        rows = [
            _simple_assist_payload_row(str(item["name"]), item["timing_seconds"])
            for item in propagation.get("workloads", [])
        ]
        sections.append(_format_simple_timing_table("ASSIST — propagation", rows))

    covariance = _load_json_artifact(covariance_path)
    if covariance:
        rows = [
            _simple_assist_payload_row(str(item["name"]), item["timing_seconds"])
            for item in covariance.get("workloads", [])
        ]
        sections.append(
            _format_simple_timing_table("ASSIST — covariance propagation", rows)
        )

    impacts = _load_json_artifact(impacts_path)
    if impacts:
        rows = []
        for lane in impacts.get("lanes", []):
            name = (
                f"detect_collisions — {lane['n_orbits']} orbits × "
                f"{lane['num_days']} days"
            )
            rows.append(
                (
                    name,
                    _simple_timing_pair(
                        lane.get("legacy_adam_core_p50_s", lane.get("python_p50_s")),
                        lane.get("legacy_adam_core_p95_s", lane.get("python_p95_s")),
                    ),
                    _simple_timing_pair(
                        lane.get("current_python_p50_s", lane.get("rust_p50_s")),
                        lane.get("current_python_p95_s", lane.get("rust_p95_s")),
                    ),
                    _simple_timing_pair(
                        lane.get("native_rust_p50_s"),
                        lane.get("native_rust_p95_s"),
                    ),
                )
            )
        sections.append(_format_simple_timing_table("ASSIST — impact detection", rows))
    return "\n\n".join(sections)


def _format_assist_section(
    residuals_path: Path | None = DEFAULT_ASSIST_RESIDUALS_ARTIFACT,
    propagation_path: Path | None = DEFAULT_ASSIST_PROPAGATION_BENCHMARK,
    covariance_path: Path | None = DEFAULT_ASSIST_COVARIANCE_BENCHMARK,
    impacts_path: Path | None = DEFAULT_ASSIST_IMPACTS_BENCHMARK,
) -> str:
    residuals = _load_json_artifact(residuals_path)
    propagation = _load_json_artifact(propagation_path)
    covariance = _load_json_artifact(covariance_path)
    impacts = _load_json_artifact(impacts_path)
    if not any([residuals, propagation, covariance, impacts]):
        return ""

    packages: dict[str, Any] = {}
    for source in (propagation, covariance, residuals, impacts):
        if source:
            packages = source.get("packages", {}) or packages
            if packages:
                break
    package_text = (
        "; ".join(f"`{name}=={ver}`" for name, ver in sorted(packages.items()))
        if packages
        else "unknown"
    )

    comparison_mode = _assist_comparison_mode(propagation, covariance, impacts)
    if "legacy_python_adam_assist" in comparison_mode:
        comparison_text = (
            "Rust GPL backend (downstream `adam_assist` over `assist-rs` + "
            "adam-core contracts) vs legacy Python "
            "`adam_assist.ASSISTPropagator` executed in the isolated "
            "`.legacy-assist-venv` runtime. The current adam_core tree no "
            "longer ships the Python Propagator composition, so refreshed "
            "benchmark artifacts use the two-runtime oracle."
        )
    else:
        comparison_text = (
            "Rust GPL backend (downstream `adam_assist` over `assist-rs` + "
            "adam-core contracts) vs the current Python "
            "`adam_assist.ASSISTPropagator` public path. Both sides are current "
            "implementations; this describes historical artifacts that predate "
            "the two-runtime oracle refresh."
        )
    lines = [
        "## ASSIST (GPL) N-Body Propagation",
        "",
        f"**Mode**: `{comparison_mode}` — {comparison_text} Both sides load "
        "the same DE440/SB441 kernels from the PyPI data packages installed "
        "with the package dependencies (`naif-de440`, "
        "`jpl-small-bodies-de441-n16`); kernel SHA-256 identity is recorded in "
        "each artifact. Rows stay artifact-driven for the same reason as the "
        "frozen legacy speed baselines: multi-minute benchmark suites behind "
        "the GPL crate build, refreshed intentionally rather than per render.",
        "",
        f"Packages: {package_text}.",
    ]
    if residuals:
        lines.append("")
        lines.extend(_format_assist_residuals(residuals))
    if propagation:
        lines.append("")
        lines.extend(_format_assist_propagation(propagation))
    if covariance:
        lines.append("")
        lines.extend(_format_assist_covariance(covariance))
    if impacts:
        lines.append("")
        lines.extend(_format_assist_impacts(impacts))
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
                max_rel_above_atol_floor=o.get("max_rel_above_atol_floor", 0.0),
                max_tolerance_ratio=o.get("max_tolerance_ratio", 0.0),
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
                max_rel_above_atol_floor=output.get("max_rel_above_atol_floor", 0.0),
                max_tolerance_ratio=output.get("max_tolerance_ratio", 0.0),
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
        "--latest-main-artifact",
        type=Path,
        default=DEFAULT_LATEST_MAIN_ARTIFACT,
        help="Supplemental parity/timing artifact for APIs added at upstream 9b756803.",
    )
    p.add_argument(
        "--latest-main-only",
        action="store_true",
        help="Render only the supplemental upstream-9b756803 section.",
    )
    p.add_argument(
        "--no-speed",
        action="store_true",
        help="Only print parity tolerance/RCA table.",
    )
    p.add_argument(
        "--no-assist",
        action="store_true",
        help="Skip the GPL ASSIST parity/performance section.",
    )
    p.add_argument(
        "--speed-long",
        action="store_true",
        help="Render speed rows in long lane-per-row form instead of pivoting by API.",
    )
    p.add_argument(
        "--simple-timings",
        action="store_true",
        help=(
            "Render only name plus legacy/current-Python/native-Rust p50/p95 "
            "tables; unavailable native measurements are blank."
        ),
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
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.latest_main_only:
        print(_format_latest_main_additions(args.latest_main_artifact))
        return 0
    if args.simple_timings and args.no_speed:
        parser.error("--simple-timings cannot be combined with --no-speed")
    if args.simple_timings and args.speed_long:
        parser.error("--simple-timings cannot be combined with --speed-long")
    if args.simple_timings and args.refresh:
        parser.error("--simple-timings is artifact-only and cannot use --refresh")

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
    requested_api_ids = set(args.apis) if args.apis is not None else None
    if requested_api_ids is not None:
        rows = [row for row in rows if row["api_id"] in requested_api_ids]
    speed_rows, speed_metadata = (
        ([], {}) if args.no_speed else _load_speed_artifact(args.speed_artifact)
    )
    if requested_api_ids is not None:
        speed_rows = [
            row for row in speed_rows if str(row.get("api_id")) in requested_api_ids
        ]

    max_text = None if args.max_text == 0 else args.max_text
    if args.simple_timings:
        sections = ["# Three-implementation performance timings"]
        if speed_rows:
            sections.extend(["", _format_simple_speed_timing_tables(speed_rows)])
        if not args.no_assist and requested_api_ids is None:
            assist_md = _format_simple_assist_timing_tables()
            if assist_md:
                sections.extend(["", assist_md])
    else:
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
        if not args.no_assist and requested_api_ids is None:
            assist_md = _format_assist_section()
            if assist_md:
                sections.extend(["", assist_md])
        if requested_api_ids is None:
            latest_main_md = _format_latest_main_additions(args.latest_main_artifact)
            if latest_main_md:
                sections.extend(["", latest_main_md])
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
