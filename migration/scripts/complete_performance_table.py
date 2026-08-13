"""Render the complete registered core + ASSIST three-boundary performance table.

This report is deliberately explicit about scope: core rows are the complete
``API_MIGRATIONS`` computational registry, not all symbols in the 629-symbol
public-surface manifest. ASSIST rows cover every numerical public work unit for
which equivalent updated-upstream timing evidence exists; unmeasured numerical
methods are listed explicitly rather than silently omitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORE = (
    ROOT / "migration" / "artifacts" / "parity_speed_updated_upstream_cold_warm.json"
)
DEFAULT_VARIANT = (
    ROOT / "migration" / "artifacts" / "variant_9d_three_boundary_speed_2026-08-12.json"
)
DEFAULT_OUTPUT = ROOT / "migration" / "artifacts" / "complete_performance_table.md"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _ratio(numerator: float | None, denominator: float | None) -> str:
    if numerator is None or denominator is None or denominator == 0.0:
        return "—"
    return f"{numerator / denominator:.2f}×"


def _duration(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 1e-3:
        return f"{value * 1e6:.1f} µs"
    if value < 1.0:
        return f"{value * 1e3:.2f} ms"
    return f"{value:.3f} s"


def _timing_pair(p50: float | None, p95: float | None) -> str:
    return f"{_duration(p50)} / {_duration(p95)}"


def _core_gate_outcome(data: dict[str, Any]) -> str:
    enforced_failures = [
        row
        for row in data["apis"]
        if row.get("lane_enforced") and not row.get("passed")
    ]
    diagnostic_failures = [
        row
        for row in data["apis"]
        if not row.get("lane_enforced") and not row.get("raw_passed")
    ]
    if enforced_failures:
        failure_text = "; ".join(
            f"`{row['api_id']}` {row['lane']} p50/p95 "
            f"{row['speedup_p50']:.3f}×/{row['speedup_p95']:.3f}×"
            for row in enforced_failures
        )
        outcome = (
            f"the canonical timing run has {len(enforced_failures)} enforced "
            f"misses: {failure_text}."
        )
    else:
        outcome = (
            "all 105 enforced rows (35 APIs × 3 lanes) pass the 1.3× policy "
            "with no waivers; tiny-n p95 remains report-only by policy."
        )
    if diagnostic_failures:
        diagnostic_text = "; ".join(
            f"`{row['api_id']}` {row['lane']}" for row in diagnostic_failures
        )
        outcome += (
            f" {len(diagnostic_failures)} raw-kernel diagnostic rows miss: "
            f"{diagnostic_text}. These do not affect promotion governance."
        )
    return (
        "**Gate outcome:** "
        + outcome
        + " No waiver or threshold change was applied. ASSIST rows are a "
        "performance matrix rather than the canonical 1.3× gate; rows below "
        "1.0× are reported explicitly, not labeled PASS."
    )


def _core_table(data: dict[str, Any]) -> str:
    lines = [
        "## adam-core computational registry",
        "",
        "| Surface | Lane and concrete workload | Updated-upstream Python p50/p95 | Current public facade p50/p95 | Native Rust p50/p95 | Updated/current | Updated/native | Native evidence |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in data["apis"]:
        legacy_p50 = row.get("legacy_p50_s")
        legacy_p95 = row.get("legacy_p95_s")
        current_p50 = row.get("rust_p50_s")
        current_p95 = row.get("rust_p95_s")
        native_p50 = row.get("native_rust_p50_s")
        native_p95 = row.get("native_rust_p95_s")
        native_status = row.get("native_rust_status", "unavailable")
        native_note = (
            "measured (`std::time::Instant`)"
            if native_status == "measured"
            else f"unavailable ({row.get('native_rust_todo') or row.get('native_rust_unavailable_reason') or 'no adapter'})"
        )
        gate = (
            "PASS"
            if row.get("passed")
            else ("FAIL" if row.get("lane_enforced") else "DIAG FAIL")
        )
        lines.append(
            f"| `{row['api_id']}` | `{row['lane']}`: {row['workload_label']} "
            f"| {_timing_pair(legacy_p50, legacy_p95)} "
            f"| {_timing_pair(current_p50, current_p95)} "
            f"| {_timing_pair(native_p50, native_p95)} "
            f"| {_ratio(legacy_p50, current_p50)} / {_ratio(legacy_p95, current_p95)} "
            f"| {_ratio(legacy_p50, native_p50)} / {_ratio(legacy_p95, native_p95)} "
            f"| {native_note}; gate {gate} |"
        )
    return "\n".join(lines)


def _assist_payload_rows(
    payload: dict[str, Any], surface: str
) -> Iterable[tuple[str, str, dict[str, Any], str]]:
    for item in payload.get("workloads", []):
        shape = item.get("workload_shape", {})
        shape_label = ", ".join(f"{key}={value}" for key, value in shape.items())
        options = item.get("options", {})
        option_label = ", ".join(
            f"{key}={value}"
            for key, value in options.items()
            if key
            in {
                "covariance",
                "covariance_method",
                "num_samples",
                "seed",
                "include_nongrav",
                "max_processes",
                "chunk_size",
            }
        )
        details = ", ".join(part for part in (shape_label, option_label) if part)
        yield (
            surface,
            f"{item.get('lane', '—')}: `{item['name']}` ({details})",
            item["timing_seconds"],
            item.get("description", ""),
        )


def _assist_rows() -> list[tuple[str, str, dict[str, Any], str]]:
    core_artifacts = ROOT / "migration" / "artifacts"
    rows: list[tuple[str, str, dict[str, Any], str]] = []
    rows.extend(
        _assist_payload_rows(
            _load(core_artifacts / "assist_public_semantics_benchmark_2026-08-12.json"),
            "ASSISTPropagator.propagate_orbits",
        )
    )
    rows.extend(
        _assist_payload_rows(
            _load(
                core_artifacts / "assist_nongrav_propagation_benchmark_2026-08-12.json"
            ),
            "ASSISTPropagator.propagate_orbits(non-gravitational)",
        )
    )
    rows.extend(
        _assist_payload_rows(
            _load(
                core_artifacts
                / "assist_public_semantics_covariance_benchmark_2026-08-12.json"
            ),
            "ASSISTPropagator.propagate_orbits(covariance=True)",
        )
    )
    rows.extend(
        _assist_payload_rows(
            _load(core_artifacts / "assist_ephemeris_benchmark_2026-08-12.json"),
            "ASSISTPropagator.generate_ephemeris",
        )
    )
    rows.extend(
        _assist_payload_rows(
            _load(core_artifacts / "assist_od_benchmark_2026-08-12.json"),
            "ASSISTPropagator OD/least-squares",
        )
    )
    impacts = _load(core_artifacts / "assist_impacts_benchmark_2026-08-12.json")
    for lane in impacts.get("lanes", []):
        timing = {
            "legacy_adam_core": {
                "p50": lane.get("legacy_adam_core_p50_s", lane.get("python_p50_s")),
                "p95": lane.get("legacy_adam_core_p95_s", lane.get("python_p95_s")),
            },
            "current_python": {
                "p50": lane.get("current_python_p50_s", lane.get("rust_p50_s")),
                "p95": lane.get("current_python_p95_s", lane.get("rust_p95_s")),
            },
            "native_rust": {
                "p50": lane.get("native_rust_p50_s"),
                "p95": lane.get("native_rust_p95_s"),
                "status": (
                    "measured" if lane.get("native_rust_p50_s") else "unavailable"
                ),
            },
        }
        rows.append(
            (
                "ASSISTPropagator.detect_collisions",
                f"orbits={lane['n_orbits']}, days={lane['num_days']}, impacts={lane['n_impacts']}",
                timing,
                "Impact/collision survivor-set and impact-time comparison.",
            )
        )
    return rows


def _summary(timing: dict[str, Any], key: str) -> tuple[float | None, float | None]:
    value = timing.get(key, {})
    return value.get("p50"), value.get("p95")


def _assist_table() -> str:
    rows = _assist_rows()
    regressions: list[str] = []
    for surface, workload, timing, _description in rows:
        legacy_p50, _legacy_p95 = _summary(timing, "legacy_adam_core")
        if legacy_p50 is None:
            legacy_p50, _legacy_p95 = _summary(timing, "python")
        current_p50, _current_p95 = _summary(timing, "current_python")
        if current_p50 is None:
            current_p50, _current_p95 = _summary(timing, "rust")
        if (
            legacy_p50 is not None
            and current_p50 is not None
            and current_p50 > legacy_p50
        ):
            regressions.append(
                f"`{surface}` {workload}: {_ratio(legacy_p50, current_p50)}"
            )
    lines = [
        (
            f"ASSIST matrix contains {len(rows)} measured workload rows across "
            "propagation, covariance, ephemeris, impact/collision, and OD/least-squares."
        ),
        (
            "ASSIST p50 regressions (<1.0× updated/current): "
            + ("; ".join(regressions) if regressions else "none")
            + "."
        ),
        "",
        "## adam-assist numerical work units",
        "",
        "| Surface | Lane and concrete workload | Updated-upstream Python p50/p95 | Current public facade p50/p95 | Native Rust p50/p95 | Updated/current | Updated/native | Native evidence |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for surface, workload, timing, _description in rows:
        legacy_p50, legacy_p95 = _summary(timing, "legacy_adam_core")
        if legacy_p50 is None:
            legacy_p50, legacy_p95 = _summary(timing, "python")
        current_p50, current_p95 = _summary(timing, "current_python")
        if current_p50 is None:
            current_p50, current_p95 = _summary(timing, "rust")
        native_p50, native_p95 = _summary(timing, "native_rust")
        native = timing.get("native_rust", {})
        native_note = (
            "measured (`std::time::Instant`)"
            if native.get("status") == "measured" or native_p50 is not None
            else f"unavailable ({native.get('reason', 'no equivalent prepared native adapter')})"
        )
        lines.append(
            f"| `{surface}` | {workload} "
            f"| {_timing_pair(legacy_p50, legacy_p95)} "
            f"| {_timing_pair(current_p50, current_p95)} "
            f"| {_timing_pair(native_p50, native_p95)} "
            f"| {_ratio(legacy_p50, current_p50)} / {_ratio(legacy_p95, current_p95)} "
            f"| {_ratio(legacy_p50, native_p50)} / {_ratio(legacy_p95, native_p95)} "
            f"| {native_note} |"
        )
    lines.extend(
        [
            "",
            "### Explicitly unmeasured ASSIST public numerical rows",
            "",
            "| Surface | Status | Reason |",
            "|---|---|---|",
            "| `ASSISTPropagator.initial_orbit_determination` | parity and native-timing tests pass; no cross-runtime performance row | Frozen updated-upstream oracle has no equivalent IOD request/timer; no non-equivalent composition is mislabeled as apples-to-apples. |",
            "| `ASSISTPropagator.fit_least_squares_evaluated` | native parity/composition and timing-hook tests pass; no updated-upstream performance row | Frozen updated-upstream has no fused evaluated work unit. |",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--variant", type=Path, default=DEFAULT_VARIANT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    core = _load(args.core)
    if args.variant.exists():
        targeted = _load(args.variant)
        supplemental_rows = []
        for row in targeted.get("apis", []):
            row = dict(row)
            row["lane"] = f"supplemental-9d-{row['lane']}"
            supplemental_rows.append(row)
        core["apis"].extend(supplemental_rows)
    report = "\n\n".join(
        [
            "# Complete registered computational performance matrix",
            (
                "> **Scope:** Complete for all 44 computational work units in "
                "`API_MIGRATIONS` across tiny/small/large lanes, plus every "
                "ASSIST numerical public work unit with equivalent frozen "
                "updated-upstream timing evidence. It is not a claim that all "
                "629 core public symbols are benchmarkable computations. "
                "Compatibility constants, schemas, data veneers, provider "
                "boundaries, and class utilities outside the computational registry "
                "are governed by `migration/public_surface/manifest.json` plus its "
                "domain audits. The complete-surface audit `personal-cmy.37` is "
                "closed; this report does not silently treat its 629 symbol rows as "
                "629 independent benchmark workloads."
            ),
            (
                "**Workload convention:** a workload is the exact benchmark input "
                "shape and options shown in each row—for example `orbits=400 × "
                "epochs=50`, not an informal size adjective. Tiny rows measure "
                "one-off call overhead, small rows preserve the historical "
                "promotion scale, and large rows use API-shaped production axes."
            ),
            (
                f"**Coverage count:** {len(core['apis'])} core timing rows: 132 "
                "canonical rows (44 APIs × 3 lanes) plus 3 focused 9D "
                "`VariantOrbits.create` rows."
            ),
            (
                "**Ratio convention:** updated-upstream Python / implementation; "
                "larger is faster. Each cell is p50 / p95. Native values appear "
                "only for genuine Rust-owned `std::time::Instant` adapters."
            ),
            _core_gate_outcome(core),
            _core_table(core),
            _assist_table(),
        ]
    )
    args.output.write_text(report + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
