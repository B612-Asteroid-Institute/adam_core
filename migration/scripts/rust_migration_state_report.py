"""Render the current state of the adam-core Rust migration.

Combines three sources into one table:

1. `src/adam_core/_rust/status.py::API_MIGRATIONS` — per-API default backend,
   status, waiver.
2. `migration/artifacts/history/rust_vs_legacy_final_snapshot_2026-04-23.json`
   — frozen historical Rust-vs-legacy speedup per API (legacy was still
   callable at capture time). See the sibling README.md for caveats.
3. `migration/artifacts/rust_latency_baseline.json` +
   `rust_latency_current.json` — post-legacy Rust-only latency baseline and
   last-run measurements from `rust_backend_benchmark_gate.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from adam_core._rust import API_MIGRATIONS, validate_api_migrations

# Benchmark key → api_id mapping (same mapping the history snapshot uses).
BENCH_TO_API_ID = {
    "cartesian_to_spherical": "coordinates.cartesian_to_spherical",
    "cartesian_to_geodetic": "coordinates.cartesian_to_geodetic",
    "cartesian_to_keplerian": "coordinates.cartesian_to_keplerian",
    "keplerian_to_cartesian": "coordinates.keplerian.to_cartesian",
    "cartesian_to_cometary": "coordinates.cartesian_to_cometary",
    "cometary_to_cartesian": "coordinates.cometary.to_cartesian",
    "spherical_to_cartesian": "coordinates.spherical.to_cartesian",
    "calc_mean_motion": "dynamics.calc_mean_motion",
    "propagate_2body": "dynamics.propagate_2body",
    "propagate_2body_with_covariance": "dynamics.propagate_2body_with_covariance",
    "generate_ephemeris_2body": "dynamics.generate_ephemeris_2body",
    "generate_ephemeris_2body_with_covariance": (
        "dynamics.generate_ephemeris_2body_with_covariance"
    ),
    "calculate_phase_angle": "photometry.calculate_phase_angle",
    "calculate_apparent_magnitude_v": "photometry.calculate_apparent_magnitude_v",
    "calculate_apparent_magnitude_v_and_phase_angle": (
        "photometry.calculate_apparent_magnitude_v_and_phase_angle"
    ),
    "calc_gibbs": "orbit_determination.calcGibbs",
    "calc_herrick_gibbs": "orbit_determination.calcHerrickGibbs",
    "calc_gauss": "orbit_determination.calcGauss",
    "transform_coordinates": "coordinates.transform_coordinates",
    "solve_lambert": "dynamics.solve_lambert",
    "gauss_iod": "orbit_determination.gaussIOD",
}

# Overrides for historical speedup numbers that are not trustworthy in the
# snapshot (e.g., bench harness was contaminated; journal has a clean number).
HISTORICAL_SPEEDUP_OVERRIDE = {
    "coordinates.transform_coordinates": {
        # From journal.md 2026-04-17 (Phase 5 promotion), pre-coord-class
        # rust_covariance_transform wiring. Clean JAX-vs-Rust comparison
        # on 10k Cartesian-with-covariance → Keplerian.
        "speedup_p50": 1.84,
        "speedup_p95": 1.70,
        "note": "historical (2026-04-17 Phase 5 promotion; snapshot value contaminated)",
    },
    "dynamics.solve_lambert": {
        # From journal.md 2026-04-22 Lambert port. Median of N={100, 1k, 10k, 100k}
        # JAX vs Rust Izzo on the same random input batches.
        "speedup_p50": 2.10,
        "speedup_p95": 2.00,
        "note": "historical (2026-04-22 Lambert port; range 1.85×–2.34× across N)",
    },
    "orbit_determination.gaussIOD": {
        # From the now-deleted rust_orbit_determination_benchmark.py final run
        # (numba/python legacy vs fused-rust kernel on real-MPC triplets).
        "speedup_p50": 2.72,
        "speedup_p95": 2.81,
        "note": "historical (2026-04-22; numba+python legacy vs fused-rust kernel)",
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def _build_rows(
    history: dict[str, Any],
    baseline: dict[str, Any],
    current: dict[str, Any],
) -> list[dict[str, Any]]:
    validate_api_migrations()
    api_id_to_bench_key = {v: k for k, v in BENCH_TO_API_ID.items()}

    rows: list[dict[str, Any]] = []
    for migration in API_MIGRATIONS:
        row: dict[str, Any] = {
            "api_id": migration.api_id,
            "default_backend": migration.default,
            "status": migration.status,
            "boundary": migration.boundary,
            "parity_coverage": migration.parity_coverage,
            "partial": bool(migration.excluded_subcases),
            "waiver": migration.waiver or "-",
            "historical_speedup_p50": None,
            "historical_speedup_p95": None,
            "historical_note": "",
            "rust_p50_baseline_s": None,
            "rust_p95_baseline_s": None,
            "rust_p50_current_s": None,
            "rust_p95_current_s": None,
            "regression_p50": None,
            "regression_p95": None,
            "perf_source": "",
        }

        override = HISTORICAL_SPEEDUP_OVERRIDE.get(migration.api_id)
        if override is not None:
            row["historical_speedup_p50"] = override["speedup_p50"]
            row["historical_speedup_p95"] = override["speedup_p95"]
            row["historical_note"] = override.get("note", "")
        else:
            bench_key = api_id_to_bench_key.get(migration.api_id)
            if bench_key and bench_key in history:
                hist = history[bench_key]
                row["historical_speedup_p50"] = hist.get("speedup_p50")
                row["historical_speedup_p95"] = hist.get("speedup_p95")

        bench_key = api_id_to_bench_key.get(migration.api_id)
        if migration.latency_gate and bench_key:
            if bench_key in baseline:
                b = baseline[bench_key]
                row["rust_p50_baseline_s"] = b.get("rust_seconds_p50")
                row["rust_p95_baseline_s"] = b.get("rust_seconds_p95")
            if bench_key in current:
                c = current[bench_key]
                row["rust_p50_current_s"] = c.get("rust_seconds_p50")
                row["rust_p95_current_s"] = c.get("rust_seconds_p95")
                if row["rust_p50_baseline_s"]:
                    row["regression_p50"] = (
                        row["rust_p50_current_s"] / row["rust_p50_baseline_s"]
                    )
                if row["rust_p95_baseline_s"]:
                    row["regression_p95"] = (
                        row["rust_p95_current_s"] / row["rust_p95_baseline_s"]
                    )
            row["perf_source"] = "rust_latency"

        rows.append(row)

    rows.sort(key=lambda r: r["api_id"])
    return rows


def _render_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "API",
        "Default",
        "Status",
        "Boundary",
        "Parity coverage",
        "Hist speedup p50/p95",
        "Rust p50 base (ms)",
        "Rust p50 cur (ms)",
        "Regr p50/p95",
        "Waiver",
        "Source",
    ]

    def _speedup_cell(p50: Any, p95: Any) -> str:
        if p50 is None or p95 is None:
            return "N/A"
        return f"{_fmt_float(p50, 2)}x / {_fmt_float(p95, 2)}x"

    def _regr_cell(p50: Any, p95: Any) -> str:
        if p50 is None or p95 is None:
            return "N/A"
        return f"{_fmt_float(p50, 2)}x / {_fmt_float(p95, 2)}x"

    def _ms_cell(value: Any) -> str:
        if value is None:
            return "N/A"
        return _fmt_float(float(value) * 1000.0, 3)

    table_rows: list[list[str]] = []
    for row in rows:
        table_rows.append(
            [
                row["api_id"],
                str(row["default_backend"]),
                str(row["status"]),
                str(row["boundary"]),
                (
                    f"{row['parity_coverage']} (partial)"
                    if row["partial"]
                    else str(row["parity_coverage"])
                ),
                _speedup_cell(
                    row["historical_speedup_p50"], row["historical_speedup_p95"]
                ),
                _ms_cell(row["rust_p50_baseline_s"]),
                _ms_cell(row["rust_p50_current_s"]),
                _regr_cell(row["regression_p50"], row["regression_p95"]),
                str(row["waiver"]),
                str(row["perf_source"]) if row["perf_source"] else "-",
            ]
        )

    widths = [len(h) for h in headers]
    for cells in table_rows:
        for idx, cell in enumerate(cells):
            widths[idx] = max(widths[idx], len(cell))

    def fmt(cells: list[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(cells))

    lines = [fmt(headers), "-+-".join("-" * w for w in widths)]
    lines.extend(fmt(cells) for cells in table_rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history",
        type=Path,
        default=Path(
            "migration/artifacts/history/rust_vs_legacy_final_snapshot_2026-04-23.json"
        ),
        help="Frozen historical rust-vs-legacy snapshot (JAX/Numba baselines).",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("migration/artifacts/rust_latency_baseline.json"),
        help="Pinned Rust-only latency baseline.",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=Path("migration/artifacts/rust_latency_current.json"),
        help="Latest Rust-only latency measurement (if available).",
    )
    parser.add_argument("--format", choices=("table", "json"), default="table")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    history = _load_json(args.history) if args.history.exists() else {}
    baseline = _load_json(args.baseline) if args.baseline.exists() else {}
    current = _load_json(args.current) if args.current.exists() else {}

    rows = _build_rows(history, baseline, current)

    if args.format == "json":
        rendered = json.dumps(rows, indent=2)
    else:
        rendered = _render_table(rows)

    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")

    print(rendered)


if __name__ == "__main__":
    main()
