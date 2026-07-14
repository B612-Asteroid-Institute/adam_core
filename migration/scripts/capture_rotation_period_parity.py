#!/usr/bin/env python3
"""Capture latest-main/current/native rotation-period migration evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEGACY_COMMIT = "936cc636096fcfefcee3e1310c21528444f39546"
RUNNER = r"""
import json, time
import numpy as np
import pyarrow as pa
from adam_core.time import Timestamp
from adam_core.photometry.rotation import RotationPeriodObservations, estimate_rotation_period

n = 60
span_days = 0.08
period_days = 0.02
mjd = np.linspace(60000.0, 60000.0 + span_days, n, dtype=np.float64)
t_rel = mjd - mjd.min()
filters = np.asarray([("LSST_g", "LSST_r")[i % 2] for i in range(n)], dtype=object)
sessions = np.asarray([f"X05:{int(60000 + (i // max(1, n // 6)))}" for i in range(n)], dtype=object)
r_au = np.full(n, 2.0)
delta_au = np.full(n, 1.5)
phase_angle = 12.0 + 4.0 * np.sin(2.0 * np.pi * t_rel / span_days)
baseline = 15.0 + 5.0 * np.log10(r_au * delta_au) + 0.015 * phase_angle + 0.0015 * phase_angle**2
phase = 2.0 * np.pi * t_rel / period_days
rotation = 0.10 * np.cos(phase) + 0.35 * np.cos(2.0 * phase) + 0.07 * np.sin(2.0 * phase)
rotation += np.asarray([0.03 if name == "LSST_g" else 0.0 for name in filters])
rng = np.random.default_rng(20260414)
mag = baseline + rotation + rng.normal(0.0, 0.01, size=n)
observations = RotationPeriodObservations.from_kwargs(
    time=Timestamp.from_mjd(mjd, scale="tdb"),
    mag=pa.array(mag, type=pa.float64()),
    mag_sigma=pa.array(np.full(n, 0.03), type=pa.float64()),
    filter=pa.array(filters.tolist(), type=pa.large_string()),
    session_id=pa.array(sessions.tolist(), type=pa.large_string()),
    r_au=pa.array(r_au, type=pa.float64()),
    delta_au=pa.array(delta_au, type=pa.float64()),
    phase_angle_deg=pa.array(phase_angle, type=pa.float64()),
)
options = dict(search_fidelity="exact_grid", fourier_orders=(2, 3), max_frequency_cycles_per_day=120.0, frequency_grid_scale=40.0)
estimate_rotation_period(observations, **options)
samples = []
result = None
for _ in range(5):
    started = time.perf_counter()
    result = estimate_rotation_period(observations, **options)
    samples.append(time.perf_counter() - started)
assert result is not None
row = {
    "period_days": result.period_days[0].as_py(),
    "period_verdict": result.period_verdict[0].as_py(),
    "reliability_code": result.reliability_code[0].as_py(),
    "is_period_doubled": result.is_period_doubled[0].as_py(),
    "fourier_order": result.fourier_order[0].as_py(),
    "n_observations": result.n_observations[0].as_py(),
    "n_fit_observations": result.n_fit_observations[0].as_py(),
    "n_clipped": result.n_clipped[0].as_py(),
    "n_significant_aliases": result.n_significant_aliases[0].as_py(),
    "amplitude_snr": result.amplitude_snr[0].as_py(),
    "phase_coverage_fraction": result.phase_coverage_fraction[0].as_py(),
    "confidence_flags": result.confidence_flags[0].as_py(),
    "insufficiency_reasons": result.insufficiency_reasons[0].as_py(),
}
native = None
try:
    from adam_core.photometry.rotation.estimator import _benchmark_rotation_period_native
    native = _benchmark_rotation_period_native(observations, reps=3, trials=2, warmup_reps=1, **options)
except ImportError:
    pass
print(json.dumps({"result": row, "public_samples_seconds": samples, "native_samples_seconds": native}, sort_keys=True))
"""


def run(python: Path, cwd: Path) -> dict[str, object]:
    completed = subprocess.run(
        [str(python), "-c", RUNNER],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy-python", type=Path, default=ROOT / ".legacy-venv/bin/python"
    )
    parser.add_argument(
        "--legacy-repo", type=Path, default=ROOT.parent / "adam-core-legacy-main"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "migration/artifacts/rotation_period_parity.json",
    )
    args = parser.parse_args()
    legacy_commit = commit(args.legacy_repo)
    if legacy_commit != LEGACY_COMMIT:
        raise SystemExit(f"legacy commit mismatch: {legacy_commit} != {LEGACY_COMMIT}")
    current = run(Path(sys.executable), ROOT)
    legacy = run(args.legacy_python, args.legacy_repo)
    current_result = current["result"]
    legacy_result = legacy["result"]
    period_relative_error = abs(
        float(current_result["period_days"]) - float(legacy_result["period_days"])
    ) / float(legacy_result["period_days"])
    exact_fields = [
        "period_verdict",
        "reliability_code",
        "is_period_doubled",
        "fourier_order",
        "n_observations",
        "n_fit_observations",
        "n_clipped",
    ]
    exact_match = all(current_result[key] == legacy_result[key] for key in exact_fields)
    parity_passed = exact_match and period_relative_error <= 0.01
    current_median = statistics.median(current["public_samples_seconds"])
    legacy_median = statistics.median(legacy["public_samples_seconds"])
    native_flat = [
        sample
        for trial in (current["native_samples_seconds"] or [])
        for sample in trial
    ]
    report = {
        "schema_version": 1,
        "identity": "rotation-period-latest-main-exact-grid-v1",
        "runner_sha256": hashlib.sha256(RUNNER.encode()).hexdigest(),
        "legacy_commit": legacy_commit,
        "current_commit": commit(ROOT),
        "parity": {
            "passed": parity_passed,
            "period_relative_error": period_relative_error,
            "exact_fields_match": exact_match,
        },
        "legacy": legacy,
        "current": current,
        "timing": {
            "legacy_public_median_seconds": legacy_median,
            "current_public_median_seconds": current_median,
            "legacy_over_current": legacy_median / current_median,
            "native_rust_median_seconds": statistics.median(native_flat),
            "native_clock": "std::time::Instant",
            "python_conversion_outside_native_samples": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}; parity_passed={parity_passed}")
    if not parity_passed:
        raise SystemExit("rotation-period parity failed")


if __name__ == "__main__":
    main()
