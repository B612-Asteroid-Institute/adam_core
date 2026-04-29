"""20% speedup gate: Rust >= 1.2x baseline main at p50 and p95.

For each measured API, time both the current Rust path (in-process) and the
baseline-main path (in the legacy-venv subprocess) on identical workloads, and
assert::

    legacy_p50 / rust_p50 >= 1.2
    legacy_p95 / rust_p95 >= 1.2

Each timing loop runs ``reps`` repetitions inside its respective
process so subprocess invocation overhead is excluded from the legacy
measurements.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from adam_core._rust.status import API_MIGRATIONS

from . import _inputs, _oracle, _rust_runner

MIN_SPEEDUP_P50 = 1.2
MIN_SPEEDUP_P95 = 1.2
PERF_WAIVERS_BY_API = {
    migration.api_id: migration.waiver
    for migration in API_MIGRATIONS
    if migration.waiver
}


def _percentile(samples: list[float], q: float) -> float:
    if not samples:
        return float("inf")
    return float(np.percentile(samples, q))


def _time_rust(api_id: str, kwargs: dict, *, reps: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        _rust_runner.run(api_id, **kwargs)
    samples: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _rust_runner.run(api_id, **kwargs)
        samples.append(time.perf_counter() - t0)
    return samples


@dataclass
class SpeedResult:
    api_id: str
    n: int
    rust_p50: float
    rust_p95: float
    legacy_p50: float
    legacy_p95: float
    speedup_p50: float
    speedup_p95: float
    raw_passed: bool
    passed: bool
    waived: bool = False
    waiver: str = ""
    error: Optional[str] = None
    # Cold-call (one-shot, includes process spawn + import + first call).
    rust_cold: Optional[float] = None
    legacy_cold: Optional[float] = None
    speedup_cold: Optional[float] = None


def _passes(
    legacy_p50: float, legacy_p95: float, rust_p50: float, rust_p95: float
) -> tuple[bool, float, float]:
    s50 = legacy_p50 / rust_p50 if rust_p50 > 0 else float("inf")
    s95 = legacy_p95 / rust_p95 if rust_p95 > 0 else float("inf")
    passed = s50 >= MIN_SPEEDUP_P50 and s95 >= MIN_SPEEDUP_P95
    return passed, s50, s95


def measure(
    api_id: str,
    *,
    n: int,
    reps: int = 7,
    warmup: int = 1,
    seed: int = 20260425,
    measure_cold: bool = False,
) -> SpeedResult:
    rng = np.random.default_rng(seed)
    try:
        sample = _inputs.make(api_id, rng, n)
    except Exception as e:
        return SpeedResult(
            api_id=api_id,
            n=n,
            rust_p50=float("inf"),
            rust_p95=float("inf"),
            legacy_p50=float("inf"),
            legacy_p95=float("inf"),
            speedup_p50=0.0,
            speedup_p95=0.0,
            raw_passed=False,
            passed=False,
            error=f"input gen: {type(e).__name__}: {e}",
        )

    try:
        rust_times = _time_rust(api_id, sample.rust_kwargs, reps=reps, warmup=warmup)
        legacy_times = _oracle.time_legacy(
            api_id, reps=reps, warmup=warmup, **sample.legacy_kwargs
        )
    except Exception as e:
        return SpeedResult(
            api_id=api_id,
            n=n,
            rust_p50=float("inf"),
            rust_p95=float("inf"),
            legacy_p50=float("inf"),
            legacy_p95=float("inf"),
            speedup_p50=0.0,
            speedup_p95=0.0,
            raw_passed=False,
            passed=False,
            error=f"timing: {type(e).__name__}: {e}",
        )

    rust_p50 = _percentile(rust_times, 50)
    rust_p95 = _percentile(rust_times, 95)
    legacy_p50 = _percentile(legacy_times, 50)
    legacy_p95 = _percentile(legacy_times, 95)
    raw_passed, s50, s95 = _passes(legacy_p50, legacy_p95, rust_p50, rust_p95)
    waiver = PERF_WAIVERS_BY_API.get(api_id, "")
    waived = bool(waiver and not raw_passed)
    passed = raw_passed or waived

    rust_cold = legacy_cold = speedup_cold = None
    if measure_cold:
        try:
            rust_cold = _oracle.time_rust_cold(api_id, **sample.rust_kwargs)
            legacy_cold = _oracle.time_legacy_cold(api_id, **sample.legacy_kwargs)
            speedup_cold = legacy_cold / rust_cold if rust_cold > 0 else float("inf")
        except Exception as e:
            # Cold timing failure shouldn't fail the gate — record and move on.
            rust_cold = legacy_cold = speedup_cold = None
            print(f"  [cold-time error for {api_id}: {e}]", file=sys.stderr)

    return SpeedResult(
        api_id=api_id,
        n=n,
        rust_p50=rust_p50,
        rust_p95=rust_p95,
        legacy_p50=legacy_p50,
        legacy_p95=legacy_p95,
        speedup_p50=s50,
        speedup_p95=s95,
        raw_passed=raw_passed,
        passed=passed,
        waived=waived,
        waiver=waiver,
        rust_cold=rust_cold,
        legacy_cold=legacy_cold,
        speedup_cold=speedup_cold,
    )


def measure_all(
    api_ids: list[str],
    *,
    n: int,
    reps: int = 7,
    warmup: int = 1,
    seed: int = 20260425,
    measure_cold: bool = False,
) -> list[SpeedResult]:
    return [
        measure(a, n=n, reps=reps, warmup=warmup, seed=seed, measure_cold=measure_cold)
        for a in api_ids
    ]


def format_summary(results: list[SpeedResult]) -> str:
    has_cold = any(r.rust_cold is not None for r in results)
    lines = []
    if has_cold:
        lines.append(
            f"{'API':50s}  {'rust warm':>10s}  {'leg warm':>10s}  {'×warm':>6s}  "
            f"{'rust cold':>11s}  {'leg cold':>10s}  {'×cold':>6s}  flag"
        )
        lines.append("-" * 130)
    else:
        lines.append(
            f"{'API':50s}  {'rust p50':>10s}  {'leg p50':>10s}  "
            f"{'×p50':>6s}  {'×p95':>6s}  flag"
        )
        lines.append("-" * 100)
    for r in results:
        flag = "" if r.passed else "FAIL"
        if r.waived and not r.raw_passed:
            flag = f"WAIVED ({r.waiver})"
        if r.error:
            flag = f"ERR ({r.error[:40]})"
        if has_cold:
            cold_str = (
                f"{r.rust_cold*1000:>10.1f}ms  {r.legacy_cold*1000:>9.1f}ms  "
                f"{r.speedup_cold:>5.2f}x"
                if r.rust_cold is not None
                else f"{'—':>11s}  {'—':>10s}  {'—':>6s}"
            )
            lines.append(
                f"{r.api_id:50s}  "
                f"{r.rust_p50*1e6:>9.1f}μs  {r.legacy_p50*1e6:>9.1f}μs  "
                f"{r.speedup_p50:>5.2f}x  {cold_str}  {flag}"
            )
        else:
            lines.append(
                f"{r.api_id:50s}  "
                f"{r.rust_p50*1e6:>9.1f}μs  {r.legacy_p50*1e6:>9.1f}μs  "
                f"{r.speedup_p50:>5.2f}x  {r.speedup_p95:>5.2f}x  {flag}"
            )
    return "\n".join(lines)


def to_json(results: list[SpeedResult]) -> dict:
    return {
        "min_speedup_p50": MIN_SPEEDUP_P50,
        "min_speedup_p95": MIN_SPEEDUP_P95,
        "apis": [
            {
                "api_id": r.api_id,
                "n": r.n,
                "rust_p50_s": r.rust_p50,
                "rust_p95_s": r.rust_p95,
                "legacy_p50_s": r.legacy_p50,
                "legacy_p95_s": r.legacy_p95,
                "speedup_p50": r.speedup_p50,
                "speedup_p95": r.speedup_p95,
                "raw_passed": r.raw_passed,
                "passed": r.passed,
                "waived": r.waived,
                "waiver": r.waiver,
                "error": r.error,
                "rust_cold_s": r.rust_cold,
                "legacy_cold_s": r.legacy_cold,
                "speedup_cold": r.speedup_cold,
            }
            for r in results
        ],
        "all_passed": all(r.passed for r in results),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rust-vs-baseline-main speedup gate.")
    p.add_argument(
        "--n",
        type=int,
        default=2000,
        help="Workload size for timing (default: 2000).",
    )
    p.add_argument(
        "--reps",
        type=int,
        default=7,
        help="Timing reps (default: 7 — odd so p50 lands on a sample).",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup reps before timing (default: 1).",
    )
    p.add_argument(
        "--apis",
        nargs="*",
        default=None,
        help="Specific API ids to gate. Defaults to ALL.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20260425,
        help="RNG seed for the timing workload.",
    )
    p.add_argument(
        "--cold",
        action="store_true",
        help="Also measure cold-call latency (fresh subprocess per call).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write JSON artifact.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    api_ids = args.apis or list(_inputs.all_api_ids())
    results = measure_all(
        api_ids,
        n=args.n,
        reps=args.reps,
        warmup=args.warmup,
        seed=args.seed,
        measure_cold=args.cold,
    )
    print(format_summary(results))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(to_json(results), indent=2))
        print(f"\nwrote {args.output}", file=sys.stderr)

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
