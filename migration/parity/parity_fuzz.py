"""Randomized fuzz parity gate.

For each rust-default API:

1. Sample ``seeds`` random workloads of size ``n``.
2. Run both the rust and the legacy implementation on identical inputs.
3. Per output, compute the worst absolute and relative diff and assert
   it satisfies the per-API tolerance from ``tolerances.py``.

A failure is recorded but the gate continues across all APIs so we get
the full picture in one run, then exits non-zero if any failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import _inputs, _oracle, _rust_runner, tolerances


@dataclass
class OutputResult:
    name: str
    max_abs: float
    max_rel: float
    atol: float
    rtol: float
    passed: bool
    nan_disagreement: int = 0


@dataclass
class SeedResult:
    seed: int
    n: int
    outputs: list[OutputResult] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.error is None and all(o.passed for o in self.outputs)


@dataclass
class ApiResult:
    api_id: str
    investigate: bool
    investigate_task: str
    seeds: list[SeedResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(s.passed for s in self.seeds)


def _max_abs_rel(rust: np.ndarray, legacy: np.ndarray) -> tuple[float, float, int]:
    """Worst |abs| and |rel| diff across finite cells; count NaN mismatches."""
    rust = np.asarray(rust, dtype=np.float64)
    legacy = np.asarray(legacy, dtype=np.float64)
    if rust.shape != legacy.shape:
        raise ValueError(f"shape mismatch: rust {rust.shape} vs legacy {legacy.shape}")

    rust_nan = ~np.isfinite(rust)
    legacy_nan = ~np.isfinite(legacy)
    nan_mismatch = int(np.sum(rust_nan ^ legacy_nan))

    finite = (~rust_nan) & (~legacy_nan)
    if not np.any(finite):
        return 0.0, 0.0, nan_mismatch

    diff = rust[finite] - legacy[finite]
    abs_max = float(np.max(np.abs(diff))) if diff.size else 0.0

    denom = np.maximum(np.abs(legacy[finite]), np.finfo(np.float64).tiny)
    rel = np.abs(diff) / denom
    rel_max = float(np.max(rel)) if rel.size else 0.0

    return abs_max, rel_max, nan_mismatch


def _check_output(
    name: str,
    rust: np.ndarray,
    legacy: np.ndarray,
    tol: tolerances.OutputTol,
) -> OutputResult:
    abs_max, rel_max, nan_mismatch = _max_abs_rel(rust, legacy)
    # Pass condition: |abs_diff| ≤ atol + rtol * |legacy|.
    # We use `np.allclose` semantics: pass if (abs ≤ atol + rtol·|legacy|).
    rust_arr = np.asarray(rust, dtype=np.float64)
    legacy_arr = np.asarray(legacy, dtype=np.float64)
    finite = np.isfinite(rust_arr) & np.isfinite(legacy_arr)
    nan_match = (~np.isfinite(rust_arr)) == (~np.isfinite(legacy_arr))
    finite_pass = bool(
        np.all(
            np.abs(rust_arr[finite] - legacy_arr[finite])
            <= tol.atol + tol.rtol * np.abs(legacy_arr[finite])
        )
    )
    passed = finite_pass and bool(np.all(nan_match))
    return OutputResult(
        name=name,
        max_abs=abs_max,
        max_rel=rel_max,
        atol=tol.atol,
        rtol=tol.rtol,
        passed=passed,
        nan_disagreement=nan_mismatch,
    )


def fuzz_one(
    api_id: str, seeds: int, n: int, base_seed: int = 0
) -> ApiResult:
    spec = tolerances.get(api_id)
    api = ApiResult(
        api_id=api_id,
        investigate=spec.investigate,
        investigate_task=spec.investigate_task,
    )

    for s in range(seeds):
        seed = base_seed + s
        rng = np.random.default_rng(seed)
        seed_result = SeedResult(seed=seed, n=n)

        try:
            sample = _inputs.make(api_id, rng, n)
            rust_out = _rust_runner.run(api_id, **sample.rust_kwargs)
            legacy_out = _oracle.parity(api_id, **sample.legacy_kwargs)
        except Exception as e:
            seed_result.error = f"{type(e).__name__}: {e}"
            api.seeds.append(seed_result)
            continue

        for out_name, tol in spec.outputs.items():
            if out_name not in rust_out:
                seed_result.error = f"missing rust output {out_name!r}"
                break
            if out_name not in legacy_out:
                seed_result.error = f"missing legacy output {out_name!r}"
                break
            seed_result.outputs.append(
                _check_output(out_name, rust_out[out_name], legacy_out[out_name], tol)
            )

        api.seeds.append(seed_result)

    return api


def fuzz_all(
    api_ids: list[str], seeds: int, n: int, base_seed: int = 0
) -> list[ApiResult]:
    results = []
    for api_id in api_ids:
        results.append(fuzz_one(api_id, seeds=seeds, n=n, base_seed=base_seed))
    return results


def format_summary(results: list[ApiResult]) -> str:
    lines = []
    lines.append(f"{'API':50s}  {'pass/total':>12s}  {'worst_abs':>12s}  {'worst_rel':>12s}  flag")
    lines.append("-" * 110)
    for r in results:
        passed = sum(1 for s in r.seeds if s.passed)
        total = len(r.seeds)
        worst_abs = max(
            (o.max_abs for s in r.seeds for o in s.outputs if not s.error),
            default=0.0,
        )
        worst_rel = max(
            (o.max_rel for s in r.seeds for o in s.outputs if not s.error),
            default=0.0,
        )
        flag = ""
        if r.investigate:
            flag = f"INVESTIGATE {r.investigate_task}"
        if not r.passed:
            flag = "FAIL " + flag
        lines.append(
            f"{r.api_id:50s}  {f'{passed}/{total}':>12s}  "
            f"{worst_abs:>12.3e}  {worst_rel:>12.3e}  {flag}"
        )
    return "\n".join(lines)


def to_json(results: list[ApiResult]) -> dict:
    return {
        "apis": [
            {
                "api_id": r.api_id,
                "passed": r.passed,
                "investigate": r.investigate,
                "investigate_task": r.investigate_task,
                "seeds": [
                    {
                        "seed": s.seed,
                        "n": s.n,
                        "passed": s.passed,
                        "error": s.error,
                        "outputs": [
                            {
                                "name": o.name,
                                "max_abs": o.max_abs,
                                "max_rel": o.max_rel,
                                "atol": o.atol,
                                "rtol": o.rtol,
                                "nan_disagreement": o.nan_disagreement,
                                "passed": o.passed,
                            }
                            for o in s.outputs
                        ],
                    }
                    for s in r.seeds
                ],
            }
            for r in results
        ],
        "all_passed": all(r.passed for r in results),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Randomized rust-vs-legacy parity gate."
    )
    p.add_argument(
        "--seeds",
        type=int,
        default=8,
        help="Number of random seeds per API (default: 8).",
    )
    p.add_argument(
        "--n",
        type=int,
        default=128,
        help="Workload size per seed (default: 128).",
    )
    p.add_argument(
        "--apis",
        nargs="*",
        default=None,
        help="Specific API ids to gate. Defaults to ALL.",
    )
    p.add_argument(
        "--base-seed",
        type=int,
        default=20260425,
        help="Base RNG seed (default: today's date).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write JSON artifact (default: do not write).",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    api_ids = args.apis or list(_inputs.all_api_ids())

    results = fuzz_all(api_ids, seeds=args.seeds, n=args.n, base_seed=args.base_seed)
    print(format_summary(results))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(to_json(results), indent=2))
        print(f"\nwrote {args.output}", file=sys.stderr)

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
