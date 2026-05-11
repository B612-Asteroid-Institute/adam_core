"""Deterministic fixed-fixture parity gate.

Randomized fuzz is the default parity mechanism, but some Rust/default APIs are
not good random-fuzz candidates.  ``orbit_determination.gaussIOD`` is the first
case: Rust uses Laguerre+deflation for the 8th-order Gauss-IOD polynomial while
baseline-main uses ``np.roots``/LAPACK, and those algorithms accept different
physical-root subsets on some random triplets.  This module keeps that random
exclusion intact while still governing well-conditioned deterministic triplets
against the baseline-main oracle.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import _inputs, _oracle, _rust_runner, parity_fuzz, tolerances


@dataclass(frozen=True)
class FixedFixtureSpec:
    api_id: str
    name: str
    description: str
    make_sample: Callable[[], _inputs.Sample]


@dataclass
class FixtureResult:
    name: str
    description: str
    n: int
    outputs: list[parity_fuzz.OutputResult] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.error is None and all(output.passed for output in self.outputs)


@dataclass
class ApiResult:
    api_id: str
    investigate: bool
    investigate_task: str
    fixtures: list[FixtureResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(fixture.passed for fixture in self.fixtures)


def _gauss_iod_well_conditioned_sample() -> _inputs.Sample:
    """Eight fixed triplets where Rust and legacy both accept a physical root.

    These values are frozen from ``_inputs.make_gauss_iod`` with seed
    ``20260425`` after confirming every triplet has a shared best |r2| ≥ 1.5 AU
    solution.  Do not replace this with live RNG generation: the point of this
    gate is a stable fixture for the well-conditioned subset while randomized
    Gauss-IOD fuzz remains excluded.
    """

    ra_deg_per_triplet = np.array(
        [
            [57.8492351997515, 57.791953931637394, 57.73672992311044],
            [17.897986937496036, 17.98326883810352, 18.059466080864983],
            [161.221095824849, 161.3759225154983, 161.5403920048698],
            [292.19304489247816, 292.07580798107136, 291.7766164617784],
            [156.29415316024665, 156.5017145339414, 156.68043443580362],
            [219.8969791658994, 219.83586591985298, 219.727945361609],
            [220.0237727074175, 219.9236948574344, 219.88138889659285],
            [76.2239867351426, 76.35894641298141, 76.54304981115668],
        ],
        dtype=np.float64,
    )
    dec_deg_per_triplet = np.array(
        [
            [-0.6765348320549983, -0.9959866403096104, -1.3053527989774307],
            [5.867435083102628, 5.902802310172872, 5.934386573196944],
            [-15.213251903451415, -15.483688683229207, -15.769630216956912],
            [-21.909899201771566, -22.05925007753948, -22.437984575929246],
            [31.98769378926163, 31.805748499604743, 31.648222885723566],
            [11.541376660991412, 11.557830090811969, 11.586836528771231],
            [-1.9123542651718335, -1.839329099540702, -1.808464059331334],
            [13.85107387473804, 13.876781218396967, 13.91170764279283],
        ],
        dtype=np.float64,
    )
    times_per_triplet = np.array(
        [
            [59901.14655104109, 59903.950165199814, 59906.66478175893],
            [59294.440593239735, 59295.70762092547, 59296.839451877524],
            [59535.42574868939, 59538.22713149738, 59541.17937261571],
            [59241.861800446284, 59243.01771144071, 59245.95123929142],
            [59502.38634995574, 59505.319284902725, 59507.85485829525],
            [59457.17248017955, 59458.43835877859, 59460.672958895135],
            [59309.274304117855, 59311.70661422978, 59312.73455691219],
            [59631.57710271599, 59633.277164528125, 59635.59150033385],
        ],
        dtype=np.float64,
    )
    obs_pos_per_triplet = np.broadcast_to(
        np.array([[[1.0, 0.0, 0.0]]], dtype=np.float64),
        (ra_deg_per_triplet.shape[0], 3, 3),
    ).copy()
    kwargs: dict[str, Any] = {
        "ra_deg_per_triplet": ra_deg_per_triplet,
        "dec_deg_per_triplet": dec_deg_per_triplet,
        "times_per_triplet": times_per_triplet,
        "obs_pos_per_triplet": obs_pos_per_triplet,
        "mu": _inputs.MU_SUN,
        "c": _inputs.C_AU_PER_DAY,
    }
    return _inputs.Sample(rust_kwargs=kwargs, legacy_kwargs=kwargs)


FIXTURES: tuple[FixedFixtureSpec, ...] = (
    FixedFixtureSpec(
        api_id="orbit_determination.gaussIOD",
        name="well_conditioned_seed_20260425",
        description=(
            "Eight deterministic Gauss-IOD triplets with shared physical best "
            "roots on Rust Laguerre+deflation and legacy np.roots/LAPACK."
        ),
        make_sample=_gauss_iod_well_conditioned_sample,
    ),
)

FIXTURES_BY_API: dict[str, tuple[FixedFixtureSpec, ...]] = {}
for fixture in FIXTURES:
    FIXTURES_BY_API.setdefault(fixture.api_id, ())
    FIXTURES_BY_API[fixture.api_id] = FIXTURES_BY_API[fixture.api_id] + (fixture,)


def all_api_ids() -> tuple[str, ...]:
    return tuple(FIXTURES_BY_API.keys())


def _sample_size(sample: _inputs.Sample) -> int:
    for value in sample.rust_kwargs.values():
        if isinstance(value, np.ndarray) and value.ndim > 0:
            return int(value.shape[0])
    return 1


def fixed_one(api_id: str) -> ApiResult:
    spec = tolerances.get(api_id)
    api = ApiResult(
        api_id=api_id,
        investigate=spec.investigate,
        investigate_task=spec.investigate_task,
    )
    if api_id not in FIXTURES_BY_API:
        raise KeyError(f"No fixed parity fixtures for {api_id!r}")

    for fixture in FIXTURES_BY_API[api_id]:
        sample = fixture.make_sample()
        result = FixtureResult(
            name=fixture.name,
            description=fixture.description,
            n=_sample_size(sample),
        )
        try:
            rust_out = _rust_runner.run(api_id, **sample.rust_kwargs)
            legacy_out = _oracle.parity(api_id, **sample.legacy_kwargs)
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            api.fixtures.append(result)
            continue

        for out_name, tol in spec.outputs.items():
            if out_name not in rust_out:
                result.error = f"missing rust output {out_name!r}"
                break
            if out_name not in legacy_out:
                result.error = f"missing legacy output {out_name!r}"
                break
            result.outputs.append(
                parity_fuzz._check_output(  # noqa: SLF001 - shared gate semantics.
                    out_name,
                    rust_out[out_name],
                    legacy_out[out_name],
                    tol,
                )
            )
        api.fixtures.append(result)
    return api


def fixed_all(api_ids: list[str]) -> list[ApiResult]:
    return [fixed_one(api_id) for api_id in api_ids]


def format_summary(results: list[ApiResult]) -> str:
    lines = [
        f"{'API':50s}  {'pass/total':>12s}  {'worst_abs':>12s}  {'worst_rel':>12s}  flag",
        "-" * 110,
    ]
    for result in results:
        passed = sum(1 for fixture in result.fixtures if fixture.passed)
        total = len(result.fixtures)
        worst_abs = max(
            (
                output.max_abs
                for fixture in result.fixtures
                for output in fixture.outputs
            ),
            default=0.0,
        )
        worst_rel = max(
            (
                output.max_rel
                for fixture in result.fixtures
                for output in fixture.outputs
            ),
            default=0.0,
        )
        flag = ""
        if result.investigate:
            flag = f"INVESTIGATE {result.investigate_task}"
        if not result.passed:
            flag = "FAIL " + flag
        lines.append(
            f"{result.api_id:50s}  {f'{passed}/{total}':>12s}  "
            f"{worst_abs:>12.3e}  {worst_rel:>12.3e}  {flag}"
        )
    return "\n".join(lines)


def to_json(results: list[ApiResult]) -> dict[str, Any]:
    return {
        "apis": [
            {
                "api_id": result.api_id,
                "passed": result.passed,
                "investigate": result.investigate,
                "investigate_task": result.investigate_task,
                "fixtures": [
                    {
                        "name": fixture.name,
                        "description": fixture.description,
                        "n": fixture.n,
                        "passed": fixture.passed,
                        "error": fixture.error,
                        "outputs": [
                            {
                                "name": output.name,
                                "max_abs": output.max_abs,
                                "max_rel": output.max_rel,
                                "atol": output.atol,
                                "rtol": output.rtol,
                                "nan_disagreement": output.nan_disagreement,
                                "passed": output.passed,
                            }
                            for output in fixture.outputs
                        ],
                    }
                    for fixture in result.fixtures
                ],
            }
            for result in results
        ],
        "all_passed": all(result.passed for result in results),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic fixed-fixture rust-vs-legacy parity gate."
    )
    parser.add_argument(
        "--apis",
        nargs="*",
        default=None,
        help="Specific fixed-fixture API ids (default: all fixed fixtures).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("migration/artifacts/parity_fixed_fixtures.json"),
        help="JSON artifact path.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    api_ids = args.apis or list(all_api_ids())
    unknown = sorted(set(api_ids) - set(all_api_ids()))
    if unknown:
        raise SystemExit("No fixed parity fixtures for: " + ", ".join(unknown))

    results = fixed_all(api_ids)
    print(format_summary(results))

    artifact = to_json(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2))
    print(f"\nwrote {args.output}")
    return 0 if artifact["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
