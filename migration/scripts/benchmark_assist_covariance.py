"""Two-runtime benchmark for legacy Python vs Rust ASSIST covariance.

This is the RM-STANDALONE-007B apples-to-apples covariance benchmark hook for
``ASSISTPropagator.propagate_orbits(..., covariance=True)`` public semantics. It
times the public sampled-covariance path (variant creation, variant
propagation, and collapse-to-nominal covariance) for legacy Python
``adam_assist`` in the isolated ``.legacy-assist-venv`` and local Rust
``adam_assist`` over identical quivr orbit/time workloads, and records
timing plus state and covariance residual metadata.

This is intentionally kept separate from the propagation-only state benchmark
(``benchmark_assist_public_semantics.py``) so the covariance numbers never get
conflated with the ``covariance=False`` state-path speed claim.

Stochastic policy
-----------------
Sigma-point and ``auto`` (when sigma points reconstruct the input covariance)
are deterministic in both Python ``adam_assist`` and ``adam_assist``, so
their collapsed covariance is compared element-wise (``parity_expected=True``).
Monte-carlo sampling uses NumPy ``default_rng`` in Python and an internal
SplitMix64 normal sampler in Rust; identical seeds do not produce identical
draws, so monte-carlo covariance is compared statistically
(``parity_expected=False``) via relative Frobenius norm and 1-sigma agreement.
The nominal mean state stays a strict parity check for every method because the
nominal orbit is propagated deterministically (not sampled) before collapse.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from adam_assist import ASSISTPropagator as RustASSISTPropagator
from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.orbits.orbits import Orbits
from adam_core.time import Timestamp
from jpl_small_bodies_de441_n16 import de441_n16
from naif_de440 import de440

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from migration.parity._assist_bench import (  # noqa: E402
    PERFORMANCE_COLUMNS,
    TWO_RUNTIME_COMPARISON_MODE,
    performance_timing_payload,
    time_native_rust,
    time_rust,
)
from migration.parity._assist_oracle import (  # noqa: E402
    LEGACY_ASSIST_VENV_PYTHON,
    LegacyAssistPropagator,
)
from migration.scripts import benchmark_assist_public_semantics as base  # noqa: E402

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "migration"
    / "artifacts"
    / "assist_public_semantics_covariance_benchmark_2026-06-20.json"
)
BENCHMARK_LANES = ("tiny", "small", "large")
CovarianceMethod = Literal["auto", "sigma-point", "monte-carlo"]


@dataclass(frozen=True)
class CovarianceWorkload:
    lane: str
    name: str
    description: str
    orbits: Orbits
    times: Timestamp
    chunk_size: int
    covariance_method: CovarianceMethod
    num_samples: int
    seed: int | None
    parity_expected: bool


def _with_covariance(
    orbits: Orbits,
    *,
    pos_sigma_au: float = 1.0e-8,
    vel_sigma_au_per_day: float = 1.0e-10,
) -> Orbits:
    """Attach a well-conditioned diagonal Cartesian covariance to every row."""
    rows = len(orbits)
    sigmas = np.tile(
        np.array(
            [
                pos_sigma_au,
                pos_sigma_au,
                pos_sigma_au,
                vel_sigma_au_per_day,
                vel_sigma_au_per_day,
                vel_sigma_au_per_day,
            ],
            dtype=np.float64,
        ),
        (rows, 1),
    )
    return orbits.set_column(
        "coordinates.covariance", CoordinateCovariances.from_sigmas(sigmas)
    )


def _workloads() -> list[CovarianceWorkload]:
    return [
        CovarianceWorkload(
            lane="tiny",
            name="tiny_cov_sigma_point_sun_ecliptic_tdb_4x3",
            description=(
                "Tiny deterministic sigma-point covariance: 4 SUN/ecliptic/TDB "
                "same-epoch orbits by 3 target epochs."
            ),
            orbits=_with_covariance(
                base._base_sun_ecliptic_orbits(4, mixed_epochs=False)
            ),
            times=base._target_times(3, scale="tdb"),
            chunk_size=100,
            covariance_method="sigma-point",
            num_samples=1000,
            seed=None,
            parity_expected=True,
        ),
        CovarianceWorkload(
            lane="small",
            name="small_cov_sigma_point_sun_ecliptic_tdb_25x20",
            description=(
                "Small deterministic sigma-point covariance: 25 SUN/ecliptic/TDB "
                "same-epoch orbits by 20 target epochs."
            ),
            orbits=_with_covariance(
                base._base_sun_ecliptic_orbits(25, mixed_epochs=False)
            ),
            times=base._target_times(20, scale="tdb"),
            chunk_size=200,
            covariance_method="sigma-point",
            num_samples=1000,
            seed=None,
            parity_expected=True,
        ),
        CovarianceWorkload(
            lane="small",
            name="small_cov_auto_sun_ecliptic_tdb_25x20",
            description=(
                "Small deterministic auto covariance (sigma-point reconstructs the "
                "well-conditioned input): 25 SUN/ecliptic/TDB same-epoch orbits by "
                "20 target epochs."
            ),
            orbits=_with_covariance(
                base._base_sun_ecliptic_orbits(25, mixed_epochs=False)
            ),
            times=base._target_times(20, scale="tdb"),
            chunk_size=200,
            covariance_method="auto",
            num_samples=1000,
            seed=None,
            parity_expected=True,
        ),
        CovarianceWorkload(
            lane="small",
            name="small_cov_monte_carlo_sun_ecliptic_tdb_10x10",
            description=(
                "Small statistical monte-carlo covariance: 10 SUN/ecliptic/TDB "
                "same-epoch orbits by 10 target epochs, 512 samples, fixed seed. "
                "Compared statistically because NumPy and SplitMix64 RNGs differ."
            ),
            orbits=_with_covariance(
                base._base_sun_ecliptic_orbits(10, mixed_epochs=False)
            ),
            times=base._target_times(10, scale="tdb"),
            chunk_size=200,
            covariance_method="monte-carlo",
            num_samples=512,
            seed=20260620,
            parity_expected=False,
        ),
        CovarianceWorkload(
            lane="large",
            name="large_cov_sigma_point_sun_ecliptic_tdb_100x50_1yr",
            description=(
                "Large deterministic sigma-point covariance over a one-year "
                "horizon: 100 SUN/ecliptic/TDB same-epoch orbits by 50 target "
                "epochs across 365 days."
            ),
            orbits=_with_covariance(
                base._base_sun_ecliptic_orbits(100, mixed_epochs=False)
            ),
            times=base._long_horizon_target_times(50, scale="tdb"),
            chunk_size=200,
            covariance_method="sigma-point",
            num_samples=1000,
            seed=None,
            parity_expected=True,
        ),
        CovarianceWorkload(
            lane="large",
            name="large_cov_sigma_point_unique_input_epochs_50x25_1yr",
            description=(
                "Large realistic unique-input-epoch sigma-point covariance over a "
                "one-year horizon: 50 SUN/ecliptic/TDB orbits with 50 unique "
                "initial epochs by 25 target epochs across 365 days. Exercises the "
                "one-by-one per-orbit covariance fallback."
            ),
            orbits=_with_covariance(
                base._base_sun_ecliptic_orbits(
                    50,
                    mixed_epochs=False,
                    epoch_offsets=base._unique_initial_epoch_offsets(50),
                )
            ),
            times=base._long_horizon_target_times(25, scale="tdb"),
            chunk_size=200,
            covariance_method="sigma-point",
            num_samples=1000,
            seed=None,
            parity_expected=True,
        ),
    ]


def _covariance_residuals(actual: Orbits, expected: Orbits) -> dict[str, Any]:
    a = np.asarray(actual.coordinates.covariance.to_matrix(), dtype=np.float64)
    e = np.asarray(expected.coordinates.covariance.to_matrix(), dtype=np.float64)
    if a.shape != e.shape:
        raise AssertionError(f"covariance shape mismatch: {a.shape} != {e.shape}")
    abs_diff = np.abs(a - e)
    e_flat = e.reshape(e.shape[0], -1)
    diff_flat = (a - e).reshape(a.shape[0], -1)
    fro_e = np.linalg.norm(e_flat, axis=1)
    fro_diff = np.linalg.norm(diff_flat, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_frobenius = np.where(fro_e > 0.0, fro_diff / fro_e, 0.0)
    diag_a = np.sqrt(np.clip(np.diagonal(a, axis1=1, axis2=2), 0.0, None))
    diag_e = np.sqrt(np.clip(np.diagonal(e, axis1=1, axis2=2), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_rel = np.where(diag_e > 0.0, np.abs(diag_a - diag_e) / diag_e, 0.0)
    return {
        "rows": int(a.shape[0]),
        "rust_finite": bool(np.isfinite(a).all()),
        "python_finite": bool(np.isfinite(e).all()),
        "max_abs": float(abs_diff.max(initial=0.0)),
        "max_rel_frobenius": float(rel_frobenius.max(initial=0.0)),
        "mean_rel_frobenius": (
            float(rel_frobenius.mean()) if rel_frobenius.size else 0.0
        ),
        "max_sigma_rel": float(sigma_rel.max(initial=0.0)),
        "mean_sigma_rel": float(sigma_rel.mean()) if sigma_rel.size else 0.0,
    }


def _benchmark_workload(
    workload: CovarianceWorkload,
    *,
    legacy_propagator: LegacyAssistPropagator,
    rust_propagator: RustASSISTPropagator,
    repeats: int,
    warmups: int,
    max_processes: int,
) -> dict[str, Any]:
    chunk_size = base._effective_chunk_size(workload, max_processes)
    kwargs = {
        "covariance": True,
        "covariance_method": workload.covariance_method,
        "num_samples": workload.num_samples,
        "seed": workload.seed,
        "max_processes": max_processes,
        "chunk_size": chunk_size,
    }

    def run_rust() -> Orbits:
        return rust_propagator.propagate_orbits(
            workload.orbits, workload.times, **kwargs
        )

    python_timings = legacy_propagator.time_propagate_orbits(
        workload.orbits,
        workload.times,
        repeats=repeats,
        warmups=warmups,
        **kwargs,
    )
    python_output = legacy_propagator.propagate_orbits(
        workload.orbits, workload.times, **kwargs
    )
    rust_timings, rust_output = time_rust(run_rust, repeats=repeats, warmups=warmups)
    native_operation, native_timings = time_native_rust(
        rust_propagator, repeats=repeats, warmups=warmups
    )
    input_rows = len(workload.orbits)
    target_rows = len(workload.times)
    return {
        "lane": workload.lane,
        "name": workload.name,
        "description": workload.description,
        "covariance": {
            "method": workload.covariance_method,
            "num_samples": workload.num_samples,
            "seed": workload.seed,
            "parity_expected": workload.parity_expected,
        },
        "input": base._orbit_metadata(workload.orbits),
        "target_times": {
            "rows": target_rows,
            "unique_rows": len(workload.times.unique()),
            "scale": workload.times.scale,
        },
        "workload_shape": {
            "n_orbits": input_rows,
            "n_target_times": target_rows,
            "output_rows": input_rows * target_rows,
        },
        "options": {
            "covariance": True,
            "chunk_size": chunk_size,
            "chunk_size_ceiling": workload.chunk_size,
            "max_processes": max_processes,
        },
        "timing_seconds": performance_timing_payload(
            python_timings,
            rust_timings,
            native_timings,
            native_operation=native_operation,
        ),
        "state_residuals": base._state_residuals(rust_output, python_output),
        "covariance_residuals": _covariance_residuals(rust_output, python_output),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--max-processes",
        type=int,
        default=mp.cpu_count(),
        help=(
            "Parallel public-call workers. Python adam-assist uses Ray worker "
            "processes; adam_assist uses the same value as its Rust Rayon "
            "thread_limit."
        ),
    )
    parser.add_argument(
        "--lanes",
        nargs="+",
        choices=(*BENCHMARK_LANES, "all"),
        default=["all"],
        help="Benchmark size lanes to run. Default: all lanes.",
    )
    parser.add_argument(
        "--skip-kernel-sha256",
        action="store_true",
        help="Skip kernel SHA256 hashing for faster local diagnostics.",
    )
    return parser


def _selected_lanes(lanes: list[str]) -> set[str]:
    if "all" in lanes:
        return set(BENCHMARK_LANES)
    return set(lanes)


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.repeats < 3:
        raise ValueError("--repeats must be at least 3 to report p50/p95")
    if args.max_processes < 1:
        raise ValueError("--max-processes must be at least 1")
    selected_lanes = _selected_lanes(args.lanes)
    workloads = [
        workload for workload in _workloads() if workload.lane in selected_lanes
    ]

    if not LEGACY_ASSIST_VENV_PYTHON.exists():
        print(
            "legacy adam_assist runtime (.legacy-assist-venv) not built; "
            "see migration/parity/README"
        )
        return 1
    legacy_propagator = LegacyAssistPropagator()
    rust_propagator = RustASSISTPropagator()

    results: list[dict[str, Any]] = []
    for workload in workloads:
        results.append(
            _benchmark_workload(
                workload,
                legacy_propagator=legacy_propagator,
                rust_propagator=rust_propagator,
                repeats=args.repeats,
                warmups=args.warmups,
                max_processes=args.max_processes,
            )
        )

    include_sha256 = not args.skip_kernel_sha256
    artifact = {
        "schema_version": 3,
        "benchmark_id": "assist_public_semantics_covariance_benchmark_2026-06-20",
        "comparison_mode": TWO_RUNTIME_COMPARISON_MODE,
        "performance_columns": PERFORMANCE_COLUMNS,
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "packages": {name: base._package_version(name) for name in base.PACKAGE_NAMES},
        "kernels": [
            base._kernel_metadata(
                de440, label="naif_de440", include_sha256=include_sha256
            ),
            base._kernel_metadata(
                de441_n16,
                label="jpl_small_bodies_de441_n16",
                include_sha256=include_sha256,
            ),
        ],
        "environment": {
            "python": base.platform.python_version(),
            "platform": base.platform.platform(),
            "machine": base.platform.machine(),
            "processor": base.platform.processor(),
            "cpu_count": mp.cpu_count(),
            "max_processes": args.max_processes,
            "thread_mode": (
                "two-runtime public calls: legacy Python adam-assist is timed "
                "inside .legacy-assist-venv (Ray worker processes) and local "
                "adam_assist uses the same public max_processes value as "
                "its Rust Rayon thread_limit"
            ),
            "covariance_methods": (
                "Public sampled covariance: VariantOrbits.create -> variant "
                "propagation -> collapse-to-nominal covariance. Legacy parity "
                "outputs are cached separately from timing samples; legacy timing "
                "loops run inside .legacy-assist-venv and Rust timing runs locally."
            ),
            "stochastic_policy": (
                "sigma-point and auto are deterministic and compared element-wise "
                "(parity_expected=true); monte-carlo uses different RNGs in Python "
                "(NumPy) and Rust (SplitMix64) so it is compared statistically "
                "(parity_expected=false). The nominal mean state is a strict "
                "parity check for every method."
            ),
        },
        "workloads": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")

    print(
        "\n| lane | workload | method | parity | p50 speedup | p95 speedup | "
        "state pos (m) | cov max rel | cov max sigma rel |"
    )
    print("|---|---|---|---|---:|---:|---:|---:|---:|")
    for row in results:
        timing = row["timing_seconds"]
        cov = row["covariance"]
        state = row["state_residuals"]
        cov_res = row["covariance_residuals"]
        print(
            f"| {row['lane']} | {row['name']} | {cov['method']} | "
            f"{cov['parity_expected']} | "
            f"{timing['speedup']['p50_python_over_rust']:.3f} | "
            f"{timing['speedup']['p95_python_over_rust']:.3f} | "
            f"{state['position_abs_m']:.6e} | "
            f"{cov_res['max_rel_frobenius']:.6e} | "
            f"{cov_res['max_sigma_rel']:.6e} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
