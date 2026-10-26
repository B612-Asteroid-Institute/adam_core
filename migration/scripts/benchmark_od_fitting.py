"""Benchmark Core-owned native and public orbit-fitting workflows.

This is a current-runtime latency diagnostic, not a Rust-vs-legacy gate.  It
measures two deliberately separate boundaries over identical deterministic
workloads:

* native Rust work timed with ``std::time::Instant`` after Arrow decoding and
  settings parsing (``refine(..., rejection='none')``'s whitened fit and the
  complete model-free ``run_od`` IOD + refinement workflow); and
* public Python end-to-end wall clock for ``refine`` and ``run_od`` including
  serialization, provenance checks and table assembly.

Workload construction, imports and correctness preflights are never timed.
Machine-specific output defaults to a gitignored local artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse

import numpy as np

import adam_core
from adam_core import _rust_native
from adam_core.coordinates import CartesianCoordinates, CoordinateCovariances
from adam_core.coordinates.origin import Origin
from adam_core.coordinates.spherical import SphericalCoordinates
from adam_core.dynamics.ephemeris import generate_ephemeris_2body
from adam_core.dynamics.propagation import propagate_2body
from adam_core.observers import Observers
from adam_core.orbit_determination._native_dispatch import native_propagator
from adam_core.orbit_determination.evaluate import (
    OrbitDeterminationObservations,
    OrbitDeterminationPhotometry,
)
from adam_core.orbit_determination.native_orbit_fitter import NativeOrbitFitter
from adam_core.orbit_determination.od_orchestration import refine, run_od
from adam_core.orbits.ephemeris import Ephemeris
from adam_core.orbits.orbits import Orbits
from adam_core.propagator import NativeBackend, Propagator
from adam_core.time import Timestamp

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "migration" / "artifacts" / "od_fitting_benchmark_local.json"
EPOCH_MJD_TDB = 61_000.0
TRUTH_STATE = np.array([0.9, 0.5, 0.05, -0.008, 0.012, 0.0005])
SIGMA_ARCSEC = 0.5
WORKLOAD_SPECS = (
    ("small_8obs_21d", 8, 21.0, 7001),
    ("larger_32obs_90d", 32, 90.0, 7002),
)


class BenchmarkTwoBodyPropagator(Propagator):
    """Propagation-only adapter over Core's native two-body provider."""

    two_body_dynamics = True
    native_backend = NativeBackend(
        name="adam_core.two_body.external",
        version=_rust_native.two_body_propagator_semver(),
        work_units=frozenset({"propagator_capsule"}),
    )

    def propagator_capsule(self):
        return _rust_native.two_body_propagator_capsule()

    def propagate_orbits(self, orbits: Orbits, times: Timestamp, **kwargs) -> Orbits:
        return propagate_2body(orbits, times)

    def generate_ephemeris(
        self, orbits: Orbits, observers: Observers, **kwargs
    ) -> Ephemeris:
        orbits = orbits.sort_by(["orbit_id"]).set_column(
            "coordinates.covariance", CoordinateCovariances.nulls(len(orbits))
        )
        rows = len(observers)
        propagated = propagate_2body(orbits, observers.coordinates.time)
        blocks = [
            generate_ephemeris_2body(
                propagated.take(list(range(i * rows, (i + 1) * rows))), observers
            )
            for i in range(len(orbits))
        ]
        if len(blocks) == 1:
            return blocks[0]
        import quivr as qv

        return qv.concatenate(blocks)


@dataclass(frozen=True)
class Workload:
    name: str
    observations: OrbitDeterminationObservations
    seed: Orbits
    arc_days: float
    rng_seed: int


@dataclass(frozen=True)
class Operation:
    name: str
    native_name: str
    public_call: Callable[[], tuple[Any, Any]]
    iod_settings: dict[str, Any] | None
    refinement: dict[str, Any] | None
    fit_settings: dict[str, Any]


def _truth_orbit() -> Orbits:
    return Orbits.from_kwargs(
        orbit_id=["truth"],
        object_id=["synthetic"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=TRUTH_STATE[0:1],
            y=TRUTH_STATE[1:2],
            z=TRUTH_STATE[2:3],
            vx=TRUTH_STATE[3:4],
            vy=TRUTH_STATE[4:5],
            vz=TRUTH_STATE[5:6],
            time=Timestamp.from_mjd([EPOCH_MJD_TDB], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )


def _seed_orbit() -> Orbits:
    state = TRUTH_STATE + np.array([1e-5, -8e-6, 5e-6, 1e-8, -1e-8, 5e-9])
    return Orbits.from_kwargs(
        orbit_id=["seed"],
        object_id=["synthetic"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=state[0:1],
            y=state[1:2],
            z=state[2:3],
            vx=state[3:4],
            vy=state[4:5],
            vz=state[5:6],
            time=Timestamp.from_mjd([EPOCH_MJD_TDB], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )


def _workload(name: str, n_obs: int, arc_days: float, seed: int) -> Workload:
    provider = BenchmarkTwoBodyPropagator()
    times = Timestamp.from_mjd(
        EPOCH_MJD_TDB + 5.0 + np.linspace(0.0, arc_days, n_obs), scale="tdb"
    )
    observers = Observers.from_code("500", times)
    ephemeris = provider.generate_ephemeris(_truth_orbit(), observers, max_processes=1)
    lon = ephemeris.coordinates.lon.to_numpy(zero_copy_only=False)
    lat = ephemeris.coordinates.lat.to_numpy(zero_copy_only=False)
    cos_lat = np.cos(np.radians(lat))
    sigma_deg = SIGMA_ARCSEC / 3600.0
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma_deg, size=(n_obs, 2))
    covariance = np.full((n_obs, 6, 6), np.nan)
    covariance[:, 1, 1] = sigma_deg**2 / cos_lat**2
    covariance[:, 2, 2] = sigma_deg**2
    observations = OrbitDeterminationObservations.from_kwargs(
        id=[f"{name}-{row:04d}" for row in range(n_obs)],
        coordinates=SphericalCoordinates.from_kwargs(
            lon=lon + noise[:, 0] / cos_lat,
            lat=lat + noise[:, 1],
            time=times,
            covariance=CoordinateCovariances.from_matrix(covariance),
            origin=Origin.from_kwargs(code=["500"] * n_obs),
            frame="equatorial",
        ),
        observers=observers,
        photometry=OrbitDeterminationPhotometry.from_kwargs(
            mag=[None] * n_obs,
            rmsmag=[None] * n_obs,
            band=[None] * n_obs,
        ),
    )
    return Workload(name, observations, _seed_orbit(), arc_days, seed)


def _state(fitted: Any) -> np.ndarray:
    if len(fitted) != 1 or not fitted.success[0].as_py():
        raise RuntimeError("correctness preflight did not produce one successful orbit")
    return np.asarray(fitted.coordinates.values[0], dtype=np.float64)


def _sample_public(
    call: Callable[[], tuple[Any, Any]], reps: int, trials: int, warmup_reps: int
) -> list[list[float]]:
    samples: list[list[float]] = []
    for _ in range(trials):
        for _ in range(warmup_reps):
            _state(call()[0])
        trial: list[float] = []
        for _ in range(reps):
            started = time.perf_counter()
            result = call()
            trial.append(time.perf_counter() - started)
            _state(result[0])
            # Keep result-table destruction outside the timed public boundary.
            del result
        samples.append(trial)
    return samples


def _summary(samples: list[list[float]]) -> dict[str, Any]:
    array = np.asarray(samples, dtype=np.float64)
    trial_p50 = np.median(array, axis=1)
    trial_p95 = np.percentile(array, 95, axis=1)
    return {
        "p50_seconds": float(np.median(trial_p50)),
        "p95_seconds": float(np.median(trial_p95)),
        "sample_trials_seconds": samples,
        "trial_p50_seconds": trial_p50.tolist(),
        "trial_p95_seconds": trial_p95.tolist(),
    }


def _sha256_parts(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _native_preflight(
    operation: Operation,
    propagator_descriptor: Any,
    orbit_ipc: bytes,
    observed_ipc: bytes,
    observers_ipc: bytes,
    obs_code: list[str],
    band: list[str | None],
    astcat: list[str | None],
) -> np.ndarray:
    if operation.native_name == "refine_none":
        output = _rust_native.fit_orbit_whitened_ipc(
            propagator_descriptor,
            orbit_ipc,
            observed_ipc,
            observers_ipc,
            [False] * len(obs_code),
            fit_settings=operation.fit_settings,
        )
    elif operation.native_name == "run_od":
        output = _rust_native.run_od_ipc(
            propagator_descriptor,
            observed_ipc,
            observers_ipc,
            obs_code,
            band,
            astcat,
            models=[],
            iod_settings=operation.iod_settings,
            refinement=operation.refinement,
            fit_settings=operation.fit_settings,
            iod_propagator=propagator_descriptor,
        )
    else:
        raise AssertionError(
            f"unsupported benchmark operation {operation.native_name!r}"
        )
    fit_output = output["fit"] if operation.native_name == "run_od" else output
    state = np.asarray(fit_output["state"], dtype=np.float64)
    if state.shape != (6,) or not np.isfinite(state).all():
        raise RuntimeError("native correctness preflight produced an invalid state")
    return state


def _benchmark_workload(
    workload: Workload, *, reps: int, trials: int, warmup_reps: int
) -> dict[str, Any]:
    from adam_core.orbit_determination.differential_correction import _native_problem

    provider = BenchmarkTwoBodyPropagator()
    fitter = NativeOrbitFitter(
        provider,
        iod_rchi2_threshold=1e6,
        contamination_percentage=0.0,
        outlier_rejection="worst_residual",
    )
    iod_settings, refinement_settings, fit_settings = fitter.native_settings()
    orbit_ipc, observed_ipc, observers_ipc = _native_problem(
        workload.seed, workload.observations
    )
    propagator_descriptor = native_propagator(provider)
    obs_code = workload.observations.observers.code.to_pylist()
    band = workload.observations.photometry.band.to_pylist()
    astcat = workload.observations.astcat.to_pylist()

    operations = [
        Operation(
            "refine_none",
            "refine_none",
            lambda: refine(
                workload.seed,
                workload.observations,
                provider,
                rejection="none",
                fit_settings=fit_settings,
            ),
            None,
            None,
            fit_settings,
        ),
        Operation(
            "run_od_model_free_worst_residual",
            "run_od",
            lambda: run_od(
                workload.observations,
                fitter,
                models=None,
                propagator=provider,
                object_id="benchmark",
            ),
            iod_settings,
            refinement_settings,
            fit_settings,
        ),
    ]

    rows = []
    for operation in operations:
        public_state = _state(operation.public_call()[0])
        native_state = _native_preflight(
            operation,
            propagator_descriptor,
            orbit_ipc,
            observed_ipc,
            observers_ipc,
            obs_code,
            band,
            astcat,
        )
        np.testing.assert_allclose(public_state, native_state, rtol=0.0, atol=1e-13)
        native = dict(
            _rust_native.benchmark_od_fit_ipc(
                propagator_descriptor,
                operation.native_name,
                orbit_ipc,
                observed_ipc,
                observers_ipc,
                obs_code,
                band,
                astcat,
                reps,
                trials,
                warmup_reps,
                iod_settings=operation.iod_settings,
                refinement=operation.refinement,
                fit_settings=operation.fit_settings,
            )
        )
        public_samples = _sample_public(
            operation.public_call, reps, trials, warmup_reps
        )
        timed_native_state = np.asarray(native.pop("state"), dtype=np.float64)
        np.testing.assert_allclose(
            native_state, timed_native_state, rtol=0.0, atol=1e-15
        )
        native_samples = native.pop("samples_seconds")
        native_timing = _summary(native_samples)
        public_timing = _summary(public_samples)
        rows.append(
            {
                "operation": operation.name,
                "configuration": {
                    "iod_settings": operation.iod_settings,
                    "refinement": operation.refinement,
                    "fit_settings": operation.fit_settings,
                    "observation_models": [],
                },
                "native_rust": {
                    **native,
                    **native_timing,
                    "timing_boundary": "Rust OD algorithm only",
                },
                "public_python": {
                    **public_timing,
                    "timer": "time.perf_counter",
                    "timing_boundary": "public API end-to-end",
                },
                "public_over_native": {
                    "p50": public_timing["p50_seconds"] / native_timing["p50_seconds"],
                    "p95": public_timing["p95_seconds"] / native_timing["p95_seconds"],
                    "interpretation": (
                        "diagnostic includes Python serialization/provenance/result assembly; "
                        "both paths use the same propagation capsule; not a speed gate"
                    ),
                },
                "correctness": {
                    "state_max_abs_difference": float(
                        np.max(np.abs(public_state - native_state))
                    ),
                    "state_tolerance": 1e-13,
                    "checked_before_timing": True,
                },
            }
        )
    return {
        "name": workload.name,
        "shape": {
            "observations": len(workload.observations),
            "arc_days": workload.arc_days,
            "state_parameters": 6,
            "station_codes": sorted(set(obs_code)),
        },
        "input_identity": {
            "combined_ipc_sha256": _sha256_parts(
                orbit_ipc, observed_ipc, observers_ipc
            ),
            "orbit_ipc_sha256": hashlib.sha256(orbit_ipc).hexdigest(),
            "observed_ipc_sha256": hashlib.sha256(observed_ipc).hexdigest(),
            "observers_ipc_sha256": hashlib.sha256(observers_ipc).hexdigest(),
            "rng_seed": workload.rng_seed,
            "truth_state_au_au_per_day": TRUTH_STATE.tolist(),
            "truth_epoch_mjd_tdb": EPOCH_MJD_TDB,
            "sigma_arcsec": SIGMA_ARCSEC,
        },
        "rows": rows,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _containing_git_checkout(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _distribution_file(distribution: metadata.Distribution, suffix: str) -> Path | None:
    matches = [
        Path(str(distribution.locate_file(item))).resolve()
        for item in distribution.files or ()
        if str(item).endswith(suffix)
    ]
    if len(matches) > 1:
        raise RuntimeError(f"installed adam-core has multiple {suffix!r} files")
    return matches[0] if matches else None


def _native_build_profile_identity() -> dict[str, Any]:
    profile, opt_level, target, debug_assertions = _rust_native.native_build_profile()
    if (
        profile != "release"
        or debug_assertions
        or opt_level not in {"2", "3", "s", "z"}
    ):
        raise RuntimeError(
            "OD fitting benchmarks require an optimized release native artifact; "
            f"got profile={profile!r}, opt_level={opt_level!r}, "
            f"debug_assertions={debug_assertions!r}"
        )
    return {
        "cargo_profile": profile,
        "cargo_opt_level": opt_level,
        "cargo_target": target,
        "debug_assertions": debug_assertions,
    }


def _installed_artifact_identity() -> dict[str, Any]:
    package_file = Path(adam_core.__file__).resolve()
    native_file = Path(_rust_native.__file__).resolve()
    for label, path in (
        ("Python package", package_file),
        ("native extension", native_file),
    ):
        checkout = _containing_git_checkout(path)
        if checkout is not None:
            raise RuntimeError(
                f"{label} executes from Git checkout {checkout}; install the release wheel "
                "into an environment outside every checkout before benchmarking"
            )

    distribution = metadata.distribution("adam-core")
    record_path = _distribution_file(distribution, ".dist-info/RECORD")
    wheel_metadata_path = _distribution_file(distribution, ".dist-info/WHEEL")
    direct_url_path = _distribution_file(distribution, ".dist-info/direct_url.json")
    release_inventory = package_file.parent / "_release_inventory.json"
    if (
        record_path is None
        or wheel_metadata_path is None
        or not release_inventory.is_file()
    ):
        raise RuntimeError("installed adam-core wheel identity files are incomplete")

    direct_url: dict[str, Any] | None = None
    wheel_artifact: dict[str, Any] | None = None
    if direct_url_path is not None:
        direct_url = json.loads(direct_url_path.read_text())
        if direct_url.get("dir_info", {}).get("editable") is True:
            raise RuntimeError("benchmark refuses an editable adam-core installation")
        parsed = urlparse(str(direct_url.get("url", "")))
        if parsed.scheme == "file" and parsed.path.endswith(".whl"):
            artifact = Path(unquote(parsed.path)).resolve()
            if artifact.is_file():
                wheel_artifact = {
                    "path": str(artifact),
                    "size_bytes": artifact.stat().st_size,
                    "sha256": _sha256_file(artifact),
                }
                declared = str(direct_url.get("archive_info", {}).get("hash", ""))
                if (
                    declared.startswith("sha256=")
                    and declared[7:] != wheel_artifact["sha256"]
                ):
                    raise RuntimeError(
                        "installed wheel direct_url hash disagrees with artifact"
                    )

    return {
        "installation_kind": "wheel",
        "python_package_file": str(package_file),
        "native_extension_file": str(native_file),
        "outside_git_checkout": True,
        "native_build_profile": _native_build_profile_identity(),
        "installed_record": {
            "path": str(record_path),
            "sha256": _sha256_file(record_path),
        },
        "wheel_metadata": {
            "path": str(wheel_metadata_path),
            "sha256": _sha256_file(wheel_metadata_path),
        },
        "release_inventory_sha256": _sha256_file(release_inventory),
        "direct_url": direct_url,
        "wheel_artifact": wheel_artifact,
    }


def _kernel_identity(path_string: str) -> dict[str, Any]:
    path = Path(path_string).resolve()
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": _sha256_file(path) if path.is_file() else None,
    }


def _runtime_state() -> dict[str, Any]:
    from adam_core.utils.spice_backend import get_backend

    backend = get_backend()
    kernels = [_kernel_identity(path) for path in backend.registered_kernels()]
    thread_environment = {
        key: os.environ.get(key)
        for key in (
            "RAYON_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    }
    return {
        "threads": {
            "logical_cpu_count": os.cpu_count(),
            "environment": thread_environment,
            "od_algorithm": "serial Core-owned candidate/optimizer loops",
            "provider_policy": "provider-controlled; two-body ABI capsule used here",
        },
        "spice": {
            "backend": type(backend).__qualname__,
            "registered_kernels": kernels,
            "registered_kernel_count": len(kernels),
            "spkez_state_cache_entries": backend.spkez_state_cache_len(),
        },
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("adam-core", "numpy", "pyarrow", "quivr", "spiceypy"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _validated_identity() -> dict[str, Any]:
    build_version, build_commit = _rust_native.build_identity()
    distribution_version = metadata.version("adam-core")
    if not (adam_core.__version__ == distribution_version == build_version):
        raise RuntimeError("Python, distribution and native build versions disagree")
    if re.fullmatch(r"[0-9a-f]{40}", build_commit) is None:
        raise RuntimeError("native build commit is not an exact Git identity")
    provider = dict(
        _rust_native.native_provider_metadata(
            native_propagator(BenchmarkTwoBodyPropagator()), {}
        )
    )
    for field in ("configuration_fingerprint", "data_fingerprint"):
        if re.fullmatch(r"[0-9a-f]{64}", str(provider.get(field, ""))) is None:
            raise RuntimeError(f"native provider has no canonical {field}")
    return {
        "adam_core_python_version": adam_core.__version__,
        "adam_core_distribution_version": distribution_version,
        "native_build_version": build_version,
        "native_build_commit": build_commit,
        "installed_artifact": _installed_artifact_identity(),
        "benchmark_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "package_versions": _package_versions(),
        "provider": provider,
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup-reps", type=int, default=1)
    parser.add_argument(
        "--quick", action="store_true", help="one sample/trial for smoke validation"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    reps = 1 if args.quick else args.reps
    trials = 1 if args.quick else args.trials
    warmup_reps = 0 if args.quick else args.warmup_reps
    if reps < 1 or trials < 1 or warmup_reps < 0:
        raise ValueError("reps/trials must be >= 1 and warmup-reps must be >= 0")

    identity = _validated_identity()
    runtime_before_setup = _runtime_state()
    workloads = [_workload(*spec) for spec in WORKLOAD_SPECS]
    runtime_after_setup = _runtime_state()
    benchmarked_workloads = [
        _benchmark_workload(workload, reps=reps, trials=trials, warmup_reps=warmup_reps)
        for workload in workloads
    ]
    runtime_after_timing = _runtime_state()
    payload = {
        "schema_version": 2,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "current Core OD latency diagnostic; non-gating",
        "timing_policy": {
            "setup_timed": False,
            "correctness_preflight_timed": False,
            "native_timer": "std::time::Instant",
            "public_timer": "time.perf_counter",
            "reps": reps,
            "trials": trials,
            "warmup_reps": warmup_reps,
            "lane_order": [
                "public correctness preflight",
                "native correctness preflight",
                "native Rust samples",
                "public Python samples",
            ],
            "public_result_destruction_timed": False,
            "native_result_destruction_timed": False,
        },
        "identity": identity,
        "runtime_state": {
            "before_setup": runtime_before_setup,
            "after_setup": runtime_after_setup,
            "after_timing": runtime_after_timing,
            "cache_policy": (
                "no explicit clearing; correctness preflights run before timing and each "
                "trial has the recorded warmup count; native timing precedes public timing"
            ),
        },
        "propagation_call_diagnostics": {
            "status": "unavailable",
            "reason": "the in-process two-body Propagator does not expose a call counter",
        },
        "workloads": benchmarked_workloads,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
