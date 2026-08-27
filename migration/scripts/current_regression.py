"""Run the complete deterministic, current-only correctness regression grid.

The fixture preserves every reviewed eight-seed parity workload plus the five
special fixed fixtures. Normal checks execute only the current public/thin
Python facade over Rust. Exact contractual outputs are pinned by SHA-256;
numerical science outputs use their reviewed tolerances, while shape, dtype,
ordering, and non-finite masks remain exact contracts.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from adam_core._rust.status import API_MIGRATIONS
from migration.parity import _inputs, _rust_runner, tolerances
from migration.parity.parity_fuzz import _check_output, _spice_kernel_provenance

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = ROOT / "migration" / "artifacts" / "current_regression_core.json.gz"
DEFAULT_BASE_SEED = 20260425
DEFAULT_SEED_COUNT = 8
DEFAULT_N = 128
EXPECTED_FIXED_FIXTURES = (
    "stiff_high_a_covariance_finite_difference",
    "distant_stellar_aberration_covariance_finite_difference",
    "identical_circular_flat_minimum",
    "well_conditioned_unique_minimum",
    "well_conditioned_seed_20260425",
)
CONTRACTUALLY_EXACT_OUTPUTS = {
    ("orbits.VariantOrbits.create", "weights"),
    ("orbits.VariantOrbits.create", "weights_cov"),
}


@dataclass(frozen=True)
class CurrentRegressionResult:
    api_count: int
    case_count: int
    fuzz_case_count: int
    fixed_fixture_count: int
    output_count: int
    exact_output_count: int


def _output_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _encode_input(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            return {
                "__object_array__": array.tolist(),
                "shape": list(array.shape),
            }
        return {
            "__ndarray__": base64.b64encode(array.tobytes()).decode("ascii"),
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }
    if isinstance(value, dict):
        return {key: _encode_input(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode_input(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported regression input type: {type(value).__name__}")


def _decode_input(value: Any) -> Any:
    if isinstance(value, dict) and "__object_array__" in value:
        return np.asarray(value["__object_array__"], dtype="object").reshape(
            value["shape"]
        )
    if isinstance(value, dict) and "__ndarray__" in value:
        array = np.frombuffer(
            base64.b64decode(value["__ndarray__"]), dtype=np.dtype(value["dtype"])
        )
        return np.ascontiguousarray(array.reshape(value["shape"]))
    if isinstance(value, dict):
        return {key: _decode_input(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_input(item) for item in value]
    return value


def _selected_api_ids(requested: list[str] | None) -> list[str]:
    all_ids = [migration.api_id for migration in API_MIGRATIONS]
    if requested is None:
        return all_ids
    unknown = sorted(set(requested) - set(all_ids))
    if unknown:
        raise ValueError(f"unknown API ids: {unknown}")
    return [api_id for api_id in all_ids if api_id in requested]


def _is_exact(api_id: str, name: str, tolerance: tolerances.OutputTol) -> bool:
    return (api_id, name) in CONTRACTUALLY_EXACT_OUTPUTS or (
        tolerance.atol == 0.0 and tolerance.rtol == 0.0
    )


def _capture_output(
    api_id: str,
    name: str,
    value: Any,
    tolerance: tolerances.OutputTol,
) -> dict[str, Any]:
    raw_value = np.asarray(value)
    numeric_value = np.asarray(raw_value, dtype=np.float64)
    exact = _is_exact(api_id, name, tolerance)
    return {
        "atol": tolerance.atol,
        "rtol": tolerance.rtol,
        "shape": list(raw_value.shape),
        "dtype": raw_value.dtype.str,
        "nan_mask_sha256": hashlib.sha256(
            np.ascontiguousarray(~np.isfinite(numeric_value)).tobytes()
        ).hexdigest(),
        "sha256": _output_digest(raw_value) if exact else None,
        "values": numeric_value.tolist(),
    }


def _capture_case(
    api_id: str,
    *,
    case_id: str,
    kind: str,
    run_kwargs: Mapping[str, Any],
    output_tolerances: Mapping[str, tolerances.OutputTol],
    expected_outputs: Mapping[str, Any] | None = None,
    n: int,
    seed: int | None = None,
    description: str = "",
    authority: str = "accepted current candidate after reviewed parity",
) -> dict[str, Any]:
    kwargs = dict(run_kwargs)
    if api_id == "coordinates.transform_coordinates":
        kwargs.pop("spice_kernels", None)
    outputs = (
        _rust_runner.run(api_id, **kwargs)
        if expected_outputs is None
        else dict(expected_outputs)
    )
    missing = sorted(set(output_tolerances) - set(outputs))
    if missing:
        raise AssertionError(f"{api_id}.{case_id}: missing outputs {missing}")
    return {
        "case_id": case_id,
        "kind": kind,
        "seed": seed,
        "n": n,
        "description": description,
        "authority": authority,
        "inputs": _encode_input(kwargs),
        "outputs": {
            name: _capture_output(api_id, name, outputs[name], tolerance)
            for name, tolerance in output_tolerances.items()
        },
    }


def _capture_fuzz_case(api_id: str, *, seed: int, n: int) -> dict[str, Any]:
    workload_n = _inputs.fuzz_n(api_id, n)
    sample = _inputs.make(api_id, np.random.default_rng(seed), workload_n)
    return _capture_case(
        api_id,
        case_id=f"fuzz-seed-{seed}",
        kind="fuzz",
        run_kwargs=sample.rust_kwargs,
        output_tolerances=tolerances.get(api_id).outputs,
        n=workload_n if isinstance(workload_n, int) else workload_n.rows,
        seed=seed,
    )


def capture_fixture(
    *,
    base_seed: int,
    seeds: int,
    n: int,
    api_ids: list[str],
) -> dict[str, Any]:
    seed_values = list(range(base_seed, base_seed + seeds))
    return {
        "schema_version": 2,
        "fixture_authority": (
            "accepted current candidate after complete reviewed baseline parity; "
            "special fixtures retain their reviewed reference authority"
        ),
        "normal_test_runtime": "current public/thin Python facade over Rust",
        "normal_test_legacy_runtime_used": False,
        "spice_kernel_provenance": _spice_kernel_provenance(),
        "base_seed": base_seed,
        "seed_count": seeds,
        "default_n": n,
        "fixed_fixture_names": [],
        "apis": {
            api_id: [_capture_fuzz_case(api_id, seed=seed, n=n) for seed in seed_values]
            for api_id in api_ids
        },
    }


def _case_tolerances(
    api_id: str, case: Mapping[str, Any]
) -> dict[str, tolerances.OutputTol]:
    stored = {
        name: tolerances.OutputTol(
            atol=float(output["atol"]), rtol=float(output["rtol"])
        )
        for name, output in case["outputs"].items()
    }
    if case["kind"] != "fuzz":
        return stored

    canonical = tolerances.get(api_id).outputs
    if set(stored) != set(canonical):
        raise AssertionError(f"{api_id}.{case['case_id']}: output policy changed")
    for name, tolerance in canonical.items():
        if stored[name] != tolerance:
            raise AssertionError(
                f"{api_id}.{case['case_id']}.{name}: tolerance policy changed"
            )
    return stored


def _check_case(api_id: str, case: Mapping[str, Any]) -> tuple[int, int]:
    actual_outputs = _rust_runner.run(api_id, **_decode_input(case["inputs"]))
    output_tolerances = _case_tolerances(api_id, case)
    output_count = 0
    exact_output_count = 0
    for name, tolerance in output_tolerances.items():
        output_count += 1
        if name not in actual_outputs:
            raise AssertionError(f"{api_id}: missing current output {name!r}")
        expected_output = case["outputs"][name]
        if "values" not in expected_output:
            raise AssertionError(
                "current regression fixture must contain frozen inputs and outputs"
            )
        raw_actual = np.asarray(actual_outputs[name])
        actual = np.asarray(raw_actual, dtype=np.float64)
        expected_values = np.asarray(expected_output["values"], dtype=np.float64)
        label = f"{api_id}.{case['case_id']}.{name}"
        if list(actual.shape) != expected_output["shape"]:
            raise AssertionError(
                f"{label}: shape {actual.shape} != {expected_output['shape']}"
            )
        if raw_actual.dtype.str != expected_output["dtype"]:
            raise AssertionError(f"{label}: output dtype changed")
        actual_nan_sha = hashlib.sha256(
            np.ascontiguousarray(~np.isfinite(actual)).tobytes()
        ).hexdigest()
        if actual_nan_sha != expected_output["nan_mask_sha256"]:
            raise AssertionError(f"{label}: non-finite mask changed")
        if _is_exact(api_id, name, tolerance):
            exact_output_count += 1
            actual_sha = _output_digest(raw_actual)
            if actual_sha != expected_output["sha256"]:
                raise AssertionError(
                    f"{label}: exact output digest changed "
                    f"({actual_sha} != {expected_output['sha256']})"
                )
            continue
        result = _check_output(name, actual, expected_values, tolerance)
        if not result.passed:
            raise AssertionError(
                f"{label}: output exceeded its science tolerance "
                f"(max_abs={result.max_abs:.3e}, max_rel={result.max_rel:.3e})"
            )
    return output_count, exact_output_count


def _read_fixture(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fixture_file:
            return json.load(fixture_file)
    return json.loads(path.read_text())


def _write_fixture(path: Path, fixture: Mapping[str, Any]) -> None:
    if path.suffix == ".gz":
        payload = (
            json.dumps(fixture, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        with path.open("wb") as raw_file:
            with gzip.GzipFile(
                fileobj=raw_file, mode="wb", compresslevel=9, mtime=0
            ) as fixture_file:
                fixture_file.write(payload)
        return
    path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")


def _kernel_identity(provenance: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "naif_eop_high_prec_version": provenance["naif_eop_high_prec_version"],
        "kernels": [
            {
                "name": Path(kernel["path"]).name,
                "size_bytes": kernel["size_bytes"],
                "sha256": kernel["sha256"],
            }
            for kernel in provenance["kernels"]
        ],
    }


def check_fixture(
    path: Path, *, requested: list[str] | None = None
) -> CurrentRegressionResult:
    fixture = _read_fixture(path)
    if fixture.get("schema_version") != 2:
        raise AssertionError("unsupported current regression fixture schema")
    if fixture.get("normal_test_runtime") != (
        "current public/thin Python facade over Rust"
    ):
        raise AssertionError("current regression fixture runtime changed")
    if fixture.get("normal_test_legacy_runtime_used") is not False:
        raise AssertionError("normal regression checks must not use a frozen runtime")
    if fixture.get("base_seed") != DEFAULT_BASE_SEED:
        raise AssertionError("current regression base seed changed")
    if fixture.get("seed_count") != DEFAULT_SEED_COUNT:
        raise AssertionError("current regression seed count changed")
    if fixture.get("default_n") != DEFAULT_N:
        raise AssertionError("current regression workload size changed")
    if _kernel_identity(fixture.get("spice_kernel_provenance", {})) != (
        _kernel_identity(_spice_kernel_provenance())
    ):
        raise AssertionError("current regression SPICE kernel identity changed")
    if tuple(fixture.get("fixed_fixture_names", ())) != EXPECTED_FIXED_FIXTURES:
        raise AssertionError("current regression fixed-fixture coverage changed")

    api_ids = _selected_api_ids(requested)
    expected_apis = fixture["apis"]
    missing = sorted(set(api_ids) - set(expected_apis))
    if missing:
        raise AssertionError(f"current regression fixture is missing APIs: {missing}")

    case_count = fuzz_case_count = fixed_fixture_count = 0
    output_count = exact_output_count = 0
    expected_seeds = list(
        range(DEFAULT_BASE_SEED, DEFAULT_BASE_SEED + DEFAULT_SEED_COUNT)
    )
    for api_id in api_ids:
        cases = expected_apis[api_id]
        fuzz_seeds = [case["seed"] for case in cases if case["kind"] == "fuzz"]
        if fuzz_seeds != expected_seeds:
            raise AssertionError(f"{api_id}: eight-seed fuzz coverage changed")
        for case in cases:
            outputs, exact = _check_case(api_id, case)
            case_count += 1
            fuzz_case_count += case["kind"] == "fuzz"
            fixed_fixture_count += case["kind"] == "fixed"
            output_count += outputs
            exact_output_count += exact

    return CurrentRegressionResult(
        api_count=len(api_ids),
        case_count=case_count,
        fuzz_case_count=fuzz_case_count,
        fixed_fixture_count=fixed_fixture_count,
        output_count=output_count,
        exact_output_count=exact_output_count,
    )


def _reviewed_fixed_cases(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load the immutable reviewed fixed cases before recapturing fuzz rows."""
    if not path.exists():
        raise FileNotFoundError(
            "--capture requires the reviewed fixed fixture file to exist"
        )
    existing = _read_fixture(path)
    if tuple(existing.get("fixed_fixture_names", ())) != EXPECTED_FIXED_FIXTURES:
        raise AssertionError("reviewed fixed-fixture authority is incomplete")
    return {
        api_id: [case for case in cases if case["kind"] == "fixed"]
        for api_id, cases in existing["apis"].items()
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--apis", nargs="*")
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEED_COUNT)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--capture", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    api_ids = _selected_api_ids(args.apis)
    if args.capture:
        fixed_cases = _reviewed_fixed_cases(args.fixture)
        fixture = capture_fixture(
            base_seed=args.base_seed,
            seeds=args.seeds,
            n=args.n,
            api_ids=api_ids,
        )
        fixture["fixed_fixture_names"] = list(EXPECTED_FIXED_FIXTURES)
        for api_id in api_ids:
            fixture["apis"][api_id].extend(fixed_cases.get(api_id, ()))
        args.fixture.parent.mkdir(parents=True, exist_ok=True)
        _write_fixture(args.fixture, fixture)
        print(args.fixture)
        return 0
    result = check_fixture(args.fixture, requested=args.apis)
    print(
        f"current regression passed: {result.api_count} APIs, "
        f"{result.case_count} cases ({result.fuzz_case_count} fuzz + "
        f"{result.fixed_fixture_count} fixed), {result.output_count} outputs, "
        f"{result.exact_output_count} exact outputs"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
