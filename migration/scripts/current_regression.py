"""Run deterministic current-only correctness regression checks.

This is the post-parity CI gate. It reuses the canonical API registry, input
builders, public/thin Python runners, and per-output correctness policy without
starting the frozen Python oracle. Exact outputs are pinned by SHA-256; numeric
science outputs are checked for deterministic repeatability using their accepted
parity tolerance, and output schemas/NaN masks are always checked exactly.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from adam_core._rust.status import API_MIGRATIONS
from migration.parity import _inputs, _rust_runner, tolerances
from migration.parity.parity_fuzz import _check_output

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = ROOT / "migration" / "artifacts" / "current_regression_core.json"
DEFAULT_SEED = 20260429
DEFAULT_N = 16
CONTRACTUALLY_EXACT_OUTPUTS = {
    ("orbits.VariantOrbits.create", "weights"),
    ("orbits.VariantOrbits.create", "weights_cov"),
}


@dataclass(frozen=True)
class CurrentRegressionResult:
    api_count: int
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


def _capture_api(api_id: str, *, seed: int, n: int) -> dict[str, Any]:
    workload_n = _inputs.fuzz_n(api_id, n)
    sample = _inputs.make(api_id, np.random.default_rng(seed), workload_n)
    run_kwargs = dict(sample.rust_kwargs)
    if api_id == "coordinates.transform_coordinates":
        run_kwargs.pop("spice_kernels", None)
    outputs = _rust_runner.run(api_id, **run_kwargs)
    spec = tolerances.get(api_id)
    captured_outputs: dict[str, Any] = {}
    for name, tolerance in spec.outputs.items():
        raw_value = np.asarray(outputs[name])
        value = np.asarray(raw_value, dtype=np.float64)
        exact = (api_id, name) in CONTRACTUALLY_EXACT_OUTPUTS or (
            tolerance.atol == 0.0 and tolerance.rtol == 0.0
        )
        captured_outputs[name] = {
            "shape": list(raw_value.shape),
            "dtype": raw_value.dtype.str,
            "nan_mask_sha256": hashlib.sha256(
                np.ascontiguousarray(~np.isfinite(value)).tobytes()
            ).hexdigest(),
            "sha256": _output_digest(raw_value) if exact else None,
            "values": value.tolist(),
        }
    return {
        "n": workload_n,
        "inputs": _encode_input(run_kwargs),
        "outputs": captured_outputs,
    }


def capture_fixture(*, seed: int, n: int, api_ids: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark_source": "current public/thin Python facade over Rust",
        "legacy_runtime_used": False,
        "seed": seed,
        "default_n": n,
        "apis": {
            api_id: _capture_api(api_id, seed=seed, n=n) for api_id in api_ids
        },
    }


def _check_api(api_id: str, expected: dict[str, Any]) -> tuple[int, int]:
    actual_outputs = _rust_runner.run(api_id, **_decode_input(expected["inputs"]))
    spec = tolerances.get(api_id)
    output_count = 0
    exact_output_count = 0
    for name, tolerance in spec.outputs.items():
        output_count += 1
        if name not in actual_outputs:
            raise AssertionError(f"{api_id}: missing current output {name!r}")
        expected_output = expected["outputs"][name]
        if "inputs" not in expected or "values" not in expected_output:
            raise AssertionError(
                "current regression fixture must contain frozen inputs and outputs; "
                "recapture it explicitly with --capture after reviewing changes"
            )
        actual = np.asarray(actual_outputs[name], dtype=np.float64)
        expected_values = np.asarray(expected_output["values"], dtype=np.float64)
        if list(actual.shape) != expected_output["shape"]:
            raise AssertionError(
                f"{api_id}.{name}: shape {actual.shape} != {expected_output['shape']}"
            )
        actual_nan_sha = hashlib.sha256(
            np.ascontiguousarray(~np.isfinite(actual)).tobytes()
        ).hexdigest()
        if actual_nan_sha != expected_output["nan_mask_sha256"]:
            raise AssertionError(f"{api_id}.{name}: non-finite mask changed")
        exact = (api_id, name) in CONTRACTUALLY_EXACT_OUTPUTS or (
            tolerance.atol == 0.0 and tolerance.rtol == 0.0
        )
        if exact:
            exact_output_count += 1
            if np.asarray(actual_outputs[name]).dtype.str != expected_output["dtype"]:
                raise AssertionError(f"{api_id}.{name}: exact output dtype changed")
            actual_sha = _output_digest(np.asarray(actual_outputs[name]))
            if actual_sha != expected_output["sha256"]:
                raise AssertionError(
                    f"{api_id}.{name}: exact output digest changed "
                    f"({actual_sha} != {expected_output['sha256']})"
                )
            continue
        result = _check_output(name, actual, expected_values, tolerance)
        if not result.passed:
            raise AssertionError(
                f"{api_id}.{name}: deterministic current output exceeded its "
                f"science tolerance (max_abs={result.max_abs:.3e}, "
                f"max_rel={result.max_rel:.3e})"
            )
    return output_count, exact_output_count


def check_fixture(path: Path, *, requested: list[str] | None = None) -> CurrentRegressionResult:
    fixture = json.loads(path.read_text())
    if fixture.get("schema_version") != 1:
        raise AssertionError("unsupported current regression fixture schema")
    if fixture.get("benchmark_source") != "current public/thin Python facade over Rust":
        raise AssertionError("current regression fixture source changed")
    if fixture.get("legacy_runtime_used") is not False:
        raise AssertionError("current regression fixture must not use a legacy runtime")
    api_ids = _selected_api_ids(requested)
    expected_apis = fixture["apis"]
    missing = sorted(set(api_ids) - set(expected_apis))
    if missing:
        raise AssertionError(f"current regression fixture is missing APIs: {missing}")
    output_count = 0
    exact_output_count = 0
    for api_id in api_ids:
        outputs, exact = _check_api(api_id, expected_apis[api_id])
        output_count += outputs
        exact_output_count += exact
    return CurrentRegressionResult(len(api_ids), output_count, exact_output_count)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--apis", nargs="*")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--capture", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    api_ids = _selected_api_ids(args.apis)
    if args.capture:
        fixture = capture_fixture(seed=args.seed, n=args.n, api_ids=api_ids)
        args.fixture.parent.mkdir(parents=True, exist_ok=True)
        args.fixture.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
        print(args.fixture)
        return 0
    result = check_fixture(args.fixture, requested=args.apis)
    print(
        f"current regression passed: {result.api_count} APIs, "
        f"{result.output_count} outputs, {result.exact_output_count} exact outputs"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
