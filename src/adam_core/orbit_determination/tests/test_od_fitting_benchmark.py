"""Governance tests for the current Core OD fitting benchmark."""

import json
from pathlib import Path

import numpy as np
import pytest

from adam_core import _rust_native
from migration.scripts import benchmark_od_fitting as benchmark


def test_native_od_benchmark_has_closed_operation_and_count_contracts() -> None:
    with pytest.raises(ValueError, match="operation must be exactly"):
        _rust_native.benchmark_od_fit_ipc(
            None, "other", b"", b"", b"", [], [], [], 1, 1, 0
        )
    with pytest.raises(ValueError, match="reps and trials must be >= 1"):
        _rust_native.benchmark_od_fit_ipc(
            None, "refine_none", b"", b"", b"", [], [], [], 0, 1, 0
        )


def test_od_benchmark_has_representative_workload_lanes() -> None:
    assert len(benchmark.WORKLOAD_SPECS) >= 2
    sizes = [spec[1] for spec in benchmark.WORKLOAD_SPECS]
    arcs = [spec[2] for spec in benchmark.WORKLOAD_SPECS]
    assert min(sizes) <= 8 and max(sizes) >= 32
    assert min(arcs) <= 21.0 and max(arcs) >= 90.0


def test_od_benchmark_timing_summary_preserves_raw_samples() -> None:
    samples = [[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]]
    summary = benchmark._summary(samples)
    assert summary["sample_trials_seconds"] == samples
    np.testing.assert_allclose(summary["trial_p50_seconds"], [2.0, 4.0])
    assert summary["p50_seconds"] == 3.0
    assert summary["p95_seconds"] > summary["p50_seconds"]


def test_od_benchmark_artifact_binds_build_provider_and_timing_policy(
    monkeypatch, tmp_path: Path
) -> None:
    fake_workloads = [object(), object()]
    monkeypatch.setattr(
        benchmark, "WORKLOAD_SPECS", (("a", 8, 21.0, 1), ("b", 32, 90.0, 2))
    )
    monkeypatch.setattr(benchmark, "_workload", lambda *args: fake_workloads.pop(0))
    monkeypatch.setattr(
        benchmark,
        "_benchmark_workload",
        lambda workload, **kwargs: {"name": "fake", "rows": []},
    )
    monkeypatch.setattr(
        benchmark._rust_native,
        "build_identity",
        lambda: ("0.7.0", "a" * 40),
    )
    monkeypatch.setattr(
        benchmark._rust_native,
        "native_provider_metadata",
        lambda propagator, settings: {
            "name": "adam_core.two_body",
            "version": "0.7.0",
            "configuration_fingerprint": "b" * 64,
            "data_fingerprint": "c" * 64,
        },
    )
    monkeypatch.setattr(benchmark.adam_core, "__version__", "0.7.0")
    monkeypatch.setattr(benchmark.metadata, "version", lambda name: "0.7.0")

    output = tmp_path / "benchmark.json"
    assert benchmark.main(["--quick", "--output", str(output)]) == 0
    payload = json.loads(output.read_text())
    assert payload["purpose"].endswith("non-gating")
    assert payload["timing_policy"]["setup_timed"] is False
    assert payload["timing_policy"]["correctness_preflight_timed"] is False
    assert payload["identity"]["native_build_commit"] == "a" * 40
    assert payload["identity"]["provider"]["configuration_fingerprint"] == "b" * 64
    assert payload["identity"]["provider"]["data_fingerprint"] == "c" * 64
    assert len(payload["workloads"]) == 2


def test_od_benchmark_rejects_identity_mismatch_before_work(
    monkeypatch, tmp_path: Path
) -> None:
    called = False

    def forbidden_workload(*args):
        nonlocal called
        called = True
        raise AssertionError("workload construction must follow identity validation")

    monkeypatch.setattr(benchmark, "_workload", forbidden_workload)
    monkeypatch.setattr(
        benchmark._rust_native,
        "build_identity",
        lambda: ("0.7.0", "a" * 40),
    )
    monkeypatch.setattr(benchmark.adam_core, "__version__", "0.7.0")
    monkeypatch.setattr(benchmark.metadata, "version", lambda name: "9.9.9")
    monkeypatch.setattr(
        benchmark._rust_native,
        "native_provider_metadata",
        lambda propagator, settings: {
            "configuration_fingerprint": "b" * 64,
            "data_fingerprint": "c" * 64,
        },
    )

    with pytest.raises(RuntimeError, match="versions disagree"):
        benchmark.main(["--quick", "--output", str(tmp_path / "no.json")])
    assert not called
