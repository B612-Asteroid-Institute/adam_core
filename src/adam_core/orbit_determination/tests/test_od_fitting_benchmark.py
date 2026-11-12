"""Governance tests for the current Core OD fitting benchmark."""

import json
from pathlib import Path

import numpy as np
import pytest

from adam_core import _rust_native
from migration.scripts import benchmark_od_fitting as benchmark


def test_native_od_benchmark_has_closed_operation_and_count_contracts() -> None:
    config = benchmark._benchmark_config()
    with pytest.raises(ValueError, match="operation must be exactly"):
        _rust_native.benchmark_od_fit_ipc(
            None, "other", b"", b"", b"", [], [], [], config._native, 1, 1, 0
        )
    with pytest.raises(ValueError, match="reps and trials must be >= 1"):
        _rust_native.benchmark_od_fit_ipc(
            None, "refine", b"", b"", b"", [], [], [], config._native, 0, 1, 0
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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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
        lambda: ("0.8.0", "a" * 40),
    )
    monkeypatch.setattr(
        benchmark._rust_native,
        "native_provider_metadata",
        lambda propagator, settings: {
            "name": "adam_core.two_body",
            "version": "0.8.0",
            "configuration_fingerprint": "b" * 64,
            "data_fingerprint": "c" * 64,
        },
    )
    monkeypatch.setattr(benchmark.adam_core, "__version__", "0.8.0")
    monkeypatch.setattr(benchmark.metadata, "version", lambda name: "0.8.0")
    monkeypatch.setattr(
        benchmark,
        "_installed_artifact_identity",
        lambda: {
            "installation_kind": "wheel",
            "outside_git_checkout": True,
            "native_build_profile": {
                "cargo_profile": "release",
                "cargo_opt_level": "3",
                "cargo_target": "test-target",
                "debug_assertions": False,
            },
        },
    )
    monkeypatch.setattr(
        benchmark,
        "_runtime_state",
        lambda: {"threads": {"logical_cpu_count": 1}, "spice": {}},
    )

    output = tmp_path / "benchmark.json"
    assert benchmark.main(["--quick", "--output", str(output)]) == 0
    payload = json.loads(output.read_text())
    assert payload["purpose"].endswith("non-gating")
    assert payload["timing_policy"]["setup_timed"] is False
    assert payload["timing_policy"]["correctness_preflight_timed"] is False
    assert payload["identity"]["native_build_commit"] == "a" * 40
    assert payload["identity"]["provider"]["configuration_fingerprint"] == "b" * 64
    assert payload["identity"]["provider"]["data_fingerprint"] == "c" * 64
    assert payload["identity"]["installed_artifact"]["outside_git_checkout"] is True
    assert (
        payload["identity"]["installed_artifact"]["native_build_profile"][
            "cargo_profile"
        ]
        == "release"
    )
    assert payload["runtime_state"]["before_setup"]["threads"]["logical_cpu_count"] == 1
    assert len(payload["workloads"]) == 2


@pytest.mark.parametrize(
    "profile",
    [
        ("dev", "0", "test-target", True),
        ("release", "0", "test-target", False),
        ("release", "3", "test-target", True),
    ],
)
def test_od_benchmark_rejects_nonoptimized_native_build(
    monkeypatch: pytest.MonkeyPatch,
    profile: tuple[str, str, str, bool],
) -> None:
    monkeypatch.setattr(benchmark._rust_native, "native_build_profile", lambda: profile)
    with pytest.raises(RuntimeError, match="optimized release native artifact"):
        benchmark._native_build_profile_identity()


def test_od_benchmark_accepts_exact_release_native_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        benchmark._rust_native,
        "native_build_profile",
        lambda: ("release", "3", "aarch64-unknown-linux-musl", False),
    )
    assert benchmark._native_build_profile_identity() == {
        "cargo_profile": "release",
        "cargo_opt_level": "3",
        "cargo_target": "aarch64-unknown-linux-musl",
        "debug_assertions": False,
    }


def test_od_benchmark_rejects_package_imported_from_git_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    package = checkout / "src" / "adam_core" / "__init__.py"
    package.parent.mkdir(parents=True)
    package.write_text("")
    monkeypatch.setattr(benchmark.adam_core, "__file__", str(package))

    with pytest.raises(RuntimeError, match="executes from Git checkout"):
        benchmark._installed_artifact_identity()


def test_od_benchmark_rejects_identity_mismatch_before_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called = False

    def forbidden_workload(*args: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("workload construction must follow identity validation")

    monkeypatch.setattr(benchmark, "_workload", forbidden_workload)
    monkeypatch.setattr(
        benchmark._rust_native,
        "build_identity",
        lambda: ("0.8.0", "a" * 40),
    )
    monkeypatch.setattr(benchmark.adam_core, "__version__", "0.8.0")
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
