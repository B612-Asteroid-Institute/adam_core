import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

from adam_core._rust.status import (
    API_MIGRATIONS,
    API_MIGRATIONS_BY_ID,
    validate_api_migrations,
)
from migration.parity import (
    _inputs,
    _native_rust_runner,
    _threading,
    comparison_metadata,
    parity_fuzz,
    tolerances,
)
from migration.parity.backend_candidates import BACKEND_CANDIDATES_BY_ID
from migration.scripts import benchmark_current, parity_table
from migration.scripts.rust_backend_benchmark_gate import (
    BENCHMARK_TO_API_ID,
    EXTERNALLY_BENCHMARKED,
    _latency_summary,
    _thread_mode_from_argv,
)


def test_rust_migration_registry_validates() -> None:
    validate_api_migrations()


def test_random_fuzz_registry_matches_generators() -> None:
    random_fuzz_ids = {
        migration.api_id
        for migration in API_MIGRATIONS
        if migration.parity_coverage == "random-fuzz"
    }
    # bridge.* rows are diagnostic backend candidates (bead personal-cmy.13.1),
    # tracked in migration/parity/backend_candidates.py rather than in the
    # public per-API migration registry; the fuzz generators cover both.
    candidate_ids = set(BACKEND_CANDIDATES_BY_ID)

    assert not (random_fuzz_ids & candidate_ids)
    assert random_fuzz_ids | candidate_ids == set(_inputs.all_api_ids())


def test_tolerance_manifest_entries_are_registered() -> None:
    tolerance_ids = set(tolerances.all_api_ids())
    candidate_ids = set(BACKEND_CANDIDATES_BY_ID)
    # Every tolerance row must belong to either the public migration registry
    # or the diagnostic backend-candidate registry (bead personal-cmy.13.1).
    assert tolerance_ids <= set(API_MIGRATIONS_BY_ID) | candidate_ids

    expected_coverage = {
        "random-fuzz",
        "fixed-fixture",
        "random-fuzz-excluded",
        "orchestration-implied",
    }
    assert {
        API_MIGRATIONS_BY_ID[api_id].parity_coverage
        for api_id in tolerance_ids - candidate_ids
    } <= expected_coverage


def test_no_dual_rows_without_current_legacy_implementation() -> None:
    assert not [
        migration.api_id
        for migration in API_MIGRATIONS
        if migration.status == "dual" and not migration.current_legacy_impl
    ]


def test_transform_coordinates_partial_coverage_is_visible() -> None:
    migration = API_MIGRATIONS_BY_ID["coordinates.transform_coordinates"]

    assert migration.parity_coverage == "random-fuzz"
    assert migration.covered_subcases
    assert migration.excluded_subcases
    assert "public quivr-object dispatcher subcase matrix" in migration.coverage_note
    assert any("CartesianCoordinates" in case for case in migration.covered_subcases)
    assert any("SphericalCoordinates" in case for case in migration.covered_subcases)
    assert any("KeplerianCoordinates" in case for case in migration.covered_subcases)
    assert any("CometaryCoordinates" in case for case in migration.covered_subcases)
    assert any("origin translations" in case for case in migration.covered_subcases)
    assert any("ITRF93" in case for case in migration.covered_subcases)
    assert any("covariance-bearing" in case for case in migration.covered_subcases)
    assert migration.boundary == "arrow"
    assert any("non-Cartesian input" in case for case in migration.excluded_subcases)
    assert any("geodetic input" in case for case in migration.excluded_subcases)


def test_transform_coordinates_parity_pins_shared_spice_kernel_paths() -> None:
    rng = np.random.default_rng(20260429)
    sample = _inputs.make("coordinates.transform_coordinates", rng, 128)

    rust_kernels = sample.rust_kwargs["spice_kernels"]
    legacy_kernels = sample.legacy_kwargs["spice_kernels"]
    assert rust_kernels == legacy_kernels
    assert len(rust_kernels) == 6
    assert any(path.endswith(".bpc") for path in rust_kernels)


def test_parity_artifact_records_spice_kernel_provenance() -> None:
    artifact = parity_fuzz.to_json([])
    provenance = artifact["spice_kernel_provenance"]

    assert provenance["naif_eop_high_prec_version"]
    kernels = provenance["kernels"]
    assert len(kernels) == 6
    assert all(len(kernel["sha256"]) == 64 for kernel in kernels)
    assert all(kernel["size_bytes"] > 0 for kernel in kernels)
    assert any(str(kernel["path"]).endswith(".bpc") for kernel in kernels)


def test_parity_output_reports_headroom_and_nan_policy() -> None:
    result = parity_fuzz._check_output(
        "out",
        np.array([1.0 + 1e-13]),
        np.array([1.0]),
        tolerances.OutputTol(atol=1e-12, rtol=0.0),
    )

    assert result.passed
    assert 0.09 < result.max_tolerance_ratio < 0.11
    assert result.max_rel_above_atol_floor > 0.0

    rows = parity_table._build_rows(
        [
            parity_fuzz.ApiResult(
                api_id="coordinates.cartesian_to_spherical",
                investigate=False,
                investigate_task="",
                seeds=[parity_fuzz.SeedResult(seed=1, n=1, outputs=[result])],
            )
        ],
        [],
    )
    row = next(
        row
        for row in rows
        if row["api_id"] == "coordinates.cartesian_to_spherical"
        and row["state"] == "measured"
    )
    assert 9.0 < row["margin"] < 11.0
    assert row["nan_disagreement"] == 0

    nan_mismatch = parity_fuzz._check_output(
        "out",
        np.array([np.nan]),
        np.array([1.0]),
        tolerances.OutputTol(atol=1e-12, rtol=0.0),
    )

    assert not nan_mismatch.passed
    assert nan_mismatch.nan_disagreement == 1
    assert np.isinf(nan_mismatch.max_tolerance_ratio)


def test_comparison_mode_metadata_labels() -> None:
    facade = comparison_metadata.for_api("coordinates.transform_coordinates")
    assert facade["comparison_mode"] == comparison_metadata.PUBLIC_PYTHON_FACADE
    assert facade["comparison_mode_short"] == "public facade"
    assert facade["rust_native_top_level"] is False

    kernel = comparison_metadata.for_api("statistics.weighted_mean")
    assert kernel["comparison_mode"] == comparison_metadata.RAW_RUST_PYO3_KERNEL
    assert kernel["speed_gate_scope"] == "diagnostic_raw_kernel"

    # bridge.* candidate lanes were retired under bead personal-cmy.36.10;
    # retired ids now resolve like any unknown id.
    retired = comparison_metadata.for_api("bridge.rotate_orbits_frame")
    assert retired["comparison_mode"] == comparison_metadata.UNKNOWN

    unknown = comparison_metadata.for_api("nonexistent.api")
    assert unknown["comparison_mode"] == comparison_metadata.UNKNOWN


def test_comparison_mode_metadata_covers_registry_and_candidates() -> None:
    for migration in API_MIGRATIONS:
        meta = comparison_metadata.for_api(migration.api_id)
        assert meta["comparison_mode"] != comparison_metadata.UNKNOWN, migration.api_id
        assert meta["comparison_mode_short"], migration.api_id
        assert meta["rust_native_top_level"] == (migration.status == "rust-only")
    for candidate_id in BACKEND_CANDIDATES_BY_ID:
        meta = comparison_metadata.for_api(candidate_id)
        assert meta["comparison_mode"] == comparison_metadata.BACKEND_CANDIDATE


def test_photometry_h_fit_random_fuzz_is_visible() -> None:
    api_ids = {
        "photometry.fit_absolute_magnitude_rows",
        "photometry.fit_absolute_magnitude_grouped",
    }

    assert api_ids <= set(_inputs.all_api_ids())
    for api_id in api_ids:
        migration = API_MIGRATIONS_BY_ID[api_id]
        assert migration.parity_coverage == "random-fuzz"
        assert "Randomized parity" in migration.coverage_note


def test_raw_statistics_kernels_are_random_fuzz_with_diagnostic_speed() -> None:
    api_ids = {
        "statistics.weighted_mean",
        "statistics.weighted_covariance",
    }

    assert api_ids <= set(_inputs.all_api_ids())
    for api_id in api_ids:
        migration = API_MIGRATIONS_BY_ID[api_id]
        assert migration.status == "raw-kernel-only"
        assert migration.parity_coverage == "random-fuzz"
        assert "diagnostic raw-kernel comparisons" in migration.coverage_note


def test_raw_coordinate_kernels_are_random_fuzz_with_diagnostic_speed() -> None:
    api_ids = {
        "coordinates.transform_coordinates_with_covariance",
        "coordinates.rotate_cartesian_time_varying",
    }

    assert api_ids <= set(_inputs.all_api_ids())
    for api_id in api_ids:
        migration = API_MIGRATIONS_BY_ID[api_id]
        assert migration.status == "raw-kernel-only"
        assert migration.parity_coverage == "random-fuzz"
        assert "diagnostic raw-kernel comparisons" in migration.coverage_note
        assert migration.covered_subcases

    transform_covariance = API_MIGRATIONS_BY_ID[
        "coordinates.transform_coordinates_with_covariance"
    ]
    assert "Any NaN" in transform_covariance.coverage_note
    assert "short-circuits" in transform_covariance.coverage_note

    rotation = API_MIGRATIONS_BY_ID["coordinates.rotate_cartesian_time_varying"]
    assert "zero-fill-then-restore" in rotation.coverage_note
    assert "non-physical" in rotation.coverage_note


def test_residual_helper_kernels_are_random_fuzz() -> None:
    api_ids = {
        "coordinates.residuals.apply_cosine_latitude_correction",
        "coordinates.residuals.bound_longitude_residuals",
    }

    assert api_ids <= set(_inputs.all_api_ids())
    for api_id in api_ids:
        migration = API_MIGRATIONS_BY_ID[api_id]
        assert migration.status == "public-rust-default"
        assert migration.parity_coverage == "random-fuzz"
        assert migration.covered_subcases


def test_tisserand_parameter_is_random_fuzz() -> None:
    api_id = "dynamics.tisserand_parameter"

    assert api_id in set(_inputs.all_api_ids())
    migration = API_MIGRATIONS_BY_ID[api_id]
    assert migration.status == "public-rust-default"
    assert migration.parity_coverage == "random-fuzz"
    assert migration.covered_subcases


def test_raw_propagation_arc_kernels_are_random_fuzz_with_diagnostic_speed() -> None:
    api_ids = {
        "dynamics.propagate_2body_along_arc",
        "dynamics.propagate_2body_arc_batch",
    }

    assert api_ids <= set(_inputs.all_api_ids())
    for api_id in api_ids:
        migration = API_MIGRATIONS_BY_ID[api_id]
        assert migration.status == "raw-kernel-only"
        assert migration.parity_coverage == "random-fuzz"
        assert "diagnostic raw-kernel comparisons" in migration.coverage_note
        assert migration.covered_subcases


def test_raw_batch_kernels_are_random_fuzz_with_diagnostic_speed() -> None:
    api_ids = {
        "dynamics.calculate_moid_batch",
        "missions.porkchop_grid",
    }

    assert api_ids <= set(_inputs.all_api_ids())
    for api_id in api_ids:
        migration = API_MIGRATIONS_BY_ID[api_id]
        assert migration.status == "raw-kernel-only"
        assert migration.parity_coverage == "random-fuzz"
        assert "diagnostic raw-kernel comparisons" in migration.coverage_note
        assert migration.covered_subcases


def test_latency_gate_registry_matches_latency_benchmark_scope() -> None:
    latency_ids = {
        migration.api_id
        for migration in API_MIGRATIONS
        if migration.default == "rust" and migration.latency_gate
    }
    benchmarked_ids = set(BENCHMARK_TO_API_ID.values()) | EXTERNALLY_BENCHMARKED

    assert latency_ids <= benchmarked_ids
    assert "coordinates.residuals.calculate_chi2" in latency_ids
    assert "dynamics.add_light_time" not in latency_ids
    assert "orbit_determination.gaussIOD" in latency_ids
    assert "dynamics.calculate_perturber_moids" not in latency_ids
    assert "dynamics.generate_porkchop_data" not in latency_ids


def test_latency_gate_defaults_to_single_thread_policy() -> None:
    # rust-latency-gate is Rust-only regression detection; single-thread Rayon
    # gives a stable measurement that does not depend on Rust-vs-JAX core
    # asymmetry. This is intentionally separate from parity_speed which now
    # defaults to multi-thread for production-realistic comparison.
    assert _thread_mode_from_argv([]) == "single"
    assert _thread_mode_from_argv(["--threads", "multi-thread"]) == "multi-thread"
    # 'native' is accepted as a deprecated alias for 'multi-thread'.
    assert _thread_mode_from_argv(["--threads", "native"]) == "native"
    assert _thread_mode_from_argv(["--threads=single"]) == "single"


def test_github_actions_latency_baseline_matches_benchmark_scope() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    baseline = json.loads(
        (
            repo_root
            / "migration"
            / "artifacts"
            / "rust_latency_baseline_github_ubuntu.json"
        ).read_text(encoding="utf-8")
    )

    benchmark_keys = {key for key in baseline if not key.startswith("_")}
    assert benchmark_keys == set(BENCHMARK_TO_API_ID)
    assert baseline["_metadata"]["thread_mode"] == "single"
    for name in benchmark_keys:
        assert baseline[name]["thread_mode"] == "single"
        assert baseline[name]["rust_seconds_p50"] > 0.0
        assert baseline[name]["rust_seconds_p95"] > 0.0


def test_current_ci_scripts_are_complete_normal_ci_legacy_free_gates() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    with (repo_root / "pyproject.toml").open("rb") as pyproject_file:
        command = tomllib.load(pyproject_file)["tool"]["pdm"]["scripts"][
            "benchmark-current-ci"
        ]
    workflow = (
        repo_root / ".github" / "workflows" / "pip-build-lint-test-coverage.yml"
    ).read_text()

    assert "--lanes tiny small large" in command
    assert "--trials 3" in command
    assert "--require-native" in command
    assert "--quick" not in command
    assert "pdm run benchmark-current-ci" in workflow
    assert "pdm run test-current-regression" in workflow
    assert "current-only-benchmark" in workflow


def test_current_benchmark_reuses_registry_and_canonical_lane_shapes() -> None:
    parser = benchmark_current._build_arg_parser()
    args = parser.parse_args([])

    assert benchmark_current._selected_api_ids(None, None) == [
        migration.api_id for migration in API_MIGRATIONS
    ]
    lanes = benchmark_current._lanes(args)
    assert [lane.name for lane in lanes] == ["tiny-n", "small-n", "large-n"]
    assert [lane.reps for lane in lanes] == [101, 21, 7]
    assert args.trials == 3
    assert args.quick is False
    help_text = parser.format_help().lower()
    source = Path(benchmark_current.__file__).read_text()
    assert "legacy-cache" not in help_text
    assert "legacy-root" not in help_text
    assert "oracle-python" not in help_text
    assert "_legacy_runner" not in source
    assert "LEGACY_REPO_ROOT" not in source
    assert "LEGACY_VENV_PYTHON" not in source


def test_thread_mode_native_is_deprecated_alias_for_multi_thread() -> None:
    assert _threading.validate_thread_mode("single") == "single"
    assert _threading.validate_thread_mode("multi-thread") == "multi-thread"
    # 'native' is accepted as a deprecated backward-compatibility alias and
    # is normalized to the canonical 'multi-thread' name.
    assert _threading.validate_thread_mode("native") == "multi-thread"


def test_multi_thread_mode_removes_caps_for_both_rust_and_legacy() -> None:
    # multi-thread mode strips harness-imposed caps from BOTH the Rust
    # (Rayon) side and the legacy baseline (NumPy/JAX/XLA/BLAS) side so
    # production-realistic scaling can be measured fairly.
    base = {key: "1" for key in _threading.THREAD_ENV_KEYS}
    base["XLA_FLAGS"] = (
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    )
    env = _threading.env_for_thread_mode("multi-thread", base_env=base)
    for key in _threading.THREAD_ENV_KEYS:
        assert key not in env, f"{key} should be uncapped in multi-thread mode"
    # Non-default external values are preserved as authored.
    base_with_external = dict(base)
    base_with_external["RAYON_NUM_THREADS"] = "4"
    env2 = _threading.env_for_thread_mode("multi-thread", base_env=base_with_external)
    assert env2["RAYON_NUM_THREADS"] == "4"
    # 'native' alias produces the same env as canonical 'multi-thread'.
    assert _threading.env_for_thread_mode("native", base_env=base) == env


def test_latency_summary_uses_median_of_trial_percentiles() -> None:
    samples = np.asarray(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 100.0],
            [2.0, 2.0, 2.0, 2.0, 2.0],
        ],
        dtype=np.float64,
    )

    summary = _latency_summary(samples)

    assert summary["latency_trials"] == 3
    assert summary["rust_seconds_p50"] == 1.0
    assert summary["rust_seconds_p95"] == 2.0
    assert summary["rust_seconds_p95_trials"][1] > 50.0
    assert summary["rust_sample_trials_seconds"] == samples.tolist()
    assert summary["latency_aggregation"] == "median-of-trial-percentiles"


def test_workload_shape_records_multi_axis_large_lanes() -> None:
    workloads = _inputs.lane_workloads()
    ephemeris = workloads["large-n"]["dynamics.generate_ephemeris_2body"]
    photometry = workloads["large-n"]["photometry.predict_magnitudes"]

    assert ephemeris.rows == 20_000
    assert ephemeris.axes() == {"orbits": 400, "epochs": 50}
    assert photometry.axes() == {"orbits": 1000, "observers": 50}
    assert "×" in ephemeris.label()


def test_simple_timing_renderer_uses_canonical_candidate_names_and_blank_native() -> (
    None
):
    rows = [
        {
            "api_id": "orbits.VariantOrbits.create",
            "lane": "tiny-n",
            "legacy_p50_s": 2.0,
            "legacy_p95_s": 2.5,
            "current_python_p50_s": 1.0,
            "current_python_p95_s": 1.5,
            "native_rust_p50_s": None,
            "native_rust_p95_s": None,
        },
        {
            "api_id": "observers.Observers.from_codes",
            "lane": "tiny-n",
            "legacy_p50_s": 0.006,
            "legacy_p95_s": 0.007,
            "current_python_p50_s": 0.0002,
            "current_python_p95_s": 0.0003,
            "native_rust_p50_s": 0.00001,
            "native_rust_p95_s": 0.00002,
        },
    ]

    rendered = parity_table._format_simple_speed_timing_tables(rows)

    assert "orbits.VariantOrbits.create" in rendered
    assert "| 2.000s / 2.500s | 1.000s / 1.500s |  |" in rendered
    assert "| 6.00ms / 7.00ms | 200.0µs / 300.0µs | 10.0µs / 20.0µs |" in rendered
    assert (
        parity_table._build_arg_parser().parse_args(["--simple-timings"]).simple_timings
    )


def test_native_rust_timer_is_internal_and_missing_surfaces_are_blank(
    monkeypatch,
) -> None:
    def fake_native_timer(**kwargs):
        assert kwargs["reps"] == 3
        assert kwargs["trials"] == 3
        return _native_rust_runner.NativeRustTiming(
            status="measured",
            sample_trials_s=[[1.0, 1.0, 1.0]] * 3,
            entrypoint="example::direct_rust",
            timing_boundary=(
                "Rust std::time::Instant; outer Python/PyO3 launch excluded"
            ),
        )

    monkeypatch.setitem(
        _native_rust_runner._ADAPTERS,
        "observers.Observers.from_codes",
        fake_native_timer,
    )
    rng = np.random.default_rng(20260709)
    observer_sample = _inputs.make("observers.Observers.from_codes", rng, 10)
    native = _native_rust_runner.measure(
        "observers.Observers.from_codes",
        observer_sample.rust_kwargs,
        reps=3,
        warmup=1,
        trials=3,
    )
    assert native.status == "measured"
    assert len(native.sample_trials_s) == 3
    assert all(len(trial) == 3 for trial in native.sample_trials_s)
    assert "Instant" in native.timing_boundary
    assert "PyO3 launch" in native.timing_boundary
    assert native.entrypoint == "example::direct_rust"

    transform_sample = _inputs.make(
        "coordinates.transform_coordinates_with_covariance", rng, 4
    )
    transform_native = _native_rust_runner.measure(
        "coordinates.transform_coordinates_with_covariance",
        transform_sample.rust_kwargs,
        reps=3,
        warmup=1,
        trials=3,
    )
    assert transform_native.status == "measured"
    assert len(transform_native.sample_trials_s) == 3
    assert all(len(trial) == 3 for trial in transform_native.sample_trials_s)
    assert "Instant" in transform_native.timing_boundary
    assert transform_native.entrypoint == (
        "adam_core_rs_coords::transform_with_covariance_flat"
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("api_id", "entrypoint"),
    [
        ("orbit_determination.calcGibbs", "calc_gibbs_row"),
        ("orbit_determination.calcHerrickGibbs", "calc_herrick_gibbs_row"),
        ("orbit_determination.calcGauss", "calc_gauss_row"),
    ],
)
def test_orbit_determination_kernel_native_rust_adapters_live(
    api_id: str, entrypoint: str
) -> None:
    """Canonical IOD lanes record only direct Rust kernel calls."""
    rng = np.random.default_rng(20260710)
    sample = _inputs.make(api_id, rng, 10)
    native = _native_rust_runner.measure(
        api_id,
        sample.rust_kwargs,
        reps=2,
        warmup=1,
        trials=2,
    )

    assert native.status == "measured", native.reason
    assert len(native.sample_trials_s) == 2
    assert all(len(trial) == 2 for trial in native.sample_trials_s)
    assert all(value > 0.0 for trial in native.sample_trials_s for value in trial)
    assert native.entrypoint == f"adam_core_rs_orbit_determination::{entrypoint}"
    assert "std::time::Instant" in native.timing_boundary
    assert "Python/PyO3 launch" in native.timing_boundary
    assert "NumPy access excluded" in native.timing_boundary


@pytest.mark.integration
@pytest.mark.parametrize(
    "api_id",
    [
        "coordinates.cartesian_to_spherical",
        "coordinates.spherical.to_cartesian",
        "coordinates.cartesian_to_geodetic",
        "coordinates.cartesian_to_keplerian",
        "coordinates.keplerian.to_cartesian",
        "coordinates.cartesian_to_cometary",
        "coordinates.cometary.to_cartesian",
        "coordinates.rotate_cartesian_time_varying",
        "coordinates.residuals.calculate_chi2",
        "coordinates.residuals.bound_longitude_residuals",
        "coordinates.residuals.apply_cosine_latitude_correction",
        "statistics.weighted_mean",
        "statistics.weighted_covariance",
        "orbits.classify_orbits",
        "dynamics.calc_mean_motion",
        "dynamics.tisserand_parameter",
        "dynamics.calculate_moid",
        "dynamics.calculate_moid_batch",
        "missions.porkchop_grid",
        "dynamics.propagate_2body_along_arc",
        "dynamics.propagate_2body_arc_batch",
        "dynamics.propagate_2body_with_covariance",
        "dynamics.solve_lambert",
        "dynamics.add_light_time",
        "photometry.calculate_phase_angle",
        "photometry.calculate_apparent_magnitude_v",
        "photometry.calculate_apparent_magnitude_v_and_phase_angle",
        "photometry.predict_magnitudes",
        "photometry.fit_absolute_magnitude_rows",
        "photometry.fit_absolute_magnitude_grouped",
    ],
)
def test_numpy_kernel_native_rust_adapters_live(api_id: str) -> None:
    """Registered NumPy-flat lanes measure direct Rust calls with Rust clocks."""
    rng = np.random.default_rng(20260711)
    sample = _inputs.make(api_id, rng, 10)
    native = _native_rust_runner.measure(
        api_id,
        sample.rust_kwargs,
        reps=2,
        warmup=1,
        trials=2,
    )

    assert native.status == "measured", native.reason
    assert len(native.sample_trials_s) == 2
    assert all(len(trial) == 2 for trial in native.sample_trials_s)
    # Sub-tick kernels (e.g. 10-row calc_mean_motion) can quantize to exactly
    # 0.0 on Apple Silicon's ~41.7ns Instant granularity; samples must be
    # finite and non-negative, and the lane must remain measurable.
    assert all(
        np.isfinite(value) and value >= 0.0
        for trial in native.sample_trials_s
        for value in trial
    )
    assert native.entrypoint.startswith("adam_core_rs_coords::")
    assert "std::time::Instant" in native.timing_boundary
    assert "Python/PyO3 launch" in native.timing_boundary


@pytest.mark.integration
def test_observer_native_rust_adapter_live() -> None:
    """A registered native adapter must not silently degrade to a blank column."""
    rng = np.random.default_rng(20260709)
    observer_sample = _inputs.make("observers.Observers.from_codes", rng, 10)
    native = _native_rust_runner.measure(
        "observers.Observers.from_codes",
        observer_sample.rust_kwargs,
        reps=2,
        warmup=1,
        trials=2,
    )

    assert native.status == "measured", native.reason
    assert len(native.sample_trials_s) == 2
    assert all(len(trial) == 2 for trial in native.sample_trials_s)
    assert all(value > 0.0 for trial in native.sample_trials_s for value in trial)
    assert native.entrypoint == (
        "adam_core_py::spice::observer_states_from_codes_record_batch"
    )
    assert "std::time::Instant" in native.timing_boundary
    assert "PyArrow conversion excluded" in native.timing_boundary


@pytest.mark.integration
def test_transform_coordinates_native_rust_adapter_live() -> None:
    """The Arrow transform adapter times only Rust-owned direct calls."""
    rng = np.random.default_rng(20260709)
    transform_sample = _inputs.make("coordinates.transform_coordinates", rng, 12)
    native = _native_rust_runner.measure(
        "coordinates.transform_coordinates",
        transform_sample.rust_kwargs,
        reps=2,
        warmup=1,
        trials=2,
    )

    assert native.status == "measured", native.reason
    assert len(native.sample_trials_s) == 2
    assert all(len(trial) == 2 for trial in native.sample_trials_s)
    assert all(value > 0.0 for trial in native.sample_trials_s for value in trial)
    assert native.entrypoint == (
        "adam_core_py::coordinates::transform_coordinates_record_batch"
    )
    assert "std::time::Instant" in native.timing_boundary
    assert "PyArrow conversion excluded" in native.timing_boundary


@pytest.mark.integration
def test_propagate_2body_native_rust_adapter_live() -> None:
    """The Arrow propagation adapter times only Rust-owned direct calls."""
    rng = np.random.default_rng(20260709)
    propagation_sample = _inputs.make("dynamics.propagate_2body", rng, 10)
    native = _native_rust_runner.measure(
        "dynamics.propagate_2body",
        propagation_sample.rust_kwargs,
        reps=2,
        warmup=1,
        trials=2,
    )

    assert native.status == "measured", native.reason
    assert len(native.sample_trials_s) == 2
    assert all(len(trial) == 2 for trial in native.sample_trials_s)
    assert all(value > 0.0 for trial in native.sample_trials_s for value in trial)
    assert native.entrypoint == (
        "adam_core_py::coordinates::propagate_orbits_record_batch"
    )
    assert "std::time::Instant" in native.timing_boundary
    assert "PyArrow conversion excluded" in native.timing_boundary


@pytest.mark.integration
def test_generate_ephemeris_native_rust_adapter_live() -> None:
    """The Arrow ephemeris adapter times only Rust-owned direct calls."""
    rng = np.random.default_rng(20260709)
    sample = _inputs.make("dynamics.generate_ephemeris_2body", rng, 10)
    native = _native_rust_runner.measure(
        "dynamics.generate_ephemeris_2body",
        sample.rust_kwargs,
        reps=2,
        warmup=1,
        trials=2,
    )

    assert native.status == "measured", native.reason
    assert len(native.sample_trials_s) == 2
    assert all(len(trial) == 2 for trial in native.sample_trials_s)
    assert all(value > 0.0 for trial in native.sample_trials_s for value in trial)
    assert native.entrypoint == (
        "adam_core_py::coordinates::generate_ephemeris_record_batch"
    )
    assert "std::time::Instant" in native.timing_boundary
    assert "PyArrow conversion excluded" in native.timing_boundary


def test_every_parity_api_has_an_intentional_native_rust_todo_bucket() -> None:
    todos = {
        api_id: _native_rust_runner._todo_for(api_id)
        for api_id in _inputs.all_api_ids()
    }
    assert set(todos.values()) <= {
        "personal-3gg",
        "personal-98v.1",
        "personal-98v.1.3",
        "personal-cmy.36.4",
        "personal-cmy.36.5",
        "personal-cmy.36.6",
        "personal-cmy.36.7",
        "personal-cmy.36.8",
        "personal-cmy.36.9",
        "personal-cmy.36.10",
    }
    # Scalar helpers and classified numpy-flat kernels use the dedicated
    # native-benchmark catch-all bead; bridge.* candidate lanes were retired
    # under bead personal-cmy.36.10.
    catch_all = {api_id for api_id, todo in todos.items() if todo == "personal-98v.1"}
    assert {
        "dynamics.calc_mean_motion",
        "dynamics.tisserand_parameter",
        "coordinates.transform_coordinates_with_covariance",
        "coordinates.rotate_cartesian_time_varying",
    } <= catch_all
    assert not any(api_id.startswith("bridge.") for api_id in todos)
