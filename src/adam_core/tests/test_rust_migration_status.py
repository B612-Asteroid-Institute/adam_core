from pathlib import Path

import numpy as np

from adam_core._rust.status import (
    API_MIGRATIONS,
    API_MIGRATIONS_BY_ID,
    validate_api_migrations,
)
from migration.parity import (
    _inputs,
    _oracle,
    _threading,
    parity_fixed,
    parity_fuzz,
    parity_main,
    parity_speed,
    tolerances,
)
from migration.scripts import parity_table
from migration.scripts.rust_backend_benchmark_gate import (
    BENCHMARK_TO_API_ID,
    EXTERNALLY_BENCHMARKED,
    _build_arg_parser as _build_latency_arg_parser,
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

    assert random_fuzz_ids == set(_inputs.all_api_ids())


def test_tolerance_manifest_entries_are_registered() -> None:
    tolerance_ids = set(tolerances.all_api_ids())
    assert tolerance_ids <= set(API_MIGRATIONS_BY_ID)

    expected_coverage = {
        "random-fuzz",
        "fixed-fixture",
        "random-fuzz-excluded",
        "orchestration-implied",
    }
    assert {
        API_MIGRATIONS_BY_ID[api_id].parity_coverage for api_id in tolerance_ids
    } <= expected_coverage


def test_no_dual_rows_without_current_legacy_implementation() -> None:
    assert not [
        migration.api_id
        for migration in API_MIGRATIONS
        if migration.status == "dual" and not migration.current_legacy_impl
    ]


def test_fixed_fixture_manifest_entries_are_registered() -> None:
    fixed_fixture_ids = set(parity_fixed.all_api_ids())

    assert fixed_fixture_ids <= set(API_MIGRATIONS_BY_ID)
    assert all(
        API_MIGRATIONS_BY_ID[api_id].parity_coverage in {"fixed-fixture", "random-fuzz"}
        for api_id in fixed_fixture_ids
    )


def test_gauss_iod_constrained_random_fuzz_is_visible() -> None:
    migration = API_MIGRATIONS_BY_ID["orbit_determination.gaussIOD"]

    assert migration.parity_coverage == "random-fuzz"
    assert migration.coverage_note
    assert "Random fuzz is constrained" in migration.coverage_note
    assert migration.covered_subcases
    assert migration.excluded_subcases
    assert "orbit_determination.gaussIOD" in _inputs.all_api_ids()
    assert "orbit_determination.gaussIOD" in parity_fixed.all_api_ids()


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
    assert any("Cartesian->Cartesian" in case for case in migration.excluded_subcases)
    assert any(
        "covariance-bearing ITRF93" in case for case in migration.excluded_subcases
    )


def test_covariance_finite_difference_fixtures_are_visible() -> None:
    api_ids = {
        "dynamics.propagate_2body_with_covariance",
        "dynamics.generate_ephemeris_2body_with_covariance",
    }

    assert api_ids <= set(parity_fixed.all_api_ids())
    for api_id in api_ids:
        migration = API_MIGRATIONS_BY_ID[api_id]
        assert migration.parity_coverage == "random-fuzz"
        assert "finite-difference covariance fixture" in migration.coverage_note
        assert any(
            "finite-difference covariance witness" in case
            for case in migration.covered_subcases
        )


def test_moid_fixed_fixtures_cover_flat_and_unique_minima() -> None:
    migration = API_MIGRATIONS_BY_ID["dynamics.calculate_moid"]
    fixture_names = {
        fixture.name
        for fixture in parity_fixed.FIXTURES_BY_API["dynamics.calculate_moid"]
    }

    assert {
        "identical_circular_flat_minimum",
        "well_conditioned_unique_minimum",
    } <= fixture_names
    assert "unique-minimum" in migration.coverage_note


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
        assert parity_speed._is_diagnostic_speed_api(api_id)


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
        assert parity_speed._is_diagnostic_speed_api(api_id)

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
        assert not parity_speed._is_diagnostic_speed_api(api_id)


def test_tisserand_parameter_is_random_fuzz() -> None:
    api_id = "dynamics.tisserand_parameter"

    assert api_id in set(_inputs.all_api_ids())
    migration = API_MIGRATIONS_BY_ID[api_id]
    assert migration.status == "public-rust-default"
    assert migration.parity_coverage == "random-fuzz"
    assert migration.covered_subcases
    assert not parity_speed._is_diagnostic_speed_api(api_id)


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
        assert parity_speed._is_diagnostic_speed_api(api_id)


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
        assert parity_speed._is_diagnostic_speed_api(api_id)


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


def test_parity_speed_default_thread_mode_is_multi_thread() -> None:
    parser = parity_speed._build_arg_parser()
    args = parser.parse_args([])
    assert args.threads == "multi-thread"


def test_parity_main_default_thread_mode_is_multi_thread() -> None:
    parser = parity_main._build_arg_parser()
    args = parser.parse_args([])
    assert args.threads == "multi-thread"


def test_speed_trial_counts_are_source_governed() -> None:
    assert parity_speed.CANONICAL_SPEED_TRIALS >= 3
    assert "--trials" not in parity_speed._build_arg_parser().format_help()
    assert "--speed-trials" not in parity_main._build_arg_parser().format_help()
    assert "--trials" not in _build_latency_arg_parser().format_help()


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


def test_parity_speed_summary_uses_median_of_trial_percentiles() -> None:
    samples = [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 100.0],
        [2.0, 2.0, 2.0, 2.0, 2.0],
    ]

    summary = parity_speed._timing_summary(samples)

    assert summary["timing_trials"] == 3
    assert summary["p50_s"] == 1.0
    assert summary["p95_s"] == 2.0
    assert summary["p95_trials_s"][1] > 50.0
    assert summary["sample_trials_s"] == samples
    assert summary["timing_aggregation"] == "median-of-trial-percentiles"


def test_single_thread_policy_sets_and_forwards_caps(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/tmp/should-not-leak")
    monkeypatch.setenv("RAYON_NUM_THREADS", "8")

    env: dict[str, str] = {}
    snapshot = _threading.apply_thread_mode("single", env)
    assert snapshot["RAYON_NUM_THREADS"] == "1"
    assert snapshot["JAX_NUM_THREADS"] == "1"
    assert snapshot["XLA_FLAGS"] == (
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    )

    subprocess_env = _oracle._subprocess_env(thread_mode="single")
    assert subprocess_env["RAYON_NUM_THREADS"] == "1"
    assert subprocess_env["OMP_NUM_THREADS"] == "1"
    assert subprocess_env["OPENBLAS_NUM_THREADS"] == "1"
    assert subprocess_env["MKL_NUM_THREADS"] == "1"
    assert subprocess_env["VECLIB_MAXIMUM_THREADS"] == "1"
    assert subprocess_env["NUMEXPR_NUM_THREADS"] == "1"
    assert subprocess_env["JAX_NUM_THREADS"] == "1"
    assert subprocess_env["XLA_FLAGS"] == (
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    )
    assert "PYTHONPATH" not in subprocess_env


def test_workload_shape_records_multi_axis_large_lanes() -> None:
    workloads = _inputs.lane_workloads()
    ephemeris = workloads["large-n"]["dynamics.generate_ephemeris_2body"]
    photometry = workloads["large-n"]["photometry.predict_magnitudes"]

    assert ephemeris.rows == 20_000
    assert ephemeris.axes() == {"orbits": 400, "epochs": 50}
    assert photometry.axes() == {"orbits": 1000, "observers": 50}
    assert "×" in ephemeris.label()


def test_speed_artifact_records_thread_metadata() -> None:
    result = parity_speed.SpeedResult(
        api_id="coordinates.cartesian_to_spherical",
        n=1,
        rust_p50=1.0,
        rust_p95=1.0,
        legacy_p50=2.0,
        legacy_p95=2.0,
        speedup_p50=2.0,
        speedup_p95=2.0,
        raw_passed=True,
        passed=True,
        timing_trials=3,
        rust_sample_trials_s=[[1.0], [1.0], [1.0]],
        rust_p50_trials_s=[1.0, 1.0, 1.0],
        rust_p95_trials_s=[1.0, 1.0, 1.0],
        legacy_sample_trials_s=[[2.0], [2.0], [2.0]],
        legacy_p50_trials_s=[2.0, 2.0, 2.0],
        legacy_p95_trials_s=[2.0, 2.0, 2.0],
        speedup_p50_trials=[2.0, 2.0, 2.0],
        speedup_p95_trials=[2.0, 2.0, 2.0],
        thread_mode="single",
        thread_env=_threading.SINGLE_THREAD_ENV.copy(),
        cold_thread_mode="native",
        cold_thread_env={key: None for key in _threading.THREAD_ENV_KEYS},
    )

    artifact = parity_speed.to_json([result])

    assert artifact["canonical_speed_trials"] == parity_speed.CANONICAL_SPEED_TRIALS
    assert artifact["timing_aggregation"] == "median-of-trial-percentiles"
    assert artifact["thread_mode"] == "single"
    assert artifact["thread_env"]["RAYON_NUM_THREADS"] == "1"
    assert artifact["cold_thread_mode"] == "native"
    assert artifact["lanes"][0]["name"] == "small-n"
    assert artifact["lanes"][0]["enforced"] is True
    assert artifact["lanes"][0]["timing_trials"] == [3]
    assert "single-trial" in artifact["timing_policy"]
    assert artifact["apis"][0]["rust_sample_trials_s"] == [[1.0], [1.0], [1.0]]
    assert artifact["apis"][0]["legacy_p95_trials_s"] == [2.0, 2.0, 2.0]
    assert "large-n" in artifact["lane_policy"]
    assert "multi-thread" in artifact["thread_policy"]


def test_parity_main_exposes_additive_legacy_cache_refresh_controls() -> None:
    parser = parity_main._build_arg_parser()
    help_text = parser.format_help()
    args = parser.parse_args(
        [
            "--speed-legacy-cache",
            "cache.json",
            "--speed-refresh-legacy-cache",
            "--speed-replace-legacy-cache",
        ]
    )

    assert args.speed_refresh_legacy_cache
    assert args.speed_replace_legacy_cache
    assert "merge" in help_text
    assert "--speed-replace-legacy-cache" in help_text


def test_legacy_relevant_untracked_status_filters_non_code(monkeypatch) -> None:
    def fake_git_output(args: list[str], *, cwd: Path) -> str:
        assert "--untracked-files=all" in args
        return "\n".join(
            [
                "?? .pi/session.json",
                "?? decisions.md",
                "?? src/adam_core/new_module.py",
                "?? adam_core/top_level.py",
                " M src/adam_core/existing.py",
            ]
        )

    monkeypatch.setattr(parity_speed, "_git_output", fake_git_output)

    status = parity_speed._legacy_relevant_untracked_status()

    assert "src/adam_core/new_module.py" in status
    assert "adam_core/top_level.py" in status
    assert ".pi/session.json" not in status
    assert "decisions.md" not in status
    assert "existing.py" not in status


def test_refresh_legacy_cache_merges_existing_entries(monkeypatch, tmp_path) -> None:
    identity = {
        "git_commit": "baseline",
        "process_version": "test",
        "benchmark_source_hash": "new-source",
        "timing_process_hash": "process-source",
        "legacy_packages_hash": "packages",
    }
    monkeypatch.setattr(parity_speed, "_legacy_identity", lambda: identity)
    cache_path = tmp_path / "legacy_cache.json"
    cache_path.write_text("""
        {
          "schema_version": 1,
          "process_version": "rm-p1-020-built-in-speed-trials-v1",
          "created_at": "2026-05-05T00:00:00+00:00",
          "updated_at": "2026-05-05T00:00:00+00:00",
          "legacy_identity": {
            "git_commit": "baseline",
            "process_version": "test",
            "benchmark_source_hash": "old-source",
            "timing_process_hash": "process-source",
            "legacy_packages_hash": "packages"
          },
          "warm": {"existing-warm": {"key_fields": {"kind": "warm"}}},
          "cold": {"existing-cold": {"key_fields": {"kind": "cold"}}}
        }
        """)

    try:
        parity_speed.prepare_legacy_timing_cache(cache_path)
    except ValueError as exc:
        assert "benchmark source hash" in str(exc)
    else:
        raise AssertionError("source-hash drift should fail without refresh")

    untouched_cache = parity_speed.prepare_legacy_timing_cache(cache_path, refresh=True)
    assert untouched_cache is not None
    parity_speed.write_legacy_timing_cache(untouched_cache)
    untouched = __import__("json").loads(cache_path.read_text())
    assert untouched["legacy_identity"]["benchmark_source_hash"] == "old-source"

    cache = parity_speed.prepare_legacy_timing_cache(cache_path, refresh=True)
    assert cache is not None
    parity_speed._write_cache_entry(
        cache,
        "warm",
        "new-warm",
        {"key_fields": {"kind": "warm", "api_id": "new"}, "samples_s": [1.0]},
    )
    parity_speed.write_legacy_timing_cache(cache)

    merged = __import__("json").loads(cache_path.read_text())
    assert merged["legacy_identity"] == identity
    assert set(merged["warm"]) == {"existing-warm", "new-warm"}
    assert set(merged["cold"]) == {"existing-cold"}
    assert merged["warm"]["new-warm"]["legacy_identity"] == identity

    replaced = parity_speed.prepare_legacy_timing_cache(
        cache_path,
        refresh=True,
        replace=True,
    )
    assert replaced is not None
    assert parity_speed._cache_section(replaced, "warm") == {}
    assert parity_speed._cache_section(replaced, "cold") == {}


def test_legacy_cache_entry_identity_allows_source_hash_only_drift() -> None:
    identity = {
        "git_commit": "baseline",
        "process_version": "test",
        "benchmark_source_hash": "new-source",
        "timing_process_hash": "process-source",
        "legacy_packages_hash": "packages",
    }
    fields = {"kind": "warm", "api_id": "api", "process_version": "test"}
    old_source_identity = dict(identity)
    old_source_identity["benchmark_source_hash"] = "old-source"
    context = {
        "data": {
            "legacy_identity": identity,
            "warm": {
                "key": {
                    "key_fields": fields,
                    "samples_s": [1.0],
                    "legacy_identity": old_source_identity,
                }
            },
        },
        "refresh": False,
        "hits": {"warm": 0, "cold": 0},
        "misses": {"warm": 0, "cold": 0},
        "writes": {"warm": 0, "cold": 0},
    }

    assert parity_speed._cached_entry(context, "warm", "key", fields) is not None
    assert context["hits"] == {"warm": 1, "cold": 0}

    del context["data"]["warm"]["key"]["legacy_identity"]
    try:
        parity_speed._cached_entry(context, "warm", "key", fields)
    except ValueError as exc:
        assert "missing legacy_identity" in str(exc)
    else:
        raise AssertionError("missing per-entry identity should fail cache lookup")

    context["data"]["warm"]["key"]["legacy_identity"] = old_source_identity
    stale_identity = dict(old_source_identity)
    stale_identity["legacy_packages_hash"] = "other-packages"
    context["data"]["warm"]["key"]["legacy_identity"] = stale_identity
    try:
        parity_speed._cached_entry(context, "warm", "key", fields)
    except ValueError as exc:
        assert "different legacy checkout" in str(exc)
    else:
        raise AssertionError("non-source identity drift should fail cache lookup")


def test_time_legacy_warm_rejects_stale_entry_identity(monkeypatch) -> None:
    identity = {
        "git_commit": "baseline",
        "process_version": "test",
        "benchmark_source_hash": "source",
        "timing_process_hash": "process-source",
        "legacy_packages_hash": "packages",
    }
    stale_identity = dict(identity)
    stale_identity["legacy_packages_hash"] = "other-packages"
    workload_shape = {"rows": 1, "axes": {}, "label": "rows=1"}
    fields = parity_speed._legacy_cache_fields(
        kind="warm",
        api_id="api",
        lane="small-n",
        workload_shape=workload_shape,
        seed=123,
        thread_mode="single",
        reps=1,
        warmup=0,
    )
    key = parity_speed._hash_json(fields)
    context = {
        "data": {
            "legacy_identity": identity,
            "warm": {
                key: {
                    "key_fields": fields,
                    "samples_s": [1.0],
                    "legacy_identity": stale_identity,
                }
            },
            "cold": {},
        },
        "refresh": False,
        "hits": {"warm": 0, "cold": 0},
        "misses": {"warm": 0, "cold": 0},
        "writes": {"warm": 0, "cold": 0},
    }

    def fail_time_legacy(*args: object, **kwargs: object) -> list[float]:
        raise AssertionError("stale cache lookup should fail before measurement")

    monkeypatch.setattr(_oracle, "time_legacy", fail_time_legacy)

    try:
        parity_speed._time_legacy_warm(
            "api",
            {},
            reps=1,
            warmup=0,
            seed=123,
            thread_mode="single",
            lane="small-n",
            workload_shape=workload_shape,
            workload_label="rows=1",
            legacy_cache=context,
        )
    except ValueError as exc:
        assert "different legacy checkout" in str(exc)
    else:
        raise AssertionError("stale per-entry identity should fail warm lookup")
    assert context["hits"] == {"warm": 0, "cold": 0}


def test_time_legacy_warm_refresh_writes_entry_identity(monkeypatch) -> None:
    identity = {
        "git_commit": "baseline",
        "process_version": "test",
        "benchmark_source_hash": "source",
        "timing_process_hash": "process-source",
        "legacy_packages_hash": "packages",
    }
    workload_shape = {"rows": 1, "axes": {}, "label": "rows=1"}
    context = {
        "data": {"legacy_identity": identity, "warm": {}, "cold": {}},
        "refresh": True,
        "dirty": False,
        "hits": {"warm": 0, "cold": 0},
        "misses": {"warm": 0, "cold": 0},
        "writes": {"warm": 0, "cold": 0},
    }
    samples = [0.1, 0.2, 0.3]

    monkeypatch.setattr(
        _oracle,
        "time_legacy",
        lambda *args, **kwargs: samples,
    )

    measured, source, key = parity_speed._time_legacy_warm(
        "api",
        {},
        reps=3,
        warmup=0,
        seed=123,
        thread_mode="single",
        lane="small-n",
        workload_shape=workload_shape,
        workload_label="rows=1",
        legacy_cache=context,
    )

    assert measured == samples
    assert source == "refreshed"
    entry = context["data"]["warm"][key]
    assert entry["legacy_identity"] == identity
    assert entry["samples_s"] == samples
    assert context["dirty"] is True
    assert context["writes"] == {"warm": 1, "cold": 0}


def test_speed_artifact_records_legacy_cache_metadata() -> None:
    result = parity_speed.SpeedResult(
        api_id="coordinates.cartesian_to_spherical",
        n=1,
        rust_p50=1.0,
        rust_p95=1.0,
        legacy_p50=2.0,
        legacy_p95=2.0,
        speedup_p50=2.0,
        speedup_p95=2.0,
        raw_passed=True,
        passed=True,
        legacy_source="cache",
        legacy_cache_key="warm-key",
    )
    cache_context = {
        "path": Path("migration/artifacts/parity_legacy_speed_baseline.json"),
        "data": {
            "schema_version": parity_speed.LEGACY_TIMING_CACHE_SCHEMA_VERSION,
            "process_version": parity_speed.LEGACY_TIMING_CACHE_PROCESS_VERSION,
            "legacy_identity": {"git_commit": "baseline"},
            "warm": {
                "warm-key": {
                    "captured_at": "2026-05-05T00:00:00+00:00",
                    "legacy_identity": {
                        "git_commit": "baseline",
                        "benchmark_source_hash": "source",
                    },
                }
            },
            "cold": {},
        },
        "refresh": False,
        "dirty": False,
        "hits": {"warm": 1, "cold": 0},
        "misses": {"warm": 0, "cold": 0},
        "writes": {"warm": 0, "cold": 0},
    }

    artifact = parity_speed.to_json([result], legacy_cache=cache_context)

    assert artifact["legacy_timing_cache"]["hits"] == {"warm": 1, "cold": 0}
    assert artifact["legacy_timing_cache"]["entry_freshness"]["warm"] == {
        "entries": 1,
        "captured_at_min": "2026-05-05T00:00:00+00:00",
        "captured_at_max": "2026-05-05T00:00:00+00:00",
        "missing_legacy_identity": 0,
        "distinct_legacy_identities": 1,
        "benchmark_source_hashes": ["source"],
    }
    assert artifact["apis"][0]["legacy_source"] == "cache"
    assert artifact["apis"][0]["legacy_cache_key"] == "warm-key"


def test_speed_artifact_enforces_large_lane_and_records_status() -> None:
    small = parity_speed.SpeedResult(
        api_id="coordinates.cartesian_to_spherical",
        n=2000,
        rust_p50=1.0,
        rust_p95=1.0,
        legacy_p50=2.0,
        legacy_p95=2.0,
        speedup_p50=2.0,
        speedup_p95=2.0,
        raw_passed=True,
        passed=True,
        lane="small-n",
        lane_enforced=True,
        workload_shape={"rows": 2000, "axes": {}, "label": "rows=2000"},
        workload_label="rows=2000",
    )
    large = parity_speed.SpeedResult(
        api_id="coordinates.cartesian_to_spherical",
        n=20_000,
        rust_p50=2.0,
        rust_p95=2.0,
        legacy_p50=2.0,
        legacy_p95=2.0,
        speedup_p50=1.0,
        speedup_p95=1.0,
        raw_passed=False,
        passed=False,
        lane="large-n",
        lane_enforced=True,
        workload_shape={"rows": 20_000, "axes": {}, "label": "rows=20000"},
        workload_label="rows=20000",
        min_speedup_p50=1.0,
        min_speedup_p95=1.0,
    )

    artifact = parity_speed.to_json([small, large])

    assert artifact["all_passed"] is False
    assert [lane["name"] for lane in artifact["lanes"]] == ["small-n", "large-n"]
    assert artifact["lane_status"]["large-n"]["passed"] is False
    assert artifact["apis"][1]["lane"] == "large-n"
    assert artifact["apis"][1]["lane_enforced"] is True
    assert artifact["apis"][1]["workload_shape"]["rows"] == 20_000
