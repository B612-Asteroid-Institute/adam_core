from adam_core._rust.status import (
    API_MIGRATIONS,
    API_MIGRATIONS_BY_ID,
    validate_api_migrations,
)
from migration.parity import _inputs, tolerances


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


def test_gauss_iod_randomized_exclusion_is_visible() -> None:
    migration = API_MIGRATIONS_BY_ID["orbit_determination.gaussIOD"]

    assert migration.parity_coverage == "random-fuzz-excluded"
    assert migration.coverage_note
    assert migration.excluded_subcases


def test_transform_coordinates_partial_coverage_is_visible() -> None:
    migration = API_MIGRATIONS_BY_ID["coordinates.transform_coordinates"]

    assert migration.parity_coverage == "random-fuzz"
    assert migration.covered_subcases
    assert migration.excluded_subcases
    assert "public quivr-object dispatcher" in migration.coverage_note
    assert any("CartesianCoordinates" in case for case in migration.covered_subcases)
    assert any("Cartesian->Cartesian" in case for case in migration.excluded_subcases)


def test_latency_gate_registry_matches_latency_benchmark_scope() -> None:
    latency_ids = {
        migration.api_id for migration in API_MIGRATIONS if migration.latency_gate
    }

    assert "dynamics.add_light_time" not in latency_ids
    assert "orbit_determination.gaussIOD" in latency_ids
    assert "dynamics.calculate_perturber_moids" not in latency_ids
    assert "dynamics.generate_porkchop_data" not in latency_ids
