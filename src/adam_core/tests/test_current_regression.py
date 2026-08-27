from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from adam_core._rust.status import API_MIGRATIONS
from migration.scripts import current_regression


def test_current_regression_fixture_covers_registry_without_legacy_runtime() -> None:
    result = current_regression.check_fixture(current_regression.DEFAULT_FIXTURE)

    assert result.api_count == len(API_MIGRATIONS) == 44
    assert result.case_count == 357
    assert result.fuzz_case_count == 352
    assert result.fixed_fixture_count == 5
    assert result.output_count == 855
    assert result.exact_output_count == 128


def test_current_regression_runner_has_no_legacy_runtime_dependency() -> None:
    source = Path(current_regression.__file__).read_text()
    help_text = current_regression._build_arg_parser().format_help().lower()

    assert "_legacy_runner" not in source
    assert "_oracle" not in source
    assert "legacy-root" not in help_text
    assert "legacy-python" not in help_text
    assert "migration.parity._oracle" not in sys.modules
    assert "migration.parity._legacy_runner" not in sys.modules
    assert "migration.parity._timing_cache" not in sys.modules


def test_current_regression_fixture_contains_every_reviewed_case() -> None:
    fixture = current_regression._read_fixture(current_regression.DEFAULT_FIXTURE)
    cases = [case for api_cases in fixture["apis"].values() for case in api_cases]

    assert [case["seed"] for case in cases if case["kind"] == "fuzz"].count(
        current_regression.DEFAULT_BASE_SEED
    ) == len(API_MIGRATIONS)
    assert tuple(fixture["fixed_fixture_names"]) == (
        current_regression.EXPECTED_FIXED_FIXTURES
    )

    metadata_path = current_regression.DEFAULT_FIXTURE.with_name(
        "current_regression_core.meta.json"
    )
    metadata = json.loads(metadata_path.read_text())
    assert hashlib.sha256(
        current_regression.DEFAULT_FIXTURE.read_bytes()
    ).hexdigest() == (metadata["fixture_sha256"])
    assert metadata["coverage"]["total_cases"] == 357
