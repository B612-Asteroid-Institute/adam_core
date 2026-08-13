from __future__ import annotations

from pathlib import Path

from adam_core._rust.status import API_MIGRATIONS
from migration.scripts import current_regression


def test_current_regression_fixture_covers_registry_without_legacy_runtime() -> None:
    result = current_regression.check_fixture(current_regression.DEFAULT_FIXTURE)

    assert result.api_count == len(API_MIGRATIONS)
    assert result.output_count == 106
    assert result.exact_output_count == 16


def test_current_regression_runner_has_no_legacy_runtime_dependency() -> None:
    source = Path(current_regression.__file__).read_text()
    help_text = current_regression._build_arg_parser().format_help().lower()

    assert "_legacy_runner" not in source
    assert "_oracle" not in source
    assert "legacy-root" not in help_text
    assert "legacy-python" not in help_text
