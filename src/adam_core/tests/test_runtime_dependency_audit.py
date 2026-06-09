from __future__ import annotations

import ast
import tomllib
from collections.abc import Iterator
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src" / "adam_core"
MIGRATION_ROOT = PROJECT_ROOT / "migration"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
RETIRED_PYTHON_BACKENDS = frozenset(
    {
        "jax",
        "jaxlib",
        "numba",
        "spiceypy",
        "spicekit",
    }
)


def _iter_audited_python_paths() -> Iterator[Path]:
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if "tests" in path.relative_to(SRC_ROOT).parts:
            continue
        yield path

    for path in sorted(MIGRATION_ROOT.rglob("*.py")):
        yield path


def _top_level_requirement_name(requirement: str) -> str:
    split_at = len(requirement)
    for delimiter in "[<>=!~; ":
        index = requirement.find(delimiter)
        if index != -1:
            split_at = min(split_at, index)
    return requirement[:split_at]


def test_package_and_migration_code_do_not_import_retired_python_backends() -> None:
    offenders: list[str] = []

    for path in _iter_audited_python_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported = alias.name.split(".")[0]
                    if imported in RETIRED_PYTHON_BACKENDS:
                        offenders.append(f"{path}: import {alias.name}")

            if isinstance(node, ast.ImportFrom):
                imported = (node.module or "").split(".")[0]
                if imported in RETIRED_PYTHON_BACKENDS:
                    offenders.append(f"{path}: from {node.module} import ...")

    assert offenders == []


def test_project_runtime_dependencies_exclude_retired_python_backends() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    dependency_names = {
        _top_level_requirement_name(requirement)
        for requirement in project.get("dependencies", [])
    }

    assert dependency_names.isdisjoint(RETIRED_PYTHON_BACKENDS)
