"""Preflight PDM scripts and CI references.

This intentionally checks for stale references, not semantic correctness of
every shell command. It catches the class of drift that broke this branch:
PDM/CI entries pointing at deleted tests, deleted benchmark scripts, or retired
live Rust-vs-legacy benchmark flags/artifacts.
"""

from __future__ import annotations

import re
import shlex
import sys
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

STALE_REFERENCES = (
    "src/adam_core/tests/test_rust_orbit_determination.py",
    "src/adam_core/tests/test_rust_parity_randomized.py",
    "src/adam_core/dynamics/tests/test_kepler.py",
    "migration/scripts/rust_orbit_determination_benchmark.py",
    "rust-parity-randomized",
    "rust-od-benchmark",
    "--max-rust-over-legacy",
    "migration/artifacts/rust_benchmark_gate.json",
)

PATH_PREFIXES = (
    "src/",
    "./src/",
    "migration/",
    "./migration/",
    "docs/",
    "./docs/",
    "rust/",
    "./rust/",
)

OUTPUT_PREFIXES = (
    "migration/artifacts/",
    "./migration/artifacts/",
    "docs/build",
    "./docs/build",
    "dist/",
    "./dist/",
)


def _load_scripts() -> dict[str, Any]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    scripts = data.get("tool", {}).get("pdm", {}).get("scripts", {})
    if not isinstance(scripts, dict):
        raise SystemExit("[tool.pdm.scripts] is missing or not a table")
    return scripts


def _script_commands(script: Any) -> list[str]:
    if isinstance(script, str):
        return [script]
    if not isinstance(script, dict):
        return []
    if isinstance(script.get("cmd"), str):
        return [script["cmd"]]
    composite = script.get("composite")
    if isinstance(composite, list):
        return [cmd for cmd in composite if isinstance(cmd, str)]
    return []


def _all_script_commands(scripts: dict[str, Any]) -> list[tuple[str, str]]:
    commands: list[tuple[str, str]] = []
    for name, script in scripts.items():
        commands.extend((name, cmd) for cmd in _script_commands(script))
    return commands


def _workflow_texts() -> list[tuple[Path, str]]:
    if not WORKFLOW_DIR.exists():
        return []
    return [
        (path, path.read_text(encoding="utf-8"))
        for path in sorted(WORKFLOW_DIR.glob("*.yml"))
    ] + [
        (path, path.read_text(encoding="utf-8"))
        for path in sorted(WORKFLOW_DIR.glob("*.yaml"))
    ]


def _tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        # Fall back to whitespace splitting so a malformed command still gets
        # checked for obvious stale paths before the shell reports the syntax.
        return command.split()


def _normalize_path_token(token: str) -> str:
    return token.strip().rstrip(",:;")


def _looks_like_repo_path(token: str) -> bool:
    token = _normalize_path_token(token)
    if not token.startswith(PATH_PREFIXES):
        return False
    if token.startswith(OUTPUT_PREFIXES):
        return False
    return "{" not in token and "}" not in token


def _missing_paths(source: str, commands: Iterable[str]) -> list[str]:
    failures: list[str] = []
    for command in commands:
        for token in _tokens(command):
            token = _normalize_path_token(token)
            if not _looks_like_repo_path(token):
                continue
            path = REPO_ROOT / token.removeprefix("./")
            if not path.exists():
                failures.append(f"{source}: missing referenced path `{token}`")
    return failures


def _stale_reference_failures(source: str, text: str) -> list[str]:
    return [
        f"{source}: stale reference `{stale}`"
        for stale in STALE_REFERENCES
        if stale in text
    ]


def _workflow_pdm_run_failures(
    source: str, text: str, script_names: set[str]
) -> list[str]:
    failures: list[str] = []
    for match in re.finditer(r"\bpdm\s+run\s+([A-Za-z0-9_.:-]+)", text):
        script = match.group(1)
        if script not in script_names:
            failures.append(f"{source}: `pdm run {script}` has no matching PDM script")
    return failures


def main() -> None:
    scripts = _load_scripts()
    script_names = set(scripts)
    script_commands = _all_script_commands(scripts)

    failures: list[str] = []
    pyproject_text = PYPROJECT.read_text(encoding="utf-8")
    failures.extend(_stale_reference_failures("pyproject.toml", pyproject_text))
    failures.extend(
        _workflow_pdm_run_failures("pyproject.toml", pyproject_text, script_names)
    )
    failures.extend(
        _missing_paths("pyproject.toml", (cmd for _, cmd in script_commands))
    )

    for path, text in _workflow_texts():
        source = str(path.relative_to(REPO_ROOT))
        failures.extend(_stale_reference_failures(source, text))
        failures.extend(_workflow_pdm_run_failures(source, text, script_names))

    if failures:
        print("PDM/CI script preflight failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        raise SystemExit(1)

    print(
        f"PDM/CI script preflight passed "
        f"({len(script_names)} PDM scripts, {len(_workflow_texts())} workflows)."
    )


if __name__ == "__main__":
    main()
