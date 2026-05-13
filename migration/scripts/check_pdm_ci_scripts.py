"""Preflight PDM scripts and CI references.

This intentionally checks for stale references, not semantic correctness of
every shell command. It catches the class of drift that broke this branch:
PDM/CI entries pointing at deleted tests, deleted benchmark scripts, or retired
live Rust-vs-legacy benchmark flags/artifacts.
"""

from __future__ import annotations

import re
import shlex
import subprocess
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
    "--trials",
    "migration/artifacts/rust_benchmark_gate.json",
    "migration/artifacts/ephemeris_wide_observer_bench.json",
    "migration/artifacts/rust_orbit_determination_benchmark.json",
)

REQUIRED_SCRIPT_OUTPUTS = {
    "rust-parity-main": "migration/artifacts/parity_gate.json",
    "rust-parity-speed-cold": "migration/artifacts/parity_speed_cold_warm.json",
    "rust-latency-gate": "migration/artifacts/rust_latency_current.json",
}

LEGACY_SPEED_CACHE = "migration/artifacts/parity_legacy_speed_baseline.json"

REQUIRED_WORKFLOW_ARTIFACTS = {
    "rust-latency-current": "migration/artifacts/rust_latency_current.json",
}

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
            elif not _has_tracked_entries(path):
                failures.append(
                    f"{source}: referenced path `{token}` exists locally but has no "
                    "tracked git entries"
                )
    return failures


def _has_tracked_entries(path: Path) -> bool:
    relative_path = path.relative_to(REPO_ROOT).as_posix()
    result = subprocess.run(
        ["git", "ls-files", "--", relative_path],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or "git ls-files failed")
    return bool(result.stdout.strip())


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


def _option_value(tokens: list[str], option: str) -> str | None:
    prefix = option + "="
    for idx, token in enumerate(tokens):
        if token == option and idx + 1 < len(tokens):
            return tokens[idx + 1]
        if token.startswith(prefix):
            return token.split("=", 1)[1]
    return None


def _int_option(tokens: list[str], option: str) -> int | None:
    value = _option_value(tokens, option)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _speed_script_policy_failures(scripts: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    main_tokens = _tokens(" ".join(_script_commands(scripts.get("rust-parity-main"))))
    if "--speed-large" not in main_tokens:
        failures.append("pyproject.toml: `rust-parity-main` must include large-n lane")
    else:
        if "--speed-large-diagnostic" in main_tokens:
            failures.append("pyproject.toml: `rust-parity-main` must enforce large-n")
        large_reps = _int_option(main_tokens, "--speed-large-reps")
        if large_reps is None or large_reps < 5:
            failures.append(
                "pyproject.toml: `rust-parity-main` large-n lane must use >=5 reps"
            )
    if "--speed-tiny" not in main_tokens:
        failures.append("pyproject.toml: `rust-parity-main` must include tiny-n lane")
    tiny_reps = _int_option(main_tokens, "--speed-tiny-reps")
    if tiny_reps is None or tiny_reps < 51:
        failures.append(
            "pyproject.toml: `rust-parity-main` tiny-n lane must use >=51 reps"
        )
    if _option_value(main_tokens, "--speed-legacy-cache") != LEGACY_SPEED_CACHE:
        failures.append(
            "pyproject.toml: `rust-parity-main` must use the legacy speed cache"
        )

    refresh_tokens = _tokens(
        " ".join(_script_commands(scripts.get("rust-parity-legacy-cache-refresh")))
    )
    if not refresh_tokens:
        failures.append(
            "pyproject.toml: `rust-parity-legacy-cache-refresh` script is required"
        )
    elif _option_value(refresh_tokens, "--legacy-cache") != LEGACY_SPEED_CACHE:
        failures.append(
            "pyproject.toml: `rust-parity-legacy-cache-refresh` must write the legacy speed cache"
        )
    elif "--refresh-legacy-cache" not in refresh_tokens:
        failures.append(
            "pyproject.toml: `rust-parity-legacy-cache-refresh` must refresh the cache"
        )
    refresh_tiny_reps = _int_option(refresh_tokens, "--tiny-reps")
    if refresh_tiny_reps is None or refresh_tiny_reps < 51:
        failures.append(
            "pyproject.toml: `rust-parity-legacy-cache-refresh` tiny-n lane must use >=51 reps"
        )

    cold_tokens = _tokens(
        " ".join(_script_commands(scripts.get("rust-parity-speed-cold")))
    )
    if "--large" not in cold_tokens:
        failures.append(
            "pyproject.toml: `rust-parity-speed-cold` must include large-n lane"
        )
    else:
        if "--large-cold" not in cold_tokens:
            failures.append(
                "pyproject.toml: `rust-parity-speed-cold` must collect large-n cold timing"
            )
        large_reps = _int_option(cold_tokens, "--large-reps")
        if large_reps is None or large_reps < 5:
            failures.append(
                "pyproject.toml: `rust-parity-speed-cold` large-n lane must use >=5 reps"
            )
    if "--tiny" not in cold_tokens:
        failures.append(
            "pyproject.toml: `rust-parity-speed-cold` must include tiny-n lane"
        )
    cold_tiny_reps = _int_option(cold_tokens, "--tiny-reps")
    if cold_tiny_reps is None or cold_tiny_reps < 51:
        failures.append(
            "pyproject.toml: `rust-parity-speed-cold` tiny-n lane must use >=51 reps"
        )
    if _option_value(cold_tokens, "--legacy-cache") != LEGACY_SPEED_CACHE:
        failures.append(
            "pyproject.toml: `rust-parity-speed-cold` must use the legacy speed cache"
        )
    return failures


def _required_script_output_failures(scripts: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for script, expected_output in REQUIRED_SCRIPT_OUTPUTS.items():
        commands = _script_commands(scripts.get(script))
        if not commands:
            failures.append(f"pyproject.toml: required PDM script `{script}` missing")
            continue
        command_text = " ".join(commands)
        if expected_output not in command_text:
            failures.append(
                f"pyproject.toml: `{script}` must write `{expected_output}`"
            )
    failures.extend(_speed_script_policy_failures(scripts))
    return failures


def _required_workflow_artifact_failures(
    workflows: list[tuple[Path, str]],
) -> list[str]:
    workflow_text = "\n".join(text for _, text in workflows)
    failures: list[str] = []
    for artifact_name, expected_path in REQUIRED_WORKFLOW_ARTIFACTS.items():
        if f"name: {artifact_name}" not in workflow_text:
            failures.append(f"workflows: artifact `{artifact_name}` is not uploaded")
        if f"path: {expected_path}" not in workflow_text:
            failures.append(
                f"workflows: artifact `{artifact_name}` must upload `{expected_path}`"
            )
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
    failures.extend(_required_script_output_failures(scripts))

    workflows = _workflow_texts()
    for path, text in workflows:
        source = str(path.relative_to(REPO_ROOT))
        failures.extend(_stale_reference_failures(source, text))
        failures.extend(_workflow_pdm_run_failures(source, text, script_names))
    failures.extend(_required_workflow_artifact_failures(workflows))

    if failures:
        print("PDM/CI script preflight failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        raise SystemExit(1)

    print(
        f"PDM/CI script preflight passed "
        f"({len(script_names)} PDM scripts, {len(workflows)} workflows)."
    )


if __name__ == "__main__":
    main()
