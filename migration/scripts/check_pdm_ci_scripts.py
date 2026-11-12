"""Preflight PDM scripts and CI references.

This intentionally checks for stale references, not semantic correctness of
every shell command. It catches the class of drift that broke this branch:
PDM/CI entries pointing at deleted tests, deleted benchmark scripts, or retired
live Rust-vs-legacy benchmark flags/artifacts.
"""

from __future__ import annotations

import json
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
UV_LOCK = REPO_ROOT / "uv.lock"
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
RELEASE_WORKFLOW = WORKFLOW_DIR / "release-candidate-wheel-matrix.yml"
TIER1_WORKFLOW = WORKFLOW_DIR / "tier1-dependent-smoke.yml"
RUST_CRATE_WORKFLOW = WORKFLOW_DIR / "rust-crate-release-candidate.yml"
EXPECTED_BUILD_TOOLS = [
    "maturin==1.15.0",
    "packaging==25.0",
    "ziglang==0.17.0",
]
EXPECTED_MANYLINUX_TARGETS = [
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
]
PDM_WHEEL_SCRIPTS = {
    "wheel-build",
    "rust-build",
    "wheel-inspect",
    "release-build-environment-check",
}

STALE_REFERENCES = (
    "src/adam_core/tests/test_rust_orbit_determination.py",
    "src/adam_core/tests/test_rust_parity_randomized.py",
    "src/adam_core/dynamics/tests/test_kepler.py",
    "migration/scripts/rust_orbit_determination_benchmark.py",
    "rust-parity-randomized",
    "rust-od-benchmark",
    "--max-rust-over-legacy",
    "migration/artifacts/rust_benchmark_gate.json",
    "migration/artifacts/ephemeris_wide_observer_bench.json",
    "migration/artifacts/rust_orbit_determination_benchmark.json",
)

REQUIRED_SCRIPT_OUTPUTS = {
    "rust-parity-main": "migration/artifacts/parity_gate.json",
    "rust-parity-speed-cold": "migration/artifacts/parity_speed_cold_warm.json",
    "rust-latency-gate": "migration/artifacts/rust_latency_current.json",
    "rust-latency-gate-ci": "migration/artifacts/rust_latency_current.json",
    "benchmark-current-ci": "migration/artifacts/benchmark_current_core_ci.json",
}

LEGACY_SPEED_CACHE = "migration/artifacts/parity_legacy_speed_baseline.json"

REQUIRED_WORKFLOW_ARTIFACTS = {
    "rust-latency-current": "migration/artifacts/rust_latency_current.json",
    "current-only-benchmark": "migration/artifacts/benchmark_current_core_ci.json",
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


def _load_project() -> dict[str, Any]:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _load_scripts(project: dict[str, Any]) -> dict[str, Any]:
    scripts = project.get("tool", {}).get("pdm", {}).get("scripts", {})
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
        if script == "python":
            # `pdm run python` deliberately selects the installed project
            # interpreter; unlike other names it is an executable, not a script.
            continue
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
    regression_commands = _script_commands(scripts.get("test-current-regression"))
    if not regression_commands:
        failures.append(
            "pyproject.toml: required PDM script `test-current-regression` missing"
        )
    else:
        import_smoke = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import migration.scripts.current_regression; "
                "assert 'migration.parity._oracle' not in sys.modules; "
                "assert 'migration.parity._legacy_runner' not in sys.modules",
            ],
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if import_smoke.returncode != 0:
            failures.append(
                "pyproject.toml: `test-current-regression` import must not load "
                "legacy oracle modules"
            )
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


def _job_prefix(text: str, offset: int) -> str:
    starts = [
        match.start()
        for match in re.finditer(r"(?m)^  [A-Za-z0-9_-]+:\s*$", text[:offset])
    ]
    return text[starts[-1] : offset] if starts else text[:offset]


def _selects_build_group(command: str) -> bool:
    tokens = _tokens(command)
    return any(
        (
            token in {"-G", "--group"}
            and index + 1 < len(tokens)
            and tokens[index + 1] == "build"
        )
        or token in {"-Gbuild", "--group=build"}
        for index, token in enumerate(tokens)
    )


def _workflow_build_dependency_failures(source: str, text: str) -> list[str]:
    failures: list[str] = []
    for match in re.finditer(
        r"\bpdm\s+run\s+(wheel-build|rust-build|wheel-inspect|release-build-environment-check)\b",
        text,
    ):
        prefix = _job_prefix(text, match.start())
        installs = re.findall(r"(?m)^\s*run:\s*(pdm install[^\n]*)$", prefix)
        if not installs or not any(
            _selects_build_group(command) for command in installs
        ):
            failures.append(
                f"{source}: `pdm run {match.group(1)}` requires a preceding "
                "same-job `pdm install ... -G build`"
            )

    direct = re.compile(
        r"python migration/scripts/(?:build_release_wheel|check_wheel_artifacts)\.py"
    )
    for match in direct.finditer(text):
        prefix = _job_prefix(text, match.start())
        if '"packaging==25.0"' not in prefix:
            failures.append(
                f"{source}: direct builder/checker invocation requires a preceding "
                'same-job exact "packaging==25.0" install'
            )

    for line in text.splitlines():
        if "pip install" in line and "packaging" in line:
            if '"packaging==25.0"' not in line:
                failures.append(
                    f"{source}: packaging installs must pin exactly "
                    '"packaging==25.0"'
                )
    return failures


def _uv_lock_failures(project: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    uv = project.get("tool", {}).get("uv", {})
    if "override-dependencies" in uv:
        failures.append(
            "pyproject.toml: self override prevents dependency groups from being locked"
        )
    if uv.get("sources", {}).get("adam-core") != {"workspace": True}:
        failures.append(
            "pyproject.toml: adam-core must retain its authoritative workspace source"
        )
    if not UV_LOCK.is_file():
        return failures + ["uv.lock is missing"]

    lock = tomllib.loads(UV_LOCK.read_text(encoding="utf-8"))
    packages = lock.get("package", [])
    by_name = {package.get("name"): package for package in packages}
    root = by_name.get("adam-core")
    if not isinstance(root, dict) or root.get("source") != {"editable": "."}:
        failures.append("uv.lock: adam-core must resolve to the editable checkout")
    for name, version in (
        ("maturin", "1.15.0"),
        ("packaging", "25.0"),
        ("ziglang", "0.17.0"),
    ):
        package = by_name.get(name)
        if not isinstance(package, dict):
            failures.append(f"uv.lock: missing locked build tool {name}=={version}")
            continue
        if package.get("version") != version or set(package.get("source", {})) != {
            "registry"
        }:
            failures.append(
                f"uv.lock: {name} must be registry-locked at exact version {version}"
            )
        artifacts = ([package["sdist"]] if "sdist" in package else []) + package.get(
            "wheels", []
        )
        if not artifacts or not all(
            str(artifact.get("hash", "")).startswith("sha256:")
            for artifact in artifacts
        ):
            failures.append(f"uv.lock: {name} must include registry artifact hashes")
    nonregistry = [
        package.get("name")
        for package in packages
        if package.get("name") != "adam-core"
        and set(package.get("source", {})) != {"registry"}
    ]
    if nonregistry:
        failures.append(
            "uv.lock: non-root packages must be registry-only: "
            + ", ".join(sorted(str(name) for name in nonregistry))
        )
    mpcq = by_name.get("mpcq", {})
    if not any(
        dependency.get("name") == "adam-core"
        for dependency in mpcq.get("dependencies", [])
    ):
        failures.append(
            "uv.lock: test-only mpcq must resolve its adam-core dependency to the root"
        )
    return failures


def _release_matrix_cell(text: str, cell: str) -> tuple[str, ...] | None:
    marker = f"          - name: {cell}\n"
    if text.count(marker) != 1:
        return None
    tail = text.split(marker, 1)[1]
    boundaries = [
        offset
        for boundary in ("\n          - name:", "\n    steps:")
        if (offset := tail.find(boundary)) >= 0
    ]
    if not boundaries:
        return None
    block = tail[: min(boundaries)]
    return tuple(line.strip() for line in block.splitlines() if line.strip())


def _release_matrix_contract_failures(artifact_acceptance: str) -> list[str]:
    failures: list[str] = []
    matrix_marker = "\n      matrix:\n"
    steps_marker = "\n    steps:\n"
    if (
        artifact_acceptance.count(matrix_marker) != 1
        or artifact_acceptance.count(steps_marker) != 1
    ):
        return ["artifact acceptance must define one matrix and one steps list"]

    matrix = artifact_acceptance.split(matrix_marker, 1)[1].split(steps_marker, 1)[0]
    axes = re.findall(r"^        ([A-Za-z0-9_-]+):", matrix, flags=re.MULTILINE)
    if axes != ["python-version", "platform"]:
        failures.append(
            "artifact acceptance matrix must contain only python-version and "
            f"platform axes, found {axes!r}"
        )

    version_matches = re.findall(
        r"^        python-version:\s*(\[.*\])\s*$", matrix, flags=re.MULTILINE
    )
    versions: object = None
    if len(version_matches) == 1:
        try:
            versions = json.loads(version_matches[0])
        except json.JSONDecodeError:
            pass
    expected_versions = ["3.11", "3.12", "3.13", "3.14"]
    if versions != expected_versions:
        failures.append(
            "artifact acceptance matrix must contain exactly CPython "
            f"{expected_versions!r}, found {versions!r}"
        )

    platforms = re.findall(r"^          - name:\s*(\S+)\s*$", matrix, re.MULTILINE)
    expected_platforms = {
        "manylinux-x86_64",
        "manylinux-aarch64",
        "macos-arm64",
        "macos-x86_64",
    }
    if len(platforms) != 4 or set(platforms) != expected_platforms:
        failures.append(
            "artifact acceptance matrix must contain exactly four supported "
            f"platforms, found {platforms!r}"
        )
    return failures


def _release_artifact_steps(artifact_acceptance: str) -> list[dict[str, str]]:
    """Read top-level step fields from the canonical release workflow shape."""

    steps_marker = "\n    steps:\n"
    if artifact_acceptance.count(steps_marker) != 1:
        return []
    steps_source = artifact_acceptance.split(steps_marker, 1)[1]
    steps: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    active_field: str | None = None
    key_pattern = r"(['\"]?)([A-Za-z0-9_-]+)\1\s*:\s*(.*)$"
    for line in steps_source.splitlines():
        step_match = re.match(r"^      - " + key_pattern, line)
        if step_match:
            if current is not None:
                steps.append(current)
            current = {step_match.group(2): step_match.group(3)}
            active_field = step_match.group(2)
            continue
        if current is None:
            continue
        field_match = re.match(r"^        " + key_pattern, line)
        if field_match:
            active_field = field_match.group(2)
            current[active_field] = field_match.group(3)
            continue
        if active_field is not None and (line.startswith("          ") or not line):
            current[active_field] += "\n" + line.strip()
    if current is not None:
        steps.append(current)
    return steps


def _release_workflow_failures(project: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    build_tools = project.get("dependency-groups", {}).get("build")
    if build_tools != EXPECTED_BUILD_TOOLS:
        failures.append(
            "pyproject.toml: build dependency group must equal "
            f"{EXPECTED_BUILD_TOOLS!r}"
        )
    if not RELEASE_WORKFLOW.is_file():
        return failures + ["release candidate wheel workflow is missing"]

    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    header = text.split("\njobs:", 1)[0]
    expected_release_env = (
        '  ADAM_CORE_RELEASE_BUILD: "1"',
        "  ADAM_CORE_BUILD_COMMIT: ${{ github.sha }}",
    )
    for declaration in expected_release_env:
        if header.count(declaration) != 1:
            failures.append(
                "release workflow must declare exact artifact-build environment "
                f"{declaration!r} once before jobs"
            )

    for forbidden in (
        "adam_assist_ref:",
        "ADAM_ASSIST_REF",
        "Checkout adam-assist candidate",
        "Build adam-assist",
        "Prepare compatible ASSIST lock",
        "pairing-cargo-config.toml",
    ):
        if forbidden in text:
            failures.append(
                "Core release acceptance must not own downstream Assist work: "
                f"found {forbidden!r}"
            )
    artifact_marker = "\n  artifact-acceptance:\n"
    extended_marker = "\n  extended-coverage:\n"
    if text.count(artifact_marker) != 1 or text.count(extended_marker) != 1:
        failures.append(
            "release workflow must contain one artifact-acceptance and one "
            "extended-coverage job"
        )
    else:
        artifact_acceptance = text.split(artifact_marker, 1)[1].split(
            extended_marker, 1
        )[0]
        failures.extend(_release_matrix_contract_failures(artifact_acceptance))
        steps = _release_artifact_steps(artifact_acceptance)
        python_condition = re.compile(
            r"matrix\s*(?:\.\s*python-version|\[\s*['\"]python-version['\"]\s*\])"
        )
        for step in steps:
            condition = step.get("if", "")
            if python_condition.search(condition):
                failures.append(
                    "artifact acceptance step "
                    f"{step.get('name', '<unnamed>')!r} must not use "
                    "matrix.python-version in an if condition"
                )

        inline_python = re.compile(
            r"(?<![A-Za-z0-9_])python(?:\d+(?:\.\d+)*)?\s+"
            r"(?:-(?=\s|$)|-c(?=\s|$|['\"]))"
        )
        for step in steps:
            run = step.get("run", "")
            if "<<" in run or inline_python.search(run):
                failures.append(
                    "artifact acceptance run blocks must not embed shell heredocs "
                    "or inline Python programs"
                )
                break

        core_steps = [
            step for step in steps if step.get("name") == "Run Core wheel acceptance"
        ]
        if len(core_steps) != 1:
            failures.append(
                "artifact acceptance must contain exactly one unconditional "
                "Core wheel acceptance step"
            )
        else:
            core_step = core_steps[0]
            if "if" in core_step:
                failures.append("Core wheel acceptance step must be unconditional")
            normalized_run = " ".join(core_step.get("run", "").split())
            required_arguments = (
                "python migration/scripts/run_core_wheel_acceptance.py",
                '--python "$(command -v python)"',
                '--wheelhouse "$GITHUB_WORKSPACE/wheelhouse"',
                '--workspace "$RUNNER_TEMP/acceptance"',
                '--report "$RUNNER_TEMP/acceptance-report.json"',
                '--expected-python "${{ matrix.python-version }}"',
                '--expected-version "$PYTHON_PREVIEW_VERSION"',
                '--expected-commit "$GITHUB_SHA"',
                '--expected-system "${{ matrix.platform.system }}"',
                '--expected-wheel-tag "${{ matrix.platform.wheel_tag }}"',
            )
            for argument in required_arguments:
                if normalized_run.count(argument) != 1:
                    failures.append(
                        "Core wheel acceptance must invoke the repository runner "
                        f"with {argument!r} exactly once"
                    )

        inspector_steps = [
            step
            for step in steps
            if step.get("name") == "Inspect adam-core wheel contents"
        ]
        if len(inspector_steps) != 1:
            failures.append(
                "artifact acceptance must contain one wheel-content inspection step"
            )
        else:
            inspector_step = inspector_steps[0]
            if "if" in inspector_step:
                failures.append("wheel-content inspection step must be unconditional")
            normalized_run = " ".join(inspector_step.get("run", "").split())
            if (
                "python migration/scripts/check_wheel_artifacts.py"
                not in normalized_run
            ):
                failures.append(
                    "wheel-content inspection must call check_wheel_artifacts.py"
                )

        extended_coverage = text.split(extended_marker, 1)[1]
        extended_header, separator, _ = extended_coverage.partition("\n    steps:\n")
        if not separator:
            failures.append("extended coverage must retain its steps")
        if "continue-on-error:" in artifact_acceptance:
            failures.append("artifact acceptance must remain hard gating")
        if (
            text.count("continue-on-error:") != 1
            or extended_header.count("    continue-on-error: true") != 1
        ):
            failures.append("only extended coverage may be non-gating at the job level")
        if extended_header.count("    needs: artifact-acceptance") != 1:
            failures.append("extended coverage must depend on artifact acceptance")

        safe_install = (
            "env -u ADAM_CORE_RELEASE_BUILD -u ADAM_CORE_BUILD_COMMIT "
            "pdm install -G test"
        )
        normalized_extended = " ".join(extended_coverage.split())
        if normalized_extended.count(safe_install) != 1:
            failures.append(
                "extended coverage must remove both release-identity variables "
                "from its ordinary editable test install"
            )
        required_extended_checks = (
            "cargo test -p adam_core_rs_kernel_data -- --ignored",
            "pdm run test-profile -- src/adam_core",
            'ADAM_CORE_LIVE_HORIZONS: "1"',
            'ADAM_CORE_LIVE_QUERY_CLIENTS: "1"',
            "pdm run test-live-integrations --",
        )
        for check in required_extended_checks:
            if extended_coverage.count(check) != 1:
                failures.append(
                    f"extended coverage must retain slow/live check {check!r}"
                )

    install_marker = "Install audited adam-core wheel builder"
    builder = "python migration/scripts/build_release_wheel.py"
    inspector = "python migration/scripts/check_wheel_artifacts.py"
    if not all(marker in text for marker in (install_marker, builder, inspector)):
        failures.append("release workflow is missing install/build/inspect steps")
    else:
        install_offset = text.index(install_marker)
        builder_offset = text.index(builder)
        inspector_offset = text.index(inspector)
        if not install_offset < builder_offset < inspector_offset:
            failures.append(
                "release workflow must install exact build tools before build/inspect"
            )
        for tool in EXPECTED_BUILD_TOOLS:
            pin = f'"{tool}"'
            if text.count(pin) != 1:
                failures.append(
                    f"release workflow must install exact build tool {tool!r} once"
                )
            elif not install_offset < text.index(pin) < builder_offset:
                failures.append(
                    f"release workflow must install {tool!r} before its first build"
                )
    for target in EXPECTED_MANYLINUX_TARGETS:
        if f"target: {target}" not in text:
            failures.append(f"release workflow is missing exact Rust target {target}")
    expected_cells = {
        "manylinux-x86_64": (
            "runner: ubuntu-latest",
            "system: Linux-x86_64",
            "target: x86_64-unknown-linux-gnu",
            "wheel_tag: manylinux*x86_64",
        ),
        "manylinux-aarch64": (
            "runner: ubuntu-24.04-arm",
            "system: Linux-aarch64",
            "target: aarch64-unknown-linux-gnu",
            "wheel_tag: manylinux*aarch64",
        ),
        "macos-arm64": (
            "runner: macos-15",
            "system: Darwin-arm64",
            "family: native",
            "wheel_tag: macosx*arm64",
        ),
        "macos-x86_64": (
            "runner: macos-15-intel",
            "system: Darwin-x86_64",
            "family: native",
            "wheel_tag: macosx*x86_64",
        ),
    }
    exact_cells = {"macos-arm64", "macos-x86_64"}
    for cell, entries in expected_cells.items():
        actual = _release_matrix_cell(text, cell)
        if actual is None:
            failures.append(f"release workflow is missing matrix cell {cell}")
            continue
        if cell in exact_cells:
            if actual != entries:
                failures.append(
                    f"release workflow matrix cell {cell} must equal {entries!r}, "
                    f"found {actual!r}"
                )
            continue
        for entry in entries:
            if entry not in actual:
                failures.append(
                    f"release workflow matrix cell {cell} must contain {entry!r}"
                )
    for shorthand in ("target: x86_64\n", "target: aarch64\n"):
        if shorthand in text:
            failures.append(
                "release workflow must not pass maturin-action target shorthands "
                "to the maturin CLI"
            )
    return failures


def _rust_msrv_dependency_failures() -> list[str]:
    failures: list[str] = []
    coords_manifest = tomllib.loads(
        (REPO_ROOT / "rust/adam_core_rs_coords/Cargo.toml").read_text(encoding="utf-8")
    )
    spice_manifest = tomllib.loads(
        (REPO_ROOT / "rust/adam_core_rs_spice/Cargo.toml").read_text(encoding="utf-8")
    )
    if "faer" in coords_manifest.get("dependencies", {}):
        failures.append("coords manifest must not retain its unused faer dependency")
    expected_faer = {
        "version": "0.22",
        "default-features": False,
        "features": ["std", "linalg", "rayon"],
    }
    if spice_manifest.get("dependencies", {}).get("faer") != expected_faer:
        failures.append(
            "spice manifest faer dependency must use std+linalg+rayon without defaults"
        )

    lock = tomllib.loads((REPO_ROOT / "Cargo.lock").read_text(encoding="utf-8"))
    resolved = {package.get("name") for package in lock.get("package", [])}
    forbidden = {"ar_archive_writer", "npyz", "pest_stack_guard", "psm"}
    leaked = sorted(forbidden & resolved)
    if leaked:
        failures.append(
            "Cargo.lock retains incompatible optional faer chain: " + ", ".join(leaked)
        )

    workflow = RUST_CRATE_WORKFLOW.read_text(encoding="utf-8")
    for required in (
        'faer feature "linalg"',
        'faer feature "rayon"',
        'faer feature "std"',
        'faer feature "(default|npy|rand|sparse|sparse-linalg)"',
        '"ar_archive_writer"',
        '"npyz"',
        '"pest_stack_guard"',
        '"psm"',
    ):
        if required not in workflow:
            failures.append(
                f"Rust clean-consumer workflow must assert dependency contract {required!r}"
            )
    return failures


def _tier1_exact_ref_failures() -> list[str]:
    if not TIER1_WORKFLOW.is_file():
        return ["Tier-1 exact-ref workflow is missing"]
    text = TIER1_WORKFLOW.read_text(encoding="utf-8")
    trigger = text.split("permissions:", 1)[0]
    failures: list[str] = []
    if "pull_request:" in trigger or "schedule:" in trigger:
        failures.append(
            "Tier-1 compatibility must be manual-only until compatible Assist exists"
        )
    for required in (
        "workflow_dispatch:",
        "adam_assist_ref:",
        "required: true",
        "ADAM_ASSIST_REF: ${{ inputs.adam_assist_ref }}",
        "ref: ${{ env.ADAM_ASSIST_REF }}",
        '[[ "$ADAM_ASSIST_REF" =~ ^[0-9a-f]{40}$ ]]',
        'test "$(git -C dependent rev-parse HEAD)" = "$ADAM_ASSIST_REF"',
        "Validate requested dependent revision syntax",
        "Verify dependent declares Core 0.8 compatibility",
        "pdm run python - <<'PY'",
        'Version("0.8.0") in core[0].specifier',
    ):
        if required not in text:
            failures.append(f"Tier-1 exact-ref workflow is missing {required!r}")
    if "c06164a18e8d90d8bf83d7ab58e479dc97e1be41" in text:
        failures.append(
            "Tier-1 workflow must not pin the Core-0.6-only Assist revision"
        )
    if "dependent/pyproject.toml" not in text:
        failures.append("Tier-1 workflow must inspect the dependent's own metadata")

    step_names = {
        "install": "Install test and exact build dependencies",
        "syntax": "Validate requested dependent revision syntax",
        "checkout": "Checkout requested dependent revision",
        "revision": "Verify exact dependent revision",
        "metadata": "Verify dependent declares Core 0.8 compatibility",
    }
    positions = {key: text.find(value) for key, value in step_names.items()}
    if all(position >= 0 for position in positions.values()):
        if not positions["syntax"] < positions["checkout"] < positions["revision"]:
            failures.append(
                "Tier-1 must validate exact-ref syntax before checkout and HEAD after checkout"
            )
        if not positions["install"] < positions["metadata"]:
            failures.append(
                "Tier-1 dependent metadata validation must follow the PDM dependency install"
            )
        metadata_end = text.find("\n      - name:", positions["metadata"] + 1)
        metadata_block = text[
            positions["metadata"] : metadata_end if metadata_end >= 0 else None
        ]
        if "pdm run python - <<'PY'" not in metadata_block:
            failures.append(
                "Tier-1 dependent metadata validation must use the installed PDM interpreter"
            )
        if "\n          python - <<'PY'" in metadata_block:
            failures.append(
                "Tier-1 dependent metadata validation must not use bare system Python"
            )
    if text.count("-m pip check") < 2:
        failures.append(
            "Tier-1 workflow must enforce the dependent's declared requirements in both environments"
        )
    if 'assist_version == "0.5.0"' in text:
        failures.append(
            "Tier-1 workflow must not claim a frozen Assist version is compatible"
        )

    return failures


def _required_workflow_artifact_failures(
    workflows: list[tuple[Path, str]],
) -> list[str]:
    workflow_text = "\n".join(text for _, text in workflows)
    failures: list[str] = []
    for artifact_name, expected_path in REQUIRED_WORKFLOW_ARTIFACTS.items():
        if f"name: {artifact_name}" not in workflow_text:
            failures.append(f"workflows: artifact `{artifact_name}` is not uploaded")
        if expected_path not in workflow_text:
            failures.append(
                f"workflows: artifact `{artifact_name}` must upload `{expected_path}`"
            )
    return failures


def main() -> None:
    project = _load_project()
    scripts = _load_scripts(project)
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
        failures.extend(_workflow_build_dependency_failures(source, text))
    failures.extend(_required_workflow_artifact_failures(workflows))
    failures.extend(_release_workflow_failures(project))
    failures.extend(_uv_lock_failures(project))
    failures.extend(_rust_msrv_dependency_failures())
    failures.extend(_tier1_exact_ref_failures())

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
