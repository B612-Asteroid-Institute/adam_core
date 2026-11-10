from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import pytest

from migration.scripts import (
    check_pdm_ci_scripts,
    develop_downstream_adam_assist,
    verify_preview_versions,
)
from migration.scripts.clean_room_artifact_smoke import _validate_runtime_versions

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "migration" / "scripts" / "write_maturin_version.py"
SPEC = importlib.util.spec_from_file_location("write_maturin_version", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    ("cargo", "python"),
    [
        ("0.5.6", "0.5.6"),
        ("0.5.6-alpha.1", "0.5.6a1"),
        ("0.5.6-beta.2", "0.5.6b2"),
        ("0.5.6-rc.1", "0.5.6rc1"),
        ("0.5.6-rc.2", "0.5.6rc2"),
        ("0.5.6-rc.3", "0.5.6rc3"),
        ("0.5.6-rc.4", "0.5.6rc4"),
        ("0.5.6-rc.5", "0.5.6rc5"),
        ("0.5.6-rc.6", "0.5.6rc6"),
        ("0.5.7", "0.5.7"),
        ("0.5.8", "0.5.8"),
        ("0.6.0", "0.6.0"),
        ("0.7.0", "0.7.0"),
    ],
)
def test_cargo_version_to_pep440(cargo: str, python: str) -> None:
    assert MODULE.cargo_version_to_pep440(cargo) == python


@pytest.mark.parametrize(
    "version",
    ["0.5", "0.5.6-preview.1", "0.5.6-rc", "0.5.6-rc.01", "0.5.6+local"],
)
def test_cargo_version_to_pep440_rejects_unsupported_forms(version: str) -> None:
    with pytest.raises(ValueError):
        MODULE.cargo_version_to_pep440(version)


def test_python_tag_validation_uses_the_python_release_line() -> None:
    assert MODULE._python_version_tag_glob("0.7.0") == "v0.7.*"


def test_stable_release_versions_align_python_and_public_rust_crates() -> None:
    verify_preview_versions.verify(ROOT, "0.7.0", "0.7.0", "stable")
    with pytest.raises(ValueError, match="not a prerelease"):
        verify_preview_versions.verify(ROOT, "0.7.0", "0.7.0", "preview")


def test_runtime_versions_must_match_distribution_metadata() -> None:
    versions = {"adam-core": "0.7.0", "adam-assist": "0.5.0"}
    assert _validate_runtime_versions(versions, versions.copy()) == versions

    with pytest.raises(AssertionError, match="runtime package version mismatch"):
        _validate_runtime_versions(
            versions,
            {"adam-core": "0.0.0dev0", "adam-assist": "0.5.0"},
        )


def test_release_matrix_generates_and_inspects_runtime_version() -> None:
    workflow = (
        ROOT / ".github/workflows/release-candidate-wheel-matrix.yml"
    ).read_text()
    builder = "python migration/scripts/build_release_wheel.py"
    inspector = "python migration/scripts/check_wheel_artifacts.py"
    acceptance = "python migration/scripts/run_core_wheel_acceptance.py"

    assert workflow.count(builder) == 2
    assert (
        workflow.index(builder) < workflow.index(acceptance) < workflow.index(inspector)
    )
    assert "python migration/scripts/write_maturin_version.py" not in workflow
    assert "adam_assist_ref:" not in workflow
    assert "ADAM_ASSIST_REF" not in workflow
    assert "adam-assist" not in workflow
    assert "adam_assist" not in workflow
    assert 'PYTHON_PREVIEW_VERSION: "0.7.0"' in workflow
    assert workflow.count('python-version: ["3.11", "3.12", "3.13", "3.14"]') == 1
    assert workflow.count("Run Core wheel acceptance") == 1
    assert "Run CPython 3.14 Core-only wheel acceptance" not in workflow
    assert workflow.count(acceptance) == 1
    assert '--expected-python "${{ matrix.python-version }}"' in workflow
    assert '--expected-system "${{ matrix.platform.system }}"' in workflow
    assert '--expected-wheel-tag "${{ matrix.platform.wheel_tag }}"' in workflow
    assert "<<'PY'" not in workflow
    assert "python - <<" not in workflow

    header = workflow.split("\njobs:", 1)[0]
    assert header.count('  ADAM_CORE_RELEASE_BUILD: "1"') == 1
    assert header.count("  ADAM_CORE_BUILD_COMMIT: ${{ github.sha }}") == 1

    artifact_acceptance = workflow.split("  artifact-acceptance:", 1)[1].split(
        "  extended-coverage:", 1
    )[0]
    extended_coverage = workflow.split("  extended-coverage:", 1)[1]
    extended_header = extended_coverage.split("\n    steps:", 1)[0]
    assert "continue-on-error:" not in artifact_acceptance
    assert workflow.count("continue-on-error:") == 1
    assert extended_header.count("    continue-on-error: true") == 1
    assert extended_header.count("    needs: artifact-acceptance") == 1
    assert "adam-assist" not in extended_coverage
    assert "adam_assist" not in extended_coverage
    assert "assist-rust-develop" not in extended_coverage
    assert (
        "env -u ADAM_CORE_RELEASE_BUILD -u ADAM_CORE_BUILD_COMMIT "
        "pdm install -G test" in " ".join(extended_coverage.split())
    )
    for check in (
        "cargo test -p adam_core_rs_kernel_data -- --ignored",
        "pdm run test-profile -- src/adam_core",
        'ADAM_CORE_LIVE_HORIZONS: "1"',
        'ADAM_CORE_LIVE_QUERY_CLIENTS: "1"',
        "pdm run test-live-integrations --",
    ):
        assert extended_coverage.count(check) == 1


def _mutated_release_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    original: str,
    replacement: str,
) -> list[str]:
    workflow = (
        ROOT / ".github/workflows/release-candidate-wheel-matrix.yml"
    ).read_text()
    assert workflow.count(original) == 1
    path = tmp_path / "release-candidate-wheel-matrix.yml"
    path.write_text(workflow.replace(original, replacement, 1))
    monkeypatch.setattr(check_pdm_ci_scripts, "RELEASE_WORKFLOW", path)
    with (ROOT / "pyproject.toml").open("rb") as source:
        project = tomllib.load(source)
    return check_pdm_ci_scripts._release_workflow_failures(project)


@pytest.mark.parametrize(
    ("step_name", "condition"),
    [
        ("Run Core wheel acceptance", "${{ matrix.python-version != '3.14' }}"),
        (
            "Run Core wheel acceptance",
            "${{ always() && matrix.python-version != '3.14' }}",
        ),
        (
            "Inspect adam-core wheel contents",
            "${{ matrix['python-version'] == '3.14' }}",
        ),
    ],
)
def test_release_policy_rejects_python_version_step_conditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    step_name: str,
    condition: str,
) -> None:
    original = f"      - name: {step_name}\n        shell: bash\n"
    replacement = (
        f"      - name: {step_name}\n"
        f"        if: {condition}\n"
        "        shell: bash\n"
    )

    failures = _mutated_release_failures(tmp_path, monkeypatch, original, replacement)

    assert any("must not use matrix.python-version" in failure for failure in failures)
    if step_name == "Run Core wheel acceptance":
        assert any(
            "Core wheel acceptance step must be unconditional" in failure
            for failure in failures
        )
    else:
        assert any(
            "wheel-content inspection step must be unconditional" in failure
            for failure in failures
        )


@pytest.mark.parametrize(
    ("original", "replacement", "message"),
    [
        (
            "  workflow_dispatch:\n",
            "  workflow_dispatch:\n"
            "    inputs:\n"
            "      adam_assist_ref:\n"
            "        required: true\n",
            "must not own downstream Assist work",
        ),
        (
            '        python-version: ["3.11", "3.12", "3.13", "3.14"]\n',
            '        python-version: ["3.11", "3.12", "3.13"]\n',
            "must contain exactly CPython",
        ),
        (
            "        platform:\n",
            "        exclude:\n"
            '          - python-version: "3.14"\n'
            "        platform:\n",
            "only python-version and platform axes",
        ),
        (
            "        platform:\n",
            "        include:\n"
            '          - python-version: "3.14"\n'
            "        platform:\n",
            "only python-version and platform axes",
        ),
        (
            "          - name: macos-x86_64\n"
            "            runner: macos-15-intel\n"
            "            system: Darwin-x86_64\n"
            "            family: native\n"
            "            wheel_tag: macosx*x86_64\n"
            "    steps:\n",
            "          - name: macos-x86_64\n"
            "            runner: macos-15-intel\n"
            "            system: Darwin-x86_64\n"
            "            family: native\n"
            "            wheel_tag: macosx*x86_64\n"
            "          - name: manylinux-ppc64le\n"
            "            runner: ubuntu-latest\n"
            "            system: Linux-ppc64le\n"
            "            family: manylinux\n"
            "            target: powerpc64le-unknown-linux-gnu\n"
            "            wheel_tag: manylinux*ppc64le\n"
            "    steps:\n",
            "exactly four supported platforms",
        ),
    ],
)
def test_release_policy_rejects_nonuniform_matrix_or_downstream_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    original: str,
    replacement: str,
    message: str,
) -> None:
    failures = _mutated_release_failures(tmp_path, monkeypatch, original, replacement)

    assert any(message in failure for failure in failures)


@pytest.mark.parametrize(
    ("original", "replacement", "message"),
    [
        (
            "      - name: Run Core wheel acceptance\n",
            "      - name: Run version-specific wheel acceptance\n",
            "exactly one unconditional Core wheel acceptance step",
        ),
        (
            "          python migration/scripts/run_core_wheel_acceptance.py\n",
            "          python migration/scripts/run_clean_room_artifact_acceptance.py\n",
            "run_core_wheel_acceptance.py",
        ),
        (
            '          --wheelhouse "$GITHUB_WORKSPACE/wheelhouse"\n',
            '          --wheelhouse "$RUNNER_TEMP/wheelhouse"\n',
            "--wheelhouse",
        ),
        (
            '          --expected-commit "$GITHUB_SHA"\n',
            '          --expected-commit "HEAD"\n',
            "--expected-commit",
        ),
        (
            '          --expected-wheel-tag "${{ matrix.platform.wheel_tag }}"\n',
            '          --ignored-wheel-tag "${{ matrix.platform.wheel_tag }}"\n',
            "--expected-wheel-tag",
        ),
        (
            "      - name: Checkout adam-core candidate\n"
            "        uses: actions/checkout@v4\n"
            "        with:\n"
            "          fetch-depth: 0\n",
            "      - name: Embedded Python test\n"
            "        run: |\n"
            "          python - <<'PY'\n"
            "          print('forbidden')\n"
            "          PY\n"
            "      - name: Checkout adam-core candidate\n"
            "        uses: actions/checkout@v4\n"
            "        with:\n"
            "          fetch-depth: 0\n",
            "must not embed shell heredocs or inline Python programs",
        ),
    ],
)
def test_release_policy_rejects_weakened_core_acceptance_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    original: str,
    replacement: str,
    message: str,
) -> None:
    failures = _mutated_release_failures(tmp_path, monkeypatch, original, replacement)

    assert any(message in failure for failure in failures)


@pytest.mark.parametrize(
    "embedded_run",
    [
        "          cat <<'EOF'\n          forbidden\n          EOF\n",
        "          python -c \"print('forbidden')\"\n",
    ],
)
def test_release_policy_rejects_other_heredoc_or_inline_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    embedded_run: str,
) -> None:
    original = (
        "      - name: Checkout adam-core candidate\n"
        "        uses: actions/checkout@v4\n"
        "        with:\n"
        "          fetch-depth: 0\n"
    )
    replacement = (
        "      - name: Embedded program\n"
        "        run: |\n"
        f"{embedded_run}"
        f"{original}"
    )

    failures = _mutated_release_failures(tmp_path, monkeypatch, original, replacement)

    assert any(
        "must not embed shell heredocs or inline Python programs" in failure
        for failure in failures
    )


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ('  ADAM_CORE_RELEASE_BUILD: "1"', '  ADAM_CORE_RELEASE_BUILD: "0"'),
        (
            "  ADAM_CORE_BUILD_COMMIT: ${{ github.sha }}",
            "  ADAM_CORE_BUILD_COMMIT: HEAD",
        ),
        (
            "env -u ADAM_CORE_RELEASE_BUILD -u ADAM_CORE_BUILD_COMMIT",
            "env -u ADAM_CORE_BUILD_COMMIT",
        ),
        (
            "env -u ADAM_CORE_RELEASE_BUILD -u ADAM_CORE_BUILD_COMMIT",
            "env -u ADAM_CORE_RELEASE_BUILD",
        ),
        (
            "env -u ADAM_CORE_RELEASE_BUILD -u ADAM_CORE_BUILD_COMMIT",
            "env",
        ),
    ],
)
def test_release_policy_rejects_artifact_or_extended_install_env_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    original: str,
    replacement: str,
) -> None:
    workflow = (
        ROOT / ".github/workflows/release-candidate-wheel-matrix.yml"
    ).read_text()
    assert original in workflow
    mutated = workflow.replace(original, replacement, 1)
    path = tmp_path / "release-candidate-wheel-matrix.yml"
    path.write_text(mutated)
    monkeypatch.setattr(check_pdm_ci_scripts, "RELEASE_WORKFLOW", path)
    with (ROOT / "pyproject.toml").open("rb") as source:
        project = tomllib.load(source)

    failures = check_pdm_ci_scripts._release_workflow_failures(project)

    assert any(
        "artifact-build environment" in failure
        or "remove both release-identity variables" in failure
        for failure in failures
    )


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ("    continue-on-error: true", "    continue-on-error: false"),
        (
            "  artifact-acceptance:\n",
            "  artifact-acceptance:\n    continue-on-error: true\n",
        ),
        ("    needs: artifact-acceptance", "    needs: lint"),
        (
            "cargo test -p adam_core_rs_kernel_data -- --ignored",
            "cargo test -p adam_core_rs_kernel_data",
        ),
        (
            "pdm run test-profile -- src/adam_core",
            "pdm run test-profile",
        ),
        ('ADAM_CORE_LIVE_HORIZONS: "1"', 'ADAM_CORE_LIVE_HORIZONS: "0"'),
        (
            'ADAM_CORE_LIVE_QUERY_CLIENTS: "1"',
            'ADAM_CORE_LIVE_QUERY_CLIENTS: "0"',
        ),
        ("pdm run test-live-integrations --", "pdm run test --"),
    ],
)
def test_release_policy_rejects_extended_coverage_boundary_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    original: str,
    replacement: str,
) -> None:
    workflow = (
        ROOT / ".github/workflows/release-candidate-wheel-matrix.yml"
    ).read_text()
    assert original in workflow
    path = tmp_path / "release-candidate-wheel-matrix.yml"
    path.write_text(workflow.replace(original, replacement, 1))
    monkeypatch.setattr(check_pdm_ci_scripts, "RELEASE_WORKFLOW", path)
    with (ROOT / "pyproject.toml").open("rb") as source:
        project = tomllib.load(source)

    failures = check_pdm_ci_scripts._release_workflow_failures(project)

    assert any(
        "hard gating" in failure
        or "non-gating" in failure
        or "depend on artifact acceptance" in failure
        or "retain slow/live check" in failure
        for failure in failures
    )


def test_release_builder_uses_declared_exact_tools_and_supported_targets() -> None:
    """The release matrix and fresh-build guard share one exact tool set."""

    with (ROOT / "pyproject.toml").open("rb") as source:
        project = tomllib.load(source)
    expected_tools = [
        "maturin==1.15.0",
        "packaging==25.0",
        "ziglang==0.17.0",
    ]
    assert project["dependency-groups"]["build"] == expected_tools
    assert "override-dependencies" not in project["tool"]["uv"]
    assert project["tool"]["uv"]["sources"]["adam-core"] == {"workspace": True}
    assert project["tool"]["pdm"]["scripts"]["release-build-environment-check"] == (
        "python migration/scripts/validate_release_build_environment.py"
    )

    with (ROOT / "uv.lock").open("rb") as source:
        lock = tomllib.load(source)
    packages = {package["name"]: package for package in lock["package"]}
    assert packages["adam-core"]["source"] == {"editable": "."}
    for name, version in (
        ("maturin", "1.15.0"),
        ("packaging", "25.0"),
        ("ziglang", "0.17.0"),
    ):
        package = packages[name]
        assert package["version"] == version
        assert package["source"] == {"registry": "https://pypi.org/simple"}
        artifacts = ([package["sdist"]] if "sdist" in package else []) + package[
            "wheels"
        ]
        assert artifacts
        assert all(artifact["hash"].startswith("sha256:") for artifact in artifacts)
    assert any(
        dependency["name"] == "adam-core"
        for dependency in packages["mpcq"]["dependencies"]
    )

    workflow = (
        ROOT / ".github/workflows/release-candidate-wheel-matrix.yml"
    ).read_text()
    install_step = workflow.index("Install audited adam-core wheel builder")
    builder = workflow.index("python migration/scripts/build_release_wheel.py")
    inspector = workflow.index("python migration/scripts/check_wheel_artifacts.py")
    assert install_step < builder < inspector
    for tool in expected_tools:
        assert workflow.count(f'"{tool}"') == 1
        assert install_step < workflow.index(f'"{tool}"') < builder
    assert "--only-binary=:all: packaging" not in workflow
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
    }
    for name, entries in expected_cells.items():
        actual = check_pdm_ci_scripts._release_matrix_cell(workflow, name)
        assert actual is not None
        assert all(entry in actual for entry in entries)
    assert check_pdm_ci_scripts._release_matrix_cell(workflow, "macos-arm64") == (
        "runner: macos-15",
        "system: Darwin-arm64",
        "family: native",
        "wheel_tag: macosx*arm64",
    )
    assert check_pdm_ci_scripts._release_matrix_cell(workflow, "macos-x86_64") == (
        "runner: macos-15-intel",
        "system: Darwin-x86_64",
        "family: native",
        "wheel_tag: macosx*x86_64",
    )
    assert "target: x86_64\n" not in workflow
    assert "target: aarch64\n" not in workflow

    pip_ci = (ROOT / ".github/workflows/pip-build-lint-test-coverage.yml").read_text()
    rust_job = pip_ci.split("  rust-smoke:", 1)[1].split("\n  benchmark:", 1)[0]
    assert "pdm install -G test -G build" in rust_job
    assert "pdm run release-build-environment-check --out dist" in rust_job
    assert "pdm run rust-build" not in rust_job
    assert (
        rust_job.index("pdm install -G test -G build")
        < rust_job.index("pdm run release-build-environment-check --out dist")
        < rust_job.index("pdm run wheel-inspect")
    )

    tier1 = (ROOT / ".github/workflows/tier1-dependent-smoke.yml").read_text()
    assert "pdm install -G test -G build" in tier1
    assert (
        tier1.index("pdm install -G test -G build")
        < tier1.index("pdm run rust-build")
        < tier1.index("pdm run wheel-inspect")
    )

    publish = (ROOT / ".github/workflows/publish.yml").read_text()
    exact_parser = 'python -m pip install --only-binary=:all: "packaging==25.0"'
    assert publish.count(exact_parser) == 1
    assert publish.index(exact_parser) < publish.index(
        "python migration/scripts/check_wheel_artifacts.py"
    )
    assert 'expected_python = {"cp311", "cp312", "cp313", "cp314"}' in publish
    assert publish.count("--expected-count 16") == 2
    assert (
        "test \"$(find dist -maxdepth 1 -name 'adam_core-*.whl' | "
        "wc -l | tr -d ' ')\" = 16" in publish
    )

    installation = (ROOT / "docs/source/getting_started/installation.rst").read_text()
    assert "pdm install -G test -G docs -G build" in installation


def test_kernel_data_constrains_icu_for_clean_rust_1_87_consumers() -> None:
    manifest_path = ROOT / "rust/adam_core_rs_kernel_data/Cargo.toml"
    with manifest_path.open("rb") as source:
        dependencies = tomllib.load(source)["dependencies"]

    expected = {"version": "=2.2.0", "default-features": False}
    for name in (
        "icu_locale_core",
        "icu_normalizer",
        "icu_properties",
        "icu_provider",
    ):
        assert dependencies[name] == expected
    assert dependencies["rawzip"] == "=0.5.2"
    assert "zip" not in dependencies
    assert dependencies["flate2"] == {
        "version": "1",
        "default-features": False,
        "features": ["rust_backend"],
    }

    with (ROOT / "rust/adam_core_rs_coords/Cargo.toml").open("rb") as source:
        coords_dependencies = tomllib.load(source)["dependencies"]
    with (ROOT / "rust/adam_core_rs_spice/Cargo.toml").open("rb") as source:
        spice_dependencies = tomllib.load(source)["dependencies"]
    assert "faer" not in coords_dependencies
    assert spice_dependencies["faer"] == {
        "version": "0.22",
        "default-features": False,
        "features": ["std", "linalg", "rayon"],
    }

    workflow = (ROOT / ".github/workflows/rust-crate-release-candidate.yml").read_text()
    assert "Verify clean no-lock Rust 1.87 consumer resolution" in workflow
    assert 'test ! -e "$consumer/Cargo.lock"' in workflow
    assert 'rustc --version | grep -F "rustc 1.87.0"' in workflow
    assert "clean Rust 1.87 consumer resolved faer std+linalg+rayon" in workflow
    for forbidden in ("ar_archive_writer", "npyz", "pest_stack_guard", "psm"):
        assert f'"{forbidden}"' in workflow
    assert 'faer feature "linalg"' in workflow
    assert 'faer feature "rayon"' in workflow
    assert 'faer feature "std"' in workflow
    assert 'faer feature "(default|npy|rand|sparse|sparse-linalg)"' in workflow
    assert 'cargo package -p "$crate" --locked' in workflow
    assert 'RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps' in workflow
    assert "Latest-stable Rust compatibility (non-authoritative)" in workflow
    setup_python = (
        "actions/setup-python@" "a26af69be951a213d495a4c3e4e4022e16d87065 # v5.6.0"
    )
    assert workflow.count("name: Set up supported PyO3 build interpreter") == 2
    assert workflow.count(f"uses: {setup_python}") == 2
    assert workflow.count('python-version: "3.13"') == 2
    package_job, latest_job = workflow.split("  latest-stable-compatibility:", 1)
    for job in (package_job, latest_job):
        assert job.count(setup_python) == 1
        assert job.count('python-version: "3.13"') == 1
        assert job.index(setup_python) < job.index("dtolnay/rust-toolchain@")
        assert job.index(setup_python) < job.index("cargo ")
    assert "PYO3_USE_ABI3_FORWARD_COMPATIBILITY" not in workflow
    assert "dtolnay/rust-toolchain@stable" in workflow
    assert "cargo test --workspace --locked --all-features" in workflow
    assert "--locked --no-default-features" in workflow
    assert 'test ! -e "$consumer/default/Cargo.lock"' in workflow
    assert 'test ! -e "$consumer/no-default/Cargo.lock"' in workflow
    assert 'adam_core = { path = "$GITHUB_WORKSPACE/rust/adam_core" }' in workflow
    assert "adam_core_rs_kernel_data = { path =" in workflow

    python_publisher = (ROOT / ".github/workflows/publish.yml").read_text()
    assert "testpypi" not in python_publisher.lower()
    assert "to pypi" in python_publisher
    assert "name: pypi" in python_publisher
    assert "prepare_pypi_upload.py" in python_publisher
    assert "packages-dir: upload-dist/" in python_publisher
    assert "skip-existing" not in python_publisher
    assert (
        "pypa/gh-action-pypi-publish@"
        "dc37677b2e1c63e2034f94d8a5b11f265b73ba33" in python_publisher
    )

    publisher = (ROOT / ".github/workflows/publish-crates.yml").read_text()
    assert "default: 0.7.0" in publisher
    assert 'PYTHON_SOURCE_VERSION: "0.7.0"' in publisher
    assert "environment: crates-io" in publisher
    assert "--channel stable" in publisher
    assert (
        "rust-lang/crates-io-auth-action@"
        "c6f97d42243bad5fab37ca0427f495c86d5b1a18" in publisher
    )
    assert 'test "$GITHUB_REF" = "refs/tags/v$EXPECTED_VERSION"' in publisher


def test_downstream_assist_build_patches_only_current_core_crates(
    tmp_path: Path,
) -> None:
    assert develop_downstream_adam_assist.PUBLIC_CORE_CRATES == (
        "adam_core",
        "adam_core_rs_autodiff",
        "adam_core_rs_coords",
        "adam_core_rs_kernel_data",
        "adam_core_rs_orbit_determination",
        "adam_core_rs_spice",
    )
    config, remove_cargo_dir = develop_downstream_adam_assist._write_core_patch(
        tmp_path
    )
    assert remove_cargo_dir
    source = config.read_text()
    assert source.startswith("[patch.crates-io]\n")
    for crate in develop_downstream_adam_assist.PUBLIC_CORE_CRATES:
        assert f"{crate} =" in source
        assert str(ROOT / "rust" / crate) in source

    with pytest.raises(SystemExit, match="refusing to replace"):
        develop_downstream_adam_assist._write_core_patch(tmp_path)

    config.unlink()
    config.parent.rmdir()


def _package(name: str, version: str, *, source: str | None = None) -> dict[str, str]:
    package = {"name": name, "version": version}
    if source is not None:
        package["source"] = source
    return package


def _core_packages(version: str) -> list[dict[str, str]]:
    return [
        _package(crate, version)
        for crate in develop_downstream_adam_assist.PUBLIC_CORE_CRATES
    ]


def _write_assist_lock(path: Path, packages: list[dict[str, str]]) -> bytes:
    lines = ["version = 4", ""]
    for package in packages:
        lines.append("[[package]]")
        for key, value in package.items():
            lines.append(f'{key} = "{value}"')
        lines.append("")
    contents = "\n".join(lines).encode()
    path.write_bytes(contents)
    return contents


def _write_assist_manifest(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "rust" / "adam_assist_rs" / "Cargo.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('[package]\nname = "adam_assist"\nversion = "0.5.0"\n')
    return manifest, manifest.with_name("Cargo.lock")


def test_downstream_core_lock_refresh_is_one_targeted_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, lock = _write_assist_manifest(tmp_path)
    before = [
        *_core_packages("0.6.0"),
        _package("unrelated", "1.0.0"),
        _package("zip", "2.4.2"),
    ]
    _write_assist_lock(lock, before)

    commands: list[list[str]] = []

    def fake_check_call(command: list[str], **_: object) -> None:
        commands.append(command)
        after = [
            *_core_packages("0.7.0"),
            _package("unrelated", "1.0.0"),
            _package("rawzip", "0.5.2"),
        ]
        _write_assist_lock(lock, after)

    monkeypatch.setattr(
        develop_downstream_adam_assist.shutil, "which", lambda _: "/cargo"
    )
    monkeypatch.setattr(
        develop_downstream_adam_assist.subprocess, "check_call", fake_check_call
    )
    monkeypatch.setattr(
        develop_downstream_adam_assist, "_current_core_version", lambda: "0.7.0"
    )

    develop_downstream_adam_assist._refresh_core_lock(tmp_path, {})

    expected = ["/cargo", "update", "--manifest-path", str(manifest)]
    for crate in develop_downstream_adam_assist.PUBLIC_CORE_CRATES:
        expected.extend(["--package", crate])
    assert commands == [expected]
    assert "--precise" not in commands[0]


def test_downstream_core_lock_generation_supports_initial_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, lock = _write_assist_manifest(tmp_path)
    commands: list[list[str]] = []

    def fake_check_call(command: list[str], **_: object) -> None:
        commands.append(command)
        assert (tmp_path / ".cargo" / "config.toml").is_file()
        _write_assist_lock(lock, _core_packages("0.7.0"))

    monkeypatch.setattr(
        develop_downstream_adam_assist.shutil, "which", lambda _: "/cargo"
    )
    monkeypatch.setattr(
        develop_downstream_adam_assist.subprocess, "check_call", fake_check_call
    )
    monkeypatch.setattr(
        develop_downstream_adam_assist, "_current_core_version", lambda: "0.7.0"
    )

    prepared = develop_downstream_adam_assist._prepare_downstream(tmp_path, {})

    assert commands == [
        ["/cargo", "generate-lockfile", "--manifest-path", str(manifest)]
    ]
    assert prepared.original_lock is None
    assert lock.is_file()
    assert prepared.config.is_file()

    develop_downstream_adam_assist._restore_downstream(prepared)
    assert not lock.exists()
    assert not prepared.config.exists()
    assert not prepared.config.parent.exists()


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "registry-source", "wrong-version"],
)
def test_generated_lock_validation_failure_restores_initial_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    manifest, lock = _write_assist_manifest(tmp_path)
    generated = _core_packages("0.7.0")
    if mutation == "missing":
        generated.pop()
    elif mutation == "duplicate":
        generated.append(dict(generated[0]))
    elif mutation == "registry-source":
        generated[0]["source"] = "registry+https://github.com/rust-lang/crates.io-index"
    elif mutation == "wrong-version":
        generated[0]["version"] = "0.6.1"
    else:  # pragma: no cover - the parameter list is closed above
        raise AssertionError(mutation)

    def fake_check_call(command: list[str], **_: object) -> None:
        assert command == [
            "/cargo",
            "generate-lockfile",
            "--manifest-path",
            str(manifest),
        ]
        assert (tmp_path / ".cargo" / "config.toml").is_file()
        _write_assist_lock(lock, generated)

    monkeypatch.setattr(
        develop_downstream_adam_assist.shutil, "which", lambda _: "/cargo"
    )
    monkeypatch.setattr(
        develop_downstream_adam_assist.subprocess, "check_call", fake_check_call
    )
    monkeypatch.setattr(
        develop_downstream_adam_assist, "_current_core_version", lambda: "0.7.0"
    )

    with pytest.raises(SystemExit, match="exactly one local-path Core 0.7.0"):
        develop_downstream_adam_assist._prepare_downstream(tmp_path, {})

    assert not lock.exists()
    assert not (tmp_path / ".cargo" / "config.toml").exists()
    assert not (tmp_path / ".cargo").exists()


def test_patched_core_lock_verification_requires_one_local_aligned_package() -> None:
    valid = _core_packages("0.7.0")
    develop_downstream_adam_assist._verify_patched_core_packages(valid, "0.7.0")

    wrong_version = [dict(package) for package in valid]
    wrong_version[0]["version"] = "0.6.0"
    with pytest.raises(SystemExit, match="exactly one local-path Core 0.7.0"):
        develop_downstream_adam_assist._verify_patched_core_packages(
            wrong_version, "0.7.0"
        )

    registry_source = [dict(package) for package in valid]
    registry_source[0][
        "source"
    ] = "registry+https://github.com/rust-lang/crates.io-index"
    with pytest.raises(SystemExit, match="exactly one local-path Core 0.7.0"):
        develop_downstream_adam_assist._verify_patched_core_packages(
            registry_source, "0.7.0"
        )

    duplicate = [*valid, dict(valid[0])]
    with pytest.raises(SystemExit, match="exactly one local-path Core 0.7.0"):
        develop_downstream_adam_assist._verify_patched_core_packages(duplicate, "0.7.0")

    missing = valid[1:]
    with pytest.raises(SystemExit, match="exactly one local-path Core 0.7.0"):
        develop_downstream_adam_assist._verify_patched_core_packages(missing, "0.7.0")


def test_unrelated_lock_guard_allows_removals_but_rejects_added_versions() -> None:
    before = [
        *_core_packages("0.6.0"),
        _package("retained", "1.0.0"),
        _package("retained", "2.0.0"),
        _package("removed", "1.0.0"),
    ]
    accepted = [
        *_core_packages("0.7.0"),
        _package("retained", "1.0.0"),
        _package("rawzip", "0.5.2"),
    ]
    develop_downstream_adam_assist._reject_unrelated_version_additions(before, accepted)

    added_version = [*accepted, _package("retained", "3.0.0")]
    with pytest.raises(SystemExit, match="added unrelated package versions"):
        develop_downstream_adam_assist._reject_unrelated_version_additions(
            before, added_version
        )


def test_lock_guard_aligns_spicekit_only_to_the_core_lock() -> None:
    core_versions = develop_downstream_adam_assist._locked_versions(
        develop_downstream_adam_assist._locked_packages(ROOT / "Cargo.lock")
    )
    (spicekit_version,) = core_versions["spicekit"]
    before = [_package("spicekit", "0.2.2"), _package("retained", "1.0.0")]
    aligned = [
        _package("spicekit", spicekit_version),
        _package("retained", "1.0.0"),
    ]
    develop_downstream_adam_assist._reject_unrelated_version_additions(before, aligned)

    for changed in (
        [_package("spicekit", "99.0.0"), _package("retained", "1.0.0")],
        [*aligned, _package("retained", "2.0.0")],
    ):
        with pytest.raises(SystemExit, match="added unrelated package versions"):
            develop_downstream_adam_assist._reject_unrelated_version_additions(
                before, changed
            )


def test_existing_lock_unrelated_addition_restores_original_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, lock = _write_assist_manifest(tmp_path)
    original = _write_assist_lock(
        lock,
        [*_core_packages("0.6.0"), _package("retained", "1.0.0")],
    )
    commands: list[list[str]] = []

    def fake_check_call(command: list[str], **_: object) -> None:
        commands.append(command)
        _write_assist_lock(
            lock,
            [
                *_core_packages("0.7.0"),
                _package("retained", "1.0.0"),
                _package("retained", "2.0.0"),
            ],
        )

    monkeypatch.setattr(
        develop_downstream_adam_assist.shutil, "which", lambda _: "/cargo"
    )
    monkeypatch.setattr(
        develop_downstream_adam_assist.subprocess, "check_call", fake_check_call
    )
    monkeypatch.setattr(
        develop_downstream_adam_assist, "_current_core_version", lambda: "0.7.0"
    )

    with pytest.raises(SystemExit, match="added unrelated package versions"):
        develop_downstream_adam_assist._prepare_downstream(tmp_path, {})

    expected = ["/cargo", "update", "--manifest-path", str(manifest)]
    for crate in develop_downstream_adam_assist.PUBLIC_CORE_CRATES:
        expected.extend(["--package", crate])
    assert commands == [expected]
    assert lock.read_bytes() == original
    assert not (tmp_path / ".cargo" / "config.toml").exists()
    assert not (tmp_path / ".cargo").exists()


def test_downstream_prepare_restores_lock_and_config_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, lock = _write_assist_manifest(tmp_path)
    original = _write_assist_lock(lock, _core_packages("0.6.0"))

    def fake_refresh(repo: Path, env: dict[str, str], paths: tuple[Path, Path]) -> None:
        assert repo == tmp_path
        assert env == {"TEST": "1"}
        assert paths[1] == lock
        lock.write_bytes(b"temporary lock")
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(
        develop_downstream_adam_assist, "_refresh_core_lock", fake_refresh
    )
    with pytest.raises(RuntimeError, match="refresh failed"):
        develop_downstream_adam_assist._prepare_downstream(tmp_path, {"TEST": "1"})

    assert lock.read_bytes() == original
    assert not (tmp_path / ".cargo" / "config.toml").exists()
    assert not (tmp_path / ".cargo").exists()


def test_downstream_prepare_restores_absent_lock_and_config_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, lock = _write_assist_manifest(tmp_path)

    def fake_refresh(repo: Path, env: dict[str, str], paths: tuple[Path, Path]) -> None:
        assert repo == tmp_path
        assert env == {"TEST": "1"}
        assert paths[1] == lock
        lock.write_bytes(b"temporary lock")
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(
        develop_downstream_adam_assist, "_refresh_core_lock", fake_refresh
    )
    with pytest.raises(RuntimeError, match="refresh failed"):
        develop_downstream_adam_assist._prepare_downstream(tmp_path, {"TEST": "1"})

    assert not lock.exists()
    assert not (tmp_path / ".cargo" / "config.toml").exists()
    assert not (tmp_path / ".cargo").exists()


def test_downstream_prepare_can_restore_successful_temporary_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, lock = _write_assist_manifest(tmp_path)
    original = _write_assist_lock(lock, _core_packages("0.6.0"))

    def fake_refresh(repo: Path, env: dict[str, str], paths: tuple[Path, Path]) -> None:
        assert repo == tmp_path
        assert env == {}
        assert paths[1] == lock
        lock.write_bytes(b"temporary lock")

    monkeypatch.setattr(
        develop_downstream_adam_assist, "_refresh_core_lock", fake_refresh
    )
    prepared = develop_downstream_adam_assist._prepare_downstream(tmp_path, {})
    assert lock.read_bytes() == b"temporary lock"
    assert prepared.config.exists()

    develop_downstream_adam_assist._restore_downstream(prepared)
    assert lock.read_bytes() == original
    assert not prepared.config.exists()
    assert not prepared.config.parent.exists()


def test_develop_restores_lock_and_config_when_maturin_returns_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, lock = _write_assist_manifest(tmp_path)
    original = _write_assist_lock(lock, _core_packages("0.6.0"))

    def fake_refresh(repo: Path, env: dict[str, str], paths: tuple[Path, Path]) -> None:
        assert repo == tmp_path
        assert "CARGO_NET_GIT_FETCH_WITH_CLI" in env
        assert paths[1] == lock
        lock.write_bytes(b"temporary lock")

    class PipCheck:
        returncode = 0

    monkeypatch.setenv("ADAM_ASSIST_RUST_REPO", str(tmp_path))
    monkeypatch.setattr(
        develop_downstream_adam_assist, "_refresh_core_lock", fake_refresh
    )
    monkeypatch.setattr(
        develop_downstream_adam_assist.shutil, "which", lambda _: "/maturin"
    )
    monkeypatch.setattr(
        develop_downstream_adam_assist.subprocess,
        "run",
        lambda *args, **kwargs: PipCheck(),
    )
    monkeypatch.setattr(
        develop_downstream_adam_assist.subprocess,
        "call",
        lambda *args, **kwargs: 17,
    )

    assert develop_downstream_adam_assist.main([]) == 17
    assert lock.read_bytes() == original
    assert not (tmp_path / ".cargo" / "config.toml").exists()
    assert not (tmp_path / ".cargo").exists()


def test_prepare_only_cli_leaves_generated_lock_and_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, lock = _write_assist_manifest(tmp_path)
    commands: list[list[str]] = []

    def fake_check_call(command: list[str], **_: object) -> None:
        commands.append(command)
        _write_assist_lock(lock, _core_packages("0.7.0"))

    monkeypatch.setenv("ADAM_ASSIST_RUST_REPO", str(tmp_path))
    monkeypatch.setattr(
        develop_downstream_adam_assist.shutil, "which", lambda _: "/cargo"
    )
    monkeypatch.setattr(
        develop_downstream_adam_assist.subprocess, "check_call", fake_check_call
    )
    monkeypatch.setattr(
        develop_downstream_adam_assist, "_current_core_version", lambda: "0.7.0"
    )

    assert develop_downstream_adam_assist.main(["--prepare-only"]) == 0
    assert commands == [
        ["/cargo", "generate-lockfile", "--manifest-path", str(manifest)]
    ]
    assert lock.is_file()
    assert (tmp_path / ".cargo" / "config.toml").is_file()


def test_prepare_only_cli_dispatches_without_maturin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_assist_manifest(tmp_path)[1].write_text("version = 4\n")
    calls: list[tuple[Path, dict[str, str]]] = []

    def fake_prepare(
        repo: Path,
        env: dict[str, str],
        paths: tuple[Path, Path],
    ) -> develop_downstream_adam_assist.PreparedDownstream:
        assert paths[0].name == "Cargo.toml"
        assert paths[1].name == "Cargo.lock"
        calls.append((repo, env))
        return develop_downstream_adam_assist.PreparedDownstream(
            tmp_path / ".cargo/config.toml", True, tmp_path / "Cargo.lock", b""
        )

    def unexpected_maturin_lookup(_: str) -> str:
        raise AssertionError("prepare-only must not look up maturin")

    monkeypatch.setenv("ADAM_ASSIST_RUST_REPO", str(tmp_path))
    monkeypatch.setenv("CARGO_NET_GIT_FETCH_WITH_CLI", "preserved")
    monkeypatch.setattr(
        develop_downstream_adam_assist, "_prepare_downstream", fake_prepare
    )
    monkeypatch.setattr(
        develop_downstream_adam_assist.shutil,
        "which",
        unexpected_maturin_lookup,
    )

    assert develop_downstream_adam_assist.main(["--prepare-only"]) == 0
    assert len(calls) == 1
    assert calls[0][0] == tmp_path
    assert calls[0][1]["CARGO_NET_GIT_FETCH_WITH_CLI"] == "preserved"


def test_rust_develop_refreshes_version_and_explains_stale_extensions() -> None:
    pyproject_path = ROOT / "pyproject.toml"
    pyproject = pyproject_path.read_text()
    with pyproject_path.open("rb") as source:
        test_dependencies = tomllib.load(source)["dependency-groups"]["test"]
    assert not any(
        dependency.startswith("adam-assist") for dependency in test_dependencies
    )
    assert "mpc-obscodes==2026.4.18" in test_dependencies

    writer = '"pdm run wheel-version"'
    builder = '"maturin develop --manifest-path rust/adam_core_py/Cargo.toml --release"'
    assert pyproject.index(writer) < pyproject.index(builder)

    rust_api = (ROOT / "src/adam_core/_rust/api.py").read_text()
    assert "pdm install -G test && pdm run rust-develop" in rust_api


def test_rust_ci_is_reproducible_and_downstream_sources_are_exact() -> None:
    workflows = ROOT / ".github/workflows"
    normal_ci = (workflows / "pip-build-lint-test-coverage.yml").read_text()
    crate_ci = (workflows / "rust-crate-release-candidate.yml").read_text()
    nightly = (workflows / "nightly-rust-benchmark.yml").read_text()
    release_matrix = (workflows / "release-candidate-wheel-matrix.yml").read_text()
    tier1 = (workflows / "tier1-dependent-smoke.yml").read_text()

    assert "dtolnay/rust-toolchain@1.87.0" in normal_ci
    assert "components: rustfmt, clippy" in normal_ci
    assert normal_ci.count("python-version: [ '3.11', '3.12', '3.13', '3.14' ]") == 1
    assert "dtolnay/rust-toolchain@1.87.0" in crate_ci
    assert "components: rustfmt, clippy" in crate_ci
    assert "dtolnay/rust-toolchain@1.87.0" in tier1
    assert "adam_assist_ref:" not in release_matrix
    assert "ADAM_ASSIST_REF" not in release_matrix
    tier1_trigger = tier1.split("permissions:", 1)[0]
    assert "workflow_dispatch:" in tier1_trigger
    assert "pull_request:" not in tier1_trigger
    assert "schedule:" not in tier1_trigger
    assert "adam_assist_ref:" in tier1_trigger
    assert "required: true" in tier1_trigger
    assert "c06164a18e8d90d8bf83d7ab58e479dc97e1be41" not in tier1
    assert "ADAM_ASSIST_REF: ${{ inputs.adam_assist_ref }}" in tier1
    assert "ref: ${{ env.ADAM_ASSIST_REF }}" in tier1
    assert tier1.count('[[ "$ADAM_ASSIST_REF" =~ ^[0-9a-f]{40}$ ]]') == 1
    assert 'test "$(git -C dependent rev-parse HEAD)" = "$ADAM_ASSIST_REF"' in tier1
    syntax_name = "Validate requested dependent revision syntax"
    checkout_name = "Checkout requested dependent revision"
    revision_name = "Verify exact dependent revision"
    install_name = "Install test and exact build dependencies"
    metadata_name = "Verify dependent declares Core 0.7 compatibility"
    assert (
        tier1.index(syntax_name)
        < tier1.index(checkout_name)
        < tier1.index(revision_name)
    )
    assert tier1.index(install_name) < tier1.index(metadata_name)
    metadata_start = tier1.index(metadata_name)
    metadata_end = tier1.index("\n      - name:", metadata_start)
    metadata_block = tier1[metadata_start:metadata_end]
    assert "pdm run python - <<'PY'" in metadata_block
    assert "\n          python - <<'PY'" not in metadata_block
    assert 'Version("0.7.0") in core[0].specifier' in metadata_block
    assert "dependent/pyproject.toml" in metadata_block
    assert "adam-assist" not in normal_ci
    assert "adam_assist" not in normal_ci
    assert "adam-assist" not in nightly
    assert "adam_assist" not in nightly
    assert "adam-assist" not in release_matrix
    assert "adam_assist" not in release_matrix
    assert "Run Core wheel acceptance" in release_matrix
    assert "run_core_wheel_acceptance.py" in release_matrix
    assert '--expected-python "${{ matrix.python-version }}"' in release_matrix
    assert "<<'PY'" not in release_matrix
    assert release_matrix.count("rust-toolchain: 1.87.0") == 0
    assert "python -m pip install --no-deps -e dependent" in tier1
    assert "jpl-small-bodies-de441-n16==2021.3.31.1" in tier1
    builder_name = "Build reference propagator in the Core test environment"
    preparer_name = "Prepare compatible ASSIST lock for dependent install"
    installer_name = "Install dependent and adam_core wheel"
    assert builder_name in tier1
    assert tier1.index(builder_name) < tier1.index(preparer_name)
    assert tier1.index(preparer_name) < tier1.index(installer_name)
    assert "ADAM_ASSIST_RUST_REPO: ${{ github.workspace }}/dependent" in tier1
    assert "run: pdm run assist-rust-develop" in tier1
    helper_command = (
        "python migration/scripts/develop_downstream_adam_assist.py --prepare-only"
    )
    assert tier1.count(helper_command) == 1
    assert "_write_core_patch" not in tier1
    assert "Verify Core test environment candidate installations" in tier1
    assert 'test_python="$(pdm info --python)"' in tier1
    assert (
        tier1.count('assert version("adam-core") == adam_core.__version__ == "0.7.0"')
        == 2
    )
    assert tier1.count('assert version("adam-assist") == assist_version') == 2
    assert 'assist_version == "0.5.0"' not in tier1
    assert tier1.count("-m pip check") == 2
    assert "Run Core integration tests with the reference propagator" in tier1
    assert "for test_path in \\" in tier1
    assert 'pdm run test -- "$test_path"' in tier1
    for test_path in (
        "src/adam_core/propagator/tests/test_propagator.py",
        "src/adam_core/propagator/tests/test_space_ephemeris.py",
        "src/adam_core/dynamics/tests/test_impact_viz_data.py",
        "src/adam_core/dynamics/tests/test_lambert.py",
        "src/adam_core/orbit_determination/tests/test_differential_correction.py",
        "src/adam_core/orbit_determination/tests/test_iod.py",
        "src/adam_core/orbit_determination/tests/test_least_squares.py",
        "src/adam_core/orbit_determination/tests/test_od.py",
        "src/adam_core/orbits/tests/test_oem.py",
        "src/adam_core/orbits/tests/test_spice_kernel.py",
    ):
        assert test_path in tier1

    downstream_builder = (
        ROOT / "migration/scripts/develop_downstream_adam_assist.py"
    ).read_text()
    assert "[patch.crates-io]" in downstream_builder
    for crate in (
        "adam_core",
        "adam_core_rs_autodiff",
        "adam_core_rs_coords",
        "adam_core_rs_kernel_data",
        "adam_core_rs_orbit_determination",
        "adam_core_rs_spice",
    ):
        assert crate in downstream_builder
    assert "refusing to replace downstream Cargo config" in downstream_builder
    assert '"--prepare-only"' in downstream_builder
    assert '"--precise"' not in downstream_builder
    assert 'command.extend(["--package", crate])' in downstream_builder
    assert "original_lock" in downstream_builder
    assert "_restore_downstream(prepared)" in downstream_builder
    assert "Prepare compatible ASSIST lock for Core candidate" not in release_matrix
    assert helper_command not in release_matrix
    assert "[patch.crates-io]" not in release_matrix
    assert 'version("adam-core") == adam_core.__version__ == "0.7.0"' in tier1

    coords_manifest = (ROOT / "rust/adam_core_rs_coords/Cargo.toml").read_text()
    native_module = (ROOT / "rust/adam_core_py/src/lib.rs").read_text()
    packaging_notes = (ROOT / "migration/packaging.md").read_text()
    release_notes = (ROOT / "docs/source/getting_started/releasing.rst").read_text()
    assert 'uuid = "=1.26.1"' in coords_manifest
    core_lock_versions = develop_downstream_adam_assist._locked_versions(
        develop_downstream_adam_assist._locked_packages(ROOT / "Cargo.lock")
    )
    assert core_lock_versions["uuid"] == {"1.26.1"}
    assert "#[pymodule(gil_used = true)]" in native_module
    for documentation in (packaging_notes, release_notes):
        assert "Arrow 58" in documentation
        assert "0.8.0" in documentation
    release_order = release_notes.split("Release order\n-------------\n", 1)[1]
    assert "adam_assist_ref" not in release_order
    first_release_step = release_order.split("#.", 1)[1].lstrip()
    assert first_release_step.startswith("after separate version-bump authorization")
    assert 'version("adam-assist") == assist_version' in tier1

    for workflow in workflows.glob("*.yml"):
        source = workflow.read_text()
        if "dtolnay/rust-toolchain@" not in source:
            continue
        if workflow.name == "rust-crate-release-candidate.yml":
            assert "dtolnay/rust-toolchain@stable" in source
        else:
            assert "dtolnay/rust-toolchain@stable" not in source, workflow.name
        assert "dtolnay/rust-toolchain@1.87.0" in source, workflow.name
        assert "components: rustfmt, clippy" in source, workflow.name
