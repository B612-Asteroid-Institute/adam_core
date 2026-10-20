from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import pytest

from migration.scripts import develop_downstream_adam_assist, verify_preview_versions
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
        ("0.6.1", "0.6.1"),
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
    assert MODULE._python_version_tag_glob("0.6.1") == "v0.6.*"


def test_stable_release_versions_align_python_and_public_rust_crates() -> None:
    verify_preview_versions.verify(ROOT, "0.6.1", "0.6.1", "stable")
    with pytest.raises(ValueError, match="not a prerelease"):
        verify_preview_versions.verify(ROOT, "0.6.1", "0.6.1", "preview")


def test_runtime_versions_must_match_distribution_metadata() -> None:
    versions = {"adam-core": "0.6.1", "adam-assist": "0.5.0"}
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
    writer = "python migration/scripts/write_maturin_version.py"
    builder = "uses: PyO3/maturin-action@e83996d129638aa358a18fbd1dfb82f0b0fb5d3b"
    inspector = "python migration/scripts/check_wheel_artifacts.py"

    assert workflow.index(writer) < workflow.index(builder) < workflow.index(inspector)
    assert "adam_assist_ref:" in workflow
    assert "ADAM_ASSIST_REF: ${{ inputs.adam_assist_ref }}" in workflow
    assert 'PYTHON_PREVIEW_VERSION: "0.6.1"' in workflow

    extended_coverage = workflow.split("  extended-coverage:", 1)[1]
    assert "adam-assist" not in extended_coverage
    assert "adam_assist" not in extended_coverage
    assert "assist-rust-develop" not in extended_coverage


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

    workflow = (ROOT / ".github/workflows/rust-crate-release-candidate.yml").read_text()
    assert "Verify clean no-lock Rust 1.87 consumer resolution" in workflow
    assert 'test ! -e "$consumer/Cargo.lock"' in workflow
    assert 'rustc --version | grep -F "rustc 1.87.0"' in workflow
    assert "clean Rust 1.87 consumer resolved ICU4X 2.2.0 and rawzip 0.5.2" in workflow
    assert 'if package["name"] in {"zip", "zopfli"}' in workflow
    assert 'cargo package -p "$crate" --locked' in workflow
    assert 'RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps' in workflow
    assert "Latest-stable Rust compatibility (non-authoritative)" in workflow
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
    assert "default: 0.6.1" in publisher
    assert 'PYTHON_SOURCE_VERSION: "0.6.1"' in publisher
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
    config, remove_cargo_dir = develop_downstream_adam_assist._write_core_patch(
        tmp_path
    )
    assert remove_cargo_dir
    source = config.read_text()
    assert source.startswith("[patch.crates-io]\n")
    assert "adam_core =" not in source
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
            *_core_packages("0.6.1"),
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
        develop_downstream_adam_assist, "_current_core_version", lambda: "0.6.1"
    )

    develop_downstream_adam_assist._refresh_core_lock(tmp_path, {})

    expected = ["/cargo", "update", "--manifest-path", str(manifest)]
    for crate in develop_downstream_adam_assist.PUBLIC_CORE_CRATES:
        expected.extend(["--package", crate])
    assert commands == [expected]
    assert "--precise" not in commands[0]


def test_patched_core_lock_verification_requires_one_local_aligned_package() -> None:
    valid = _core_packages("0.6.1")
    develop_downstream_adam_assist._verify_patched_core_packages(valid, "0.6.1")

    wrong_version = [dict(package) for package in valid]
    wrong_version[0]["version"] = "0.6.0"
    with pytest.raises(SystemExit, match="exactly one local-path Core 0.6.1"):
        develop_downstream_adam_assist._verify_patched_core_packages(
            wrong_version, "0.6.1"
        )

    registry_source = [dict(package) for package in valid]
    registry_source[0][
        "source"
    ] = "registry+https://github.com/rust-lang/crates.io-index"
    with pytest.raises(SystemExit, match="exactly one local-path Core 0.6.1"):
        develop_downstream_adam_assist._verify_patched_core_packages(
            registry_source, "0.6.1"
        )

    duplicate = [*valid, dict(valid[0])]
    with pytest.raises(SystemExit, match="exactly one local-path Core 0.6.1"):
        develop_downstream_adam_assist._verify_patched_core_packages(duplicate, "0.6.1")

    missing = valid[1:]
    with pytest.raises(SystemExit, match="exactly one local-path Core 0.6.1"):
        develop_downstream_adam_assist._verify_patched_core_packages(missing, "0.6.1")


def test_unrelated_lock_guard_allows_removals_but_rejects_added_versions() -> None:
    before = [
        *_core_packages("0.6.0"),
        _package("retained", "1.0.0"),
        _package("retained", "2.0.0"),
        _package("removed", "1.0.0"),
    ]
    accepted = [
        *_core_packages("0.6.1"),
        _package("retained", "1.0.0"),
        _package("rawzip", "0.5.2"),
    ]
    develop_downstream_adam_assist._reject_unrelated_version_additions(before, accepted)

    added_version = [*accepted, _package("retained", "3.0.0")]
    with pytest.raises(SystemExit, match="added unrelated package versions"):
        develop_downstream_adam_assist._reject_unrelated_version_additions(
            before, added_version
        )


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
    assist_sha = "c06164a18e8d90d8bf83d7ab58e479dc97e1be41"

    assert "dtolnay/rust-toolchain@1.87.0" in normal_ci
    assert "components: rustfmt, clippy" in normal_ci
    assert "dtolnay/rust-toolchain@1.87.0" in crate_ci
    assert "components: rustfmt, clippy" in crate_ci
    assert "dtolnay/rust-toolchain@1.87.0" in tier1
    assert "adam_assist_ref:" in release_matrix
    assert assist_sha in tier1
    assert "adam-assist" not in normal_ci
    assert "adam_assist" not in normal_ci
    assert "adam-assist" not in nightly
    assert "adam_assist" not in nightly
    assert "Build adam-assist manylinux wheel" in release_matrix
    assert "Build adam-assist native wheel" in release_matrix
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
        tier1.count('assert version("adam-core") == adam_core.__version__ == "0.6.1"')
        == 2
    )
    assert (
        tier1.count('assert version("adam-assist") == assist_version == "0.5.0"') == 2
    )
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
    assert "Prepare compatible ASSIST lock for Core candidate" in release_matrix
    assert release_matrix.count(helper_command) == 1
    assert "[patch.crates-io]" not in release_matrix
    matrix_preparer = release_matrix.index(
        "Prepare compatible ASSIST lock for Core candidate"
    )
    assert release_matrix.index("Set up Rust") < matrix_preparer
    assert matrix_preparer < release_matrix.index("Build adam-assist manylinux wheel")
    assert matrix_preparer < release_matrix.index("Build adam-assist native wheel")
    assert 'version("adam-core") == adam_core.__version__ == "0.6.1"' in tier1
    assert 'version("adam-assist") == assist_version == "0.5.0"' in tier1

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
