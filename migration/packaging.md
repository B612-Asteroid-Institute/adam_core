# adam-core Rust Packaging Notes

Last updated: 2026-11-10.

## Supported Build Path

- `pdm run wheel-build` is the authoritative local and CI Core-wheel command.
  It invokes `migration/scripts/build_release_wheel.py`; direct release
  `maturin build`, `pdm build`, or a hand-assembled release environment is not
  a supported release path.
- The wrapper mirrors the Cargo version, creates a unique per-invocation source
  digest and nonce immediately before Maturin, builds, and runs
  `check_wheel_artifacts.py` before returning success.
- `pdm run wheel-inspect` may be run again before upload. It rejects build-only
  runtime dependencies, packaged tests, version/native-extension errors, a
  missing release attestation, or any packaged Python file/hash not present in
  that attestation.
- `pdm run rust-build` is an alias for `pdm run wheel-build`. For explicit
  manylinux arguments call the same wrapper, for example
  `python migration/scripts/build_release_wheel.py --out wheelhouse -- --zig
  --target aarch64-unknown-linux-gnu --manylinux 2_17 -i python3.14`.

## Version Source

- The wheel version source of truth is `rust/adam_core_py/Cargo.toml` `[package].version`.
- This is intentional because maturin uses that Cargo version for wheel metadata when `project.version` is dynamic.
- `migration/scripts/write_maturin_version.py` mirrors the Cargo version into `src/adam_core/_version.py` before build.
- The Python distribution and unpublished `adam_core_py` wheel crate share a version. Supported Cargo prereleases are normalized explicitly, for example `0.5.6-rc.1` to PEP 440 `0.5.6rc1`.
- The published `0.7.0` release shipped on 2026-11-06 as 12 CPython 3.11-3.13 wheels and six public Rust crates built against Arrow 54. Its files, versions, and `v0.7.0` tag are immutable.
- Version `0.8.0` adds CPython 3.14 and aligns the Python distribution, `adam_core_py`, and all six public Rust crates. This 0.x minor release is semver-breaking because `adam_core_rs_coords` exposes Arrow types that move from Arrow 54 to Arrow 58 and the public Rust OD surface changes. The expected `v0.8.0` tag date is 2026-11-10.
- `pyproject.toml` does not declare `[tool.pdm.version]`; PDM SCM versioning is not part of the native wheel path.
- If the checkout is exactly on a `vX.Y.Z` or prerelease tag, `write_maturin_version.py` fails when the normalized tag and Cargo versions differ.
- `migration/scripts/verify_preview_versions.py` enforces explicit preview/stable channel semantics and exact public-crate pins. Stable version, tag, protected-environment, resumable-publication, and release-order policy is canonical in `docs/source/getting_started/releasing.rst`.

## uv Status

- `uv lock --check` is expected to pass with the workspace source mapping for `adam-core`; test-only dependents must resolve the editable checkout, never the public distribution.
- `uv export --frozen --only-group build` must contain exactly the three direct build-tool pins declared in `pyproject.toml`, including `packaging==25.0` from the public registry with hashes.
- `pdm run wheel-build` remains the sole authoritative build command.
- A clean-wheel validation with uv must run outside the repository and pass
  `--no-config`, for example `uv pip install --no-config --python <venv-python>
  dist/adam_core-<version>-<python>-<abi>-<platform>.whl`. Running outside the
  checkout prevents workspace source configuration from replacing the wheel
  under test.
- Validate the installed distribution/native identity and run
  `uv pip check --no-config --python <venv-python>` after installation.
- Do not use `maturin develop --uv` as release evidence.

## Core-before-Assist compatibility sequencing

- `.github/workflows/tier1-dependent-smoke.yml` is manual-only. It rejects a
  non-40-hex Assist revision before checkout, verifies the resulting HEAD, and
  requires the dependent to declare Core 0.8 compatibility. `pip check` then
  enforces that unmodified requirement. Use this workflow after the downstream
  candidate is updated; Core publication does not depend on it. Add no
  pull-request or scheduled trigger until a separately reviewed compatible
  Assist candidate exists.
- Removing the former `tier1-smoke` pull-request job also removes that check
  context from new PR commits. Public repository API results verified on
  2026-11-03 showed that `main` was unprotected and that no repository rulesets
  applied, so no required `tier1-smoke` context existed at that date. Recheck
  branch protection and rulesets if repository policy changes.
- The release-candidate wheel matrix remains a separate manually dispatched
  workflow. It builds and cleanly accepts one Core wheel for every CPython
  3.11-3.14 and supported-platform cell with identical Core-owned checks. The
  workflow delegates fresh-venv installation, identity, platform-tag, and native
  smoke validation to the dedicated `run_core_wheel_acceptance.py` and copied
  `core_wheel_smoke.py` scripts rather than embedding test programs in YAML.
  Assist and other downstream
  projects validate Core compatibility in their own repositories; they are not
  built or lock-patched by the Core release matrix.

## Expected Validation Shape

- Normal CI runs the deterministic Python suite on CPython 3.11-3.14; only the
  CPython 3.12 row collects coverage. Lint and documentation gates also run on
  CPython 3.12.
- The release-candidate workflow first builds and accepts the uniform 16-wheel
  artifact matrix. Its dependent extended-coverage job runs the six
  profile-marked scientific fixtures, both live Rust kernel-data integrations,
  and the live Horizons/NEOCC/Scout/SBDB integration tests on CPython 3.12.
- Dedicated benchmark jobs execute every benchmark row.
- No expected xfail remains in the rotation-period confidence suite: every committed standard-candle fixture enforces the strict confidence contract.
- To inspect skip reasons, run `pdm run test-rust-full -- -rs`.


## Source inventory and native identity

Bare `cargo`, Clippy, and Rust test builds identify the current commit and watch
Git refs plus exact tracked inputs. They do **not** watch a nonexistent
sentinel, so repeated unchanged developer builds can reach `Fresh`.

Every official release invocation instead uses the audited wrapper. Immediately
before Cargo it requires a clean tracked checkout; enumerates tracked inputs and
all package/Rust source directly (including ignored files); writes a full source
digest, unique nonce, and exact Python-file hashes; and passes those values to
Cargo. The build script verifies and consumes that single-use attestation. A
second direct release build cannot reuse it. The packaged JSON attestation is
then checked against the wheel's exact Python file set and content hashes.
Missing, mismatched, reused, or stale attestations fail closed.

Source-overlay `run_od` performs an independent runtime check: checkout `HEAD`
must equal the embedded native commit, tracked/staged runtime source must be
clean, and no untracked or ignored package Python/Rust source may be present.
Generated `_version.py`, native-extension artifacts, tests, caches, `target/`,
wheelhouse, and the independently pinned nested `adam-assist/` checkout are
excluded where appropriate. Installed wheels have no checkout and rely on the
embedded exact release identity and packaged attestation.
