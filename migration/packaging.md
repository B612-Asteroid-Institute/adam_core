# adam-core Rust Packaging Notes

Last updated: 2026-04-28.

## Supported Build Path

- `pdm run wheel-build` is the authoritative local and CI wheel build command.
- `pdm run wheel-build` runs `pdm run wheel-version`, then `pdm build --no-sdist --dest dist`.
- `pdm run wheel-inspect` must pass before uploading or using `dist/*.whl` as a release artifact.
- `pdm run rust-build` is an alias for `pdm run wheel-build`.

## Version Source

- The wheel version source of truth is `rust/adam_core_py/Cargo.toml` `[package].version`.
- This is intentional because maturin uses that Cargo version for wheel metadata when `project.version` is dynamic.
- `migration/scripts/write_maturin_version.py` mirrors the Cargo version into `src/adam_core/_version.py` before build.
- `pyproject.toml` does not declare `[tool.pdm.version]`; PDM SCM versioning is not part of the native wheel path.
- If the checkout is exactly on a `vX.Y.Z` release tag, `write_maturin_version.py` fails when the tag version and Cargo version differ.

## uv Status

- `uv lock --check` is expected to pass with the workspace override for `adam-core`.
- PDM plus standard pip are currently the authoritative local build/install path.
- Do not use `maturin develop --uv` or direct `uv pip install dist/*.whl` as the authoritative local validation path until uv local-wheel behavior is revalidated on the intended current uv release.
- The observed local caveat was with `uv 0.6.16`: direct uv local-wheel/develop testing selected or retained PyPI `adam_core 0.5.5`, which did not contain `adam_core._rust_native`.

## Expected Validation Shape

- `pdm run test-rust-full` currently runs under Python 3.13 in this checkout.
- The observed full-suite shape after the baseline merge is `708 passed, 144 skipped, 2 deselected`.
- The 144 skips are expected for the current command: 139 benchmark cases skipped by `--benchmark-skip`, two explicitly skipped Lambert scenarios, and three optional PYOORB cases.
- The two deselected tests are profile-marked tests excluded by `-m 'not profile'`.
- To inspect skip reasons, run `pdm run test-rust-full -- -rs`.
