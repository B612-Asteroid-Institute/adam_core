# adam-assist / assist / REBOUND CI install failure handoff

Date: 2026-05-18

## Purpose

This document captures the current CI dependency failure affecting the `rust-migration-waves-d-e` branch, the evidence collected so far, and the recommended fix path so another agent can continue independently once new `adam-assist` versions are available.

## Current branch state

- Repository: `B612-Asteroid-Institute/adam_core`
- Branch: `rust-migration-waves-d-e`
- Relevant recent commits:
  - `a4b5d88e Align standalone propagator planning`
  - `db859275 Fix propagator review follow-ups`
- Local validation for the docs/planning work passed, but GitHub CI is red during dependency installation before project tests run.

## Symptom

GitHub Actions fails while installing the `test` dependency group:

- CI run `26029276496`: failed.
- Tier-1 Dependent Smoke run `26029276474`: failed.
- Earlier runs `26027998870` and `26027998944` failed the same way.

Representative failure from the GitHub logs:

```text
✖ Install assist 1.2.0 failed
...
gcc ... -I/tmp/pdm-build-env-.../site-packages/rebound -c src/ascii_ephem.c ...
In file included from src/spk.h:10,
                 from src/ascii_ephem.c:12:
src/assist.h:30:10: fatal error: rebound.h: No such file or directory
   30 | #include "rebound.h"
      |          ^~~~~~~~~~~
compilation terminated.
error: command '/usr/bin/gcc' failed with exit code 1
```

The failure occurs before adam-core tests, docs, or Rust code run. It is dependency/build hygiene, not a regression in the standalone Rust SPICE/time/propagator planning work.

## Dependency chain

Current adam-core `test` dependency group includes:

```toml
"adam-assist>=0.3.5"
```

The latest published `adam-assist` at inspection time is `0.3.8`. Its metadata depends on:

```text
assist==1.2.0
rebound>=4.4.10
```

`assist 1.2.0` has build metadata in `setup.py` equivalent to:

```python
setup_requires=['rebound>=4.4.11', 'numpy']
install_requires=['rebound>=4.4.11', 'numpy']
```

Because the `setup_requires` constraint has no upper bound, the isolated build environment for `assist` can choose `rebound 5.0.0` even when the main PDM environment is constrained differently.

## Root cause

REBOUND changed where `rebound.h` is packaged:

- `rebound 4.6.0` Linux wheels contain `rebound/rebound.h`.
- `rebound 5.0.0` Linux wheels contain `src/rebound.h`.

`assist 1.2.0` computes the include directory from `inspect.getfile(rebound)`, appends the package directory itself, and includes:

```c
#include "rebound.h"
```

That works with REBOUND 4.x because `rebound.h` is directly under the `rebound` package directory. It fails with REBOUND 5.0.0 because the header is under `src/`.

## What was already tried

Commit `db859275` added a direct adam-core test dependency:

```toml
"rebound>=4.4.11,<5"
```

This made PDM install `rebound 4.6.0` in the **main** environment, as shown in the latest CI logs:

```text
✔ Install rebound 4.6.0 successful
```

However, CI still failed because `assist 1.2.0` builds inside PDM's isolated build environment, and its own `setup_requires=['rebound>=4.4.11', ...]` remains unconstrained. The direct adam-core pin is therefore useful for the main env but insufficient for assist's isolated build step.

A local temporary PDM project confirmed that installing `assist==1.2.0` with an explicit top-level `rebound>=4.4.11,<5` can succeed in a simple environment, but GitHub/PDM's isolated build behavior for the adam-core group still failed. Do not treat the direct top-level pin alone as a complete CI fix.

## Important package-version findings

PyPI versions inspected on 2026-05-18:

- `assist`: latest `1.2.3`
- `adam-assist`: latest `0.3.8`
- `rebound`: latest `5.0.0`

`assist 1.2.1` and `assist 1.2.2` still have unconstrained REBOUND build/runtime requirements:

```python
setup_requires=['rebound>=4.4.11', 'numpy']
install_requires=['rebound>=4.4.11', 'numpy']
```

`assist 1.2.3` fixes this by adding an upper bound:

```python
setup_requires=['rebound>=4.4.11,<5.0.0', 'numpy']
install_requires=['rebound>=4.4.11,<5.0.0', 'numpy']
tests_require=['numpy', 'matplotlib', 'rebound>=4.4.11,<5.0.0']
```

This is the key upstream fix.

## Preferred solution

Release a new `adam-assist` version that depends on `assist 1.2.3` or later in the 1.2 line.

Recommended `adam-assist` dependency update:

```toml
assist >= 1.2.3, < 1.3
```

or, if exact pins are preferred:

```toml
assist == 1.2.3
```

Optionally add the explicit REBOUND cap in `adam-assist` too for clarity and resolver hygiene:

```toml
rebound >= 4.4.11, < 5
```

After publishing the new `adam-assist` release, update adam-core's `pyproject.toml` test dependency from:

```toml
"adam-assist>=0.3.5"
```

to the new version, for example:

```toml
"adam-assist>=0.3.9"
```

Then run:

```bash
uv lock
pdm run script-preflight
```

and push to exercise GitHub CI.

The direct adam-core `"rebound>=4.4.11,<5"` test dependency added in `db859275` can either remain as an explicit guard or be removed once the new `adam-assist` metadata owns the constraint. Prefer keeping dependency ownership in `adam-assist` if possible because it is the package that actually requires ASSIST/REBOUND compatibility.

## Temporary adam-core-only workaround if no new adam-assist release is available

PDM supports resolver overrides. A temporary adam-core-only workaround is:

```toml
[tool.pdm.resolution.overrides]
assist = "==1.2.3"
```

A temporary PDM project using:

```toml
[dependency-groups]
test = [
  "adam-assist>=0.3.5",
]

[tool.pdm.resolution.overrides]
assist = "==1.2.3"
```

successfully locked `assist 1.2.3`, overriding the transitive `adam-assist 0.3.8` `assist==1.2.0` pin.

This workaround is less desirable than a new `adam-assist` release because it makes adam-core override a dependent package's transitive dependency. Use it only if unblocking CI is urgent before `adam-assist` is released.

## Validation plan after new adam-assist release

1. Update `pyproject.toml`:

   ```toml
   "adam-assist>=<new-version>"
   ```

2. Decide whether to keep or remove the direct adam-core REBOUND cap:

   ```toml
   "rebound>=4.4.11,<5"
   ```

   Recommended: remove it if the new `adam-assist` package owns the `assist>=1.2.3` dependency and any needed REBOUND cap; keep it only if CI still needs an explicit top-level guard.

3. Refresh lock metadata:

   ```bash
   uv lock
   uv lock --check
   ```

4. Validate local project metadata and scripts:

   ```bash
   python3 - <<'PY'
   import tomllib
   from pathlib import Path
   for path in ['pyproject.toml', 'uv.lock']:
       with Path(path).open('rb') as f:
           tomllib.load(f)
   print('toml parse ok')
   PY
   pdm run script-preflight
   git diff --check
   ```

5. Validate a clean temp dependency install that mirrors the relevant CI dependency group enough to prove assist imports:

   ```bash
   tmp=$(mktemp -d /tmp/adam-assist-ci-check.XXXXXX)
   cat > "$tmp/pyproject.toml" <<'TOML'
   [project]
   name = "adam-assist-ci-check"
   version = "0.0.0"
   requires-python = ">=3.11,<3.14"
   dependencies = []

   [dependency-groups]
   test = [
     "adam-assist>=<new-version>",
   ]
   TOML
   cd "$tmp"
   PDM_IGNORE_ACTIVE_VENV=1 pdm install -G test
   pdm run python - <<'PY'
   import assist
   import rebound
   import adam_assist
   print('imports ok', rebound.__version__)
   PY
   ```

6. Push and check GitHub Actions:

   ```bash
   gh run list -R B612-Asteroid-Institute/adam_core --branch rust-migration-waves-d-e --limit 6
   ```

Expected CI result: the install phase should no longer fail on `assist` / `rebound.h`. If new failures occur after install, treat them as the next independent CI issue.

## Related GitHub runs and commits

- `26027998870` CI: failed on `assist 1.2.0` / `rebound.h`.
- `26027998944` Tier-1: failed on same issue.
- `26029276496` CI after `db859275`: still failed on same issue, despite main-env `rebound 4.6.0`.
- `26029276474` Tier-1 after `db859275`: still failed on same issue.
- `db859275 Fix propagator review follow-ups`: added direct `rebound>=4.4.11,<5`, swept stale planning docs, and tightened propagator RFC pseudocode. The dependency pin in this commit did not fully fix CI because assist's isolated build env still used its own unconstrained setup requirements.

## Summary recommendation

Wait for or create a new `adam-assist` release that depends on `assist>=1.2.3,<1.3` (or `assist==1.2.3`). Then update adam-core to require that new `adam-assist` version and rerun CI. This keeps dependency ownership where it belongs and avoids maintaining a PDM transitive override in adam-core.
