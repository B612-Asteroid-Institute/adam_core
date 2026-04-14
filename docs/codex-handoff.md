# Objective
Implement and harden `adam_core` rotation-period estimation with a Fourier-based solver that is scientifically usable on Rubin/LSST-like sparse cadence data, while improving agreement with Greenstreet et al. (2026) and keeping the core codebase focused (moving one-off report/test artifacts out of repo scope).

# Current Status
- What is complete. The core solver and API are implemented (`estimate_rotation_period`, detection wrappers, grouped wrappers), the main Fourier path was refactored and reduced in size, shared kernel primitives were split into `rotation_period_fourier_core.py`, and the primary core unit suite passes (`27 passed`).
- What is in progress. Paper-faithful alignment is not finished (remaining method gaps vs paper), and the repo cleanup is mid-stream (one-off/diagnostic files have been moved out and are currently staged as deletions, pending commit decisions).

# Decisions & Constraints
- Key technical decisions and why. Kept public API stable; made `surrogate_refine` the effective optimized path; retained `"coarse_to_fine"` only as a compatibility alias to reduce maintenance burden; separated low-level Fourier primitives into `rotation_period_fourier_core.py` to cut duplication in `rotation_period_fourier.py`; moved exploratory/diagnostic report generators and optional paper/PDS regression artifacts to an external archive folder to keep repo surface clean.
- Non-negotiable constraints/requirements. Follow the papers more closely in next iteration; keep core solver testable and deterministic; avoid committing one-off generated HTML/CSV/report artifacts into repo; preserve exact reproducibility commands for core verification.

# Changed Files
- `src/adam_core/photometry/rotation_period_fourier.py` — refactored main solver/orchestration, removed duplicated kernel code, removed separate coarse-to-fine implementation, kept API compatibility.
- `src/adam_core/photometry/rotation_period_fourier_core.py` — added shared Fourier kernel primitives/types used by main solver (new dependency file).
- `src/adam_core/photometry/tests/conftest.py` — removed CLI options for moved optional rotation-period suites.
- `docs/generate_rotation_period_2025_mm81_gallery.py` — moved out of repo (one-off report generation).
- `docs/generate_rotation_period_plotly_gallery.py` — moved out of repo (one-off report generation).
- `docs/generate_rotation_period_x05_gallery.py` — moved out of repo (one-off report generation).
- `docs/rotation_period_validation.md` — moved out of repo (project narrative writeup, not core package docs).
- `src/adam_core/photometry/tests/data/generate_mpc_rotation_period_fixtures.py` — moved out of repo (optional fixture generation path).
- `src/adam_core/photometry/tests/data/generate_pds_rotation_period_fixtures.py` — moved out of repo (optional fixture generation path).
- `src/adam_core/photometry/tests/data/rotation_period_fixture_1011_071_R.npz` — moved out of repo (optional real-data fixture).
- `src/adam_core/photometry/tests/data/rotation_period_fixture_289_071_R.npz` — moved out of repo (optional real-data fixture).
- `src/adam_core/photometry/tests/data/rotation_period_fixture_702_071_R.npz` — moved out of repo (optional real-data fixture).
- `src/adam_core/photometry/tests/data/rotation_period_pds_fixture_1011_Laodamia.npz` — moved out of repo (optional PDS fixture).
- `src/adam_core/photometry/tests/data/rotation_period_pds_fixture_1282_Utopia.npz` — moved out of repo (optional PDS fixture).
- `src/adam_core/photometry/tests/data/rotation_period_pds_fixture_1323_Tugela.npz` — moved out of repo (optional PDS fixture).
- `src/adam_core/photometry/tests/data/rotation_period_pds_fixture_1627_Ivar.npz` — moved out of repo (optional PDS fixture).
- `src/adam_core/photometry/tests/data/rotation_period_pds_fixture_289_Nenetta.npz` — moved out of repo (optional PDS fixture).
- `src/adam_core/photometry/tests/data/rotation_period_pds_fixture_511_Davida.npz` — moved out of repo (optional PDS fixture).
- `src/adam_core/photometry/tests/data/rotation_period_pds_fixture_702_Alauda.npz` — moved out of repo (optional PDS fixture).
- `src/adam_core/photometry/tests/data/rotation_period_pds_fixture_886_Washingtonia.npz` — moved out of repo (optional PDS fixture).
- `src/adam_core/photometry/tests/test_rotation_period_pds.py` — moved out of repo (optional PDS regression suite).
- `src/adam_core/photometry/tests/test_rotation_period_real_data.py` — moved out of repo (optional real-data regression suite).
- `uv.lock` — present as untracked env lock artifact; not part of solver logic.

# Open Issues / Risks
- `rotation_period_fourier_core.py` is now required by `rotation_period_fourier.py`; if not committed together, imports will break.
- Agreement with paper results is still incomplete for a subset of objects, especially alias/harmonic-adjacent cases.
- Some paper-described steps are still not mirrored exactly (manual adjudication workflow, reliability-threshold process, and full paper-specific preprocessing details).
- Moving optional real-data/PDS tests out of repo reduces in-repo scientific regression coverage until a final policy is chosen.

# Next Steps
- I will dictate these to the next chat

# Run / Verify
- Lint: `cd /Users/natetellis/codex_workspaces/adam_rotation_period_analysis/adam_core && .venv/bin/ruff check src/adam_core/photometry/rotation_period_fourier.py src/adam_core/photometry/tests/conftest.py` (currently passes).
- Bytecode check: `cd /Users/natetellis/codex_workspaces/adam_rotation_period_analysis/adam_core && .venv/bin/python -m py_compile src/adam_core/photometry/rotation_period_fourier.py src/adam_core/photometry/rotation_period_fourier_core.py` (currently passes).
- Core unit tests: `cd /Users/natetellis/codex_workspaces/adam_rotation_period_analysis/adam_core && .venv/bin/pytest -q src/adam_core/photometry/tests/test_rotation_period.py` (currently `27 passed`, 1 Ray warning).
- Optional moved suites: `test_rotation_period_pds.py` and `test_rotation_period_real_data.py` were moved to archive and are not currently runnable in-repo without restoring those files/fixtures.
