Rust Backend Contracts
======================

This document defines boundary and behavior contracts for migrated APIs in ``adam_core``.

Boundary Selection Rule
-----------------------

- Use a NumPy boundary for dense numerical kernels where contiguous array inputs are natural.
- Use an Arrow boundary for table-centric APIs that require schema/null semantics.
- For high-level entrypoints, use a single Python->Rust boundary crossing per call and execute the full internal pipeline in Rust before returning results.

Migration Surface Scope Rule
----------------------------

- Default migration target is the highest-level atomic function entrypoint used by ``adam-core`` callers or downstream users.
- Internal-only helper functions behind that entrypoint are not required migration targets by default.
- Helper-level Rust replacements should be added only when they are needed for measurable performance/correctness gains and do not fragment the public migration surface.
- High-level orchestrators (for example ``transform_coordinates``) are only considered migrated when orchestration + sub-transform execution happen end-to-end in Rust without Python<->Rust ping-pong.

Current Migrated APIs
---------------------

- ``coordinates.cartesian_to_spherical``
  - Boundary: NumPy ``float64`` array with shape ``(N, 6)``.
  - Rust entrypoint: ``adam_core._rust_native.cartesian_to_spherical_numpy``.
  - Error behavior: raises ``ValueError`` on shape mismatch.

- ``coordinates.cartesian_to_geodetic``
  - Boundary: NumPy ``float64`` array with shape ``(N, 6)`` and scalar ``a``/``f`` parameters.
  - Rust entrypoint: ``adam_core._rust_native.cartesian_to_geodetic_numpy``.
  - Error behavior: raises ``ValueError`` on shape mismatch.

- ``coordinates.cartesian_to_keplerian``
  - Boundary: NumPy ``float64`` arrays with shape ``(N, 6)`` for ``coords`` and ``(N,)`` for ``t0``/``mu``.
  - Rust entrypoint: ``adam_core._rust_native.cartesian_to_keplerian_numpy``.
  - Error behavior: raises ``ValueError`` on shape/length mismatch.

- ``coordinates.keplerian.to_cartesian``
  - Boundary: NumPy ``float64`` arrays with shape ``(N, 6)`` for ``coords`` and ``(N,)`` for ``mu``.
  - Rust entrypoint: ``adam_core._rust_native.keplerian_to_cartesian_numpy``.
  - Error behavior: raises ``ValueError`` on shape/length mismatch.

- ``coordinates.transform_coordinates`` (partial Rust high-level dispatcher)
  - Boundary: Python coordinate tables, with a single internal crossing into ``adam_core._rust_native.transform_coordinates_numpy`` for supported paths.
  - Current Rust-supported single-crossing paths: ``cartesian|spherical|keplerian -> cartesian|spherical|geodetic|keplerian`` with unchanged origin and NaN covariances; frame support includes unchanged frame and ``equatorial <-> ecliptic``.
  - Unsupported paths remain whole-call Python execution (no mixed Python<->Rust ping-pong within one call).

- ``coordinates.spherical.from_cartesian``
  - Boundary: Arrow table with columns ``x, y, z, vx, vy, vz`` (float64-compatible).
  - Rust entrypoint: ``adam_core._rust.api.cartesian_to_spherical_arrow`` -> ``adam_core._rust_native.cartesian_to_spherical_numpy``.
  - Null behavior: Arrow nulls are converted to ``NaN`` in the numeric kernel path.
  - Error behavior: raises ``ValueError`` when required columns are missing.

- ``coordinates.spherical.to_cartesian``
  - Boundary: NumPy ``float64`` array with shape ``(N, 6)``.
  - Rust entrypoint: ``adam_core._rust_native.spherical_to_cartesian_numpy``.
  - Error behavior: raises ``ValueError`` on shape mismatch.

- ``dynamics.calc_mean_motion``
  - Boundary: NumPy ``float64`` arrays with shape ``(N,)`` for ``a`` and ``mu``.
  - Rust entrypoint: ``adam_core._rust_native.calc_mean_motion_numpy``.
  - Error behavior: raises ``ValueError`` when lengths differ.

- ``orbit_determination.calcGibbs``
  - Boundary: NumPy ``float64`` arrays with shape ``(3,)`` for ``r1``, ``r2``, and ``r3``.
  - Rust entrypoint: ``adam_core._rust_native.calc_gibbs_numpy``.
  - Error behavior: raises ``ValueError`` on shape mismatch.

- ``orbit_determination.calcHerrickGibbs``
  - Boundary: NumPy ``float64`` arrays with shape ``(3,)`` for ``r1``, ``r2``, and ``r3``; ``t1``, ``t2``, ``t3`` as scalar float inputs.
  - Rust entrypoint: ``adam_core._rust_native.calc_herrick_gibbs_numpy``.
  - Error behavior: raises ``ValueError`` on shape mismatch.

- ``orbit_determination.calcGauss``
  - Boundary: NumPy ``float64`` arrays with shape ``(3,)`` for ``r1``, ``r2``, and ``r3``; ``t1``, ``t2``, ``t3`` as scalar float inputs.
  - Rust entrypoint: ``adam_core._rust_native.calc_gauss_numpy``.
  - Error behavior: raises ``ValueError`` on shape mismatch.

- ``orbit_determination.gaussIOD``
  - Boundary: NumPy arrays for ``coords`` ``(3, 2)``, ``observation_times`` ``(3,)``, and ``coords_obs`` ``(3, 3)``.
  - Rust entrypoint: ``adam_core._rust_native.gauss_iod_orbits_numpy`` (candidate-generation kernel from precomputed roots).
  - Backend default: legacy path, with Rust path held in ``dual`` mode pending +20% p50/p95 perf gate.
  - Error behavior: velocity-method validation raises ``ValueError`` for unsupported solver names.

Fallback and Waivers
--------------------

- If Rust backend cannot satisfy parity/performance criteria, keep legacy behavior under waiver control.
- Waivers are tracked in ``migration/waivers.yaml`` with owner and review date.

Status Registry
---------------

- Single source of truth for migration state: ``adam_core._rust.status`` (``API_MIGRATIONS``).
- ``API_MIGRATION_STATUS`` is a compatibility projection of the same data used by runtime dispatch.
- Migration governance scripts (``migration/scripts/*``) import from this module; there is no separate YAML registry.

Validation Contract
-------------------

- The standard Python test suite command ``pytest --benchmark-skip -m 'not profile'`` must run in a Rust-enabled environment for migration validation.
- Rust-required runs must fail fast if the Rust extension is unavailable.
- High-level migrated APIs must include a contract test that enforces single-crossing execution (one Python->Rust entry and one Rust->Python return).

Engineering Requirements Contract
---------------------------------

- This migration is bound to ``/Users/aleck/Code/AGENTS.md`` as a normative engineering policy.
- Control flow must prioritize readability: guard clauses, early returns, and decomposition of large functions.
- Prefer functions over classes unless using classes is clearly less awkward for the local design.
- Reuse existing functionality before adding new entrypoints; avoid duplicate abstractions.
- Do not add fallback behavior unless explicitly requested; fail loudly when contracts are not met.
- Prefer vectorized and batched operations in Python-facing numerical paths.
- Keep scripts minimal: use inline terminal checks for quick validation and avoid leaving throwaway top-level scripts.
- Require Python type hints in migration code, with strong concrete types when feasible.
