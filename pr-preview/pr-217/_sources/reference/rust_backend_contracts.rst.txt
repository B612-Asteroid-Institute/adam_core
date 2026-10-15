Rust Backend Contracts
======================

This document defines boundary and behavior contracts for migrated APIs in ``adam_core``.

Runtime Availability Contract
-----------------------------

- The compiled ``adam_core._rust_native`` extension is mandatory for this
  migration branch.
- Importing ``adam_core._rust.api`` raises ``ImportError`` immediately if the
  native extension is missing or if the installed extension lacks a required
  native symbol.
- Python wrappers around native kernels return concrete results or propagate
  native exceptions. They do not return ``None`` to signal backend
  unavailability.
- CI and local validation scripts rely on the mandatory import contract; they
  no longer need ``ADAM_CORE_REQUIRE_RUST_BACKEND``.

Boundary Selection Rule
-----------------------

- Use a NumPy boundary for dense numerical kernels where contiguous array inputs are natural.
- Use an Arrow boundary for table-centric APIs that require schema/null semantics.
- For high-level entrypoints, use a single Python->Rust boundary crossing per call and execute the full internal pipeline in Rust before returning results.

Complete Surface Status
-----------------------

The authoritative latest-main inventory is
``migration/public_surface/manifest.json`` (595 symbols / 67 constants at
upstream ``9b756803ab3afbe11e33df9e57d30a28e7976b92``). The
selected parity registry is a benchmark set, not the public-surface count.
Every adam-core-owned non-plotting operation is either Rust-backed, a thin
single-crossing compatibility veneer, or an explicitly documented provider
boundary. The API examples below describe boundary contracts; they are not an
exhaustive migration-status list.

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

- ``coordinates.transform_coordinates`` (Rust high-level dispatcher with explicit provider boundaries)
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

- ``coordinates.residuals.calculate_chi2``
  - Boundary: NumPy ``float64`` arrays with shapes ``(N, D)`` for
    residuals and ``(N, D, D)`` for covariances.
  - Rust entrypoint: ``adam_core._rust_native.calculate_chi2_numpy``.
  - Contract: covariance matrices must be symmetric positive definite;
    the Rust kernel solves with Cholesky rather than forming an explicit inverse.
  - NaN behavior: NaN diagonal entries raise ``ValueError``; NaN off-diagonal
    entries are treated as zero with a Python ``UserWarning`` for legacy compatibility.
  - Error behavior: raises ``ValueError`` on shape mismatch or
    non-positive-definite covariance input.

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

- ``orbit_determination.fit_least_squares`` (whitened residuals, analytic Jacobian, robust loss)
  - Boundary: nested Arrow IPC of the orbit, observed coordinates and
    observers plus a ``fit_settings`` dict in; a dict of scalars, lists and
    NumPy arrays out. The complete work unit (solver, Jacobian, loss,
    covariance probe, final evaluation) runs in Rust; Python converts tables
    and re-emits the driver's diagnostics as ``RuntimeWarning``. The supplied
    ``Propagator`` is driven from Rust: through its own fused work unit when it
    has one, otherwise through the callback route
    (``adam_core.orbit_determination._native_callback.predict_spherical``
    calling its ``generate_ephemeris``).
  - Rust entrypoints: ``adam_core._rust_native.observation_whitening_matrices_numpy``
    (``(N, 6, 6)`` covariance + ``(N,)`` latitude -> ``(N, 2, 2)`` inverse Cholesky
    factors), ``whiten_residual_pairs_numpy``, ``whitened_2body_jacobian_numpy``
    (``(2N, 6)`` forward-mode autodiff Jacobian of the whitened 2-body model;
    barycentric ecliptic observer and Sun states in), ``robust_cost_numpy`` /
    ``robust_weights_numpy`` / ``robust_jacobian_scale_numpy`` and
    ``validate_robust_loss``.
  - Error behavior: raises ``ValueError`` naming the observation whose
    (lon, lat) covariance block is non-finite or not positive definite, on shape
    mismatch, and on an unknown loss or non-positive ``f_scale``.
  - Dispatch: a Rust-backed propagator exposing ``fit_least_squares_whitened
    (orbit, observations, ignore_mask, fit_settings=...)`` runs the complete
    work unit natively in one crossing for every Jacobian and loss option
    (``adam_core_rs_coords::fit_orbit_whitened_barycentric``: whitened
    residuals through the barycentric ephemeris workflow, analytic / central /
    2-point Jacobian, IRLS Huber, Levenberg-Marquardt with scipy's stopping
    rules, ``inv(JᵀJ)`` with the weak-direction probe and central-difference
    fallback, the fused final evaluation; diagnostics return as strings the
    veneer re-emits as ``RuntimeWarning``). The in-tree implementation over the
    two-body backend is ``adam_core._rust_native.fit_orbit_whitened_2body_ipc``.
    Every other propagator runs the same driver through the callback route
    (``adam_core._rust_native.fit_orbit_whitened_ipc`` with the propagator
    object; ``propagator=None`` selects the two-body backend). Only the
    solver settings ``xtol``, ``ftol``, ``gtol`` and ``max_nfev`` are
    accepted; any other keyword raises ``ValueError`` (there is no scipy
    path). ``jacobian="2-point"`` with ``loss="linear"`` on a propagator
    exposing only the older ``fit_least_squares_evaluated`` /
    ``fit_least_squares`` work units still routes there (forward-difference
    Gauss-Newton), with its covariance probed by
    ``validate_fit_covariance_ipc``.

- ``orbit_determination.iterative_fit`` / ``NativeOrbitFitter.full_od`` /
  ``run_od`` (native orchestration)
  - Boundary: nested Arrow IPC of the orbit, observed coordinates and observers
    plus dicts of settings (``fit_settings``, ``iod_settings``, ``refinement``)
    and, for ``run_od``, the per-observation station / band / catalog lists and
    the models' ``_native_spec()`` dicts; outputs are dicts of scalars, lists
    and NumPy arrays the veneer wraps into tables without further computation.
  - Rust entrypoints (``adam_core_rs_coords::propagation``, generic over the
    ``Propagator`` trait; two-body implementations in ``_rust_native``):
    ``iterative_fit_barycentric`` / ``iterative_fit_2body_ipc`` (worst-residual
    loop), ``cmc2003_fit_barycentric`` / ``cmc2003_fit_2body_ipc`` (CMC2003
    loop), ``full_od_barycentric`` / ``full_od_2body_ipc`` (Gauss IOD decision
    loop followed by the refinement loop) and ``run_od_barycentric`` /
    ``run_od_2body_ipc`` (observation models in order, full OD on the used
    observations, original / used astrometry snapshots for the members).
  - Dispatch: ``iterative_fit``, ``cmc2003_fit_detailed`` and ``iod`` route
    to a propagator's ``iterative_fit`` / ``cmc2003_fit`` /
    ``initial_orbit_determination`` work units when it has them and otherwise
    run the same Rust loop through the callback route
    (``iterative_fit_ipc`` / ``cmc2003_fit_ipc`` / ``iod_fit_ipc`` with the
    propagator object); ``NativeOrbitFitter.full_od`` routes to ``full_od``
    when its ``rejection_kwargs`` are all fit or loop settings and otherwise
    chains ``initial_fit`` and ``refine_fit`` (two native crossings);
    ``run_od`` routes to ``run_od`` for a ``NativeOrbitFitter`` when every
    model provides a native spec, otherwise it composes model application
    (Rust kernels), ``fitter.full_od`` and the provenance join. No orbit
    determination loop remains in Python: Python owns ids, the
    ``OrbitFitter`` plugin boundary, table loading and table assembly.
    Settings that only the scipy solver understood raise ``ValueError``.
  - Error behavior: driver errors raise ``ValueError`` for invalid inputs and
    ``RuntimeError`` for backend failures; an exception raised by a Python
    propagator inside the callback is re-raised unchanged.
  - CMC2003 finalization uses one central-difference Jacobian through the
    full predictor for all observations. Selected rows form the covariance;
    all rows supply rejection/recovery projections. The solver Jacobian
    setting affects optimizer steps only. The twelve finalization candidate
    predictions are outside ``max_nfev``; ``validate_covariance`` controls
    the additional weak-direction probe. Huber plus CMC2003 remains a
    heuristic combination without calibrated rejection statistics.

- ``orbit_determination.cmc2003_fit`` / ``cmc2003_fit_detailed``
  - Boundary: the whole loop is the ``cmc2003_fit`` work unit above (fused
    or callback route); the decision kernels below remain callable on their
    own for diagnostics.
  - Rust entrypoints: ``adam_core._rust_native.cmc2003_apparitions_numpy``,
    ``cmc2003_expected_residual_chi2_numpy`` (``(N, 2)`` whitened residuals,
    ``(2N, 6)`` Jacobian, optional ``(6, 6)`` covariance, selection mask),
    ``cmc2003_select_numpy``.
  - Error behavior: raises ``ValueError`` on shape mismatch.

- ``orbit_determination`` observation uncertainty models
  - Boundary: NumPy ``(N, 6, 6)`` covariance plus Python lists of station codes,
    bands and star catalogs; the quivr table is rebuilt in Python only when Rust
    reports a change (``None`` return = input object returned untouched).
  - Rust entrypoints: ``adam_core._rust_native.bias_table_model_apply_numpy``
    (``empirical_covariance`` / ``performance_weighted`` / ``sigma_floor``),
    ``night_batch_deweighting_model_apply_numpy``,
    ``efcc18_debias_model_apply_numpy`` (positions), ``veres_model_apply_numpy``
    (``floor`` / ``replace`` / ``fill``), ``veres_sigma_lookup_numpy``,
    ``ades_angular_covariance_numpy``. No sigma table is bundled: the Python
    veneer resolves a default table by importing the private
    ``observatory_uncertainties`` data package (a soft import, never a
    dependency) and raises ``ImportError`` naming it when absent.
  - Position rule: every model except ``efcc18_debias_model_apply`` leaves the
    lon/lat columns untouched; ``efcc18_debias_model_apply`` leaves the
    covariance untouched.
  - Error behavior: raises ``ValueError`` on duplicate ``(obs_code, band)`` bias
    rows, an unknown ``mode``/``model``, a non-positive ``cap`` or sigma, and
    on shape mismatch.

- ``observations.efcc18`` (star-catalog debiasing)
  - Boundary: NumPy ``float64`` RA/Dec/JD arrays and Python catalog lists; the
    ``(49152, 26, 4)`` float32 table crosses as a contiguous NumPy array.
  - Rust entrypoints: ``adam_core._rust_native.efcc18_ra_dec_to_healpix_numpy``
    (``N_side = 64`` RING, the healpix_cxx ``ang2pix`` port),
    ``efcc18_parse_bias_dat``, ``efcc18_read_bias_version``,
    ``efcc18_catalog_columns_numpy``, ``efcc18_corrections_numpy``,
    ``healpix_ang2pix_lonlat_numpy``.
  - Contract: row ``k`` of ``bias.dat`` is HEALPix ring pixel ``k``; the Rust
    test suite pins the 8 published ``tiles.dat`` anchors and, when
    ``ADAM_CORE_EFCC18_TILES_DAT`` names a local copy, all 49152 tile centres.
  - Error behavior: raises ``ValueError`` on a malformed table layout, a wrong
    table shape, or a colatitude outside ``[0, pi]``.

Fallback and Waivers
--------------------

- If a Rust-backed API cannot satisfy parity/performance criteria, keep that
  API off the migrated production surface or record an explicit waiver. Do not
  add rustless production fallbacks. Optional Astropy/UT1, Astroquery
  monkeypatch, Healpy, plotting, external propagator, and live provider calls
  are explicit compatibility/provider boundaries rather than default backends.
- Waivers are tracked in ``migration/waivers.yaml`` with owner and review date.

Status Registry
---------------

- Single source of truth for migration state: ``adam_core._rust.status`` (``API_MIGRATIONS``).
- ``API_MIGRATION_STATUS`` is a compatibility projection of the same data used by runtime dispatch.
- Migration governance scripts (``migration/scripts/*``) import from this module; there is no separate YAML registry.

Validation Contract
-------------------

- The standard Python test suite command ``pytest --benchmark-skip -m 'not profile'`` must run in a Rust-enabled environment for migration validation.
- Release candidates additionally run every profile-marked scientific fixture and the opt-in live external-service integration gates after wheel acceptance.
- Validation runs fail during import if the Rust extension is unavailable.
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
