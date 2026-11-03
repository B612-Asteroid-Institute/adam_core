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

The authoritative inventory for this source tree is
``migration/public_surface/manifest.json`` (734 symbols / 101 constants).
Its frozen-upstream reconciliation records the independently classified
retirements. The selected parity registry is a benchmark set, not the
public-surface count.
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

- ``orbit_determination.fit_least_squares`` (whitened residuals, Jacobian that follows the dynamics, robust loss)
  - Derivatives: ``jacobian="auto"`` (default) resolves to ``"analytic"`` for
    declared two-body dynamics, ``"propagator"`` when the backend supplies
    state-transition matrices (``Propagator::propagate_stm``), and
    ``"central"`` otherwise. The propagator route takes optimizer steps with
    the two-body Jacobian, then converges and evaluates the covariance with
    the backend's STMs, so the reported solution and covariance carry the
    selected dynamics' exact derivatives.
  - Boundary: nested Arrow IPC of the orbit, observed coordinates and
    observers plus a ``fit_settings`` dict in; a dict of scalars, lists and
    NumPy arrays out. The complete work unit (solver, Jacobian, loss,
    covariance probe, final evaluation) runs in Rust; Python converts tables
    and re-emits the driver's diagnostics as ``RuntimeWarning``. The supplied
    ``Propagator`` contributes only its external-propagator capsule
    (``propagator_capsule``). The solver and evaluation remain adam-core code;
    a propagator without a capsule raises ``TypeError`` and is never called
    back from Rust.
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
  - Dispatch: ``fit_least_squares`` always invokes adam-core's
    ``fit_orbit_whitened_ipc`` driver with the selected capsule
    (``propagator=None`` selects the in-tree two-body backend). The generic
    Rust implementation is
    ``adam_core_rs_coords::fit_orbit_whitened_barycentric`` and remains the
    single definition of whitening, robust loss, Jacobian policy, covariance
    and final evaluation. Providers do not implement least-squares work
    units. Only ``xtol``, ``ftol``, ``gtol`` and ``max_nfev`` are accepted;
    any other solver keyword raises ``ValueError`` (there is no scipy path).

- OD configuration and canonical Rust batch workflows
  - Status: canonical batch ``run_iod`` / ``run_od`` / ``refine`` /
    ``evaluate_orbits`` are exported. Every route returns fitted orbits,
    explicit fitted members, and typed scientific failures; expert numerical
    kernels remain under ``adam_core.orbit_determination.algorithms``.
  - Boundary: nested Arrow IPC plus explicit typed configuration. Rust returns
    authoritative output disposition, success/status, solution/outlier flags,
    state retention, failure classification, fingerprints, and provenance;
    Python only wraps those columns into quivr tables. Route, complete
    configuration (including profile selection), input identity, and propagated
    IOD metadata are covered by a validated consistency digest. This unkeyed
    digest detects inconsistent fields; it is not authentication or a
    tamper-proof signature. Python does not infer composition eligibility from
    a mutable route string.
  - Python
    binds semantic resource slots and marshals tables; Rust validates exact
    resource identity, parses tables, executes the ordered observation
    pipeline, and owns numerical loops. There are no model callback/native-spec
    dictionaries.
  - Rust configuration types live under ``adam_core_rs_coords::od``; profiles
    live under ``od::profiles``. Implemented focused Rust composition functions
    under ``od::algorithms`` are ``prepare_observations``,
    ``run_iod_with_predictor``, ``refine_with_predictor``, and
    ``run_od_with_predictors``. The last applies the pipeline once and gives
    both stages the same prepared coordinates.
  - External native propagators expose the propagation-only
    ``adam_core.propagator.v2`` capsule. Capsule adoption validates its ABI,
    identity fields, and capabilities. The current predictor-level composition
    accepts an already-adopted predictor and does not itself sequence global
    provider/resource/ID preflight. Providers never select transforms, profiles,
    grouping, fitting, rejection, covariance, or OD algorithms.
  - The canonical batch layer rejects invalid IDs/member mappings and
    malformed resource/config/provider contracts
    atomically before pipeline or numerical work. Configuration JSON uses
    correctly rounded float parsing, so every finite Python ``float`` bit is
    retained through native persistence and consistency validation. Canonical
    Python calls make
    one near-linear Rust planning/preparation pass and carry a deferred provider
    resolver that is invoked only after that pass; empty and malformed batches
    never bind a provider. The layer defines empty-group behavior, typed
    per-group scientific failures, deterministic overlapping grouping, and
    canonical Rust runtime provenance.
  - CMC2003 finalization uses one central-difference Jacobian through the full
    predictor for all observations. Selected rows form the covariance; all
    rows supply rejection/recovery projections. Huber plus CMC2003 remains a
    heuristic combination without calibrated rejection statistics.
  - Weak-direction covariance validation first probes both signs at 1 sigma.
    After a failed probe it repeats at 0.1 sigma: a curvature signal scales by
    about 0.01, while propagation/evaluation noise does not. The latter is
    reported with an integration-tolerance hint and does not trigger the
    analytic-Jacobian central-difference fallback. Provider errors propagate;
    only in-band rejected/non-finite probe outcomes become diagnostic warnings.
    This is a fit-surface consistency check, not an uncertainty-realism claim.
  - Python may resolve the installed public EFCC18 package's exported path, but
    passes that path explicitly. The standalone Rust route receives explicit
    ``ResourceBindings`` and performs the identical byte/schema/structure
    validation. Neither Rust route downloads, discovers packages, or falls back.
    See ``rust/adam_core/examples/validate_core_od_resource.rs``.

- ``orbit_determination.algorithms.cmc2003_fit_detailed``
  - Boundary: the whole loop is the adam-core ``cmc2003_fit_ipc`` driver
    above over the selected propagation capsule; the decision kernels below
    remain callable on their own for diagnostics.
  - Rust entrypoints: ``adam_core._rust_native.cmc2003_apparitions_numpy``,
    ``cmc2003_expected_residual_chi2_numpy`` (``(N, 2)`` whitened residuals,
    ``(2N, 6)`` Jacobian, optional ``(6, 6)`` covariance, selection mask),
    ``cmc2003_select_numpy``.
  - Error behavior: raises ``ValueError`` on shape mismatch.

- ``orbit_determination.ObservationPipeline``
  - Boundary: an ordered, immutable declaration of native transforms plus
    explicit bindings from semantic resource slots to local paths. Python does
    not discover data packages, interpret transforms, or execute callbacks.
  - Rust entrypoint: ``adam_core._rust_native.apply_observation_pipeline_ipc``
    validates every required resource before applying any transform and returns
    both original and used astrometry with canonical configuration provenance.
  - Contract: declaration order and repeated transforms are authoritative.
    Deliberate no-transform behavior is represented only by
    ``ObservationPipeline.as_supplied()`` and is recorded as
    ``transforms=[]``. Resource paths are excluded from canonical provenance.
  - Error behavior: missing resources, byte-identity mismatches, schema errors,
    and structural mismatches are aggregated before propagation or fitting;
    an explicitly selected profile never falls back to another profile.

- ``observations.efcc18`` (star-catalog debiasing)
  - Boundary: NumPy ``float64`` RA/Dec/JD arrays and Python catalog lists; the
    ``(49152, 26, 4)`` float32 table crosses as a contiguous NumPy array.
  - Rust implementation: profile binding parses ``bias.dat`` internally and
    the closed pipeline applies catalog corrections. No public native model
    dictionary or in-memory bias-table bypass is exposed.
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
