# adam_core Rust Surface Gap Report (2026-05-15)

Status: persisted audit and conversion-priority analysis for the `rust-migration-waves-d-e` branch after `d93ab234` (`Optimize least-squares inner loop`).

Sources used:

- `src/adam_core/_rust/status.py`
- `migration/parity/tolerances.py`
- `migration/artifacts/parity_table_rca.json`
- `migration/artifacts/parity_speed_cold_warm.json`
- `docs/source/reference/rust_public_compatibility.rst`
- AST inventory of public-ish definitions under `src/adam_core` excluding tests and `_rust` internals
- Current grounding decisions/journal through 2026-05-15

This report intentionally covers more than known hot paths. It distinguishes:

1. surfaces already Rust-backed and canonically governed;
2. Rust-backed surfaces that are not yet canonical parity/speed rows;
3. surfaces without Rust equivalents or without direct canonical governance;
4. which remaining surfaces should be converted, should only be gated, should stay Python, or should be deferred to future architecture work.

## Executive summary

| Area | Current state |
|---|---:|
| Public-ish Python definitions/classes/methods found by AST inventory | 507 |
| Canonical Rust-governed API IDs | 42 |
| Canonical speed rows | 126 = 42 APIs × tiny/small/large |
| Enforced speed rows per lane | 34 |
| Diagnostic/raw-kernel speed rows per lane | 8 |
| Targeted-test-only registry rows | 0 |
| Active performance waivers | 0 |

The key interpretation is: **canonical Rust migration governance is complete for the registered migrated numerical/Rust surfaces, but not for all of adam_core.** Large parts of adam_core are PyArrow/quivr table orchestration, network clients, plotting, textual I/O, timestamp scale handling, public workflow glue, and ASSIST/n-body propagation. Those are not automatically appropriate Rust-port targets.

## Conversion-priority legend

| Label | Meaning |
|---|---|
| **Gate now** | Rust already exists; add canonical parity/speed or latency governance if we want full accounting. |
| **Convert small** | Straightforward Rust conversion possible, but likely low impact from Python due PyO3 overhead; useful mainly for future Rust-to-Rust coverage. |
| **Convert if benchmark-driven** | Do not port speculatively. Convert only if RM-PERF-001 or profiling shows material cost and a clean semantic boundary. |
| **Keep Python** | Python/PyArrow/network/plotting/I/O orchestration is the right implementation; Rust would add complexity without a meaningful correctness/performance win. |
| **Defer major** | Feasible in Rust, but requires separate project architecture and validation. Do not mix into merge cleanup. |
| **Already governed / intentional diagnostic** | Rust exists and is fuzzed, but speed rows are diagnostic/raw-kernel rather than public promotion gates by design. |

## 1. Rust-backed and canonically fuzz/speed governed

These have canonical randomized parity tolerance specs and tiny/small/large speed rows. The public/default/orchestration rows below are enforced, subject to tiny-n p95 being diagnostic by policy.

### 1.1 Enforced public/default/orchestration rows

- `coordinates.cartesian_to_spherical(coords_cartesian)`
- `coordinates.cartesian_to_geodetic(coords_cartesian, a, f, max_iter=100, tol=1e-15)`
- `coordinates.cartesian_to_keplerian(coords_cartesian, t0, mu)`
- `coordinates.keplerian.to_cartesian(coords_keplerian, mu, max_iter=100, tol=1e-15)`
- `coordinates.cartesian_to_cometary(coords_cartesian, t0, mu)`
- `coordinates.cometary.to_cartesian(coords_cometary, t0, mu, max_iter=100, tol=1e-15)`
- `coordinates.spherical.to_cartesian(coords_spherical)`
- `coordinates.transform_coordinates(coords, representation_out=None, frame_out=None, origin_out=None)` — partial public-dispatch matrix; exclusions listed below
- `coordinates.residuals.apply_cosine_latitude_correction(lat, residuals, covariances)`
- `coordinates.residuals.bound_longitude_residuals(observed, residuals)`
- `coordinates.residuals.Residuals.calculate(observed, predicted, custom_coordinates=False, use_predicted_covariance=True)`
- `coordinates.residuals.calculate_chi2(residuals, covariances)`
- `dynamics.calc_mean_motion(a, mu)`
- `dynamics.propagate_2body(orbits, times, max_iter=1000, tol=1e-14, *, max_processes=1, chunk_size=100)`
- `dynamics.propagate_2body_with_covariance(...)`
- `dynamics.generate_ephemeris_2body(propagated_orbits, observers, lt_tol=1e-10, max_iter=1000, tol=1e-15, stellar_aberration=False, predict_magnitudes=True, *, predict_phase_angle=False, max_processes=1, chunk_size=100)`
- `dynamics.generate_ephemeris_2body_with_covariance(...)`
- `dynamics.add_light_time(orbits, t0, observer_positions, lt_tol=1e-10, mu=MU, max_iter=1000, tol=1e-15, max_lt_iter=10)`
- `dynamics.calculate_moid(primary_ellipse, secondary_ellipse)`
- `dynamics.calculate_perturber_moids(orbits, perturber, chunk_size=100, max_processes=1)`
- `dynamics.solve_lambert(r1, r2, tof, mu=MU, prograde=True, max_iter=35, tol=1e-10)`
- `dynamics.tisserand_parameter(a, e, i, third_body='jupiter')`
- `dynamics.generate_porkchop_data(departure_orbits, arrival_orbits, propagation_origin=SUN, prograde=True, max_iter=35, tol=1e-10, max_processes=1)`
- `photometry.calculate_phase_angle(object_coords, observers)`
- `photometry.calculate_apparent_magnitude_v(H_v, object_coords, observer, G=0.15)`
- `photometry.calculate_apparent_magnitude_v_and_phase_angle(H_v, object_coords, observer, G=0.15)`
- `photometry.predict_magnitudes(H, object_coords, exposures, G=0.15, reference_filter='V', *, composition)`
- `photometry.fit_absolute_magnitude_rows(h_rows, sigma_rows)`
- `photometry.fit_absolute_magnitude_grouped(h_rows, sigma_rows, group_offsets)`
- `orbit_determination.calcGibbs(r1, r2, r3)`
- `orbit_determination.calcHerrickGibbs(r1, r2, r3, t1, t2, t3)`
- `orbit_determination.calcGauss(r1, r2, r3, t1, t2, t3)`
- `orbit_determination.gaussIOD(coords, observation_times, coords_obs, velocity_method='gibbs', light_time=True, mu=MU, max_iter=10, tol=1e-15)`
- `orbits.classify_orbits(elements)` / `orbits.classification.calc_orbit_class(elements)`

### 1.2 Random-fuzzed, but diagnostic/raw-kernel speed rows by design

These have canonical fuzz parity, but speed is not a public-promotion gate. Some are intentionally slower than the public NumPy/BLAS path and are kept for future Rust-to-Rust coverage.

- `coordinates.transform_coordinates_with_covariance(coords, covariances, representation_in, representation_out, ...)`
- `coordinates.rotate_cartesian_time_varying(coords, time_index, matrices, covariances=None)`
- `dynamics.propagate_2body_along_arc(orbit, dts, mu, max_iter=100, tol=1e-15)`
- `dynamics.propagate_2body_arc_batch(orbits, dts, mus, max_iter=100, tol=1e-15)`
- `dynamics.calculate_moid_batch(primary_orbits, secondary_orbits, mus, max_iter=100, xtol=1e-10)`
- `missions.porkchop_grid(dep_states, dep_mjds, arr_states, arr_mjds, mu, prograde=True, maxiter=35, atol=1e-10, rtol=1e-10)`
- `statistics.weighted_mean(samples, W)`
- `statistics.weighted_covariance(mean, samples, W_cov)`

Conversion analysis: **do not promote the statistics kernels to public Python defaults now.** The diagnostic artifacts show BLAS-backed NumPy remains faster for public coordinate covariance/statistics workflows. Keep the Rust versions for future Rust-side pipelines and correctness coverage.

## 2. Rust exists, but no canonical random-fuzz/speed row yet

These are the clearest remaining governance gap if the policy becomes “every Rust equivalent must appear in canonical accounting.” Many have smoke/unit tests and compatibility guardrails, but they are not present in `API_MIGRATIONS` / `tolerances.py` / canonical speed artifacts.

### 2.1 Dynamics compatibility/helper APIs

Rust wrappers exist; canonical randomized governance does not, except for `calc_mean_motion`.

- `dynamics.kepler.calc_period(a, mu)`
- `dynamics.kepler.calc_periapsis_distance(a, e)`
- `dynamics.kepler.calc_apoapsis_distance(a, e)`
- `dynamics.kepler.calc_semi_major_axis(q, e)`
- `dynamics.kepler.calc_semi_latus_rectum(a, e)`
- `dynamics.kepler.calc_mean_anomaly(nu, e)`
- `dynamics.kepler.solve_kepler(e, M, max_iter=100, tol=1e-15)`
- `dynamics.barker.solve_barker(M)`
- `dynamics.stumpff.calc_stumpff(psi)`
- `dynamics.chi.calc_chi(r, v, dt, mu=MU, max_iter=100, tol=1e-15)`
- `dynamics.chi.calc_chi_diagnostics(r, v, dt, mu=MU, max_iter=100, tol=1e-15)`
- `dynamics.lagrange.calc_lagrange_coefficients(r, v, dt, mu=MU, max_iter=100, tol=1e-15)`
- `dynamics.lagrange.apply_lagrange_coefficients(r, v, f, g, f_dot, g_dot)`
- `dynamics.aberrations.add_stellar_aberration(orbits, observer_states)`

Recommendation: **Gate now before new algorithmic ports.** These are already Rust-backed and small. Add canonical fuzz rows and probably diagnostic/enforced speed rows depending on whether they are public defaults or compatibility helper wrappers. Expect several tiny one-line formulas to have marginal Python-call speed due PyO3 overhead; for those, label carefully so Rust-to-Rust coverage is not confused with Python promotion evidence.

### 2.2 Raw/internal Rust OD helper

- `_rust.api.gauss_iod_orbits_numpy(...)`

Recommendation: **Probably do not add a standalone public governance row unless we expose/commit to it.** `gaussIOD` is already governed at the fused public boundary, and root-finding sub-helper behavior is only meaningful inside that full pipeline.

### 2.3 SPICE / NAIF native wrappers

Rust/spicekit backs these, with targeted tests and spicekit-side CSpice parity ownership, but they are not canonical adam-core parity/speed rows.

- `_rust.api.naif_spk_open(path)`
- `_rust.api.naif_pck_open(path)`
- `_rust.api.naif_bodn2c(name)`
- `_rust.api.naif_bodc2n(code)`
- `_rust.api.naif_parse_text_kernel_bindings(path)`
- `_rust.api.naif_spk_writer(locifn='adam-core')`
- `_rust.api.adam_core_spice_backend()`
- `utils.spice_backend.RustBackend.furnsh(path)`
- `utils.spice_backend.RustBackend.unload(path)`
- `RustBackend.spkez(target, et, frame, observer)`
- `RustBackend.spkez_batch(target, observer, frame, ets)`
- `RustBackend.pxform_batch(frame_from, frame_to, ets)`
- `RustBackend.sxform_batch(frame_from, frame_to, ets)`
- `RustBackend.bodn2c(name)`

Recommendation: **Gate selectively, not by forcing all into the baseline-main parity harness.** The CSpice oracle lives in `spicekit-bench` by decision; adam-core should govern wiring/semantics and maybe Rust-only latency for public wrappers. The most useful adam-core rows would be `get_perturber_state`, `get_spice_body_state`, `get_observer_state`, and `RustBackend.{spkez_batch,pxform_batch,sxform_batch,bodn2c}` using fixed kernel fixtures, not random fuzz.

## 3. Governed, but not bitwise/exhaustive

These are covered and should remain covered, but their tolerance model is intentionally not bitwise across the whole input domain.

### `coordinates.transform_coordinates(...)`

Covered representative public-dispatch matrix:

- Cartesian constant-frame inverse directions
- Spherical/Keplerian/Cometary non-Cartesian inputs
- representative covariance-bearing Cartesian/Keplerian dispatcher paths
- SUN↔EARTH origin translations
- Earth-centered ITRF93 time-varying rotations at vetted PCK epochs

Explicit exclusions:

- Cartesian→Cartesian frame-only fallthrough, intentionally faster through cached path
- covariance-bearing ITRF93 public dispatcher
- unsupported SPICE-origin/time-varying paths where Rust SPICE parity/perf is not yet the governing oracle

Recommendation: **keep the current representative public gate and documented exclusions.** Add rows only when a currently excluded path becomes a Rust-default public path. Do not force the pure Cartesian frame-only fast path through Rust while Python cached dispatch remains faster.

### `orbit_determination.gaussIOD(...)`

The public `gaussIOD` row is fixed-fixture governed for well-conditioned shared-root cases. It is intentionally not random-fuzzed over all possible ambiguous root geometries. Recommendation: **keep the constrained oracle.** Expanding this to arbitrary root-selection behavior would be a numerical-method policy project, not a Rust coverage cleanup.

### Other tolerance-based rows

- MOID `dt_at_min` uses tolerance-based comparison, aligned with scalar MOID policy.
- Covariance outputs are tolerance-based because Rust AD, JAX, and NumPy differ in accumulation/order.
- Photometry magnitudes are tolerance-based because `log10` / `powf` / transcendental stacks differ by ulps.
- Covariance-transform NaN policy is intentional: NaN covariance input yields all-NaN output covariance row while transformed state values still propagate.

Recommendation: **do not chase bitwise parity for these rows.** The current tolerances preserve scientific semantics while avoiding false failures from equivalent floating-point implementations.

## 4. Full persisted public-ish inventory

The exact machine-generated inventory is persisted at:

- `migration/artifacts/adam_core_public_surface_inventory_2026-05-15.json`

It contains 507 entries with `module`, `qualname`, `kind`, `signature`, `path`, and source `line`. This JSON is the full list; the analysis below groups it by conversion priority. The inventory is intentionally public-ish, not “must convert”: it includes table models, properties, plotting helpers, network clients, and workflow orchestration.

Module counts from the persisted inventory:

| Count | Module |
|---:|---|
| 37 | `adam_core.time.time` |
| 34 | `adam_core.coordinates.cartesian` |
| 29 | `adam_core.coordinates.cometary` |
| 29 | `adam_core.coordinates.keplerian` |
| 20 | `adam_core.parallel` |
| 16 | `adam_core.coordinates.spherical` |
| 16 | `adam_core.missions.porkchop` |
| 15 | `adam_core.coordinates.covariances` |
| 12 | `adam_core.coordinates.transform` |
| 11 | `adam_core.coordinates.geodetics` |
| 11 | `adam_core.dynamics.impacts` |
| 11 | `adam_core.photometry.bandpasses.api` |
| 10 | `adam_core.orbits.openspace.renderable` |
| 10 | `adam_core.orbits.variants` |
| 10 | `adam_core.utils.spice_backend` |
| 9 | `adam_core.observations.ades` |
| 9 | `adam_core.utils.mpc` |
| 8 | `adam_core.coordinates.residuals` |
| 8 | `adam_core.coordinates.units` |
| 8 | `adam_core.dynamics.kepler` |
| 8 | `adam_core.observations.source_catalog` |
| 7 | `adam_core.coordinates.origin` |
| 7 | `adam_core.dynamics.plots` |
| 7 | `adam_core.observers.observers` |
| 7 | `adam_core.utils.spice` |
| 6 | `adam_core.coordinates.variants` |
| 6 | `adam_core.dynamics._rust_compat` |
| 6 | `adam_core.orbits.openspace.assets` |
| 6 | `adam_core.propagator.propagator` |
| 5 | `adam_core.observations.detections` |
| 5 | `adam_core.orbit_determination.fitted_orbits` |
| 5 | `adam_core.orbit_determination.iod` |
| 5 | `adam_core.orbits.query.scout` |
| 5 | `adam_core.photometry.bandpasses.vendor` |
| 5 | `adam_core.photometry.magnitude` |
| 5 | `adam_core.photometry.magnitude_common` |
| 4 | `adam_core.observations.exposures` |
| 4 | `adam_core.orbits.openspace.translation` |
| 4 | `adam_core.orbits.orbits` |
| 4 | `adam_core.orbits.spice_kernel` |
| 4 | `adam_core.photometry.bandpasses.tables` |
| 3 | `adam_core.dynamics.chi` |
| 3 | `adam_core.dynamics.moid` |
| 3 | `adam_core.observations.associations` |
| 3 | `adam_core.observers.state` |
| 3 | `adam_core.orbit_determination.evaluate` |
| 3 | `adam_core.orbit_determination.od` |
| 3 | `adam_core.orbits.oem_io` |
| 3 | `adam_core.orbits.openspace.lua` |
| 3 | `adam_core.orbits.plots` |
| 3 | `adam_core.orbits.query.sbdb` |
| 3 | `adam_core.photometry.absolute_magnitude` |
| 2 | `adam_core.dynamics.aberrations` |
| 2 | `adam_core.dynamics.lagrange` |
| 2 | `adam_core.dynamics.lambert` |
| 2 | `adam_core.dynamics.propagation` |
| 2 | `adam_core.orbit_determination.differential_correction` |
| 2 | `adam_core.orbit_determination.gauss` |
| 2 | `adam_core.orbit_determination.least_squares` |
| 2 | `adam_core.orbit_determination.orbit_fitter` |
| 2 | `adam_core.orbit_determination.outliers` |
| 2 | `adam_core.orbits.ephemeris` |
| 2 | `adam_core.orbits.query.horizons` |
| 2 | `adam_core.propagator.utils` |
| 2 | `adam_core.utils.bounded_lru` |
| 2 | `adam_core.utils.chunking` |
| 2 | `adam_core.utils.helpers.orbits` |
| 1 | `adam_core.dynamics.barker` |
| 1 | `adam_core.dynamics.ephemeris` |
| 1 | `adam_core.dynamics.exceptions` |
| 1 | `adam_core.dynamics.stumpff` |
| 1 | `adam_core.dynamics.tisserand` |
| 1 | `adam_core.observations.photometry` |
| 1 | `adam_core.observers.utils` |
| 1 | `adam_core.orbit_determination.gibbs` |
| 1 | `adam_core.orbit_determination.herrick_gibbs` |
| 1 | `adam_core.orbits.classification` |
| 1 | `adam_core.orbits.physical_parameters` |
| 1 | `adam_core.orbits.query.neocc` |
| 1 | `adam_core.photometry.bandpasses.constants` |
| 1 | `adam_core.ray_cluster` |
| 1 | `adam_core.utils.helpers.observations` |
| 1 | `adam_core.utils.plots.logos` |

## 5. Conversion analysis for non-canonical or non-Rust surfaces

### 5.1 Gate now: Rust exists and governance is the main gap

Recommended near-term governance tasks:

1. **Dynamics scalar/helper compatibility gates** for Kepler, Barker, Stumpff, chi, Lagrange, and stellar-aberration helpers listed in Section 2.1.
   - Why: already Rust-backed; small incremental cost; closes “Rust exists but not in canonical accounting.”
   - Caveat: tiny scalar helpers may not meet public Python speed promotion after PyO3 overhead. Label as helper/diagnostic rows if the measured scope is a single scalar call.
2. **SPICE/NAIF wiring gates** for fixed-kernel fixtures.
   - Why: Rust backend is active and user-visible through `RustBackend` plus observer/SPICE utilities.
   - Caveat: do not duplicate spicekit’s CSpice parity suite in adam-core. Test adam-core marshaling, batching, kernel load/unload semantics, and selected latency.
3. **Documentation rows for high-level orchestrators that compose Rust primitives.**
   - Examples: `evaluate_orbits`, `LeastSquares.least_squares`, `iod`, `od`, `Propagator.propagate_orbits`, `EphemerisMixin.generate_ephemeris`.
   - Why: prevents overclaiming that a high-level workflow is Rust-default just because inner kernels are Rust-backed.
   - Caveat: document/benchmark wall-clock scope separately from inner microbenchmarks.

### 5.2 Convert small only if a future Rust-to-Rust path needs it

These are technically portable, but Python-level speedups are uncertain or unlikely because they are tiny scalar formulas or mostly data reshaping.

- Remaining coordinate unit-conversion helpers in `coordinates.units.*`.
- Small residual/statistical utility wrappers not already governed, such as `compute_residuals_ndarray(observed, predicted)` and `calculate_reduced_chi2(residuals, parameters)`.
- MPC designation/string helpers in `utils.mpc.*`.
- Simple orbit physical-parameter/classification helpers not already covered by `calc_orbit_class`.

Recommendation: **do not port these solely for migration completeness.** Port only when a larger Rust pipeline would otherwise have to call back into Python.

### 5.3 Convert only if RM-PERF-001 or profiling shows material cost

These are plausible Rust candidates, but the current evidence does not justify porting them as cleanup work.

- `coordinates.covariances.*` and `coordinates.variants.create_coordinate_variants(...)`.
  - Current evidence: recent RM-WE2-003 win came from batching PyArrow assembly while preserving SciPy `sqrtm` and NumPy/BLAS weighted statistics.
  - Recommendation: keep Python/NumPy/SciPy unless profiling shows linalg kernels, not table orchestration, are the bottleneck.
- Higher-level coordinate transform dispatcher cases not already covered.
  - Recommendation: add only when a specific public path is promoted to Rust-default or appears in slow-row audit.
- `Timestamp` vector operations, cache/key helpers, and time-scale conversion wrappers.
  - Recommendation: keep Python/Astropy/ERFA. A Rust time system would be a standards-compliance project, not a small port.
- Photometry bandpass data table/lookup layer.
  - Current evidence: numerical magnitude fitting/prediction kernels are already governed in Rust.
  - Recommendation: keep Python data plumbing unless benchmark-action slow rows show table lookup dominates.
- Source-catalog and observation table transformations.
  - Recommendation: keep Python until an end-to-end ingestion benchmark identifies a narrow, stable kernel.

### 5.4 Defer major architecture work

These may be good future Rust projects, but not last-mile migration cleanup.

- **Generic n-body / ASSIST propagation:** `EphemerisMixin.generate_ephemeris`, `Propagator.propagate_orbits`, `propagation_worker`, impact/collision helpers, and orbit propagation workflows.
  - Why defer: wall-clock OD/LSQ/IOD remains dominated by `generate_ephemeris` / ASSIST propagation. Meaningful conversion likely requires `assist-rs` or a new Rust propagation architecture.
- **OD orchestration:** `fit_least_squares`, `LeastSquares.least_squares`, `evaluate_orbits`, outlier helpers, `iod`, `od`, `differential_correction`, and `OrbitFitter.initial_fit`.
  - Why defer: recent RM-WE2-004/RM-WE3-001 wins were from Python/NumPy orchestration cleanup while preserving deterministic OD behavior. Full Rust port would need ownership of propagation, residual table semantics, convergence diagnostics, and fit reporting.
- **Impact probability/collision workflows.**
  - Why defer: tied to n-body propagation, observer geometry, and stochastic workflow semantics.
- **SPK/OEM/OpenSpace generation as complete Rust workflows.**
  - Why defer: mostly textual/file-format orchestration around already-covered numerical state generation.

### 5.5 Keep Python / should not convert in this migration

These are poor Rust-port targets for this migration because they are dominated by external Python APIs, file/network I/O, plotting, PyArrow table modeling, or user-facing workflow glue.

- Coordinate model classes, accessors, and property methods on `CartesianCoordinates`, `SphericalCoordinates`, `KeplerianCoordinates`, `CometaryCoordinates`, `GeodeticCoordinates`, `Origin`, and covariance table classes.
- Quivr/PyArrow table models across orbits, observations, exposures, detections, associations, fitted-orbit tables, physical parameters, source catalogs, bandpass tables, and OpenSpace renderable records.
- `parallel.*`, `ray_cluster.*`, `utils.chunking.*`, `utils.iter.*`, `utils.bounded_lru.*`, and simple helper modules.
- Plotting and visualization modules: `dynamics.plots`, `orbits.plots`, `utils.plots.*`, OpenSpace Lua/renderable/asset text emitters.
- Network query clients: JPL Horizons, Scout, SBDB, NEOCC, and other source-catalog download/query helpers.
- ADES/OEM textual I/O and mostly-formatting surfaces.
- Bandpass vendor data ingestion and constants loading.
- Public convenience constructors/accessors whose primary job is schema validation, Arrow conversion, or user ergonomics.

Recommendation: **do not convert these unless a separate product/architecture decision changes the boundary.** Rust would not preserve the same ecosystem advantages and would likely increase maintenance burden.

## 6. Cannot or should-not-convert reasons by dependency class

| Dependency / behavior | Why it blocks or discourages direct Rust conversion |
|---|---|
| Astropy/ERFA time-scale semantics | Reimplementing time standards is high-risk; parity would require a dedicated standards suite. |
| ASSIST / n-body propagation | Dominant wall-clock cost, but requires `assist-rs` or a new propagation backend rather than piecemeal wrappers. |
| PyArrow/quivr table ergonomics | The public API is Python table modeling; moving it to Rust would not avoid Python object construction at the boundary. |
| Network services | Latency/error semantics are external; Rust numeric speed is irrelevant. |
| Plotting/OpenSpace text output | Primarily formatting/UI; Python ecosystem is appropriate. |
| SPICE C-oracle parity | Already owned by spicekit benches; adam-core should only gate wrapper semantics and fixtures. |
| Tiny scalar helpers | PyO3 call overhead can dominate; good for Rust-to-Rust use, not necessarily Python speed promotion. |

## 7. Recommended backlog from this audit

1. **RM-PERF-001:** Complete PR #195 benchmark-action slow-row audit for broad surface clues.
2. **RM-GOV-HELPERS:** Add canonical governance for Rust-backed dynamics helper/compatibility APIs if the goal is “every Rust equivalent is counted.”
3. **RM-GOV-SPICE-WIRING:** Add fixed-kernel adam-core wiring gates for Rust SPICE backend wrappers without duplicating spicekit’s CSpice oracle suite.
4. **RM-DOC-SCOPE:** Document that the 42 canonical API IDs cover migrated numerical/Rust surfaces, not every public `adam_core` surface.
5. **Future project:** Evaluate `assist-rs` / n-body propagation only as a dedicated project, because propagation dominates OD/LSQ/IOD wall-clock.

## 8. Reporting rule for future benchmark summaries

When reporting follow-up performance evidence, explicitly state:

- measured scope;
- excluded work;
- dataset size;
- units;
- whether the result is an isolated microbenchmark or end-to-end wall-clock.

This avoids overclaiming inner-loop speedups as full workflow acceleration.