Orbit determination
===================

Core 0.7's canonical batch functions are ``run_iod``, ``run_od``, ``refine``,
and ``evaluate_orbits``. Each returns ``FittedOrbits``,
``FittedOrbitMembers``, and typed ``OrbitDeterminationFailures``. Raw IOD/OD
uses a ``SourceCatalog`` plus explicit ``Associations``; refinement and
evaluation use explicit ``FittedOrbitMembers``. One object is one group through
the same batch path; there are no primary one-group aliases.

Rust-owned configuration and execution
--------------------------------------

Core Rust owns complete OD configuration, profile expansion, ordered
observation transforms, resource validation/parsing, fitting and rejection
settings, and path-free canonical configuration provenance. Python declares
immutable values, binds explicit local paths, marshals tables, and assembles
result tables. A propagation provider supplies only its negotiated native
capsule.

The complete fused configuration is ``OdConfig``. ``run_od`` receives it
directly; ``run_iod`` receives its bound ``config.for_iod()`` projection.
Standalone ``refine`` receives ``RefinementRunConfig`` and
``evaluate_orbits`` receives an independent ``EvaluationConfig`` (often
``config.for_evaluation()``). The types have deliberately separate scopes:

* ``IodStageConfig`` owns search, selection, velocity, acceptance, correction,
  ranging, candidate-fit, solver, Jacobian, covariance, and two-body settings;
* ``IodRunConfig`` binds one ``IodStageConfig`` to an ordered, already-bound
  ``ObservationPipeline`` and optional immutable profile identity;
* ``OdConfig`` binds one ``IodStageConfig``, one ``RefinementRunConfig``
  (complete CMC2003 or worst-residual settings), and one ordered pipeline;
* ``EvaluationConfig`` independently owns covariance dimension and the exact
  prediction-fit settings used for residual evaluation.

Explicit callers construct ``IodRunConfig(pipeline, iod_stage, resources)``
or ``OdConfig.explicit(...)`` only after choosing every scientific setting.
``EvaluationConfig(6, prediction_fit)`` is directly constructible and does not
carry a hidden ``OdConfig``.

Python exposes typed immutable declarations such as ``IodSearchConfig``,
``SpanSearchConfig``, ``CorrectionConfig``, ``RangingConfig``,
``WhitenedFitConfig``, ``TwoBodyFitConfig``, ``Cmc2003Refinement``, and
``WorstResidualRefinement``. Their canonical JSON is parsed, validated, and
re-emitted by Rust; Python does not maintain a second validation/default path.
There is no implicit scientific ``Default`` at the high-level boundary.

Python batch example
~~~~~~~~~~~~~~~~~~~~

The public Core profile uses the installed public EFCC18 data package only to
obtain an explicit path; Rust validates that file before adopting a propagation
provider::

   from adam_core.orbit_determination import (
       CORE_OD_V1,
       core_od_v1_resource_bindings,
       run_od,
   )

   # source_catalog is a SourceCatalog and associations is an Associations
   # table. The propagator supplies propagation only.
   config = CORE_OD_V1.bind(core_od_v1_resource_bindings())
   fitted_orbits, fitted_members, failures = run_od(
       source_catalog,
       associations,
       propagator,
       config,
   )

Use ``config.for_iod()``, ``config.for_refinement()``, and
``config.for_evaluation()`` for the corresponding standalone stage. Never omit
``config`` and never pass a profile itself where a bound stage configuration is
required.

Observation pipelines
---------------------

The closed transform set is:

* ``FillMissingAstrometricUncertainty``;
* ``Efcc18CatalogBiasCorrection``;
* ``StationBandResidualCovarianceAdjustment`` with explicit Add/Replace and a
  minimum residual count;
* ``NightlyBatchDeweighting``.

Declaration order and repeated transforms are authoritative. Python callbacks,
dynamic registration, model dictionaries, and external science plugin ABIs are
not supported. Deliberate no-transform behavior is expressed only by
``ObservationPipeline.as_supplied()`` and is recorded as ``transforms=[]``.

Profiles and resources
----------------------

``CORE_OD_V1`` and ``ASTEROID_INSTITUTE_OD_V1`` are immutable path-free Rust
profile definitions. A caller binds every required semantic slot explicitly::

   from adam_core.orbit_determination import (
       ASTEROID_INSTITUTE_OD_V1,
       ResourceBindings,
   )

   config = ASTEROID_INSTITUTE_OD_V1.bind(
       ResourceBindings(
           {
               "efcc18_catalog_bias_table": efcc18_path,
               "station_catalog_uncertainty_table": station_sigma_path,
               "station_band_residual_covariance_table": station_covariance_path,
           }
       )
   )

The Institute profile applies those resources in this exact order: fill
missing uncertainty, EFCC18 correction, station/band residual covariance Add
with minimum count 30, then nightly deweighting with cap 4. ``CORE_OD_V1``
applies EFCC18 correction and nightly deweighting only. It intentionally has no
missing-uncertainty fallback while public-profile qualification remains open;
inputs without finite positive astrometric variances fail rather than receiving
an invented default.

Validation checks semantic slot, product/version, schema/version, SHA-256, byte
size, row count/structure, and table schema. Missing, corrupt, swapped, parse,
and structural failures are typed and aggregated by slot before science work.
Unused bindings fail. Absolute paths are never included in profile or bound
configuration fingerprints.

The profiles preserve the accepted d21 effective numerical constants:
``lt_tol=1e-12``, ephemeris maximum iterations 1000, and light-time maximum
iterations 10. Changing settings, transform order, resource bytes, or schemas
requires a new version/fingerprint decision.

Choose ``CORE_OD_V1`` when observations already carry finite positive
astrometric uncertainty and the public EFCC18 correction plus nightly
batch-deweighting policy is appropriate. Choose
``ASTEROID_INSTITUTE_OD_V1`` only when the separately supplied, licensed
station resources are available and its fill/add/deweight policy is appropriate
for the data. The names do not establish comparative accuracy, and Core 0.7
does not make a profile-superiority claim.

Standalone Rust resource validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rust does not import Python packages or discover resources. Pass absolute paths
through semantic slots::

   use adam_core::coords::{ResourceBindings, ResourceSlot, CORE_OD_V1};

   let bindings = ResourceBindings::new().bind(
       ResourceSlot::Efcc18CatalogBiasTable,
       "/absolute/path/to/bias.dat",
   );
   let config = CORE_OD_V1.bind(&bindings)?;

The repository includes the complete runnable example
``rust/adam_core/examples/validate_core_od_resource.rs``::

   cargo run -p adam_core --example validate_core_od_resource -- \
       /absolute/path/to/bias.dat

It validates bytes and structure without downloading, caching, or falling back
to another source.

Propagation integration tolerance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Profiles never select or configure a propagator. When ASSIST is the negotiated
propagation-only provider, its ordinary ``epsilon=1e-6`` propagation default
can produce evaluation noise under sub-metre perturbations on some long,
heavily observed main-belt arcs. Existing qualification found ``epsilon=1e-8``
to be converged and faster than tighter settings on a 30-object sample; use it
as an evidence-based starting point for that workload and re-qualify other
workloads. Avoid assuming that ``1e-12`` is safer: ASSIST state-transition
integration stalled at that setting in the qualification. Core's two-scale
weak-direction covariance diagnostic distinguishes approximately quadratic
curvature from step-independent evaluation noise and reports an integration-
tolerance hint; it checks consistency with the fit's own chi-square surface,
not physical uncertainty realism.

Canonical batch workflows
-------------------------

Rust parses and validates complete memberships before provider or pipeline
work. Null group IDs, unknown references, duplicate pairs, non-unique IDs, and
orbits without explicit members abort atomically. Groups follow first
appearance; members follow source-table order; overlapping hypotheses are
supported.

``run_od`` plans and prepares each group once, runs IOD, and refines the
accepted seed on the exact same rows. Original and used astrometry, including
exact diagonal variances, are persisted on members so standalone ``refine``
consumes a bitwise-lossless prepared handoff. Refinement and
evaluation fail closed when those snapshots are missing or when current source
astrometry differs from the persisted original snapshot. Per-member Rust
fingerprints also bind detection metadata, epochs/scale, observatory and observer
geometry, prepared astrometry, explicit orbit membership, and exact orbit state.
Expert kernels remain
under ``adam_core.orbit_determination.algorithms``. Standalone Rust exposes
predictor-level building blocks under ``adam_core_rs_coords::od::algorithms``.

Provenance
----------

Canonical batch results populate, where applicable:

* ordered complete source/member, membership, and exact orbit-state fingerprints;
* a validated provenance consistency digest covering route, complete
  configuration (including profile selection), input fingerprint, and every
  propagated IOD provider/backend/diagnostic/ambiguity field. The digest is
  unkeyed consistency validation, not authentication or a tamper-proof
  signature;
* focused route;
* native IOD/refinement provider identity and metadata;
* IOD and refinement diagnostics;
* complete canonical Rust configuration and its SHA-256;
* ambiguity, rejection count, and rejected fraction;
* original and used astrometry on member rows.

Resource paths are excluded. Lower-level numerical kernels that do not run a
complete configured workflow may legitimately leave runtime provenance fields
null.

Failure boundary
----------------

Malformed configuration, resources, IDs/member mappings, unsupported provider
capabilities and malformed provider responses are invocation-fatal with no
partial output. Only safely isolated scientific
no-solution/refinement/evaluation failures continue per group. They are rows of
``OrbitDeterminationFailures`` with typed stage counters, not opaque JSON.
``orbit_id`` is present only for an emitted state; ``input_orbit_id`` separately
identifies a refine/evaluate input when a state-less failure occurs. IOD
failures emit no placeholder orbit. A non-converged refinement remains in
``FittedOrbits`` with ``success=False`` only when Rust returned an actual best
state; local refinement/evaluation errors never relabel an untouched input seed.

Canonical IOD IDs include the self-contained ``IodRunConfig`` science
fingerprint and deliberately exclude later refinement settings. ``run_iod`` →
``refine`` composes that validated IOD configuration with the requested
``RefinementRunConfig``. A named profile identity is retained only when the
refinement matches that immutable profile; cross-configuration composition
drops the profile and exactly equals an explicit unprofiled ``run_od``.
Complete original membership is still required. Direct ``run_iod`` results are
valid evaluation inputs after full provenance/member/state validation, and
evaluation may use an explicit member subset. Evaluation uses only the
prediction fit and covariance dimension in its independent
``EvaluationConfig``. Provider ``RowFailure`` and other provider/protocol
errors remain invocation-fatal rather than being converted to group failures.
Evaluation outputs record evaluation-stage configuration/provider identity; they
do not copy prior-stage backend fields or member weights.
