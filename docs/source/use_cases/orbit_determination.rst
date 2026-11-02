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

The complete configuration is ``OdConfig``. ``run_od`` receives it directly;
``run_iod`` receives its bound ``config.for_iod()`` projection. Standalone
``refine`` receives ``RefinementRunConfig`` and ``evaluate_orbits`` receives
the bound ``config.for_evaluation()`` projection. ``OdConfig`` contains:

* ``IodRunConfig``: search, selection, velocity, acceptance, correction,
  ranging, candidate-fit, solver, Jacobian, covariance, and two-body settings;
* ``RefinementRunConfig``: either complete CMC2003 or worst-residual settings;
* ``EvaluationConfig``;
* one ordered ``ObservationPipeline``.

Python exposes typed immutable declarations such as ``IodSearchConfig``,
``SpanSearchConfig``, ``CorrectionConfig``, ``RangingConfig``,
``WhitenedFitConfig``, ``TwoBodyFitConfig``, ``Cmc2003Refinement``, and
``WorstResidualRefinement``. Their canonical JSON is parsed, validated, and
re-emitted by Rust; Python does not maintain a second validation/default path.
There is no implicit scientific ``Default`` at the high-level boundary.

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

Canonical batch results fail closed on unknown or dirty native build identity
and populate, where applicable:

* adam-core version and exact source commit;
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
capabilities, malformed provider responses, and unavailable build identity are
invocation-fatal with no partial output. Only safely isolated scientific
no-solution/refinement/evaluation failures continue per group. They are rows of
``OrbitDeterminationFailures`` with typed stage counters, not opaque JSON.
``orbit_id`` is present only for an emitted state; ``input_orbit_id`` separately
identifies a refine/evaluate input when a state-less failure occurs. IOD
failures emit no placeholder orbit. A non-converged refinement remains in
``FittedOrbits`` with ``success=False`` only when Rust returned an actual best
state; local refinement/evaluation errors never relabel an untouched input seed.

Canonical IOD IDs deliberately include the complete ``OdConfig``. Exact
``run_iod`` → ``refine`` composition therefore requires the same refinement
projection and complete original membership. Direct ``run_iod`` results are
valid evaluation inputs after full provenance/member/state validation, and
evaluation may use an explicit member subset. Evaluation uses predictor settings
from the complete configuration's refinement fit and the covariance dimension
from ``EvaluationConfig``. Provider ``RowFailure`` and other provider/protocol
errors remain invocation-fatal rather than being converted to group failures.
Evaluation outputs record evaluation-stage configuration/provider identity; they
do not copy prior-stage backend fields or member weights.
