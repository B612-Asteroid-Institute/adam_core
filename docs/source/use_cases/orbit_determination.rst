Orbit-determination configuration foundation
============================================

.. important::

   Core 0.7's canonical batch ``run_iod``, ``run_od``, ``refine``, and
   ``evaluate_orbits`` API is **not implemented by this foundation**. Those
   names are deliberately absent from ``adam_core.orbit_determination`` until
   the Associations/FittedOrbitMembers grouping and typed-failure work lands.
   The intended batch result is ``FittedOrbits``, ``FittedOrbitMembers``, and
   ``OrbitDeterminationFailures``; no one-group three-value signature is being
   frozen as the public API.

What this foundation implements
-------------------------------

Core Rust owns complete OD configuration, profile expansion, ordered
observation transforms, resource validation/parsing, fitting and rejection
settings, and path-free canonical configuration provenance. Python declares
immutable values, binds explicit local paths, marshals tables, and assembles
result tables. A propagation provider supplies only its negotiated native
capsule.

The complete configuration is ``OdConfig``. It contains:

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

Configured workflow foundation
------------------------------

This foundation does not expose interim Python one-group workflow functions.
Doing so would put orchestration and provenance behavior in Python or freeze a
signature that conflicts with the forthcoming batch contract. Expert numerical
kernels remain under ``adam_core.orbit_determination.algorithms``; they are not
complete configured workflows.

Standalone Rust exposes predictor-level configured building blocks under
``adam_core_rs_coords::od::algorithms``:
``prepare_observations``, ``run_iod_with_predictor``,
``refine_with_predictor``, and ``run_od_with_predictors``. They consume the
closed ``OdConfig`` path; the old executable ``dyn ObservationUncertaintyModel``
route is not public.

Provenance
----------

The forthcoming canonical batch results fail closed on unknown or dirty native
build identity and populate, where applicable:

* adam-core version and exact source commit;
* ordered input/used-astrometry fingerprint and focused route;
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

The current foundation validates configuration and bound resources and adopts
external capsules through the native ABI checks, but it does not yet own batch
IDs/member mappings, empty-group behavior, or the ordering of all invocation
preflights.

The forthcoming canonical batch layer will make malformed
configuration, resources, IDs/member mappings, unsupported provider
capabilities, malformed provider responses, and unavailable build identity
invocation-fatal before pipeline/numerical work, with no partial output. It will
classify only safely isolated scientific no-solution/refinement failures per
group and will populate canonical runtime provenance.
