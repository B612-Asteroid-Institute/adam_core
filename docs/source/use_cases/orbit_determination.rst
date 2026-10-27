.. meta::
   :description: Explicit Rust-owned orbit-determination configuration, profiles, resources, and observation transforms in adam_core.

Orbit determination
===================

Core 0.7 owns observation preparation, IOD, differential correction,
rejection, covariance, configuration, and provenance.  A propagation provider
supplies propagation only.  Every high-level call requires an explicit complete
:class:`~adam_core.orbit_determination.OdConfig`; no scientific default is
selected when ``config`` is omitted.

The four canonical workflow names are ``run_iod``, ``run_od``, ``refine``, and
``evaluate_orbits``.  A one-object call is a one-group use of the same staged
contract. ``run_od`` prepares each group once and is exactly IOD followed by
refinement on those already-prepared observations.

Choosing a complete configuration
---------------------------------

Named profiles are immutable, path-free Rust definitions mirrored from
``adam_core_rs_coords::od::profiles`` into
``adam_core.orbit_determination.profiles``.  Profiles never select a
propagator, discover packages, download data, or fall back to another profile.
Callers bind concrete local files to semantic resource slots explicitly::

   from adam_core.orbit_determination import (
       CORE_OD_V1,
       ResourceBindings,
       run_od,
   )

   resources = ResourceBindings({
       "efcc18_catalog_bias_table": bias_dat_path,
   })
   config = CORE_OD_V1.bind(resources)
   fitted_orbits, fitted_members = run_od(
       observations,
       propagator,
       config=config,
   )

All required resources are read and validated before propagation or fitting.
Failures are aggregated and identify each semantic slot. Validation checks the
profile's product/version, schema/version, exact byte length, SHA-256, row
count, and structural schema. Absolute paths are deployment details: they are
never included in scientific identity or provenance.

``CORE_OD_V1``
~~~~~~~~~~~~~~

``CORE_OD_V1`` is the portable Core reference configuration. Its ordered
observation pipeline is:

#. :class:`~adam_core.orbit_determination.Efcc18CatalogBiasCorrection`, bound
   to ``efcc18_catalog_bias_table``;
#. :class:`~adam_core.orbit_determination.NightlyBatchDeweighting` with
   ``effective_observation_cap=4``.

It uses the Core-v1 bounded corrected-IOD policy, Core-v1 CMC2003 refinement,
and six-dimensional covariance evaluation. Selection is always explicit;
qualification work may recommend this profile but Core does not auto-select it.

``ASTEROID_INSTITUTE_OD_V1``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ASTEROID_INSTITUTE_OD_V1`` is a separately selected reference profile. Its
ordered pipeline is:

#. :class:`~adam_core.orbit_determination.FillMissingAstrometricUncertainty`,
   bound to ``station_catalog_uncertainty_table``;
#. :class:`~adam_core.orbit_determination.Efcc18CatalogBiasCorrection`, bound
   to ``efcc18_catalog_bias_table``;
#. :class:`~adam_core.orbit_determination.StationBandResidualCovarianceAdjustment`
   in ``Add`` mode with ``minimum_residual_count=30``, bound to
   ``station_band_residual_covariance_table``;
#. :class:`~adam_core.orbit_determination.NightlyBatchDeweighting` with
   ``effective_observation_cap=4``.

The profile uses the same complete numerical stage policies as
``CORE_OD_V1``. Its extra resources are supplied separately under their
semantic slots. Core's public documentation intentionally does not prescribe a
private distribution, repository, credential, or acquisition mechanism.
Explicit Institute selection never falls back to ``CORE_OD_V1``.

Manual configuration
--------------------

Manual callers construct every stage explicitly.  Deliberate no-transform
behavior is spelled :meth:`ObservationPipeline.as_supplied`; its provenance is
``transforms=[]``::

   from adam_core.orbit_determination import (
       EvaluationConfig,
       IodRunConfig,
       OdConfig,
       ObservationPipeline,
       RefinementRunConfig,
       ResourceBindings,
   )

   config = OdConfig.explicit(
       observation_pipeline=ObservationPipeline.as_supplied(),
       iod=IodRunConfig.core_v1(),
       refinement=RefinementRunConfig.core_v1(),
       evaluation=EvaluationConfig.six_dimensional(),
       resources=ResourceBindings({}),
   )

For a transformed manual pipeline, use ``ObservationPipeline.ordered`` with
Core-defined immutable transform values, then bind every referenced resource
through ``ResourceBindings``. Declaration order and repeated transforms are
scientifically meaningful and are never sorted, deduplicated, or combined.
Python transform objects are declarative only; Rust validates, parses, and
executes them once per group.

Observation provenance
----------------------

``FittedOrbitMembers`` retains both the caller's original astrometry and the
astrometry consumed by the numerical kernels. ``FittedOrbits.provenance``
contains canonical expanded configuration JSON: profile identity and
fingerprint, ordered transforms and their parameters, expected and actual
resource identities, IOD/refinement/evaluation settings, and no absolute
paths.

Nightly deweighting groups by observatory and ``floor(UTC MJD)``. For ``N``
rows and cap ``C``, it multiplies the complete RA/Dec covariance block by
``max(N/C, 1)``; correlation is preserved. Metadata/resource covariance
adjustments belong in the observation pipeline. Residual-adaptive weighting
and rejection belong in refinement.

Migration from Core 0.6
-----------------------

Core 0.7 removes the executable Python ``models=`` and
``observatory_bias_model=`` paths, ``models="default"``,
``ObservationUncertaintyModel``, ``CompositeModel``,
``apply_observation_models``, ``initial_orbit_determination``, ``iod``,
``differential_correction``, ``NativeOrbitFitter``, ``OrbitFitter``, and public
linkage helpers. There are no aliases or silent translations. Useful expert
numerical primitives are organized under
``adam_core.orbit_determination.algorithms``; they are not alternate profile or
high-level workflow APIs.
