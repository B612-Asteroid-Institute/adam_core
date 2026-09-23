.. meta::
   :description: Orbit determination in adam_core with initial orbit determination and least-squares refinement.

Orbit Determination from Linked Observations
============================================

Problem
-------

You have linked detections and need fitted orbits with quantitative quality
metrics and outlier-aware membership outputs.

What You Get Back
-----------------

The orbit-determination APIs return two synchronized tables:

* ``FittedOrbits``: one row per solved candidate with quality metrics
  (``orbit_id``, ``arc_length``, ``num_obs``, ``chi2``, ``reduced_chi2``,
  convergence/status fields, and best-fit Cartesian state).
* ``FittedOrbitMembers``: one row per observation assignment with
  ``orbit_id`` + ``obs_id`` plus per-observation residuals and outlier flags.

You use them together to rank candidates, inspect residual structure, remove
poor fits, and feed validated solutions downstream.

Implementation Options and Tradeoffs
------------------------------------

* ``initial_orbit_determination`` only:
  Fast triage and coarse fits for many candidates.
* ``initial_orbit_determination`` + ``fit_least_squares``:
  Higher-quality solutions and covariance updates, more compute and tuning.

Runnable Example
----------------

.. code-block:: python

   import pyarrow.compute as pc
   from adam_assist import ASSISTPropagator
   from adam_core.orbit_determination import (
       FittedOrbitMembers,
       FittedOrbits,
       OrbitDeterminationObservations,
       evaluate_orbits,
       fit_least_squares,
       initial_orbit_determination,
   )

   observations: OrbitDeterminationObservations
   linkage_members: FittedOrbitMembers
   # These are typically produced by your detection association pipeline.

   propagator_cls: type[ASSISTPropagator] = ASSISTPropagator

   iod_orbits: FittedOrbits
   iod_members: FittedOrbitMembers
   iod_orbits, iod_members = initial_orbit_determination(
       observations=observations,
       linkage_members=linkage_members,
       propagator=propagator_cls,
       min_obs=6,
       min_arc_length=1.0,
       chunk_size=1,
       max_processes=4,
   )

   propagator = ASSISTPropagator()

   # Inspect top candidates by quality metric.
   ranked_orbits = iod_orbits.sort_by([("reduced_chi2", "ascending")])

   # Refine one candidate orbit with differential correction.
   # (Converts one fitted row to Orbits for the least-squares fitter.)
   fitted_orbit, fitted_members = fit_least_squares(
       orbit=ranked_orbits.take([0]).to_orbits(),
       observations=observations,
       propagator=propagator,
   )

   # Evaluate an orbit set against the same observation bundle.
   evaluated_orbits, evaluated_members = evaluate_orbits(
       orbits=iod_orbits.to_orbits(),
       observations=observations,
       propagator=propagator,
   )

   # Example: count non-outlier members by orbit_id.
   non_outlier = evaluated_members.apply_mask(pc.invert(evaluated_members.outlier))
   print(non_outlier.group_by("orbit_id").aggregate([("obs_id", "count")]))

Fit-Time Observation Models and Provenance
------------------------------------------

``run_od`` is the blessed entry point when the provenance of a fit matters.
It applies observation models to the ORIGINAL observations at fit time,
drives any ``OrbitFitter`` backend (``NativeOrbitFitter``: Gauss IOD followed
by differential correction; plugins such as adam_fo override ``full_od``), and
returns members carrying the original and the used astrometry of every
observation. The models never reach the fitter; it receives pre-transformed
observations, so the ``OrbitFitter`` interface stays flag-free.

* ``SigmaFillModel`` fills ONLY the per-axis sigmas an observation lacks (MPC
  records without ``rmsRACosDec`` / ``rmsDec``) from a station/catalog sigma
  table (``VERES2017_SIGMA_TABLE_SCHEMA``; rows may be (station, catalog),
  station-only, catalog-only or one global row). Reported sigmas are never
  touched. The Asteroid Institute default table is ``v2_sigma_fill`` from the
  private ``adam-observatory-uncertainties`` package; the bundled Veres et al.
  (2017) table is the legacy reference.
* ``EFCC18DebiasModel`` subtracts the Eggl et al. (2020) star-catalog bias
  from the observed positions (JPL's ``bias.dat`` in HEALPix RING order,
  keyed on the ``astcat`` column carried by
  ``OrbitDeterminationObservations.from_ades``). It is the one shipped model
  that moves positions.
* ``EmpiricalCovarianceModel``, ``PerformanceWeightedModel`` and
  ``SigmaFloorModel`` interpret an observatory bias table
  (``BIAS_TABLE_SCHEMA``) as covariance only; ``NightBatchDeweightingModel``
  and the VFC2017 ``VeresFloorModel`` / ``VeresReplaceModel`` need no table.
  None of them changes a position.
* ``CompositeModel`` (or a plain sequence) applies models left to right.

Recommended defaults (Asteroid Institute 100-object walk-forward study,
decision 2026-09-23): fill missing sigmas with ``SigmaFillModel`` first so
every later model sees a finite covariance, debias positions with
``EFCC18DebiasModel`` before anything reads them, add the station's empirical
residual covariance with ``EmpiricalCovarianceModel(bias_table, mode="add")``,
deweight same-station same-night batches with
``NightBatchDeweightingModel(cap=4)`` (sigma scaled by sqrt(N/4) for N > 4),
and fit with ``NativeOrbitFitter(outlier_rejection="cmc2003")``. Kept as
options but not recommended as defaults: ``loss="huber"`` (open covariance
pathology on short arcs), ``VeresFloorModel`` / ``VeresReplaceModel``,
``SigmaFloorModel`` (no-op), ``PerformanceWeightedModel`` (over-inflates by
about 1.5x) and ``NightBatchDeweightingModel(cap=1)`` (sqrt(N), about 1.9x
inflation with no position benefit). The models act on the fit only: judge a
held-out observation against its nominal position and original sigma.

.. code-block:: python

   from adam_core.orbit_determination import (
       CompositeModel,
       EFCC18DebiasModel,
       EmpiricalCovarianceModel,
       NativeOrbitFitter,
       NightBatchDeweightingModel,
       SigmaFillModel,
       run_od,
   )

   fitter = NativeOrbitFitter(
       propagator_class=ASSISTPropagator,
       outlier_rejection="cmc2003",      # Carpino-Milani-Chesley reject/re-include
   )
   models = CompositeModel(
       SigmaFillModel(sigma_fill_table, fallback_sigma_arcsec=None),  # fill first
       EFCC18DebiasModel(),                                           # RING order
       EmpiricalCovarianceModel(bias_table),                          # mode="add"
       NightBatchDeweightingModel(cap=4),
   )
   fitted_orbits, members = run_od(
       observations,
       fitter,
       models,
       propagator=ASSISTPropagator(),
       object_id="2024 XY",
   )
   # members.original_astrometry vs members.used_astrometry show what the fit saw;
   # members.weight holds per-observation weights, members.astcat the star catalog.

``fit_least_squares`` itself minimizes the whitened (lon, lat) residuals with
an exact two-body Jacobian (Rust autodiff) and validates the covariance along
its weakest direction, so line-of-sight uncertainties are no longer fabricated
by finite differences. Pass ``jacobian="2-point"`` to reach a propagator's
fused Rust Gauss-Newton work unit with the legacy forward-difference
covariance.

When to Use This Pattern
------------------------

Use this for candidate confirmation, orbit quality scoring, and iterative
cleanup of observation-to-orbit assignments.

Related Documentation
---------------------

* :doc:`../reference/api/adam_core.orbit_determination`
* :doc:`../reference/api/adam_core.observations`
* :doc:`../reference/api/adam_core.propagator`
