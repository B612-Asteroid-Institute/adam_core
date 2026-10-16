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

* ``run_od``: the whole step for one object (observation models, IOD,
  refinement, provenance); the blessed entry point.
* ``run_iod`` then ``refine``: the same two stages called separately, e.g. to
  inspect the IOD candidates (``IodReport``) or choose the rejection loop.
* ``initial_orbit_determination``: ``run_iod`` over many linkages at once.

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
       initial_orbit_determination,
       refine,
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

   # Refine one candidate orbit (CMC2003 rejection by default;
   # rejection="worst_residual" or "none" select the other loops).
   fitted_orbit, fitted_members = refine(
       ranked_orbits.take([0]),
       observations,
       propagator,
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
  touched. adam_core bundles no sigma table: with no ``sigma_table`` the model
  imports the private ``adam-observatory-uncertainties`` data package and reads
  its ``v2_sigma_fill`` table (the Asteroid Institute default), the way
  EFCC18 imports ``jpl_debias_2018``; ``VeresFloorModel`` /
  ``VeresReplaceModel`` read that package's legacy ``veres2017_working`` table.
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

The shipped defaults
~~~~~~~~~~~~~~~~~~~~

Calling the pieces with no arguments gives the Asteroid Institute default
configuration (the outcome of the 100-object walk-forward study, decision
2026-09-23). Every default is the default value of the signature that owns
it, and ``run_od`` called without ``models`` applies the default model stack,
which is written out in its body and explained in its docstring:

.. code-block:: python

   from adam_core.orbit_determination import NativeOrbitFitter, run_od

   fitted_orbits, members = run_od(
       observations,
       NativeOrbitFitter(ASSISTPropagator()),
       propagator=ASSISTPropagator(),
   )

Some of the default models import data packages for their tables.
``SigmaFillModel()`` and ``EmpiricalCovarianceModel()`` load
``v2_sigma_fill`` and ``v2_full`` from the private
``adam-observatory-uncertainties`` package, and ``EFCC18DebiasModel()``
locates JPL's ``bias.dat`` through the ``jpl_debias_2018`` package, the
``ADAM_CORE_EFCC18_BIAS_DAT`` environment variable or the EFCC18 cache.
adam_core does not depend on these packages: without them ``run_od`` raises
an ``ImportError`` / ``FileNotFoundError`` naming the missing one.
``models=None`` fits the observations exactly as supplied, and an explicit
model list built with your own tables needs no package.

.. list-table::
   :header-rows: 1
   :widths: 22 44 34

   * - Lever
     - Default
     - Owned by
   * - Sigma where the MPC reports none
     - ``SigmaFillModel`` with the ``v2_sigma_fill`` table, 0.75" fallback
       (fills ONLY missing sigmas; reported sigmas are never touched)
     - ``SigmaFillModel()`` / ``load_sigma_table()``
   * - Star-catalog debias (positions)
     - ``EFCC18DebiasModel``: JPL ``bias.dat`` in HEALPix RING order, every
       tabulated catalog corrected
     - ``EFCC18DebiasModel()`` (``exclude_astcats=()``)
   * - Station weighting
     - ``EmpiricalCovarianceModel``: the station's measured 2x2 residual
       covariance is ADDED to the reported one; stations with fewer than 30
       residuals pass through
     - ``EmpiricalCovarianceModel()`` (``mode="add"``, ``min_resid_cov_n=30``)
       / ``load_bias_table()`` (``v2_full``)
   * - Nightly deweighting
     - sigma scaled by sqrt(N/4) for same-station same-night batches with N > 4
     - ``NightBatchDeweightingModel()`` (``cap=4``)
   * - Model order
     - fill, then EFCC18, then empirical covariance, then nightly deweighting
       (fill first so every later model sees a finite covariance; debias
       positions before anything reads them)
     - ``run_od(models="default")``
   * - Outlier rejection
     - Carpino-Milani-Chesley (2003) with OrbFit's ``reject.def`` constants
       (reject 8, recover 7, frac 0.25, 15 passes, 50% max rejected, 180-day
       apparitions, 5% eigenvalue floor). Each refit shares a full-predictor
       central-difference Jacobian between covariance and rejection, even
       when optimizer steps use the analytic 2-body Jacobian.
     - ``NativeOrbitFitter()`` (``outlier_rejection="cmc2003"``) / ``refine()``
   * - Differential correction
     - whitened residuals, derivatives that follow the dynamics,
       weak-direction covariance probe, linear loss
     - ``refine()`` ``fit_settings`` (``jacobian="auto"``: analytic for
       two-body propagators, the backend's state-transition matrices for
       native backends such as ASSIST, central differences otherwise;
       ``validate_covariance=True``, ``loss="linear"``)
   * - IOD
     - Gauss on span-bounded triplets (about 6 days), each orbit a seed for
       staged differential corrections to the whole arc; at least 6
       observations over 1 day. See *The IOD policy* below.
     - ``NativeOrbitFitter()`` / ``run_iod()``
       (``observation_selection_method="spans"``, ``iod_acceptance="corrected"``,
       ``iod_rchi2_threshold=200``, ``min_obs=6``, ``min_arc_length=1``)
   * - Provenance
     - ``FittedOrbitMembers`` record original vs used astrometry, ``astcat``
       and ``weight``; the models act on the fit only. Judge a held-out
       observation against its nominal position and original sigma.
       ``FittedOrbits.provenance`` (``OdProvenance``) records the
       adam-core version, route, fitter, IOD and refinement backends,
       settings, model descriptors with table hashes, IOD ambiguity, and the
       number and fraction of observations rejected (CMC2003 may reject up
       to half the arc; the fraction shows how much it did).
     - ``run_od``

Kept as options but not defaults: ``loss="huber"`` (open covariance pathology
on short arcs), ``VeresFloorModel`` / ``VeresReplaceModel``, ``SigmaFloorModel``
(no-op on the study), ``PerformanceWeightedModel`` (over-inflates by about
1.5x), ``NightBatchDeweightingModel(cap=1)`` (sqrt(N), about 1.9x inflation
with no position benefit) and ``outlier_rejection="worst_residual"`` (the
pre-2026-09 loop). The lower-level entry points (``initial_orbit_determination``,
``differential_correction``, ``iterative_fit``) take an optional
``observatory_bias_model`` that defaults to None: they are building blocks,
not the shipped pipeline, and work without the data packages. The same notes
are in the ``run_od`` docstring.

The IOD policy
~~~~~~~~~~~~~~

A Gauss preliminary orbit is a seed for differential correction, not the
answer (Milani & Gronchi 2010), as in OrbFit, Find_Orb and JPL's pipeline.
Every number is a Rust ``Default`` that documents its source; Python reads
them from ``IOD_DEFAULTS`` (``from adam_core.orbit_determination import
IOD_DEFAULTS``), so the signature defaults of ``run_iod`` and
``NativeOrbitFitter`` cannot drift from Rust.

* **Ordering and determinism.** Observations are searched in a canonical
  order (epoch, station, input row); the same input gives the same orbit
  whatever its row order. A triplet never repeats an epoch.
* **Inputs.** Ecliptic observations and equatorial observers are rotated to
  the frames Gauss uses; observers not centred on the Sun are refused. An
  arc shorter than ``min_arc_length`` (1 day) is not searched, and a fit
  whose rejected observations leave a shorter arc is not accepted.
* **Nights.** A station's observations form one night until a gap of more
  than 8 hours (or 16 hours of continuous coverage), so nights that straddle
  0h UTC are not split.
* **Triplets.** Gauss truncates the f and g series, so a preliminary orbit
  degrades with the arc its triplet spans (on 2012 TF79 the full-arc reduced
  chi2 grows from about 1e4 at a 3-day span to 1e9 at 67 days). The search
  visits the densest 10 days, the densest 30 days, the last and first 30
  days and the whole arc, and in each prefers triplets spanning about 6
  days, at most one per pair of nights, 32 triplets in all
  (``span_settings``).
* **Velocity.** ``velocity_method="auto"`` by default, which switches to
  Herrick-Gibbs when consecutive positions are within 5 degrees (Vallado
  2013, sec. 7.5.2).
* **Correction.** Each raw Gauss orbit is scored on the observations near
  its own triplet (``prefit_scope="local"``; ``"whole_arc"`` is the legacy
  order), which is how good it is where it was built rather than how far
  it extrapolates. In that order, up to 8 candidates are corrected in
  stages: the triplet's span (at least 6 observations), then
  spans grown fourfold (an apparition gap jumps to the nearest night), each
  stage within ``rchi2_threshold``. Stages before the whole arc reject
  nothing and down-weight with Huber's loss (``correction_settings=
  {"stage_loss": "linear"}`` turns that off); the whole-arc stage is a plain
  fit with Carpino, Milani & Chesley (2003) rejection when above the
  threshold, rejecting at most ``contamination_percentage`` (20 %) of the
  arc and never leaving fewer than ``min_obs`` (6); a fit that keeps fewer
  than ``min_obs`` is not accepted. The search stops after 2 candidates reach the whole arc
  (``correction_settings``).
* **Multiple roots.** Every physical root of a triplet's Gauss polynomial is
  a candidate (Charlier: up to three): real to a relative 1e-6, so a
  near-double root is kept, and with every topocentric range at least
  1e-4 au, so the spurious zero-range root is dropped. The other roots of a
  converged candidate's triplet are corrected too.
* **Ranking and ambiguity.** Candidates are compared on the whole arc:
  every observation's chi2 clipped at the CMC2003 rejection threshold, so a
  rejected observation still counts. The lowest score wins (ties within
  1e-6 go to the earliest candidate). ``IodReport.ambiguous_with`` lists
  converged solutions outside the accepted orbit's 3-sigma region whose
  score is within the same margin (Gronchi 2009).
* **Short arcs and Gauss failures: ranging.** An arc too short for an
  orbit (``min_arc_length``) returns no orbit but the systematic-ranging
  sample (``IodReport.ranging``): the attributable (position and rates) is
  completed at every range and range rate of the admissible region (bound
  to the Sun, not to the Earth; Milani et al. 2004), each orbit scored on
  the observations and weighted by ``exp(-delta_chi2/2)`` (Farnocchia,
  Chesley & Micheli 2015). ``RangingSample.to_variant_orbits`` gives the
  sample as weighted ``VariantOrbits``. When no Gauss candidate converges,
  the best 3 ranging orbits are corrected exactly like Gauss candidates
  (``IodCandidate.source == "ranging"``). Ranging runs on arcs of at most
  5 days (a few nights; ``ranging_max_arc_days``); ``ranging_settings``
  tunes or disables it.
* **Failure.** A candidate whose prediction fails is skipped and counted;
  when nothing is accepted, ``IodReport.not_found_reason`` says why.
* **Evidence.** On the first solutions of 126 walk-forward objects
  (ASSIST, MPC observations) the policy seeded 116 of 122, against 16 for
  the legacy ``"combinations"`` search with the prefit gate and 114 for the
  study's seeding ladder, at the ladder's final reduced chi2; single-object
  reruns since recover 433 Eros and 3200 Phaethon. The legacy
  policy stays selectable (``observation_selection_method="combinations"``,
  ``acceptance="prefit"``); ``iod()`` keeps it.

Migrating to the native stages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The overlapping entry points converge on three stages, ``run_iod``,
``refine`` and their composition ``run_od``. The old names still work and
emit a ``DeprecationWarning`` naming their replacement; they will be removed
in a later release.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Deprecated
     - Use
   * - ``iod(observations, PropagatorClass, ...)``
     - ``run_iod(observations, propagator, ...)``; the legacy policy is
       ``acceptance="prefit", observation_selection_method="combinations",
       velocity_method="gibbs"``
   * - ``iod_worker(...)``
     - ``initial_orbit_determination(...)`` (``run_iod`` per linkage)
   * - ``fit_least_squares(orbit, observations, propagator, ...)``
     - ``refine(orbit, observations, propagator, rejection="none",
       fit_settings={...})``
   * - ``iterative_fit(...)``
     - ``refine(..., rejection="worst_residual", rejection_settings={...})``
   * - ``cmc2003_fit(...)``
     - ``refine(...)`` (``rejection="cmc2003"``, the default);
       ``cmc2003_fit_detailed`` remains for the loop diagnostics
   * - ``od.od``, ``od.differential_correction``, ``LeastSquares``
     - ``refine`` (or ``run_od`` for the whole step). They already delegate
       to it (``rejection="worst_residual"`` for ``od``, ``"none"`` for
       ``LeastSquares``); their finite-difference settings are ignored
   * - ``NativeOrbitFitter(PropagatorClass, propagator_kwargs)``
     - ``NativeOrbitFitter(PropagatorClass(**propagator_kwargs))``: the IOD
       propagator is an instance
   * - OD with a Python-only propagator (predictions through
       ``generate_ephemeris``)
     - Not supported: OD runs natively and raises ``TypeError``. Use
       adam-core's two-body propagator or a native backend that declares an
       external propagator (adam-assist >= 0.5's ``ASSISTPropagator``)

``run_od``, ``run_iod``, ``refine``, ``initial_orbit_determination``,
``evaluate_orbits``, ``cmc2003_fit_detailed`` and the ``OrbitFitter``
interface (``NativeOrbitFitter``, plugins such as adam_fo) are not
deprecated.

The pieces, one by one
~~~~~~~~~~~~~~~~~~~~~~

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
       ASSISTPropagator(),
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

Custom models and fitters
~~~~~~~~~~~~~~~~~~~~~~~~

``run_od`` can combine model application and fitting into one Rust call when
their native specifications describe the Python implementations. Subclass
overrides are honored through the ordinary composition by default:

* A model overriding ``apply`` must also define ``_native_specs`` to opt into
  the combined call. The specification must reproduce the complete custom
  transformation. This check also applies inside nested ``CompositeModel``
  instances.
* A ``NativeOrbitFitter`` overriding ``full_od``, ``initial_fit`` or
  ``refine_fit`` must also define ``native_settings`` to opt into the combined
  call. Those settings must fully describe the custom behavior in the Rust
  drivers. Custom logic that cannot be expressed by those drivers should
  leave ``native_settings`` inherited.

Subclasses that inherit the fitting/model methods unchanged remain eligible
for fusion. Further subclasses that change those methods must renew the
explicit opt-in. The ordinary composition still uses Rust numerical drivers
inside the built-in fitting stages.

The refinement solver (``refine``) minimizes the whitened (lon, lat) residuals with
an exact two-body Jacobian (Rust autodiff) and validates the covariance along
its weakest direction, so line-of-sight uncertainties are no longer fabricated
by finite differences. Pass ``jacobian="2-point"`` to reach a propagator's
fused Rust Gauss-Newton work unit with the legacy forward-difference
covariance.

Solver budgets
~~~~~~~~~~~~~~

The solver accepts two independent limits (``refine`` ``fit_settings``): ``max_iterations``
(default 100) caps optimizer iterations, while ``max_nfev`` (default None)
caps solver residual-vector evaluations. Supplying both enforces both.
The evaluation budget includes the initial state, rejected trial steps, and
every finite-difference candidate: a forward-difference Jacobian costs six
evaluations and a central-difference Jacobian costs twelve, even when batched
into one backend call. A batch is only started if it fits in the remaining
budget. Final covariance calculation, covariance validation, and fit reporting
are outside this solver budget.

Budget exhaustion returns the current solution with ``success=False`` and
``status_code=0`` unless convergence was established on the final permitted
evaluation. The legacy ``FittedOrbits.iterations`` column records solver
residual-vector evaluations, including numerical Jacobians. Each fit within
a rejection loop gets a fresh evaluation budget; CMC2003's ``max_iterations``
continues to control its rejection loop.

When to Use This Pattern
------------------------

Use this for candidate confirmation, orbit quality scoring, and iterative
cleanup of observation-to-orbit assignments.

Related Documentation
---------------------

* :doc:`../reference/api/adam_core.orbit_determination`
* :doc:`../reference/api/adam_core.observations`
* :doc:`../reference/api/adam_core.propagator`
