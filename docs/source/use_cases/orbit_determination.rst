.. meta::
   :description: Orbit determination in adam_core with initial orbit determination and least-squares refinement.

Orbit Determination from Linked Observations
============================================

Problem
-------

You have linked detections and need fitted orbits with quantitative quality
metrics and outlier-aware membership outputs.

Implementation Options and Tradeoffs
------------------------------------

* ``initial_orbit_determination`` only:
  Fast triage and coarse fits for many candidates.
* ``initial_orbit_determination`` + ``fit_least_squares``:
  Higher-quality solutions and covariance updates, more compute and tuning.

Runnable Example
----------------

.. code-block:: python

   from adam_assist import ASSISTPropagator
   from adam_core.orbit_determination import (
       evaluate_orbits,
       fit_least_squares,
       initial_orbit_determination,
   )

   # observations: OrbitDeterminationObservations
   # linkage_members: FittedOrbitMembers with cluster/linkage assignments
   # These are typically produced by your detection association pipeline.

   propagator_cls = ASSISTPropagator

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

   # Refine one candidate orbit with differential correction.
   fitted_orbit, fitted_members = fit_least_squares(
       orbit=iod_orbits.take([0]).to_orbits(),
       observations=observations,
       propagator=propagator,
   )

   # Evaluate an orbit set against the same observation bundle.
   evaluated_orbits, evaluated_members = evaluate_orbits(
       orbits=iod_orbits.to_orbits(),
       observations=observations,
       propagator=propagator,
   )

When to Use This Pattern
------------------------

Use this for candidate confirmation, orbit quality scoring, and iterative
cleanup of observation-to-orbit assignments.

Related Documentation
---------------------

* :doc:`../reference/orbit_determination`
* :doc:`../reference/observations`
* :doc:`../reference/propagator`

Input Types
-----------
.. code-block:: python

   # initial_orbit_determination(observations: OrbitDeterminationObservations, linkage_members: FittedOrbitMembers, propagator: type[Propagator], ...) -> tuple[FittedOrbits, FittedOrbitMembers]
   # fit_least_squares(orbit: Orbits, observations: OrbitDeterminationObservations, propagator: Propagator, ...) -> tuple[FittedOrbits, FittedOrbitMembers]
   # evaluate_orbits(orbits: Orbits, observations: OrbitDeterminationObservations, propagator: Propagator, ...) -> tuple[FittedOrbits, FittedOrbitMembers]
