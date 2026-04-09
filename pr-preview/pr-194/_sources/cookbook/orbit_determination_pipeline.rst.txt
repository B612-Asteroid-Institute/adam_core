Orbit Determination Pipeline
============================

Simple Case (IOD)
-----------------

.. code-block:: python

   from adam_assist import ASSISTPropagator
   from adam_core.orbit_determination import initial_orbit_determination

   iod_orbits, iod_members = initial_orbit_determination(
       observations=observations,
       linkage_members=linkage_members,
       propagator=ASSISTPropagator,
       min_obs=6,
       min_arc_length=1.0,
       max_processes=4,
   )

Advanced Case (Least Squares + Evaluation)
------------------------------------------

.. code-block:: python

   from adam_assist import ASSISTPropagator
   from adam_core.orbit_determination import evaluate_orbits, fit_least_squares

   propagator = ASSISTPropagator()

   fitted_orbit, fitted_members = fit_least_squares(
       orbit=iod_orbits.take([0]).to_orbits(),
       observations=observations,
       propagator=propagator,
   )

   evaluated_orbits, evaluated_members = evaluate_orbits(
       orbits=iod_orbits.to_orbits(),
       observations=observations,
       propagator=propagator,
   )

When to Use
-----------

* IOD only for fast triage at scale.
* Add least-squares when publishing, ranking, or scheduling follow-up.
* ``evaluate_orbits`` is useful for consistent quality scoring across candidates.

Related Reference
-----------------

* :doc:`../reference/orbit_determination`
* :doc:`../reference/propagator`

Input Types
-----------
.. code-block:: python

   # initial_orbit_determination(observations: OrbitDeterminationObservations, linkage_members: FittedOrbitMembers, propagator: type[Propagator], ...) -> tuple[FittedOrbits, FittedOrbitMembers]
   # fit_least_squares(orbit: Orbits, observations: OrbitDeterminationObservations, propagator: Propagator, ...) -> tuple[FittedOrbits, FittedOrbitMembers]
   # evaluate_orbits(orbits: Orbits, observations: OrbitDeterminationObservations, propagator: Propagator, ...) -> tuple[FittedOrbits, FittedOrbitMembers]
