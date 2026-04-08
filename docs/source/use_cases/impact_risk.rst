.. meta::
   :description: Impact risk analysis in adam_core using sampled variants, collision conditions, and impact probability summaries.

Impact Risk Analysis
====================

Problem
-------

You need to estimate impact likelihood and impact corridor behavior from a
nominal orbit with uncertainty, while preserving reproducibility.

Implementation Options and Tradeoffs
------------------------------------

* Lower sample counts (for example ``num_samples=1000``):
  Faster turnaround, lower statistical confidence in rare-event tails.
* Higher sample counts (for example ``num_samples=10000+``):
  Better confidence for low probabilities, increased compute cost.

Runnable Example
----------------

.. code-block:: python

   from adam_core.coordinates import Origin
   from adam_core.dynamics.impacts import (
       CollisionConditions,
       calculate_impacts,
       calculate_impact_probabilities,
   )
   from adam_core.orbits.query import query_sbdb
   from adam_core.time import Timestamp
   from adam_assist import ASSISTPropagator

   orbit = query_sbdb(["2024 YR4"])

   approx_impact_date = Timestamp.from_iso8601(["2032-12-22"], scale="tdb")
   thirty_days_after = approx_impact_date.add_days(30)
   days_to_run, _ = thirty_days_after.difference(orbit.coordinates.time)

   conditions = CollisionConditions.from_kwargs(
       condition_id=["Earth", "Moon"],
       collision_object=Origin.from_kwargs(code=["EARTH", "MOON"]),
       collision_distance=[6420, 1740],
       stopping_condition=[True, True],
   )

   propagator = ASSISTPropagator()

   variants, impacts = calculate_impacts(
       orbit,
       days_to_run[0].as_py(),
       propagator,
       num_samples=10000,
       processes=60,
       conditions=conditions,
   )

   impact_probabilities = calculate_impact_probabilities(
       variants,
       impacts,
       conditions=conditions,
   )
   print(impact_probabilities.to_dataframe())

When to Use This Pattern
------------------------

Use this when quantifying planetary defense risk, evaluating follow-up urgency,
or communicating risk outcomes to decision-makers.

Related Documentation
---------------------

* :doc:`../examples/2024_yr4_impact_risk`
* :doc:`../reference/dynamics`
* :doc:`../reference/orbits`

Input Types
-----------
.. code-block:: python

   # CollisionConditions.from_kwargs(...) -> CollisionConditions
   # calculate_impacts(orbits: Orbits, num_days: int, propagator: Propagator, ...) -> tuple[VariantOrbits, CollisionEvent]
   # calculate_impact_probabilities(variants: VariantOrbits, collision_events: CollisionEvent, conditions: CollisionConditions | None = None) -> ImpactProbabilities
