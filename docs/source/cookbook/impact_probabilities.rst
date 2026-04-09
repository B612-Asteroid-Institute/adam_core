Impact Probabilities
====================

This section covers end-to-end impact probability estimation from uncertain orbits.

Core Pattern
------------

1. sample variants from orbit covariance
2. propagate variants and detect collisions
3. aggregate collision outcomes into cumulative probabilities

Simple End-to-End Example
-------------------------

.. code-block:: python

   from adam_assist import ASSISTPropagator
   from adam_core.dynamics.impacts import calculate_impacts, calculate_impact_probabilities

   propagator = ASSISTPropagator()

   variants, collisions = calculate_impacts(
       orbits=orbit,
       num_days=365,
       propagator=propagator,
       num_samples=5000,
       processes=16,
       seed=42,
   )

   impact_probabilities = calculate_impact_probabilities(variants, collisions)
   print(impact_probabilities.to_dataframe())

Custom Collision Conditions
---------------------------

.. code-block:: python

   from adam_core.coordinates import Origin
   from adam_core.dynamics.impacts import CollisionConditions

   conditions = CollisionConditions.from_kwargs(
       condition_id=["Earth", "Moon"],
       collision_object=Origin.from_kwargs(code=["EARTH", "MOON"]),
       collision_distance=[6420.0, 1740.0],
       stopping_condition=[True, True],
   )

   variants, collisions = calculate_impacts(
       orbits=orbit,
       num_days=365,
       propagator=propagator,
       num_samples=10000,
       processes=16,
       conditions=conditions,
   )

   impact_probabilities = calculate_impact_probabilities(
       variants,
       collisions,
       conditions=conditions,
   )

What the Output Represents
--------------------------

``ImpactProbabilities`` includes per-orbit, per-condition summaries such as:

* ``impacts`` and ``variants``
* ``cumulative_probability``
* ``mean_impact_time`` / ``stddev_impact_time``
* minimum and maximum impact times

Propagator Guidance
-------------------

``calculate_impacts`` requires a propagator that supports impact detection
(``ImpactMixin`` behavior). In practice, use a production propagator such as
``adam_assist.ASSISTPropagator`` for meaningful risk estimates.

When to Use
-----------

* probabilistic risk assessment
* comparing intervention/follow-up urgency across objects
* scenario analysis with alternate collision conditions

Related Reference
-----------------

* :doc:`../reference/dynamics`
* :doc:`moid_analysis`
* :doc:`../use_cases/impact_risk`

