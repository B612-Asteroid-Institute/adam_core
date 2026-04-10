.. meta::
   :description: Impact risk analysis in adam_core using sampled variants, collision conditions, and impact probability summaries.

Impact Risk Analysis
====================

Goal
----

Estimate impact probability from an uncertain orbit, then inspect where/when
impacts occur for downstream planning and communication.

What This Workflow Produces
---------------------------

* propagated variants from covariance sampling
* ``CollisionEvent`` rows (which condition was hit, when, and where)
* per-object ``ImpactProbabilities`` summary metrics

Backend Requirement
-------------------

Use a propagator that implements ``ImpactMixin`` collision detection.
For production analyses this typically means ``adam_assist.ASSISTPropagator``.

Pattern 1: Default Earth-Only Risk Estimate (Fastest Path)
-----------------------------------------------------------

.. code-block:: python

   from adam_assist import ASSISTPropagator
   from adam_core.dynamics.impacts import (
       CollisionEvent,
       ImpactProbabilities,
       calculate_impacts,
       calculate_impact_probabilities,
   )
   from adam_core.orbits import Orbits, VariantOrbits
   from adam_core.orbits.query import query_sbdb

   orbit: Orbits = query_sbdb(["2024 YR4"])
   propagator = ASSISTPropagator()

   variants: VariantOrbits
   impacts: CollisionEvent
   variants, impacts = calculate_impacts(
       orbits=orbit,
       num_days=365,
       propagator=propagator,
       num_samples=5000,
       processes=16,
       seed=42,
   )

   probabilities: ImpactProbabilities = calculate_impact_probabilities(variants, impacts)
   print(probabilities.to_dataframe())

Pattern 2: Custom Conditions + Explicit Stopping Semantics
----------------------------------------------------------

``CollisionConditions`` controls what counts as a collision and whether
propagation stops after each condition is met.

.. code-block:: python

   from adam_core.coordinates import Origin
   from adam_core.dynamics.impacts import (
       CollisionConditions,
       CollisionEvent,
       ImpactProbabilities,
       calculate_impacts,
       calculate_impact_probabilities,
   )
   from adam_core.orbits import VariantOrbits

   conditions: CollisionConditions = CollisionConditions.from_kwargs(
       condition_id=["Earth", "Moon"],
       collision_object=Origin.from_kwargs(code=["EARTH", "MOON"]),
       collision_distance=[6420.0, 1740.0],  # km thresholds
       stopping_condition=[True, False],
   )

   variants: VariantOrbits
   impacts: CollisionEvent
   variants, impacts = calculate_impacts(
       orbits=orbit,
       num_days=365,
       propagator=propagator,
       num_samples=10000,
       processes=32,
       seed=7,
       conditions=conditions,
   )

   probabilities: ImpactProbabilities = calculate_impact_probabilities(
       variants,
       impacts,
       conditions=conditions,
   )
   print(probabilities.to_dataframe())

Stopping Conditions: What They Mean
-----------------------------------

* ``stopping_condition=True``:
  after that collision is detected for a variant, propagation can terminate for
  that variant.
* ``stopping_condition=False``:
  propagation continues, allowing later events to be detected (backend behavior
  determines exact event sequencing).
* mixed settings are valid per condition and useful for "terminal Earth impact"
  plus "continue tracking Moon events" style scenarios.

Pattern 3: Bring Your Own Variant Strategy
------------------------------------------

Use this when you want direct control over sampling method before collision
detection (for example ``sigma-point`` vs ``monte-carlo``).

.. code-block:: python

   from adam_core.dynamics.impacts import (
       CollisionEvent,
       ImpactProbabilities,
       calculate_impact_probabilities,
   )
   from adam_core.orbits import Orbits, VariantOrbits

   variants: VariantOrbits = VariantOrbits.create(
       orbit,
       method="sigma-point",  # or "monte-carlo"
       num_samples=5000,
       seed=11,
   )

   propagated: Orbits
   impacts: CollisionEvent
   propagated, impacts = propagator.detect_collisions(
       variants,
       num_days=365,
       conditions=conditions,
       max_processes=16,
   )

   probabilities: ImpactProbabilities = calculate_impact_probabilities(
       variants,
       impacts,
       conditions=conditions,
   )

Visualization Options
---------------------

.. code-block:: python

   import plotly.graph_objects as go
   from adam_core.dynamics.plots import (
       generate_impact_visualization_data,
       plot_impact_simulation,
       plot_risk_corridor,
   )

   corridor_fig: go.Figure = plot_risk_corridor(
       impacts,
       title="Risk Corridor",
       map_style="carto-positron",
   )

   # Earth/Moon collision visualizations:
   times, propagated_orbit, propagated_variants = generate_impact_visualization_data(
       orbit,
       variants,
       impacts,
       propagator,
       time_step=5.0,
       time_range=60.0,
       max_processes=8,
   )
   sim_fig: go.Figure = plot_impact_simulation(
       times,
       propagated_orbit,
       propagated_variants,
       impacts,
       title="Impact Simulation",
       logo=False,
   )

Performance and Reproducibility
-------------------------------

* ``num_samples``: larger values improve rare-event confidence but increase cost.
* ``processes`` / ``max_processes``: primary runtime scaling lever.
* ``seed``: fixes stochastic variant sampling for repeatable runs.
* start with lower sample counts for triage, then rerun with higher counts for
  decision-grade outputs.

Related Reference
-----------------

* :doc:`../reference/dynamics`
* :doc:`../reference/orbits`
* :doc:`../cookbook/impact_probabilities`
* :doc:`../cookbook/moid_analysis`
