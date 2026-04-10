.. meta::
   :description: Impact risk analysis in adam_core using sampled variants, collision conditions, and impact probability summaries.

Impact Risk Analysis
====================

Goal
----

Estimate impact likelihood from uncertain orbits, then inspect event timing and
impact geography for operational decision-making.

Backend Requirement
-------------------

Impact analysis requires a propagator that implements ``ImpactMixin`` collision
detection. For production use, prefer ``adam_assist.ASSISTPropagator``.

Core Objects and Their Roles
----------------------------

* ``CollisionConditions``:
  defines what event surfaces to watch during propagation.
* ``CollisionEvent``:
  event rows returned by detection (which condition was hit, when, where,
  whether that condition is terminal).
* ``ImpactProbabilities``:
  per-orbit summary table with counts and cumulative probabilities.

Collision Conditions: What They Are
-----------------------------------

Each row in ``CollisionConditions`` is one monitored boundary:

* ``condition_id``:
  label used in outputs and summaries.
* ``collision_object``:
  body code (for example ``EARTH``, ``MOON``).
* ``collision_distance``:
  radial threshold in **km** from the body center.
* ``stopping_condition``:
  whether propagation should stop for a variant after this condition is met.

Minimal explicit setup:

.. code-block:: python

   from adam_core.coordinates import Origin
   from adam_core.dynamics.impacts import CollisionConditions

   conditions: CollisionConditions = CollisionConditions.from_kwargs(
       condition_id=["Earth", "Moon"],
       collision_object=Origin.from_kwargs(code=["EARTH", "MOON"]),
       collision_distance=[6420.0, 1740.0],
       stopping_condition=[True, True],
   )

Stopping Conditions: How To Use Them
------------------------------------

``stopping_condition`` is per-condition and controls termination behavior after
that boundary is crossed:

* ``True``:
  treat this condition as terminal for a variant.
* ``False``:
  keep propagating after crossing this boundary, so later events can still be
  recorded.
* mixed configurations:
  common for "terminal Earth impact, non-terminal Moon/other event tracking."

Example mixed configuration:

.. code-block:: python

   conditions: CollisionConditions = CollisionConditions.from_kwargs(
       condition_id=["Earth", "Moon"],
       collision_object=Origin.from_kwargs(code=["EARTH", "MOON"]),
       collision_distance=[6420.0, 1740.0],
       stopping_condition=[True, False],
   )

YR4-Style End-to-End Workflow (Expanded)
----------------------------------------

This is the consolidated operational path that combines the previous notebook
flow and narrative page into one guide.

.. code-block:: python

   from adam_assist import ASSISTPropagator
   from adam_core.coordinates import Origin
   from adam_core.dynamics.impacts import (
       CollisionConditions,
       CollisionEvent,
       ImpactProbabilities,
       calculate_impacts,
       calculate_impact_probabilities,
   )
   from adam_core.orbits import Orbits, VariantOrbits
   from adam_core.orbits.query import query_sbdb
   from adam_core.time import Timestamp

   orbit: Orbits = query_sbdb(["2024 YR4"])

   # Size propagation window from an operationally relevant date.
   approx_impact_date: Timestamp = Timestamp.from_iso8601(["2032-12-22"], scale="tdb")
   analysis_end: Timestamp = approx_impact_date.add_days(30)
   days_to_run, _ = analysis_end.difference(orbit.coordinates.time)
   num_days: int = int(days_to_run[0].as_py())

   conditions: CollisionConditions = CollisionConditions.from_kwargs(
       condition_id=["Earth", "Moon"],
       collision_object=Origin.from_kwargs(code=["EARTH", "MOON"]),
       collision_distance=[6420.0, 1740.0],
       stopping_condition=[True, True],
   )

   propagator = ASSISTPropagator()

   variants: VariantOrbits
   impacts: CollisionEvent
   variants, impacts = calculate_impacts(
       orbits=orbit,
       num_days=num_days,
       propagator=propagator,
       num_samples=10000,
       processes=60,
       seed=42,
       conditions=conditions,
   )

   probabilities: ImpactProbabilities = calculate_impact_probabilities(
       variants,
       impacts,
       conditions=conditions,
   )
   print(probabilities.to_dataframe())

Inspect What You Got Back
-------------------------

Before summarizing, inspect raw event rows to understand condition behavior.

.. code-block:: python

   impact_df = impacts.to_dataframe()
   print(impact_df[["orbit_id", "variant_id", "condition_id", "stopping_condition"]].head())
   print("event counts by condition:")
   print(impact_df["condition_id"].value_counts())

Alternative Usage Patterns
--------------------------

1. Quick Earth-only triage:
   use default conditions and moderate ``num_samples`` for throughput.
2. Decision-grade rerun:
   increase ``num_samples`` and tighten/expand ``num_days`` around relevant windows.
3. Custom event surfaces:
   define multiple conditions with distinct ``stopping_condition`` values to encode
   mission-specific logic.
4. Bring your own variant strategy:
   generate variants explicitly with ``VariantOrbits.create(method=...)`` and call
   ``propagator.detect_collisions(...)`` directly.

.. code-block:: python

   from adam_core.orbits import Orbits, VariantOrbits

   variants_direct: VariantOrbits = VariantOrbits.create(
       orbit,
       method="sigma-point",  # or "monte-carlo"
       num_samples=5000,
       seed=11,
   )

   propagated: Orbits
   impacts_direct: CollisionEvent
   propagated, impacts_direct = propagator.detect_collisions(
       variants_direct,
       num_days=num_days,
       conditions=conditions,
       max_processes=16,
   )

Visualization Workflow
----------------------

.. code-block:: python

   import plotly.graph_objects as go
   from adam_core.dynamics.plots import (
       generate_impact_visualization_data,
       plot_impact_simulation,
       plot_risk_corridor,
   )

   corridor_fig: go.Figure = plot_risk_corridor(
       impacts,
       title="Risk Corridor for 2024 YR4",
       map_style="carto-positron",
   )
   corridor_fig.show()

   # Earth/Moon impact simulation visualization.
   propagation_times, propagated_best_fit_orbit, propagated_variants = (
       generate_impact_visualization_data(
           orbit,
           variants,
           impacts,
           propagator,
           time_step=5.0,
           time_range=60.0,
           max_processes=8,
       )
   )

   sim_fig: go.Figure = plot_impact_simulation(
       propagation_times,
       propagated_best_fit_orbit,
       propagated_variants,
       impacts,
       title="2024 YR4 Impact Simulation",
       logo=False,
   )
   sim_fig.show()

Performance and Reproducibility
-------------------------------

* ``num_samples``:
  larger sample counts increase confidence in low-probability tails.
* ``processes`` / ``max_processes``:
  primary runtime scaling controls.
* ``seed``:
  fix for reproducible Monte Carlo variant generation.
* practical approach:
  run a fast triage pass first, then rerun higher-fidelity for final products.

Related Reference
-----------------

* :doc:`../reference/dynamics`
* :doc:`../reference/orbits`
* :doc:`../cookbook/impact_probabilities`
* :doc:`../cookbook/moid_analysis`
