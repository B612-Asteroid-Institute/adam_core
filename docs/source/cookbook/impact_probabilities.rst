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

Risk Corridor Visualization
---------------------------

Use ``plot_risk_corridor`` to visualize Earth impact geography over time.
Prefer ``map_style="carto-positron"`` for docs/CI/browser stability (avoid
OpenStreetMap tile policy blocks).

.. code-block:: python

   import plotly.graph_objects as go
   from adam_core.dynamics.impacts import CollisionEvent
   from adam_core.dynamics.plots import plot_risk_corridor

   collisions: CollisionEvent
   corridor_fig: go.Figure = plot_risk_corridor(
       collisions,
       title="Earth Risk Corridor",
       map_style="carto-positron",
   )
   corridor_fig.show()

For a deterministic static preview image (for docs pages/PR assets), snapshot a
late animation frame before exporting:

.. code-block:: python

   if corridor_fig.frames:
       corridor_fig.update(data=corridor_fig.frames[-1].data)
   corridor_fig.write_image(
       "impact_risk_corridor_preview.png",
       width=1400,
       height=900,
       scale=1,
   )

.. figure:: ../_static/impact_risk_corridor_preview.png
   :alt: Impact risk corridor preview generated from collision events using carto-positron basemap.

   Corridor preview with impacts colored by relative impact time.

Impact Simulation Plot
----------------------

Use ``generate_impact_visualization_data`` + ``plot_impact_simulation`` to
build an animated Earth/Moon approach and impact sequence.

.. code-block:: python

   import plotly.graph_objects as go
   from adam_core.dynamics.impacts import CollisionEvent
   from adam_core.dynamics.plots import (
       generate_impact_visualization_data,
       plot_impact_simulation,
   )
   from adam_core.orbits import Orbits, VariantOrbits
   from adam_core.propagator import Propagator
   from adam_core.time import Timestamp

   orbit: Orbits
   variants: VariantOrbits
   collisions: CollisionEvent
   propagator: Propagator

   propagation_times: Timestamp
   propagated_best_fit_orbit: Orbits
   propagated_variants: dict[str, Orbits]
   propagation_times, propagated_best_fit_orbit, propagated_variants = (
       generate_impact_visualization_data(
           orbit,
           variants,
           collisions,
           propagator,
           time_step=5.0,
           time_range=60.0,
           max_processes=8,
       )
   )

   simulation_fig: go.Figure = plot_impact_simulation(
       propagation_times,
       propagated_best_fit_orbit,
       propagated_variants,
       collisions,
       title="Impact Simulation",
       sample_impactors=None,
       sample_non_impactors=0.1,
       logo=False,
   )
   simulation_fig.show()

   # Shareable artifact for reviewers.
   simulation_fig.write_html("impact_simulation.html")

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
