Use Cases
=========

Long-form guides for production asteroid/orbital analysis and service design.
The section is ordered from simpler operational tasks to more complex risk and
decision analysis.

Start Here
----------

* :doc:`real_world_use_cases` for the full simple-to-advanced progression map.
* :doc:`../cookbook/index` when you need atomic APIs and type-level behavior.

Narrative Guides
----------------

.. toctree::
   :maxdepth: 2

   real_world_use_cases
   source_multiplexing
   ephemeris_service
   neo_tracking
   observability_and_light_curves
   orbit_determination
   impact_risk

Notebook Demonstrations
-----------------------

Rendered notebooks are grouped with the same use-case material and include
runtime/dependency notes.

.. list-table::
   :header-rows: 1
   :widths: 22 30 28 20

   * - Notebook
     - Purpose
     - Dependencies
     - Runtime / Constraints
   * - :doc:`../examples/track_neo`
     - Track NEOCP candidates and generate uncertainty-aware follow-up pointings.
     - ``adam_core``, ``adam-assist``
     - Network/API dependent; multiprocessing recommended for speed.
   * - :doc:`../examples/preview_orbit`
     - Quickly inspect and preview candidate orbits.
     - ``adam_core``, ``adam-assist``
     - Lightweight; depends on SBDB availability.
   * - :doc:`../examples/2024_yr4_impact_risk`
     - Simulate variants and summarize Earth/Moon impact probabilities.
     - ``adam_core[plots]``, ``adam-assist``
     - CPU intensive at high sample counts; includes optional visualization outputs.

.. toctree::
   :maxdepth: 1

   ../examples/track_neo
   ../examples/preview_orbit
   ../examples/2024_yr4_impact_risk
