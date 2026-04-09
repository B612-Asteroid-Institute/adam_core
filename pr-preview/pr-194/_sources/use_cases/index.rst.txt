Use Cases
=========

All narrative and atomic guides live here in one flat list. Start with the
goal closest to your task, then move deeper into uncertainty, scaling, and
mission-analysis details.

Guide Index
-----------

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
   * - :doc:`../examples/2024_yr4_impact_risk`
     - Simulate variants and summarize Earth/Moon impact probabilities.
     - ``adam_core[plots]``, ``adam-assist``
     - CPU intensive at high sample counts; includes optional visualization outputs.

.. toctree::
   :maxdepth: 1

   real_world_use_cases
   ../examples/track_neo
   ../examples/2024_yr4_impact_risk
   source_multiplexing
   ephemeris_service
   neo_tracking
   orbit_determination
   observability_and_light_curves
   impact_risk
   ../cookbook/time_and_windows
   ../cookbook/orbit_sources_and_state_tables
   ../cookbook/observations_and_observers
   ../cookbook/propagation_and_ephemeris
   ../cookbook/oem_and_spk_io
   ../cookbook/openspace_visualization
   ../cookbook/transfer_and_porkchop
   ../cookbook/moid_analysis
   ../cookbook/impact_probabilities
   ../cookbook/photometry_and_magnitude
   ../cookbook/coordinates_and_residuals
   ../cookbook/coordinates_classes
   ../cookbook/coordinate_covariances
   ../cookbook/coordinate_transforms
   ../cookbook/residuals
   ../cookbook/variant_sampling_and_collapse
   ../cookbook/scaling_and_utils
