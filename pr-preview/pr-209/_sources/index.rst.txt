.. meta::
   :description: adam_core is the core scientific library for ADAM, providing orbit, coordinate, time, residual, photometry, and uncertainty tooling for production asteroid analysis.
   :keywords: adam_core, asteroid, orbit propagation, ephemeris, impact risk, orbit determination, planetary science

adam_core
=======================

``adam_core`` exists to provide a single, reliable scientific foundation for asteroid
analysis, astrodynamics, and solar system astronomy. It centralizes core representations
(time, coordinates, covariances, orbits, observers, residuals, and photometry) so
different applications use the same physical assumptions and data contracts. We built this library
because existing tools did not meet the precision, accuracy, or engineering performance required for
large scale analysis or scientifically critical applications like impact risk assessment.

Many ``adam_core`` tables are implemented with `quivr <https://quivr.readthedocs.io/en/latest/>`_,
which provides typed, Arrow-backed scientific tables. This gives explicit schemas, efficient
columnar operations, and predictable behavior when moving between in-memory analysis and
file/service boundaries.

.. list-table::
   :class: visual-grid
   :widths: 34 33 33

   * - .. figure:: _static/using_orbits_preview.png
          :alt: Orbit preview from the Using Orbits guide.
          :target: cookbook/orbit_sources_and_state_tables.html
          :width: 100%
          :align: center

          **Orbit inspection** Quick state checks and geometry inspection from :doc:`cookbook/orbit_sources_and_state_tables`.
     - .. figure:: _static/transfer_porkchop_preview.png
          :alt: Mars to Apophis transfer porkchop from the transfer guide.
          :target: cookbook/transfer_and_porkchop.html
          :width: 100%
          :align: center

          **Mars-Apophis transfer** Lambert trade-space snapshot from :doc:`cookbook/transfer_and_porkchop`.
     - .. figure:: _static/impact_risk_corridor_preview.png
          :alt: Impact risk corridor visualization from the impact guide.
          :target: use_cases/impact_risk.html
          :width: 100%
          :align: center

          **Impact corridor** Earth entry corridor visualization from :doc:`use_cases/impact_risk`.

Engineering Standards
---------------------

* Strict quality gates in development: linting/formatting, type checking, and tests.
* Aim for machine precision when possible and benchmarked CPU/GPU performance with bounded memory usage.

Development Philosophy
----------------------

* Build small, composable scientific objects first; compose behavior from those objects.
* Prefer explicitness over hidden heuristics for frames, scales, time systems, and uncertainty handling.
* Start simple, then expose advanced controls for scale, performance, and mission-specific constraints.
* Keep domain logic close to data models so behavior remains inspectable and testable.

Documentation Map
-----------------

* ``Getting Started``: installation and environment setup.
* ``Reference``: module-by-module API coverage and inventory.

Use Cases (Complete List)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Guide
     - Focus
   * - :doc:`cookbook/time_and_windows`
     - Working with time scales, windows, and time arithmetic.
   * - :doc:`cookbook/orbit_sources_and_state_tables`
     - Using Orbits: ingesting orbit sources, constructing state tables, and inspecting orbit content.
   * - :doc:`cookbook/coordinates_classes`
     - Coordinate class tour with class-specific methods and usage boundaries.
   * - :doc:`cookbook/coordinate_covariances`
     - Coordinate covariance construction, interpretation, and uncertainty propagation patterns.
   * - :doc:`cookbook/coordinate_transforms`
     - High-level and low-level coordinate transformations, including translation cache behavior.
   * - :doc:`cookbook/observations_and_observers`
     - Observation and observer modeling, including custom SPICE kernel observer workflows.
   * - :doc:`cookbook/propagation_and_ephemeris`
     - Propagation and ephemeris generation with practical solver/propagator guidance.
   * - :doc:`cookbook/residuals`
     - Residual primitives and ``Residuals.calculate*`` workflows.
   * - :doc:`cookbook/coordinates_and_residuals`
     - Integrated coordinate/residual analysis path from representation to fit diagnostics.
   * - :doc:`cookbook/variant_sampling_and_collapse`
     - Variant sampling/collapse workflows for uncertainty propagation pipelines.
   * - :doc:`cookbook/photometry_and_magnitude`
     - Magnitude prediction, phase-angle modeling, and photometric analysis functions.
   * - :doc:`cookbook/moid_analysis`
     - MOID analysis workflows and screening patterns.
   * - :doc:`cookbook/transfer_and_porkchop`
     - Lambert solutions, transfer analysis, porkchop dataset generation, and plotting.
   * - :doc:`cookbook/oem_and_spk_io`
     - Ephemeris exchange formats (OEM/SPK): generating and consuming mission-analysis artifacts.
   * - :doc:`cookbook/openspace_visualization`
     - OpenSpace export and visualization setup for trajectory and orbit communication.
   * - :doc:`cookbook/ades_module`
     - ADES PSV parsing/building/export boundaries and validation patterns.
   * - :doc:`use_cases/orbit_determination`
     - Orbit determination flows and residual-driven refinement loops.
   * - :doc:`use_cases/neo_tracking`
     - NEO candidate tracking from uncertain orbit state to follow-up planning products.
   * - :doc:`use_cases/source_multiplexing`
     - Source selection patterns across SBDB/NEOCC/JPL-facing ingestion routes.
   * - :doc:`use_cases/ephemeris_service`
     - Service-style ephemeris generation patterns and operational concerns.
   * - :doc:`use_cases/observability_and_light_curves`
     - Observability constraints, phase-angle analysis, and light-curve tooling.
   * - :doc:`use_cases/impact_risk`
     - Impact-risk analysis with collision conditions, stopping behavior, and visual products.
   * - :doc:`cookbook/scaling_and_utils`
     - Scaling/runtime controls and utility patterns for high-throughput analysis.

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   Getting Started <getting_started/index>
   Use Cases <use_cases/index>
   Reference <reference/index>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
