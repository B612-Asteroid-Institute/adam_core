.. meta::
   :description: Complete inventory of adam_core functionality and mapping to cookbook and reference coverage.

Functionality Inventory
=======================

This inventory is generated from the current ``src/adam_core`` tree (excluding tests) and maps each package to narrative and API coverage.

Coverage Map
------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Package
     - Narrative Coverage
     - Reference Coverage
   * - ``adam_core.constants``
     - :doc:`../cookbook/scaling_and_utils`
     - :doc:`constants`
   * - ``adam_core.coordinates``
     - :doc:`../cookbook/coordinates_and_residuals`,
       :doc:`../cookbook/coordinates_classes`,
       :doc:`../cookbook/coordinate_covariances`,
       :doc:`../cookbook/coordinate_transforms`,
       :doc:`../cookbook/residuals`
     - :doc:`coordinates`
   * - ``adam_core.dynamics``
     - :doc:`../cookbook/propagation_and_ephemeris`,
       :doc:`../cookbook/transfer_and_porkchop`,
       :doc:`../cookbook/moid_analysis`,
       :doc:`../cookbook/impact_probabilities`,
       :doc:`../cookbook/dynamics_and_impact_analysis`
     - :doc:`dynamics`
   * - ``adam_core.missions``
     - :doc:`../cookbook/transfer_and_porkchop`,
       :doc:`../cookbook/mission_and_export`
     - :doc:`missions`
   * - ``adam_core.observations``
     - :doc:`../cookbook/observations_and_observers`
     - :doc:`observations`
   * - ``adam_core.observers``
     - :doc:`../cookbook/observations_and_observers`
     - :doc:`observers`
   * - ``adam_core.orbit_determination``
     - :doc:`../cookbook/orbit_determination_pipeline`
     - :doc:`orbit_determination`
   * - ``adam_core.orbits``
     - :doc:`../cookbook/orbit_sources_and_state_tables`,
       :doc:`../cookbook/variant_sampling_and_collapse`,
       :doc:`../cookbook/oem_and_spk_io`,
       :doc:`../cookbook/openspace_visualization`
     - :doc:`orbits`
   * - ``adam_core.photometry``
     - :doc:`../cookbook/photometry_and_magnitude`,
       :doc:`../use_cases/observability_and_light_curves`
     - :doc:`photometry`
   * - ``adam_core.propagator``
     - :doc:`../cookbook/propagation_and_ephemeris`
     - :doc:`propagator`
   * - ``adam_core.ray_cluster``
     - :doc:`../cookbook/scaling_and_utils`
     - :doc:`ray_cluster`
   * - ``adam_core.time``
     - :doc:`../cookbook/time_and_windows`
     - :doc:`time`
   * - ``adam_core.utils``
     - :doc:`../cookbook/scaling_and_utils`
     - :doc:`utils`

Module Inventory
----------------

``adam_core.constants``
^^^^^^^^^^^^^^^^^^^^^^^

* ``adam_core.constants``

``adam_core.coordinates``
^^^^^^^^^^^^^^^^^^^^^^^^^

* ``adam_core.coordinates``
* ``adam_core.coordinates.cartesian``
* ``adam_core.coordinates.cometary``
* ``adam_core.coordinates.covariances``
* ``adam_core.coordinates.geodetics``
* ``adam_core.coordinates.jacobian``
* ``adam_core.coordinates.keplerian``
* ``adam_core.coordinates.origin``
* ``adam_core.coordinates.residuals``
* ``adam_core.coordinates.spherical``
* ``adam_core.coordinates.transform``
* ``adam_core.coordinates.types``
* ``adam_core.coordinates.units``
* ``adam_core.coordinates.variants``

``adam_core.dynamics``
^^^^^^^^^^^^^^^^^^^^^^

* ``adam_core.dynamics``
* ``adam_core.dynamics.aberrations``
* ``adam_core.dynamics.barker``
* ``adam_core.dynamics.chi``
* ``adam_core.dynamics.ephemeris``
* ``adam_core.dynamics.exceptions``
* ``adam_core.dynamics.impacts``
* ``adam_core.dynamics.kepler``
* ``adam_core.dynamics.lagrange``
* ``adam_core.dynamics.lambert``
* ``adam_core.dynamics.moid``
* ``adam_core.dynamics.plots``
* ``adam_core.dynamics.propagation``
* ``adam_core.dynamics.stumpff``
* ``adam_core.dynamics.tisserand``

``adam_core.missions``
^^^^^^^^^^^^^^^^^^^^^^

* ``adam_core.missions``
* ``adam_core.missions.porkchop``

``adam_core.observations``
^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``adam_core.observations``
* ``adam_core.observations.ades``
* ``adam_core.observations.associations``
* ``adam_core.observations.detections``
* ``adam_core.observations.exposures``
* ``adam_core.observations.photometry``
* ``adam_core.observations.source_catalog``

``adam_core.observers``
^^^^^^^^^^^^^^^^^^^^^^^

* ``adam_core.observers``
* ``adam_core.observers.observers``
* ``adam_core.observers.state``
* ``adam_core.observers.utils``

``adam_core.orbit_determination``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``adam_core.orbit_determination``
* ``adam_core.orbit_determination.differential_correction``
* ``adam_core.orbit_determination.evaluate``
* ``adam_core.orbit_determination.fitted_orbits``
* ``adam_core.orbit_determination.gauss``
* ``adam_core.orbit_determination.gibbs``
* ``adam_core.orbit_determination.herrick_gibbs``
* ``adam_core.orbit_determination.iod``
* ``adam_core.orbit_determination.least_squares``
* ``adam_core.orbit_determination.od``
* ``adam_core.orbit_determination.orbit_fitter``
* ``adam_core.orbit_determination.outliers``

``adam_core.orbits``
^^^^^^^^^^^^^^^^^^^^

* ``adam_core.orbits``
* ``adam_core.orbits.classification``
* ``adam_core.orbits.ephemeris``
* ``adam_core.orbits.oem_io``
* ``adam_core.orbits.openspace``
* ``adam_core.orbits.openspace.assets``
* ``adam_core.orbits.openspace.lua``
* ``adam_core.orbits.openspace.renderable``
* ``adam_core.orbits.openspace.translation``
* ``adam_core.orbits.orbits``
* ``adam_core.orbits.physical_parameters``
* ``adam_core.orbits.plots``
* ``adam_core.orbits.query``
* ``adam_core.orbits.query.horizons``
* ``adam_core.orbits.query.neocc``
* ``adam_core.orbits.query.sbdb``
* ``adam_core.orbits.query.scout``
* ``adam_core.orbits.spice_kernel``
* ``adam_core.orbits.variants``

``adam_core.photometry``
^^^^^^^^^^^^^^^^^^^^^^^^

* ``adam_core.photometry``
* ``adam_core.photometry.absolute_magnitude``
* ``adam_core.photometry.bandpasses``
* ``adam_core.photometry.bandpasses.api``
* ``adam_core.photometry.bandpasses.constants``
* ``adam_core.photometry.bandpasses.tables``
* ``adam_core.photometry.bandpasses.vendor``
* ``adam_core.photometry.magnitude``
* ``adam_core.photometry.magnitude_common``

``adam_core.propagator``
^^^^^^^^^^^^^^^^^^^^^^^^

* ``adam_core.propagator``
* ``adam_core.propagator.propagator``
* ``adam_core.propagator.types``
* ``adam_core.propagator.utils``

``adam_core.ray_cluster``
^^^^^^^^^^^^^^^^^^^^^^^^^

* ``adam_core.ray_cluster``

``adam_core.time``
^^^^^^^^^^^^^^^^^^

* ``adam_core.time``
* ``adam_core.time.time``

``adam_core.utils``
^^^^^^^^^^^^^^^^^^^

* ``adam_core.utils``
* ``adam_core.utils.bounded_lru``
* ``adam_core.utils.chunking``
* ``adam_core.utils.helpers``
* ``adam_core.utils.helpers.data``
* ``adam_core.utils.helpers.data.get_test_data``
* ``adam_core.utils.helpers.observations``
* ``adam_core.utils.helpers.orbits``
* ``adam_core.utils.iter``
* ``adam_core.utils.mpc``
* ``adam_core.utils.plots``
* ``adam_core.utils.plots.data``
* ``adam_core.utils.plots.logos``
* ``adam_core.utils.spice``
