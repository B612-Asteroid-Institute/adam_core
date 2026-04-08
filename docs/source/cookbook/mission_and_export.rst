Mission and Export
==================

Mission planning and export guides are split into dedicated sections.

Dedicated Guides
----------------

* :doc:`transfer_and_porkchop` for Lambert solutions, porkchop data generation,
  and porkchop visualization.
* :doc:`oem_and_spk_io` for OEM write/read and SPK kernel generation.
* :doc:`openspace_visualization` for OpenSpace asset generation.

Propagator Guidance
-------------------

For production-quality exports and mission design studies, use a robust
propagator implementation (for example, ``adam_assist.ASSISTPropagator``)
when propagation is required.

Related Reference
-----------------

* :doc:`../reference/missions`
* :doc:`../reference/orbits`

Input Types
-----------
.. code-block:: python

   # prepare_and_propagate_orbits(body: Orbits | OriginCodes, start_time: Timestamp, end_time: Timestamp, propagation_origin: OriginCodes = OriginCodes.SUN, step_size: float = 1.0, propagator_class: type[Propagator] | None = None, max_processes: int | None = 1) -> Orbits
   # orbit_to_oem(orbits: Orbits, output_file: str, originator: str = "ADAM CORE USER") -> str
   # orbits_to_spk(orbits: Orbits, output_file: str, start_time: Timestamp, end_time: Timestamp, propagator: Propagator | None = None, ...) -> dict[str, int]
