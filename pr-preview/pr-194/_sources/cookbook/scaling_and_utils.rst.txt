Scaling and Utilities
=====================

Simple Case
-----------

.. code-block:: python

   from adam_core.ray_cluster import initialize_use_ray

   use_ray = initialize_use_ray(num_cpus=8)
   print("ray_enabled", use_ray)

Advanced Options
----------------

.. code-block:: python

   from adam_core.coordinates.origin import OriginCodes
   from adam_core.utils import get_perturber_state, setup_SPICE

   setup_SPICE()
   earth_state = get_perturber_state(
       OriginCodes.EARTH,
       times,
       frame="ecliptic",
       origin=OriginCodes.SUN,
   )

   # internal chunking helpers are useful for large fan-out pipelines
   from adam_core.utils.iter import _iterate_chunks

   for idx_chunk in _iterate_chunks(range(0, 10000), chunk_size=500):
       pass

When to Use
-----------

* ``initialize_use_ray`` for multiprocessing/distributed fan-out.
* ``setup_SPICE`` and ``get_perturber_state`` for deterministic state lookup.
* Iter/chunking helpers for memory-safe large-batch processing.

Related Reference
-----------------

* :doc:`../reference/api/adam_core.ray_cluster`
* :doc:`../reference/api/adam_core.utils`

