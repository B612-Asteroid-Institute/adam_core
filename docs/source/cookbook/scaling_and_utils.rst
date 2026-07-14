Scaling and Utilities
=====================

Simple Case
-----------

Rust kernels and ``adam-assist`` parallelize eligible batch work internally;
no Python cluster needs to be initialized. To cap Rayon worker threads, set
``RAYON_NUM_THREADS`` before starting Python:

.. code-block:: console

   $ RAYON_NUM_THREADS=8 python my_pipeline.py

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

* Native batch kernels for in-process parallel fan-out; historical Ray
  scheduling arguments remain compatibility no-ops.
* ``setup_SPICE`` and ``get_perturber_state`` for deterministic state lookup.
* Iter/chunking helpers for memory-safe large-batch processing.

Related Reference
-----------------

* :doc:`../reference/api/adam_core.utils`
