.. meta::
   :description: Service-oriented ephemeris generation design using adam_core orbits, observers, and propagator interfaces.

Ephemeris Generation for Services
=================================

Problem
-------

You need a repeatable service endpoint that transforms object identifiers and
observation requests into ephemerides with uncertainty context.

Implementation Options and Tradeoffs
------------------------------------

* Propagate first, then generate ephemerides:
  Better for reuse when many observer requests share the same propagated states.
* Direct ephemeris generation from initial orbits:
  Simpler request path, potentially more repeated propagation work.

Runnable Example
----------------

.. code-block:: python

   import numpy as np

   from adam_core.orbits import Orbits
   from adam_core.observers import Observers
   from adam_core.orbits.query import query_horizons
   from adam_core.time import Timestamp
   from adam_assist import ASSISTPropagator

   request_object_ids: list[str] = ["Apophis", "Bennu", "Eros"]
   t0: Timestamp = Timestamp.from_mjd([60000.0], scale="tdb")
   request_times: Timestamp = Timestamp.from_mjd(
       t0.mjd().to_numpy() + np.arange(0, 20), scale="tdb"
   )

   orbits: Orbits = query_horizons(request_object_ids, t0)
   observers: Observers = Observers.from_code("I11", request_times)

   propagator = ASSISTPropagator()

   # Pattern A: direct output for one request.
   ephemeris = propagator.generate_ephemeris(
       orbits,
       observers,
       predict_magnitudes=True,
       predict_phase_angle=True,
       chunk_size=100,
       max_processes=4,
   )

   # Pattern B: reusable propagated cache for multiple observer sets.
   propagated = propagator.propagate_orbits(orbits, request_times, max_processes=4)

   print(ephemeris.to_dataframe().head())

When to Use This Pattern
------------------------

Use this when building observatory planning APIs, nightly scheduling pipelines,
or batch prediction systems that need deterministic behavior.

Notes
-----

This pattern treats these objects as propagated targets. If your use case is
about perturbing-body states (for example Earth, Moon, planets), use
``adam_core.utils.get_perturber_state`` and origin codes instead.

Related Documentation
---------------------

* :doc:`../reference/api/adam_core.orbits`
* :doc:`../reference/api/adam_core.observers`
* :doc:`../reference/api/adam_core.propagator`
* :doc:`../reference/api/adam_core.photometry`
* :doc:`../reference/api/adam_core.time`
