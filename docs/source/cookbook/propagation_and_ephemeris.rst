Propagation and Ephemeris
=========================

Propagator Requirement
----------------------

High-fidelity operational propagation requires a concrete ``Propagator``
implementation. In practice, use a production backend such as
``adam_assist.ASSISTPropagator``.

Simple Case (Production Propagator)
-----------------------------------

.. code-block:: python

   import numpy as np
   from adam_assist import ASSISTPropagator
   from adam_core.observers import Observers
   from adam_core.orbits.query import query_sbdb
   from adam_core.time import Timestamp

   orbits = query_sbdb(["Apophis", "Bennu"])
   t0 = Timestamp.from_mjd([60200.0], scale="tdb")
   times = Timestamp.from_mjd(t0.mjd().to_numpy() + np.arange(0, 10), scale="tdb")

   propagator = ASSISTPropagator()
   propagated = propagator.propagate_orbits(orbits, times, max_processes=1)

   observers = Observers.from_code("I11", times)
   ephem = propagator.generate_ephemeris(propagated, observers, max_processes=1)

Advanced Options (Two-Body Fast Path)
-------------------------------------

.. code-block:: python

   from adam_core.dynamics.ephemeris import generate_ephemeris_2body
   from adam_core.dynamics.propagation import propagate_2body

   propagated_fast = propagate_2body(
       orbits,
       times,
       max_processes=8,
       chunk_size=250,
       max_iter=1000,
       tol=1e-14,
   )

   ephem_fast = generate_ephemeris_2body(
       propagated_fast,
       observers,
       max_processes=8,
       chunk_size=250,
       stellar_aberration=False,
       predict_magnitudes=True,
       predict_phase_angle=True,
   )

When to Use
-----------

* High-fidelity operational runs: ``ASSISTPropagator`` pattern.
* Large-scale prefiltering or candidate fan-out: two-body path first.
* Use propagate-then-ephemeris when observer requests are repeated.
* For spacecraft observers from custom SPICE kernels (for example JWST), see
  :doc:`observations_and_observers`.

Related Reference
-----------------

* :doc:`../reference/propagator`
* :doc:`../reference/dynamics`
* :doc:`../reference/observers`

