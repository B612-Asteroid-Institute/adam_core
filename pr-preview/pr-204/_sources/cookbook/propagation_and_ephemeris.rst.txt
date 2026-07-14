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
   from adam_core.orbits import Orbits
   from adam_core.orbits.ephemeris import Ephemeris
   from adam_core.observers import Observers
   from adam_core.orbits.query import query_sbdb
   from adam_core.time import Timestamp

   orbits: Orbits = query_sbdb(["Apophis", "Bennu"])
   t0: Timestamp = Timestamp.from_mjd([60200.0], scale="tdb")
   times: Timestamp = Timestamp.from_mjd(t0.mjd().to_numpy() + np.arange(0, 10), scale="tdb")

   propagator = ASSISTPropagator()
   propagated: Orbits = propagator.propagate_orbits(orbits, times, max_processes=1)

   observers: Observers = Observers.from_code("I11", times)
   ephem: Ephemeris = propagator.generate_ephemeris(propagated, observers, max_processes=1)

Advanced N-Body Pattern: Parallel Throughput Controls
-----------------------------------------------------

.. code-block:: python

   # Scale out orbit propagation work.
   propagated_parallel: Orbits = propagator.propagate_orbits(
       orbits,
       times,
       max_processes=8,
       chunk_size=256,
   )

   # Generate ephemerides in parallel with brightness/phase prediction enabled.
   ephem_parallel: Ephemeris = propagator.generate_ephemeris(
       propagated_parallel,
       observers,
       max_processes=8,
       chunk_size=256,
       predict_magnitudes=True,
       predict_phase_angle=True,
   )

Advanced N-Body Pattern: Covariance-Aware Products
--------------------------------------------------

.. code-block:: python

   # Covariance propagation on state vectors (returns Orbits with propagated covariance).
   propagated_cov: Orbits = propagator.propagate_orbits(
       orbits,
       times,
       covariance=True,
       covariance_method="auto",
       num_samples=2000,
       seed=42,
       max_processes=8,
       chunk_size=256,
   )

   # Covariance-aware ephemeris generation (sample, propagate, collapse).
   ephem_cov: Ephemeris = propagator.generate_ephemeris(
       orbits,
       observers,
       covariance=True,
       covariance_method="auto",
       num_samples=2000,
       seed=42,
       max_processes=8,
       chunk_size=256,
       predict_magnitudes=True,
       predict_phase_angle=True,
   )

Advanced N-Body Pattern: Variant Ensembles
------------------------------------------

.. code-block:: python

   from adam_core.orbits.variants import VariantEphemeris, VariantOrbits
   from adam_core.orbits.query import query_scout

   scout_variants: VariantOrbits = query_scout(["P10vY9r"])
   variant_ephem: VariantEphemeris = propagator.generate_ephemeris(
       scout_variants,
       observers,
       covariance=False,  # Variants are already explicit samples.
       max_processes=8,
       chunk_size=512,
   )

Basic Two-Body Baseline (Fast Screening)
----------------------------------------

.. code-block:: python

   from adam_core.dynamics.ephemeris import generate_ephemeris_2body
   from adam_core.dynamics.propagation import propagate_2body

   propagated_2body = propagate_2body(
       orbits,
       times,
       max_processes=8,
       chunk_size=250,
       max_iter=1000,
       tol=1e-14,
   )

   ephem_2body = generate_ephemeris_2body(
       propagated_2body,
       observers,
       max_processes=8,
       chunk_size=250,
       stellar_aberration=False,
       predict_magnitudes=True,
       predict_phase_angle=True,
   )

When to Use
-----------

* Default operational path: n-body propagator interface (for example ``ASSISTPropagator``).
* Advanced production tuning: ``max_processes`` + ``chunk_size`` on n-body methods.
* Uncertainty-aware outputs: ``covariance=True`` on n-body methods.
* Variant workflows: pass ``VariantOrbits`` directly to n-body ephemeris generation.
* Fast pre-screening only: two-body baseline before high-fidelity reruns.
* For spacecraft observers from custom SPICE kernels (for example JWST), see
  :doc:`observations_and_observers`.

Related Reference
-----------------

* :doc:`../reference/api/adam_core.propagator`
* :doc:`../reference/api/adam_core.dynamics`
* :doc:`../reference/api/adam_core.observers`
