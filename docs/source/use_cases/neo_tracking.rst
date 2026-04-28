.. meta::
   :description: Track NEOCP candidates with Scout samples, observer geometry, and uncertainty-aware ephemerides using adam_core.

NEO Tracking for Follow-Up
==========================

Goal
----

Generate follow-up pointings for uncertain NEOCP candidates quickly and with
explicit uncertainty handling.

Operational Prerequisites
-------------------------

* Install a production propagator: ``adam-assist`` (recommended).
* Query access to CNEOS Scout for active NEOCP candidates.
* Observatory code(s) or explicit observer state vectors.

Canonical End-to-End Pattern
----------------------------

.. code-block:: python

   import pyarrow.compute as pc
   from adam_assist import ASSISTPropagator
   from adam_core.observers import Observers
   from adam_core.orbits import Orbits, VariantOrbits
   from adam_core.orbits.query.scout import get_scout_objects, query_scout
   from adam_core.time import Timestamp

   # 1) Select candidate(s) from Scout's current NEOCP list.
   scout_objects = get_scout_objects()
   object_of_interest = scout_objects[10]

   # 2) Fetch Scout posterior samples (usually ~1000 variants per object).
   samples: VariantOrbits = query_scout(object_of_interest.objectName)

   # 3) Define exposure times and observer geometry.
   times: Timestamp = Timestamp.from_iso8601(
       [
           "2025-02-23T00:00:00Z",
           "2025-02-23T00:05:00Z",
           "2025-02-23T00:10:00Z",
       ],
       scale="utc",
   )
   observers: Observers = Observers.from_code("T08", times)
   propagator = ASSISTPropagator()

   # 4A) Fast operational path: collapse variants and propagate covariance.
   collapsed_orbits: Orbits = samples.collapse_by_object_id()
   collapsed_ephemeris = propagator.generate_ephemeris(
       collapsed_orbits,
       observers,
       covariance=True,
       num_samples=1000,
       max_processes=10,
   )

   # 4B) High-fidelity path: propagate each variant directly.
   sample_ephemeris = propagator.generate_ephemeris(
       samples,
       observers,
       max_processes=10,
   )

   # 5) Derive per-time on-sky envelopes for pointing decisions.
   for t in sample_ephemeris.coordinates.time.unique():
       at_t = sample_ephemeris.apply_mask(sample_ephemeris.coordinates.time.equals(t))
       print(
           at_t.coordinates.time.to_iso8601()[0],
           "RA range:",
           pc.min(at_t.coordinates.lon).as_py(),
           pc.max(at_t.coordinates.lon).as_py(),
           "Dec range:",
           pc.min(at_t.coordinates.lat).as_py(),
           pc.max(at_t.coordinates.lat).as_py(),
       )

Implementation Choices
----------------------

* ``collapse_by_object_id()`` + ``covariance=True``:
  Lowest compute cost and easiest to deploy for routine scheduling.
* direct ``VariantOrbits`` propagation:
  Preserves multimodal structure and heavy tails in uncertainty.
* ``max_processes``:
  Primary speed control for short-notice campaign planning.

What You Get Back
-----------------

* ``collapsed_ephemeris``:
  Central track with covariance-derived uncertainty columns.
* ``sample_ephemeris``:
  Variant-wise tracks suitable for envelope/percentile summarization.
* Observer-aware sky coordinates at each requested time for immediate
  telescope planning.

When To Use
-----------

Use this path when you are planning near-term follow-up for newly discovered
candidates and need a defensible uncertainty-aware pointing recommendation.

Related Reference
-----------------

* :doc:`../reference/api/adam_core.orbits`
* :doc:`../reference/api/adam_core.observers`
* :doc:`../reference/api/adam_core.propagator`
* :doc:`../cookbook/variant_sampling_and_collapse`
