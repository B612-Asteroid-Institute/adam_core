.. meta::
   :description: Track NEOCP candidates with Scout samples, observer geometry, and uncertainty-aware ephemerides using adam_core.

NEO Tracking for Follow-Up
==========================

Problem
-------

You need actionable pointings for newly discovered NEO candidates with uncertain
orbits, often under tight operational timelines.

Implementation Options and Tradeoffs
------------------------------------

* Collapse variants to a single covariance orbit:
  Faster and easier to operationalize; can smooth multimodal uncertainty.
* Propagate full Scout variant ensembles:
  Better uncertainty fidelity; higher CPU and memory cost.

Runnable Example
----------------

.. code-block:: python

   import pyarrow.compute as pc

   from adam_core.observers import Observers
   from adam_core.orbits.query.scout import get_scout_objects, query_scout
   from adam_core.time import Timestamp
   from adam_assist import ASSISTPropagator

   scout_objects = get_scout_objects()
   object_of_interest = scout_objects[10]
   samples = query_scout(object_of_interest.objectName)

   times = Timestamp.from_iso8601(
       [
           "2025-02-23T00:00:00Z",
           "2025-02-23T00:05:00Z",
           "2025-02-23T00:10:00Z",
       ],
       scale="utc",
   )
   observers = Observers.from_code("T08", times)

   propagator = ASSISTPropagator()

   # Option A: collapse sampled orbits into one uncertainty-aware orbit.
   orbits = samples.collapse_by_object_id()
   collapsed_ephemeris = propagator.generate_ephemeris(
       orbits,
       observers,
       covariance=True,
       num_samples=1000,
       max_processes=10,
   )

   # Option B: propagate all samples and derive uncertainty envelopes directly.
   sample_ephemeris = propagator.generate_ephemeris(samples, observers, max_processes=10)

   for t in sample_ephemeris.coordinates.time.unique():
       at_t = sample_ephemeris.apply_mask(sample_ephemeris.coordinates.time.equals(t))
       print(
           at_t.coordinates.time.to_iso8601()[0],
           pc.min(at_t.coordinates.lon).as_py(),
           pc.max(at_t.coordinates.lon).as_py(),
       )

When to Use This Pattern
------------------------

Use this when your primary output is follow-up telescope pointing guidance and
the input orbit uncertainty is still evolving.

Related Documentation
---------------------

* :doc:`../examples/track_neo`
* :doc:`../reference/orbits`
* :doc:`../reference/observers`
* :doc:`../reference/propagator`

Input Types
-----------
.. code-block:: python

   # get_scout_objects() -> table-like scout object collection
   # query_scout(object_name: str) -> VariantOrbits
   # VariantOrbits.collapse_by_object_id() -> Orbits
   # ASSISTPropagator.generate_ephemeris(orbits: Orbits | VariantOrbits, observers: Observers, ...) -> Ephemeris | VariantEphemeris
