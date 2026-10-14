.. meta::
   :description: Multi-source orbit ingest pattern for service APIs using SBDB, Scout, NEOCC, and Horizons in adam_core.

Multi-Source Orbit Ingest for APIs
==================================

Problem
-------

A service endpoint needs one stable contract but must accept object inputs from
multiple upstream sources (SBDB, Scout, NEOCC, Horizons).

Implementation Options and Tradeoffs
------------------------------------

* Source-specific endpoints:
  Simple for implementation, but duplicates validation and downstream logic.
* One endpoint with source routing:
  Cleaner operational model and cache strategy, with slightly more source-specific branching.

Runnable Example
----------------

.. code-block:: python

   from adam_core.orbits import Orbits
   from adam_core.orbits.query import query_horizons, query_neocc, query_sbdb, query_scout
   from adam_core.time import Timestamp

   def load_orbits(source: str, object_ids: list[str]) -> Orbits:
       if source == "sbdb":
           return query_sbdb(object_ids)
       if source == "scout":
           # scout returns variants; collapse for deterministic downstream service logic
           return query_scout(object_ids).collapse_by_object_id()
       if source == "neocc":
           return query_neocc(object_ids)
       if source == "horizons":
           t0 = Timestamp.from_mjd([60200.0], scale="tdb")
           return query_horizons(object_ids, t0)
       raise ValueError(f"Unsupported source: {source}")

When to Use This Pattern
------------------------

Use this for public or internal APIs where clients choose data source at request
time but downstream pipelines require one normalized ``Orbits`` table type.

Related Documentation
---------------------

* :doc:`../cookbook/orbit_sources_and_state_tables`
* :doc:`../reference/api/adam_core.orbits`
