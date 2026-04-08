Orbit Sources and State Tables
==============================

Simple Case
-----------

.. code-block:: python

   from adam_core.orbits.query import query_sbdb

   # Common bootstrap pattern for catalog-driven orbit ingest.
   orbits = query_sbdb(["Apophis", "Eros"])
   print(len(orbits), orbits.orbit_id.to_pylist()[:2])

Advanced Source Multiplexing
----------------------------

.. code-block:: python

   from adam_core.orbits import Orbits
   from adam_core.orbits.query import query_horizons, query_neocc, query_sbdb, query_scout
   from adam_core.time import Timestamp

   def get_orbits(source: str, object_ids: list[str]) -> Orbits:
       if source == "sbdb":
           return query_sbdb(object_ids)
       if source == "horizons":
           t0 = Timestamp.from_mjd([60200.0], scale="tdb")
           return query_horizons(object_ids, t0)
       if source == "neocc":
           return query_neocc(object_ids)
       if source == "scout":
           return query_scout(object_ids).collapse_by_object_id()
       raise ValueError(f"Unsupported source: {source}")

This is the same operational pattern used in service-style APIs where clients
can request multiple orbit sources behind one endpoint.

When to Use
-----------

* ``query_sbdb`` for stable catalog orbits.
* ``query_scout`` for uncertain candidate ensembles, then collapse when needed.
* ``query_horizons`` when specific epochs and Horizons consistency matter.
* ``query_neocc`` for ESA NEOCC-based ingest.

Related Reference
-----------------

* :doc:`../reference/orbits`
* :doc:`../reference/functionality_inventory`

Input Types
-----------
.. code-block:: python

   # query_sbdb(object_ids: list[str]) -> Orbits
   # query_horizons(object_ids: list[str], epoch: Timestamp) -> Orbits
   # query_neocc(object_ids: list[str]) -> Orbits
   # query_scout(object_ids: list[str] | str) -> VariantOrbits
