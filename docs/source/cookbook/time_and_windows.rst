Time and Windows
================

``Timestamp`` is the core time primitive across ``adam_core``. It stores integer
MJD days + integer nanoseconds with an explicit scale, which keeps joins,
windowing, and cache keys deterministic.

Construction Patterns
---------------------

Simple: operator-facing ISO input.

.. code-block:: python

   from adam_core.time import Timestamp

   t_utc = Timestamp.from_iso8601(
       ["2026-01-01T00:00:00", "2026-01-02T12:30:00"],
       scale="utc",
   )

Numeric: simulation and propagation pipelines.

.. code-block:: python

   import numpy as np
   from adam_core.time import Timestamp

   t_tdb = Timestamp.from_mjd(np.array([60200.0, 60200.25, 60201.0]), scale="tdb")
   t_jd = Timestamp.from_jd([2460200.5], scale="tt")

Interoperability: Astropy and ET.

.. code-block:: python

   from adam_core.time import Timestamp

   t_et = Timestamp.from_et([0.0, 86400.0], scale="tdb")
   astropy_time = t_et.to_astropy()
   round_trip = Timestamp.from_astropy(astropy_time)

Atomic Operations
-----------------

Representation and formatting.

.. code-block:: python

   iso = t_tdb.to_iso8601()
   mjd = t_tdb.mjd()
   jd = t_tdb.jd()
   et = t_tdb.et()
   as_numpy_tdb_mjd = t_tdb.to_numpy()

Arithmetic.

.. code-block:: python

   shifted = (
       t_tdb
       .add_days(3)
       .add_seconds(45)
       .add_millis(10)
       .add_micros(5)
       .add_nanos(25)
   )

   plus_fractional = t_tdb.add_fractional_days([0.125, 0.125, 0.125])

Differences and equality.

.. code-block:: python

   delta_days, delta_nanos = shifted.difference(t_tdb)
   exact_equal = shifted.equals(t_tdb, precision="ns")
   second_equal = shifted.equals(t_tdb, precision="s")

Min/max/unique for time windows.

.. code-block:: python

   earliest = t_tdb.min()
   latest = t_tdb.max()
   unique_epochs = t_tdb.unique()

Scale Management
----------------

Use explicit scales whenever data crosses system boundaries.

.. code-block:: python

   tdb = t_utc.rescale("tdb")
   tai = tdb.rescale("tai")
   tt = tai.rescale("tt")

If you need the astropy implementation for a specific validation path:

.. code-block:: python

   tdb_astropy = t_utc.rescale_astropy("tdb")

Linking and Window Joins
------------------------

``Timestamp.link`` is useful when joining tables on time with controlled precision.

.. code-block:: python

   # Match rows at millisecond precision.
   link = t_utc.link(t_utc.rounded("ms"), precision="ms")

Cache and Grouping Keys
-----------------------

High-throughput pipelines can use ``key``, ``signature``, and ``cache_digest``
for stable cache keys and quick identity checks.

.. code-block:: python

   keys = t_tdb.key(scale="tdb")
   signature = t_tdb.signature(scale="tdb")
   digest = t_tdb.cache_digest(scale="tdb")

When to Use Which Constructor
-----------------------------

* ``from_iso8601``: API inputs, operator tooling, external payloads.
* ``from_mjd``/``from_jd``: numerical pipelines and propagation windows.
* ``from_et``: SPICE-style integration points.
* ``from_astropy``: interoperability with existing astronomy tooling.

Related Reference
-----------------

* :doc:`../reference/time`
* :doc:`../reference/functionality_inventory`

Input Types
-----------
.. code-block:: python

   # Timestamp.from_iso8601(times: list[str], scale: str) -> Timestamp
   # Timestamp.from_mjd(mjd: np.ndarray | list[float], scale: str) -> Timestamp
   # Timestamp.rescale(scale: str) -> Timestamp
   # Timestamp.difference(other: Timestamp) -> tuple[pyarrow.Array, pyarrow.Array]
