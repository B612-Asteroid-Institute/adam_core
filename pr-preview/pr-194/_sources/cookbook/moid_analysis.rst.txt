MOID Analysis
=============

MOID (Minimum Orbit Intersection Distance) is a fast geometric risk-screening
metric between two orbits.

Single-Pair MOID
----------------

.. code-block:: python

   from adam_core.dynamics.moid import calculate_moid
   from adam_core.orbits import Orbits
   from adam_core.coordinates.origin import OriginCodes
   from adam_core.time import Timestamp
   from adam_core.utils.spice import get_perturber_state

   # Example: object orbit and Earth state at a common epoch.
   epoch = Timestamp.from_mjd([60200.0], scale="tdb")
   earth_state = get_perturber_state(OriginCodes.EARTH, epoch)
   earth_orbit = Orbits.from_kwargs(orbit_id=["EARTH"], coordinates=earth_state)

   moid_au, moid_time = calculate_moid(object_orbit, earth_orbit)
   print(moid_au, moid_time.to_iso8601()[0].as_py())

Batch MOIDs Against Perturbers
------------------------------

.. code-block:: python

   from adam_core.coordinates.origin import OriginCodes
   from adam_core.dynamics.moid import calculate_perturber_moids

   moids = calculate_perturber_moids(
       orbits=candidate_orbits,
       perturber=[OriginCodes.EARTH, OriginCodes.MARS_BARYCENTER, OriginCodes.VENUS],
       chunk_size=100,
       max_processes=8,
   )

   print(moids.to_dataframe().head())

Interpretation Guidance
-----------------------

* MOID is geometric, not a probability.
* Use MOID for rapid triage and candidate ranking.
* Follow with full variant propagation + impact probability for decision-quality risk.

When to Use
-----------

* First-pass impact-risk filtering.
* Dynamical neighborhood analysis.
* Ranking follow-up targets before expensive Monte Carlo impact runs.

Related Reference
-----------------

* :doc:`../reference/dynamics`
* :doc:`impact_probabilities`

Input Types
-----------
.. code-block:: python

   # get_perturber_state(perturber: OriginCodes, times: Timestamp, frame: str = "ecliptic", origin: OriginCodes = OriginCodes.SUN) -> CartesianCoordinates
   # calculate_moid(primary_ellipse: Orbits, secondary_ellipse: Orbits) -> tuple[float, Timestamp]
   # calculate_perturber_moids(orbits: Orbits, perturber: OriginCodes | list[OriginCodes], chunk_size: int = 100, max_processes: int | None = 1) -> PerturberMOIDs
