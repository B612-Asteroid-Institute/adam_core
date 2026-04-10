Coordinate Transforms
=====================

Most production flows should start with ``transform_coordinates``.
Use direct low-level converters only when you need strict control over each step.

High-Level API: transform_coordinates
-------------------------------------

.. code-block:: python

   from adam_core.coordinates import CartesianCoordinates, SphericalCoordinates, transform_coordinates
   from adam_core.coordinates.origin import OriginCodes

   # Canonicalize to ecliptic heliocentric Cartesian before dynamics work.
   canonical = transform_coordinates(
       coords,
       CartesianCoordinates,
       frame_out="ecliptic",
       origin_out=OriginCodes.SUN,
   )

   # Convert to spherical for observation-space comparisons.
   on_sky = transform_coordinates(
       canonical,
       SphericalCoordinates,
       frame_out="equatorial",
   )

Atomic Frame/Origin Operations
------------------------------

.. code-block:: python

   from adam_core.coordinates.transform import cartesian_to_origin, cartesian_to_frame
   from adam_core.coordinates.origin import OriginCodes

   # Translation only.
   bary = cartesian_to_origin(cartesian_coords, OriginCodes.SOLAR_SYSTEM_BARYCENTER)

   # Rotation only.
   eq = cartesian_to_frame(bary, "equatorial")

Time-Varying Rotation Path
--------------------------

Rotations involving ``itrf93`` are time-varying and use SPICE transforms
per unique epoch.

.. code-block:: python

   from adam_core.coordinates.transform import apply_time_varying_rotation

   itrf = apply_time_varying_rotation(eq, frame_out="itrf93")

Low-Level Representation Converters
-----------------------------------

Use these if you are operating directly on NumPy arrays.

.. code-block:: python

   import numpy as np
   from adam_core.coordinates.transform import (
       cartesian_to_spherical,
       spherical_to_cartesian,
       cartesian_to_keplerian,
       keplerian_to_cartesian,
       cartesian_to_cometary,
       cometary_to_cartesian,
       cartesian_to_geodetic,
   )

   x = np.array([[1.0, 0.0, 0.2, 0.0, 0.01, 0.0]])
   t_mjd_tdb = np.array([60200.0])
   mu_sun = np.array([2.959122082855911e-04])

   sph = cartesian_to_spherical(x)
   x2 = spherical_to_cartesian(sph)

   kep13 = cartesian_to_keplerian(x, t_mjd_tdb, mu=mu_sun)
   kep6 = kep13[:, [0, 4, 5, 6, 7, 8]]  # a, e, i, raan, ap, M
   x3 = keplerian_to_cartesian(kep6, mu=mu_sun)

   com = cartesian_to_cometary(x, t_mjd_tdb, mu=mu_sun)
   x4 = cometary_to_cartesian(com, t0=t_mjd_tdb, mu=mu_sun)

Translation Cache: Why It Exists
--------------------------------

The expensive part of many origin transforms is repeated perturber-state lookup
for the same time grid. ``adam_core`` keeps an in-process LRU cache for common
translations (currently SUN <-> SOLAR_SYSTEM_BARYCENTER) keyed by:

* input/output origins
* frame
* time-series signature (size, first, last)
* order-sensitive time digest

This is why repeated bulk transforms on identical time arrays become much faster.

Cache Controls
--------------

Environment variables are read at module import time.

.. code-block:: bash

   export ADAM_CORE_TRANSLATION_CACHE=1
   export ADAM_CORE_TRANSLATION_CACHE_MAXSIZE=4096

Testing/benchmarking helper:

.. code-block:: python

   from adam_core.coordinates.transform import clear_translation_cache

   clear_translation_cache()

Recommended Usage Pattern
-------------------------

1. Normalize origin and frame once at pipeline boundaries.
2. Keep internal operations in Cartesian where possible.
3. Convert to output representation (spherical/keplerian/cometary/geodetic) at the edge.

Related Reference
-----------------

* :doc:`../reference/api/adam_core.coordinates`
* :doc:`../reference/api/adam_core.propagator`

