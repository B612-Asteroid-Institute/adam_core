Coordinate Classes Guide
========================

This guide covers all coordinate classes in ``adam_core`` and emphasizes their
class-specific methods. Start with ``CartesianCoordinates`` as the neutral
interchange format, then convert to specialized representations as needed.

Shared Setup
------------

.. code-block:: python

   import numpy as np
   from adam_core.coordinates import (
       CartesianCoordinates,
       CoordinateCovariances,
       Origin,
       OriginCodes,
   )
   from adam_core.time import Timestamp

   t = Timestamp.from_mjd(np.array([60200.0]), scale="tdb")
   cov = CoordinateCovariances.from_sigmas(np.array([[1e-9, 1e-9, 1e-9, 1e-11, 1e-11, 1e-11]]))

   cart = CartesianCoordinates.from_kwargs(
       x=[1.1], y=[0.2], z=[0.1],
       vx=[0.0], vy=[0.015], vz=[0.001],
       time=t,
       covariance=cov,
       origin=Origin.from_OriginCodes(OriginCodes.SUN, size=1),
       frame="ecliptic",
   )

CartesianCoordinates
--------------------

Use this class for propagation, frame/origin transforms, and low-level vector math.

.. code-block:: python

   # Core vector accessors.
   state = cart.values
   r = cart.r
   v = cart.v
   r_mag = cart.r_mag
   v_mag = cart.v_mag
   r_hat = cart.r_hat
   v_hat = cart.v_hat

   # Angular momentum diagnostics.
   h = cart.h
   h_mag = cart.h_mag

   # Unit helpers.
   state_km = cart.values_km
   r_km = cart.r_km
   v_km_s = cart.v_km_s

   # Local orbital frame rotations.
   ric3 = cart.ric3_matrix
   ric6 = cart.ric6_matrix

   # Explicit transforms on raw state vectors.
   translated = cart.translate(np.array([[0.001, 0.0, 0.0, 0.0, 0.0, 0.0]]), origin_out="SUN")

SphericalCoordinates
--------------------

Use for topocentric or on-sky style representations (``rho/lon/lat`` + rates).

.. code-block:: python

   from adam_core.coordinates import SphericalCoordinates

   sph = cart.to_spherical()

   # Fill missing rho/vrho for direction-only data on a unit sphere.
   sph_unit = sph.to_unit_sphere(only_missing=True)

   # Convert back when Cartesian dynamics are required.
   cart_back = sph.to_cartesian()

KeplerianCoordinates
--------------------

Use for orbital-element diagnostics and reporting.

.. code-block:: python

   kep = cart.to_keplerian()

   # Derived orbital quantities.
   q = kep.q      # periapsis distance
   Q = kep.Q      # apoapsis distance
   p = kep.p      # semi-latus rectum
   P = kep.P      # period
   n = kep.n      # mean motion

   # Convert back for propagation.
   cart_from_kep = kep.to_cartesian()

CometaryCoordinates
-------------------

Use when perihelion-time representation (``tp``) is the more natural interface.

.. code-block:: python

   com = cart.to_cometary()

   # Derived quantities available directly.
   a = com.a
   Q = com.Q
   p = com.p
   P = com.P
   n = com.n

   # Conversions retain time context, needed for tp-based transformations.
   cart_from_com = com.to_cartesian()

GeodeticCoordinates
-------------------

Use for Earth-fixed interpretation (``itrf93`` frame and ``EARTH`` origin).

.. code-block:: python

   from adam_core.coordinates import GeodeticCoordinates

   # Approximate Earth surface point in AU (position) and AU/day (velocity).
   earth_fixed = CartesianCoordinates.from_kwargs(
       x=[4.26e-5], y=[0.0], z=[0.0],
       vx=[0.0], vy=[0.0], vz=[0.0],
       time=t,
       covariance=CoordinateCovariances.nulls(1),
       origin=Origin.from_OriginCodes(OriginCodes.EARTH, size=1),
       frame="itrf93",
   )

   geodetic = GeodeticCoordinates.from_cartesian(earth_fixed)
   map_urls = geodetic.google_maps_url(zoom=14)

Origin and Gravitational Parameter
----------------------------------

``Origin`` carries the center body and supplies ``mu()`` used by element
conversions.

.. code-block:: python

   mu = cart.origin.mu()
   is_solar = (cart.origin == OriginCodes.SUN)

Cross-Class Conversion Matrix
-----------------------------

.. code-block:: python

   sph = cart.to_spherical()
   kep = cart.to_keplerian()
   com = cart.to_cometary()

   # Alternate constructors from another representation:
   from_kep = CartesianCoordinates.from_keplerian(kep)
   from_com = CartesianCoordinates.from_cometary(com)
   from_sph = CartesianCoordinates.from_spherical(sph)

When to Use This Pattern
------------------------

* Do dynamics and transformations in ``CartesianCoordinates``.
* Use ``SphericalCoordinates`` for observation-space comparisons.
* Use ``KeplerianCoordinates`` / ``CometaryCoordinates`` for interpretation and reporting.
* Use ``GeodeticCoordinates`` only for Earth-centered, Earth-fixed states.

Related Reference
-----------------

* :doc:`../reference/coordinates`
* :doc:`../reference/orbits`

