Observations and Observers
==========================

Simple Case
-----------

.. code-block:: python

   from adam_core.observations import Exposures, PointSourceDetections

   exposures = Exposures.from_kwargs(
       id=["exp-1"],
       start_time=start_times,
       duration=[30.0],
       filter=["r"],
       observatory_code=["I11"],
   )

   detections = PointSourceDetections.from_kwargs(
       id=["det-1"],
       exposure_id=["exp-1"],
       time=start_times,
       ra=[120.1],
       dec=[-2.3],
       ra_sigma=[0.15],
       dec_sigma=[0.15],
       mag=[20.3],
       mag_sigma=[0.1],
   )

Advanced Options
----------------

.. code-block:: python

   import pyarrow as pa
   from adam_core.observers import Observers

   # Vectorized code+time mapping for mixed-observatory observation streams.
   codes = pa.array(["I11", "X05", "I11"])
   observers = Observers.from_codes(codes, times)

   # Exposure midpoint observer states for ephemeris generation.
   exposure_observers = exposures.observers(frame="equatorial")

Custom SPICE Kernels for Observer States (JWST or Custom Spacecraft)
--------------------------------------------------------------------

Use this when observer states come from SPICE kernels instead of MPC station
codes.

.. code-block:: python

   import numpy as np
   from adam_core.observers import Observers
   from adam_core.time import Timestamp
   from adam_core.utils.spice import (
       get_spice_body_state,
       register_spice_kernel,
       unregister_spice_kernel,
   )

   times = Timestamp.from_mjd(np.array([60200.0, 60200.25]), scale="tdb")

   # 1) Load custom/SPICE mission kernel (example path).
   register_spice_kernel("/path/to/jwst_or_custom_spacecraft.bsp")

   # 2a) Name-based lookup via SPICE body name table (recommended when available).
   # If the kernel exposes this body name, Observers.from_code resolves it.
   jwst_observers = Observers.from_code("JWST", times)

   # 2b) Explicit NAIF ID lookup (works even if name mapping is unavailable).
   custom_states = get_spice_body_state(
       body_id=-170,  # example: JWST NAIF ID
       times=times,
       frame="ecliptic",
   )
   custom_observers = Observers.from_kwargs(
       code=["JWST"] * len(times),
       coordinates=custom_states,
   )

   # 3) Use these observers directly in ephemeris generation.
   # ephem = propagator.generate_ephemeris(orbits, jwst_observers, ...)

   unregister_spice_kernel("/path/to/jwst_or_custom_spacecraft.bsp")

When to Use
-----------

* ``Exposures`` + ``PointSourceDetections`` for clean ingest boundaries.
* ``Observers.from_codes`` for mixed-station production batches.
* ``Exposures.observers`` when building prediction-at-exposure-time pipelines.

Related Reference
-----------------

* :doc:`../reference/observations`
* :doc:`../reference/observers`

Input Types
-----------
.. code-block:: python

   # Exposures.from_kwargs(id: list[str], start_time: Timestamp, duration: list[float], filter: list[str], observatory_code: list[str], ...) -> Exposures
   # PointSourceDetections.from_kwargs(id: list[str], exposure_id: list[str], time: Timestamp, ra: list[float], dec: list[float], ...) -> PointSourceDetections
   # Observers.from_codes(codes: pa.Array, times: Timestamp) -> Observers
   # Observers.from_code(code: str | OriginCodes, times: Timestamp) -> Observers
   # exposures.observers(frame: str = "equatorial") -> Observers
   # register_spice_kernel(kernel_path: str) -> None
   # get_spice_body_state(body_id: int, times: Timestamp, frame: str = "ecliptic", origin: OriginCodes = OriginCodes.SUN) -> CartesianCoordinates
