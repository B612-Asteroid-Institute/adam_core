.. meta::
   :description: Brightness-aware observability analysis in adam_core with phase angle, bandpass conversion, and light-curve generation.

Observability and Light Curves
==============================

Problem
-------

You need to prioritize follow-up observations by expected brightness and phase
geometry, not just sky-plane position.

Implementation Options and Tradeoffs
------------------------------------

* V-band only predictions:
  Simple and fast; good when ranking is relative and filter effects are small.
* Bandpass-aware predictions:
  Better survey realism; requires canonical filter mapping and composition assumptions.
* Nominal orbit only:
  Lower compute cost; misses uncertainty-driven brightness spread.
* Variant ensemble collapse:
  Better uncertainty characterization; higher runtime and memory cost.

Runnable Example
----------------

.. code-block:: python

   import numpy as np
   import pyarrow as pa
   import pyarrow.compute as pc
   from adam_assist import ASSISTPropagator
   from adam_core.orbits import Orbits
   from adam_core.observers import Observers
   from adam_core.orbits.query import query_sbdb
   from adam_core.photometry import convert_magnitude
   from adam_core.photometry.bandpasses import map_to_canonical_filter_bands
   from adam_core.time import Timestamp

   orbits: Orbits = query_sbdb(["Apophis"])
   times: Timestamp = Timestamp.from_mjd(np.arange(60200.0, 60240.0, 1.0), scale="utc")
   observers: Observers = Observers.from_code("I41", times)

   propagator = ASSISTPropagator()
   ephem = propagator.generate_ephemeris(
       orbits,
       observers,
       predict_magnitudes=True,
       predict_phase_angle=True,
       max_processes=4,
   )

   # Example: convert predicted V magnitudes to a specific canonical filter.
   target_filter_id = map_to_canonical_filter_bands(
       observers.code,
       pa.array(["r"] * len(observers), type=pa.large_string()),
       allow_fallback_filters=True,
   )
   mag_v = pc.fill_null(ephem.predicted_magnitude_v, np.nan).to_numpy(
       zero_copy_only=False
   )
   mag_r = convert_magnitude(
       magnitude=mag_v,
       source_filter_id=np.array(["V"] * len(mag_v), dtype=object),
       target_filter_id=np.asarray(target_filter_id, dtype=object),
       composition="NEO",
   )

   # Light-curve-like table
   df = ephem.to_dataframe()
   df["predicted_magnitude_target"] = mag_r
   print(df[["coordinates.time.days", "predicted_magnitude_v", "alpha"]].head())

When to Use This Pattern
------------------------

Use this for follow-up scheduling, limiting-magnitude gating, and ranking
candidate observability across nights and observatories.

Related Documentation
---------------------

* :doc:`../cookbook/photometry_and_magnitude`
* :doc:`../reference/photometry`
* :doc:`../reference/propagator`
