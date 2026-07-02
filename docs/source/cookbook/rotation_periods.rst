Rotation Periods and Measured Confidence
========================================

``adam_core.photometry`` can recover an asteroid's rotation period from sparse,
multi-band photometry and, crucially, return a **measured confidence verdict**
instead of a bare number. The estimator fits a truncated-harmonic Fourier model to
the distance-reduced, light-time-corrected lightcurve, searches a frequency grid,
clusters harmonic aliases, and classifies the outcome.

The headline is the verdict, not the period: the API is built so that a caller can
tell, per object, whether to trust the number.

What You Get Back
-----------------

Every solve returns a one-row ``RotationPeriodResult``
whose fields fall into three groups:

* **The answer** -- ``period_hours`` / ``period_days``,
  ``frequency_cycles_per_day``, an uncertainty interval
  (``period_lower_days`` / ``period_upper_days`` /
  ``relative_period_uncertainty``), and ``alternate_period_days`` (the other
  candidates considered -- this is where harmonic aliases live).
* **How sure it is** -- ``period_verdict`` and ``reliability_code`` (read these
  first; see below).
* **Why** -- ``confidence_flags`` and ``insufficiency_reasons`` (machine-readable
  rationale), plus diagnostics you can sanity-check yourself: ``amplitude_snr``,
  ``phase_coverage_fraction``, ``n_rotations_spanned``, ``n_observations``.

Confidence Model and Guarantees
-------------------------------

``period_verdict`` takes one of
three values, and it is a decision guide:

* ``single_period`` -- one period is believed. In validation against LCDB/DAMIT
  standard-candle asteroids these calls are correct the large majority of the time.
  Use the value directly.
* ``period_family`` -- there is a real signal, but a harmonic ambiguity (typically
  2x / 0.5x) cannot be ruled out. The reported period may be off by an integer
  factor; consult ``alternate_period_days`` for the candidate family.
* ``insufficient_data`` -- the data cannot responsibly support a period. The period
  fields are ``NaN``; ``insufficiency_reasons`` gives a first reason as to why a 
  reliable period could not be fit (too few observations, too little phase 
  coverage, spans too few rotations, ...).

``reliability_code`` mirrors the LCDB "U" quality scale as a **string** --
``"3"`` (secure), ``"2"`` (some ambiguity), ``"1"`` (weak). It is deterministic
from the verdict (``single_period`` -> ``"3"``, ``period_family`` -> ``"2"``,
``insufficient_data`` -> ``"1"``). Do not sort or compare it numerically.

Note that ``single_period`` is high-confidence but **not a zero-alias guarantee.** A
  confident call can, rarely, be a harmonic alias of the true period. If a period
  that is wrong by an integer factor would be costly, cross-check a
  ``single_period`` result against ``alternate_period_days`` and
  ``reliability_code`` rather than treating it as infallible.

Data Requirements
-----------------

The solver works on ``RotationPeriodObservations``: one
row per photometric measurement, carrying the observing geometry so the intrinsic
rotational variation can be isolated.

* Required per row: ``time``, ``mag``, ``r_au`` (heliocentric distance),
  ``delta_au`` (observer distance), ``phase_angle_deg``.
* Optional: ``mag_sigma`` (per-point uncertainty), ``filter`` (multi-band is fit
  jointly with per-band offsets), ``session_id`` (per-night labels used for
  session-offset handling).
* Reductions are applied internally: distances are removed via
  ``mag - 5 * log10(r_au * delta_au)`` and times are light-time corrected.

Estimating From Arrays
----------------------

The most direct path: build the observations table and call the estimator.

.. code-block:: python

   import numpy as np
   from adam_core.photometry import (
       RotationPeriodObservations,
       estimate_rotation_period,
   )
   from adam_core.time import Timestamp

   observations = RotationPeriodObservations.from_kwargs(
       time=Timestamp.from_mjd(mjd, scale="tdb"),
       mag=mag,
       mag_sigma=mag_sigma,          # optional
       filter=filters,               # optional; e.g. ["g", "r", "g", ...]
       session_id=session_ids,       # optional; per-night labels
       r_au=r_au,
       delta_au=delta_au,
       phase_angle_deg=phase_angle_deg,
   )

   result = estimate_rotation_period(observations)

   verdict = result.period_verdict[0].as_py()
   if verdict == "single_period":
       print(f"P = {result.period_hours[0].as_py():.3f} h "
             f"(reliability {result.reliability_code[0].as_py()})")
   elif verdict == "period_family":
       print("period family; candidates:",
             result.alternate_period_days[0].as_py())
   else:
       print("insufficient:", result.insufficiency_reasons[0].as_py())

Estimating From Detections and Exposures
----------------------------------------

If you are already working with ``adam_core`` observation primitives, the geometry
is derived for you from aligned detections, exposures, and heliocentric object
coordinates.

.. code-block:: python

   from adam_core.photometry import estimate_rotation_period_from_detections

   # object_coords must be heliocentric (origin=SUN), row-aligned with detections.
   result = estimate_rotation_period_from_detections(
       detections=detections,
       exposures=exposures,
       object_coords=object_coords,
   )

The same geometry pipeline is available directly as
``RotationPeriodObservations.from_point_source_observations(detections, exposures,
object_coords)`` if you want the observations table rather than a solve.

Survey Scale: One Row Per Object
--------------------------------

For many objects at once, the grouped wrapper solves per object id and returns a
``GroupedRotationPeriodResults`` table.

.. code-block:: python

   from adam_core.photometry import estimate_rotation_period_from_detections_grouped

   grouped = estimate_rotation_period_from_detections_grouped(
       detections=detections,
       exposures=exposures,
       object_coords=object_coords,
       object_ids=object_ids,
   )

   # grouped.object_id  -> the object ids
   # grouped.result     -> a RotationPeriodResult column, one row per id

The grouped API **never silently drops an object**: an object whose solve fails on
expected bad/insufficient data comes back as an ``insufficient_data`` row carrying a
``solve_error`` flag, so the output always has exactly one row per distinct input
id. Unexpected (programmer/contract) errors are re-raised with the offending object
id attached rather than swallowed.

Performance
-----------

The defaults are validated and correct out of the box. The solver has an optional
JAX backend that is numerically identical to the NumPy path but faster on large
frequency grids:

.. code-block:: python

   result = estimate_rotation_period(
       observations,
       search_fidelity="validated_staged",   # default; coarse-then-refine on big grids
       exact_evaluation_backend="jax",        # ~2-5x faster than "numpy", same numbers
   )

``session_mode`` (default ``"auto"``) controls how multi-night magnitude offsets
are handled; the default decides per object whether to fit per-session offsets.

Entry Points
-----------

* ``estimate_rotation_period``: you already have a per-object RotationPeriodObservations table.
* ``estimate_rotation_period_from_detections``: single object, from detections /
  exposures / heliocentric coordinates.
* ``estimate_rotation_period_from_detections_grouped``: many objects; guarantees one
  result row per id.
* ``RotationPeriodObservations.from_point_source_observations``: build the
  observations table (e.g. to inspect or reuse) without solving.

Related Documentation
---------------------

* :doc:`photometry_and_magnitude`
* :doc:`../use_cases/observability_and_light_curves`
* :doc:`../reference/api/adam_core.photometry`
