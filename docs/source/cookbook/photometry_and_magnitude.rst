Photometry, Phase Angle, and Light Curves
=========================================

This guide covers the full ``adam_core`` photometry stack:

1. geometric phase-angle and V-band magnitude primitives
2. bandpass-aware magnitude prediction and conversion
3. inverse fitting of ``H_v`` from detections
4. light-curve construction from nominal and variant ephemerides

Photometry Model and Data Requirements
--------------------------------------

Key assumptions used by ``adam_core.photometry``:

* H-G model for apparent magnitude.
* ``H_v`` and ``G`` are read from orbit physical parameters for ephemeris-level prediction.
* Direct geometry helpers expect object and observer states in a shared heliocentric frame.
* Bandpass-aware APIs require canonical filter IDs (map from reported bands first).

Atomic Geometry Functions
-------------------------

Use these when you already have row-aligned object and observer states.

.. code-block:: python

   from adam_core.photometry import (
       calculate_apparent_magnitude_v,
       calculate_apparent_magnitude_v_and_phase_angle,
       calculate_phase_angle,
   )

   # object_coords and observers should be aligned row-wise (same length),
   # in a shared heliocentric frame and origin.
   alpha_deg = calculate_phase_angle(object_coords, observers)
   mag_v = calculate_apparent_magnitude_v(
       H_v=20.1,
       object_coords=object_coords,
       observer=observers,
       G=0.15,
   )

   # Faster when you need both outputs.
   mag_v2, alpha_deg2 = calculate_apparent_magnitude_v_and_phase_angle(
       H_v=20.1,
       object_coords=object_coords,
       observer=observers,
       G=0.15,
   )

Ephemeris-Level Photometry and Phase Angle
------------------------------------------

For operational pipelines, usually compute these through ephemeris generation.

.. code-block:: python

   import numpy as np
   from adam_assist import ASSISTPropagator
   from adam_core.observers import Observers
   from adam_core.orbits.query import query_sbdb
   from adam_core.time import Timestamp

   orbits = query_sbdb(["Apophis"])
   times = Timestamp.from_mjd(
       np.arange(60200.0, 60230.0, 1.0),
       scale="utc",
   )
   observers = Observers.from_code("I41", times)

   propagator = ASSISTPropagator()
   ephemeris = propagator.generate_ephemeris(
       orbits,
       observers,
       predict_magnitudes=True,
       predict_phase_angle=True,
       max_processes=4,
   )

   # Ephemeris columns now include:
   # - ephemeris.predicted_magnitude_v
   # - ephemeris.alpha

Light-Curve Construction
------------------------

``adam_core`` does not require a dedicated light-curve class. Build light-curve
products directly from ephemeris tables.

.. code-block:: python

   import matplotlib.pyplot as plt
   import pyarrow.compute as pc

   df = ephemeris.to_dataframe()
   df = df.sort_values("coordinates.time.days")

   # Filter to rows with valid predicted magnitude.
   valid = ~df["predicted_magnitude_v"].isna()
   lc = df.loc[valid, ["coordinates.time.days", "predicted_magnitude_v", "alpha"]]

   fig, ax = plt.subplots(figsize=(8, 4))
   ax.plot(lc["coordinates.time.days"], lc["predicted_magnitude_v"], marker=".")
   ax.invert_yaxis()  # brighter objects are lower magnitudes
   ax.set_xlabel("MJD (UTC)")
   ax.set_ylabel("Predicted magnitude (V)")
   ax.set_title("Predicted V-band light curve")

Bandpass-Aware Prediction
-------------------------

For survey-specific filters, map reported bands to canonical filter IDs and then
predict magnitudes per exposure.

.. code-block:: python

   import numpy as np
   import pyarrow as pa
   from adam_core.photometry import predict_magnitudes
   from adam_core.photometry.bandpasses import map_to_canonical_filter_bands

   # exposures has observatory_code and reported filter strings.
   canonical = map_to_canonical_filter_bands(
       exposures.observatory_code,
       exposures.filter,
       allow_fallback_filters=True,
   )
   exposures_canon = exposures.set_column(
       "filter",
       pa.array(canonical, type=pa.large_string()),
   )

   mags = predict_magnitudes(
       H=20.1,
       object_coords=object_coords,
       exposures=exposures_canon,
       G=0.15,
       reference_filter="V",
       composition="NEO",
   )

Bandpass Conversion and Color Terms
-----------------------------------

Use these when you already have magnitudes in one canonical filter and need
another.

.. code-block:: python

   import numpy as np
   from adam_core.photometry import convert_magnitude
   from adam_core.photometry.bandpasses import (
       bandpass_color_terms,
       bandpass_delta_mag,
       register_custom_template,
   )

   m_v = np.array([20.1, 20.4, 20.8], dtype=float)
   m_r = convert_magnitude(
       magnitude=m_v,
       source_filter_id=np.array(["V", "V", "V"], dtype=object),
       target_filter_id=np.array(["LSST_r", "LSST_r", "LSST_r"], dtype=object),
       composition="NEO",
   )

   delta_v_to_g = bandpass_delta_mag("NEO", "V", "LSST_g")
   color_terms = bandpass_color_terms("NEO", source_filter_id="V")

   # Optional: register a custom reflectance template for local analysis.
   register_custom_template(
       template_id="CUSTOM_RED",
       wavelength_nm=np.array([400.0, 600.0, 900.0]),
       reflectance=np.array([0.8, 1.0, 1.3]),
   )

Inspecting Vendored Bandpass Data
---------------------------------

For validation and debugging, inspect the packaged curves/maps/integrals tables.

.. code-block:: python

   import numpy as np
   from adam_core.photometry.bandpasses import (
       compute_mix_integrals,
       get_integrals,
       load_asteroid_templates,
       load_bandpass_curves,
       load_observatory_band_map,
       load_template_integrals,
   )

   curves = load_bandpass_curves()
   mapping = load_observatory_band_map()
   templates = load_asteroid_templates()
   integrals = load_template_integrals()

   ids = np.array(["V", "LSST_r", "DECam_g"], dtype=object)
   neo_int = get_integrals("NEO", ids)
   custom_mix = compute_mix_integrals(0.7, 0.3, ids)

Estimating ``H_v`` From Detections
----------------------------------

Use inverse fitting when detections and exposure metadata are available.

.. code-block:: python

   from adam_core.photometry import (
       estimate_absolute_magnitude_v_from_detections,
       estimate_absolute_magnitude_v_from_detections_grouped,
   )

   # Single-object fit.
   fitted = estimate_absolute_magnitude_v_from_detections(
       detections=detections,
       exposures=exposures,
       object_coords=object_coords,
       composition="NEO",
       G=0.15,
       strict_band_mapping=False,
   )

   # Multi-object grouped fit (survey scale).
   grouped = estimate_absolute_magnitude_v_from_detections_grouped(
       detections=detections,
       exposures=exposures,
       object_coords=object_coords,
       object_ids=object_ids,
       composition="NEO",
       G=0.15,
       strict_band_mapping=False,
   )

Variant Light Curves and Collapse
---------------------------------

For uncertain orbits, propagate variant ensembles and collapse back to a mean
light curve with covariance.

.. code-block:: python

   from adam_core.orbits import VariantOrbits

   variants = VariantOrbits.create(orbits, method="sigma-point")
   variant_ephemeris = propagator.generate_ephemeris(
       variants,
       observers,
       predict_magnitudes=True,
       predict_phase_angle=True,
       max_processes=4,
   )

   collapsed = variant_ephemeris.collapse_by_object_id(aberration_mode="recompute")
   print(collapsed.to_dataframe().head())

When to Use
-----------

* ``calculate_*`` primitives: custom geometry and debugging.
* ``generate_ephemeris(..., predict_magnitudes=True, predict_phase_angle=True)``:
  production light-curve generation.
* ``predict_magnitudes`` + bandpass mapping: survey/filter-aware prediction.
* ``estimate_absolute_magnitude_*``: calibrating ``H_v`` from detection history.
* ``VariantEphemeris.collapse_by_object_id``: uncertainty-aware light curves.

Related Documentation
---------------------

* :doc:`../use_cases/observability_and_light_curves`
* :doc:`../reference/api/adam_core.photometry`
* :doc:`../reference/api/adam_core.propagator`
* :doc:`../reference/api/adam_core.orbits`
