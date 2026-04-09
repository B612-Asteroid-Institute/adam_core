Photometry
==========

Apparent/absolute magnitude modeling helpers, bandpass tooling, grouped fitting,
and light-curve building blocks.

.. automodule:: adam_core.photometry

Key Interfaces
--------------

* ``calculate_phase_angle``
* ``calculate_apparent_magnitude_v``
* ``calculate_apparent_magnitude_v_and_phase_angle``
* ``predict_magnitudes``
* ``convert_magnitude``
* ``estimate_absolute_magnitude_v_from_detections``
* ``estimate_absolute_magnitude_v_from_detections_grouped``
* ``GroupedPhysicalParameters``
* ``bandpasses.map_to_canonical_filter_bands``
* ``bandpasses.assert_filter_ids_have_curves``
* ``bandpasses.load_bandpass_curves``
* ``bandpasses.load_observatory_band_map``
* ``bandpasses.load_asteroid_templates``
* ``bandpasses.load_template_integrals``
* ``bandpasses.get_integrals``
* ``bandpasses.compute_mix_integrals``
* ``bandpasses.bandpass_delta_mag``
* ``bandpasses.bandpass_color_terms``
* ``bandpasses.register_custom_template``

Related Guides
--------------

* :doc:`../cookbook/photometry_and_magnitude`
* :doc:`../use_cases/observability_and_light_curves`
* :doc:`../use_cases/ephemeris_service`
