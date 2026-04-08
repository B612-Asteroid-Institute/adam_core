Use-Case Progression (Simple to Advanced)
=========================================

This page maps common astronomy operations to ``adam_core`` capabilities in a
practical progression. Start at the smallest operational question, then move
toward higher-cost, higher-confidence analyses.

How to Read This
----------------

* Start with the first level you actually need.
* Skip ahead only when your decisions require more uncertainty modeling.
* Use :doc:`../cookbook/index` for atomic API details and type-level behavior.

1. Object Attribution and Candidate Confirmation
------------------------------------------------

Question:
Does this detection belong to this orbit candidate?

Core capabilities:

* ``Propagator.generate_ephemeris`` for predicted sky-plane states at observation times.
* ``Residuals.calculate`` and chi-square diagnostics for match quality.
* ``Observers`` + ``Observations`` tables for consistent observer/time geometry.

Use when:

* triaging new detections
* assigning detections to candidate objects
* deciding whether to escalate to orbit fitting

See:

* :doc:`orbit_determination`
* :doc:`../cookbook/residuals`
* :doc:`../cookbook/observations_and_observers`

2. Multi-Source Orbit Ingest and Normalization
-----------------------------------------------

Question:
How do I accept orbit inputs from mixed sources but keep one stable downstream contract?

Core capabilities:

* ``query_sbdb`` / ``query_scout`` / ``query_neocc`` / ``query_horizons``
* ``VariantOrbits.collapse_by_object_id`` when source data arrives as ensembles

Use when:

* running service endpoints with selectable upstream sources
* normalizing ingest before propagation, ranking, or export

See:

* :doc:`source_multiplexing`
* :doc:`../cookbook/orbit_sources_and_state_tables`

3. Ephemeris Generation for Planning and Services
-------------------------------------------------

Question:
What will the object look like from a specific observatory over time?

Core capabilities:

* ``Propagator.propagate_orbits``
* ``Propagator.generate_ephemeris``
* ``Observers.from_code`` / ``Observers.from_codes``

Use when:

* generating observatory planning products
* powering internal/public prediction APIs
* batching nightly schedules

See:

* :doc:`ephemeris_service`
* :doc:`../cookbook/propagation_and_ephemeris`

4. Precovery-Style Uncertainty Search
-------------------------------------

Question:
Where should this uncertain orbit have appeared in historical survey data?

Core capabilities:

* covariance-aware state handling via ``CoordinateCovariances``
* ``VariantOrbits.create`` and ``collapse_by_object_id``
* observer-aware ephemeris generation over long time windows

Use when:

* searching archives with uncertain early-arc orbits
* balancing recall vs compute by switching sigma-point vs Monte Carlo variants

See:

* :doc:`neo_tracking`
* :doc:`../cookbook/variant_sampling_and_collapse`
* :doc:`../cookbook/coordinate_covariances`

5. Orbit Determination and Fit Refinement
-----------------------------------------

Question:
Can we turn linked observations into a defensible orbit with quality metrics?

Core capabilities:

* ``initial_orbit_determination``
* ``fit_least_squares``
* ``evaluate_orbits``

Use when:

* confirming candidates
* improving state/covariance quality
* outlier management and iterative fit loops

See:

* :doc:`orbit_determination`
* :doc:`../cookbook/orbit_determination_pipeline`

6. Observability and Brightness Ranking
---------------------------------------

Question:
Which targets are actually observable, not just geometrically visible?

Core capabilities:

* phase angle and magnitude prediction during ephemeris generation
* bandpass mapping and magnitude conversion APIs
* table-first light-curve construction

Use when:

* ranking nightly follow-up targets
* limiting-magnitude gating by site/filter
* comparing observing plans across facilities

See:

* :doc:`observability_and_light_curves`
* :doc:`../cookbook/photometry_and_magnitude`

7. Risk Triage and Decision-Grade Impact Analysis
-------------------------------------------------

Question:
Is this object only geometrically close, or probabilistically dangerous?

Core capabilities:

* fast geometric triage with ``calculate_moid``
* Monte Carlo impact simulation with ``calculate_impacts``
* probability summarization with ``calculate_impact_probabilities``

Use when:

* screening many objects quickly (MOID first)
* producing decision-grade impact probabilities (sampling + collision conditions)

See:

* :doc:`impact_risk`
* :doc:`../cookbook/moid_analysis`
* :doc:`../cookbook/impact_probabilities`

8. Interchange and Operations Outputs
-------------------------------------

Question:
How do I hand off products to other mission/ops tools?

Core capabilities:

* OEM write/read for standards-based orbit exchange
* SPK generation + SPICE state queries
* OpenSpace asset generation for visualization

Use when:

* integrating with SPICE-native systems
* publishing trajectory products
* building visualization/briefing artifacts

See:

* :doc:`../cookbook/oem_and_spk_io`
* :doc:`../cookbook/openspace_visualization`

Common Architecture Choice: Propagator Backend
----------------------------------------------

Most operational use cases above assume a concrete propagator backend.
For production-quality n-body propagation and impact/event handling, use a
backend such as ``adam-assist``.

* ``adam_core`` defines stable data models and interfaces.
* Backend propagators provide integration fidelity and event handling.
* Keep this separation explicit in system design.
