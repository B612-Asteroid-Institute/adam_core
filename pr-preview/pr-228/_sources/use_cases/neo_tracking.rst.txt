.. meta::
   :description: Track NEOCP candidates with Scout samples, observer geometry, and uncertainty-aware ephemerides using adam_core.

NEO Tracking for Follow-Up
==========================

Goal
----

Generate follow-up pointings for uncertain NEOCP candidates quickly and with
explicit uncertainty handling.

Operational Prerequisites
-------------------------

* Install a production propagator: ``adam-assist`` (recommended).
* Query access to CNEOS Scout for active NEOCP candidates.
* Observatory code(s) or explicit observer state vectors.

Canonical End-to-End Pattern
----------------------------

.. code-block:: python

   import pyarrow.compute as pc
   from adam_assist import ASSISTPropagator
   from adam_core.observers import Observers
   from adam_core.observations import ScoutObservations
   from adam_core.orbits import Orbits, VariantOrbits
   from adam_core.orbits.query.scout import (
       get_scout_objects,
       query_scout,
       query_scout_observations,
   )
   from adam_core.time import Timestamp

   # 1) Select candidate(s) from Scout's current NEOCP list.
   scout_objects = get_scout_objects()
   object_id = scout_objects.objectName[10].as_py()

   # 2) Capture the exact fitted-observation snapshot used by Scout.
   observations: ScoutObservations = query_scout_observations(object_id)
   print(observations.snapshot_sha256[0].as_py())
   print(observations.observation.time.to_iso8601())

   # 3) Fetch Scout posterior samples (usually ~1000 variants per object).
   # query_scout accepts an array-like collection of designations.
   samples: VariantOrbits = query_scout([object_id])

   # 4) Define exposure times and observer geometry.
   times: Timestamp = Timestamp.from_iso8601(
       [
           "2025-02-23T00:00:00Z",
           "2025-02-23T00:05:00Z",
           "2025-02-23T00:10:00Z",
       ],
       scale="utc",
   )
   observers: Observers = Observers.from_code("T08", times)
   propagator = ASSISTPropagator()

   # 5A) Fast operational path: collapse variants and propagate covariance.
   collapsed_orbits: Orbits = samples.collapse_by_object_id()
   collapsed_ephemeris = propagator.generate_ephemeris(
       collapsed_orbits,
       observers,
       covariance=True,
       num_samples=1000,
       max_processes=10,
   )

   # 5B) High-fidelity path: propagate each variant directly.
   sample_ephemeris = propagator.generate_ephemeris(
       samples,
       observers,
       max_processes=10,
   )

   # 6) Derive per-time on-sky envelopes for pointing decisions.
   for t in sample_ephemeris.coordinates.time.unique():
       at_t = sample_ephemeris.apply_mask(sample_ephemeris.coordinates.time.equals(t))
       print(
           at_t.coordinates.time.to_iso8601()[0],
           "RA range:",
           pc.min(at_t.coordinates.lon).as_py(),
           pc.max(at_t.coordinates.lon).as_py(),
           "Dec range:",
           pc.min(at_t.coordinates.lat).as_py(),
           pc.max(at_t.coordinates.lat).as_py(),
       )

Authoritative Scout Observation Snapshots
-----------------------------------------

``query_scout_observations()`` requests Scout's ``file=mpc`` representation
for each designation and returns a typed ``ScoutObservations`` table. The nested
``observation`` column contains ``OpticalObs80`` rows with UTC timestamps,
RA/Dec in degrees, observatory codes, the original 80-column records, and
nullable photometric and reference fields.

The returned provenance is deliberately repeated on every observation so that
filtering or slicing a table does not detach rows from their source snapshot:

* ``snapshot_sha256`` identifies the exact ``fileMPC`` text;
* ``solution_date_utc`` records Scout's ``lastRun`` value;
* ``signature_version`` and ``signature_source`` record the API signature;
* ``snapshot_observation_count`` records the parsed file membership; and
* ``observation_index`` preserves Scout's file order within the snapshot.

Scout's ``fileMPC`` rows, in their returned order, are authoritative for
snapshot membership. ``declared_n_obs`` preserves the separate ``nObs`` summary
value, but it does not add or remove observations when that value differs from
the file. Signature version ``1.3`` is required; missing or unsupported
signatures fail the request rather than silently changing the interpretation.
Multiple requested designations are concatenated in request order.

Strict MPC 80-Column Parsing
----------------------------

You can use the same parser for an MPC-format optical file obtained elsewhere:

.. code-block:: python

   from adam_core.observations import OpticalObs80
   from adam_core.observations.obs80 import parse_optical_obs80_file

   mpc_text = (
       "     A11EpSe*0C2026 07 08.17725719 41 24.185-30 19 19.42"
       "         19.35oVNEOCPW68\n"
   )
   optical: OpticalObs80 = parse_optical_obs80_file(mpc_text)

   print(optical.designation[0].as_py())       # A11EpSe
   print(optical.observatory_code[0].as_py())  # W68
   print(optical.time.to_iso8601()[0])
   print(optical.ra_deg[0].as_py(), optical.dec_deg[0].as_py())

The default ``strict=True`` is intentional. Every nonblank row must be a valid,
self-contained optical astrometry record; malformed rows and unsupported
radar, satellite, or roving-observer companion-line records raise
``Obs80ParseError`` with the file line number. The parser reads the full
12-column designation field, including numbered designations such as
``00001`` and packed/provisional values. Use ``strict=False`` only when a
best-effort generic file import is explicitly acceptable. Scout ingestion
always uses strict parsing so it never exposes a partial fitted observation
set.

Live NEOCP Semantics and Recovery
---------------------------------

Scout is a live service, not a durable NEOCP catalog. A designation returned by
``get_scout_objects()`` can be updated, replaced, or disappear before the next
request. Transport errors, HTTP 429 responses, and server errors receive
bounded retries. An exhausted transient failure raises
``ScoutServiceUnavailableError`` with ``http_status=503`` and
``retryable=True``; a valid error payload for a vanished designation raises
``ScoutObjectNotFoundError`` with ``http_status=404``. A malformed response, an
empty or invalid ``fileMPC`` value, a designation mismatch, or an unsupported
signature fails closed as ``ScoutResponseError``. Each structured exception
also carries the requested ``object_id`` and any upstream HTTP status.

For operational ingestion, treat such a failure as a reason to refresh the
active-object list and reconcile the candidate's lifecycle. Do not substitute
``nObs`` for file membership or retain a partially parsed snapshot. Persist the
snapshot table (especially its hash and solution date) alongside downstream
products so a later live response cannot silently rewrite their provenance.

``query_scout_observations()`` and ``query_scout()`` serve different products:
the former returns the observations fitted by Scout, while the latter returns
Scout's sampled orbit variants. They are separate live HTTP requests and are
not an atomic service snapshot. Capture their retrieval times and provenance
when a workflow requires both, and do not infer orbit-sample membership from
the observation count.

Implementation Choices
----------------------

* ``collapse_by_object_id()`` + ``covariance=True``:
  Lowest compute cost and easiest to deploy for routine scheduling.
* direct ``VariantOrbits`` propagation:
  Preserves multimodal structure and heavy tails in uncertainty.
* ``max_processes``:
  Primary speed control for short-notice campaign planning.

What You Get Back
-----------------

* ``collapsed_ephemeris``:
  Central track with covariance-derived uncertainty columns.
* ``sample_ephemeris``:
  Variant-wise tracks suitable for envelope/percentile summarization.
* Observer-aware sky coordinates at each requested time for immediate
  telescope planning.

When To Use
-----------

Use this path when you are planning near-term follow-up for newly discovered
candidates and need a defensible uncertainty-aware pointing recommendation.

Related Reference
-----------------

* :doc:`../reference/api/adam_core.orbits`
* :doc:`../reference/api/adam_core.observations`
* :doc:`../reference/api/adam_core.observers`
* :doc:`../reference/api/adam_core.propagator`
* :doc:`../cookbook/variant_sampling_and_collapse`
