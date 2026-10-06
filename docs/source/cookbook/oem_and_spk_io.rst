Trajectory Interchange: OEM and SPK Workflows
=============================================

This page explains why OEM and SPK exist, when to use each format, and how to
generate them from ``adam_core`` state histories.

Format Background
-----------------

* OEM (CCSDS Orbit Ephemeris Message):
  human-readable interchange format for timestamped Cartesian state vectors
  (and optional covariance), useful for auditability and cross-tool exchange.
* SPK (NAIF/SPICE kernel):
  binary ephemeris format optimized for SPICE ecosystems and operational
  mission tooling.

In ``adam_core``, both formats are generated from Cartesian state histories
represented as ``Orbits`` rows over time.

Input Model: Orbit Seed vs Ephemeris State History
--------------------------------------------------

You can start from:

* a seed orbit + propagation window, or
* an already propagated state history (often called an ephemeris trajectory).

If your "ephemeris set" is already Cartesian states through time, convert that
table into ``Orbits`` and export directly.

Generate OEM from a Pre-Propagated State History
------------------------------------------------

.. code-block:: python

   from adam_core.orbits import Orbits
   from adam_core.orbits.oem_io import orbit_to_oem

   # `propagated_orbits` must represent one object_id with multiple epochs.
   propagated_orbits: Orbits = propagated_orbits.sort_by("coordinates.time")

   oem_path: str = orbit_to_oem(
       orbits=propagated_orbits,
       output_file="apophis.oem",
       originator="ADAM CORE USER",
   )
   print(oem_path)

Generate OEM from a Seed Orbit (Propagation Included)
-----------------------------------------------------

.. code-block:: python

   import numpy as np
   from adam_assist import ASSISTPropagator
   from adam_core.orbits import Orbits
   from adam_core.orbits.oem_io import orbit_to_oem_propagated
   from adam_core.time import Timestamp

   seed_orbit: Orbits = seed_orbit
   times: Timestamp = Timestamp.from_mjd(np.arange(60200.0, 60210.0, 1.0), scale="tdb")

   oem_path: str = orbit_to_oem_propagated(
       orbits=seed_orbit,
       output_file="apophis_propagated.oem",
       times=times,
       propagator_klass=ASSISTPropagator,
       originator="ADAM CORE USER",
   )

OEM with Explicit CCSDS Metadata
--------------------------------

``orbit_to_oem`` writes one fixed shape of file: OEM version 2.0 with
``REF_FRAME = EME2000``. When a consumer specifies the labels, build an
:class:`~adam_core.orbits.oem.OrbitEphemerisMessage` instead. Its metadata is
explicit, it writes OEM 3.0, and it reads any header and metadata keyword back.

The input is the same multi-epoch ``Orbits`` table ``propagate_orbits`` returns.
adam_core's ``"equatorial"`` frame is the J2000 frame SPICE and DE440 deliver,
which NAIF aligns with the ICRF, so the default label for it is ``ICRF``.
``EME2000`` is available as an override. Epochs are written with three decimal
places of seconds, and an epoch that is not on a millisecond boundary raises
unless ``allow_epoch_rounding=True`` is passed.

.. code-block:: python

   from adam_core.orbits import OrbitEphemerisMessage

   # propagated: Orbits with one object_id, N epochs, frame "equatorial",
   # origin SUN, time scale "tdb", from propagate_orbits(..., covariance=True).
   message: OrbitEphemerisMessage = OrbitEphemerisMessage.from_orbits(
       propagated,
       originator="B612 ASTEROID INSTITUTE",
       object_name="99942 Apophis (2004 MN4)",
       object_id="99942",
       ref_frame="ICRF",        # default for equatorial states
       time_system="TDB",       # default is the Timestamp scale
   )
   message.write("nominal_states.oem")   # states only, CENTER_NAME = SUN

   # Read it back. Every segment is kept, states are in AU and AU/day.
   loaded = OrbitEphemerisMessage.from_kvn("nominal_states.oem")
   print(loaded.segments[0].metadata)
   states = loaded.to_orbits()

A covariance block is written only on request. ``covariance_frame`` names the
frame of the block.

.. code-block:: python

   # State covariance in the OEM reference frame. Conformant.
   message.write("nominal_states_icrf_cov.oem", covariance_frame="ICRF")

   # Covariance rotated into a local orbital frame at every epoch. The OEM
   # standard lists RSW, RTN and TNW for COV_REF_FRAME.
   message.write("nominal_states_tnw_cov.oem", covariance_frame="TNW")

   # VNC_ROTATING is a registered SANA frame outside that OEM list. The
   # writer records that with a COMMENT line. Pass strict=True to refuse it.
   message.write("nominal_states_vnc_cov.oem", covariance_frame="VNC_ROTATING")

The separate covariance product for local orbital frames is described in
:doc:`local_orbital_frames`.

Read OEM Back into ``Orbits``
-----------------------------

.. code-block:: python

   from adam_core.orbits import Orbits
   from adam_core.orbits.oem_io import orbit_from_oem

   loaded_orbits: Orbits = orbit_from_oem("apophis_propagated.oem")
   print(len(loaded_orbits))

Generate SPK from ``Orbits``
----------------------------

``orbits_to_spk`` can propagate internally if you pass ``propagator=...``.
For production products, use a high-fidelity propagator such as
``adam_assist.ASSISTPropagator``.

.. code-block:: python

   from adam_assist import ASSISTPropagator
   from adam_core.orbits import Orbits
   from adam_core.orbits.spice_kernel import orbits_to_spk
   from adam_core.time import Timestamp

   seed_orbits: Orbits = seed_orbits
   start_time: Timestamp = Timestamp.from_iso8601(["2028-01-01T00:00:00"], scale="tdb")
   end_time: Timestamp = Timestamp.from_iso8601(["2028-06-01T00:00:00"], scale="tdb")

   target_id_map: dict[str, int] = orbits_to_spk(
       orbits=seed_orbits,
       output_file="objects.bsp",
       start_time=start_time,
       end_time=end_time,
       propagator=ASSISTPropagator(),
       step_days=0.25,
       window_days=32.0,
       kernel_type="w03",
       max_processes=8,
   )
   print(target_id_map)

From Ephemeris-Like State Tables to OEM/SPK
-------------------------------------------

If you already have Cartesian state rows (for example from an internal
trajectory service), build ``Orbits`` and export.

.. code-block:: python

   from adam_core.coordinates.cartesian import CartesianCoordinates
   from adam_core.orbits import Orbits

   state_history: CartesianCoordinates = state_history
   export_orbits: Orbits = Orbits.from_kwargs(
       orbit_id=["traj-001"] * len(state_history),
       object_id=["Apophis"] * len(state_history),
       coordinates=state_history,
   )

   # Then reuse orbit_to_oem(...) or orbits_to_spk(...).

Load Custom SPKs for Observer/Ephemeris Workflows
-------------------------------------------------

Custom kernels (for example JWST or self-generated ``.bsp`` files) can be
registered and then used by observer/ephemeris workflows.

.. code-block:: python

   from adam_core.observers import Observers
   from adam_core.time import Timestamp
   from adam_core.utils.spice import register_spice_kernel, unregister_spice_kernel

   times: Timestamp = Timestamp.from_mjd([60200.0, 60200.25], scale="tdb")

   register_spice_kernel("objects.bsp")
   # If a SPICE body name is present (e.g. "JWST"), observer lookup can use it directly.
   custom_observers: Observers = Observers.from_code("JWST", times)
   # Ephemeris generation can now use these observers.
   # ephemeris = propagator.generate_ephemeris(orbits, custom_observers, ...)
   unregister_spice_kernel("objects.bsp")

How These Products Are Used
---------------------------

* OEM:
  reviewable trajectory exchange, validation artifacts, and handoff between
  teams/tools that prefer text standards.
* SPK:
  mission operations, SPICE-native analysis, OpenSpace pipelines, and
  external tools that consume NAIF kernels directly.

Practical Notes
---------------

* OEM export requires one ``object_id`` per file call and benefits from
  multi-epoch state history.
* SPK export uses NAIF target IDs mapped per orbit via ``target_id_map``.
* For decision-grade trajectories, propagation quality is dominated by the
  propagator backend and force model choices.

Related Reference
-----------------

* :doc:`../reference/api/adam_core.orbits`
* :doc:`../reference/api/adam_core.propagator`
* :doc:`observations_and_observers`
