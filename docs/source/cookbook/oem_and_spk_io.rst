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

OEM 3.0 with Explicit Labels and Local Frame Covariances
---------------------------------------------------------

``orbit_to_oem`` writes OEM 2.0 with ``REF_FRAME = EME2000``. An
:class:`~adam_core.orbits.oem.OrbitEphemerisMessage` built from the same
``Orbits`` table writes OEM 3.0 with ``REF_FRAME = ICRF`` for equatorial
states, ``CENTER_NAME`` from the origin and ``TIME_SYSTEM`` from the Timestamp
scale.

.. code-block:: python

   from adam_core.orbits import OrbitEphemerisMessage
   from adam_core.coordinates import LocalFrameCovariances

   # propagated: Orbits of one object, frame "equatorial", origin SUN, scale "tdb"
   message = OrbitEphemerisMessage.from_orbits(
       propagated, originator="B612 ASTEROID INSTITUTE", object_name="99942 Apophis"
   )

   # States with a covariance block per epoch in VNC_ROTATING, a SANA frame
   # CCSDS 502.0-B-3 admits through annex B5. VNC_INERTIAL is the pure rotation,
   # TNW and RSW (RTN, RIC) work the same way, "ICRF" writes the state covariance.
   message.write("states_cov_vnc_rotating.oem", covariance_frame="VNC_ROTATING")

   # strict=True keeps to the RSW, RTN, TNW set of table 5-4 for readers that
   # only know that list. VNC_ROTATING is TNW_ROTATING with rows (T, W, -N).
   message.write("states_cov_tnw.oem", covariance_frame="TNW", strict=True)

   # States only, with the covariance as a separate product at full epoch
   # precision in AU and AU/day, for consumers that want it outside the OEM.
   message.write("states.oem")
   LocalFrameCovariances.from_orbits(propagated, frame="VNC_ROTATING").to_parquet(
       "covariance_vnc_rotating.parquet"
   )

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
