OEM and SPK I/O
===============

This guide covers standards-based ephemeris export/import:

* OEM (read/write)
* SPK creation for SPICE-native consumers

OEM: Write Pre-Propagated States
--------------------------------

.. code-block:: python

   from adam_core.orbits.oem_io import orbit_to_oem

   output_path = orbit_to_oem(
       orbits=propagated_orbits,
       output_file="apophis.oem",
       originator="ADAM CORE USER",
   )

OEM: Write with Propagation Included
------------------------------------

Use ``orbit_to_oem_propagated`` when you have an initial orbit state and want
OEM generation to include propagation.

.. code-block:: python

   import numpy as np
   from adam_assist import ASSISTPropagator
   from adam_core.orbits.oem_io import orbit_to_oem_propagated
   from adam_core.time import Timestamp

   times = Timestamp.from_mjd(np.arange(60200.0, 60210.0, 1.0), scale="tdb")

   output_path = orbit_to_oem_propagated(
       orbits=seed_orbit,
       output_file="apophis_propagated.oem",
       times=times,
       propagator_klass=ASSISTPropagator,
       originator="ADAM CORE USER",
   )

OEM: Read Back Into Orbits
--------------------------

.. code-block:: python

   from adam_core.orbits.oem_io import orbit_from_oem

   loaded_orbits = orbit_from_oem("apophis_propagated.oem")

SPK: Create Kernel
------------------

``orbits_to_spk`` can propagate internally if you provide a propagator instance.
For realistic mission products, use a high-fidelity propagator (for example,
``adam_assist.ASSISTPropagator``).

.. code-block:: python

   from adam_assist import ASSISTPropagator
   from adam_core.orbits.spice_kernel import orbits_to_spk

   target_id_map = orbits_to_spk(
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

SPK: Read/Validate via SPICE
----------------------------

``adam_core`` includes SPICE helper functions for loading kernels and querying
state vectors. Use raw ``spiceypy`` for lower-level or specialized SPICE calls.

.. code-block:: python

   from adam_core.time import Timestamp
   from adam_core.utils.spice import (
       get_spice_body_state,
       register_spice_kernel,
       unregister_spice_kernel,
   )

   register_spice_kernel("objects.bsp")

   times = Timestamp.from_mjd([60200.0], scale="tdb")
   apophis_state = get_spice_body_state(
       body_id=target_id_map["Apophis"],
       times=times,
       frame="equatorial",
   )

   unregister_spice_kernel("objects.bsp")

Propagator Guidance
-------------------

* OEM pre-propagated: any source of valid state history.
* OEM propagated and SPK creation: prefer ``adam_assist.ASSISTPropagator`` or
  another robust ``Propagator`` implementation.
* If propagation quality matters, avoid relying on sparse single-epoch inputs.

When to Use
-----------

* OEM for CCSDS-style interchange and interoperability.
* SPK for SPICE ecosystems, OpenSpace SpiceTranslation, and mission operations tooling.

Related Reference
-----------------

* :doc:`../reference/orbits`
* :doc:`../reference/propagator`

Input Types
-----------
.. code-block:: python

   # orbit_to_oem(orbits: Orbits, output_file: str, originator: str = "ADAM CORE USER") -> str
   # orbit_to_oem_propagated(orbits: Orbits, output_file: str, times: Timestamp, propagator_klass: type[Propagator], originator: str = "ADAM CORE USER") -> str
   # orbit_from_oem(input_file: str) -> Orbits
   # get_spice_body_state(body_id: int, times: Timestamp, frame: str = "ecliptic", origin: OriginCodes = OriginCodes.SUN) -> CartesianCoordinates
