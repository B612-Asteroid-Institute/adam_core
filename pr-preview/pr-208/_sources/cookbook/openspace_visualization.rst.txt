OpenSpace Visualization Assets
==============================

``adam_core`` includes helpers to generate OpenSpace-ready asset files from
``Orbits`` tables.

Simple: RenderableOrbitalKepler
-------------------------------

Best for large numbers of orbits.

.. code-block:: python

   from adam_core.orbits.openspace import create_renderable_orbital_kepler

   create_renderable_orbital_kepler(
       orbits=orbits,
       out_dir="openspace_assets",
       identifier="main_belt_sample",
       gui_name="Main Belt Sample",
       gui_path="/ADAM",
       color=(0.9, 0.9, 1.0),
       segment_quality=10,
   )

This creates an ``.asset`` file and a CSV orbital-elements source file.

Trail Orbits with Kepler Translation
------------------------------------

Best for smaller sets where per-object trail styling is useful.

.. code-block:: python

   from adam_core.orbits.openspace import create_renderable_trail_orbit

   create_renderable_trail_orbit(
       orbits=orbits,
       out_dir="openspace_assets",
       identifier="neo_trails",
       translation_type="Kepler",
       rendering="Lines+Points",
       line_width=2.0,
       trail_head=True,
       gui_path="/ADAM/NEOs",
   )

Trail Orbits with Spice Translation
-----------------------------------

Use this mode when you want OpenSpace to drive positions from an SPK kernel.

.. code-block:: python

   from adam_assist import ASSISTPropagator
   from adam_core.orbits.openspace import create_renderable_trail_orbit
   from adam_core.orbits.spice_kernel import orbits_to_spk

   spice_id_mappings = orbits_to_spk(
       orbits=orbits,
       output_file="openspace_assets/neo_kernel.bsp",
       start_time=start_time,
       end_time=end_time,
       propagator=ASSISTPropagator(),
       step_days=0.25,
       window_days=32.0,
       kernel_type="w03",
   )

   create_renderable_trail_orbit(
       orbits=orbits,
       out_dir="openspace_assets",
       identifier="neo_spice_trails",
       translation_type="Spice",
       spice_kernel_path="openspace_assets/neo_kernel.bsp",
       spice_id_mappings=spice_id_mappings,
       gui_path="/ADAM/NEOs",
   )

Practical Notes
---------------

* Keep identifiers path-safe and stable.
* For Spice translation, SPK kernel + ``spice_id_mappings`` are required.
* For production kernels, use a high-fidelity propagator such as
  ``adam_assist.ASSISTPropagator``.

When to Use
-----------

* ``create_renderable_orbital_kepler``: large catalog visual context.
* ``create_renderable_trail_orbit`` with Kepler: lightweight per-object trails.
* ``create_renderable_trail_orbit`` with Spice: kernel-driven cinematic/ops scenes.

Related Reference
-----------------

* :doc:`../reference/api/adam_core.orbits`
* :doc:`oem_and_spk_io`

