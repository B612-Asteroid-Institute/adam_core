Lambert Solutions, Transfers, and Porkchop Plots
=================================================

This guide separates transfer analysis into three layers:

1. solve Lambert solutions
2. generate porkchop data over time grids
3. visualize and inspect porkchop trade spaces

Lambert Solutions (Atomic)
--------------------------

Use ``solve_lambert`` when you already have paired departure/arrival states and
time-of-flight values.

.. code-block:: python

   import numpy as np
   from adam_core.dynamics.lambert import solve_lambert, calculate_c3

   # Two transfer opportunities (AU, AU/day, days)
   r1: np.ndarray = np.array([[1.0, 0.0, 0.0], [0.95, 0.1, 0.0]])
   r2: np.ndarray = np.array([[1.5, 0.2, 0.0], [1.4, 0.35, 0.0]])
   tof: np.ndarray = np.array([180.0, 230.0])

   v1, v2 = solve_lambert(r1, r2, tof, prograde=True, max_iter=35, tol=1e-10)

   # If body velocity at departure is known, pass it here for C3.
   body_v_departure: np.ndarray = np.zeros_like(v1)
   c3 = calculate_c3(v1, body_v_departure)

Generate Porkchop Data
----------------------

For planetary transfers, first generate departure/arrival orbit grids, then call
``generate_porkchop_data``.

.. code-block:: python

   from adam_core.coordinates.origin import OriginCodes
   from adam_core.missions.porkchop import (
       LambertSolutions,
       generate_porkchop_data,
       prepare_and_propagate_orbits,
   )
   from adam_core.orbits import Orbits
   from adam_core.time import Timestamp

   departure_start: Timestamp = Timestamp.from_iso8601(["2028-01-01T00:00:00"], scale="tdb")
   departure_end: Timestamp = Timestamp.from_iso8601(["2028-12-31T00:00:00"], scale="tdb")
   arrival_start: Timestamp = Timestamp.from_iso8601(["2028-06-01T00:00:00"], scale="tdb")
   arrival_end: Timestamp = Timestamp.from_iso8601(["2030-01-01T00:00:00"], scale="tdb")

   departure_orbits: Orbits = prepare_and_propagate_orbits(
       body=OriginCodes.EARTH,
       start_time=departure_start,
       end_time=departure_end,
       propagation_origin=OriginCodes.SUN,
       step_size=2.0,
   )

   arrival_orbits: Orbits = prepare_and_propagate_orbits(
       body=OriginCodes.MARS_BARYCENTER,
       start_time=arrival_start,
       end_time=arrival_end,
       propagation_origin=OriginCodes.SUN,
       step_size=2.0,
   )

   lambert_solutions: LambertSolutions = generate_porkchop_data(
       departure_orbits=departure_orbits,
       arrival_orbits=arrival_orbits,
       propagation_origin=OriginCodes.SUN,
       prograde=True,
       max_processes=8,
   )

   c3_departure = lambert_solutions.c3_departure()
   vinf_arrival = lambert_solutions.vinf_arrival()
   tof_days = lambert_solutions.time_of_flight()

Creating and Viewing Porkchops
------------------------------

.. code-block:: python

   import plotly.graph_objects as go
   from adam_core.missions.porkchop import plot_porkchop_plotly

   fig: go.Figure = plot_porkchop_plotly(
       lambert_solutions,
       title="Earth to Mars Porkchop",
       c3_departure_min=0.0,
       c3_departure_max=80.0,
       vinf_arrival_min=0.0,
       vinf_arrival_max=20.0,
       tof_min=60.0,
       tof_max=500.0,
       show_optimal=True,
       show_hover=True,
   )

   fig.show()
   # fig.write_html("earth_mars_porkchop.html")

Propagator Guidance
-------------------

When using ``prepare_and_propagate_orbits`` with ``body`` as an ``Orbits`` table
(instead of an ``OriginCodes`` major body), you should provide a propagator class.
For production-quality trajectories, use a high-fidelity propagator such as
``adam_assist.ASSISTPropagator``.

.. code-block:: python

   from adam_assist import ASSISTPropagator

   propagated_custom: Orbits = prepare_and_propagate_orbits(
       body=custom_orbit,
       start_time=departure_start,
       end_time=departure_end,
       propagation_origin=OriginCodes.SUN,
       step_size=1.0,
       propagator_class=ASSISTPropagator,
       max_processes=8,
   )

When to Use
-----------

* ``solve_lambert``: targeted transfer solutions.
* ``generate_porkchop_data``: window-wide trade studies.
* ``plot_porkchop_plotly``: mission design communication and decision support.

Related Reference
-----------------

* :doc:`../reference/api/adam_core.missions`
* :doc:`../reference/api/adam_core.dynamics`
