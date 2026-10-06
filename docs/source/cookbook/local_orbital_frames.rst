Local Orbital Frames: Covariance in VNC, TNW and RSW
=====================================================

Navigation consumers often want state uncertainties split along the velocity,
across the orbit plane, and in the remaining in-plane direction, rather than
in inertial x, y, z. adam_core keeps state vectors in inertial frames. The
only quantity it expresses in a local orbital frame is the covariance, as a
separate product keyed to the same epochs.

Frame Definitions
-----------------

Names follow the SANA orbit-relative reference frames registry used by the
CCSDS navigation data messages.

* ``RSW``: x along the position vector, z along the orbital angular momentum,
  y completes the right-handed set. Also called RTN and RIC.
* ``TNW``: x along the velocity vector, z along the orbital angular momentum,
  y completes the right-handed set.
* ``VNC``: x along the velocity vector, y along the orbital angular momentum,
  z completes the right-handed set.

VNC and TNW share two axes and differ in the third, so a TNW covariance must
never be labelled VNC.

Each family has two variants. ``_INERTIAL`` treats the frame as fixed at the
epoch, and the covariance is rotated with the same 3x3 rotation on position and
velocity. ``_ROTATING`` carries the frame angular velocity into the velocity
rows, so velocity uncertainties are relative to the rotating axes. The frame
rate is computed from two-body motion about the coordinate origin, with
``mu`` taken from the origin unless given. The bare names ``RSW``, ``RTN``,
``RIC``, ``TNW`` and ``VNC`` resolve to the ``_INERTIAL`` variants, which is
how the CCSDS orbit data messages use them.

Covariance Product
------------------

.. code-block:: python

   from adam_core.coordinates import LocalFrameCovariances

   # propagated: Orbits with covariances, frame "equatorial", origin SUN.
   vnc: LocalFrameCovariances = LocalFrameCovariances.from_orbits(
       propagated, frame="VNC_ROTATING"
   )
   print(vnc.frame, vnc.reference_frame)      # VNC_ROTATING equatorial
   print(vnc.to_matrix().shape)               # (N, 6, 6) in AU and AU/day
   print(vnc.to_matrix_km().shape)            # (N, 6, 6) in km and km/s
   vnc.to_parquet("covariance_vnc_rotating.parquet")

Rows are aligned one to one with the input orbits, in the same order. The
table carries ``orbit_id``, ``object_id``, ``time`` and ``origin`` so a
consumer can match each covariance to its state without the OEM file. Rows
whose source covariance was missing stay NaN.

Rotation Matrices and Jacobians
-------------------------------

The building blocks are available for other uses.

.. code-block:: python

   from adam_core.coordinates.local_orbital_frames import (
       local_frame_angular_velocity,
       local_frame_jacobians,
       local_frame_rotation_matrices,
   )

   coords = propagated.coordinates
   rotation = local_frame_rotation_matrices(coords, "VNC")        # (N, 3, 3)
   omega = local_frame_angular_velocity(coords, "VNC_ROTATING")   # (N, 3) rad/day
   jacobian = local_frame_jacobians(coords, "VNC_ROTATING")       # (N, 6, 6)

``rotation @ x`` maps an inertial vector into the local frame. The Jacobian of
the rotating variant is ``[[R, 0], [-R [omega]x, R]]`` and a covariance in the
local frame is ``J @ covariance @ J.T``.

Related Reference
-----------------

* :doc:`oem_and_spk_io`
* :doc:`coordinate_covariances`
* :doc:`../reference/api/adam_core.coordinates`
