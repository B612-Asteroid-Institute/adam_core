Non-Gravitational Support
=========================

This note captures the current source-format audit and the design used for
non-gravitational parameter support in ``adam_core``.

Source Matrix
-------------

SBDB
~~~~

- Non-gravitational parameters are provided in ``orbit.model_pars``.
- Observed parameter names in local fixtures include ``A1``, ``A2``, ``A3``,
  ``DT``, ``ALN``, ``NK``, ``NM``, ``R0``, ``AMRAT``, and ``RHO``.
- ``orbit.covariance.labels`` may extend beyond the six orbital elements to
  include solved non-grav parameters.
- Representative local fixtures:
  ``99942_phys.json``, ``67P_phys.json``, ``81P_phys.json``,
  ``101955_phys.json``, ``2022OB5_phys.json``.

NEOCC
~~~~~

- Non-gravitational metadata is provided by ``LSP`` and ``NGR`` records in OEF
  files.
- Local fixtures include both 6D orbital solutions and 7D solutions with a
  Yarkovsky parameter in the non-grav vector.
- Representative local fixtures: ``99942.ke0``, ``99942.ke1``,
  ``101955.ke0``, ``101955.ke1``.
- The local OEF examples document the Yarkovsky parameter in units of
  ``1E-10 au/day^2``.

MPCQ
~~~~

- ``mpcq`` versions with non-grav support expose orbit-side ``a1``, ``a2``,
  and ``a3`` columns on ``MPCOrbits``, and ``MPCOrbits.orbits()`` maps those
  values into ``adam_core.Orbits.non_gravitational_parameters`` using the
  canonical ``A1/A2/A3`` fields.
- This handoff requires a companion ``mpcq`` release; older versions do not
  carry the columns.

Canonical Schema
----------------

``adam_core.orbits.NonGravitationalParameters`` stores only the Marsden-style
radial/transverse/normal accelerations supported for ingestion and storage:

- ``source``: provenance of the solution (for example ``"SBDB"``,
  ``"NEOCC"``, ``"MPCQ"``).
- ``A1``, ``A2``, ``A3`` in au / d^2.

Source parameters outside this set (``DT``, ``AMRAT``, ``RHO``, Marsden
``g(r)`` constants, ...) are not stored: importers drop their values and
marginalize their covariance dimensions out, logging a warning.

Extended Coordinate Covariance
------------------------------

Uncertainties for the non-gravitational parameters live in the coordinate
covariance itself, which is the single source of truth for solution
uncertainty. ``CoordinateCovariances`` rows hold either the familiar 6x6
coordinate covariance or a 9x9 matrix over the fixed basis
``(coordinates, A1, A2, A3)`` — the orbital state plus the non-gravitational
parameters, including their cross-covariances. Parameters that were not
estimated carry zero rows/columns (held fixed); rows without a
non-gravitational solution store only the 6x6 block.

- ``CoordinateCovariances.to_matrix()`` returns the (leading) 6x6 coordinate
  block, so existing consumers are unaffected. ``to_full_matrix()`` returns
  the ``(N, 9, 9)`` matrices with NaN trailing dimensions where no
  non-gravitational block is present, and ``nongrav_block_mask()`` reports
  which rows carry the block.
- ``transform_coordinates`` (and the coordinate classes' ``to_cartesian`` /
  ``from_cartesian``) transform the full matrix alongside representation and
  frame changes: the coordinate block is transformed with the coordinate
  Jacobian while the non-gravitational dimensions (and their
  cross-covariances) are carried through an identity block. No separate
  covariance transform entry point is needed.
- ``VariantOrbits.create`` jointly samples the full 9-dimensional state for
  orbits whose covariance carries the non-gravitational block (sampling is
  done in a whitened basis so the ~1e-26-scale non-grav variances are not
  lost to numerical truncation), and ``collapse``/``collapse_by_object_id``
  rebuild the full matrix from the variants.
- The SBDB and NEOCC importers build the 9x9 covariance in their native
  element basis (cometary and Keplerian respectively), converted to canonical
  units, and the ordinary coordinate transform carries it to Cartesian.

Current limits:

- NEOCC non-grav solutions are decoded only for the Yarkovsky model
  (``AMRAT``/``A2``); the ``AMRAT`` dimension is marginalized out with a
  warning and other models are degraded to value-free rows with a warning.

Two-Body Note
-------------

``propagate_2body`` does not model non-gravitational accelerations.
It rejects orbits that carry non-zero non-gravitational parameter values
instead of silently propagating them with a gravity-only model. For orbits
whose parameters are zero-valued but carry a non-gravitational covariance
block, the covariance is propagated with the 2-body state-transition
Jacobian and the non-gravitational block is carried through as dynamically
inert — consistent with the 2-body force model.

ASSIST Boundary
---------------

``adam_assist`` versions with non-grav support handle propagation when the
usable estimated non-gravitational parameters are limited to Marsden-style
``A1``, ``A2``, and ``A3``. This requires a companion ``adam_assist``
release; ``adam_core`` itself only carries the parameters through its
``Propagator`` interface (propagators that do not declare
``supports_non_gravitational_forces`` log a warning when handed orbits with
non-zero parameter values or a non-gravitational covariance block).

- Real regression coverage includes SBDB ``99942`` end-to-end propagation
  and a NEOCC ``99942`` parsing-to-ASSIST handoff check.
