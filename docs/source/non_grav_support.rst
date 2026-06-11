Non-Gravitational Support
=========================

This note captures the current source-format audit and the MVP design used for
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

Canonical MVP Schema
--------------------

``adam_core.orbits.NonGravitationalParameters`` stores:

- Source and coarse model metadata.
- Larger-state metadata such as ``solution_dimension`` and
  ``estimated_parameter_names``.
- Canonical scalar parameters and their sigmas when available:
  ``A1``, ``A2``, ``A3``, ``DT``, ``R0``, ``ALN``, ``NK``, ``NM``, ``NN``,
  ``AMRAT``, ``RHO``.

Canonical units
---------------

- ``A1``, ``A2``, ``A3``: au / d^2
- ``DT``: d
- ``R0``: au
- ``AMRAT``: m^2 / kg
- ``RHO``: kg / m^3

Solved-State Covariances
------------------------

In addition to the 6x6 coordinate covariance, orbits can carry the full
``(6 + k) x (6 + k)`` solved-state covariance — the orbital state plus the
fitted non-gravitational parameters, including their cross-covariances:

- ``adam_core.orbits.SolvedStateCovariances`` stores the full matrices along
  with the per-row parameter names (the first six entries are always the
  orbital basis).
- ``transform_coordinates(..., solved_state_covariances=...)`` transforms the
  full matrix alongside representation and frame changes: the orbital block is
  transformed with the coordinate Jacobian while the extra parameter
  dimensions (and their cross-covariances) are preserved through an identity
  block. ``Orbits.solved_state_covariance_to`` is the orbit-level convenience
  wrapper.
- ``VariantOrbits.create`` jointly samples the full ``6 + k`` state for orbits
  that carry a solved-state covariance (sampling is done in a whitened basis
  so the ~1e-26-scale non-grav variances are not lost to numerical
  truncation), and ``collapse``/``collapse_by_object_id`` rebuild the full
  matrix from the variants.
- The SBDB and NEOCC importers populate the column from the full fitted
  covariance, converted to canonical units.

Current limits:

- Propagated outputs do not carry an epoch-updated solved-state covariance:
  the matrix is epoch-specific and propagating the full ``6 + k`` covariance
  is not yet implemented, so ``propagate_2body`` nulls the column on its
  output while preserving the (time-invariant) non-grav parameter values.
- The 6x6 ``coordinates.covariance`` and the leading block of the solved-state
  covariance are stored separately; importers keep them consistent at
  ingestion, but downstream code that modifies one is responsible for the
  other.
- NEOCC solved states are decoded only for the Yarkovsky model
  (``AMRAT``/``A2``); other models are preserved as metadata with a warning.

Two-Body Note
-------------

``propagate_2body`` does not model non-gravitational accelerations.
It now rejects orbits that carry non-gravitational parameter values instead of
silently propagating them with a gravity-only model.

ASSIST Boundary
---------------

``adam_assist`` versions with non-grav support handle propagation when the
usable estimated non-gravitational parameters are limited to Marsden-style
``A1``, ``A2``, and ``A3``. This requires a companion ``adam_assist``
release; ``adam_core`` itself only carries the parameters through its
``Propagator`` interface (propagators that do not declare
``supports_non_gravitational_forces`` log a warning when handed non-grav
orbits).

- Fixed metadata that accompanies those models, such as SBDB Marsden constants
  (for example ``R0``, ``ALN``, ``NK``, ``NM``) or a non-estimated
  ``AMRAT = 0`` from NEOCC Yarkovsky records, is preserved and ignored by the
  ASSIST handoff.
- Solved models that require unsupported estimated parameters, such as ``DT``,
  ``AMRAT``, or ``RHO``, are rejected with a clear error.
- Real regression coverage now includes SBDB ``99942`` end-to-end propagation
  and a NEOCC ``99942`` parsing-to-ASSIST handoff check.
