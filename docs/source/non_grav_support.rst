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

- The installed ``mpcq`` package exposes orbit-side ``a1``, ``a2``, and ``a3``
  columns on ``MPCOrbits``.
- ``MPCOrbits.orbits()`` now maps those values into
  ``adam_core.Orbits.non_gravitational_parameters`` using the canonical
  ``A1/A2/A3`` fields.

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

Current Covariance Boundary
---------------------------

The current covariance machinery in ``adam_core`` remains 6D and tied to the
orbital state vector only.

For the MVP:

- Larger solved-state dimensions are preserved as metadata.
- Source importers continue to use the leading orbital 6x6 block for
  coordinate covariance.
- Cross-covariances between orbital and non-grav parameters are not yet
  propagated through coordinate transforms or variant collapse.

Two-Body Note
-------------

``propagate_2body`` does not model non-gravitational accelerations.
It now rejects orbits that carry non-gravitational parameter values instead of
silently propagating them with a gravity-only model.

ASSIST Boundary
---------------

``adam_assist`` currently supports propagation when the usable estimated
non-gravitational parameters are limited to Marsden-style ``A1``, ``A2``,
and ``A3``.

- Fixed metadata that accompanies those models, such as SBDB Marsden constants
  (for example ``R0``, ``ALN``, ``NK``, ``NM``) or a non-estimated
  ``AMRAT = 0`` from NEOCC Yarkovsky records, is preserved and ignored by the
  ASSIST handoff.
- Solved models that require unsupported estimated parameters, such as ``DT``,
  ``AMRAT``, or ``RHO``, are rejected with a clear error.
- Real regression coverage now includes SBDB ``99942`` end-to-end propagation
  and a NEOCC ``99942`` parsing-to-ASSIST handoff check.
