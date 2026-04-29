Rust Migration Public Compatibility
===================================

This page records the support decision for Python module paths restored during
the Rust migration. Restoring an import path is not the same as making the
Python implementation part of the long-term hot path.

Classification meanings:

- Supported Python API: downstream code may keep using the Python symbol.
- Deprecated/private compatibility shim: retained only to avoid an accidental
  break in this release; scheduled for removal in the next planned breaking
  release.
- Reference-only compatibility helper: retained for parity tests, benchmark
  references, or diagnostics. New production code must not depend on it.

Rust implementation rule:

- Rust code must call Rust functions directly. It must never call back through
  these Python compatibility modules for Kepler, Stumpff, chi, Lagrange, or
  light-time helper math.

Release Note
------------

The following baseline module paths remain importable in this migration branch:
``adam_core.dynamics.aberrations``, ``adam_core.dynamics.barker``,
``adam_core.dynamics.chi``, ``adam_core.dynamics.kepler``,
``adam_core.dynamics.lagrange``, ``adam_core.dynamics.stumpff``, and
``adam_core.coordinates.jacobian``.

The private light-time helpers
``adam_core.dynamics.aberrations._add_light_time`` and
``adam_core.dynamics.aberrations._add_light_time_vmap`` are deprecated/private
compatibility shims. Use
``adam_core.dynamics.aberrations.add_light_time`` at the Python level. Internal
code should use ``adam_core._rust.api.add_light_time_numpy`` or the Rust
``adam_core_rs_coords::add_light_time_row`` /
``adam_core_rs_coords::add_light_time_batch_flat`` functions directly.

``adam_core.coordinates.jacobian.calc_jacobian`` is retained as a reference-only
JAX helper. Production covariance transforms use Rust forward-mode AD and must
not route through this module.

Inventory
---------

.. list-table::
   :header-rows: 1
   :widths: 28 16 28 28

   * - Symbol
     - Classification
     - Python implementation
     - Rust-native path
   * - ``adam_core.dynamics.aberrations._add_light_time``
     - Deprecated/private compatibility shim
     - Thin wrapper over ``adam_core._rust.api.add_light_time_numpy``.
     - ``adam_core_rs_coords::add_light_time_row``.
   * - ``adam_core.dynamics.aberrations._add_light_time_vmap``
     - Deprecated/private compatibility shim
     - Thin wrapper over ``adam_core.dynamics.aberrations.add_light_time``.
     - ``adam_core_rs_coords::add_light_time_batch_flat``.
   * - ``adam_core.dynamics.aberrations.add_light_time``
     - Supported Python API
     - Thin wrapper over ``adam_core._rust.api.add_light_time_numpy``.
     - ``adam_core_rs_coords::add_light_time_row`` and ``adam_core_rs_coords::add_light_time_batch_flat``.
   * - ``adam_core.dynamics.aberrations.add_stellar_aberration``
     - Supported Python API
     - Thin wrapper over ``adam_core._rust.api.add_stellar_aberration_numpy``.
     - ``adam_core_rs_coords::apply_stellar_aberration_row``; fused ephemeris kernels use the same Rust helper internally.
   * - ``adam_core.dynamics.barker.solve_barker``
     - Supported Python API
     - Thin wrapper over ``adam_core._rust.api.solve_barker_numpy``.
     - ``adam_core_rs_coords::solve_barker``.
   * - ``adam_core.dynamics.chi.ChiDiagnostics``
     - Supported Python diagnostic API
     - Frozen dataclass for host-side diagnostic reporting.
     - Not a Rust algorithm dependency.
   * - ``adam_core.dynamics.chi.calc_chi``
     - Supported Python API
     - Thin wrapper over ``adam_core._rust.api.calc_chi_numpy``.
     - ``adam_core_rs_coords::calc_chi`` and ``adam_core_rs_coords::calc_chi_with_init``.
   * - ``adam_core.dynamics.chi.calc_chi_diagnostics``
     - Supported Python diagnostic API
     - Host-side diagnostic wrapper around Rust-backed ``calc_chi``.
     - Production propagation diagnostics compute host metadata directly and do not re-enter this module.
   * - ``adam_core.dynamics.kepler.calc_period``
     - Supported Python API
     - Thin wrapper over ``adam_core._rust.api.calc_period_numpy``.
     - ``adam_core_rs_coords::calc_period``.
   * - ``adam_core.dynamics.kepler.calc_periapsis_distance``
     - Supported Python API
     - Thin wrapper over ``adam_core._rust.api.calc_periapsis_distance_numpy``.
     - ``adam_core_rs_coords::calc_periapsis_distance``.
   * - ``adam_core.dynamics.kepler.calc_apoapsis_distance``
     - Supported Python API
     - Thin wrapper over ``adam_core._rust.api.calc_apoapsis_distance_numpy``.
     - ``adam_core_rs_coords::calc_apoapsis_distance``.
   * - ``adam_core.dynamics.kepler.calc_semi_major_axis``
     - Supported Python API
     - Thin wrapper over ``adam_core._rust.api.calc_semi_major_axis_numpy``.
     - ``adam_core_rs_coords::calc_semi_major_axis``.
   * - ``adam_core.dynamics.kepler.calc_semi_latus_rectum``
     - Supported Python API
     - Thin wrapper over ``adam_core._rust.api.calc_semi_latus_rectum_numpy``.
     - ``adam_core_rs_coords::calc_semi_latus_rectum``.
   * - ``adam_core.dynamics.kepler.calc_mean_motion``
     - Supported Python API
     - Thin wrapper over ``adam_core._rust.api.calc_mean_motion_numpy``.
     - ``adam_core_rs_coords::calc_mean_motion`` and ``adam_core_rs_coords::calc_mean_motion_batch``.
   * - ``adam_core.dynamics.kepler.calc_mean_anomaly``
     - Supported Python API
     - Thin wrapper over ``adam_core._rust.api.calc_mean_anomaly_numpy``.
     - ``adam_core_rs_coords::calc_mean_anomaly``.
   * - ``adam_core.dynamics.kepler.solve_kepler``
     - Supported Python API
     - Thin wrapper over ``adam_core._rust.api.solve_kepler_numpy``.
     - ``adam_core_rs_coords::solve_kepler_true_anomaly``.
   * - ``adam_core.dynamics.lagrange.calc_lagrange_coefficients``
     - Supported Python API
     - Thin wrapper over ``adam_core._rust.api.calc_lagrange_coefficients_numpy``.
     - ``adam_core_rs_coords::calc_lagrange_coefficients``.
   * - ``adam_core.dynamics.lagrange.apply_lagrange_coefficients``
     - Supported Python API
     - Thin wrapper over ``adam_core._rust.api.apply_lagrange_coefficients_numpy``.
     - ``adam_core_rs_coords::apply_lagrange_coefficients``.
   * - ``adam_core.dynamics.stumpff.calc_stumpff``
     - Supported Python API
     - Thin wrapper over ``adam_core._rust.api.calc_stumpff_numpy``.
     - ``adam_core_rs_coords::calc_stumpff``.
   * - ``adam_core.coordinates.jacobian.calc_jacobian``
     - Reference-only compatibility helper
     - JAX ``jacfwd``/``vmap`` helper retained for parity and tests.
     - Production covariance transforms use ``adam_core_rs_autodiff`` through the Rust coordinate kernels.

Guardrails
----------

``src/adam_core/tests/test_public_module_compatibility.py`` checks that every
restored symbol above remains importable and documented. It also statically
rejects production imports of the restored compatibility modules, so future
production code keeps using the Rust-backed entrypoints directly instead of
creating Python-to-Rust-to-Python ping-pong.
