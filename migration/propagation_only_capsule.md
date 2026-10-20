# Migration: propagation-only native providers

This branch replaces the provisional PR #217 backend-owned OD protocol with a
single private propagation-provider capsule.

## Application code

Keep calling adam-core APIs (`run_iod`, `refine`, `run_od`,
`fit_least_squares`, rejection and linkage APIs). Pass an adam-assist
propagator for each stage exactly as before. No bridge package is installed.
Python-only propagators are no longer accepted by native OD because Rust never
calls Python for predictions.

## Provider code

Declare only:

```python
NativeBackend(
    name="adam_assist",
    version=__version__,
    work_units=frozenset({"propagator_capsule"}),
)
```

Remove provider-owned methods and PyO3 exports for fitting, IOD, OD,
rejection, and linkage orchestration. In particular, migrate away from
`fit_least_squares`, `fit_least_squares_evaluated`, `od_fit`,
`initial_orbit_determination`, `run_iod_linkages`, and `run_od_fit`. Implement
the 64-bit v2 table/request layout (the v1 prototype is intentionally
incompatible) and lifecycle contract in
`docs/source/reference/native_propagator_abi.rst`. ``NativeBackend`` rejects
algorithm work units at construction time. Consumers also reject unknown
capability bits and non-zero advertised reserved words. A successful
propagation request may report row-local failures through ``out_valid`` and the
optional stable ``out_failure_codes`` buffer; malformed requests, callback
errors, panics and protocol errors are fatal to the entire workflow (including
a linkage batch).

The provider must supply batched propagation, the full optional physical
parameter batch, complete active non-gravitational inputs when it advertises
that capability, and dynamics-consistent STMs when
it advertises STM support. The STM callback receives the same fixed A1/A2/A3
and Marsden-law row as state propagation. Its identity must match `NativeBackend`; callbacks
must be thread-safe, must not retain request buffers or invoke Python, and
must catch all panics/exceptions before returning across the C ABI.

## Rust-only code

No capsule is needed. Continue compiling adam-core's generic algorithms with a
concrete `Propagator`, such as `AssistPropagator`. This path keeps all
algorithm policy in adam-core and all provider physics in adam-assist without
linking adam-core itself to GPL code.

## Result schema and IOD policy

``FittedOrbits`` gains nullable ``physical_parameters`` and
``non_gravitational_parameters`` columns. Readers of older serialized tables
receive null columns; code that compares exact Arrow schemas must update its
expected schema. Propagation providers receive the physical batch and every
active non-gravitational row, and adam-core preserves both on fitted outputs.

``acceptance="prefit"`` is the explicit legacy IOD policy: raw Gauss candidates
are considered in prefit-score order and must pass the configured outlier trial
and whole-arc threshold. The default ``run_iod`` policy remains
``acceptance="corrected"``; deprecated ``iod`` requests ``"prefit"``.
