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
the table/request layout and lifecycle contract in
`docs/source/reference/native_propagator_abi.rst`.

The provider must supply batched propagation, complete non-gravitational
inputs when it advertises that capability, and dynamics-consistent STMs when
it advertises STM support. The STM callback receives the same fixed A1/A2/A3
and Marsden-law row as state propagation. Its identity must match `NativeBackend`; callbacks
must be thread-safe, must not retain request buffers or invoke Python, and
must catch all panics/exceptions before returning across the C ABI.

## Rust-only code

No capsule is needed. Continue compiling adam-core's generic algorithms with a
concrete `Propagator`, such as `AssistPropagator`. This path keeps all
algorithm policy in adam-core and all provider physics in adam-assist without
linking adam-core itself to GPL code.
