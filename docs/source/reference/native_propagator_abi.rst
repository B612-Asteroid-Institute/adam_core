Native propagation-provider ABI
===============================

Scope and ownership
-------------------

``adam-core`` owns orbit determination: IOD candidate policy, least squares,
rejection, ranging, linkage orchestration, settings, result tables and
provenance.  A separately built package such as ``adam-assist`` may contribute
only propagation and derivatives through the private capsule described here.
It must not provide ``run_iod``, ``run_od``, fitting or rejection work units.
Python enters an adam-core workflow once (including a linkage batch);
prediction calls inside that workflow are batched C-ABI calls and never Python
callbacks.

Rust applications do not need this capsule. They compile the generic
adam-core algorithms against a concrete ``Propagator`` implementation (for
example ``AssistPropagator``) directly. The capsule is the binary boundary
between separately built Python extension modules and does not make
adam-core link to a GPL provider.

Discovery and identity
----------------------

A Python propagator declares ``NativeBackend(name, version,
work_units=frozenset({"propagator_capsule"}))``. Its ``propagator_capsule()``
returns a ``PyCapsule`` named exactly ``adam_core.propagator.v1``. Adam-core
retains the capsule for the complete native call and compares the declaration's
name, version and two-body declaration with the immutable identity in the
table. Empty or mismatched identities fail before propagation. Subclasses that override
propagation or the capsule method need a new declaration.

The capsule points to ``ExternalPropagatorV1`` in
``adam_core_rs_coords::propagation``. ``abi_version`` must be 1 and
``struct_size`` must cover the v1 table. Tables and requests may be extended by
appending fields; a v1 consumer ignores a larger size and capability bits it
does not understand. Existing fields must never be reordered or retyped.
Providers must build against the exact published v1 definitions rather than
copying an inferred Rust layout.

Capabilities
------------

``capabilities`` is a bit set:

* ``EXTERNAL_PROPAGATOR_CAP_THREAD_SAFE`` is mandatory. All callbacks and the
  context they reference may be used concurrently from arbitrary native
  threads while the capsule is alive.
* ``EXTERNAL_PROPAGATOR_CAP_STM`` requires a non-null ``propagate_stm``
  callback. Its ``PropagateStmRequestV1`` has independent ABI-version and
  structure-size fields. It returns states and row-major 6-by-6 derivatives
  ``d state(t) / d state(epoch)`` for a batch of target times. The callback
  receives the orbit's fixed non-gravitational row, so the STM follows the
  same dynamics as propagation.
* ``EXTERNAL_PROPAGATOR_CAP_NON_GRAVITATIONAL`` means ``propagate`` accepts the
  per-orbit A1, A2, A3, ALN, NK, NM, NN and R0 array. A NaN means that field is
  null. Adam-core fails closed instead of dropping non-gravitational values
  when this bit is absent.

The current state-fitting API consumes state STMs. Parameter-partial support
must be introduced as an appended callback plus a new capability bit; it must
not be smuggled into the state STM layout.

Propagation request
-------------------

``PropagateRequestV1`` and ``PropagateStmRequestV1`` each carry their own
``abi_version`` and ``struct_size``.
States use au and au/day. Epochs and target times are exact ``(days,
nanoseconds)`` pairs. Time scales, frame and origin are NUL-terminated strings.
Adam-core converts both time arrays to the table's ``integration_time_scale``
before the call. Output states stay in the requested frame and origin.

The callback receives all orbits and target times in one request. For
``pairwise == 0`` output is orbit-major with ``n_orbits * n_times`` rows. For
``pairwise == 1`` orbit *i* maps to time *i* and both lengths must match.
``out_valid == 0`` marks an individual failed row; a successful callback may
therefore return partial row failures without failing the request.

A non-null non-gravitational pointer has
``n_orbits * EXTERNAL_NON_GRAVITATIONAL_WIDTH`` values in the order listed
above. Providers must apply the complete row consistently in propagation and in
any derivatives they advertise. Adam-core carries the fixed row through every
candidate, finite-difference batch, rejection refit and returned
``FittedOrbits`` table; it never silently substitutes gravity-only dynamics.

Lifetime, errors and panics
---------------------------

The provider owns the table, identity strings and context. They remain valid
and immutable until the capsule destructor runs. Adam-core owns every request
and output buffer, which is valid only for the duration of its callback. A
provider must neither retain those pointers nor call Python from a callback.

Callback status values are ``OK=0``, ``ERROR=1``, ``PANIC=2`` and
``INVALID=3``. On ``ERROR`` or ``PANIC`` the provider writes a NUL-terminated
message to the supplied error buffer when one is available. Messages may be
truncated. No Rust panic, C++ exception or other language unwind may cross the
C ABI; providers catch it, report ``PANIC`` and return. Adam-core maps invalid
requests to ``ValueError`` and provider/protocol failures to ``RuntimeError``.
Null required pointers, unsupported capabilities, malformed versions/sizes and
non-finite rows marked valid are treated as invalid rows.

Migration from backend-owned OD methods
---------------------------------------

Provider packages should delete or deprecate Python and PyO3 methods such as
``fit_least_squares``, ``fit_least_squares_evaluated``, ``od_fit``,
``initial_orbit_determination``, ``run_iod_linkages`` and ``run_od_fit``.
Users call the unchanged adam-core APIs and pass the provider as the stage
propagator. There is no bridge package: Python users install ``adam-core`` and
the propagation provider only.
