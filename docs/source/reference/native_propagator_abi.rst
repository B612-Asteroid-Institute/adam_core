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
returns a ``PyCapsule`` named exactly ``adam_core.propagator.v2``. Adam-core
retains the capsule for the complete native call and compares the declaration's
name, version and two-body declaration with the immutable identity in the
table. Empty or mismatched identities fail before propagation. Subclasses that override
propagation or the capsule method need a new declaration.

The capsule points to ``ExternalPropagatorV2`` in
``adam_core_rs_coords::propagation``. ``abi_version`` must equal **2** exactly;
there is no major/minor compatibility rule. The distinct v2 capsule name and
version intentionally prevent provisional v1 tables from being interpreted as
the stabilized layout.

Every structure has a frozen required prefix and four zeroed reserved
pointer-sized words. Providers set ``struct_size`` to the bytes they actually
supply. Consumers require ``EXTERNAL_PROPAGATOR_V2_MIN_SIZE`` for the mandatory
table fields and ``EXTERNAL_PROPAGATOR_V2_STM_SIZE`` before reading the optional
STM slot. Requests similarly require ``PROPAGATE_REQUEST_V2_MIN_SIZE`` or
``PROPAGATE_STM_REQUEST_V2_MIN_SIZE``. A larger structure is accepted and the
unknown suffix is ignored; a shorter prefix is rejected before any omitted
field is read. Existing prefix fields must never be reordered or retyped.
ABI v2 is defined for 64-bit extension processes. Its frozen prefix sizes are
``EXTERNAL_PROPAGATOR_V2_MIN_SIZE = 72``,
``EXTERNAL_PROPAGATOR_V2_STM_SIZE = 80``,
``PROPAGATE_REQUEST_V2_MIN_SIZE = 160`` and
``PROPAGATE_STM_REQUEST_V2_MIN_SIZE = 112`` bytes. Providers must compile
against the exact exported v2 definitions and constants, not copy an inferred
Rust layout.

Capabilities
------------

``capabilities`` is a bit set:

* ``EXTERNAL_PROPAGATOR_CAP_THREAD_SAFE`` is mandatory. All callbacks and the
  context they reference may be used concurrently from arbitrary native
  threads while the capsule is alive.
* ``EXTERNAL_PROPAGATOR_CAP_STM`` requires a non-null ``propagate_stm``
  callback. Its ``PropagateStmRequestV2`` has independent ABI-version and
  structure-size fields. It returns states and row-major 6-by-6 derivatives
  ``d state(t) / d state(epoch)`` for a batch of target times. The callback
  receives the orbit's fixed non-gravitational row, so the STM follows the
  same dynamics as propagation.
* ``EXTERNAL_PROPAGATOR_CAP_NON_GRAVITATIONAL`` means ``propagate`` accepts the
  per-orbit A1, A2, A3, ALN, NK, NM, NN and R0 array. A NaN means that field is
  null. A row is dynamically active only when at least one of A1/A2/A3 is
  non-zero, matching adam-core's in-process two-body check. For an active row,
  null Marsden constants mean the asteroid convention used by ASSIST:
  ``ALN=1``, ``NK=0``, ``NM=2``, ``NN=5.093``, ``R0=1 au``. Adam-core fails
  closed instead of dropping active non-gravitational acceleration when this
  bit is absent.

The current state-fitting API consumes state STMs. Parameter-partial support
must be introduced as an appended callback plus a new capability bit; it must
not be smuggled into the state STM layout.

Propagation request
-------------------

``PropagateRequestV2`` and ``PropagateStmRequestV2`` each carry their own
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

A non-null physical-parameter pointer has
``n_orbits * EXTERNAL_PHYSICAL_PARAMETERS_WIDTH`` values in
``H_v, H_v_sigma, G, G_sigma, sigma_eff, chi2_red`` order. It reconstructs the
same optional ``PhysicalParametersBatch`` that an in-process ``Propagator``
receives; these values may affect provider physics and are preserved on output.
A non-null non-gravitational pointer has
``n_orbits * EXTERNAL_NON_GRAVITATIONAL_WIDTH`` values in the order listed
above. Providers must apply the complete active row consistently in propagation
and in any derivatives they advertise. Adam-core carries both typed batches
through every candidate, finite-difference batch, rejection refit and returned
``FittedOrbits`` table.

All non-empty input and output arrays, including the error buffer, must occupy
non-overlapping address ranges and must have the lengths implied by the request.
Providers validate checked products before constructing slices. Input strings
are NUL terminated and immutable for the call. Output buffers are writable only
for the call and must not alias each other or any input.

Lifetime, errors and panics
---------------------------

The provider owns the table, identity strings and context. They remain valid
and immutable until the capsule destructor runs. Adam-core owns every request
and output buffer, which is valid only for the duration of its callback. A
provider must neither retain those pointers nor call Python from a callback.

Callback status values are ``OK=0``, ``ERROR=1``, ``PANIC=2`` and
``INVALID=3`` and are not interchangeable. ``ERROR`` is a provider operation
failure, ``PANIC`` is a caught provider panic/exception, and ``INVALID`` is a
malformed ABI request or protocol violation. On any non-OK status the whole
request fails; adam-core never converts it into candidate-local row invalidity.
The provider writes a NUL-terminated message when the validated prefix includes
an error buffer. Messages may be truncated. No Rust panic, C++ exception or
other language unwind may cross the C ABI.

Only ``OK`` may contain candidate-local failures. In that case ``out_valid=0``
marks a failed row. ABI v2 has no per-row message or provider failure-code
buffer; adam-core records the generic ``IntegratorFailure`` code and no row
message, so providers must not rely on diagnostics that the ABI does not carry.
A row marked valid but containing non-finite state/STM values is also treated as
row-invalid. Unknown statuses and ``INVALID`` map to a fatal protocol error;
``ERROR`` and ``PANIC`` map to fatal backend errors with distinct text.

Migration from backend-owned OD methods
---------------------------------------

Provider packages should delete or deprecate Python and PyO3 methods such as
``fit_least_squares``, ``fit_least_squares_evaluated``, ``od_fit``,
``initial_orbit_determination``, ``run_iod_linkages`` and ``run_od_fit``.
Users call the unchanged adam-core APIs and pass the provider as the stage
propagator. There is no bridge package: Python users install ``adam-core`` and
the propagation provider only.
