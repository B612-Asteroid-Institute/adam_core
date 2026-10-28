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
work_units=frozenset({"propagator_capsule"}))``. ``version`` is the canonical
Cargo SemVer exported by that provider's native extension (for example
``provider_native.propagator_semver()``), not the Python distribution's PEP
440-normalized ``__version__``. Thus a native ``1.2.3-rc.4`` remains exactly
that spelling even when Python package metadata says ``1.2.3rc4``. Its
``propagator_capsule()`` returns a ``PyCapsule`` named exactly
``adam_core.propagator.v2``. Adam-core retains the capsule for the complete
native call and compares the declaration's name, native SemVer and two-body
declaration with the immutable identity in the table. Constructing ``NativeBackend`` with an algorithm work unit (rather than
only ``propagator_capsule``), or with an empty identity field, raises
``ValueError``. Empty or mismatched capsule identities fail before propagation.
Subclasses that override propagation or the capsule method need a new
declaration.

The capsule points to ``ExternalPropagatorV2`` in
``adam_core_rs_coords::propagation``. ``abi_version`` must equal **2** exactly;
there is no major/minor compatibility rule. The distinct v2 capsule name and
version intentionally prevent provisional v1 tables from being interpreted as
the stabilized layout.

Every structure has a frozen required prefix and four reserved pointer-sized
words. Providers set ``struct_size`` to the bytes they actually supply. Every
reserved word completely covered by that advertised size must be zero; a
non-zero word is a protocol error. Consumers require
``EXTERNAL_PROPAGATOR_V2_MIN_SIZE`` for the mandatory
table fields and ``EXTERNAL_PROPAGATOR_V2_STM_SIZE`` before reading the optional
STM slot. Requests similarly require ``PROPAGATE_REQUEST_V2_MIN_SIZE`` or
``PROPAGATE_STM_REQUEST_V2_MIN_SIZE``. A larger structure is accepted and the
unknown suffix is ignored; a shorter prefix is rejected before any omitted
field is read. Existing prefix fields must never be reordered or retyped.
ABI v2 uses the process target's C pointer/``size_t`` layout; provider and
consumer necessarily share that target because their extensions are loaded in
one process. The exported prefix constants are computed as the exact end of the
last required field. Compile-time assertions prove that each copied mandatory
``repr(C)`` prefix ends there without unread tail padding; after its field-end
size check the optional STM callback is read as one aligned field, never as a
padded whole-prefix structure. On 64-bit targets the frozen values are
``EXTERNAL_PROPAGATOR_V2_MIN_SIZE = 72``,
``EXTERNAL_PROPAGATOR_V2_STM_SIZE = 80``,
``EXTERNAL_PROPAGATOR_V2_LEGACY_FULL_SIZE = 112``,
``EXTERNAL_PROPAGATOR_V2_METADATA_SIZE = 128``,
``PROPAGATE_REQUEST_V2_MIN_SIZE = 168`` and
``PROPAGATE_STM_REQUEST_V2_MIN_SIZE = 112`` bytes. Generic adam-core Rust APIs
remain available on every supported target. Providers must compile against the
exact exported v2 definitions and target-specific constants, not copy an
inferred layout.

Reproducibility metadata suffix
-------------------------------

The frozen table prefix and four reserved words retain their original offsets.
A complete legacy table ends at ``EXTERNAL_PROPAGATOR_V2_LEGACY_FULL_SIZE``
(112 bytes on 64-bit targets). The append-only suffix adds two immutable
NUL-terminated lowercase 64-character SHA-256 strings,
``configuration_fingerprint`` and ``data_fingerprint``; a complete suffix ends
at ``EXTERNAL_PROPAGATOR_V2_METADATA_SIZE`` (128 bytes on 64-bit targets).
Sizes between those boundaries are rejected because they advertise only part
of an atomic metadata contract. General propagation remains compatible with a
short or legacy v2 table, but native OD fails clearly unless both hashes are
present. The provider hashes every dynamics-affecting configuration input and
all external ephemeris/model data. Adam-core decodes and copies this metadata
from the exact retained capsule used by each IOD/refinement stage before
releasing the GIL; provenance never asks Python for a second capsule afterward.
Unknown fields appended after the complete suffix remain ignored.

``adam-assist`` adaptation contract
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``ASSISTPropagator`` remains a propagation-only capsule provider; Core does
not add an OD callback or import ASSIST. Its provider table must append both
fields after the frozen reserved words, retain stable NUL-terminated storage
for the capsule lifetime, and advertise at least
``EXTERNAL_PROPAGATOR_V2_METADATA_SIZE``. The configuration hash covers every
setting that can change propagated states (including integration and force
model choices); the data hash covers the exact immutable ephemeris/kernel and
other dynamics data content. An older adam-assist table remains usable for
general propagation but is intentionally rejected by native OD as
provenance-incomplete. This is the only required downstream ABI adaptation.

Capabilities
------------

``capabilities`` is a bit set. Unknown bits advertise optional future
provider support and are ignored by older consumers. Known bits retain their
specified fail-closed validation, so a provider cannot omit semantics required
by the current request:

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

The current state-fitting API consumes state STMs. The v2 STM request carries
the fixed non-gravitational row but **does not carry physical parameters**.
STM capability is table-wide, not conditional on a request: a provider whose
supported propagation model permits physical fields to affect its state
variational equations must not advertise v2 STM support at all. Adam-core must
use central finite differences through full propagation instead.
Parameter-partial or physical-parameter-aware STM support requires an appended
request/callback plus a new known capability bit; it must not be smuggled into
the v2 state STM layout.

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
``out_failure_codes`` is an optional caller-owned byte buffer with one entry per
output row. When it is non-null, providers write 0 for every valid row and one
of these stable values for every invalid row: 1 ``NonFiniteInputState``, 2
``NonFiniteOutputState``, 3 ``NonFiniteCovariance``, 4
``SolverZeroDerivative``, 5 ``SolverMaxIterations`` or 6
``IntegratorFailure``. Unknown codes, non-zero codes on valid rows, and zero
codes on invalid rows are fatal protocol errors. When the pointer is null,
validity remains available but the caller has elected not to receive a code.

A non-null physical-parameter pointer has
``n_orbits * EXTERNAL_PHYSICAL_PARAMETERS_WIDTH`` values in
``H_v, H_v_sigma, G, G_sigma, sigma_eff, chi2_red`` order. It reconstructs the
same optional ``PhysicalParametersBatch`` that an in-process ``Propagator``
receives; these values may affect provider physics and are preserved on output.
A non-null non-gravitational pointer has
``n_orbits * EXTERNAL_NON_GRAVITATIONAL_WIDTH`` values in the order listed
above. Before encoding nulls as NaN, the adam-core consumer rejects every
present non-finite physical or non-gravitational value; ``Some(NaN)`` can never
silently become null. A provider decodes NaN as the specified null sentinel but
must reject every other non-finite value (positive or negative infinity) as
``INVALID`` before constructing typed batches or invoking either propagation
callback. Providers must apply the complete active non-gravitational row
consistently in propagation and in any derivatives they advertise. Adam-core
carries both typed batches through every candidate, finite-difference batch,
rejection refit and returned ``FittedOrbits`` table.

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
The provider writes a NUL-terminated message when the validated v2 prefix
includes an error buffer. Messages may be truncated. An ABI-version mismatch
must not write through v2-only error offsets because only the two-field header
has been established. No Rust panic, C++ exception or other language unwind may
cross the C ABI.

Only ``OK`` may contain candidate-local failures. In that case
``out_valid=0`` marks a failed row and the optional stable per-row code preserves
its ``PropagationFailureCode``; ABI v2 has no per-row message. Adam-core
classifies a state row marked valid but containing non-finite output as
``NonFiniteOutputState``. A non-finite state/STM row is never accepted as valid.
Unknown statuses and ``INVALID`` map to a fatal protocol error; ``ERROR`` and
``PANIC`` map to fatal backend errors with distinct text. The same distinction
applies to linkage batches: row-local failures may reject one candidate, while
a malformed request, callback error, panic or protocol violation aborts the
whole core-owned batch.

Migration from backend-owned OD methods
---------------------------------------

Provider packages should delete or deprecate Python and PyO3 methods such as
``fit_least_squares``, ``fit_least_squares_evaluated``, ``od_fit``,
``initial_orbit_determination``, ``run_iod_linkages`` and ``run_od_fit``.
The Core 0.7 Rust foundation accepts an explicit complete ``OdConfig`` in its
predictor-level configured algorithms and passes the provider only as the
stage propagator. Canonical Python batch workflows are forthcoming; providers must not
publish backend-owned OD compatibility routes in the meantime. There is no
bridge package.
