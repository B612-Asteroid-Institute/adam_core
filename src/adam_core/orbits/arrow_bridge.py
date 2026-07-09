"""Arrow bridge: quivr ``Orbits`` <-> Rust-canonical ``OrbitBatch``.

W1 data-model keystone (Beads personal-cmy.13). Carries the *complete* nested
``Orbits`` schema (coordinates incl. covariance, time, origin, physical
parameters) to/from the Rust data model in a single Python<->Rust crossing, and
exposes Rust-native workflows that run entirely on the ``OrbitBatch``.

Two transports share the same Rust workflow code:

* **IPC bytes** (``*_ipc`` helpers): schema-faithful, one serialize/deserialize
  copy; works on any supported pyo3/pyarrow.
* **Arrow C Data Interface** (``*_record_batch`` helpers): zero-copy buffer
  sharing via pyarrow ``RecordBatch`` <-> arrow-rs.

quivr stores ``frame`` and the timestamp ``scale`` as ``StringAttribute`` values
in Arrow schema metadata (``coordinates.frame`` / ``coordinates.time.scale``);
the Rust codec reads canonical ``adam_core_*`` keys. The metadata translation
lives here so callers exchange ordinary ``Orbits``/``VariantOrbits`` objects.
"""

from __future__ import annotations

import pyarrow as pa

from adam_core import _rust_native as _rn
from adam_core._rust.arrow import stamp_adam_core_metadata as _stamp_adam_core_metadata
from adam_core._rust.arrow import to_quivr_metadata as _to_quivr_metadata
from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_core.orbits.variants import VariantOrbits

_NESTED_SCHEMA = "OrbitBatch.cartesian.nested.quivr.v1"
_VARIANT_NESTED_SCHEMA = "OrbitVariantBatch.cartesian.nested.quivr.v1"
_OBSERVER_SCHEMA = "ObserverBatch.cartesian.nested.quivr.v1"


def _with_adam_core_metadata(orbits: Orbits) -> pa.Table:
    """Combine chunks and stamp the canonical ``adam_core_*`` schema metadata."""
    coordinates = orbits.coordinates
    return _stamp_adam_core_metadata(
        orbits.table.combine_chunks(),
        "cartesian",
        coordinates.frame,
        coordinates.time.scale,
        _NESTED_SCHEMA,
    )


# --- IPC-bytes transport -------------------------------------------------------


def orbits_to_ipc(orbits: Orbits) -> bytes:
    """Serialize ``Orbits`` to Arrow IPC bytes with the metadata Rust needs."""
    table = _with_adam_core_metadata(orbits)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def variants_to_ipc(variants: VariantOrbits) -> bytes:
    """Serialize ``VariantOrbits`` to Arrow IPC bytes with the metadata Rust needs."""
    coordinates = variants.coordinates
    table = _stamp_adam_core_metadata(
        variants.table.combine_chunks(),
        "cartesian",
        coordinates.frame,
        coordinates.time.scale,
        _VARIANT_NESTED_SCHEMA,
    )
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _read_ipc(raw: bytes) -> pa.Table:
    with pa.ipc.open_stream(pa.py_buffer(raw)) as reader:
        return reader.read_all().combine_chunks()


def orbits_from_ipc(raw: bytes) -> Orbits:
    """Reconstruct ``Orbits`` from Rust-produced Arrow IPC bytes."""
    return Orbits.from_pyarrow(_to_quivr_metadata(_read_ipc(raw)))


def variants_from_ipc(raw: bytes) -> VariantOrbits:
    """Reconstruct ``VariantOrbits`` from Rust-produced Arrow IPC bytes."""
    return VariantOrbits.from_pyarrow(_to_quivr_metadata(_read_ipc(raw)))


def observers_to_ipc(observers: Observers) -> bytes:
    """Serialize ``Observers`` to Arrow IPC bytes with the metadata Rust needs."""
    coordinates = observers.coordinates
    table = _stamp_adam_core_metadata(
        observers.table.combine_chunks(),
        "cartesian",
        coordinates.frame,
        coordinates.time.scale,
        _OBSERVER_SCHEMA,
    )
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def observers_from_ipc(raw: bytes) -> Observers:
    """Reconstruct ``Observers`` from Rust-produced Arrow IPC bytes."""
    return Observers.from_pyarrow(_to_quivr_metadata(_read_ipc(raw)))


def coordinates_to_ipc(coordinates, representation: str) -> bytes:
    """Serialize a coordinate table (``cartesian`` or ``spherical``) to Arrow IPC."""
    table = _stamp_adam_core_metadata(
        coordinates.table.combine_chunks(),
        representation,
        coordinates.frame,
        coordinates.time.scale,
        "CoordinateBatch.nested.quivr.v1",
    )
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


# --- Arrow C Data Interface (zero-copy) transport ------------------------------


def orbits_to_record_batch(orbits: Orbits) -> pa.RecordBatch:
    """Materialize ``Orbits`` as a single pyarrow ``RecordBatch`` for zero-copy hand-off."""
    table = _with_adam_core_metadata(orbits)
    arrays = [column.combine_chunks() for column in table.columns]
    return pa.RecordBatch.from_arrays(arrays, schema=table.schema)


def orbits_from_record_batch(record_batch: pa.RecordBatch) -> Orbits:
    """Reconstruct ``Orbits`` from a Rust-produced pyarrow ``RecordBatch``."""
    table = pa.Table.from_batches([record_batch])
    return Orbits.from_pyarrow(_to_quivr_metadata(table))


# --- Workflows -----------------------------------------------------------------


def round_trip_orbits(orbits: Orbits) -> Orbits:
    """Decode to the Rust ``OrbitBatch`` and back via IPC (identity bridge check)."""
    return orbits_from_ipc(_rn.orbits_nested_ipc_round_trip(orbits_to_ipc(orbits)))


def round_trip_observers(observers: Observers) -> Observers:
    """Decode to the Rust ``ObserverBatch`` and back via IPC (transport check)."""
    return observers_from_ipc(
        _rn.observers_nested_ipc_round_trip(observers_to_ipc(observers))
    )


def _evaluate_residuals_2body_ipc_candidate(
    orbits: Orbits,
    observed_coordinates,
    observers: Observers,
    lt_tol: float = 1e-10,
    max_iter: int = 1000,
    tol: float = 1e-15,
    stellar_aberration: bool = False,
    max_lt_iter: int = 10,
):
    """Diagnostic Arrow-IPC candidate for the OD residual inner loop.

    Private per the bridge-naming policy (bead personal-cmy.13.1.4): the
    canonical public surfaces are ``coordinates.residuals.Residuals.calculate``
    and ``orbit_determination.evaluate_orbits``; this workflow is their fused
    2-body transport experiment, kept for the parity/speed harness.

    ``orbits`` must already be at the observation times (1:1 with the observed
    astrometry and observers). Composes the same 2-body ephemeris + residual
    kernels as adam_core's ``generate_ephemeris_2body`` + ``Residuals.calculate``,
    including its barycentric (SSB / ecliptic) light-time convention: orbits and
    observers are transformed to the solar-system barycenter first (identity when
    already barycentric). Returns ``(chi2 (N,), residuals (N, 6))`` numpy arrays.
    """
    from adam_core.coordinates import CartesianCoordinates, transform_coordinates
    from adam_core.coordinates.origin import OriginCodes

    orbits = orbits.set_column(
        "coordinates",
        transform_coordinates(
            orbits.coordinates,
            CartesianCoordinates,
            frame_out="ecliptic",
            origin_out=OriginCodes.SOLAR_SYSTEM_BARYCENTER,
        ),
    )
    observers = observers.set_column(
        "coordinates",
        transform_coordinates(
            observers.coordinates,
            CartesianCoordinates,
            frame_out="ecliptic",
            origin_out=OriginCodes.SOLAR_SYSTEM_BARYCENTER,
        ),
    )
    return _rn.evaluate_residuals_2body_ipc(
        orbits_to_ipc(orbits),
        coordinates_to_ipc(observed_coordinates, "spherical"),
        observers_to_ipc(observers),
        lt_tol,
        max_iter,
        tol,
        stellar_aberration,
        max_lt_iter,
    )


def _fit_orbit_least_squares_2body_candidate(
    orbit: Orbits,
    observed_coordinates,
    observers: Observers,
    xtol: float = 1e-12,
    ftol: float = 1e-12,
    max_iterations: int = 100,
    lt_tol: float = 1e-10,
    eph_max_iter: int = 1000,
    eph_tol: float = 1e-15,
    stellar_aberration: bool = False,
    max_lt_iter: int = 10,
):
    """Rust-native Gauss-Newton least-squares orbit determination over the bridge.

    Differentially corrects a single ``orbit`` (at its epoch) against astrometric
    observations, reusing the slice-3 residual evaluation as the inner loop. Like
    adam_core's ``generate_ephemeris_2body``, inputs are transformed to the
    barycentric (SSB / ecliptic) frame first. Returns
    ``(fitted_orbit, chi2, iterations, converged)`` where ``fitted_orbit`` is an
    ``Orbits`` (SSB / ecliptic) carrying the ``inv(JᵀJ)`` parameter covariance.
    """
    from adam_core.coordinates import (
        CartesianCoordinates,
        CoordinateCovariances,
        Origin,
        transform_coordinates,
    )
    from adam_core.coordinates.origin import OriginCodes

    orbit = orbit.set_column(
        "coordinates",
        transform_coordinates(
            orbit.coordinates,
            CartesianCoordinates,
            frame_out="ecliptic",
            origin_out=OriginCodes.SOLAR_SYSTEM_BARYCENTER,
        ),
    )
    observers = observers.set_column(
        "coordinates",
        transform_coordinates(
            observers.coordinates,
            CartesianCoordinates,
            frame_out="ecliptic",
            origin_out=OriginCodes.SOLAR_SYSTEM_BARYCENTER,
        ),
    )
    state, covariance, chi2, iterations, converged = (
        _rn.fit_orbit_2body_least_squares_ipc(
            orbits_to_ipc(orbit),
            coordinates_to_ipc(observed_coordinates, "spherical"),
            observers_to_ipc(observers),
            xtol,
            ftol,
            max_iterations,
            lt_tol,
            eph_max_iter,
            eph_tol,
            stellar_aberration,
            max_lt_iter,
        )
    )
    fitted = Orbits.from_kwargs(
        orbit_id=orbit.orbit_id,
        object_id=orbit.object_id,
        coordinates=CartesianCoordinates.from_kwargs(
            x=[state[0]],
            y=[state[1]],
            z=[state[2]],
            vx=[state[3]],
            vy=[state[4]],
            vz=[state[5]],
            time=orbit.coordinates.time,
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SOLAR_SYSTEM_BARYCENTER"]),
            covariance=CoordinateCovariances.from_matrix(covariance.reshape(1, 6, 6)),
        ),
    )
    return fitted, chi2, iterations, converged


def round_trip_orbits_zero_copy(orbits: Orbits) -> Orbits:
    """Identity round-trip via the zero-copy Arrow C Data Interface transport."""
    out = _rn.orbits_nested_round_trip_arrow(orbits_to_record_batch(orbits))
    return orbits_from_record_batch(out)


def _rotate_orbits_frame_ipc_candidate(orbits: Orbits, frame: str) -> Orbits:
    """Diagnostic Arrow-IPC candidate for ``transform_coordinates(..., frame_out=...)``.

    This is intentionally private: the canonical public API for frame changes is
    ``adam_core.coordinates.transform_coordinates``. The parity/speed harness
    may still time this Orbits-level workflow as a backend-candidate experiment.
    """
    return orbits_from_ipc(_rn.orbits_rotate_frame_ipc(orbits_to_ipc(orbits), frame))


def _sample_orbit_variants_ipc_candidate(
    orbits: Orbits,
    method: str = "sigma-point",
    num_samples: int = 10000,
    seed: int | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    kappa: float = 0.0,
) -> VariantOrbits:
    """Private Rust Arrow-IPC backend for ``VariantOrbits.create``.

    All three methods (``sigma-point``, ``auto``, ``monte-carlo``) run
    Rust-side in one crossing. Monte Carlo draws (including the auto-mode
    fallback) use the Rust-native RNG: statistically equivalent to, but not
    bit-identical with, the legacy scipy path (decision 2026-07-03: exact
    scipy RNG parity is not required). Per-orbit physical parameters are
    carried by the canonical Rust variant schema itself (bead
    personal-cmy.13.2): the sampler gathers them per source orbit Rust-side
    and they arrive in the IPC payload, so no Python reattachment is needed
    even for variable-count auto-mode outputs.
    """
    raw, _source_indices = _rn.orbits_sample_variants_ipc(
        orbits_to_ipc(orbits), method, num_samples, seed, alpha, beta, kappa
    )
    return variants_from_ipc(raw)


def _propagate_orbits_typed_ipc_candidate(
    orbits,
    times,
    covariance: bool = False,
    max_iter: int = 1000,
    tol: float = 1e-14,
    chunk_size: int | None = None,
    thread_limit: int | None = None,
):
    """W12 typed-propagation adapter candidate (bead personal-cmy.15).

    Runs a quivr ``Orbits`` or ``VariantOrbits`` table through the
    Rust-canonical typed ``TwoBodyPropagator`` ``PropagationRequest`` pipeline
    in one crossing: cross-product epoch policy over ``times``, optional
    linearized covariance transport, variant metadata preservation, and
    provider-owned ERFA rescaling for non-TDB epochs (UT1/GPS fail loudly).
    Private: the canonical public API remains ``dynamics.propagate_2body`` /
    ``Propagator.propagate_orbits``; this is the typed-contract adapter
    boundary. Returns ``(table, per_row_valid)``.
    """
    is_variants = isinstance(orbits, VariantOrbits)
    raw_in = variants_to_ipc(orbits) if is_variants else orbits_to_ipc(orbits)
    raw, valid = _rn.orbits_propagate_typed_ipc(
        raw_in,
        is_variants,
        times.scale,
        times.days.to_pylist(),
        times.nanos.to_pylist(),
        covariance,
        max_iter,
        tol,
        chunk_size,
        thread_limit,
    )
    table = variants_from_ipc(raw) if is_variants else orbits_from_ipc(raw)
    return table, valid


def _propagate_orbits_2body_ipc_candidate(
    orbits: Orbits, time, max_iter: int = 100, tol: float = 1e-14
) -> Orbits:
    """Diagnostic Arrow-IPC candidate for ``dynamics.propagate_2body``.

    Private per the bridge-naming policy (bead personal-cmy.13.1.4): the
    canonical public API is ``dynamics.propagate_2body``; this Orbits-level
    single-shared-time transport experiment is kept for the harness.
    Propagates ``orbits`` to a single shared ``time`` (length-1 Timestamp)
    with 2-body dynamics Rust-side in one crossing (state only); orbit epochs
    and ``time`` must share the dynamics time scale (typically TDB).
    """
    days = int(time.days.to_numpy(zero_copy_only=False)[0])
    nanos = int(time.nanos.to_numpy(zero_copy_only=False)[0])
    raw = _rn.orbits_propagate_2body_ipc(
        orbits_to_ipc(orbits), days, nanos, time.scale, max_iter, tol
    )
    return orbits_from_ipc(raw)
