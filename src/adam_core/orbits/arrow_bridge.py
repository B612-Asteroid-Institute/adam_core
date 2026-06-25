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
from adam_core.observers import Observers
from adam_core.orbits import Orbits
from adam_core.orbits.variants import VariantOrbits

_NESTED_SCHEMA = "OrbitBatch.cartesian.nested.quivr.v1"
_OBSERVER_SCHEMA = "ObserverBatch.cartesian.nested.quivr.v1"


def _stamp_adam_core_metadata(
    table: pa.Table, representation: str, frame: str, scale: str, schema_name: str
) -> pa.Table:
    """Stamp the canonical ``adam_core_*`` schema metadata the Rust codec reads."""
    metadata = dict(table.schema.metadata or {})
    metadata.update(
        {
            b"adam_core_schema": schema_name.encode(),
            b"adam_core_schema_version": b"1",
            b"adam_core_representation": representation.encode(),
            b"adam_core_frame": frame.encode(),
            b"adam_core_time_scale": scale.encode(),
        }
    )
    return table.replace_schema_metadata(metadata)


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


def _to_quivr_metadata(table: pa.Table) -> pa.Table:
    """Translate Rust ``adam_core_*`` metadata back into quivr attribute keys."""
    metadata = dict(table.schema.metadata or {})
    frame = metadata.get(b"adam_core_frame", b"unspecified").decode()
    scale = metadata.get(b"adam_core_time_scale", b"utc").decode()
    return table.replace_schema_metadata(
        {
            b"coordinates.frame": frame.encode(),
            b"coordinates.time.scale": scale.encode(),
        }
    )


# --- IPC-bytes transport -------------------------------------------------------


def orbits_to_ipc(orbits: Orbits) -> bytes:
    """Serialize ``Orbits`` to Arrow IPC bytes with the metadata Rust needs."""
    table = _with_adam_core_metadata(orbits)
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


def evaluate_residuals_2body(
    orbits: Orbits,
    observed_coordinates,
    observers: Observers,
    lt_tol: float = 1e-10,
    max_iter: int = 1000,
    tol: float = 1e-15,
    stellar_aberration: bool = False,
    max_lt_iter: int = 10,
):
    """Rust-native OD residual evaluation (the OD inner loop) over the bridge.

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


def fit_orbit_least_squares(
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


def sample_orbit_variants(
    orbits: Orbits,
    method: str = "sigma-point",
    num_samples: int = 10000,
    seed: int | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    kappa: float = 0.0,
) -> VariantOrbits:
    """Sample covariance variants of ``orbits`` Rust-side in one crossing.

    Mirrors ``VariantOrbits.create`` semantics; ``method`` is one of ``auto``,
    ``sigma-point``, or ``monte-carlo``.
    """
    raw = _rn.orbits_sample_variants_ipc(
        orbits_to_ipc(orbits), method, num_samples, seed, alpha, beta, kappa
    )
    return variants_from_ipc(raw)


def propagate_orbits_2body(
    orbits: Orbits, time, max_iter: int = 100, tol: float = 1e-14
) -> Orbits:
    """Propagate ``orbits`` to a single shared ``time`` (length-1 Timestamp) with
    2-body dynamics Rust-side in one crossing (state only). Orbit epochs and
    ``time`` must share the dynamics time scale (typically TDB).
    """
    days = int(time.days.to_numpy(zero_copy_only=False)[0])
    nanos = int(time.nanos.to_numpy(zero_copy_only=False)[0])
    raw = _rn.orbits_propagate_2body_ipc(
        orbits_to_ipc(orbits), days, nanos, time.scale, max_iter, tol
    )
    return orbits_from_ipc(raw)
