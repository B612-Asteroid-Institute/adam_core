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
from adam_core.orbits import Orbits
from adam_core.orbits.variants import VariantOrbits

_NESTED_SCHEMA = "OrbitBatch.cartesian.nested.quivr.v1"


def _with_adam_core_metadata(orbits: Orbits) -> pa.Table:
    """Combine chunks and stamp the canonical ``adam_core_*`` schema metadata."""
    table = orbits.table.combine_chunks()
    coordinates = orbits.coordinates
    metadata = dict(table.schema.metadata or {})
    metadata.update(
        {
            b"adam_core_schema": _NESTED_SCHEMA.encode(),
            b"adam_core_schema_version": b"1",
            b"adam_core_representation": b"cartesian",
            b"adam_core_frame": coordinates.frame.encode(),
            b"adam_core_time_scale": coordinates.time.scale.encode(),
        }
    )
    return table.replace_schema_metadata(metadata)


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


def round_trip_orbits_zero_copy(orbits: Orbits) -> Orbits:
    """Identity round-trip via the zero-copy Arrow C Data Interface transport."""
    out = _rn.orbits_nested_round_trip_arrow(orbits_to_record_batch(orbits))
    return orbits_from_record_batch(out)


def rotate_orbits_frame(orbits: Orbits, frame: str) -> Orbits:
    """Rotate orbit coordinates and covariance into ``frame`` Rust-side, one crossing."""
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
