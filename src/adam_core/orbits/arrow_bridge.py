"""Arrow IPC bridge: quivr ``Orbits`` <-> Rust-canonical ``OrbitBatch``.

W1 data-model keystone (Beads personal-cmy.13). Provides a single Python<->Rust
crossing that carries the *complete* nested ``Orbits`` schema (coordinates incl.
covariance, time, origin, and physical parameters) to/from the Rust data model
via Arrow IPC bytes, plus Rust-native workflows that run entirely on the
``OrbitBatch`` between encode and decode.

quivr stores ``frame`` and the timestamp ``scale`` as ``StringAttribute``
values in Arrow *schema metadata* (``coordinates.frame`` /
``coordinates.time.scale``); the Rust decoder reads canonical ``adam_core_*``
metadata keys. This module owns that metadata translation in both directions so
callers exchange ordinary ``Orbits`` objects.
"""

from __future__ import annotations

import pyarrow as pa

from adam_core import _rust_native as _rn
from adam_core.orbits import Orbits
from adam_core.orbits.variants import VariantOrbits

_NESTED_SCHEMA = "OrbitBatch.cartesian.nested.quivr.v1"


def orbits_to_ipc(orbits: Orbits) -> bytes:
    """Serialize ``Orbits`` to Arrow IPC bytes with the metadata Rust needs."""
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
    table = table.replace_schema_metadata(metadata)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _table_with_quivr_metadata(raw: bytes) -> pa.Table:
    """Read Rust IPC bytes and translate ``adam_core_*`` metadata to quivr keys."""
    with pa.ipc.open_stream(pa.py_buffer(raw)) as reader:
        table = reader.read_all().combine_chunks()
    metadata = dict(table.schema.metadata or {})
    frame = metadata.get(b"adam_core_frame", b"unspecified").decode()
    scale = metadata.get(b"adam_core_time_scale", b"utc").decode()
    return table.replace_schema_metadata(
        {
            b"coordinates.frame": frame.encode(),
            b"coordinates.time.scale": scale.encode(),
        }
    )


def orbits_from_ipc(raw: bytes) -> Orbits:
    """Reconstruct ``Orbits`` from Rust-produced Arrow IPC bytes."""
    return Orbits.from_pyarrow(_table_with_quivr_metadata(raw))


def variants_from_ipc(raw: bytes) -> VariantOrbits:
    """Reconstruct ``VariantOrbits`` from Rust-produced Arrow IPC bytes."""
    return VariantOrbits.from_pyarrow(_table_with_quivr_metadata(raw))


def round_trip_orbits(orbits: Orbits) -> Orbits:
    """Decode to the Rust ``OrbitBatch`` and back (identity bridge check)."""
    return orbits_from_ipc(_rn.orbits_nested_ipc_round_trip(orbits_to_ipc(orbits)))


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
