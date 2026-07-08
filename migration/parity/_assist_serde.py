"""Portable quivr <-> Arrow-IPC serialization for the adam_assist parity oracle.

The main venv (rust adam_core) and ``.legacy-assist-venv`` (legacy,
composition-bearing adam_core) both run quivr 0.8.1 with schema-compatible
adam_core tables, so any quivr ``Table`` round-trips losslessly through Arrow
IPC bytes (verified: an ``Orbits`` serialized under rust adam_core reconstructs
exactly under legacy adam_core via ``from_pyarrow``). This lets the subprocess
oracle move Orbits / VariantOrbits / Observers / Ephemeris / CollisionEvent /
Timestamp between the two isolated runtimes without coupling to either
adam_core's Python object identity.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa


def table_to_ipc(table: Any) -> bytes:
    """Serialize a quivr ``Table`` to Arrow IPC stream bytes."""
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.table.schema) as writer:
        writer.write_table(table.table)
    return sink.getvalue().to_pybytes()


def table_from_ipc(cls: Any, data: bytes) -> Any:
    """Reconstruct a quivr ``Table`` subclass ``cls`` from Arrow IPC bytes."""
    reader = pa.ipc.open_stream(pa.BufferReader(data))
    return cls.from_pyarrow(reader.read_all())
