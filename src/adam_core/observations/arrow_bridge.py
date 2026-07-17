"""Arrow bridge: quivr observation tables <-> Rust-canonical observation batches.

W8 observations data model (bead personal-cmy.20). Decision (user, 2026-07-05):
the Rust-canonical observation schemas mirror the existing quivr tables 1:1.
Each supported table crosses to Rust as nested quivr-layout Arrow IPC in a
single crossing; Timestamp column scales travel in schema metadata
(``adam_core_time_scale_<column>``) exactly like the orbit bridge carries
``coordinates.time.scale``.
"""

from __future__ import annotations

import pyarrow as pa
import quivr as qv

from adam_core import _rust_native as _rn

from .ades import ADESObservations
from .associations import Associations
from .detections import PointSourceDetections
from .exposures import Exposures
from .photometry import Photometry
from .source_catalog import SourceCatalog

_SCHEMA_NAMES: dict[type[qv.Table], str] = {
    ADESObservations: "AdesObservationBatch.nested.quivr.v1",
    PointSourceDetections: "PointSourceDetectionBatch.nested.quivr.v1",
    Exposures: "ExposureBatch.nested.quivr.v1",
    Associations: "AssociationBatch.nested.quivr.v1",
    Photometry: "PhotometryBatch.nested.quivr.v1",
    SourceCatalog: "SourceCatalogBatch.nested.quivr.v1",
}

_TIME_COLUMNS: dict[type[qv.Table], tuple[str, ...]] = {
    ADESObservations: ("obsTime",),
    PointSourceDetections: ("time",),
    Exposures: ("start_time",),
    Associations: (),
    Photometry: ("time",),
    SourceCatalog: ("time", "exposure_start_time"),
}


def _supported_class(table: qv.Table) -> type[qv.Table]:
    cls = type(table)
    if cls not in _SCHEMA_NAMES:
        raise TypeError(f"Unsupported observation table type: {cls.__name__}")
    return cls


def observations_to_ipc(table: qv.Table) -> bytes:
    """Serialize a supported observation table to Arrow IPC bytes with the
    ``adam_core_*`` schema metadata the Rust codec reads."""
    cls = _supported_class(table)
    metadata = {
        b"adam_core_schema": _SCHEMA_NAMES[cls].encode(),
        b"adam_core_schema_version": b"1",
    }
    for column in _TIME_COLUMNS[cls]:
        scale = getattr(table, column).scale
        metadata[f"adam_core_time_scale_{column}".encode()] = scale.encode()
    arrow_table = table.table.combine_chunks().replace_schema_metadata(metadata)
    batches = arrow_table.to_batches()
    if not batches:
        # Zero-row tables serialize to zero batches; emit one explicit empty
        # batch so the Rust reader always sees the schema + row count.
        batches = [pa.RecordBatch.from_pylist([], schema=arrow_table.schema)]
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, arrow_table.schema) as writer:
        for batch in batches:
            writer.write_batch(batch)
    return sink.getvalue().to_pybytes()


def observations_from_ipc(raw: bytes, cls: type[qv.Table]) -> qv.Table:
    """Reconstruct a quivr observation table from Rust-produced IPC bytes."""
    if cls not in _SCHEMA_NAMES:
        raise TypeError(f"Unsupported observation table type: {cls.__name__}")
    with pa.ipc.open_stream(pa.py_buffer(raw)) as reader:
        arrow_table = reader.read_all().combine_chunks()
    metadata = dict(arrow_table.schema.metadata or {})
    quivr_metadata = {}
    for column in _TIME_COLUMNS[cls]:
        key = f"adam_core_time_scale_{column}".encode()
        if key in metadata:
            quivr_metadata[f"{column}.scale".encode()] = metadata[key]
    return cls.from_pyarrow(arrow_table.replace_schema_metadata(quivr_metadata))


def round_trip_observations(table: qv.Table) -> qv.Table:
    """Identity round-trip through the Rust-canonical observation batch."""
    cls = _supported_class(table)
    return observations_from_ipc(
        _rn.observations_nested_ipc_round_trip(observations_to_ipc(table)), cls
    )
