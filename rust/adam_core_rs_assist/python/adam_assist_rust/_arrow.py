"""Canonical flat Arrow mapping for the W12 typed propagation adapter.

This is the Python side of the typed adapter (bead personal-cmy.10). It maps a
quivr ``Orbits`` table to the Rust-canonical ``OrbitBatch.cartesian.flat.v1``
RecordBatch layout defined in ``adam_core_rs_coords::types::arrow`` and back.

Slice (1) pins this mapping as a tested contract with no new Rust dependency.
Slice (2) will hand the flat RecordBatch to Rust over the Arrow C data
interface so ``OrbitBatch::try_from_record_batch`` owns schema validation and
the typed contracts end to end, replacing the current dict/numpy boundary.

The flat schema (field order, dtypes, nullability, metadata) intentionally
mirrors ``orbit_schema`` in the Rust crate. The schema-name parity is checked
against ``adam_core._rust`` ``orbit_schema_metadata`` in the contract test.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.coordinates.origin import Origin
from adam_core.orbits.orbits import Orbits
from adam_core.time import Timestamp

SCHEMA_NAME = "OrbitBatch.cartesian.flat.v1"
SCHEMA_VERSION = "1"
REPRESENTATION = "cartesian"

_STATE_FIELDS = ("x", "y", "z", "vx", "vy", "vz")
_COVARIANCE_FIELDS = tuple(
    f"covariance_{row}{col}" for row in range(6) for col in range(6)
)
ORBIT_FLAT_FIELDS = (
    "orbit_id",
    "object_id",
    *_STATE_FIELDS,
    "time_days",
    "time_nanos",
    "origin_code",
    *_COVARIANCE_FIELDS,
)


def _flat_schema(*, frame: str, time_scale: str, has_covariance: bool) -> pa.Schema:
    fields = [
        pa.field("orbit_id", pa.large_string(), nullable=False),
        pa.field("object_id", pa.large_string(), nullable=True),
    ]
    for name in _STATE_FIELDS:
        fields.append(pa.field(name, pa.float64(), nullable=False))
    fields.append(pa.field("time_days", pa.int64(), nullable=True))
    fields.append(pa.field("time_nanos", pa.int64(), nullable=True))
    fields.append(pa.field("origin_code", pa.large_string(), nullable=False))
    for name in _COVARIANCE_FIELDS:
        fields.append(pa.field(name, pa.float64(), nullable=True))
    metadata = {
        "adam_core_schema": SCHEMA_NAME,
        "adam_core_schema_version": SCHEMA_VERSION,
        "adam_core_representation": REPRESENTATION,
        "adam_core_frame": frame,
        "adam_core_time_scale": time_scale,
        "adam_core_covariance": "present" if has_covariance else "absent",
    }
    return pa.schema(fields, metadata=metadata)


def _metadata_value(metadata: dict | None, key: str) -> str:
    if not metadata:
        raise ValueError(f"flat OrbitBatch RecordBatch is missing metadata key {key!r}")
    value = (
        metadata.get(key.encode()) if metadata.get(key.encode()) else metadata.get(key)
    )
    if value is None:
        raise ValueError(f"flat OrbitBatch RecordBatch is missing metadata key {key!r}")
    return value.decode() if isinstance(value, bytes) else value


def orbits_to_flat_record_batch(orbits: Orbits) -> pa.RecordBatch:
    """Flatten a quivr ``Orbits`` table into the canonical flat RecordBatch."""
    coordinates = orbits.coordinates
    rows = len(orbits)
    values = np.ascontiguousarray(coordinates.values, dtype=np.float64)
    days = coordinates.time.days.to_numpy(zero_copy_only=False).astype(np.int64)
    nanos = coordinates.time.nanos.to_numpy(zero_copy_only=False).astype(np.int64)
    has_covariance = not coordinates.covariance.is_all_nan()
    covariance = (
        coordinates.covariance.to_matrix().reshape(rows, 36) if has_covariance else None
    )

    arrays: list[pa.Array] = [
        pa.array(orbits.orbit_id.to_pylist(), type=pa.large_string()),
        pa.array(orbits.object_id.to_pylist(), type=pa.large_string()),
    ]
    for column in range(6):
        arrays.append(pa.array(values[:, column], type=pa.float64()))
    arrays.append(pa.array(days, type=pa.int64()))
    arrays.append(pa.array(nanos, type=pa.int64()))
    arrays.append(pa.array(coordinates.origin.code.to_pylist(), type=pa.large_string()))
    for element in range(36):
        if covariance is None:
            arrays.append(pa.array([None] * rows, type=pa.float64()))
        else:
            arrays.append(pa.array(covariance[:, element], type=pa.float64()))

    schema = _flat_schema(
        frame=coordinates.frame,
        time_scale=coordinates.time.scale,
        has_covariance=has_covariance,
    )
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def orbits_from_flat_record_batch(batch: pa.RecordBatch) -> Orbits:
    """Rebuild a quivr ``Orbits`` table from a canonical flat RecordBatch."""
    names = batch.schema.names
    if tuple(names) != ORBIT_FLAT_FIELDS:
        raise ValueError(
            "flat OrbitBatch RecordBatch columns do not match "
            f"{SCHEMA_NAME}; got {names}"
        )
    metadata = batch.schema.metadata
    frame = _metadata_value(metadata, "adam_core_frame")
    time_scale = _metadata_value(metadata, "adam_core_time_scale")
    has_covariance = _metadata_value(metadata, "adam_core_covariance") == "present"

    def column(name: str) -> pa.Array:
        return batch.column(names.index(name))

    rows = batch.num_rows
    states = np.empty((rows, 6), dtype=np.float64)
    for index, name in enumerate(_STATE_FIELDS):
        states[:, index] = column(name).to_numpy(zero_copy_only=False)
    days = column("time_days").to_numpy(zero_copy_only=False).astype(np.int64)
    nanos = column("time_nanos").to_numpy(zero_copy_only=False).astype(np.int64)

    covariance = None
    if has_covariance:
        flat = np.empty((rows, 36), dtype=np.float64)
        for element, name in enumerate(_COVARIANCE_FIELDS):
            flat[:, element] = column(name).to_numpy(zero_copy_only=False)
        covariance = CoordinateCovariances.from_matrix(flat.reshape(rows, 6, 6))

    coordinates = CartesianCoordinates.from_kwargs(
        x=states[:, 0],
        y=states[:, 1],
        z=states[:, 2],
        vx=states[:, 3],
        vy=states[:, 4],
        vz=states[:, 5],
        covariance=covariance,
        time=Timestamp.from_kwargs(
            days=days.tolist(), nanos=nanos.tolist(), scale=time_scale
        ),
        origin=Origin.from_kwargs(code=column("origin_code").to_pylist()),
        frame=frame,
    )
    return Orbits.from_kwargs(
        orbit_id=column("orbit_id").to_pylist(),
        object_id=column("object_id").to_pylist(),
        coordinates=coordinates,
    )


__all__ = [
    "ORBIT_FLAT_FIELDS",
    "SCHEMA_NAME",
    "orbits_to_flat_record_batch",
    "orbits_from_flat_record_batch",
]
