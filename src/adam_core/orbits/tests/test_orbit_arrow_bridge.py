"""W1 data-model bridge parity (Beads personal-cmy.13, mechanism C).

Round-trips a real quivr ``Orbits`` table through the Rust-canonical ``OrbitBatch``
via Arrow IPC bytes (a single Python<->Rust crossing of the complete nested
schema) and asserts the data AND the Arrow schema (ignoring metadata) survive
exactly. This exercises the Rust ``OrbitBatch`` nested quivr-compatible
round-trip end-to-end through the compiled ``_rust_native`` extension.

The small ``_orbits_to_ipc`` shim injects the ``adam_core_*`` schema metadata
(frame, time scale) the Rust decoder needs -- this is the "quivr adapts"
boundary that a future bridge module will own.
"""

import numpy as np
import pyarrow as pa

from adam_core import _rust_native as rn
from adam_core.coordinates import CartesianCoordinates, CoordinateCovariances, Origin
from adam_core.orbits import Orbits
from adam_core.orbits.orbits import PhysicalParameters
from adam_core.time import Timestamp


def _orbits_to_ipc(orbits: Orbits) -> bytes:
    table = orbits.table.combine_chunks()
    coords = orbits.coordinates
    table = table.replace_schema_metadata(
        {
            "adam_core_schema": "OrbitBatch.cartesian.nested.quivr.v1",
            "adam_core_schema_version": "1",
            "adam_core_representation": "cartesian",
            "adam_core_frame": coords.frame,
            "adam_core_time_scale": coords.time.scale,
        }
    )
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _read_ipc_table(raw: bytes) -> pa.Table:
    with pa.ipc.open_stream(pa.py_buffer(raw)) as reader:
        return reader.read_all().combine_chunks()


def _assert_lossless(orbits: Orbits) -> None:
    tin = orbits.table.combine_chunks()
    tout = _read_ipc_table(rn.orbits_nested_ipc_round_trip(_orbits_to_ipc(orbits)))
    # Full nested data survives quivr -> IPC -> Rust OrbitBatch -> IPC -> quivr.
    assert tout.to_pylist() == tin.to_pylist()
    # Arrow schema (types + nullability) is byte-identical, ignoring metadata.
    assert tout.schema.equals(tin.schema, check_metadata=False)


def _cartesian(with_covariance: bool) -> CartesianCoordinates:
    n = 3
    kwargs = dict(
        x=[1.0, 4.0, 7.0],
        y=[2.0, 5.0, 8.0],
        z=[3.0, 6.0, 9.0],
        vx=[0.1, 0.4, 0.7],
        vy=[0.2, 0.5, 0.8],
        vz=[0.3, 0.6, 0.9],
        time=Timestamp.from_mjd([60000.0, 60001.0, 60002.0], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN", "SUN", "SUN"]),
        frame="ecliptic",
    )
    if with_covariance:
        cov = np.stack(
            [np.arange(36, dtype=float).reshape(6, 6) + i * 100 for i in range(n)]
        )
        kwargs["covariance"] = CoordinateCovariances.from_matrix(cov)
    return CartesianCoordinates.from_kwargs(**kwargs)


def test_orbits_nested_ipc_round_trip_full_with_covariance():
    orbits = Orbits.from_kwargs(
        orbit_id=["o1", "o2", "o3"],
        object_id=["a", None, "c"],
        coordinates=_cartesian(with_covariance=True),
    )
    _assert_lossless(orbits)


def test_orbits_nested_ipc_round_trip_without_covariance():
    orbits = Orbits.from_kwargs(
        orbit_id=["o1", "o2", "o3"],
        object_id=["a", "b", "c"],
        coordinates=_cartesian(with_covariance=False),
    )
    _assert_lossless(orbits)


def test_orbits_nested_ipc_round_trip_with_physical_parameters():
    physical_parameters = PhysicalParameters.from_kwargs(
        H_v=[15.5, 16.0, 17.0],
        H_v_sigma=[0.1, None, 0.3],
        G=[0.15, 0.15, 0.15],
        G_sigma=[None, None, None],
        sigma_eff=[0.05, 0.06, 0.07],
        chi2_red=[1.2, 1.1, 1.0],
    )
    orbits = Orbits.from_kwargs(
        orbit_id=["o1", "o2", "o3"],
        object_id=["a", "b", "c"],
        coordinates=_cartesian(with_covariance=True),
        physical_parameters=physical_parameters,
    )
    _assert_lossless(orbits)
