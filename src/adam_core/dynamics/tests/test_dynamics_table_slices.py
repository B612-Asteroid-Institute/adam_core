"""
Regression: a sliced quivr table (``orbits[i : i + 1]``, Arrow offset > 0)
handed to the typed Rust crossings must be read as the sliced rows. The Rust
nested-schema decoders read struct children from the start of the shared
buffers, so the Python bridge materializes offset arrays
(`adam_core._rust.arrow.contiguous_record_batch`) before every crossing.
"""

import numpy as np
import pyarrow as pa

from ..._rust.arrow import contiguous_record_batch
from ...coordinates import CartesianCoordinates, Origin
from ...observers import Observers
from ...orbits import Orbits
from ...orbits.arrow_bridge import (
    observers_to_record_batch,
    orbits_from_ipc,
    orbits_to_ipc,
    orbits_to_record_batch,
)
from ...time import Timestamp
from ..ephemeris import generate_ephemeris_2body
from ..propagation import propagate_2body


def _orbits() -> Orbits:
    states = np.array(
        [
            [0.9, 0.5, 0.05, -0.008, 0.012, 0.0005],
            [0.95, 0.5, 0.05, -0.008, 0.012, 0.0005],
            [1.0, 0.5, 0.05, -0.008, 0.012, 0.0005],
        ]
    )
    return Orbits.from_kwargs(
        orbit_id=["a", "b", "c"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=states[:, 0],
            y=states[:, 1],
            z=states[:, 2],
            vx=states[:, 3],
            vy=states[:, 4],
            vz=states[:, 5],
            time=Timestamp.from_mjd([61000.0] * 3, scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN"] * 3),
            frame="ecliptic",
        ),
    )


def test_contiguous_record_batch_materializes_offsets() -> None:
    orbits = _orbits()
    sliced = orbits[1:3]
    raw = sliced.table.combine_chunks()
    assert any(column.chunk(0).offset != 0 for column in raw.columns)
    batch = contiguous_record_batch(raw)
    assert all(column.offset == 0 for column in batch.columns)
    assert batch.num_rows == 2
    assert batch.column(batch.schema.get_field_index("orbit_id")).to_pylist() == [
        "b",
        "c",
    ]
    coordinates = batch.column(batch.schema.get_field_index("coordinates"))
    assert coordinates.field(0).to_pylist() == [0.95, 1.0]
    # An unsliced table passes through unchanged.
    plain = contiguous_record_batch(orbits.table.combine_chunks())
    assert plain.num_rows == 3
    # Empty tables are fine.
    assert contiguous_record_batch(orbits[:0].table.combine_chunks()).num_rows == 0


def test_record_batch_and_ipc_routes_agree_on_a_slice() -> None:
    orbits = _orbits()
    batch = orbits_to_record_batch(orbits[1:2])
    via_ipc = orbits_from_ipc(orbits_to_ipc(orbits[1:2]))
    assert batch.column(batch.schema.get_field_index("orbit_id")).to_pylist() == ["b"]
    assert via_ipc.orbit_id.to_pylist() == ["b"]
    observers = Observers.from_code(
        "500", Timestamp.from_mjd([61010.0, 61020.0, 61030.0], scale="tdb")
    )
    observer_batch = observers_to_record_batch(observers[1:3])
    assert observer_batch.num_rows == 2
    assert all(column.offset == 0 for column in observer_batch.columns)


def test_sliced_orbits_propagate_the_requested_row() -> None:
    orbits = _orbits()
    times = Timestamp.from_mjd([61010.0], scale="tdb")
    full = propagate_2body(orbits, times)
    sliced = propagate_2body(orbits[1:2], times)
    taken = propagate_2body(orbits.take([1]), times)
    np.testing.assert_array_equal(
        sliced.coordinates.values, full.coordinates.values[1:2]
    )
    np.testing.assert_array_equal(
        taken.coordinates.values, full.coordinates.values[1:2]
    )
    assert sliced.orbit_id.to_pylist() == ["b"]


def test_sliced_orbits_and_observers_generate_the_requested_ephemeris() -> None:
    orbits = _orbits()
    times = Timestamp.from_mjd([61010.0, 61020.0, 61030.0], scale="tdb")
    observers = Observers.from_code("500", times)
    propagated = propagate_2body(orbits.take([0]), times)
    full = generate_ephemeris_2body(propagated, observers).coordinates.values
    from_slice = generate_ephemeris_2body(
        propagated[1:3], observers[1:3]
    ).coordinates.values
    from_take = generate_ephemeris_2body(
        propagated.take([1, 2]), observers.take([1, 2])
    ).coordinates.values
    np.testing.assert_array_equal(from_slice, full[1:3])
    np.testing.assert_array_equal(from_take, full[1:3])


def test_sliced_target_times_are_honoured() -> None:
    orbits = _orbits()
    times = Timestamp.from_mjd([61005.0, 61010.0], scale="tdb")
    full = propagate_2body(orbits.take([0]), times).coordinates.values
    sliced = propagate_2body(orbits.take([0]), times[1:2]).coordinates.values
    np.testing.assert_array_equal(sliced, full[1:2])
    assert isinstance(times[1:2].table, pa.Table)
