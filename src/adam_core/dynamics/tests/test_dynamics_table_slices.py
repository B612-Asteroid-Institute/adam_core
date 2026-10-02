"""
Regression pin: a sliced quivr table (``orbits[i : i + 1]``, Arrow offset > 0)
handed to the typed Rust crossings is read from row 0. ``take`` materializes
the rows and is unaffected. Strict expected failures: fixing the record-batch
decode turns them into passes, at which point remove the markers.
"""

import numpy as np
import pytest

from ...coordinates import CartesianCoordinates, Origin
from ...orbits import Orbits
from ...time import Timestamp
from ..ephemeris import generate_ephemeris_2body
from ..propagation import propagate_2body


def _orbits() -> Orbits:
    states = np.array(
        [
            [0.9, 0.5, 0.05, -0.008, 0.012, 0.0005],
            [0.95, 0.5, 0.05, -0.008, 0.012, 0.0005],
        ]
    )
    return Orbits.from_kwargs(
        orbit_id=["a", "b"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=states[:, 0],
            y=states[:, 1],
            z=states[:, 2],
            vx=states[:, 3],
            vy=states[:, 4],
            vz=states[:, 5],
            time=Timestamp.from_mjd([61000.0, 61000.0], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN", "SUN"]),
            frame="ecliptic",
        ),
    )


def test_take_selects_the_requested_row() -> None:
    orbits = _orbits()
    times = Timestamp.from_mjd([61010.0], scale="tdb")
    full = propagate_2body(orbits, times)
    taken = propagate_2body(orbits.take([1]), times)
    np.testing.assert_array_equal(
        taken.coordinates.values, full.coordinates.values[1:2]
    )


@pytest.mark.xfail(
    strict=True,
    reason="the record-batch crossing ignores the Arrow slice offset and propagates row 0",
)
def test_slice_selects_the_requested_row() -> None:
    orbits = _orbits()
    times = Timestamp.from_mjd([61010.0], scale="tdb")
    full = propagate_2body(orbits, times)
    sliced = propagate_2body(orbits[1:2], times)
    np.testing.assert_array_equal(
        sliced.coordinates.values, full.coordinates.values[1:2]
    )


@pytest.mark.xfail(
    strict=True,
    reason="the record-batch crossing ignores the Arrow slice offset in generate_ephemeris_2body",
)
def test_ephemeris_slice_selects_the_requested_rows() -> None:
    from ...observers import Observers

    orbits = _orbits()
    times = Timestamp.from_mjd([61010.0, 61020.0], scale="tdb")
    observers = Observers.from_code("500", times)
    propagated = propagate_2body(orbits, times)
    from_take = generate_ephemeris_2body(propagated.take([2, 3]), observers)
    from_slice = generate_ephemeris_2body(propagated[2:4], observers)
    np.testing.assert_array_equal(
        from_slice.coordinates.values, from_take.coordinates.values
    )
