from __future__ import annotations

import numpy as np
import pytest

from adam_assist_rust import ASSISTPropagator as RustASSISTPropagator
from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin
from adam_core.orbits.orbits import Orbits
from adam_core.time import Timestamp


def _same_epoch_orbits() -> Orbits:
    coordinates = CartesianCoordinates.from_kwargs(
        x=[1.05, 1.10, 1.15],
        y=[0.0, 0.02, -0.03],
        z=[0.0, 0.001, -0.002],
        vx=[0.0, -0.0002, 0.0003],
        vy=[0.016787, 0.0159, 0.0152],
        vz=[0.0, 0.0001, -0.0001],
        time=Timestamp.from_mjd([60000.0, 60000.0, 60000.0], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN", "SUN", "SUN"]),
        frame="ecliptic",
    )
    return Orbits.from_kwargs(
        orbit_id=["fast-0000", "fast-0001", "fast-0002"],
        object_id=["fast-0000", "fast-0001", "fast-0002"],
        coordinates=coordinates,
    )


def test_same_epoch_multi_orbit_fast_path_matches_python_public() -> None:
    python_assist = pytest.importorskip("adam_assist")
    orbits = _same_epoch_orbits()
    times = Timestamp.from_mjd([60000.25, 60001.0], scale="tdb")

    expected = python_assist.ASSISTPropagator().propagate_orbits(
        orbits,
        times,
        max_processes=1,
        chunk_size=10,
    )
    actual = RustASSISTPropagator().propagate_orbits(
        orbits,
        times,
        max_processes=1,
        chunk_size=10,
    )

    assert actual.orbit_id.to_pylist() == expected.orbit_id.to_pylist()
    np.testing.assert_array_equal(
        actual.coordinates.time.mjd().to_numpy(zero_copy_only=False),
        expected.coordinates.time.mjd().to_numpy(zero_copy_only=False),
    )
    np.testing.assert_allclose(
        actual.coordinates.values,
        expected.coordinates.values,
        atol=1.0e-13,
        rtol=0,
    )
