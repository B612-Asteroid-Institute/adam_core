from __future__ import annotations

import numpy as np
import pytest

from adam_assist_rust import ASSISTPropagator as RustASSISTPropagator
from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.origin import Origin
from adam_core.observers.observers import Observers
from adam_core.orbits.orbits import Orbits
from adam_core.time import Timestamp


def _orbits() -> Orbits:
    coordinates = CartesianCoordinates.from_kwargs(
        x=[1.05, 1.35],
        y=[0.02, -0.08],
        z=[0.01, 0.03],
        vx=[-0.0005, 0.001],
        vy=[0.0165, 0.014],
        vz=[0.0002, -0.0001],
        time=Timestamp.from_mjd([60000.0, 60000.0], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN", "SUN"]),
        frame="ecliptic",
    )
    return Orbits.from_kwargs(
        orbit_id=["fixture-a", "fixture-b"],
        object_id=["fixture-a", "fixture-b"],
        coordinates=coordinates,
    )


def _sorted_values(ephemeris) -> np.ndarray:
    ordered = ephemeris.sort_by(
        [
            "orbit_id",
            "coordinates.time.days",
            "coordinates.time.nanos",
            "coordinates.origin.code",
        ]
    )
    return np.asarray(ordered.coordinates.values, dtype=np.float64)


def test_generate_ephemeris_matches_python_adam_assist() -> None:
    python_assist = pytest.importorskip("adam_assist")
    orbits = _orbits()
    times = Timestamp.from_mjd([60000.5, 60001.0], scale="utc")
    observers = Observers.from_code("X05", times)

    expected = python_assist.ASSISTPropagator().generate_ephemeris(
        orbits,
        observers,
        covariance=False,
        max_processes=1,
        predict_magnitudes=False,
        predict_phase_angle=False,
    )
    actual = RustASSISTPropagator().generate_ephemeris(
        orbits,
        observers,
        predict_magnitudes=False,
        predict_phase_angle=False,
        max_processes=1,
    )

    expected_values = _sorted_values(expected)
    actual_values = _sorted_values(actual)
    assert actual_values.shape == expected_values.shape
    np.testing.assert_allclose(actual_values, expected_values, atol=1.0e-9, rtol=0)
