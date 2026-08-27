import json
from pathlib import Path

import numpy as np

from adam_core.coordinates import CartesianCoordinates, Origin
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

from ..utils import ensure_input_origin_and_frame, ensure_input_time_scale


def test_ensure_input_time_scale():
    # Test that ensure_input_time_scale works
    times = Timestamp.from_mjd(np.array([1.0, 2.0, 3.0]), scale="tdb")
    orbits = Orbits.from_kwargs(
        coordinates=CartesianCoordinates.from_kwargs(
            x=np.array([1.0, 2.0, 3.0]),
            y=np.array([1.0, 2.0, 3.0]),
            z=np.array([1.0, 2.0, 3.0]),
            vx=np.array([1.0, 2.0, 3.0]),
            vy=np.array([1.0, 2.0, 3.0]),
            vz=np.array([1.0, 2.0, 3.0]),
            time=times,
        ),
    )

    results = orbits.set_column(
        "coordinates.time", orbits.coordinates.time.rescale("utc")
    )
    assert results.coordinates.time.scale == "utc"

    results = ensure_input_time_scale(orbits, times)
    assert results.coordinates.time.scale == "tdb"


def test_ensure_input_origin_and_frame():
    # Test that ensure_input_origin_and_frame works
    times = Timestamp.from_mjd(np.array([60000.0, 60001.0, 60002.0]), scale="tdb")
    orbits = Orbits.from_kwargs(
        coordinates=CartesianCoordinates.from_kwargs(
            x=np.array([1.0, 2.0, 3.0]),
            y=np.array([1.0, 2.0, 3.0]),
            z=np.array([1.0, 2.0, 3.0]),
            vx=np.array([1.0, 2.0, 3.0]),
            vy=np.array([1.0, 2.0, 3.0]),
            vz=np.array([1.0, 2.0, 3.0]),
            time=times,
            origin=Origin.from_kwargs(code=["EARTH", "SUN", "MOON"]),
            frame="equatorial",
        ),
    )

    results = Orbits.from_kwargs(
        orbit_id=orbits.orbit_id,
        coordinates=CartesianCoordinates.from_kwargs(
            x=orbits.coordinates.x,
            y=orbits.coordinates.y,
            z=orbits.coordinates.z,
            vx=orbits.coordinates.vx,
            vy=orbits.coordinates.vy,
            vz=orbits.coordinates.vz,
            time=orbits.coordinates.time,
            origin=Origin.from_kwargs(
                code=["SOLAR_SYSTEM_BARYCENTER", "SUN", "SOLAR_SYSTEM_BARYCENTER"]
            ),
            frame="ecliptic",
        ),
    )

    results = ensure_input_origin_and_frame(orbits, results)
    assert results.coordinates.origin.code.to_pylist() == ["EARTH", "SUN", "MOON"]
    assert results.coordinates.frame == "equatorial"


def _make_parity_cases():
    input_times = Timestamp.from_mjd([60000.0, 60001.0, 60002.0], scale="tdb")
    inputs = Orbits.from_kwargs(
        orbit_id=["a", "b", "c"],
        object_id=["A", "B", "C"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0, 2.0, 3.0],
            y=[0.1, 0.2, 0.3],
            z=[-0.1, -0.2, -0.3],
            vx=[0.01, 0.02, 0.03],
            vy=[0.001, 0.002, 0.003],
            vz=[-0.001, -0.002, -0.003],
            time=input_times,
            origin=Origin.from_kwargs(code=["EARTH", "SUN", "MOON"]),
            frame="equatorial",
        ),
    )
    result_times = Timestamp.from_mjd([60002.0, 60000.0, 60001.0, 60000.5], scale="tdb")
    results = Orbits.from_kwargs(
        orbit_id=["c", "a", "b", "a"],
        object_id=["C", "A", "B", "A"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[3.0, 1.0, 2.0, 1.5],
            y=[0.3, 0.1, 0.2, 0.15],
            z=[-0.3, -0.1, -0.2, -0.15],
            vx=[0.03, 0.01, 0.02, 0.015],
            vy=[0.003, 0.001, 0.002, 0.0015],
            vz=[-0.003, -0.001, -0.002, -0.0015],
            time=result_times,
            origin=Origin.from_kwargs(code=["SOLAR_SYSTEM_BARYCENTER"] * 4),
            frame="ecliptic",
        ),
    )
    return inputs, results


def test_propagator_utils_frozen_legacy_parity():
    fixture_path = (
        Path(__file__).resolve().parents[4]
        / "migration"
        / "artifacts"
        / "propagator_utils_fixture_2026-07-12.json"
    )
    fixture = json.loads(fixture_path.read_text())
    inputs, results = _make_parity_cases()

    utc_results = results.set_column(
        "coordinates.time", results.coordinates.time.rescale("utc")
    )
    time_output = ensure_input_time_scale(utc_results, inputs.coordinates.time)
    expected_time = fixture["time_scale"]
    assert time_output.coordinates.time.days.to_pylist() == expected_time["days"]
    assert time_output.coordinates.time.nanos.to_pylist() == expected_time["nanos"]
    assert time_output.coordinates.time.scale == expected_time["scale"]

    origin_output = ensure_input_origin_and_frame(inputs, results)
    expected = fixture["origin_and_frame"]
    assert origin_output.orbit_id.to_pylist() == expected["orbit_id"]
    assert origin_output.object_id.to_pylist() == expected["object_id"]
    assert origin_output.coordinates.time.days.to_pylist() == expected["days"]
    assert origin_output.coordinates.time.nanos.to_pylist() == expected["nanos"]
    assert origin_output.coordinates.time.scale == expected["scale"]
    assert origin_output.coordinates.origin.code.to_pylist() == expected["origins"]
    assert origin_output.coordinates.frame == expected["frame"]
    np.testing.assert_allclose(
        origin_output.coordinates.values, expected["values"], rtol=1e-13, atol=1e-13
    )


def test_ensure_input_origin_and_frame_empty_inputs_returns_none():
    inputs, results = _make_parity_cases()
    assert ensure_input_origin_and_frame(inputs[:0], results) is None


def test_propagator_utils_rust_native_timing():
    from adam_core import _rust_native
    from adam_core._rust.arrow import ensure_spice_backend
    from adam_core.coordinates.transform import _coordinate_record_batch

    inputs, results = _make_parity_cases()
    utc = results.coordinates.time.rescale("utc")
    time_samples = _rust_native.benchmark_ensure_input_time_scale(
        np.ascontiguousarray(utc.days.to_numpy()),
        np.ascontiguousarray(utc.nanos.to_numpy()),
        "utc",
        "tdb",
        2,
        2,
        0,
    )
    ensure_spice_backend()
    origin_samples = _rust_native.benchmark_ensure_input_origin_and_frame_arrow(
        _coordinate_record_batch(results.coordinates, "cartesian"),
        inputs.orbit_id.to_pylist(),
        inputs.coordinates.origin.code.to_pylist(),
        results.orbit_id.to_pylist(),
        inputs.coordinates.frame,
        2,
        2,
        0,
    )
    for samples in (time_samples, origin_samples):
        assert len(samples) == 2
        assert all(len(trial) == 2 for trial in samples)
        assert all(sample >= 0 for trial in samples for sample in trial)
