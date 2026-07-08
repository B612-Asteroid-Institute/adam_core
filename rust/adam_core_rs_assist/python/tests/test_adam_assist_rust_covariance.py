from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from adam_assist_rust import ASSISTPropagator
from adam_assist_rust import _coordinates_from_result, _time_parts
from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.coordinates.origin import Origin
from adam_core.orbits.orbits import Orbits
from adam_core.time import Timestamp


def _covariance_orbits() -> Orbits:
    sigmas = np.array(
        [[1.0e-9, 2.0e-9, 3.0e-9, 1.0e-10, 2.0e-10, 3.0e-10]],
        dtype=np.float64,
    )
    coordinates = CartesianCoordinates.from_kwargs(
        x=[1.05],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[0.016787],
        vz=[0.0],
        covariance=CoordinateCovariances.from_sigmas(sigmas),
        time=Timestamp.from_mjd([60000.0], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN"]),
        frame="ecliptic",
    )
    return Orbits.from_kwargs(
        orbit_id=["cov-test-0001"],
        object_id=["cov-test-0001"],
        coordinates=coordinates,
    )


def _native_covariance_result(
    propagator: ASSISTPropagator,
    orbits: Orbits,
    times: Timestamp,
    *,
    covariance_method: str,
    num_samples: int = 16,
    seed: int | None = 42,
) -> dict[str, Any]:
    input_scale, input_days, input_nanos = _time_parts(orbits.coordinates.time)
    target_scale, target_days, target_nanos = _time_parts(times)
    states: npt.NDArray[np.float64] = np.ascontiguousarray(
        orbits.coordinates.values, dtype=np.float64
    )
    covariances: npt.NDArray[np.float64] = np.ascontiguousarray(
        orbits.coordinates.covariance.to_matrix().reshape(len(orbits), 36),
        dtype=np.float64,
    )
    return propagator._native.propagate_orbits(
        ["cov-test-0001"],
        ["cov-test-0001"],
        states,
        ["SUN"],
        "ecliptic",
        input_scale,
        input_days,
        input_nanos,
        target_scale,
        target_days,
        target_nanos,
        True,
        covariances=covariances,
        covariance_method=covariance_method,
        num_samples=num_samples,
        seed=seed,
        chunk_size=10,
        thread_limit=1,
    )


def test_coordinates_from_result_reconstructs_covariance() -> None:
    covariance = np.arange(72, dtype=np.float64).reshape(2, 36)
    result: dict[str, Any] = {
        "states": np.arange(12, dtype=np.float64).reshape(2, 6),
        "covariances": covariance,
        "time_days": [60000, 60001],
        "time_nanos": [0, 123],
        "time_scale": "tdb",
        "origin_codes": ["SUN", "SUN"],
        "frame": "ecliptic",
    }

    coordinates = _coordinates_from_result(result)

    assert coordinates.frame == "ecliptic"
    assert coordinates.time.scale == "tdb"
    np.testing.assert_array_equal(
        coordinates.covariance.to_matrix(), covariance.reshape(2, 6, 6)
    )


def test_native_covariance_output_dictionary_has_flattened_covariance_rows() -> None:
    orbits = _covariance_orbits()
    propagator = ASSISTPropagator()
    result = _native_covariance_result(
        propagator,
        orbits,
        orbits.coordinates.time,
        covariance_method="sigma-point",
    )

    assert result["covariances"].shape == (1, 36)
    assert result["states"].shape == (1, 6)
    assert result["validity"] == [True]
    assert np.isfinite(result["covariances"]).all()


def test_sigma_point_covariance_matches_python_public_same_epoch(
    python_reference_propagator,
) -> None:
    orbits = _covariance_orbits()
    times = orbits.coordinates.time

    expected = python_reference_propagator.propagate_orbits(
        orbits,
        times,
        covariance=True,
        covariance_method="sigma-point",
        max_processes=1,
        chunk_size=10,
    )
    actual = ASSISTPropagator().propagate_orbits(
        orbits,
        times,
        covariance=True,
        covariance_method="sigma-point",
        max_processes=1,
        chunk_size=10,
    )

    np.testing.assert_allclose(
        actual.coordinates.values,
        expected.coordinates.values,
        atol=1.0e-14,
        rtol=0,
    )
    np.testing.assert_allclose(
        actual.coordinates.covariance.to_matrix(),
        expected.coordinates.covariance.to_matrix(),
        atol=1.0e-18,
        rtol=0,
    )


def test_sigma_point_covariance_matches_python_public_multiple_target_epochs(
    python_reference_propagator,
) -> None:
    orbits = _covariance_orbits()
    times = Timestamp.from_mjd([60000.25, 60001.0], scale="tdb")

    expected = python_reference_propagator.propagate_orbits(
        orbits,
        times,
        covariance=True,
        covariance_method="sigma-point",
        max_processes=1,
        chunk_size=10,
    )
    actual = ASSISTPropagator().propagate_orbits(
        orbits,
        times,
        covariance=True,
        covariance_method="sigma-point",
        max_processes=1,
        chunk_size=10,
    )

    np.testing.assert_allclose(
        actual.coordinates.values,
        expected.coordinates.values,
        atol=1.0e-14,
        rtol=0,
    )
    np.testing.assert_allclose(
        actual.coordinates.covariance.to_matrix(),
        expected.coordinates.covariance.to_matrix(),
        atol=1.0e-25,
        rtol=0,
    )


def test_auto_and_monte_carlo_covariance_methods_return_covariance() -> None:
    orbits = _covariance_orbits()
    propagator = ASSISTPropagator()
    for method in ("auto", "monte-carlo"):
        result = propagator.propagate_orbits(
            orbits,
            orbits.coordinates.time,
            covariance=True,
            covariance_method=method,
            num_samples=16,
            seed=123,
            max_processes=1,
            chunk_size=10,
        )
        assert not result.coordinates.covariance.is_all_nan()
        assert np.isfinite(result.coordinates.covariance.to_matrix()).all()
