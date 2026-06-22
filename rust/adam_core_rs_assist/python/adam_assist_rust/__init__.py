"""Experimental Rust-backed ASSIST adapter.

This package is the GPL Python boundary for benchmarking the private
``assist-rs`` adapter against the public ``adam_assist.ASSISTPropagator``
semantics. It currently covers public orbit propagation plus sampled covariance
expansion/collapse; ephemeris and collision parity remain separate
RM-STANDALONE-007B work items.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.coordinates.origin import Origin
from adam_core.coordinates.spherical import SphericalCoordinates
from adam_core.observers.observers import Observers
from adam_core.orbits.ephemeris import Ephemeris
from adam_core.orbits.orbits import Orbits
from adam_core.orbits.variants import VariantOrbits
from adam_core.time import Timestamp
from jpl_small_bodies_de441_n16 import de441_n16
from naif_de440 import de440

from ._native import NativeAssistPropagator

OrbitTable = Orbits | VariantOrbits


def _column_to_list(column: Any) -> list[Any]:
    if hasattr(column, "to_pylist"):
        return column.to_pylist()
    if hasattr(column, "to_numpy"):
        return column.to_numpy(zero_copy_only=False).tolist()
    return list(column)


def _string_column_to_list(column: Any) -> list[str]:
    return [str(value) for value in _column_to_list(column)]


def _optional_string_column_to_list(column: Any) -> list[str | None]:
    return [None if value is None else str(value) for value in _column_to_list(column)]


def _optional_float_column_to_list(column: Any) -> list[float | None]:
    return [
        None if value is None else float(value) for value in _column_to_list(column)
    ]


def _time_parts(times: Timestamp) -> tuple[str, list[int], list[int]]:
    days = times.days.to_numpy(zero_copy_only=False).astype(np.int64).tolist()
    nanos = times.nanos.to_numpy(zero_copy_only=False).astype(np.int64).tolist()
    return times.scale, days, nanos


def _output_times(result: dict[str, Any]) -> Timestamp:
    return Timestamp.from_kwargs(
        days=result["time_days"],
        nanos=result["time_nanos"],
        scale=result["time_scale"],
    )


def _raise_if_failed_rows(result: dict[str, Any]) -> None:
    validity = result["validity"]
    if all(validity):
        return
    messages = result["messages"]
    first_failure = next(
        (message for is_valid, message in zip(validity, messages) if not is_valid),
        "assist-rs propagation failed",
    )
    raise RuntimeError(first_failure)


def _physical_parameters_for_output(orbits: OrbitTable, result: dict[str, Any]) -> Any:
    input_indices = np.asarray(result["input_orbit_indices"], dtype=np.int64)
    return orbits.physical_parameters.take(input_indices)


def _coordinates_from_result(result: dict[str, Any]) -> CartesianCoordinates:
    states = np.asarray(result["states"], dtype=np.float64)
    covariances = result.get("covariances")
    covariance = None
    if covariances is not None:
        covariance_rows = np.asarray(covariances, dtype=np.float64).reshape(
            states.shape[0], 6, 6
        )
        covariance = CoordinateCovariances.from_matrix(covariance_rows)
    return CartesianCoordinates.from_kwargs(
        x=states[:, 0],
        y=states[:, 1],
        z=states[:, 2],
        vx=states[:, 3],
        vy=states[:, 4],
        vz=states[:, 5],
        covariance=covariance,
        time=_output_times(result),
        origin=Origin.from_kwargs(code=result["origin_codes"]),
        frame=result["frame"],
    )


def _ephemeris_from_result(result: dict[str, Any]) -> Ephemeris:
    states = np.asarray(result["states"], dtype=np.float64)
    coordinates = SphericalCoordinates.from_kwargs(
        rho=states[:, 0],
        lon=states[:, 1],
        lat=states[:, 2],
        vrho=states[:, 3],
        vlon=states[:, 4],
        vlat=states[:, 5],
        time=Timestamp.from_kwargs(
            days=result["time_days"],
            nanos=result["time_nanos"],
            scale=result["time_scale"],
        ),
        origin=Origin.from_kwargs(code=result["origin_codes"]),
        frame=result["frame"],
    )
    kwargs: dict[str, Any] = {
        "orbit_id": result["orbit_id"],
        "object_id": result["object_id"],
        "coordinates": coordinates,
        "light_time": result["light_time"],
    }
    if result["alpha"] is not None:
        kwargs["alpha"] = result["alpha"]
    if result["predicted_magnitude_v"] is not None:
        kwargs["predicted_magnitude_v"] = result["predicted_magnitude_v"]
    if result["aberrated_states"] is not None:
        aberrated_states = np.asarray(result["aberrated_states"], dtype=np.float64)
        kwargs["aberrated_coordinates"] = CartesianCoordinates.from_kwargs(
            x=aberrated_states[:, 0],
            y=aberrated_states[:, 1],
            z=aberrated_states[:, 2],
            vx=aberrated_states[:, 3],
            vy=aberrated_states[:, 4],
            vz=aberrated_states[:, 5],
            time=Timestamp.from_kwargs(
                days=result["aberrated_time_days"],
                nanos=result["aberrated_time_nanos"],
                scale=result["aberrated_time_scale"],
            ),
            origin=Origin.from_kwargs(code=result["aberrated_origin_codes"]),
            frame="ecliptic",
        )
    return Ephemeris.from_kwargs(**kwargs)


class ASSISTPropagator:
    """Rust-backed propagation subset of ``adam_assist.ASSISTPropagator``.

    The public method mirrors the Python propagator's ``propagate_orbits``
    signature for state propagation. ``max_processes`` controls the Rust Rayon
    thread limit rather than launching Ray workers; this keeps the benchmark
    Python-callable while measuring the Rust adapter boundary directly.
    """

    def __init__(
        self,
        *,
        planets_path: str | Path | None = None,
        asteroids_path: str | Path | None = None,
        min_dt: float = 1e-9,
        initial_dt: float = 1e-6,
        adaptive_mode: int = 1,
        epsilon: float = 1e-6,
    ) -> None:
        self.planets_path = str(planets_path if planets_path is not None else de440)
        self.asteroids_path = str(
            asteroids_path if asteroids_path is not None else de441_n16
        )
        self.min_dt = min_dt
        self.initial_dt = initial_dt
        self.adaptive_mode = adaptive_mode
        self.epsilon = epsilon
        self._native = NativeAssistPropagator(
            self.planets_path,
            self.asteroids_path,
            min_dt=min_dt,
            initial_dt=initial_dt,
            adaptive_mode=adaptive_mode,
            epsilon=epsilon,
        )

    def propagate_orbits(
        self,
        orbits: OrbitTable,
        times: Timestamp,
        covariance: bool = False,
        covariance_method: Literal[
            "auto", "sigma-point", "monte-carlo"
        ] = "monte-carlo",
        num_samples: int = 1000,
        chunk_size: int | None = 100,
        max_processes: int | None = 1,
        seed: int | None = None,
    ) -> OrbitTable:
        if covariance and isinstance(orbits, VariantOrbits):
            raise AssertionError("Covariance is not supported for VariantOrbits")
        sorted_times = times.sort_by(["days", "nanos"])
        result = self._propagate_orbits_native(
            orbits,
            sorted_times,
            covariance=covariance,
            covariance_method=covariance_method,
            num_samples=num_samples,
            seed=seed,
            chunk_size=chunk_size,
            thread_limit=max_processes,
        )
        if isinstance(result, VariantOrbits):
            return result.sort_by(
                [
                    "orbit_id",
                    "variant_id",
                    "coordinates.time.days",
                    "coordinates.time.nanos",
                ]
            )
        return result.sort_by(
            ["orbit_id", "coordinates.time.days", "coordinates.time.nanos"]
        )

    def generate_ephemeris(
        self,
        orbits: Orbits,
        observers: Observers,
        *,
        lt_tol: float = 1.0e-12,
        max_iter: int = 1000,
        tol: float = 1.0e-15,
        stellar_aberration: bool = False,
        max_lt_iter: int = 10,
        output_time_scale: str | None = None,
        predict_magnitudes: bool = False,
        predict_phase_angle: bool = False,
        chunk_size: int | None = 100,
        max_processes: int | None = 1,
    ) -> Ephemeris:
        """Rust-native ASSIST ephemeris matching adam_assist public semantics.

        Observer Cartesian states come from the ``observers`` table (e.g.
        ``Observers.from_codes``); the Rust path performs the barycentric
        light-time geometry natively. Mirrors ``adam_assist`` light-time-only
        (no stellar aberration) defaults.
        """
        orbit_coordinates = orbits.coordinates
        orbit_scale, orbit_days, orbit_nanos = _time_parts(orbit_coordinates.time)
        orbit_states: npt.NDArray[np.float64] = np.ascontiguousarray(
            orbit_coordinates.values, dtype=np.float64
        )
        observer_coordinates = observers.coordinates
        observer_scale, observer_days, observer_nanos = _time_parts(
            observer_coordinates.time
        )
        observer_states: npt.NDArray[np.float64] = np.ascontiguousarray(
            observer_coordinates.values, dtype=np.float64
        )
        out_scale = output_time_scale or observer_coordinates.time.scale

        h_v: list[float | None] | None = None
        g: list[float | None] | None = None
        if predict_magnitudes:
            h_v = _optional_float_column_to_list(orbits.physical_parameters.H)
            g = _optional_float_column_to_list(orbits.physical_parameters.G)

        native = self._native.generate_ephemeris(
            _string_column_to_list(orbits.orbit_id),
            _optional_string_column_to_list(orbits.object_id),
            orbit_states,
            _string_column_to_list(orbit_coordinates.origin.code),
            orbit_coordinates.frame,
            orbit_scale,
            orbit_days,
            orbit_nanos,
            _string_column_to_list(observers.code),
            observer_states,
            _string_column_to_list(observer_coordinates.origin.code),
            observer_coordinates.frame,
            observer_scale,
            observer_days,
            observer_nanos,
            out_scale,
            lt_tol,
            max_iter,
            tol,
            stellar_aberration,
            max_lt_iter,
            predict_magnitudes,
            predict_phase_angle,
            h_v,
            g,
            chunk_size,
            max_processes,
        )
        return _ephemeris_from_result(native)

    def _propagate_orbits_native(
        self,
        orbits: OrbitTable,
        times: Timestamp,
        *,
        covariance: bool,
        covariance_method: Literal["auto", "sigma-point", "monte-carlo"],
        num_samples: int,
        seed: int | None,
        chunk_size: int | None,
        thread_limit: int | None,
    ) -> OrbitTable:
        coordinates = orbits.coordinates
        input_scale, input_days, input_nanos = _time_parts(coordinates.time)
        target_scale, target_days, target_nanos = _time_parts(times)
        states: npt.NDArray[np.float64] = np.ascontiguousarray(
            coordinates.values, dtype=np.float64
        )
        variant_ids: list[str | None] | None
        weights: list[float | None] | None
        weights_cov: list[float | None] | None
        native_covariances: npt.NDArray[np.float64] | None = None
        native_covariance = False
        if isinstance(orbits, VariantOrbits):
            variant_ids = _optional_string_column_to_list(orbits.variant_id)
            weights = _optional_float_column_to_list(orbits.weights)
            weights_cov = _optional_float_column_to_list(orbits.weights_cov)
        else:
            variant_ids = None
            weights = None
            weights_cov = None
            if covariance and not coordinates.covariance.is_all_nan():
                native_covariance = True
                native_covariances = np.ascontiguousarray(
                    coordinates.covariance.to_matrix().reshape(len(orbits), 36),
                    dtype=np.float64,
                )

        native_result = self._native.propagate_orbits(
            _string_column_to_list(orbits.orbit_id),
            _optional_string_column_to_list(orbits.object_id),
            states,
            _string_column_to_list(coordinates.origin.code),
            coordinates.frame,
            input_scale,
            input_days,
            input_nanos,
            target_scale,
            target_days,
            target_nanos,
            native_covariance,
            covariances=native_covariances,
            covariance_method=covariance_method,
            num_samples=num_samples,
            seed=seed,
            chunk_size=chunk_size,
            thread_limit=thread_limit,
            variant_ids=variant_ids,
            weights=weights,
            weights_cov=weights_cov,
        )
        _raise_if_failed_rows(native_result)
        output_coordinates = _coordinates_from_result(native_result)
        physical_parameters = _physical_parameters_for_output(orbits, native_result)
        if isinstance(orbits, VariantOrbits):
            return VariantOrbits.from_kwargs(
                orbit_id=native_result["orbit_id"],
                object_id=native_result["object_id"],
                variant_id=native_result["variant_id"],
                weights=native_result["weights"],
                weights_cov=native_result["weights_cov"],
                coordinates=output_coordinates,
                physical_parameters=physical_parameters,
            )
        return Orbits.from_kwargs(
            orbit_id=native_result["orbit_id"],
            object_id=native_result["object_id"],
            coordinates=output_coordinates,
            physical_parameters=physical_parameters,
        )


__all__ = ["ASSISTPropagator", "NativeAssistPropagator"]
