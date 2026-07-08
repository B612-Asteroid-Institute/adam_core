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
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv

from adam_core.coordinates.cartesian import CartesianCoordinates
from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.coordinates.spherical import SphericalCoordinates
from adam_core.coordinates.transform import transform_coordinates
from adam_core.dynamics.impacts import CollisionConditions, CollisionEvent, ImpactMixin
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
    spherical_covariance = result.get("covariance")
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
        covariance=(
            CoordinateCovariances.from_matrix(
                np.asarray(spherical_covariance, dtype=np.float64).reshape(-1, 6, 6)
            )
            if spherical_covariance is not None
            else None
        ),
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
        aberrated_covariance = result.get("aberrated_covariance")
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
            covariance=(
                CoordinateCovariances.from_matrix(
                    np.asarray(aberrated_covariance, dtype=np.float64).reshape(-1, 6, 6)
                )
                if aberrated_covariance is not None
                else None
            ),
        )
    return Ephemeris.from_kwargs(**kwargs)


def _collision_rows(
    source: OrbitTable,
    indices: npt.NDArray[np.int64],
    states: npt.NDArray[np.float64],
    times: Timestamp,
) -> OrbitTable:
    """Source rows re-coordinated at collision-loop states/times.

    Preserves orbit/variant identity, weights, and physical parameters from
    the input table while replacing coordinates with barycentric-equatorial
    states from the Rust collision loop (matching the legacy step-loop table
    construction).
    """
    rows = source.take(pa.array(indices, type=pa.int64()))
    coordinates = CartesianCoordinates.from_kwargs(
        x=states[:, 0],
        y=states[:, 1],
        z=states[:, 2],
        vx=states[:, 3],
        vy=states[:, 4],
        vz=states[:, 5],
        time=times,
        origin=Origin.from_kwargs(
            code=pa.repeat("SOLAR_SYSTEM_BARYCENTER", len(states))
        ),
        frame="equatorial",
    )
    return rows.set_column("coordinates", coordinates)


class ASSISTPropagator(ImpactMixin):
    """Rust-backed propagation subset of ``adam_assist.ASSISTPropagator``.

    The public method mirrors the Python propagator's ``propagate_orbits``
    signature for state propagation. ``max_processes`` controls the Rust Rayon
    thread limit rather than launching Ray workers; this keeps the benchmark
    Python-callable while measuring the Rust adapter boundary directly.

    ``ImpactMixin`` support: ``_detect_collisions`` mirrors the Python
    ``adam_assist`` step loop through a single Rust crossing, so
    ``detect_collisions``/``calculate_impacts`` work with this propagator.
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
        covariance: bool = False,
        covariance_method: Literal["auto", "sigma-point", "monte-carlo"] = "monte-carlo",
        num_samples: int = 1000,
        seed: int | None = None,
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
            h_v = _optional_float_column_to_list(orbits.physical_parameters.H_v)
            g = _optional_float_column_to_list(orbits.physical_parameters.G)

        # Covariance ephemeris: pass the orbit covariance matrix so the Rust
        # backend can sample variants + collapse the variant ephemeris to
        # per-row covariance in one crossing (mirrors the propagate path).
        native_covariances: npt.NDArray[np.float64] | None = None
        if covariance and not orbit_coordinates.covariance.is_all_nan():
            native_covariances = np.ascontiguousarray(
                orbit_coordinates.covariance.to_matrix().reshape(len(orbits), 36),
                dtype=np.float64,
            )

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
            covariance=covariance and native_covariances is not None,
            covariances=native_covariances,
            covariance_method=covariance_method,
            num_samples=num_samples,
            seed=seed,
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

    def fit_least_squares(
        self,
        orbit: Orbits,
        observations: Any,
        *,
        xtol: float = 1e-12,
        ftol: float = 1e-12,
        max_iterations: int = 100,
        lt_tol: float = 1.0e-12,
        eph_max_iter: int = 1000,
        eph_tol: float = 1.0e-15,
        stellar_aberration: bool = False,
        max_lt_iter: int = 10,
    ) -> tuple[Orbits, float, int, bool]:
        """Backend-generic Gauss-Newton OD with the ASSIST propagator.

        Mirrors the semantics of
        ``adam_core.orbit_determination.fit_least_squares(orbit, observations,
        propagator)`` for a single orbit: differentially corrects the orbit
        state at its epoch against ``observations``
        (``OrbitDeterminationObservations``: spherical astrometry with
        covariance + observers). The Gauss-Newton driver lives in the
        permissive core; this GPL package only supplies the ASSIST propagator
        (each iteration batches the base + six perturbed candidates into one
        same-epoch multi-particle ephemeris crossing). Returns
        ``(fitted_orbit, chi2, iterations, converged)`` where the fitted orbit
        carries the ``inv(J^T J)`` covariance in the input frame/origin.
        """
        assert len(orbit) == 1, "Only one orbit can be differentially corrected"
        coordinates = orbit.coordinates
        orbit_scale, orbit_days, orbit_nanos = _time_parts(coordinates.time)
        observed = observations.coordinates
        observers = observations.observers
        observer_coordinates = observers.coordinates
        observer_scale, observer_days, observer_nanos = _time_parts(
            observer_coordinates.time
        )
        state, covariance, chi2, iterations, converged = (
            self._native.fit_orbit_least_squares(
                _string_column_to_list(orbit.orbit_id),
                _optional_string_column_to_list(orbit.object_id),
                np.ascontiguousarray(coordinates.values, dtype=np.float64),
                _string_column_to_list(coordinates.origin.code),
                coordinates.frame,
                orbit_scale,
                orbit_days,
                orbit_nanos,
                np.ascontiguousarray(observed.values, dtype=np.float64),
                np.ascontiguousarray(
                    observed.covariance.to_matrix().reshape(len(observed), 36),
                    dtype=np.float64,
                ),
                _string_column_to_list(observers.code),
                np.ascontiguousarray(observer_coordinates.values, dtype=np.float64),
                _string_column_to_list(observer_coordinates.origin.code),
                observer_coordinates.frame,
                observer_scale,
                observer_days,
                observer_nanos,
                xtol=xtol,
                ftol=ftol,
                max_iterations=max_iterations,
                lt_tol=lt_tol,
                eph_max_iter=eph_max_iter,
                eph_tol=eph_tol,
                stellar_aberration=stellar_aberration,
                max_lt_iter=max_lt_iter,
            )
        )
        state = np.asarray(state, dtype=np.float64)
        covariance = np.asarray(covariance, dtype=np.float64).reshape(1, 6, 6)
        fitted = Orbits.from_kwargs(
            orbit_id=orbit.orbit_id,
            object_id=orbit.object_id,
            coordinates=CartesianCoordinates.from_kwargs(
                x=[state[0]],
                y=[state[1]],
                z=[state[2]],
                vx=[state[3]],
                vy=[state[4]],
                vz=[state[5]],
                time=coordinates.time,
                covariance=CoordinateCovariances.from_matrix(covariance),
                origin=coordinates.origin,
                frame=coordinates.frame,
            ),
        )
        return fitted, float(chi2), int(iterations), bool(converged)

    def _detect_collisions(
        self,
        orbits: OrbitTable,
        num_days: int,
        conditions: CollisionConditions,
    ) -> tuple[OrbitTable, CollisionEvent]:
        """Rust-native mirror of ``adam_assist.ASSISTPropagator._detect_collisions``.

        The input transform (SSB/equatorial/TDB), the TDB Julian-date horizon
        arithmetic, and the returned table shapes match the legacy Python
        loop; the per-step integration and distance checks run in one Rust
        crossing. Survivors are reported at the final executed (overshooting)
        integrator step exactly like legacy.
        """
        assert len(pc.unique(orbits.coordinates.time.mjd())) == 1

        coords = transform_coordinates(
            orbits.coordinates,
            origin_out=OriginCodes.SOLAR_SYSTEM_BARYCENTER,
            frame_out="equatorial",
        )
        coords = coords.set_column("time", coords.time.rescale("tdb"))
        orbits = orbits.set_column("coordinates", coords)

        epoch_jd = float(coords.time.jd().to_numpy(zero_copy_only=False)[0])
        final_jd = float(
            coords.time.add_days(num_days).jd().to_numpy(zero_copy_only=False)[0]
        )

        native = self._native.detect_collisions(
            np.ascontiguousarray(coords.values, dtype=np.float64),
            epoch_jd,
            final_jd,
            _string_column_to_list(conditions.collision_object.code),
            [float(value) for value in conditions.collision_distance.to_pylist()],
            [bool(value) for value in conditions.stopping_condition.to_pylist()],
        )

        impact_indices = np.asarray(native["impact_indices"], dtype=np.int64)
        impact_states = np.asarray(native["impact_states"], dtype=np.float64).reshape(
            -1, 6
        )
        impact_condition_indices = np.asarray(
            native["impact_condition_indices"], dtype=np.int64
        )
        impact_times = np.asarray(native["impact_times_jd_tdb"], dtype=np.float64)

        events: list[CollisionEvent] = []
        for condition_index in range(len(conditions)):
            mask = impact_condition_indices == condition_index
            if not mask.any():
                continue
            condition = conditions[condition_index : condition_index + 1]
            rows = _collision_rows(
                orbits,
                impact_indices[mask],
                impact_states[mask],
                Timestamp.from_jd(pa.array(impact_times[mask]), scale="tdb"),
            )
            kwargs: dict[str, Any] = dict(
                orbit_id=rows.orbit_id,
                coordinates=rows.coordinates,
                condition_id=pa.repeat(condition.condition_id[0].as_py(), len(rows)),
                collision_coordinates=transform_coordinates(
                    rows.coordinates,
                    representation_out=SphericalCoordinates,
                    origin_out=condition.collision_object.as_OriginCodes(),
                    frame_out="ecliptic",
                ),
                collision_object=condition.collision_object.take(
                    [0 for _ in range(len(rows))]
                ),
                stopping_condition=pa.repeat(
                    bool(condition.stopping_condition[0].as_py()), len(rows)
                ),
            )
            if isinstance(orbits, VariantOrbits):
                kwargs["variant_id"] = rows.variant_id
            events.append(CollisionEvent.from_kwargs(**kwargs))
        collision_events = qv.concatenate(events) if events else CollisionEvent.empty()

        # Results: rows removed by stopping conditions (at their impact-step
        # states/times, in removal order) followed by the survivors at the
        # final executed step, matching the legacy accumulation order.
        stopping_by_condition = np.asarray(
            [bool(value) for value in conditions.stopping_condition.to_pylist()],
            dtype=bool,
        )
        removed_mask = stopping_by_condition[impact_condition_indices]
        results_parts: list[OrbitTable] = []
        if removed_mask.any():
            results_parts.append(
                _collision_rows(
                    orbits,
                    impact_indices[removed_mask],
                    impact_states[removed_mask],
                    Timestamp.from_jd(
                        pa.array(impact_times[removed_mask]), scale="tdb"
                    ),
                )
            )
        final_indices = np.asarray(native["final_indices"], dtype=np.int64)
        final_states = np.asarray(native["final_states"], dtype=np.float64).reshape(
            -1, 6
        )
        if len(final_indices) > 0:
            final_jds = np.full(len(final_indices), native["final_time_jd_tdb"])
            results_parts.append(
                _collision_rows(
                    orbits,
                    final_indices,
                    final_states,
                    Timestamp.from_jd(pa.array(final_jds), scale="tdb"),
                )
            )
        if results_parts:
            results = (
                qv.concatenate(results_parts)
                if len(results_parts) > 1
                else results_parts[0]
            )
        else:
            results = (
                Orbits.empty() if isinstance(orbits, Orbits) else VariantOrbits.empty()
            )
        return results, collision_events


__all__ = ["ASSISTPropagator", "NativeAssistPropagator"]
