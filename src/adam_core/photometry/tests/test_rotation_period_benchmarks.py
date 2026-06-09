from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import Origin
from ...observations.detections import PointSourceDetections
from ...observations.exposures import Exposures
from ...observers.observers import Observers
from ...time import Timestamp
from ..rotation_period_wrappers import estimate_rotation_period_from_detections_grouped


def _phase_angle_deg(object_coords: CartesianCoordinates, observers: Observers) -> np.ndarray:
    obj = np.asarray(object_coords.r, dtype=np.float64)
    obs = np.asarray(observers.coordinates.r, dtype=np.float64)
    delta = obj - obs
    r = np.linalg.norm(obj, axis=1)
    d = np.linalg.norm(delta, axis=1)
    sun = np.linalg.norm(obs, axis=1)
    cos_alpha = np.clip((r * r + d * d - sun * sun) / (2.0 * r * d), -1.0, 1.0)
    return np.degrees(np.arccos(cos_alpha))


def _make_detection_bundle() -> tuple[PointSourceDetections, Exposures, CartesianCoordinates, Observers, pa.Array]:
    n_per_object = 32
    object_ids = (["a"] * n_per_object) + (["b"] * n_per_object)
    periods = {"a": 0.02, "b": 0.025}
    filters = (["LSST_r", "LSST_g"] * n_per_object)[:n_per_object] + (["LSST_r", "LSST_g"] * n_per_object)[:n_per_object]
    mjd = np.linspace(60100.0, 60100.08, len(object_ids), dtype=np.float64)
    times = Timestamp.from_mjd(mjd, scale="tdb")
    object_x = np.asarray([2.0 if oid == "a" else 2.2 for oid in object_ids], dtype=np.float64)
    object_y = np.asarray([0.1 if oid == "a" else 0.15 for oid in object_ids], dtype=np.float64)
    object_coords = CartesianCoordinates.from_kwargs(
        x=object_x,
        y=object_y,
        z=np.zeros(len(object_ids), dtype=np.float64),
        vx=np.zeros(len(object_ids), dtype=np.float64),
        vy=np.zeros(len(object_ids), dtype=np.float64),
        vz=np.zeros(len(object_ids), dtype=np.float64),
        time=times,
        frame="ecliptic",
        origin=Origin.from_kwargs(code=["SUN"] * len(object_ids)),
    )
    observers = Observers.from_kwargs(
        code=["X05"] * len(object_ids),
        coordinates=CartesianCoordinates.from_kwargs(
            x=np.ones(len(object_ids), dtype=np.float64),
            y=np.zeros(len(object_ids), dtype=np.float64),
            z=np.zeros(len(object_ids), dtype=np.float64),
            vx=np.zeros(len(object_ids), dtype=np.float64),
            vy=np.zeros(len(object_ids), dtype=np.float64),
            vz=np.zeros(len(object_ids), dtype=np.float64),
            time=times,
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"] * len(object_ids)),
        ),
    )
    phase_angle = _phase_angle_deg(object_coords, observers)
    r_au = np.linalg.norm(np.asarray(object_coords.r, dtype=np.float64), axis=1)
    delta_au = np.linalg.norm(
        np.asarray(object_coords.r, dtype=np.float64) - np.asarray(observers.coordinates.r, dtype=np.float64),
        axis=1,
    )
    baseline = 15.0 + 5.0 * np.log10(r_au * delta_au) + 0.015 * phase_angle + 0.0015 * np.square(phase_angle)
    t_rel = mjd - mjd.min()
    mag = baseline.copy()
    for idx, oid in enumerate(object_ids):
        phase = 2.0 * np.pi * t_rel[idx] / periods[oid]
        mag[idx] += 0.10 * np.cos(phase) + 0.30 * np.cos(2.0 * phase) + 0.05 * np.sin(2.0 * phase)
    exposures = Exposures.from_kwargs(
        id=[f"e{idx}" for idx in range(len(object_ids))],
        start_time=times,
        duration=np.zeros(len(object_ids), dtype=np.float64),
        filter=filters,
        observatory_code=["X05"] * len(object_ids),
        seeing=[None] * len(object_ids),
        depth_5sigma=[None] * len(object_ids),
    )
    detections = PointSourceDetections.from_kwargs(
        id=[f"d{idx}" for idx in range(len(object_ids))],
        exposure_id=[f"e{idx}" for idx in range(len(object_ids))],
        time=times,
        ra=np.zeros(len(object_ids), dtype=np.float64),
        dec=np.zeros(len(object_ids), dtype=np.float64),
        mag=mag,
        mag_sigma=np.full(len(object_ids), 0.03, dtype=np.float64),
    )
    return detections, exposures, object_coords, observers, pa.array(object_ids, type=pa.large_string())


def test_exact_grid_reference_recovers_fast_period(monkeypatch):
    detections, exposures, object_coords, observers, object_ids = _make_detection_bundle()

    def fake_observers(self, *args, **kwargs):  # noqa: ARG001
        return observers

    monkeypatch.setattr(Exposures, "observers", fake_observers)
    result = estimate_rotation_period_from_detections_grouped(
        detections,
        exposures,
        object_coords,
        object_ids=object_ids,
        method_mode="fourier",
        search_fidelity="exact_grid",
        max_frequency_cycles_per_day=120.0,
        frequency_grid_scale=40.0,
    )

    assert len(result) == 2
    periods = [result.result.period_days[idx].as_py() for idx in range(len(result))]
    assert periods[0] == pytest.approx(0.02, rel=0.15)
    assert periods[1] == pytest.approx(0.025, rel=0.15)


def test_exact_and_staged_reference_match_on_same_case(monkeypatch):
    detections, exposures, object_coords, observers, object_ids = _make_detection_bundle()

    def fake_observers(self, *args, **kwargs):  # noqa: ARG001
        return observers

    monkeypatch.setattr(Exposures, "observers", fake_observers)
    grouped_exact = estimate_rotation_period_from_detections_grouped(
        detections,
        exposures,
        object_coords,
        object_ids=object_ids,
        method_mode="fourier",
        search_fidelity="exact_grid",
        max_frequency_cycles_per_day=120.0,
        frequency_grid_scale=40.0,
    )
    grouped_staged = estimate_rotation_period_from_detections_grouped(
        detections,
        exposures,
        object_coords,
        object_ids=object_ids,
        method_mode="fourier",
        search_fidelity="validated_staged",
        max_frequency_cycles_per_day=120.0,
        frequency_grid_scale=40.0,
    )

    for idx in range(len(grouped_exact)):
        assert grouped_exact.result.period_days[idx].as_py() == pytest.approx(
            grouped_staged.result.period_days[idx].as_py(),
            rel=0.05,
        )


def test_from_point_source_observations_classmethod_matches_wrapper(monkeypatch):
    # RotationPeriodObservations.from_point_source_observations is the explicit
    # constructor linking the table to the adam_core observation primitives; it
    # delegates to build_rotation_period_observations_from_detections, so the two
    # must produce identical observations from the same detections/exposures/coords.
    detections, exposures, object_coords, observers, _ = _make_detection_bundle()

    def fake_observers(self, *args, **kwargs):  # noqa: ARG001
        return observers

    monkeypatch.setattr(Exposures, "observers", fake_observers)

    from ..rotation_period_types import RotationPeriodObservations
    from ..rotation_period_wrappers import (
        build_rotation_period_observations_from_detections,
    )

    via_method = RotationPeriodObservations.from_point_source_observations(
        detections, exposures, object_coords
    )
    via_wrapper = build_rotation_period_observations_from_detections(
        detections, exposures, object_coords
    )

    assert isinstance(via_method, RotationPeriodObservations)
    assert len(via_method) == len(detections)
    assert via_method.table.equals(via_wrapper.table)
