from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import adam_core.photometry as photometry
from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import Origin
from ...observations.detections import PointSourceDetections
from ...observations.exposures import Exposures
from ...observers.observers import Observers
from ...time import Timestamp
from .. import GroupedRotationPeriodResults, RotationPeriodObservations, RotationPeriodResult

FAST_SEARCH_KWARGS = {
    "max_frequency_cycles_per_day": 1.0,
    "frequency_grid_scale": 40.0,
}


def _api(name: str):
    value = getattr(photometry, name, None)
    if value is None:
        pytest.skip(f"{name} is not implemented yet")
    return value


def _scalar(value) -> object:
    if hasattr(value, "as_py"):
        return value.as_py()
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size != 1:
        raise ValueError(f"Expected scalar-like value, got shape {arr.shape}")
    return arr.reshape(-1)[0].item()


def _make_times(n: int, *, start_mjd: float = 60000.0, span_days: float = 12.0) -> Timestamp:
    mjd = np.linspace(start_mjd, start_mjd + span_days, n, dtype=np.float64)
    return Timestamp.from_mjd(mjd, scale="tdb")


def _make_heliocentric_state(
    *,
    times: Timestamp,
    object_x: np.ndarray,
    object_y: np.ndarray,
    observer_x: np.ndarray,
    observer_y: np.ndarray,
) -> tuple[CartesianCoordinates, Observers]:
    n = len(times)
    object_coords = CartesianCoordinates.from_kwargs(
        x=np.asarray(object_x, dtype=np.float64),
        y=np.asarray(object_y, dtype=np.float64),
        z=np.zeros(n, dtype=np.float64),
        vx=np.zeros(n, dtype=np.float64),
        vy=np.zeros(n, dtype=np.float64),
        vz=np.zeros(n, dtype=np.float64),
        time=times,
        frame="ecliptic",
        origin=Origin.from_kwargs(code=["SUN"] * n),
    )
    observers = Observers.from_kwargs(
        code=["500"] * n,
        coordinates=CartesianCoordinates.from_kwargs(
            x=np.asarray(observer_x, dtype=np.float64),
            y=np.asarray(observer_y, dtype=np.float64),
            z=np.zeros(n, dtype=np.float64),
            vx=np.zeros(n, dtype=np.float64),
            vy=np.zeros(n, dtype=np.float64),
            vz=np.zeros(n, dtype=np.float64),
            time=times,
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"] * n),
        ),
    )
    return object_coords, observers


def _phase_angle_deg(object_coords: CartesianCoordinates, observers: Observers) -> np.ndarray:
    obj = np.asarray(object_coords.r, dtype=np.float64)
    obs = np.asarray(observers.coordinates.r, dtype=np.float64)
    delta = obj - obs
    r = np.linalg.norm(obj, axis=1)
    d = np.linalg.norm(delta, axis=1)
    sun = np.linalg.norm(obs, axis=1)
    cos_alpha = np.clip((r * r + d * d - sun * sun) / (2.0 * r * d), -1.0, 1.0)
    return np.degrees(np.arccos(cos_alpha))


def _make_rotation_observations(
    *,
    n: int = 36,
    period_days: float = 3.2,
    filters: list[str] | None = None,
    session_ids: list[str | None] | None = None,
    filter_offsets: dict[str, float] | None = None,
    session_offsets: dict[str, float] | None = None,
    amplitude: float = 0.35,
    single_peaked: bool = False,
    include_phase_terms: bool = True,
    noise_sigma: float = 0.01,
    outlier_indices: tuple[int, ...] = (),
    outlier_offset: float = 1.5,
    mag_sigma: float | None = 0.05,
) -> RotationPeriodObservations:
    if filters is not None:
        n = len(filters)
    times = _make_times(n)
    t_days = np.asarray(times.mjd(), dtype=np.float64) - float(np.asarray(times.mjd(), dtype=np.float64)[0])
    object_coords, observers = _make_heliocentric_state(
        times=times,
        object_x=2.0 + 0.05 * np.sin(2.0 * np.pi * t_days / 9.0),
        object_y=0.2 * np.cos(2.0 * np.pi * t_days / 7.0),
        observer_x=np.full(n, 1.0, dtype=np.float64),
        observer_y=np.zeros(n, dtype=np.float64),
    )
    alpha = _phase_angle_deg(object_coords, observers)
    reduced = np.full(n, 15.0, dtype=np.float64)
    if include_phase_terms:
        reduced = reduced + 0.015 * alpha + 0.0015 * np.square(alpha)

    phase = 2.0 * np.pi * t_days / period_days
    if single_peaked:
        reduced = reduced + amplitude * np.cos(phase)
    else:
        reduced = (
            reduced
            + 0.12 * np.cos(phase)
            + amplitude * np.cos(2.0 * phase)
            + 0.08 * np.sin(2.0 * phase)
        )

    if filters is None:
        filters = ["LSST_r"] * n
    if session_ids is None:
        session_ids = [None] * n
    elif len(session_ids) != n:
        raise ValueError("session_ids must align with the observation count")
    if filter_offsets is None:
        filter_offsets = {f: 0.0 for f in set(filters)}
    if session_offsets is None:
        session_offsets = {}

    reduced = reduced + np.asarray([filter_offsets.get(f, 0.0) for f in filters], dtype=np.float64)
    reduced = reduced + np.asarray(
        [0.0 if sid is None else session_offsets.get(sid, 0.0) for sid in session_ids],
        dtype=np.float64,
    )
    r_au = np.linalg.norm(np.asarray(object_coords.r, dtype=np.float64), axis=1)
    delta_au = np.linalg.norm(
        np.asarray(object_coords.r, dtype=np.float64)
        - np.asarray(observers.coordinates.r, dtype=np.float64),
        axis=1,
    )
    distance_term = 5.0 * np.log10(r_au * delta_au)

    rng = np.random.default_rng(20260326)
    mag = reduced + distance_term + rng.normal(0.0, noise_sigma, size=n)
    if outlier_indices:
        mag = mag.copy()
        for idx in outlier_indices:
            mag[idx] += outlier_offset

    if mag_sigma is None:
        sigma = None
    else:
        sigma = np.full(n, mag_sigma, dtype=np.float64)

    return RotationPeriodObservations.from_kwargs(
        time=times,
        mag=mag,
        mag_sigma=sigma,
        filter=filters,
        session_id=session_ids,
        r_au=r_au,
        delta_au=delta_au,
        phase_angle_deg=alpha,
    )


def _make_detection_bundle(
    *,
    object_ids: list[str],
    periods_by_object: dict[str, float],
    filters: list[str],
    mag_sigma: float = 0.05,
    single_peaked_objects: set[str] | None = None,
) -> tuple[PointSourceDetections, Exposures, CartesianCoordinates, Observers]:
    if single_peaked_objects is None:
        single_peaked_objects = set()

    rows_per_object = len(filters)
    total = rows_per_object * len(object_ids)
    times = _make_times(total, span_days=10.0)
    t_days = np.asarray(times.mjd(), dtype=np.float64) - float(np.asarray(times.mjd(), dtype=np.float64)[0])
    object_coords = []
    exposure_ids = []
    det_ids = []
    det_filters = []
    obj_labels = []
    mag = []
    sigma = np.full(total, mag_sigma, dtype=np.float64)

    for obj_idx, object_id in enumerate(object_ids):
        start = obj_idx * rows_per_object
        stop = start + rows_per_object
        local_times = times.take(pa.array(np.arange(start, stop, dtype=np.int32)))
        local_t_days = t_days[start:stop] - t_days[start]
        period_days = periods_by_object[object_id]
        local_object = CartesianCoordinates.from_kwargs(
            x=np.full(rows_per_object, 2.0 + 0.1 * obj_idx, dtype=np.float64),
            y=0.15 * np.sin(2.0 * np.pi * local_t_days / (period_days * 1.7)),
            z=np.zeros(rows_per_object, dtype=np.float64),
            vx=np.zeros(rows_per_object, dtype=np.float64),
            vy=np.zeros(rows_per_object, dtype=np.float64),
            vz=np.zeros(rows_per_object, dtype=np.float64),
            time=local_times,
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"] * rows_per_object),
        )
        local_observers = Observers.from_kwargs(
            code=["500"] * rows_per_object,
            coordinates=CartesianCoordinates.from_kwargs(
                x=np.full(rows_per_object, 1.0, dtype=np.float64),
                y=np.zeros(rows_per_object, dtype=np.float64),
                z=np.zeros(rows_per_object, dtype=np.float64),
                vx=np.zeros(rows_per_object, dtype=np.float64),
                vy=np.zeros(rows_per_object, dtype=np.float64),
                vz=np.zeros(rows_per_object, dtype=np.float64),
                time=local_times,
                frame="ecliptic",
                origin=Origin.from_kwargs(code=["SUN"] * rows_per_object),
            ),
        )
        alpha = _phase_angle_deg(local_object, local_observers)
        t_local = local_t_days
        phase = 2.0 * np.pi * t_local / period_days
        if object_id in single_peaked_objects:
            reduced = 15.0 + 0.015 * alpha + 0.0015 * np.square(alpha) + 0.3 * np.cos(phase)
        else:
            reduced = (
                15.0
                + 0.015 * alpha
                + 0.0015 * np.square(alpha)
                + 0.12 * np.cos(phase)
                + 0.3 * np.cos(2.0 * phase)
                + 0.05 * np.sin(2.0 * phase)
            )
        reduced = reduced + np.asarray([0.02 if f == "LSST_i" else 0.0 for f in filters], dtype=np.float64)
        r_au = np.linalg.norm(np.asarray(local_object.r, dtype=np.float64), axis=1)
        delta_au = np.linalg.norm(
            np.asarray(local_object.r, dtype=np.float64)
            - np.asarray(local_observers.coordinates.r, dtype=np.float64),
            axis=1,
        )
        distance_term = 5.0 * np.log10(r_au * delta_au)
        rng = np.random.default_rng(20260326 + obj_idx)
        obs_mag = reduced + distance_term + rng.normal(0.0, 0.01, size=rows_per_object)

        object_coords.append(local_object)
        exposure_ids.extend([f"e_{object_id}_{i}" for i in range(rows_per_object)])
        det_ids.extend([f"d_{object_id}_{i}" for i in range(rows_per_object)])
        det_filters.extend(filters)
        obj_labels.extend([object_id] * rows_per_object)
        mag.extend(obs_mag.tolist())

    object_coords = CartesianCoordinates.from_kwargs(
        x=np.concatenate([np.asarray(chunk.r[:, 0], dtype=np.float64) for chunk in object_coords]),
        y=np.concatenate([np.asarray(chunk.r[:, 1], dtype=np.float64) for chunk in object_coords]),
        z=np.concatenate([np.asarray(chunk.r[:, 2], dtype=np.float64) for chunk in object_coords]),
        vx=np.zeros(total, dtype=np.float64),
        vy=np.zeros(total, dtype=np.float64),
        vz=np.zeros(total, dtype=np.float64),
        time=times,
        frame="ecliptic",
        origin=Origin.from_kwargs(code=["SUN"] * total),
    )
    observer = Observers.from_kwargs(
        code=["500"] * total,
        coordinates=CartesianCoordinates.from_kwargs(
            x=np.ones(total, dtype=np.float64),
            y=np.zeros(total, dtype=np.float64),
            z=np.zeros(total, dtype=np.float64),
            vx=np.zeros(total, dtype=np.float64),
            vy=np.zeros(total, dtype=np.float64),
            vz=np.zeros(total, dtype=np.float64),
            time=times,
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"] * total),
        ),
    )
    exposures = Exposures.from_kwargs(
        id=exposure_ids,
        start_time=times,
        duration=np.zeros(total, dtype=np.float64),
        filter=det_filters,
        observatory_code=["500"] * total,
        seeing=[None] * total,
        depth_5sigma=[None] * total,
    )
    detections = PointSourceDetections.from_kwargs(
        id=det_ids,
        exposure_id=exposure_ids,
        time=times,
        ra=np.zeros(total, dtype=np.float64),
        dec=np.zeros(total, dtype=np.float64),
        mag=np.asarray(mag, dtype=np.float64),
        mag_sigma=sigma,
    )
    return detections, exposures, object_coords, observer


def _patch_exposure_observers(monkeypatch, observer: Observers) -> None:
    def fake_observers(self, *args, **kwargs):  # noqa: ARG001
        idx = pa.array(np.arange(len(self), dtype=np.int32))
        return observer.take(idx)

    monkeypatch.setattr(Exposures, "observers", fake_observers)


def test_estimate_rotation_period_recovers_single_filter_period():
    estimate_rotation_period = _api("estimate_rotation_period")
    observations = _make_rotation_observations(period_days=3.25, filters=["LSST_r"] * 48)

    result = estimate_rotation_period(observations, **FAST_SEARCH_KWARGS)

    assert isinstance(result, RotationPeriodResult)
    assert len(result) == 1
    assert float(_scalar(result.period_days[0])) == pytest.approx(3.25, rel=0.03)
    assert float(_scalar(result.period_hours[0])) == pytest.approx(3.25 * 24.0, rel=0.03)
    assert float(_scalar(result.frequency_cycles_per_day[0])) == pytest.approx(1.0 / 3.25, rel=0.03)
    assert int(_scalar(result.fourier_order[0])) == 2
    assert int(_scalar(result.n_observations[0])) == len(observations)
    assert int(_scalar(result.n_fit_observations[0])) >= len(observations) - 2


def test_estimate_rotation_period_handles_multifilter_offsets():
    estimate_rotation_period = _api("estimate_rotation_period")
    filters = ["LSST_r", "LSST_i"] * 24
    observations = _make_rotation_observations(
        period_days=4.0,
        filters=filters,
        filter_offsets={"LSST_r": 0.0, "LSST_i": 0.18},
    )

    result = estimate_rotation_period(observations, **FAST_SEARCH_KWARGS)

    assert isinstance(result, RotationPeriodResult)
    assert len(result) == 1
    assert float(_scalar(result.period_days[0])) == pytest.approx(4.0, rel=0.03)
    assert int(_scalar(result.n_filters[0])) == 2
    assert int(_scalar(result.fourier_order[0])) == 2


def test_estimate_rotation_period_handles_session_offsets():
    estimate_rotation_period = _api("estimate_rotation_period")
    observations = _make_rotation_observations(
        period_days=2.7,
        filters=["LSST_r"] * 48,
        session_ids=(["s1"] * 16) + (["s2"] * 16) + (["s3"] * 16),
        session_offsets={"s2": 0.18, "s3": -0.12},
    )

    result = estimate_rotation_period(observations, **FAST_SEARCH_KWARGS)

    assert isinstance(result, RotationPeriodResult)
    assert len(result) == 1
    assert float(_scalar(result.period_days[0])) == pytest.approx(2.7, rel=0.03)
    assert int(_scalar(result.n_filters[0])) == 1


def test_estimate_rotation_period_doubles_single_peaked_alias():
    estimate_rotation_period = _api("estimate_rotation_period")
    observations = _make_rotation_observations(
        period_days=2.2,
        filters=["LSST_r"] * 40,
        single_peaked=True,
        amplitude=0.28,
    )

    result = estimate_rotation_period(observations, **FAST_SEARCH_KWARGS)

    assert isinstance(result, RotationPeriodResult)
    assert len(result) == 1
    assert float(_scalar(result.period_days[0])) == pytest.approx(4.4, rel=0.04)
    assert bool(_scalar(result.is_period_doubled[0])) is True


def test_estimate_rotation_period_rejects_outliers_with_clipping():
    estimate_rotation_period = _api("estimate_rotation_period")
    observations = _make_rotation_observations(
        period_days=3.6,
        filters=["LSST_r"] * 60,
        outlier_indices=(4, 11, 29),
        outlier_offset=3.0,
        noise_sigma=0.008,
    )

    result = estimate_rotation_period(
        observations,
        **FAST_SEARCH_KWARGS,
        clip_sigma=2.0,
    )

    assert isinstance(result, RotationPeriodResult)
    assert len(result) == 1
    assert float(_scalar(result.period_days[0])) == pytest.approx(3.6, rel=0.05)


def test_estimate_rotation_period_prefers_minimum_sufficient_order():
    estimate_rotation_period = _api("estimate_rotation_period")
    observations = _make_rotation_observations(
        period_days=5.0,
        filters=["LSST_r"] * 60,
        amplitude=0.25,
        noise_sigma=0.002,
    )

    result = estimate_rotation_period(observations, **FAST_SEARCH_KWARGS)

    assert isinstance(result, RotationPeriodResult)
    assert len(result) == 1
    assert int(_scalar(result.fourier_order[0])) == 2
    assert float(_scalar(result.period_days[0])) == pytest.approx(5.0, rel=0.03)


def test_build_rotation_period_observations_from_detections(monkeypatch):
    build = _api("build_rotation_period_observations_from_detections")
    times = _make_times(24, span_days=8.0)
    detections, exposures, object_coords, observer = _make_detection_bundle(
        object_ids=["A"],
        periods_by_object={"A": 3.0},
        filters=["LSST_r", "LSST_i", "LSST_r", "LSST_i"] * 6,
    )
    _patch_exposure_observers(monkeypatch, observer)

    observations = build(detections, exposures, object_coords)

    assert isinstance(observations, RotationPeriodObservations)
    assert len(observations) == len(detections)
    assert np.all(np.isfinite(np.asarray(observations.r_au, dtype=np.float64)))
    assert np.all(np.isfinite(np.asarray(observations.delta_au, dtype=np.float64)))
    assert np.all(np.isfinite(np.asarray(observations.phase_angle_deg, dtype=np.float64)))


def test_estimate_rotation_period_from_detections(monkeypatch):
    estimate_from_detections = _api("estimate_rotation_period_from_detections")
    detections, exposures, object_coords, observer = _make_detection_bundle(
        object_ids=["A"],
        periods_by_object={"A": 3.4},
        filters=["LSST_r", "LSST_i", "LSST_r", "LSST_i"] * 8,
    )
    _patch_exposure_observers(monkeypatch, observer)

    result = estimate_from_detections(
        detections,
        exposures,
        object_coords,
        **FAST_SEARCH_KWARGS,
    )

    assert isinstance(result, RotationPeriodResult)
    assert len(result) == 1
    assert float(_scalar(result.period_days[0])) == pytest.approx(3.4, rel=0.04)


def test_estimate_rotation_period_grouped_from_detections(monkeypatch):
    grouped_api = _api("estimate_rotation_period_from_detections_grouped")
    detections, exposures, object_coords, observer = _make_detection_bundle(
        object_ids=["A", "B"],
        periods_by_object={"A": 3.4, "B": 4.6},
        filters=["LSST_r", "LSST_i", "LSST_r", "LSST_i"] * 8,
    )
    _patch_exposure_observers(monkeypatch, observer)

    result = grouped_api(
        detections,
        exposures,
        object_coords,
        pa.array(["A"] * 32 + ["B"] * 32, type=pa.large_string()),
        **FAST_SEARCH_KWARGS,
    )

    assert isinstance(result, GroupedRotationPeriodResults)
    assert len(result) == 2
    rows = {str(row["object_id"]): row for row in result.table.to_pylist()}
    assert float(rows["A"]["result"]["period_days"]) == pytest.approx(3.4, rel=0.04)
    assert float(rows["B"]["result"]["period_days"]) == pytest.approx(4.6, rel=0.04)


def test_rotation_period_invalid_inputs_raise(monkeypatch):
    build = _api("build_rotation_period_observations_from_detections")
    estimate_rotation_period = _api("estimate_rotation_period")

    detections, exposures, object_coords, observer = _make_detection_bundle(
        object_ids=["A"],
        periods_by_object={"A": 3.1},
        filters=["LSST_r"] * 8,
    )
    _patch_exposure_observers(monkeypatch, observer)

    with pytest.raises(ValueError):
        build(
            detections,
            exposures,
            object_coords.set_column(
                "origin",
                Origin.from_kwargs(code=["EARTH"] * len(object_coords)),
            ),
        )

    with pytest.raises(ValueError):
        build(
            detections.set_column("exposure_id", pa.array(["missing"] * len(detections), type=pa.large_string())),
            exposures,
            object_coords,
        )

    with pytest.raises(ValueError):
        build(
            detections,
            exposures,
            object_coords.take(pa.array(np.arange(len(object_coords) - 1), type=pa.int32())),
        )

    too_few = _make_rotation_observations(n=3, period_days=2.4, filters=["LSST_r"] * 3)
    with pytest.raises(ValueError):
        estimate_rotation_period(too_few, **FAST_SEARCH_KWARGS)


def test_rotation_period_uses_unit_weights_when_sigma_is_invalid():
    estimate_rotation_period = _api("estimate_rotation_period")
    observations = _make_rotation_observations(
        period_days=3.9,
        filters=["LSST_r"] * 42,
        mag_sigma=None,
        noise_sigma=0.006,
    )
    observations = observations.set_column(
        "mag_sigma",
        pa.array([None, np.nan, -1.0] + [0.05] * (len(observations) - 3), type=pa.float64()),
    )

    result = estimate_rotation_period(observations, **FAST_SEARCH_KWARGS)

    assert isinstance(result, RotationPeriodResult)
    assert len(result) == 1
    assert float(_scalar(result.period_days[0])) == pytest.approx(3.9, rel=0.05)
