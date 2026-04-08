from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import adam_core.photometry as photometry
import adam_core.photometry.rotation_period_fourier as rotation_period_fourier
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
        observatory_code=["X05"] * total,
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


@pytest.mark.parametrize("search_strategy", ["grid", "surrogate_refine", "coarse_to_fine"])
def test_estimate_rotation_period_search_strategies_match_short_period(search_strategy: str):
    estimate_rotation_period = _api("estimate_rotation_period")
    observations = _make_rotation_observations(period_days=3.25, filters=["LSST_r"] * 48)

    result = estimate_rotation_period(
        observations,
        search_strategy=search_strategy,
        **FAST_SEARCH_KWARGS,
    )

    assert isinstance(result, RotationPeriodResult)
    assert len(result) == 1
    assert float(_scalar(result.period_days[0])) == pytest.approx(3.25, rel=0.03)
    assert int(_scalar(result.fourier_order[0])) == 2


def test_estimate_rotation_period_parallel_surrogate_matches_serial():
    estimate_rotation_period = _api("estimate_rotation_period")
    observations = _make_rotation_observations(period_days=3.25, filters=["LSST_r"] * 48)

    serial = estimate_rotation_period(
        observations,
        search_strategy="surrogate_refine",
        max_processes=None,
        **FAST_SEARCH_KWARGS,
    )
    parallel = estimate_rotation_period(
        observations,
        search_strategy="surrogate_refine",
        max_processes=2,
        **FAST_SEARCH_KWARGS,
    )

    assert float(_scalar(parallel.period_days[0])) == pytest.approx(
        float(_scalar(serial.period_days[0])),
        rel=1.0e-9,
    )
    assert int(_scalar(parallel.fourier_order[0])) == int(_scalar(serial.fourier_order[0]))
    assert bool(_scalar(parallel.used_session_offsets[0])) is bool(
        _scalar(serial.used_session_offsets[0])
    )


def test_estimate_rotation_period_jax_backend_matches_numpy():
    estimate_rotation_period = _api("estimate_rotation_period")
    observations = _make_rotation_observations(period_days=3.25, filters=["LSST_r"] * 48)

    numpy_result = estimate_rotation_period(
        observations,
        search_strategy="surrogate_refine",
        exact_evaluation_backend="numpy",
        session_mode="ignore",
        **FAST_SEARCH_KWARGS,
    )
    jax_result = estimate_rotation_period(
        observations,
        search_strategy="surrogate_refine",
        exact_evaluation_backend="jax",
        session_mode="ignore",
        jax_frequency_batch_size=64,
        jax_row_pad_multiple=32,
        **FAST_SEARCH_KWARGS,
    )

    assert float(_scalar(jax_result.period_days[0])) == pytest.approx(
        float(_scalar(numpy_result.period_days[0])),
        rel=1.0e-9,
    )
    assert int(_scalar(jax_result.fourier_order[0])) == int(_scalar(numpy_result.fourier_order[0]))
    assert float(_scalar(jax_result.residual_sigma_mag[0])) == pytest.approx(
        float(_scalar(numpy_result.residual_sigma_mag[0])),
        rel=1.0e-8,
    )


def test_estimate_rotation_period_jax_backend_parallel_matches_serial():
    estimate_rotation_period = _api("estimate_rotation_period")
    observations = _make_rotation_observations(period_days=3.25, filters=["LSST_r"] * 48)

    serial = estimate_rotation_period(
        observations,
        search_strategy="surrogate_refine",
        exact_evaluation_backend="jax",
        session_mode="ignore",
        max_processes=None,
        jax_frequency_batch_size=64,
        jax_row_pad_multiple=32,
        **FAST_SEARCH_KWARGS,
    )
    parallel = estimate_rotation_period(
        observations,
        search_strategy="surrogate_refine",
        exact_evaluation_backend="jax",
        session_mode="ignore",
        max_processes=2,
        jax_frequency_batch_size=64,
        jax_row_pad_multiple=32,
        **FAST_SEARCH_KWARGS,
    )

    assert float(_scalar(parallel.period_days[0])) == pytest.approx(
        float(_scalar(serial.period_days[0])),
        rel=1.0e-9,
    )
    assert int(_scalar(parallel.fourier_order[0])) == int(_scalar(serial.fourier_order[0]))
    assert float(_scalar(parallel.residual_sigma_mag[0])) == pytest.approx(
        float(_scalar(serial.residual_sigma_mag[0])),
        rel=1.0e-8,
    )


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

    result = estimate_rotation_period(
        observations,
        session_mode="use",
        **FAST_SEARCH_KWARGS,
    )

    assert isinstance(result, RotationPeriodResult)
    assert len(result) == 1
    assert float(_scalar(result.period_days[0])) == pytest.approx(2.7, rel=0.03)
    assert int(_scalar(result.n_filters[0])) == 1
    assert int(_scalar(result.n_sessions[0])) == 3
    assert bool(_scalar(result.used_session_offsets[0])) is True


def test_estimate_rotation_period_auto_uses_dense_sessions_for_short_period():
    estimate_rotation_period = _api("estimate_rotation_period")
    observations = _make_rotation_observations(
        period_days=0.25,
        filters=["LSST_r"] * 48,
        session_ids=(["s1"] * 8) + (["s2"] * 8) + (["s3"] * 8) + (["s4"] * 8) + (["s5"] * 8) + (["s6"] * 8),
        session_offsets={"s2": 0.35, "s4": -0.28, "s6": 0.22},
        noise_sigma=0.005,
    )
    baseline = estimate_rotation_period(
        observations,
        max_frequency_cycles_per_day=8.0,
        frequency_grid_scale=40.0,
        session_mode="ignore",
    )

    result = estimate_rotation_period(
        observations,
        max_frequency_cycles_per_day=8.0,
        frequency_grid_scale=40.0,
        session_mode="auto",
        auto_session_bic_improvement=0.0,
    )

    assert bool(_scalar(result.used_session_offsets[0])) is True
    assert float(_scalar(result.residual_sigma_mag[0])) < float(_scalar(baseline.residual_sigma_mag[0]))


def test_estimate_rotation_period_auto_ignores_sparse_sessions_for_long_period():
    estimate_rotation_period = _api("estimate_rotation_period")
    observations = _make_rotation_observations(
        period_days=4.2,
        filters=["LSST_r"] * 48,
        session_ids=sum(([f"s{i}"] * 4 for i in range(12)), start=[]),
        session_offsets={f"s{i}": (0.08 if i % 2 == 0 else -0.06) for i in range(12)},
        noise_sigma=0.005,
    )

    result = estimate_rotation_period(observations, **FAST_SEARCH_KWARGS)

    assert float(_scalar(result.period_days[0])) == pytest.approx(4.2, rel=0.04)
    assert bool(_scalar(result.used_session_offsets[0])) is False


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


def test_estimate_rotation_period_default_orders_cap_at_four():
    estimate_rotation_period = _api("estimate_rotation_period")
    observations = _make_rotation_observations(
        period_days=3.8,
        filters=["LSST_r"] * 64,
        amplitude=0.35,
        noise_sigma=0.01,
    )

    result = estimate_rotation_period(
        observations,
        max_frequency_cycles_per_day=1.0,
        frequency_grid_scale=60.0,
    )

    assert int(_scalar(result.fourier_order[0])) <= 4


def test_estimate_rotation_period_sets_high_confidence_without_ambiguity():
    estimate_rotation_period = _api("estimate_rotation_period")
    observations = _make_rotation_observations(
        period_days=3.4,
        filters=["LSST_r"] * 48,
    )

    result = estimate_rotation_period(
        observations,
        enable_harmonic_adjudication=False,
        enable_lsm_crosscheck=False,
        enable_global_near_tie_check=False,
        enable_window_alias_check=False,
        **FAST_SEARCH_KWARGS,
    )

    assert bool(_scalar(result.is_ambiguous[0])) is False
    assert str(_scalar(result.confidence_label[0])) == "high"
    assert _scalar(result.ambiguity_reason[0]) is None


def test_estimate_rotation_period_marks_ambiguous_when_lsm_disagrees(monkeypatch):
    estimate_rotation_period = _api("estimate_rotation_period")
    observations = _make_rotation_observations(
        period_days=3.25,
        filters=["LSST_r"] * 48,
    )
    monkeypatch.setattr(
        "adam_core.photometry.rotation_period_fourier._estimate_lsm_frequency",
        lambda **kwargs: 0.5,
    )

    result = estimate_rotation_period(
        observations,
        enable_harmonic_adjudication=False,
        enable_lsm_crosscheck=True,
        harmonic_grid_fallback_on_near_tie=False,
        **FAST_SEARCH_KWARGS,
    )

    assert bool(_scalar(result.is_ambiguous[0])) is True
    assert str(_scalar(result.confidence_label[0])) == "ambiguous"
    assert "lsm_disagreement" in str(_scalar(result.ambiguity_reason[0]))
    assert bool(_scalar(result.lsm_harmonic_agreement[0])) is False


def test_estimate_rotation_period_marks_harmonic_ambiguous_for_harmonic_near_tie(monkeypatch):
    estimate_rotation_period = _api("estimate_rotation_period")
    observations = _make_rotation_observations(
        period_days=3.25,
        filters=["LSST_r"] * 48,
    )

    def fake_adjudication(*, chosen_fit, **kwargs):  # noqa: ARG001
        return rotation_period_fourier._HarmonicAdjudication(
            selected=rotation_period_fourier._fit_with_period(chosen_fit),
            near_tie_candidates=2,
            had_near_tie=True,
        )

    monkeypatch.setattr(
        "adam_core.photometry.rotation_period_fourier._adjudicate_harmonic_aliases",
        fake_adjudication,
    )

    result = estimate_rotation_period(
        observations,
        enable_harmonic_adjudication=True,
        harmonic_grid_fallback_on_near_tie=False,
        enable_lsm_crosscheck=False,
        enable_global_near_tie_check=False,
        enable_window_alias_check=False,
        **FAST_SEARCH_KWARGS,
    )

    assert bool(_scalar(result.is_ambiguous[0])) is True
    assert str(_scalar(result.confidence_label[0])) == "harmonic_ambiguous"
    assert "harmonic_near_tie" in str(_scalar(result.ambiguity_reason[0]))
    assert int(_scalar(result.n_harmonic_near_ties[0])) == 2


def test_estimate_rotation_period_marks_ambiguous_for_global_near_tie(monkeypatch):
    estimate_rotation_period = _api("estimate_rotation_period")
    observations = _make_rotation_observations(
        period_days=3.25,
        filters=["LSST_r"] * 48,
    )
    monkeypatch.setattr(
        "adam_core.photometry.rotation_period_fourier._count_nonharmonic_near_ties",
        lambda **kwargs: 1,
    )
    monkeypatch.setattr(
        "adam_core.photometry.rotation_period_fourier._count_window_alias_near_ties",
        lambda **kwargs: 0,
    )

    result = estimate_rotation_period(
        observations,
        enable_harmonic_adjudication=False,
        enable_lsm_crosscheck=False,
        enable_global_near_tie_check=True,
        enable_window_alias_check=False,
        **FAST_SEARCH_KWARGS,
    )

    assert bool(_scalar(result.is_ambiguous[0])) is True
    assert str(_scalar(result.confidence_label[0])) == "ambiguous"
    assert "global_near_tie" in str(_scalar(result.ambiguity_reason[0]))


def test_estimate_rotation_period_marks_ambiguous_for_window_alias_near_tie(monkeypatch):
    estimate_rotation_period = _api("estimate_rotation_period")
    observations = _make_rotation_observations(
        period_days=3.25,
        filters=["LSST_r"] * 48,
    )
    monkeypatch.setattr(
        "adam_core.photometry.rotation_period_fourier._count_nonharmonic_near_ties",
        lambda **kwargs: 0,
    )
    monkeypatch.setattr(
        "adam_core.photometry.rotation_period_fourier._count_window_alias_near_ties",
        lambda **kwargs: 1,
    )

    result = estimate_rotation_period(
        observations,
        enable_harmonic_adjudication=False,
        enable_lsm_crosscheck=False,
        enable_global_near_tie_check=False,
        enable_window_alias_check=True,
        **FAST_SEARCH_KWARGS,
    )

    assert bool(_scalar(result.is_ambiguous[0])) is True
    assert str(_scalar(result.confidence_label[0])) == "ambiguous"
    assert "window_alias_near_tie" in str(_scalar(result.ambiguity_reason[0]))


def test_estimate_rotation_period_uses_grid_fallback_for_harmonic_near_tie(monkeypatch):
    estimate_rotation_period = _api("estimate_rotation_period")
    observations = _make_rotation_observations(
        period_days=3.25,
        filters=["LSST_r"] * 48,
    )

    adjudication_calls = {"count": 0}
    grid_calls = {"count": 0}
    original_targeted_grid = rotation_period_fourier._run_period_search_targeted_grid

    def fake_adjudication(*, chosen_fit, **kwargs):  # noqa: ARG001
        adjudication_calls["count"] += 1
        selected = rotation_period_fourier._fit_with_period(chosen_fit)
        had_near_tie = adjudication_calls["count"] == 1
        return rotation_period_fourier._HarmonicAdjudication(
            selected=selected,
            near_tie_candidates=1 if had_near_tie else 0,
            had_near_tie=had_near_tie,
        )

    def fake_targeted_grid(**kwargs):
        grid_calls["count"] += 1
        return original_targeted_grid(**kwargs)

    monkeypatch.setattr(
        "adam_core.photometry.rotation_period_fourier._adjudicate_harmonic_aliases",
        fake_adjudication,
    )
    monkeypatch.setattr(
        "adam_core.photometry.rotation_period_fourier._run_period_search_targeted_grid",
        fake_targeted_grid,
    )

    result = estimate_rotation_period(
        observations,
        search_strategy="surrogate_refine",
        enable_lsm_crosscheck=False,
        harmonic_grid_fallback_on_near_tie=True,
        enable_global_near_tie_check=False,
        enable_window_alias_check=False,
        max_frequency_cycles_per_day=30.0,
        frequency_grid_scale=20.0,
    )

    assert grid_calls["count"] == 1
    assert adjudication_calls["count"] >= 2
    assert bool(_scalar(result.used_grid_fallback[0])) is True


def test_build_rotation_period_observations_from_detections(monkeypatch):
    build = _api("build_rotation_period_observations_from_detections")
    detections, exposures, object_coords, observer = _make_detection_bundle(
        object_ids=["A"],
        periods_by_object={"A": 3.0},
        filters=["LSST_r", "LSST_i", "LSST_r", "LSST_i"] * 6,
    )
    _patch_exposure_observers(monkeypatch, observer)
    monkeypatch.setattr(
        "adam_core.photometry.rotation_period_wrappers.calculate_observing_night",
        lambda codes, times: pa.array(
            [61000 + (idx // 4) for idx in range(len(times))],
            type=pa.int64(),
        ),
    )

    observations = build(detections, exposures, object_coords)

    assert isinstance(observations, RotationPeriodObservations)
    assert len(observations) == len(detections)
    assert np.all(np.isfinite(np.asarray(observations.r_au, dtype=np.float64)))
    assert np.all(np.isfinite(np.asarray(observations.delta_au, dtype=np.float64)))
    assert np.all(np.isfinite(np.asarray(observations.phase_angle_deg, dtype=np.float64)))
    assert observations.session_id.to_pylist()[:8] == [
        "X05:61000",
        "X05:61000",
        "X05:61000",
        "X05:61000",
        "X05:61001",
        "X05:61001",
        "X05:61001",
        "X05:61001",
    ]


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


def test_estimate_rotation_period_from_detections_forwards_parallel_kwargs(monkeypatch):
    estimate_from_detections = _api("estimate_rotation_period_from_detections")
    detections, exposures, object_coords, observer = _make_detection_bundle(
        object_ids=["A"],
        periods_by_object={"A": 3.4},
        filters=["LSST_r"] * 8,
    )
    _patch_exposure_observers(monkeypatch, observer)

    seen: dict[str, object] = {}

    def fake_estimate(observations, **kwargs):
        seen["n_observations"] = len(observations)
        seen["kwargs"] = kwargs
        return RotationPeriodResult.from_kwargs(
            period_days=[3.4],
            period_hours=[81.6],
            frequency_cycles_per_day=[1.0 / 3.4],
            fourier_order=[2],
            phase_c1=[0.0],
            phase_c2=[0.0],
            residual_sigma_mag=[0.1],
            n_observations=[len(observations)],
            n_fit_observations=[len(observations)],
            n_clipped=[0],
            n_filters=[1],
            n_sessions=[1],
            used_session_offsets=[False],
            is_period_doubled=[False],
            is_ambiguous=[False],
            confidence_label=["high"],
            ambiguity_reason=[None],
            n_harmonic_near_ties=[0],
            used_grid_fallback=[False],
            harmonic_sigma_tolerance_mag=[0.02],
            lsm_period_days=[3.4],
            lsm_period_hours=[81.6],
            lsm_frequency_cycles_per_day=[1.0 / 3.4],
            lsm_harmonic_agreement=[True],
        )

    monkeypatch.setattr(
        "adam_core.photometry.rotation_period_fourier.estimate_rotation_period",
        fake_estimate,
    )

    result = estimate_from_detections(
        detections,
        exposures,
        object_coords,
        max_processes=3,
        parallel_chunk_size=128,
        **FAST_SEARCH_KWARGS,
    )

    assert isinstance(result, RotationPeriodResult)
    assert seen["n_observations"] == len(detections)
    assert seen["kwargs"]["max_processes"] == 3
    assert seen["kwargs"]["parallel_chunk_size"] == 128
    assert seen["kwargs"]["max_frequency_cycles_per_day"] == FAST_SEARCH_KWARGS["max_frequency_cycles_per_day"]
    assert seen["kwargs"]["frequency_grid_scale"] == FAST_SEARCH_KWARGS["frequency_grid_scale"]


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

    with pytest.raises(ValueError):
        estimate_rotation_period(
            _make_rotation_observations(period_days=2.4, filters=["LSST_r"] * 12),
            session_mode="nope",
            **FAST_SEARCH_KWARGS,
        )

    with pytest.raises(ValueError):
        estimate_rotation_period(
            _make_rotation_observations(period_days=2.4, filters=["LSST_r"] * 12),
            search_strategy="teleport",
            **FAST_SEARCH_KWARGS,
        )

    with pytest.raises(ValueError):
        estimate_rotation_period(
            _make_rotation_observations(period_days=2.4, filters=["LSST_r"] * 12),
            exact_evaluation_backend="cuda_magic",
            **FAST_SEARCH_KWARGS,
        )

    with pytest.raises(ValueError):
        estimate_rotation_period(
            _make_rotation_observations(period_days=2.4, filters=["LSST_r"] * 12),
            max_processes=0,
            **FAST_SEARCH_KWARGS,
        )

    with pytest.raises(ValueError):
        estimate_rotation_period(
            _make_rotation_observations(period_days=2.4, filters=["LSST_r"] * 12),
            parallel_chunk_size=0,
            **FAST_SEARCH_KWARGS,
        )


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
