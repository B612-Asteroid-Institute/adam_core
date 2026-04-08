from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from ...ray_cluster import initialize_use_ray
from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import Origin
from ...observers.observers import Observers
from ...time import Timestamp
from ..rotation_period_fourier import (
    _build_fixed_design,
    _fit_frequency,
    _validate_inputs,
    estimate_rotation_period,
)
from ..rotation_period_types import RotationPeriodObservations

DATA_DIR = Path(__file__).parent / "data"


@dataclass(frozen=True)
class _RotationBenchmarkCase:
    name: str
    observations: RotationPeriodObservations
    expected_period_hours: float
    relative_tolerance: float | None
    search_kwargs: dict[str, object]
    use_sessions_for_kernel: bool
    expected_used_session_offsets: bool | None
    approx_frequency_count: int

    @property
    def pytest_id(self) -> str:
        return f"{self.name}-nf~{self.approx_frequency_count}"


def _make_clustered_times(
    *,
    start_mjd: float,
    night_starts_days: np.ndarray,
    observations_per_night: int,
    intra_night_span_days: float,
    rng: np.random.Generator,
) -> Timestamp:
    offsets: list[float] = []
    for night_start in np.asarray(night_starts_days, dtype=np.float64):
        if observations_per_night == 1:
            local = np.asarray([0.0], dtype=np.float64)
        else:
            local = np.linspace(
                0.0,
                intra_night_span_days,
                observations_per_night,
                endpoint=False,
                dtype=np.float64,
            )
        jitter = rng.normal(
            loc=0.0,
            scale=max(intra_night_span_days / max(8 * observations_per_night, 1), 1.0e-5),
            size=local.size,
        )
        local = np.clip(local + jitter, 0.0, None)
        offsets.extend((night_start + local).tolist())
    mjd = start_mjd + np.asarray(offsets, dtype=np.float64)
    return Timestamp.from_mjd(mjd, scale="tdb")


def _phase_angle_deg(object_coords: CartesianCoordinates, observers: Observers) -> np.ndarray:
    obj = np.asarray(object_coords.r, dtype=np.float64)
    obs = np.asarray(observers.coordinates.r, dtype=np.float64)
    delta = obj - obs
    r = np.linalg.norm(obj, axis=1)
    d = np.linalg.norm(delta, axis=1)
    sun = np.linalg.norm(obs, axis=1)
    cos_alpha = np.clip((r * r + d * d - sun * sun) / (2.0 * r * d), -1.0, 1.0)
    return np.degrees(np.arccos(cos_alpha))


def _make_observations(
    *,
    times: Timestamp,
    period_days: float,
    filters: list[str],
    session_ids: list[str | None],
    filter_offsets: dict[str, float],
    session_offsets: dict[str, float],
    amplitude: float,
    noise_sigma: float,
    mag_sigma: float,
    seed: int,
) -> RotationPeriodObservations:
    n = len(times)
    t_mjd = np.asarray(times.mjd(), dtype=np.float64)
    t_rel = t_mjd - float(t_mjd.min())

    object_coords = CartesianCoordinates.from_kwargs(
        x=2.1 + 0.12 * np.sin(2.0 * np.pi * t_rel / 90.0),
        y=0.25 * np.cos(2.0 * np.pi * t_rel / 70.0),
        z=0.04 * np.sin(2.0 * np.pi * t_rel / 120.0),
        vx=np.zeros(n, dtype=np.float64),
        vy=np.zeros(n, dtype=np.float64),
        vz=np.zeros(n, dtype=np.float64),
        time=times,
        frame="ecliptic",
        origin=Origin.from_kwargs(code=["SUN"] * n),
    )
    observers = Observers.from_kwargs(
        code=["X05"] * n,
        coordinates=CartesianCoordinates.from_kwargs(
            x=np.full(n, 1.0, dtype=np.float64),
            y=np.zeros(n, dtype=np.float64),
            z=np.zeros(n, dtype=np.float64),
            vx=np.zeros(n, dtype=np.float64),
            vy=np.zeros(n, dtype=np.float64),
            vz=np.zeros(n, dtype=np.float64),
            time=times,
            frame="ecliptic",
            origin=Origin.from_kwargs(code=["SUN"] * n),
        ),
    )
    alpha = _phase_angle_deg(object_coords, observers)
    phase = 2.0 * np.pi * t_rel / period_days

    reduced = (
        19.1
        + 0.015 * alpha
        + 0.0015 * np.square(alpha)
        + 0.10 * np.cos(phase)
        + amplitude * np.cos(2.0 * phase)
        + 0.07 * np.sin(2.0 * phase)
    )
    reduced = reduced + np.asarray(
        [filter_offsets.get(filter_name, 0.0) for filter_name in filters],
        dtype=np.float64,
    )
    reduced = reduced + np.asarray(
        [0.0 if session_id is None else session_offsets.get(session_id, 0.0) for session_id in session_ids],
        dtype=np.float64,
    )

    r_au = np.linalg.norm(np.asarray(object_coords.r, dtype=np.float64), axis=1)
    delta_au = np.linalg.norm(
        np.asarray(object_coords.r, dtype=np.float64)
        - np.asarray(observers.coordinates.r, dtype=np.float64),
        axis=1,
    )
    distance_term = 5.0 * np.log10(r_au * delta_au)

    rng = np.random.default_rng(seed)
    mag = reduced + distance_term + rng.normal(0.0, noise_sigma, size=n)
    sigma = np.full(n, mag_sigma, dtype=np.float64)

    return RotationPeriodObservations.from_kwargs(
        time=times,
        mag=mag,
        mag_sigma=pa.array(sigma, type=pa.float64()),
        filter=filters,
        session_id=session_ids,
        r_au=r_au,
        delta_au=delta_au,
        phase_angle_deg=alpha,
    )


def _approx_frequency_count(
    observations: RotationPeriodObservations,
    *,
    min_rotations_in_span: float,
    max_frequency_cycles_per_day: float,
    frequency_grid_scale: float,
) -> int:
    time = np.asarray(observations.time.rescale("tdb").mjd().to_numpy(False), dtype=np.float64)
    span = float(np.max(time) - np.min(time))
    f_min = float(min_rotations_in_span / span)
    return max(
        int(ceil(frequency_grid_scale * span * (max_frequency_cycles_per_day - f_min)) + 1),
        2,
    )


def _case_frequency_count(
    observations: RotationPeriodObservations,
    search_kwargs: dict[str, object],
) -> int:
    return _approx_frequency_count(
        observations,
        min_rotations_in_span=float(search_kwargs["min_rotations_in_span"]),
        max_frequency_cycles_per_day=float(search_kwargs["max_frequency_cycles_per_day"]),
        frequency_grid_scale=float(search_kwargs["frequency_grid_scale"]),
    )


def _make_dense_fast_case() -> _RotationBenchmarkCase:
    rng = np.random.default_rng(20260406)
    night_starts = np.arange(0.0, 6.0, 1.0, dtype=np.float64)
    observations_per_night = 24
    times = _make_clustered_times(
        start_mjd=61000.0,
        night_starts_days=night_starts,
        observations_per_night=observations_per_night,
        intra_night_span_days=0.22,
        rng=rng,
    )
    filters = (["LSST_g", "LSST_r", "LSST_i"] * ((len(times) + 2) // 3))[: len(times)]
    session_ids = []
    session_offsets: dict[str, float] = {}
    for night_index, night_start in enumerate(night_starts):
        session_id = f"X05:{61000 + int(night_start)}"
        session_offsets[session_id] = 0.05 * np.sin(0.7 * night_index)
        session_ids.extend([session_id] * observations_per_night)

    observations = _make_observations(
        times=times,
        period_days=1.10 / 24.0,
        filters=filters,
        session_ids=session_ids,
        filter_offsets={"LSST_g": 0.18, "LSST_r": 0.0, "LSST_i": -0.07},
        session_offsets=session_offsets,
        amplitude=0.26,
        noise_sigma=0.025,
        mag_sigma=0.03,
        seed=202604061,
    )
    search_kwargs = {
        "max_frequency_cycles_per_day": 72.0,
        "frequency_grid_scale": 6.0,
        "min_rotations_in_span": 2.0,
        "session_mode": "auto",
    }
    return _RotationBenchmarkCase(
        name="dense_fast_auto_sessions",
        observations=observations,
        expected_period_hours=1.10,
        relative_tolerance=0.08,
        search_kwargs=search_kwargs,
        use_sessions_for_kernel=True,
        expected_used_session_offsets=None,
        approx_frequency_count=_case_frequency_count(observations, search_kwargs),
    )


def _make_sparse_lsst_case() -> _RotationBenchmarkCase:
    rng = np.random.default_rng(20260407)
    night_starts = np.arange(0.0, 180.0, 4.0, dtype=np.float64)
    observations_per_night = 2
    times = _make_clustered_times(
        start_mjd=61100.0,
        night_starts_days=night_starts,
        observations_per_night=observations_per_night,
        intra_night_span_days=0.0208,
        rng=rng,
    )
    filters = ["LSST_r"] * len(times)
    session_ids = []
    for night_start in night_starts:
        session_ids.extend([f"X05:{61100 + int(night_start)}"] * observations_per_night)

    observations = _make_observations(
        times=times,
        period_days=14.0 / 24.0,
        filters=filters,
        session_ids=session_ids,
        filter_offsets={"LSST_r": 0.0},
        session_offsets={},
        amplitude=0.18,
        noise_sigma=0.035,
        mag_sigma=0.04,
        seed=202604071,
    )
    search_kwargs = {
        "max_frequency_cycles_per_day": 24.0,
        "frequency_grid_scale": 1.25,
        "min_rotations_in_span": 2.0,
        "session_mode": "auto",
    }
    return _RotationBenchmarkCase(
        name="sparse_lsst_generic_auto",
        observations=observations,
        expected_period_hours=14.0,
        relative_tolerance=None,
        search_kwargs=search_kwargs,
        use_sessions_for_kernel=False,
        expected_used_session_offsets=False,
        approx_frequency_count=_case_frequency_count(observations, search_kwargs),
    )


def _make_sparse_long_case() -> _RotationBenchmarkCase:
    rng = np.random.default_rng(20260408)
    night_starts = np.arange(0.0, 240.0, 6.0, dtype=np.float64)
    observations_per_night = 2
    times = _make_clustered_times(
        start_mjd=61350.0,
        night_starts_days=night_starts,
        observations_per_night=observations_per_night,
        intra_night_span_days=0.018,
        rng=rng,
    )
    filters = ["LSST_r"] * len(times)
    session_ids = []
    for night_start in night_starts:
        session_ids.extend([f"X05:{61350 + int(night_start)}"] * observations_per_night)

    observations = _make_observations(
        times=times,
        period_days=3.6,
        filters=filters,
        session_ids=session_ids,
        filter_offsets={"LSST_r": 0.0},
        session_offsets={},
        amplitude=0.22,
        noise_sigma=0.04,
        mag_sigma=0.05,
        seed=202604081,
    )
    search_kwargs = {
        "max_frequency_cycles_per_day": 8.0,
        "frequency_grid_scale": 1.0,
        "min_rotations_in_span": 2.0,
        "session_mode": "auto",
    }
    return _RotationBenchmarkCase(
        name="sparse_long_period_auto",
        observations=observations,
        expected_period_hours=3.6 * 24.0,
        relative_tolerance=None,
        search_kwargs=search_kwargs,
        use_sessions_for_kernel=False,
        expected_used_session_offsets=False,
        approx_frequency_count=_case_frequency_count(observations, search_kwargs),
    )


def _benchmark_cases() -> tuple[_RotationBenchmarkCase, ...]:
    return (
        _make_dense_fast_case(),
        _make_sparse_lsst_case(),
        _make_sparse_long_case(),
    )


SEARCH_STRATEGIES = ("grid", "surrogate_refine", "coarse_to_fine")
REAL_MAX_PROCESSES = (1, 4)
HOT_BACKENDS = ("numpy", "jax")
HOT_BACKEND_MAX_PROCESSES = (1, 4)
REAL_FIXTURES: list[str] = sorted(
    p.name for p in DATA_DIR.glob("rotation_period_benchmark_fixture_*.npz")
)
if not REAL_FIXTURES:
    REAL_FIXTURES = ["__NO_FIXTURES__"]


def _load_rotation_fixture(
    fixture_name: str,
) -> tuple[RotationPeriodObservations, dict[str, object], float, int, bool]:
    fx = np.load(DATA_DIR / fixture_name, allow_pickle=True)
    mag_sigma = np.asarray(fx["mag_sigma"], dtype=np.float64)
    observations = RotationPeriodObservations.from_kwargs(
        time=Timestamp.from_iso8601(fx["time_iso"].astype(object).tolist(), scale="utc"),
        mag=np.asarray(fx["mag_obs"], dtype=np.float64),
        mag_sigma=pa.array(
            mag_sigma,
            mask=~np.isfinite(mag_sigma),
            type=pa.float64(),
        ),
        filter=fx["filter"].astype(object).tolist(),
        session_id=fx["session_id"].astype(object).tolist(),
        r_au=np.asarray(fx["r_au"], dtype=np.float64),
        delta_au=np.asarray(fx["delta_au"], dtype=np.float64),
        phase_angle_deg=np.asarray(fx["phase_angle_deg"], dtype=np.float64),
    )
    search_kwargs = {
        "frequency_grid_scale": float(fx["frequency_grid_scale"][0]),
        "max_frequency_cycles_per_day": float(fx["max_frequency_cycles_per_day"][0]),
        "min_rotations_in_span": float(fx["min_rotations_in_span"][0]),
        "session_mode": str(fx["session_mode"][0]),
    }
    return (
        observations,
        search_kwargs,
        float(fx["expected_period_hours"][0]),
        int(fx["expected_fourier_order"][0]),
        bool(fx["expected_used_session_offsets"][0]),
    )


@pytest.mark.parametrize(
    "case",
    _benchmark_cases(),
    ids=lambda case: case.pytest_id,
)
@pytest.mark.benchmark(group="rotation_period_fit_frequency")
def test_benchmark_fit_frequency_kernel(benchmark, case: _RotationBenchmarkCase):
    time, mag, r_au, delta_au, phase_angle, filter_labels, session_labels, mag_sigma = _validate_inputs(
        case.observations
    )
    y = mag - 5.0 * np.log10(r_au * delta_au)
    time_rel = time - float(np.min(time))
    weights = 1.0 / np.square(np.asarray(mag_sigma, dtype=np.float64))
    active_sessions = session_labels if case.use_sessions_for_kernel else None
    design_info = _build_fixed_design(filter_labels, active_sessions, phase_angle)
    frequency = 24.0 / float(case.expected_period_hours)

    def run():
        return _fit_frequency(
            time_rel,
            y,
            design_info,
            frequency,
            5,
            clip_sigma=3.0,
            weights=weights,
        )

    fit = benchmark(run)
    assert fit is not None
    assert fit.n_fit > fit.n_par


@pytest.mark.parametrize(
    "search_strategy",
    SEARCH_STRATEGIES,
    ids=lambda strategy: f"strategy={strategy}",
)
@pytest.mark.parametrize(
    "case",
    _benchmark_cases(),
    ids=lambda case: case.pytest_id,
)
@pytest.mark.benchmark(group="rotation_period_estimate")
def test_benchmark_estimate_rotation_period(
    benchmark,
    case: _RotationBenchmarkCase,
    search_strategy: str,
):
    search_kwargs = dict(case.search_kwargs)
    search_kwargs["search_strategy"] = search_strategy

    def run():
        return estimate_rotation_period(case.observations, **search_kwargs)

    result = benchmark(run)
    period_hours = float(result.period_hours[0].as_py())
    assert np.isfinite(period_hours)
    assert period_hours > 0.0
    if case.expected_used_session_offsets is not None:
        assert bool(result.used_session_offsets[0].as_py()) is case.expected_used_session_offsets
    if case.relative_tolerance is not None:
        rel_error = abs(period_hours - case.expected_period_hours) / case.expected_period_hours
        assert rel_error <= case.relative_tolerance


@pytest.mark.parametrize(
    "search_strategy",
    SEARCH_STRATEGIES,
    ids=lambda strategy: f"strategy={strategy}",
)
@pytest.mark.parametrize(
    "max_processes",
    REAL_MAX_PROCESSES,
    ids=lambda count: f"procs={count}",
)
@pytest.mark.parametrize(
    "fixture_name",
    REAL_FIXTURES,
    ids=lambda name: name.replace(".npz", ""),
)
@pytest.mark.benchmark(group="rotation_period_estimate_real")
def test_benchmark_estimate_rotation_period_real_fixture(
    benchmark,
    fixture_name: str,
    search_strategy: str,
    max_processes: int,
):
    if fixture_name == "__NO_FIXTURES__":
        pytest.skip("No frozen real-data rotation-period benchmark fixtures found on disk.")

    observations, search_kwargs, expected_period_hours, expected_order, expected_used_session_offsets = (
        _load_rotation_fixture(fixture_name)
    )
    search_kwargs["search_strategy"] = search_strategy
    search_kwargs["max_processes"] = None if max_processes == 1 else max_processes
    if max_processes > 1:
        initialize_use_ray(num_cpus=max_processes)

    def run():
        return estimate_rotation_period(observations, **search_kwargs)

    result = benchmark(run)
    period_hours = float(result.period_hours[0].as_py())
    assert np.isfinite(period_hours)
    assert period_hours > 0.0
    rel_error = abs(period_hours - expected_period_hours) / expected_period_hours
    assert rel_error <= 1.0e-9
    assert int(result.fourier_order[0].as_py()) == expected_order
    assert bool(result.used_session_offsets[0].as_py()) is expected_used_session_offsets


@pytest.mark.parametrize(
    "session_mode",
    ("ignore", "auto"),
    ids=lambda mode: f"session={mode}",
)
@pytest.mark.parametrize(
    "exact_evaluation_backend",
    HOT_BACKENDS,
    ids=lambda backend: f"backend={backend}",
)
@pytest.mark.parametrize(
    "max_processes",
    HOT_BACKEND_MAX_PROCESSES,
    ids=lambda count: f"procs={count}",
)
@pytest.mark.parametrize(
    "fixture_name",
    REAL_FIXTURES,
    ids=lambda name: name.replace(".npz", ""),
)
@pytest.mark.benchmark(group="rotation_period_estimate_real_backend")
def test_benchmark_estimate_rotation_period_real_fixture_backend(
    benchmark,
    fixture_name: str,
    exact_evaluation_backend: str,
    session_mode: str,
    max_processes: int,
):
    if fixture_name == "__NO_FIXTURES__":
        pytest.skip("No frozen real-data rotation-period benchmark fixtures found on disk.")

    observations, search_kwargs, expected_period_hours, expected_order, _ = _load_rotation_fixture(
        fixture_name
    )
    search_kwargs["search_strategy"] = "surrogate_refine"
    search_kwargs["session_mode"] = session_mode
    search_kwargs["max_processes"] = None if max_processes == 1 else max_processes

    if max_processes > 1:
        initialize_use_ray(num_cpus=max_processes)

    if exact_evaluation_backend == "jax":
        search_kwargs["exact_evaluation_backend"] = "jax"
        search_kwargs["jax_frequency_batch_size"] = 128
        search_kwargs["jax_row_pad_multiple"] = 64
        # Benchmark hot-path throughput, not one-time JIT compile cost.
        estimate_rotation_period(observations, **search_kwargs)

    def run():
        return estimate_rotation_period(observations, **search_kwargs)

    result = benchmark(run)
    period_hours = float(result.period_hours[0].as_py())
    assert np.isfinite(period_hours)
    assert period_hours > 0.0
    rel_error = abs(period_hours - expected_period_hours) / expected_period_hours
    assert rel_error <= 1.0e-9
    assert int(result.fourier_order[0].as_py()) == expected_order
