from __future__ import annotations

import numpy as np

from adam_core import _rust_native as _rn

from ..rotation.estimator import (
    _benchmark_rotation_period_native,
    estimate_rotation_period,
)
from ..rotation.wrappers import (
    estimate_rotation_period_from_detections,
    estimate_rotation_period_from_detections_grouped,
)
from .test_rotation_period import _make_rotation_observations
from .test_rotation_period_benchmarks import _make_detection_bundle


def test_rotation_period_native_timing_uses_positive_rust_samples() -> None:
    observations = _make_rotation_observations()
    options = {
        "search_fidelity": "exact_grid",
        "fourier_orders": (2,),
        "max_frequency_cycles_per_day": 80.0,
        "frequency_grid_scale": 10.0,
    }
    expected = estimate_rotation_period(observations, **options)
    samples = _benchmark_rotation_period_native(
        observations,
        reps=1,
        trials=2,
        warmup_reps=1,
        **options,
    )

    assert len(samples) == 2
    assert all(len(trial) == 1 for trial in samples)
    assert all(np.isfinite(trial[0]) and trial[0] > 0.0 for trial in samples)
    assert expected.period_days[0].as_py() > 0.0


def test_detection_rotation_facades_use_one_fused_crossing(monkeypatch) -> None:
    detections, exposures, object_coords, _, object_ids = _make_detection_bundle()
    original = _rn.rotation_period_estimate_from_detections
    calls = 0

    def counted(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(_rn, "rotation_period_estimate_from_detections", counted)
    options = {
        "search_fidelity": "exact_grid",
        "fourier_orders": (2,),
        "max_frequency_cycles_per_day": 80.0,
        "frequency_grid_scale": 10.0,
    }
    single = estimate_rotation_period_from_detections(
        detections, exposures, object_coords, **options
    )
    assert len(single) == 1
    assert calls == 1

    calls = 0
    grouped = estimate_rotation_period_from_detections_grouped(
        detections,
        exposures,
        object_coords,
        object_ids,
        **options,
    )
    assert len(grouped) == 2
    assert calls == 1
