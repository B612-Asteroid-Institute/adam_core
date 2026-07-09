from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from ...time import Timestamp
from ..rotation.core import RotationPeriodObservations, RotationPeriodResult
from ..rotation.wrappers import (
    _apparition_index_groups,
    estimate_rotation_period_best_apparition,
)

# Fast, deterministic solver settings used throughout (mirrors the other
# rotation tests: small exact grid, no staged search).
_FAST: dict[str, Any] = dict(
    search_fidelity="exact_grid",
    max_frequency_cycles_per_day=120.0,
    frequency_grid_scale=40.0,
)


def _make_apparition(
    *,
    t0_mjd: float,
    n: int = 60,
    span_days: float = 0.08,
    period_days: float | None = 0.02,
    amplitude: float = 0.35,
    noise_sigma: float = 0.01,
) -> dict[str, np.ndarray]:
    """Raw column arrays for one apparition; period_days=None means flat noise."""
    mjd = np.linspace(t0_mjd, t0_mjd + span_days, n, dtype=np.float64)
    t_rel = mjd - mjd.min()
    r_au = np.full(n, 2.0, dtype=np.float64)
    delta_au = np.full(n, 1.5, dtype=np.float64)
    phase_angle = np.full(n, 12.0, dtype=np.float64)
    baseline = 15.0 + 5.0 * np.log10(r_au * delta_au) + 0.015 * phase_angle
    rng = np.random.default_rng(int(t0_mjd))
    if period_days is None:
        rotation = np.zeros(n, dtype=np.float64)
        noise = rng.normal(0.0, 0.005, size=n)  # scatter below mag_sigma
    else:
        phase = 2.0 * np.pi * t_rel / period_days
        rotation = (
            0.10 * np.cos(phase)
            + amplitude * np.cos(2.0 * phase)
            + 0.07 * np.sin(2.0 * phase)
        )
        noise = rng.normal(0.0, noise_sigma, size=n)
    return {
        "mjd": mjd,
        "mag": baseline + rotation + noise,
        "mag_sigma": np.full(n, 0.03 if period_days is not None else 0.05),
        "r_au": r_au,
        "delta_au": delta_au,
        "phase_angle_deg": phase_angle,
    }


def _to_observations(*apparitions: dict[str, np.ndarray]) -> RotationPeriodObservations:
    cols = {
        key: np.concatenate([a[key] for a in apparitions])
        for key in apparitions[0].keys()
    }
    n = len(cols["mjd"])
    return RotationPeriodObservations.from_kwargs(
        time=Timestamp.from_mjd(cols["mjd"], scale="tdb"),
        mag=pa.array(cols["mag"], type=pa.float64()),
        mag_sigma=pa.array(cols["mag_sigma"], type=pa.float64()),
        filter=pa.array(["LSST_r"] * n, type=pa.large_string()),
        session_id=pa.array(
            [f"X05:{int(m)}" for m in cols["mjd"]], type=pa.large_string()
        ),
        r_au=pa.array(cols["r_au"], type=pa.float64()),
        delta_au=pa.array(cols["delta_au"], type=pa.float64()),
        phase_angle_deg=pa.array(cols["phase_angle_deg"], type=pa.float64()),
    )


def test_apparition_index_groups_partitions_on_gaps():
    # Unsorted input; gaps of >120 d split into three chronological groups whose
    # indices refer to the ORIGINAL order.
    mjd = np.array([60300.0, 60000.0, 60001.5, 60301.0, 60900.0])
    groups = _apparition_index_groups(mjd, 120.0)
    assert [list(g) for g in groups] == [[1, 2], [0, 3], [4]]
    # One group when no gap exceeds the threshold.
    assert len(_apparition_index_groups(np.array([60000.0, 60100.0]), 120.0)) == 1


def test_best_apparition_picks_the_clean_apparition():
    clean = _make_apparition(t0_mjd=60000.0, period_days=0.02)
    flat = _make_apparition(t0_mjd=60200.0, period_days=None)  # signal-free
    combined = _to_observations(clean, flat)

    result = estimate_rotation_period_best_apparition(combined, **_FAST)
    alone = estimate_rotation_period_best_apparition(_to_observations(clean), **_FAST)

    assert isinstance(result, RotationPeriodResult)
    assert len(result) == 1
    # The chosen row is the clean apparition's solve: same period, same verdict.
    assert result.period_hours[0].as_py() == pytest.approx(
        alone.period_hours[0].as_py(), rel=1e-9
    )
    assert result.period_verdict[0].as_py() == alone.period_verdict[0].as_py()
    # The clean apparition is chronologically first of two.
    flags = result.confidence_flags[0].as_py() or []
    assert "apparition_selected_1_of_2" in flags


def test_single_apparition_matches_direct_solve():
    clean = _make_apparition(t0_mjd=60000.0, period_days=0.02)
    observations = _to_observations(clean)

    from ..rotation.estimator import estimate_rotation_period

    direct = estimate_rotation_period(observations, **_FAST)
    via_helper = estimate_rotation_period_best_apparition(observations, **_FAST)

    assert via_helper.period_hours[0].as_py() == pytest.approx(
        direct.period_hours[0].as_py(), rel=1e-9
    )
    assert via_helper.period_verdict[0].as_py() == direct.period_verdict[0].as_py()
    flags = via_helper.confidence_flags[0].as_py() or []
    assert "apparition_selected_1_of_1" in flags
    # Aside from the appended flag, the row is the direct result.
    direct_flags = direct.confidence_flags[0].as_py() or []
    assert flags == direct_flags + ["apparition_selected_1_of_1"]


def test_all_apparitions_failing_returns_single_insufficient(monkeypatch):
    import adam_core.photometry.rotation.estimator as rpe

    def always_fail(observations, **kwargs):
        raise ValueError("simulated insufficient data")

    monkeypatch.setattr(rpe, "estimate_rotation_period", always_fail)

    combined = _to_observations(
        _make_apparition(t0_mjd=60000.0, period_days=0.02),
        _make_apparition(t0_mjd=60200.0, period_days=None),
    )
    result = estimate_rotation_period_best_apparition(combined, **_FAST)

    assert result.period_verdict[0].as_py() == "insufficient_data"
    flags = result.confidence_flags[0].as_py() or []
    assert "solve_error" in flags
    assert any(f.startswith("apparition_selected_") for f in flags)


def test_unexpected_error_reraised_with_apparition_context(monkeypatch):
    import adam_core.photometry.rotation.estimator as rpe

    def boom(observations, **kwargs):
        raise RuntimeError("unexpected solver bug")

    monkeypatch.setattr(rpe, "estimate_rotation_period", boom)

    combined = _to_observations(
        _make_apparition(t0_mjd=60000.0, period_days=0.02),
        _make_apparition(t0_mjd=60200.0, period_days=None),
    )
    with pytest.raises(RuntimeError, match="apparition 1 of 2"):
        estimate_rotation_period_best_apparition(combined, **_FAST)
