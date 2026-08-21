"""Committed validation gates for rotation-period estimation.

Auto-discovers the committed standard-candle fixtures
(``rotation_period_validation_fixture_*.npz``) and runs the Fourier fast-path
solver on each. Two gates are exercised:

1. A schema/smoke gate (parametrized) asserting every fixture solves and returns
   a coherent confidence result (``period_verdict`` in the valid enum,
   ``reliability_code`` in ``{"1", "2", "3"}``, and the validity/reliability/period
   fields consistent with the verdict).
2. A zero-false-confidence gate on every fixture: no result labelled
   ``single_period`` may exceed the fixture's strict, unadjusted period tolerance.
   This includes harmonic and cadence aliases. The per-order frequency-consensus
   guard prevents a statistically preferred high-order daily alias from overriding
   the physical-frequency family shared by a majority of Fourier orders.

Slow fixtures are marked ``@pytest.mark.profile`` and run by the extended
release-candidate coverage job.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from adam_core.photometry.rotation.core import (
    RotationPeriodObservations,
    alias_bucket,
    harmonic_adjusted_error_pct,
    relative_error_pct,
    within_tolerance,
)
from adam_core.photometry.rotation.estimator import estimate_rotation_period
from adam_core.time import Timestamp

DATA_DIR = Path(__file__).parent / "data"

# Object substrings whose fixtures are slow to solve (~29-217 s); excluded from
# the default run via @pytest.mark.profile.
_SLOW_OBJECTS = ("Rotraut", "Nenetta", "Alauda", "Murakami")

_ALL_FIXTURES: list[str] = sorted(
    p.name for p in DATA_DIR.glob("rotation_period_validation_fixture_*.npz")
)
if not _ALL_FIXTURES:
    _ALL_FIXTURES = ["__NO_FIXTURES__"]


def _is_slow(fixture_name: str) -> bool:
    return any(obj in fixture_name for obj in _SLOW_OBJECTS)


# Parametrize so slow fixtures carry the `profile` marker individually.
_FIXTURE_PARAMS = [
    pytest.param(name, marks=pytest.mark.profile) if _is_slow(name) else name
    for name in _ALL_FIXTURES
]


def _load_fixture(path: Path) -> tuple[RotationPeriodObservations, dict[str, object]]:
    """Load a committed validation fixture into observations + scoring metadata."""
    z = np.load(path, allow_pickle=True)
    time = Timestamp.from_iso8601(
        np.asarray(z["time_iso"], dtype=object).tolist(), scale="utc"
    )
    mag = np.asarray(z["mag_obs"], dtype=np.float64)
    sigma = np.asarray(z["mag_sigma"], dtype=np.float64)
    observations = RotationPeriodObservations.from_kwargs(
        time=time,
        mag=mag,
        mag_sigma=pa.array(sigma, mask=~np.isfinite(sigma), type=pa.float64()),
        filter=[str(v) for v in np.asarray(z["filter"], dtype=object).tolist()],
        session_id=[str(v) for v in np.asarray(z["session_id"], dtype=object).tolist()],
        r_au=np.asarray(z["r_au"], dtype=np.float64),
        delta_au=np.asarray(z["delta_au"], dtype=np.float64),
        phase_angle_deg=np.asarray(z["phase_angle_deg"], dtype=np.float64),
    )
    meta: dict[str, object] = {
        "object": f"{int(z['object_number'][0])} {z['object_name'][0]}",
        "expected_hours": float(z["expected_period_hours"][0]),
        "tolerance_fraction": float(z["tolerance_fraction"][0]),
        "frequency_grid_scale": float(z["frequency_grid_scale"][0]),
        "max_frequency_cycles_per_day": float(z["max_frequency_cycles_per_day"][0]),
        "min_rotations_in_span": float(z["min_rotations_in_span"][0]),
    }
    return observations, meta


def _solve(observations: RotationPeriodObservations, meta: dict[str, object]):
    return estimate_rotation_period(
        observations,
        search_fidelity="validated_staged",
        exact_evaluation_backend="jax",
        frequency_grid_scale=float(meta["frequency_grid_scale"]),
        max_frequency_cycles_per_day=float(meta["max_frequency_cycles_per_day"]),
        min_rotations_in_span=float(meta["min_rotations_in_span"]),
    )


@pytest.mark.parametrize("fixture_name", _FIXTURE_PARAMS)
def test_validation_fixture_schema(fixture_name: str) -> None:
    """Every committed fixture solves and returns a coherent confidence result."""
    if fixture_name == "__NO_FIXTURES__":
        pytest.skip("No rotation-period validation fixtures found on disk.")

    observations, meta = _load_fixture(DATA_DIR / fixture_name)
    result = _solve(observations, meta)

    verdict = str(result.period_verdict[0].as_py())
    assert verdict in {"single_period", "period_family", "insufficient_data"}
    assert result.is_period_doubled[0].as_py() is not None

    # The verdict fully determines validity, reliability, and reliability_code --
    # assert the whole mapping, not merely that the columns are non-null.
    is_valid = bool(result.is_valid[0].as_py())
    is_reliable = bool(result.is_reliable[0].as_py())
    reliability = str(result.reliability_code[0].as_py())
    expected_by_verdict = {
        "single_period": (True, True, "3"),
        "period_family": (True, False, "2"),
        "insufficient_data": (False, False, "1"),
    }
    assert (is_valid, is_reliable, reliability) == expected_by_verdict[verdict]

    # Period finiteness must track the verdict: a believed/family call reports a
    # finite positive period; an insufficient_data call reports a non-finite period.
    p_hours = result.period_hours[0].as_py()
    if verdict == "insufficient_data":
        assert p_hours is None or not np.isfinite(float(p_hours))
    else:
        assert (
            p_hours is not None and np.isfinite(float(p_hours)) and float(p_hours) > 0.0
        )
    if str(meta["object"]).startswith("1627 Ivar"):
        assert "diurnal_order_consensus" in result.confidence_flags[0].as_py()
        assert result.fourier_order[0].as_py() == 5
    if str(meta["object"]).startswith("3295 Murakami"):
        assert verdict == "period_family"
        assert "fourier_order_disagreement" in result.insufficiency_reasons[0].as_py()

    if verdict == "single_period":
        p_rec = float(p_hours)
        p_true = float(meta["expected_hours"])
        tolerance = float(meta["tolerance_fraction"])
        if not within_tolerance(p_rec, p_true, tolerance):
            _, best_factor = harmonic_adjusted_error_pct(p_rec, p_true)
            pytest.fail(
                f"{meta['object']}: confident period {p_rec:.4f}h differs from "
                f"{p_true:.4f}h by {relative_error_pct(p_rec, p_true):.2f}% "
                f"(tolerance {tolerance * 100.0:.2f}%, alias "
                f"{alias_bucket(best_factor)})"
            )

    # Committed fixtures are curated sparse subsets and can behave differently from
    # a full calibration corpus. This is a per-fixture scientific regression lock.
