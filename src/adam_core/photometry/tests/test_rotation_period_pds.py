from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from adam_core.photometry.rotation_period_fourier import estimate_rotation_period
from adam_core.photometry.rotation_period_types import RotationPeriodObservations
from adam_core.time import Timestamp

DATA_DIR = Path(__file__).parent / "data"

PDS_FIXTURES: list[str] = sorted(
    p.name for p in DATA_DIR.glob("rotation_period_pds_fixture_*.npz")
)
if not PDS_FIXTURES:
    PDS_FIXTURES = ["__NO_FIXTURES__"]


def _scalar(value) -> object:
    if hasattr(value, "as_py"):
        return value.as_py()
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size != 1:
        raise ValueError(f"Expected scalar-like value, got shape {arr.shape}")
    return arr.reshape(-1)[0].item()


def test_rotation_period_pds_fixture_inventory() -> None:
    assert "__NO_FIXTURES__" not in PDS_FIXTURES, "No PDS rotation-period fixtures found on disk."

    tiers: list[str] = []
    for fixture_name in PDS_FIXTURES:
        fx = np.load(DATA_DIR / fixture_name, allow_pickle=True)
        tiers.append(str(fx["tier"][0]))

    assert len(PDS_FIXTURES) >= 8
    assert tiers.count("gold") >= 6
    assert tiers.count("challenge") >= 2


@pytest.mark.parametrize("fixture_name", PDS_FIXTURES)
def test_rotation_period_from_pds_fixture(
    fixture_name: str,
    pytestconfig: pytest.Config,
) -> None:
    if not pytestconfig.getoption("--run-rotation-period-pds"):
        pytest.skip("PDS rotation-period regression tests are opt-in.")
    if fixture_name == "__NO_FIXTURES__":
        pytest.skip("No PDS rotation-period fixtures found on disk.")

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

    result = estimate_rotation_period(
        observations,
        frequency_grid_scale=float(fx["frequency_grid_scale"][0]),
        max_frequency_cycles_per_day=float(fx["max_frequency_cycles_per_day"][0]),
        min_rotations_in_span=float(fx["min_rotations_in_span"][0]),
    )
    period_hours = float(_scalar(result.period_hours[0]))
    expected_hours = float(fx["expected_period_hours"][0])
    tolerance_fraction = float(fx["tolerance_fraction"][0])

    rel_err = abs(period_hours - expected_hours) / expected_hours
    assert rel_err <= tolerance_fraction, (
        f"{fixture_name}: fitted {period_hours:.6f} h, expected {expected_hours:.6f} h, "
        f"relative error {rel_err:.3%}, allowed {tolerance_fraction:.3%}. "
        f"Source: {fx['source_title'][0]} ({fx['source_url'][0]})"
    )
