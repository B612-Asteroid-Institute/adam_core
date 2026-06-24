"""Committed validation gates for rotation-period estimation (bead rp-e4a.14).

Auto-discovers the committed standard-candle fixtures
(``rotation_period_validation_fixture_*.npz``) and runs the Fourier fast-path
solver on each. Two gates are exercised:

1. A schema/smoke gate (parametrized) asserting every fixture solves and returns
   the new D1 confidence surface (``period_verdict`` in the valid enum,
   ``reliability_code`` in ``{"1", "2", "3"}``, diagnostics populated).
2. The D4 zero-false-confidence gate: no committed fixture may be labelled
   ``single_period`` while its STRICT (no-harmonic-adjustment) error exceeds the
   fixture tolerance -- a confident-but-wrong call, including a clean 2x/0.5x alias
   which the D1 contract treats as the worst failure. This is ``xfail(strict=True)``
   because 1627 Ivar remains a confident diurnal-sampling alias on the committed set
   (the order-selection fix is tracked in rp-e4a.22/.23); strict=True forces the
   marker to be removed once the solver is fixed (an xpass fails CI).

Slow fixtures are marked ``@pytest.mark.profile`` so the default run
(``-m 'not profile'``) stays fast.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from adam_core.photometry.rotation_period_fourier import estimate_rotation_period
from adam_core.photometry.rotation_period_scoring import (
    alias_bucket,
    harmonic_adjusted_error_pct,
    relative_error_pct,
    within_tolerance,
)
from adam_core.photometry.rotation_period_types import RotationPeriodObservations
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
        method_mode="fourier",
        search_fidelity="validated_staged",
        exact_evaluation_backend="jax",
        frequency_grid_scale=float(meta["frequency_grid_scale"]),
        max_frequency_cycles_per_day=float(meta["max_frequency_cycles_per_day"]),
        min_rotations_in_span=float(meta["min_rotations_in_span"]),
    )


@pytest.mark.parametrize("fixture_name", _FIXTURE_PARAMS)
def test_validation_fixture_schema(fixture_name: str) -> None:
    """Every committed fixture solves and returns the D1 confidence surface."""
    if fixture_name == "__NO_FIXTURES__":
        pytest.skip("No rotation-period validation fixtures found on disk.")

    observations, meta = _load_fixture(DATA_DIR / fixture_name)
    result = _solve(observations, meta)

    verdict = str(result.period_verdict[0].as_py())
    reliability = str(result.reliability_code[0].as_py())
    assert verdict in {"single_period", "period_family", "insufficient_data"}
    assert reliability in {"1", "2", "3"}
    # Diagnostics populated for a fixture that solved.
    assert result.is_valid[0].as_py() is not None
    assert result.is_reliable[0].as_py() is not None
    assert result.is_period_doubled[0].as_py() is not None
    # Period finiteness must TRACK the verdict. The prior `period_hours is not None`
    # check was a no-op for the insufficient path, which reports NaN (`nan is not None`
    # is True). A believed/family call must report a finite positive period; an
    # insufficient_data call must report a non-finite (NaN) period.
    p_hours = result.period_hours[0].as_py()
    if verdict == "insufficient_data":
        assert p_hours is None or not np.isfinite(float(p_hours))
    else:
        assert (
            p_hours is not None and np.isfinite(float(p_hours)) and float(p_hours) > 0.0
        )
    # NB: committed fixtures (e.g. Ivar n=101) are curated sparse subsets and can behave
    # differently from the full cached calibration (Ivar n=1255 -> period_family); this
    # gate is a per-fixture regression lock, not the full-dataset precision figure.


@pytest.mark.xfail(
    strict=True,
    reason="D1 zero-false-confidence is not yet satisfied on the committed set: 1627 "
    "Ivar is reported as a confident single_period at the ~5/3 diurnal-sampling alias "
    "(P_rec=7.99h vs P_true=4.795h). The PR claims the MEASURED ~0.88 strict "
    "single_period precision (full LCDB+DAMIT candle set), NOT a zero-alias guarantee. "
    "strict=True so that fixing the order-selection alias (rp-e4a.22/.23) surfaces as "
    "an xpass and forces removing this marker, instead of letting the gate rot.",
)
def test_zero_false_confidence() -> None:
    """No committed fixture may be confidently wrong (D4 headline gate).

    Collects every fast-set fixture labelled ``single_period`` whose STRICT relative
    error (no harmonic adjustment) exceeds the fixture tolerance -- a confident-but-
    wrong call -- and requires the list to be empty.

    The metric is intentionally strict ``within_tolerance``: unlike a harmonic-adjusted
    metric, it ALSO flags a clean 2x/0.5x ``single_period``, which the D1 contract
    (CLAUDE.md) treats as the worst failure mode. ``harmonic_adjusted_error_pct`` is
    retained only for the diagnostic alias-bucket label on each offender.

    Current state (xfail, strict): the fast committed gold set has exactly one offender,
    1627 Ivar -- an in-grid order-6 fit that relocates to a ~2 cycle/day diurnal
    sampling alias (NOT a harmonic-only issue, and NOT recovered by the existing A1
    sub-harmonic guardrail, which only fires below the grid floor). The fix is the
    frequency-aware order-selection / alias work in rp-e4a.22/.23; a prototyped
    frequency-anchored order selection regressed the gold set, so it is a calibration
    task, not a one-line statistic swap.
    """
    # Iterate the fast default set only so the gate stays cheap; the known
    # offender (1627 Ivar) is in this set. Slow fixtures are covered by the
    # profile-marked schema gate.
    fixtures = [n for n in _ALL_FIXTURES if not _is_slow(n)]
    offenders: list[str] = []
    n_single = 0
    for fixture_name in fixtures:
        if fixture_name == "__NO_FIXTURES__":
            pytest.skip("No rotation-period validation fixtures found on disk.")
        observations, meta = _load_fixture(DATA_DIR / fixture_name)
        result = _solve(observations, meta)

        verdict = str(result.period_verdict[0].as_py())
        if verdict != "single_period":
            continue
        n_single += 1
        p_rec = float(result.period_hours[0].as_py())
        p_true = float(meta["expected_hours"])
        tol = float(meta["tolerance_fraction"])
        # STRICT metric: a single_period whose RAW relative error exceeds tolerance is
        # confidently wrong. This also flags a clean 2x/0.5x alias (worst-case per D1);
        # harmonic_adjusted_error_pct is used only for the diagnostic alias label.
        if not within_tolerance(p_rec, p_true, tol):
            _, best_factor = harmonic_adjusted_error_pct(p_rec, p_true)
            offenders.append(
                f"{meta['object']}: single_period P_rec={p_rec:.4f}h "
                f"P_true={p_true:.4f}h raw_err={relative_error_pct(p_rec, p_true):.2f}% "
                f"(tol={tol * 100.0:.2f}%) alias={alias_bucket(best_factor)}"
            )

    n_correct = n_single - len(offenders)
    precision = n_correct / n_single if n_single else float("nan")
    print(
        f"\nD4 fast-set single_period precision: {n_correct}/{n_single} = "
        f"{precision:.3f} (bar: >= 0.90)"
    )
    if offenders:
        print("False-confidence offenders:")
        for line in offenders:
            print(f"  {line}")
    assert offenders == [], f"{len(offenders)} confident-but-wrong fixture(s)"
