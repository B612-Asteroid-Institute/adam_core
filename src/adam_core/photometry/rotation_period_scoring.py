"""Scoring helpers for the rotation-period standard-candle validation (D2 policy).

This small module collects the period-recovery scoring primitives shared by the
committed validation gates (``test_rotation_period_validation.py``) and the
out-of-package calibration report. It implements the D2 standard-candle scoring
policy: a strict relative error, a harmonic-tolerant error over a fixed factor
set, an alias-bucket label for the best-fitting harmonic, a per-fixture tolerance
check on the *raw* error, and a diurnal-cadence (near-day) alias flag.

All periods are in hours. Recovered (``p_rec``) and LCDB truth (``p_true``) are
both synodic, so no sidereal correction is applied (see D2).
"""

from __future__ import annotations

import numpy as np

HARMONIC_FACTORS: list[float] = [
    0.25,
    1.0 / 3.0,
    0.5,
    2.0 / 3.0,
    0.75,
    1.0,
    4.0 / 3.0,
    1.5,
    2.0,
    3.0,
    4.0,
]
"""Harmonic factors applied to the recovered period when diagnosing aliases."""


def relative_error_pct(p_rec: float, p_true: float) -> float:
    """Strict relative error of a recovered period, in percent.

    Parameters
    ----------
    p_rec : float
        Recovered period (hours).
    p_true : float
        True (LCDB) period (hours); must be positive.

    Returns
    -------
    float
        ``100 * |p_rec - p_true| / p_true``.
    """
    return float(100.0 * abs(float(p_rec) - float(p_true)) / float(p_true))


def harmonic_adjusted_error_pct(p_rec: float, p_true: float) -> tuple[float, float]:
    """Harmonic-tolerant relative error and the best-fitting harmonic factor.

    Minimises ``|p_rec * f - p_true| / p_true`` over :data:`HARMONIC_FACTORS`.

    Parameters
    ----------
    p_rec : float
        Recovered period (hours).
    p_true : float
        True (LCDB) period (hours); must be positive.

    Returns
    -------
    tuple of (float, float)
        The minimum harmonic-adjusted error in percent, and the factor ``f``
        that achieves it.
    """
    factors = np.asarray(HARMONIC_FACTORS, dtype=np.float64)
    errors = np.abs(float(p_rec) * factors - float(p_true)) / float(p_true)
    best = int(np.argmin(errors))
    return float(100.0 * errors[best]), float(factors[best])


def alias_bucket(best_factor: float) -> str:
    """Label the harmonic-alias bucket for the best-fitting factor.

    Parameters
    ----------
    best_factor : float
        The harmonic factor returned by :func:`harmonic_adjusted_error_pct`.

    Returns
    -------
    str
        One of ``"1x"``, ``"1/4x"``, ``"1/3x"``, ``"1/2x"``, ``"2/3x"``,
        ``"3/4x"``, ``"4/3x"``, ``"3/2x"``, ``"2x"``, ``"3x"``, ``"4x"``, or
        ``"other"`` if no factor is within 5% of ``best_factor``.
    """
    labels: list[tuple[float, str]] = [
        (1.0, "1x"),
        (0.25, "1/4x"),
        (1.0 / 3.0, "1/3x"),
        (0.5, "1/2x"),
        (2.0 / 3.0, "2/3x"),
        (0.75, "3/4x"),
        (4.0 / 3.0, "4/3x"),
        (1.5, "3/2x"),
        (2.0, "2x"),
        (3.0, "3x"),
        (4.0, "4x"),
    ]
    factor = float(best_factor)
    for value, label in labels:
        if abs(factor - value) <= 0.05 * value:
            return label
    return "other"


def within_tolerance(p_rec: float, p_true: float, tolerance_fraction: float) -> bool:
    """Whether the raw relative error is within the fixture's tolerance.

    Uses the strict :func:`relative_error_pct` (no harmonic adjustment) compared
    against the per-fixture ``tolerance_fraction`` from the npz.

    Parameters
    ----------
    p_rec : float
        Recovered period (hours).
    p_true : float
        True (LCDB) period (hours); must be positive.
    tolerance_fraction : float
        Allowed fractional error (e.g. ``0.01`` for a tight U=3 fixture).

    Returns
    -------
    bool
        ``True`` if ``relative_error_pct(p_rec, p_true) <= tolerance_fraction * 100``.
    """
    return bool(relative_error_pct(p_rec, p_true) <= float(tolerance_fraction) * 100.0)


def near_day_alias(
    p_rec_hours: float,
    p_true_hours: float,
    tolerance_fraction: float = 0.02,
) -> bool:
    """Whether the recovered frequency is a diurnal-cadence alias of the truth.

    Flags lock-on to a day-aliased frequency (cycles per day): the recovered
    frequency sits within ``tolerance_fraction`` of ``f_true +/- n`` for ``n`` in
    ``{1, 2}`` cycles/day, i.e. ``|f_rec - (f_true +/- n)| / f_true <= tol``.

    Parameters
    ----------
    p_rec_hours : float
        Recovered period (hours); must be positive.
    p_true_hours : float
        True (LCDB) period (hours); must be positive.
    tolerance_fraction : float, optional
        Fractional tolerance on the day-aliased frequency match (default 0.02).

    Returns
    -------
    bool
        ``True`` if a day-alias relationship holds.
    """
    p_rec = float(p_rec_hours)
    p_true = float(p_true_hours)
    if p_rec <= 0.0 or p_true <= 0.0:
        return False
    f_rec = 24.0 / p_rec
    f_true = 24.0 / p_true
    tol = float(tolerance_fraction)
    for n in (1, 2):
        for aliased in (f_true + n, f_true - n):
            if aliased <= 0.0:
                continue
            if abs(f_rec - aliased) / f_true <= tol:
                return True
    return False
