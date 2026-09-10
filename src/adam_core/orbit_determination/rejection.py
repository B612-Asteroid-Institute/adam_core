"""
Outlier rejection with re-inclusion after Carpino, Milani & Chesley (2003).

Carpino, Milani & Chesley (2003, Icarus 166, 248; "CMC2003") reject and
re-include observations by testing each observation's post-fit residual
against its EXPECTED residual covariance rather than against its reported
uncertainty alone. The scheme is the one implemented in OrbFit
(``src/propag/least_squares.f90`` ``reject_obs``, defaults in
``lib/reject.def``) and used by NEODyS/AstDyS; this module ports it as an
outlier treatment for adam_core's differential correction, selectable as
``outlier_rejection="cmc2003"`` on `NativeOrbitFitter`.

Algorithm (one pass; passes repeat until no observation changes state)
----------------------------------------------------------------------
1. Fit the currently selected observations (`fit_least_squares`), warm-started
   from the previous pass.
2. For every observation i, compute the whitened residual ``r_i`` (two
   components, so ``|r_i|**2`` is the observation's chi2) and the covariance
   of the fitted prediction projected onto its sky plane, ``P_i = J_i C J_i^T``,
   with ``J_i`` the whitened observation Jacobian and ``C`` the fit
   covariance. The expected residual covariance is ``I - P_i`` for an
   observation INSIDE the fit (the fit absorbs part of its error) and
   ``I + P_i`` for one OUTSIDE it; ``chi2_i = r_i^T (I -/+ P_i)^-1 r_i``.
   The projection is OrbFit's linear ``A Gamma A^T`` in whitened units,
   evaluated with the analytic 2-body Jacobian of `fit_least_squares`.
3. Re-include an excluded observation when ``chi2_i <= chi2_recover +
   0.75 * fudge``; reject a selected one when ``chi2_i >= threshold`` and
   ``chi2_i > chi2_reject + fudge``, where ``threshold`` is ``chi2_frac`` times
   the worst selected chi2 (so only observations within a fraction of the
   worst offender go in one pass) or the worst chi2 itself in
   one-at-a-time mode (few observations, ``n_selected <= 6 * min_obs``, or
   four passes with the same number of modifications). The small-sample
   fudge ``400 * 3**(-n_selected)`` follows OrbFit's error-model form and is
   negligible beyond ~8 observations.
4. Guards: no observation is rejected when ``n <= min_obs``; the last
   selected observation of an apparition (time-sorted groups split at gaps
   longer than ``apparition_gap_days``) is never rejected; rejection stops
   when the selected fraction would drop below
   ``1 - max_rejected_fraction``.

Every call starts with all observations selected, so decisions do not
cascade across calls.

Constants are OrbFit's ``reject.def`` defaults: chi2_reject 8, chi2_recover 7,
chi2_frac 0.25, at most 15 passes, at most 50% rejected, 180-day apparition
gap, and a 5% eigenvalue floor on the expected residual covariance (whitened
units) protecting against ``I - P_i`` losing positive definiteness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ..orbits.orbits import Orbits
from ..propagator.propagator import Propagator
from .differential_correction import (
    HUBER_F_SCALE_DEFAULT,
    LossType,
    _analytic_jacobian,
    _analytic_jacobian_terms,
    _observation_whitening_matrices,
    fit_least_squares,
)
from .evaluate import OrbitDeterminationObservations
from .fitted_orbits import FittedOrbitMembers, FittedOrbits

logger = logging.getLogger(__name__)

__all__ = [
    "CMC2003_APPARITION_GAP_DAYS",
    "CMC2003_CHI2_FRAC",
    "CMC2003_CHI2_RECOVER",
    "CMC2003_CHI2_REJECT",
    "CMC2003_MAX_ITERATIONS",
    "CMC2003_MAX_REJECTED_FRACTION",
    "CMC2003_PSD_FLOOR_FRAC",
    "CMC2003Fit",
    "cmc2003_fit",
    "cmc2003_fit_detailed",
]

# OrbFit lib/reject.def defaults
CMC2003_CHI2_REJECT = 8.0
CMC2003_CHI2_RECOVER = 7.0
CMC2003_CHI2_FRAC = 0.25
CMC2003_MAX_ITERATIONS = 15
CMC2003_MAX_REJECTED_FRACTION = 0.5
CMC2003_APPARITION_GAP_DAYS = 180.0
# Eigenvalue floor on the expected residual covariance, as a fraction of the
# (whitened, i.e. unit) observation variance.
CMC2003_PSD_FLOOR_FRAC = 0.05

_N_PARAMETERS = 6
# OrbFit: no rejection at all when n <= round(0.5 * n_params)
_MIN_OBS = round(0.5 * _N_PARAMETERS)


@dataclass(frozen=True)
class CMC2003Fit:
    """
    Result of `cmc2003_fit_detailed`: the final fit, members for every input
    observation, and run diagnostics.
    """

    #: Final fit to the selected observations (`FittedOrbits`, 1 row).
    fitted_orbit: FittedOrbits
    #: One row per input observation, in input order: residuals with respect
    #: to the final orbit, ``outlier`` True for rejected observations,
    #: ``solution`` True for selected ones, and the fit ``weight``.
    fitted_orbit_members: FittedOrbitMembers
    #: Number of fit passes performed.
    n_iterations: int
    #: Observations excluded from the final fit.
    n_rejected: int
    #: Re-inclusion events summed over all passes.
    n_recovered: int
    #: Diagnostic flags raised during the run (sorted, may be empty):
    #: ``"too_few_observations"``, ``"max_iterations"``,
    #: ``"max_rejected_fraction"``, ``"kept_last_in_apparition"``,
    #: ``"psd_floor"``, ``"no_fit_covariance"``,
    #: ``"singular_residual_covariance"``, ``"non_finite_chi2"``.
    flags: tuple[str, ...]


def _apparitions(
    mjd: npt.NDArray[np.float64], gap_days: float
) -> npt.NDArray[np.int64]:
    """Apparition index per observation: time-sorted groups split at gaps
    longer than ``gap_days``."""
    order = np.argsort(mjd, kind="stable").tolist()
    apparition = np.zeros(len(mjd), dtype=np.int64)
    current = 0
    previous: float | None = None
    for i in order:
        if previous is not None and mjd[i] - previous > gap_days:
            current += 1
        apparition[i] = current
        previous = float(mjd[i])
    return apparition


def _expected_residual_chi2(
    residuals: npt.NDArray[np.float64],
    jacobian: npt.NDArray[np.float64],
    covariance: npt.NDArray[np.float64] | None,
    selected: npt.NDArray[np.bool_],
    psd_floor_frac: float = CMC2003_PSD_FLOOR_FRAC,
) -> tuple[npt.NDArray[np.float64], set[str]]:
    """
    Per-observation chi2 against the expected post-fit residual covariance.

    Parameters
    ----------
    residuals : (N, 2)
        Whitened (lon, lat) residual components of every observation with
        respect to the fitted orbit.
    jacobian : (2N, 6)
        Jacobian of the whitened residual vector with respect to the fitted
        state (rows 2i, 2i+1 belong to observation i).
    covariance : (6, 6) or None
        Fit covariance. None, or a non-finite matrix, is treated as zero
        prediction uncertainty (chi2 reduces to the plain residual chi2) and
        raises the ``"no_fit_covariance"`` flag.
    selected : (N,) bool
        Whether each observation is inside the current fit (``I - P``) or
        outside it (``I + P``).
    psd_floor_frac : float
        Eigenvalue floor for the expected residual covariance in whitened
        units.

    Returns
    -------
    chi2 : (N,)
        Expected-covariance chi2 per observation (non-negative).
    flags : set of str
        Diagnostic flags raised while computing.
    """
    n = len(residuals)
    chi2 = np.zeros(n, dtype=np.float64)
    flags: set[str] = set()
    identity = np.eye(2)

    use_covariance = covariance is not None and bool(np.all(np.isfinite(covariance)))
    if not use_covariance:
        flags.add("no_fit_covariance")

    for i in range(n):
        r = residuals[i]
        if use_covariance:
            assert covariance is not None
            j = jacobian[2 * i : 2 * i + 2]
            projected = j @ covariance @ j.T
            expected = identity - projected if selected[i] else identity + projected
        else:
            expected = identity
        eigenvalues, eigenvectors = np.linalg.eigh(expected)
        if eigenvalues[0] < psd_floor_frac:
            flags.add("psd_floor")
            eigenvalues = np.maximum(eigenvalues, psd_floor_frac)
            expected = (eigenvectors * eigenvalues) @ eigenvectors.T
        try:
            value = float(r @ np.linalg.solve(expected, r))
        except np.linalg.LinAlgError:
            flags.add("singular_residual_covariance")
            value = 0.0
        if not np.isfinite(value):
            flags.add("non_finite_chi2")
            value = 0.0
        chi2[i] = max(value, 0.0)
    return chi2, flags


def _cmc2003_select(
    chi2: npt.NDArray[np.float64],
    selected: npt.NDArray[np.bool_],
    apparitions: npt.NDArray[np.int64],
    *,
    chi2_reject: float,
    chi2_recover: float,
    chi2_frac: float,
    max_rejected_fraction: float,
    one_at_a_time: bool,
    min_obs: int = _MIN_OBS,
) -> tuple[npt.NDArray[np.bool_], int, int, set[str]]:
    """
    One CMC2003 reject / re-include decision pass.

    Returns the new selection mask, the number of rejections, the number of
    re-inclusions and any diagnostic flags. See the module docstring for the
    rules; ``one_at_a_time`` selects the single-worst-observation batch rule.
    """
    n = len(chi2)
    n_selected = int(selected.sum())
    flags: set[str] = set()
    new_selected = selected.copy()

    chi2_max = float(chi2[selected].max()) if n_selected else 0.0
    threshold = chi2_max if one_at_a_time else chi2_max * chi2_frac
    fudge = 400.0 * 3.0 ** (-n_selected)

    n_recovered = 0
    for i in np.flatnonzero(~selected).tolist():
        if chi2[i] <= chi2_recover + 0.75 * fudge:
            new_selected[i] = True
            n_recovered += 1

    n_rejected = 0
    min_keep = int(np.ceil(n * (1.0 - max_rejected_fraction)))
    candidates = [
        i
        for i in np.flatnonzero(selected).tolist()
        if chi2[i] >= threshold and chi2[i] > chi2_reject + fudge
    ]
    candidates.sort(key=lambda i: -chi2[i])
    for i in candidates:
        if int(new_selected.sum()) - 1 < max(min_keep, min_obs + 1):
            flags.add("max_rejected_fraction")
            break
        same_apparition = new_selected & (apparitions == apparitions[i])
        if int(same_apparition.sum()) <= 1:
            flags.add("kept_last_in_apparition")
            continue
        new_selected[i] = False
        n_rejected += 1

    return new_selected, n_rejected, n_recovered, flags


def _whitened_residuals(
    members: FittedOrbitMembers, whiteners: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """(N, 2) whitened (lon, lat) residual components from fitted members."""
    values = np.stack(members.residuals.values.to_numpy(zero_copy_only=False))[:, 1:3]
    return np.asarray(np.einsum("nij,nj->ni", whiteners, values), dtype=np.float64)


def cmc2003_fit_detailed(
    orbit: Orbits,
    observations: OrbitDeterminationObservations,
    propagator: Propagator,
    *,
    chi2_reject: float = CMC2003_CHI2_REJECT,
    chi2_recover: float = CMC2003_CHI2_RECOVER,
    chi2_frac: float = CMC2003_CHI2_FRAC,
    max_iterations: int = CMC2003_MAX_ITERATIONS,
    max_rejected_fraction: float = CMC2003_MAX_REJECTED_FRACTION,
    apparition_gap_days: float = CMC2003_APPARITION_GAP_DAYS,
    psd_floor_frac: float = CMC2003_PSD_FLOOR_FRAC,
    loss: LossType = "linear",
    f_scale: float = HUBER_F_SCALE_DEFAULT,
    validate_covariance: bool = True,
    **kwargs: Any,
) -> CMC2003Fit:
    """
    Differentially correct an orbit with CMC2003 outlier rejection and
    re-inclusion, returning the fit together with run diagnostics.

    Parameters
    ----------
    orbit : `~adam_core.orbits.Orbits` (1)
        Initial orbit (e.g. from IOD); its epoch is the fit epoch.
    observations : `OrbitDeterminationObservations` (N)
        Observations believed to belong to the object. All start selected.
    propagator : `~adam_core.propagator.Propagator`
        Propagator used by `fit_least_squares`.
    chi2_reject, chi2_recover : float
        Hysteresis thresholds on the expected-covariance chi2 (OrbFit
        defaults 8 and 7).
    chi2_frac : float
        Batch rule: only selected observations with chi2 within this fraction
        of the worst offender are rejected in one pass (0.25).
    max_iterations : int
        Maximum number of fit passes (15). If reached, a final fit to the
        final selection is performed and ``"max_iterations"`` is flagged.
    max_rejected_fraction : float
        Never reject more than this fraction of the observations (0.5).
    apparition_gap_days : float
        Gap splitting apparitions; the last selected observation of an
        apparition is never rejected (180 days).
    psd_floor_frac : float
        Eigenvalue floor on the expected residual covariance in whitened
        units (0.05).
    loss, f_scale : see `fit_least_squares`
        Loss used by every fit pass; ``"huber"`` composes robust fitting with
        rejection.
    validate_covariance : bool
        Forwarded to `fit_least_squares` for every pass.
    **kwargs
        Further keyword arguments for `fit_least_squares` (e.g. ``jacobian``,
        ``max_nfev``). ``ignore`` is managed by this function and must not be
        passed.

    Returns
    -------
    result : `CMC2003Fit`
        Final fit, members for all N observations, and diagnostics.

    Raises
    ------
    ValueError
        If ``observations`` is empty or ``ignore`` is passed in ``kwargs``.
    """
    n = len(observations)
    if n == 0:
        raise ValueError("cmc2003_fit requires at least one observation")
    if "ignore" in kwargs:
        raise ValueError("ignore is managed by cmc2003_fit and cannot be passed")
    assert len(orbit) == 1, "Only one orbit can be fitted"

    ids = observations.id.to_numpy(zero_copy_only=False)
    epoch_mjd_tdb = float(
        orbit.coordinates.time.rescale("tdb").mjd().to_numpy(zero_copy_only=False)[0]
    )
    mjd_utc = (
        observations.coordinates.time.rescale("utc")
        .mjd()
        .to_numpy(zero_copy_only=False)
    )
    apparitions = _apparitions(mjd_utc, apparition_gap_days)
    whiteners = _observation_whitening_matrices(observations)
    jacobian_terms = _analytic_jacobian_terms(observations)

    def fit(
        seed: Orbits, selected: npt.NDArray[np.bool_]
    ) -> tuple[FittedOrbits, FittedOrbitMembers]:
        ignore = [str(i) for i in ids[~selected]]
        return fit_least_squares(
            seed,
            observations,
            propagator,
            ignore=ignore if ignore else None,
            loss=loss,
            f_scale=f_scale,
            validate_covariance=validate_covariance,
            **kwargs,
        )

    selected = np.ones(n, dtype=bool)
    flags: set[str] = set()
    n_recovered_total = 0
    stuck_passes = 0
    previous_modifications: int | None = None
    seed = orbit
    n_iterations = 0
    converged = False

    for iteration in range(max_iterations):
        n_iterations = iteration + 1
        fitted_orbit, fitted_orbit_members = fit(seed, selected)
        seed = fitted_orbit.to_orbits()

        if n <= _MIN_OBS:
            flags.add("too_few_observations")
            converged = True
            break

        state = np.concatenate(
            [fitted_orbit.coordinates.r[0], fitted_orbit.coordinates.v[0]]
        )
        covariance = fitted_orbit.coordinates.covariance.to_matrix()[0]
        jacobian = _analytic_jacobian(state, epoch_mjd_tdb, jacobian_terms)
        residuals = _whitened_residuals(fitted_orbit_members, whiteners)
        chi2, pass_flags = _expected_residual_chi2(
            residuals, jacobian, covariance, selected, psd_floor_frac
        )
        flags |= pass_flags

        n_selected = int(selected.sum())
        one_at_a_time = n_selected <= 6 * _MIN_OBS or stuck_passes >= 4
        new_selected, n_rejected, n_recovered, select_flags = _cmc2003_select(
            chi2,
            selected,
            apparitions,
            chi2_reject=chi2_reject,
            chi2_recover=chi2_recover,
            chi2_frac=chi2_frac,
            max_rejected_fraction=max_rejected_fraction,
            one_at_a_time=one_at_a_time,
        )
        flags |= select_flags
        n_modifications = n_rejected + n_recovered
        n_recovered_total += n_recovered
        logger.debug(
            "CMC2003 pass %d: %d selected, worst chi2 %.2f, %d rejected, %d recovered",
            n_iterations,
            n_selected,
            float(chi2[selected].max()) if n_selected else 0.0,
            n_rejected,
            n_recovered,
        )
        stuck_passes = (
            stuck_passes + 1
            if previous_modifications == n_modifications and n_modifications > 0
            else 0
        )
        previous_modifications = n_modifications
        if n_modifications == 0:
            converged = True
            break
        selected = new_selected

    if not converged:
        # The selection changed after the last fit: refit so that the returned
        # orbit and members describe the final selection.
        flags.add("max_iterations")
        fitted_orbit, fitted_orbit_members = fit(seed, selected)

    return CMC2003Fit(
        fitted_orbit=fitted_orbit,
        fitted_orbit_members=fitted_orbit_members,
        n_iterations=n_iterations,
        n_rejected=int((~selected).sum()),
        n_recovered=n_recovered_total,
        flags=tuple(sorted(flags)),
    )


def cmc2003_fit(
    orbit: Orbits,
    observations: OrbitDeterminationObservations,
    propagator: Propagator,
    **kwargs: Any,
) -> tuple[FittedOrbits, FittedOrbitMembers]:
    """
    Differentially correct an orbit with CMC2003 outlier rejection and
    re-inclusion.

    Convenience wrapper around `cmc2003_fit_detailed` returning only the
    fitted orbit and members (rejected observations have ``outlier`` True).
    All keyword arguments are forwarded; see `cmc2003_fit_detailed`.

    Returns
    -------
    fitted_orbit : `FittedOrbits` (1)
    fitted_orbit_members : `FittedOrbitMembers` (N)
    """
    result = cmc2003_fit_detailed(orbit, observations, propagator, **kwargs)
    return result.fitted_orbit, result.fitted_orbit_members
