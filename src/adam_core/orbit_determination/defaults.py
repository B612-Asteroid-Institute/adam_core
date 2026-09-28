"""
The shipped orbit-determination defaults, and the notes that explain them.

There is no configuration object: every default is the default value of the
signature that owns it, so calling the pieces with no arguments gives the
shipped behaviour. The one thing a signature cannot express is the
observation-model STACK (an ordered composition with external tables), which
`default_observation_models` builds and `run_od` applies when it is called
without ``models``.

NOTES: the default configuration (decision 2026-09-23)
======================================================

Outcome of the Asteroid Institute 100-object walk-forward study
(``adam_od_experiments``, defaults handoff of 2026-09-23). Every lever, its
default, and the signature that owns it:

Observation models, applied in this order by `run_od` (default ``models``)
---------------------------------------------------------------------------
1. Sigma where the MPC reports none
       `SigmaFillModel` with the ``v2_sigma_fill`` table of the private
       ``adam-observatory-uncertainties`` package (`load_sigma_table`), global
       fallback 0.75" (`VERES2017_FALLBACK_SIGMA_ARCSEC`). Fills ONLY missing
       sigmas; reported sigmas are never touched. Runs first so every later
       model sees a finite covariance.
2. Star-catalog debiasing (the only default that MOVES positions)
       `EFCC18DebiasModel`: JPL ``bias.dat`` (``jpl_debias_2018`` package,
       env var or cache; see `adam_core.observations.efcc18`), HEALPix RING
       order, every tabulated catalog corrected (``exclude_astcats=()``;
       `EFCC18_JPL_UNDEBIASED_ASTCATS` is opt-in).
3. Station weighting
       `EmpiricalCovarianceModel` with the ``v2_full`` bias table
       (`load_bias_table`), ``mode="add"``, ``min_resid_cov_n=30``: the
       station's measured 2x2 residual covariance is ADDED to the reported
       covariance; stations with fewer residuals pass through.
4. Nightly deweighting
       `NightBatchDeweightingModel(cap=4)`: sigma scaled by sqrt(N/4) for
       same-station same-night batches with N > 4.

Fitter (`NativeOrbitFitter` defaults)
-------------------------------------
- ``outlier_rejection="cmc2003"``: Carpino-Milani-Chesley (2003) rejection
  with re-inclusion (`cmc2003_fit`), OrbFit ``reject.def`` constants:
  chi2 reject 8, recover 7, frac 0.25, 15 passes, at most 50 % rejected,
  180-day apparitions, 5 % eigenvalue floor (`CMC2003_*` constants).
- ``loss="linear"``, ``f_scale=1.345`` (Huber is available, not default).
- Differential correction `fit_least_squares` defaults: whitened 2N residuals,
  ``jacobian="analytic"`` (exact 2-body STM, autodiff), ``validate_covariance``
  True (weak-direction probe with central-difference fallback).
- IOD: Gauss on every triplet (``observation_selection_method="combinations"``),
  ``iod_rchi2_threshold=200``, ``min_obs=6``, ``min_arc_length=1`` day,
  ``contamination_percentage=20``, ``rchi2_threshold=10`` (under CMC2003 the
  last four bound IOD only).

Provenance and scoring
----------------------
- The models act on the FIT only. `FittedOrbitMembers` record original vs
  used astrometry, ``astcat`` and ``weight``. Judge a held-out observation
  against its nominal position and ORIGINAL sigma (2026-09-16 rule).

Options kept, not defaults
--------------------------
- ``loss="huber"`` (open covariance pathology on short arcs).
- `VeresFloorModel` / `VeresReplaceModel` (``veres2017_working`` table is the
  legacy reference).
- `SigmaFloorModel` (no-op on the study), `PerformanceWeightedModel`
  (over-inflates by about 1.5x), ``NightBatchDeweightingModel(cap=1)``
  (about 1.9x inflation, no position benefit), ``mode="replace"``.
- ``outlier_rejection="worst_residual"`` (the pre-2026-09 loop).

Opting out
----------
- ``run_od(..., models=None)`` fits the observations exactly as supplied;
  pass your own model list to pin a study. The tables can be injected
  (``default_observation_models(bias_table=..., sigma_table=...)``).
- The lower-level entry points (`initial_orbit_determination`,
  `differential_correction`, `iterative_fit`, ...) take an optional
  ``observatory_bias_model`` that defaults to None (identity): they are the
  building blocks, not the shipped pipeline, and must work without the data
  packages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from .observation_uncertainty import (
    EFCC18DebiasModel,
    EmpiricalCovarianceModel,
    NightBatchDeweightingModel,
    ObservationUncertaintyModel,
    load_bias_table,
)
from .veres2017 import SigmaFillModel, load_sigma_table

__all__ = ["default_observation_models"]


def default_observation_models(
    bias_table: Optional[pa.Table] = None,
    sigma_table: Optional[pa.Table] = None,
    efcc18_bias_table: Optional[npt.NDArray[np.floating]] = None,
    efcc18_bias_dat: Optional[Union[str, Path]] = None,
) -> list[ObservationUncertaintyModel]:
    """
    The default observation-model stack, in application order: what `run_od`
    applies when called without ``models`` (see the module notes).

    Parameters
    ----------
    bias_table : `pyarrow.Table`, optional
        Observatory bias table (`BIAS_TABLE_SCHEMA`) for the empirical
        covariance model. Default: the ``v2_full`` table of the
        ``adam-observatory-uncertainties`` data package (`load_bias_table`).
    sigma_table : `pyarrow.Table`, optional
        Station/catalog sigma table (`VERES2017_SIGMA_TABLE_SCHEMA`) for the
        sigma fill. Default: the ``v2_sigma_fill`` table of the data package
        (`load_sigma_table`).
    efcc18_bias_table : `numpy.ndarray` (49152, 26, 4), optional
        Pre-loaded EFCC18 table; otherwise located from ``efcc18_bias_dat``,
        the environment, the ``jpl_debias_2018`` package or the cache.
    efcc18_bias_dat : path, optional
        Explicit ``bias.dat`` location (only when ``efcc18_bias_table`` is None).

    Returns
    -------
    models : list of `ObservationUncertaintyModel`
        ``[SigmaFillModel, EFCC18DebiasModel, EmpiricalCovarianceModel,
        NightBatchDeweightingModel]``.

    Raises
    ------
    ImportError
        If a table is not passed and the ``adam-observatory-uncertainties``
        package is not installed.
    FileNotFoundError
        If no EFCC18 ``bias.dat`` can be located.
    """
    if sigma_table is None:
        sigma_table = load_sigma_table()
    if bias_table is None:
        bias_table = load_bias_table()
    return [
        SigmaFillModel(sigma_table),
        EFCC18DebiasModel(bias_table=efcc18_bias_table, bias_dat=efcc18_bias_dat),
        EmpiricalCovarianceModel(bias_table),
        NightBatchDeweightingModel(),
    ]
