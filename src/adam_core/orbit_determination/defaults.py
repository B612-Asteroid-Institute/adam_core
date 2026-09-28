"""
The Asteroid Institute default orbit-determination settings, in one place.

Every lever of the fit-time observation handling and of the fitter has a
value here (`OD_DEFAULTS`), and two factories build the shipped configuration
from those values so that downstream pipelines reproduce it by construction
instead of re-assembling it:

* `default_observation_models` -- the model stack applied to the ORIGINAL
  observations at fit time by `~adam_core.orbit_determination.run_od`, in the
  prescribed order: fill sigmas the MPC does not report (`SigmaFillModel`,
  ``v2_sigma_fill``), debias positions (`EFCC18DebiasModel`, RING order), add
  the station's empirical residual covariance (`EmpiricalCovarianceModel`,
  ``v2_full``, ``mode="add"``), deweight same-station same-night batches
  (`NightBatchDeweightingModel`, ``cap=4``).
* `default_orbit_fitter` -- `NativeOrbitFitter` with Carpino-Milani-Chesley
  (2003) rejection, the linear loss, and the analytic-Jacobian /
  validated-covariance differential correction.

The values are the outcome of the Asteroid Institute 100-object walk-forward
study (decision 2026-09-23, ``adam_od_experiments`` defaults handoff). Kept as
options but NOT defaults: ``loss="huber"`` (open covariance pathology on short
arcs), `VeresFloorModel` / `VeresReplaceModel`, `SigmaFloorModel` (no-op on
that study), `PerformanceWeightedModel` (over-inflates by about 1.5x) and
``NightBatchDeweightingModel(cap=1)`` (about 1.9x inflation, no position
benefit). The models act on the fit only: judge a held-out observation against
its nominal position and original sigma.

The tables are data of the private ``adam-observatory-uncertainties`` package
and JPL's ``bias.dat`` (``jpl_debias_2018``); the factories resolve them by
import when no table is passed and raise an `ImportError` / `FileNotFoundError`
naming the missing data otherwise. Pass tables explicitly to pin a study.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Type, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from ..propagator.propagator import Propagator
from .differential_correction import HUBER_F_SCALE_DEFAULT, LossType
from .native_orbit_fitter import NativeOrbitFitter
from .observation_uncertainty import (
    V2_FULL_BIAS_TABLE,
    EFCC18DebiasModel,
    EmpiricalCovarianceModel,
    NightBatchDeweightingModel,
    ObservationUncertaintyModel,
    load_bias_table,
)
from .rejection import (
    CMC2003_APPARITION_GAP_DAYS,
    CMC2003_CHI2_FRAC,
    CMC2003_CHI2_RECOVER,
    CMC2003_CHI2_REJECT,
    CMC2003_MAX_ITERATIONS,
    CMC2003_MAX_REJECTED_FRACTION,
    CMC2003_PSD_FLOOR_FRAC,
)
from .veres2017 import (
    V2_SIGMA_FILL_TABLE,
    VERES2017_FALLBACK_SIGMA_ARCSEC,
    SigmaFillModel,
    load_sigma_table,
)

__all__ = [
    "OD_DEFAULTS",
    "OrbitDeterminationDefaults",
    "default_observation_models",
    "default_orbit_fitter",
]


@dataclass(frozen=True)
class OrbitDeterminationDefaults:
    """
    Every default lever of the shipped orbit-determination configuration.

    Observation models (applied in this order by `default_observation_models`)
    -------------------------------------------------------------------------
    sigma_fill_table : str
        Data-package sigma table `SigmaFillModel` fills missing MPC sigmas from.
    sigma_fill_fallback_arcsec : float
        Sigma used when nothing in that table matches (both axes).
    efcc18_exclude_astcats : tuple of str
        Catalogs `EFCC18DebiasModel` leaves uncorrected although tabulated.
        Empty: every tabulated catalog is debiased (the study's behaviour;
        pass `EFCC18_JPL_UNDEBIASED_ASTCATS` to follow JPL's practice).
    bias_table : str
        Data-package observatory bias table for `EmpiricalCovarianceModel`.
    empirical_covariance_mode : {"add", "replace"}
        Combine the measured residual covariance with the reported one.
    min_resid_cov_n : int
        Stations with fewer residuals than this pass through unchanged.
    night_batch_cap : int
        Same-station same-night batches larger than this are scaled by N/cap.

    Fitter (`default_orbit_fitter`)
    --------------------------------
    outlier_rejection : {"worst_residual", "cmc2003"}
        Outlier treatment of the differential correction.
    cmc2003_chi2_reject, cmc2003_chi2_recover, cmc2003_chi2_frac,
    cmc2003_max_iterations, cmc2003_max_rejected_fraction,
    cmc2003_apparition_gap_days, cmc2003_psd_floor_frac
        OrbFit ``reject.def`` constants used by the CMC2003 scheme.
    loss : {"linear", "huber"}, f_scale : float
        Loss of the differential correction (Huber is an option, not default).
    jacobian : {"analytic", "central", "2-point"}, validate_covariance : bool
        Jacobian and covariance validation of `fit_least_squares`.
    min_obs, min_arc_length, contamination_percentage, rchi2_threshold
        `NativeOrbitFitter` IOD / worst-residual loop bounds (rchi2 and
        contamination apply to IOD only under CMC2003 rejection).
    iod_rchi2_threshold, observation_selection_method
        Gauss IOD acceptance threshold and triplet selection.
    """

    sigma_fill_table: str = V2_SIGMA_FILL_TABLE
    sigma_fill_fallback_arcsec: float = VERES2017_FALLBACK_SIGMA_ARCSEC
    efcc18_exclude_astcats: tuple[str, ...] = ()
    bias_table: str = V2_FULL_BIAS_TABLE
    empirical_covariance_mode: Literal["add", "replace"] = "add"
    min_resid_cov_n: int = 30
    night_batch_cap: int = 4

    outlier_rejection: Literal["worst_residual", "cmc2003"] = "cmc2003"
    cmc2003_chi2_reject: float = CMC2003_CHI2_REJECT
    cmc2003_chi2_recover: float = CMC2003_CHI2_RECOVER
    cmc2003_chi2_frac: float = CMC2003_CHI2_FRAC
    cmc2003_max_iterations: int = CMC2003_MAX_ITERATIONS
    cmc2003_max_rejected_fraction: float = CMC2003_MAX_REJECTED_FRACTION
    cmc2003_apparition_gap_days: float = CMC2003_APPARITION_GAP_DAYS
    cmc2003_psd_floor_frac: float = CMC2003_PSD_FLOOR_FRAC
    loss: LossType = "linear"
    f_scale: float = HUBER_F_SCALE_DEFAULT
    jacobian: Literal["analytic", "central", "2-point"] = "analytic"
    validate_covariance: bool = True
    min_obs: int = 6
    min_arc_length: float = 1.0
    contamination_percentage: float = 20.0
    rchi2_threshold: float = 10.0
    iod_rchi2_threshold: float = 200.0
    observation_selection_method: Literal[
        "combinations", "first+middle+last", "thirds"
    ] = "combinations"

    def model_order(self) -> tuple[str, ...]:
        """Class names of the default model stack, in application order."""
        return (
            "SigmaFillModel",
            "EFCC18DebiasModel",
            "EmpiricalCovarianceModel",
            "NightBatchDeweightingModel",
        )


#: The shipped defaults (decision 2026-09-23).
OD_DEFAULTS = OrbitDeterminationDefaults()


def default_observation_models(
    bias_table: Optional[pa.Table] = None,
    sigma_table: Optional[pa.Table] = None,
    efcc18_bias_table: Optional[npt.NDArray[np.floating]] = None,
    efcc18_bias_dat: Optional[Union[str, Path]] = None,
    defaults: OrbitDeterminationDefaults = OD_DEFAULTS,
) -> list[ObservationUncertaintyModel]:
    """
    The default observation-model stack, in application order, for
    `~adam_core.orbit_determination.run_od`.

    Parameters
    ----------
    bias_table : `pyarrow.Table`, optional
        Observatory bias table (`BIAS_TABLE_SCHEMA`). Default: the
        ``defaults.bias_table`` table of the ``adam-observatory-uncertainties``
        data package.
    sigma_table : `pyarrow.Table`, optional
        Station/catalog sigma table (`VERES2017_SIGMA_TABLE_SCHEMA`). Default:
        the ``defaults.sigma_fill_table`` table of the data package.
    efcc18_bias_table : `numpy.ndarray` (49152, 26, 4), optional
        Pre-loaded EFCC18 table; otherwise located from ``efcc18_bias_dat``,
        the environment, the ``jpl_debias_2018`` package or the cache.
    efcc18_bias_dat : path, optional
        Explicit ``bias.dat`` location (only when ``efcc18_bias_table`` is None).
    defaults : `OrbitDeterminationDefaults`
        Lever values; `OD_DEFAULTS` unless overridden.

    Returns
    -------
    models : list of `ObservationUncertaintyModel`
        ``[SigmaFillModel, EFCC18DebiasModel, EmpiricalCovarianceModel,
        NightBatchDeweightingModel]`` configured from ``defaults``.
    """
    if sigma_table is None:
        sigma_table = load_sigma_table(defaults.sigma_fill_table)
    if bias_table is None:
        bias_table = load_bias_table(defaults.bias_table)
    return [
        SigmaFillModel(
            sigma_table, fallback_sigma_arcsec=defaults.sigma_fill_fallback_arcsec
        ),
        EFCC18DebiasModel(
            bias_table=efcc18_bias_table,
            bias_dat=efcc18_bias_dat,
            exclude_astcats=defaults.efcc18_exclude_astcats,
        ),
        EmpiricalCovarianceModel(
            bias_table,
            mode=defaults.empirical_covariance_mode,
            min_resid_cov_n=defaults.min_resid_cov_n,
        ),
        NightBatchDeweightingModel(cap=defaults.night_batch_cap),
    ]


def default_orbit_fitter(
    propagator_class: Type[Propagator],
    propagator_kwargs: Optional[dict[str, Any]] = None,
    defaults: OrbitDeterminationDefaults = OD_DEFAULTS,
    **overrides: Any,
) -> NativeOrbitFitter:
    """
    `NativeOrbitFitter` configured with the shipped defaults: CMC2003 outlier
    rejection with OrbFit's constants, the linear loss, and the
    analytic-Jacobian / validated-covariance differential correction.

    Parameters
    ----------
    propagator_class : type
        Propagator class used for IOD ephemerides (an instance of it, or any
        other propagator, is passed separately to `run_od` for refinement).
    propagator_kwargs : dict, optional
        Constructor arguments for that class.
    defaults : `OrbitDeterminationDefaults`
        Lever values; `OD_DEFAULTS` unless overridden.
    **overrides
        Any `NativeOrbitFitter` keyword to override a default with.
    """
    rejection_kwargs: dict[str, Any] = {
        "jacobian": defaults.jacobian,
        "validate_covariance": defaults.validate_covariance,
    }
    if defaults.outlier_rejection == "cmc2003":
        rejection_kwargs.update(
            chi2_reject=defaults.cmc2003_chi2_reject,
            chi2_recover=defaults.cmc2003_chi2_recover,
            chi2_frac=defaults.cmc2003_chi2_frac,
            max_iterations=defaults.cmc2003_max_iterations,
            max_rejected_fraction=defaults.cmc2003_max_rejected_fraction,
            apparition_gap_days=defaults.cmc2003_apparition_gap_days,
            psd_floor_frac=defaults.cmc2003_psd_floor_frac,
        )
    settings: dict[str, Any] = {
        "propagator_kwargs": propagator_kwargs,
        "min_obs": defaults.min_obs,
        "min_arc_length": defaults.min_arc_length,
        "contamination_percentage": defaults.contamination_percentage,
        "rchi2_threshold": defaults.rchi2_threshold,
        "iod_rchi2_threshold": defaults.iod_rchi2_threshold,
        "observation_selection_method": defaults.observation_selection_method,
        "loss": defaults.loss,
        "f_scale": defaults.f_scale,
        "outlier_rejection": defaults.outlier_rejection,
        "rejection_kwargs": rejection_kwargs,
    }
    settings.update(overrides)
    return NativeOrbitFitter(propagator_class, **settings)
