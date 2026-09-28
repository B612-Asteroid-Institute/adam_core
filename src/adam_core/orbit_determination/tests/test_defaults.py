"""
Tests for the shipped orbit-determination defaults (`OD_DEFAULTS`,
`default_observation_models`, `default_orbit_fitter`): the stack order and
lever values of the 2026-09-23 decision and the data-package resolution. The
end-to-end `run_od` with the default stack lives next to its two-body fixture
in ``test_od_orchestration.py``.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pyarrow as pa
import pytest

from ...observations import efcc18
from ..defaults import (
    OD_DEFAULTS,
    OrbitDeterminationDefaults,
    default_observation_models,
    default_orbit_fitter,
)
from ..observation_uncertainty import (
    BIAS_TABLE_PACKAGE,
    EFCC18DebiasModel,
    EmpiricalCovarianceModel,
    NightBatchDeweightingModel,
    load_bias_table,
)
from ..veres2017 import SIGMA_TABLE_PACKAGE, SigmaFillModel, load_sigma_table
from .test_differential_correction import TwoBodyPropagator
from .test_observation_uncertainty import make_bias_table

EFCC18_SHAPE = (efcc18.EFCC18_N_TILES, efcc18.EFCC18_N_CATALOGS, 4)


def _zero_efcc18_table() -> np.ndarray:
    return np.zeros(EFCC18_SHAPE, dtype=np.float32)


def _small_sigma_table() -> pa.Table:
    return pa.table(
        {
            "obs_code": pa.array([None], pa.large_string()),
            "astcat": pa.array([None], pa.large_string()),
            "sigma_ra_arcsec": pa.array([0.3], pa.float64()),
            "sigma_dec_arcsec": pa.array([0.3], pa.float64()),
        }
    )


class TestDefaultValues:
    def test_decision_2026_09_23(self) -> None:
        d = OD_DEFAULTS
        assert d.model_order() == (
            "SigmaFillModel",
            "EFCC18DebiasModel",
            "EmpiricalCovarianceModel",
            "NightBatchDeweightingModel",
        )
        assert d.sigma_fill_table == "v2_sigma_fill"
        assert d.bias_table == "v2_full"
        assert d.empirical_covariance_mode == "add"
        assert d.min_resid_cov_n == 30
        assert d.night_batch_cap == 4
        assert d.efcc18_exclude_astcats == ()
        assert d.outlier_rejection == "cmc2003"
        assert d.loss == "linear"
        assert d.jacobian == "analytic"
        assert d.validate_covariance is True
        assert (d.cmc2003_chi2_reject, d.cmc2003_chi2_recover) == (8.0, 7.0)

    def test_defaults_are_frozen_and_overridable_by_copy(self) -> None:
        with pytest.raises(AttributeError):
            OD_DEFAULTS.night_batch_cap = 1  # type: ignore[misc]
        custom = OrbitDeterminationDefaults(
            night_batch_cap=1, outlier_rejection="worst_residual"
        )
        assert custom.night_batch_cap == 1 and OD_DEFAULTS.night_batch_cap == 4


class TestDefaultObservationModels:
    def test_stack_with_explicit_tables_needs_no_package(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, SIGMA_TABLE_PACKAGE, None)
        models = default_observation_models(
            bias_table=make_bias_table([{"obs_code": "500"}]),
            sigma_table=_small_sigma_table(),
            efcc18_bias_table=_zero_efcc18_table(),
        )
        assert [type(m).__name__ for m in models] == list(OD_DEFAULTS.model_order())
        fill, debias, empirical, night = models
        assert isinstance(fill, SigmaFillModel)
        assert (
            fill.lookup.fallback_sigma_arcsec == OD_DEFAULTS.sigma_fill_fallback_arcsec
        )
        assert isinstance(debias, EFCC18DebiasModel) and debias.exclude_astcats == ()
        assert isinstance(empirical, EmpiricalCovarianceModel)
        assert empirical.mode == "add" and empirical.min_resid_cov_n == 30
        assert isinstance(night, NightBatchDeweightingModel) and night.cap == 4

    def test_missing_package_is_named(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, SIGMA_TABLE_PACKAGE, None)
        monkeypatch.setitem(sys.modules, BIAS_TABLE_PACKAGE, None)
        with pytest.raises(ImportError, match="adam-observatory-uncertainties"):
            default_observation_models(efcc18_bias_table=_zero_efcc18_table())
        with pytest.raises(ImportError, match="adam-observatory-uncertainties"):
            load_bias_table()
        with pytest.raises(ImportError):
            importlib.import_module(BIAS_TABLE_PACKAGE)

    def test_tables_resolve_from_the_package(self) -> None:
        package = pytest.importorskip(BIAS_TABLE_PACKAGE)
        models = default_observation_models(efcc18_bias_table=_zero_efcc18_table())
        sigma_rows = package.load_sigma_table("v2_sigma_fill").num_rows
        bias_rows = package.load_bias_table("v2_full").table.num_rows
        assert models[0].lookup.sigma_table.num_rows == sigma_rows > 0
        assert load_sigma_table("v2_sigma_fill").num_rows == sigma_rows
        assert models[2].bias_table.num_rows == bias_rows > 0
        assert load_bias_table("v2_full").num_rows == bias_rows


class TestDefaultOrbitFitter:
    def test_fitter_settings(self) -> None:
        fitter = default_orbit_fitter(TwoBodyPropagator)
        assert fitter.propagator_class is TwoBodyPropagator
        assert fitter.outlier_rejection == "cmc2003"
        assert fitter.loss == "linear"
        assert fitter.min_obs == 6 and fitter.iod_rchi2_threshold == 200.0
        assert fitter.rejection_kwargs["jacobian"] == "analytic"
        assert fitter.rejection_kwargs["validate_covariance"] is True
        assert fitter.rejection_kwargs["chi2_reject"] == 8.0
        assert fitter.rejection_kwargs["apparition_gap_days"] == 180.0

    def test_overrides_win(self) -> None:
        fitter = default_orbit_fitter(TwoBodyPropagator, loss="huber", min_obs=5)
        assert fitter.loss == "huber" and fitter.min_obs == 5
        worst = default_orbit_fitter(
            TwoBodyPropagator,
            defaults=OrbitDeterminationDefaults(outlier_rejection="worst_residual"),
        )
        assert worst.outlier_rejection == "worst_residual"
        assert "chi2_reject" not in worst.rejection_kwargs
