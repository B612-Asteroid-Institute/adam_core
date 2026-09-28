"""
Tests for `default_observation_models`, the shipped observation-model stack:
its order and settings, the data-package resolution of its tables, and the
error naming the package when it is missing. The end-to-end `run_od` with
the defaults lives next to its two-body fixture in ``test_od_orchestration.py``.
"""

from __future__ import annotations

import sys

import numpy as np
import pyarrow as pa
import pytest

from ...observations import efcc18
from ..defaults import default_observation_models
from ..observation_uncertainty import (
    BIAS_TABLE_PACKAGE,
    EFCC18DebiasModel,
    EmpiricalCovarianceModel,
    NightBatchDeweightingModel,
    load_bias_table,
)
from ..veres2017 import (
    SIGMA_TABLE_PACKAGE,
    VERES2017_FALLBACK_SIGMA_ARCSEC,
    SigmaFillModel,
    load_sigma_table,
)
from .test_observation_uncertainty import make_bias_table

EFCC18_SHAPE = (efcc18.EFCC18_N_TILES, efcc18.EFCC18_N_CATALOGS, 4)
DEFAULT_MODEL_ORDER = (
    SigmaFillModel,
    EFCC18DebiasModel,
    EmpiricalCovarianceModel,
    NightBatchDeweightingModel,
)


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
        assert tuple(type(m) for m in models) == DEFAULT_MODEL_ORDER
        fill, debias, empirical, night = models
        assert isinstance(fill, SigmaFillModel)
        assert fill.lookup.fallback_sigma_arcsec == VERES2017_FALLBACK_SIGMA_ARCSEC
        assert isinstance(debias, EFCC18DebiasModel)
        assert debias.exclude_astcats == ()
        assert isinstance(empirical, EmpiricalCovarianceModel)
        assert empirical.mode == "add"
        assert empirical.min_resid_cov_n == 30
        assert isinstance(night, NightBatchDeweightingModel)
        assert night.cap == 4

    def test_missing_package_is_named(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, SIGMA_TABLE_PACKAGE, None)
        monkeypatch.setitem(sys.modules, BIAS_TABLE_PACKAGE, None)
        with pytest.raises(ImportError, match="adam-observatory-uncertainties"):
            default_observation_models(efcc18_bias_table=_zero_efcc18_table())
        with pytest.raises(ImportError, match="adam-observatory-uncertainties"):
            load_bias_table()

    def test_tables_resolve_from_the_package(self) -> None:
        package = pytest.importorskip(BIAS_TABLE_PACKAGE)
        models = default_observation_models(efcc18_bias_table=_zero_efcc18_table())
        sigma_rows = package.load_sigma_table("v2_sigma_fill").num_rows
        bias_rows = package.load_bias_table("v2_full").table.num_rows
        fill, _, empirical, _ = models
        assert isinstance(fill, SigmaFillModel)
        assert isinstance(empirical, EmpiricalCovarianceModel)
        assert fill.lookup.sigma_table.num_rows == sigma_rows > 0
        assert load_sigma_table().num_rows == sigma_rows
        assert empirical.bias_table.num_rows == bias_rows > 0
        assert load_bias_table().num_rows == bias_rows
