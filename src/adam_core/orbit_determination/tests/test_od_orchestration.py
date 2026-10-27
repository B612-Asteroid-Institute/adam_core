"""Focused one-group orchestration: prevalidate, prepare once, compose exactly."""

import numpy as np
import pyarrow as pa
import pytest

from ..algorithms import workflows
from ..algorithms.iod import run_iod_one_group
from ..algorithms.workflows import refine_one_group, run_od_one_group
from ._focused import explicit_config
from .test_differential_correction import TwoBodyPropagator, make_synthetic_observations


def _clean_identity(monkeypatch) -> None:
    from ..algorithms import _provenance

    monkeypatch.setattr(
        _provenance._rust_native,
        "build_identity",
        lambda: ("0.7.0", "a" * 40),
    )


def test_duplicate_observation_ids_fail_before_pipeline(monkeypatch) -> None:
    _clean_identity(monkeypatch)
    observations = make_synthetic_observations().set_column(
        "id", pa.array(["duplicate"] * 25, type=pa.large_string())
    )
    monkeypatch.setattr(
        workflows,
        "apply_pipeline_once",
        lambda *_: (_ for _ in ()).throw(AssertionError("pipeline ran")),
    )
    with pytest.raises(ValueError, match="unique"):
        run_od_one_group(
            observations,
            TwoBodyPropagator(),
            config=explicit_config(),
        )


def test_run_od_prepares_once(monkeypatch) -> None:
    _clean_identity(monkeypatch)
    observations = make_synthetic_observations(n_obs=16)
    calls = 0
    original = workflows.apply_pipeline_once

    def record(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(workflows, "apply_pipeline_once", record)
    run_od_one_group(observations, TwoBodyPropagator(), config=explicit_config())
    assert calls == 1


def test_run_od_matches_explicit_focused_stage_composition(monkeypatch) -> None:
    _clean_identity(monkeypatch)
    observations = make_synthetic_observations(n_obs=16)
    config = explicit_config()
    fused, _ = run_od_one_group(observations, TwoBodyPropagator(), config=config)
    seed, _, report = run_iod_one_group(
        observations, TwoBodyPropagator(), config=config
    )
    assert report.found
    composed, _ = refine_one_group(
        seed, observations, TwoBodyPropagator(), config=config
    )
    np.testing.assert_array_equal(fused.coordinates.values, composed.coordinates.values)
    np.testing.assert_array_equal(
        fused.coordinates.covariance.to_matrix(),
        composed.coordinates.covariance.to_matrix(),
    )
