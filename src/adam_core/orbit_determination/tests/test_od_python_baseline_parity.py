"""Rust-owned configured OD determinism and accepted d21 constant pins."""

import numpy as np

from ..algorithms import _provenance
from ..algorithms.workflows import run_od_one_group
from ._focused import explicit_config
from .test_differential_correction import TwoBodyPropagator, make_synthetic_observations


def test_core_config_pins_accepted_d21_two_body_constants() -> None:
    two_body = explicit_config().for_iod().candidate_correction_fit.two_body
    assert two_body.lt_tol == 1.0e-12
    assert two_body.ephemeris_max_iter == 1000
    assert two_body.max_lt_iter == 10


def test_configured_rust_od_is_bitwise_deterministic(monkeypatch) -> None:
    monkeypatch.setattr(
        _provenance._rust_native,
        "build_identity",
        lambda: ("0.7.0", "a" * 40),
    )
    observations = make_synthetic_observations(n_obs=16, seed=91)
    config = explicit_config()
    first = run_od_one_group(observations, TwoBodyPropagator(), config=config)
    second = run_od_one_group(observations, TwoBodyPropagator(), config=config)
    np.testing.assert_array_equal(
        first[0].coordinates.values, second[0].coordinates.values
    )
    np.testing.assert_array_equal(
        first[0].coordinates.covariance.to_matrix(),
        second[0].coordinates.covariance.to_matrix(),
    )
    assert first[1].outlier.to_pylist() == second[1].outlier.to_pylist()
