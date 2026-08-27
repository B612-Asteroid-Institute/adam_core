"""Rust-owned Instant timing smoke coverage for the coordinate property
kernels. Semantic behavior is covered by the existing per-class test modules;
these assertions only guarantee each qualifying kernel family has a working
Rust-internal benchmark entrypoint."""

import numpy as np

from adam_core import _rust_native


def test_coordinate_ops_have_rust_owned_timing():
    rng = np.random.default_rng(3)
    values = np.ascontiguousarray(rng.normal(size=(16, 6)))
    covariances = np.ascontiguousarray(rng.normal(size=(16, 6, 6)))
    sigmas = np.ascontiguousarray(np.abs(rng.normal(size=(16, 6))))
    q = np.ascontiguousarray(np.abs(rng.normal(size=16)) + 0.1)
    e = np.ascontiguousarray(np.abs(rng.normal(size=16)) * 0.5)
    mu = np.full(16, 2.9591220828411956e-4)

    for samples in (
        _rust_native.benchmark_ric_matrices_numpy(values, 2, 2, 1),
        _rust_native.benchmark_cartesian_unit_conversions_numpy(
            values, covariances, 2, 2, 1
        ),
        _rust_native.benchmark_covariance_sigmas_numpy(covariances, 2, 2, 1),
        _rust_native.benchmark_sigmas_to_covariances_numpy(sigmas, 2, 2, 1),
        _rust_native.benchmark_derived_elements_numpy(q, e, mu, 2, 2, 1),
        _rust_native.benchmark_origin_mu_numpy(["SUN", "EARTH"] * 8, 2, 2, 1),
    ):
        assert len(samples) == 2
        assert all(len(trial) == 2 for trial in samples)
        assert all(value > 0.0 for trial in samples for value in trial)
