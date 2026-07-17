"""Rust-owned Instant timing coverage for the public scalar dynamics helpers.

Each public scalar helper is a veneer over a Rust kernel; parity for the
kernels themselves is pinned by the dynamics fixture suites. This module
asserts every helper also has a live Rust-owned timing lane through the
packed-column scalar benchmark dispatcher.
"""

import numpy as np
import pytest

from adam_core import _rust_native

RNG = np.random.default_rng(19)
# Large enough that each rep exceeds the platform timer granularity.
N = 4096


def _packed(kernel: str) -> np.ndarray:
    a = RNG.uniform(0.5, 5.0, size=N)
    e = RNG.uniform(0.0, 0.9, size=N)
    mu = RNG.uniform(0.01, 0.05, size=N)
    angles = RNG.uniform(0.0, 2 * np.pi, size=N)
    if kernel in ("calc_period", "calc_mean_motion"):
        return np.column_stack([a, mu])
    if kernel in (
        "calc_periapsis_distance",
        "calc_apoapsis_distance",
        "calc_semi_latus_rectum",
        "calc_semi_major_axis",
    ):
        return np.column_stack([a, e])
    if kernel == "calc_mean_anomaly":
        return np.column_stack([angles, e])
    if kernel == "solve_kepler":
        return np.column_stack([e, angles])
    if kernel == "solve_barker":
        return angles.reshape(-1, 1)
    if kernel == "calc_stumpff":
        return RNG.uniform(-5.0, 5.0, size=N).reshape(-1, 1)
    if kernel in ("calc_chi", "calc_lagrange_coefficients"):
        r = RNG.uniform(0.5, 2.5, size=(N, 3))
        v = RNG.uniform(-0.02, 0.02, size=(N, 3))
        dt = RNG.uniform(1.0, 30.0, size=N)
        return np.column_stack([r, v, dt, mu])
    if kernel == "apply_lagrange_coefficients":
        r = RNG.uniform(0.5, 2.5, size=(N, 3))
        v = RNG.uniform(-0.02, 0.02, size=(N, 3))
        coeffs = RNG.uniform(0.9, 1.1, size=(N, 4))
        return np.column_stack([r, v, coeffs])
    if kernel == "add_stellar_aberration":
        orbits = RNG.uniform(-2.0, 2.0, size=(N, 6))
        observers = RNG.uniform(-1.0, 1.0, size=(N, 6))
        return np.column_stack([orbits, observers])
    raise AssertionError(f"unhandled kernel {kernel}")


SCALAR_KERNELS = [
    "calc_period",
    "calc_mean_motion",
    "calc_periapsis_distance",
    "calc_apoapsis_distance",
    "calc_semi_latus_rectum",
    "calc_semi_major_axis",
    "calc_mean_anomaly",
    "solve_kepler",
    "solve_barker",
    "calc_stumpff",
    # calc_chi_diagnostics shares the calc_chi kernel lane.
    "calc_chi",
    "calc_lagrange_coefficients",
    "apply_lagrange_coefficients",
    "add_stellar_aberration",
]


@pytest.mark.parametrize("kernel", SCALAR_KERNELS)
def test_scalar_dynamics_kernel_native_timing_live(kernel):
    packed = np.ascontiguousarray(_packed(kernel), dtype=np.float64)
    trials = _rust_native.benchmark_scalar_dynamics_kernel_numpy(
        kernel, packed, 3, 2, 1
    )
    assert len(trials) == 2
    assert all(len(samples) == 3 for samples in trials)
    assert all(sample >= 0.0 for samples in trials for sample in samples)
    assert any(sample > 0.0 for samples in trials for sample in samples)


def test_scalar_dynamics_kernel_rejects_unknown_kernel():
    packed = np.zeros((4, 2))
    with pytest.raises(ValueError, match="unknown scalar dynamics kernel"):
        _rust_native.benchmark_scalar_dynamics_kernel_numpy("nope", packed, 1, 1, 1)


def test_scalar_dynamics_kernel_rejects_wrong_columns():
    packed = np.zeros((4, 3))
    with pytest.raises(ValueError, match="expects 2 packed columns"):
        _rust_native.benchmark_scalar_dynamics_kernel_numpy(
            "calc_period", packed, 1, 1, 1
        )
