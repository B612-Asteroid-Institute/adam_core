import numpy as np
import pytest

from ..origin import Origin
from ..transform import (
    cartesian_to_cometary,
    cartesian_to_keplerian,
    cometary_to_cartesian,
    keplerian_to_cartesian,
)


def _solar_mu(rows: int) -> np.ndarray:
    return Origin.from_kwargs(code=["SUN"] * rows).mu()


@pytest.mark.parametrize("mean_anomaly", [-10.0, 10.0])
def test_hyperbolic_keplerian_roundtrip_preserves_mean_anomaly_sign(
    mean_anomaly: float,
) -> None:
    keplerian = np.array(
        [[-1.272345007428081, 1.201133796102373, 20.0, 45.0, 30.0, mean_anomaly]]
    )
    mu = _solar_mu(1)
    cartesian = keplerian_to_cartesian(keplerian, mu)
    actual = cartesian_to_keplerian(cartesian, np.array([59_000.0]), mu)

    np.testing.assert_allclose(actual[0, [0, 4, 5, 6, 7]], keplerian[0, :5], rtol=1e-11)
    np.testing.assert_allclose(actual[0, 8], mean_anomaly, rtol=0.0, atol=1e-10)
    assert np.signbit(actual[0, 8]) == np.signbit(mean_anomaly)


def test_hyperbolic_cometary_roundtrip_preserves_both_perihelion_sides() -> None:
    perihelion = 59_000.0
    cometary = np.array(
        [
            [0.2559115812959117, 1.201133796102373, 20.0, 45.0, 30.0, perihelion],
            [0.2559115812959117, 1.201133796102373, 20.0, 45.0, 30.0, perihelion],
        ]
    )
    epochs = np.array([perihelion - 100.0, perihelion + 100.0])
    mu = _solar_mu(2)
    cartesian = cometary_to_cartesian(cometary, epochs, mu)
    actual = cartesian_to_cometary(cartesian, epochs, mu)

    assert np.all(np.isfinite(cartesian))
    np.testing.assert_allclose(actual, cometary, rtol=1e-10, atol=1e-9)
    keplerian = cartesian_to_keplerian(cartesian, epochs, mu)
    assert keplerian[0, 8] < 0.0
    assert keplerian[1, 8] > 0.0


def test_c2022_e3_preperihelion_state_is_physical() -> None:
    epoch = np.array([59_873.0])
    perihelion = 59_956.7851988296
    cometary = np.array(
        [
            [
                1.11224437022534,
                1.000301905819192,
                109.169480756749,
                302.5550197168474,
                145.81492879,
                perihelion,
            ]
        ]
    )
    mu = _solar_mu(1)
    cartesian = cometary_to_cartesian(cometary, epoch, mu)

    # Independent reference: JPL Horizons C/2022 E3, source JPL#82,
    # Sun center (DE441), JD 2459873.5 TDB, geometric vectors in the J2000
    # ecliptic frame and AU-D units (Horizons API, 2026-11-10):
    # https://ssd.jpl.nasa.gov/api/horizons.api
    horizons_state = np.array(
        [
            -1.860700831657873e-1,
            -7.098312947206076e-1,
            1.549882327926310,
            -7.712229663917785e-3,
            1.585906203592981e-2,
            -5.849336545320489e-3,
        ]
    )
    # The SBDB fixture rounds the angular elements. A 1e-10 au position
    # tolerance is about 15 m; 2e-12 au/day is about 3.5 micrometers/second.
    np.testing.assert_allclose(
        cartesian[0, :3], horizons_state[:3], rtol=0.0, atol=1e-10
    )
    np.testing.assert_allclose(
        cartesian[0, 3:], horizons_state[3:], rtol=0.0, atol=2e-12
    )
    assert np.dot(cartesian[0, :3], cartesian[0, 3:]) < 0.0

    actual_cometary = cartesian_to_cometary(cartesian, epoch, mu)
    np.testing.assert_allclose(actual_cometary[0, 0], cometary[0, 0], rtol=1e-10)
    np.testing.assert_allclose(actual_cometary[0, 1], cometary[0, 1], rtol=1e-12)
    np.testing.assert_allclose(actual_cometary[0, 5], perihelion, atol=1e-9)

    actual_keplerian = cartesian_to_keplerian(cartesian, epoch, mu)
    np.testing.assert_allclose(
        actual_keplerian[0, 8], -0.0003692991264304336, atol=1e-9
    )
    assert actual_keplerian[0, 8] < 0.0
