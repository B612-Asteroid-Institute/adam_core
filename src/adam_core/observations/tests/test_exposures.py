import os
import pathlib

import numpy as np
import quivr as qv

from ...coordinates.cartesian import CartesianCoordinates
from ...time import Timestamp
from ..exposures import Exposures


def test_exposure_midpoints():
    start_times = Timestamp.from_iso8601(
        [
            "2000-01-01T00:00:00",
            "2000-01-02T00:00:00",
        ],
    )
    exp = Exposures.from_kwargs(
        id=["e1", "e2"],
        start_time=start_times,
        duration=[60, 30],
        filter=["g", "r"],
        observatory_code=["I41", "I41"],
    )

    midpoints = exp.midpoint()
    assert midpoints == Timestamp.from_iso8601(
        ["2000-01-01T00:00:30", "2000-01-02T00:00:15"]
    )


def test_exposure_states():
    observer_state_dir = (
        pathlib.Path(os.path.dirname(__file__))
        / ".."
        / ".."
        / "observers"
        / "tests"
        / "testdata"
    )

    w84_state_data = CartesianCoordinates.from_parquet(
        observer_state_dir / "W84_sun.parquet"
    )
    i41_state_data = CartesianCoordinates.from_parquet(
        observer_state_dir / "I41_sun.parquet"
    )

    # Mix up w84 and i41 in one big exposure table
    codes = ["W84", "I41", "W84", "I41", "W84"]
    state_times = qv.concatenate(
        [
            w84_state_data.time[0],
            i41_state_data.time[0],
            w84_state_data.time[3],
            i41_state_data.time[1],
            w84_state_data.time[2],
        ]
    )
    expected = qv.concatenate(
        [
            w84_state_data[0],
            i41_state_data[0],
            w84_state_data[3],
            i41_state_data[1],
            w84_state_data[2],
        ]
    )
    exp = Exposures.from_kwargs(
        id=["e1", "e2", "e3", "e4", "e5"],
        start_time=state_times,
        duration=[0, 0, 0, 0, 0],
        filter=["g", "r", "g", "r", "g"],
        observatory_code=codes,
    )

    obs = exp.observers()
    codes = obs.code

    assert codes.to_pylist() == ["W84", "I41", "W84", "I41", "W84"]
    states = obs.coordinates

    # Assert that states are same, to within expected number of bits
    # of precision.
    #
    # ideally, we'd like everything to be exactly equal to double
    # precision. That would require extremely careful numerical
    # analysis, though. The current values of maxulp are just
    # empirical.
    np.testing.assert_array_max_ulp(
        expected.x.to_numpy(), states.x.to_numpy(), dtype=np.float32
    )
    np.testing.assert_array_max_ulp(
        expected.y.to_numpy(), states.y.to_numpy(), dtype=np.float32
    )
    np.testing.assert_array_max_ulp(
        expected.z.to_numpy(), states.z.to_numpy(), maxulp=156, dtype=np.float32
    )
    np.testing.assert_array_max_ulp(
        expected.vx.to_numpy(), states.vx.to_numpy(), dtype=np.float32
    )
    np.testing.assert_array_max_ulp(
        expected.vy.to_numpy(), states.vy.to_numpy(), maxulp=2, dtype=np.float32
    )
    np.testing.assert_array_max_ulp(
        expected.vz.to_numpy(), states.vz.to_numpy(), maxulp=31, dtype=np.float32
    )

    np.testing.assert_allclose(expected.x.to_numpy(), states.x.to_numpy())
    np.testing.assert_allclose(expected.y.to_numpy(), states.y.to_numpy())
    np.testing.assert_allclose(expected.z.to_numpy(), states.z.to_numpy(), rtol=1e-4)
    np.testing.assert_allclose(expected.vx.to_numpy(), states.vx.to_numpy())
    np.testing.assert_allclose(expected.vy.to_numpy(), states.vy.to_numpy(), rtol=1e-6)
    np.testing.assert_allclose(expected.vz.to_numpy(), states.vz.to_numpy(), rtol=1e-5)


def test_empty_exposures():
    have = Exposures.empty()
    assert len(have) == 0
