import os
import pathlib

import astropy.time
import numpy as np
import quivr as qv

from ...coordinates.cartesian import CartesianCoordinates
from ...coordinates.origin import Origin
from ...coordinates.times import Times
from ..exposures import Exposures


def test_exposure_midpoints():
    start_times = astropy.time.Time(
        [
            "2000-01-01T00:00:00",
            "2000-01-02T00:00:00",
        ],
        scale="utc",
    )
    exp = Exposures.from_kwargs(
        id=["e1", "e2"],
        start_time=Times.from_astropy(start_times),
        duration=[60, 30],
        filter=["g", "r"],
        observatory_code=["I41", "I41"],
    )

    midpoints = exp.midpoint()
    midpoints_at = midpoints.to_astropy()
    assert midpoints_at[0] == astropy.time.Time("2000-01-01T00:00:30", scale="utc")
    assert midpoints_at[1] == astropy.time.Time("2000-01-02T00:00:15", scale="utc")


def test_exposure_states():
    class StateData(qv.Table):
        # Represents the data in ../../observers/tests/testdata/*.csv
        jd1_tdb = qv.Float64Column()
        jd2_tdb = qv.Float64Column()

        x = qv.Float64Column()
        y = qv.Float64Column()
        z = qv.Float64Column()
        vx = qv.Float64Column()
        vy = qv.Float64Column()
        vz = qv.Float64Column()

        origin = Origin.as_column()

        def to_astropy(self):
            return astropy.time.Time(
                val=self.jd1_tdb, val2=self.jd2_tdb, format="jd", scale="tdb"
            )

        def to_cartesian(self):
            return CartesianCoordinates.from_kwargs(
                x=self.x,
                y=self.y,
                z=self.z,
                vx=self.vx,
                vy=self.vy,
                vz=self.vz,
                origin=self.origin,
                frame="ecliptic",
            )

    observer_state_dir = (
        pathlib.Path(os.path.dirname(__file__))
        / ".."
        / ".."
        / "observers"
        / "tests"
        / "testdata"
    )

    w84_state_data = StateData.from_csv(observer_state_dir / "W84_sun.csv")
    i41_state_data = StateData.from_csv(observer_state_dir / "I41_sun.csv")

    # Mix up w84 and i41 in one big exposure table
    codes = ["W84", "I41", "W84", "I41", "W84"]
    state_times = astropy.time.Time(
        [
            w84_state_data.to_astropy()[0],
            i41_state_data.to_astropy()[0],
            w84_state_data.to_astropy()[3],
            i41_state_data.to_astropy()[1],
            w84_state_data.to_astropy()[2],
        ],
        scale="tdb",
    )
    expected = qv.concatenate(
        [
            w84_state_data.to_cartesian()[0],
            i41_state_data.to_cartesian()[0],
            w84_state_data.to_cartesian()[3],
            i41_state_data.to_cartesian()[1],
            w84_state_data.to_cartesian()[2],
        ]
    )
    exp = Exposures.from_kwargs(
        id=["e1", "e2", "e3", "e4", "e5"],
        start_time=Times.from_astropy(state_times),
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
        expected.z.to_numpy(), states.z.to_numpy(), maxulp=64, dtype=np.float32
    )
    np.testing.assert_array_max_ulp(
        expected.vx.to_numpy(), states.vx.to_numpy(), dtype=np.float32
    )
    np.testing.assert_array_max_ulp(
        expected.vy.to_numpy(), states.vy.to_numpy(), maxulp=2, dtype=np.float32
    )
    np.testing.assert_array_max_ulp(
        expected.vz.to_numpy(), states.vz.to_numpy(), maxulp=22, dtype=np.float32
    )

    np.testing.assert_allclose(expected.x.to_numpy(), states.x.to_numpy())
    np.testing.assert_allclose(expected.y.to_numpy(), states.y.to_numpy())
    np.testing.assert_allclose(expected.z.to_numpy(), states.z.to_numpy(), rtol=1e-5)
    np.testing.assert_allclose(expected.vx.to_numpy(), states.vx.to_numpy())
    np.testing.assert_allclose(expected.vy.to_numpy(), states.vy.to_numpy(), rtol=1e-6)
    np.testing.assert_allclose(expected.vz.to_numpy(), states.vz.to_numpy(), rtol=1e-5)


def test_empty_exposures():
    have = Exposures.empty()
    assert len(have) == 0
