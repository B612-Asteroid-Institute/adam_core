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
    groups = list(exp.group_by_observatory_code())
    assert len(groups) == 1
    assert groups[0][0] == "I41"
    assert groups[0][1].id.to_pylist() == ["e1", "e2"]

    from adam_core import _rust_native
    from adam_core.observations.arrow_bridge import observations_to_ipc

    raw = observations_to_ipc(exp)
    group_samples = _rust_native.benchmark_exposure_groups_ipc(raw, 2, 2, 1)
    midpoint_samples = _rust_native.benchmark_exposure_midpoint_ipc(raw, 2, 2, 1)
    assert all(sample > 0.0 for trial in group_samples for sample in trial)
    assert all(sample > 0.0 for trial in midpoint_samples for sample in trial)


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

    import pyarrow as pa

    from adam_core._rust.arrow import ensure_spice_backend

    time_table = exp.start_time.table.combine_chunks()
    exposure_table = exp.table.combine_chunks()
    batch = pa.RecordBatch.from_arrays(
        [
            exposure_table.column("observatory_code").chunk(0),
            time_table.column("days").chunk(0),
            time_table.column("nanos").chunk(0),
            exposure_table.column("duration").chunk(0),
        ],
        names=["code", "days", "nanos", "duration"],
    )
    samples = (
        ensure_spice_backend().benchmark_observer_states_from_exposures_arrow_rust(
            batch, exp.start_time.scale, "ecliptic", "SUN", 2, 2, 1
        )
    )
    assert all(sample > 0.0 for trial in samples for sample in trial)

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
