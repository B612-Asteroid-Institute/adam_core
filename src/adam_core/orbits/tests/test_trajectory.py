import numpy as np
import pytest

from adam_core import _rust_native

from ...coordinates import CartesianCoordinates
from ...coordinates.origin import Origin
from ...time import Timestamp
from ..orbits import Orbits
from ..trajectory import Trajectory


def _orbits(object_id: str, epochs: list[float]) -> Orbits:
    n = len(epochs)
    return Orbits.from_kwargs(
        orbit_id=[f"{object_id}-{i}" for i in range(n)],
        object_id=[object_id] * n,
        coordinates=CartesianCoordinates.from_kwargs(
            x=[1.0] * n,
            y=[0.1] * n,
            z=[0.0] * n,
            vx=[0.0] * n,
            vy=[0.017] * n,
            vz=[0.0] * n,
            time=Timestamp.from_mjd(epochs, scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN"] * n),
            frame="ecliptic",
        ),
    )


def _trajectory(
    object_id: str,
    epochs: list[float],
    starts: list[float],
    ends: list[float],
) -> Trajectory:
    n = len(epochs)
    return Trajectory.from_kwargs(
        object_id=[object_id] * n,
        segment_id=[f"{object_id}.seg{i}" for i in range(n)],
        coverage_start=Timestamp.from_mjd(starts, scale="tdb"),
        coverage_end=Timestamp.from_mjd(ends, scale="tdb"),
        orbit=_orbits(object_id, epochs),
        source=["spk:test"] * n,
        source_version=["v1"] * n,
        max_propagation_days=[5.0] * n,
        is_maneuver_boundary=[False] * n,
    )


def _contiguous() -> Trajectory:
    return _trajectory(
        "A",
        epochs=[59000.0, 59010.0, 59020.0],
        starts=[58995.0, 59005.0, 59015.0],
        ends=[59005.0, 59015.0, 59025.0],
    )


def test_trajectory_nests_orbits_and_round_trips(tmp_path):
    traj = _contiguous()
    assert len(traj) == 3
    assert isinstance(traj.orbit, Orbits)
    assert len(traj.orbit) == 3
    np.testing.assert_allclose(traj.epoch_mjd(), [59000.0, 59010.0, 59020.0])

    path = tmp_path / "traj.parquet"
    traj.to_parquet(str(path))
    rt = Trajectory.from_parquet(str(path))
    assert rt.segment_id.to_pylist() == ["A.seg0", "A.seg1", "A.seg2"]
    assert isinstance(rt.orbit, Orbits)


def test_validate_coverage_accepts_contiguous_non_overlapping():
    assert _contiguous().validate_coverage() is not None


def test_empty_trajectory_preserves_upstream_defaults():
    trajectory = Trajectory.empty()
    np.testing.assert_array_equal(trajectory.coverage_start_mjd(), [])
    np.testing.assert_array_equal(trajectory.coverage_end_mjd(), [])
    np.testing.assert_array_equal(trajectory.epoch_mjd(), [])
    assert trajectory.object_ids() == []
    assert trajectory.validate_coverage() is trajectory
    assert trajectory.segment_for(59000.0) is None


def test_validate_coverage_rejects_non_positive_window():
    traj = _trajectory("A", epochs=[59000.0], starts=[59000.0], ends=[59000.0])
    with pytest.raises(ValueError, match="strictly after"):
        traj.validate_coverage()


def test_validate_coverage_rejects_epoch_outside_window():
    traj = _trajectory("A", epochs=[59100.0], starts=[58995.0], ends=[59005.0])
    with pytest.raises(ValueError, match="within its coverage window"):
        traj.validate_coverage()


def test_validate_coverage_rejects_overlap():
    traj = _trajectory(
        "A",
        epochs=[59000.0, 59002.0],
        starts=[58995.0, 59000.0],
        ends=[59005.0, 59010.0],
    )
    with pytest.raises(ValueError, match="overlapping coverage"):
        traj.validate_coverage()


def test_segment_for_hits_and_half_open_boundary():
    traj = _contiguous().validate_coverage()
    assert traj.segment_for(59000.0).segment_id.to_pylist() == ["A.seg0"]
    # Half-open: the shared boundary belongs to the later segment.
    assert traj.segment_for(59005.0).segment_id.to_pylist() == ["A.seg1"]
    assert traj.segment_for(59020.0).segment_id.to_pylist() == ["A.seg2"]


def test_segment_for_gap_returns_none_fail_closed():
    traj = _contiguous().validate_coverage()
    assert traj.segment_for(58990.0) is None  # before first window
    assert traj.segment_for(59025.0) is None  # == last end, excluded (half-open)


def test_segment_for_multi_object_requires_object_id():
    a = _contiguous()
    b = _trajectory("B", epochs=[59000.0], starts=[58990.0], ends=[59030.0])
    from quivr import concatenate

    both = concatenate([a, b])
    assert both.object_ids() == ["A", "B"]
    with pytest.raises(ValueError, match="requires object_id"):
        both.segment_for(59000.0)
    assert both.segment_for(59000.0, object_id="A").segment_id.to_pylist() == ["A.seg0"]
    assert both.segment_for(59000.0, object_id="B").segment_id.to_pylist() == ["B.seg0"]


def test_trajectory_public_methods_have_rust_owned_timing():
    trajectory = _contiguous().validate_coverage()
    batch = trajectory._native_batch()
    for operation in [
        "coverage_start_mjd",
        "coverage_end_mjd",
        "epoch_mjd",
        "object_ids",
        "validate_coverage",
        "segment_for",
    ]:
        samples = np.asarray(
            _rust_native.benchmark_trajectory_arrow(
                batch,
                operation,
                2,
                2,
                1,
                59000.0 if operation == "segment_for" else None,
                "A" if operation == "segment_for" else None,
            )
        )
        assert samples.shape == (2, 2)
        assert np.all(samples > 0.0)
