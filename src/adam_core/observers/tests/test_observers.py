import pyarrow as pa
import pyarrow.compute as pc
import pytest

from ...time import Timestamp
from ..observers import Observers


@pytest.fixture
def codes_times() -> tuple[pa.Array, Timestamp]:
    codes = pa.array(
        ["500", "X05", "I41", "X05", "I41", "W84", "500"],
    )

    times = Timestamp.from_kwargs(
        days=[59000, 59001, 59002, 59003, 59004, 59005, 59006],
        nanos=[0, 0, 0, 0, 0, 0, 0],
        scale="tdb",
    )
    return codes, times


def test_Observers_from_codes(codes_times) -> None:
    # Test that observers from code returns the correct number of observers
    # and in the order that they were requested
    codes, times = codes_times

    observers = Observers.from_codes(codes, times)
    assert len(observers) == 7
    assert pc.all(pc.equal(observers.code, codes)).as_py()
    assert pc.all(pc.equal(observers.coordinates.time.days, times.days)).as_py()
    assert pc.all(pc.equal(observers.coordinates.time.nanos, times.nanos)).as_py()


def test_Observers_from_codes_non_pyarrow(codes_times) -> None:
    # Test that observers from code returns the correct number of observers
    # and in the order that they were requested
    codes, times = codes_times

    observers = Observers.from_codes(codes.to_numpy(zero_copy_only=False), times)
    assert len(observers) == 7
    assert pc.all(pc.equal(observers.code, codes)).as_py()
    assert pc.all(pc.equal(observers.coordinates.time.days, times.days)).as_py()
    assert pc.all(pc.equal(observers.coordinates.time.nanos, times.nanos)).as_py()

    observers = Observers.from_codes(codes.to_pylist(), times)
    assert len(observers) == 7
    assert pc.all(pc.equal(observers.code, codes)).as_py()
    assert pc.all(pc.equal(observers.coordinates.time.days, times.days)).as_py()
    assert pc.all(pc.equal(observers.coordinates.time.nanos, times.nanos)).as_py()


def test_Observers_from_codes_raises(codes_times) -> None:
    # Test that observers from code raises an error if the codes and times
    # are not the same length
    codes, times = codes_times

    with pytest.raises(ValueError, match="codes and times must have the same length."):
        Observers.from_codes(codes[:3], times)
    with pytest.raises(ValueError, match="codes and times must have the same length."):
        Observers.from_codes(codes, times[:3])
