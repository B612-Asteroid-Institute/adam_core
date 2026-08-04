import pyarrow as pa
import pytest
import quivr as qv

from ...coordinates.origin import Origin
from ...coordinates.spherical import SphericalCoordinates
from ...observers.observers import Observers
from ...time import Timestamp
from ..ephemeris import Ephemeris


def test_Ephemeris_link_to_observers():
    # Test that we can link ephemerides to observers with different
    # precisions
    observer_01 = Observers.from_code(
        "X05",
        Timestamp.from_kwargs(
            days=[59000, 59000, 59000],
            nanos=[9, 999, 999_999],
            scale="utc",
        ),
    )
    observer_02 = Observers.from_code(
        "500",
        Timestamp.from_kwargs(
            days=[59000, 59000, 59000],
            nanos=[9, 999, 999_999],
            scale="utc",
        ),
    )
    observers = qv.concatenate([observer_01, observer_02])

    ephemeris = Ephemeris.from_kwargs(
        orbit_id=pa.array(["00000" for i in range(6)]),
        object_id=pa.array(["00000" for i in range(6)]),
        coordinates=SphericalCoordinates.from_kwargs(
            time=Timestamp.from_kwargs(
                days=[59000, 59000, 59000, 59000, 59000, 59000],
                nanos=[9, 999, 999_999, 9, 999, 999_999],
                scale="utc",
            ),
            lon=[0, 0, 0, 0, 0, 0],
            lat=[0, 0, 0, 0, 0, 0],
            frame="equatorial",
            origin=Origin.from_kwargs(
                code=pa.array(["X05", "X05", "X05", "500", "500", "500"])
            ),
        ),
    )

    linkage = ephemeris.link_to_observers(observers, precision="ns")
    assert len(linkage.all_unique_values) == 6
    e1, o1 = linkage.select((59000, 9, "X05"))
    assert len(e1) == len(o1) == 1
    e2, o2 = linkage.select((59000, 999, "X05"))
    assert len(e2) == len(o2) == 1
    e3, o3 = linkage.select((59000, 999_999, "X05"))
    assert len(e3) == len(o3) == 1
    e4, o4 = linkage.select((59000, 9, "500"))
    assert len(e4) == len(o4) == 1
    e5, o5 = linkage.select((59000, 999, "500"))
    assert len(e5) == len(o5) == 1
    e6, o6 = linkage.select((59000, 999_999, "500"))
    assert len(e6) == len(o6) == 1

    # Reduce precision to microseconds
    with pytest.warns(UserWarning):
        linkage = ephemeris.link_to_observers(observers, precision="us")

    # First two times should be grouped together
    # Last 2 times should be round-down to the previous microsecond
    assert len(linkage.all_unique_values) == 4
    e1, o1 = linkage.select((59000, 0, "X05"))
    assert len(e1) == len(o1) == 2
    e2, o2 = linkage.select((59000, 0, "500"))
    assert len(e2) == len(o2) == 2
    e3, o3 = linkage.select((59000, 999_000, "X05"))
    assert len(e3) == len(o3) == 1
    e4, o4 = linkage.select((59000, 999_000, "500"))
    assert len(e4) == len(o4) == 1

    # Reduce precision to milliseconds
    with pytest.warns(UserWarning):
        linkage = ephemeris.link_to_observers(observers, precision="ms")

    assert len(linkage.all_unique_values) == 2
    e1, o1 = linkage.select((59000, 0, "X05"))
    assert len(e1) == len(o1) == 3
    e2, o2 = linkage.select((59000, 0, "500"))
    assert len(e2) == len(o2) == 3
