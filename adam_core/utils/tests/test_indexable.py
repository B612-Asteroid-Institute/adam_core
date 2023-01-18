from copy import deepcopy

import numpy as np
from astropy.time import Time

from ..indexable import Indexable

SLICES = [
    slice(0, 1, 1),  # 0
    slice(1, 2, 1),  # 1
    slice(8, 9, 1),  # -1
    slice(7, 8, 1),  # -2
    slice(0, 10, 1),
    slice(0, 5, 1),
    slice(5, 10, 1),
]

ATTRIBUTES = ["array", "masked_array", "times"]
N = 100


class TestIndexable(Indexable):
    def __init__(self, N: int = 100):
        self.array = np.arange(0, N)
        self.masked_array = np.ma.arange(0, N)
        self.masked_array.mask = np.zeros(N)
        self.masked_array.mask[0:N:2] = 1
        self.times = Time(np.arange(59000, 59000 + N), scale="utc", format="mjd")

        Indexable.__init__(self)


def assert_equal(a, b):
    if isinstance(a, (np.ndarray)) and isinstance(b, (np.ndarray)):
        np.testing.assert_equal(a, b)
    elif isinstance(a, (np.ma.masked_array)) and isinstance(b, (np.ma.masked_array)):
        np.testing.assert_equal(a.data, b.data)
        np.testing.assert_equal(a.mask, b.mask)
    elif isinstance(a, (Time)) and isinstance(b, (Time)):
        np.testing.assert_equal(a.mjd, b.mjd)
        assert a.scale == b.scale
        assert a.format == b.format
    else:
        assert a == b

    return


def test_Indexable_slicing():
    # Create test indexable
    indexable = TestIndexable()

    # For each slice, slice the test indexable and check that this operation
    # is equivalent to slicing the individual attributes
    for s in SLICES:
        for attribute_i in ATTRIBUTES:
            indexable_i = indexable[s]
            assert_equal(
                indexable_i.__dict__[attribute_i], indexable.__dict__[attribute_i][s]
            )

    return


def test_Indexable_iteration():
    # Create test indexable
    indexable = TestIndexable()

    # Iterate through the indexable and check that this operation
    # is equivalent to iterating through the individual attributes
    for i, indexable_i in enumerate(indexable):
        for attribute_i in ATTRIBUTES:
            assert_equal(
                indexable_i.__dict__[attribute_i], indexable.__dict__[attribute_i][i]
            )


def test_Indexable_deletion():
    # Create test indexable
    indexable = TestIndexable()

    # Delete the first element of the indexable and check that this operation
    # is equivalent to deleting the first element of the individual attributes
    indexable_mod = deepcopy(indexable)
    del indexable_mod[0]
    for attribute_i in ATTRIBUTES:
        assert_equal(
            indexable_mod.__dict__[attribute_i], indexable.__dict__[attribute_i][1:]
        )

    # Delete the last 10 elements of the indexable and check that this operation
    # is equivalent to deleting the last 10 elements of the individual attributes
    del indexable_mod[-10:]
    for attribute_i in ATTRIBUTES:
        assert_equal(
            indexable_mod.__dict__[attribute_i],
            indexable.__dict__[attribute_i][1 : N - 9],
        )

    return
