from copy import deepcopy

import numpy as np
from astropy.time import Time

from ..indexable import Indexable, concatenate

SLICES = [
    slice(0, 1, 1),
    slice(1, 2, 1),
    slice(8, 9, 1),
    slice(7, 8, 1),
    slice(0, 10, 1),
    slice(0, 5, 1),
    slice(5, 10, 1),
    slice(50, 60, 2),
    slice(50, 60, 2),
    slice(99, 10, -2),
]

ATTRIBUTES = [
    "index_array_int",
    "index_array_str",
    "array",
    "array_2d",
    "array_3d",
    "masked_array",
    "times",
]
N = 100


class TestIndexable(Indexable):
    def __init__(self):
        self.index_array_int = np.concatenate(
            [np.arange(10, dtype=int) for i in range(10)]
        )
        self.index_array_str = np.concatenate(
            [np.arange(10, dtype=int).astype(str) for i in range(10)]
        )
        self.array = np.arange(0, N)
        self.array_2d = np.random.random((N, 6))
        self.array_3d = np.random.random((N, 6, 6))
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
                indexable_i.__dict__[attribute_i],
                indexable.__dict__[attribute_i][i : i + 1],
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
            indexable.__dict__[attribute_i][1 : N - 10],
        )

    # Delete the first 20 elements of the indexable and check that this operation
    # is equivalent to deleting the first 20 elements of the individual attributes
    del indexable_mod[:20]
    for attribute_i in ATTRIBUTES:
        assert_equal(
            indexable_mod.__dict__[attribute_i],
            # We've removed the first 1 element in the first test, and now
            # we remove the next 20
            indexable.__dict__[attribute_i][21 : N - 10],
        )

    # Delete the first and last element and check that this operation is equivalent to
    # deleting the first and last element of the individual attributes
    array_slice = np.array([0, -1])
    del indexable_mod[array_slice]
    for attribute_i in ATTRIBUTES:
        assert_equal(
            indexable_mod.__dict__[attribute_i],
            # We've removed the first 21 elements in test 1 and 3, and now
            # we remove the next one and the last element
            indexable.__dict__[attribute_i][22 : N - 11],
        )

    return


def test_Indexable_yield_chunks():
    # Create test indexable
    indexable = TestIndexable()

    # Iterate through the indexable and check that this operation
    # is equivalent to iterating through the individual attributes
    for i, indexable_i in enumerate(indexable.yield_chunks(10)):
        for attribute_i in ATTRIBUTES:
            assert_equal(
                indexable_i.__dict__[attribute_i],
                indexable.__dict__[attribute_i][i * 10 : (i + 1) * 10],
            )

    return


def test_Indexable_set_index_int_unsorted():
    # Create test indexable
    indexable = TestIndexable()
    indexable.set_index("index_array_int")

    # The index is unsorted with the numbers 0-9 repeated 10 times
    assert indexable._class_index_attribute == "index_array_int"
    np.testing.assert_equal(
        indexable._class_index, np.unique(indexable.index_array_int)
    )

    # We expect the class index to be the unique values of the index
    np.testing.assert_equal(
        indexable._class_index_to_members, indexable.index_array_int
    )
    assert indexable._class_index_to_members_is_slice == False

    # If we grab the first element of the class now we expect each member array to the every 10th element
    # in the original members
    indexable_i = indexable[0]
    for attribute_i in ATTRIBUTES:
        assert_equal(
            indexable_i.__dict__[attribute_i], indexable.__dict__[attribute_i][::10]
        )

    return


def test_Indexable_set_index_int_sorted():
    # Create test indexable
    indexable = TestIndexable()
    indexable.index_array_int.sort()
    indexable.set_index("index_array_int")

    # The index is sorted with the with the numbers 0-9 repeated 10 times in order
    assert indexable._class_index_attribute == "index_array_int"
    np.testing.assert_equal(
        indexable._class_index, np.unique(indexable.index_array_int)
    )

    # Because the class index is sorted we now expect the class index to members mapping to be
    # an array of slices
    np.testing.assert_equal(
        indexable._class_index_to_members,
        np.array([slice(10 * i, 10 * (i + 1)) for i in range(10)]),
    )
    assert indexable._class_index_to_members_is_slice == True

    # If we grab the first element of the class now we expect each member array to the every 10th element
    # in the original members
    indexable_i = indexable[0]
    for attribute_i in ATTRIBUTES:
        assert_equal(
            indexable_i.__dict__[attribute_i], indexable.__dict__[attribute_i][:10]
        )


def test_Indexable_set_index_str_unsorted():
    # Create test indexable
    indexable = TestIndexable()
    indexable.set_index("index_array_str")

    # The index is unsorted with the numbers 0-9 repeated 10 times
    assert indexable._class_index_attribute == "index_array_str"
    np.testing.assert_equal(indexable._class_index, np.arange(0, 10))

    # We expect the class index to be the unique values of the index
    np.testing.assert_equal(
        indexable._class_index_to_members, indexable.index_array_int
    )
    assert indexable._class_index_to_members_is_slice == False

    # If we grab the first element of the class now we expect each member array to the every 10th element
    # in the original members
    indexable_i = indexable[0]
    for attribute_i in ATTRIBUTES:
        assert_equal(
            indexable_i.__dict__[attribute_i], indexable.__dict__[attribute_i][::10]
        )

    return


def test_Indexable_set_index_str_sorted():
    # Create test indexable
    indexable = TestIndexable()
    indexable.index_array_str.sort()
    indexable.set_index("index_array_str")

    # The index is unsorted with the numbers 0-9 repeated 10 times
    assert indexable._class_index_attribute == "index_array_str"
    np.testing.assert_equal(indexable._class_index, np.arange(0, 10))

    # We expect the class index to be the unique values of the index
    np.testing.assert_equal(
        indexable._class_index_to_members,
        np.array([slice(10 * i, 10 * (i + 1)) for i in range(10)]),
    )
    assert indexable._class_index_to_members_is_slice == True

    # If we grab the first element of the class now we expect each member array to the every 10th element
    # in the original members
    indexable_i = indexable[0]
    for attribute_i in ATTRIBUTES:
        assert_equal(
            indexable_i.__dict__[attribute_i], indexable.__dict__[attribute_i][:10]
        )

    return


def test_Indexable_concatenate():
    indexable1 = TestIndexable()
    indexable2 = TestIndexable()
    indexable3 = TestIndexable()
    # Slightly modify the second and third indexable with the exception of the array of strings
    for attribute_i in ATTRIBUTES:
        if attribute_i != "index_array_str":
            indexable2.__dict__[attribute_i] = indexable2.__dict__[attribute_i] + 100
            indexable3.__dict__[attribute_i] = indexable3.__dict__[attribute_i] + 200

    indexable = concatenate([indexable1, indexable2, indexable3])
    for attribute_i in ATTRIBUTES:
        assert_equal(
            indexable.__dict__[attribute_i][:100],
            indexable1.__dict__[attribute_i][:100],
        )
        assert_equal(
            indexable.__dict__[attribute_i][100:200],
            indexable2.__dict__[attribute_i][:100],
        )
        assert_equal(
            indexable.__dict__[attribute_i][200:300],
            indexable3.__dict__[attribute_i][:100],
        )

    return
