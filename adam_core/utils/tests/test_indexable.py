from copy import deepcopy

import numpy as np
import pytest
from astropy.time import Time

from ..indexable import (
    Indexable,
    _check_slices_are_consecutive,
    _convert_grouped_array_to_slices,
    _map_values_to_integers,
    concatenate,
)

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


def test__convert_grouped_array_to_slices_raises():
    # Test that the function raises a ValueError if the array is not grouped
    # (sorted but sort order does not matter)
    array = np.array([0, 1, 0, 1])

    with pytest.raises(ValueError):
        _convert_grouped_array_to_slices(array)


def test__convert_grouped_array_to_slices():
    # Test that the function correctly converts a sorted array to slices
    array = np.arange(0, 10)
    slices = _convert_grouped_array_to_slices(array)
    desired = np.array(
        [
            slice(0, 1, 1),
            slice(1, 2, 1),
            slice(2, 3, 1),
            slice(3, 4, 1),
            slice(4, 5, 1),
            slice(5, 6, 1),
            slice(6, 7, 1),
            slice(7, 8, 1),
            slice(8, 9, 1),
            slice(9, 10, 1),
        ]
    )
    np.testing.assert_equal(slices, desired)

    # The order of the sorted array should not matter as long as all elements that
    # are supposed to be in the same slice are grouped together
    array = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    slices = _convert_grouped_array_to_slices(array)
    np.testing.assert_equal(slices, desired)

    # Test that the function correctly converts a sorted array to slices
    # this time with duplicates in the array
    array = np.array([0, 0, 1, 1])
    slices = _convert_grouped_array_to_slices(array)
    desired = np.array(
        [
            slice(0, 2, 1),
            slice(2, 4, 1),
        ]
    )
    np.testing.assert_equal(slices, desired)

    # Test that the function correctly converts a sorted array to slices
    # this time with only some duplicates in the array
    array = np.array([0, 0, 0, 1, 1, 2, 3])
    array.sort()
    slices = _convert_grouped_array_to_slices(array)
    desired = np.array(
        [
            slice(0, 3, 1),
            slice(3, 5, 1),
            slice(5, 6, 1),
            slice(6, 7, 1),
        ]
    )
    np.testing.assert_equal(slices, desired)


def test__map_values_to_integers():
    # Test that the function correctly maps integer values to integers
    values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    mapping = _map_values_to_integers(values)
    desired = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    np.testing.assert_equal(mapping, desired)

    # Test that the function correctly maps string values to integers
    values = np.array(["a", "a", "b", "c", "d", "e", "e", "e"])
    mapping = _map_values_to_integers(values)
    desired = np.array([0, 0, 1, 2, 3, 4, 4, 4])
    np.testing.assert_equal(mapping, desired)

    # Test that the function correctly maps float values with duplicates to integers
    values = np.array([0, 1, 0, 1], dtype=float)
    mapping = _map_values_to_integers(values)
    desired = np.array([0, 1, 0, 1])
    np.testing.assert_equal(mapping, desired)

    # Test that the function correctly maps unsorted strings with duplicates to integers
    values = np.array(["a", "b", "a"])
    mapping = _map_values_to_integers(values)
    desired = np.array([0, 1, 0])
    np.testing.assert_equal(mapping, desired)


def test__check_slices_are_consecutive():
    # Slices are consecutive and share the same step
    slices = np.array([slice(0, 1, 1), slice(1, 2, 1), slice(2, 10, 1)])
    assert _check_slices_are_consecutive(slices) is True

    # Step is not consistent
    slices = np.array([slice(0, 1, 1), slice(1, 2, 2), slice(2, 10, 1)])
    assert _check_slices_are_consecutive(slices) is False

    # Step is consistent but slices are not consecutive
    slices = np.array([slice(0, 1, 1), slice(1, 2, 1), slice(0, 1, 1)])
    assert _check_slices_are_consecutive(slices) is False


def test__check_member_validity_raises():
    # Create invalid test indexable
    class InvalidTestIndexable(Indexable):
        def __init__(self):
            self.index_array_1 = np.arange(11, dtype=int)
            self.index_array_2 = np.arange(10, dtype=int)

            Indexable.__init__(self)

    # Check that the invalid test indexable raises a ValueError
    with pytest.raises(ValueError):
        InvalidTestIndexable()


def test__check_member_validity():
    # Create invalid test indexable
    indexable = TestIndexable()
    member_length, member_index = indexable._check_member_validity()

    # Assert that the function correctly computes the length of the member
    # attributes and sets the according member length and index
    assert member_length == N
    np.testing.assert_equal(member_index, np.arange(0, 100))


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

    # The class index to members mapping should be an array of integers
    # since the index is cannot be represented as slices
    np.testing.assert_equal(
        indexable._class_index_to_integers, indexable.index_array_int
    )
    assert indexable._class_index_to_slices is None

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

    # The class index to slices mapping should be an array of slices
    # since the index is can be represented as slices
    np.testing.assert_equal(
        indexable._class_index_to_slices,
        np.array([slice(10 * i, 10 * (i + 1), 1) for i in range(10)]),
    )
    assert indexable._class_index_to_integers is None

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

    # Test that the length of the class is the same as the length of the index
    assert len(indexable) == 100

    # Set the index
    indexable.set_index("index_array_str")

    # The index is unsorted with the numbers 0-9 repeated 10 times
    assert indexable._class_index_attribute == "index_array_str"
    np.testing.assert_equal(indexable._class_index, np.arange(0, 10))

    # The class index to members mapping should be an array of integers
    # since the index is cannot be represented as slices
    np.testing.assert_equal(
        indexable._class_index_to_integers, indexable.index_array_int
    )
    assert indexable._class_index_to_slices is None

    # If we grab the first element of the class now we expect each member array to the every 10th element
    # in the original members
    indexable_i = indexable[0]
    for attribute_i in ATTRIBUTES:
        assert_equal(
            indexable_i.__dict__[attribute_i], indexable.__dict__[attribute_i][::10]
        )

    # Test that the length of the class is the same as the length of the index
    assert len(indexable) == 10
    return


def test_Indexable_set_index_str_sorted():
    # Create test indexable
    indexable = TestIndexable()

    # Test that the length of the class is the same as the length of the index
    assert len(indexable) == 100

    # Set the index
    indexable.index_array_str.sort()
    indexable.set_index("index_array_str")

    # The index is unsorted with the numbers 0-9 repeated 10 times
    assert indexable._class_index_attribute == "index_array_str"
    np.testing.assert_equal(indexable._class_index, np.arange(0, 10))

    # The class index to slices mapping should be an array of slices
    # since the index is can be represented as slices
    np.testing.assert_equal(
        indexable._class_index_to_slices,
        np.array([slice(10 * i, 10 * (i + 1), 1) for i in range(10)]),
    )
    assert indexable._class_index_to_integers is None

    # If we grab the first element of the class now we expect each member array to the every 10th element
    # in the original members
    indexable_i = indexable[0]
    for attribute_i in ATTRIBUTES:
        assert_equal(
            indexable_i.__dict__[attribute_i], indexable.__dict__[attribute_i][:10]
        )

    # Test that the length of the class is the same as the length of the index
    assert len(indexable) == 10
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


def test_Indexable_sort_values():
    # Create test indexable
    indexable = TestIndexable()

    # Sort the indexable by the index array
    indexable_sorted = indexable.sort_values("index_array_int", inplace=False)

    # Test that the attributes are correctly sorted
    for attribute_i in ATTRIBUTES:
        assert_equal(
            indexable_sorted.__dict__[attribute_i][:10],
            indexable.__dict__[attribute_i][::10],
        )

    return


def test_Indexable_sort_values_inplace():
    # Create test indexable
    indexable = TestIndexable()
    indexable_unsorted = deepcopy(indexable)

    # Sort the indexable by the index array
    indexable.sort_values("index_array_int", inplace=True)

    # Test that the attributes are correctly sorted
    for attribute_i in ATTRIBUTES:
        assert_equal(
            indexable.__dict__[attribute_i][:10],
            indexable_unsorted.__dict__[attribute_i][::10],
        )

    return
