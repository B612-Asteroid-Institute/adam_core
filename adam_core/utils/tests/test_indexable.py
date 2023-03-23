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

N = 100
SLICES = [
    slice(0, 1, 1),
    slice(1, 2, 1),
    slice(7, 8, 1),
    slice(8, 9, 1),
    slice(0, N, 1),
    slice(0, 5, 1),
    slice(5, 10, 1),
    slice(50, 60, 2),
    slice(50, N, 2),
    slice(N - 1, 10, -2),
    slice(-5, -1, 1),
    slice(-10, -5, 2),
    slice(-N + 1, -1, 1),
    slice(-N + 1, -10, 2),
    slice(-N + 1, -N + 2, 1),
    # Out of bounds slices
    slice(N, N + 1, 1),
    slice(2 * N, 3 * N, 1),
    slice(-2 * N, -N, 1),
]

SLICEABLE_ATTRIBUTES = [
    "index_array_int",
    "index_array_str",
    "array",
    "array_2d",
    "array_3d",
    "masked_array",
    "times",
]
UNSLICEABLE_ATTRIBUTES = [
    "tuple",
    "int",
    "float",
    "str",
    "dict",
    "set",
]


class TestIndexable(Indexable):
    def __init__(self, size: int = N):
        self.index_array_int = np.array([i % 10 for i in range(size)])
        self.index_array_str = self.index_array_int.astype("str")
        self.array = np.arange(0, size)
        self.array_2d = np.random.random((size, 6))
        self.array_3d = np.random.random((size, 6, 6))
        self.masked_array = np.ma.arange(0, size)
        self.masked_array.mask = np.zeros(size)
        self.masked_array.mask[0:size:2] = 1
        self.times = Time(np.arange(59000, 59000 + size), scale="utc", format="mjd")

        self.tuple = (1, 2, 3)
        self.int = 1
        self.float = 1.0
        self.str = "test"
        self.dict = {"a": 1, "b": 2}
        self.set = {1, 2, 3}

        super().__init__()


class DummyIndexable(Indexable):
    def __init__(self, array):
        self.array = array

        super().__init__()


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
    member_length = indexable._check_member_validity()

    # Assert that the function correctly computes the length of the member
    # attributes and sets the according member length and index
    assert member_length == N


def test__convert_to_array():
    # Test that the function correctly converts strings, floats, ints to an array
    # as a single element in the array
    value = "test str"
    desired = np.array(["test str"])
    np.testing.assert_equal(Indexable._convert_to_array(value), desired)

    value = 23
    desired = np.array([23])
    np.testing.assert_equal(Indexable._convert_to_array(value), desired)

    value = 3.14
    desired = np.array([3.14])
    np.testing.assert_equal(Indexable._convert_to_array(value), desired)

    value = [3, 2, 1]
    desired = np.array([3, 2, 1])
    np.testing.assert_equal(Indexable._convert_to_array(value), desired)

    value = np.array([1, 2, 3])
    desired = np.array([1, 2, 3])
    np.testing.assert_equal(Indexable._convert_to_array(value), desired)


def test_Indexable__eq__array():
    # Numpy array
    indexable_1 = DummyIndexable(np.array([0, 1, 2]))
    indexable_2 = DummyIndexable(np.array([0, 1, 2]))
    indexable_3 = DummyIndexable(np.array([0, 1, 3]))

    assert indexable_1 == indexable_2
    assert indexable_1 != indexable_3


def test_Indexable__eq__masked_array():
    # Numpy masked array
    values = np.ma.array([0, 1, 2])
    values.mask = [True, False, True]
    indexable_1 = DummyIndexable(values)
    indexable_2 = DummyIndexable(values)
    values3 = np.ma.array([0, 1, 3])
    values3.mask = [True, False, True]
    indexable_3 = DummyIndexable(values3)

    assert indexable_1 == indexable_2
    assert indexable_1 != indexable_3

    # Numpy masked array (same values but different mask)
    values = np.ma.array([0, 1, 2])
    values.mask = [True, False, True]
    indexable_1 = DummyIndexable(values)
    indexable_2 = DummyIndexable(values)
    values3 = np.ma.array([0, 1, 2])
    values3.mask = [True, True, True]
    indexable_3 = DummyIndexable(values3)

    assert indexable_1 == indexable_2
    assert indexable_1 != indexable_3


def test_Indexable__eq__Time():
    # Astropy Time object
    indexable_1 = DummyIndexable(Time([59000.0, 59001.0, 59002.0], format="mjd"))
    indexable_2 = DummyIndexable(Time([59000.0, 59001.0, 59002.0], format="mjd"))
    indexable_3 = DummyIndexable(Time([59000.0, 59001.0, 59003.0], format="mjd"))

    assert indexable_1 == indexable_2
    assert indexable_1 != indexable_3


def test_Indexable__eq__Indexable():
    # Other Indexable
    indexable_1 = DummyIndexable(DummyIndexable(np.array([0, 1, 2])))
    indexable_2 = DummyIndexable(DummyIndexable(np.array([0, 1, 2])))
    indexable_3 = DummyIndexable(DummyIndexable(np.array([0, 1, 3])))
    indexable_4 = DummyIndexable(DummyIndexable(np.array([0, 1, 2])))
    indexable_4.__dict__["test"] = "test"

    assert indexable_1 == indexable_2
    assert indexable_1 != indexable_3
    assert indexable_1 != indexable_4


def test_Indexable__eq__None():
    # Other types (None)
    indexable_1 = DummyIndexable(np.array([0, 1, 2]))
    indexable_1.__dict__["test"] = None
    indexable_2 = DummyIndexable(np.array([0, 1, 2]))
    indexable_2.__dict__["test"] = None
    indexable_3 = DummyIndexable(np.array([0, 1, 2]))

    assert indexable_1 == indexable_2
    assert indexable_1 != indexable_3


def test_Indexable__eq__str():
    # Other types (str)
    indexable_1 = DummyIndexable(np.array([0, 1, 2]))
    indexable_1.__dict__["test"] = "test"
    indexable_2 = DummyIndexable(np.array([0, 1, 2]))
    indexable_2.__dict__["test"] = "test"
    indexable_3 = DummyIndexable(np.array([0, 1, 2]))
    indexable_3.__dict__["test"] = "test2"

    assert indexable_1 == indexable_2
    assert indexable_1 != indexable_3


def test_Indexable__eq__dict():
    # Other types (dict)
    indexable_1 = DummyIndexable(np.array([0, 1, 2]))
    indexable_1.__dict__["test"] = {"a": 0, "b": "1"}
    indexable_2 = DummyIndexable(np.array([0, 1, 2]))
    indexable_2.__dict__["test"] = {"a": 0, "b": "1"}
    indexable_3 = DummyIndexable(np.array([0, 1, 2]))
    indexable_3.__dict__["test"] = {"a": 0, "b": "2"}

    assert indexable_1 == indexable_2
    assert indexable_1 != indexable_3


def test_Indexable__eq__float():
    # Other types (float)
    indexable_1 = DummyIndexable(np.array([0, 1, 2]))
    indexable_1.__dict__["test"] = 0.1
    indexable_2 = DummyIndexable(np.array([0, 1, 2]))
    indexable_2.__dict__["test"] = 0.1
    indexable_3 = DummyIndexable(np.array([0, 1, 2]))
    indexable_3.__dict__["test"] = 0.2

    assert indexable_1 == indexable_2
    assert indexable_1 != indexable_3


def test_Indexable__eq__int():
    # Other types (int)
    indexable_1 = DummyIndexable(np.array([0, 1, 2]))
    indexable_1.__dict__["test"] = 1
    indexable_2 = DummyIndexable(np.array([0, 1, 2]))
    indexable_2.__dict__["test"] = 1
    indexable_3 = DummyIndexable(np.array([0, 1, 2]))

    assert indexable_1 == indexable_2
    assert indexable_1 != indexable_3


def test_Indexable__eq__set():
    # Other types (set)
    indexable_1 = DummyIndexable(np.array([0, 1, 2]))
    indexable_1.__dict__["test"] = set([0, 1])
    indexable_2 = DummyIndexable(np.array([0, 1, 2]))
    indexable_2.__dict__["test"] = set([0, 1])
    indexable_3 = DummyIndexable(np.array([0, 1, 2]))
    indexable_3.__dict__["test"] = set([0, 2])

    assert indexable_1 == indexable_2
    assert indexable_1 != indexable_3


def test_Indexable__eq__tuple():
    # Other types (tuple)
    indexable_1 = DummyIndexable(np.array([0, 1, 2]))
    indexable_1.__dict__["test"] = (0, 1)
    indexable_2 = DummyIndexable(np.array([0, 1, 2]))
    indexable_2.__dict__["test"] = (0, 1)
    indexable_3 = DummyIndexable(np.array([0, 1, 2]))
    indexable_3.__dict__["test"] = (0, 2)

    assert indexable_1 == indexable_2
    assert indexable_1 != indexable_3


def test_Indexable__check_index_int():
    # Check that an integer is correctly converted to a slice
    indexable = DummyIndexable(np.array([0, 1, 2]))
    assert indexable._check_index(0) == slice(0, 1, None)


def test_Indexable__check_index_slice():
    # Check that a slice is correctly returned
    indexable = DummyIndexable(np.array([0, 1, 2]))
    assert indexable._check_index(slice(0, 1, None)) == slice(0, 1, None)


def test_Indexable__check_index_array():
    # Check that an array is correctly returned
    indexable = DummyIndexable(np.array([0, 1, 2]))
    assert np.array_equal(
        indexable._check_index(np.array([0, 1, 2])), np.array([0, 1, 2])
    )


def test_Indexable__check_index_list():
    # Check that a list is correctly converted to an array
    indexable = DummyIndexable(np.array([0, 1, 2]))
    assert np.array_equal(indexable._check_index([0, 1, 2]), np.array([0, 1, 2]))


def test_Indexable__query_index_slice():
    # Array is grouped and can be represented by slices
    indexable = DummyIndexable(np.array([0, 1, 2, 3, 4, 5]))
    indexable.set_index("array")
    index = indexable._query_index(slice(0, 2, 1))
    assert index == slice(0, 2, 1)

    # Array is not grouped and can be not be represented by slices
    indexable = DummyIndexable(np.array([0, 1, 0]))
    indexable.set_index("array")
    # I want the first two unique values in array which is the entire array
    index = indexable._query_index(slice(0, 2, 1))
    np.testing.assert_equal(index, np.array([0, 1, 2]))


def test_Indexable__query_index_array():
    # Array is grouped and can be represented by slices
    indexable = DummyIndexable(np.array([0, 1, 2, 3, 4, 5]))
    indexable.set_index("array")
    index = indexable._query_index(np.array([0, 1]))
    assert index == slice(0, 2, 1)

    # Array is not grouped and can be not be represented by slices
    indexable = DummyIndexable(np.array([0, 1, 0]))
    indexable.set_index("array")
    # I want the first two unique values in array which is the entire array
    index = indexable._query_index(np.array([0, 1]))
    np.testing.assert_equal(index, np.array([0, 1, 2]))


def test_Indexable__query_index_raises():
    # Try to query with an integer
    indexable = DummyIndexable(np.array([0, 1, 2, 3, 4, 5]))
    with pytest.raises(TypeError):
        indexable._query_index(2)

    # Try to query with a list
    indexable = DummyIndexable(np.array([0, 1, 2, 3, 4, 5]))
    with pytest.raises(TypeError):
        indexable._query_index([0, 2])


def test_Indexable_slicing():
    # Create test indexable
    indexable = TestIndexable()

    # For each slice, slice the test indexable and check that this operation
    # is equivalent to slicing the individual attributes
    for s in SLICES:
        for attribute_i in SLICEABLE_ATTRIBUTES:
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
        for attribute_i in SLICEABLE_ATTRIBUTES:
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
    for attribute_i in SLICEABLE_ATTRIBUTES:
        assert_equal(
            indexable_mod.__dict__[attribute_i], indexable.__dict__[attribute_i][1:]
        )
    for attribute_i in UNSLICEABLE_ATTRIBUTES:
        assert_equal(
            indexable_mod.__dict__[attribute_i],
            indexable.__dict__[attribute_i],
        )

    # Delete the last 10 elements of the indexable and check that this operation
    # is equivalent to deleting the last 10 elements of the individual attributes
    del indexable_mod[-10:]
    for attribute_i in SLICEABLE_ATTRIBUTES:
        assert_equal(
            indexable_mod.__dict__[attribute_i],
            indexable.__dict__[attribute_i][1 : N - 10],
        )
    for attribute_i in UNSLICEABLE_ATTRIBUTES:
        assert_equal(
            indexable_mod.__dict__[attribute_i],
            indexable.__dict__[attribute_i],
        )

    # Delete the first 20 elements of the indexable and check that this operation
    # is equivalent to deleting the first 20 elements of the individual attributes
    del indexable_mod[:20]
    for attribute_i in SLICEABLE_ATTRIBUTES:
        assert_equal(
            indexable_mod.__dict__[attribute_i],
            # We've removed the first 1 element in the first test, and now
            # we remove the next 20
            indexable.__dict__[attribute_i][21 : N - 10],
        )
    for attribute_i in UNSLICEABLE_ATTRIBUTES:
        assert_equal(
            indexable_mod.__dict__[attribute_i],
            indexable.__dict__[attribute_i],
        )

    # Delete the first and last element and check that this operation is equivalent to
    # deleting the first and last element of the individual attributes
    array_slice = np.array([0, -1])
    del indexable_mod[array_slice]
    for attribute_i in SLICEABLE_ATTRIBUTES:
        assert_equal(
            indexable_mod.__dict__[attribute_i],
            # We've removed the first 21 elements in test 1 and 3, and now
            # we remove the next one and the last element
            indexable.__dict__[attribute_i][22 : N - 11],
        )
    for attribute_i in UNSLICEABLE_ATTRIBUTES:
        assert_equal(
            indexable_mod.__dict__[attribute_i],
            indexable.__dict__[attribute_i],
        )

    return


def test_Indexable_yield_chunks():
    # Create test indexable
    indexable = TestIndexable()

    # Iterate through the indexable and check that this operation
    # is equivalent to iterating through the individual attributes
    for i, indexable_i in enumerate(indexable.yield_chunks(10)):
        for attribute_i in SLICEABLE_ATTRIBUTES:
            assert_equal(
                indexable_i.__dict__[attribute_i],
                indexable.__dict__[attribute_i][i * 10 : (i + 1) * 10],
            )
        for attribute_i in UNSLICEABLE_ATTRIBUTES:
            assert_equal(
                indexable_i.__dict__[attribute_i],
                indexable.__dict__[attribute_i],
            )

    return


def test_Indexable_set_index_int_unsorted():
    # Create test indexable
    indexable = TestIndexable()
    indexable.set_index("index_array_int")

    # The index is unsorted with the numbers 0-9 repeated 10 times
    assert indexable._index_attribute == "index_array_int"
    np.testing.assert_equal(indexable._index, np.unique(indexable.index_array_int))

    # The class index to members mapping should be an array of integers
    # since the index is cannot be represented as slices
    np.testing.assert_equal(indexable._index_to_integers, indexable.index_array_int)
    assert indexable._index_to_slices is None

    # If we grab the first element of the class now we expect each member array to the every 10th element
    # in the original members
    indexable_i = indexable[0]
    for attribute_i in SLICEABLE_ATTRIBUTES:
        assert_equal(
            indexable_i.__dict__[attribute_i], indexable.__dict__[attribute_i][::10]
        )
    for attribute_i in UNSLICEABLE_ATTRIBUTES:
        assert_equal(
            indexable_i.__dict__[attribute_i],
            indexable.__dict__[attribute_i],
        )

    return


def test_Indexable_set_index_int_sorted():
    # Create test indexable
    indexable = TestIndexable()
    indexable.index_array_int.sort()
    indexable.set_index("index_array_int")

    # The index is sorted with the with the numbers 0-9 repeated 10 times in order
    assert indexable._index_attribute == "index_array_int"
    np.testing.assert_equal(indexable._index, np.unique(indexable.index_array_int))

    # The class index to slices mapping should be an array of slices
    # since the index is can be represented as slices
    np.testing.assert_equal(
        indexable._index_to_slices,
        np.array([slice(10 * i, 10 * (i + 1), 1) for i in range(10)]),
    )
    assert indexable._index_to_integers is None

    # If we grab the first element of the class now we expect each member array to the every 10th element
    # in the original members
    indexable_i = indexable[0]
    for attribute_i in SLICEABLE_ATTRIBUTES:
        assert_equal(
            indexable_i.__dict__[attribute_i], indexable.__dict__[attribute_i][:10]
        )
    for attribute_i in UNSLICEABLE_ATTRIBUTES:
        assert_equal(
            indexable_i.__dict__[attribute_i],
            indexable.__dict__[attribute_i],
        )


def test_Indexable_set_index_str_unsorted():
    # Create test indexable
    indexable = TestIndexable()

    # Test that the length of the class is the same as the length of the index
    assert len(indexable) == 100

    # Set the index
    indexable.set_index("index_array_str")

    # The index is unsorted with the numbers 0-9 repeated 10 times
    assert indexable._index_attribute == "index_array_str"
    np.testing.assert_equal(indexable._index, np.arange(0, 10))

    # The class index to members mapping should be an array of integers
    # since the index is cannot be represented as slices
    np.testing.assert_equal(indexable._index_to_integers, indexable.index_array_int)
    assert indexable._index_to_slices is None

    # If we grab the first element of the class now we expect each member array to the every 10th element
    # in the original members
    indexable_i = indexable[0]
    for attribute_i in SLICEABLE_ATTRIBUTES:
        assert_equal(
            indexable_i.__dict__[attribute_i], indexable.__dict__[attribute_i][::10]
        )
    for attribute_i in UNSLICEABLE_ATTRIBUTES:
        assert_equal(
            indexable_i.__dict__[attribute_i],
            indexable.__dict__[attribute_i],
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
    assert indexable._index_attribute == "index_array_str"
    np.testing.assert_equal(indexable._index, np.arange(0, 10))

    # The class index to slices mapping should be an array of slices
    # since the index is can be represented as slices
    np.testing.assert_equal(
        indexable._index_to_slices,
        np.array([slice(10 * i, 10 * (i + 1), 1) for i in range(10)]),
    )
    assert indexable._index_to_integers is None

    # If we grab the first element of the class now we expect each member array to the every 10th element
    # in the original members
    indexable_i = indexable[0]
    for attribute_i in SLICEABLE_ATTRIBUTES:
        assert_equal(
            indexable_i.__dict__[attribute_i], indexable.__dict__[attribute_i][:10]
        )
    for attribute_i in UNSLICEABLE_ATTRIBUTES:
        assert_equal(
            indexable_i.__dict__[attribute_i],
            indexable.__dict__[attribute_i],
        )

    # Test that the length of the class is the same as the length of the index
    assert len(indexable) == 10
    return


def test_Indexable_concatenate():
    indexable1 = TestIndexable()
    indexable2 = TestIndexable()
    indexable3 = TestIndexable()
    # Slightly modify the second and third indexable with the exception of the array of strings
    for attribute_i in SLICEABLE_ATTRIBUTES:
        if attribute_i != "index_array_str":
            indexable2.__dict__[attribute_i] = indexable2.__dict__[attribute_i] + 100
            indexable3.__dict__[attribute_i] = indexable3.__dict__[attribute_i] + 200

    indexable = concatenate([indexable1, indexable2, indexable3])
    for attribute_i in SLICEABLE_ATTRIBUTES:
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
    for attribute_i in UNSLICEABLE_ATTRIBUTES:
        assert_equal(
            indexable.__dict__[attribute_i],
            indexable1.__dict__[attribute_i],
        )
        assert_equal(
            indexable.__dict__[attribute_i],
            indexable2.__dict__[attribute_i],
        )
        assert_equal(
            indexable.__dict__[attribute_i],
            indexable3.__dict__[attribute_i],
        )

    return


def test_Indexable_sort_values():
    # Create test indexable
    indexable = TestIndexable()

    # Sort the indexable by the index array
    indexable_sorted = indexable.sort_values("index_array_int", inplace=False)

    # Test that the attributes are correctly sorted
    for attribute_i in SLICEABLE_ATTRIBUTES:
        assert_equal(
            indexable_sorted.__dict__[attribute_i][:10],
            indexable.__dict__[attribute_i][::10],
        )
    for attribute_i in UNSLICEABLE_ATTRIBUTES:
        assert_equal(
            indexable_sorted.__dict__[attribute_i],
            indexable.__dict__[attribute_i],
        )

    return


def test_Indexable_sort_values_inplace():
    # Create test indexable
    indexable = TestIndexable()
    indexable_unsorted = deepcopy(indexable)

    # Sort the indexable by the index array
    indexable.sort_values("index_array_int", inplace=True)

    # Test that the attributes are correctly sorted
    for attribute_i in SLICEABLE_ATTRIBUTES:
        assert_equal(
            indexable.__dict__[attribute_i][:10],
            indexable_unsorted.__dict__[attribute_i][::10],
        )
    for attribute_i in UNSLICEABLE_ATTRIBUTES:
        assert_equal(
            indexable.__dict__[attribute_i],
            indexable_unsorted.__dict__[attribute_i],
        )

    return


def test_Indexable_getitem_index_copying():
    indexable = TestIndexable()
    one_row = indexable[0]
    assert one_row._index.shape == (1,)
    assert one_row._index[0] == 0
