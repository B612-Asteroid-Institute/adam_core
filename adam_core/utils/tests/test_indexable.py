import pytest
import numpy as np
from astropy.time import Time
from abc import ABC, abstractmethod

from ..indexable import Indexable

SLICES = [
    slice(0, 1, 1),  #  0
    slice(1, 2, 1),  #  1
    slice(8, 9, 1),  # -1
    slice(7, 8, 1),  # -2
    slice(0, 10, 1),
    slice(0, 5, 1),
    slice(5, 10, 1),
]


class TestIndexable(Indexable):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        Indexable.__init__(self)


class SliceableDataStructureTester(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def assert_equal(self, a, b):
        pass


class NumpyArrayTester(SliceableDataStructureTester):
    def __init__(self):
        array = np.arange(0, 100)
        self.values = array

    def assert_equal(self, a, b):
        if isinstance(a, (np.int_, np.float_)):
            assert a == b
        else:
            np.testing.assert_equal(a, b)


class NumpyMaskedArrayTester(SliceableDataStructureTester):
    def __init__(self):
        masked_array = np.ma.arange(0, 10)
        masked_array.mask = np.zeros(len(masked_array))
        masked_array.mask[0:10:2] = 1
        self.values = masked_array

    def assert_equal(self, a, b):
        if isinstance(a, (np.int_, np.float_)):
            assert a == b
        else:
            np.testing.assert_equal(a.data, b.data)
            np.testing.assert_equal(a.mask, b.mask)


class AstropyTimeTester(SliceableDataStructureTester):
    def __init__(self):
        times = Time(np.arange(59000, 59100), scale="utc", format="mjd")
        self.values = times

    def assert_equal(self, a, b):
        if isinstance(a, (np.int_, np.float_)):
            assert a == b
        else:
            np.testing.assert_equal(a.mjd, b.mjd)
            assert a.scale == b.scale
            assert a.format == b.format


def test_Indexable_slicing_array():

    tester = NumpyArrayTester()
    indexable = TestIndexable(values=tester.values)
    for s in SLICES:
        tester.assert_equal(indexable[s].values, indexable.values[s])

    return


def test_Indexable_slicing_marray():

    tester = NumpyMaskedArrayTester()
    indexable = TestIndexable(values=tester.values)
    for s in SLICES:
        tester.assert_equal(indexable[s].values, indexable.values[s])

    return


def test_Indexable_slicing_time():

    tester = AstropyTimeTester()
    indexable = TestIndexable(values=tester.values)
    for s in SLICES:
        tester.assert_equal(indexable[s].values, indexable.values[s])

    return

def test_Indexable_iteration_array():

    tester = NumpyArrayTester()
    indexable = TestIndexable(values=tester.values)
    for i, indexable_i in enumerate(indexable):
        tester.assert_equal(indexable_i.values[0], tester.values[i])

    return

def test_Indexable_iteration_marray():

    tester = NumpyMaskedArrayTester()
    indexable = TestIndexable(values=tester.values)
    for i, indexable_i in enumerate(indexable):
        tester.assert_equal(indexable_i.values[0], tester.values[i])

    return

def test_Indexable_iteration_time():

    tester = AstropyTimeTester()
    indexable = TestIndexable(values=tester.values)
    for i, indexable_i in enumerate(indexable):
        tester.assert_equal(indexable_i.values[0], tester.values[i])

    return
