import pytest
import numpy as np
from astropy.time import Time

from ..indexable import Indexable

class TestIndexable(Indexable):

    def __init__(self, values):
        self.values = values
        Indexable.__init__(self)

SLICES = [
    slice(0, 1, 1),   #  0
    slice(1, 2, 1),   #  1
    slice(8, 9, 1),   # -1
    slice(7, 8, 1),   # -2
    slice(0, 10, 1),
    slice(0, 5, 1),
    slice(5, 10, 1),
]

def test_Indexable_slicing_array():

    array = np.arange(0, 10)

    indexable = TestIndexable(array)
    for s in SLICES:
        np.testing.assert_equal(indexable[s].values, indexable.values[s])

    return

def test_Indexable_slicing_marray():

    masked_array = np.ma.arange(0, 10)
    masked_array.mask = np.zeros(len(masked_array))
    masked_array.mask[0:10:2] = 1

    indexable = TestIndexable(masked_array)
    for s in SLICES:
        np.testing.assert_equal(indexable[s].values.data, indexable.values[s].data)
        np.testing.assert_equal(indexable[s].values.mask, indexable.values[s].mask)

    return

def test_Indexable_slicing_time():

    times = Time(np.arange(59000, 59010), scale="utc", format="mjd")

    indexable = TestIndexable(times)
    for s in SLICES:
        np.testing.assert_equal(indexable[s].values.mjd, indexable.values[s].mjd)
        assert indexable[s].values.scale == indexable.values[s].scale
        assert indexable[s].values.format == indexable.values[s].format

    return
