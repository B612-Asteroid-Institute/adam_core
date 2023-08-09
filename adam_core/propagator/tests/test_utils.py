import numpy as np
from quivr import Float64Column, Table

from ..utils import _iterate_chunks


class SampleTable(Table):
    a = Float64Column()


def test__iterate_chunks():
    # Test that _iterate_chunks works with numpy arrays and lists
    a = np.arange(0, 10)
    for chunk in _iterate_chunks(a, 2):
        assert len(chunk) == 2

    a = [i for i in range(10)]
    for chunk in _iterate_chunks(a, 2):
        assert len(chunk) == 2


def test__iterate_chunks_table():
    # Test that _iterate_chunks works with quivr tables
    table = SampleTable.from_kwargs(a=np.arange(0, 11))
    chunks = list(_iterate_chunks(table, 2))
    assert len(chunks) == 6

    for i, chunk in enumerate(chunks):
        if i != 5:
            assert len(chunk) == 2
            assert isinstance(chunk, SampleTable)
            np.testing.assert_equal(chunk.a.to_numpy(), np.arange(i * 2, i * 2 + 2))
        else:
            assert len(chunk) == 1
            assert isinstance(chunk, SampleTable)
            np.testing.assert_equal(chunk.a.to_numpy(), np.arange(i * 2, i * 2 + 1))
