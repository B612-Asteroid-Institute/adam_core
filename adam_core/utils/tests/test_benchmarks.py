import pytest

from .test_indexable import TestIndexable, concatenate


@pytest.mark.parametrize(
    ["size", "slicer"],
    [
        (10, 0),
        (100, 1),
        (100, 99),
        (100, slice(5, 10, 1)),
        (100, slice(99, 10, -2)),
        (10000, 9999),
        (100000, 99999),
    ],
    ids=repr
)
@pytest.mark.benchmark(group="slice")
def test_benchmark_slice(benchmark, size, slicer):
    indexable = TestIndexable(size)
    benchmark(indexable.__getitem__, 1)


@pytest.mark.parametrize("size", [10, 100, 1000, 10000])
@pytest.mark.benchmark(group="iterate")
def test_benchmark_iterate(benchmark, size):
    indexable = TestIndexable(size)

    def noop_iterate(iterable):
        for x in iterable:
            pass

    benchmark(noop_iterate, indexable)


@pytest.mark.parametrize(
    "indexer",
    [1, [3, 5, 7], slice(2, 5, 1)],
    ids=lambda x: f"(idx={repr(x)})"
)
@pytest.mark.parametrize(
    "size",
    [100, 1000, 10000],
    ids=lambda x: f"(size={x})"
)
@pytest.mark.benchmark(group="slice")
def test_benchmark_indirect_indexing(benchmark, size, indexer):
    indexable = TestIndexable(size)
    indexable.set_index("index_array_int")
    benchmark(indexable.__getitem__, indexer)


@pytest.mark.parametrize(
    "size",
    [10, 100, 1000, 10000],
    ids=lambda x: f"(size={x})"
)
@pytest.mark.parametrize(
    "n",
    [2, 5, 10],
    ids=lambda x: f"(n={x})"    
)
@pytest.mark.benchmark(group="concat")
def test_benchmark_concatenate(benchmark, size, n):
    benchmark(concatenate, [TestIndexable(size)] * n)
