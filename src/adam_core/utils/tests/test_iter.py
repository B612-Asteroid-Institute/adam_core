import numpy as np
import quivr as qv

from ..iter import ChunkedParquetWriter, _iterate_chunk_indices, _iterate_chunks, qv_table_iter


class SampleTable(qv.Table):
    a = qv.Float64Column()


class AttrTable(qv.Table):
    a = qv.Int64Column()
    foo = qv.StringAttribute(default="default")


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


def test__iterate_chunk_indices():
    # Test that _iterate_chunk_indices works with numpy arrays and lists
    a = np.arange(0, 10)
    for i, chunk in enumerate(_iterate_chunk_indices(a, 2)):
        assert chunk == (i * 2, i * 2 + 2)

    a = [i for i in range(10)]
    for i, chunk in enumerate(_iterate_chunk_indices(a, 2)):
        assert chunk == (i * 2, i * 2 + 2)

    table = SampleTable.from_kwargs(a=np.arange(0, 11))
    for i, chunk in enumerate(_iterate_chunk_indices(table, 2)):
        if i != 5:
            assert chunk == (i * 2, i * 2 + 2)
        else:
            assert chunk == (i * 2, i * 2 + 1)


def test_qv_table_iter_preserves_attributes_from_parquet(tmp_path):
    t = AttrTable.from_kwargs(a=np.arange(0, 5), foo="bar")
    p = tmp_path / "attr_table.parquet"
    t.to_parquet(p)

    chunks = list(qv_table_iter(AttrTable, p, max_chunk_size=2))
    assert len(chunks) == 3
    assert all(isinstance(c, AttrTable) for c in chunks)
    assert all(c.foo == "bar" for c in chunks)


def test_qv_table_iter_preserves_attributes_for_in_memory_tables():
    t = AttrTable.from_kwargs(a=np.arange(0, 5), foo="bar")
    chunks = list(qv_table_iter(AttrTable, t, max_chunk_size=2))
    assert len(chunks) == 3
    assert all(isinstance(c, AttrTable) for c in chunks)
    assert all(c.foo == "bar" for c in chunks)


def test_chunked_parquet_writer_preserves_attributes(tmp_path):
    t = AttrTable.from_kwargs(a=np.arange(0, 5), foo="bar")
    out = tmp_path / "out.parquet"

    w = ChunkedParquetWriter(out)
    w.write(t[:2])
    w.write(t[2:])
    w.close()

    reread = AttrTable.from_parquet(out)
    assert reread.foo == "bar"
