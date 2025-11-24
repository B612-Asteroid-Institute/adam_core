from typing import Iterable, Sequence, Tuple
import pathlib
import pyarrow as pa
import pyarrow.parquet as pq
import quivr as qv
import logging

logger = logging.getLogger(__name__)


def _iterate_chunks(iterable: Sequence, chunk_size: int) -> Iterable:
    """
    Generator that yields chunks of size chunk_size from sized iterable
    (Sequence).

    Parameters
    ----------
    iterable : Sequence
        Iterable to chunk.
    chunk_size : int
        Size of chunks.

    Yields
    ------
    chunk : iterable
        Chunk of size chunk_size from iterable.
    """
    N = len(iterable)
    for i in range(0, N, chunk_size):
        yield iterable[i : i + chunk_size]


def _iterate_chunk_indices(iterable: Sequence, chunk_size: int) -> Iterable[Tuple[int, int]]:
    """
    Generator that yields indices for chunks of size chunk_size from sized
    iterable (Sequence).

    Parameters
    ----------
    iterable : Sequence
        Iterable to chunk.
    chunk_size : int
        Size of chunks.

    Yields
    ------
    chunk : iterable
        Chunk of size chunk_size from iterable.
    """
    N = len(iterable)
    for i in range(0, N, chunk_size):
        yield i, min(i + chunk_size, N)


def qv_table_iter(
    table_cls: type[qv.Table],
    file_or_dir: str | pathlib.Path,
    filename_pattern: str = "*.parquet",
    max_chunk_size: int = 1000,
) -> Iterable[qv.Table]:
    """
    Generator that yields chunks of qv.Table from a file or directory

    Parameters:
        table_cls: type[qv.Table]
            The class of the table to yield
        file_or_dir: str | pathlib.Path
            The file or directory to yield tables from
        row_batch_size: int
            The number of rows to batch in each table

    Yields:
        table: qv.Table
            A chunk of the table
    """
    # Discover all the files
    files = list(pathlib.Path(file_or_dir).glob(filename_pattern))
    logger.debug(f"Found {len(files)} files")
    for file in files:
        pf = pq.ParquetFile(file)
        logger.debug(f"Processing file {file}")
        for batch in pf.iter_batches(batch_size=max_chunk_size):
            tbl = pa.Table.from_batches([batch])
            yield table_cls.from_pyarrow(tbl)
        del pf
        del batch
        del tbl