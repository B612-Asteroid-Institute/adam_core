from __future__ import annotations

from typing import Iterable, Iterator, Sequence, Tuple, Union
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


def _discover_parquet_files(
    file_or_dir: str | pathlib.Path,
    *,
    filename_pattern: str,
) -> list[pathlib.Path]:
    """
    Return a sorted list of Parquet files for a path that may be a file or directory.
    """
    p = pathlib.Path(file_or_dir)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted([f for f in p.glob(filename_pattern) if f.is_file()])
    # Best-effort: return empty list if the path doesn't exist / isn't accessible.
    return []


def qv_table_iter(
    table_cls: type[qv.Table],
    file_or_dir: str | pathlib.Path | qv.Table,
    filename_pattern: str = "*.parquet",
    max_chunk_size: int = 1000,
    *,
    preserve_schema_metadata: bool = True,
) -> Iterable[qv.Table]:
    """
    Generator that yields chunks of qv.Table from a file or directory

    Parameters:
        table_cls: type[qv.Table]
            The class of the table to yield
        file_or_dir: str | pathlib.Path | qv.Table
            The file, directory, or in-memory table to yield chunks from
        row_batch_size: int
            The number of rows to batch in each table
        preserve_schema_metadata: bool
            When True, attach the source Parquet file's Arrow schema metadata to each
            yielded batch so Quivr attributes (e.g., time scale, frames, provenance)
            round-trip correctly.

    Yields:
        table: qv.Table
            A chunk of the table
    """
    # In-memory table: yield slices.
    if isinstance(file_or_dir, qv.Table):
        # Best-effort type guard: allow subclasses, but warn if mismatch.
        if not isinstance(file_or_dir, table_cls):
            logger.debug(
                "qv_table_iter: got in-memory %s but table_cls=%s; proceeding",
                type(file_or_dir).__name__,
                getattr(table_cls, "__name__", str(table_cls)),
            )
        for s, e in _iterate_chunk_indices(file_or_dir, max_chunk_size):
            yield file_or_dir[s:e]
        return

    # Discover all the files (file or directory input).
    files = _discover_parquet_files(file_or_dir, filename_pattern=filename_pattern)
    logger.debug("qv_table_iter: found %d parquet files under %s", len(files), file_or_dir)

    # Mirror Quivr's parquet loading behavior by only requesting the columns
    # that exist in the target table schema. This allows Parquet files to have
    # extra columns without breaking `from_pyarrow` due to schema mismatches.
    column_names = [field.name for field in table_cls.schema]
    for file in files:
        # Read the file schema metadata once and reattach it to every batch table.
        # This is critical for preserving Quivr attributes stored in Arrow schema
        # key/value metadata.
        file_metadata = None
        if preserve_schema_metadata:
            try:
                file_schema = pq.read_schema(file)
                file_metadata = (file_schema.metadata or None)
            except Exception:
                file_metadata = None

        pf = pq.ParquetFile(file)
        logger.debug(f"Processing file {file}")
        for batch in pf.iter_batches(batch_size=max_chunk_size, columns=column_names):
            tbl = pa.Table.from_batches([batch])
            if file_metadata:
                tbl = tbl.replace_schema_metadata(file_metadata)
            yield table_cls.from_pyarrow(tbl)
        del pf
        del batch
        del tbl


class ChunkedParquetWriter:
    """
    Minimal helper to stream Arrow/Quivr tables into a single Parquet file (or
    Arrow output stream), while preserving schema metadata (Quivr attributes).

    The writer schema defaults to the schema of the first written chunk (i.e. a
    "populated" schema), which is essential to preserve Quivr attributes that
    are stored in Arrow schema key/value metadata.
    """

    def __init__(
        self,
        destination: Union[str, pathlib.Path, pa.NativeFile],
        *,
        schema: pa.Schema | None = None,
        compression: str | None = "snappy",
        **parquet_writer_kwargs: object,
    ) -> None:
        self.destination = destination
        self.schema = schema
        self.compression = compression
        self._writer: pq.ParquetWriter | None = None
        self._parquet_writer_kwargs = parquet_writer_kwargs
        self.rows_written: int = 0

    def _ensure_writer(self, tbl: pa.Table) -> None:
        if self._writer is not None:
            return
        if self.schema is None:
            self.schema = tbl.schema
        self._writer = pq.ParquetWriter(
            self.destination,
            self.schema,
            compression=self.compression,
            **self._parquet_writer_kwargs,
        )

    def write(self, chunk: Union[qv.Table, pa.Table, pa.RecordBatch]) -> None:
        if isinstance(chunk, qv.Table):
            tbl = chunk.table
        elif isinstance(chunk, pa.RecordBatch):
            tbl = pa.Table.from_batches([chunk])
        else:
            tbl = chunk

        if tbl.num_rows == 0:
            return

        self._ensure_writer(tbl)
        assert self._writer is not None
        assert self.schema is not None

        if tbl.schema != self.schema:
            tbl = tbl.cast(self.schema, safe=False)

        self._writer.write_table(tbl)
        self.rows_written += int(tbl.num_rows)

    # Backwards-compatible spelling for existing call sites that use ParquetWriter.
    def write_table(self, tbl: pa.Table) -> None:
        self.write(tbl)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def write_parquet_chunks(
    destination: Union[str, pathlib.Path, pa.NativeFile],
    chunks: Iterable[Union[qv.Table, pa.Table, pa.RecordBatch]],
    *,
    schema: pa.Schema | None = None,
    compression: str | None = "snappy",
    **parquet_writer_kwargs: object,
) -> int:
    """
    Convenience wrapper around :class:`ChunkedParquetWriter` that writes an
    iterable of chunks into a single Parquet file/stream.

    Returns the total number of rows written.
    """
    w = ChunkedParquetWriter(
        destination,
        schema=schema,
        compression=compression,
        **parquet_writer_kwargs,
    )
    try:
        for chunk in chunks:
            w.write(chunk)
        return int(w.rows_written)
    finally:
        w.close()