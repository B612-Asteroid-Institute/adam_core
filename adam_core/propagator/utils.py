from typing import Iterable, Sequence


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
