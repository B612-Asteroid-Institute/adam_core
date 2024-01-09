from typing import Iterable, Sequence, Tuple

import numpy as np

MILLISECOND_IN_DAYS = 1 / 86400 / 1000


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


def _iterate_chunk_indices(
    iterable: Sequence, chunk_size: int
) -> Iterable[Tuple[int, int]]:
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


def _assert_times_almost_equal(
    have: np.ndarray, want: np.ndarray, tolerance: float = 0.1
):
    """
    Raises a ValueError if the time arrays (in units of days such as MJD) are not within the
    tolerance in milliseconds of each other.

    Parameters
    ----------
    have : `~numpy.ndarray`
        Times (in units of days) to check.
    want : `~numpy.ndarray`
        Times (in units of days) to check.

    Raises
    ------
    ValueError: If the time arrays are not within the tolerance in milliseconds of each other.
    """
    tolerance_in_days = tolerance * MILLISECOND_IN_DAYS

    diff = np.abs(have - want)
    if np.any(diff > tolerance_in_days):
        raise ValueError(f"Times were not within {tolerance:.6f} ms of each other.")
