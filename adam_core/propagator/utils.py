from typing import Iterable, Sequence

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc

from ..orbits.ephemeris import Ephemeris
from ..orbits.orbits import Orbits

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
        raise ValueError(f"Times were not within {tolerance:.2f} ms of each other.")


def sort_propagated_orbits(propagated_orbits: Orbits) -> Orbits:
    """
    Sort propagated orbits by orbit_id, object_id, and time.

    Parameters
    ----------
    propagated_orbits : `~adam_core.orbits.orbits.Orbits`
        Orbits to sort.

    Returns
    -------
    Orbits : `~adam_core.orbits.orbits.Orbits`
        Sorted orbits.
    """
    # Build table with orbit_ids, object_ids, and times
    table = pa.table(
        [
            propagated_orbits.orbit_id,
            propagated_orbits.object_id,
            pc.add(
                pc.struct_field(
                    pc.struct_field(propagated_orbits.table["coordinates"], "time"),
                    "jd1",
                ),
                pc.struct_field(
                    pc.struct_field(propagated_orbits.table["coordinates"], "time"),
                    "jd2",
                ),
            ),
        ],
        names=["orbit_id", "object_id", "time"],
    )

    indices = pc.sort_indices(
        table,
        (
            ("orbit_id", "ascending"),
            ("object_id", "ascending"),
            ("time", "ascending"),
        ),
    )
    return propagated_orbits.take(indices)


def sort_ephemeris(ephemeris: Ephemeris) -> Ephemeris:
    """
    Sort ephemeris by orbit_id, object_id, time, and observatory code.

    Parameters
    ----------
    ephemeris : `~adam_core.orbits.ephemeris.Ephemeris`
        Ephemerides to sort.

    Returns
    -------
    ephemeris : `~adam_core.orbits.ephemeris.Ephemeris`
        Sorted ephemerides.
    """
    # Build table with orbit_ids, object_ids, times and observatory codes
    coords_array = ephemeris.table["observer"].combine_chunks().field("coordinates")

    table = pa.table(
        [
            ephemeris.orbit_id,
            ephemeris.object_id,
            pc.add(
                pc.struct_field(
                    pc.struct_field(coords_array, "time"),
                    "jd1",
                ),
                pc.struct_field(
                    pc.struct_field(coords_array, "time"),
                    "jd2",
                ),
            ),
            ephemeris.observer.code,
        ],
        names=["orbit_id", "object_id", "time", "observatory_code"],
    )

    indices = pc.sort_indices(
        table,
        (
            ("orbit_id", "ascending"),
            ("object_id", "ascending"),
            ("time", "ascending"),
            ("observatory_code", "ascending"),
        ),
    )
    return ephemeris.take(indices)
