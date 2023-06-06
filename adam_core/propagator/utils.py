from typing import Iterable, Sequence

import pyarrow as pa
from pyarrow import compute as pc

from ..orbits import Orbits


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


def sort_propagated_orbits(propagated_orbits: Orbits) -> Orbits:
    """
    Sort propagated orbits by orbit_id, object_id, and time.

    Parameters
    ----------
    propagated_orbits : Orbits
        Orbits to sort.

    Returns
    -------
    Orbits
        Sorted orbits.
    """
    # Build table with orbit_ids, object_ids, and times
    table = pa.table(
        [
            propagated_orbits.orbit_ids,
            propagated_orbits.object_ids,
            pc.add(
                pc.struct_field(
                    pc.struct_field(propagated_orbits.table["coordinates"], "times"),
                    "jd1",
                ),
                pc.struct_field(
                    pc.struct_field(propagated_orbits.table["coordinates"], "times"),
                    "jd2",
                ),
            ),
        ],
        names=["orbit_ids", "object_ids", "times"],
    )

    indices = pc.sort_indices(
        table,
        (
            ("orbit_ids", "ascending"),
            ("object_ids", "ascending"),
            ("times", "ascending"),
        ),
    )
    return propagated_orbits.take(indices)
