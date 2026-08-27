from typing import List, Optional, Union

import numpy.typing as npt
import pyarrow as pa

from ...observers import Observers
from ...time import Timestamp
from ...utils.http import _raise_compatible_http_error
from ..ephemeris import Ephemeris
from ..orbits import Orbits


def _time_batch(times: Timestamp) -> pa.RecordBatch:
    table = times.table.combine_chunks()
    return pa.RecordBatch.from_arrays(
        [table.column("days").chunk(0), table.column("nanos").chunk(0)],
        names=["days", "nanos"],
    )


def query_horizons_ephemeris(
    object_ids: Union[List, npt.ArrayLike], observers: Observers
) -> Ephemeris:
    """Query JPL Horizons for predicted ephemerides at observer epochs."""
    from adam_core import _rust_native

    from ..._rust.arrow import table_from_record_batch

    times = observers.coordinates.time
    time = times.table.combine_chunks()
    observer_table = observers.table.combine_chunks()
    observer_batch = pa.RecordBatch.from_arrays(
        [
            observer_table.column("code").chunk(0),
            time.column("days").chunk(0),
            time.column("nanos").chunk(0),
        ],
        names=["code", "days", "nanos"],
    )
    try:
        batch = _rust_native.query_horizons_ephemeris_arrow(
            [str(value) for value in object_ids],
            observer_batch,
            times.scale,
        )
    except RuntimeError as error:
        _raise_compatible_http_error(error)
    return table_from_record_batch(Ephemeris, batch)


def query_horizons(
    object_ids: Union[List, npt.ArrayLike],
    times: Timestamp,
    coordinate_type: str = "cartesian",
    location: str = "@sun",
    aberrations: str = "geometric",
    id_type: Optional[str] = None,
) -> Orbits:
    """Query JPL Horizons for state vectors or osculating elements."""
    from adam_core import _rust_native

    from ..._rust.arrow import table_from_record_batch

    try:
        batch = _rust_native.query_horizons_arrow(
            [str(value) for value in object_ids],
            _time_batch(times),
            times.scale,
            str(coordinate_type),
            str(location),
            str(aberrations),
            id_type,
        )
    except RuntimeError as error:
        _raise_compatible_http_error(error)
    return table_from_record_batch(Orbits, batch)
