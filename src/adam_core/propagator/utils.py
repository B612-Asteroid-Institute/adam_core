from typing import Union

from ..coordinates import CartesianCoordinates
from ..time import Timestamp
from .types import EphemerisType, OrbitType


def ensure_input_time_scale(
    results: Union[OrbitType, EphemerisType], times: Timestamp
) -> Union[OrbitType, EphemerisType]:
    """
    Ensure the time scale of the results is the same as the input.
    """
    return results.set_column(
        "coordinates.time", results.coordinates.time.rescale(times.scale)
    )


def ensure_input_origin_and_frame(
    inputs: Union[OrbitType, EphemerisType], results: Union[OrbitType, EphemerisType]
) -> Union[OrbitType, EphemerisType]:
    """
    Ensure the input origin and frame of the results are the same as the input.
    """
    from adam_core import _rust_native

    from .._rust.arrow import ensure_spice_backend, table_from_record_batch
    from ..coordinates.transform import _coordinate_record_batch

    ensure_spice_backend()
    output = _rust_native.ensure_input_origin_and_frame_arrow(
        _coordinate_record_batch(results.coordinates, "cartesian"),
        inputs.orbit_id.to_pylist(),
        inputs.coordinates.origin.code.to_pylist(),
        results.orbit_id.to_pylist(),
        inputs.coordinates.frame,
    )
    if output is None:
        return None
    coordinate_batch, rows = output
    coordinates = table_from_record_batch(CartesianCoordinates, coordinate_batch)
    return results.take(rows).set_column("coordinates", coordinates)
