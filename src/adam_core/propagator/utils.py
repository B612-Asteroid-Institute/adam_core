from typing import Union

import pyarrow.compute as pc
import quivr as qv

from ..coordinates import OriginCodes, transform_coordinates
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
    final_results = None
    unique_origins = inputs.coordinates.origin.code.unique()
    for origin_code in unique_origins:
        origin_orbits = inputs.select("coordinates.origin.code", origin_code)
        result_origin = results.apply_mask(
            pc.is_in(results.orbit_id, origin_orbits.orbit_id)
        )
        partial_results = result_origin.set_column(
            "coordinates",
            transform_coordinates(
                result_origin.coordinates,
                origin_out=OriginCodes[origin_code.as_py()],
                frame_out=inputs.coordinates.frame,
            ),
        )
        if final_results is None:
            final_results = partial_results
        else:
            final_results = qv.concatenate([final_results, partial_results])

    return final_results
