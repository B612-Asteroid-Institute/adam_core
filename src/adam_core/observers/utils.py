from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ..time import Timestamp
from .observers import OBSERVATORY_PARALLAX_COEFFICIENTS


def calculate_observing_night(codes: pa.Array, times: Timestamp) -> pa.Array:
    """
    Compute the observing night for a given set of observatory codes and times. An observing night is defined as the night
    during which the observation is made, in the local time of the observatory +- 12 hours. The observing night is defined
    as the integer MJD of the night in the local time of the observatory.

    Parameters
    ----------
    codes : pyarrow.Array (N)
        An array of observatory codes.
    times : Timestamp (N)
        An array of times.

    Returns
    -------
    observing_night : pyarrow.Array (N)
        An array of observing nights.
    """
    from adam_core import _rust_native as _rn

    timezone_names = np.empty(len(times), dtype=object)
    for code in pc.unique(codes):
        parallax_coefficients = OBSERVATORY_PARALLAX_COEFFICIENTS.select("code", code)
        if len(parallax_coefficients) == 0:
            raise ValueError(f"Unknown observatory code: {code}")

        timezone_name = str(parallax_coefficients.timezone()[0])
        # Preserve ZoneInfo's public validation/error boundary before the one
        # Rust crossing owns historical UTC-offset/DST lookup and MJD math.
        ZoneInfo(timezone_name)
        mask = pc.equal(codes, code).to_numpy(zero_copy_only=False)
        timezone_names[mask] = timezone_name

    observing_nights = _rn.calculate_observing_nights_numpy(
        times.days.to_numpy(zero_copy_only=False),
        times.nanos.to_numpy(zero_copy_only=False),
        timezone_names.tolist(),
    )
    return pa.array(observing_nights, type=pa.int64())
