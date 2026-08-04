from datetime import timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from astropy.time import Time

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
    half_day = timedelta(hours=12)
    observing_night = np.empty(len(times), dtype=np.int64)

    for code in pc.unique(codes):

        parallax_coefficients = OBSERVATORY_PARALLAX_COEFFICIENTS.select("code", code)
        if len(parallax_coefficients) == 0:
            raise ValueError(f"Unknown observatory code: {code}")

        tz = ZoneInfo(parallax_coefficients.timezone()[0])

        mask = pc.equal(codes, code)
        times_code = times.apply_mask(mask)

        observing_night_code = Time(
            [
                time + time.astimezone(tz).utcoffset() - half_day
                for time in times_code.to_astropy().datetime
            ],
            format="datetime",
            scale="utc",
        )
        observing_night_code = observing_night_code.mjd.astype(int)

        observing_night[mask.to_numpy(zero_copy_only=False)] = observing_night_code

    return pa.array(observing_night)
