from copy import deepcopy

import pandas as pd
from astropy.time import Time

__all__ = ["times_from_df", "times_to_df"]


def times_from_df(df: pd.DataFrame) -> Time:
    """
    Read times from a `~pandas.DataFrame`.

    Parameters
    ----------
    df : `~pandas.DataFrame`
        DataFrame containing times.

    Returns
    -------
    times : `~astropy.time.core.Time`
        Astropy time object containing times read from dataframe.
    """
    time_col = None
    cols = [f"mjd_{s}" for s in Time.SCALES]
    for col in cols:
        if col in df.columns:
            time_col = col
            format, scale = time_col.split("_")
            break

    times = Time(df[time_col].values, format=format, scale=scale)
    return times


def times_to_df(times: Time, time_scale: str = "utc") -> pd.DataFrame:
    """
    Store times as a `~pandas.DataFrame`.

    Parameters
    ----------
    times : `~astropy.time.core.Time`
        Astropy time object.
    time_scale : str
        Store times with this time scale.

    Returns
    -------
    df : `~pandas.DataFrame`
        DataFrame containing times.
    """
    data = {}
    time = deepcopy(times)
    time._set_scale(time_scale)
    data[f"mjd_{time.scale}"] = time.mjd

    return pd.DataFrame(data)
