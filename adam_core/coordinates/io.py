from typing import TYPE_CHECKING, List, Optional, Type, Union

import numpy as np
import pandas as pd

from .covariances import CoordinateCovariances
from .origin import Origin
from .times import Times

if TYPE_CHECKING:
    from .cartesian import CartesianCoordinates
    from .cometary import CometaryCoordinates
    from .keplerian import KeplerianCoordinates
    from .spherical import SphericalCoordinates

    CoordinateType = Union[
        CartesianCoordinates,
        CometaryCoordinates,
        KeplerianCoordinates,
        SphericalCoordinates,
    ]


def coords_to_dataframe(
    coords: "CoordinateType",
    coord_names: List[str],
    sigmas: Optional[bool] = None,
    covariances: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Store coordinates as a pandas DataFrame.

    Parameters
    ----------
    coords : {CartesianCoordinates, CometaryCoordinates, KeplerianCoordinates, SphericalCoordinates}
        Coordinates to store.
    coord_names : list of str
        Names of the coordinates to store. The coordinate reference frame will be appended to the
        coordinate names, e.g., "x" will become "x_ec" for ecliptic coordinates and "x_eq" for
        equatorial coordinates.
    sigmas : bool, optional
        If None, will check if any sigmas are defined (via covariance matrix) and add them to the dataframe.
        If True, include 1-sigma uncertainties in the DataFrame regardless. If False, do not include 1-sigma
        uncertainties in the DataFrame.
    covariances : bool, optional
        If None, will check if any of the covariance terms are defined and add them to the datframe.
        If True, include covariance matrices in the DataFrame regardless. Covariance matrices
        will be split into 21 columns, with the lower triangular elements stored. If False, do not
        include covariance matrices in the DataFrame.

    Returns
    -------
    df : `~pandas.Dataframe`
        DataFrame containing coordinates and covariances.
    """
    # Gather times and put them into a dataframe
    df_times = coords.time.to_dataframe(flatten=True)

    if coords.frame == "ecliptic":
        coord_names = [f"{coord}_ec" for coord in coord_names]
    elif coords.frame == "equatorial":
        coord_names = [f"{coord}_eq" for coord in coord_names]
    else:
        raise ValueError(f"Frame {coords.frame} not recognized.")

    df_coords = pd.DataFrame(
        data=coords.values,
        columns=coord_names,
    )

    # Gather the origin and put it into a dataframe
    df_origin = coords.origin.to_dataframe(flatten=True)
    origin_dict = {col: f"origin.{col}" for col in df_origin.columns}
    df_origin.rename(columns=origin_dict, inplace=True)

    if covariances is None:
        covariances = np.all(~np.isnan(coords.covariance.to_matrix()))

    if sigmas is None:
        sigmas = np.all(~np.isnan(coords.covariance.sigmas))

    # Gather the covariances and put them into a dataframe
    if covariances or sigmas:
        df_cov = coords.covariance.to_dataframe(
            coord_names=coord_names,
            sigmas=sigmas,
        )
        if not covariances:
            cov_cols = [col for col in df_cov.columns if "cov_" in col]
            df_cov = df_cov.drop(columns=cov_cols)

    # Join the dataframes
    df = pd.concat([df_times, df_coords, df_origin], axis=1)
    if covariances or sigmas:
        df = df.join(df_cov)

    return df


def coords_from_dataframe(
    cls: "Type[CoordinateType]",
    df: pd.DataFrame,
    coord_names: List[str],
) -> "CoordinateType":
    """
    Return coordinates from a pandas DataFrame that was generated with
    `~adam_core.coordinates.io.coords_to_dataframe` (or any coordinate type's
    `.to_dataframe` method).

    Parameters
    ----------
    df : `~pandas.Dataframe`
        DataFrame containing coordinates and covariances.
    coord_names : list of str
        Names of the coordinate dimensions. The coordinate reference frame will be appended to the
        coordinate names, e.g., "x" will become "x_ec" for ecliptic coordinates and "x_eq" for
        equatorial coordinates.

    Returns
    -------
    coords : {CartesianCoordinates, CometaryCoordinates, KeplerianCoordinates, SphericalCoordinates}
        Coordinates read from the DataFrame.

    Raises
    ------
    ValueError : If the DataFrame does not contain the expected columns for the times of the coordinates.
    """
    times = Times.from_dataframe(df)

    origin = Origin.from_dataframe(
        df[["origin.code"]].rename(columns={"origin.code": "code"})
    )
    covariances = CoordinateCovariances.from_dataframe(df, coord_names=coord_names)

    coord_names_ec = [f"{coord}_ec" for coord in coord_names]
    coord_names_eq = [f"{coord}_eq" for coord in coord_names]
    if all(col in df.columns for col in coord_names_ec):
        coords = {
            col: df[col_frame].values
            for col, col_frame in zip(coord_names, coord_names_ec)
        }
        frame = "ecliptic"
    elif all(col in df.columns for col in coord_names_eq):
        coords = {
            col: df[col_frame].values
            for col, col_frame in zip(coord_names, coord_names_eq)
        }
        frame = "equatorial"
    else:
        raise ValueError(
            "DataFrame does not contain the expected columns for the coordinate values:\n"
            f"Expected: {coord_names_ec} or {coord_names_eq}\n"
        )

    return cls.from_kwargs(
        **coords, time=times, origin=origin, frame=frame, covariance=covariances
    )
