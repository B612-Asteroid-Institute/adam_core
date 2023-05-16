import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time

from ..utils import Indexable, times_from_df, times_to_df
from .covariances import (
    COVARIANCE_FILL_VALUE,
    covariances_from_df,
    covariances_to_df,
    sigmas_to_df,
)

logger = logging.getLogger(__name__)

__all__ = [
    "_ingest_covariance",
    "Coordinates",
]

COORD_FILL_VALUE = np.NaN


def _ingest_covariance(
    coords: np.ma.masked_array,
    covariance: Union[np.ndarray, np.ma.masked_array],
) -> np.ma.masked_array:
    """
    Ingest a set of covariance matrices.

    Parameters
    ----------
    coords : `~numpy.ma.masked_array` (N, D)
        Masked array of 6D coordinate measurements with q measurements ingested.
    covariance : `~numpy.ndarray` or `~numpy.ma.masked_array` (N, <=D, <=D)
        Covariance matrices for each coordinate. These matrices may have fewer dimensions
        than D. If so, additional dimensions will be added for each masked or missing coordinate
        dimension.

    Returns
    -------
    covariance : `~numpy.ma.masked_array` (N, D, D)
        Masked array of covariance matrices.

    Raises
    ------
    ValueError
        If not every coordinate has an associated covariance.
        If the number of covariance dimensions does not match
            the number of unmasked or missing coordinate dimensions.
    """
    N, D = coords.shape
    axes = D - np.sum(coords.mask.all(axis=0))

    if covariance.shape[0] != len(coords):
        err = "Every coordinate in coords should have an associated covariance."
        raise ValueError(err)

    if covariance.shape[1] != covariance.shape[2] != axes:
        err = (
            f"Coordinates have {axes} defined dimensions, expected covariance matrix\n"
            f"shapes of (N, {axes}, {axes}."
        )
        raise ValueError(err)

    if isinstance(covariance, np.ma.masked_array) and (
        covariance.shape[1] == covariance.shape[2] == coords.shape[1]
    ):
        return covariance

    if isinstance(covariance, np.ndarray) and (
        covariance.shape[1] == covariance.shape[2] == coords.shape[1]
    ):
        covariance_ = np.ma.masked_array(
            covariance,
            dtype=np.float64,
            fill_value=COVARIANCE_FILL_VALUE,
            mask=np.isnan(covariance),
        )
        return covariance_

    covariance_ = np.ma.zeros((N, D, D), dtype=np.float64)
    covariance_.fill_value = COVARIANCE_FILL_VALUE
    covariance_.mask = np.ones(covariance_.shape, dtype=bool)

    for n in range(len(coords)):
        covariance_[n].mask[coords[n].mask, :] = True
        covariance_[n].mask[:, coords[n].mask] = True
        covariance_[n][~covariance_[n].mask] = covariance[n][
            ~covariance_[n].mask
        ].flatten()

    return covariance_


class Coordinates(Indexable):
    def __init__(
        self,
        covariances: Optional[Union[np.ndarray, np.ma.masked_array, List]] = None,
        sigmas: Optional[Union[tuple, np.ndarray, np.ma.masked_array]] = None,
        times: Optional[Time] = None,
        origin: Union[np.ndarray, str] = "heliocenter",
        frame: str = "ecliptic",
        names: dict = {},
        units: dict = {},
        **kwargs,
    ):
        # Total number of coordinate dimensions passed (D)
        D = len(kwargs.keys())
        if D == 0:
            raise ValueError("No coordinates were passed.")

        # Total number of coordinate measurements passed (N)
        # Lets grab the first coordinate dimension and use that
        # to determine the number of measurements to expect.
        # And error will be raised later if the number of measurements in other
        # coordinate dimensions do not match.
        q = list(kwargs.values())[0]
        q_ = self._convert_to_array(q)
        N = len(q_)

        units_ = {}
        coords = np.ma.zeros(
            (N, D),
            dtype=np.float64,
        )
        coords.fill_value = COORD_FILL_VALUE
        coords.mask = np.ones((N, D), dtype=bool)

        for d, (name, q) in enumerate(kwargs.items()):
            q_ = self._convert_to_array(q)

            # If the coordinate dimension is not the same length as the
            # other coordinate dimensions raise an error.
            if len(q_) != N:
                err = (
                    f"Coordinate dimension {name} has {len(q_)} measurements, "
                    f"expected {N}."
                )
                raise ValueError(err)

            # If the coordinate dimension has a coresponding unit
            # then use that unit. If it does not look for the unit
            # in the units kwarg.
            if isinstance(q_, u.Quantity):
                units_[name] = q_.unit
                q_ = q_.value
            else:
                logger.debug(
                    f"Coordinate dimension {name} does not have a corresponding unit, "
                    f"using unit defined in units kwarg ({units[name]})."
                )
                units_[name] = units[name]

            q_ = np.asarray(q)
            coords[:, d] = q_
            coords.mask[:, d] = np.where(np.isnan(q_), True, False)

        self._values = coords
        if isinstance(times, Time):
            if isinstance(times.value, (int, float)):
                times_ = Time([times.value], scale=times.scale, format=times.format)
            else:
                times_ = times

            if len(self.values) != len(times_):
                err = (
                    f"coordinates (N = {len(self._values)}) and times (N = {len(times_)})"
                    "do not have the same length.\n"
                    "If times are defined, each coordinate must have a corresponding time.\n"
                )
                raise ValueError(err)

            self._times = times_
        else:
            self._times = None

        if isinstance(origin, str):
            self._origin = np.empty(len(self._values), dtype="<U16")
            self._origin.fill(origin)
        elif isinstance(origin, np.ndarray):
            assert len(origin) == len(self._values)
            self._origin = origin
        else:
            err = "Origin should be a str or `~numpy.ndarray`"
            raise TypeError(err)

        if isinstance(frame, str):
            self._frame = frame
        else:
            err = "frame should be a str"
            raise TypeError(err)

        self._frame = frame
        self._names = names
        self._units = units_

        if (
            not isinstance(sigmas, (tuple, np.ndarray, np.ma.masked_array))
            and sigmas is not None
        ):
            err = "sigmas should be one of {None, `~numpy.ndarray`, `~numpy.ma.masked_array`, tuple}"
            raise TypeError(err)

        if covariances is not None:
            if (isinstance(sigmas, tuple) and all(sigmas)) or isinstance(
                sigmas, (np.ndarray, np.ma.masked_array)
            ):
                logger.warning(
                    "Both covariances and sigmas have been given. Sigmas will be ignored "
                    "and the covariance matrices will be used instead."
                )
            self._covariances = _ingest_covariance(coords, covariances)

        elif covariances is None and isinstance(
            sigmas, (tuple, np.ndarray, np.ma.masked_array)
        ):
            if isinstance(sigmas, tuple):
                N = len(self._values)
                sigmas_ = np.zeros((N, D), dtype=float)
                for i, sigma in enumerate(sigmas):
                    if sigma is None:
                        sigmas_[:, i] = np.sqrt(COVARIANCE_FILL_VALUE)
                    else:
                        sigmas_[:, i] = sigma

            else:  # isinstance(sigmas, (np.ndarray, np.ma.masked_array)):
                sigmas_ = sigmas

            # self._covariances = sigmas_to_covariance(sigmas_)

        # Both covariances and sigmas are None
        else:
            N, D = coords.shape
            self._covariances = np.ma.zeros(
                (N, D, D),
                dtype=np.float64,
            )
            self._covariances.fill_value = COVARIANCE_FILL_VALUE
            self._covariances.mask = np.ones((N, D, D), dtype=bool)

        index = np.arange(0, len(self._values), 1)
        Indexable.__init__(self, index)
        return

    @property
    def times(self) -> Time:
        return self._times

    @property
    def values(self) -> np.ma.masked_array:
        return self._values

    @property
    def covariances(self) -> np.ma.masked_array:
        return self._covariances

    @property
    def sigmas(self) -> np.ma.masked_array:
        sigmas = None
        if self._covariances is not None:
            cov_diag = np.diagonal(self._covariances, axis1=1, axis2=2)
            sigmas = np.sqrt(cov_diag)

        return sigmas

    @property
    def origin(self) -> np.ndarray:
        return self._origin

    @property
    def frame(self) -> str:
        return self._frame

    @property
    def names(self) -> Dict[str, str]:
        return self._names

    @property
    def units(self) -> Dict[str, u.Unit]:
        return self._units

    def has_units(self, units: dict) -> bool:
        """
        Check if these coordinate have the given units.

        Parameters
        ----------
        units : dict
            Dictionary containing coordinate dimension names as keys
            and astropy units as values.

        Returns
        -------
        bool :
            True if these coordinates have the given units, False otherwise.
        """
        for dim, unit in self.units.items():
            if units[dim] != unit:
                logger.debug(
                    f"Coordinate dimension {dim} has units in {unit}, not the given units of {units[dim]}."
                )
                return False
        return True

    def to_cartesian(self):
        raise NotImplementedError

    def from_cartesian(cls, cartesian):
        raise NotImplementedError

    def to_df(
        self,
        time_scale: Optional[str] = None,
        sigmas: Optional[bool] = None,
        covariances: Optional[bool] = None,
        origin_col: str = "origin",
        frame_col: str = "frame",
    ) -> pd.DataFrame:
        """
        Represent Coordinates as a `~pandas.DataFrame`.

        Parameters
        ----------
        time_scale : {"tdb", "tt", "utc"}, optional
            Desired timescale of the output MJDs. If None, will default to
            the time scale of the current instance.
        sigmas : bool, optional
            Include 1-sigma uncertainty columns. If None, will determine if
            any uncertainties are present and add them if so.
        covariances : bool, optional
            Include lower triangular covariance matrix columns. If None, will
            determine if any uncertainties are present and add them if so.
        origin_col : str
            Name of the column to store the origin of each coordinate.
        frame_col : str
            Name of the column to store the coordinate frame.

        Returns
        -------
        df : `~pandas.DataFrame`
            `~pandas.DataFrame` containing coordinates and optionally their 1-sigma uncertainties
            and lower triangular covariance matrix elements.
        """
        data = {}
        N, D = self.values.shape

        if self.times is not None:
            if isinstance(time_scale, str):
                time_scale_ = time_scale
            else:
                time_scale_ = self.times.scale
            df = times_to_df(self.times, time_scale=time_scale_)
        else:
            df = pd.DataFrame(index=np.arange(0, len(self)))

        for i, (k, v) in enumerate(self.names.items()):
            data[v] = self.values.filled()[:, i]

        df = df.join(pd.DataFrame(data))

        # If the sigmas are all NaN don't add them unless explicitly requested
        if isinstance(sigmas, bool):
            add_sigmas = sigmas
        elif sigmas is None:
            if np.isnan(self.sigmas.filled()).all():
                add_sigmas = False
            else:
                add_sigmas = True
        else:
            raise TypeError(f"sigmas must be a bool or None, not {type(sigmas)}.")

        if add_sigmas:
            df_sigmas = sigmas_to_df(self.sigmas, coord_names=list(self.names.values()))
            df = df.join(df_sigmas)

        # If the covariances are all NaN don't add them unless
        # explicitly requested
        if isinstance(covariances, bool):
            add_covariances = covariances
        elif covariances is None:
            if np.isnan(self.covariances.filled()).all():
                add_covariances = False
            else:
                add_covariances = True
        else:
            raise TypeError(
                f"covariances must be a bool or None, not {type(covariances)}."
            )

        if add_covariances:
            df_covariances = covariances_to_df(
                self.covariances, list(self.names.values()), kind="lower"
            )
            df = df.join(df_covariances)

        df.insert(len(df.columns), origin_col, self.origin)
        df.insert(len(df.columns), frame_col, self.frame)
        return df

    @staticmethod
    def _dict_from_df(
        df: pd.DataFrame,
        coord_cols: dict,
        origin_col: str = "origin",
        frame_col: str = "frame",
    ) -> dict:
        """
        Create a dictionary from a `pandas.DataFrame`.

        Parameters
        ----------
        df : `~pandas.DataFrame`
            Pandas DataFrame containing coordinates and optionally their
            times and covariances.
        coord_cols : dict
            Dictionary containing the coordinate dimensions as keys and their equivalent columns
            as values. For example,
                coord_cols = {}
                coord_cols["a"] = Column name of semi-major axis values
                coord_cols["e"] = Column name of eccentricity values
                coord_cols["i"] = Column name of inclination values
                coord_cols["raan"] = Column name of longitude of ascending node values
                coord_cols["ap"] = Column name of argument of periapsis values
                coord_cols["M"] = Column name of mean anomaly values
        origin_col : str
            Name of the column containing the origin of each coordinate.
        frame_col : str
            Name of the column containing the coordinate frame.

        Returns
        -------
        data : dict
            Dictionary containing attributes extracted from the given Pandas DataFrame.
        """
        data = {}
        data["times"] = times_from_df(df)
        for i, (k, v) in enumerate(coord_cols.items()):
            if v in df.columns:
                data[k] = df[v].values
            else:
                data[k] = None

        if origin_col in df.columns:
            data["origin"] = df[origin_col].values
        else:
            logger.debug(
                f"origin_col ({origin_col}) has not been found in given dataframe."
            )

        if frame_col in df.columns:
            frame = df[frame_col].values
            unique_frames = np.unique(frame)
            assert len(unique_frames) == 1
            data["frame"] = unique_frames[0]
        else:
            logger.debug(
                f"frame_col ({frame_col}) has not been found in given dataframe."
            )

        # Try to read covariances from the dataframe
        covariances = covariances_from_df(
            df, coord_names=list(coord_cols.keys()), kind="lower"
        )

        # If the covariance matrices are fully masked out then try reading covariances
        # using the standard deviation columns
        # if (
        #     isinstance(covariances, np.ma.masked_array)
        #     and (np.all(covariances.mask) is True)
        # ) or (covariances is None):
        #     sigmas = sigmas_from_df(
        #         df,
        #         coord_names=list(coord_cols.keys()),
        #     )
        #     covariances = sigmas_to_covariance(sigmas)
        #     if isinstance(covariances, np.ma.masked_array) and (
        #         np.all(covariances.mask) is True
        #     ):
        #         covariances = None

        data["covariances"] = covariances
        data["names"] = coord_cols

        return data
