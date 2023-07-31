import logging
from typing import Callable, List

import numpy as np
import pandas as pd
import pyarrow as pa
from astropy import units as u
from astropy.table import Table as AstropyTable
from quivr import FixedSizeListColumn, Table
from scipy.stats import multivariate_normal

from .jacobian import calc_jacobian

logger = logging.getLogger(__name__)

__all__ = [
    "CoordinateCovariances",
    "sigmas_to_covariances",
    "sample_covariance",
    "transform_covariances_sampling",
    "transform_covariances_jacobian",
    "sigmas_to_df",
    "sigmas_from_df",
    "covariances_to_df",
    "covariances_from_df",
    "covariances_to_table",
]

COVARIANCE_FILL_VALUE = np.NaN


def sigmas_to_covariances(sigmas: np.ndarray) -> np.ndarray:
    """
    Convert an array of standard deviations to an array of covariance matrices.
    Non-diagonal elements are set to zero.

    Parameters
    ----------
    sigmas : `numpy.ndarray` (N, D)
        Standard deviations for N coordinates in D dimensions.

    Returns
    -------
    covariances : `numpy.ndarray` (N, D, D)
        Covariance matrices for N coordinates in D dimensions.
    """
    D = sigmas.shape[1]
    identity = np.identity(D, dtype=sigmas.dtype)
    covariances = np.einsum("kj,ji->kij", sigmas**2, identity, order="C")
    return covariances


class CoordinateCovariances(Table):
    # TODO: Would be interesting if the dimensionality can be generalized
    #      to D dimensions, so (N, D, D) instead of (N, 6, 6). We would be
    #      able to use this class for the covariance matrices of different
    #      measurments like projections (D = 4) and photometry (D = 1).

    # This is temporary while we await the implementation of
    # https://github.com/apache/arrow/issues/35599
    values = FixedSizeListColumn(pa.float64(), list_size=36)
    # When fixed, we should revert to:
    # values = Column(pa.fixed_shape_tensor(pa.float64(), (6, 6)))

    @property
    def sigmas(self):
        cov_diag = np.diagonal(self.to_matrix(), axis1=1, axis2=2)
        sigmas = np.sqrt(cov_diag)
        return sigmas

    def to_matrix(self) -> np.ndarray:
        """
        Return the covariance matrices as a 3D array of shape (N, 6, 6).

        Returns
        -------
        covariances : `numpy.ndarray` (N, 6, 6)
            Covariance matrices for N coordinates in 6 dimensions.
        """
        # return self.values.combine_chunks().to_numpy_ndarray()
        cov = np.stack(self.values.to_numpy(zero_copy_only=False))
        if np.all(cov == None):  # noqa: E711
            return np.full((len(self), 6, 6), np.nan)
        else:
            cov = np.stack(cov).reshape(-1, 6, 6)
        return cov

    @classmethod
    def from_matrix(cls, covariances: np.ndarray) -> "CoordinateCovariances":
        """
        Create a Covariances object from a 3D array of covariance matrices.

        Parameters
        ----------
        covariances : `numpy.ndarray` (N, 6, 6)
            Covariance matrices for N coordinates in 6 dimensions.

        Returns
        -------
        covariances : `Covariances`
            Covariance matrices for N coordinates in 6 dimensions.
        """
        # cov = pa.FixedShapeTensorArray.from_numpy_ndarray(covariances)
        cov = covariances.reshape(-1, 36)
        return cls.from_kwargs(values=list(cov))

    @classmethod
    def from_sigmas(cls, sigmas: np.ndarray) -> "CoordinateCovariances":
        """
        Create a Covariances object from a 2D array of sigmas.

        Parameters
        ----------
        sigmas : `numpy.ndarray` (N, 6)
            Array of 1-sigma uncertainties for N coordinates in 6
            dimensions.

        Returns
        -------
        covariances : `Covariances`
            Covariance matrices with the diagonal elements set to the
            squares of the input sigmas.
        """
        return cls.from_matrix(sigmas_to_covariances(sigmas))

    def to_dataframe(
        self,
        coord_names: List[str] = ["x", "y", "z", "vx", "vy", "vz"],
        sigmas: bool = False,
    ) -> pd.DataFrame:
        """
        Return the covariance matrices represented as lower triangular columns in a pandas DataFrame.

        Parameters
        ----------
        coord_names : `list` of `str`, optional
            Names of the coordinate axes. Default is ["x", "y", "z", "vx", "vy", "vz"].
        sigmas : `bool`, optional
            If True, the standard deviations are added as columns to the DataFrame. Default is False.

        Returns
        -------
        df : `pandas.DataFrame`
            Covariance matrices (lower triangular) for N coordinates in 6 dimensions.
        """
        df = covariances_to_df(self.to_matrix(), coord_names=coord_names, kind="lower")
        if sigmas:
            df_sigmas = sigmas_to_df(self.sigmas, coord_names=coord_names)
            df = df_sigmas.join(df)

        return df

    @classmethod
    def from_dataframe(
        cls, df, coord_names: List[str] = ["x", "y", "z", "vx", "vy", "vz"]
    ) -> "CoordinateCovariances":
        """
        Create a Covariances object from a pandas DataFrame.

        Parameters
        ----------
        df : `pandas.DataFrame`
            Covariance matrices (lower triangular) for N coordinates in 6 dimensions.
        coord_names : `list` of `str`, optional
            Names of the coordinate axes. Default is ["x", "y", "z", "vx", "vy", "vz"].

        Returns
        -------
        covariances : `CoordinateCovariances`
            Covariance matrices for N coordinates in 6 dimensions.
        """
        try:
            covariances = covariances_from_df(df, coord_names=coord_names, kind="lower")
        except KeyError:
            sigmas = sigmas_from_df(df, coord_names=coord_names)
            covariances = sigmas_to_covariances(sigmas)

        return cls.from_matrix(covariances)

    def is_all_nan(self) -> bool:
        """
        Check if all covariance matrix values are NaN.

        Returns
        -------
        is_all_nan : bool
            True if all covariance matrix elements are NaN, False otherwise.
        """
        return np.all(np.isnan(self.to_matrix()))


def sample_covariance(
    mean: np.ndarray, cov: np.ndarray, num_samples: int = 100000
) -> np.ndarray:
    """
    Sample a multivariate Gaussian distribution with given
    mean and covariances.

    Parameters
    ----------
    mean : `~numpy.ndarray` (D)
        Multivariate mean of the Gaussian distribution.
    cov : `~numpy.ndarray` (D, D)
        Multivariate variance-covariance matrix of the Gaussian distribution.
    num_samples : int, optional
        Number of samples to draw from the distribution.

    Returns
    -------
    samples : `~numpy.ndarray` (num_samples, D)
        The drawn samples row-by-row.
    """
    normal = multivariate_normal(mean=mean, cov=cov, allow_singular=True)
    samples = normal.rvs(num_samples)
    return samples


def transform_covariances_sampling(
    coords: np.ndarray,
    covariances: np.ndarray,
    func: Callable,
    num_samples: int = 100000,
) -> np.ma.masked_array:
    """
    Transform covariance matrices by sampling the transformation function.

    Parameters
    ----------
    coords : `~numpy.ndarray` (N, D)
        Coordinates that correspond to the input covariance matrices.
    covariances : `~numpy.ndarray` (N, D, D)
        Covariance matrices to transform via sampling.
    func : function
        A function that takes coords (N, D) as input and returns the transformed
        coordinates (N, D). See for example: `thor.coordinates.cartesian_to_spherical`
        or `thor.coordinates.cartesian_to_keplerian`.
    num_samples : int, optional
        The number of samples to draw.

    Returns
    -------
    covariances_out : `~numpy.ndarray` (N, D, D)
        Transformed covariance matrices.
    """
    covariances_out = []
    for coord, covariance in zip(coords, covariances):
        samples = sample_covariance(coord, covariance, num_samples)
        samples_converted = func(samples)
        covariances_out.append(np.cov(samples_converted.T))

    covariances_out = np.stack(covariances_out)
    return covariances_out


def transform_covariances_jacobian(
    coords: np.ndarray,
    covariances: np.ndarray,
    _func: Callable,
    **kwargs,
) -> np.ndarray:
    """
    Transform covariance matrices by calculating the Jacobian of the transformation function
    using `~jax.jacfwd`.

    Parameters
    ----------
    coords : `~numpy.ndarray` (N, D)
        Coordinates that correspond to the input covariance matrices.
    covariances : `~numpy.ndarray` (N, D, D)
        Covariance matrices to transform via numerical differentiation.
    _func : function
        A function that takes a single coord (D) as input and return the transformed
        coordinate (D). See for example: `thor.coordinates._cartesian_to_spherical`
        or `thor.coordinates._cartesian_to_keplerian`.

    Returns
    -------
    covariances_out : `~numpy.ndarray` (N, D, D)
        Transformed covariance matrices.
    """
    jacobian = calc_jacobian(coords, _func, **kwargs)
    covariances = jacobian @ covariances @ np.transpose(jacobian, axes=(0, 2, 1))
    return covariances


def sigmas_to_df(
    sigmas: np.ndarray,
    coord_names: List[str] = ["x", "y", "z", "vx", "vy", "vz"],
) -> pd.DataFrame:
    """
    Place sigmas into a `pandas.DataFrame`.

    Parameters
    ----------
    sigmas : `~numpy.ndarray` (N, D)
        1-sigma uncertainty values for each coordinate dimension D.
    coord_names : List[str]
        Names of the coordinate columns, will be used to determine column names for
        each element in the triangular covariance matrix.

    Returns
    -------
    df : `~pandas.DataFrame`
        DataFrame containing covariances in either upper or lower triangular
        form.
    """
    N, D = sigmas.shape

    data = {}
    for i in range(D):
        data[f"sigma_{coord_names[i]}"] = sigmas[:, i]

    return pd.DataFrame(data)


def sigmas_from_df(
    df: pd.DataFrame,
    coord_names: List[str] = ["x", "y", "z", "vx", "vy", "vz"],
) -> np.ndarray:
    """
    Read sigmas from a `~pandas.DataFrame`.

    Parameters
    ----------
    df : `~pandas.DataFrame`
        DataFrame containing covariances in either upper or lower triangular
        form.
    coord_names : List[str]
        Names of the coordinate columns, will be used to determine column names for
        each element in the triangular covariance matrix.

    Returns
    -------
    sigmas : `~numpy.ndarray` (N, D)
        1-sigma uncertainty values for each coordinate dimension D.
    """
    N = len(df)
    D = len(coord_names)
    sigmas = np.zeros((N, D), dtype=np.float64)
    sigmas.fill(COVARIANCE_FILL_VALUE)

    for i in range(D):
        try:
            sigmas[:, i] = df[f"sigma_{coord_names[i]}"].values

        except KeyError:
            logger.debug(f"No sigma column found for dimension {coord_names[i]}.")

    return sigmas


def covariances_to_df(
    covariances: np.ndarray,
    coord_names: List[str] = ["x", "y", "z", "vx", "vy", "vz"],
    kind: str = "lower",
) -> pd.DataFrame:
    """
    Place covariance matrices into a `pandas.DataFrame`. Splits the covariance matrices
    into either upper or lower triangular form and then adds one column per dimension.

    Parameters
    ----------
    covariances : `~numpy.ndarray` (N, D, D)
        3D array of covariance matrices.
    coord_names : List[str]
        Names of the coordinate columns, will be used to determine column names for
        each element in the triangular covariance matrix.
    kind : {'upper', 'lower'}
        The orientation of the triangular representation.

    Returns
    -------
    df : `~pandas.DataFrame`
        DataFrame containing covariances in either upper or lower triangular
        form.

    Raises
    ------
    ValueError : If kind is not one of {'upper', 'lower'}

    """
    N, D, D = covariances.shape

    if kind == "upper":
        ii, jj = np.triu_indices(D)
    elif kind == "lower":
        ii, jj = np.tril_indices(D)
    else:
        err = "kind should be one of {'upper', 'lower'}"
        raise ValueError(err)

    data = {}
    for i, j in zip(ii, jj):
        data[f"cov_{coord_names[i]}_{coord_names[j]}"] = covariances[:, i, j]

    return pd.DataFrame(data)


def covariances_from_df(
    df: pd.DataFrame,
    coord_names: List[str] = ["x", "y", "z", "vx", "vy", "vz"],
    kind: str = "lower",
) -> np.ndarray:
    """
    Read covariance matrices from a `~pandas.DataFrame`.

    Parameters
    ----------
    df : `~pandas.DataFrame`
        DataFrame containing covariances in either upper or lower triangular
        form.
    coord_names : List[str]
        Names of the coordinate columns, will be used to determine column names for
        each element in the triangular covariance matrix.
    kind : {'upper', 'lower'}
        The orientation of the triangular representation.

    Returns
    -------
    covariances : `~numpy.ndarray` (N, D, D)
        3D array of covariance matrices.

    Raises
    ------
    ValueError : If kind is not one of {'upper', 'lower'}
    """
    N = len(df)
    D = len(coord_names)
    covariances = np.zeros((N, D, D), dtype=np.float64)
    covariances.fill(COVARIANCE_FILL_VALUE)

    if kind == "upper":
        ii, jj = np.triu_indices(D)
    elif kind == "lower":
        ii, jj = np.tril_indices(D)
    else:
        err = "kind should be one of {'upper', 'lower'}"
        raise ValueError(err)

    for i, j in zip(ii, jj):
        try:
            covariances[:, i, j] = df[f"cov_{coord_names[i]}_{coord_names[j]}"].values
            covariances[:, j, i] = covariances[:, i, j]
        except KeyError:
            logger.debug(
                "No covariance column found for dimensions"
                f" {coord_names[i]},{coord_names[j]}."
            )

    return covariances


def covariances_to_table(
    covariances: np.ma.masked_array,
    coord_names: List[str] = ["x", "y", "z", "vx", "vy", "vz"],
    coord_units=[u.au, u.au, u.au, u.au / u.d, u.au / u.d, u.au / u.d],
    kind: str = "lower",
) -> AstropyTable:
    """
    Place covariance matrices into a `astropy.table.table.Table`. Splits the covariance matrices
    into either upper or lower triangular form and then adds one column per dimension.

    Parameters
    ----------
    covariances : `~numpy.ma.masked_array` (N, D, D)
        3D array of covariance matrices.
    coord_names : List[str]
        Names of the coordinate columns, will be used to determine column names for
        each element in the triangular covariance matrix.
    coord_units : List[]
        The unit for each coordinate, will be used to determination the units for
        element in the triangular covariance matrix.
    kind : {'upper', 'lower'}
        The orientation of the triangular representation.

    Returns
    -------
    table : `~astropy.table.table.Table`
        Table containing covariances in either upper or lower triangular
        form.

    Raises
    ------
    ValueError : If kind is not one of {'upper', 'lower'}
    """
    N, D, D = covariances.shape

    if kind == "upper":
        ii, jj = np.triu_indices(D)
    elif kind == "lower":
        ii, jj = np.tril_indices(D)
    else:
        err = "kind should be one of {'upper', 'lower'}"
        raise ValueError(err)

    data = {}
    for i, j in zip(ii, jj):
        data[f"cov_{coord_names[i]}_{coord_names[j]}"] = (
            covariances[:, i, j] * coord_units[i] * coord_units[j]
        )

    return AstropyTable(data)


def covariances_from_table(
    table: AstropyTable,
    coord_names: List[str] = ["x", "y", "z", "vx", "vy", "vz"],
    kind: str = "lower",
) -> np.ma.masked_array:
    """
    Read covariance matrices from a `~astropy.table.table.Table`.

    Parameters
    ----------
    table : `~astropy.table.table.Table`
        Table containing covariances in either upper or lower triangular
        form.
    coord_names : List[str]
        Names of the coordinate columns, will be used to determine column names for
        each element in the triangular covariance matrix.
    kind : {'upper', 'lower'}
        The orientation of the triangular representation.

    Returns
    -------
    covariances : `~numpy.ma.masked_array` (N, D, D)
        3D array of covariance matrices.

    Raises
    ------
    ValueError : If kind is not one of {'upper', 'lower'}
    """
    N = len(table)
    D = len(coord_names)
    covariances = np.ma.zeros((N, D, D), dtype=np.float64)
    covariances.fill_value = COVARIANCE_FILL_VALUE
    covariances.mask = np.ones((N, D, D), dtype=bool)

    if kind == "upper":
        ii, jj = np.triu_indices(D)
    elif kind == "lower":
        ii, jj = np.tril_indices(D)
    else:
        err = "kind should be one of {'upper', 'lower'}"
        raise ValueError(err)

    for i, j in zip(ii, jj):
        try:
            covariances[:, i, j] = table[
                f"cov_{coord_names[i]}_{coord_names[j]}"
            ].values
            covariances[:, j, i] = covariances[:, i, j]
        except KeyError:
            logger.debug(
                "No covariance column found for dimensions"
                f" {coord_names[i]},{coord_names[j]}."
            )

    return covariances
