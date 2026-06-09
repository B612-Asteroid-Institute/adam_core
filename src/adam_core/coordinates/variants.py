from __future__ import annotations

from typing import Generic, Literal, Optional, Protocol, TypeVar

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import quivr as qv

from . import types
from .cartesian import CartesianCoordinates
from .cometary import CometaryCoordinates
from .covariances import (
    sample_covariance_random,
    sample_covariance_sigma_points,
    weighted_covariance,
    weighted_mean,
)
from .keplerian import KeplerianCoordinates
from .spherical import SphericalCoordinates

T = TypeVar("T", bound=types.CoordinateType, covariant=True)


class VariantCoordinatesTable(Generic[T], Protocol):
    """
    A protocol for a generic table of variant coordinates.
    """

    @property
    def index(self) -> pa.Int64Array: ...

    @property
    def sample(self) -> T: ...

    @property
    def weight(self) -> pa.lib.DoubleArray: ...

    @property
    def weight_cov(self) -> pa.lib.DoubleArray: ...


def _coordinate_dimensions(coordinates: types.CoordinateType) -> list[str]:
    if isinstance(coordinates, CartesianCoordinates):
        return ["x", "y", "z", "vx", "vy", "vz"]
    if isinstance(coordinates, SphericalCoordinates):
        return ["rho", "lon", "lat", "vrho", "vlon", "vlat"]
    if isinstance(coordinates, KeplerianCoordinates):
        return ["a", "e", "i", "raan", "ap", "M"]
    if isinstance(coordinates, CometaryCoordinates):
        return ["q", "e", "i", "raan", "ap", "tp"]
    raise ValueError(f"Unsupported coordinate type: {type(coordinates)}")


def _sample_coordinate_row(
    mean: npt.NDArray[np.float64],
    cov: npt.NDArray[np.float64],
    method: Literal["auto", "sigma-point", "monte-carlo"],
    num_samples: int,
    alpha: float,
    beta: float,
    kappa: float,
    seed: Optional[int],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if np.any(np.isnan(cov)):
        raise ValueError(
            "Cannot sample coordinate covariances when some covariance elements are undefined."
        )
    if np.any(np.isnan(mean)):
        raise ValueError(
            "Cannot sample coordinate covariances when some coordinate dimensions are undefined."
        )

    if method == "sigma-point":
        return sample_covariance_sigma_points(
            mean, cov, alpha=alpha, beta=beta, kappa=kappa
        )

    if method == "monte-carlo":
        return sample_covariance_random(mean, cov, num_samples=num_samples, seed=seed)

    if method != "auto":
        raise ValueError(f"Unknown coordinate covariance sampling method: {method}")

    # Sample with sigma points.
    samples, weights, weights_cov = sample_covariance_sigma_points(
        mean, cov, alpha=alpha, beta=beta, kappa=kappa
    )

    # Check if the sigma point sampling is good enough by seeing if we can
    # recover the mean and covariance from the sigma points.
    mean_sg = weighted_mean(samples, weights)
    cov_sg = weighted_covariance(mean_sg, samples, weights_cov)

    # If the sigma point sampling is not good enough, then sample with monte carlo.
    # Though it is not guaranteed that monte carlo will actually be better.
    diff_mean = np.abs(mean_sg - mean)
    diff_cov = np.abs(cov_sg - cov)
    if np.any(diff_mean >= 1e-12) or np.any(diff_cov >= 1e-12):
        # Preserve the historical auto-mode behavior: the user-supplied seed is
        # not threaded into the Monte Carlo fallback.
        return sample_covariance_random(mean, cov, num_samples=num_samples)

    return samples, weights, weights_cov


def _sample_coordinate_rows(
    means: npt.NDArray[np.float64],
    covariances: npt.NDArray[np.float64],
    method: Literal["auto", "sigma-point", "monte-carlo"],
    num_samples: int,
    alpha: float,
    beta: float,
    kappa: float,
    seed: Optional[int],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
]:
    samples_blocks: list[npt.NDArray[np.float64]] = []
    weights_blocks: list[npt.NDArray[np.float64]] = []
    weights_cov_blocks: list[npt.NDArray[np.float64]] = []
    index_blocks: list[npt.NDArray[np.int64]] = []

    for row_index, (mean, cov) in enumerate(zip(means, covariances)):
        samples, weights, weights_cov = _sample_coordinate_row(
            mean,
            cov,
            method=method,
            num_samples=num_samples,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
            seed=seed,
        )
        samples_blocks.append(samples)
        weights_blocks.append(weights)
        weights_cov_blocks.append(weights_cov)
        index_blocks.append(np.full(len(samples), row_index, dtype=np.int64))

    return (
        np.concatenate(samples_blocks),
        np.concatenate(weights_blocks),
        np.concatenate(weights_cov_blocks),
        np.concatenate(index_blocks),
    )


def create_coordinate_variants(
    coordinates: types.CoordinateType,
    method: Literal["auto", "sigma-point", "monte-carlo"] = "auto",
    num_samples: int = 10000,
    alpha: float = 1,
    beta: float = 0,
    kappa: float = 0,
    seed: Optional[int] = None,
) -> VariantCoordinatesTable[types.CoordinateType]:
    """
    Sample and create variants for the given coordinates by sampling the covariance matrices.
    There are three supported methods:

    1. sigma-point: Sample the covariance matrix using sigma points. This is the fastest method,
       but can be inaccurate if the covariance matrix is not well behaved.
    2. monte-carlo: Sample the covariance matrix using a monte carlo method.
       This is the slowest method, but is the most accurate.
    3. auto: Automatically select the best method based on the covariance matrix.
       If the covariance matrix is well behaved then sigma-point sampling will be used.
       If the covariance matrix is not well behaved then monte-carlo sampling will be used.

    When sampling with monte-carlo, 10k samples are drawn. Sigma-point sampling draws 13 samples
    for 6-dimensional coordinates.

    .. warning::

        This function does not yet handle sampling of covariances and coordinates with missing values.

    Parameters
    ----------
    coordinates :
        The coordinates to sample.
    method:
        The method to use for sampling the covariance matrix. If 'auto' is selected then the method
        will be automatically selected based on the covariance matrix. The default is 'auto'.
    num_samples : int, optional
        The number of samples to draw when sampling with monte-carlo.
    alpha:
        Spread of the sigma points between 1e^-2 and 1.
    beta:
        Prior knowledge of the distribution when generating sigma points usually set to 2 for a Gaussian.
    kappa:
        Secondary scaling parameter when generating sigma points usually set to 0.


    Returns
    -------
        The variant coordinates.

    Raises
    ------
    ValueError:
        If the covariance matrices are all undefined.
        If the input coordinates are not supported.
    """

    covariances = coordinates.covariance.to_matrix()
    if np.all(np.isnan(covariances)):
        raise ValueError(
            "Cannot sample coordinate covariances when covariances are all undefined."
        )

    class VariantCoordinates(qv.Table):
        index = qv.Int64Column()
        sample = coordinates.as_column()
        weight = qv.Float64Column()
        weight_cov = qv.Float64Column()

    dimensions = _coordinate_dimensions(coordinates)
    means = coordinates.values
    samples, weights, weights_cov, index = _sample_coordinate_rows(
        means,
        covariances,
        method=method,
        num_samples=num_samples,
        alpha=alpha,
        beta=beta,
        kappa=kappa,
        seed=seed,
    )
    take_index = pa.array(index, type=pa.int64())

    return VariantCoordinates.from_kwargs(
        index=index,
        sample=coordinates.from_kwargs(
            origin=coordinates.origin.take(take_index),
            time=coordinates.time.take(take_index),
            frame=coordinates.frame,
            **{dim: samples[:, dim_index] for dim_index, dim in enumerate(dimensions)},
        ),
        weight=weights,
        weight_cov=weights_cov,
    )
