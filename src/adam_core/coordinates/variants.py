from __future__ import annotations

from typing import Generic, Literal, Optional, Protocol, TypeVar

import numpy as np
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

    if coordinates.covariance.is_all_nan():
        raise ValueError(
            "Cannot sample coordinate covariances when covariances are all undefined."
        )

    class VariantCoordinates(qv.Table):
        index = qv.Int64Column()
        sample = coordinates.as_column()
        weight = qv.Float64Column()
        weight_cov = qv.Float64Column()

    if isinstance(coordinates, CartesianCoordinates):
        dimensions = ["x", "y", "z", "vx", "vy", "vz"]
    elif isinstance(coordinates, SphericalCoordinates):
        dimensions = ["rho", "lon", "lat", "vrho", "vlon", "vlat"]
    elif isinstance(coordinates, KeplerianCoordinates):
        dimensions = ["a", "e", "i", "raan", "ap", "M"]
    elif isinstance(coordinates, CometaryCoordinates):
        dimensions = ["q", "e", "i", "raan", "ap", "tp"]
    else:
        raise ValueError(f"Unsupported coordinate type: {type(coordinates)}")

    variants_list = []
    for i, coordinate_i in enumerate(coordinates):

        mean = coordinate_i.values[0]
        cov = coordinate_i.covariance.to_matrix()[0]

        if np.any(np.isnan(cov)):
            raise ValueError(
                "Cannot sample coordinate covariances when some covariance elements are undefined."
            )
        if np.any(np.isnan(mean)):
            raise ValueError(
                "Cannot sample coordinate covariances when some coordinate dimensions are undefined."
            )

        if method == "sigma-point":
            samples, W, W_cov = sample_covariance_sigma_points(
                mean, cov, alpha=alpha, beta=beta, kappa=kappa
            )

        elif method == "monte-carlo":
            samples, W, W_cov = sample_covariance_random(
                mean, cov, num_samples=num_samples, seed=seed
            )

        elif method == "auto":
            # Sample with sigma points
            samples, W, W_cov = sample_covariance_sigma_points(
                mean, cov, alpha=alpha, beta=beta, kappa=kappa
            )

            # Check if the sigma point sampling is good enough by seeing if we can
            # recover the mean and covariance from the sigma points
            mean_sg = weighted_mean(samples, W)
            cov_sg = weighted_covariance(mean_sg, samples, W_cov)

            # If the sigma point sampling is not good enough, then sample with monte carlo
            # Though it is not guaranteed that monte carlo will actually be better
            diff_mean = np.abs(mean_sg - mean)
            diff_cov = np.abs(cov_sg - cov)
            if np.any(diff_mean >= 1e-12) or np.any(diff_cov >= 1e-12):
                samples, W, W_cov = sample_covariance_random(
                    mean, cov, num_samples=num_samples
                )

        else:
            raise ValueError(f"Unknown coordinate covariance sampling method: {method}")

        variants_list.append(
            VariantCoordinates.from_kwargs(
                index=np.full(len(samples), i),
                sample=coordinates.from_kwargs(
                    origin=qv.concatenate(
                        [coordinate_i.origin for i in range(len(samples))]
                    ),
                    time=qv.concatenate(
                        [coordinate_i.time for i in range(len(samples))]
                    ),
                    frame=coordinate_i.frame,
                    **{dim: samples[:, i] for i, dim in enumerate(dimensions)},
                ),
                weight=W,
                weight_cov=W_cov,
            )
        )

    return qv.concatenate(variants_list)
