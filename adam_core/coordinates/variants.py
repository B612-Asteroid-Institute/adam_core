from typing import Literal, Tuple, Union

import numpy as np
import quivr as qv

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

CoordinateType = Union[
    CartesianCoordinates,
    KeplerianCoordinates,
    CometaryCoordinates,
    SphericalCoordinates,
]


def create_coordinate_variants(
    coordinates: CoordinateType, method: Literal["auto", "sigma-point", "monte-carlo"]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, CoordinateType]:
    """
    Sample and create variants for the given coordinates by sampling the covariance matrices.
    There are three supported methods:
    - sigma-point: Sample the covariance matrix using sigma points. This is the fastest method,
      but can be inaccurate if the covariance matrix is not well behaved.
    - monte-carlo: Sample the covariance matrix using a monte carlo method.
      This is the slowest method, but is the most accurate.
    - auto: Automatically select the best method based on the covariance matrix.
      If the covariance matrix is well behaved then sigma-point sampling will be used.
      If the covariance matrix is not well behaved then monte-carlo sampling will be used.

    When sampling with monte-carlo, 10k samples are drawn. Sigma-point sampling draws 13 samples
    for 6-dimensional coordinates.

    TODO:
        This function does not yet handle sampling of covariances and coordinates with missing values.

    Parameters
    ----------
    coordinates : {'~adam_core.coordinates.cartesian.CartesianCoordinates',
                   '~adam_core.coordinates.keplerian.KeplerianCoordinates',
                   '~adam_core.coordinates.cometary.CometaryCoordinates',
                   '~adam_core.coordinates.spherical.SphericalCoordinates'}
        The coordinates to sample.
    method : {'sigma-point', 'monte-carlo', 'auto'}, optional
        The method to use for sampling the covariance matrix. If 'auto' is selected then the method
        will be automatically selected based on the covariance matrix. The default is 'auto'.

    Returns
    -------
    idx : '~numpy.ndarray'
        The index of the coordinate that each sample belongs to.
    weights : '~numpy.ndarray'
        Weights of each sample.
    cov_weights : '~numpy.ndarray'
        Weights of the samples to reconstruct covariance matrix.
    samples : {'~adam_core.coordinates.cartesian.CartesianCoordinates',
                     '~adam_core.coordinates.keplerian.KeplerianCoordinates',
                     '~adam_core.coordinates.cometary.CometaryCoordinates',
                     '~adam_core.coordinates.spherical.SphericalCoordinates'}
        The samples drawn from the coordinate covariance matrices.

    Raises
    ------
    ValueError:
        If the covariance matrices are all undefined.
        If the input coordinates are not supported.
    """
    idx_list = []
    samples_list = []
    weights_list = []
    cov_weights_list = []
    origins_list = []
    times_list = []

    if coordinates.covariance.is_all_nan():
        raise ValueError(
            "Cannot sample coordinate covariances when covariances are all undefined."
        )

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
            samples, W, W_cov = sample_covariance_sigma_points(mean, cov)

        elif method == "monte-carlo":
            samples, W, W_cov = sample_covariance_random(mean, cov, 10000)

        elif method == "auto":
            # Sample with sigma points
            samples, W, W_cov = sample_covariance_sigma_points(mean, cov)

            # Check if the sigma point sampling is good enough by seeing if we can
            # recover the mean and covariance from the sigma points
            mean_sg = weighted_mean(samples, W)
            cov_sg = weighted_covariance(mean_sg, samples, W_cov)

            # If the sigma point sampling is not good enough, then sample with monte carlo
            # Though it is not guaranteed that monte carlo will actually be better
            diff_mean = np.abs(mean_sg - mean)
            diff_cov = np.abs(cov_sg - cov)
            if np.any(diff_mean >= 1e-12) or np.any(diff_cov >= 1e-12):
                samples, W, W_cov = sample_covariance_random(mean, cov, 10000)

        else:
            raise ValueError(f"Unknown coordinate covariance sampling method: {method}")

        origins_list += [coordinate_i.origin for i in range(len(samples))]
        times_list += [coordinate_i.time for i in range(len(samples))]
        samples_list.append(samples)
        weights_list.append(W)
        cov_weights_list.append(W_cov)
        idx_list.append(np.full(len(samples), i))

    samples = np.concatenate(samples_list)
    idx = np.concatenate(idx_list)
    weights = np.concatenate(weights_list)
    cov_weights = np.concatenate(cov_weights_list)
    origins = qv.concatenate(origins_list)
    times = qv.concatenate(times_list)

    if isinstance(coordinates, CartesianCoordinates):
        return (
            idx,
            weights,
            cov_weights,
            CartesianCoordinates.from_kwargs(
                x=samples[:, 0],
                y=samples[:, 1],
                z=samples[:, 2],
                vx=samples[:, 3],
                vy=samples[:, 4],
                vz=samples[:, 5],
                time=times,
                covariance=None,
                origin=origins,
                frame=coordinates.frame,
            ),
        )
    elif isinstance(coordinates, SphericalCoordinates):
        return (
            idx,
            weights,
            cov_weights,
            SphericalCoordinates.from_kwargs(
                rho=samples[:, 0],
                lon=samples[:, 1],
                lat=samples[:, 2],
                vrho=samples[:, 3],
                vlon=samples[:, 4],
                vlat=samples[:, 5],
                time=times,
                covariance=None,
                origin=origins,
                frame=coordinates.frame,
            ),
        )
    elif isinstance(coordinates, KeplerianCoordinates):
        return (
            idx,
            weights,
            cov_weights,
            KeplerianCoordinates.from_kwargs(
                a=samples[:, 0],
                e=samples[:, 1],
                i=samples[:, 2],
                raan=samples[:, 3],
                ap=samples[:, 4],
                M=samples[:, 5],
                time=times,
                covariance=None,
                origin=origins,
                frame=coordinates.frame,
            ),
        )
    elif isinstance(coordinates, CometaryCoordinates):
        return (
            idx,
            weights,
            cov_weights,
            CometaryCoordinates.from_kwargs(
                q=samples[:, 0],
                e=samples[:, 1],
                i=samples[:, 2],
                raan=samples[:, 3],
                ap=samples[:, 4],
                tp=samples[:, 5],
                time=times,
                covariance=None,
                origin=origins,
                frame=coordinates.frame,
            ),
        )

    else:
        raise ValueError(f"Unsupported coordinate type: {type(coordinates)}")
