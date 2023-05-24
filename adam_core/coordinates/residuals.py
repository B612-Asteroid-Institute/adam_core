from typing import Union

import numpy as np

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
SUPPORTED_COORDINATES = (
    CartesianCoordinates,
    CometaryCoordinates,
    KeplerianCoordinates,
    SphericalCoordinates,
)

__all__ = ["calculate_chi2"]


def calculate_chi2(residuals: np.ndarray, covariances: np.ndarray) -> np.ndarray:
    """
    Calculate the vectorized normalized squared residual for each residual and covariance pair.
    This normalized residual is equivalent to the Mahalanobis distance squared.
    For residuals with no covariance (all non-diagonal covariance elements are zero) this
    is exactly equivalent to chi2.

    Parameters
    ----------
    residuals : `~numpy.ndarray` (N, D)
        Array of N residuals in D dimensions (observed - predicted).
    covariances : `~numpy.ndarray` (N, D, D)
        Array of N covariance matrices in D dimensions.

    Returns
    -------
    chi2 : `~numpy.ndarray` (N)
        Array of chi2 values for each residual and covariance pair.

    References
    ----------
    [1] Carpino, M. et al. (2003). Error statistics of asteroid optical astrometric observations.
        Icarus, 166, 248-270. https://doi.org/10.1016/S0019-1035(03)00051-4
    """
    W = np.linalg.inv(covariances)
    chi2 = np.einsum("ij,ji->i", np.einsum("ij,ijk->ik", residuals, W), residuals.T)
    return chi2
