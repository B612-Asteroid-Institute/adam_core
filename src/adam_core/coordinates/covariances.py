import logging
from typing import Callable, Optional, Tuple

import numpy as np
import pyarrow as pa
import quivr as qv
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal

from .jacobian import calc_jacobian

logger = logging.getLogger(__name__)

COVARIANCE_FILL_VALUE = np.nan


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


class CoordinateCovariances(qv.Table):
    # TODO: Would be interesting if the dimensionality can be generalized
    #      to D dimensions, so (N, D, D) instead of (N, 6, 6). We would be
    #      able to use this class for the covariance matrices of different
    #      measurments like projections (D = 4) and photometry (D = 1).

    values = qv.LargeListColumn(pa.float64(), nullable=True)
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
        values = self.values.to_numpy(zero_copy_only=False)

        # Try and see if all covariance matrices are None, if so return
        # an array of NaNs.
        try:
            if np.all(values == None):  # noqa: E711
                return np.full((len(self), 6, 6), np.nan)

        except ValueError as e:
            err_str = (
                "The truth value of an array with more than one element is ambiguous."
                " Use a.any() or a.all()"
            )
            if str(e) != err_str:
                raise e

        # Try to stack the values into a 3D array. If this works, then
        # all covariance matrices are the same size and we can return
        # the stacked matrices.
        try:
            cov = np.stack(values).reshape(-1, 6, 6)

        # If not then some of the arrays might be None. Lets loop through
        # the values and fill in the arrays that are missing (None) with NaNs.
        except ValueError as e:
            # If we don't get the error we expect, then raise it.
            if str(e) != "all input arrays must have the same shape":
                raise e
            else:
                for i in range(len(values)):
                    if values[i] is None:
                        values[i] = np.full(36, np.nan)

            # Try stacking again
            cov = np.stack(values).reshape(-1, 6, 6)

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

        Raises
        ------
        ValueError : If the covariance matrices are not (N, 6, 6)
        """
        # cov = pa.FixedShapeTensorArray.from_numpy_ndarray(covariances)
        if covariances.shape[1:] != (6, 6):
            raise ValueError(
                f"Covariance matrices should have shape (N, 6, 6) but got {covariances.shape}"
            )
        cov = covariances.flatten()
        offsets = np.arange(0, (len(covariances) + 1) * 36, 36, dtype=np.int64)
        return cls.from_kwargs(values=pa.LargeListArray.from_arrays(offsets, cov))

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

    @classmethod
    def nulls(cls, length: int) -> "CoordinateCovariances":
        """
        Create a Covariances object with all covariance matrix elements set to NaN.
        Parameters
        ----------
        length : `int`
            Number of coordinates.

        Returns
        -------
        covariances : `CoordinateCovariances`
            Covariance matrices for N coordinates in 6 dimensions.
        """
        return cls.from_kwargs(
            values=pa.ListArray.from_arrays(
                pa.array(np.arange(0, 36 * (length + 1), 36)),
                pa.nulls(36 * length, pa.float64()),
            )
        )

    def is_all_nan(self) -> bool:
        """
        Check if all covariance matrix values are NaN.

        Returns
        -------
        is_all_nan : bool
            True if all covariance matrix elements are NaN, False otherwise.
        """
        return np.all(np.isnan(self.to_matrix()))


def make_positive_semidefinite(
    cov: np.ndarray, semidef_tol: float = 1e-15
) -> np.ndarray:
    """
    Adjust a covariance matrix that is non positive semidefinite
    within a given tolerance, by flipping the sign of the negative
    eigenvalues. This can occur when the covariance matrix inludes
    values that are close to zero, which results in very small
    negative numbers.

    Parameters
    ----------
    cov : `~numpy.ndarray` (D, D)
        Covariance matrix to adjust.
    tol : float, optional
        Tolerance for eigenvalues close to zero.

    Returns
    -------
    cov_psd : `~numpy.ndarray` (D, D)
        Positive semidefinite covariance matrix.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if np.any(eigenvalues < -semidef_tol):
        raise ValueError(
            f"Covariance matrix is not positive semidefinite, {eigenvalues} above the tolerance of: {semidef_tol}"
        )
    mask = (eigenvalues < 0) & (np.abs(eigenvalues) < semidef_tol)
    eigenvalues[mask] = -eigenvalues[mask]
    cov_psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return cov_psd


def sample_covariance_random(
    mean: np.ndarray,
    cov: np.ndarray,
    num_samples: int = 10000,
    seed: Optional[int] = None,
    semidef_tol: Optional[float] = 1e-15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample a multivariate Gaussian distribution with given
    mean and covariances.

    The returned weights will be equal to 1 / num_samples so that
    each sample is equally weighted. Weights are returned from this function
    so its interface is identical to that of
    `~adam_core.coordinates.covariances.sample_covariance_sigma_points`.

    Parameters
    ----------
    mean : `~numpy.ndarray` (D)
        Multivariate mean of the Gaussian distribution.
    cov : `~numpy.ndarray` (D, D)
        Multivariate variance-covariance matrix of the Gaussian distribution.
    num_samples : int, optional
        Number of samples to draw from the distribution.
    seed : int, optional
        Seed for the random number generator.
    semidef_tol : float, optional
        Tolerance for eigenvalues close to zero.

    Returns
    -------
    samples : `~numpy.ndarray` (num_samples, D)
        The drawn samples row-by-row.
    W: `~numpy.ndarray` (num_samples)
        Weights of the samples.
    W_cov: `~numpy.ndarray` (num_samples)
        Weights of the samples to reconstruct covariance matrix.
    Raises
    ------
    ValueError : If the covariance matrix is not positive semidefinite, within the given tolerance.
    """
    # Check if the covariance matrix is non positive semidefinite. This is usually a sign
    # that something has gone wrong with the covariance. However, when the negative eigenvalues
    # are very close to zero, they can be flipped to positive without an issue. This is due to
    # the way the covariance matrix is calculated.
    if np.any(np.linalg.eigvals(cov) < 0):
        if np.any(np.linalg.eigvals(cov) < -1 * semidef_tol):
            raise ValueError(
                f"Covariance matrix is not positive semidefinite, below the tolerance of: {semidef_tol}"
            )
        else:
            logger.warning(
                "Covariance matrix is not positive semidefinite, but within tolerance, adjusting..."
            )
            cov = make_positive_semidefinite(cov)
    normal = multivariate_normal(mean=mean, cov=cov, allow_singular=True, seed=seed)
    samples = normal.rvs(num_samples)
    W = np.full(num_samples, 1 / num_samples)
    W_cov = np.full(num_samples, 1 / num_samples)
    return samples, W, W_cov


def sample_covariance_sigma_points(
    mean: np.ndarray,
    cov: np.ndarray,
    alpha: float = 1,
    beta: float = 0.0,
    kappa: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sigma-point samples of a multivariate Gaussian distribution
    with given mean and covariances.

    Parameters
    ----------
    mean : `~numpy.ndarray` (D)
        Multivariate mean of the Gaussian distribution.
    cov : `~numpy.ndarray` (D, D)
        Multivariate variance-covariance matrix of the Gaussian distribution.
    alpha : float, optional
        Spread of the sigma points between 1e^-2 and 1.
    beta : float, optional
        Prior knowledge of the distribution usually set to 2 for a Gaussian.
    kappa : float, optional
        Secondary scaling parameter usually set to 0.

    Returns
    -------
    sigma_points : `~numpy.ndarray` (2 * D + 1, D)
        Sigma points drawn from the distribution.
    W: `~numpy.ndarray` (2 * D + 1)
        Weights of the sigma points.
    W_cov: `~numpy.ndarray` (2 * D + 1)
        Weights of the sigma points to reconstruct covariance matrix.

    References
    ----------
    [1] Wan, E. A; Van Der Merwe, R. (2000). The unscented Kalman filter for nonlinear estimation.
        Proceedings of the IEEE 2000 Adaptive Systems for Signal Processing,
        Communications, and Control Symposium, 153-158.
        https://doi.org/10.1109/ASSPCC.2000.882463
    """
    # Calculate the dimensionality of the distribution
    D = mean.shape[0]

    # See equation 15 in Wan & Van Der Merwe (2000) [1]
    N = 2 * D + 1
    sigma_points = np.empty((N, D))
    W = np.empty(N)
    W_cov = np.empty(N)

    # Calculate the scaling parameter lambda
    lambd = alpha**2 * (D + kappa) - D

    # First sigma point is the mean
    sigma_points[0] = mean

    # Beta is used to encode prior knowledge about the distribution.
    # If the distribution is a well-constrained Gaussian, beta = 2 is optimal
    # but lets set beta to 0 for now which has the effect of not weighting the mean state
    # with 0 for the covariance matrix. This is generally better for more distributions.
    # Calculate the weights for mean and the covariance matrix
    # Weight are used to reconstruct the mean and covariance matrix from the sigma points
    W[0] = lambd / (D + lambd)
    W_cov[0] = W[0] + (1 - alpha**2 + beta)

    # Take the matrix square root of the scaled covariance matrix.
    # Sometimes you'll see this done with a Cholesky decomposition for speed
    # but sqrtm is sufficiently optimized for this use case and typically provides
    # better results
    L = sqrtm((D + lambd) * cov)

    # Calculate the remaining sigma points
    for i in range(D):
        offset = L[i]
        sigma_points[i + 1] = mean + offset
        sigma_points[i + 1 + D] = mean - offset

    # The weights for the remaining sigma points are the same
    # for the mean and the covariance matrix
    W[1:] = 1 / (2 * (D + lambd))
    W_cov[1:] = 1 / (2 * (D + lambd))

    return sigma_points, W, W_cov


def weighted_mean(samples: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Calculate the weighted mean of a set of samples.

    Parameters
    ----------
    samples : `~numpy.ndarray` (N, D)
        Samples drawn from the distribution (these can be randomly drawn
        or sigma points)
    W: `~numpy.ndarray` (N)
        Weights of the samples.

    Returns
    -------
    mean : `~numpy.ndarray` (D)
        Mean calculated from the samples and weights.
    """
    return np.dot(W, samples)


def weighted_covariance(
    mean: np.ndarray, samples: np.ndarray, W_cov: np.ndarray
) -> np.ndarray:
    """
    Calculate a covariance matrix from samples and their corresponding weights.

    Parameters
    ----------
    mean : `~numpy.ndarray` (D)
        Mean calculated from the samples and weights.
        See `~adam_core.coordinates.covariances.weighted_mean`.
    samples : `~numpy.ndarray` (N, D)
        Samples drawn from the distribution (these can be randomly drawn
        or sigma points)
    W_cov: `~numpy.ndarray` (N)
        Weights of the samples to reconstruct covariance matrix.

    Returns
    -------
    cov : `~numpy.ndarray` (D, D)
        Covariance matrix calculated from the samples and weights.
    """
    # Calculate the covariance matrix from the sigma points and weights
    # `~numpy.cov` does not support negative weights so we will calculate
    # the covariance manually
    # cov = np.cov(samples, aweights=W_cov, rowvar=False, bias=True)
    residual = samples - mean
    cov = (W_cov * residual.T) @ residual
    return cov


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
        samples, W, W_cov = sample_covariance_random(coord, covariance, num_samples)
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
