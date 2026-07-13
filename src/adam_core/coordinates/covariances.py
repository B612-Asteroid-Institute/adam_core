import logging
from typing import Callable, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal

from .jacobian import calc_jacobian

logger = logging.getLogger(__name__)

COVARIANCE_FILL_VALUE = np.nan

# Dimensionality of the coordinate (orbital) block of a covariance matrix.
COORD_DIM = 6
# Orbits fitted with non-gravitational parameters extend the covariance with
# the fixed trailing dimensions (A1, A2, A3), giving a 9x9 matrix.
FULL_DIM = 9


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
    # Each row holds either a 6x6 coordinate covariance (36 values) or, for
    # orbits fitted with non-gravitational parameters, a 9x9 covariance
    # (81 values) over the fixed basis (x, y, z, vx, vy, vz, A1, A2, A3) --
    # or the equivalent first-six elements of the current representation.
    # Non-gravitational parameters that were not estimated carry zero
    # rows/columns (held fixed); rows without a non-gravitational solution
    # store only the 6x6 block.

    values = qv.LargeListColumn(pa.float64(), nullable=True)

    @property
    def sigmas(self):
        cov_diag = np.diagonal(self.to_matrix(), axis1=1, axis2=2)
        sigmas = np.sqrt(cov_diag)
        return sigmas

    def nongrav_block_mask(self) -> npt.NDArray[np.bool_]:
        """
        Return a per-row boolean mask that is True where the covariance
        carries the trailing non-gravitational (A1, A2, A3) block.
        """
        lengths = pc.list_value_length(self.values)
        return (
            pc.fill_null(pc.equal(lengths, FULL_DIM * FULL_DIM), False)
            .to_numpy(zero_copy_only=False)
            .astype(bool)
        )

    def has_nongrav_block(self) -> bool:
        """
        Return True if any row carries the non-gravitational (A1, A2, A3) block.
        """
        return bool(self.nongrav_block_mask().any())

    def to_matrix(self) -> np.ndarray:
        """
        Return the coordinate block of the covariance matrices as a 3D array
        of shape (N, 6, 6). For rows carrying the non-gravitational block,
        this is the leading 6x6 block; use `to_full_matrix` to retrieve the
        full 9x9 matrices.

        Returns
        -------
        covariances : `numpy.ndarray` (N, 6, 6)
            Covariance matrices for N coordinates in 6 dimensions.
        """
        if self.has_nongrav_block():
            return self.to_full_matrix()[:, :COORD_DIM, :COORD_DIM]

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

    def to_full_matrix(self) -> np.ndarray:
        """
        Return the covariance matrices as a 3D array of shape (N, 9, 9) over
        the basis (coordinates, A1, A2, A3). Rows without a non-gravitational
        block have their trailing rows and columns filled with NaN.

        Returns
        -------
        covariances : `numpy.ndarray` (N, 9, 9)
            Covariance matrices for N coordinates plus the three
            non-gravitational parameters.
        """
        full = np.full((len(self), FULL_DIM, FULL_DIM), np.nan)
        if not self.has_nongrav_block():
            full[:, :COORD_DIM, :COORD_DIM] = self.to_matrix()
            return full

        values = self.values.to_numpy(zero_copy_only=False)
        for i, value in enumerate(values):
            if value is None:
                continue
            value = np.asarray(value, dtype=np.float64)
            if value.size == COORD_DIM * COORD_DIM:
                full[i, :COORD_DIM, :COORD_DIM] = value.reshape(COORD_DIM, COORD_DIM)
            elif value.size == FULL_DIM * FULL_DIM:
                full[i] = value.reshape(FULL_DIM, FULL_DIM)
            else:
                raise ValueError(
                    f"Covariance row {i} has {value.size} values; expected "
                    f"{COORD_DIM * COORD_DIM} or {FULL_DIM * FULL_DIM}."
                )
        return full

    def to_transform_matrix(self) -> np.ndarray:
        """
        Return the covariance matrices in the widest layout present:
        (N, 6, 6) when no row carries the non-gravitational block, otherwise
        (N, 9, 9). Intended for feeding `transform_covariances_jacobian` or
        `apply_linear_covariance_transform`, which carry the
        non-gravitational block through the transform when it is present.
        """
        if self.has_nongrav_block():
            return self.to_full_matrix()
        return self.to_matrix()

    @classmethod
    def from_matrix(cls, covariances: np.ndarray) -> "CoordinateCovariances":
        """
        Create a Covariances object from a 3D array of covariance matrices.

        Parameters
        ----------
        covariances : `numpy.ndarray` (N, 6, 6) or (N, 9, 9)
            Covariance matrices for N coordinates in 6 dimensions, or in
            6 dimensions plus the non-gravitational parameters (A1, A2, A3).
            For (N, 9, 9) input, rows whose trailing non-gravitational rows
            and columns are all NaN are stored as plain 6x6 covariances.

        Returns
        -------
        covariances : `Covariances`
            Covariance matrices for N coordinates.

        Raises
        ------
        ValueError : If the covariance matrices are not (N, 6, 6) or (N, 9, 9)
        """
        if covariances.shape[1:] == (COORD_DIM, COORD_DIM):
            cov = covariances.flatten()
            offsets = np.arange(0, (len(covariances) + 1) * 36, 36, dtype=np.int64)
            return cls.from_kwargs(values=pa.LargeListArray.from_arrays(offsets, cov))

        if covariances.shape[1:] != (FULL_DIM, FULL_DIM):
            raise ValueError(
                "Covariance matrices should have shape (N, 6, 6) or (N, 9, 9) "
                f"but got {covariances.shape}"
            )

        rows = []
        offsets = [0]
        for cov in covariances:
            if (
                np.isnan(cov[COORD_DIM:, :]).all()
                and np.isnan(cov[:, COORD_DIM:]).all()
            ):
                rows.append(cov[:COORD_DIM, :COORD_DIM].reshape(-1))
            else:
                rows.append(np.asarray(cov, dtype=np.float64).reshape(-1))
            offsets.append(offsets[-1] + rows[-1].size)
        flat = np.concatenate(rows) if rows else np.empty(0, dtype=np.float64)
        return cls.from_kwargs(
            values=pa.LargeListArray.from_arrays(
                pa.array(offsets, type=pa.int64()),
                pa.array(flat, type=pa.float64()),
            )
        )

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


def apply_linear_covariance_transform(
    transform_matrices: np.ndarray,
    covariances: np.ndarray,
) -> np.ndarray:
    """
    Apply a linear 6x6 coordinate transform to covariance matrices.

    For (N, 6, 6) covariances this is the usual similarity transform. For
    (N, 9, 9) covariances the coordinate block is transformed, the
    coordinate/non-gravitational cross-covariances are rotated with it, and
    the non-gravitational (A1, A2, A3) block is preserved unchanged --
    equivalent to a block-diagonal transform with an identity block on the
    non-gravitational dimensions.

    Parameters
    ----------
    transform_matrices : `~numpy.ndarray` (6, 6) or (N, 6, 6)
        Linear transform(s) applied to the coordinate block.
    covariances : `~numpy.ndarray` (N, 6, 6) or (N, 9, 9)
        Covariance matrices to transform.

    Returns
    -------
    covariances_out : `~numpy.ndarray`
        Transformed covariance matrices with the same shape as the input.
    """
    matrices = np.asarray(transform_matrices, dtype=np.float64)
    if matrices.shape == (COORD_DIM, COORD_DIM):
        matrices = matrices.reshape(1, COORD_DIM, COORD_DIM)
    if matrices.ndim != 3 or matrices.shape[1:] != (COORD_DIM, COORD_DIM):
        raise ValueError(
            "transform_matrices must have shape (6, 6) or (N, 6, 6), "
            f"got {matrices.shape}"
        )
    if len(matrices) not in (1, len(covariances)):
        raise ValueError(
            "Number of transform matrices must be 1 or match the number of covariances."
        )
    matrices_T = np.transpose(matrices, axes=(0, 2, 1))

    if covariances.shape[1:] == (COORD_DIM, COORD_DIM):
        return matrices @ covariances @ matrices_T

    if covariances.shape[1:] != (FULL_DIM, FULL_DIM):
        raise ValueError(
            f"Covariance matrices should have shape (N, 6, 6) or (N, 9, 9), "
            f"got {covariances.shape}"
        )

    out = np.empty_like(covariances)
    out[:, :COORD_DIM, :COORD_DIM] = (
        matrices @ covariances[:, :COORD_DIM, :COORD_DIM] @ matrices_T
    )
    out[:, :COORD_DIM, COORD_DIM:] = matrices @ covariances[:, :COORD_DIM, COORD_DIM:]
    out[:, COORD_DIM:, :COORD_DIM] = np.transpose(
        out[:, :COORD_DIM, COORD_DIM:], axes=(0, 2, 1)
    )
    out[:, COORD_DIM:, COORD_DIM:] = covariances[:, COORD_DIM:, COORD_DIM:]
    return out


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
    coords : `~numpy.ndarray` (N, 6)
        Coordinates that correspond to the input covariance matrices.
    covariances : `~numpy.ndarray` (N, 6, 6) or (N, 9, 9)
        Covariance matrices to transform via numerical differentiation.
        For (N, 9, 9) input the trailing non-gravitational dimensions
        (A1, A2, A3) are carried through an identity block: the cross
        covariances rotate with the coordinate Jacobian and the
        non-gravitational block is preserved.
    _func : function
        A function that takes a single coord (6) as input and return the transformed
        coordinate (6). See for example: `thor.coordinates._cartesian_to_spherical`
        or `thor.coordinates._cartesian_to_keplerian`.

    Returns
    -------
    covariances_out : `~numpy.ndarray`
        Transformed covariance matrices with the same shape as the input.
    """
    jacobian = calc_jacobian(coords, _func, **kwargs)
    return apply_linear_covariance_transform(jacobian, covariances)


def _upper_triangular_to_full(
    upper_triangular: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Convert an upper triangular matrix containing 21 elements to a full 6x6 matrix.
    """
    assert (
        len(upper_triangular) == 21
    ), "Upper triangular matrix must be a 21 element vector"

    full = np.zeros((6, 6))
    full[np.triu_indices(6)] = upper_triangular
    full[np.tril_indices(6, -1)] = full.T[np.tril_indices(6, -1)]
    return full


def _lower_triangular_to_full(
    lower_triangular: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Convert a lower triangular matrix containing 21 elements to a full 6x6 matrix.
    """
    assert (
        len(lower_triangular) == 21
    ), "Lower triangular matrix must be a 21 element vector"

    full = np.zeros((6, 6))
    full[np.tril_indices(6)] = lower_triangular
    full[np.triu_indices(6, -1)] = full.T[np.triu_indices(6, -1)]
    return full
