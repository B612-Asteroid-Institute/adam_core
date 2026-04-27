import logging
from typing import Callable, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import quivr as qv
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal


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

    def _fast_to_matrix(self) -> Optional[np.ndarray]:
        """
        Return (N, 6, 6) by reshaping the underlying flat buffer when the
        LargeListArray has uniform stride-36 offsets and no nulls. Returns
        None to signal the caller to use the slower stack-based fallback
        when those invariants do not hold (ragged rows, arrow-level nulls,
        or chunked arrays that fail to combine).
        """
        n = len(self)
        arr = self.values
        if hasattr(arr, "combine_chunks"):
            arr = arr.combine_chunks()
        if not hasattr(arr, "offsets") or not hasattr(arr, "values"):
            return None
        if arr.null_count != 0:
            return None
        offsets = np.asarray(arr.offsets)
        if len(offsets) != n + 1 or offsets[0] != 0 or offsets[-1] != n * 36:
            return None
        if n > 0 and not np.array_equal(
            np.diff(offsets), np.full(n, 36, dtype=offsets.dtype)
        ):
            return None
        flat = arr.values.to_numpy(zero_copy_only=False)
        if len(flat) != n * 36:
            return None
        # np.copy() so the caller owns a writable, arrow-independent buffer —
        # matching the existing to_matrix contract (callers routinely mutate
        # the result, e.g. near-zero cleanup in CartesianCoordinates.rotate).
        return np.ascontiguousarray(flat).reshape(n, 6, 6).copy()

    def to_matrix(self) -> np.ndarray:
        """
        Return the covariance matrices as a 3D array of shape (N, 6, 6).

        Returns
        -------
        covariances : `numpy.ndarray` (N, 6, 6)
            Covariance matrices for N coordinates in 6 dimensions.
        """
        # Fast path: LargeListArray with uniform stride-36 offsets and no nulls
        # (the normal shape produced by `from_matrix`). Reshape the underlying
        # flat buffer instead of stacking per-row pyarrow objects.
        fast = self._fast_to_matrix()
        if fast is not None:
            return fast

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

    NOTE: this dispatches to numpy `np.dot`, NOT a rust kernel. Apple
    Accelerate / OpenBLAS GEMV is hand-tuned NEON/AVX SIMD and beats
    every pure-rust loop we tried (faer, hand-rolled rayon, hand-rolled
    serial — see journal 2026-04-27 perf measurements). Rust would need
    sleef-vectorized FMA to compete; deferred until we add SIMD math
    crate. The function exists as a vectored entry point but is BLAS
    underneath.

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

    NOTE: this dispatches to numpy (BLAS GEMM), NOT rust — same reason
    as `weighted_mean` above (BLAS hand-tuned SIMD wins until we add a
    rust SIMD math crate).

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
    residual = samples - mean
    return (W_cov * residual.T) @ residual


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


def rust_covariance_transform(
    coords_values: np.ndarray,
    covariances: np.ndarray,
    representation_in: str,
    representation_out: str,
    *,
    t0: Optional[np.ndarray] = None,
    mu: Optional[np.ndarray] = None,
    a: Optional[float] = None,
    f: Optional[float] = None,
    frame_in: str = "ecliptic",
    frame_out: Optional[str] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Run the Rust forward-mode autodiff covariance transform in a single batched
    pass. Returns ``(coords_out [N, 6], cov_out [N, 6, 6])`` when the Rust
    backend is available, else ``None`` so callers can fall back to the JAX
    `transform_covariances_jacobian` path.

    The kernel evaluates every rep-in -> cartesian(frame_in) -> cartesian(frame_out)
    -> rep-out function as ``Dual<6>`` and reads the propagated covariance as
    ``J @ Sigma @ J^T``. NaN covariance rows pass NaN through (consistent with
    the legacy policy).
    """
    # Local import to avoid a module-level cycle and to keep the JAX fallback
    # working when the native extension is not built.
    from .._rust.api import transform_coordinates_with_covariance_numpy

    coords_values = np.ascontiguousarray(coords_values, dtype=np.float64)
    if coords_values.ndim != 2 or coords_values.shape[1] != 6:
        raise ValueError("coords_values must have shape (N, 6)")
    n = coords_values.shape[0]
    if covariances.shape != (n, 6, 6):
        raise ValueError("covariances must have shape (N, 6, 6)")

    cov_flat = np.ascontiguousarray(
        np.asarray(covariances, dtype=np.float64).reshape(n, 36)
    )
    result = transform_coordinates_with_covariance_numpy(
        coords_values,
        cov_flat,
        representation_in,
        representation_out,
        t0=t0,
        mu=mu,
        a=a,
        f=f,
        max_iter=100,
        tol=1e-15,
        frame_in=frame_in,
        frame_out=frame_out,
    )
    if result is None:
        return None
    coords_out, cov_flat_out = result
    return (
        np.asarray(coords_out, dtype=np.float64),
        np.asarray(cov_flat_out, dtype=np.float64).reshape(n, 6, 6),
    )


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
