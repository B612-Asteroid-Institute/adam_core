import warnings
from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import quivr as qv
from scipy import stats

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

__all__ = [
    "Residuals",
    "calculate_chi2",
    "_batch_coords_and_covariances",
]


def apply_cosine_latitude_correction(
    lat: npt.NDArray[np.float64],
    residuals: npt.NDArray[np.float64],
    covariances: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Apply a correction factor of cosine latitude to the residuals and covariance matrix.
    This is designed to account for the fact that longitudes get smaller as you approach
    the poles.

    Parameters
    ----------
    lat : `~numpy.ndarray` (N)
        Latitudes in degrees.
    residuals : `~numpy.ndarray` (N, D)
        Spherical residuals.
    covariances : `~numpy.ndarray` (N, D, D)
        Covariance matrices for spherical residuals.

    Returns
    -------
    residuals : `~numpy.ndarray` (N, D)
        Spherical residuals with the correction factor applied.
    covariances : `~numpy.ndarray` (N, D, D)
        Covariance matrices for spherical residuals with the correction factor applied.
    """
    N = len(lat)
    cos_lat = np.cos(np.radians(lat))

    identity = np.identity(6, dtype=np.float64)

    # Populate the diagonal of the matrix with cos(latitude) for
    # the longitude and longitudinal velocity dimensions
    diag = np.ones((N, 6))
    diag[:, 1] = cos_lat
    diag[:, 4] = cos_lat

    # Calculate the cos(latitude) factor for the covariance matrix
    cos_lat_cov = np.einsum("kj,ji->kij", diag, identity, order="C")

    # Apply the cos(latitude) factor to the residuals in longitude
    # and longitudinal velocity
    residuals[:, 1] *= cos_lat
    residuals[:, 4] *= cos_lat

    # Identify locations where the covariance matrix is NaN and set
    # those values to 0.0
    nan_cov = np.isnan(covariances)
    covariances_masked = np.where(nan_cov, 0.0, covariances)

    # Apply the cos(latitude) factor to the covariance matrix.
    covariances_masked = (
        cos_lat_cov @ covariances_masked @ cos_lat_cov.transpose(0, 2, 1)
    )

    # Set the covariance matrix to nan where it was nan before.
    # Note that when we apply cos(latitude) to the covariance matrix in the previous statement,
    # we are only modifying the longitude and longitudinal velocity dimensions. The remaining
    # dimensions are left unchanged which is why we can use a mask to reset the
    # the missing values in the resulting matrix to NaN.
    covariances = np.where(nan_cov, np.nan, covariances_masked)
    return residuals, covariances


def bound_longitude_residuals(
    observed: npt.NDArray[np.float64], residuals: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Bound the longitude residuals to the range [-180, 180] degrees and adjust the
    signs of the residuals that cross the 0/360 degree boundary. By convention, we define
    residuals that cross from <360 to >0 as a positive residual (355 - 5 = +10) and cross
    from >0 to <360 as a negative residual (5 - 355 = -10).

    Parameters
    ----------
    observed : `~numpy.ndarray` (N, D)
        Observed coordinates in degrees. We use the observed coordinates to
        determine whether the residuals cross the 0/360 degree boundary.
    residuals : `~numpy.ndarray` (N, D)
        Residuals in degrees.

    Returns
    -------
    residuals : `~numpy.ndarray` (N, D)
        Residuals wrapped to the range [-180, 180] degrees.
    """
    # Extract the observed longitude and residuals
    observed_longitude = observed[:, 1]
    longitude_residual = residuals[:, 1]

    # Identify residuals that exceed a half circle
    longitude_residual_g180 = longitude_residual > 180
    longitude_residual_l180 = longitude_residual < -180

    # Wrap the residuals to the range [-180, 180] degrees
    longitude_residual = np.where(
        longitude_residual_g180, longitude_residual - 360, longitude_residual
    )
    longitude_residual = np.where(
        longitude_residual_l180, longitude_residual + 360, longitude_residual
    )

    # Adjust the signs of the residuals that cross the 0/360 degree boundary
    # We want it so that crossing from <360 to >0 is a positive residual (355 - 5 = +10)
    # and crossing from >0 to <360 is a negative residual (5 - 355 = -10)
    longitude_residual = np.where(
        longitude_residual_g180 & (observed_longitude > 180),
        -longitude_residual,
        longitude_residual,
    )
    longitude_residual = np.where(
        longitude_residual_l180 & (observed_longitude < 180),
        -longitude_residual,
        longitude_residual,
    )

    residuals[:, 1] = longitude_residual
    return residuals


class Residuals(qv.Table):

    values = qv.ListColumn(pa.float64(), nullable=True)
    chi2 = qv.Float64Column(nullable=True)
    dof = qv.Int64Column(nullable=True)
    probability = qv.Float64Column(nullable=True)

    @classmethod
    def calculate(
        cls, observed: CoordinateType, predicted: CoordinateType
    ) -> "Residuals":
        """
        Calculate the residuals between the observed and predicted coordinates. Residuals
        are defined as the observed coordinates minus the predicted coordinates. The observed
        coordinate's covariance matrix is used to calculate the chi2 and degrees of freedom.

        TODO: We may want to add support for a single predicted coordinate and covariance
            compared to multiple observed coordinates and covariances.
            Add support for cases where both the observed and predicted coordinates have
            covariances. Maybe a Kolmogorov-Smirnov test?

        Parameters
        ----------
        observed : CoordinateType (N, D)
            Observed coordinates.
        predicted : CoordinateType (N, D) or (1, D)
            Predicted coordinates. If a single coordinate is provided, it is broadcasted
            to the same shape as the observed coordinates.

        Returns
        -------
        residuals : `~adam_core.coordinates.residuals.Residuals`
            Residuals between the observed and predicted coordinates.
        """
        if not isinstance(observed, SUPPORTED_COORDINATES):
            raise TypeError(
                f"Observed coordinates must be one of {SUPPORTED_COORDINATES}, not {type(observed)}."
            )
        if not isinstance(predicted, SUPPORTED_COORDINATES):
            raise TypeError(
                f"Predicted coordinates must be one of {SUPPORTED_COORDINATES}, not {type(predicted)}."
            )
        if type(observed) is not type(predicted):
            raise TypeError(
                "Observed and predicted coordinates must be the same type, "
                f"not {type(observed)} and {type(predicted)}."
            )
        if (observed.origin != predicted.origin).all():
            raise ValueError(
                "Observed and predicted coordinates must have the same origin."
            )
        if observed.frame != predicted.frame:
            raise ValueError(
                f"Observed ({observed.frame}) and predicted ({predicted.frame}) coordinates must have the same frame."
            )
        # Extract the observed and predicted values
        observed_values = observed.values
        observed_covariances = observed.covariance.to_matrix()
        predicted_values = predicted.values

        # Create the output arrays
        N, D = observed_values.shape
        p = np.empty(N, dtype=np.float64)
        chi2s = np.empty(N, dtype=np.float64)

        # Calculate the degrees of freedom for every coordinate
        # Number of coordinate dimensions less the number of quantities that are NaN
        dof = D - np.sum(np.isnan(observed_values), axis=1)

        # Calculate the array of residuals
        residuals = observed_values - predicted_values

        # If the coordinates are spherical then we do some extra work:
        # 1. Bound residuals in longitude to the range [-180, 180] degrees and
        #    adjust the signs of the residuals that cross the 0/360 degree boundary.
        # 2. Apply the cos(latitude) factor to the residuals in longitude and longitudinal
        #    velocity
        if isinstance(observed, SphericalCoordinates):
            # Bound the longitude residuals to the range [-180, 180] degrees
            residuals = bound_longitude_residuals(observed_values, residuals)

            # Apply the cos(latitude) factor to the residuals and covariance matrix
            residuals, observed_covariances = apply_cosine_latitude_correction(
                observed_values[:, 2], residuals, observed_covariances
            )

        # Batch the coordinates and covariances into groups that have the same
        # number of dimensions that have missing values (represented by NaNs)
        (
            batch_indices,
            batch_dimensions,
            batch_coords,
            batch_covariances,
        ) = _batch_coords_and_covariances(observed_values, observed_covariances)

        for indices, dimensions, coords, covariances in zip(
            batch_indices, batch_dimensions, batch_coords, batch_covariances
        ):
            if not np.all(np.isnan(covariances)):
                # Filter residuals by dimensions that have values
                residuals_i = residuals[:, dimensions]

                # Then filter by rows that belong to this batch
                residuals_i = residuals_i[indices, :]

                # Calculate the chi2 for each coordinate (this is actually
                # calculating mahalanobis distance squared -- both are equivalent
                # when the covariance matrix is diagonal, mahalanobis distance is more
                # general as it allows for covariates between dimensions)
                chi2_values = calculate_chi2(residuals_i, covariances)

                # For a normally distributed random variable, the mahalanobis distance
                # squared in D dimesions follows a chi2 distribution with D degrees of freedom.
                # So for each coordinate, calculate the probability that you would
                # get a chi2 value greater than or equal to that coordinate's chi2 value.
                # At a residual of zero this probability is 1.0, and at a residual of
                # 1 sigma (for 1 degree of freedom) this probability is ~0.3173.
                p[indices] = 1 - stats.chi2.cdf(chi2_values, dof[indices])

                # Set the chi2 for each coordinate
                chi2s[indices] = chi2_values

            else:
                # If the covariance matrix is all NaNs, then the chi2 is NaN
                chi2s[indices] = np.nan
                p[indices] = np.nan

        return cls.from_kwargs(
            values=residuals.tolist(),
            chi2=chi2s,
            dof=dof,
            probability=p,
        )

    def to_array(self) -> npt.NDArray[np.float64]:
        """
        Convert the residuals to a numpy array.

        Returns
        -------
        residuals : `~numpy.ndarray` (N, D)
            Array of residuals.
        """
        return np.stack(self.values.to_numpy(zero_copy_only=False))


def calculate_chi2(
    residuals: npt.NDArray[np.float64], covariances: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Calculate the vectorized normalized squared residual for each residual and covariance pair.
    This normalized residual is equivalent to the Mahalanobis distance squared.
    For residuals with no covariance (all non-diagonal covariance elements are zero) this
    is exactly equivalent to chi2.

    If the off-diagonal covariance elements (the covariate terms) are missing (represented by NaNs),
    then they will assumed to be 0.0.

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
    # Raise error if any of the diagonal elements are nan
    if np.any(np.isnan(np.diagonal(covariances, axis1=1, axis2=2))):
        raise ValueError("Covariance matrix has NaNs on the diagonal.")

    # Warn if any of the non-diagonal elements are nan
    if np.any(np.isnan(covariances)):
        warnings.warn(
            "Covariance matrix has NaNs on the off-diagonal (these will be assumed to be 0.0).",
            UserWarning,
        )
        covariances = np.where(np.isnan(covariances), 0.0, covariances)

    W = np.linalg.inv(covariances)
    chi2 = np.einsum("ij,ji->i", np.einsum("ij,ijk->ik", residuals, W), residuals.T)
    return chi2


def calculate_reduced_chi2(residuals: Residuals, parameters: int) -> float:
    """
    Calculate the reduced chi2 for a set of residuals.

    Parameters
    ----------
    residuals : `~adam_core.coordinates.residuals.Residuals`
        Residuals.
    parameters : int
        Number of parameters in the model.

    Returns
    -------
    reduced_chi2 : float
        Reduced chi2.
    """
    chi2_total = residuals.chi2.to_numpy().sum()
    dof_total = residuals.dof.to_numpy().sum() - parameters
    return chi2_total / dof_total


def _batch_coords_and_covariances(
    coords: npt.NDArray[np.float64], covariances: npt.NDArray[np.float64]
) -> Tuple[
    List[npt.NDArray[np.float64]],
    List[npt.NDArray[np.float64]],
    List[npt.NDArray[np.float64]],
    List[npt.NDArray[np.float64]],
]:
    """
    Batch coordinates and covariances into groups that have the same
    number of dimensions that have missing values (represented by NaNs).

    Parameters
    ----------
    coords : `~numpy.ndarray` (N, D)
        Array of N coordinates in D dimensions.
    covariances : `~numpy.ndarray` (N, D, D)
        Array of N covariance matrices in D dimensions.

    Returns
    -------
    batch_indices : List[`~numpy.ndarray` (<=N)]
        List of arrays of indices into the original arrays that correspond to
        coordinates with the same number of dimensions that have missing values.
    batch_dimensions : List[`~numpy.ndarray` (<=D)]
        List of arrays of dimensions that have values for each batch.
    batch_coords : List[`~numpy.ndarray` (<=N, <=D)]
        Array of N coordinates in D dimensions with missing dimensions removed.
    batch_covariances : List[`~numpy.ndarray` (<=N, <=D, <=D)]
        Array of N covariance matrices in D dimensions with missing dimensions removed.
    """
    N, D = coords.shape
    assert covariances.shape == (N, D, D)

    # Find the indices of the coordinates that have the same missing dimensions
    # (if any) and select the coordinates and covariances for those indices
    nans = np.isnan(coords)
    batch_masks = np.unique(nans, axis=0)
    indices = np.arange(0, N, dtype=np.int64)

    batch_indices: List[np.ndarray] = []
    batch_dimensions: List[np.ndarray] = []
    batch_coords: List[np.ndarray] = []
    batch_covariances: List[np.ndarray] = []
    for batch in batch_masks:
        # Find the indices of the coordinates that have the same missing dimensions
        # and select the coordinates and covariances for those indices
        batch_mask_i = indices[np.where(np.all(nans == batch, axis=1))[0]]
        coords_i = coords[batch_mask_i]
        covariances_i = covariances[batch_mask_i]

        # Remove the missing dimensions from the coordinates and covariances
        dimensions_with_values = np.where(~batch)[0]
        coords_i = coords_i[:, dimensions_with_values]
        covariances_i = covariances_i[:, dimensions_with_values, :]
        covariances_i = covariances_i[:, :, dimensions_with_values]

        batch_indices.append(batch_mask_i)
        batch_dimensions.append(dimensions_with_values)
        batch_coords.append(coords_i)
        batch_covariances.append(covariances_i)

    return batch_indices, batch_dimensions, batch_coords, batch_covariances
