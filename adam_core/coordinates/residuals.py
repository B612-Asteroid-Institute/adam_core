from typing import List, Tuple, Union

import numpy as np
import pyarrow as pa
from quivr import Float64Column, Int64Column, ListColumn, Table
from scipy.stats import chi2

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


class Residuals(Table):

    values = ListColumn(pa.float64(), nullable=True)
    chi2 = Float64Column(nullable=True)
    dof = Int64Column(nullable=True)
    probability = Float64Column(nullable=True)

    @classmethod
    def calculate(
        cls, observed: CoordinateType, predicted: CoordinateType
    ) -> "Residuals":
        """
        Calculate the residuals between the observed and predicted coordinates. The observed
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
        if type(observed) != type(predicted):
            raise TypeError(
                "Observed and predicted coordinates must be the same type, "
                f"not {type(observed)} and {type(predicted)}."
            )

        N, D = observed.values.shape
        p = np.empty(N, dtype=np.float64)
        chi2s = np.empty(N, dtype=np.float64)

        # Caclulate the degrees of freedom for every coordinate
        # Number of coordinate dimensions less the number of quantities that are NaN
        dof = D - np.sum(np.isnan(observed.values), axis=1)

        # Calculate the array of residuals
        residuals = observed.values - predicted.values

        # Batch the coordinates and covariances into groups that have the same
        # number of dimensions that have missing values (represented by NaNs)
        (
            batch_indices,
            batch_dimensions,
            batch_coords,
            batch_covariances,
        ) = _batch_coords_and_covariances(
            observed.values, observed.covariance.to_matrix()
        )

        for indices, dimensions, coords, covariances in zip(
            batch_indices, batch_dimensions, batch_coords, batch_covariances
        ):
            if not np.all(np.isnan(covariances)):
                # Calculate the chi2 for each coordinate
                chi2_values = calculate_chi2(residuals[indices], covariances)

                # Calculate the probability for each coordinate
                p[indices] = 1 - chi2.cdf(np.sqrt(chi2_values), dof[indices])

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


def _batch_coords_and_covariances(
    coords: np.ndarray, covariances: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
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
