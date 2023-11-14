import numpy as np
import pytest
from scipy.spatial.distance import mahalanobis

from ..cartesian import CartesianCoordinates, CoordinateCovariances
from ..origin import Origin
from ..residuals import (
    Residuals,
    _batch_coords_and_covariances,
    bound_longitude_residuals,
    calculate_chi2,
)


def test_calculate_chi2():
    # Einsums are tricky to follow, so here's a test case
    # with a small number of dimensions and residuals to check that
    # the chi2 calculation is correct.
    residuals = np.array(
        [
            [1, 2, 3],
            [2, 1, 1],
        ]
    )
    covariances = np.array(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ]
    )
    np.testing.assert_allclose(calculate_chi2(residuals, covariances), [14, 6])

    # Test that the chi2 calculation works for a single residual and is
    # exactly equal to chi2 (observed - expected)^2 / sigma^2
    residuals = np.array([[3]])
    covariances = np.array([[[1]]])
    np.testing.assert_allclose(calculate_chi2(residuals, covariances), [9])


def test_calculate_chi2_mahalanobis():
    # Test that the calculate_chi2 is equivalent to the Mahalanobis distance squared
    observed = np.array([[1, 1, 1], [2, 2, 2]])
    predicted = np.array([[0, 0, 0], [0, 0, 0]])

    covariances = np.array(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ]
    )

    mahalanobis_distance = np.zeros(2)
    for i in range(2):
        mahalanobis_distance[i] = mahalanobis(
            observed[i], predicted[i], np.linalg.inv(covariances[i])
        )

    assert np.allclose(
        calculate_chi2(observed - predicted, covariances), mahalanobis_distance**2
    )


def test_batch_coords_and_covariances_single_batch_no_missing_values():
    # Single batch, all dimensions have values
    coords = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
        ]
    )
    covariances = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
            [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
        ]
    )

    (
        batch_indices,
        batch_dimensions,
        batch_coords,
        batch_covariances,
    ) = _batch_coords_and_covariances(coords, covariances)
    assert (
        len(batch_indices)
        == len(batch_dimensions)
        == len(batch_coords)
        == len(batch_covariances)
        == 1
    )
    np.testing.assert_equal(batch_indices[0], np.array([0, 1]))
    np.testing.assert_equal(batch_dimensions[0], np.array([0, 1, 2]))
    np.testing.assert_equal(batch_coords[0], coords)
    np.testing.assert_equal(batch_covariances[0], covariances)


def test_batch_coords_and_covariances_single_batch_missing_values():
    # Single batch, one dimension has no values
    coords = np.array(
        [
            [1.0, np.NaN, 3.0],
            [2.0, np.NaN, 4.0],
        ]
    )
    covariances = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
            [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
        ]
    )

    (
        batch_indices,
        batch_dimensions,
        batch_coords,
        batch_covariances,
    ) = _batch_coords_and_covariances(coords, covariances)
    assert (
        len(batch_indices)
        == len(batch_dimensions)
        == len(batch_coords)
        == len(batch_covariances)
        == 1
    )
    np.testing.assert_equal(batch_indices[0], np.array([0, 1]))
    np.testing.assert_equal(batch_dimensions[0], np.array([0, 2]))
    np.testing.assert_equal(batch_coords[0], np.array([[1.0, 3.0], [2.0, 4.0]]))
    np.testing.assert_equal(
        batch_covariances[0],
        np.array([[[1.0, 0.0], [0.0, 3.0]], [[2.0, 0.0], [0.0, 4.0]]]),
    )

    # Single batch, two dimensions have no values
    coords = np.array(
        [
            [np.NaN, np.NaN, 3.0],
            [np.NaN, np.NaN, 4.0],
        ]
    )
    covariances = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
            [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
        ]
    )

    (
        batch_indices,
        batch_dimensions,
        batch_coords,
        batch_covariances,
    ) = _batch_coords_and_covariances(coords, covariances)
    assert (
        len(batch_indices)
        == len(batch_dimensions)
        == len(batch_coords)
        == len(batch_covariances)
        == 1
    )
    np.testing.assert_equal(batch_indices[0], np.array([0, 1]))
    np.testing.assert_equal(batch_dimensions[0], np.array([2]))
    np.testing.assert_equal(batch_coords[0], np.array([[3.0], [4.0]]))
    np.testing.assert_equal(batch_covariances[0], np.array([[[3.0]], [[4.0]]]))


def test_batch_coords_and_covariances_multiple_batches():
    # Multiple batches, different rows have different missing values
    coords = np.array(
        [
            [1.0, np.NaN, 3.0],
            [np.NaN, 3.0, 4.0],
        ]
    )
    covariances = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
            [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
        ]
    )

    (
        batch_indices,
        batch_dimensions,
        batch_coords,
        batch_covariances,
    ) = _batch_coords_and_covariances(coords, covariances)
    assert (
        len(batch_indices)
        == len(batch_dimensions)
        == len(batch_coords)
        == len(batch_covariances)
        == 2
    )
    np.testing.assert_equal(batch_indices[0], np.array([0]))
    np.testing.assert_equal(batch_dimensions[0], np.array([0, 2]))
    np.testing.assert_equal(batch_coords[0], np.array([[1.0, 3.0]]))
    np.testing.assert_equal(batch_covariances[0], np.array([[[1.0, 0.0], [0.0, 3.0]]]))

    np.testing.assert_equal(batch_indices[1], np.array([1]))
    np.testing.assert_equal(batch_dimensions[1], np.array([1, 2]))
    np.testing.assert_equal(batch_coords[1], np.array([[3.0, 4.0]]))
    np.testing.assert_equal(batch_covariances[1], np.array([[[3.0, 0.0], [0.0, 4.0]]]))


def test_Residuals_calculate():
    # Test that Residuals.calculate correctly identifies the number of degrees of freedom,
    # and correctly identifies the dimensions that have valid values and those that do not.
    observed_array = np.array(
        [
            [0.2, np.nan, np.nan, np.nan, np.nan, np.nan],
            [0.6, 1.0, 2.0, np.nan, np.nan, np.nan],
            [np.nan, 3.0, np.nan, 4.0, np.nan, np.nan],
            [0.5, 3.0, 0.5, 4.5, 0.1, 0.1],
        ]
    )
    observed = CartesianCoordinates.from_kwargs(
        x=observed_array[:, 0],
        y=observed_array[:, 1],
        z=observed_array[:, 2],
        vx=observed_array[:, 3],
        vy=observed_array[:, 4],
        vz=observed_array[:, 5],
        covariance=CoordinateCovariances.from_sigmas(np.full((4, 6), 0.1)),
        origin=Origin.from_kwargs(code=np.full(4, "SUN")),
        frame="ecliptic",
    )
    predicted_array = np.array(
        [
            [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [0.5, 1.1, 1.9, 0.2, 0.1, 0.1],
            [0.5, 2.9, 0.2, 4.1, 0.1, 0.1],
            [0.5, 3.0, 0.5, 4.5, 0.1, 0.1],
        ]
    )
    predicted = CartesianCoordinates.from_kwargs(
        x=predicted_array[:, 0],
        y=predicted_array[:, 1],
        z=predicted_array[:, 2],
        vx=predicted_array[:, 3],
        vy=predicted_array[:, 4],
        vz=predicted_array[:, 5],
        origin=Origin.from_kwargs(code=np.full(4, "SUN")),
        frame="ecliptic",
    )

    residuals = Residuals.calculate(observed, predicted)

    # Calculate the expected residuals
    desired_residuals = observed_array - predicted_array
    np.testing.assert_almost_equal(
        desired_residuals[0], np.array([-0.1, np.nan, np.nan, np.nan, np.nan, np.nan])
    )
    np.testing.assert_almost_equal(
        desired_residuals[1], np.array([0.1, -0.1, 0.1, np.nan, np.nan, np.nan])
    )
    np.testing.assert_almost_equal(
        desired_residuals[2], np.array([np.nan, 0.1, np.nan, -0.1, np.nan, np.nan])
    )
    np.testing.assert_almost_equal(
        desired_residuals[3], np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )

    assert len(residuals) == 4
    assert residuals.to_array().shape == (4, 6)
    np.testing.assert_equal(residuals.to_array(), desired_residuals)
    assert residuals.dof.to_pylist() == [1, 3, 2, 6]
    np.testing.assert_almost_equal(
        residuals.chi2.to_numpy(zero_copy_only=False), np.array([1, 3, 2, 0])
    )

    # Test that the probabilities for the first and last case are correct (these are more well known examples)
    actual_probabilities = residuals.probability.to_numpy(zero_copy_only=False)
    np.testing.assert_almost_equal(actual_probabilities[0], 0.31731050786291415)
    np.testing.assert_almost_equal(actual_probabilities[3], 1.0)


def test_Residuals_calculate_raises_frames():
    # Test that an error is raised when the frames are not equal
    observed_array = np.random.random((10, 6))
    observed = CartesianCoordinates.from_kwargs(
        x=observed_array[:, 0],
        y=observed_array[:, 1],
        z=observed_array[:, 2],
        vx=observed_array[:, 3],
        vy=observed_array[:, 4],
        vz=observed_array[:, 5],
        origin=Origin.from_kwargs(code=np.full(10, "SUN")),
        frame="ecliptic",
    )

    predicted_array = np.random.random((10, 6))
    predicted = CartesianCoordinates.from_kwargs(
        x=predicted_array[:, 0],
        y=predicted_array[:, 1],
        z=predicted_array[:, 2],
        vx=predicted_array[:, 3],
        vy=predicted_array[:, 4],
        vz=predicted_array[:, 5],
        origin=Origin.from_kwargs(code=np.full(10, "SUN")),
        frame="equatorial",
    )

    with pytest.raises(ValueError, match=r"coordinates must have the same frame."):
        Residuals.calculate(observed, predicted)


def test_Residuals_calculate_raises_origins():
    # Test that an error is raised when the frames are not equal
    observed_array = np.random.random((10, 6))
    observed = CartesianCoordinates.from_kwargs(
        x=observed_array[:, 0],
        y=observed_array[:, 1],
        z=observed_array[:, 2],
        vx=observed_array[:, 3],
        vy=observed_array[:, 4],
        vz=observed_array[:, 5],
        origin=Origin.from_kwargs(code=np.full(10, "SUN")),
        frame="equatorial",
    )

    predicted_array = np.random.random((10, 6))
    predicted = CartesianCoordinates.from_kwargs(
        x=predicted_array[:, 0],
        y=predicted_array[:, 1],
        z=predicted_array[:, 2],
        vx=predicted_array[:, 3],
        vy=predicted_array[:, 4],
        vz=predicted_array[:, 5],
        origin=Origin.from_kwargs(code=np.full(10, "EARTH")),
        frame="equatorial",
    )

    with pytest.raises(ValueError, match=r"coordinates must have the same origin."):
        Residuals.calculate(observed, predicted)


def test_bound_longitude_residuals():
    # Test that bound_longitude_residuals correctly bounds the longitude residuals
    # to within -180 and 180 degrees and that it handles the signs correctly across the
    # 0/360 degree boundary. Typically residuals that are positive mean that the observed
    # value is greater than the predicted value, and vice versa for negative residuals.

    # Observed  Predicted  Arithmetic  Expected
    #    1        359        -358         -2
    #  359          1         358          2
    #   60        240        -180       -180
    #  240         60         180        180
    #   10        190        -180       -180
    #  190         10         180        180
    #   60        250        -190       -170
    #  250         60         190        170
    #   60        230        -170       -170
    #  230         60         170        170
    observed_array = np.ones((10, 6))
    observed_array[:, 1] = np.array([1, 359, 60, 240, 10, 190, 60, 250, 60, 230])

    predicted_array = np.ones((10, 6))
    predicted_array[:, 1] = np.array([359, 1, 240, 60, 190, 10, 250, 60, 230, 60])

    residuals_array = observed_array - predicted_array
    residuals = bound_longitude_residuals(observed_array, residuals_array)

    np.testing.assert_equal(
        residuals[:, 1], np.array([-2, 2, -180, 180, -180, 180, -170, 170, -170, 170])
    )
