import numpy as np
from scipy.spatial.distance import mahalanobis

from ..residuals import _batch_coords_and_covariances, calculate_chi2


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
