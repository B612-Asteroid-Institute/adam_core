import numpy as np
from scipy.spatial.distance import mahalanobis

from ..residuals import calculate_chi2


def test_calculate_chi2():
    # Einsums are tricky to follow, so here's a test case
    # with a small number of dimensions and residuals to check that
    # the chi2 calculation is correct.
    residuals = np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
        ]
    )
    covariances = np.array(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ]
    )
    assert np.allclose(calculate_chi2(residuals, covariances), [3, 3])

    # Test that the chi2 calculation works for a single residual and is
    # exactly equal to chi2 (observed - expected)^2 / sigma^2
    residuals = np.array([[1]])
    covariances = np.array([[[1]]])
    assert np.allclose(calculate_chi2(residuals, covariances), [1])


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
