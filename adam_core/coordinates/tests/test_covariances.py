import numpy as np

from ..covariances import CoordinateCovariances


def test_CoordinateCovariances_from_sigmas():
    # Given an array of sigmas, test that the
    # covariance matrices are correctly calculated
    sigmas = np.array(
        [
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ]
    )
    covariances = np.array(
        [
            [
                [4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 9.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 16.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 25.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 36.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 49.0],
            ],
            [
                [9.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 16.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 25.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 36.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 49.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 64.0],
            ],
        ]
    )

    cov = CoordinateCovariances.from_sigmas(sigmas)
    np.testing.assert_equal(cov.to_matrix(), covariances)


def test_CoordinateCovariances_to_from_matrix():
    # Given an array of covariances test that the
    # covariance matrices are correctly returned
    covariances = np.zeros((10, 6, 6))
    for i in range(10):
        covariances[i, :, :] = np.diag(np.arange(1, 7))

    cov = CoordinateCovariances.from_matrix(covariances)
    np.testing.assert_equal(cov.to_matrix(), covariances)
