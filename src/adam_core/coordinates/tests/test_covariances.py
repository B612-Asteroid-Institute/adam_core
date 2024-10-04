import numpy as np
import pytest

from ...utils.helpers.orbits import make_real_orbits
from ..covariances import (
    CoordinateCovariances,
    make_positive_semidefinite,
    sample_covariance_random,
    sample_covariance_sigma_points,
    weighted_covariance,
    weighted_mean,
)


def test_sample_covariance_sigma_points():
    # Get a sample of real orbits and test that sigma point sampling
    # allows the state vector and its covariance to be reconstructed
    orbits = make_real_orbits()

    # Limit to first 10 orbits
    for orbit in orbits[:10]:
        mean = orbit.coordinates.values[0]
        covariance = orbit.coordinates.covariance.to_matrix()[0]

        samples, W, W_cov = sample_covariance_sigma_points(mean, covariance)

        # In a 6 dimensional space we expect 13 sigma point samples
        assert len(samples) == 13
        np.testing.assert_almost_equal(np.sum(W), 1.0)
        np.testing.assert_almost_equal(np.sum(W_cov), 1.0)
        # The first sample should be the mean
        np.testing.assert_equal(samples[0], mean)
        # The first weight should be 0.0
        assert W[0] == 0.0
        # The first weight for the covariance should be 0
        # since beta = 0 internally
        assert W_cov[0] == 0.0

        # Reconstruct the mean and covariance and test that they match
        # the original inputs to within 1e-14
        mean_sg = weighted_mean(samples, W)
        covariance_sg = weighted_covariance(mean, samples, W_cov)

        np.testing.assert_allclose(mean_sg, mean, rtol=0, atol=1e-14)
        np.testing.assert_allclose(covariance_sg, covariance, rtol=0, atol=1e-14)


def test_make_positive_semidefinite():
    non_psd_matrix = np.array(
        [
            [1e-11, 0, 0, 0, 0, 0],
            [0, -1e-11, 0, 0, 0, 0],
            [0, 0, -5e-12, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1.5, 0],
            [0, 0, 0, 0, 0, 2],
        ]
    )

    psd_matrix = make_positive_semidefinite(non_psd_matrix, 1e-10)

    assert np.all(np.linalg.eigvals(psd_matrix) >= 0)


def test_make_positive_semidefinite_fail():
    # Case where eigenvalues exceed the tolerance and should not be flipped
    non_psd_matrix_fail = np.array(
        [
            [1e-11, 0, 0, 0, 0, 0],
            [0, -1e-11, 0, 0, 0, 0],
            [0, 0, -5e-12, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1.5, 0],
            [0, 0, 0, 0, 0, 2],
        ]
    )

    with pytest.raises(ValueError):
        make_positive_semidefinite(non_psd_matrix_fail, semidef_tol=1e-15)


def test_sample_covariance_random():
    # Get a sample of real orbits and test that random sampling
    # allows the state vector and its covariance to be reconstructed
    orbits = make_real_orbits()

    np.random.seed(0)
    # Limit to first 10 orbits
    for orbit in orbits[:10]:
        mean = orbit.coordinates.values[0]
        covariance = orbit.coordinates.covariance.to_matrix()[0]

        samples, W, W_cov = sample_covariance_random(mean, covariance, 1000000)

        # In a 6 dimensional space we expect 1000000 samples
        assert len(samples) == 1000000
        np.testing.assert_almost_equal(np.sum(W), 1.0)
        np.testing.assert_almost_equal(np.sum(W_cov), 1.0)

        # Reconstruct the mean and covariance and test that they match
        # the original inputs to within 1e-8 and 1e-14 respectively
        mean_rs = weighted_mean(samples, W)
        covariance_rs = weighted_covariance(mean, samples, W_cov)

        # Note how many samples are needed to get the covariance to
        # match to within 1e-14... (the same tolerance as sigma point sampling)
        np.testing.assert_allclose(mean_rs, mean, rtol=0, atol=1e-8)
        np.testing.assert_allclose(covariance_rs, covariance, rtol=0, atol=1e-14)


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

    # Test when covariances are mixed with None and np.array
    covariances = [None, np.ones((6, 6)).flatten()]
    cov = CoordinateCovariances.from_kwargs(values=covariances)
    cov_expected = np.ones((2, 6, 6))
    cov_expected[0, :, :] = np.nan
    np.testing.assert_equal(cov.to_matrix(), cov_expected)

    # Test when covariances are only None
    covariances = [None, None]
    cov = CoordinateCovariances.from_kwargs(values=covariances)
    cov_expected = np.full((2, 6, 6), np.nan)
    np.testing.assert_equal(cov.to_matrix(), cov_expected)


def test_CoordinateCovariances_is_all_nan():
    # Test that all_nan convenience method works as intended
    # No NaNs at all
    covariances = np.ones((10, 6, 6))
    cov = CoordinateCovariances.from_matrix(covariances)
    assert not cov.is_all_nan()

    # Single NaN value
    covariances = np.ones((10, 6, 6))
    covariances[0, 0, 0] = np.nan
    cov = CoordinateCovariances.from_matrix(covariances)
    assert not cov.is_all_nan()

    # Null covariance (None)
    covariances = [None, np.ones((6, 6)).flatten()]
    cov = CoordinateCovariances.from_kwargs(values=covariances)
    assert not cov.is_all_nan()

    # All NaNs
    covariances = np.ones((10, 6, 6)) * np.nan
    cov = CoordinateCovariances.from_matrix(covariances)
    assert cov.is_all_nan()


def test_CoordinateCovariances_from_matrix_invalid_shape():
    # Test that an error is raised when the shape of the input
    # covariances is invalid
    covariances = np.zeros((10, 6, 5))
    with pytest.raises(ValueError):
        CoordinateCovariances.from_matrix(covariances)

    covariances = np.zeros((6, 5))
    with pytest.raises(ValueError):
        CoordinateCovariances.from_matrix(covariances)
