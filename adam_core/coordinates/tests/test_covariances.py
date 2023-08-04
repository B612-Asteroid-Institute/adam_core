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

    # Test when covariances are mixed with None and np.array
    covariances = [None, np.ones((6, 6)).flatten()]
    cov = CoordinateCovariances.from_kwargs(values=covariances)
    cov_expected = np.ones((2, 6, 6))
    cov_expected[0, :, :] = np.NaN
    np.testing.assert_equal(cov.to_matrix(), cov_expected)

    # Test when covariances are only None
    covariances = [None, None]
    cov = CoordinateCovariances.from_kwargs(values=covariances)
    cov_expected = np.full((2, 6, 6), np.NaN)
    np.testing.assert_equal(cov.to_matrix(), cov_expected)


def test_CoordinateCovariances_to_dataframe():
    # Given an array of covariances test that the dataframe
    # is correctly populated
    covariances = np.array(
        [
            [
                [4.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, 9.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 16.0, -3.0, 0.0, 0.0],
                [0.0, 0.0, -3.0, 25.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 4.0, 36.0, -5.0],
                [0.0, 0.0, 0.0, 0.0, -5.0, 49.0],
            ],
            [
                [9.0, -2.0, 0.0, 0.0, 0.0, 0.0],
                [-2.0, 16.0, 3.0, 0.0, 0.0, 0.0],
                [0.0, 3.0, 25.0, -4.0, 0.0, 0.0],
                [0.0, 0.0, -4.0, 36.0, 5.0, 0.0],
                [0.0, 0.0, 0.0, 5.0, 49.0, -6.0],
                [0.0, 0.0, 0.0, 0.0, -6.0, 64.0],
            ],
        ]
    )
    cov = CoordinateCovariances.from_matrix(covariances)
    df1 = cov.to_dataframe(coord_names=["x", "y", "z", "vx", "vy", "vz"])
    df2 = cov.to_dataframe(coord_names=["x", "y", "z", "vx", "vy", "vz"], sigmas=True)
    for df in [df1, df2]:
        np.testing.assert_equal(df["cov_x_x"].values, np.array([4.0, 9.0]))
        np.testing.assert_equal(df["cov_y_y"].values, np.array([9.0, 16.0]))
        np.testing.assert_equal(df["cov_z_z"].values, np.array([16.0, 25.0]))
        np.testing.assert_equal(df["cov_vx_vx"].values, np.array([25.0, 36.0]))
        np.testing.assert_equal(df["cov_vy_vy"].values, np.array([36.0, 49.0]))
        np.testing.assert_equal(df["cov_vz_vz"].values, np.array([49.0, 64.0]))
        np.testing.assert_equal(df["cov_y_x"].values, np.array([-1.0, -2.0]))
        np.testing.assert_equal(df["cov_z_x"].values, np.array([0.0, 0.0]))
        np.testing.assert_equal(df["cov_vx_x"].values, np.array([0.0, 0.0]))
        np.testing.assert_equal(df["cov_vy_x"].values, np.array([0.0, 0.0]))
        np.testing.assert_equal(df["cov_vz_x"].values, np.array([0.0, 0.0]))
        np.testing.assert_equal(df["cov_z_y"].values, np.array([2.0, 3.0]))
        np.testing.assert_equal(df["cov_vx_y"].values, np.array([0.0, 0.0]))
        np.testing.assert_equal(df["cov_vy_y"].values, np.array([0.0, 0.0]))
        np.testing.assert_equal(df["cov_vz_y"].values, np.array([0.0, 0.0]))
        np.testing.assert_equal(df["cov_vx_z"].values, np.array([-3.0, -4.0]))
        np.testing.assert_equal(df["cov_vy_z"].values, np.array([0.0, 0.0]))
        np.testing.assert_equal(df["cov_vz_z"].values, np.array([0.0, 0.0]))
        np.testing.assert_equal(df["cov_vy_vx"].values, np.array([4.0, 5.0]))
        np.testing.assert_equal(df["cov_vz_vx"].values, np.array([0.0, 0.0]))
        np.testing.assert_equal(df["cov_vz_vy"].values, np.array([-5.0, -6.0]))

    np.testing.assert_equal(df["sigma_x"].values, np.sqrt(np.array([4.0, 9.0])))
    np.testing.assert_equal(df["sigma_y"].values, np.sqrt(np.array([9.0, 16.0])))
    np.testing.assert_equal(df["sigma_z"].values, np.sqrt(np.array([16.0, 25.0])))
    np.testing.assert_equal(df["sigma_vx"].values, np.sqrt(np.array([25.0, 36.0])))
    np.testing.assert_equal(df["sigma_vy"].values, np.sqrt(np.array([36.0, 49.0])))
    np.testing.assert_equal(df["sigma_vz"].values, np.sqrt(np.array([49.0, 64.0])))


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
