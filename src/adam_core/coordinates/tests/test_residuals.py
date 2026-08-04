import numpy as np
import pytest
from scipy.spatial.distance import mahalanobis

from ..cartesian import CartesianCoordinates, CoordinateCovariances
from ..origin import Origin
from ..residuals import (
    Residuals,
    _batch_coords_and_covariances,
    apply_cosine_latitude_correction,
    bound_longitude_residuals,
    calculate_chi2,
    calculate_reduced_chi2,
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


def test_calculate_chi2_missing_diagonal_covariance_values():
    # Lets make sure we raise an error if the covariance matrix has NaNs on the diagonal
    residuals = np.array(
        [
            [1, 2, 3],
            [2, 1, 1],
        ]
    )
    covariances = np.array(
        [
            [[np.nan, 0, 0], [0, np.nan, 0], [0, 0, 1]],
            [[np.nan, 0, 0], [0, np.nan, 0], [0, 0, 1]],
        ]
    )

    with pytest.raises(
        ValueError, match=r"Covariance matrix has NaNs on the diagonal."
    ):
        calculate_chi2(residuals, covariances)


def test_calculate_chi2_missing_off_diagonal_covariance_values():
    # Lets make sure we raise an error if the covariance matrix has NaNs on the off-diagonal
    residuals = np.array(
        [
            [1, 2, 3],
            [2, 1, 1],
        ]
    )
    covariances = np.array(
        [
            [[1, np.nan, 0], [np.nan, 1, 0], [0, 0, 1]],
            [[1, np.nan, 0], [np.nan, 1, 0], [0, 0, 1]],
        ]
    )

    with pytest.warns(
        UserWarning,
        match=r"Covariance matrix has NaNs on the off-diagonal \(these will be assumed to be 0.0\).",
    ):
        np.testing.assert_allclose(calculate_chi2(residuals, covariances), [14, 6])


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
            [1.0, np.nan, 3.0],
            [2.0, np.nan, 4.0],
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
            [np.nan, np.nan, 3.0],
            [np.nan, np.nan, 4.0],
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
            [1.0, np.nan, 3.0],
            [np.nan, 3.0, 4.0],
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


@pytest.fixture
def observed_array():
    observed_array = np.array(
        [
            [0.2, np.nan, np.nan, np.nan, np.nan, np.nan],
            [0.6, 1.0, 2.0, np.nan, np.nan, np.nan],
            [np.nan, 3.0, np.nan, 4.0, np.nan, np.nan],
            [0.5, 3.0, 0.5, 4.5, 0.1, 0.1],
        ]
    )
    return observed_array


@pytest.fixture
def predicted_array():
    predicted_array = np.array(
        [
            [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [0.5, 1.1, 1.9, 0.2, 0.1, 0.1],
            [0.5, 2.9, 0.2, 4.1, 0.1, 0.1],
            [0.5, 3.0, 0.5, 4.5, 0.1, 0.1],
        ]
    )
    return predicted_array


@pytest.fixture
def expected_residuals(observed_array, predicted_array):
    return observed_array - predicted_array


def test_Residuals_calculate(observed_array, predicted_array, expected_residuals):
    # Test that Residuals.calculate correctly identifies the number of degrees of freedom,
    # and correctly identifies the dimensions that have valid values and those that do not.
    observed = CartesianCoordinates.from_kwargs(
        x=observed_array[:, 0],
        y=observed_array[:, 1],
        z=observed_array[:, 2],
        vx=observed_array[:, 3],
        vy=observed_array[:, 4],
        vz=observed_array[:, 5],
        covariance=CoordinateCovariances.from_sigmas(np.full((4, 6), 0.1)),
        origin=Origin.from_kwargs(code=np.full(4, "SUN", dtype="object")),
        frame="ecliptic",
    )
    predicted = CartesianCoordinates.from_kwargs(
        x=predicted_array[:, 0],
        y=predicted_array[:, 1],
        z=predicted_array[:, 2],
        vx=predicted_array[:, 3],
        vy=predicted_array[:, 4],
        vz=predicted_array[:, 5],
        origin=Origin.from_kwargs(code=np.full(4, "SUN", dtype="object")),
        frame="ecliptic",
    )

    residuals = Residuals.calculate(observed, predicted)

    assert len(residuals) == 4
    assert residuals.to_array().shape == (4, 6)
    np.testing.assert_equal(residuals.to_array(), expected_residuals)
    assert residuals.dof.to_pylist() == [1, 3, 2, 6]
    np.testing.assert_almost_equal(
        residuals.chi2.to_numpy(zero_copy_only=False), np.array([1, 3, 2, 0])
    )

    # Test that the probabilities for the first and last case are correct (these are more well known examples)
    actual_probabilities = residuals.probability.to_numpy(zero_copy_only=False)
    np.testing.assert_almost_equal(actual_probabilities[0], 0.31731050786291415)
    np.testing.assert_almost_equal(actual_probabilities[3], 1.0)


def test_Residuals_calculate_missing_covariance_values(
    observed_array, predicted_array, expected_residuals
):
    # Test that Residuals.calculate correctly identifies the number of degrees of freedom,
    # and correctly identifies the dimensions that have valid values and those that do not.
    # Here all covariance values (both variates and covariates) are defined
    # for those dimensions that have values, the rest are NaN
    observed_covariances = np.array(
        [
            [
                [0.01, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
            [
                [0.01, 0.0, 0.0, np.nan, np.nan, np.nan],
                [0.0, 0.01, 0.0, np.nan, np.nan, np.nan],
                [0.0, 0.0, 0.01, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, 0.01, np.nan, 0.0, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, 0.0, np.nan, 0.01, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
            [
                [0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.01, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.01, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.01, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.01, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.01],
            ],
        ]
    )

    observed = CartesianCoordinates.from_kwargs(
        x=observed_array[:, 0],
        y=observed_array[:, 1],
        z=observed_array[:, 2],
        vx=observed_array[:, 3],
        vy=observed_array[:, 4],
        vz=observed_array[:, 5],
        covariance=CoordinateCovariances.from_matrix(observed_covariances),
        origin=Origin.from_kwargs(code=np.full(4, "SUN", dtype="object")),
        frame="ecliptic",
    )
    predicted = CartesianCoordinates.from_kwargs(
        x=predicted_array[:, 0],
        y=predicted_array[:, 1],
        z=predicted_array[:, 2],
        vx=predicted_array[:, 3],
        vy=predicted_array[:, 4],
        vz=predicted_array[:, 5],
        origin=Origin.from_kwargs(code=np.full(4, "SUN", dtype="object")),
        frame="ecliptic",
    )

    residuals = Residuals.calculate(observed, predicted)

    assert len(residuals) == 4
    assert residuals.to_array().shape == (4, 6)
    np.testing.assert_equal(residuals.to_array(), expected_residuals)
    assert residuals.dof.to_pylist() == [1, 3, 2, 6]
    np.testing.assert_almost_equal(
        residuals.chi2.to_numpy(zero_copy_only=False), np.array([1, 3, 2, 0])
    )

    # Test that the probabilities for the first and last case are correct (these are more well known examples)
    actual_probabilities = residuals.probability.to_numpy(zero_copy_only=False)
    np.testing.assert_almost_equal(actual_probabilities[0], 0.31731050786291415)
    np.testing.assert_almost_equal(actual_probabilities[3], 1.0)


def test_Residuals_calculate_missing_off_diagonal_covariance_values(
    observed_array, predicted_array, expected_residuals
):
    # Test that Residuals.calculate correctly identifies the number of degrees of freedom,
    # and correctly identifies the dimensions that have valid values and those that do not.
    # Here only variance values are defined
    # for those dimensions that have values, the covariates and the rest are NaN
    observed_covariances = np.array(
        [
            [
                [0.01, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
            [
                [0.01, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, 0.01, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, 0.01, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, 0.01, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, 0.01, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
            [
                [0.01, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, 0.01, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, 0.01, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, 0.01, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, 0.01, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, 0.01],
            ],
        ]
    )

    observed = CartesianCoordinates.from_kwargs(
        x=observed_array[:, 0],
        y=observed_array[:, 1],
        z=observed_array[:, 2],
        vx=observed_array[:, 3],
        vy=observed_array[:, 4],
        vz=observed_array[:, 5],
        covariance=CoordinateCovariances.from_matrix(observed_covariances),
        origin=Origin.from_kwargs(code=np.full(4, "SUN", dtype="object")),
        frame="ecliptic",
    )
    predicted = CartesianCoordinates.from_kwargs(
        x=predicted_array[:, 0],
        y=predicted_array[:, 1],
        z=predicted_array[:, 2],
        vx=predicted_array[:, 3],
        vy=predicted_array[:, 4],
        vz=predicted_array[:, 5],
        origin=Origin.from_kwargs(code=np.full(4, "SUN", dtype="object")),
        frame="ecliptic",
    )

    with pytest.warns(
        UserWarning,
        match=r"Covariance matrix has NaNs on the off-diagonal \(these will be assumed to be 0.0\).",
    ):
        residuals = Residuals.calculate(observed, predicted)

    assert len(residuals) == 4
    assert residuals.to_array().shape == (4, 6)
    np.testing.assert_equal(residuals.to_array(), expected_residuals)
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
        origin=Origin.from_kwargs(code=np.full(10, "SUN", dtype="object")),
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
        origin=Origin.from_kwargs(code=np.full(10, "SUN", dtype="object")),
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
        origin=Origin.from_kwargs(code=np.full(10, "SUN", dtype="object")),
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
        origin=Origin.from_kwargs(code=np.full(10, "EARTH", dtype="object")),
        frame="equatorial",
    )

    with pytest.raises(ValueError, match=r"coordinates must have the same origin."):
        Residuals.calculate(observed, predicted)


def test_apply_cosine_latitude_correction():
    # Test that apply_cosine_latitude_correction correctly applies the cosine latitude correction
    # to the residuals and covariances as a function of the latitude.
    residual_array = np.array(
        [
            [0, 10, 0, 0, 1, 0],
            [0, 10, 0, 0, 1, 0],
            [0, 10, 0, 0, 1, 0],
            [0, 10, 0, 0, 1, 0],
        ],
        dtype=np.float64,
    )

    # Create covariances that are all 1
    covariance_array = np.ones((4, 6, 6), dtype=np.float64)

    # Cosine of 0 degrees is 1
    # Cosine of 45 degrees is 1/sqrt(2)
    # Cosien of 60 degrees is 1/2
    # Cosine of 90 degrees is 0
    # The latter represents an unphysical case
    latitude_array = np.array([0, 45, 60, 90])
    cos_latitude = np.cos(np.radians(latitude_array))
    expected_residual_array = np.array(
        [
            [0, 10, 0, 0, 1, 0],
            [0, 10 / np.sqrt(2), 0, 0, 1 / np.sqrt(2), 0],
            [0, 5, 0, 0, 1 / 2, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )

    # Covariance matrix elements
    # rho_rho  (:,0,0), rho_lon  (: 0,1), rho_lat  (:,0,2), rho_vrho  (:,0,3), rho_vlon  (:,0,4), rho_vlat  (:,0,5)
    # lon_rho  (:,1,0), lon_lon  (:,1,1), lon_lat  (:,1,2), lon_vrho  (:,1,3), lon_vlon  (:,1,4), lon_vlat  (:,1,5)
    # lat_rho  (:,2,0), lat_lon  (:,2,1), lat_lat  (:,2,2), lat_vrho  (:,2,3), lat_vlon  (:,2,4), lat_vlat  (:,2,5)
    # vrho_rho (:,3,0), vrho_lon (:,3,1), vrho_lat (:,3,2), vrho_vrho (:,3,3), vrho_vlon (:,3,4), vrho_vlat (:,3,5)
    # vlon_rho (:,4,0), vlon_lon (:,4,1), vlon_lat (:,4,2), vlon_vrho (:,4,3), vlon_vlon (:,4,4), vlon_vlat (:,4,5)
    # vlat_rho (:,5,0), vlat_lon (:,5,1), vlat_lat (:,5,2), vlat_vrho (:,5,3), vlat_vlon (:,5,4), vlat_vlat (:,5,5)
    # The only elements that should change are those rows and columns containing longitude and longitudinal velocity
    # Not that for cov_lon_lon and cov_vlon_vlon the correction is applied twice as expected (we touch both quantities
    # as rows and columns)
    expected_covariance = np.ones_like(covariance_array)
    expected_covariance[:, :, 1] *= cos_latitude[:, np.newaxis]
    expected_covariance[:, :, 4] *= cos_latitude[:, np.newaxis]
    expected_covariance[:, 1, :] *= cos_latitude[:, np.newaxis]
    expected_covariance[:, 4, :] *= cos_latitude[:, np.newaxis]

    corrected_residuals, corrected_covariances = apply_cosine_latitude_correction(
        latitude_array, residual_array, covariance_array
    )

    # Test that the residuals are corrected correctly
    np.testing.assert_almost_equal(
        corrected_residuals, expected_residual_array, decimal=15
    )

    # Test that the covariances are corrected correctly
    np.testing.assert_almost_equal(
        corrected_covariances, expected_covariance, decimal=15
    )

    # Add additional assertions for clarity
    # The above are sufficient to test the code but obfuscate the linear algebra

    # We expect that the first covariance matrix is completely unchanged
    np.testing.assert_equal(corrected_covariances[:, 0, 0], covariance_array[:, 0, 0])

    # Define the expected diagonal and off-diagonal elements that would have changed
    expected_covariates = np.array([1, 1 / np.sqrt(2), 1 / 2, 0])
    expected_variates = expected_covariates**2

    # Test_cov_lon_rho and rho_lon (1, 2)
    np.testing.assert_almost_equal(
        corrected_covariances[:, 1, 0], expected_covariates, decimal=15
    )
    np.testing.assert_almost_equal(
        corrected_covariances[:, 0, 1], expected_covariates, decimal=15
    )
    # Test cov_lon_lon (3)
    np.testing.assert_almost_equal(
        corrected_covariances[:, 1, 1], expected_variates, decimal=15
    )
    # Test cov_lon_lat and lat_lon (4, 5)
    np.testing.assert_almost_equal(
        corrected_covariances[:, 1, 2], expected_covariates, decimal=15
    )
    np.testing.assert_almost_equal(
        corrected_covariances[:, 2, 1], expected_covariates, decimal=15
    )
    # Test cov_lon_vrho and vrho_lon (6, 7)
    np.testing.assert_almost_equal(
        corrected_covariances[:, 1, 3], expected_covariates, decimal=15
    )
    np.testing.assert_almost_equal(
        corrected_covariances[:, 3, 1], expected_covariates, decimal=15
    )
    # Test cov_lon_vlon and vlon_lon (8, 9) - note that this is applied twice
    np.testing.assert_almost_equal(
        corrected_covariances[:, 1, 4], expected_variates, decimal=15
    )
    np.testing.assert_almost_equal(
        corrected_covariances[:, 4, 1], expected_variates, decimal=15
    )
    # Test cov_lon_vlat and vlat_lon (10, 11)
    np.testing.assert_almost_equal(
        corrected_covariances[:, 1, 5], expected_covariates, decimal=15
    )
    np.testing.assert_almost_equal(
        corrected_covariances[:, 5, 1], expected_covariates, decimal=15
    )

    # Test cov_vlon_rho and rho_vlon (12, 13)
    np.testing.assert_almost_equal(
        corrected_covariances[:, 4, 0], expected_covariates, decimal=15
    )
    np.testing.assert_almost_equal(
        corrected_covariances[:, 0, 4], expected_covariates, decimal=15
    )
    # Test cov_vlon_lat and lat_vlon (14, 15)
    np.testing.assert_almost_equal(
        corrected_covariances[:, 4, 2], expected_covariates, decimal=15
    )
    np.testing.assert_almost_equal(
        corrected_covariances[:, 2, 4], expected_covariates, decimal=15
    )
    # Test cov_vlon_vlon (16)
    np.testing.assert_almost_equal(
        corrected_covariances[:, 4, 4], expected_variates, decimal=15
    )
    # Test cov vlon_vrho and vrho_vlon (17, 18)
    np.testing.assert_almost_equal(
        corrected_covariances[:, 4, 3], expected_covariates, decimal=15
    )
    np.testing.assert_almost_equal(
        corrected_covariances[:, 3, 4], expected_covariates, decimal=15
    )
    # Test cov_vlon_vlat and vlat_vlon (19, 20)
    np.testing.assert_almost_equal(
        corrected_covariances[:, 4, 5], expected_covariates, decimal=15
    )
    np.testing.assert_almost_equal(
        corrected_covariances[:, 5, 4], expected_covariates, decimal=15
    )

    # Define the expected diagonal and off-diagonal elements that would not have changed
    expected_covariates_unchanged = expected_variates_unchanged = np.ones(4)
    # Test cov_rho_rho (21)
    np.testing.assert_almost_equal(
        corrected_covariances[:, 0, 0], expected_variates_unchanged, decimal=15
    )
    # Test cov_rho_lat and lat_rho (22, 23)
    np.testing.assert_almost_equal(
        corrected_covariances[:, 0, 2], expected_covariates_unchanged, decimal=15
    )
    np.testing.assert_almost_equal(
        corrected_covariances[:, 2, 0], expected_covariates_unchanged, decimal=15
    )
    # Test cov_rho_vrho and vrho_rho (24, 25)
    np.testing.assert_almost_equal(
        corrected_covariances[:, 0, 3], expected_covariates_unchanged, decimal=15
    )
    np.testing.assert_almost_equal(
        corrected_covariances[:, 3, 0], expected_covariates_unchanged, decimal=15
    )
    # Test cov_rho_vlat and vlat_rho (26, 27)
    np.testing.assert_almost_equal(
        corrected_covariances[:, 0, 5], expected_covariates_unchanged, decimal=15
    )
    np.testing.assert_almost_equal(
        corrected_covariances[:, 5, 0], expected_covariates_unchanged, decimal=15
    )
    # Test cov_lat_lat (28)
    np.testing.assert_almost_equal(
        corrected_covariances[:, 2, 2], expected_variates_unchanged, decimal=15
    )
    # Test cov_lat_vrho and vrho_lat (29, 30)
    np.testing.assert_almost_equal(
        corrected_covariances[:, 2, 3], expected_covariates_unchanged, decimal=15
    )
    np.testing.assert_almost_equal(
        corrected_covariances[:, 3, 2], expected_covariates_unchanged, decimal=15
    )
    # Test cov_lat_vlat and vlat_lat (31, 32)
    np.testing.assert_almost_equal(
        corrected_covariances[:, 2, 5], expected_covariates_unchanged, decimal=15
    )
    np.testing.assert_almost_equal(
        corrected_covariances[:, 5, 2], expected_covariates_unchanged, decimal=15
    )
    # Test cov_vrho_vrho (33)
    np.testing.assert_almost_equal(
        corrected_covariances[:, 3, 3], expected_variates_unchanged, decimal=15
    )
    # Test cov_vrho_vlat and vlat_vrho (34, 35)
    np.testing.assert_almost_equal(
        corrected_covariances[:, 3, 5], expected_covariates_unchanged, decimal=15
    )
    np.testing.assert_almost_equal(
        corrected_covariances[:, 5, 3], expected_covariates_unchanged, decimal=15
    )
    # Test cov_vlat_vlat (36)
    np.testing.assert_almost_equal(
        corrected_covariances[:, 5, 5], expected_variates_unchanged, decimal=15
    )


def test_apply_cosine_latitude_correction_missing_off_diagonal_covariance_values():
    # Test that apply_cosine_latitude_correction correctly applies the cosine latitude correction
    # to the residuals and covariances as a function of the latitude. Specifically, lets make sure
    # that if the covariance matrix has NaNs on the off-diagonal that these are correctly handled
    residual_array = np.array(
        [
            [0, 10, 0, 0, 1, 0],
            [0, 10, 0, 0, 1, 0],
            [0, 10, 0, 0, 1, 0],
            [0, 10, 0, 0, 1, 0],
        ],
        dtype=np.float64,
    )

    # Create covariances that only have the diagonal defined
    covariance_array = np.full((4, 6, 6), np.nan, dtype=np.float64)
    covariance_array[:, np.arange(6), np.arange(6)] = 1.0

    # Cosine of 0 degrees is 1
    # Cosine of 45 degrees is 1/sqrt(2)
    # Cosien of 60 degrees is 1/2
    # Cosine of 90 degrees is 0
    # The latter represents an unphysical case
    latitude_array = np.array([0, 45, 60, 90])
    cos_latitude = np.cos(np.radians(latitude_array))
    expected_residual_array = np.array(
        [
            [0, 10, 0, 0, 1, 0],
            [0, 10 / np.sqrt(2), 0, 0, 1 / np.sqrt(2), 0],
            [0, 5, 0, 0, 1 / 2, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )

    # Covariance matrix elements
    # rho_rho  (:,0,0), rho_lon  (: 0,1), rho_lat  (:,0,2), rho_vrho  (:,0,3), rho_vlon  (:,0,4), rho_vlat  (:,0,5)
    # lon_rho  (:,1,0), lon_lon  (:,1,1), lon_lat  (:,1,2), lon_vrho  (:,1,3), lon_vlon  (:,1,4), lon_vlat  (:,1,5)
    # lat_rho  (:,2,0), lat_lon  (:,2,1), lat_lat  (:,2,2), lat_vrho  (:,2,3), lat_vlon  (:,2,4), lat_vlat  (:,2,5)
    # vrho_rho (:,3,0), vrho_lon (:,3,1), vrho_lat (:,3,2), vrho_vrho (:,3,3), vrho_vlon (:,3,4), vrho_vlat (:,3,5)
    # vlon_rho (:,4,0), vlon_lon (:,4,1), vlon_lat (:,4,2), vlon_vrho (:,4,3), vlon_vlon (:,4,4), vlon_vlat (:,4,5)
    # vlat_rho (:,5,0), vlat_lon (:,5,1), vlat_lat (:,5,2), vlat_vrho (:,5,3), vlat_vlon (:,5,4), vlat_vlat (:,5,5)
    # The only elements that should change are those rows and columns containing longitude and longitudinal velocity
    # Not that for cov_lon_lon and cov_vlon_vlon the correction is applied twice as expected (we touch both quantities
    # as rows and columns)
    expected_covariance = covariance_array.copy()
    expected_covariance[:, :, 1] *= cos_latitude[:, np.newaxis]
    expected_covariance[:, :, 4] *= cos_latitude[:, np.newaxis]
    expected_covariance[:, 1, :] *= cos_latitude[:, np.newaxis]
    expected_covariance[:, 4, :] *= cos_latitude[:, np.newaxis]

    corrected_residuals, corrected_covariances = apply_cosine_latitude_correction(
        latitude_array, residual_array, covariance_array
    )

    # Test that the residuals are corrected correctly
    np.testing.assert_almost_equal(
        corrected_residuals, expected_residual_array, decimal=15
    )

    # Test that the covariances are corrected correctly
    np.testing.assert_almost_equal(
        corrected_covariances, expected_covariance, decimal=15
    )


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


def test_calculate_reduced_chi2(observed_array, predicted_array):
    # Test that Residuals.calculate correctly identifies the number of degrees of freedom,
    # and correctly identifies the dimensions that have valid values and those that do not.
    observed = CartesianCoordinates.from_kwargs(
        x=observed_array[:, 0],
        y=observed_array[:, 1],
        z=observed_array[:, 2],
        vx=observed_array[:, 3],
        vy=observed_array[:, 4],
        vz=observed_array[:, 5],
        covariance=CoordinateCovariances.from_sigmas(np.full((4, 6), 0.1)),
        origin=Origin.from_kwargs(code=np.full(4, "SUN", dtype="object")),
        frame="ecliptic",
    )
    predicted = CartesianCoordinates.from_kwargs(
        x=predicted_array[:, 0],
        y=predicted_array[:, 1],
        z=predicted_array[:, 2],
        vx=predicted_array[:, 3],
        vy=predicted_array[:, 4],
        vz=predicted_array[:, 5],
        origin=Origin.from_kwargs(code=np.full(4, "SUN", dtype="object")),
        frame="ecliptic",
    )

    residuals = Residuals.calculate(observed, predicted)

    assert residuals.dof.to_pylist() == [1, 3, 2, 6]
    np.testing.assert_almost_equal(
        residuals.chi2.to_numpy(zero_copy_only=False), np.array([1, 3, 2, 0])
    )

    reduced_chi2 = calculate_reduced_chi2(residuals, 6)
    np.testing.assert_almost_equal(reduced_chi2, 6 / 6)

    reduced_chi2 = calculate_reduced_chi2(residuals, 4)
    np.testing.assert_almost_equal(reduced_chi2, 6 / 8)
