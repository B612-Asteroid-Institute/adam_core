import numpy as np

from ..cartesian import CartesianCoordinates
from ..covariances import CoordinateCovariances
from ..origin import Origin


def test_CartesianCoordinates_rotate():
    # Test rotation of Cartesian coordinates. Rotating coordinates and their
    # covariances with an identity matrix should return the same coordinates and covariances.
    N, D = 1000, 6
    values = np.random.random((N, D))
    covariances = np.random.random((N, D, D))

    coords = CartesianCoordinates.from_kwargs(
        x=values[:, 0],
        y=values[:, 1],
        z=values[:, 2],
        vx=values[:, 3],
        vy=values[:, 4],
        vz=values[:, 5],
        covariance=CoordinateCovariances.from_matrix(covariances),
        origin=Origin.from_kwargs(code=["origin"] * N),
        frame="equatorial",
    )

    rot_matrix = np.identity(D)
    coords_rotated = coords.rotate(rot_matrix, "identity")

    np.testing.assert_equal(coords_rotated.values, values)
    np.testing.assert_equal(coords_rotated.covariance.to_matrix(), covariances)

    # Now repeat the same with a non-identity matrix (rotation by 90 degrees
    # about the x-axis)
    theta = np.radians(90)
    rot_matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, np.cos(theta), -np.sin(theta), 0.0, 0.0, 0.0],
            [0.0, np.sin(theta), np.cos(theta), 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, np.cos(theta), -np.sin(theta)],
            [0.0, 0.0, 0.0, 0.0, np.sin(theta), np.cos(theta)],
        ]
    )

    coords_rotated = coords.rotate(rot_matrix, "x-axis-90")

    # x position and velocity should be unchanged
    np.testing.assert_equal(coords_rotated.values[:, 0], values[:, 0])
    np.testing.assert_equal(coords_rotated.values[:, 3], values[:, 3])

    # y rotates to z, z rotates to -y, vy rotates to -vz, vz rotates to vy
    np.testing.assert_almost_equal(coords_rotated.values[:, 2], values[:, 1])
    np.testing.assert_almost_equal(coords_rotated.values[:, 1], -values[:, 2])
    np.testing.assert_almost_equal(coords_rotated.values[:, 4], -values[:, 5])
    np.testing.assert_almost_equal(coords_rotated.values[:, 5], values[:, 4])

    # Covariances should also be rotated (this is a bit harder to test) but we
    # write out the explicit matrix multiplication)
    covariances_rotated = rot_matrix @ covariances @ rot_matrix.T
    np.testing.assert_almost_equal(
        coords_rotated.covariance.to_matrix(), covariances_rotated
    )
    return


def test_CartesianCoordinates_translate():
    # Test translation of Cartesian coordinates. Translating coordinates and their
    # covariances with a zero vector should return the same coordinates and covariances.
    N, D = 1000, 6
    values = np.random.random((N, D))
    covariances = np.random.random((N, D, D))

    coords = CartesianCoordinates.from_kwargs(
        x=values[:, 0],
        y=values[:, 1],
        z=values[:, 2],
        vx=values[:, 3],
        vy=values[:, 4],
        vz=values[:, 5],
        covariance=CoordinateCovariances.from_matrix(covariances),
        origin=Origin.from_kwargs(code=["origin"] * N),
        frame="equatorial",
    )

    translation_vector = np.zeros(D)
    coords_translated = coords.translate(translation_vector, "zero")

    np.testing.assert_equal(coords_translated.values, values)
    np.testing.assert_equal(coords_translated.covariance.to_matrix(), covariances)

    # Now repeat the same with a non-zero vector
    translation_vector = np.array([1.0, -2.0, 3.0, 4.0, -5.0, 6.0])
    coords_translated = coords.translate(translation_vector, "nonzero")

    np.testing.assert_equal(coords_translated.values, values + translation_vector)
    np.testing.assert_equal(coords_translated.covariance.to_matrix(), covariances)
