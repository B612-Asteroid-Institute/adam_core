import numpy as np
import pytest

from ..cartesian import SPECIFIC_ANGULAR_MOMENTUM_TOLERANCE, CartesianCoordinates
from ..covariances import CoordinateCovariances
from ..origin import Origin


def test_CartesianCoordinates_rotate() -> None:
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


def _cartesian_with_covariance_rows(
    covariance: np.ndarray,
) -> CartesianCoordinates:
    rows = len(covariance)
    values = np.tile(np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]), (rows, 1))
    return CartesianCoordinates.from_kwargs(
        x=values[:, 0],
        y=values[:, 1],
        z=values[:, 2],
        vx=values[:, 3],
        vy=values[:, 4],
        vz=values[:, 5],
        covariance=CoordinateCovariances.from_matrix(covariance),
        origin=Origin.from_kwargs(code=["origin"] * rows),
        frame="equatorial",
    )


def _nonidentity_xy_rotation() -> np.ndarray:
    cosine = np.sqrt(0.5)
    matrix = np.eye(6)
    matrix[:2, :2] = [[cosine, -cosine], [cosine, cosine]]
    return matrix


def test_cartesian_rotate_6d_nonidentity_preserves_ieee_infinity_policy() -> None:
    covariance = np.zeros((3, 6, 6), dtype=np.float64)
    covariance[0, 0, 0], covariance[0, 1, 1] = np.inf, -np.inf
    covariance[1, 0, 0], covariance[1, 1, 1] = -np.inf, np.inf
    covariance[2] = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    covariance[2, 4, 5] = np.nan
    matrix = _nonidentity_xy_rotation()

    rotated = _cartesian_with_covariance_rows(covariance).rotate(
        matrix, "nonidentity-6d"
    )
    actual = rotated.covariance.to_matrix()

    expected_nan = np.ones_like(covariance, dtype=bool)
    expected_nan[:2, :2, :2] = np.array([[True, False], [False, True]])
    expected_nan[2] = False
    expected_nan[2, 4, 5] = True
    np.testing.assert_array_equal(np.isnan(actual), expected_nan)

    expected_positive_infinity = np.zeros_like(expected_nan)
    expected_positive_infinity[0, 0, 1] = True
    expected_positive_infinity[0, 1, 0] = True
    np.testing.assert_array_equal(np.isposinf(actual), expected_positive_infinity)
    expected_negative_infinity = np.zeros_like(expected_nan)
    expected_negative_infinity[1, 0, 1] = True
    expected_negative_infinity[1, 1, 0] = True
    np.testing.assert_array_equal(np.isneginf(actual), expected_negative_infinity)

    finite_expected = np.diag([1.5000000000000004, 1.5000000000000004, 3, 4, 5, 6])
    finite_expected[0, 1] = finite_expected[1, 0] = -0.5000000000000001
    finite_expected[4, 5] = np.nan
    expected = np.full_like(covariance, np.nan)
    expected[0, 0, 1] = expected[0, 1, 0] = np.inf
    expected[1, 0, 1] = expected[1, 1, 0] = -np.inf
    expected[2] = finite_expected
    np.testing.assert_array_equal(actual, expected)


def test_cartesian_rotate_9d_nonidentity_normalizes_infinities_before_rotation() -> (
    None
):
    covariance = np.zeros((3, 9, 9), dtype=np.float64)
    covariance[0, 0, 0], covariance[0, 1, 1] = np.inf, -np.inf
    covariance[1, 0, 0], covariance[1, 1, 1] = -np.inf, np.inf
    covariance[0, 2, 3] = np.nan
    covariance[0, 6, 6] = np.inf
    covariance[1, 7, 7] = -np.inf
    covariance[2] = np.diag(np.arange(1.0, 10.0))
    covariance[2, 4, 5] = np.nan
    matrix = _nonidentity_xy_rotation()

    rotated = _cartesian_with_covariance_rows(covariance).rotate(
        matrix[None, :, :], "nonidentity-9d"
    )
    actual = rotated.covariance.to_full_matrix()

    expected_nan = np.zeros_like(covariance, dtype=bool)
    expected_nan[0, 2, 3] = True
    expected_nan[2, 4, 5] = True
    np.testing.assert_array_equal(np.isnan(actual), expected_nan)

    expected_positive_infinity = np.zeros_like(expected_nan)
    expected_positive_infinity[0, 0, 1] = True
    expected_positive_infinity[0, 1, 0] = True
    np.testing.assert_array_equal(np.isposinf(actual), expected_positive_infinity)
    expected_negative_infinity = np.zeros_like(expected_nan)
    expected_negative_infinity[1, 0, 1] = True
    expected_negative_infinity[1, 1, 0] = True
    np.testing.assert_array_equal(np.isneginf(actual), expected_negative_infinity)

    assert actual[0, 0, 0] == 0.0
    assert actual[0, 1, 1] == 0.0
    assert actual[1, 0, 0] == 0.0
    assert actual[1, 1, 1] == 0.0
    assert actual[0, 6, 6] == np.finfo(np.float64).max
    assert actual[1, 7, 7] == np.finfo(np.float64).min

    finite_expected = np.diag(
        [1.5000000000000004, 1.5000000000000004, 3, 4, 5, 6, 7, 8, 9]
    )
    finite_expected[0, 1] = finite_expected[1, 0] = -0.5000000000000001
    finite_expected[4, 5] = np.nan
    expected = np.zeros_like(covariance)
    expected[0, 0, 1] = expected[0, 1, 0] = np.inf
    expected[0, 2, 3] = np.nan
    expected[0, 6, 6] = np.finfo(np.float64).max
    expected[1, 0, 1] = expected[1, 1, 0] = -np.inf
    expected[1, 7, 7] = np.finfo(np.float64).min
    expected[2] = finite_expected
    np.testing.assert_array_equal(actual, expected)


def test_cartesian_rotate_extended_special_values_and_matrix_shapes() -> None:
    covariance = np.zeros((1, 9, 9), dtype=np.float64)
    covariance[0, 0, 0] = np.inf
    covariance[0, 0, 1] = -np.inf
    covariance[0, 2, 2] = np.nan
    covariance[0, 3, 3] = 1e-30
    covariance[0, 6, 6] = 1e-30
    covariance[0, 6, 7] = np.inf
    coordinates = CartesianCoordinates.from_kwargs(
        x=[1.0],
        y=[2.0],
        z=[3.0],
        vx=[0.1],
        vy=[0.2],
        vz=[0.3],
        covariance=CoordinateCovariances.from_matrix(covariance),
        origin=Origin.from_kwargs(code=["origin"]),
        frame="equatorial",
    )
    matrix = np.eye(6)

    rotated_2d = coordinates.rotate(matrix, "identity")
    rotated_3d = coordinates.rotate(matrix[None, :, :], "identity")

    expected = np.nan_to_num(covariance, nan=0.0)
    expected[np.isnan(covariance)] = np.nan
    coordinate_block = expected[:, :6, :6]
    expected[:, :6, :6] = np.where(
        np.abs(coordinate_block) < 1e-25, 0.0, coordinate_block
    )
    np.testing.assert_equal(rotated_2d.covariance.to_full_matrix(), expected)
    np.testing.assert_equal(
        rotated_3d.covariance.to_full_matrix(),
        rotated_2d.covariance.to_full_matrix(),
    )
    assert rotated_2d.covariance.to_full_matrix()[0, 0, 0] == np.finfo(float).max
    assert rotated_2d.covariance.to_full_matrix()[0, 0, 1] == np.finfo(float).min
    assert np.isnan(rotated_2d.covariance.to_full_matrix()[0, 2, 2])
    assert rotated_2d.covariance.to_full_matrix()[0, 3, 3] == 0.0
    assert rotated_2d.covariance.to_full_matrix()[0, 6, 6] == 1e-30
    assert rotated_2d.covariance.to_full_matrix()[0, 6, 7] == np.finfo(float).max


@pytest.mark.parametrize("shape", [(36,), (5, 5), (2, 6, 6), (6, 6, 1)])
def test_cartesian_rotate_rejects_noncanonical_matrix_shapes(
    shape: tuple[int, ...],
) -> None:
    coordinates = CartesianCoordinates.from_kwargs(
        x=[1.0],
        y=[2.0],
        z=[3.0],
        vx=[0.1],
        vy=[0.2],
        vz=[0.3],
        covariance=CoordinateCovariances.from_matrix(np.eye(6)[None]),
        origin=Origin.from_kwargs(code=["origin"]),
        frame="equatorial",
    )
    with pytest.raises(
        ValueError,
        match=r"rotation_matrix must have shape \(6, 6\) or \(1, 6, 6\)",
    ):
        coordinates.rotate(np.zeros(shape), "invalid")


def test_CartesianCoordinates_translate() -> None:
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


def test_CartesianCoordinates_attributes() -> None:
    # Test that the correct position attributes are returned both for individual
    # dimensions but also multiple dimensions (as vectors)
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

    # Test coordinate dimensions
    np.testing.assert_equal(coords.x, values[:, 0])
    np.testing.assert_equal(coords.y, values[:, 1])
    np.testing.assert_equal(coords.z, values[:, 2])
    np.testing.assert_equal(coords.vx, values[:, 3])
    np.testing.assert_equal(coords.vy, values[:, 4])
    np.testing.assert_equal(coords.vz, values[:, 5])

    # Test coordinate vectors and vector magnitudes
    r_mag = np.linalg.norm(values[:, :3], axis=1)
    r_hat = values[:, :3] / r_mag[:, None]
    v_mag = np.linalg.norm(values[:, 3:], axis=1)
    v_hat = values[:, 3:] / v_mag[:, None]
    np.testing.assert_equal(coords.r, values[:, :3])
    np.testing.assert_equal(coords.r_mag, r_mag)
    np.testing.assert_equal(coords.r_hat, r_hat)
    np.testing.assert_equal(coords.v, values[:, 3:])
    np.testing.assert_equal(coords.v_mag, v_mag)
    np.testing.assert_equal(coords.v_hat, v_hat)

    # Test covariance sigmas
    np.testing.assert_equal(coords.sigma_x, np.sqrt(covariances[:, 0, 0]))
    np.testing.assert_equal(coords.sigma_y, np.sqrt(covariances[:, 1, 1]))
    np.testing.assert_equal(coords.sigma_z, np.sqrt(covariances[:, 2, 2]))
    np.testing.assert_equal(coords.sigma_vx, np.sqrt(covariances[:, 3, 3]))
    np.testing.assert_equal(coords.sigma_vy, np.sqrt(covariances[:, 4, 4]))
    np.testing.assert_equal(coords.sigma_vz, np.sqrt(covariances[:, 5, 5]))

    # Test position and velocity sigma vectors
    sigma_r = np.empty((N, 3))
    sigma_r[:, 0] = np.sqrt(covariances[:, 0, 0])
    sigma_r[:, 1] = np.sqrt(covariances[:, 1, 1])
    sigma_r[:, 2] = np.sqrt(covariances[:, 2, 2])
    np.testing.assert_equal(coords.sigma_r, sigma_r)

    sigma_r_mag = np.linalg.norm(sigma_r, axis=1)
    np.testing.assert_equal(coords.sigma_r_mag, sigma_r_mag)

    sigma_v = np.empty((N, 3))
    sigma_v[:, 0] = np.sqrt(covariances[:, 3, 3])
    sigma_v[:, 1] = np.sqrt(covariances[:, 4, 4])
    sigma_v[:, 2] = np.sqrt(covariances[:, 5, 5])
    np.testing.assert_equal(coords.sigma_v, sigma_v)

    sigma_v_mag = np.linalg.norm(sigma_v, axis=1)
    np.testing.assert_equal(coords.sigma_v_mag, sigma_v_mag)


def test_CartesianCoordinates_ric() -> None:
    N, D, d = 500, 6, 3
    values = np.random.random((N, D)) * 10 - 5
    covariances = np.random.random((N, D, D))

    # Make sure we have at least one 0,0,0 for r and v
    values[0, :d] = 0
    values[1, d:] = 0

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
    ric6 = coords.ric6_matrix

    # Rotations are per line, so iterate
    h_mag = coords.h_mag
    for i in range(N):
        rot = ric6[i]
        # Check the structure
        assert (rot[:d, :d] == rot[d:, d:]).all()
        assert (rot[:d, d:] == 0).all()
        assert (rot[d:, :d] == 0).all()
        # If orbital plane cannot be established, we should have identity
        if h_mag[i] < SPECIFIC_ANGULAR_MOMENTUM_TOLERANCE:
            assert (rot == np.eye(D)).all()
            continue
        # For all "normal" matrices, we should have a proper rotation matrix
        assert np.allclose(rot.T @ rot, np.eye(D))
        rotated = coords[i].rotate(rot, "ric")
        # Rotated r_hat should be the new X axis
        assert np.allclose(rotated.r_hat, [1, 0, 0])
        # Rotated h_hat should be the new Z axis
        assert np.allclose(rotated.h / rotated.h_mag, [0, 0, 1])
        # In-track should be in the same direction as rotated V, but they may not perfectly align
        assert np.dot(rotated.v_hat, [0, 1, 0]) > 0
