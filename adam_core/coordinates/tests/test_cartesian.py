import numpy as np

from ..cartesian import CartesianCoordinates
from ..covariances import CoordinateCovariances
from ..frame import Frame
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
        covariances=CoordinateCovariances.from_matrix(covariances),
        origin=Origin.from_kwargs(code=["origin"] * N),
        frame=Frame.from_kwargs(name=["frame"] * N),
    )

    rot_matrix = np.identity(D)
    coords_rotated = coords.rotate(rot_matrix, "identity")

    np.testing.assert_equal(coords_rotated.values, values)
    np.testing.assert_equal(coords_rotated.covariances.to_matrix(), covariances)
    return
