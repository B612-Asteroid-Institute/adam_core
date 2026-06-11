import numpy as np

from ...orbits.solved_state_covariances import SolvedStateCovariances
from ...time import Timestamp
from .. import (
    CartesianCoordinates,
    CometaryCoordinates,
    CoordinateCovariances,
    KeplerianCoordinates,
    Origin,
    transform_coordinates,
)


def test_transform_coordinates_roundtrip_cometary_preserves_solved_state_covariance():
    cometary_covariance = np.diag(
        [1e-10, 2e-10, 3e-8, 4e-8, 5e-8, 6e-6, 9e-26]
    ).reshape(1, 7, 7)
    cometary_covariance[0, 0, 6] = 2e-18
    cometary_covariance[0, 6, 0] = 2e-18
    cometary_covariance[0, 4, 6] = -3e-18
    cometary_covariance[0, 6, 4] = -3e-18

    coords = CometaryCoordinates.from_kwargs(
        q=[0.9],
        e=[0.1],
        i=[5.0],
        raan=[120.0],
        ap=[35.0],
        tp=[60010.0],
        covariance=CoordinateCovariances.from_matrix(cometary_covariance[:, :6, :6]),
        time=Timestamp.from_mjd([60000.0], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN"]),
        frame="ecliptic",
    )
    solved = SolvedStateCovariances.from_matrix(
        cometary_covariance,
        [["q", "e", "i", "raan", "ap", "tp", "A2"]],
    )

    cartesian, solved_cartesian = transform_coordinates(
        coords,
        representation_out=CartesianCoordinates,
        solved_state_covariances=solved,
    )
    cometary_roundtrip, solved_roundtrip = transform_coordinates(
        cartesian,
        representation_out=CometaryCoordinates,
        solved_state_covariances=solved_cartesian,
    )

    assert solved_cartesian.dimension[0].as_py() == 7
    assert solved_cartesian.parameter_names[0].as_py() == "x,y,z,vx,vy,vz,A2"
    assert solved_roundtrip.parameter_names[0].as_py() == "q,e,i,raan,ap,tp,A2"
    # The one-way transformed orbital block must match the independently
    # transformed 6x6 coordinate covariance (same Jacobian) — without this,
    # a no-op covariance transform would still pass the roundtrip checks.
    np.testing.assert_allclose(
        solved_cartesian.to_orbital_covariances().to_matrix(),
        cartesian.covariance.to_matrix(),
        rtol=1e-12,
        atol=0,
    )
    np.testing.assert_allclose(
        solved.to_orbital_covariances().to_matrix(),
        coords.covariance.to_matrix(),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        solved_roundtrip.to_orbital_covariances().to_matrix(),
        coords.covariance.to_matrix(),
        rtol=0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        solved_roundtrip.to_orbital_covariances().to_matrix(),
        cometary_roundtrip.covariance.to_matrix(),
        rtol=0,
        atol=1e-10,
    )
    np.testing.assert_allclose(cometary_roundtrip.values, coords.values, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        solved_roundtrip.to_matrix()[0],
        cometary_covariance[0],
        rtol=0,
        atol=1e-9,
    )


def test_transform_coordinates_roundtrip_keplerian_preserves_9x9_solved_state_covariance():
    keplerian_covariance = np.diag(
        [1e-10, 2e-10, 3e-8, 4e-8, 5e-8, 6e-8, 9e-26, 4e-26, 1e-26]
    ).reshape(1, 9, 9)
    keplerian_covariance[0, 1, 6] = 2e-18
    keplerian_covariance[0, 6, 1] = 2e-18
    keplerian_covariance[0, 5, 7] = -1e-18
    keplerian_covariance[0, 7, 5] = -1e-18
    keplerian_covariance[0, 3, 8] = 5e-19
    keplerian_covariance[0, 8, 3] = 5e-19

    coords = KeplerianCoordinates.from_kwargs(
        a=[1.2],
        e=[0.15],
        i=[7.0],
        raan=[30.0],
        ap=[45.0],
        M=[12.0],
        covariance=CoordinateCovariances.from_matrix(keplerian_covariance[:, :6, :6]),
        time=Timestamp.from_mjd([60000.0], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN"]),
        frame="ecliptic",
    )
    solved = SolvedStateCovariances.from_matrix(
        keplerian_covariance,
        [["a", "e", "i", "raan", "ap", "M", "A1", "A2", "A3"]],
    )

    cartesian, solved_cartesian = transform_coordinates(
        coords,
        representation_out=CartesianCoordinates,
        solved_state_covariances=solved,
    )
    keplerian_roundtrip, solved_roundtrip = transform_coordinates(
        cartesian,
        representation_out=KeplerianCoordinates,
        solved_state_covariances=solved_cartesian,
    )

    assert solved_cartesian.dimension[0].as_py() == 9
    assert solved_cartesian.parameter_names[0].as_py() == "x,y,z,vx,vy,vz,A1,A2,A3"
    assert solved_roundtrip.parameter_names[0].as_py() == "a,e,i,raan,ap,M,A1,A2,A3"
    # One-way check: the transformed orbital block must match the 6x6
    # coordinate covariance transformed through the standard path.
    np.testing.assert_allclose(
        solved_cartesian.to_orbital_covariances().to_matrix(),
        cartesian.covariance.to_matrix(),
        rtol=1e-12,
        atol=0,
    )
    np.testing.assert_allclose(
        solved.to_orbital_covariances().to_matrix(),
        coords.covariance.to_matrix(),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        solved_roundtrip.to_orbital_covariances().to_matrix(),
        coords.covariance.to_matrix(),
        rtol=0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        solved_roundtrip.to_orbital_covariances().to_matrix(),
        keplerian_roundtrip.covariance.to_matrix(),
        rtol=0,
        atol=1e-10,
    )
    np.testing.assert_allclose(keplerian_roundtrip.values, coords.values, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        solved_roundtrip.to_matrix()[0],
        keplerian_covariance[0],
        rtol=0,
        atol=1e-9,
    )


def test_transform_coordinates_frame_rotation_preserves_solved_state_covariance():
    covariance = np.diag([1e-8, 2e-8, 3e-8, 4e-10, 5e-10, 6e-10, 9e-26]).reshape(1, 7, 7)
    covariance[0, 0, 6] = 7e-18
    covariance[0, 6, 0] = 7e-18
    covariance[0, 2, 6] = -2e-18
    covariance[0, 6, 2] = -2e-18

    coords = CartesianCoordinates.from_kwargs(
        x=[1.0],
        y=[0.5],
        z=[0.2],
        vx=[0.01],
        vy=[0.02],
        vz=[0.03],
        covariance=CoordinateCovariances.from_matrix(covariance[:, :6, :6]),
        time=Timestamp.from_mjd([60000.0], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN"]),
        frame="ecliptic",
    )
    solved = SolvedStateCovariances.from_matrix(
        covariance,
        [["x", "y", "z", "vx", "vy", "vz", "A2"]],
    )

    rotated, solved_rotated = transform_coordinates(
        coords, frame_out="equatorial", solved_state_covariances=solved
    )
    roundtrip, solved_roundtrip = transform_coordinates(
        rotated, frame_out="ecliptic", solved_state_covariances=solved_rotated
    )

    assert solved_rotated.parameter_names[0].as_py() == "x,y,z,vx,vy,vz,A2"
    # One-way check: the rotated orbital block must match the independently
    # rotated 6x6 coordinate covariance. A no-op rotation of the solved-state
    # covariance would pass the roundtrip assertions below but fail here.
    np.testing.assert_allclose(
        solved_rotated.to_orbital_covariances().to_matrix(),
        rotated.covariance.to_matrix(),
        rtol=1e-12,
        atol=0,
    )
    np.testing.assert_allclose(
        solved.to_orbital_covariances().to_matrix(),
        coords.covariance.to_matrix(),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        solved_roundtrip.to_orbital_covariances().to_matrix(),
        coords.covariance.to_matrix(),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        solved_roundtrip.to_orbital_covariances().to_matrix(),
        roundtrip.covariance.to_matrix(),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(roundtrip.values, coords.values, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        solved_roundtrip.to_matrix()[0],
        covariance[0],
        rtol=0,
        atol=1e-12,
    )


def test_solved_state_covariances_to_orbital_covariances_handles_null_rows():
    solved = SolvedStateCovariances.from_matrix(
        [None, np.eye(7, dtype=np.float64)],
        [None, ["x", "y", "z", "vx", "vy", "vz", "A2"]],
    )

    orbital = solved.to_orbital_covariances().to_matrix()

    assert orbital.shape == (2, 6, 6)
    assert np.isnan(orbital[0]).all()
    np.testing.assert_allclose(orbital[1], np.eye(6), rtol=0, atol=0)
