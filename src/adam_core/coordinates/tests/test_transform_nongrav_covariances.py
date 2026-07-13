import numpy as np

from ...time import Timestamp
from ..cartesian import CartesianCoordinates
from ..cometary import CometaryCoordinates
from ..covariances import CoordinateCovariances
from ..keplerian import KeplerianCoordinates
from ..origin import Origin
from ..transform import transform_coordinates


def _extended_covariance(seed: int = 5) -> np.ndarray:
    """
    Build a well-conditioned 9x9 covariance over (x, y, z, vx, vy, vz, A1,
    A2, A3) with realistic per-dimension scales and non-zero orbital/non-grav
    cross-covariances.
    """
    rng = np.random.default_rng(seed)
    scales = np.array([1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6, 1e-13, 1e-14, 1e-14])
    R = rng.normal(size=(9, 9)) * 0.1
    corr = np.eye(9) + (R + R.T) / 2
    np.fill_diagonal(corr, 1.0)
    w, V = np.linalg.eigh(corr)
    corr = V @ np.diag(np.clip(w, 1e-3, None)) @ V.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    return corr * np.outer(scales, scales)


def _cartesian_with_covariance(covariance: np.ndarray) -> CartesianCoordinates:
    return CartesianCoordinates.from_kwargs(
        x=[0.85],
        y=[-0.4],
        z=[0.02],
        vx=[0.005],
        vy=[0.014],
        vz=[0.0003],
        time=Timestamp.from_mjd([60000.0], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN"]),
        frame="ecliptic",
        covariance=CoordinateCovariances.from_matrix(covariance),
    )


def test_transform_coordinates_roundtrip_cometary_preserves_nongrav_block():
    covariance = _extended_covariance().reshape(1, 9, 9)
    cartesian = _cartesian_with_covariance(covariance)

    cometary = transform_coordinates(cartesian, CometaryCoordinates)

    # The non-grav block is invariant under the coordinate transform and the
    # cross-covariances rotate with the orbital Jacobian.
    cometary_full = cometary.covariance.to_full_matrix()
    np.testing.assert_allclose(
        cometary_full[0, 6:, 6:], covariance[0, 6:, 6:], rtol=1e-12
    )
    assert not np.allclose(
        cometary_full[0, :6, 6:], covariance[0, :6, 6:], rtol=1e-3, atol=0
    )

    # The one-way transformed orbital block must match the independently
    # transformed 6x6 coordinate covariance (same Jacobian) -- without this,
    # a no-op covariance transform would still pass the roundtrip checks.
    cartesian_6x6 = _cartesian_with_covariance(covariance[:, :6, :6])
    cometary_6x6 = transform_coordinates(cartesian_6x6, CometaryCoordinates)
    np.testing.assert_allclose(
        cometary_full[:, :6, :6],
        cometary_6x6.covariance.to_matrix(),
        rtol=1e-10,
    )

    back = transform_coordinates(cometary, CartesianCoordinates)
    np.testing.assert_allclose(back.covariance.to_full_matrix(), covariance, rtol=1e-6)


def test_transform_coordinates_roundtrip_keplerian_preserves_nongrav_block():
    covariance = _extended_covariance(seed=11).reshape(1, 9, 9)
    cartesian = _cartesian_with_covariance(covariance)

    keplerian = transform_coordinates(cartesian, KeplerianCoordinates)
    keplerian_full = keplerian.covariance.to_full_matrix()
    np.testing.assert_allclose(
        keplerian_full[0, 6:, 6:], covariance[0, 6:, 6:], rtol=1e-12
    )

    # One-way check: the transformed orbital block must match the 6x6
    # coordinate covariance transformed through the standard path.
    cartesian_6x6 = _cartesian_with_covariance(covariance[:, :6, :6])
    keplerian_6x6 = transform_coordinates(cartesian_6x6, KeplerianCoordinates)
    np.testing.assert_allclose(
        keplerian_full[:, :6, :6],
        keplerian_6x6.covariance.to_matrix(),
        rtol=1e-10,
    )

    back = transform_coordinates(keplerian, CartesianCoordinates)
    np.testing.assert_allclose(back.covariance.to_full_matrix(), covariance, rtol=1e-6)


def test_transform_coordinates_frame_rotation_preserves_nongrav_block():
    covariance = _extended_covariance(seed=23).reshape(1, 9, 9)
    cartesian = _cartesian_with_covariance(covariance)

    equatorial = transform_coordinates(cartesian, frame_out="equatorial")
    equatorial_full = equatorial.covariance.to_full_matrix()
    np.testing.assert_allclose(
        equatorial_full[0, 6:, 6:], covariance[0, 6:, 6:], rtol=1e-12
    )

    # One-way check: the rotated orbital block must match the independently
    # rotated 6x6 coordinate covariance. A no-op rotation of the extended
    # covariance would pass the roundtrip assertions below but fail here.
    cartesian_6x6 = _cartesian_with_covariance(covariance[:, :6, :6])
    equatorial_6x6 = transform_coordinates(cartesian_6x6, frame_out="equatorial")
    np.testing.assert_allclose(
        equatorial_full[:, :6, :6],
        equatorial_6x6.covariance.to_matrix(),
        rtol=1e-10,
    )
    assert not np.allclose(
        equatorial_full[0, :6, 6:], covariance[0, :6, 6:], rtol=1e-3, atol=0
    )

    back = transform_coordinates(equatorial, frame_out="ecliptic")
    np.testing.assert_allclose(
        back.covariance.to_full_matrix(), covariance, rtol=1e-9, atol=1e-40
    )


def test_transform_coordinates_mixed_rows_keep_plain_6x6():
    # A table mixing a row with the non-grav block and a plain 6x6 row must
    # transform both correctly: the plain row keeps NaN trailing dimensions.
    covariance9 = _extended_covariance(seed=31)
    covariance = np.full((2, 9, 9), np.nan)
    covariance[0] = covariance9
    covariance[1, :6, :6] = covariance9[:6, :6]

    cartesian = CartesianCoordinates.from_kwargs(
        x=[0.85, 1.2],
        y=[-0.4, 0.3],
        z=[0.02, -0.01],
        vx=[0.005, -0.004],
        vy=[0.014, 0.012],
        vz=[0.0003, 0.0001],
        time=Timestamp.from_mjd([60000.0, 60000.0], scale="tdb"),
        origin=Origin.from_kwargs(code=["SUN", "SUN"]),
        frame="ecliptic",
        covariance=CoordinateCovariances.from_matrix(covariance),
    )

    keplerian = transform_coordinates(cartesian, KeplerianCoordinates)
    assert keplerian.covariance.nongrav_block_mask().tolist() == [True, False]
    full = keplerian.covariance.to_full_matrix()
    np.testing.assert_allclose(full[0, 6:, 6:], covariance9[6:, 6:], rtol=1e-12)
    assert np.isnan(full[1, 6:, :]).all()
    assert np.isnan(full[1, :, 6:]).all()
    assert not np.isnan(full[1, :6, :6]).any()


def test_coordinate_covariances_full_matrix_round_trip():
    covariance9 = _extended_covariance(seed=41)
    covariance = np.full((3, 9, 9), np.nan)
    covariance[0] = covariance9
    covariance[1, :6, :6] = covariance9[:6, :6]
    # Row 2 stays all NaN (no covariance at all).

    table = CoordinateCovariances.from_matrix(covariance)
    assert table.nongrav_block_mask().tolist() == [True, False, False]
    assert table.has_nongrav_block()

    full = table.to_full_matrix()
    np.testing.assert_allclose(full[0], covariance9)
    np.testing.assert_allclose(full[1, :6, :6], covariance9[:6, :6])
    assert np.isnan(full[1, 6:, :]).all()
    assert np.isnan(full[2]).all()

    matrix = table.to_matrix()
    assert matrix.shape == (3, 6, 6)
    np.testing.assert_allclose(matrix[0], covariance9[:6, :6])
    assert np.isnan(matrix[2]).all()
