import numpy as np
import pyarrow as pa
import pytest

from ...constants import KM_P_AU
from ...dynamics.propagation import propagate_2body
from ...orbits import Orbits
from ...time import Timestamp
from ...utils.helpers.orbits import make_real_orbits
from .. import CartesianCoordinates, CoordinateCovariances, Origin
from ..local_orbital_frames import (
    LOCAL_ORBITAL_FRAMES,
    LocalFrameCovariances,
    _basis_and_rates,
    local_frame_angular_velocity,
    local_frame_jacobians,
    local_frame_rotation_matrices,
    resolve_local_orbital_frame,
)
from ..origin import OriginGravitationalParameters
from ..transform import transform_coordinates

MU_SUN = float(OriginGravitationalParameters.SUN)


@pytest.fixture
def heliocentric_orbits() -> Orbits:
    orbits = make_real_orbits(5)
    coordinates = transform_coordinates(orbits.coordinates, frame_out="equatorial")
    return orbits.set_column("coordinates", coordinates)


def circular_state(a: float = 1.3, frame: str = "equatorial") -> CartesianCoordinates:
    n = np.sqrt(MU_SUN / a**3)
    return CartesianCoordinates.from_kwargs(
        x=[a],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[n * a],
        vz=[0.0],
        time=Timestamp.from_mjd([60000.0], scale="tdb"),
        frame=frame,
        origin=Origin.from_kwargs(code=["SUN"]),
    )


def test_resolve_local_orbital_frame_aliases():
    assert resolve_local_orbital_frame("rtn") == "RSW_INERTIAL"
    assert resolve_local_orbital_frame("RIC") == "RSW_INERTIAL"
    assert resolve_local_orbital_frame("RSW") == "RSW_INERTIAL"
    assert resolve_local_orbital_frame("TNW") == "TNW_INERTIAL"
    assert resolve_local_orbital_frame("VNC") == "VNC_INERTIAL"
    assert resolve_local_orbital_frame("vnc_rotating") == "VNC_ROTATING"
    for frame in LOCAL_ORBITAL_FRAMES:
        assert resolve_local_orbital_frame(frame) == frame
    with pytest.raises(ValueError, match="Unknown local orbital frame"):
        resolve_local_orbital_frame("LVLH")


@pytest.mark.parametrize("frame", LOCAL_ORBITAL_FRAMES)
def test_rotation_matrices_are_proper_rotations(heliocentric_orbits, frame):
    rotation = local_frame_rotation_matrices(heliocentric_orbits.coordinates, frame)
    assert rotation.shape == (len(heliocentric_orbits), 3, 3)
    identity = np.einsum("nij,nkj->nik", rotation, rotation)
    np.testing.assert_allclose(
        identity, np.broadcast_to(np.eye(3), rotation.shape), atol=1e-14
    )
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-14)


def test_axis_definitions_follow_the_registry(heliocentric_orbits):
    coords = heliocentric_orbits.coordinates
    r_hat = coords.r_hat
    v_hat = coords.v_hat
    w_hat = coords.h / np.linalg.norm(coords.h, axis=1)[:, None]

    rsw = local_frame_rotation_matrices(coords, "RSW_INERTIAL")
    np.testing.assert_allclose(rsw[:, 0], r_hat, atol=1e-14)
    np.testing.assert_allclose(rsw[:, 2], w_hat, atol=1e-14)
    np.testing.assert_allclose(rsw[:, 1], np.cross(w_hat, r_hat), atol=1e-14)

    tnw = local_frame_rotation_matrices(coords, "TNW_INERTIAL")
    np.testing.assert_allclose(tnw[:, 0], v_hat, atol=1e-14)
    np.testing.assert_allclose(tnw[:, 2], w_hat, atol=1e-14)
    np.testing.assert_allclose(tnw[:, 1], np.cross(w_hat, v_hat), atol=1e-14)

    vnc = local_frame_rotation_matrices(coords, "VNC_INERTIAL")
    np.testing.assert_allclose(vnc[:, 0], v_hat, atol=1e-14)
    np.testing.assert_allclose(vnc[:, 1], w_hat, atol=1e-14)
    np.testing.assert_allclose(vnc[:, 2], np.cross(v_hat, w_hat), atol=1e-14)


def test_vnc_is_not_tnw(heliocentric_orbits):
    coords = heliocentric_orbits.coordinates
    vnc = local_frame_rotation_matrices(coords, "VNC")
    tnw = local_frame_rotation_matrices(coords, "TNW")
    # VNC rows are (T, W, -N) of TNW: two shared axes, the third flipped
    # and moved. A TNW covariance must never be labelled VNC.
    expected = np.stack([tnw[:, 0], tnw[:, 2], -tnw[:, 1]], axis=1)
    np.testing.assert_allclose(vnc, expected, atol=1e-14)
    assert not np.allclose(vnc, tnw)


def test_rsw_inertial_matches_rust_ric(heliocentric_orbits):
    coords = heliocentric_orbits.coordinates
    np.testing.assert_allclose(
        local_frame_rotation_matrices(coords, "RSW_INERTIAL"),
        coords.ric3_matrix,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        local_frame_jacobians(coords, "RSW_INERTIAL"), coords.ric6_matrix, atol=1e-15
    )


@pytest.mark.parametrize("family", ["RSW", "TNW", "VNC"])
def test_circular_orbit_rate_is_the_mean_motion(family):
    a = 1.3
    coords = circular_state(a)
    n = np.sqrt(MU_SUN / a**3)
    omega = local_frame_angular_velocity(coords, f"{family}_ROTATING")
    np.testing.assert_allclose(omega[0], [0.0, 0.0, n], atol=1e-15)
    np.testing.assert_array_equal(
        local_frame_angular_velocity(coords, f"{family}_INERTIAL"), np.zeros((1, 3))
    )


@pytest.mark.parametrize("frame", ["RSW_ROTATING", "TNW_ROTATING", "VNC_ROTATING"])
def test_basis_rates_match_finite_differences(heliocentric_orbits, frame):
    orbits = heliocentric_orbits[:2]
    epoch = orbits.coordinates.time[0].rescale("tdb")
    dt = 1e-3  # days
    t0 = epoch.mjd().to_numpy(zero_copy_only=False)[0]
    times = Timestamp.from_mjd([t0 - dt, t0, t0 + dt], scale="tdb")
    propagated = propagate_2body(orbits, times)
    for orbit_id in orbits.orbit_id.to_pylist():
        rows = propagated.apply_mask(
            np.asarray(propagated.orbit_id.to_numpy(zero_copy_only=False) == orbit_id)
        )
        rows = rows.sort_by(
            [
                ("coordinates.time.days", "ascending"),
                ("coordinates.time.nanos", "ascending"),
            ]
        )
        coords = rows.coordinates
        basis, rates = _basis_and_rates(coords, frame)
        finite_difference = (basis[2] - basis[0]) / (2 * dt)
        np.testing.assert_allclose(rates[1], finite_difference, atol=1e-8)
        # The rates are the rotation of the triad: de_i = omega x e_i.
        omega = local_frame_angular_velocity(coords, frame)[1]
        np.testing.assert_allclose(rates[1], np.cross(omega, basis[1]), atol=1e-14)


def test_rotating_jacobian_differs_only_in_velocity_rows(heliocentric_orbits):
    coords = heliocentric_orbits.coordinates
    inertial = local_frame_jacobians(coords, "VNC_INERTIAL")
    rotating = local_frame_jacobians(coords, "VNC_ROTATING")
    np.testing.assert_array_equal(rotating[:, :3, :], inertial[:, :3, :])
    np.testing.assert_array_equal(rotating[:, 3:, 3:], inertial[:, 3:, 3:])
    assert not np.allclose(rotating[:, 3:, :3], 0.0)
    # With no central attraction the velocity direction does not turn, so
    # the VNC frame rate vanishes and the two variants agree.
    np.testing.assert_allclose(
        local_frame_jacobians(coords, "VNC_ROTATING", mu=0.0), inertial, atol=1e-15
    )
    # The lower left block is -R [omega]x.
    omega = local_frame_angular_velocity(coords, "VNC_ROTATING")
    rotation = local_frame_rotation_matrices(coords, "VNC_ROTATING")
    for i in range(len(coords)):
        skew = np.array(
            [
                [0.0, -omega[i, 2], omega[i, 1]],
                [omega[i, 2], 0.0, -omega[i, 0]],
                [-omega[i, 1], omega[i, 0], 0.0],
            ]
        )
        np.testing.assert_allclose(rotating[i, 3:, :3], -rotation[i] @ skew, atol=1e-18)


def test_corotating_circular_velocity_vanishes():
    coords = circular_state()
    jacobian = local_frame_jacobians(coords, "RSW_ROTATING")[0]
    local_state = jacobian @ coords.values[0]
    np.testing.assert_allclose(local_state[3:], 0.0, atol=1e-15)
    np.testing.assert_allclose(local_state[:3], [1.3, 0.0, 0.0], atol=1e-15)


def test_inertial_rotation_preserves_covariance_invariants(heliocentric_orbits):
    source = heliocentric_orbits.coordinates.covariance.to_matrix()
    product = LocalFrameCovariances.from_orbits(heliocentric_orbits, "VNC_INERTIAL")
    rotated = product.to_matrix()
    np.testing.assert_allclose(
        np.trace(rotated[:, :3, :3], axis1=1, axis2=2),
        np.trace(source[:, :3, :3], axis1=1, axis2=2),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.sqrt(np.trace(rotated[:, :3, :3], axis1=1, axis2=2)),
        heliocentric_orbits.coordinates.sigma_r_mag,
        rtol=1e-12,
    )
    assert np.all(np.linalg.eigvalsh(rotated) > -1e-20)
    jacobians = local_frame_jacobians(heliocentric_orbits.coordinates, "VNC_INERTIAL")
    recovered = np.einsum("nji,njk,nkl->nil", jacobians, rotated, jacobians)
    np.testing.assert_allclose(recovered, source, rtol=1e-10, atol=1e-30)


def test_from_orbits_table_shape_and_attributes(heliocentric_orbits):
    product = LocalFrameCovariances.from_orbits(heliocentric_orbits, "vnc_rotating")
    assert len(product) == len(heliocentric_orbits)
    assert product.frame == "VNC_ROTATING"
    assert product.reference_frame == "equatorial"
    assert product.orbit_id.to_pylist() == heliocentric_orbits.orbit_id.to_pylist()
    assert product.object_id.to_pylist() == heliocentric_orbits.object_id.to_pylist()
    assert product.time.scale == heliocentric_orbits.coordinates.time.scale
    assert product.time.days.equals(heliocentric_orbits.coordinates.time.days)
    assert product.origin.code.to_pylist() == ["SUN"] * len(product)
    alias = LocalFrameCovariances.from_orbits(heliocentric_orbits, "VNC")
    assert alias.frame == "VNC_INERTIAL"
    km = product.to_matrix_km()
    np.testing.assert_allclose(
        km[:, :3, :3], product.to_matrix()[:, :3, :3] * KM_P_AU**2
    )


def test_from_orbits_keeps_missing_rows_nan(heliocentric_orbits):
    matrices = heliocentric_orbits.coordinates.covariance.to_matrix()
    matrices[2] = np.nan
    orbits = heliocentric_orbits.set_column(
        "coordinates.covariance", CoordinateCovariances.from_matrix(matrices)
    )
    product = LocalFrameCovariances.from_orbits(orbits, "VNC_ROTATING")
    assert len(product) == len(orbits)
    assert np.isnan(product.to_matrix()[2]).all()
    assert not np.isnan(product.to_matrix()[[0, 1, 3, 4]]).any()


def test_from_orbits_errors(heliocentric_orbits):
    nulls = heliocentric_orbits.set_column(
        "coordinates.covariance", CoordinateCovariances.nulls(len(heliocentric_orbits))
    )
    with pytest.raises(ValueError, match="no covariance"):
        LocalFrameCovariances.from_orbits(nulls)

    unspecified = heliocentric_orbits.set_column(
        "coordinates",
        CartesianCoordinates.from_kwargs(
            x=heliocentric_orbits.coordinates.x,
            y=heliocentric_orbits.coordinates.y,
            z=heliocentric_orbits.coordinates.z,
            vx=heliocentric_orbits.coordinates.vx,
            vy=heliocentric_orbits.coordinates.vy,
            vz=heliocentric_orbits.coordinates.vz,
            time=heliocentric_orbits.coordinates.time,
            covariance=heliocentric_orbits.coordinates.covariance,
            origin=heliocentric_orbits.coordinates.origin,
            frame="unspecified",
        ),
    )
    with pytest.raises(ValueError, match="inertial frame"):
        LocalFrameCovariances.from_orbits(unspecified)

    itrf = heliocentric_orbits.set_column(
        "coordinates",
        transform_coordinates(
            heliocentric_orbits.coordinates, frame_out="itrf93", origin_out=None
        ),
    )
    with pytest.raises(ValueError, match="inertial frame"):
        LocalFrameCovariances.from_orbits(itrf)


def test_degenerate_state_raises():
    coords = CartesianCoordinates.from_kwargs(
        x=[1.0],
        y=[0.0],
        z=[0.0],
        vx=[0.01],
        vy=[0.0],
        vz=[0.0],
        time=Timestamp.from_mjd([60000.0], scale="tdb"),
        frame="equatorial",
        origin=Origin.from_kwargs(code=["SUN"]),
    )
    with pytest.raises(ValueError, match="no defined orbit plane"):
        local_frame_rotation_matrices(coords, "VNC_ROTATING")


def test_mu_argument_shapes(heliocentric_orbits):
    coords = heliocentric_orbits.coordinates
    scalar = local_frame_angular_velocity(coords, "VNC_ROTATING", mu=MU_SUN)
    default = local_frame_angular_velocity(coords, "VNC_ROTATING")
    np.testing.assert_allclose(scalar, default, rtol=1e-12)
    per_row = local_frame_angular_velocity(
        coords, "VNC_ROTATING", mu=np.full(len(coords), MU_SUN)
    )
    np.testing.assert_allclose(per_row, default, rtol=1e-12)
    with pytest.raises(ValueError, match="mu must be"):
        local_frame_angular_velocity(coords, "VNC_ROTATING", mu=np.ones(2))


def test_parquet_round_trip(heliocentric_orbits, tmp_path):
    product = LocalFrameCovariances.from_orbits(heliocentric_orbits, "VNC_ROTATING")
    path = tmp_path / "covariance_vnc.parquet"
    product.to_parquet(path)
    loaded = LocalFrameCovariances.from_parquet(path)
    assert loaded.frame == "VNC_ROTATING"
    assert loaded.reference_frame == "equatorial"
    assert loaded.time.scale == product.time.scale
    np.testing.assert_array_equal(loaded.to_matrix(), product.to_matrix())
    assert isinstance(loaded.orbit_id, pa.ChunkedArray) or loaded.orbit_id is not None


def test_inertial_frames_do_not_need_the_origin_mu():
    # MARS has no gravitational parameter in Origin.mu(), so the basis and
    # the inertial Jacobian must not ask for one.
    coords = CartesianCoordinates.from_kwargs(
        x=[2.0e-5],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[1.0e-3],
        vz=[0.0],
        time=Timestamp.from_mjd([60000.0], scale="tdb"),
        frame="equatorial",
        origin=Origin.from_kwargs(code=["MARS"]),
    )
    rotation = local_frame_rotation_matrices(coords, "RTN")
    np.testing.assert_allclose(rotation[0], np.eye(3), atol=1e-15)
    jacobian = local_frame_jacobians(coords, "RSW_INERTIAL")
    np.testing.assert_allclose(jacobian[0], np.eye(6), atol=1e-15)
    np.testing.assert_array_equal(
        local_frame_angular_velocity(coords, "VNC_INERTIAL"), np.zeros((1, 3))
    )
    orbits = Orbits.from_kwargs(
        orbit_id=["m"],
        object_id=["m"],
        coordinates=coords.set_column(
            "covariance", CoordinateCovariances.from_sigmas(np.ones((1, 6)) * 1e-6)
        ),
    )
    product = LocalFrameCovariances.from_orbits(orbits, "RSW")
    assert product.frame == "RSW_INERTIAL"

    with pytest.raises(ValueError, match="Pass mu="):
        local_frame_angular_velocity(coords, "RSW_ROTATING")
    omega = local_frame_angular_velocity(coords, "RSW_ROTATING", mu=1e-12)
    assert omega.shape == (1, 3)
