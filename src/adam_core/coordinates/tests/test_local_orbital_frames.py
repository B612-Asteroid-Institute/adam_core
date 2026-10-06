import numpy as np
import pytest

from ...constants import KM_P_AU
from ...dynamics.propagation import propagate_2body
from ...orbits import Orbits
from ...time import Timestamp
from ...utils.helpers.orbits import make_real_orbits
from .. import CartesianCoordinates, CoordinateCovariances, Origin
from ..local_orbital_frames import (
    LocalFrameCovariances,
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


def circular_state(a: float = 1.3, origin: str = "SUN") -> CartesianCoordinates:
    n = np.sqrt(MU_SUN / a**3)
    return CartesianCoordinates.from_kwargs(
        x=[a],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[n * a],
        vz=[0.0],
        time=Timestamp.from_mjd([60000.0], scale="tdb"),
        frame="equatorial",
        origin=Origin.from_kwargs(code=[origin]),
    )


def test_axes_follow_the_registry_and_match_rust_ric(heliocentric_orbits):
    coords = heliocentric_orbits.coordinates
    r_hat = coords.r_hat
    v_hat = coords.v_hat
    w_hat = coords.h / np.linalg.norm(coords.h, axis=1)[:, None]

    # Bare names are the quasi-inertial variants, as the CCSDS messages use them.
    assert resolve_local_orbital_frame("rtn") == "RSW_INERTIAL"
    assert resolve_local_orbital_frame("RIC") == "RSW_INERTIAL"
    assert resolve_local_orbital_frame("TNW") == "TNW_INERTIAL"
    assert resolve_local_orbital_frame("vnc") == "VNC_INERTIAL"
    assert resolve_local_orbital_frame("VNC_ROTATING") == "VNC_ROTATING"

    rsw = local_frame_rotation_matrices(coords, "RSW")
    np.testing.assert_allclose(rsw[:, 0], r_hat, atol=1e-14)
    np.testing.assert_allclose(rsw[:, 1], np.cross(w_hat, r_hat), atol=1e-14)
    np.testing.assert_allclose(rsw[:, 2], w_hat, atol=1e-14)
    # The existing Rust RIC matrices are the same frame.
    np.testing.assert_allclose(rsw, coords.ric3_matrix, atol=1e-15)
    np.testing.assert_allclose(
        local_frame_jacobians(coords, "RSW_INERTIAL"), coords.ric6_matrix, atol=1e-15
    )

    tnw = local_frame_rotation_matrices(coords, "TNW")
    np.testing.assert_allclose(tnw[:, 0], v_hat, atol=1e-14)
    np.testing.assert_allclose(tnw[:, 1], np.cross(w_hat, v_hat), atol=1e-14)
    np.testing.assert_allclose(tnw[:, 2], w_hat, atol=1e-14)

    vnc = local_frame_rotation_matrices(coords, "VNC")
    np.testing.assert_allclose(vnc[:, 0], v_hat, atol=1e-14)
    np.testing.assert_allclose(vnc[:, 1], w_hat, atol=1e-14)
    np.testing.assert_allclose(vnc[:, 2], np.cross(v_hat, w_hat), atol=1e-14)

    # VNC rows are (T, W, -N) of TNW. A TNW covariance must never be labelled VNC.
    np.testing.assert_allclose(
        vnc, np.stack([tnw[:, 0], tnw[:, 2], -tnw[:, 1]], axis=1), atol=1e-14
    )
    assert not np.allclose(vnc, tnw)


def test_circular_orbit_rate_and_corotating_velocity():
    a = 1.3
    coords = circular_state(a)
    n = np.sqrt(MU_SUN / a**3)
    for family in ("RSW", "TNW", "VNC"):
        omega = local_frame_angular_velocity(coords, f"{family}_ROTATING")
        np.testing.assert_allclose(omega[0], [0.0, 0.0, n], atol=1e-15)
        np.testing.assert_array_equal(
            local_frame_angular_velocity(coords, f"{family}_INERTIAL"),
            np.zeros((1, 3)),
        )
    # A per-row mu array is accepted and gives the same rate.
    np.testing.assert_allclose(
        local_frame_angular_velocity(coords, "VNC_ROTATING", mu=np.array([MU_SUN])),
        [[0.0, 0.0, n]],
        atol=1e-15,
    )
    # In the rotating RSW frame a circular orbit is at rest: R (v - omega x r) = 0.
    local_state = local_frame_jacobians(coords, "RSW_ROTATING")[0] @ coords.values[0]
    np.testing.assert_allclose(local_state, [a, 0.0, 0.0, 0.0, 0.0, 0.0], atol=1e-15)


def test_frame_rates_match_finite_differences(heliocentric_orbits):
    orbits = heliocentric_orbits[:2]
    t0 = orbits.coordinates.time[0].rescale("tdb").mjd().to_numpy(False)[0]
    dt = 1e-3  # days
    propagated = propagate_2body(
        orbits, Timestamp.from_mjd([t0 - dt, t0, t0 + dt], scale="tdb")
    )
    for orbit_id in orbits.orbit_id.to_pylist():
        rows = propagated.apply_mask(
            propagated.orbit_id.to_numpy(zero_copy_only=False) == orbit_id
        ).sort_by(
            [
                ("coordinates.time.days", "ascending"),
                ("coordinates.time.nanos", "ascending"),
            ]
        )
        for frame in ("RSW_ROTATING", "TNW_ROTATING", "VNC_ROTATING"):
            # The triad rotates with the frame rate: de_i/dt = omega x e_i.
            basis = local_frame_rotation_matrices(rows.coordinates, frame)
            omega = local_frame_angular_velocity(rows.coordinates, frame)[1]
            np.testing.assert_allclose(
                np.cross(omega, basis[1]), (basis[2] - basis[0]) / (2 * dt), atol=1e-8
            )


def test_inertial_rotation_preserves_covariance(heliocentric_orbits):
    source = heliocentric_orbits.coordinates.covariance.to_matrix()
    rotated = LocalFrameCovariances.from_orbits(
        heliocentric_orbits, "VNC_INERTIAL"
    ).covariance.to_matrix()
    np.testing.assert_allclose(
        np.trace(rotated[:, :3, :3], axis1=1, axis2=2),
        np.trace(source[:, :3, :3], axis1=1, axis2=2),
        rtol=1e-12,
    )
    jacobians = local_frame_jacobians(heliocentric_orbits.coordinates, "VNC_INERTIAL")
    recovered = np.einsum("nji,njk,nkl->nil", jacobians, rotated, jacobians)
    np.testing.assert_allclose(recovered, source, rtol=1e-10, atol=1e-30)


def test_from_orbits_product_contract(heliocentric_orbits):
    matrices = heliocentric_orbits.coordinates.covariance.to_matrix()
    matrices[2] = np.nan
    orbits = heliocentric_orbits.set_column(
        "coordinates.covariance", CoordinateCovariances.from_matrix(matrices)
    )
    product = LocalFrameCovariances.from_orbits(orbits, "vnc_rotating")
    assert len(product) == len(orbits)
    assert product.frame == "VNC_ROTATING"
    assert product.reference_frame == "equatorial"
    assert product.orbit_id.to_pylist() == orbits.orbit_id.to_pylist()
    assert product.object_id.to_pylist() == orbits.object_id.to_pylist()
    assert product.time.scale == orbits.coordinates.time.scale
    assert product.time.days.equals(orbits.coordinates.time.days)
    assert product.origin.code.to_pylist() == ["SUN"] * len(product)
    # Rows stay aligned with the input, missing covariances stay NaN.
    rotated = product.covariance.to_matrix()
    assert np.isnan(rotated[2]).all()
    assert not np.isnan(rotated[[0, 1, 3, 4]]).any()
    np.testing.assert_allclose(
        product.to_matrix_km()[:, :3, :3], rotated[:, :3, :3] * KM_P_AU**2
    )
    assert LocalFrameCovariances.from_orbits(orbits, "VNC").frame == "VNC_INERTIAL"


def test_errors(heliocentric_orbits):
    with pytest.raises(ValueError, match="Unknown local orbital frame"):
        resolve_local_orbital_frame("LVLH")

    nulls = heliocentric_orbits.set_column(
        "coordinates.covariance", CoordinateCovariances.nulls(len(heliocentric_orbits))
    )
    with pytest.raises(ValueError, match="no covariance"):
        LocalFrameCovariances.from_orbits(nulls)

    itrf = heliocentric_orbits.set_column(
        "coordinates",
        transform_coordinates(heliocentric_orbits.coordinates, frame_out="itrf93"),
    )
    with pytest.raises(ValueError, match="inertial frame"):
        LocalFrameCovariances.from_orbits(itrf)

    radial = circular_state()
    radial = radial.set_column("vx", radial.vy).set_column("vy", radial.vx)
    with pytest.raises(ValueError, match="no defined orbit plane"):
        local_frame_rotation_matrices(radial, "VNC")


def test_inertial_frames_do_not_need_the_origin_mu():
    # MARS has no gravitational parameter in Origin.mu(), so the basis and
    # the inertial Jacobian must not ask for one.
    coords = circular_state(2.0e-5, origin="MARS")
    np.testing.assert_allclose(
        local_frame_rotation_matrices(coords, "RTN")[0], np.eye(3), atol=1e-15
    )
    np.testing.assert_allclose(
        local_frame_jacobians(coords, "RSW_INERTIAL")[0], np.eye(6), atol=1e-15
    )
    orbits = Orbits.from_kwargs(
        orbit_id=["m"],
        object_id=["m"],
        coordinates=coords.set_column(
            "covariance", CoordinateCovariances.from_sigmas(np.ones((1, 6)) * 1e-6)
        ),
    )
    assert LocalFrameCovariances.from_orbits(orbits, "RSW").frame == "RSW_INERTIAL"
    with pytest.raises(ValueError, match="Pass mu="):
        local_frame_angular_velocity(coords, "RSW_ROTATING")
    assert local_frame_angular_velocity(coords, "RSW_ROTATING", mu=1e-12).shape == (
        1,
        3,
    )
