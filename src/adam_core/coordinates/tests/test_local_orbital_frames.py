import numpy as np
import pytest

from ...dynamics.propagation import propagate_2body
from ...orbits import Orbits
from ...time import Timestamp
from ...utils.helpers.orbits import make_real_orbits
from .. import CartesianCoordinates, CoordinateCovariances, Origin
from ..local_orbital_frames import LocalFrameCovariances, local_frame_jacobians
from ..origin import OriginGravitationalParameters
from ..transform import transform_coordinates

MU_SUN = float(OriginGravitationalParameters.SUN)


def axes_of(coords, frame):
    return local_frame_jacobians(coords, frame)[:, :3, :3]


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
    r_hat, v_hat = coords.r_hat, coords.v_hat
    w_hat = coords.h / np.linalg.norm(coords.h, axis=1)[:, None]

    rsw = axes_of(coords, "rtn")
    np.testing.assert_allclose(rsw[:, 0], r_hat, atol=1e-14)
    np.testing.assert_allclose(rsw[:, 1], np.cross(w_hat, r_hat), atol=1e-14)
    np.testing.assert_allclose(rsw[:, 2], w_hat, atol=1e-14)
    # RSW, RTN and RIC are the frame the existing Rust RIC matrices give.
    np.testing.assert_allclose(
        local_frame_jacobians(coords, "RIC"), coords.ric6_matrix, atol=1e-15
    )

    tnw = axes_of(coords, "TNW")
    np.testing.assert_allclose(tnw[:, 0], v_hat, atol=1e-14)
    np.testing.assert_allclose(tnw[:, 1], np.cross(w_hat, v_hat), atol=1e-14)
    np.testing.assert_allclose(tnw[:, 2], w_hat, atol=1e-14)

    vnc = axes_of(coords, "VNC")
    np.testing.assert_allclose(vnc[:, 0], v_hat, atol=1e-14)
    np.testing.assert_allclose(vnc[:, 1], w_hat, atol=1e-14)
    np.testing.assert_allclose(vnc[:, 2], np.cross(v_hat, w_hat), atol=1e-14)
    # VNC rows are (T, W, -N) of TNW. A TNW covariance must never be labelled VNC.
    np.testing.assert_allclose(
        vnc, np.stack([tnw[:, 0], tnw[:, 2], -tnw[:, 1]], axis=1), atol=1e-14
    )
    assert not np.allclose(vnc, tnw)


def test_rotating_frames_follow_two_body_motion(heliocentric_orbits):
    a = 1.3
    coords = circular_state(a)
    n = np.sqrt(MU_SUN / a**3)
    for family in ("RSW", "TNW", "VNC"):
        jacobian = local_frame_jacobians(coords, f"{family}_ROTATING")[0]
        np.testing.assert_allclose(
            jacobian[3:, :3], np.cross([0, 0, n], jacobian[:3, :3]), atol=1e-15
        )
        assert not local_frame_jacobians(coords, f"{family}_INERTIAL")[0, 3:, :3].any()
    at_rest = local_frame_jacobians(coords, "RSW_ROTATING")[0] @ coords.values[0]
    np.testing.assert_allclose(at_rest, [a, 0, 0, 0, 0, 0], atol=1e-15)

    # Velocity block rows omega x e_i match finite differences of the axes.
    orbits = heliocentric_orbits[:2]
    t0 = orbits.coordinates.time[0].rescale("tdb").mjd().to_numpy(False)[0]
    dt = 1e-3
    times = Timestamp.from_mjd([t0 - dt, t0, t0 + dt], scale="tdb")
    propagated = propagate_2body(orbits, times)
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
            basis = axes_of(rows.coordinates, frame)
            rate = local_frame_jacobians(rows.coordinates, frame)[1, 3:, :3]
            np.testing.assert_allclose(
                rate, (basis[2] - basis[0]) / (2 * dt), atol=1e-8
            )


def test_covariance_product(heliocentric_orbits):
    source = heliocentric_orbits.coordinates.covariance.to_matrix()
    source[2] = np.nan
    orbits = heliocentric_orbits.set_column(
        "coordinates.covariance", CoordinateCovariances.from_matrix(source)
    )
    product = LocalFrameCovariances.from_orbits(orbits, "vnc_inertial")
    rotated = product.covariance.to_matrix()
    assert product.frame == "VNC_INERTIAL"
    assert product.inertial_frame == "equatorial"
    assert product.orbit_id.to_pylist() == orbits.orbit_id.to_pylist()
    assert product.time.days.equals(orbits.coordinates.time.days)
    assert product.origin.code.to_pylist() == ["SUN"] * 5
    assert np.isnan(rotated[2]).all() and not np.isnan(rotated[[0, 1, 3, 4]]).any()
    # A rotation preserves the position variance and inverts exactly.
    keep = [0, 1, 3, 4]
    np.testing.assert_allclose(
        np.trace(rotated[keep, :3, :3], axis1=1, axis2=2),
        np.trace(source[keep, :3, :3], axis1=1, axis2=2),
        rtol=1e-12,
    )
    jacobians = local_frame_jacobians(orbits.coordinates, "VNC_INERTIAL")[keep]
    recovered = np.einsum("nji,njk,nkl->nil", jacobians, rotated[keep], jacobians)
    np.testing.assert_allclose(recovered, source[keep], rtol=1e-10, atol=1e-30)
    assert LocalFrameCovariances.from_orbits(orbits).frame == "VNC_ROTATING"


def test_errors(heliocentric_orbits):
    with pytest.raises(ValueError, match="Unknown local orbital frame"):
        local_frame_jacobians(heliocentric_orbits.coordinates, "LVLH")
    nulls = heliocentric_orbits.set_column(
        "coordinates.covariance", CoordinateCovariances.nulls(5)
    )
    with pytest.raises(ValueError, match="no covariance"):
        LocalFrameCovariances.from_orbits(nulls)
    itrf = transform_coordinates(heliocentric_orbits.coordinates, frame_out="itrf93")
    with pytest.raises(ValueError, match="inertial frame"):
        local_frame_jacobians(itrf, "VNC")
    radial = circular_state()
    radial = radial.set_column("vx", radial.vy).set_column("vy", radial.vx)
    with pytest.raises(ValueError, match="no orbit plane"):
        local_frame_jacobians(radial, "VNC")

    mars = circular_state(2.0e-5, origin="MARS")
    np.testing.assert_allclose(
        local_frame_jacobians(mars, "RSW")[0], np.eye(6), atol=1e-15
    )
    with pytest.raises(ValueError, match="Pass mu="):
        local_frame_jacobians(mars, "RSW_ROTATING")
    assert local_frame_jacobians(mars, "RSW_ROTATING", mu=1e-12).shape == (1, 6, 6)
