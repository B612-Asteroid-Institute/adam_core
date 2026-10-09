from decimal import Decimal, getcontext

import numpy as np
import pytest

from ...dynamics.propagation import propagate_2body
from ...time import Timestamp
from ...utils.helpers.orbits import make_real_orbits
from .. import CoordinateCovariances, Origin
from ..local_orbital_frames import LocalFrameCovariances, local_frame_jacobians
from ..transform import transform_coordinates


@pytest.fixture
def orbits():
    orbits = make_real_orbits(5)
    coordinates = transform_coordinates(orbits.coordinates, frame_out="equatorial")
    return orbits.set_column("coordinates", coordinates)


def with_covariances(orbits, matrices):
    covariance = CoordinateCovariances.from_matrix(matrices)
    return orbits.set_column("coordinates.covariance", covariance)


def test_table_and_jacobians(orbits):
    source = orbits.coordinates.covariance.to_matrix()
    source[2] = np.nan
    orbits = with_covariances(orbits, source)
    product = LocalFrameCovariances.from_orbits(orbits, " rtn ")
    assert (product.frame, product.inertial_frame) == ("RSW_INERTIAL", "equatorial")
    assert product.orbit_id.equals(orbits.orbit_id)
    assert product.time.equals(orbits.coordinates.time)
    assert product.origin.code.to_pylist() == ["SUN"] * 5
    rotated = product.covariance.to_matrix()
    assert np.isnan(rotated[2]).all() and not np.isnan(rotated[[0, 1, 3, 4]]).any()
    jacobians = local_frame_jacobians(orbits.coordinates, "RSW")
    recovered = np.einsum("nji,njk,nkl->nil", jacobians, rotated, jacobians)
    np.testing.assert_allclose(recovered[:2], source[:2], rtol=1e-10, atol=1e-30)
    assert LocalFrameCovariances.from_orbits(orbits).frame == "VNC_ROTATING"


def test_errors(orbits):
    coords = orbits.coordinates
    with pytest.raises(ValueError, match="Unknown local orbital frame 'LVLH'"):
        local_frame_jacobians(coords, "LVLH")
    nan = with_covariances(orbits, np.full((5, 6, 6), np.nan))
    with pytest.raises(ValueError, match="no covariance"):
        LocalFrameCovariances.from_orbits(nan)
    with pytest.raises(ValueError, match="inertial frame, got 'itrf93'"):
        local_frame_jacobians(transform_coordinates(coords, frame_out="itrf93"), "VNC")
    radial = coords.set_column("vx", coords.x).set_column("vy", coords.y)
    with pytest.raises(ValueError, match="State 0 is not finite or has no orbit plane"):
        local_frame_jacobians(radial.set_column("vz", coords.z), "VNC")
    mars = coords.set_column("origin", Origin.from_kwargs(code=["MARS"] * 5))
    assert local_frame_jacobians(mars, "RSW").shape == (5, 6, 6)
    with pytest.raises(ValueError, match="Unknown origin code"):
        local_frame_jacobians(mars, "RSW_ROTATING")
    assert local_frame_jacobians(mars, "RSW_ROTATING", mu=1e-12).shape == (5, 6, 6)


def test_jacobian_and_product_match_a_decimal_reference(orbits):
    """The double-double evaluation against 50 digit Decimal arithmetic: Jacobian
    blocks within one ulp of their largest entry, the product within one ulp each."""
    getcontext().prec = 50
    times = Timestamp.from_mjd([60000.0, 60400.0], scale="tdb")
    orbits = propagate_2body(orbits[:1], times)
    # The second covariance adds a one day timing error along the orbit, whose
    # rotating frame velocity rows cancel so deeply that a Jacobian rounded to
    # double precision misses the bound below by billions of ulps.
    values, mu = orbits.coordinates.values, float(orbits.coordinates.origin.mu()[0])
    r, v = values[1, :3], values[1, 3:]
    along = np.concatenate([v, -mu * r / np.linalg.norm(r) ** 3])
    timing = np.stack([0 * np.eye(6), np.outer(along, along)])
    stored = orbits.coordinates.covariance.to_matrix() + timing
    orbits = with_covariances(orbits, stored)
    jacobians = local_frame_jacobians(orbits.coordinates, "VNC_ROTATING")
    rotated = LocalFrameCovariances.from_orbits(orbits).covariance.to_matrix()

    exact = np.vectorize(lambda x: Decimal(float(x)), otypes=[object])
    for k in range(2):
        r, v = exact(values[k]).reshape(2, 3)
        v_hat, h = v / np.sqrt(v @ v), np.cross(r, v)
        h_hat = h / np.sqrt(h @ h)
        rows = np.array([v_hat, h_hat, np.cross(v_hat, h_hat)])
        acceleration = -Decimal(mu) * r / np.sqrt(r @ r) ** 3
        v_hat_dot = (acceleration - (acceleration @ v_hat) * v_hat) / np.sqrt(v @ v)
        rates = [v_hat_dot, 0 * v_hat, np.cross(v_hat_dot, h_hat)]
        omega = sum(np.cross(e, de) for e, de in zip(rows, rates)) / 2
        J = np.block([[rows, 0 * rows], [np.cross(omega, rows), rows]])
        # the library symmetrises the stored covariance exactly before rotating
        C = (exact(stored[k]) + exact(stored[k]).T) / 2
        exact_J, exact_P = J.astype(float), (J @ C @ J.T).astype(float)
        # one ulp of each block's largest entry: the orbit normal row of the rate
        # block is zero in exact arithmetic and carries only reference noise
        for block in (np.s_[:3, :3], np.s_[3:, :3], np.s_[3:, 3:]):
            tolerance = np.spacing(np.abs(exact_J[block]).max())
            assert np.all(np.abs(jacobians[k][block] - exact_J[block]) <= tolerance)
        assert np.all(np.abs(rotated[k] - exact_P) <= np.spacing(np.abs(exact_P)))
