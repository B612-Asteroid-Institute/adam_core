import json

import numpy as np
import pytest

from adam_core import _rust_native

from ...coordinates import CoordinateCovariances, LocalFrameCovariances
from ...coordinates.transform import transform_coordinates
from ...coordinates.units import convert_cartesian_covariance_au_to_km
from ...dynamics.propagation import propagate_2body
from ...time import Timestamp
from ...utils.helpers.orbits import make_real_orbits
from ..oem_io import orbit_from_oem, orbit_to_oem


@pytest.fixture
def history():
    seed = make_real_orbits(1)
    coordinates = transform_coordinates(seed.coordinates, frame_out="equatorial")
    times = Timestamp.from_mjd(61465.0 + np.array([0, 14.5, 31, 366]), scale="tdb")
    orbits = propagate_2body(seed.set_column("coordinates", coordinates), times)
    factors = np.random.default_rng(1).normal(size=(4, 6, 6)) * 1e-7
    factors[:, :, 3:] *= 1e-2
    covariances = np.einsum("nij,nkj->nik", factors, factors)
    return orbits.set_column(
        "coordinates.covariance", CoordinateCovariances.from_matrix(covariances)
    )


def write(orbits, path, **options) -> str:
    date = "2026-10-06T00:00:00"
    orbit_to_oem(orbits, str(path), "ME", version="3.0", creation_date=date, **options)
    return open(path).read()


def test_round_trip(history, tmp_path):
    text = write(history, tmp_path / "states.oem")
    assert text.startswith(
        "CCSDS_OEM_VERS = 3.0\nCREATION_DATE = 2026-10-06T00:00:00\nORIGINATOR = ME\n"
    )
    assert "\nREF_FRAME = ICRF\nTIME_SYSTEM = TDB\n" in text
    loaded = orbit_from_oem(str(tmp_path / "states.oem")).coordinates
    assert loaded.frame == "equatorial"
    assert loaded.time.equals(history.coordinates.time)
    np.testing.assert_allclose(loaded.values, history.coordinates.values, rtol=1e-15)
    expected = history.coordinates.covariance.to_matrix()
    np.testing.assert_allclose(loaded.covariance.to_matrix(), expected, rtol=1e-15)
    with pytest.raises(TypeError):  # a string is not a list of comments
        write(history, tmp_path / "x.oem", comments="abc")


def test_vnc_rotating_blocks_hold_the_local_frame_product(history, tmp_path):
    path = tmp_path / "vnc.oem"
    text = write(history, path, covariance_frame="VNC_ROTATING")
    assert text.count("COV_REF_FRAME = VNC_ROTATING") == 4
    segment = json.loads(_rust_native.oem_parse_kvn(str(path)))["segments"][0]
    written = np.array([block["matrix"] for block in segment["covariances"]])
    expected = convert_cartesian_covariance_au_to_km(
        LocalFrameCovariances.from_orbits(history).covariance.to_matrix()
    )
    sigma = np.sqrt(np.diagonal(expected, axis1=1, axis2=2))
    bound = 1e-12 * sigma[:, :, None] * sigma[:, None, :]
    assert np.all(np.abs(written.reshape(4, 6, 6) - expected) <= bound)
    # The reader keeps the states and ignores the block.
    loaded = orbit_from_oem(str(path)).coordinates
    assert loaded.time.equals(history.coordinates.time)
    assert loaded.covariance.is_all_nan()


def test_off_grid_epochs_are_rounded_with_a_warning(history, tmp_path):
    time = history.coordinates.time
    nanos = time.nanos.to_numpy(zero_copy_only=False).copy()
    nanos[1] += 123_456
    shifted = Timestamp.from_kwargs(days=time.days, nanos=nanos, scale="tdb")
    with pytest.warns(UserWarning, match="1 of 4 epochs rounded to the millisecond"):
        write(history.set_column("coordinates.time", shifted), tmp_path / "x.oem")
    assert orbit_from_oem(str(tmp_path / "x.oem")).coordinates.time.equals(time)


def test_nine_by_nine_covariances_write_their_state_block(history, tmp_path):
    # Orbits with non-gravitational parameters carry 9x9 covariances.
    full = np.full((4, 9, 9), 1e-22)
    full[:, :6, :6] = history.coordinates.covariance.to_matrix()
    nongrav = history.set_column(
        "coordinates.covariance", CoordinateCovariances.from_matrix(full)
    )
    for options in ({}, {"covariance_frame": "VNC_ROTATING"}):
        text = write(nongrav, tmp_path / "nongrav.oem", **options)
        assert text.count("EPOCH = ") == 4
        assert text == write(history, tmp_path / "states.oem", **options)
