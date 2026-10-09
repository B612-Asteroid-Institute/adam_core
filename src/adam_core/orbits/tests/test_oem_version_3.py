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


def with_covariances(orbits, matrices):
    covariance = CoordinateCovariances.from_matrix(matrices)
    return orbits.set_column("coordinates.covariance", covariance)


@pytest.fixture
def history():
    seed = make_real_orbits(1)
    coordinates = transform_coordinates(seed.coordinates, frame_out="equatorial")
    times = Timestamp.from_mjd(61465.0 + np.array([0, 14.5, 31, 366]), scale="tdb")
    orbits = propagate_2body(seed.set_column("coordinates", coordinates), times)
    scale = np.repeat([1e-7, 1e-9], 3)
    factors = np.random.default_rng(1).normal(size=(4, 6, 6)) * scale
    return with_covariances(orbits, np.einsum("nij,nkj->nik", factors, factors))


def write(orbits, path, **options) -> str:
    date = "2026-10-06T00:00:00"
    orbit_to_oem(orbits, str(path), "ME", version="3.0", creation_date=date, **options)
    return open(path).read()


def test_round_trip_with_labels_and_rounded_epochs(history, tmp_path):
    time = history.coordinates.time
    nanos = time.nanos.to_numpy(zero_copy_only=False) + [0, 123_456, 0, 0]
    shifted = history.set_column("coordinates.time.nanos", nanos)
    with pytest.warns(UserWarning, match="1 of 4 epochs rounded to the millisecond"):
        text = write(shifted, tmp_path / "x.oem", object_name="NAME", object_id="ID")
    assert text.startswith("CCSDS_OEM_VERS = 3.0\nCREATION_DATE = 2026-10-06T00:00:00")
    assert "\nORIGINATOR = ME\n\nMETA_START\nOBJECT_NAME = NAME\nOBJECT_ID = ID" in text
    assert "\nCENTER_NAME = SUN\nREF_FRAME = ICRF\nTIME_SYSTEM = TDB\n" in text
    loaded = orbit_from_oem(str(tmp_path / "x.oem")).coordinates
    assert loaded.frame == "equatorial" and loaded.time.equals(time)
    np.testing.assert_allclose(loaded.values, history.coordinates.values, rtol=1e-15)
    expected = history.coordinates.covariance.to_matrix()
    np.testing.assert_allclose(loaded.covariance.to_matrix(), expected, rtol=1e-15)
    with pytest.raises(TypeError):  # a string is not a list of comments
        write(history, tmp_path / "x.oem", comments="abc")


def test_vnc_rotating_blocks_hold_the_local_frame_product(history, tmp_path):
    # 9x9 covariances (non-gravitational parameters) write their state block.
    full = np.full((4, 9, 9), 1e-22)
    full[:, :6, :6] = history.coordinates.covariance.to_matrix()
    for options in ({}, {"covariance_frame": "VNC_ROTATING"}):
        text = write(with_covariances(history, full), tmp_path / "vnc.oem", **options)
        assert text == write(history, tmp_path / "six.oem", **options)
    assert text.count("COV_REF_FRAME = VNC_ROTATING") == 4
    segment = json.loads(_rust_native.oem_parse_kvn(str(tmp_path / "vnc.oem")))
    written = [block["matrix"] for block in segment["segments"][0]["covariances"]]
    local = LocalFrameCovariances.from_orbits(history).covariance.to_matrix()
    expected = convert_cartesian_covariance_au_to_km(local)
    sigma = np.sqrt(np.diagonal(expected, axis1=1, axis2=2))
    bound = 1e-12 * sigma[:, :, None] * sigma[:, None, :]
    assert np.all(np.abs(np.reshape(written, (4, 6, 6)) - expected) <= bound)
    # The reader keeps the states and ignores the block.
    loaded = orbit_from_oem(str(tmp_path / "vnc.oem")).coordinates
    assert loaded.time.equals(history.coordinates.time)
    assert loaded.covariance.is_all_nan()
