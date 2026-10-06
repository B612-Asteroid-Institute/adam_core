import json
import re

import numpy as np
import pyarrow as pa
import pytest

from adam_core import _rust_native

from ...coordinates import (
    CoordinateCovariances,
    LocalFrameCovariances,
    Origin,
    transform_coordinates,
)
from ...coordinates.units import convert_cartesian_covariance_au_to_km
from ...dynamics.propagation import propagate_2body
from ...time import Timestamp
from ...utils.helpers.orbits import make_real_orbits
from .. import OrbitEphemerisMessage, Orbits
from ..oem_io import orbit_from_oem, orbit_to_oem

ORIGINATOR = "TEST ORIGINATOR"
EPOCHS = [
    "2027-03-01T00:00:00",
    "2027-03-15T12:00:00",
    "2027-04-01T00:00:00",
    "2028-03-01T00:00:00",
    "2028-03-15T12:00:00",
    "2028-04-01T00:00:00",
]


@pytest.fixture
def history() -> Orbits:
    """One object, six heliocentric equatorial TDB epochs with covariances."""
    seed = make_real_orbits(1)
    seed = seed.set_column(
        "coordinates", transform_coordinates(seed.coordinates, frame_out="equatorial")
    )
    seed = seed.set_column(
        "object_id", pa.array(["TEST OBJECT"], type=pa.large_string())
    )
    propagated = propagate_2body(seed, Timestamp.from_iso8601(EPOCHS, scale="tdb"))
    factors = np.random.default_rng(1).normal(size=(6, 6, 6)) * 1e-7
    factors[:, :, 3:] *= 1e-2
    return propagated.set_column(
        "coordinates.covariance",
        CoordinateCovariances.from_matrix(np.einsum("nij,nkj->nik", factors, factors)),
    )


def test_round_trip_heliocentric_icrf(history, tmp_path):
    message = OrbitEphemerisMessage.from_orbits(
        history, ORIGINATOR, creation_date="2026-10-06T00:00:00", comments=["a comment"]
    )
    assert (message.object_name, message.object_id) == ("TEST OBJECT", "TEST OBJECT")
    assert (message.center_name, message.ref_frame, message.time_system) == (
        "SUN",
        "ICRF",
        "TDB",
    )

    path = message.write(tmp_path / "states.oem")
    lines = open(path).read().split("\n")
    assert lines[:3] == [
        "CCSDS_OEM_VERS = 3.0",
        "CREATION_DATE = 2026-10-06T00:00:00",
        f"ORIGINATOR = {ORIGINATOR}",
    ]
    for line in (
        "COMMENT a comment",
        "CENTER_NAME = SUN",
        "REF_FRAME = ICRF",
        "TIME_SYSTEM = TDB",
        "START_TIME = 2027-03-01T00:00:00.000",
        "STOP_TIME = 2028-04-01T00:00:00.000",
    ):
        assert line in lines
    assert len([line for line in lines if re.match(r"^\d{4}-", line)]) == 6
    assert "COVARIANCE_START" not in lines

    loaded = orbit_from_oem(message.write(path, covariance_frame="ICRF"))
    assert loaded.coordinates.frame == "equatorial"
    assert loaded.coordinates.time.scale == "tdb"
    assert loaded.coordinates.origin.code.to_pylist() == ["SUN"] * 6
    assert loaded.coordinates.time.days.equals(history.coordinates.time.days)
    assert loaded.coordinates.time.nanos.equals(history.coordinates.time.nanos)
    np.testing.assert_allclose(
        loaded.coordinates.values, history.coordinates.values, rtol=1e-12
    )
    np.testing.assert_allclose(
        loaded.coordinates.covariance.to_matrix(),
        history.coordinates.covariance.to_matrix(),
        rtol=1e-12,
    )


def test_local_frame_covariance_blocks(history, tmp_path):
    message = OrbitEphemerisMessage.from_orbits(history, ORIGINATOR)
    # TNW is in the OEM covariance frame set, VNC is not.
    tnw = open(
        message.write(tmp_path / "tnw.oem", covariance_frame="TNW", strict=True)
    ).read()
    assert tnw.count("COV_REF_FRAME = TNW") == 6 and "SANA" not in tnw
    with pytest.raises(ValueError, match="outside the OEM covariance frame set"):
        message.write(tmp_path / "x.oem", covariance_frame="VNC_ROTATING", strict=True)
    path = message.write(tmp_path / "vnc.oem", covariance_frame="VNC_ROTATING")
    text = open(path).read()
    assert text.count("COV_REF_FRAME = VNC_ROTATING") == 6
    assert "COMMENT COV_REF_FRAME VNC_ROTATING is a SANA orbit-relative" in text

    # The block holds the separate product's matrices in km, and the legacy
    # reader ignores a block that is not in REF_FRAME.
    records = json.loads(_rust_native.oem_parse_kvn(path))["segments"][0]["covariances"]
    written = np.array([record["matrix"] for record in records]).reshape(6, 6, 6)
    product = LocalFrameCovariances.from_orbits(history, "VNC_ROTATING")
    np.testing.assert_allclose(
        written,
        convert_cartesian_covariance_au_to_km(product.covariance.to_matrix()),
        rtol=1e-12,
    )
    assert orbit_from_oem(path).coordinates.covariance.is_all_nan()

    with pytest.raises(ValueError, match="Unknown local orbital frame"):
        message.write(tmp_path / "x.oem", covariance_frame="LVLH")
    nulls = history.set_column("coordinates.covariance", CoordinateCovariances.nulls(6))
    with pytest.raises(ValueError, match="no covariance"):
        OrbitEphemerisMessage.from_orbits(nulls, ORIGINATOR).write(
            tmp_path / "x.oem", covariance_frame="ICRF"
        )


def test_input_rules(history, tmp_path):
    bad = [
        (
            "Transform to equatorial first",
            history.set_column(
                "coordinates",
                transform_coordinates(history.coordinates, frame_out="ecliptic"),
            ),
        ),
        (
            "one object about one center",
            history.set_column(
                "object_id", pa.array(["A"] * 3 + ["B"] * 3, type=pa.large_string())
            ),
        ),
        (
            "one object about one center",
            history.set_column(
                "coordinates.origin",
                Origin.from_kwargs(code=["SUN"] * 5 + ["SOLAR_SYSTEM_BARYCENTER"]),
            ),
        ),
        (
            "needs an object_id",
            history.set_column("object_id", pa.nulls(6, pa.large_string())),
        ),
        ("at least one state", Orbits.empty()),
        ("unique", history.take(pa.array([0, 0, 1, 2, 3, 4]))),
    ]
    for match, orbits in bad:
        with pytest.raises(ValueError, match=match):
            OrbitEphemerisMessage.from_orbits(orbits, ORIGINATOR)
    # Unsorted input is sorted by time.
    states = OrbitEphemerisMessage.from_orbits(
        history.take(pa.array([5, 4, 3, 2, 1, 0])), ORIGINATOR
    ).states
    assert states.coordinates.time.days.equals(history.coordinates.time.days)
    # Epochs off the millisecond grid are moved onto it with a warning.
    time = history.coordinates.time
    nanos = time.nanos.to_numpy(zero_copy_only=False).copy()
    nanos[1] += 123_456
    shifted = history.set_column(
        "coordinates.time",
        Timestamp.from_kwargs(days=time.days, nanos=nanos, scale="tdb"),
    )
    with pytest.warns(UserWarning, match="1 of 6 epochs .* 123.5 microseconds"):
        message = OrbitEphemerisMessage.from_orbits(shifted, ORIGINATOR)
    assert message.states.coordinates.time.nanos.equals(time.nanos)


def test_matches_legacy_writer_formatting(history, tmp_path):
    legacy_path = str(tmp_path / "legacy.oem")
    orbit_to_oem(history, legacy_path, originator=ORIGINATOR)
    legacy = open(legacy_path).read()
    creation_date = re.search(r"^CREATION_DATE = (.*)$", legacy, re.M).group(1)
    message = OrbitEphemerisMessage.from_orbits(
        history, ORIGINATOR, creation_date=creation_date
    )
    # Only the version and frame labels differ from the legacy writer.
    expected = legacy.replace("CCSDS_OEM_VERS = 2.0", "CCSDS_OEM_VERS = 3.0").replace(
        "REF_FRAME = EME2000", "REF_FRAME = ICRF"
    )
    assert (
        open(message.write(tmp_path / "new.oem", covariance_frame="ICRF")).read()
        == expected
    )
